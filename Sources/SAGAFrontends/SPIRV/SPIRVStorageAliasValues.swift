import SAGAShaderIR

extension SPVFunctionLowerer {
    mutating func reinterpretStorageAlias(
        _ value: IRHandle<IRExpression>, from source: IRHandle<IRType>, to target: IRHandle<IRType>,
        _ i: SPVInstruction, _ statements: inout [IRHandle<IRStatement>]
    ) throws -> IRHandle<IRExpression> {
        if source == target { return value }
        try budget.enterDepth()
        defer { try? budget.leaveDepth() }
        let sourceKind = module.types[source.index].kind
        let targetKind = module.types[target.index].kind
        if sourceKind == targetKind {
            switch sourceKind {
            case .scalar, .vector, .matrix: return value
            default: break
            }
        }
        if let left = IRBitcastShape(sourceKind), let right = IRBitcastShape(targetKind),
           left.size == right.size, left.permits(right) {
            return try emitValue(target, .bitcast(expression: value, type: target), &statements, i)
        }
        let sourceFields = try componentTypes(source, i)
        let targetFields = try componentTypes(target, i)
        guard sourceFields.count == targetFields.count else { throw i.fail("Storage alias value shape mismatch") }
        var fields: [IRHandle<IRExpression>] = []
        for (index, pair) in zip(sourceFields, targetFields).enumerated() {
            let member = try emitValue(pair.0, .accessIndex(base: value, index: UInt32(index)), &statements, i)
            fields.append(try reinterpretStorageAlias(member, from: pair.0, to: pair.1, i, &statements))
        }
        return try emitValue(target, .compose(type: target, components: fields), &statements, i)
    }
}
