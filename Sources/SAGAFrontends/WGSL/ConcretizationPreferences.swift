import SAGAShaderIR

extension WGSLParser {
    // Naga concretizes abstract integers using the first type that represents every
    // component: i32, then u32, then f32. Inspect pure constants before constructing
    // any IR so a failed preference cannot leave partially emitted expressions.
    func defaultIntegerConcretization(_ value: WGSLValue, at token: WGSLToken) throws -> WType? {
        switch types[value.type.index].kind {
        case .scalar(.abstractInt), .vector(_, .abstractInt), .array(_, .constant, _): break
        default: return nil
        }
        let kind = try constantKind(value.type, at: token)
        func replacingLeaf(_ kind: WGSLConstKind, with leaf: WGSLConstKind) -> WGSLConstKind? {
            switch kind {
            case .integer(.abstract): return leaf
            case .vector(let n, let element):
                return replacingLeaf(element, with: leaf).map { .vector(n, $0) }
            case .array(let n, let element):
                return replacingLeaf(element, with: leaf).map { .array(n, $0) }
            default: return nil
            }
        }
        guard replacingLeaf(kind, with: .integer(.sint)) != nil else { return nil }
        var fitsI32 = true
        var fitsU32 = true
        func visit(_ handle: WExpr) throws -> Bool {
            try budget.consume(.work)
            guard body.expressions.indices.contains(handle.index) else { throw ShaderFailure.invalidHandle }
            let expression = body.expressions[handle.index]
            switch types[expression.type.index].kind {
            case .scalar(.abstractInt):
                guard case .abstractInt(let value)? = try abstractScalarLeaf(handle, at: token) else { return false }
                fitsI32 = fitsI32 && Int32(exactly: value) != nil
                fitsU32 = fitsU32 && UInt32(exactly: value) != nil
                return true
            case .vector(_, .abstractInt):
                guard let values = try abstractVectorLeaves(handle, at: token) else { return false }
                for literal in values {
                    try budget.consume(.work)
                    guard case .abstractInt(let value) = literal else { return false }
                    fitsI32 = fitsI32 && Int32(exactly: value) != nil
                    fitsU32 = fitsU32 && UInt32(exactly: value) != nil
                }
                return true
            case .array:
                guard case .compose(_, let elements) = expression.kind else { return false }
                try enter()
                defer { nesting -= 1 }
                for element in elements {
                    if try !visit(element) { return false }
                }
                return true
            default: return false
            }
        }
        guard try visit(value.handle) else { return nil }
        let leaf: WGSLConstKind = fitsI32 ? .integer(.sint) : fitsU32 ? .integer(.uint) : .floating(.f32)
        guard let target = replacingLeaf(kind, with: leaf) else { return nil }
        return try constantType(target, at: token)
    }
}
