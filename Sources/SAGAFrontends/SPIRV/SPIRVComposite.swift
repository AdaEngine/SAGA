import SAGAShaderIR

extension SPVFunctionLowerer {
    mutating func lowerDynamicVector(_ i: SPVInstruction, _ statements: inout [IRHandle<IRStatement>]) throws {
        let inserts = i.op == 78
        try i.count(inserts ? 5 : 4)
        let vector = try value(i.a[2], i, &statements)
        let vectorType = try module.type(vector.type, i)
        let resultType = try module.type(i.a[0], i)
        guard case .vector(let width, let scalar) = module.types[vectorType.index].kind else {
            throw i.fail("Dynamic vector operation requires a vector operand")
        }
        let componentType = try indexedType(vectorType, 0, i)
        guard resultType == (inserts ? vectorType : componentType) else {
            throw i.fail("Dynamic vector result type mismatch")
        }
        let index = try value(i.a[inserts ? 4 : 3], i, &statements)
        let indexType = try module.type(index.type, i)
        guard case .scalar(let integer) = module.types[indexType.index].kind,
              integer.kind == .sint || integer.kind == .uint else {
            throw i.fail("Dynamic vector index requires a scalar integer")
        }
        let component = inserts ? try value(i.a[3], i, &statements) : nil
        if let component, try module.kind(component.type, i) != .scalar(scalar) {
            throw i.fail("Dynamic vector component type mismatch")
        }
        let boolean = try internLoweredType(.scalar(.bool))
        var extracted = inserts ? nil : try expr(componentType, .zeroValue(componentType), i)
        var inserted: [IRHandle<IRExpression>] = []
        for lane in 0..<UInt32(width.rawValue) {
            let literal: IRLiteral = switch (integer.kind, integer.width) {
            case (.sint, 2): .i16(Int16(lane))
            case (.uint, 2): .u16(UInt16(lane))
            case (.sint, 4): .i32(Int32(lane))
            case (.uint, 4): .u32(lane)
            case (.sint, 8): .i64(Int64(lane))
            case (.uint, 8): .u64(UInt64(lane))
            default: throw i.fail("Unsupported dynamic vector index width", code: .unsupportedConstruct)
            }
            let position = try expr(indexType, .literal(literal), i)
            let matches = try emitValue(boolean, .binary(op: .equal, left: index.expression, right: position), &statements, i)
            let original = try emitValue(componentType, .accessIndex(base: vector.expression, index: lane), &statements, i)
            if let component {
                let select = IRExpressionKind.select(condition: matches, accept: component.expression, reject: original)
                inserted.append(try emitValue(componentType, select, &statements, i))
            } else if let previous = extracted {
                let select = IRExpressionKind.select(condition: matches, accept: original, reject: previous)
                if lane + 1 == UInt32(width.rawValue) {
                    try result(i, select, &statements)
                } else {
                    extracted = try emitValue(componentType, select, &statements, i)
                }
            }
        }
        if inserts { try result(i, .compose(type: resultType, components: inserted), &statements) }
    }

    mutating func lowerVectorShuffle(_ i: SPVInstruction, _ statements: inout [IRHandle<IRStatement>]) throws {
        try i.minimum(4)
        let type = try module.type(i.a[0], i)
        let left = try value(i.a[2], i, &statements)
        let right = try value(i.a[3], i, &statements)
        guard case .vector(let width, let scalar) = module.types[type.index].kind,
              case .vector(let leftWidth, let leftScalar) = try module.kind(left.type, i), leftScalar == scalar,
              case .vector(let rightWidth, let rightScalar) = try module.kind(right.type, i), rightScalar == scalar,
              i.a.count - 4 == Int(width.rawValue) else {
            throw i.fail("VectorShuffle operand types or selector count do not match the result")
        }
        let componentType = try indexedType(type, 0, i)
        let leftCount = UInt32(leftWidth.rawValue)
        let limit = leftCount + UInt32(rightWidth.rawValue)
        var components: [IRHandle<IRExpression>] = []
        for lane in i.a.dropFirst(4) {
            if lane == .max {
                components.append(try expr(componentType, .zeroValue(componentType), i))
            } else {
                guard lane < limit else { throw i.fail("VectorShuffle selector out of bounds") }
                let base = lane < leftCount ? left.expression : right.expression
                let index = lane < leftCount ? lane : lane - leftCount
                components.append(try emitValue(componentType, .accessIndex(base: base, index: index), &statements, i))
            }
        }
        try result(i, .compose(type: type, components: components), &statements)
    }

    mutating func lowerCompositeConstruct(_ i: SPVInstruction, _ statements: inout [IRHandle<IRStatement>]) throws {
        try i.minimum(2)
        let type = try module.type(i.a[0], i)
        let ids = i.a.dropFirst(2)
        var expressions: [IRHandle<IRExpression>] = []
        if case .vector(let width, let scalar) = module.types[type.index].kind {
            guard (2...Int(width.rawValue)).contains(ids.count) else { throw i.fail("Vector constructor arity") }
            var lanes = 0
            for id in ids {
                let value = try value(id, i, &statements)
                switch try module.kind(value.type, i) {
                case .scalar(let component) where component == scalar: lanes += 1
                case .vector(let size, let component) where component == scalar: lanes += Int(size.rawValue)
                default: throw i.fail("Vector constructor component type mismatch")
                }
                expressions.append(value.expression)
            }
            guard lanes == Int(width.rawValue) else { throw i.fail("Vector constructor lane count") }
        } else {
            let components = try componentTypes(type, i, expectedArity: ids.count)
            guard ids.count == components.count else { throw i.fail("CompositeConstruct arity") }
            for (id, expected) in zip(ids, components) {
                let value = try value(id, i, &statements)
                guard try module.type(value.type, i) == expected else { throw i.fail("CompositeConstruct type mismatch") }
                expressions.append(value.expression)
            }
        }
        try result(i, .compose(type: type, components: expressions), &statements)
    }

    mutating func lowerCompositeInsert(_ i: SPVInstruction, _ statements: inout [IRHandle<IRStatement>]) throws {
        try i.minimum(5)
        let type = try module.type(i.a[0], i)
        let source = try typed(i.a[3], i.a[0], i, &statements)
        let replacement = try value(i.a[2], i, &statements)
        let compose = try replacingComposite(source, type: type, at: i.a.dropFirst(4), with: replacement, i, &statements)
        try result(i, compose, &statements)
    }

    private mutating func replacingComposite(
        _ source: IRHandle<IRExpression>, type: IRHandle<IRType>, at path: ArraySlice<UInt32>, with replacement: SPVValue,
        _ i: SPVInstruction, _ statements: inout [IRHandle<IRStatement>]
    ) throws -> IRExpressionKind {
        try budget.enterDepth()
        defer { try? budget.leaveDepth() }
        guard let index = path.first else { throw i.fail("Missing CompositeInsert index") }
        let childType = try indexedType(type, index, i)
        let child: IRHandle<IRExpression>
        if path.count == 1 {
            guard try module.type(replacement.type, i) == childType else { throw i.fail("CompositeInsert type mismatch") }
            child = replacement.expression
        } else {
            let original = try emitValue(childType, .accessIndex(base: source, index: index), &statements, i)
            let compose = try replacingComposite(original, type: childType, at: path.dropFirst(), with: replacement, i, &statements)
            child = try emitValue(childType, compose, &statements, i)
        }
        let types = try componentTypes(type, i)
        var components: [IRHandle<IRExpression>] = []
        for (offset, componentType) in types.enumerated() {
            let offset = UInt32(offset)
            if offset == index { components.append(child) }
            else {
                components.append(try emitValue(componentType, .accessIndex(base: source, index: offset), &statements, i))
            }
        }
        return .compose(type: type, components: components)
    }
}
