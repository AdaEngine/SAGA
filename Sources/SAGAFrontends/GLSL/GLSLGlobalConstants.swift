import SAGAShaderIR

extension GLSLParser {
    func globalConstant(_ base: IRHandle<IRType>) throws {
        let prefix = try arraySuffix(base)
        let name = try identifier()
        guard !hasGlobalName(name.text), !name.text.hasPrefix("gl_") else {
            throw glslError(name, .duplicateName, "Duplicate/reserved global")
        }
        guard prefix == base || text != "[" else {
            throw glslError(name, .unsupportedConstruct, "Multidimensional constant arrays are not implemented")
        }
        let type = try arraySuffix(prefix)
        let element: IRHandle<IRType>
        if case .array(let child, .constant, _) = kind(type) { element = child }
        else { element = type }
        guard let shape = IRBitcastShape(kind(element)), [.bool, .i32, .u32, .f16, .f32].contains(shape.scalar) else {
            throw glslError(name, .unsupportedConstruct, "Expected a scalar, vector, or fixed array constant type")
        }

        try expect("=")
        context = GLSLFunctionContext(result: nil)
        defer { context = nil }
        let initializer: IRHandle<IRExpression>
        if kind(type) == .scalar(.i32) || kind(type) == .scalar(.u32) {
            let value = try integerConstant(at: name, convertingTo: type)
            initializer = try appendConstantExpression(type, .literal(value.literal), at: name)
        } else {
            let root = try constantInitializer(type, at: name)
            let value = try evaluateGlobalConstant(root, at: name)
            initializer = try materializeGlobalConstant(value, type: type, at: name)
        }
        try expect(";")
        try budget.consume(.nodes)
        constantNames[name.text] = try IRHandle(index: constants.count)
        constants.append(.init(name: name.text, type: type, initializer: initializer, location: try name.location))
    }

    func globalConstantLiteral(_ handle: IRHandle<IRConstant>) -> IRLiteral? {
        let initializer = constants[handle.index].initializer
        if case .literal(let literal) = globalExpressions[initializer.index].kind { return literal }
        return nil
    }

    private func constantInitializer(_ type: IRHandle<IRType>, at name: GLSLToken) throws -> GLSLValue {
        try enter()
        defer { depth -= 1 }
        if try take("{") {
            switch kind(type) {
            case .array(let element, .constant(let count), _):
                return try constantElements(type, element: element, count: Int(count), terminator: "}", at: name)
            case .vector(let size, let scalar):
                return try constantElements(type, element: addType(.scalar(scalar)), count: Int(size.rawValue),
                                            terminator: "}", at: name)
            default:
                throw glslError(name, .typeMismatch, "Initializer list requires a vector or fixed array")
            }
        }
        if case .array(let element, .constant(let count), _) = kind(type),
           tokens.indices.contains(i + 1), tokens[i + 1].text == "[", let base = try resolveType(token) {
            let constructor = try advance()
            guard try arraySuffix(base, globalLength: true) == type else {
                throw glslError(constructor, .typeMismatch, "Array constructor type does not match the declaration")
            }
            try expect("(")
            return try constantElements(type, element: element, count: Int(count), terminator: ")", at: constructor)
        }
        return try convertedConstantExpression(type, at: name)
    }

    private func convertedConstantExpression(_ type: IRHandle<IRType>, at name: GLSLToken) throws -> GLSLValue {
        let source = try rvalue(expression(), at: name)
        guard let value = try implicitlyConvert(source, to: type, at: name) else {
            throw glslError(name, .typeMismatch, "Constant initializer type mismatch")
        }
        return value
    }

    private func constantElements(
        _ type: IRHandle<IRType>, element: IRHandle<IRType>, count: Int, terminator: String, at name: GLSLToken
    ) throws -> GLSLValue {
        var elements: [IRHandle<IRExpression>] = []
        if text != terminator {
            repeat {
                guard elements.count < count else {
                    throw glslError(token, .typeMismatch, "Too many elements in constant initializer")
                }
                let value = try terminator == "}" ? constantInitializer(element, at: name)
                                                 : convertedConstantExpression(element, at: name)
                elements.append(value.expression)
                if !(try take(",")) || terminator == "}" && text == "}" { break }
            } while true
        }
        try expect(terminator)
        guard elements.count == count else {
            throw glslError(name, .typeMismatch, "Constant initializer element count does not match the declaration")
        }
        return try expressionNode(type, .compose(type: type, components: elements), at: name)
    }

    private func evaluateGlobalConstant(_ root: GLSLValue, at name: GLSLToken) throws -> WGSLPipelineValue {
        let expressions = try requireContext().expressions
        var pending = [root.expression]
        var visited = Set(pending)
        while let handle = pending.popLast() {
            try budget.consume(.work)
            let expression = expressions[handle.index]
            guard kind(expression.type).isPipelineValueType, let operands = expression.kind.pipelineOperands else {
                throw glslError(name, .unsupportedConstruct, "Global initializer requires a constant expression")
            }
            for operand in operands where visited.insert(operand).inserted { pending.append(operand) }
        }
        let start = globalExpressions.count
        defer { globalExpressions.removeSubrange(start...) }
        var remap: [IRHandle<IRExpression>: IRHandle<IRExpression>] = [:]
        for handle in visited.sorted(by: { $0.index < $1.index }) {
            let expression = expressions[handle.index]
            guard let operation = try expression.kind.remappingPipelineOperands({ operand in
                guard let mapped = remap[operand] else { throw ShaderFailure.invalidHandle }
                return mapped
            }) else { throw ShaderFailure.invalidHandle }
            remap[handle] = try appendConstantExpression(expression.type, operation, at: name)
        }
        guard let result = remap[root.expression] else { throw ShaderFailure.invalidHandle }
        do {
            return try evaluateWGSLPipelineValue(result, in: IRModule(
                types: .init(types), constants: .init(constants), globalExpressions: .init(globalExpressions)),
                overrides: [:], budget: &budget)
        } catch ShaderFailure.invalidHandle {
            throw glslError(name, .unsupportedConstruct, "Unsupported global constant expression")
        }
    }

    private func materializeGlobalConstant(
        _ value: WGSLPipelineValue, type: IRHandle<IRType>, at name: GLSLToken
    ) throws -> IRHandle<IRExpression> {
        switch (value, kind(type)) {
        case (.scalar(let literal), .scalar):
            return try appendConstantExpression(type, .literal(literal), at: name)
        case (.vector(let lanes), .vector(let size, let scalar)) where lanes.count == Int(size.rawValue):
            let component = try addType(.scalar(scalar))
            let children = try lanes.map { try appendConstantExpression(component, .literal($0), at: name) }
            return try appendConstantExpression(type, .compose(type: type, components: children), at: name)
        case (.composite(let values), .array(let element, .constant(let count), _)) where values.count == Int(count):
            let children = try values.map { try materializeGlobalConstant($0, type: element, at: name) }
            return try appendConstantExpression(type, .compose(type: type, components: children), at: name)
        default:
            throw ShaderFailure.invalidHandle
        }
    }

    private func appendConstantExpression(
        _ type: IRHandle<IRType>, _ operation: IRExpressionKind, at name: GLSLToken
    ) throws -> IRHandle<IRExpression> {
        try budget.consume(.nodes)
        let handle = try IRHandle<IRExpression>(index: globalExpressions.count)
        globalExpressions.append(.init(type: type, kind: operation, location: try name.location))
        return handle
    }
}
