import SAGAShaderIR

extension GLSLParser {
    func specializationConstant(_ type: IRHandle<IRType>, id: UInt32) throws {
        guard case .scalar(let scalar) = kind(type), [.bool, .i32, .u32, .f16, .f32].contains(scalar),
              text != "[" else {
            throw glslError(token, .typeMismatch, "Specialization constant requires a supported scalar type")
        }
        let name = try identifier()
        guard text != "[" else {
            throw glslError(token, .typeMismatch, "Specialization constant cannot be an array")
        }
        guard let id = UInt16(exactly: id) else {
            throw glslError(name, .invalidBinding, "Specialization constant ID must be in 0...65535")
        }
        guard !hasGlobalName(name.text), !name.text.hasPrefix("gl_") else {
            throw glslError(name, .duplicateName, "Duplicate/reserved global")
        }
        try expect("=")
        let initializerContext = GLSLFunctionContext(result: nil)
        context = initializerContext
        defer { context = nil }
        let root = try convertedInitializerExpression(type, at: name)
        for statement in initializerContext.statements {
            try budget.consume(.work)
            guard case .emit = statement.kind else {
                throw glslError(name, .unsupportedConstruct, "Unsupported specialization constant initializer")
            }
        }
        let initializer = try lowerGlobalConstant(root, at: name, allowsOverrides: true)
        try expect(";")
        try budget.consume(.nodes)
        overrideNames[name.text] = try IRHandle(index: overrides.count)
        overrides.append(.init(name: name.text, id: id, type: type, initializer: initializer, location: try name.location))
    }

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
        guard kind(type).isPipelineValueType else {
            throw glslError(name, .unsupportedConstruct, "Unsupported global constant type")
        }

        try expect("=")
        context = GLSLFunctionContext(result: nil)
        defer { context = nil }
        let initializer: IRHandle<IRExpression>
        if kind(type) == .scalar(.i32) || kind(type) == .scalar(.u32) {
            let value = try integerConstant(at: name, convertingTo: type)
            initializer = try appendConstantExpression(type, .literal(value.literal), at: name)
        } else {
            let root = try typedInitializer(type, at: name, globalLength: true)
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

    func typedInitializer(
        _ type: IRHandle<IRType>, at name: GLSLToken, globalLength: Bool = false
    ) throws -> GLSLValue {
        if try take("{") {
            try enter()
            defer { depth -= 1 }
            let count: Int
            let elementType: (Int) -> IRHandle<IRType>
            switch kind(type) {
            case .array(let element, .constant(let length), _):
                count = Int(length)
                elementType = { _ in element }
            case .vector(let size, let scalar):
                let element = try addType(.scalar(scalar))
                count = Int(size.rawValue)
                elementType = { _ in element }
            case .matrix(let columns, let rows, let scalar):
                let column = try addType(.vector(size: rows, scalar: scalar))
                count = Int(columns.rawValue)
                elementType = { _ in column }
            case .structure(let members, _):
                count = members.count
                elementType = { members[$0].type }
            default:
                throw glslError(name, .typeMismatch, "Initializer list requires a vector, matrix, fixed array, or structure")
            }
            return try initializerElements(type, count: count, terminator: "}", at: name) { index in
                try typedInitializer(elementType(index), at: name, globalLength: globalLength)
            }
        }
        if globalLength, case .array = kind(type),
           tokens.indices.contains(i + 1), tokens[i + 1].text == "[", let base = try resolveType(token) {
            let constructor = try advance()
            guard try arraySuffix(base, globalLength: true) == type else {
                throw glslError(constructor, .typeMismatch, "Array constructor type does not match the declaration")
            }
            return try arrayConstructor(type, at: constructor)
        }
        return try convertedInitializerExpression(type, at: name)
    }

    func arrayConstructor(_ type: IRHandle<IRType>, at name: GLSLToken) throws -> GLSLValue {
        guard case .array(let element, .constant(let count), _) = kind(type) else {
            throw glslError(name, .typeMismatch, "Array constructor requires a fixed array type")
        }
        try expect("(")
        return try initializerElements(type, count: Int(count), terminator: ")", at: name) { _ in
            try convertedInitializerExpression(element, at: name)
        }
    }

    private func convertedInitializerExpression(_ type: IRHandle<IRType>, at name: GLSLToken) throws -> GLSLValue {
        let source = try rvalue(expression(), at: name)
        guard let value = try implicitlyConvert(source, to: type, at: name) else {
            throw glslError(name, .typeMismatch, "Initializer type mismatch")
        }
        return value
    }

    private func initializerElements(
        _ type: IRHandle<IRType>, count: Int, terminator: String, at name: GLSLToken,
        parseElement: (Int) throws -> GLSLValue
    ) throws -> GLSLValue {
        var elements: [IRHandle<IRExpression>] = []
        if text != terminator {
            repeat {
                guard elements.count < count else {
                    throw glslError(token, .typeMismatch, "Too many initializer elements")
                }
                let value = try parseElement(elements.count)
                elements.append(value.expression)
                if !(try take(",")) || terminator == "}" && text == "}" { break }
            } while true
        }
        try expect(terminator)
        guard elements.count == count else {
            throw glslError(name, .typeMismatch, "Initializer element count does not match its type")
        }
        return try expressionNode(type, .compose(type: type, components: elements), at: name)
    }

    func evaluateGlobalConstant(_ root: GLSLValue, at name: GLSLToken) throws -> WGSLPipelineValue {
        let start = globalExpressions.count
        defer { globalExpressions.removeSubrange(start...) }
        let result = try lowerGlobalConstant(root, at: name, allowsOverrides: false)
        do {
            return try evaluateWGSLPipelineValue(result, in: IRModule(
                types: .init(types), constants: .init(constants), globalExpressions: .init(globalExpressions)),
                overrides: [:], budget: &budget)
        } catch ShaderFailure.invalidHandle {
            throw glslError(name, .unsupportedConstruct, "Unsupported global constant expression")
        }
    }

    private func lowerGlobalConstant(
        _ root: GLSLValue, at name: GLSLToken, allowsOverrides: Bool
    ) throws -> IRHandle<IRExpression> {
        let expressions = try requireContext().expressions
        var pending = [root.expression]
        var visited = Set(pending)
        while let handle = pending.popLast() {
            try budget.consume(.work)
            let expression = expressions[handle.index]
            guard kind(expression.type).isPipelineValueType, let operands = expression.kind.pipelineOperands else {
                throw glslError(name, .unsupportedConstruct, "Global initializer requires a constant expression")
            }
            if case .override = expression.kind, !allowsOverrides {
                throw glslError(name, .unsupportedConstruct, "Ordinary global constants cannot depend on specialization constants")
            }
            for operand in operands where visited.insert(operand).inserted { pending.append(operand) }
        }
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
        return result
    }

    private func materializeGlobalConstant(
        _ value: WGSLPipelineValue, type: IRHandle<IRType>, at name: GLSLToken
    ) throws -> IRHandle<IRExpression> {
        try enter()
        defer { depth -= 1 }
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
        case (.composite(let values), .structure(let members, _)) where values.count == members.count:
            let children = try zip(values, members).map { value, member in
                try materializeGlobalConstant(value, type: member.type, at: name)
            }
            return try appendConstantExpression(type, .compose(type: type, components: children), at: name)
        case (.matrix(let values), .matrix(let columns, let rows, let scalar)) where values.count == Int(columns.rawValue):
            let column = try addType(.vector(size: rows, scalar: scalar))
            let children = try values.map { try materializeGlobalConstant(.vector($0), type: column, at: name) }
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
