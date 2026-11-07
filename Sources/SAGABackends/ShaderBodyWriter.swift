import SAGAShaderIR

struct ShaderStorage {
    let reference: String
    let count: UInt32?
    let writable: Bool
}

private struct ShaderAddress {
    let reference: String
    let boundsCheck: String?
    let writable: Bool
}

enum ShaderResultTarget {
    case direct
    case metalStructure(String)
    case glslPosition
    case glslColor
    case glslVertexStructure(String, destinations: [String])
}

struct ShaderBodyWriter {
    let module: IRModule
    let function: IRFunction
    let arguments: [String]
    let storage: [ShaderStorage]
    let outputFormat: ShaderOutputFormat
    let resultTarget: ShaderResultTarget?
    let structures: [IRHandle<IRType>: ShaderStructure]
    let argumentFields: [Int: [String]]
    let callCaptures: [Int: [String]]
    private(set) var budget: ShaderBudget
    private(set) var integerHelpers: Set<IntegerShaderHelper> = []
    private(set) var integerVectorHelpers: Set<IntegerVectorHelper> = []
    private var temporaries: Set<IRHandle<IRExpression>> = []
    private var temporaryOrder: [IRHandle<IRExpression>] = []
    private var indentation = "    "

    init(module: IRModule, function: IRFunction, arguments: [String], storage: [ShaderStorage],
         outputFormat: ShaderOutputFormat, resultTarget: ShaderResultTarget? = nil,
         structures: [IRHandle<IRType>: ShaderStructure] = [:], argumentFields: [Int: [String]] = [:],
         callCaptures: [Int: [String]] = [:], budget: ShaderBudget = ShaderBudget()) {
        self.module = module
        self.function = function
        self.arguments = arguments
        self.storage = storage
        self.outputFormat = outputFormat
        self.resultTarget = resultTarget
        self.structures = structures
        self.argumentFields = argumentFields
        self.callCaptures = callCaptures
        self.budget = budget
    }

    mutating func write() throws -> [String] {
        var lines: [String] = []
        for (index, local) in function.localVariables.values.enumerated() {
            let type = try typeName(local.type)
            let initial = try local.initializer.map { try value($0) } ?? zeroValue(local.type)
            try append("\(declaration(type, named: localName(index))) = \(initial);", to: &lines)
        }
        try writeBlock(function.body, nested: false, to: &lines)
        return lines
    }

    private mutating func writeBlock(
        _ block: IRHandle<IRBlock>, nested: Bool = true, breakIf: IRHandle<IRExpression>? = nil,
        to lines: inout [String]
    ) throws {
        try budget.enterDepth()
        let mark = temporaryOrder.count
        if nested { indentation += "    " }
        defer {
            for temporary in temporaryOrder[mark...] { temporaries.remove(temporary) }
            temporaryOrder.removeSubrange(mark...)
            if nested { indentation.removeLast(4) }
            try? budget.leaveDepth()
        }
        for handle in try function.blocks.value(for: block).statements {
            try budget.consume(.work)
            switch try function.statements.value(for: handle).kind {
            case .emit(let handles):
                for expression in handles { try emit(expression, to: &lines) }
            case .call(let callee, let args, let result):
                try writeCall(callee, arguments: args, result: result, to: &lines)
            case .returnValue(nil):
                try append("return;", to: &lines)
            case .returnValue(let result?):
                let value = try value(result)
                try writeResult(value, to: &lines)
            case .break:
                try append("break;", to: &lines)
            case .continue:
                try append("continue;", to: &lines)
            case .block(let child):
                try append("{", to: &lines)
                try writeBlock(child, to: &lines)
                try append("}", to: &lines)
            case .ifCondition(let condition, let accept, let reject):
                let test = try value(condition, parenthesized: false)
                try append("if (\(test)) {", to: &lines)
                try writeBlock(accept, to: &lines)
                if !(try function.blocks.value(for: reject).statements.isEmpty) {
                    try append("} else {", to: &lines)
                    try writeBlock(reject, to: &lines)
                }
                try append("}", to: &lines)
            case .loop(let body, let continuing, let condition):
                try writeLoop(body, continuing: continuing, breakIf: condition,
                              index: handle.index, to: &lines)
            case .store(let pointer, let expression):
                let target = try address(pointer)
                guard target.writable else { throw unsupported("store to read-only storage") }
                let rhs = try value(expression)
                try assign(target.reference, rhs, when: target.boundsCheck, to: &lines)
            default: throw unsupported("statement")
            }
        }
        if let breakIf {
            let test = try value(breakIf, parenthesized: false)
            try append("if (\(test)) { break; }", to: &lines)
        }
    }

    private mutating func writeCall(
        _ callee: IRHandle<IRFunction>, arguments: [IRHandle<IRExpression>],
        result: IRHandle<IRExpression>?, to lines: inout [String]
    ) throws {
        guard let captures = callCaptures[callee.index] else { throw unsupported("unplanned helper call") }
        let arguments = try arguments.map { try value($0) } + captures
        let call = "saga_fn_\(callee.index)(\(arguments.joined(separator: ", ")))"
        if let result {
            let expression = try function.expressions.value(for: result)
            let type = try typeName(expression.type)
            let name = "saga_call_\(result.index)"
            let declaration = outputFormat == .wgsl ? "let \(name): \(type)" : "\(type) \(name)"
            try append("\(declaration) = \(call);", to: &lines)
            remember(result)
        } else {
            try append("\(call);", to: &lines)
        }
    }

    private mutating func remember(_ handle: IRHandle<IRExpression>) {
        temporaries.insert(handle)
        temporaryOrder.append(handle)
    }

    private mutating func writeResult(_ value: String, to lines: inout [String]) throws {
        switch resultTarget {
        case .direct:
            try append("return \(value);", to: &lines)
        case .metalStructure(let type):
            try append("return \(type){\(value)};", to: &lines)
        case .glslPosition:
            try append("gl_Position = \(value);", to: &lines)
            try finishGLSLVertex(to: &lines)
        case .glslColor:
            try append("saga_color = \(value);", to: &lines)
            try append("return;", to: &lines)
        case .glslVertexStructure(let type, let destinations):
            try append("\(type) saga_result = \(value);", to: &lines)
            for (index, destination) in destinations.enumerated() {
                try append("\(destination) = saga_result.saga_member_\(index);", to: &lines)
            }
            try finishGLSLVertex(to: &lines)
        case nil:
            throw unsupported("unexpected return value")
        }
    }

    private mutating func finishGLSLVertex(to lines: inout [String]) throws {
        try append("gl_Position.y = -gl_Position.y;", to: &lines)
        try append("gl_Position.z = gl_Position.z * 2.0 - gl_Position.w;", to: &lines)
        try append("return;", to: &lines)
    }

    private mutating func writeLoop(
        _ body: IRHandle<IRBlock>, continuing: IRHandle<IRBlock>, breakIf: IRHandle<IRExpression>?,
        index: Int, to lines: inout [String]
    ) throws {
        let continuingBlock = try function.blocks.value(for: continuing)
        let hasContinuing = !continuingBlock.statements.isEmpty || breakIf != nil
        let first = "saga_first_\(index)"
        if hasContinuing {
            try append("\(declaration("bool", named: first)) = true;", to: &lines)
        }
        try append("while (true) {", to: &lines)
        do {
            indentation += "    "
            defer { indentation.removeLast(4) }
            if hasContinuing {
                // A native continue starts the next iteration, so run WGSL continuing at its head.
                try append("if (!\(first)) {", to: &lines)
                try writeBlock(continuing, breakIf: breakIf, to: &lines)
                try append("}", to: &lines)
                try append("\(first) = false;", to: &lines)
            }
            try writeBlock(body, nested: false, to: &lines)
        }
        try append("}", to: &lines)
    }

    private mutating func emit(_ handle: IRHandle<IRExpression>, to lines: inout [String]) throws {
        let expression = try function.expressions.value(for: handle)
        switch expression.kind {
        case .load(let pointer):
            // Loads are evaluated at their IR emit, before any later writes.
            let source = try address(pointer)
            let name = "saga_load_\(handle.index)"
            let type = try typeName(expression.type)
            let declaration = declaration(type, named: name)
            if let condition = source.boundsCheck {
                let fallback = try zeroValue(expression.type)
                try append("\(declaration) = \(fallback);", to: &lines)
                try assign(name, source.reference, when: condition, to: &lines)
            } else {
                try append("\(declaration) = \(source.reference);", to: &lines)
            }
            remember(handle)
        case .access, .accessIndex:
            if case .pointer = try module.types.value(for: expression.type).kind {
                _ = try address(handle)
            } else {
                _ = try value(handle)
            }
        default:
            _ = try value(handle)
        }
    }

    private mutating func assign(
        _ destination: String, _ value: String, when condition: String?, to lines: inout [String]
    ) throws {
        let assignment = "\(destination) = \(value);"
        if let condition {
            try append("if (\(condition)) { \(assignment) }", to: &lines)
        } else {
            try append(assignment, to: &lines)
        }
    }

    private mutating func address(_ handle: IRHandle<IRExpression>) throws -> ShaderAddress {
        try budget.consume(.work)
        try budget.enterDepth()
        defer { try? budget.leaveDepth() }
        let pointer = try function.expressions.value(for: handle)
        if case .localVariable(let local) = pointer.kind {
            _ = try function.localVariables.value(for: local)
            return .init(reference: localName(local.index), boundsCheck: nil, writable: true)
        }
        if case .accessIndex(let base, let index) = pointer.kind {
            let source = try function.expressions.value(for: base)
            if case .pointer(let pointee, .function) = try module.types.value(for: source.type).kind {
                let parent = try address(base)
                let field = try member(index, of: pointee)
                return .init(reference: parent.reference + field,
                             boundsCheck: parent.boundsCheck, writable: parent.writable)
            }
        }
        let base: IRExpressionKind
        let index: String?
        switch pointer.kind {
        case .access(let array, let element):
            base = try function.expressions.value(for: array).kind
            index = try value(element)
        case .accessIndex(let array, let element):
            base = try function.expressions.value(for: array).kind
            index = "\(element)u"
        default:
            base = pointer.kind
            index = nil
        }
        guard case .globalVariable(let global) = base, storage.indices.contains(global.index) else {
            throw unsupported("storage address")
        }
        let binding = storage[global.index]
        switch (index, binding.count) {
        case (.some(let index), .some(let count)):
            return .init(reference: "\(binding.reference)[\(index)]",
                         boundsCheck: "\(index) < \(count)u", writable: binding.writable)
        case (nil, nil):
            return .init(reference: binding.reference, boundsCheck: nil, writable: binding.writable)
        default:
            throw unsupported("storage access must name a scalar or an indexed array element")
        }
    }

    private mutating func append(_ line: String, to lines: inout [String]) throws {
        try budget.consume(.sourceBytes, amount: indentation.utf8.count + line.utf8.count + 1)
        lines.append(indentation + line)
    }

    private func scalar(_ handle: IRHandle<IRType>) throws -> IRScalar {
        guard case .scalar(let type) = try module.types.value(for: handle).kind,
              type == .u32 || type == .i32 || type == .bool || type == .f32
        else { throw unsupported("expected u32, i32, f32 or bool scalar") }
        return type
    }

    private func localName(_ index: Int) -> String { "saga_local_\(index)" }
    private func scalarZero(_ type: IRScalar) -> String {
        if type == .bool { return "false" }
        if type == .f32 { return floatLiteral(0) }
        if type == .i32 { return outputFormat == .wgsl ? "0i" : "0" }
        return "0u"
    }

    private mutating func zeroValue(_ type: IRHandle<IRType>) throws -> String {
        try budget.consume(.work)
        try budget.enterDepth()
        defer { try? budget.leaveDepth() }
        switch try module.types.value(for: type).kind {
        case .scalar: return try scalarZero(scalar(type))
        case .vector(let size, let scalar) where [.f32, .i32, .u32, .bool].contains(scalar):
            return try construct(type, components: Array(repeating: scalarZero(scalar), count: Int(size.rawValue)))
        case .structure(let members, _) where structures[type] != nil:
            let components = try members.map { try zeroValue($0.type) }
            return try construct(type, components: components)
        default: throw unsupported("zero value for unsupported type")
        }
    }

    private func floatLiteral(_ value: Float) -> String {
        outputFormat == .glsl ? String(value) : "\(value)f"
    }

    private func declaration(_ type: String, named name: String) -> String {
        outputFormat == .wgsl ? "var \(name): \(type)" : "\(type) \(name)"
    }

    private func typeName(_ type: IRHandle<IRType>) throws -> String {
        try shaderValueType(type, module: module, format: outputFormat, structures: structures)
    }

    private mutating func construct(_ type: IRHandle<IRType>, components: [String]) throws -> String {
        let constructor = try structures[type]?.constructor ?? typeName(type)
        let aggregate = structures[type] != nil && outputFormat == .msl
        try budget.consume(.sourceBytes, amount: constructor.utf8.count + components.count * 2 + 2)
        let values = components.joined(separator: ", ")
        return aggregate ? "\(constructor){\(values)}" : "\(constructor)(\(values))"
    }

    private func member(_ index: UInt32, of type: IRHandle<IRType>) throws -> String {
        if case .vector(let width, let scalar) = try module.types.value(for: type).kind,
           [.f32, .u32, .i32, .bool].contains(scalar), index < UInt32(width.rawValue) {
            return [".x", ".y", ".z", ".w"][Int(index)]
        }
        if structures[type] != nil { return ".saga_member_\(index)" }
        throw unsupported("access outside a supported vector or render structure")
    }

    private mutating func value(
        _ handle: IRHandle<IRExpression>, parenthesized: Bool = true
    ) throws -> String {
        try budget.consume(.work)
        try budget.enterDepth()
        defer { try? budget.leaveDepth() }
        let expression = try function.expressions.value(for: handle)
        switch expression.kind {
        case .literal(.u32(let number)): return "\(number)u"
        case .literal(.i32(let number)):
            if number == .min { return integerBits("2147483648u", signed: true, in: outputFormat) }
            let literal = outputFormat == .wgsl ? "\(number)i" : String(number)
            return number < 0 ? "(\(literal))" : literal
        case .literal(.f32(let number)) where number.isFinite: return floatLiteral(number)
        case .literal(.bool(let flag)): return flag ? "true" : "false"
        case .zeroValue(let type): return try zeroValue(type)
        case .load where temporaries.contains(handle): return "saga_load_\(handle.index)"
        case .callResult where temporaries.contains(handle): return "saga_call_\(handle.index)"
        case .functionArgument(let argument):
            guard let index = Int(exactly: argument), arguments.indices.contains(index) else {
                throw ShaderFailure.invalidHandle
            }
            guard !arguments[index].isEmpty else { throw unsupported("whole structure input value") }
            return arguments[index]
        case .compose(let type, let components):
            // Charge before expansion so an expression shared by several constructors stays bounded.
            try budget.consume(.sourceBytes, amount: components.count + 1)
            let values = try components.map { try value($0) }
            return try construct(type, components: values)
        case .splat(let size, let operand):
            let lane = try value(operand)
            if outputFormat == .hlsl {
                return "(\(lane)).\(String(repeating: "x", count: Int(size.rawValue)))"
            }
            return try construct(expression.type, components: [lane])
        case .swizzle(let size, let vector, let pattern):
            let suffix = [pattern.x, pattern.y, pattern.z, pattern.w]
                .prefix(Int(size.rawValue)).map(\.rawValue).joined()
            return try value(vector) + "." + suffix
        case .accessIndex(let base, let index):
            let source = try function.expressions.value(for: base)
            if case .functionArgument(let argument) = source.kind, let slot = Int(exactly: argument),
               let fields = argumentFields[slot], let field = Int(exactly: index), fields.indices.contains(field) {
                return fields[field]
            }
            let field = try member(index, of: source.type)
            return try value(base) + field
        case .cast(let operand, let target):
            let operandType = try function.expressions.value(for: operand).type
            if case .vector(let size, let input) = try module.types.value(for: operandType).kind {
                guard (input == .i32 || input == .u32) && (target == .i32 || target == .u32) else {
                    throw unsupported("vector cast outside the i32/u32 profile")
                }
                let source = try value(operand)
                return input == target ? source :
                    integerBits(source, signed: target == .i32, width: size.rawValue, in: outputFormat)
            }
            let input = try scalar(operandType)
            guard (input == .i32 || input == .u32) && (target == .i32 || target == .u32) else {
                throw unsupported("cast outside the i32/u32 profile")
            }
            let source = try value(operand)
            return input == target ? source : integerBits(source, signed: target == .i32, in: outputFormat)
        case .unary(let op, let argument):
            if case .vector(let size, let lane) = try module.types.value(for: expression.type).kind,
               lane == .i32 || lane == .u32 {
                let operand = try value(argument)
                if op == .bitwiseNot { return "(~\(operand))" }
                guard op == .negate && lane == .i32 else { throw unsupported("integer vector unary operation") }
                let helper = IntegerVectorHelper(operation: .negate, signed: true, width: size.rawValue)
                integerVectorHelpers.insert(helper)
                return "\(helper.name)(\(operand))"
            }
            let type = try scalar(expression.type)
            if op == .negate && type == .i32 {
                return signedNegation(try value(argument), in: outputFormat)
            }
            let symbol: String
            switch op {
            case .logicalNot where type == .bool: symbol = "!"
            case .bitwiseNot where type == .u32 || type == .i32: symbol = "~"
            case .negate where type == .f32 || type == .i32: symbol = "-"
            default: throw unsupported("unary operation \(op.rawValue)")
            }
            try budget.consume(.sourceBytes, amount: 3)
            let unary = "\(symbol)\(try value(argument))"
            return parenthesized ? "(\(unary))" : unary
        case .binary(let op, let left, let right):
            let operandType = try function.expressions.value(for: left).type
            if case .vector(let size, let lane) = try module.types.value(for: operandType).kind,
               lane == .i32 || lane == .u32 {
                let helper = IntegerVectorHelper(operation: .binary(op), signed: lane == .i32, width: size.rawValue)
                try budget.consume(.sourceBytes, amount: helper.name.utf8.count + 4)
                let a = try value(left)
                let b = try value(right)
                integerVectorHelpers.insert(helper)
                return "\(helper.name)(\(a), \(b))"
            }
            let type = try scalar(operandType)
            let shift = op == .shiftLeft || op == .shiftRight
            let rightType = try scalar(function.expressions.value(for: right).type)
            guard rightType == (shift ? .u32 : type) else {
                throw unsupported("binary operand types")
            }
            // Charge each expansion before traversing children so shared IR cannot inflate output unchecked.
            try budget.consume(.sourceBytes, amount: 8)
            let lhs = try value(left)
            let rhs = try value(right)
            if let lowered = try integerBinary(op, scalar: type, left: lhs, right: rhs,
                                               in: outputFormat, helpers: &integerHelpers) {
                return lowered
            }
            let symbol = try shaderBinarySymbol(op, type: type)
            let binary = "\(lhs) \(symbol) \(rhs)"
            return parenthesized ? "(\(binary))" : binary
        default: throw unsupported("expression")
        }
    }

}

func shaderBinarySymbol(_ op: IRBinaryOperator, type: IRScalar) throws -> String {
    switch (op, type) {
    case (.equal, _): "=="
    case (.notEqual, _): "!="
    case (.logicalAnd, .bool): "&&"
    case (.logicalOr, .bool): "||"
    case (.add, .u32), (.add, .i32), (.add, .f32): "+"
    case (.subtract, .u32), (.subtract, .i32), (.subtract, .f32): "-"
    case (.multiply, .u32), (.multiply, .i32), (.multiply, .f32): "*"
    case (.less, .u32), (.less, .i32), (.less, .f32): "<"
    case (.lessEqual, .u32), (.lessEqual, .i32), (.lessEqual, .f32): "<="
    case (.greater, .u32), (.greater, .i32), (.greater, .f32): ">"
    case (.greaterEqual, .u32), (.greaterEqual, .i32), (.greaterEqual, .f32): ">="
    case (.and, .u32), (.and, .i32): "&"
    case (.inclusiveOr, .u32), (.inclusiveOr, .i32): "|"
    case (.exclusiveOr, .u32), (.exclusiveOr, .i32): "^"
    default: throw unsupported("binary operation \(op.rawValue)")
    }
}

private func unsupported(_ reason: String) -> ShaderFailure {
    .unsupported("Shader body code generation: \(reason)")
}
