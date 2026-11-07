import SAGAShaderIR

struct ShaderStructure {
    let typeName: String
    let constructor: String
}

struct ShaderParameter {
    let type: String
    let name: String
}

/// Emits reachable helpers in IR order. Validation guarantees that every callee precedes its caller.
struct ShaderFunctions {
    private let order: [Int]
    private let parameters: [Int: [ShaderParameter]]
    private(set) var budget: ShaderBudget
    private(set) var integerHelpers: Set<IntegerShaderHelper> = []
    private(set) var integerVectorHelpers: Set<IntegerVectorHelper> = []

    init(module: IRModule, entry: IRHandle<IRFunction>, metalStorage: [ShaderParameter] = []) throws {
        var budget = ShaderBudget()
        var pending = [entry]
        var seen = Set([entry.index])
        var calls: [Int: Set<Int>] = [:]
        var captures: [Int: Set<Int>] = [:]
        while let handle = pending.popLast() {
            try budget.consume(.work)
            let function = try module.functions.value(for: handle)
            for statement in function.statements.values {
                try budget.consume(.work)
                guard case .call(let callee, _, _) = statement.kind else { continue }
                calls[handle.index, default: []].insert(callee.index)
                if seen.insert(callee.index).inserted { pending.append(callee) }
            }
            if !metalStorage.isEmpty {
                for expression in function.expressions.values {
                    try budget.consume(.work)
                    if case .globalVariable(let global) = expression.kind {
                        guard metalStorage.indices.contains(global.index) else { throw ShaderFailure.invalidHandle }
                        captures[handle.index, default: []].insert(global.index)
                    }
                }
            }
        }
        seen.remove(entry.index)
        let order = seen.sorted()
        var parameters: [Int: [ShaderParameter]] = [:]
        for index in order {
            try budget.consume(.work)
            var bindings = captures[index] ?? []
            for callee in calls[index] ?? [] {
                try budget.consume(.work)
                bindings.formUnion(captures[callee] ?? [])
            }
            captures[index] = bindings
            parameters[index] = bindings.sorted().map { metalStorage[$0] }
        }
        self.order = order
        self.parameters = parameters
        self.budget = budget
    }

    var callCaptures: [Int: [String]] { parameters.mapValues { $0.map(\.name) } }

    mutating func declarations(
        module: IRModule, format: ShaderOutputFormat, storage: [ShaderStorage],
        structures: [IRHandle<IRType>: ShaderStructure] = [:]
    ) throws -> [String] {
        var lines: [String] = []
        let calls = callCaptures
        for index in order {
            let handle = try module.functions.handle(at: index)
            let function = try module.functions.value(for: handle)
            let explicit = try function.arguments.enumerated().map { slot, argument in
                try budget.consume(.work)
                let type = try shaderValueType(argument.type, module: module, format: format, structures: structures)
                return ShaderParameter(type: type, name: "saga_param_\(slot)")
            }
            let resultType = try function.result.map {
                try shaderValueType($0.type, module: module, format: format, structures: structures)
            }
            let all = explicit + (parameters[index] ?? [])
            let name = "saga_fn_\(index)"
            let signature: String
            if format == .wgsl {
                let args = all.map { "\($0.name): \($0.type)" }.joined(separator: ", ")
                let result = resultType.map { " -> \($0)" } ?? ""
                signature = "fn \(name)(\(args))\(result) {"
            } else {
                let args = all.map { "\($0.type) \($0.name)" }.joined(separator: ", ")
                signature = "\(resultType ?? "void") \(name)(\(args)) {"
            }
            try budget.consume(.sourceBytes, amount: signature.utf8.count + 3)
            var writer = ShaderBodyWriter(module: module, function: function, arguments: explicit.map(\.name),
                                          storage: storage, outputFormat: format, resultTarget: .direct,
                                          structures: structures, callCaptures: calls, budget: budget)
            lines.append(signature)
            lines += try writer.write()
            lines.append("}")
            budget = writer.budget
            integerHelpers.formUnion(writer.integerHelpers)
            integerVectorHelpers.formUnion(writer.integerVectorHelpers)
        }
        return lines
    }
}

func shaderValueType(
    _ handle: IRHandle<IRType>, module: IRModule, format: ShaderOutputFormat,
    structures: [IRHandle<IRType>: ShaderStructure]
) throws -> String {
    if let structure = structures[handle] { return structure.typeName }
    switch try module.types.value(for: handle).kind {
    case .scalar(.bool): return "bool"
    case .scalar(.u32): return format == .wgsl ? "u32" : "uint"
    case .scalar(.i32): return format == .wgsl ? "i32" : "int"
    case .scalar(.f32): return floatType(1, in: format)
    case .vector(let size, let scalar): return try shaderVectorType(size.rawValue, scalar: scalar, in: format)
    default: throw ShaderFailure.unsupported("Shader value type")
    }
}

func shaderVectorType(_ width: UInt8, scalar: IRScalar, in format: ShaderOutputFormat) throws -> String {
    if scalar == .f32 { return floatType(width, in: format) }
    let native: String
    let glsl: String
    let wgsl: String
    switch scalar {
    case .i32: (native, glsl, wgsl) = ("int", "ivec", "i32")
    case .u32: (native, glsl, wgsl) = ("uint", "uvec", "u32")
    case .bool: (native, glsl, wgsl) = ("bool", "bvec", "bool")
    default: throw ShaderFailure.unsupported("Shader vector lane type")
    }
    switch format {
    case .wgsl: return "vec\(width)<\(wgsl)>"
    case .glsl: return "\(glsl)\(width)"
    default: return "\(native)\(width)"
    }
}
