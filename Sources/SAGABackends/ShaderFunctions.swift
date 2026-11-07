import SAGAShaderIR

struct ShaderStructure {
    let typeName: String
    let constructor: String

    func declarations(memberTypes: [String], in format: ShaderOutputFormat) -> [String] {
        var lines = ["struct \(typeName) {"]
        for (index, type) in memberTypes.enumerated() {
            let name = "saga_member_\(index)"
            lines.append(format == .wgsl ? "    \(name): \(type)," : "    \(type) \(name);")
        }
        lines.append(format == .wgsl ? "}" : "};")
        if format == .hlsl { lines += hlslConstructor(memberTypes: memberTypes) }
        return lines
    }

    func hlslConstructor(memberTypes: [String]) -> [String] {
        let params = memberTypes.enumerated().map { "\($0.element) arg\($0.offset)" }.joined(separator: ", ")
        return ["\(typeName) \(constructor)(\(params)) {", "    \(typeName) result;"] +
            memberTypes.indices.map { "    result.saga_member_\($0) = arg\($0);" } +
            ["    return result;", "}"]
    }
}

struct ShaderParameter {
    let type: String
    let name: String
}

struct ShaderTextureSamplerPair: Hashable {
    let texture: Int
    let sampler: Int
    let isComparison: Bool
    var name: String { "saga_pair_\(texture)_\(sampler)" }

    init(_ sample: IRImageSample, in function: IRFunction) throws {
        texture = try Self.global(sample.image, in: function)
        sampler = try Self.global(sample.sampler, in: function)
        isComparison = sample.depthReference != nil
    }

    static func global(_ handle: IRHandle<IRExpression>, in function: IRFunction) throws -> Int {
        guard case .globalVariable(let global) = try function.expressions.value(for: handle).kind else {
            throw ShaderFailure.unsupported("Expected a global texture or sampler binding")
        }
        return global.index
    }
}

/// Emits reachable helpers in IR order. Validation guarantees that every callee precedes its caller.
struct ShaderFunctions {
    private let order: [Int]
    private let capturedGlobals: [Int: [Int]]
    private var parameters: [Int: [ShaderParameter]]
    let controlBarriers: IRBarrier
    private(set) var budget: ShaderBudget
    private(set) var integerHelpers: Set<IntegerShaderHelper> = []
    private(set) var integerVectorHelpers: Set<IntegerVectorHelper> = []
    private(set) var glslExtensions: [String] = []

    init(module: IRModule, entry: IRHandle<IRFunction>, captureGlobals: Bool = false) throws {
        var budget = ShaderBudget()
        var pending = [entry]
        var seen = Set([entry.index])
        var calls: [Int: Set<Int>] = [:]
        var captures: [Int: Set<Int>] = [:]
        var barriers: IRBarrier = []
        while let handle = pending.popLast() {
            try budget.consume(.work)
            let function = try module.functions.value(for: handle)
            for statement in function.statements.values {
                try budget.consume(.work)
                switch statement.kind {
                case .call(let callee, _, _):
                    calls[handle.index, default: []].insert(callee.index)
                    if seen.insert(callee.index).inserted { pending.append(callee) }
                case .controlBarrier(let flags):
                    barriers.formUnion(flags)
                default: break
                }
            }
            if captureGlobals {
                for expression in function.expressions.values {
                    try budget.consume(.work)
                    if case .globalVariable(let global) = expression.kind {
                        captures[handle.index, default: []].insert(global.index)
                    }
                }
            }
        }
        seen.remove(entry.index)
        let order = seen.sorted()
        for index in order {
            try budget.consume(.work)
            var bindings = captures[index] ?? []
            for callee in calls[index] ?? [] {
                try budget.consume(.work)
                bindings.formUnion(captures[callee] ?? [])
            }
            captures[index] = bindings
        }
        self.order = order
        self.capturedGlobals = captures.mapValues { $0.sorted() }
        self.parameters = Dictionary(uniqueKeysWithValues: order.map { ($0, []) })
        self.controlBarriers = barriers
        self.budget = budget
    }

    mutating func nonFilteringSamplers(
        module: IRModule, entry: IRHandle<IRFunction>, sampledTextures: [IRGlobalVariable]
    ) throws -> Set<Int> {
        func requiresNonFiltering(_ type: IRHandle<IRType>) throws -> Bool {
            let kind = try module.types.value(for: type).kind
            return kind.isSampledTexture && !kind.isMultisampled2D && (kind.isDepthTexture || kind.sampledTextureScalar != .f32)
        }
        guard try sampledTextures.contains(where: { try requiresNonFiltering($0.type) }) else { return [] }
        var samplers: Set<Int> = []
        for index in order + [entry.index] {
            let function = module.functions.values[index]
            for expression in function.expressions.values {
                try budget.consume(.work)
                guard case .imageSample(let sample) = expression.kind, sample.depthReference == nil,
                      try requiresNonFiltering(function.expressions.value(for: sample.image).type)
                else { continue }
                samplers.insert(try ShaderTextureSamplerPair.global(sample.sampler, in: function))
            }
        }
        return samplers
    }

    mutating func glslTextureSamplerBindings(
        module: IRModule, entry: IRHandle<IRFunction>, textureSlots: Set<UInt32>,
        sharedEntries: [IRHandle<IRFunction>] = []
    ) throws -> [(declaration: String, reflection: ComputeShaderArtifact.GLSLTextureSamplerBinding)] {
        let stageFunctions = order + [entry.index]
        let (stagePairs, standalone, needsShadowLod) = try glslTextureUses(module: module, functions: stageFunctions)
        if needsShadowLod { glslExtensions = ["#extension GL_EXT_texture_shadow_lod : require"] }
        let limit = ComputeShaderArtifact.sampledTextureBindingCapacity
        guard stagePairs.count + standalone <= Int(limit) else {
            throw ShaderFailure.unsupported("GLSL combined texture units exceed the per-stage limit")
        }
        guard !stagePairs.isEmpty else { return [] }

        // Stages compiled from the same render module must agree on each combined unit.
        let additional = try additionalFunctions(module: module, entries: sharedEntries, excluding: Set(stageFunctions))
        let additionalPairs = try glslTextureUses(module: module, functions: additional).pairs
        let pairs = stagePairs.union(additionalPairs).sorted {
            ($0.texture, $0.sampler) < ($1.texture, $1.sampler)
        }
        let capacity = sharedEntries.isEmpty ? limit : 2 * limit
        let units = (0..<(2 * capacity)).filter { !textureSlots.contains($0) }
        guard pairs.count <= Int(capacity), pairs.count <= units.count else {
            throw ShaderFailure.unsupported("GLSL combined texture units exceed the linked-stage limit")
        }
        let globals = module.globalVariables.values
        return try zip(pairs, units).compactMap { pair, unit in
            guard stagePairs.contains(pair) else { return nil }
            guard let texture = globals[pair.texture].binding, let sampler = globals[pair.sampler].binding else {
                throw ShaderFailure.unsupported("Invalid GLSL texture and sampler pair")
            }
            let kind = try module.types.value(for: globals[pair.texture].type).kind
            let scalar = kind.sampledTextureScalar
            let prefix = scalar == .i32 ? "i" : scalar == .u32 ? "u" : ""
            let shape = kind.isCubeTexture ? "samplerCube" : kind.isVolumeTexture ? "sampler3D" : "sampler2D"
            let type = prefix + shape + (kind.isArrayedTexture ? "Array" : "") + (pair.isComparison ? "Shadow" : "")
            return ("layout(binding=\(unit)) uniform \(type) \(pair.name);",
                    .init(texture: texture, sampler: sampler, flatBindingIndex: unit, isComparison: pair.isComparison))
        }
    }

    mutating func glslSharedStorageTextures(
        module: IRModule, entry: IRHandle<IRFunction>, sharedEntries: [IRHandle<IRFunction>]
    ) throws -> [IRGlobalVariable] {
        let additional = try additionalFunctions(module: module, entries: sharedEntries, excluding: Set(order + [entry.index]))
        var globals = reachableGlobals
        for index in additional {
            for expression in module.functions.values[index].expressions.values {
                try budget.consume(.work)
                if case .globalVariable(let global) = expression.kind { globals.insert(global.index) }
            }
        }
        return try globals.sorted().compactMap { index in
            let global = module.globalVariables.values[index]
            return try module.types.value(for: global.type).kind.storageTexture2D == nil ? nil : global
        }
    }

    private mutating func additionalFunctions(
        module: IRModule, entries: [IRHandle<IRFunction>], excluding: Set<Int>
    ) throws -> [Int] {
        var pending = entries
        var seen = excluding
        var additional: [Int] = []
        while let handle = pending.popLast() {
            try budget.consume(.work)
            guard seen.insert(handle.index).inserted else { continue }
            additional.append(handle.index)
            let function = try module.functions.value(for: handle)
            for statement in function.statements.values {
                try budget.consume(.work)
                if case .call(let callee, _, _) = statement.kind { pending.append(callee) }
            }
        }
        return additional
    }

    private mutating func glslTextureUses(
        module: IRModule, functions: [Int]
    ) throws -> (pairs: Set<ShaderTextureSamplerPair>, standaloneCount: Int, needsShadowLod: Bool) {
        var pairs = Set<ShaderTextureSamplerPair>()
        var standalone = Set<Int>()
        var needsShadowLod = false
        for index in functions {
            let function = module.functions.values[index]
            for expression in function.expressions.values {
                try budget.consume(.work)
                switch expression.kind {
                case .imageSample(let sample):
                    pairs.insert(try .init(sample, in: function))
                    if sample.depthReference != nil && sample.gather == nil && sample.level == .zero && sample.arrayIndex != nil {
                        let imageType = try function.expressions.value(for: sample.image).type
                        if try module.types.value(for: imageType).kind.isCubeTexture { needsShadowLod = true }
                    }
                case .imageLoad(let load):
                    let expression = try function.expressions.value(for: load.image)
                    if try module.types.value(for: expression.type).kind.isSampledTexture {
                        standalone.insert(try ShaderTextureSamplerPair.global(load.image, in: function))
                    }
                case .imageQuery(let image, _):
                    let expression = try function.expressions.value(for: image)
                    if try module.types.value(for: expression.type).kind.isSampledTexture {
                        standalone.insert(try ShaderTextureSamplerPair.global(image, in: function))
                    }
                default: break
                }
            }
        }
        return (pairs, standalone.count, needsShadowLod)
    }

    mutating func bindMetalStorage(_ storage: [[ShaderParameter]]) throws {
        for index in order {
            var names: Set<String> = []
            parameters[index] = try capturedGlobals[index, default: []].flatMap { global in
                guard storage.indices.contains(global) else { throw ShaderFailure.invalidHandle }
                return storage[global]
            }.filter { names.insert($0.name).inserted }
        }
    }

    var callCaptures: [Int: [String]] { parameters.mapValues { $0.map(\.name) } }
    var reachableGlobals: Set<Int> { Set(capturedGlobals.values.joined()) }

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
    case .matrix(let columns, let rows, .f32):
        return shaderMatrixType(columns.rawValue, rows.rawValue, in: format)
    default: throw ShaderFailure.unsupported("Shader value type")
    }
}

func shaderMatrixType(_ columns: UInt8, _ rows: UInt8, in format: ShaderOutputFormat) -> String {
    let shape = "\(columns)x\(rows)"
    switch format {
    case .wgsl: return "mat\(shape)<f32>"
    case .glsl: return "mat\(shape)"
    default: return "float\(shape)"
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
