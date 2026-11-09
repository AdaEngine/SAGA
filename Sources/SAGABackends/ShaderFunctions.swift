import SAGAShaderIR
import SAGAValidation

struct ShaderStructure {
    let typeName: String
    let constructor: String
    var memberNames: [String]? = nil

    func memberName(_ index: Int) -> String {
        memberNames?[index] ?? "saga_member_\(index)"
    }

    func declarations(memberTypes: [String], in format: ShaderOutputFormat) -> [String] {
        var lines = ["struct \(typeName) {"]
        for (index, type) in memberTypes.enumerated() {
            let name = memberName(index)
            lines.append(format == .wgsl ? "    \(name): \(type)," : "    \(type) \(name);")
        }
        lines.append(format == .wgsl ? "}" : "};")
        if format == .hlsl { lines += hlslConstructor(memberTypes: memberTypes) }
        return lines
    }

    func hlslConstructor(memberTypes: [String]) -> [String] {
        let params = memberTypes.enumerated().map { "\($0.element) arg\($0.offset)" }.joined(separator: ", ")
        return ["\(typeName) \(constructor)(\(params)) {", "    \(typeName) result;"] +
            memberTypes.indices.map { "    result.\(memberName($0)) = arg\($0);" } +
            ["    return result;", "}"]
    }
}

func atomicComparisonDeclarations(
    module: IRModule, format: ShaderOutputFormat, webGPU: Bool,
    structures: inout [IRHandle<IRType>: ShaderStructure]
) throws -> [String] {
    var declarations: [String] = []
    for entry in module.specialTypes.predeclared {
        guard case .atomicCompareExchangeWeakResult(let scalar) = entry.kind else { continue }
        if webGPU && scalar.width == 8 { continue }
        let suffix = "\(scalar.kind == .sint ? "I" : "U")\(Int(scalar.width) * 8)"
        let name = "SagaAtomicResult\(suffix)"
        let constructor = format == .hlsl ? "saga_make_atomic_\(suffix)" : name
        let structure = ShaderStructure(typeName: name, constructor: constructor)
        let lane = try shaderVectorType(1, scalar: scalar, in: format)
        declarations += structure.declarations(memberTypes: [lane, "bool"], in: format)
        structures[entry.type] = structure
    }
    return declarations
}

struct ShaderParameter {
    let type: String
    let name: String
}

struct PrivateShaderBindings {
    var storage: [Int: ShaderStorage] = [:]
    var captures: [Int: [ShaderParameter]] = [:]
    var declarations: [String] = []
    var initialization: [String] = []
}

struct SplitImageAccess {
    let base: IRHandle<IRExpression>
    let dynamic: IRHandle<IRExpression>?
    let fixed: UInt32?

    init?(_ image: IRHandle<IRExpression>, in function: IRFunction) throws {
        switch try function.expressions.value(for: image).kind {
        case .accessIndex(let array, let index):
            (base, dynamic, fixed) = (array, nil, index)
        case .access(let array, let index):
            (base, dynamic) = (array, index)
            switch try function.expressions.value(for: index).kind {
            case .literal(.u32(let value)): fixed = value
            case .literal(.i32(let value)): fixed = UInt32(max(0, value))
            default: fixed = nil
            }
        default: return nil
        }
    }

    func fixedElement(count: UInt32) -> UInt32? {
        if count == 1 { return 0 }
        return fixed.map { min($0, count - 1) }
    }
}

struct GLSLResourceElement: Hashable {
    let global: Int
    let element: UInt32?

    static func choices(
        _ handle: IRHandle<IRExpression>, in function: IRFunction, module: IRModule,
        bindingArrayLengths: [IRResourceBinding: UInt32]
    ) throws -> [Self] {
        guard let global = try ShaderTextureSamplerPair.resourceGlobal(handle, in: function) else {
            throw ShaderFailure.unsupported("Expected a global texture or sampler binding")
        }
        guard let access = try SplitImageAccess(handle, in: function) else {
            return [.init(global: global, element: nil)]
        }
        let resource = module.globalVariables.values[global]
        guard let binding = resource.binding,
              case .bindingArray(_, let size) = try module.types.value(for: resource.type).kind else {
            throw ShaderFailure.invalidHandle
        }
        let count = try resourceBindingArrayCount(size, binding: binding, location: resource.location,
                                                  bindingArrayLengths: bindingArrayLengths)
        if let fixed = access.fixedElement(count: count) { return [.init(global: global, element: fixed)] }
        return (0..<count).map { .init(global: global, element: $0) }
    }
}

struct ShaderTextureSamplerPair: Hashable {
    let texture: GLSLResourceElement
    let sampler: GLSLResourceElement
    let isComparison: Bool
    let externalPlane: UInt32?
    var name: String {
        let textureSuffix = texture.element.map { "_t\($0)" } ?? ""
        let samplerSuffix = sampler.element.map { "_s\($0)" } ?? ""
        let planeSuffix = externalPlane.map { "_p\($0)" } ?? ""
        return "saga_pair_\(texture.global)_\(sampler.global)\(textureSuffix)\(samplerSuffix)\(planeSuffix)"
    }

    init(texture: GLSLResourceElement, sampler: GLSLResourceElement, isComparison: Bool, externalPlane: UInt32? = nil) {
        self.texture = texture
        self.sampler = sampler
        self.isComparison = isComparison
        self.externalPlane = externalPlane
    }

    init(_ sample: IRImageSample, in function: IRFunction, textureElement: UInt32?, samplerElement: UInt32?,
         externalPlane: UInt32? = nil) throws {
        guard let texture = try Self.resourceGlobal(sample.image, in: function),
              let sampler = try Self.resourceGlobal(sample.sampler, in: function) else {
            throw ShaderFailure.unsupported("Expected a global texture or sampler binding")
        }
        self.init(texture: .init(global: texture, element: textureElement),
                  sampler: .init(global: sampler, element: samplerElement), isComparison: sample.depthReference != nil,
                  externalPlane: externalPlane)
    }

    static func resourceGlobal(_ handle: IRHandle<IRExpression>, in function: IRFunction) throws -> Int? {
        switch try function.expressions.value(for: handle).kind {
        case .globalVariable(let global): return global.index
        case .access(let base, _), .accessIndex(let base, _):
            guard case .globalVariable(let global) = try function.expressions.value(for: base).kind else { return nil }
            return global.index
        default: return nil
        }
    }
}

/// Emits reachable helpers in IR order. Validation guarantees that every callee precedes its caller.
struct ShaderFunctions {
    private let order: [Int]
    private let capturedGlobals: [Int: [Int]]
    private var parameters: [Int: [ShaderParameter]]
    let controlBarriers: IRBarrier
    let requiresShaderInt64AtomicAllOps: Bool
    let demotesToHelperInvocation: Bool
    let requiresDebugPrintf: Bool
    let subgroupExtensions: Set<String>
    let subgroupUsesFloat16: Bool
    let floatTextureSamplers: [Int: Set<IRResourceBinding>]
    let pointerWrites: [Int: Set<Int>]
    private(set) var budget: ShaderBudget
    private(set) var integerHelpers: Set<IntegerShaderHelper> = []
    private(set) var integerVectorHelpers: Set<IntegerVectorHelper> = []
    private(set) var glslExtensions: [String] = []

    init(module: IRModule, entry: IRHandle<IRFunction>) throws {
        var budget = ShaderBudget()
        var pending = [entry]
        var seen = Set([entry.index])
        var calls: [Int: Set<Int>] = [:]
        var captures: [Int: Set<Int>] = [:]
        var barriers: IRBarrier = []
        var needsFullInt64Atomics = false
        var demotes = false
        var prints = false
        var subgroupExtensions: Set<String> = []
        var subgroupFloat16 = false
        var floatTextureSamplers: [Int: Set<IRResourceBinding>] = [:]
        func wideAtomicSpace(_ pointer: IRHandle<IRExpression>, in function: IRFunction) throws -> IRAddressSpace? {
            let type = try function.expressions.value(for: pointer).type
            guard case .pointer(let base, let space) = try module.types.value(for: type).kind,
                  case .atomic(let scalar) = try module.types.value(for: base).kind,
                  scalar.width == 8 else { return nil }
            return space
        }
        while let handle = pending.popLast() {
            try budget.consume(.work)
            let function = try module.functions.value(for: handle)
            for statement in function.statements.values {
                try budget.consume(.work)
                var subgroupOperand: IRHandle<IRExpression>?
                switch statement.kind {
                case .call(let callee, _, _):
                    calls[handle.index, default: []].insert(callee.index)
                    if seen.insert(callee.index).inserted { pending.append(callee) }
                case .controlBarrier(let flags):
                    barriers.formUnion(flags)
                case .debugPrintf:
                    prints = true
                case .atomic(let pointer, let operation, _, let result) where !needsFullInt64Atomics:
                    if let space = try wideAtomicSpace(pointer, in: function) {
                        if case .storage = space, result == nil, operation == .min || operation == .max { break }
                        needsFullInt64Atomics = true
                    }
                case .store(let pointer, _), .workgroupUniformLoad(let pointer, _):
                    if !needsFullInt64Atomics, try wideAtomicSpace(pointer, in: function) != nil {
                        needsFullInt64Atomics = true
                    }
                case .demoteToHelperInvocation:
                    demotes = true
                case .subgroupGather(let mode, let argument, _):
                    switch mode {
                    case .broadcast, .broadcastFirst: subgroupExtensions.insert("ballot")
                    case .shuffle, .shuffleXor: subgroupExtensions.insert("shuffle")
                    case .shuffleUp, .shuffleDown: subgroupExtensions.insert("shuffle_relative")
                    case .quadBroadcast, .quadBroadcastDynamic, .quadSwap: subgroupExtensions.insert("quad")
                    }
                    subgroupOperand = argument
                case .subgroupBallot:
                    subgroupExtensions.insert("ballot")
                case .subgroupCollectiveOperation(let operation, _, let argument, _):
                    subgroupExtensions.insert(operation.isVote ? "vote" : "arithmetic")
                    subgroupOperand = argument
                default: break
                }
                if let subgroupOperand {
                    let type = try function.expressions.value(for: subgroupOperand).type
                    if try module.types.value(for: type).kind.usesFloat16 { subgroupFloat16 = true }
                }
            }
            for expression in function.expressions.values {
                try budget.consume(.work)
                if case .globalVariable(let global) = expression.kind {
                    captures[handle.index, default: []].insert(global.index)
                }
                if !needsFullInt64Atomics, case .load(let pointer) = expression.kind,
                   try wideAtomicSpace(pointer, in: function) != nil {
                    needsFullInt64Atomics = true
                }
                if case .imageSample(let sample) = expression.kind {
                    let image = try function.expressions.value(for: sample.image)
                    let kind = try module.types.value(for: image.type).kind
                    if !kind.isDepthTexture, kind.sampledTextureScalar == .f32,
                       let global = try ShaderTextureSamplerPair.resourceGlobal(sample.image, in: function),
                       let sampler = try ShaderTextureSamplerPair.resourceGlobal(sample.sampler, in: function),
                       let binding = module.globalVariables.values[sampler].binding {
                        floatTextureSamplers[global, default: []].insert(binding)
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
        var hasPointerParameters = false
        for index in order {
            try budget.consume(.work)
            for argument in module.functions.values[index].arguments {
                try budget.consume(.work)
                if case .pointer = try module.types.value(for: argument.type).kind { hasPointerParameters = true }
            }
            if hasPointerParameters { break }
        }
        var written: [Int: Set<Int>] = [:]
        if hasPointerParameters {
            var accesses: [Int: ShaderPointerAccesses] = [:]
            for index in order {
                try budget.consume(.work)
                let summary = try ShaderPointerAccesses.checking(module.functions.values[index], in: module,
                                                                callees: accesses, budget: &budget)
                accesses[index] = summary
                written[index] = summary.writtenParameters
            }
        }
        self.order = order
        self.pointerWrites = written
        self.capturedGlobals = captures.mapValues { $0.sorted() }
        self.parameters = Dictionary(uniqueKeysWithValues: order.map { ($0, []) })
        self.controlBarriers = barriers
        self.requiresShaderInt64AtomicAllOps = needsFullInt64Atomics
        self.demotesToHelperInvocation = demotes
        self.requiresDebugPrintf = prints
        self.subgroupExtensions = subgroupExtensions
        self.subgroupUsesFloat16 = subgroupFloat16
        self.floatTextureSamplers = floatTextureSamplers
        self.budget = budget
    }

    func glslSubgroupExtensions(requiresSubgroups: Bool) -> [String] {
        guard requiresSubgroups else { return [] }
        let standard = (["basic"] + subgroupExtensions.sorted()).map {
            "#extension GL_KHR_shader_subgroup_\($0) : require"
        }
        return standard + (subgroupUsesFloat16 ? ["#extension GL_EXT_shader_subgroup_extended_types_float16 : require"] : [])
    }

    mutating func nonFilteringSamplers(
        module: IRModule, entry: IRHandle<IRFunction>, sampledTextures: [IRGlobalVariable]
    ) throws -> Set<Int> {
        func requiresNonFiltering(_ type: IRHandle<IRType>) throws -> Bool {
            var kind = try module.types.value(for: type).kind
            if case .bindingArray(let base, _) = kind { kind = try module.types.value(for: base).kind }
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
                guard let global = try ShaderTextureSamplerPair.resourceGlobal(sample.sampler, in: function) else {
                    throw ShaderFailure.unsupported("Expected a bound sampler")
                }
                samplers.insert(global)
            }
        }
        return samplers
    }

    mutating func glslTextureSamplerBindings(
        module: IRModule, entry: IRHandle<IRFunction>, textureSlots: Set<UInt32>,
        bindingArrayLengths: [IRResourceBinding: UInt32],
        sharedEntries: [IRHandle<IRFunction>] = []
    ) throws -> [(declaration: String, reflection: ComputeShaderArtifact.GLSLTextureSamplerBinding)] {
        let stageFunctions = order + [entry.index]
        let (stagePairs, standalone, needsShadowLod) = try glslTextureUses(
            module: module, functions: stageFunctions, bindingArrayLengths: bindingArrayLengths)
        if needsShadowLod { glslExtensions = ["#extension GL_EXT_texture_shadow_lod : require"] }
        let limit = ComputeShaderArtifact.sampledTextureBindingCapacity
        guard stagePairs.count + standalone <= Int(limit) else {
            throw ShaderFailure.unsupported("GLSL combined texture units exceed the per-stage limit")
        }
        guard !stagePairs.isEmpty else { return [] }

        // Stages compiled from the same render module must agree on each combined unit.
        let additional = try additionalFunctions(module: module, entries: sharedEntries, excluding: Set(stageFunctions))
        let additionalPairs = try glslTextureUses(module: module, functions: additional,
                                                   bindingArrayLengths: bindingArrayLengths).pairs
        let pairs = stagePairs.union(additionalPairs).sorted {
            ($0.texture.global, $0.texture.element ?? 0, $0.sampler.global, $0.sampler.element ?? 0, $0.externalPlane ?? 0) <
            ($1.texture.global, $1.texture.element ?? 0, $1.sampler.global, $1.sampler.element ?? 0, $1.externalPlane ?? 0)
        }
        let capacity = sharedEntries.isEmpty ? limit : 2 * limit
        let units = (0..<(2 * capacity)).filter { !textureSlots.contains($0) }
        guard pairs.count <= Int(capacity), pairs.count <= units.count else {
            throw ShaderFailure.unsupported("GLSL combined texture units exceed the linked-stage limit")
        }
        let globals = module.globalVariables.values
        return try zip(pairs, units).compactMap { pair, unit in
            guard stagePairs.contains(pair) else { return nil }
            guard let texture = globals[pair.texture.global].binding, let sampler = globals[pair.sampler.global].binding else {
                throw ShaderFailure.unsupported("Invalid GLSL texture and sampler pair")
            }
            let kind = try bindingElementKind(globals[pair.texture.global].type, module: module)
            let scalar = kind.sampledTextureScalar
            let prefix = scalar == .i32 ? "i" : scalar == .u32 ? "u" : ""
            let shape = kind.isCubeTexture ? "samplerCube" : kind.isVolumeTexture ? "sampler3D" :
                kind.isLineTexture ? "sampler1D" : "sampler2D"
            let type = prefix + shape + (kind.isArrayedTexture ? "Array" : "") + (pair.isComparison ? "Shadow" : "")
            return ("layout(binding=\(unit)) uniform \(type) \(pair.name);",
                    .init(texture: texture, sampler: sampler, flatBindingIndex: unit, isComparison: pair.isComparison,
                          textureArrayElement: pair.texture.element, samplerArrayElement: pair.sampler.element,
                          externalTexturePlane: pair.externalPlane))
        }
    }

    mutating func glslLinkedGlobals(
        module: IRModule, entry: IRHandle<IRFunction>, sharedEntries: [IRHandle<IRFunction>]
    ) throws -> Set<Int> {
        let additional = try additionalFunctions(module: module, entries: sharedEntries, excluding: Set(order + [entry.index]))
        var globals = reachableGlobals
        for index in additional {
            for expression in module.functions.values[index].expressions.values {
                try budget.consume(.work)
                if case .globalVariable(let global) = expression.kind { globals.insert(global.index) }
            }
        }
        return globals
    }

    mutating func aggregateBufferGlobals(
        module: IRModule, entry: IRHandle<IRFunction>, sharedEntries: [IRHandle<IRFunction>] = []
    ) throws -> Set<Int> {
        let active = order + [entry.index]
        let additional = try additionalFunctions(module: module, entries: sharedEntries, excluding: Set(active))
        var roots: Set<Int> = []
        func record(_ pointer: IRHandle<IRExpression>, in function: IRFunction) throws {
            let expression = try function.expressions.value(for: pointer)
            guard case .pointer(let type, let space) = try module.types.value(for: expression.type).kind,
                  space.isBufferAddressSpace, space != .immediate,
                  try module.types.value(for: type).kind.isShaderCompositeValue else { return }
            var current = expression
            while true {
                try budget.consume(.work)
                switch current.kind {
                case .globalVariable(let global):
                    roots.insert(global.index)
                    return
                case .access(let base, _), .accessIndex(let base, _):
                    current = try function.expressions.value(for: base)
                default:
                    return
                }
            }
        }
        for index in active + additional {
            let function = module.functions.values[index]
            for expression in function.expressions.values {
                try budget.consume(.work)
                if case .load(let pointer) = expression.kind { try record(pointer, in: function) }
            }
            for statement in function.statements.values {
                try budget.consume(.work)
                if case .store(let pointer, _) = statement.kind { try record(pointer, in: function) }
            }
        }
        return roots
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
        module: IRModule, functions: [Int], bindingArrayLengths: [IRResourceBinding: UInt32]
    ) throws -> (pairs: Set<ShaderTextureSamplerPair>, standaloneCount: Int, needsShadowLod: Bool) {
        var pairs = Set<ShaderTextureSamplerPair>()
        var standalone = Set<GLSLResourceElement>()
        var needsShadowLod = false
        for index in functions {
            let function = module.functions.values[index]
            for expression in function.expressions.values {
                try budget.consume(.work)
                switch expression.kind {
                case .imageSample(let sample):
                    let image = try function.expressions.value(for: sample.image)
                    let external = try module.types.value(for: image.type).kind.isExternalTexture
                    let textures = try GLSLResourceElement.choices(sample.image, in: function, module: module,
                                                                   bindingArrayLengths: bindingArrayLengths)
                    let samplers = try GLSLResourceElement.choices(sample.sampler, in: function, module: module,
                                                                   bindingArrayLengths: bindingArrayLengths)
                    try budget.consume(.work, amount: textures.count * samplers.count * (external ? 3 : 1))
                    for texture in textures {
                        for sampler in samplers {
                            if external {
                                for plane in UInt32(0)..<3 {
                                    pairs.insert(.init(texture: texture, sampler: sampler, isComparison: false, externalPlane: plane))
                                }
                            } else {
                                pairs.insert(.init(texture: texture, sampler: sampler, isComparison: sample.depthReference != nil))
                            }
                        }
                    }
                    if sample.depthReference != nil && sample.gather == nil && sample.level == .zero && sample.arrayIndex != nil {
                        let imageType = try function.expressions.value(for: sample.image).type
                        if try module.types.value(for: imageType).kind.isCubeTexture { needsShadowLod = true }
                    }
                case .imageLoad(let load):
                    let expression = try function.expressions.value(for: load.image)
                    let kind = try module.types.value(for: expression.type).kind
                    if kind.isSampledTexture {
                        standalone.formUnion(try GLSLResourceElement.choices(
                            load.image, in: function, module: module, bindingArrayLengths: bindingArrayLengths))
                    } else if kind.isExternalTexture,
                              let global = try ShaderTextureSamplerPair.resourceGlobal(load.image, in: function) {
                        for plane in UInt32(0)..<3 { standalone.insert(.init(global: global, element: plane)) }
                    }
                case .imageQuery(let image, _):
                    let expression = try function.expressions.value(for: image)
                    let kind = try module.types.value(for: expression.type).kind
                    if kind.isSampledTexture {
                        standalone.formUnion(try GLSLResourceElement.choices(
                            image, in: function, module: module, bindingArrayLengths: bindingArrayLengths))
                    } else if kind.isExternalTexture,
                              let global = try ShaderTextureSamplerPair.resourceGlobal(image, in: function) {
                        standalone.insert(.init(global: global, element: 0))
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

    mutating func nativeIntegerWidths(
        in module: IRModule, entry: IRHandle<IRFunction>, resources: [IRGlobalVariable]
    ) throws -> Set<UInt8> {
        guard module.types.values.contains(where: { $0.kind.nativeShaderIntegerWidth != nil }) else { return [] }
        try budget.consume(.work, amount: resources.count)
        var types = Set(resources.map(\.type))
        for index in order + [entry.index] {
            let function = module.functions.values[index]
            try budget.consume(.work, amount: 1 + function.arguments.count + function.localVariables.count + function.expressions.count)
            types.formUnion(function.arguments.map(\.type))
            if let result = function.result { types.insert(result.type) }
            types.formUnion(function.localVariables.values.map(\.type))
            types.formUnion(function.expressions.values.map(\.type))
        }
        var pending = Array(types)
        var widths: Set<UInt8> = []
        while let type = pending.popLast() {
            try budget.consume(.work)
            let kind = try module.types.value(for: type).kind
            if let width = kind.nativeShaderIntegerWidth { widths.insert(width) }
            switch kind {
            case .pointer(let base, _), .array(let base, _, _), .bindingArray(let base, _):
                if types.insert(base).inserted { pending.append(base) }
            case .structure(let members, _):
                try budget.consume(.work, amount: members.count)
                for member in members where types.insert(member.type).inserted { pending.append(member.type) }
            default: break
            }
        }
        return widths
    }

    mutating func wgslDiagnosticLines(module: IRModule, function: IRFunction? = nil) throws -> [String] {
        var cursor = function == nil ? module.diagnosticFilter : function?.diagnosticFilter
        let parent = function == nil ? nil : module.diagnosticFilter
        var lines: [String] = []
        while let handle = cursor, handle != parent {
            try budget.consume(.work)
            let filter = try module.diagnosticFilters.value(for: handle)
            let directive = "diagnostic(\(filter.severity.rawValue), \(filter.rule.wgslName))"
            let line = function == nil ? directive + ";" : "@" + directive
            try budget.consume(.sourceBytes, amount: line.utf8.count + 1)
            lines.append(line)
            cursor = filter.parent
        }
        return Array(lines.reversed())
    }

    mutating func compositeDeclarations(
        module: IRModule, entry: IRHandle<IRFunction>, format: ShaderOutputFormat,
        additionalTypes: Set<IRHandle<IRType>> = [], structures: inout [IRHandle<IRType>: ShaderStructure]
    ) throws -> [String] {
        var types = additionalTypes
        var constructed: Set<IRHandle<IRType>> = []
        var reconstructed: Set<IRHandle<IRType>> = []
        var initializers: Set<IRHandle<IRExpression>> = []
        for index in reachableGlobals {
            try budget.consume(.work)
            let global = module.globalVariables.values[index]
            if global.space == .private {
                types.insert(global.type)
                if let initializer = global.initializer { initializers.insert(initializer) }
            }
        }
        for index in order + [entry.index] {
            let function = module.functions.values[index]
            for argument in function.arguments {
                try budget.consume(.work)
                let kind = try module.types.value(for: argument.type).kind
                if case .pointer(let pointee, _) = kind {
                    types.insert(pointee)
                } else if kind.isShaderCompositeValue {
                    types.insert(argument.type)
                }
            }
            if let result = function.result, try module.types.value(for: result.type).kind.isShaderCompositeValue {
                types.insert(result.type)
            }
            for local in function.localVariables.values {
                try budget.consume(.work)
                if try module.types.value(for: local.type).kind.isShaderCompositeValue { types.insert(local.type) }
            }
            for expression in function.expressions.values {
                try budget.consume(.work)
                let type: IRHandle<IRType>
                switch expression.kind {
                case .compose:
                    type = expression.type
                    if try module.types.value(for: type).kind.isShaderCompositeValue { constructed.insert(type) }
                case .constant(let handle):
                    type = expression.type
                    let constant = try module.constants.value(for: handle)
                    if try module.types.value(for: type).kind.isShaderCompositeValue {
                        initializers.insert(constant.initializer)
                    }
                case .load(let pointer):
                    type = expression.type
                    let source = try function.expressions.value(for: pointer)
                    if case .pointer(_, let space) = try module.types.value(for: source.type).kind,
                       space.isBufferAddressSpace, try module.types.value(for: type).kind.isShaderCompositeValue {
                        reconstructed.insert(type)
                    }
                case .runtimeValue:
                    type = expression.type
                case .access(let base, _), .accessIndex(let base, _):
                    type = try function.expressions.value(for: base).type
                default:
                    continue
                }
                if try module.types.value(for: type).kind.isShaderCompositeValue { types.insert(type) }
            }
        }

        if format.wrapsShaderValueArrays {
            var remaining = Array(reconstructed)
            while let handle = remaining.popLast() {
                try budget.consume(.work)
                let children: [IRHandle<IRType>]
                switch try module.types.value(for: handle).kind {
                case .array(let element, .constant, _): children = [element]
                case .structure(let members, _):
                    try budget.consume(.work, amount: members.count)
                    children = members.map(\.type)
                default: continue
                }
                constructed.insert(handle)
                for child in children where reconstructed.insert(child).inserted { remaining.append(child) }
            }
            var pending = Array(initializers)
            while let handle = pending.popLast() {
                try budget.consume(.work)
                let expression = try module.globalExpressions.value(for: handle)
                if case .compose(let type, let components) = expression.kind {
                    if try module.types.value(for: type).kind.isShaderCompositeValue { constructed.insert(type) }
                    for child in components {
                        try budget.consume(.work)
                        if initializers.insert(child).inserted { pending.append(child) }
                    }
                }
            }
        }
        return try compositeValueDeclarations(module: module, format: format, roots: types, constructed: constructed,
                                              structures: &structures, budget: &budget)
    }

    mutating func privateBindings(
        module: IRModule, function: IRFunction, format: ShaderOutputFormat, structures: [IRHandle<IRType>: ShaderStructure]
    ) throws -> PrivateShaderBindings {
        var writer = ShaderBodyWriter(module: module, function: function, arguments: [], storage: [],
                                      outputFormat: format, resultTarget: .direct, structures: structures, budget: budget)
        let bindings = try writer.privateBindings(reachableGlobals)
        budget = writer.budget
        return bindings
    }

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
                let type: String
                if case .pointer(let pointee, let space) = try module.types.value(for: argument.type).kind {
                    let value = try shaderValueType(pointee, module: module, format: format, structures: structures)
                    switch format {
                    case .wgsl:
                        let addressSpace = module.allowsPointerSubviews || space == .function ? "function" : "private"
                        type = "ptr<\(addressSpace), \(value)>"
                    case .msl: type = "thread \(value)&"
                    default:
                        let qualifier = pointerWrites[index]?.contains(slot) == true ? "inout" : "in"
                        type = "\(qualifier) \(value)"
                    }
                } else {
                    type = try shaderValueType(argument.type, module: module, format: format, structures: structures)
                }
                return ShaderParameter(type: type, name: "saga_param_\(slot)")
            }
            let resultType = try function.result.map {
                try shaderValueType($0.type, module: module, format: format, structures: structures)
            }
            let inputs: [ShaderParameter]
            if module.allowsPointerSubviews {
                inputs = try zip(explicit, function.arguments).flatMap { lowered, argument in
                    try budget.consume(.work)
                    if case .pointer = try module.types.value(for: argument.type).kind {
                        return [lowered, .init(type: "bool", name: lowered.name + "_valid")]
                    }
                    return [lowered]
                }
            } else {
                inputs = explicit
            }
            let all = inputs + (parameters[index] ?? [])
            let name = "saga_fn_\(index)"
            let signature: String
            if format == .wgsl {
                let args = all.map { "\($0.name): \($0.type)" }.joined(separator: ", ")
                let result = resultType.map { " -> \($0)" } ?? ""
                signature = "fn \(name)(\(args))\(result) {"
                lines += try wgslDiagnosticLines(module: module, function: function)
            } else {
                let args = all.map { "\($0.type) \($0.name)" }.joined(separator: ", ")
                signature = "\(resultType ?? "void") \(name)(\(args)) {"
            }
            try budget.consume(.sourceBytes, amount: signature.utf8.count + 3)
            var writer = ShaderBodyWriter(module: module, function: function, arguments: explicit.map(\.name),
                                          storage: storage, outputFormat: format, resultTarget: .direct,
                                          structures: structures, callCaptures: calls, pointerWrites: pointerWrites, budget: budget)
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
    case .scalar(.i16): return integer16Type(signed: true, in: format)
    case .scalar(.u16): return integer16Type(signed: false, in: format)
    case .scalar(.u32): return format == .wgsl ? "u32" : "uint"
    case .scalar(.u64): return unsigned64Type(in: format)
    case .scalar(.i64): return signed64Type(in: format)
    case .scalar(.i32): return format == .wgsl ? "i32" : "int"
    case .scalar(.f32): return floatType(1, in: format)
    case .scalar(.f16): return halfType(1, in: format)
    case .vector(let size, let scalar): return try shaderVectorType(size.rawValue, scalar: scalar, in: format)
    case .matrix(let columns, let rows, let scalar) where [.f16, .f32].contains(scalar):
        return shaderMatrixType(columns.rawValue, rows.rawValue, scalar: scalar, in: format)
    default: throw ShaderFailure.unsupported("Shader value type")
    }
}

func integer16Type(signed: Bool, in format: ShaderOutputFormat) -> String {
    switch format {
    case .wgsl: signed ? "i16" : "u16"
    case .msl: signed ? "short" : "ushort"
    default: signed ? "int16_t" : "uint16_t"
    }
}

func unsigned64Type(in format: ShaderOutputFormat) -> String {
    switch format {
    case .wgsl: "u64"
    case .msl: "ulong"
    default: "uint64_t"
    }
}

func unsigned64Literal(_ value: UInt64, in format: ShaderOutputFormat) -> String {
    switch format {
    case .wgsl: "\(value)lu"
    case .glsl: "\(value)ul"
    default: "\(value)uL"
    }
}

func signed64Type(in format: ShaderOutputFormat) -> String {
    switch format {
    case .wgsl: "i64"
    case .msl: "long"
    default: "int64_t"
    }
}

func signed64Literal(_ value: Int64, in format: ShaderOutputFormat) -> String {
    if format == .wgsl { return "\(value)li" }
    let suffix = format == .glsl ? "l" : "L"
    if value == .min { return "(-9223372036854775807\(suffix) - 1\(suffix))" }
    return "\(value)\(suffix)"
}

func shaderMatrixType(_ columns: UInt8, _ rows: UInt8, scalar: IRScalar, in format: ShaderOutputFormat) -> String {
    let shape = "\(columns)x\(rows)"
    let half = scalar == .f16
    switch format {
    case .wgsl: return "mat\(shape)<\(half ? "f16" : "f32")>"
    case .glsl: return "\(half ? "f16mat" : "mat")\(shape)"
    default: return "\(half ? "half" : "float")\(shape)"
    }
}

func shaderVectorType(_ width: UInt8, scalar: IRScalar, in format: ShaderOutputFormat) throws -> String {
    if scalar == .f32 { return floatType(width, in: format) }
    if scalar == .f16 { return halfType(width, in: format) }
    if scalar == .i16 || scalar == .u16 {
        let signed = scalar == .i16
        let lane = integer16Type(signed: signed, in: format)
        if width == 1 { return lane }
        switch format {
        case .wgsl: return "vec\(width)<\(lane)>"
        case .msl: return "\(lane)\(width)"
        case .glsl: return "\(signed ? "i" : "u")16vec\(width)"
        default: return "vector<\(lane), \(width)>"
        }
    }
    if scalar == .i64 || scalar == .u64 {
        let signed = scalar == .i64
        let lane = signed ? signed64Type(in: format) : unsigned64Type(in: format)
        if width == 1 { return lane }
        switch format {
        case .wgsl: return "vec\(width)<\(lane)>"
        case .msl: return "\(lane)\(width)"
        case .glsl: return "\(signed ? "i" : "u")64vec\(width)"
        default: return "vector<\(lane), \(width)>"
        }
    }
    let native: String
    let glsl: String
    let wgsl: String
    switch scalar {
    case .i32: (native, glsl, wgsl) = ("int", "ivec", "i32")
    case .u32: (native, glsl, wgsl) = ("uint", "uvec", "u32")
    case .bool: (native, glsl, wgsl) = ("bool", "bvec", "bool")
    default: throw ShaderFailure.unsupported("Shader vector lane type")
    }
    if width == 1 { return format == .wgsl ? wgsl : native }
    switch format {
    case .wgsl: return "vec\(width)<\(wgsl)>"
    case .glsl: return "\(glsl)\(width)"
    default: return "\(native)\(width)"
    }
}
