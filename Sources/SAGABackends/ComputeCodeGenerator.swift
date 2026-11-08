import SAGAShaderIR
import SAGAValidation

/// Initial compute writer. All unsupported IR fails closed before an artifact is returned.
package struct ComputeCodeGenerator: ShaderCodeGenerator {
    package let outputFormat: ShaderOutputFormat
    private let bindingArrayLengths: [IRResourceBinding: UInt32]
    private let webGPU: Bool

    package init?(_ output: ShaderOutputFormat, bindingArrayLengths: [IRResourceBinding: UInt32] = [:], webGPU: Bool = false) {
        switch output {
        case .msl, .hlsl, .glsl, .wgsl: outputFormat = output
        default: return nil
        }
        self.bindingArrayLengths = bindingArrayLengths
        self.webGPU = webGPU
    }

    package func generate(_ validated: ValidatedModule<IRModule>) throws -> ShaderArtifact {
        try generateCompute(validated).artifact
    }

    package func generateCompute(
        _ validated: ValidatedModule<IRModule>, onlyReachableBindings: Bool = false
    ) throws -> ComputeShaderArtifact {
        let module = validated.module
        let usesFloat16 = module.types.values.contains { $0.kind.usesFloat16 }
        let requiresStandardUniformLayout = outputFormat == .wgsl && module.uniformBufferStandardLayout
        guard module.entryPoints.count == 1, let entry = module.entryPoints.first,
              entry.stage == .compute, entry.workgroupOverrides == nil,
              module.overrides.count == 0
        else { throw unsupported("expected one compute entry point with no overrides") }
        let function = try module.functions.value(for: entry.function)
        guard function.result == nil else {
            throw unsupported("compute return values")
        }

        let size = entry.workgroupSize
        guard size.x > 0, size.y > 0, size.z > 0,
              size.x <= 1024, size.y <= 1024, size.z <= 64,
              size.x * size.y * size.z <= 1024
        else { throw unsupported("workgroup size exceeds the compute profile") }
        let subgroupSize: UInt32?
        switch entry.subgroupSize {
        case .fixed(let required):
            guard size.x.isMultiple(of: required) else {
                throw ShaderFailure.diagnostics([.init(severity: .error, code: .invalidBinding,
                    message: "Workgroup x dimension must be a multiple of the subgroup size", location: entry.location)])
            }
            subgroupSize = required
        case .overrideExpression: throw unsupported("unspecialized subgroup size")
        case nil: subgroupSize = nil
        }

        var inputs = try ComputeInputs(arguments: function.arguments, workgroupSize: size, module: module, format: outputFormat)
        var helpers = try ShaderFunctions(module: module, entry: entry.function)
        let aggregateBuffers = try helpers.aggregateBufferGlobals(module: module, entry: entry.function)
        let usesSubgroupBarrier = helpers.controlBarriers.contains(.subgroup)
        if webGPU && usesSubgroupBarrier {
            throw unsupported("Subgroup barriers require a native subgroup synchronization target")
        }
        let requiresSubgroups = subgroupSize != nil || inputs.usesSubgroups || !helpers.subgroupExtensions.isEmpty || usesSubgroupBarrier
        var declarations = inputs.declarations
        var structures = inputs.structures
        declarations += try decompositionDeclarations(module: module, format: outputFormat, structures: &structures)
        declarations += try atomicComparisonDeclarations(module: module, format: outputFormat, structures: &structures)
        let reachableGlobals = helpers.reachableGlobals
        var compositeWorkgroupSizes: [IRHandle<IRType>: UInt64] = [:]
        for index in reachableGlobals {
            let global = module.globalVariables.values[index]
            if global.space == .workgroup, let bytes = try workgroupCompositeByteSize(global.type, in: module) {
                compositeWorkgroupSizes[global.type] = bytes
            }
        }
        declarations += try helpers.compositeDeclarations(module: module, entry: entry.function, format: outputFormat,
                                                          additionalTypes: Set(compositeWorkgroupSizes.keys), structures: &structures)
        let privateBindings = try helpers.privateBindings(module: module, function: function, format: outputFormat,
                                                          structures: structures)
        declarations += privateBindings.declarations
        var resourceParameters: [String] = []

        var storage: [ShaderStorage] = []
        var metalStorage: [[ShaderParameter]] = []
        var layout: [ComputeShaderArtifact.StorageBinding] = []
        var uniformLayout: [ComputeShaderArtifact.UniformBinding] = []
        var sampledTextures: [ComputeShaderArtifact.SampledTextureBinding] = []
        var externalTextures: [IRResourceBinding] = []
        var externalLayouts: [ComputeShaderArtifact.ExternalTextureLayout] = []
        var storageTextures: [ComputeShaderArtifact.StorageTextureBinding] = []
        var samplers: [ComputeShaderArtifact.SamplerBinding] = []
        let globals = module.globalVariables.values
        let reusedBindings = onlyReachableBindings ? [] : reusedResourceBindings(globals)
        let unusedBindings = Set(try globals.indices.filter { index in
            guard !reachableGlobals.contains(index) else { return false }
            if let binding = globals[index].binding, onlyReachableBindings || reusedBindings.contains(binding) { return true }
            let kind = try module.types.value(for: globals[index].type).kind
            if case .bindingArray(_, .dynamic) = kind { return true }
            return kind.isExternalTexture
        })
        let resources = globals.enumerated().filter { !unusedBindings.contains($0.offset) }.map(\.element)
        let handles = try handleResourceGlobals(resources, module: module, bindingArrayLengths: bindingArrayLengths)
        if webGPU { try requireWebGPUShaderTypes(handles.storageTextures, module: module) }
        let arrayCounts = handles.arrayCounts
        let nativeExternal = outputFormat == .wgsl ? [] : handles.externalTextures
        let textureCounts = externalPlaneCounts(nativeExternal, in: arrayCounts)
        let webBases = webGPU ? try webBindingArrayBases(module: module, activeCounts: arrayCounts) : [:]
        let flatBindings = try flatResourceBindingIndices(
            resources.filter { global in
                if case .storage = global.space { return true }
                return false
            },
            capacity: ComputeShaderArtifact.storageBindingCapacity, arrayCounts: arrayCounts)
        let flatUniforms = try flatResourceBindingIndices(
            resources.filter { $0.space == .uniform } + nativeExternal,
            capacity: ComputeShaderArtifact.uniformBindingCapacity, arrayCounts: arrayCounts)
        let flatTextures = try flatResourceBindingIndices(handles.textures + nativeExternal,
            capacity: ComputeShaderArtifact.sampledTextureBindingCapacity, arrayCounts: textureCounts)
        let flatStorageTextures = try flatResourceBindingIndices(handles.storageTextures,
            capacity: ComputeShaderArtifact.storageTextureBindingCapacity, arrayCounts: arrayCounts)
        let flatSamplers = try flatResourceBindingIndices(handles.samplers,
            capacity: ComputeShaderArtifact.samplerBindingCapacity, arrayCounts: arrayCounts)
        let nonFilteringSamplers = try helpers.nonFilteringSamplers(
            module: module, entry: entry.function, sampledTextures: handles.textures)
        let glslBindings = outputFormat == .glsl
            ? try helpers.glslTextureSamplerBindings(module: module, entry: entry.function,
                  textureSlots: expandedBindingIndices(flatTextures, counts: textureCounts),
                  bindingArrayLengths: bindingArrayLengths)
            : []
        declarations += glslBindings.map { $0.declaration }
        if !nativeExternal.isEmpty { declarations += externalTexturePrelude(outputFormat) }
        let metalLengths = ShaderParameter(type: "constant uint*", name: "saga_lengths")
        var runtimeIndex = 0
        var usesGLSLHalfStorage = false
        var workgroupVariables: [WorkgroupVariable] = []
        var workgroupBytes: UInt64 = 0
        var nestedStorageBudget = ShaderBudget()
        let immediate = try ImmediateShaderBinding(module: module, reachable: reachableGlobals, stage: .compute,
                                                   format: outputFormat, budget: &nestedStorageBudget)
        declarations += immediate?.declarations ?? []
        resourceParameters += immediate?.parameters ?? []
        for (index, global) in globals.enumerated() {
            if unusedBindings.contains(index) {
                storage.append(.init(reference: "", writable: false))
                if outputFormat == .msl { metalStorage.append([]) }
                continue
            }
            if global.space == .private {
                storage.append(privateBindings.storage[index] ?? .init(reference: "", writable: false))
                if outputFormat == .msl { metalStorage.append(privateBindings.captures[index] ?? []) }
                continue
            }
            if global.space == .immediate {
                let binding = immediate.flatMap { $0.index == index ? $0 : nil }
                storage.append(binding?.storage ?? .init(reference: "", writable: false))
                if outputFormat == .msl { metalStorage.append(binding?.captures ?? []) }
                continue
            }
            if global.space == .workgroup {
                if !reachableGlobals.contains(index) {
                    storage.append(.init(reference: "", writable: false))
                    if outputFormat == .msl { metalStorage.append([]) }
                    continue
                }
                guard global.memoryDecorations.isEmpty else {
                    throw unsupported("expected a fixed workgroup variable")
                }
                let variable: WorkgroupVariable
                let name = "saga_shared_\(index)"
                let bytes: UInt64
                if let compositeBytes = compositeWorkgroupSizes[global.type] {
                    let type = try shaderValueType(global.type, module: module, format: outputFormat, structures: structures)
                    variable = WorkgroupVariable(name: name, layout: .composite(global.type), format: outputFormat, type: type)
                    bytes = compositeBytes
                } else {
                    let field = try ComputeStorageField(type: global.type, offset: 0, module: module)
                    guard !field.runtime else { throw unsupported("expected a fixed workgroup variable") }
                    if outputFormat == .msl, field.atomic, field.scalar.width == 8 {
                        throw unsupported("Metal does not support 64-bit workgroup atomics")
                    }
                    let type = try field.valueType(in: outputFormat, module: module)
                    variable = WorkgroupVariable(name: name, layout: .flat(field), format: outputFormat, type: type)
                    bytes = field.minimumByteSize
                }
                let (padded, paddingOverflow) = bytes.addingReportingOverflow(15)
                let (total, totalOverflow) = workgroupBytes.addingReportingOverflow(padded & ~UInt64(15))
                guard !paddingOverflow, !totalOverflow, total <= UInt64(UInt32.max) else {
                    throw unsupported("workgroup storage exceeds the representable device limit")
                }
                workgroupBytes = total
                workgroupVariables.append(variable)
                if outputFormat != .msl { declarations.append(variable.declaration) }
                storage.append(.init(reference: variable.reference, bound: variable.field?.count.map { "\($0)u" }, writable: true))
                if outputFormat == .msl {
                    metalStorage.append([ShaderParameter(type: "threadgroup \(variable.type)*", name: variable.name)])
                }
                continue
            }
            if global.space == .uniform {
                guard let binding = global.binding, let flat = flatUniforms[binding] else {
                    throw unsupported("invalid compute uniform binding")
                }
                let uniform = try UniformShaderBinding(global: global, index: index, flat: flat, module: module,
                                                       format: outputFormat, arrayCount: arrayCounts[binding],
                                                       webBindingBase: webBases[binding],
                                                       aggregateAccess: aggregateBuffers.contains(index))
                declarations += uniform.declarations
                resourceParameters += uniform.parameters
                storage.append(uniform.storage)
                metalStorage.append(uniform.captures)
                uniformLayout.append(uniform.reflection)
                usesGLSLHalfStorage = usesGLSLHalfStorage || uniform.usesGLSLHalfStorage
                continue
            }
            if global.space == .handle, let binding = global.binding, let flat = flatSamplers[binding] {
                let sampler = try SamplerShaderBinding(global: global, index: index, flat: flat, module: module,
                                                      format: outputFormat, requiresNonFiltering: nonFilteringSamplers.contains(index),
                                                      arrayCount: arrayCounts[binding], webBindingBase: webBases[binding])
                declarations += sampler.declarations
                resourceParameters += sampler.parameters
                storage.append(sampler.storage)
                metalStorage.append(sampler.captures)
                samplers.append(sampler.reflection)
                continue
            }
            if global.space == .handle, let binding = global.binding, let flat = flatStorageTextures[binding] {
                let texture = try StorageTextureShaderBinding(
                    global: global, index: index, flat: flat, module: module, format: outputFormat,
                    coherent: helpers.controlBarriers.contains(.texture), arrayCount: arrayCounts[binding],
                    webBindingBase: webBases[binding])
                declarations += texture.declarations
                resourceParameters += texture.parameters
                storage.append(texture.storage)
                metalStorage.append(texture.captures)
                storageTextures.append(texture.reflection)
                continue
            }
            if let binding = global.binding, try module.types.value(for: global.type).kind.isExternalTexture {
                let external = try ExternalTextureShaderBinding(global: global, index: index, format: outputFormat,
                    planeBase: flatTextures[binding],
                    parametersIndex: flatUniforms[binding].map { ComputeShaderArtifact.uniformBindingBase + $0 })
                declarations += external.declarations
                resourceParameters += external.parameters
                metalStorage.append(external.captures)
                storage.append(external.storage)
                externalTextures.append(external.binding)
                if let layout = external.layout { externalLayouts.append(layout) }
                continue
            }
            if global.space == .handle {
                guard let binding = global.binding, let flat = flatTextures[binding] else {
                    throw unsupported("invalid compute sampled texture binding")
                }
                let texture = try SampledTextureShaderBinding(global: global, index: index, flat: flat,
                                                              module: module, format: outputFormat,
                                                              samplingSamplers: helpers.floatTextureSamplers[index] ?? [],
                                                              arrayCount: arrayCounts[binding], webBindingBase: webBases[binding])
                declarations += texture.declarations
                resourceParameters += texture.parameters
                storage.append(texture.storage)
                metalStorage.append(texture.captures)
                sampledTextures.append(texture.reflection)
                continue
            }
            guard let binding = global.binding, let flatBinding = flatBindings[binding] else {
                throw unsupported("invalid compute buffer binding")
            }
            let buffer = try StorageBufferShaderBinding(global: global, index: index, flatBinding: flatBinding,
                                                        runtimeIndex: runtimeIndex, metalLengths: metalLengths,
                                                        coherent: helpers.controlBarriers.contains(.storage),
                                                        module: module, format: outputFormat, arrayCount: arrayCounts[binding],
                                                        webBindingBase: webBases[binding], aggregateAccess: aggregateBuffers.contains(index),
                                                        budget: &nestedStorageBudget)
            declarations += buffer.declarations
            resourceParameters += buffer.parameters
            metalStorage.append(buffer.captures)
            storage.append(buffer.storage)
            layout.append(buffer.reflection)
            usesGLSLHalfStorage = usesGLSLHalfStorage || buffer.usesGLSLHalfStorage
            if buffer.reflection.runtimeArrayStride != nil { runtimeIndex += Int(buffer.reflection.bindingArray?.count ?? 1) }
        }

        var workgroupSetup: [String] = []
        if outputFormat != .wgsl && !workgroupVariables.isEmpty {
            let lane = inputs.workgroupInitializationIndex()
            if outputFormat == .msl { workgroupSetup = workgroupVariables.map(\.declaration) }
            let threads = size.x * size.y * size.z
            var setupBudget = ShaderBudget()
            for variable in workgroupVariables {
                workgroupSetup += try variable.initialize(lane: lane, threads: threads, module: module,
                                                           structures: structures, budget: &setupBudget)
            }
            workgroupSetup += try shaderControlBarrier(.workgroup, in: outputFormat).map { "    \($0)" }
        }
        let subgroupSetup = inputs.subgroupSetup()
        declarations += subgroupSetup.declarations
        workgroupSetup += subgroupSetup.body
        var parameters = inputs.parameters + resourceParameters
        let lengthBuffer = outputFormat == .msl && runtimeIndex > 0 ? ComputeShaderArtifact.storageBindingCapacity : nil
        if let lengthBuffer {
            parameters.append("\(metalLengths.type) \(metalLengths.name) [[buffer(\(lengthBuffer))]]")
        }
        let parameterList = parameters.joined(separator: ", ")
        let entryPoint = outputFormat == .glsl ? "main" : "saga_main"
        if outputFormat == .msl { try helpers.bindMetalStorage(metalStorage) }
        let helperDeclarations = try helpers.declarations(module: module, format: outputFormat,
                                                          storage: storage, structures: structures)
        let signature: String
        let attributes: [String]
        switch outputFormat {
        case .msl:
            declarations.insert(contentsOf: ["#include <metal_stdlib>", "using namespace metal;"], at: 0)
            attributes = []
            signature = "kernel void \(entryPoint)(\(parameterList))"
        case .hlsl:
            attributes = (subgroupSize.map { ["[WaveSize(\($0))]"] } ?? []) +
                ["[numthreads(\(size.x), \(size.y), \(size.z))]"]
            signature = "void \(entryPoint)(\(parameterList))"
        case .wgsl:
            let requiresStorageFormats = storageTextures.contains { $0.format.requiresTextureFormatsTier1 }
            let directives: [(Bool, String)] = [
                (!arrayCounts.isEmpty && !webGPU, "enable wgpu_binding_array;"),
                (requiresStandardUniformLayout, "requires uniform_buffer_standard_layout;"),
                (requiresStorageFormats, "requires texture_formats_tier1;"),
                (inputs.requiresSubgroupID, "requires subgroup_id;"),
                (module.requiresSubgroupUniformity, "requires subgroup_uniformity;"),
                (inputs.requiresLinearIndexing, "requires linear_indexing;"),
                (immediate != nil, "requires immediate_address_space;"),
                (usesFloat16, "enable f16;"),
                (requiresSubgroups, "enable subgroups;"),
                (subgroupSize != nil, "enable subgroup_size_control;"),
            ]
            let extensions = directives.compactMap { required, directive in required ? directive : nil }
            declarations.insert(contentsOf: try extensions + helpers.wgslDiagnosticLines(module: module), at: 0)
            attributes = try helpers.wgslDiagnosticLines(module: module, function: function) +
                ["@compute @workgroup_size(\(size.x), \(size.y), \(size.z))"] +
                (subgroupSize.map { ["@subgroup_size(\($0))"] } ?? [])
            signature = "fn \(entryPoint)(\(parameterList))"
        default:
            declarations.insert(contentsOf: [
                "#version 450 core",
            ] + glslUnsigned64Extensions(module: module, storageTextures: storageTextures) +
                (usesFloat16 ? [glslFloat16Extension] : []) +
                (usesGLSLHalfStorage ? [glslStorageFloat16Extension] : []) + helpers.glslExtensions +
                helpers.glslSubgroupExtensions(requiresSubgroups: requiresSubgroups) + [
                "layout(local_size_x = \(size.x), local_size_y = \(size.y), local_size_z = \(size.z)) in;",
            ], at: 0)
            attributes = []
            signature = "void \(entryPoint)()"
        }

        var body = ShaderBodyWriter(module: module, function: function, arguments: inputs.references,
                                    storage: storage, outputFormat: outputFormat,
                                    structures: structures, argumentFields: inputs.argumentFields,
                                    callCaptures: helpers.callCaptures, pointerWrites: helpers.pointerWrites, budget: helpers.budget)
        let lines = try body.write()
        let numeric = try integerPrelude(helpers.integerHelpers.union(body.integerHelpers),
                                         vectors: helpers.integerVectorHelpers.union(body.integerVectorHelpers),
                                         in: outputFormat)
        let source = (declarations + numeric + helperDeclarations + attributes + [signature + " {"] +
                      workgroupSetup + privateBindings.initialization + inputs.initializers + lines + ["}", ""])
            .joined(separator: "\n")
        let atomic64 = usesInt64Atomics(module)
        let needsHLSL66 = subgroupSize != nil || atomic64 || storageTextures.contains { $0.format == .r64Uint }
        let profile = needsHLSL66 ? "cs_6_6" : usesFloat16 ? "cs_6_2" : "cs_6_0"
        return .init(
            artifact: .text(format: outputFormat, source: source), sourceEntryPoint: entry.name,
            entryPoint: entryPoint, workgroupSize: size, storageBindings: layout, uniformBindings: uniformLayout,
            sampledTextures: sampledTextures, storageTextures: storageTextures, externalTextures: externalTextures,
            externalTextureLayouts: externalLayouts,
            samplerBindings: samplers,
            glslTextureSamplerBindings: glslBindings.map { $0.reflection },
            workgroupStorageByteSize: workgroupBytes, metalRuntimeLengthBuffer: lengthBuffer,
            hlslWorkgroupCountBinding: inputs.hlslWorkgroupCountBinding,
            usesTextureBarrier: helpers.controlBarriers.contains(.texture), requiresShaderF16: usesFloat16,
            requiresUniformBufferStandardLayout: requiresStandardUniformLayout, requiresSubgroups: requiresSubgroups,
            requiresSubgroupID: inputs.requiresSubgroupID, hlslProfile: outputFormat == .hlsl ? profile : nil,
            requiresShaderInt64AtomicMinMax: atomic64 && !helpers.requiresShaderInt64AtomicAllOps,
            requiresShaderInt64AtomicAllOps: helpers.requiresShaderInt64AtomicAllOps,
            requiresShaderInt64: moduleUsesWideIntegers(module), immediateLayout: immediate?.layout,
            usesGlobalInvocationIndex: inputs.usesGlobalInvocationIndex, usesWorkgroupIndex: inputs.usesWorkgroupIndex,
            requiredSubgroupSize: subgroupSize, requiresSubgroupUniformity: module.requiresSubgroupUniformity)
    }
}

private struct WorkgroupVariable {
    enum Layout {
        case flat(ComputeStorageField)
        case composite(IRHandle<IRType>)
    }

    let name: String
    let layout: Layout
    let format: ShaderOutputFormat
    let type: String

    var field: ComputeStorageField? {
        guard case .flat(let field) = layout else { return nil }
        return field
    }

    var declaration: String {
        switch format {
        case .msl: return "    threadgroup \(type) \(name)[\(field?.count ?? 1)];"
        case .hlsl: return "groupshared \(type) \(name)\(field?.glslSuffix ?? "");"
        case .glsl: return "shared \(type) \(name)\(field?.glslSuffix ?? "");"
        default: return "var<workgroup> \(name): \(type);"
        }
    }

    var reference: String {
        format == .msl && field?.isArray != true ? "\(name)[0]" : name
    }

    func initialize(
        lane: String, threads: UInt32, module: IRModule, structures: [IRHandle<IRType>: ShaderStructure],
        budget: inout ShaderBudget
    ) throws -> [String] {
        if case .composite(let composite) = layout {
            return try initializeCompositeWorkgroup(composite, reference: reference, lane: lane, threads: threads,
                                                    module: module, format: format, structures: structures, budget: &budget)
        }
        guard let field else { throw ShaderFailure.invalidHandle }
        let backingScalar = field.atomic && field.scalar == .f32 ? .u32 : field.scalar
        let zero: String = switch (backingScalar, format) {
        case (.u32, _): "0u"
        case (.i32, _): "0"
        case (.u64, _): unsigned64Literal(0, in: format)
        case (.i64, _): signed64Literal(0, in: format)
        case (.f16, .hlsl): "((half)0.0f)"
        case (.f16, .msl): "half(0.0f)"
        case (.f16, .glsl): "float16_t(0.0)"
        case (_, .glsl): "0.0"
        default: "0.0f"
        }
        let components = Array(repeating: zero, count: Int(field.width)).joined(separator: ", ")
        let value: String
        if let columns = field.columns {
            let columnType = field.scalar == .f16 ? halfType(field.width, in: format) : floatType(field.width, in: format)
            let column = "\(columnType)(\(components))"
            value = "\(type)(\(Array(repeating: column, count: Int(columns)).joined(separator: ", ")))"
        } else {
            value = field.width == 1 ? zero : "\(type)(\(components))"
        }
        func store(_ location: String) -> String {
            if field.atomic && format == .msl {
                return "atomic_store_explicit(&\(location), \(value), memory_order_relaxed);"
            }
            return "\(location) = \(value);"
        }
        guard let count = field.count else {
            return ["    if (\(lane) == 0u) { \(store(reference)) }"]
        }
        return [
            "    for (uint saga_init = \(lane); saga_init < \(count)u; saga_init += \(threads)u) {",
            "        \(store("\(name)[saga_init]"))",
            "    }",
        ]
    }
}

func shaderControlBarrier(_ flags: IRBarrier, in format: ShaderOutputFormat) throws -> [String] {
    guard !flags.isEmpty, flags.subtracting([.workgroup, .storage, .texture, .subgroup]).isEmpty else {
        throw unsupported("unsupported control barrier memory scope")
    }
    let subgroup = flags.contains(.subgroup)
    let shared = flags.contains(.workgroup)
    let storage = flags.contains(.storage)
    let texture = flags.contains(.texture)
    switch format {
    case .msl:
        let memory = [shared || subgroup ? "mem_flags::mem_threadgroup" : nil,
                      storage ? "mem_flags::mem_device" : nil,
                      texture ? "mem_flags::mem_texture" : nil]
            .compactMap { $0 }.joined(separator: " | ")
        let barrier = subgroup ? "simdgroup_barrier" : "threadgroup_barrier"
        return ["\(barrier)(\(memory));"]
    case .hlsl:
        guard !subgroup else { throw unsupported("HLSL subgroup execution barriers") }
        if shared && (storage || texture) { return ["AllMemoryBarrierWithGroupSync();"] }
        return [shared ? "GroupMemoryBarrierWithGroupSync();" : "DeviceMemoryBarrierWithGroupSync();"]
    case .glsl:
        if subgroup { return ["subgroupBarrier();"] }
        return [shared ? "memoryBarrierShared();" : nil, storage ? "memoryBarrierBuffer();" : nil,
                texture ? "memoryBarrierImage();" : nil, "barrier();"].compactMap { $0 }
    default:
        if subgroup {
            guard flags == .subgroup else { throw unsupported("Combined subgroup memory kinds in WGSL") }
            return ["subgroupBarrier();"]
        }
        return [shared ? "workgroupBarrier();" : nil, storage ? "storageBarrier();" : nil,
                texture ? "textureBarrier();" : nil].compactMap { $0 }
    }
}

func reusedResourceBindings(_ globals: [IRGlobalVariable]) -> Set<IRResourceBinding> {
    var seen: Set<IRResourceBinding> = []
    var reused: Set<IRResourceBinding> = []
    for global in globals {
        if let binding = global.binding, !seen.insert(binding).inserted { reused.insert(binding) }
    }
    return reused
}

func flatResourceBindingIndices(
    _ globals: [IRGlobalVariable], capacity: UInt32, arrayCounts: [IRResourceBinding: UInt32] = [:]
) throws -> [IRResourceBinding: UInt32] {
    let declared = globals.compactMap(\.binding)
    let bindings = Set(declared)
    guard bindings.count <= Int(capacity) else { throw unsupported("resource binding count exceeds \(capacity)") }
    guard declared.count == globals.count, bindings.allSatisfy({ $0.group < ComputeShaderArtifact.portableBindGroupCount &&
              $0.binding < ComputeShaderArtifact.portableBindingIndexLimit })
    else { throw unsupported("expected bindings in groups 0..<4 and binding indices 0..<1000") }

    if bindings.contains(where: { arrayCounts[$0] != nil }) {
        // Pack the entire namespace so that fixed arrays always get contiguous slots.
        var indices: [IRResourceBinding: UInt32] = [:]
        var next: UInt32 = 0
        for binding in bindings.sorted(by: { ($0.group, $0.binding) < ($1.group, $1.binding) }) {
            let count = arrayCounts[binding] ?? 1
            guard count > 0, count <= capacity - next else {
                throw unsupported("resource binding count exceeds \(capacity) after expanding binding arrays")
            }
            indices[binding] = next
            next += count
        }
        return indices
    }

    // Keep existing low group-zero indices, then fill free slots in logical binding order.
    let legacy = bindings.filter { $0.group == 0 && $0.binding < capacity }
    var indices = Dictionary(uniqueKeysWithValues: legacy.map { ($0, $0.binding) })
    let occupied = Set(indices.values)
    let remaining = bindings.filter { indices[$0] == nil }.sorted {
        ($0.group, $0.binding) < ($1.group, $1.binding)
    }
    for (binding, slot) in zip(remaining, (0..<capacity).filter { !occupied.contains($0) }) {
        indices[binding] = slot
    }
    return indices
}

func expandedBindingIndices(_ bases: [IRResourceBinding: UInt32], counts: [IRResourceBinding: UInt32]) -> Set<UInt32> {
    Set(bases.flatMap { binding, base in (0..<(counts[binding] ?? 1)).map { base + $0 } })
}

func externalPlaneCounts(_ globals: [IRGlobalVariable], in counts: [IRResourceBinding: UInt32]) -> [IRResourceBinding: UInt32] {
    var result = counts
    for global in globals {
        if let binding = global.binding { result[binding] = max(result[binding] ?? 0, 3) }
    }
    return result
}

func bindingElementKind(_ type: IRHandle<IRType>, module: IRModule) throws -> IRTypeKind {
    let kind = try module.types.value(for: type).kind
    if case .bindingArray(let base, _) = kind { return try module.types.value(for: base).kind }
    return kind
}

func requireWebGPUShaderTypes(_ textures: [IRGlobalVariable], module: IRModule) throws {
    if moduleUsesWideIntegers(module) {
        throw ShaderFailure.unsupported("64-bit integer types require native shader output")
    }
    if try textures.contains(where: {
        let storage = try bindingElementKind($0.type, module: module).storageTexture
        return storage?.access.contains(.atomic) == true || storage?.format == .r64Uint
    }) {
        throw ShaderFailure.unsupported("Atomic and R64Uint storage textures require native shader output")
    }
}

func usesInt64Atomics(_ module: IRModule) -> Bool {
    module.types.values.contains {
        if case .atomic(let scalar) = $0.kind { return scalar.width == 8 }
        return false
    }
}

func moduleUsesWideIntegers(_ module: IRModule) -> Bool {
    module.types.values.contains {
        switch $0.kind {
        case .scalar(.u64), .scalar(.i64), .vector(_, .u64), .vector(_, .i64), .atomic(.u64), .atomic(.i64): true
        default: false
        }
    }
}

func glslUnsigned64Extensions(module: IRModule, storageTextures: [ComputeShaderArtifact.StorageTextureBinding]) -> [String] {
    let image64 = storageTextures.contains { $0.format == .r64Uint }
    guard image64 || moduleUsesWideIntegers(module) else { return [] }
    var extensions = ["#extension GL_ARB_gpu_shader_int64 : require"]
    if image64 { extensions.append("#extension GL_EXT_shader_image_int64 : require") }
    if usesInt64Atomics(module) { extensions.append("#extension GL_EXT_shader_atomic_int64 : require") }
    return extensions
}

func resourceBindingArrayCount(
    _ size: IRArraySize, binding: IRResourceBinding, location: ShaderLocation?,
    bindingArrayLengths: [IRResourceBinding: UInt32]
) throws -> UInt32 {
    switch size {
    case .constant(let count): return count
    case .dynamic:
        guard let count = bindingArrayLengths[binding], count > 0 else {
            throw ShaderFailure.diagnostics([.init(severity: .error, code: .invalidBinding,
                message: "Missing layout count for binding array at group \(binding.group), binding \(binding.binding)",
                location: location)])
        }
        return count
    case .override: throw ShaderFailure.unsupported("Unspecialized binding array length")
    }
}

func handleResourceGlobals(
    _ globals: [IRGlobalVariable], module: IRModule,
    bindingArrayLengths: [IRResourceBinding: UInt32] = [:]
) throws
    -> (textures: [IRGlobalVariable], samplers: [IRGlobalVariable], storageTextures: [IRGlobalVariable],
        externalTextures: [IRGlobalVariable],
        arrayCounts: [IRResourceBinding: UInt32])
{
    var textures: [IRGlobalVariable] = []
    var samplers: [IRGlobalVariable] = []
    var storageTextures: [IRGlobalVariable] = []
    var externalTextures: [IRGlobalVariable] = []
    var arrayCounts: [IRResourceBinding: UInt32] = [:]
    for global in globals {
        var kind = try module.types.value(for: global.type).kind
        if case .bindingArray(let base, let size) = kind {
            guard global.space == .handle || global.space.isBufferAddressSpace,
                  let binding = global.binding else {
                throw ShaderFailure.unsupported("Binding array shape or address space is unsupported by this shader target")
            }
            arrayCounts[binding] = try resourceBindingArrayCount(size, binding: binding, location: global.location,
                                                                 bindingArrayLengths: bindingArrayLengths)
            kind = try module.types.value(for: base).kind
        }
        guard global.space == .handle else { continue }
        switch kind {
        case let kind where kind.isSampledTexture: textures.append(global)
        case let kind where kind.isExternalTexture: externalTextures.append(global)
        case let kind where kind.storageTexture != nil: storageTextures.append(global)
        case .sampler: samplers.append(global)
        default: throw ShaderFailure.unsupported("Unsupported texture or sampler binding")
        }
    }
    return (textures, samplers, storageTextures, externalTextures, arrayCounts)
}

func hlslBindingArrayRegisterBase(_ binding: IRResourceBinding) -> UInt32 {
    // Reserve a disjoint range per logical binding so stages agree even with different live globals.
    let stride = max(ComputeShaderArtifact.sampledTextureBindingCapacity, ComputeShaderArtifact.samplerBindingCapacity,
                     ComputeShaderArtifact.storageTextureBindingCapacity, ComputeShaderArtifact.storageBindingCapacity,
                     ComputeShaderArtifact.uniformBindingCapacity)
    return 1024 + binding.binding * stride
}

private struct ShaderHandleBinding {
    let kind: IRTypeKind
    let count: UInt32?
    let webBindingBase: UInt32?

    init(_ global: IRGlobalVariable, module: IRModule, arrayCount: UInt32?, webBindingBase: UInt32?) throws {
        self.webBindingBase = webBindingBase
        let declared = try module.types.value(for: global.type).kind
        if case .bindingArray(let base, _) = declared {
            guard let arrayCount else { throw ShaderFailure.invalidHandle }
            kind = try module.types.value(for: base).kind
            count = arrayCount
        } else {
            kind = declared
            count = nil
        }
    }

    func metalType(_ element: String) -> String {
        count.map { "array<\(element), \($0)>" } ?? element
    }

    func wgslDeclarations(_ element: String, name: String, binding: IRResourceBinding) -> [String] {
        if let webBindingBase, let count {
            return (0..<count).map {
                "@group(\(binding.group)) @binding(\(webBindingBase + $0)) var \(name)_\($0): \(element);"
            }
        }
        let type = count.map { "binding_array<\(element), \($0)>" } ?? element
        return ["@group(\(binding.group)) @binding(\(binding.binding)) var \(name): \(type);"]
    }

    func glslDeclarations(name: String, flat: UInt32, declaration: (String, UInt32) -> String) -> [String] {
        guard let count else { return [declaration(name, flat)] }
        return (0..<count).map { declaration("\(name)_\($0)", flat + $0) }
    }

    func splitsArray(in format: ShaderOutputFormat) -> Bool {
        webBindingBase != nil || (format == .glsl && count != nil)
    }

    func reflection(format: ShaderOutputFormat, binding: IRResourceBinding) -> ComputeShaderArtifact.BindingArrayLayout? {
        count.map { .init(count: $0, hlslRegisterBase: format == .hlsl ? hlslBindingArrayRegisterBase(binding) : nil,
                          webBindingBase: webBindingBase) }
    }

    func hlslDeclaration(_ type: String, name: String, register: Character, binding: IRResourceBinding) -> String {
        let suffix = count.map { "[\($0)]" } ?? ""
        let index = count == nil ? binding.binding : hlslBindingArrayRegisterBase(binding)
        return "\(type) \(name)\(suffix) : register(\(register)\(index), space\(binding.group));"
    }

}

struct SamplerShaderBinding {
    var declarations: [String] = []
    var parameters: [String] = []
    var captures: [ShaderParameter] = []
    let storage: ShaderStorage
    let reflection: ComputeShaderArtifact.SamplerBinding

    init(global: IRGlobalVariable, index: Int, flat: UInt32, module: IRModule, format: ShaderOutputFormat,
         requiresNonFiltering: Bool, arrayCount: UInt32?, webBindingBase: UInt32?) throws {
        let resource = try ShaderHandleBinding(global, module: module, arrayCount: arrayCount, webBindingBase: webBindingBase)
        guard let binding = global.binding, case .sampler(let comparison) = resource.kind else {
            throw unsupported("invalid sampler binding")
        }
        let name = "saga_sampler_\(index)"
        switch format {
        case .msl:
            let type = resource.metalType("sampler")
            parameters.append("\(type) \(name) [[sampler(\(flat))]]")
            captures.append(.init(type: type, name: name))
        case .wgsl:
            let type = comparison ? "sampler_comparison" : "sampler"
            declarations += resource.wgslDeclarations(type, name: name, binding: binding)
        case .hlsl:
            let type = comparison ? "SamplerComparisonState" : "SamplerState"
            declarations.append(resource.hlslDeclaration(type, name: name, register: "s", binding: binding))
        default: break
        }
        storage = .init(reference: name, writable: false, bindingArrayCount: resource.count, splitBindingArray: resource.splitsArray(in: format))
        reflection = .init(binding: binding, flatBindingIndex: format == .msl ? flat : nil, isComparison: comparison,
                           requiresNonFiltering: requiresNonFiltering, bindingArray: resource.reflection(format: format, binding: binding))
    }
}

struct SampledTextureShaderBinding {
    var declarations: [String] = []
    var parameters: [String] = []
    var captures: [ShaderParameter] = []
    let storage: ShaderStorage
    let reflection: ComputeShaderArtifact.SampledTextureBinding

    init(global: IRGlobalVariable, index: Int, flat: UInt32, module: IRModule, format: ShaderOutputFormat,
         samplingSamplers: Set<IRResourceBinding>, arrayCount: UInt32?, webBindingBase: UInt32?) throws {
        let resource = try ShaderHandleBinding(global, module: module, arrayCount: arrayCount, webBindingBase: webBindingBase)
        let kind = resource.kind
        guard let scalar = kind.sampledTextureScalar, let binding = global.binding else {
            throw unsupported("invalid sampled texture binding")
        }
        let name = "saga_texture_\(index)"
        let depth = kind.isDepthTexture
        let multisampled = kind.isMultisampled2D
        let arrayed = kind.isArrayedTexture
        let cube = kind.isCubeTexture
        let volume = kind.isVolumeTexture
        let line = kind.isLineTexture
        let shapeName = cube ? "cube" : volume ? "3d" : line ? "1d" : "2d"
        let nativeScalar = scalar == .i32 ? "int" : scalar == .u32 ? "uint" : "float"
        switch format {
        case .msl:
            let shape = (depth ? "depth" : "texture") + shapeName +
                (multisampled ? "_ms" : arrayed ? "_array" : "")
            let access = multisampled ? "read" : "sample"
            let type = resource.metalType("\(shape)<\(nativeScalar), access::\(access)>")
            let parameter = ShaderParameter(type: type, name: name)
            parameters.append("\(parameter.type) \(name) [[texture(\(flat))]]")
            captures.append(parameter)
        case .hlsl:
            let type = depth ? nativeScalar : "\(nativeScalar)4"
            let shape = "Texture" + (cube ? "Cube" : shapeName.uppercased()) +
                (multisampled ? "MS" : arrayed ? "Array" : "")
            declarations.append(resource.hlslDeclaration("\(shape)<\(type)>", name: name, register: "t", binding: binding))
        case .glsl:
            let prefix = scalar == .i32 ? "i" : scalar == .u32 ? "u" : ""
            let type = prefix + "sampler" + (cube ? "Cube" : shapeName.uppercased()) +
                (multisampled ? "MS" : arrayed ? "Array" : "")
            declarations += resource.glslDeclarations(name: name, flat: flat) {
                "layout(binding=\($1)) uniform \(type) \($0);"
            }
        case .wgsl:
            let shape = (multisampled ? "_multisampled_2d" : "_\(shapeName)") + (arrayed ? "_array" : "")
            let element = scalar == .i32 ? "i32" : scalar == .u32 ? "u32" : "f32"
            let type = depth ? "texture_depth\(shape)" : "texture\(shape)<\(element)>"
            declarations += resource.wgslDeclarations(type, name: name, binding: binding)
        default: throw unsupported("sampled texture target")
        }
        storage = .init(reference: name, writable: false, bindingArrayCount: resource.count, splitBindingArray: resource.splitsArray(in: format))
        reflection = .init(binding: binding, flatBindingIndex: format == .msl || format == .glsl ? flat : nil,
                           isDepth: depth, isMultisampled: multisampled, isArrayed: arrayed, sampleKind: scalar.kind,
                           dimension: cube ? .cube : volume ? .d3 : line ? .d1 : .d2, samplingSamplers: samplingSamplers,
                           bindingArray: resource.reflection(format: format, binding: binding))
    }
}

struct StorageTextureShaderBinding {
    var declarations: [String] = []
    var parameters: [String] = []
    var captures: [ShaderParameter] = []
    let storage: ShaderStorage
    let reflection: ComputeShaderArtifact.StorageTextureBinding

    init(global: IRGlobalVariable, index: Int, flat: UInt32, module: IRModule, format: ShaderOutputFormat,
         coherent: Bool = false, arrayCount: UInt32?, webBindingBase: UInt32?,
         metalAtomicWriteGuard: ShaderParameter? = nil) throws {
        let resource = try ShaderHandleBinding(global, module: module, arrayCount: arrayCount, webBindingBase: webBindingBase)
        let kind = resource.kind
        guard let binding = global.binding, let texture = kind.storageTexture else {
            throw unsupported("storage texture type")
        }
        let arrayed = kind.isArrayedTexture
        let volume = kind.isVolumeTexture
        let line = kind.isLineTexture
        let shapeName = volume ? "3d" : line ? "1d" : "2d"
        let name = "saga_texture_\(index)"
        let scalar = storageTextureScalarName(texture.scalar, format: format)
        let access: String
        let qualifier: String
        switch texture.access {
        case .load: (access, qualifier) = ("read", "readonly ")
        case .store: (access, qualifier) = ("write", "writeonly ")
        default: (access, qualifier) = ("read_write", "")
        }
        let emittedIndex: UInt32?
        let writeGuard = texture.access.contains(.atomic) ? metalAtomicWriteGuard : nil
        switch format {
        case .msl:
            let slot = ComputeShaderArtifact.sampledTextureBindingCapacity + flat
            let shape = "texture" + shapeName + (arrayed ? "_array" : "")
            let type = resource.metalType("\(shape)<\(scalar), access::\(access)>")
            let parameter = ShaderParameter(type: type, name: name)
            parameters.append("\(parameter.type) \(name) [[texture(\(slot))]]")
            captures.append(parameter)
            if let writeGuard { captures.append(writeGuard) }
            emittedIndex = slot
        case .hlsl:
            let normalization = switch texture.format {
            case .r8Unorm, .rg8Unorm, .rgba8Unorm, .bgra8Unorm, .r16Unorm, .rg16Unorm, .rgba16Unorm, .rgb10a2Unorm: "unorm "
            case .r8Snorm, .rg8Snorm, .rgba8Snorm, .r16Snorm, .rg16Snorm, .rgba16Snorm: "snorm "
            default: ""
            }
            // HLSL typed UAVs use a four-lane value for R11G11B10Float.
            let components = texture.components == 3 ? 4 : texture.components
            let element = components == 1 ? scalar : "\(scalar)\(components)"
            let shape = "RWTexture" + shapeName.uppercased() + (arrayed ? "Array" : "")
            declarations.append(resource.hlslDeclaration("\(shape)<\(normalization)\(element)>", name: name,
                                                         register: "u", binding: binding))
            emittedIndex = nil
        case .glsl:
            let layout = try glslStorageTextureLayout(texture.format)
            let prefix = texture.scalar == .u64 ? "u64" : texture.scalar == .u32 ? "u" : texture.scalar == .i32 ? "i" : ""
            let type = prefix + "image" + shapeName.uppercased() + (arrayed ? "Array" : "")
            let memory = coherent && texture.access.contains(.store) ? "coherent " : ""
            declarations += resource.glslDeclarations(name: name, flat: flat) {
                "layout(\(layout), binding=\($1)) \(memory)\(qualifier)uniform \(type) \($0);"
            }
            emittedIndex = flat
        case .wgsl:
            let layout = texture.format.rawValue.lowercased()
            let shape = "texture_storage_" + shapeName + (arrayed ? "_array" : "")
            let wgslAccess = texture.access.contains(.atomic) ? "atomic" : access
            declarations += resource.wgslDeclarations("\(shape)<\(layout), \(wgslAccess)>", name: name, binding: binding)
            emittedIndex = nil
        default: throw unsupported("storage texture target")
        }
        storage = .init(reference: name, writable: texture.access.contains(.store),
                        bindingArrayCount: resource.count, splitBindingArray: resource.splitsArray(in: format),
                        metalAtomicWriteGuard: writeGuard?.name)
        reflection = .init(binding: binding, format: texture.format, access: texture.access,
                           flatBindingIndex: emittedIndex, isArrayed: arrayed, dimension: volume ? .d3 : line ? .d1 : .d2,
                           bindingArray: resource.reflection(format: format, binding: binding))
    }
}

private func storageTextureScalarName(_ scalar: IRScalar, format: ShaderOutputFormat) -> String {
    switch scalar {
    case .u32: "uint"
    case .u64: unsigned64Type(in: format)
    case .i32: "int"
    default: "float"
    }
}

private func glslStorageTextureLayout(_ format: IRStorageFormat) throws -> String {
    switch format {
    case .r8Unorm: "r8"
    case .r8Snorm: "r8_snorm"
    case .r8Uint: "r8ui"
    case .r8Sint: "r8i"
    case .rg8Unorm: "rg8"
    case .rg8Snorm: "rg8_snorm"
    case .rg8Uint: "rg8ui"
    case .rg8Sint: "rg8i"
    case .rgba8Unorm: "rgba8"
    case .bgra8Unorm: throw unsupported("GLSL storage images do not support BGRA8Unorm")
    case .rgba8Snorm: "rgba8_snorm"
    case .rgba8Uint: "rgba8ui"
    case .rgba8Sint: "rgba8i"
    case .rgba16Uint: "rgba16ui"
    case .rgba16Sint: "rgba16i"
    case .rgba16Float: "rgba16f"
    case .r32Uint: "r32ui"
    case .r64Uint: "r64ui"
    case .r32Sint: "r32i"
    case .rgb10a2Unorm: "rgb10_a2"
    case .rgb10a2Uint: "rgb10_a2ui"
    case .rg11b10Ufloat: "r11f_g11f_b10f"
    case .r16Unorm: "r16"
    case .r16Snorm: "r16_snorm"
    case .rg16Unorm: "rg16"
    case .rg16Snorm: "rg16_snorm"
    case .rgba16Unorm: "rgba16"
    case .rgba16Snorm: "rgba16_snorm"
    case .r16Uint: "r16ui"
    case .r16Sint: "r16i"
    case .rg16Uint: "rg16ui"
    case .rg16Sint: "rg16i"
    case .r16Float: "r16f"
    case .r32Float: "r32f"
    case .rg32Uint: "rg32ui"
    case .rg32Sint: "rg32i"
    case .rg16Float: "rg16f"
    case .rg32Float: "rg32f"
    case .rgba32Uint: "rgba32ui"
    case .rgba32Sint: "rgba32i"
    case .rgba32Float: "rgba32f"
    }
}

struct UniformShaderBinding {
    var declarations: [String] = []
    var parameters: [String] = []
    var captures: [ShaderParameter] = []
    let storage: ShaderStorage
    let reflection: ComputeShaderArtifact.UniformBinding
    let usesGLSLHalfStorage: Bool

    init(global: IRGlobalVariable, index: Int, flat: UInt32, module: IRModule, format: ShaderOutputFormat,
         arrayCount: UInt32?, webBindingBase: UInt32?, aggregateAccess: Bool) throws {
        guard global.space == .uniform, let binding = global.binding else { throw unsupported("invalid uniform buffer") }
        let flatBinding = ComputeShaderArtifact.uniformBindingBase + flat
        let name = "saga_buffer_\(index)"
        let type = "SagaUniform\(index)"
        if let arrayCount {
            var budget = ShaderBudget()
            let array = try BufferBindingArray(global: global, index: index, flat: flatBinding, module: module,
                                               format: format, count: arrayCount, webBindingBase: webBindingBase, budget: &budget)
            declarations = array.declarations
            parameters = array.parameters
            captures = array.captures
            storage = array.storage
            usesGLSLHalfStorage = false
            reflection = .init(binding: binding, flatBindingIndex: format == .msl || format == .glsl ? flatBinding : nil,
                               minimumByteSize: array.minimumByteSize, bindingArray: array.layout)
            return
        }
        if try aggregateAccess || module.uniformBufferStandardLayout || hasNestedBufferLayout(global.type, module: module) {
            var budget = ShaderBudget()
            var types = NestedBufferTypes(module: module, format: format, prefix: "SagaNestedUniform\(index)_")
            let root = try types.describe(global.type, budget: &budget)
            guard root.size <= 65_536 else { throw unsupported("uniform buffer exceeds 64 KiB") }
            let registers = (root.size + 15) / 16
            switch format {
            case .msl:
                captures = [.init(type: "constant uint*", name: name)]
                parameters = ["constant uint* \(name) [[buffer(\(flatBinding))]]"]
            case .hlsl:
                declarations = [
                    "cbuffer \(type) : register(b\(binding.binding), space\(binding.group)) {",
                    "    uint4 \(name)[\(registers)];", "};",
                ]
            case .wgsl:
                declarations = types.declarations
                declarations.append("@group(\(binding.group)) @binding(\(binding.binding)) var<uniform> \(name): \(root.name);")
            case .glsl:
                declarations = ["layout(std140, binding = \(flatBinding)) uniform \(type) { uvec4 words[\(registers)]; } \(name);"]
            default: throw unsupported("nested uniform target")
            }
            usesGLSLHalfStorage = false
            storage = .init(reference: name, writable: false,
                            nested: .init(name: name, type: global.type, runtimeLength: nil, space: .uniform))
            reflection = .init(binding: binding, flatBindingIndex: format == .msl || format == .glsl ? flatBinding : nil,
                               minimumByteSize: root.size)
            return
        }
        let span: UInt32?
        let fields: [ComputeStorageField]
        if case .structure(let members, let size) = try module.types.value(for: global.type).kind {
            span = size
            fields = try members.map { try ComputeStorageField(type: $0.type, offset: $0.offset, module: module) }
        } else {
            span = nil
            fields = [try ComputeStorageField(type: global.type, offset: 0, module: module)]
        }
        guard let first = fields.first, fields.allSatisfy({ !$0.atomic && !$0.runtime }) else {
            throw unsupported("expected a fixed uniform buffer")
        }
        usesGLSLHalfStorage = format == .glsl && fields.contains { $0.scalar == .f16 }
        let minimum = span.map(UInt64.init) ?? first.minimumByteSize
        guard minimum <= 65_536 else { throw unsupported("uniform buffer exceeds 64 KiB") }
        let names = span == nil ? ["value"] : fields.indices.map { "saga_member_\($0)" }
        // std140 rounds matrix column strides up to sixteen bytes.
        let packedGLSL = format == .glsl && fields.contains {
            $0.columns != nil && ($0.width == 2 || $0.scalar == .f16)
        }
        switch format {
        case .msl:
            captures = [.init(type: "constant uint*", name: name)]
            parameters = ["constant uint* \(name) [[buffer(\(flatBinding))]]"]
        case .hlsl:
            declarations = [
                "cbuffer \(type) : register(b\(binding.binding), space\(binding.group)) {",
                "    uint4 \(name)[\((minimum + 15) / 16)];", "};",
            ]
        case .wgsl:
            if let span {
                declarations.append("struct \(type) {")
                for (position, field) in fields.enumerated() {
                    let end = position + 1 < fields.count ? fields[position + 1].offset : span
                    let size = end - field.offset
                    let padding = UInt64(size) > field.minimumByteSize ? "@size(\(size)) " : ""
                    declarations.append("    \(padding)\(names[position]): \(try field.valueType(in: format, module: module)),")
                }
                declarations.append("}")
            }
            let valueType = try span == nil ? first.valueType(in: format, module: module) : type
            declarations.append("@group(\(binding.group)) @binding(\(binding.binding)) var<uniform> \(name): \(valueType);")
        default:
            let members: String
            if packedGLSL {
                members = "uvec4 words[\((minimum + 15) / 16)];"
            } else {
                members = try fields.enumerated().map { position, field in
                    "layout(offset = \(field.offset)) \(try field.valueType(in: format, module: module)) \(names[position])\(field.glslSuffix);"
                }.joined(separator: " ")
            }
            declarations = ["layout(std140, binding = \(flatBinding)) uniform \(type) { \(members) } \(name);"]
        }
        let locations = fields.enumerated().map { position, field in
            let bound = field.count.map { "\($0)u" }
            if format == .msl || format == .hlsl || packedGLSL {
                let reference: String
                if packedGLSL { reference = "\(name).words" }
                else if format == .msl, field.scalar == .f16 || field.scalar.width == 8 {
                    reference = "reinterpret_cast<constant \(field.scalar == .f16 ? "half" : "ulong")*>(\(name))"
                }
                else { reference = name }
                return ShaderStorage(reference: reference, bound: bound, writable: false,
                                     words: field.wordLayout(addressing: format == .msl ? .structured : .uniformRegisters))
            }
            let reference = format == .wgsl && span == nil ? name : "\(name).\(names[position])"
            return ShaderStorage(reference: reference, bound: bound, writable: false)
        }
        storage = span == nil ? locations[0] : .init(reference: name, writable: false, members: locations)
        reflection = .init(binding: binding, flatBindingIndex: format == .msl || format == .glsl ? flatBinding : nil,
                           minimumByteSize: minimum)
    }
}

struct StorageBufferShaderBinding {
    var declarations: [String] = []
    var parameters: [String] = []
    var captures: [ShaderParameter] = []
    let storage: ShaderStorage
    let reflection: ComputeShaderArtifact.StorageBinding
    let usesGLSLHalfStorage: Bool

    init(
        global: IRGlobalVariable, index: Int, flatBinding: UInt32, runtimeIndex: Int,
        metalLengths: ShaderParameter, coherent: Bool, module: IRModule, format outputFormat: ShaderOutputFormat,
        arrayCount: UInt32?, webBindingBase: UInt32?, aggregateAccess: Bool,
        budget: inout ShaderBudget
    ) throws {
        let writable: Bool
        switch global.space {
        case .storage(.load): writable = false
        case .storage([.load, .store]): writable = true
        default: throw unsupported("expected a uniform or storage buffer binding")
        }
        guard let binding = global.binding,
              global.initializer == nil, global.memoryDecorations.isEmpty
        else { throw unsupported("invalid compute buffer binding") }
        if let arrayCount {
            let array = try BufferBindingArray(global: global, index: index, flat: flatBinding, module: module,
                                               format: outputFormat, count: arrayCount, runtimeSlot: runtimeIndex, metalLengths: metalLengths,
                                               webBindingBase: webBindingBase, coherent: coherent, budget: &budget)
            declarations = array.declarations
            parameters = array.parameters
            captures = array.captures
            storage = array.storage
            usesGLSLHalfStorage = array.usesGLSLHalfStorage
            reflection = .init(binding: binding, flatBindingIndex: outputFormat == .msl || outputFormat == .glsl ? flatBinding : nil,
                               minimumByteSize: array.minimumByteSize, isReadOnly: !writable,
                               runtimeArrayStride: array.runtimeStride, runtimeArrayOffset: array.runtimeOffset,
                               hlslUsesByteAddressBuffer: outputFormat == .hlsl, bindingArray: array.layout)
            return
        }
        if try aggregateAccess || hasNestedBufferLayout(global.type, module: module) {
            let nested = try NestedStorageBinding(global: global, index: index, binding: binding, flat: flatBinding,
                                                  runtimeSlot: runtimeIndex, metalLengths: metalLengths, writable: writable,
                                                  coherent: coherent, module: module, format: outputFormat, budget: &budget)
            declarations = nested.declarations
            parameters = nested.parameters
            captures = nested.captures
            storage = nested.storage
            reflection = nested.reflection
            usesGLSLHalfStorage = nested.usesGLSLHalfStorage
            return
        }
        let structureSpan: UInt32?
        let fields: [ComputeStorageField]
        if case .structure(let members, let span) = try module.types.value(for: global.type).kind {
            structureSpan = span
            fields = try members.map { try ComputeStorageField(type: $0.type, offset: $0.offset, module: module) }
        } else {
            structureSpan = nil
            fields = [try ComputeStorageField(type: global.type, offset: 0, module: module)]
        }
        guard let tail = fields.last else { throw unsupported("empty storage structure") }
        if outputFormat == .msl, fields.contains(where: { $0.atomic && $0.scalar == .i64 }) {
            throw ShaderFailure.unsupported("Metal storage min/max atomics require unsigned 64-bit values")
        }
        let structured = structureSpan != nil
        let hasHalfFields = fields.contains { $0.scalar == .f16 }
        usesGLSLHalfStorage = outputFormat == .glsl && hasHalfFields
        let hasWideValues = fields.contains { !$0.atomic && $0.scalar.width == 8 }
        let rawHLSL = outputFormat == .hlsl && (hasHalfFields || hasWideValues || (structured && fields.contains(where: \.atomic)))
        let name = "saga_buffer_\(index)"
        let typeName = "SagaBuffer\(index)"
        let minimum = structureSpan.map(UInt64.init) ?? tail.minimumByteSize
        let fieldNames = structured ? fields.indices.map { "saga_member_\($0)" } : ["value"]
        let runtimeLength: String?
        switch outputFormat {
        case .msl:
            let qualifier = writable ? "device" : "device const"
            let element = tail.atomic && !structured ? tail.metalAtomicType : "uint"
            let pointer = ShaderParameter(type: "\(qualifier) \(element)*", name: name)
            parameters.append("\(pointer.type) \(name) [[buffer(\(flatBinding))]]")
            captures = tail.runtime ? [pointer, metalLengths] : [pointer]
            runtimeLength = tail.runtime ? "\(metalLengths.name)[\(runtimeIndex)]" : nil
        case .hlsl:
            let register = writable ? "u" : "t"
            let type: String
            if rawHLSL {
                type = writable ? "RWByteAddressBuffer" : "ByteAddressBuffer"
            } else {
                let buffer = writable ? "RWStructuredBuffer" : "StructuredBuffer"
                let element = tail.atomic ? try tail.valueType(in: .hlsl, module: module) : "uint"
                type = "\(buffer)<\(element)>"
            }
            declarations.append("\(type) \(name) : register(\(register)\(binding.binding), space\(binding.group));")
            if tail.runtime {
                let helper = "saga_length_\(index)"
                let unit: UInt32 = rawHLSL ? 1 : 4
                let offset = tail.offset / unit
                let stride = tail.stride / unit
                let elements = offset == 0 ? "count" : "(count - \(offset)u)"
                let count = stride == 1 ? elements : "\(elements) / \(stride)u"
                declarations += [
                    "uint \(helper)() {",
                    rawHLSL ? "    uint count;" : "    uint count, stride;",
                    "    \(name).GetDimensions(\(rawHLSL ? "count" : "count, stride"));",
                ]
                if offset > 0 { declarations.append("    if (count < \(offset)u) { return 0u; }") }
                declarations += ["    return \(count);", "}"]
                runtimeLength = "\(helper)()"
            } else {
                runtimeLength = nil
            }
        case .wgsl:
            let access = writable ? "read_write" : "read"
            let type = try structured ? typeName : tail.valueType(in: .wgsl, module: module)
            if let structureSpan {
                let firstAlignment: String
                if tail.runtime {
                    let natural = fields.map(\.alignment).max() ?? 4
                    let end = UInt64(tail.offset) + tail.minimumByteSize
                    var align = UInt64(natural)
                    func rounded() -> UInt64 { (end + align - 1) / align * align }
                    while rounded() < UInt64(structureSpan) { align *= 2 }
                    guard rounded() == UInt64(structureSpan) else { throw unsupported("runtime structure alignment") }
                    firstAlignment = align > UInt64(natural) ? "@align(\(align)) " : ""
                } else { firstAlignment = "" }
                declarations.append("struct \(typeName) {")
                for (position, field) in fields.enumerated() {
                    var padding = ""
                    let end = position + 1 < fields.count ? fields[position + 1].offset :
                        (field.runtime ? nil : structureSpan)
                    if let end {
                        let memberSize = end - field.offset
                        if UInt64(memberSize) > field.minimumByteSize { padding = "@size(\(memberSize)) " }
                    }
                    let type = try field.valueType(in: .wgsl, module: module)
                    let align = position == 0 ? firstAlignment : ""
                    declarations.append("    \(align)\(padding)\(fieldNames[position]): \(type),")
                }
                declarations.append("}")
            }
            declarations.append("@group(\(binding.group)) @binding(\(binding.binding)) var<storage, \(access)> \(name): \(type);")
            let array = structured ? "\(name).\(fieldNames.last!)" : name
            runtimeLength = tail.runtime ? "arrayLength(&\(array))" : nil
        default:
            let coherence = coherent && writable ? "coherent " : ""
            let qualifier = coherence + (writable ? "buffer" : "readonly buffer")
            let members = try fields.enumerated().map { position, field in
                let offset = structured ? "layout(offset = \(field.offset)) " : ""
                let type = try field.valueType(in: .glsl, module: module)
                return "\(offset)\(type) \(fieldNames[position])\(field.glslSuffix);"
            }.joined(separator: " ")
            declarations.append("layout(std430, binding = \(flatBinding)) \(qualifier) \(typeName) { \(members) } \(name);")
            runtimeLength = tail.runtime ? "uint(\(name).\(fieldNames.last!).length())" : nil
        }
        let locations = fields.enumerated().map { position, field in
            let reference: String
            var words: ShaderStorage.WordLayout?
            switch outputFormat {
            case .hlsl where rawHLSL:
                reference = name
                words = field.wordLayout(addressing: .byteAddressed)
            case .msl where !field.atomic && (field.scalar == .f16 || field.scalar.width == 8):
                let qualifier = writable ? "device" : "device const"
                reference = "reinterpret_cast<\(qualifier) \(field.scalar == .f16 ? "half" : "ulong")*>(\(name))"
                words = field.wordLayout()
            case .msl, .hlsl:
                if field.atomic {
                    let pointer = structured ? "reinterpret_cast<device \(field.metalAtomicType)*>(\(name) + \(field.offset / 4)u)" : name
                    reference = field.isArray ? pointer : "\(pointer)[0]"
                } else if field.isArray || field.width > 1 || field.scalar != .u32 {
                    reference = name
                    words = field.wordLayout()
                } else if outputFormat == .msl && !structured {
                    reference = "*\(name)"
                } else {
                    reference = "\(name)[\(field.offset / 4)]"
                }
            case .wgsl where !structured:
                reference = name
            default:
                reference = "\(name).\(fieldNames[position])"
            }
            let bound = field.runtime ? runtimeLength : field.count.map { "\($0)u" }
            return ShaderStorage(reference: reference, bound: bound, writable: writable, words: words)
        }
        storage = structured ? ShaderStorage(reference: name, writable: writable, members: locations) : locations[0]
        let targetIndex = outputFormat == .msl || outputFormat == .glsl ? flatBinding : nil
        reflection = .init(binding: binding, flatBindingIndex: targetIndex, minimumByteSize: minimum,
                           isReadOnly: !writable, runtimeArrayStride: tail.runtime ? tail.stride : nil,
                           runtimeArrayOffset: tail.runtime ? tail.offset : 0,
                           hlslUsesByteAddressBuffer: rawHLSL)
    }
}

private struct ComputeStorageField {
    let offset: UInt32
    let element: IRHandle<IRType>
    let scalar: IRScalar
    let width: UInt8
    let columns: UInt8?
    let stride: UInt32
    let count: UInt32?
    let runtime: Bool
    let atomic: Bool

    init(type: IRHandle<IRType>, offset: UInt32, module: IRModule) throws {
        self.offset = offset
        let arrayStride: UInt32?
        if case .array(let base, let size, let bytes) = try module.types.value(for: type).kind {
            element = base
            arrayStride = bytes
            switch size {
            case .constant(let length) where length > 0 && length <= 32 * 1024 * 1024:
                count = length
                runtime = false
            case .dynamic:
                count = nil
                runtime = true
            default:
                throw unsupported("storage array size")
            }
        } else {
            element = type
            arrayStride = nil
            count = nil
            runtime = false
        }
        switch try module.types.value(for: element).kind {
        case .scalar(let lane):
            scalar = lane
            width = 1
            columns = nil
            atomic = false
        case .atomic(let lane):
            scalar = lane
            width = 1
            columns = nil
            atomic = true
        case .vector(let size, let lane):
            scalar = lane
            width = size.rawValue
            columns = nil
            atomic = false
        case .matrix(let count, let rows, let lane) where [.f16, .f32].contains(lane):
            scalar = lane
            width = rows.rawValue
            columns = count.rawValue
            atomic = false
        default:
            throw unsupported("expected a supported scalar, vector or matrix buffer element")
        }
        guard [.u32, .i32, .u64, .i64, .f32, .f16].contains(scalar) else {
            throw unsupported("storage scalar type")
        }
        let columnSize = UInt32(width == 3 ? 4 : width) * UInt32(scalar.width)
        let byteSize = columns.map { UInt32($0) * columnSize } ?? UInt32(width) * UInt32(scalar.width)
        let naturalStride = max(byteSize, columnSize)
        guard arrayStride == nil || arrayStride == naturalStride else {
            throw unsupported("nonstandard storage array stride")
        }
        stride = arrayStride ?? byteSize
    }

    var isArray: Bool { runtime || count != nil }
    var alignment: UInt32 { UInt32(width == 3 ? 4 : width) * UInt32(scalar.width) }
    var metalAtomicType: String {
        switch scalar {
        case .u64: "atomic_ulong"
        case .i32: "atomic_int"
        default: "atomic_uint"
        }
    }
    var minimumByteSize: UInt64 { UInt64(count ?? 1) * UInt64(stride) }
    var glslSuffix: String { runtime ? "[]" : (count.map { "[\($0)]" } ?? "") }

    func wordLayout(addressing: ShaderStorage.WordAddressing = .structured) -> ShaderStorage.WordLayout {
        let laneBytes = UInt32(scalar.width)
        return .init(offset: offset / laneBytes, stride: stride / laneBytes, width: width, scalar: scalar,
                     addressing: addressing, columns: columns)
    }

    func valueType(in format: ShaderOutputFormat, module: IRModule) throws -> String {
        let base: String
        if atomic {
            let value = scalar == .i64 ? signed64Type(in: format) : scalar == .u64 ? unsigned64Type(in: format) :
                scalar == .i32 ? (format == .wgsl ? "i32" : "int") : (format == .wgsl ? "u32" : "uint")
            switch format {
            case .msl: base = metalAtomicType
            case .wgsl: base = "atomic<\(value)>"
            default: base = value
            }
        } else {
            base = try shaderValueType(element, module: module, format: format, structures: [:])
        }
        guard format == .wgsl else { return base }
        if runtime { return "array<\(base)>" }
        return count.map { "array<\(base), \($0)>" } ?? base
    }
}

private func unsupported(_ reason: String) -> ShaderFailure {
    .unsupported("Compute code generation: \(reason)")
}
