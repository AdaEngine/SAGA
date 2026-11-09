import SAGAShaderIR
import SAGATypes
import SAGAValidation

private struct RenderField {
    let index: Int
    let type: IRHandle<IRType>
    let varying: RenderStageArtifact.Varying?
    var perVertexElement: IRHandle<IRType>? = nil
    var clipDistanceCount: Int? = nil
    var invariantPosition = false
    var name: String { "saga_member_\(index)" }
    var global: String { varying.map { "saga_varying_\($0.location)" } ?? "gl_Position" }
    var perVertexLocal: String { "saga_per_vertex_\(index)" }
}

struct VertexClipDistances {
    let member: Int
    let count: Int

    var hlslGroups: [Range<Int>] {
        stride(from: 0, to: count, by: 4).map { $0..<min(count, $0 + 4) }
    }
}

enum FragmentOutputField: Equatable {
    case color(location: UInt32, width: UInt8, scalar: IRScalar)
    case blendSource(index: UInt32, width: UInt8, scalar: IRScalar)
    case depth
    case sampleMask

    var colorScalar: IRScalar? {
        switch self {
        case .color(_, _, let scalar), .blendSource(_, _, let scalar): return scalar
        default: return nil
        }
    }

    var colorAttachmentLocation: UInt32? {
        switch self {
        case .color(let location, _, _): return location
        case .blendSource(0, _, _): return 0
        default: return nil
        }
    }

    var colorComponentCount: UInt8? {
        switch self {
        case .color(_, let width, _), .blendSource(_, let width, _): return width
        default: return nil
        }
    }

    func type(in format: ShaderOutputFormat) throws -> String {
        switch self {
        case .color(_, let width, let scalar): return try shaderVectorType(width, scalar: scalar, in: format)
        case .blendSource(_, let width, let scalar): return try shaderVectorType(width, scalar: scalar, in: format)
        case .depth: return floatType(1, in: format)
        case .sampleMask: return format == .wgsl ? "u32" : "uint"
        }
    }

    func binding(in format: ShaderOutputFormat, conservativeDepth: IRConservativeDepth? = nil) -> String {
        switch (format, self, conservativeDepth) {
        case (.msl, .depth, .greaterEqual): "depth(greater)"
        case (.msl, .depth, .lessEqual): "depth(less)"
        case (.hlsl, .depth, .greaterEqual): "SV_DepthGreaterEqual"
        case (.hlsl, .depth, .lessEqual): "SV_DepthLessEqual"
        case (.msl, .color(let location, _, _), _): "color(\(location))"
        case (.msl, .blendSource(let index, _, _), _): "color(0) index(\(index))"
        case (.msl, .depth, _): "depth(any)"
        case (.msl, .sampleMask, _): "sample_mask"
        case (.hlsl, .color(let location, _, _), _): "SV_Target\(location)"
        case (.hlsl, .blendSource(let index, _, _), _): "SV_Target\(index)"
        case (.hlsl, .depth, _): "SV_Depth"
        case (.hlsl, .sampleMask, _): "SV_Coverage"
        case (.wgsl, .color(let location, _, _), _): "@location(\(location))"
        case (.wgsl, .blendSource(let index, _, _), _): "@location(0) @blend_src(\(index))"
        case (.wgsl, .depth, _): "@builtin(frag_depth)"
        case (.wgsl, .sampleMask, _): "@builtin(sample_mask)"
        case (.glsl, .color(let location, _, _), _): location == 0 ? "saga_color" : "saga_color_\(location)"
        case (.glsl, .blendSource(let index, _, _), _): "saga_blend_source_\(index)"
        case (.glsl, .depth, _): "gl_FragDepth"
        case (.glsl, .sampleMask, _): "gl_SampleMask[0]"
        default: ""
        }
    }
}

package struct RenderCodeGenerator {
    private static let metalPointOutput = [
        "#ifdef \(RenderStageArtifact.metalPointTopologyMacro)",
        "    float saga_point_size [[point_size]] = 1.0f;",
        "#endif",
    ]

    private static let metalPipelineMaskField =
        "    uint saga_pipeline_sample_mask [[sample_mask]] = uint(\(RenderStageArtifact.metalPipelineSampleMaskMacro));"

    private static let metalPipelineMaskOutput = [
        "#ifdef \(RenderStageArtifact.metalPipelineSampleMaskMacro)",
        metalPipelineMaskField,
        "#endif",
    ]

    let format: ShaderOutputFormat
    private let bindingArrayLengths: [IRResourceBinding: UInt32]
    private let webGPU: Bool
    private let metalVertexBuffers: [VertexBufferLayout]?

    package init(
        _ format: ShaderOutputFormat, bindingArrayLengths: [IRResourceBinding: UInt32] = [:],
        webGPU: Bool = false, metalVertexBuffers: [VertexBufferLayout]? = nil
    ) {
        self.format = format
        self.bindingArrayLengths = bindingArrayLengths
        self.webGPU = webGPU
        self.metalVertexBuffers = metalVertexBuffers
    }

    package func generate(_ validated: ValidatedModule<IRModule>, entryPoint: String) throws -> RenderStageArtifact {
        let module = validated.module
        guard let entry = module.entryPoints.first(where: { $0.name == entryPoint }) else {
            throw unsupported("missing entry point '\(entryPoint)'")
        }
        let vertex = entry.stage == .vertex
        guard vertex || entry.stage == .fragment else { throw unsupported("expected a vertex or fragment stage") }
        if metalVertexBuffers != nil, !vertex || format != .msl {
            throw unsupported("vertex pulling requires an MSL vertex entry point")
        }
        guard module.overrides.count == 0 else { throw unsupported("overrides") }

        var helpers = try ShaderFunctions(module: module, entry: entry.function)
        if webGPU && helpers.requiresDebugPrintf { throw unsupported("WebGPU shader debugPrintf") }
        let globals = module.globalVariables.values
        let reachable = helpers.reachableGlobals
        let active = globals.enumerated().filter { reachable.contains($0.offset) }.map(\.element)
        guard active.allSatisfy({ $0.space.isBufferAddressSpace || $0.space == .handle || $0.space == .private }) else {
            throw unsupported("unsupported render resource address space")
        }
        func isStorageBuffer(_ global: IRGlobalVariable) -> Bool {
            if case .storage = global.space { return true }
            return false
        }
        let activeStorageBuffers = active.filter(isStorageBuffer)
        let activeUniforms = active.filter { $0.space == .uniform }
        let activeHandles = try handleResourceGlobals(active, module: module, bindingArrayLengths: bindingArrayLengths)
        if webGPU {
            let widths = try helpers.nativeIntegerWidths(in: module, entry: entry.function, resources: active)
            try requireWebGPUShaderTypes(activeHandles.storageTextures, module: module, nativeIntegerWidths: widths)
        }
        let needsMetalAtomicGuard = try format == .msl && !vertex && activeHandles.storageTextures.contains {
            try bindingElementKind($0.type, module: module).storageTexture?.access.contains(.atomic) == true
        }
        let metalAtomicWriteGuard: ShaderParameter? = needsMetalAtomicGuard
            ? .init(type: "constant uint&", name: "saga_texture_atomic_guard") : nil
        let arrayCounts = activeHandles.arrayCounts
        let webBases = webGPU ? try webBindingArrayBases(module: module, activeCounts: arrayCounts) : [:]
        let externalCount = format == .wgsl ? 0 : UInt64(activeHandles.externalTextures.count)
        func slotCount(_ globals: [IRGlobalVariable]) -> UInt64 {
            globals.reduce(0) { $0 + UInt64($1.binding.flatMap { arrayCounts[$0] } ?? 1) }
        }
        guard slotCount(activeUniforms) + externalCount <= UInt64(ComputeShaderArtifact.uniformBindingCapacity),
              slotCount(activeStorageBuffers) <= UInt64(ComputeShaderArtifact.storageBindingCapacity),
              slotCount(activeHandles.textures) + 3 * externalCount <= UInt64(ComputeShaderArtifact.sampledTextureBindingCapacity),
              slotCount(activeHandles.storageTextures) <= UInt64(ComputeShaderArtifact.storageTextureBindingCapacity),
              slotCount(activeHandles.samplers) <= UInt64(ComputeShaderArtifact.samplerBindingCapacity) else {
            throw unsupported("render resource count exceeds the per-stage limit")
        }
        let renderEntries = module.entryPoints.filter { $0.stage == .vertex || $0.stage == .fragment }.map(\.function)
        let aggregateBuffers = try helpers.aggregateBufferGlobals(module: module, entry: entry.function,
                                                                 sharedEntries: format == .glsl ? renderEntries : [])
        let linkedGlobals = format == .glsl
            ? try helpers.glslLinkedGlobals(module: module, entry: entry.function, sharedEntries: renderEntries) : reachable
        let reusedBindings = format == .glsl ? reusedResourceBindings(globals) : []
        // OpenGL uses shared namespaces across linked stages. Unused arrays need no range.
        let mapped = format == .glsl ? try globals.enumerated().compactMap { index, global in
            if !linkedGlobals.contains(index) {
                if let binding = global.binding, reusedBindings.contains(binding) { return nil }
                let kind = try module.types.value(for: global.type).kind
                if case .bindingArray = kind { return nil }
                if kind.isExternalTexture { return nil }
            }
            return global
        } : active
        let namespaceCount: UInt32 = format == .glsl ? 2 : 1
        let sampled = format == .glsl
            ? try handleResourceGlobals(mapped, module: module, bindingArrayLengths: bindingArrayLengths)
            : activeHandles
        let nativeExternal = format == .wgsl ? [] : sampled.externalTextures
        func linkedArrayCounts(_ resources: [IRGlobalVariable]) throws -> [IRResourceBinding: UInt32] {
            guard !reusedBindings.isEmpty else { return sampled.arrayCounts }
            // OpenGL slots are shared across stages. Reserve the largest declaration at each slot in this namespace.
            var counts: [IRResourceBinding: UInt32] = [:]
            for global in resources {
                guard let binding = global.binding,
                      case .bindingArray(_, let size) = try module.types.value(for: global.type).kind else { continue }
                let count = try resourceBindingArrayCount(size, binding: binding, location: global.location,
                                                          bindingArrayLengths: bindingArrayLengths)
                counts[binding] = max(counts[binding] ?? 0, count)
            }
            return counts
        }
        let uniformResources = mapped.filter { $0.space == .uniform } + nativeExternal
        let flatUniforms = try flatResourceBindingIndices(uniformResources,
            capacity: ComputeShaderArtifact.uniformBindingCapacity * namespaceCount, arrayCounts: linkedArrayCounts(uniformResources))
        let textureResources = sampled.textures + nativeExternal
        let textureCounts = try externalPlaneCounts(nativeExternal, in: linkedArrayCounts(textureResources))
        let flatTextures = try flatResourceBindingIndices(textureResources,
            capacity: ComputeShaderArtifact.sampledTextureBindingCapacity * namespaceCount, arrayCounts: textureCounts)
        let mappedStorage: [IRGlobalVariable]
        if format == .glsl, !activeHandles.storageTextures.isEmpty {
            let linkedBindings = Set(linkedGlobals.compactMap { globals[$0].binding })
            mappedStorage = sampled.storageTextures.filter { $0.binding.map(linkedBindings.contains) == true }
        } else {
            mappedStorage = activeHandles.storageTextures
        }
        let flatStorageTextures = try flatResourceBindingIndices(mappedStorage,
            capacity: ComputeShaderArtifact.storageTextureBindingCapacity * namespaceCount, arrayCounts: linkedArrayCounts(mappedStorage))
        let flatSamplers = try flatResourceBindingIndices(sampled.samplers,
            capacity: ComputeShaderArtifact.samplerBindingCapacity * namespaceCount, arrayCounts: linkedArrayCounts(sampled.samplers))
        let glslBindings = format == .glsl ? try helpers.glslTextureSamplerBindings(
            module: module, entry: entry.function, textureSlots: expandedBindingIndices(flatTextures, counts: textureCounts),
            bindingArrayLengths: bindingArrayLengths, sharedEntries: renderEntries) : []
        let bufferResources = mapped.filter(isStorageBuffer)
        let flatStorageBuffers = try flatResourceBindingIndices(bufferResources,
            capacity: ComputeShaderArtifact.storageBindingCapacity * namespaceCount, arrayCounts: linkedArrayCounts(bufferResources))
        let metalLengths = ShaderParameter(type: "constant uint*", name: "saga_lengths")
        var runtimeIndex = 0
        var storageBudget = ShaderBudget()
        let immediate = try ImmediateShaderBinding(module: module, reachable: reachable, stage: entry.stage,
                                                   format: format, budget: &storageBudget)
        let storageBuffers = try globals.enumerated().map { index, global -> StorageBufferShaderBinding? in
            guard isStorageBuffer(global), reachable.contains(index),
                  let binding = global.binding, let flat = flatStorageBuffers[binding] else {
                return nil
            }
            let nativeIndex = format == .msl ? RenderStageArtifact.metalStorageBufferBase + flat : flat
            let buffer = try StorageBufferShaderBinding(global: global, index: index, flatBinding: nativeIndex,
                                                        runtimeIndex: runtimeIndex, metalLengths: metalLengths,
                                                        coherent: false, module: module, format: format,
                                                        arrayCount: arrayCounts[binding], webGPU: webGPU, webBindingBase: webBases[binding],
                                                        aggregateAccess: aggregateBuffers.contains(index),
                                                        budget: &storageBudget)
            if buffer.reflection.runtimeArrayStride != nil { runtimeIndex += Int(buffer.reflection.bindingArray?.count ?? 1) }
            return buffer
        }
        let usedStorageBuffers = storageBuffers.compactMap { $0 }
        let lengthBuffer = format == .msl && runtimeIndex > 0 ? RenderStageArtifact.metalStorageLengthBufferIndex : nil
        let uniforms = try globals.enumerated().map { index, global -> UniformShaderBinding? in
            guard global.space == .uniform, reachable.contains(index),
                  let binding = global.binding, let slot = flatUniforms[binding] else { return nil }
            return try UniformShaderBinding(global: global, index: index, flat: slot, module: module,
                                            format: format, arrayCount: arrayCounts[binding], webBindingBase: webBases[binding],
                                            aggregateAccess: aggregateBuffers.contains(index))
        }
        let sampledSlots = Set(activeHandles.textures.compactMap(\.binding))
        let textures = try globals.enumerated().map { index, global -> SampledTextureShaderBinding? in
            guard reachable.contains(index), let binding = global.binding, let slot = flatTextures[binding],
                  sampledSlots.contains(binding) else { return nil }
            return try SampledTextureShaderBinding(global: global, index: index, flat: slot, module: module, format: format,
                                                   samplingSamplers: helpers.floatTextureSamplers[index] ?? [],
                                                   arrayCount: arrayCounts[binding], webBindingBase: webBases[binding],
                                                   glslAlias: helpers.glslTextureAliases[index])
        }
        let imageSlots = Set(activeHandles.storageTextures.compactMap(\.binding))
        let storageTextures = try globals.enumerated().map { index, global -> StorageTextureShaderBinding? in
            guard reachable.contains(index), let binding = global.binding, let slot = flatStorageTextures[binding],
                  imageSlots.contains(binding) else {
                return nil
            }
            return try StorageTextureShaderBinding(global: global, index: index, flat: slot, module: module,
                                                   format: format, arrayCount: arrayCounts[binding], webBindingBase: webBases[binding],
                                                   metalAtomicWriteGuard: metalAtomicWriteGuard)
        }
        let nonFilteringSamplers = try helpers.nonFilteringSamplers(
            module: module, entry: entry.function, sampledTextures: activeHandles.textures)
        let samplerSlots = Set(activeHandles.samplers.compactMap(\.binding))
        let samplers = try globals.enumerated().map { index, global -> SamplerShaderBinding? in
            guard reachable.contains(index), let binding = global.binding, let slot = flatSamplers[binding],
                  samplerSlots.contains(binding) else { return nil }
            return try SamplerShaderBinding(global: global, index: index, flat: slot, module: module,
                                            format: format, requiresNonFiltering: nonFilteringSamplers.contains(index),
                                            arrayCount: arrayCounts[binding], webBindingBase: webBases[binding])
        }
        let externalTextures = try globals.enumerated().map { index, global -> ExternalTextureShaderBinding? in
            guard reachable.contains(index), let binding = global.binding,
                  try module.types.value(for: global.type).kind.isExternalTexture else {
                return nil
            }
            return try ExternalTextureShaderBinding(global: global, index: index, format: format,
                planeBase: flatTextures[binding],
                parametersIndex: flatUniforms[binding].map { ComputeShaderArtifact.uniformBindingBase + $0 })
        }
        let usedExternalTextures = externalTextures.compactMap { $0 }
        let usedUniforms = uniforms.compactMap { $0 }
        let usedTextures = textures.compactMap { $0 }
        let usedStorageTextures = storageTextures.compactMap { $0 }
        let usedSamplers = samplers.compactMap { $0 }
        let function = try module.functions.value(for: entry.function)
        let result = function.result
        let outputFields: [RenderField]
        if vertex {
            guard let result else { throw unsupported("missing vertex stage result") }
            outputFields = try vertexOutputs(result, module: module)
        } else {
            outputFields = []
        }
        let invariantPosition = vertex && (result?.binding == .builtIn(.position(invariant: true)) ||
            outputFields.contains(where: \.invariantPosition))
        let clipDistances = outputFields.compactMap { field in
            field.clipDistanceCount.map { VertexClipDistances(member: field.index, count: $0) }
        }.first
        let fragmentOutput = try vertex ? nil : result.map { try fragmentResult($0, module: module) }
        let fragmentFields = fragmentOutput?.fields ?? []
        let conservativeDepth: IRConservativeDepth?
        if case .allow(let depth) = entry.earlyDepthTest { conservativeDepth = depth }
        else { conservativeDepth = nil }
        // Metal disallows depth export with forced early tests. Both Metal and HLSL
        // can use rasterized depth directly when the shader promises to leave it unchanged.
        let dropsDepth = fragmentFields.contains(.depth) &&
            ((format == .msl && entry.earlyDepthTest == .force) ||
             ((format == .msl || format == .hlsl) && conservativeDepth == .unchanged))
        let outputFragmentFields = fragmentFields.enumerated().filter { !dropsDepth || $0.element != .depth }
        let returnsVoid = result == nil || (dropsDepth && outputFragmentFields.isEmpty)
        let dualSourceComponentCount: UInt8? = fragmentFields.compactMap {
            if case .blendSource(_, let width, _) = $0 { return width }
            return nil
        }.first
        let directField = fragmentFields.first ?? .color(location: 0, width: 4, scalar: .f32)
        let directType = try returnsVoid ? "void" : directField.type(in: format)
        let directBinding = directField.binding(in: format, conservativeDepth: conservativeDepth)
        let colors = fragmentFields.compactMap { field -> (location: UInt32, kind: IRScalarKind, width: UInt8)? in
            guard let location = field.colorAttachmentLocation, let scalar = field.colorScalar,
                  let width = field.colorComponentCount else { return nil }
            return (location, scalar.kind, width)
        }

        var parameter = ""
        var arguments: [String] = []
        var argumentFields: [Int: [String]] = [:]
        var inputFields: [RenderField] = []
        var inputDeclarations: [String] = []
        var inputInitializers: [String] = []
        var vertexInputs: [RenderStageArtifact.VertexInput] = []
        var vertexBuiltins: [IRBuiltIn] = []
        var fragmentBuiltins: [IRBuiltIn] = []
        if vertex {
            let inputs = try vertexArguments(function.arguments, module: module)
            parameter = inputs.parameter
            arguments = inputs.references
            argumentFields = inputs.fields
            inputDeclarations = inputs.declarations
            inputInitializers = inputs.initializers
            vertexInputs = inputs.inputs
            vertexBuiltins = inputs.builtins
        } else {
            let inputs = try fragmentArguments(function.arguments, module: module)
            parameter = inputs.parameter
            arguments = inputs.references
            argumentFields = inputs.fields
            inputDeclarations = inputs.declarations
            inputFields = inputs.varyings
            fragmentBuiltins = inputs.builtinInputs
        }
        let readsPrimitiveIndex = fragmentBuiltins.contains(.primitiveIndex)
        let readsViewIndex = vertexBuiltins.contains(.viewIndex) || fragmentBuiltins.contains(.viewIndex)
        let requiresBarycentrics = fragmentBuiltins.contains {
            if case .barycentric = $0 { return true }
            return false
        }
        let perVertexInputs = inputFields.filter { $0.perVertexElement != nil }
        guard !webGPU || !requiresBarycentrics else { throw unsupported("WebGPU fragment barycentrics") }
        guard !webGPU || !readsViewIndex else { throw unsupported("WebGPU multiview index") }
        guard !webGPU || perVertexInputs.isEmpty else { throw unsupported("WebGPU per-vertex interpolation") }
        let requiresSubgroups = fragmentBuiltins.contains(.subgroupSize) ||
            fragmentBuiltins.contains(.subgroupInvocationId) || !helpers.subgroupExtensions.isEmpty
        let fragmentInputBuiltinSlots = fragmentBuiltins.filter {
            switch $0 {
            case .position, .viewIndex: false
            default: true
            }
        }.count

        let stage: ShaderStage = vertex ? .vertex : .fragment
        var parameters = parameter.isEmpty ? [] : [parameter]
        parameters += immediate?.parameters ?? []
        parameters += usedUniforms.flatMap(\.parameters)
        parameters += usedStorageBuffers.flatMap(\.parameters)
        if let lengthBuffer {
            parameters.append("\(metalLengths.type) \(metalLengths.name) [[buffer(\(lengthBuffer))]]")
        }
        parameters += usedTextures.flatMap(\.parameters)
        parameters += usedStorageTextures.flatMap(\.parameters)
        parameters += usedSamplers.flatMap(\.parameters)
        parameters += usedExternalTextures.flatMap(\.parameters)
        if let metalAtomicWriteGuard {
            let slot = RenderStageArtifact.metalTextureAtomicGuardSlot
            parameters.append("\(metalAtomicWriteGuard.type) \(metalAtomicWriteGuard.name) [[buffer(\(slot))]]")
        }
        if format == .msl && !dropsDepth && fragmentFields.contains(.depth) {
            parameters.append("constant float2& saga_viewport_depth [[buffer(\(RenderStageArtifact.metalViewportDepthBufferIndex))]]")
        }
        parameter = parameters.joined(separator: ", ")
        let name = format == .glsl ? "main" : "saga_\(stage.rawValue)"
        var signature: String
        var declarations: [String] = []
        var resultTarget: ShaderResultTarget
        var structures: [IRHandle<IRType>: ShaderStructure] = [:]
        let usesFloat16 = module.types.values.contains { $0.kind.usesFloat16 }
        let usesInt16 = !webGPU && moduleUsesInt16(module)
        let requiresStandardUniformLayout = format == .wgsl && module.uniformBufferStandardLayout
        var hlslProfile: String?
        switch format {
        case .msl:
            let output = vertex ? "SagaVertexOutput" : "SagaFragmentOutput"
            let member = directField.colorScalar == nil ? "value" : "color"
            let field = vertex ? "position [[position\(invariantPosition ? ", invariant" : "")]]" :
                "\(member) [[\(directBinding)]]"
            declarations = ["#include <metal_stdlib>", "using namespace metal;"]
            if returnsVoid {
                let mask = RenderStageArtifact.metalPipelineSampleMaskMacro
                declarations += ["#ifdef \(mask)", "struct \(output) {", Self.metalPipelineMaskField, "};", "#endif"]
                signature = [
                    "#ifdef \(mask)", "fragment \(output) \(name)(\(parameter))",
                    "#else", "fragment void \(name)(\(parameter))", "#endif", "",
                ].joined(separator: "\n")
                resultTarget = .metalVoidFragment(output)
            } else {
                if vertex {
                    declarations += ["struct \(output) {", "    \(directType) \(field);"] + Self.metalPointOutput + ["};"]
                } else if fragmentOutput?.isStructure != true {
                    let maskOutput = directField == .sampleMask ? [] : Self.metalPipelineMaskOutput
                    declarations += ["struct \(output) {", "    \(directType) \(field);"] + maskOutput + ["};"]
                }
                signature = "\(stage.rawValue) \(output) \(name)(\(parameter))"
                resultTarget = vertex ? .metalStructure(output) : .fragmentValue(directField, metalType: output)
            }
        case .hlsl:
            let needsHLSL66 = usesInt64Atomics(module) || usedStorageTextures.contains { $0.reflection.format == .r64Uint }
            let profile: String
            if helpers.comparisonSampleHLSLMinorVersion > 0 {
                profile = "6_\(helpers.comparisonSampleHLSLMinorVersion)"
            } else if needsHLSL66 { profile = "6_6" }
            else if usesFloat16 || usesInt16 { profile = "6_2" }
            else { profile = requiresBarycentrics || readsViewIndex || !perVertexInputs.isEmpty ? "6_1" : "6_0" }
            let needsHLSL68 = vertexBuiltins.contains(.vertexIndex) || vertexBuiltins.contains(.instanceIndex)
            hlslProfile = vertex ? "vs_\(needsHLSL68 ? "6_8" : profile)" : "ps_\(profile)"
            let semantic = returnsVoid ? "" : " : \(vertex ? "SV_Position" : directBinding)"
            signature = "\(invariantPosition ? "precise " : "")\(directType) \(name)(\(parameter))\(semantic)"
            if dropsDepth && returnsVoid {
                resultTarget = .discardedFragmentDepth
            } else {
                resultTarget = vertex ? .direct : .fragmentValue(directField, metalType: nil)
            }
        case .glsl:
            declarations = [vertexBuiltins.contains(.instanceIndex) ? "#version 460 core" : "#version 450 core"]
            declarations += glslUnsigned64Extensions(module: module, storageTextures: usedStorageTextures.map(\.reflection))
            if usesFloat16 { declarations.append(glslFloat16Extension) }
            if usesInt16 { declarations.append(glslInt16Extension) }
            if usedUniforms.contains(where: \.usesGLSL16BitStorage) || usedStorageBuffers.contains(where: \.usesGLSL16BitStorage) { declarations.append(glslStorage16BitExtension) }
            declarations += helpers.glslExtensions
            declarations += helpers.glslSubgroupExtensions(requiresSubgroups: requiresSubgroups)
            if requiresBarycentrics { declarations.append("#extension GL_EXT_fragment_shader_barycentric : require") }
            if readsViewIndex { declarations.append("#extension GL_EXT_multiview : require") }
            if helpers.demotesToHelperInvocation {
                declarations.append("#extension GL_EXT_demote_to_helper_invocation : require")
            }
            if invariantPosition { declarations.append("invariant gl_Position;") }
            if let mode = entry.earlyDepthTest {
                let layout = switch mode {
                case .force: "layout(early_fragment_tests) in;"
                case .allow(.greaterEqual): "layout(depth_greater) out float gl_FragDepth;"
                case .allow(.lessEqual): "layout(depth_less) out float gl_FragDepth;"
                case .allow(.unchanged): "layout(depth_unchanged) out float gl_FragDepth;"
                }
                declarations.append(layout)
            }
            for field in fragmentFields {
                let layout: String
                switch field {
                case .color(let location, _, _): layout = "location = \(location)"
                case .blendSource(let index, _, _): layout = "location = 0, index = \(index)"
                default: continue
                }
                let type = try field.type(in: format)
                declarations.append("layout(\(layout)) out \(type) \(field.binding(in: format));")
            }
            signature = "void \(name)()"
            resultTarget = vertex ? .glslPosition : .fragmentValue(directField, metalType: nil)
        case .wgsl:
            let directives: [(Bool, String)] = [
                (!arrayCounts.isEmpty && !webGPU, "enable wgpu_binding_array;"),
                (requiresStandardUniformLayout, "requires uniform_buffer_standard_layout;"),
                (module.requiresSubgroupUniformity, "requires subgroup_uniformity;"),
                (immediate != nil, "requires immediate_address_space;"),
                (usedStorageTextures.contains { $0.reflection.format.requiresTextureFormatsTier1 }, "requires texture_formats_tier1;"),
                (usesFloat16, "enable f16;"),
                (usesInt16, "enable wgpu_int16;"),
                (helpers.requiresDebugPrintf, "enable wgpu_debug_printf;"),
                (clipDistances != nil, "enable clip_distances;"),
                (readsPrimitiveIndex, "enable primitive_index;"),
                (vertexBuiltins.contains(.drawIndex), "enable draw_index;"),
                (!perVertexInputs.isEmpty, "enable wgpu_per_vertex;"),
                (requiresSubgroups, "enable subgroups;"),
                (dualSourceComponentCount != nil, "enable dual_source_blending;"),
            ]
            let extensions = directives.compactMap { required, directive in required ? directive : nil }
            declarations = try extensions + helpers.wgslDiagnosticLines(module: module)
            let binding = vertex ? "\(invariantPosition ? "@invariant " : "")@builtin(position)" : directBinding
            let returnType = returnsVoid ? "" : " -> \(binding) \(directType)"
            signature = "@\(stage.rawValue) fn \(name)(\(parameter))\(returnType)"
            resultTarget = .direct
        default: throw unsupported("output format \(format.rawValue)")
        }
        if let result, !outputFields.isEmpty {
            let typeName = "SagaVertexOutput"
            let make = format == .hlsl ? "saga_make_vertex_output" : typeName
            let output = ShaderStructure(typeName: typeName, constructor: make)
            let usesValueStructure = clipDistances != nil && format != .wgsl
            if !usesValueStructure { structures[result.type] = output }
            switch format {
            case .msl:
                declarations = ["#include <metal_stdlib>", "using namespace metal;"]
                signature = "vertex \(typeName) \(name)(\(parameter))"
                resultTarget = .direct
            case .hlsl: signature = "\(typeName) \(name)(\(parameter))"
            case .wgsl: signature = "@vertex fn \(name)(\(parameter)) -> \(typeName)"
            case .glsl:
                declarations += try glslVaryings(outputFields, output: true, module: module)
                if let clipDistances { declarations.append("out float gl_ClipDistance[\(clipDistances.count)];") }
                resultTarget = .glslVertexStructure(typeName, destinations: outputFields.map(\.global))
            default: break
            }
            if !usesValueStructure || format != .glsl {
                declarations += try structure(typeName, fields: outputFields, vertexOutput: true, module: module)
            }
            if format == .hlsl && !usesValueStructure {
                let types = try outputFields.map {
                    try shaderValueType($0.type, module: module, format: format, structures: [:])
                }
                declarations += output.hlslConstructor(memberTypes: types)
            }
        }
        if let result, fragmentOutput?.isStructure == true, !returnsVoid {
            let typeName = "SagaFragmentOutput"
            let structure = ShaderStructure(typeName: typeName,
                constructor: format == .hlsl ? "saga_make_fragment_output" : typeName)
            if !dropsDepth { structures[result.type] = structure }
            let types = try outputFragmentFields.map { try $0.element.type(in: format) }
            let members = zip(outputFragmentFields, types).map { field, type -> String in
                let member = "saga_member_\(field.offset)"
                let binding = field.element.binding(in: format, conservativeDepth: conservativeDepth)
                switch format {
                case .msl: return "    \(type) \(member) [[\(binding)]];"
                case .hlsl: return "    \(type) \(member) : \(binding);"
                case .wgsl: return "    \(binding) \(member): \(type),"
                default: return "    \(type) \(member);"
                }
            }
            let maskOutput = format == .msl && !fragmentFields.contains(.sampleMask) ? Self.metalPipelineMaskOutput : []
            declarations += ["struct \(typeName) {"] + members + maskOutput + [format == .wgsl ? "}" : "};"]
            switch format {
            case .msl: signature = "fragment \(typeName) \(name)(\(parameter))"
            case .hlsl:
                signature = "\(typeName) \(name)(\(parameter))"
                if !dropsDepth { declarations += structure.hlslConstructor(memberTypes: types) }
            case .wgsl: signature = "@fragment fn \(name)(\(parameter)) -> \(typeName)"
            default: break
            }
            resultTarget = .fragmentStructure(typeName, fields: fragmentFields)
        }
        if format == .msl && readsViewIndex {
            let mask = RenderStageArtifact.metalMultiviewMaskMacro
            declarations += [
                "inline uint saga_view_index(uint amplification) {",
                "#ifdef \(mask)",
                "    uint views = uint(\(mask));",
                "    for (uint index = 0u; index < amplification; ++index) views &= views - 1u;",
                "    return ctz(views);",
                "#else", "    return amplification;", "#endif", "}",
            ]
        }
        if format == .msl && fragmentFields.contains(.sampleMask) {
            let mask = RenderStageArtifact.metalPipelineSampleMaskMacro
            declarations += ["inline uint saga_apply_pipeline_mask(uint mask) {",
                             "#ifdef \(mask)", "    return mask & uint(\(mask));",
                             "#else", "    return mask;", "#endif", "}"]
        }
        declarations += try decompositionDeclarations(module: module, format: format, structures: &structures)
        declarations += try atomicComparisonDeclarations(module: module, format: format, webGPU: webGPU, structures: &structures)
        let extraTypes: Set<IRHandle<IRType>>
        if let clipDistances, let result {
            extraTypes = [format == .wgsl ? outputFields[clipDistances.member].type : result.type]
        } else if dropsDepth, fragmentOutput?.isStructure == true, let result {
            extraTypes = [result.type]
        } else {
            extraTypes = []
        }
        declarations += try helpers.compositeDeclarations(module: module, entry: entry.function,
                                                          format: format, additionalTypes: extraTypes, structures: &structures)
        for field in perVertexInputs where format != .wgsl {
            let type = try shaderValueType(field.type, module: module, format: format, structures: structures)
            let source = "saga_input.\(field.name)"
            let local = field.perVertexLocal
            switch format {
            case .msl:
                let values = ["first", "second", "third"].map { "\(source).get(metal::vertex_index::\($0))" }
                inputInitializers.append("    const \(type) \(local) = { \(values.joined(separator: ", ")) };")
            case .hlsl:
                inputInitializers.append("    \(type) \(local);")
                inputInitializers += (0..<3).map { index in
                    "    \(local).elements[\(index)] = GetAttributeAtVertex(\(source), \(index));"
                }
            default: throw unsupported("per-vertex interpolation target")
            }
        }
        if let clipDistances, let result, format != .wgsl {
            let valueType = try shaderValueType(result.type, module: module, format: format, structures: structures)
            let destinations = format == .glsl ? outputFields.map(\.global) : outputFields.map(\.name)
            resultTarget = .vertexClipStructure(valueType, destinations: destinations, clipDistances: clipDistances)
        }
        if dropsDepth, fragmentOutput?.isStructure == true, !returnsVoid, let result {
            let valueType = try shaderValueType(result.type, module: module, format: format, structures: structures)
            resultTarget = .fragmentWithoutDepth(valueType: valueType, fields: outputFragmentFields)
        }
        if let mode = entry.earlyDepthTest {
            switch (format, mode) {
            case (.msl, .force): signature = "[[early_fragment_tests]]\n" + signature
            case (.hlsl, .force): signature = "[earlydepthstencil]\n" + signature
            case (.wgsl, _):
                let argument = switch mode {
                case .force: "force"
                case .allow(.greaterEqual): "greater_equal"
                case .allow(.lessEqual): "less_equal"
                case .allow(.unchanged): "unchanged"
                }
                signature = "@early_depth_test(\(argument))\n" + signature
            default: break
            }
        }
        declarations += inputDeclarations + usedUniforms.flatMap(\.declarations) + usedTextures.flatMap(\.declarations)
        declarations += immediate?.declarations ?? []
        declarations += usedStorageBuffers.flatMap(\.declarations)
        declarations += usedStorageTextures.flatMap(\.declarations) + usedSamplers.flatMap(\.declarations)
        declarations += usedExternalTextures.flatMap(\.declarations)
        if !usedExternalTextures.isEmpty { declarations += externalTexturePrelude(format) }
        declarations += glslBindings.map { $0.declaration }

        let privateBindings = try helpers.privateBindings(module: module, function: function, format: format,
                                                          structures: structures)
        declarations += privateBindings.declarations
        let storage = globals.indices.map {
            if let immediate, immediate.index == $0 { return immediate.storage }
            if let external = externalTextures[$0] { return external.storage }
            return storageBuffers[$0]?.storage ?? uniforms[$0]?.storage ?? textures[$0]?.storage ?? storageTextures[$0]?.storage ??
                samplers[$0]?.storage ?? privateBindings.storage[$0] ?? ShaderStorage(reference: "", writable: false)
        }
        if format == .msl {
            try helpers.bindMetalStorage(globals.indices.map {
                if let immediate, immediate.index == $0 { return immediate.captures }
                if let external = externalTextures[$0] { return external.captures }
                return storageBuffers[$0]?.captures ?? uniforms[$0]?.captures ?? textures[$0]?.captures ?? storageTextures[$0]?.captures ??
                    samplers[$0]?.captures ?? privateBindings.captures[$0] ?? []
            })
        }

        let helperDeclarations = try helpers.declarations(module: module, format: format, storage: storage, structures: structures)
        let diagnosticAttributes = format == .wgsl ? try helpers.wgslDiagnosticLines(module: module, function: function) : []
        var writer = ShaderBodyWriter(module: module, function: function, arguments: arguments,
                                      storage: storage, outputFormat: format, resultTarget: resultTarget,
                                      structures: structures, argumentFields: argumentFields,
                                      callCaptures: helpers.callCaptures, pointerWrites: helpers.pointerWrites, budget: helpers.budget)
        let body = try writer.write()
        let numeric = try integerPrelude(helpers.integerHelpers.union(writer.integerHelpers),
                                         vectors: helpers.integerVectorHelpers.union(writer.integerVectorHelpers),
                                         in: format)
        let source = (declarations + numeric + helperDeclarations + diagnosticAttributes + [signature + " {"] +
                      inputInitializers + privateBindings.initialization + body + ["}", ""])
            .joined(separator: "\n")
        return .init(artifact: .text(format: format, source: source), stage: stage,
                     sourceEntryPoint: entry.name, entryPoint: name,
                     varyings: (vertex ? outputFields : inputFields).compactMap(\.varying).sorted { $0.location < $1.location },
                     vertexInputs: vertexInputs, hlslProfile: hlslProfile,
                     uniformBindings: usedUniforms.map(\.reflection), storageBindings: usedStorageBuffers.map(\.reflection),
                     metalRuntimeLengthBuffer: lengthBuffer, sampledTextures: usedTextures.map(\.reflection),
                     storageTextures: usedStorageTextures.map(\.reflection),
                     externalTextures: usedExternalTextures.map(\.binding), externalTextureLayouts: usedExternalTextures.compactMap(\.layout),
                     samplerBindings: usedSamplers.map(\.reflection),
                     glslTextureSamplerBindings: glslBindings.map { $0.reflection },
                     colorOutputKinds: Dictionary(uniqueKeysWithValues: colors.map { ($0.location, $0.kind) }),
                     colorOutputComponentCounts: Dictionary(uniqueKeysWithValues: colors.map { ($0.location, $0.width) }),
                     writesDepth: !dropsDepth && fragmentFields.contains(.depth),
                     writesSampleMask: fragmentFields.contains(.sampleMask), invariantPosition: invariantPosition,
                     clipDistanceCount: clipDistances?.count ?? 0,
                     readsPrimitiveIndex: readsPrimitiveIndex, requiresBarycentrics: requiresBarycentrics,
                     readsViewIndex: readsViewIndex,
                     fragmentInputBuiltinSlots: fragmentInputBuiltinSlots,
                     requiresShaderF16: usesFloat16,
                     requiresUniformBufferStandardLayout: requiresStandardUniformLayout, requiresSubgroups: requiresSubgroups,
                     dualSourceComponentCount: dualSourceComponentCount, earlyDepthTest: entry.earlyDepthTest,
                     metalTextureAtomicGuardBufferIndex: needsMetalAtomicGuard ? RenderStageArtifact.metalTextureAtomicGuardSlot : nil,
                     requiresShaderInt64AtomicMinMax: !webGPU && usesInt64Atomics(module) && !helpers.requiresShaderInt64AtomicAllOps,
                     requiresShaderInt64AtomicAllOps: helpers.requiresShaderInt64AtomicAllOps,
                     requiresShaderInt64: !webGPU && moduleUsesWideIntegers(module), requiresShaderInt16: usesInt16,
                     requiresDebugPrintf: helpers.requiresDebugPrintf,
                     immediateLayout: immediate?.layout,
                     requiresSubgroupUniformity: module.requiresSubgroupUniformity,
                     metalVertexPullingLayout: metalVertexBuffers.map(VertexBufferLayout.metalPullingSignature))
    }

    private func vertexArguments(
        _ arguments: [IRFunctionArgument], module: IRModule
    ) throws -> (parameter: String, references: [String], fields: [Int: [String]],
                 declarations: [String], initializers: [String],
                 inputs: [RenderStageArtifact.VertexInput], builtins: [IRBuiltIn]) {
        var attributes: [(input: RenderStageArtifact.VertexInput, type: IRHandle<IRType>)] = []
        var references: [String] = []
        var fields: [Int: [String]] = [:]
        var builtinParameters: [String] = []
        var builtins: [IRBuiltIn] = []

        func reference(type: IRHandle<IRType>, binding: IRBinding?) throws -> String {
            let kind = try module.types.value(for: type).kind
            if binding == .builtIn(.vertexIndex), kind == .scalar(.u32) {
                builtins.append(.vertexIndex)
                switch format {
                case .msl: builtinParameters.append("uint saga_index [[vertex_id]]")
                case .hlsl:
                    builtinParameters += ["uint saga_index : SV_VertexID", "int saga_first_vertex : SV_StartVertexLocation"]
                    return "(saga_index + asuint(saga_first_vertex))"
                case .wgsl: builtinParameters.append("@builtin(vertex_index) saga_index: u32")
                default: return "uint(gl_VertexID)"
                }
                return "saga_index"
            }
            if binding == .builtIn(.instanceIndex), kind == .scalar(.u32) {
                builtins.append(.instanceIndex)
                switch format {
                case .msl: builtinParameters.append("uint saga_instance [[instance_id]]")
                case .hlsl:
                    builtinParameters += ["uint saga_instance : SV_InstanceID", "uint saga_first_instance : SV_StartInstanceLocation"]
                    return "(saga_instance + saga_first_instance)"
                case .wgsl: builtinParameters.append("@builtin(instance_index) saga_instance: u32")
                default: return "(uint(gl_InstanceID) + uint(gl_BaseInstance))"
                }
                return "saga_instance"
            }
            if binding == .builtIn(.drawIndex), kind == .scalar(.u32) {
                guard format == .wgsl, !webGPU else {
                    throw unsupported("draw_index requires native WGSL or SPIR-V output")
                }
                builtins.append(.drawIndex)
                builtinParameters.append("@builtin(draw_index) saga_draw_index: u32")
                return "saga_draw_index"
            }
            if binding == .builtIn(.viewIndex), kind == .scalar(.u32) {
                builtins.append(.viewIndex)
                switch format {
                case .msl:
                    builtinParameters.append("uint saga_amplification_id [[amplification_id]]")
                    return "saga_view_index(saga_amplification_id)"
                case .hlsl: builtinParameters.append("uint saga_view_index : SV_ViewID")
                case .wgsl: builtinParameters.append("@builtin(view_index) saga_view_index: u32")
                default: return "uint(gl_ViewIndex)"
                }
                return "saga_view_index"
            }
            guard case .location(let location, nil, nil, nil, false) = binding else {
                throw unsupported("expected vertex_index, instance_index, view_index or a vertex attribute")
            }
            if format == .msl, metalVertexBuffers == nil, location >= RenderStageArtifact.metalVertexAttributeCapacity {
                throw unsupported("vertex attribute location exceeds the Metal descriptor limit")
            }
            let width: UInt8
            let scalar: IRScalar
            switch kind {
            case .scalar(let lane): (width, scalar) = (1, lane)
            case .vector(let size, let lane): (width, scalar) = (size.rawValue, lane)
            default: throw unsupported("vertex attribute must be a scalar or vector")
            }
            guard [.f16, .f32, .i32, .u32].contains(scalar) else {
                throw unsupported("vertex attribute must use f16, f32, i32 or u32")
            }
            attributes.append((.init(location: location, componentCount: width, scalar: scalar), type))
            let name = "saga_attribute_\(location)"
            return format == .glsl ? name : "saga_vertex_input.\(name)"
        }

        for (index, argument) in arguments.enumerated() {
            if case .structure(let members, _) = try module.types.value(for: argument.type).kind {
                fields[index] = try members.map { try reference(type: $0.type, binding: $0.binding) }
                references.append("")
            } else {
                references.append(try reference(type: argument.type, binding: argument.binding))
            }
        }

        attributes.sort { $0.input.location < $1.input.location }
        var parameters: [String] = []
        var declarations: [String] = []
        var initializers: [String] = []
        if !attributes.isEmpty {
            let lines = try attributes.map { attribute -> String in
                let input = attribute.input
                let type = try shaderValueType(attribute.type, module: module, format: format, structures: [:])
                let name = "saga_attribute_\(input.location)"
                switch format {
                case .msl:
                    let binding = metalVertexBuffers == nil ? " [[attribute(\(input.location))]]" : ""
                    return "    \(type) \(name)\(binding);"
                case .hlsl: return "    \(type) \(name) : TEXCOORD\(input.location);"
                case .wgsl: return "    @location(\(input.location)) \(name): \(type),"
                default: return "layout(location = \(input.location)) in \(type) \(name);"
                }
            }
            if format == .glsl {
                declarations = lines
            } else {
                declarations = ["struct SagaVertexInput {"] + lines + [format == .wgsl ? "}" : "};"]
                switch format {
                case .msl:
                    if metalVertexBuffers == nil { parameters.append("SagaVertexInput saga_vertex_input [[stage_in]]") }
                case .hlsl: parameters.append("SagaVertexInput saga_vertex_input")
                case .wgsl: parameters.append("saga_vertex_input: SagaVertexInput")
                default: break
                }
            }
        }
        if let metalVertexBuffers {
            let inputs = attributes.map(\.input)
            let pulling = try MetalVertexPulling(layouts: metalVertexBuffers).generate(inputs, builtins: builtins)
            parameters += pulling.parameters
            declarations += pulling.declarations
            initializers = pulling.initializers
        }
        parameters += builtinParameters
        return (parameters.joined(separator: ", "), references, fields, declarations, initializers,
                attributes.map(\.input), builtins)
    }

    private func fragmentArguments(
        _ arguments: [IRFunctionArgument], module: IRModule
    ) throws -> (parameter: String, references: [String], fields: [Int: [String]],
                 declarations: [String], varyings: [RenderField], builtinInputs: [IRBuiltIn]) {
        var references: [String] = []
        var fields: [Int: [String]] = [:]
        var varyings: [RenderField] = []
        var builtins: [String] = []
        var structuredBuiltins: [String] = []
        var builtinInputs: [IRBuiltIn] = []
        func reference(type: IRHandle<IRType>, binding: IRBinding?) throws -> String {
            if let subgroup = subgroupInput(binding, name: "saga_subgroup_input_\(builtinInputs.count)", format: format),
               case .builtIn(let input) = binding {
                builtinInputs.append(input)
                if let parameter = subgroup.parameter { builtins.append(parameter) }
                return subgroup.value
            }
            let builtin: (name: String, msl: String, hlsl: String, wgsl: String, glsl: String)?
            switch binding {
            case .builtIn(.sampleIndex):
                builtin = ("saga_sample_index", "sample_id", "SV_SampleIndex", "sample_index", "uint(gl_SampleID)")
            case .builtIn(.sampleMask):
                builtin = ("saga_sample_mask", "sample_mask", "SV_Coverage", "sample_mask", "uint(gl_SampleMaskIn[0])")
            case .builtIn(.position):
                builtin = ("saga_position", "position", "SV_Position", "position", "gl_FragCoord")
            case .builtIn(.frontFacing):
                builtin = ("saga_front_facing", "front_facing", "SV_IsFrontFace", "front_facing", "gl_FrontFacing")
            case .builtIn(.primitiveIndex):
                builtin = ("saga_primitive_index", "primitive_id", "SV_PrimitiveID", "primitive_index", "uint(gl_PrimitiveID)")
            case .builtIn(.viewIndex):
                builtin = ("saga_amplification_id", "amplification_id", "SV_ViewID", "view_index", "uint(gl_ViewIndex)")
            case .builtIn(.barycentric(let perspective)):
                builtin = ("saga_barycentric", perspective ? "barycentric_coord" : "barycentric_coord, center_no_perspective",
                           "SV_Barycentrics", perspective ? "barycentric" : "barycentric_no_perspective",
                           perspective ? "gl_BaryCoordEXT" : "gl_BaryCoordNoPerspEXT")
            default:
                builtin = nil
            }
            if let builtin, case .builtIn(let input) = binding {
                builtinInputs.append(input)
                let type = try shaderValueType(type, module: module, format: format, structures: [:])
                switch format {
                case .msl:
                    if input == .barycentric(perspective: false) {
                        // Metal 3.1 requires the interpolation qualifier on a structure member.
                        structuredBuiltins.append("    \(type) \(builtin.name) [[\(builtin.msl)]];")
                        return "saga_input.\(builtin.name)"
                    }
                    builtins.append("\(type) \(builtin.name) [[\(builtin.msl)]]")
                    if input == .viewIndex { return "saga_view_index(\(builtin.name))" }
                case .hlsl:
                    let interpolation = input == .barycentric(perspective: false) ? "noperspective " : ""
                    structuredBuiltins.append("    \(interpolation)\(type) \(builtin.name) : \(builtin.hlsl);")
                    let reference = "saga_input.\(builtin.name)"
                    if case .builtIn(.position) = binding {
                        // SV_Position.w contains clip-space w, while WGSL exposes its reciprocal.
                        return "float4(\(reference).xyz, 1.0f / \(reference).w)"
                    }
                    return reference
                case .wgsl:
                    let qualifier = input == .position(invariant: true) ? "@invariant " : ""
                    builtins.append("\(qualifier)@builtin(\(builtin.wgsl)) \(builtin.name): \(type)")
                default: return builtin.glsl
                }
                return builtin.name
            }
            let varying = try field(varyings.count, type: type, binding: binding, module: module)
            varyings.append(varying)
            if varying.perVertexElement != nil && format != .wgsl { return varying.perVertexLocal }
            return format == .glsl ? varying.global : "saga_input.\(varying.name)"
        }
        for (index, argument) in arguments.enumerated() {
            if case .structure(let members, _) = try module.types.value(for: argument.type).kind {
                fields[index] = try members.map { try reference(type: $0.type, binding: $0.binding) }
                references.append("")
            } else {
                references.append(try reference(type: argument.type, binding: argument.binding))
            }
        }
        let hasPosition = builtinInputs.contains {
            if case .position = $0 { return true }
            return false
        }
        if format == .msl, builtinInputs.contains(.barycentric(perspective: true)), !hasPosition {
            // Metal can omit reciprocal-W interpolation when the entry has no
            // position input. Declare it to keep perspective weights normalized.
            builtins.append("float4 saga_barycentric_position [[position]]")
        }
        if format == .glsl {
            return ("", references, fields, try glslVaryings(varyings, output: false, module: module), varyings, builtinInputs)
        }
        var parameters: [String] = []
        var declarations: [String] = []
        if !varyings.isEmpty || !structuredBuiltins.isEmpty {
            declarations = try structure("SagaFragmentInput", fields: varyings, vertexOutput: false, module: module)
            declarations.insert(contentsOf: structuredBuiltins, at: declarations.count - 1)
            switch format {
            case .msl: parameters.append("SagaFragmentInput saga_input [[stage_in]]")
            case .hlsl: parameters.append("SagaFragmentInput saga_input")
            case .wgsl: parameters.append("saga_input: SagaFragmentInput")
            default: break
            }
        }
        return ((parameters + builtins).joined(separator: ", "), references, fields, declarations, varyings, builtinInputs)
    }

    private func fragmentResult(
        _ result: IRFunctionResult, module: IRModule
    ) throws -> (fields: [FragmentOutputField], isStructure: Bool) {
        func field(type: IRHandle<IRType>, binding: IRBinding?) throws -> FragmentOutputField {
            switch binding {
            case .builtIn(.fragDepth): return .depth
            case .builtIn(.sampleMask): return .sampleMask
            case .location(0, nil, nil, let source?, false):
                switch try module.types.value(for: type).kind {
                case .scalar(let scalar): return .blendSource(index: source, width: 1, scalar: scalar)
                case .vector(let size, let scalar): return .blendSource(index: source, width: size.rawValue, scalar: scalar)
                default: throw unsupported("expected a numeric dual-source output")
                }
            case .location(let location, nil, nil, nil, false):
                if (format == .msl || format == .hlsl), location >= RenderStageArtifact.nativeColorAttachmentCapacity {
                    throw unsupported("color target location exceeds the native output register limit")
                }
                switch try module.types.value(for: type).kind {
                case .scalar(let scalar) where [.f16, .f32, .i32, .u32].contains(scalar):
                    return .color(location: location, width: 1, scalar: scalar)
                case .vector(let size, let scalar) where [.f16, .f32, .i32, .u32].contains(scalar):
                    return .color(location: location, width: size.rawValue, scalar: scalar)
                default:
                    throw unsupported("expected a scalar or vector color with f16, f32, i32 or u32 components")
                }
            default: throw unsupported("expected a color location, frag_depth or sample_mask")
            }
        }
        if result.binding == nil, case .structure(let members, _) = try module.types.value(for: result.type).kind {
            return (try members.map { try field(type: $0.type, binding: $0.binding) }, true)
        }
        return ([try field(type: result.type, binding: result.binding)], false)
    }

    private func vertexOutputs(_ result: IRFunctionResult, module: IRModule) throws -> [RenderField] {
        let type = try module.types.value(for: result.type).kind
        if case .structure(let members, _) = type, result.binding == nil {
            return try members.enumerated().map { index, member in
                if case .builtIn(.position(let invariant)) = member.binding,
                   try module.types.value(for: member.type).kind == .vector(size: .four, scalar: .f32) {
                    return RenderField(index: index, type: member.type, varying: nil, invariantPosition: invariant)
                }
                if member.binding == .builtIn(.clipDistances),
                   case .array(_, .constant(let count), _) = try module.types.value(for: member.type).kind {
                    return RenderField(index: index, type: member.type, varying: nil, clipDistanceCount: Int(count))
                }
                return try field(index, type: member.type, binding: member.binding, module: module)
            }
        }
        guard case .builtIn(.position) = result.binding, type == .vector(size: .four, scalar: .f32) else {
            throw unsupported("expected a vec4<f32> position")
        }
        return []
    }

    private func field(
        _ index: Int, type: IRHandle<IRType>, binding: IRBinding?, module: IRModule
    ) throws -> RenderField {
        guard case .location(let location, let declaredInterpolation, let declaredSampling, nil, false) = binding else {
            throw unsupported("expected a varying with a location")
        }
        let interpolation = declaredInterpolation ?? .perspective
        let sampling = declaredSampling ?? (interpolation == .flat ? .first : .center)
        var element: IRHandle<IRType>?
        switch interpolation {
        case .flat:
            guard sampling == .first || sampling == .either else { throw unsupported("unsupported flat sampling") }
        case .perspective, .linear:
            guard sampling == .center || sampling == .centroid || sampling == .sample else {
                throw unsupported("unsupported interpolation sampling")
            }
        case .perVertex:
            guard format != .glsl else { throw unsupported("GLSL per-vertex interpolation") }
            guard sampling == .center, case .array(let base, .constant(3), _) = try module.types.value(for: type).kind else {
                throw unsupported("per-vertex varying must be an array of three elements")
            }
            element = base
        }
        let width: UInt8
        let scalar: IRScalar
        switch try module.types.value(for: element ?? type).kind {
        case .scalar(let lane): (width, scalar) = (1, lane)
        case .vector(let size, let lane): (width, scalar) = (size.rawValue, lane)
        default: throw unsupported("expected a scalar or vector varying")
        }
        let uninterpolated = interpolation == .flat || interpolation == .perVertex
        guard scalar == .f16 || scalar == .f32 || (uninterpolated && (scalar == .i32 || scalar == .u32)) else {
            throw unsupported("unsupported varying scalar type")
        }
        return RenderField(index: index, type: type, varying: .init(
            location: location, componentCount: width, scalar: scalar, interpolation: interpolation, sampling: sampling),
            perVertexElement: element)
    }

    private func structure(
        _ name: String, fields: [RenderField], vertexOutput: Bool, module: IRModule
    ) throws -> [String] {
        let lines = try fields.flatMap { field -> [String] in
            if let count = field.clipDistanceCount {
                switch format {
                case .wgsl: return ["    @builtin(clip_distances) \(field.name): array<f32, \(count)>,"]
                case .msl: return ["    float \(field.name) [[clip_distance]] [\(count)];"]
                case .hlsl:
                    return VertexClipDistances(member: field.index, count: count).hlslGroups.enumerated().map { group, range in
                        "    \(floatType(UInt8(range.count), in: .hlsl)) saga_clip_\(group) : SV_ClipDistance\(group);"
                    }
                default: throw unsupported("clip distances require a vertex output interface")
                }
            }
            if let element = field.perVertexElement, let varying = field.varying {
                let type = try shaderValueType(element, module: module, format: format, structures: [:])
                switch format {
                case .msl:
                    return ["    metal::vertex_value<\(type)> \(field.name) [[user(locn\(varying.location))]];"]
                case .hlsl:
                    return ["    nointerpolation \(type) \(field.name) : TEXCOORD\(varying.location);"]
                case .wgsl:
                    return ["    @location(\(varying.location)) @interpolate(per_vertex) \(field.name): array<\(type), 3>,"]
                default: throw unsupported("per-vertex interpolation target")
                }
            }
            let type = try shaderValueType(field.type, module: module, format: format, structures: [:])
            guard let varying = field.varying else {
                switch format {
                case .msl: return ["    float4 \(field.name) [[position\(field.invariantPosition ? ", invariant" : "")]];"]
                case .hlsl: return ["    \(field.invariantPosition ? "precise " : "")float4 \(field.name) : SV_Position;"]
                case .wgsl: return ["    \(field.invariantPosition ? "@invariant " : "")@builtin(position) \(field.name): \(type),"]
                default: return ["    \(type) \(field.name);"]
                }
            }
            switch format {
            case .msl:
                let modifier = varying.interpolation == .flat ? "flat" :
                    "\(varying.sampling.rawValue)_\(varying.interpolation == .linear ? "no_perspective" : "perspective")"
                let attribute = vertexOutput ? "" : ", \(modifier)"
                return ["    \(type) \(field.name) [[user(locn\(varying.location))\(attribute)]];"]
            case .hlsl:
                let qualifier = varyingQualifiers(varying, flat: "nointerpolation")
                return ["    \(qualifier)\(type) \(field.name) : TEXCOORD\(varying.location);"]
            case .wgsl:
                let attribute: String
                switch (varying.interpolation, varying.sampling) {
                case (.perspective, .center): attribute = ""
                case (.linear, .center), (.flat, .first):
                    attribute = " @interpolate(\(varying.interpolation.rawValue))"
                default:
                    attribute = " @interpolate(\(varying.interpolation.rawValue), \(varying.sampling.rawValue))"
                }
                return ["    @location(\(varying.location))\(attribute) \(field.name): \(type),"]
            default:
                return ["    \(type) \(field.name);"]
            }
        }
        let pointOutput = format == .msl && vertexOutput ? Self.metalPointOutput : []
        return ["struct \(name) {"] + lines + pointOutput + [format == .wgsl ? "}" : "};"]
    }

    private func varyingQualifiers(_ varying: RenderStageArtifact.Varying, flat: String) -> String {
        if varying.interpolation == .flat { return flat + " " }
        let interpolation = varying.interpolation == .linear ? "noperspective " : ""
        let sampling = varying.sampling == .center ? "" : "\(varying.sampling.rawValue) "
        return interpolation + sampling
    }

    private func glslVaryings(_ fields: [RenderField], output: Bool, module: IRModule) throws -> [String] {
        try fields.compactMap { field in
            guard let varying = field.varying else { return nil }
            let type = try shaderValueType(field.type, module: module, format: .glsl, structures: [:])
            let qualifier = varyingQualifiers(varying, flat: "flat")
            return "layout(location = \(varying.location)) \(qualifier)\(output ? "out" : "in") \(type) \(field.global);"
        }
    }

}

func floatType(_ width: UInt8, in format: ShaderOutputFormat) -> String {
    if width == 1 { return format == .wgsl ? "f32" : "float" }
    switch format {
    case .wgsl: return "vec\(width)<f32>"
    case .glsl: return "vec\(width)"
    default: return "float\(width)"
    }
}

private func unsupported(_ reason: String) -> ShaderFailure {
    .unsupported("Render code generation: \(reason)")
}
