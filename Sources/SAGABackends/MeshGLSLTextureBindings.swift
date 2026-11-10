import SAGAShaderIR
import SAGAValidation

/// GLSL mesh handle declarations and shader body mappings.
struct MeshGLSLTextureBindings {
    var declarations: [String] = []
    var storage: [Int: ShaderStorage] = [:]
    var storageTextures: [ComputeShaderArtifact.StorageTextureBinding] = []

    init(module: IRModule, entry: IREntryPoint, helpers: inout ShaderFunctions,
         bindingArrayLengths: [IRResourceBinding: UInt32] = [:]) throws {
        let indices = helpers.reachableGlobals.sorted().filter { module.globalVariables.values[$0].space == .handle }
        let resources = indices.map { module.globalVariables.values[$0] }
        let handles = try handleResourceGlobals(resources, module: module, bindingArrayLengths: bindingArrayLengths)
        guard handles.externalTextures.isEmpty else {
            throw ShaderFailure.unsupported("GLSL mesh external textures")
        }
        // Preserve per-stage limits even though linked stages share a larger namespace.
        let counts = handles.arrayCounts
        _ = try flatResourceBindingIndices(handles.textures, capacity: ComputeShaderArtifact.sampledTextureBindingCapacity, arrayCounts: counts)
        _ = try flatResourceBindingIndices(handles.storageTextures, capacity: ComputeShaderArtifact.storageTextureBindingCapacity, arrayCounts: counts)
        _ = try flatResourceBindingIndices(handles.samplers, capacity: ComputeShaderArtifact.samplerBindingCapacity, arrayCounts: counts)
        let linkedEntries = module.entryPoints.filter { $0.stage == .task || $0.stage == .mesh || $0.stage == .fragment }.map(\.function)
        let linkedIndices = try helpers.glslLinkedGlobals(module: module, entry: entry.function, sharedEntries: linkedEntries)
        let linkedResources = linkedIndices.sorted().map { module.globalVariables.values[$0] }.filter { $0.space == .handle }
        // Different declarations at one logical binding may have incompatible types or
        // array extents. Do not silently select one when linking stages.
        let linkedBindings = linkedResources.compactMap(\.binding)
        guard Set(linkedBindings).count == linkedBindings.count else {
            throw ShaderFailure.unsupported("GLSL task, mesh, and fragment handle bindings alias across linked stages")
        }
        let linked = try handleResourceGlobals(linkedResources, module: module, bindingArrayLengths: bindingArrayLengths)
        guard linked.externalTextures.isEmpty else { throw ShaderFailure.unsupported("GLSL mesh external textures") }
        let linkedCounts = linked.arrayCounts
        let textures = try flatResourceBindingIndices(linked.textures,
            capacity: 3 * ComputeShaderArtifact.sampledTextureBindingCapacity, arrayCounts: linkedCounts)
        let images = try flatResourceBindingIndices(linked.storageTextures,
            capacity: 3 * ComputeShaderArtifact.storageTextureBindingCapacity, arrayCounts: linkedCounts)
        let samplers = try flatResourceBindingIndices(linked.samplers,
            capacity: 3 * ComputeShaderArtifact.samplerBindingCapacity, arrayCounts: linkedCounts)
        let nonFiltering = try helpers.nonFilteringSamplers(module: module, entry: entry.function, sampledTextures: handles.textures)
        let pairs = try helpers.glslTextureSamplerBindings(module: module, entry: entry.function,
            textureSlots: expandedBindingIndices(textures, counts: linkedCounts), bindingArrayLengths: bindingArrayLengths,
            sharedEntries: linkedEntries)
        declarations += pairs.map(\.declaration)
        for index in indices {
            let global = module.globalVariables.values[index]
            guard let binding = global.binding else { throw ShaderFailure.unsupported("GLSL mesh texture binding") }
            if let flat = samplers[binding] {
                let value = try SamplerShaderBinding(global: global, index: index, flat: flat, module: module,
                    format: .glsl, requiresNonFiltering: nonFiltering.contains(index), arrayCount: counts[binding], webBindingBase: nil)
                declarations += value.declarations
                storage[index] = value.storage
            } else if let flat = images[binding] {
                let value = try StorageTextureShaderBinding(global: global, index: index, flat: flat, module: module,
                    format: .glsl, coherent: helpers.controlBarriers.contains(.texture), arrayCount: counts[binding], webBindingBase: nil)
                declarations += value.declarations
                storage[index] = value.storage
                storageTextures.append(value.reflection)
            } else if let flat = textures[binding] {
                let value = try SampledTextureShaderBinding(global: global, index: index, flat: flat, module: module,
                    format: .glsl, samplingSamplers: helpers.floatTextureSamplers[index] ?? [], arrayCount: counts[binding],
                    webBindingBase: nil, glslAlias: helpers.glslTextureAliases[index])
                declarations += value.declarations
                storage[index] = value.storage
            } else {
                throw ShaderFailure.unsupported("GLSL mesh handle binding")
            }
        }
    }
}
