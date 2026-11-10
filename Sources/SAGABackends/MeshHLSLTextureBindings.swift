import SAGAShaderIR
import SAGAValidation

/// HLSL task and mesh texture and sampler declarations.
struct MeshHLSLTextureBindings {
    var declarations: [String] = []
    var storage: [Int: ShaderStorage] = [:]
    var textures: [ComputeShaderArtifact.SampledTextureBinding] = []
    var samplers: [ComputeShaderArtifact.SamplerBinding] = []
    var storageTextures: [ComputeShaderArtifact.StorageTextureBinding] = []

    init(module: IRModule, entry: IREntryPoint, helpers: inout ShaderFunctions,
         bindingArrayLengths: [IRResourceBinding: UInt32] = [:]) throws {
        let indices = helpers.reachableGlobals.sorted().filter { module.globalVariables.values[$0].space == .handle }
        let globals = indices.map { module.globalVariables.values[$0] }
        let handles = try handleResourceGlobals(globals, module: module, bindingArrayLengths: bindingArrayLengths)
        guard handles.externalTextures.isEmpty else {
            throw ShaderFailure.unsupported("HLSL mesh texture resource profile")
        }
        let textureSlots = try flatResourceBindingIndices(handles.textures, capacity: ComputeShaderArtifact.sampledTextureBindingCapacity, arrayCounts: handles.arrayCounts)
        let imageSlots = try flatResourceBindingIndices(handles.storageTextures, capacity: ComputeShaderArtifact.storageTextureBindingCapacity, arrayCounts: handles.arrayCounts)
        let samplerSlots = try flatResourceBindingIndices(handles.samplers, capacity: ComputeShaderArtifact.samplerBindingCapacity, arrayCounts: handles.arrayCounts)
        let nonFiltering = try helpers.nonFilteringSamplers(module: module, entry: entry.function, sampledTextures: handles.textures)
        for index in indices {
            let global = module.globalVariables.values[index]
            guard let binding = global.binding else { throw ShaderFailure.unsupported("HLSL mesh texture binding") }
            if let flat = samplerSlots[binding] {
                let value = try SamplerShaderBinding(global: global, index: index, flat: flat, module: module, format: .hlsl,
                    requiresNonFiltering: nonFiltering.contains(index), arrayCount: handles.arrayCounts[binding], webBindingBase: nil)
                declarations += value.declarations; storage[index] = value.storage; samplers.append(value.reflection)
            } else if let flat = imageSlots[binding] {
                let value = try StorageTextureShaderBinding(global: global, index: index, flat: flat, module: module, format: .hlsl,
                    coherent: helpers.controlBarriers.contains(.texture), arrayCount: handles.arrayCounts[binding], webBindingBase: nil)
                declarations += value.declarations; storage[index] = value.storage; storageTextures.append(value.reflection)
            } else if let flat = textureSlots[binding] {
                let value = try SampledTextureShaderBinding(global: global, index: index, flat: flat, module: module, format: .hlsl,
                    samplingSamplers: helpers.floatTextureSamplers[index] ?? [], arrayCount: handles.arrayCounts[binding], webBindingBase: nil, glslAlias: nil)
                declarations += value.declarations; storage[index] = value.storage; textures.append(value.reflection)
            } else { throw ShaderFailure.unsupported("HLSL mesh handle binding") }
        }
    }
}
