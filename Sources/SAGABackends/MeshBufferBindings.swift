import SAGAShaderIR
import SAGAValidation

/// Buffer bindings shared by the Metal object and mesh stages.
struct MeshBufferBindings {
    var declarations: [String] = []
    var parameters: [String] = []
    var captures: [[ShaderParameter]]
    var storage: [ShaderStorage]
    var uniforms: [ComputeShaderArtifact.UniformBinding] = []
    var buffers: [ComputeShaderArtifact.StorageBinding] = []
    var textures: [ComputeShaderArtifact.SampledTextureBinding] = []
    var samplers: [ComputeShaderArtifact.SamplerBinding] = []
    var storageTextures: [ComputeShaderArtifact.StorageTextureBinding] = []

    init(module: IRModule, entry: IREntryPoint, helpers: inout ShaderFunctions, excluded: Set<Int>, bindingArrayLengths: [IRResourceBinding: UInt32] = [:]) throws {
        captures = module.globalVariables.values.map { _ in [] }
        storage = module.globalVariables.values.map { _ in ShaderStorage(reference: "", writable: false) }
        let indices = helpers.reachableGlobals.subtracting(excluded).sorted()
        let resources = indices.map { module.globalVariables.values[$0] }
        guard resources.allSatisfy({ global in
            if global.space == .uniform || global.space == .handle { return true }
            if case .storage = global.space { return true }
            return false
        }) else { throw ShaderFailure.unsupported("Mesh stage resource profile") }
        let handles = try handleResourceGlobals(resources, module: module, bindingArrayLengths: bindingArrayLengths)
        guard handles.externalTextures.isEmpty else {
            throw ShaderFailure.unsupported("Mesh stage texture resource profile")
        }
        let textureSlots = try flatResourceBindingIndices(handles.textures, capacity: ComputeShaderArtifact.sampledTextureBindingCapacity, arrayCounts: handles.arrayCounts)
        let storageTextureSlots = try flatResourceBindingIndices(handles.storageTextures, capacity: ComputeShaderArtifact.storageTextureBindingCapacity, arrayCounts: handles.arrayCounts)
        let samplerSlots = try flatResourceBindingIndices(handles.samplers, capacity: ComputeShaderArtifact.samplerBindingCapacity, arrayCounts: handles.arrayCounts)
        let nonFiltering = try helpers.nonFilteringSamplers(module: module, entry: entry.function, sampledTextures: handles.textures)
        let uniformSlots = try flatResourceBindingIndices(resources.filter { $0.space == .uniform }, capacity: ComputeShaderArtifact.uniformBindingCapacity, arrayCounts: handles.arrayCounts)
        let storageSlots = try flatResourceBindingIndices(resources.filter { if case .storage = $0.space { return true }; return false }, capacity: ComputeShaderArtifact.storageBindingCapacity, arrayCounts: handles.arrayCounts)
        let aggregates = try helpers.aggregateBufferGlobals(module: module, entry: entry.function)
        var budget = ShaderBudget()
        var runtimeIndex = 0
        let metalLengths = ShaderParameter(type: "constant uint*", name: "saga_lengths")
        for index in indices {
            let global = module.globalVariables.values[index]
            guard let binding = global.binding else { throw ShaderFailure.unsupported("Mesh stage buffer binding") }
            if global.space == .handle {
                if let flat = textureSlots[binding] {
                    let value = try SampledTextureShaderBinding(global: global, index: index, flat: flat, module: module, format: .msl, samplingSamplers: helpers.floatTextureSamplers[index] ?? [], arrayCount: handles.arrayCounts[binding], webBindingBase: nil, glslAlias: nil)
                    declarations += value.declarations; parameters += value.parameters
                    captures[index] = value.captures; storage[index] = value.storage; textures.append(value.reflection)
                } else if let flat = storageTextureSlots[binding] {
                    let value = try StorageTextureShaderBinding(global: global, index: index, flat: flat, module: module, format: .msl, coherent: helpers.controlBarriers.contains(.texture), arrayCount: handles.arrayCounts[binding], webBindingBase: nil)
                    declarations += value.declarations; parameters += value.parameters
                    captures[index] = value.captures; storage[index] = value.storage; storageTextures.append(value.reflection)
                } else if let flat = samplerSlots[binding] {
                    let value = try SamplerShaderBinding(global: global, index: index, flat: flat, module: module, format: .msl, requiresNonFiltering: nonFiltering.contains(index), arrayCount: handles.arrayCounts[binding], webBindingBase: nil)
                    declarations += value.declarations; parameters += value.parameters
                    captures[index] = value.captures; storage[index] = value.storage; samplers.append(value.reflection)
                } else { throw ShaderFailure.unsupported("Mesh stage texture binding") }
            } else if global.space == .uniform {
                guard let flat = uniformSlots[binding] else { throw ShaderFailure.unsupported("Mesh stage uniform binding") }
                let value = try UniformShaderBinding(global: global, index: index, flat: flat, module: module, format: .msl, arrayCount: handles.arrayCounts[binding], webBindingBase: nil, aggregateAccess: aggregates.contains(index))
                declarations += value.declarations; parameters += value.parameters
                captures[index] = value.captures; storage[index] = value.storage; uniforms.append(value.reflection)
            } else {
                guard let flat = storageSlots[binding] else { throw ShaderFailure.unsupported("Mesh stage storage binding") }
                let value = try StorageBufferShaderBinding(global: global, index: index, flatBinding: flat, runtimeIndex: runtimeIndex, metalLengths: metalLengths, coherent: helpers.controlBarriers.contains(.storage), module: module, format: .msl, arrayCount: handles.arrayCounts[binding], webGPU: false, webBindingBase: nil, aggregateAccess: aggregates.contains(index), budget: &budget)
                declarations += value.declarations; parameters += value.parameters
                captures[index] = value.captures; storage[index] = value.storage; buffers.append(value.reflection)
                if value.reflection.runtimeArrayStride != nil { runtimeIndex += Int(value.reflection.bindingArray?.count ?? 1) }
            }
        }
        if runtimeIndex > 0 {
            parameters.append("\(metalLengths.type) \(metalLengths.name) [[buffer(\(ComputeShaderArtifact.storageBindingCapacity))]]")
        }
    }
}
