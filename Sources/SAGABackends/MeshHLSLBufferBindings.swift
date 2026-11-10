import SAGAShaderIR
import SAGAValidation

/// HLSL task and mesh buffer declarations and resource mappings.
struct MeshHLSLBufferBindings {
    var declarations: [String] = []
    var parameters: [String] = []
    var captures: [[ShaderParameter]]
    var storage: [ShaderStorage]
    var uniforms: [ComputeShaderArtifact.UniformBinding] = []
    var buffers: [ComputeShaderArtifact.StorageBinding] = []

    init(module: IRModule, entry: IREntryPoint, helpers: inout ShaderFunctions, excluded: Set<Int>,
         bindingArrayLengths: [IRResourceBinding: UInt32] = [:]) throws {
        captures = module.globalVariables.values.map { _ in [] }
        storage = module.globalVariables.values.map { _ in ShaderStorage(reference: "", writable: false) }
        let indices = helpers.reachableGlobals.subtracting(excluded).sorted()
        let resources = indices.map { module.globalVariables.values[$0] }
        guard resources.allSatisfy({ global in
            if global.space == .uniform { return true }
            if case .storage = global.space { return true }
            return false
        }) else { throw ShaderFailure.unsupported("Mesh stage resource profile") }
        let counts = try handleResourceGlobals(resources, module: module, bindingArrayLengths: bindingArrayLengths).arrayCounts
        let uniformSlots = try flatResourceBindingIndices(resources.filter { $0.space == .uniform }, capacity: ComputeShaderArtifact.uniformBindingCapacity, arrayCounts: counts)
        let storageSlots = try flatResourceBindingIndices(resources.filter { if case .storage = $0.space { return true }; return false }, capacity: ComputeShaderArtifact.storageBindingCapacity, arrayCounts: counts)
        let aggregates = try helpers.aggregateBufferGlobals(module: module, entry: entry.function)
        var budget = ShaderBudget()
        var runtimeIndex = 0
        for index in indices {
            let global = module.globalVariables.values[index]
            guard let binding = global.binding else { throw ShaderFailure.unsupported("Mesh stage buffer binding") }
            if global.space == .uniform {
                guard let flat = uniformSlots[binding] else { throw ShaderFailure.unsupported("Mesh stage uniform binding") }
                let value = try UniformShaderBinding(global: global, index: index, flat: flat, module: module, format: .hlsl, arrayCount: counts[binding], webBindingBase: nil, aggregateAccess: aggregates.contains(index))
                declarations += value.declarations; parameters += value.parameters
                captures[index] = value.captures; storage[index] = value.storage; uniforms.append(value.reflection)
            } else {
                guard let flat = storageSlots[binding] else { throw ShaderFailure.unsupported("Mesh stage storage binding") }
                let value = try StorageBufferShaderBinding(global: global, index: index, flatBinding: flat, runtimeIndex: runtimeIndex, metalLengths: .init(type: "constant uint*", name: "saga_lengths"), coherent: helpers.controlBarriers.contains(.storage), module: module, format: .hlsl, arrayCount: counts[binding], webGPU: false, webBindingBase: nil, aggregateAccess: aggregates.contains(index), budget: &budget)
                declarations += value.declarations; parameters += value.parameters
                captures[index] = value.captures; storage[index] = value.storage; buffers.append(value.reflection)
                if value.reflection.runtimeArrayStride != nil { runtimeIndex += Int(value.reflection.bindingArray?.count ?? 1) }
            }
        }
    }
}
