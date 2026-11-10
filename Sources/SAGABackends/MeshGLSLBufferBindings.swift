import SAGAShaderIR
import SAGAValidation

/// GLSL mesh buffer declarations and their mappings for the shader body writer.
struct MeshGLSLBufferBindings {
    var declarations: [String] = []
    var storage: [ShaderStorage]
    var uniforms: [ComputeShaderArtifact.UniformBinding] = []
    var buffers: [ComputeShaderArtifact.StorageBinding] = []
    var usesGLSL16BitStorage = false

    init(module: IRModule, entry: IREntryPoint, helpers: inout ShaderFunctions, excluded: Set<Int>,
         bindingArrayLengths: [IRResourceBinding: UInt32] = [:]) throws {
        storage = module.globalVariables.values.map { _ in ShaderStorage(reference: "", writable: false) }
        let indices = helpers.reachableGlobals.subtracting(excluded).sorted()
        let resources = indices.map { module.globalVariables.values[$0] }
        guard resources.allSatisfy({ global in
            if global.space == .uniform { return true }
            if case .storage = global.space { return true }
            return false
        }) else { throw ShaderFailure.unsupported("Mesh stage resource profile") }
        let counts = try handleResourceGlobals(resources, module: module, bindingArrayLengths: bindingArrayLengths).arrayCounts
        let uniformBuffers = resources.filter { $0.space == .uniform }
        let storageBuffers = resources.filter { if case .storage = $0.space { return true }; return false }
        // Check each stage's limits before allocating a shared linked namespace.
        _ = try flatResourceBindingIndices(uniformBuffers, capacity: ComputeShaderArtifact.uniformBindingCapacity, arrayCounts: counts)
        _ = try flatResourceBindingIndices(storageBuffers, capacity: ComputeShaderArtifact.storageBindingCapacity, arrayCounts: counts)
        let linkedEntries = module.entryPoints.filter { $0.stage == .task || $0.stage == .mesh || $0.stage == .fragment }.map(\.function)
        let linkedIndices = try helpers.glslLinkedGlobals(module: module, entry: entry.function, sharedEntries: linkedEntries)
        let linkedResources = linkedIndices.sorted().map { module.globalVariables.values[$0] }.filter {
            if $0.space == .uniform { return true }
            if case .storage = $0.space { return true }
            return false
        }
        let linkedBindings = linkedResources.compactMap(\.binding)
        guard Set(linkedBindings).count == linkedBindings.count else {
            throw ShaderFailure.unsupported("GLSL task, mesh, and fragment buffer bindings alias across linked stages")
        }
        let linkedCounts = try handleResourceGlobals(linkedResources, module: module, bindingArrayLengths: bindingArrayLengths).arrayCounts
        let uniformSlots = try flatResourceBindingIndices(linkedResources.filter { $0.space == .uniform }, capacity: 3 * ComputeShaderArtifact.uniformBindingCapacity, arrayCounts: linkedCounts)
        let storageSlots = try flatResourceBindingIndices(linkedResources.filter { if case .storage = $0.space { return true }; return false }, capacity: 3 * ComputeShaderArtifact.storageBindingCapacity, arrayCounts: linkedCounts)
        let aggregates = try helpers.aggregateBufferGlobals(module: module, entry: entry.function,
                                                             sharedEntries: linkedEntries)
        var budget = ShaderBudget()
        var runtimeIndex = 0
        for index in indices {
            let global = module.globalVariables.values[index]
            guard let binding = global.binding else { throw ShaderFailure.unsupported("Mesh stage buffer binding") }
            if global.space == .uniform {
                guard let flat = uniformSlots[binding] else { throw ShaderFailure.unsupported("Mesh stage uniform binding") }
                let value = try UniformShaderBinding(global: global, index: index, flat: flat, module: module, format: .glsl, arrayCount: counts[binding], webBindingBase: nil, aggregateAccess: aggregates.contains(index))
                declarations += value.declarations
                storage[index] = value.storage
                uniforms.append(value.reflection)
                usesGLSL16BitStorage = usesGLSL16BitStorage || value.usesGLSL16BitStorage
            } else {
                guard let flat = storageSlots[binding] else { throw ShaderFailure.unsupported("Mesh stage storage binding") }
                let value = try StorageBufferShaderBinding(global: global, index: index, flatBinding: flat, runtimeIndex: runtimeIndex, metalLengths: .init(type: "constant uint*", name: "saga_lengths"), coherent: helpers.controlBarriers.contains(.storage), module: module, format: .glsl, arrayCount: counts[binding], webGPU: false, webBindingBase: nil, aggregateAccess: aggregates.contains(index), budget: &budget)
                declarations += value.declarations
                storage[index] = value.storage
                buffers.append(value.reflection)
                usesGLSL16BitStorage = usesGLSL16BitStorage || value.usesGLSL16BitStorage
                if value.reflection.runtimeArrayStride != nil { runtimeIndex += Int(value.reflection.bindingArray?.count ?? 1) }
            }
        }
    }
}
