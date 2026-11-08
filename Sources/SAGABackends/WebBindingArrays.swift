import SAGAShaderIR

func webBindingArrayBases(
    module: IRModule, activeCounts: [IRResourceBinding: UInt32]
) throws -> [IRResourceBinding: UInt32] {
    guard !activeCounts.isEmpty else { return [:] }
    var occupied: [UInt32: Set<UInt32>] = [:]
    var arrays: [IRResourceBinding: UInt32] = [:]
    for global in module.globalVariables.values {
        guard let binding = global.binding else { continue }
        occupied[binding.group, default: []].insert(binding.binding)
        guard case .bindingArray(let base, _) = try module.types.value(for: global.type).kind else { continue }
        let element = try module.types.value(for: base).kind
        let span: UInt32
        switch global.space {
        case .uniform: span = ComputeShaderArtifact.uniformBindingCapacity
        case .storage: span = ComputeShaderArtifact.storageBindingCapacity
        case .handle:
            if element.isSampler { span = ComputeShaderArtifact.samplerBindingCapacity }
            else if element.storageTexture != nil { span = ComputeShaderArtifact.storageTextureBindingCapacity }
            else { span = ComputeShaderArtifact.sampledTextureBindingCapacity }
        default: continue
        }
        arrays[binding] = max(arrays[binding] ?? 0, span)
    }

    // Reserve the same spans for every stage of this module, including inactive declarations.
    var upperBounds: [UInt32: UInt32] = [:]
    var bases: [IRResourceBinding: UInt32] = [:]
    for (binding, span) in arrays.sorted(by: { ($0.0.group, $0.0.binding) < ($1.0.group, $1.0.binding) }) {
        var upper = upperBounds[binding.group] ?? ComputeShaderArtifact.portableBindingIndexLimit
        let groupBindings = occupied[binding.group, default: []]
        while upper >= span {
            let base = upper - span
            if (base..<upper).allSatisfy({ !groupBindings.contains($0) }) {
                upper = base
                if activeCounts[binding] != nil { bases[binding] = base }
                break
            }
            upper -= 1
        }
        upperBounds[binding.group] = upper
    }
    guard bases.count == activeCounts.count else {
        throw ShaderFailure.unsupported("WebGPU binding arrays exceed available physical binding indices")
    }
    return bases
}
