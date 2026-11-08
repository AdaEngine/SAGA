import SAGAShaderIR

func subgroupInput(
    _ binding: IRBinding?, name: String, format: ShaderOutputFormat
) -> (parameter: String?, value: String)? {
    let builtin: (wgsl: String, msl: String, hlsl: String, glsl: String)
    switch binding {
    case .builtIn(.subgroupSize):
        builtin = ("subgroup_size", "threads_per_simdgroup", "WaveGetLaneCount()", "gl_SubgroupSize")
    case .builtIn(.subgroupInvocationId):
        builtin = ("subgroup_invocation_id", "thread_index_in_simdgroup", "WaveGetLaneIndex()", "gl_SubgroupInvocationID")
    case .builtIn(.subgroupId):
        builtin = ("subgroup_id", "simdgroup_index_in_threadgroup", "saga_subgroup_id", "gl_SubgroupID")
    case .builtIn(.numSubgroups):
        builtin = ("num_subgroups", "simdgroups_per_threadgroup", "saga_num_subgroups", "gl_NumSubgroups")
    default: return nil
    }
    switch format {
    case .wgsl: return ("@builtin(\(builtin.wgsl)) \(name): u32", name)
    case .msl: return ("uint \(name) [[\(builtin.msl)]]", name)
    case .hlsl: return (nil, builtin.hlsl)
    case .glsl: return (nil, builtin.glsl)
    default: return nil
    }
}
