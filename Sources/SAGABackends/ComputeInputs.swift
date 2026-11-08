import SAGAShaderIR

/// Lowers each entry input to a target builtin and reconstructs structure arguments in the entry body.
struct ComputeInputs {
    private let format: ShaderOutputFormat
    private let workgroupSize: IRWorkgroupSize
    private var builtinReferences: [IRBuiltIn: String] = [:]
    private(set) var parameters: [String] = []
    private(set) var declarations: [String] = []
    private(set) var initializers: [String] = []
    private(set) var references: [String] = []
    private(set) var argumentFields: [Int: [String]] = [:]
    private(set) var structures: [IRHandle<IRType>: ShaderStructure] = [:]
    private(set) var hlslWorkgroupCountBinding: IRResourceBinding?
    private(set) var usesSubgroups = false
    private(set) var usesGlobalInvocationIndex = false
    private(set) var usesWorkgroupIndex = false
    var requiresLinearIndexing: Bool { usesGlobalInvocationIndex || usesWorkgroupIndex }
    private var readsSubgroupID = false
    private var readsSubgroupCount = false
    var requiresSubgroupID: Bool { readsSubgroupID || readsSubgroupCount }
    private var localIndex: String?

    init(arguments: [IRFunctionArgument], workgroupSize: IRWorkgroupSize, module: IRModule, format: ShaderOutputFormat) throws {
        self.format = format
        self.workgroupSize = workgroupSize
        for (index, argument) in arguments.enumerated() {
            let name = "saga_arg_\(index)"
            if case .structure(let members, _) = try module.types.value(for: argument.type).kind {
                let fields = try members.enumerated().map { field, member in
                    try reference(type: member.type, binding: member.binding, name: "\(name)_\(field)", module: module)
                }
                if format == .wgsl { argumentFields[index] = fields }
                let structure = try declareStructure(argument.type, members: members, module: module)
                let values = fields.joined(separator: ", ")
                switch format {
                case .wgsl: initializers.append("    let \(name) = \(structure.constructor)(\(values));")
                case .msl: initializers.append("    \(structure.typeName) \(name) = {\(values)};")
                default: initializers.append("    \(structure.typeName) \(name) = \(structure.constructor)(\(values));")
                }
                references.append(name)
            } else {
                let value = try reference(type: argument.type, binding: argument.binding, name: name, module: module)
                references.append(value)
            }
        }
    }

    private mutating func reference(
        type: IRHandle<IRType>, binding: IRBinding?, name: String, module: IRModule
    ) throws -> String {
        if let subgroup = subgroupInput(binding, name: name, format: format) {
            usesSubgroups = true
            if binding == .builtIn(.subgroupId) { readsSubgroupID = true }
            if binding == .builtIn(.numSubgroups) { readsSubgroupCount = true }
            if let parameter = subgroup.parameter { parameters.append(parameter) }
            return subgroup.value
        }
        guard case .builtIn(let builtin) = binding else {
            throw ShaderFailure.unsupported("Compute code generation: unsupported entry point input")
        }
        if builtin == .globalInvocationIndex || builtin == .workgroupIndex {
            return try linearReference(builtin, name: name)
        }
        let valueType = try shaderValueType(type, module: module, format: format, structures: [:])
        return try builtinReference(builtin, type: valueType, name: name)
    }

    private mutating func linearReference(_ builtin: IRBuiltIn, name: String) throws -> String {
        let global = builtin == .globalInvocationIndex
        if global { usesGlobalInvocationIndex = true } else { usesWorkgroupIndex = true }
        if format == .wgsl {
            let attribute = global ? "global_invocation_index" : "workgroup_index"
            parameters.append("@builtin(\(attribute)) \(name): u32")
        } else {
            let vector = try shaderVectorType(3, scalar: .u32, in: format)
            let counts = try builtinReference(.numWorkgroups, type: vector, name: "saga_grid_groups")
            let coordinate = try builtinReference(global ? .globalInvocationId : .workgroupId,
                                                  type: vector, name: global ? "saga_grid_invocation" : "saga_grid_group")
            let width = global ? "(\(counts).x * \(workgroupSize.x)u)" : "\(counts).x"
            let height = global ? "(\(counts).y * \(workgroupSize.y)u)" : "\(counts).y"
            initializers.append("    uint \(name) = \(coordinate).x + \(width) * (\(coordinate).y + \(height) * \(coordinate).z);")
        }
        return name
    }

    private mutating func builtinReference(_ builtin: IRBuiltIn, type valueType: String, name: String) throws -> String {
        if let reference = builtinReferences[builtin] { return reference }
        let wgsl: String
        let msl: String
        let hlsl: String?
        let glsl: String
        switch builtin {
        case .globalInvocationId:
            (wgsl, msl, hlsl, glsl) = ("global_invocation_id", "thread_position_in_grid", "SV_DispatchThreadID", "gl_GlobalInvocationID")
        case .localInvocationId:
            (wgsl, msl, hlsl, glsl) = ("local_invocation_id", "thread_position_in_threadgroup", "SV_GroupThreadID", "gl_LocalInvocationID")
        case .localInvocationIndex:
            (wgsl, msl, hlsl, glsl) = ("local_invocation_index", "thread_index_in_threadgroup", "SV_GroupIndex", "gl_LocalInvocationIndex")
        case .workgroupId:
            (wgsl, msl, hlsl, glsl) = ("workgroup_id", "threadgroup_position_in_grid", "SV_GroupID", "gl_WorkGroupID")
        case .numWorkgroups:
            (wgsl, msl, hlsl, glsl) = ("num_workgroups", "threadgroups_per_grid", nil, "gl_NumWorkGroups")
        default:
            throw ShaderFailure.unsupported("Compute code generation: unsupported entry point input")
        }
        var reference = name
        switch format {
        case .msl: parameters.append("\(valueType) \(name) [[\(msl)]]")
        case .hlsl:
            if let hlsl {
                parameters.append("\(valueType) \(name) : \(hlsl)")
            } else {
                let target = IRResourceBinding(group: ComputeShaderArtifact.portableBindGroupCount, binding: 0)
                hlslWorkgroupCountBinding = target
                declarations += [
                    "cbuffer SagaDispatch : register(b\(target.binding), space\(target.group)) {",
                    "    uint4 saga_dispatch;",
                    "};",
                ]
                reference = "(saga_dispatch.xyz)"
            }
        case .wgsl: parameters.append("@builtin(\(wgsl)) \(name): \(valueType)")
        default: reference = glsl
        }
        if builtin == .localInvocationIndex { localIndex = reference }
        builtinReferences[builtin] = reference
        return reference
    }

    mutating func subgroupSetup() -> (declarations: [String], body: [String]) {
        guard format == .hlsl, requiresSubgroupID else { return ([], []) }
        let lane = workgroupInitializationIndex()
        var body = [
            "    if (\(lane) == 0u) { saga_subgroup_count = 0u; }",
            "    GroupMemoryBarrierWithGroupSync();",
        ]
        if readsSubgroupID { body.append("    uint saga_subgroup_id = 0u;") }
        let result = readsSubgroupID ? ", saga_subgroup_id" : ""
        body.append("    if (WaveIsFirstLane()) { InterlockedAdd(saga_subgroup_count, 1u\(result)); }")
        if readsSubgroupID { body.append("    saga_subgroup_id = WaveReadLaneFirst(saga_subgroup_id);") }
        if readsSubgroupCount {
            body += [
                "    GroupMemoryBarrierWithGroupSync();",
                "    uint saga_num_subgroups = saga_subgroup_count;",
            ]
        }
        return (["groupshared uint saga_subgroup_count;"], body)
    }

    mutating func workgroupInitializationIndex() -> String {
        if let localIndex { return localIndex }
        let name = "saga_init_lane"
        switch format {
        case .msl: parameters.append("uint \(name) [[thread_index_in_threadgroup]]")
        case .hlsl: parameters.append("uint \(name) : SV_GroupIndex")
        case .wgsl: parameters.append("@builtin(local_invocation_index) \(name): u32")
        default: return "gl_LocalInvocationIndex"
        }
        localIndex = name
        return name
    }

    private mutating func declareStructure(
        _ handle: IRHandle<IRType>, members: [IRStructMember], module: IRModule
    ) throws -> ShaderStructure {
        if let existing = structures[handle] { return existing }
        let type = "SagaComputeInput\(handle.index)"
        let constructor = format == .hlsl ? "saga_make_input_\(handle.index)" : type
        let fieldTypes = try members.map { try shaderValueType($0.type, module: module, format: format, structures: [:]) }
        let structure = ShaderStructure(typeName: type, constructor: constructor)
        declarations += structure.declarations(memberTypes: fieldTypes, in: format)
        structures[handle] = structure
        return structure
    }
}
