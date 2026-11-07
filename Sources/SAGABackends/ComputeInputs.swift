import SAGAShaderIR

/// Lowers each entry input to a target builtin and reconstructs structure arguments in the entry body.
struct ComputeInputs {
    private let format: ShaderOutputFormat
    private(set) var parameters: [String] = []
    private(set) var declarations: [String] = []
    private(set) var initializers: [String] = []
    private(set) var references: [String] = []
    private(set) var structures: [IRHandle<IRType>: ShaderStructure] = [:]
    private(set) var hlslWorkgroupCountBinding: IRResourceBinding?
    private var localIndex: String?

    init(arguments: [IRFunctionArgument], module: IRModule, format: ShaderOutputFormat) throws {
        self.format = format
        for (index, argument) in arguments.enumerated() {
            let name = "saga_arg_\(index)"
            if case .structure(let members, _) = try module.types.value(for: argument.type).kind {
                let fields = try members.enumerated().map { field, member in
                    try reference(type: member.type, binding: member.binding, name: "\(name)_\(field)", module: module)
                }
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
        let wgsl: String
        let msl: String
        let hlsl: String?
        let glsl: String
        switch binding {
        case .builtIn(.globalInvocationId):
            (wgsl, msl, hlsl, glsl) = ("global_invocation_id", "thread_position_in_grid", "SV_DispatchThreadID", "gl_GlobalInvocationID")
        case .builtIn(.localInvocationId):
            (wgsl, msl, hlsl, glsl) = ("local_invocation_id", "thread_position_in_threadgroup", "SV_GroupThreadID", "gl_LocalInvocationID")
        case .builtIn(.localInvocationIndex):
            (wgsl, msl, hlsl, glsl) = ("local_invocation_index", "thread_index_in_threadgroup", "SV_GroupIndex", "gl_LocalInvocationIndex")
            localIndex = format == .glsl ? glsl : name
        case .builtIn(.workgroupId):
            (wgsl, msl, hlsl, glsl) = ("workgroup_id", "threadgroup_position_in_grid", "SV_GroupID", "gl_WorkGroupID")
        case .builtIn(.numWorkgroups):
            (wgsl, msl, hlsl, glsl) = ("num_workgroups", "threadgroups_per_grid", nil, "gl_NumWorkGroups")
        default:
            throw ShaderFailure.unsupported("Compute code generation: unsupported entry point input")
        }
        let valueType = try shaderValueType(type, module: module, format: format, structures: [:])
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
                return "(saga_dispatch.xyz)"
            }
        case .wgsl: parameters.append("@builtin(\(wgsl)) \(name): \(valueType)")
        default: return glsl
        }
        return name
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
