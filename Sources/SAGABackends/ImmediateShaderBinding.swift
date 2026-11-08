import SAGAShaderIR

struct ImmediateShaderBinding {
    let index: Int
    let storage: ShaderStorage
    let parameters: [String]
    let captures: [ShaderParameter]
    let declarations: [String]
    let layout: ShaderImmediateLayout

    init?(module: IRModule, reachable: Set<Int>, stage: IRStage, format: ShaderOutputFormat,
          budget: inout ShaderBudget) throws {
        guard let index = reachable.first(where: { module.globalVariables.values[$0].space == .immediate }) else { return nil }
        self.index = index
        let global = module.globalVariables.values[index]
        let name = format == .glsl ? "saga_immediate_\(stage.rawValue)" : "saga_immediate"
        var types = NestedBufferTypes(module: module, format: format, prefix: "SagaImmediateType")
        let root = try types.describe(global.type, budget: &budget)
        storage = .init(reference: name, writable: false,
                        nested: .init(name: name, type: global.type, runtimeLength: nil, space: .immediate))
        let registers = (root.size + 15) / 16
        switch format {
        case .msl:
            let slot: UInt32
            switch stage {
            case .compute: slot = ComputeShaderArtifact.metalImmediateBufferIndex
            case .vertex: slot = RenderStageArtifact.metalVertexImmediateBufferIndex
            case .fragment: slot = RenderStageArtifact.metalFragmentImmediateBufferIndex
            default: throw ShaderFailure.unsupported("Immediate data requires a compute, vertex or fragment stage")
            }
            let pointer = ShaderParameter(type: "constant uint*", name: name)
            parameters = ["\(pointer.type) \(name) [[buffer(\(slot))]]"]
            captures = [pointer]
            declarations = []
            layout = .init(byteSize: root.size, metalBufferIndex: slot)
        case .hlsl:
            // Space 4 is reserved for shader data outside the four resource groups.
            let binding = IRResourceBinding(group: ComputeShaderArtifact.portableBindGroupCount, binding: 1)
            parameters = []
            captures = []
            declarations = [
                "cbuffer SagaImmediate : register(b\(binding.binding), space\(binding.group)) {",
                "    uint4 \(name)[\(registers)];",
                "};",
            ]
            layout = .init(byteSize: root.size, hlslBinding: binding)
        case .glsl:
            // Linked render stages can occupy twice the per-stage uniform range.
            let binding = ComputeShaderArtifact.uniformBindingBase + 2 * ComputeShaderArtifact.uniformBindingCapacity
            parameters = []
            captures = []
            declarations = [
                "layout(std140, binding = \(binding)) uniform SagaImmediate_\(stage.rawValue) { uvec4 words[\(registers)]; } \(name);",
            ]
            layout = .init(byteSize: root.size, glslUniformBinding: binding)
        case .wgsl:
            parameters = []
            captures = []
            declarations = types.declarations + ["var<immediate> \(name): \(root.name);"]
            layout = .init(byteSize: root.size)
        default: throw ShaderFailure.unsupported("Immediate data target")
        }
    }
}
