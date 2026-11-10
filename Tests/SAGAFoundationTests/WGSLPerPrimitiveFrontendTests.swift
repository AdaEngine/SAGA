import SAGAFrontends
import SAGAShaderIR
import Testing

@Test func wgslPerPrimitivePreservesLocationBinding() throws {
    let source = """
    enable wgpu_mesh_shader;
    struct Primitive { @location(2) @per_primitive @interpolate(flat) value: u32, }
    """
    let module = try WGSLFrontend(options: .init(capabilities: [.meshShader])).parse(.wgsl(source))
    let type = try #require(module.types.values.first { $0.name == "Primitive" })
    guard case .structure(let members, _) = type.kind else { Issue.record("Expected structure"); return }
    #expect(members.first?.binding == .location(index: 2, interpolation: .flat, sampling: .first, blendSource: nil, perPrimitive: true))
}

@Test(arguments: [
    "struct P { @location(0) @per_primitive value: u32, }",
    "enable wgpu_mesh_shader; struct P { @per_primitive value: u32, }",
    "enable wgpu_mesh_shader; struct P { @builtin(position) @per_primitive value: vec4f, }",
    "enable wgpu_mesh_shader; struct P { @location(0) @per_primitive() value: u32, }",
    "enable wgpu_mesh_shader; struct P { @location(0) @per_primitive @per_primitive value: u32, }",
])
func wgslPerPrimitiveRejectsInvalidAttribute(_ source: String) {
    #expect(throws: ShaderFailure.self) {
        try WGSLFrontend(options: .init(capabilities: [.meshShader])).parse(.wgsl(source))
    }
}
