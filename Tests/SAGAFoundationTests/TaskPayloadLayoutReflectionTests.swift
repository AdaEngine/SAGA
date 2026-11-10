import Testing
import SAGACompiler
import SAGAShaderIR

@Test func taskPayloadReflectsNestedMetalLayout() throws {
    let source = """
    enable wgpu_mesh_shader;
    struct Nested { xy: vec2u, z: u32, }
    struct Payload { items: array<Nested, 2>, color: vec4f, }
    var<task_payload> payload: Payload;
    @task @payload(payload) @workgroup_size(1)
    fn main() -> @builtin(mesh_task_size) vec3u { return vec3u(1u); }
    """
    let layout = try #require(IndependentShaderCompiler().compileReflectedTaskStage(.wgsl(source), entryPoint: "main", to: .msl).payload)
    #expect(layout.size == 48)
    #expect(layout.shape == .structure([
        .init(offset: 0, shape: .array(element: .structure([
            .init(offset: 0, shape: .vector(scalar: .u32, count: 2)),
            .init(offset: 8, shape: .scalar(.u32)),
        ]), count: 2, stride: 16)),
        .init(offset: 32, shape: .vector(scalar: .f32, count: 4)),
    ]))
}

@Test func taskPayloadAcceptsMetalPackedVec3() throws {
    let source = """
    enable wgpu_mesh_shader;
    struct Payload { value: vec3u, tail: u32, }
    var<task_payload> payload: Payload;
    @task @payload(payload) @workgroup_size(1)
    fn main() -> @builtin(mesh_task_size) vec3u { return vec3u(1u); }
    """
    let layout = try #require(IndependentShaderCompiler().compileReflectedTaskStage(.wgsl(source), entryPoint: "main", to: .msl).payload)
    #expect(layout.size == 16)
}
