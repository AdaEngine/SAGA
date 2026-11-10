import Testing
import SAGACompiler
import SAGAShaderIR

@Test func taskPayloadReflectsPackedVec3AndScalar() throws {
    let source = """
    enable wgpu_mesh_shader;
    struct Payload { value: vec3u, tail: u32, }
    var<task_payload> payload: Payload;
    @task @payload(payload) @workgroup_size(1)
    fn main() -> @builtin(mesh_task_size) vec3u { return vec3u(1u); }
    """
    let layout = try #require(IndependentShaderCompiler().compileReflectedTaskStage(.wgsl(source), entryPoint: "main", to: .msl).payload)
    #expect(layout.size == 16)
    #expect(layout.shape == .structure([
        .init(offset: 0, shape: .vector(scalar: .u32, count: 3)),
        .init(offset: 12, shape: .scalar(.u32)),
    ]))
}

@Test func taskPayloadReflectsExplicitVec3MemberSize() throws {
    let source = """
    enable wgpu_mesh_shader;
    struct Payload { @size(16) value: vec3u, tail: u32, }
    var<task_payload> payload: Payload;
    @task @payload(payload) @workgroup_size(1)
    fn main() -> @builtin(mesh_task_size) vec3u { return vec3u(1u); }
    """
    let layout = try #require(IndependentShaderCompiler().compileReflectedTaskStage(.wgsl(source), entryPoint: "main", to: .msl).payload)
    #expect(layout.size == 32)
    #expect(layout.shape == .structure([
        .init(offset: 0, shape: .vector(scalar: .u32, count: 3)),
        .init(offset: 16, shape: .scalar(.u32)),
    ]))
}

@Test func taskPayloadReflectsExplicitTailAlignment() throws {
    let source = """
    enable wgpu_mesh_shader;
    struct Payload { value: vec3u, @align(32) tail: u32, }
    var<task_payload> payload: Payload;
    @task @payload(payload) @workgroup_size(1)
    fn main() -> @builtin(mesh_task_size) vec3u { return vec3u(1u); }
    """
    let compiler = IndependentShaderCompiler()
    let layout = try #require(compiler.compileReflectedTaskStage(.wgsl(source), entryPoint: "main", to: .msl).payload)
    #expect(layout.size == 64)
    #expect(layout.shape == .structure([
        .init(offset: 0, shape: .vector(scalar: .u32, count: 3)),
        .init(offset: 32, shape: .scalar(.u32)),
    ]))
    _ = try compiler.compileTaskStage(.wgsl(source), entryPoint: "main", to: .msl)
}
