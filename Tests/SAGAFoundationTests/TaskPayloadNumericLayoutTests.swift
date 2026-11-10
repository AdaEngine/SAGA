import Testing
import SAGACompiler
import SAGAShaderIR

@Test func taskPayloadReflectsHalfAndNarrowIntegerLayouts() throws {
    let source = """
    enable wgpu_mesh_shader, wgpu_int16, f16;
    struct Payload { a: f16, b: vec2<u16>, c: vec3<i16>, d: array<vec2<f16>, 2>, }
    var<task_payload> payload: Payload;
    @task @payload(payload) @workgroup_size(1)
    fn main() -> @builtin(mesh_task_size) vec3u { return vec3u(1u); }
    """
    let layout = try #require(IndependentShaderCompiler().compileReflectedTaskStage(.wgsl(source), entryPoint: "main", to: .msl).payload)
    #expect(layout.size == 24)
    #expect(layout.shape == .structure([
        .init(offset: 0, shape: .scalar(.f16)),
        .init(offset: 4, shape: .vector(scalar: .u16, count: 2)),
        .init(offset: 8, shape: .vector(scalar: .i16, count: 3)),
        .init(offset: 16, shape: .array(element: .vector(scalar: .f16, count: 2), count: 2, stride: 4)),
    ]))
}

@Test func taskPayloadAcceptsHalfVec3TailPacking() throws {
    let source = """
    enable wgpu_mesh_shader, f16;
    struct Payload { value: vec3<f16>, tail: f16, }
    var<task_payload> payload: Payload;
    @task @payload(payload) @workgroup_size(1)
    fn main() -> @builtin(mesh_task_size) vec3u { return vec3u(1u); }
    """
    let layout = try #require(IndependentShaderCompiler().compileReflectedTaskStage(.wgsl(source), entryPoint: "main", to: .msl).payload)
    #expect(layout.size == 8)
}

@Test func taskPayloadReflectsWideIntegerLayouts() throws {
    let source = """
    enable wgpu_mesh_shader;
    struct Payload { a: i64, b: vec2<u64>, c: array<vec3<i64>, 2>, }
    var<task_payload> payload: Payload;
    @task @payload(payload) @workgroup_size(1)
    fn main() -> @builtin(mesh_task_size) vec3u { return vec3u(1u); }
    """
    let layout = try #require(IndependentShaderCompiler().compileReflectedTaskStage(.wgsl(source), entryPoint: "main", to: .msl).payload)
    #expect(layout.size == 96)
    #expect(layout.shape == .structure([
        .init(offset: 0, shape: .scalar(.i64)),
        .init(offset: 16, shape: .vector(scalar: .u64, count: 2)),
        .init(offset: 32, shape: .array(element: .vector(scalar: .i64, count: 3), count: 2, stride: 32)),
    ]))
}
