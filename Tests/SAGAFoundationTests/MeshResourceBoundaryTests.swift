import Testing
import SAGACompiler
import SAGAShaderIR

@Test func reflectedMeshIncludesReachableBufferResources() throws {
    let declarations = """
    enable wgpu_mesh_shader;
    struct Vertex { @builtin(position) position: vec4f, }
    struct Primitive { @builtin(triangle_indices) indices: vec3u, }
    struct Output {
        @builtin(vertices) vertices: array<Vertex, 3>,
        @builtin(primitives) primitives: array<Primitive, 1>,
        @builtin(vertex_count) vertex_count: u32,
        @builtin(primitive_count) primitive_count: u32,
    }
    var<workgroup> output: Output;
    """
    for (index, resource) in [
        "@group(0) @binding(0) var<uniform> value: u32;",
        "@group(0) @binding(0) var<storage, read> value: u32;",
    ].enumerated() {
        let source = declarations + resource + """

        @mesh(output) @workgroup_size(1) fn main() {
            output.vertex_count = value;
        }
        """
        let stage = try IndependentShaderCompiler().compileReflectedMeshStage(.wgsl(source), entryPoint: "main", to: .msl)
        #expect(stage.uniformBindings.count == (index == 0 ? 1 : 0))
        #expect(stage.storageBindings.count == (index == 1 ? 1 : 0))
        guard case .text(_, let msl) = stage.artifact else { throw ShaderFailure.incompatibleArtifact }
        #expect(msl.contains("[[buffer("))
    }
}

@Test func reflectedTaskIncludesReachableBufferResources() throws {
    for (index, resource) in [
        "@group(0) @binding(0) var<uniform> value: u32;",
        "@group(0) @binding(0) var<storage, read> value: u32;",
    ].enumerated() {
        let source = """
        enable wgpu_mesh_shader;
        struct Payload { value: u32, }
        var<task_payload> payload: Payload;
        \(resource)
        @task @payload(payload) @workgroup_size(1)
        fn main() -> @builtin(mesh_task_size) vec3u {
            payload.value = value;
            return vec3u(1);
        }
        """
        let stage = try IndependentShaderCompiler().compileReflectedTaskStage(.wgsl(source), entryPoint: "main", to: .msl)
        #expect(stage.uniformBindings.count == (index == 0 ? 1 : 0))
        #expect(stage.storageBindings.count == (index == 1 ? 1 : 0))
        guard case .text(_, let msl) = stage.artifact else { throw ShaderFailure.incompatibleArtifact }
        #expect(msl.contains("[[buffer("))
    }
}

