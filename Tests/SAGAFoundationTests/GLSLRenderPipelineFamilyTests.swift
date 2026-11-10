import Testing
import SAGACompiler
import SAGAShaderIR

private func familyShader(useOverride: Bool, includeIrrelevant: Bool) -> ShaderInput {
    let count = useOverride ? "override fragmentCount = 2u;" : "const fragmentCount = 2u;"
    let extra = includeIrrelevant ? """
        @group(0) @binding(4) var<uniform> unrelated: binding_array<Data, 4>;
        @compute @workgroup_size(1) fn compute() { let ignored = unrelated[0].value; }
        """ : ""
    return .wgsl("""
        enable wgpu_mesh_shader;
        enable wgpu_binding_array;
        \(count)
        struct Data { value: vec4f, }
        struct V { @builtin(position) position: vec4f, }
        struct P { @builtin(triangle_indices) indices: vec3u, }
        struct O {
          @builtin(vertices) vertices: array<V, 3>,
          @builtin(primitives) primitives: array<P, 1>,
          @builtin(vertex_count) vertex_count: u32,
          @builtin(primitive_count) primitive_count: u32,
        }
        var<workgroup> out: O;
        @group(0) @binding(0) var<uniform> vertexData: binding_array<Data, 2>;
        @group(0) @binding(1) var<uniform> meshData: binding_array<Data, 3>;
        @group(0) @binding(2) var<uniform> sharedData: Data;
        @group(0) @binding(3) var<uniform> fragmentData: binding_array<Data, fragmentCount>;
        \(extra)
        @vertex fn vertex() -> @builtin(position) vec4f {
          return vertexData[1].value + sharedData.value;
        }
        @mesh(out) @workgroup_size(1) fn mesh() {
          out.vertex_count = 3u; out.primitive_count = 1u;
          out.vertices[0].position = meshData[2].value + sharedData.value;
          out.primitives[0].indices = vec3u(0u, 1u, 2u);
        }
        @fragment fn fragment(@builtin(position) p: vec4f) -> @location(0) vec4f {
          return sharedData.value + fragmentData[u32(p.x)].value;
        }
        """)
}

@Test(arguments: [false, true], [false, true])
func glslFragmentSelectsLinkedPipelineFamily(useOverride: Bool, includeIrrelevant: Bool) throws {
    let input = familyShader(useOverride: useOverride, includeIrrelevant: includeIrrelevant)
    let compiler = IndependentShaderCompiler()
    let constants: [String: Double] = useOverride ? ["fragmentCount": 3] : [:]
    let vertex = try compiler.compileRenderStage(input, entryPoint: "vertex", to: .glsl, constants: constants)
    let mesh = try compiler.compileMeshStage(input, entryPoint: "mesh", to: .glsl, constants: constants)
    let vertexFragment = try compiler.compileRenderStage(input, entryPoint: "fragment", to: .glsl,
        glslPipelineFamily: .vertex, constants: constants)
    let meshFragment = try compiler.compileRenderStage(input, entryPoint: "fragment", to: .glsl,
        glslPipelineFamily: .mesh, constants: constants)

    let vertexShared = try #require(vertex.uniformBindings.first { $0.binding.binding == 2 })
    let vertexFragmentShared = try #require(vertexFragment.uniformBindings.first { $0.binding.binding == 2 })
    let meshFragmentShared = try #require(meshFragment.uniformBindings.first { $0.binding.binding == 2 })
    #expect(vertexShared.flatBindingIndex == vertexFragmentShared.flatBindingIndex)
    guard case .text(.glsl, let meshSource) = mesh else {
        Issue.record("Expected GLSL mesh stage")
        return
    }
    // The mesh-stage public API returns an artifact, so inspect its shared buffer declaration.
    let sharedLine = try #require(meshSource.split(separator: "\n").first {
        $0.contains("layout(") && $0.contains("saga_buffer_3;")
    })
    let meshSharedIndex = try #require(meshFragmentShared.flatBindingIndex)
    #expect(sharedLine.contains("binding = \(meshSharedIndex)"))
    #expect(vertexFragmentShared.flatBindingIndex != meshFragmentShared.flatBindingIndex)
    #expect(vertexFragment.uniformBindings.first { $0.binding.binding == 3 }?.bindingArray?.count == (useOverride ? 3 : 2))
    #expect(meshFragment.uniformBindings.first { $0.binding.binding == 3 }?.bindingArray?.count == (useOverride ? 3 : 2))
}

@Test(arguments: [false, true], [false, true])
func glslFragmentRequiresFamilyWhenVertexAndMeshExist(useOverride: Bool, includeIrrelevant: Bool) {
    let constants: [String: Double] = useOverride ? ["fragmentCount": 3] : [:]
    #expect(throws: ShaderFailure.self) {
        try IndependentShaderCompiler().compileRenderStage(
            familyShader(useOverride: useOverride, includeIrrelevant: includeIrrelevant),
            entryPoint: "fragment", to: .glsl, constants: constants)
    }
}
