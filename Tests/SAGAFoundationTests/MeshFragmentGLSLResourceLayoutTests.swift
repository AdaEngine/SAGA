import Foundation
import Testing
import SAGACompiler
import SAGAShaderIR

// Compare emitted declarations by their stable global-derived names. GLSL has
// independent namespaces for buffers, combined samplers, and storage images.
private func meshFragmentBindings(_ source: String, containing marker: String) -> [String: Int] {
    var result: [String: Int] = [:]
    for line in source.split(separator: "\n") where line.contains("layout(") && line.contains(marker) {
        // Sampled textures and storage images have separate OpenGL namespaces.
        if marker == "saga_texture_" && !line.contains("image2D") { continue }
        guard let binding = line.range(of: "binding"),
              let equal = line[binding.upperBound...].firstIndex(of: "=") else { continue }
        let digits = line[line.index(after: equal)...].drop(while: { $0 == " " }).prefix(while: { $0.isNumber })
        guard let unit = Int(digits), let nameRange = line.range(of: marker) else { continue }
        let name = line[nameRange.lowerBound...].prefix(while: { $0.isLetter || $0.isNumber || $0 == "_" })
        result[String(name)] = unit
    }
    return result
}

private func assertLinkedMeshFragmentBindings(_ mesh: String, _ fragment: String, marker: String, sharedCount: Int) {
    let meshBindings = meshFragmentBindings(mesh, containing: marker)
    let fragmentBindings = meshFragmentBindings(fragment, containing: marker)
    let shared = Set(meshBindings.keys).intersection(fragmentBindings.keys)
    #expect(shared.count == sharedCount)
    for name in shared { #expect(meshBindings[name] == fragmentBindings[name]) }
    let combined = meshBindings.merging(fragmentBindings) { first, _ in first }
    #expect(Set(combined.values).count == combined.count)
    #expect(meshBindings.keys.contains { !shared.contains($0) })
    #expect(fragmentBindings.keys.contains { !shared.contains($0) })
}

@Test(arguments: [false, true])
func glslMeshFragmentAllocateSharedResourceRangesTogether(useOverride: Bool) throws {
    let countDeclaration = useOverride ? "override fragmentCount = 2u;" : "const fragmentCount = 2u;"
    let input = ShaderInput.wgsl("""
        enable wgpu_mesh_shader;
        enable wgpu_binding_array;
        \(countDeclaration)
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
        @group(0) @binding(0) var<uniform> meshData: binding_array<Data, 2>;
        @group(0) @binding(1) var<uniform> fragmentData: binding_array<Data, fragmentCount>;
        @group(0) @binding(2) var<uniform> sharedData: Data;
        @group(1) @binding(0) var meshImages: binding_array<texture_2d<f32>, 2>;
        @group(1) @binding(1) var fragmentImages: binding_array<texture_2d<f32>, fragmentCount>;
        @group(1) @binding(2) var sharedImage: texture_2d<f32>;
        @group(1) @binding(3) var state: sampler;
        @group(2) @binding(0) var meshTargets: binding_array<texture_storage_2d<rgba8unorm, write>, 2>;
        @group(2) @binding(1) var fragmentTargets: binding_array<texture_storage_2d<rgba8unorm, write>, fragmentCount>;
        @group(2) @binding(2) var sharedTarget: texture_storage_2d<rgba8unorm, write>;
        @mesh(out) @workgroup_size(1) fn mesh() {
          let data = sharedData;
          let color = data.value + meshData[1].value + textureSampleLevel(meshImages[1], state, vec2f(0.5), 0.0)
              + textureSampleLevel(sharedImage, state, vec2f(0.5), 0.0);
          textureStore(meshTargets[1], vec2i(0), color);
          textureStore(sharedTarget, vec2i(0), color);
          out.vertex_count = 3u; out.primitive_count = 1u;
          out.vertices[0].position = color;
          out.primitives[0].indices = vec3u(0u, 1u, 2u);
        }
        @fragment fn fragment(@builtin(position) p: vec4f) -> @location(0) vec4f {
          let data = sharedData;
          let color = data.value + fragmentData[u32(p.x)].value + textureSample(fragmentImages[u32(p.x)], state, p.xy)
              + textureSample(sharedImage, state, p.xy);
          textureStore(fragmentTargets[u32(p.x)], vec2i(0), color);
          textureStore(sharedTarget, vec2i(0), color);
          return color;
        }
        """)
    let compiler = IndependentShaderCompiler()
    let constants: [String: Double] = useOverride ? ["fragmentCount": 3] : [:]
    let mesh = try compiler.compileMeshStage(input, entryPoint: "mesh", to: .glsl, constants: constants)
    let fragment = try compiler.compileRenderStage(input, entryPoint: "fragment", to: .glsl, constants: constants)
    guard case .text(.glsl, let meshSource) = mesh,
          case .text(.glsl, let fragmentSource) = fragment.artifact else {
        Issue.record("Expected GLSL stages")
        return
    }
    assertLinkedMeshFragmentBindings(meshSource, fragmentSource, marker: "saga_buffer_", sharedCount: 1)
    assertLinkedMeshFragmentBindings(meshSource, fragmentSource, marker: "saga_pair_", sharedCount: 1)
    assertLinkedMeshFragmentBindings(meshSource, fragmentSource, marker: "saga_texture_", sharedCount: 1)
    #expect(meshFragmentBindings(fragmentSource, containing: "saga_buffer_").count == (useOverride ? 4 : 3))
    #expect(meshFragmentBindings(fragmentSource, containing: "saga_texture_").count == (useOverride ? 4 : 3))
}

@Test func glslFragmentRejectsAmbiguousVertexAndMeshPipelineContext() throws {
    let input = ShaderInput.wgsl("""
        enable wgpu_mesh_shader;
        struct V { @builtin(position) position: vec4f, }
        struct P { @builtin(triangle_indices) indices: vec3u, }
        struct O {
          @builtin(vertices) vertices: array<V, 3>,
          @builtin(primitives) primitives: array<P, 1>,
          @builtin(vertex_count) vertex_count: u32,
          @builtin(primitive_count) primitive_count: u32,
        }
        var<workgroup> out: O;
        @vertex fn vertex() -> @builtin(position) vec4f { return vec4f(0.0); }
        @mesh(out) @workgroup_size(1) fn mesh() {
          out.vertex_count = 3u; out.primitive_count = 1u;
          out.primitives[0].indices = vec3u(0u, 1u, 2u);
        }
        @fragment fn fragment() -> @location(0) vec4f { return vec4f(1.0); }
        """)
    #expect(throws: ShaderFailure.self) {
        try IndependentShaderCompiler().compileRenderStage(input, entryPoint: "fragment", to: .glsl)
    }
}
