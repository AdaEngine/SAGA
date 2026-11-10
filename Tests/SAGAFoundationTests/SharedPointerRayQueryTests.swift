import Testing
import SAGACompiler
import SAGAShaderIR

// The ray-query operations mirror the pinned Naga ray-query fixture; the
// storage pointer forces SAGA to specialize the helper's global parameters.
@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .glsl])
func sharedPointerSpecializationPreservesRayQueries(_ format: ShaderOutputFormat) throws {
    let source = """
        enable wgpu_ray_query;
        requires unrestricted_pointer_parameters;
        @group(0) @binding(0) var scene: acceleration_structure;
        @group(0) @binding(1) var<storage, read_write> output: array<u32, 2>;
        fn trace(destination: ptr<storage, u32, read_write>) {
            var query: ray_query;
            rayQueryInitialize(&query, scene, RayDesc(0u, 255u, 0.1, 10.0, vec3f(0.0), vec3f(1.0)));
            while rayQueryProceed(&query) { rayQueryConfirmIntersection(&query); }
            *destination = rayQueryGetCommittedIntersection(&query).kind;
            rayQueryTerminate(&query);
        }
        @compute @workgroup_size(1) fn main() { trace(&output[1]); }
        """
    let artifact = try IndependentShaderCompiler().compileCompute(.wgsl(source), to: format)
    #expect(artifact.accelerationStructures.count == 1)
    #expect(artifact.storageBindings.count == 1)
}
