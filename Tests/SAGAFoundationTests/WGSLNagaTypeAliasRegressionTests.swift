import SAGACompiler
import SAGAShaderIR
import Testing

// Adapted from naga/tests/in/wgsl/type-alias.wgsl at the pinned revision.
@Test func nagaTypeAliasConstructors() throws {
    let source = ShaderInput.wgsl("""
        alias FVec3 = vec3<f32>;
        alias IVec3 = vec3i;
        alias Mat2 = mat2x2<f32>;
        alias I32 = i32;
        alias F32 = f32;
        @compute @workgroup_size(1) fn main() {
            let a = FVec3(0.0, 0.0, 0.0);
            let b = FVec3(vec2<f32>(0.0), 0.0);
            let e = IVec3(b);
            let f = Mat2(1.0, 2.0, 3.0, 4.0);
            let h = vec2<I32>();
            let i = mat2x2<F32>();
        }
        """)
    _ = try IndependentShaderCompiler().parseAndValidate(source)
}
