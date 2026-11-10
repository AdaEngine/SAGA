import Testing
import SAGACompiler
import SAGAShaderIR

@Test func hlslInverseHyperbolicsPreserveSmallInputs() throws {
    let source = """
        enable f16;
        @group(0) @binding(0) var<storage, read_write> data: array<vec4f>;
        @compute @workgroup_size(1) fn main() {
            let x = data[0];
            data[1] = asinh(x);
            data[2] = atanh(x);
            data[3] = vec4f(asinh(x.x), atanh(x.y), f32(asinh(f16(x.z))), f32(atanh(f16(x.w))));
        }
        """
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(source), to: .hlsl)
    guard case .text(_, let output) = shader.artifact else { Issue.record("Expected HLSL"); return }
    #expect(output.indices.filter { output[$0...].hasPrefix("1.0e-4f") }.count == 12)
    #expect(output.contains(".x) < 1.0e-4f ? "))
    #expect(output.contains(".w) < 1.0e-4f ? "))
    #expect(output.contains("sqrt(") && output.contains("log("))
}
