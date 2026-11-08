import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let matrixCompoundShader = """
    @group(0) @binding(0) var<storage, read_write> out: array<u32, 14>;
    var<private> matrices: array<mat2x3f, 2>;
    var<private> index: u32;

    fn lhsIndex() -> u32 {
        out[0] += 1u;
        return index;
    }
    fn rhsMatrix() -> mat2x3f {
        out[1] = out[0];
        matrices[0] = mat2x3f(vec3f(99), vec3f(99));
        index = 1u;
        return mat2x3f(vec3f(1), vec3f(2));
    }
    @compute @workgroup_size(1) fn run() {
        matrices[0] = mat2x3f(vec3f(1, 2, 3), vec3f(4, 5, 6));
        matrices[1] = mat2x3f(vec3f(9), vec3f(8));
        matrices[lhsIndex()] += rhsMatrix();
        matrices[0] -= mat2x3(vec3(1.0), vec3(1.0));
        matrices[0] *= 2;
        matrices[0] *= mat2x2f(vec2f(2, 1), vec2f(1, 3));
        var row = vec3f(1, 2, 3);
        row *= mat3x3f(vec3f(1, 0, 0), vec3f(2, 1, 0), vec3f(0, 1, 2));
        out[2] = index;
        out[3] = bitcast<u32>(matrices[0][0].x);
        out[4] = bitcast<u32>(matrices[0][0].y);
        out[5] = bitcast<u32>(matrices[0][0].z);
        out[6] = bitcast<u32>(matrices[0][1].x);
        out[7] = bitcast<u32>(matrices[0][1].y);
        out[8] = bitcast<u32>(matrices[0][1].z);
        out[9] = bitcast<u32>(row.x);
        out[10] = bitcast<u32>(row.y);
        out[11] = bitcast<u32>(row.z);
        out[12] = bitcast<u32>(matrices[1][0].x);
        out[13] = bitcast<u32>(matrices[1][1].x);
    }
    """

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .glsl, .wgsl])
func emitsMatrixCompoundAssignments(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(matrixCompoundShader), to: format)
    guard case .text(_, let source) = shader.artifact else { Issue.record("Expected shader source"); return }
    #expect(shader.storageBindings.count == 1)
    #expect(!source.isEmpty)
    if format == .wgsl { _ = try compiler.parseAndValidate(.wgsl(source)) }
}

@Test(arguments: [2, 3, 4], [2, 3, 4])
func matrixCompoundOperatorsPreserveAllMatrixShapes(columns: Int, rows: Int) throws {
    for scalar in ["f32", "f16"] {
        let source = """
            \(scalar == "f16" ? "enable f16;" : "")
            fn update(a: mat\(columns)x\(rows)<\(scalar)>, b: mat\(columns)x\(columns)<\(scalar)>)
                -> mat\(columns)x\(rows)<\(scalar)> {
                var result = a;
                result += a;
                result -= a;
                result *= 2;
                result *= b;
                return result;
            }
            """
        _ = try IndependentShaderCompiler().parseAndValidate(.wgsl(source), capabilities: [.shaderFloat16])
    }
}

@Test(arguments: [
    "a *= mat3x2f();",
    "a *= vec2f();",
    "a += mat2x2f();",
    "a += 2f;",
    "a /= 2f;",
    "a *= 2i;",
    "b *= mat2x3f();",
])
func rejectsMatrixCompoundAssignmentsWithInvalidResultType(statement: String) {
    let source = "fn update() { var a: mat2x3f; var b: vec3f; \(statement) }"
    do {
        _ = try IndependentShaderCompiler().parseAndValidate(.wgsl(source))
        Issue.record("Accepted invalid matrix compound assignment")
    } catch ShaderFailure.diagnostics(let diagnostics) {
        #expect(diagnostics.first?.code == .typeMismatch)
    } catch {
        Issue.record("Expected a typed shader diagnostic")
    }
}

#if canImport(Metal)
@Test func matrixCompoundAssignmentReadsBeforeRHSAndStoresToOriginalIndexOnMetal() async throws {
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(matrixCompoundShader), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    let pipeline = try await device.createComputePipeline(shader)
    let output = try await device.createBuffer(.init(size: 56, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 56, usage: [.mapRead, .copyDestination]))
    let snapshots = try await device.submitCommands([
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
        .copy(source: output, destination: read, byteCount: 56),
        .read(buffer: read, byteCount: 56),
    ]).wait()
    let values: [Float] = [14, 20, 26, 32, 40, 48, 1, 4, 8, 9, 8]
    #expect(snapshots == [[1, 1, 1] + values.map(\.bitPattern)])
    await device.close()
}
#endif
