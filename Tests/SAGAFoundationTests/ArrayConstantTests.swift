import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let arrayConstantShader = """
    alias Row = array<u32, 2>;
    struct Record { row: Row, tag: u32, }
    const empty = Record();
    const table = array(Record(Row(2u, 3u), 8u), Record(array(4u, 5), 10u));
    const_assert table[1].row[0] == 4u && empty.row[1] == 0u;
    override lane = 0u;
    override width = table[lane].row[1];
    @group(0) @binding(0) var<storage, read_write> output: array<u32, 2>;
    @compute @workgroup_size(width)
    fn run(@builtin(local_invocation_index) id: u32) {
        if id >= 2u { return; }
        const local: Row = array(6, 7);
        const_assert local[1] == 7u;
        let values = array(table[lane].tag, local[1] + id);
        output[id] = values[id];
    }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func compilesNestedArrayConstants(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl(arrayConstantShader)
    let shader = try compiler.compileCompute(source, to: format, constants: ["lane": 1])
    #expect(shader.workgroupSize == .init(x: 5))
    guard case .text(_, let generated) = shader.artifact else { Issue.record("Expected source"); return }
    if format == .wgsl { _ = try compiler.parseAndValidate(.wgsl(generated)) }
}

@Test func arrayConstantsPreserveAbstractElementsUntilContextualConversion() throws {
    let source = ShaderInput.wgsl("""
        const big = array(4294967295, 1);
        const unsigned: array<u32, 2> = big;
        const_assert unsigned[0] == 4294967295u;
        const nested: array<array<f32, 2>, 2> = array(array(1, 2), array(3.5, 4));
        const_assert nested[1][0] == 3.5f;
        const count = array(1, 2)[1];
        alias Slots = array<u32, count>;
        @compute @workgroup_size(count) fn run() {
            let x: array<u32, 2> = array(4294967295, 1);
            _ = x[0];
        }
        """)
    let shader = try IndependentShaderCompiler().compileCompute(source, to: .wgsl)
    #expect(shader.workgroupSize == .init(x: 2))
}

@Test func constNameCanMatchStructureField() throws {
    _ = try IndependentShaderCompiler().parseAndValidate(.wgsl("""
        struct Record { count: u32, }
        const count = Record(2u).count;
        const_assert count == 2u;
        """))
}

@Test(arguments: [
    "const bad = array();",
    "const bad = array(1i, 2u);",
    "const bad: array<u32, 2> = array(-1, 2);",
    "const bad = array<u32, 2>(1u);",
    "const bad = array(1, 2)[-1];",
    """
    const divisors = array(1u, 0u);
    @compute @workgroup_size(1) fn run(@builtin(local_invocation_index) id: u32) {
        _ = id / divisors[1];
    }
    """,
    "struct R { value: u32, } const bad = R(1u).unknown;",
])
func rejectsInvalidArrayAndStructureConstants(source: String) {
    #expect(throws: ShaderFailure.self) {
        try IndependentShaderCompiler().parseAndValidate(.wgsl(source))
    }
}

#if canImport(Metal)
@Test func readsArrayConstantsOnMetal() async throws {
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(arrayConstantShader), to: .msl,
                                                                 constants: ["lane": 1])
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let output = try await device.createBuffer(.init(size: 8, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: 8, usage: [.copyDestination, .mapRead]))
        let ticket = try await device.submitCommands([
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
            .copy(source: output, destination: read, byteCount: 8), .read(buffer: read, byteCount: 8),
        ])
        #expect(try await ticket.wait() == [[10, 8]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
