import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing

#if canImport(Metal)
import SAGAMetal
#endif

private let multipartForFragment = ShaderInput.glsl("""
    #version 460
    layout(location=0) out vec4 color;
    void main() {
        float a = 1.0;
        float b = 0.25;
        float c = 1.5;
        int i = 20;
        for (i = 0, c -= 1.0; i < 25; i++, b += 0.01) {
            a -= 0.02;
        }
        color = vec4(a, b, c, 1.0);
    }
    """, stage: .fragment, defines: [:])

private let multipartForCompute = ShaderInput.glsl("""
    #version 460
    layout(local_size_x=1) in;
    layout(set=0,binding=0,std430) buffer Result { int values[5]; } result;
    void stamp(int first, int second) {
        result.values[0] = result.values[0] * 10 + first + second;
    }
    void main() {
        int i = 99;
        int state = 0;
        int visits = 0;
        int increments = 0;
        result.values[0] = 0;
        for (i = 0, state = i + 1, stamp(1, 0); i < 5;
             i++, ++increments, state = state * 2 + i + increments, stamp(i, increments)) {
            if (i == 1) { continue; }
            if (i == 3) { break; }
            visits = visits * 10 + i + 1;
        }
        result.values[1] = i;
        result.values[2] = state;
        result.values[3] = visits;
        result.values[4] = increments;
    }
    """, stage: .compute, defines: [:])

@Test func translatesGLSLMultipartForFragment() throws {
    let compiler = IndependentShaderCompiler()
    let fragment = try compiler.compileRenderStage(multipartForFragment, entryPoint: "main", to: .wgsl)
    guard case .text(_, let wgsl) = fragment.artifact else {
        Issue.record("Expected WGSL fragment source")
        return
    }
    #expect(fragment.colorOutputComponentCounts == [0: 4])
    _ = try compiler.parseAndValidate(.wgsl(wgsl))
}

@Test func translatesGLSLMultipartForContinuingToWebGPU() throws {
    let compiler = IndependentShaderCompiler()
    let module = try compiler.parseAndValidate(multipartForCompute).module
    let main = try #require(module.functions.values.first { $0.name == "main" })
    let continuing = try #require(main.statements.values.compactMap { statement -> IRHandle<IRBlock>? in
        if case .loop(_, let block, _) = statement.kind { return block }
        return nil
    }.first)
    let statements = try main.blocks.value(for: continuing).statements
    let effects = try statements.map { try main.statements.value(for: $0).kind }.filter {
        switch $0 {
        case .store, .call: true
        default: false
        }
    }
    #expect(effects.count == 4)
    guard case .call(_, _, nil)? = effects.last else {
        Issue.record("Expected the final clause call in the loop continuation")
        return
    }
    let shader = try compiler.compileWebGPUCompute(multipartForCompute, entryPoint: "main")
    guard case .text(_, let wgsl) = shader.artifact else {
        Issue.record("Expected WGSL compute source")
        return
    }
    _ = try compiler.parseAndValidate(.wgsl(wgsl))
}

#if canImport(Metal)
@Test func runsGLSLMultipartForClausesInOrderOnMetal() async throws {
    let shader = try IndependentShaderCompiler().compileCompute(multipartForCompute, entryPoint: "main", to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let output = try await device.createBuffer(.init(size: 20, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 20, usage: [.copyDestination, .mapRead]))
        let ticket = try await device.submitCommands([
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
            .copy(source: output, destination: readback, byteCount: 20),
            .read(buffer: readback, byteCount: 20),
        ])
        #expect(try await ticket.wait() == [[1246, 3, 30, 13, 3]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif

@Test(arguments: [
    "int i; for (i = 0,; i < 1; i++) {}",
    "int i; for (i = 0; i < 1; i++,) {}",
    "const int i = 1; for (; false; ++i) {}",
])
func rejectsMalformedGLSLMultipartForClauses(body: String) {
    let source = ShaderInput.glsl("#version 460\nvoid main() { \(body) }", stage: .fragment, defines: [:])
    #expect(throws: ShaderFailure.self) {
        try IndependentShaderCompiler().compileRenderStage(source, entryPoint: "main", to: .wgsl)
    }
}
