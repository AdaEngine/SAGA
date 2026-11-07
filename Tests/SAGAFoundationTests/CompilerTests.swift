import Testing
import SAGACompiler
import SAGAShaderIR
import SAGAValidation
import SAGAFrontends
import SAGABackends
import SAGAMetal
import SAGATypes

private struct TestIR: ShaderIRModule {
    let schemaVersion: UInt32 = 1
    let valid: Bool
}
private struct TestFrontend: ShaderFrontend {
    let inputFormat: ShaderInputFormat = .wgsl
    let valid: Bool
    func parse(_ input: ShaderInput) throws -> TestIR { TestIR(valid: valid) }
}
private struct TestValidator: IRValidator {
    func validate(_ module: TestIR) throws {
        if !module.valid { throw ShaderFailure.diagnostics([.init(severity: .error, message: "invalid test IR")]) }
    }
}
private struct TestGenerator: ShaderCodeGenerator {
    let outputFormat: ShaderOutputFormat = .msl
    func generate(_ module: ValidatedModule<TestIR>) throws -> ShaderArtifact {
        #expect(module.module.valid)
        return .text(format: .msl, source: "test artifact, not MSL")
    }
}
private struct WrongContainerGenerator: ShaderCodeGenerator {
    let outputFormat: ShaderOutputFormat = .msl
    func generate(_ module: ValidatedModule<TestIR>) throws -> ShaderArtifact { .spirv([]) }
}

@Test func pipelineRequiresValidatedIRBeforeCodegen() throws {
    let pipeline = CompilerPipeline(frontend: TestFrontend(valid: true), validator: TestValidator(), generator: TestGenerator())
    #expect(try pipeline.compile(.wgsl("test input")) == .text(format: .msl, source: "test artifact, not MSL"))
    #expect(throws: ShaderFailure.self) { try pipeline.compile(.spirv([])) }
    let invalid = CompilerPipeline(frontend: TestFrontend(valid: false), validator: TestValidator(), generator: TestGenerator())
    #expect(throws: ShaderFailure.self) { try invalid.compile(.wgsl("invalid")) }
}

@Test func pipelineRejectsWrongOutputContainer() throws {
    let pipeline = CompilerPipeline(frontend: TestFrontend(valid: true), validator: TestValidator(), generator: WrongContainerGenerator())
    #expect(throws: ShaderFailure.incompatibleArtifact) { try pipeline.compile(.wgsl("test")) }
}

@Test(arguments: ShaderInputFormat.allCases, ShaderOutputFormat.allCases)
func emptyOrUnsupportedTranslationNeverReturnsAnArtifact(input: ShaderInputFormat, output: ShaderOutputFormat) throws {
    let source: ShaderInput
    switch input {
    case .wgsl: source = .wgsl("")
    case .glsl: source = .glsl("", stage: .compute, defines: [:])
    case .spirv: source = .spirv([])
    }
    #expect(throws: ShaderFailure.self) { try IndependentShaderCompiler().compile(source, to: output) }
}

@Test func diagnosticSpansRejectInvalidRanges() throws {
    #expect(throws: ShaderFailure.invalidSpan) { try SourceSpan(startUTF8: -1, endUTF8: 1) }
    #expect(throws: ShaderFailure.invalidSpan) { try SourceSpan(startUTF8: 2, endUTF8: 1) }
    #expect(try SourceSpan(startUTF8: 0, endUTF8: 0).startUTF8 == 0)
}

@Test func containersDistinguishTextAndBinary() throws {
    try ShaderArtifact.spirv([0x07230203]).validateContainer(for: .spirv)
    #expect(throws: ShaderFailure.incompatibleArtifact) { try ShaderArtifact.text(format: .spirv, source: "bad").validateContainer(for: .spirv) }
}

@Test func metalFactoryRejectsMissingAdapterWithoutFallback() {
    #expect(throws: GPUError.adapterUnavailable) { try MetalBackend.makeDevice(adapter: nil) }
}

@Test func cancelledCompilerDoesNotEnterPipeline() async {
    let start = Gate()
    let task = Task {
        await start.wait()
        return try IndependentShaderCompiler().compile(.wgsl(""), to: .msl)
    }
    task.cancel(); await start.release()
    await #expect(throws: CancellationError.self) { try await task.value }
}
