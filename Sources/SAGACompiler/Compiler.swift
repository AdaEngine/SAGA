public import SAGAShaderIR
public import SAGAValidation
public import SAGAFrontends
public import SAGABackends

public struct CompilerPipeline<Frontend: ShaderFrontend, Validator: IRValidator, Generator: ShaderCodeGenerator>: Sendable
where Frontend.Module == Validator.Module, Validator.Module == Generator.Module {
    private let frontend: Frontend
    private let validator: Validator
    private let generator: Generator
    public init(frontend: Frontend, validator: Validator, generator: Generator) {
        self.frontend = frontend; self.validator = validator; self.generator = generator
    }
    public func compile(_ input: ShaderInput) throws -> ShaderArtifact {
        try Task.checkCancellation()
        guard input.format == frontend.inputFormat else { throw ShaderFailure.unsupported("Frontend input format mismatch") }
        let module = try frontend.parse(input)
        try Task.checkCancellation()
        let validated = try ValidatedModule.checking(module, with: validator)
        try Task.checkCancellation()
        let artifact = try generator.generate(validated)
        try artifact.validateContainer(for: generator.outputFormat)
        try Task.checkCancellation()
        return artifact
    }
}

public struct IndependentShaderCompiler: Sendable {
    public init() {}
    public func compile(_ input: ShaderInput, to output: ShaderOutputFormat) throws -> ShaderArtifact {
        let generator = try generator(for: input, output: output)
        let validated = try parseAndValidate(input)
        let artifact: ShaderArtifact
        if validated.module.entryPoints.count == 1, let entry = validated.module.entryPoints.first,
           entry.stage == .vertex || entry.stage == .fragment {
            artifact = try RenderCodeGenerator(output).generate(validated, entryPoint: entry.name).artifact
        } else {
            artifact = try generator.generate(validated)
        }
        try artifact.validateContainer(for: output)
        try Task.checkCancellation()
        return artifact
    }

    public func compileCompute(_ input: ShaderInput, to output: ShaderOutputFormat) throws -> ComputeShaderArtifact {
        let generator = try generator(for: input, output: output)
        let module = try parseAndValidate(input)
        let compiled = try generator.generateCompute(module)
        try compiled.artifact.validateContainer(for: output)
        try Task.checkCancellation()
        return compiled
    }

    public func compileRenderStage(
        _ input: ShaderInput, entryPoint: String, to output: ShaderOutputFormat
    ) throws -> RenderStageArtifact {
        try Task.checkCancellation()
        guard input.format == .wgsl else {
            throw ShaderFailure.unsupported("Translation from \(input.format.rawValue) to \(output.rawValue)")
        }
        let module = try parseAndValidate(input)
        let compiled = try RenderCodeGenerator(output).generate(module, entryPoint: entryPoint)
        try compiled.artifact.validateContainer(for: output)
        try Task.checkCancellation()
        return compiled
    }

    private func generator(for input: ShaderInput, output: ShaderOutputFormat) throws -> ComputeCodeGenerator {
        try Task.checkCancellation()
        guard input.format == .wgsl, let generator = ComputeCodeGenerator(output) else {
            throw ShaderFailure.unsupported("Translation from \(input.format.rawValue) to \(output.rawValue)")
        }
        return generator
    }
}
