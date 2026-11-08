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
    private static let compilationCapabilities = ShaderCapabilities.referenceDefault.union([
        .storageTexture16BitNormFormats, .linearInterpolation, .shaderFloat16, .clipDistances, .primitiveIndex, .subgroup, .dualSourceBlending, .earlyDepthTest, .immediates,
    ]).union(.bindingArrays).union(.textureExternal).union([
        .textureAtomic, .textureInt64Atomic, .shaderInt64, .shaderInt64AtomicMinMax, .shaderFloat32Atomic, .subgroupBarrier, .shaderBarycentrics,
    ])

    let nagaFloatRounding: Bool

    public init() {
        self.init(nagaFloatRounding: false)
    }

    /// Allows abstract values just outside a float's finite range when Naga rounds them to finite values.
    public init(nagaFloatRounding: Bool) {
        self.nagaFloatRounding = nagaFloatRounding
    }

    public func compile(_ input: ShaderInput, to output: ShaderOutputFormat) throws -> ShaderArtifact {
        try compile(input, to: output, constants: [:])
    }

    /// Pipeline constant keys are decimal IDs when present and WGSL names otherwise.
    public func compile(
        _ input: ShaderInput, to output: ShaderOutputFormat, constants: [String: Double]
    ) throws -> ShaderArtifact {
        try compile(input, to: output, constants: constants, bindingArrayLengths: [:])
    }

    /// The layout supplies the element count for each binding array whose WGSL type omits a count.
    public func compile(
        _ input: ShaderInput, to output: ShaderOutputFormat, constants: [String: Double] = [:],
        bindingArrayLengths: [IRResourceBinding: UInt32]
    ) throws -> ShaderArtifact {
        let generator = try generator(for: input, output: output, bindingArrayLengths: bindingArrayLengths)
        let validated = try pipelineModule(input, output: output, constants: constants, bindingArrayLengths: bindingArrayLengths)
        let artifact: ShaderArtifact
        if validated.module.entryPoints.count == 1, let entry = validated.module.entryPoints.first,
           entry.stage == .vertex || entry.stage == .fragment {
            artifact = try RenderCodeGenerator(output, bindingArrayLengths: bindingArrayLengths)
                .generate(validated, entryPoint: entry.name).artifact
        } else {
            artifact = try generator.generate(validated)
        }
        try artifact.validateContainer(for: output)
        try Task.checkCancellation()
        return artifact
    }

    public func compileCompute(_ input: ShaderInput, to output: ShaderOutputFormat) throws -> ComputeShaderArtifact {
        try compileCompute(input, to: output, constants: [:])
    }

    public func compileCompute(
        _ input: ShaderInput, to output: ShaderOutputFormat, constants: [String: Double]
    ) throws -> ComputeShaderArtifact {
        try compileCompute(input, to: output, constants: constants, bindingArrayLengths: [:])
    }

    public func compileCompute(
        _ input: ShaderInput, to output: ShaderOutputFormat, constants: [String: Double] = [:],
        bindingArrayLengths: [IRResourceBinding: UInt32]
    ) throws -> ComputeShaderArtifact {
        try compileCompute(input, entryPoint: nil, to: output, constants: constants,
                           bindingArrayLengths: bindingArrayLengths, webGPU: false)
    }

    /// Compiles the named compute entry and reflects the bindings reachable from it.
    public func compileCompute(
        _ input: ShaderInput, entryPoint: String, to output: ShaderOutputFormat,
        constants: [String: Double] = [:], bindingArrayLengths: [IRResourceBinding: UInt32] = [:]
    ) throws -> ComputeShaderArtifact {
        try compileCompute(input, entryPoint: entryPoint, to: output, constants: constants,
                           bindingArrayLengths: bindingArrayLengths, webGPU: false)
    }

    private func compileCompute(
        _ input: ShaderInput, entryPoint: String?, to output: ShaderOutputFormat, constants: [String: Double],
        bindingArrayLengths: [IRResourceBinding: UInt32], webGPU: Bool
    ) throws -> ComputeShaderArtifact {
        let generator = try generator(for: input, output: output, bindingArrayLengths: bindingArrayLengths, webGPU: webGPU)
        let (validated, capabilities) = try validatePipelineInput(input, output: output, bindingArrayLengths: bindingArrayLengths)
        let entries = validated.module.entryPoints
        let entry = try selectComputeEntryPoint(entries, named: entryPoint)
        let specialized = try specializeOverrides(validated, entry: entry, constants: constants,
                                                   retainLinkedStages: false, capabilities: capabilities)
        let module = try specializeGlobalParameters(specialized, capabilities: capabilities)
        let compiled = try generator.generateCompute(module, onlyReachableBindings: entryPoint != nil || entries.count > 1)
        try compiled.artifact.validateContainer(for: output)
        try Task.checkCancellation()
        return compiled
    }

    public func compileRenderStage(
        _ input: ShaderInput, entryPoint: String, to output: ShaderOutputFormat
    ) throws -> RenderStageArtifact {
        try compileRenderStage(input, entryPoint: entryPoint, to: output, constants: [:])
    }

    public func compileRenderStage(
        _ input: ShaderInput, entryPoint: String, to output: ShaderOutputFormat, constants: [String: Double]
    ) throws -> RenderStageArtifact {
        try compileRenderStage(input, entryPoint: entryPoint, to: output, constants: constants, bindingArrayLengths: [:])
    }

    public func compileRenderStage(
        _ input: ShaderInput, entryPoint: String, to output: ShaderOutputFormat, constants: [String: Double] = [:],
        bindingArrayLengths: [IRResourceBinding: UInt32]
    ) throws -> RenderStageArtifact {
        try compileRenderStage(input, entryPoint: entryPoint, to: output, constants: constants,
                               bindingArrayLengths: bindingArrayLengths, webGPU: false)
    }

    private func compileRenderStage(
        _ input: ShaderInput, entryPoint: String, to output: ShaderOutputFormat, constants: [String: Double],
        bindingArrayLengths: [IRResourceBinding: UInt32], webGPU: Bool
    ) throws -> RenderStageArtifact {
        try Task.checkCancellation()
        let module = try pipelineModule(input, output: output, entryPoint: entryPoint, constants: constants,
                                        bindingArrayLengths: bindingArrayLengths)
        let compiled = try RenderCodeGenerator(output, bindingArrayLengths: bindingArrayLengths, webGPU: webGPU)
            .generate(module, entryPoint: entryPoint)
        try compiled.artifact.validateContainer(for: output)
        try Task.checkCancellation()
        return compiled
    }

    public func compileWebGPUCompute(
        _ input: ShaderInput, constants: [String: Double] = [:], bindingArrayLengths: [IRResourceBinding: UInt32] = [:]
    ) throws -> ComputeShaderArtifact {
        try compileCompute(input, entryPoint: nil, to: .wgsl, constants: constants,
                           bindingArrayLengths: bindingArrayLengths, webGPU: true)
    }

    public func compileWebGPUCompute(
        _ input: ShaderInput, entryPoint: String, constants: [String: Double] = [:],
        bindingArrayLengths: [IRResourceBinding: UInt32] = [:]
    ) throws -> ComputeShaderArtifact {
        try compileCompute(input, entryPoint: entryPoint, to: .wgsl, constants: constants,
                           bindingArrayLengths: bindingArrayLengths, webGPU: true)
    }

    public func compileWebGPURenderStage(
        _ input: ShaderInput, entryPoint: String, constants: [String: Double] = [:],
        bindingArrayLengths: [IRResourceBinding: UInt32] = [:]
    ) throws -> RenderStageArtifact {
        try compileRenderStage(input, entryPoint: entryPoint, to: .wgsl, constants: constants,
                               bindingArrayLengths: bindingArrayLengths, webGPU: true)
    }

    private func pipelineModule(
        _ input: ShaderInput, output: ShaderOutputFormat, entryPoint: String? = nil, constants: [String: Double],
        bindingArrayLengths: [IRResourceBinding: UInt32]
    ) throws -> ValidatedModule<IRModule> {
        let (module, capabilities) = try validatePipelineInput(input, output: output, bindingArrayLengths: bindingArrayLengths)
        let entries = module.module.entryPoints
        let entry = if let entryPoint {
            entries.first { $0.name.utf8.elementsEqual(entryPoint.utf8) }
        } else {
            entries.count == 1 ? entries.first : nil
        }
        let specialized = try specializeOverrides(module, entry: entry, constants: constants,
                                                  retainLinkedStages: output == .glsl, capabilities: capabilities)
        return try specializeGlobalParameters(specialized, capabilities: capabilities)
    }

    private func validatePipelineInput(
        _ input: ShaderInput, output: ShaderOutputFormat, bindingArrayLengths: [IRResourceBinding: UInt32]
    ) throws -> (ValidatedModule<IRModule>, ShaderCapabilities) {
        var capabilities = Self.compilationCapabilities
        if output != .msl { capabilities.insert(.shaderInt64AtomicAllOps) }
        let module = try parseAndValidate(input, capabilities: capabilities,
                                          allow1DTextureSampleLevelOffset: output == .hlsl || output == .glsl)
        if !bindingArrayLengths.isEmpty {
            try validateBindingArrayLengths(bindingArrayLengths, in: module.module)
        }
        return (module, capabilities)
    }

    private func selectComputeEntryPoint(_ entries: [IREntryPoint], named name: String?) throws -> IREntryPoint {
        let compute = entries.filter { $0.stage == .compute }
        if let name {
            if let entry = compute.first(where: { $0.name.utf8.elementsEqual(name.utf8) }) { return entry }
            if let other = entries.first(where: { $0.name.utf8.elementsEqual(name.utf8) }) {
                throw ShaderFailure.diagnostics([.init(severity: .error, code: .invalidBinding,
                    message: "Entry point '\(name)' is not a compute shader", location: other.location)])
            }
            throw ShaderFailure.diagnostics([.init(severity: .error, code: .unresolvedName,
                message: "Unknown compute entry point '\(name)'")])
        }
        guard compute.count == 1, let entry = compute.first else {
            let message = compute.isEmpty ? "Shader has no compute entry point" : "Select a compute entry point by name"
            throw ShaderFailure.diagnostics([.init(severity: .error, code: .invalidBinding, message: message)])
        }
        return entry
    }

    private func validateBindingArrayLengths(_ lengths: [IRResourceBinding: UInt32], in module: IRModule) throws {
        var unbounded = Set<IRResourceBinding>()
        for global in module.globalVariables.values {
            if let binding = global.binding,
               case .bindingArray(_, .dynamic) = try module.types.value(for: global.type).kind {
                unbounded.insert(binding)
            }
        }
        for (binding, count) in lengths {
            guard count > 0, unbounded.contains(binding) else {
                throw ShaderFailure.diagnostics([.init(severity: .error, code: .invalidBinding,
                    message: "Expected a positive layout count for an unbounded binding array at group \(binding.group), binding \(binding.binding)")])
            }
        }
    }

    private func generator(
        for input: ShaderInput, output: ShaderOutputFormat, bindingArrayLengths: [IRResourceBinding: UInt32],
        webGPU: Bool = false
    ) throws -> ComputeCodeGenerator {
        try Task.checkCancellation()
        guard let generator = ComputeCodeGenerator(output, bindingArrayLengths: bindingArrayLengths, webGPU: webGPU) else {
            throw ShaderFailure.unsupported("Translation from \(input.format.rawValue) to \(output.rawValue)")
        }
        return generator
    }
}
