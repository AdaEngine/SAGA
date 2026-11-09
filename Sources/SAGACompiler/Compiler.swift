public import SAGAShaderIR
public import SAGAValidation
public import SAGAFrontends
public import SAGABackends
public import SAGATypes

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
        .multiview, .shaderInt16, .perVertex, .drawIndex, .memoryDecorationCoherent, .memoryDecorationVolatile, .debugPrintf,
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
        if output == .dot {
            guard constants.isEmpty, bindingArrayLengths.isEmpty else {
                throw ShaderFailure.unsupported("DOT uses pre-specialization IR and does not accept pipeline options")
            }
            return try compileDOT(input)
        }
        if output == .spirv {
            let validated = try pipelineModule(input, output: output, constants: constants,
                                               bindingArrayLengths: bindingArrayLengths)
            return try SPIRVCodeGenerator(bindingArrayLengths: bindingArrayLengths).generate(validated)
        }
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

    /// Describes the entire validated module, including symbolic overrides and unused functions.
    public func compileDOT(_ input: ShaderInput, cfgOnly: Bool = false) throws -> ShaderArtifact {
        let (validated, _) = try validatePipelineInput(input, output: .dot, bindingArrayLengths: [:])
        guard validated.module.types.count > 0 || validated.module.functions.count > 0 else {
            throw ShaderFailure.unsupported("DOT export requires a module with declarations")
        }
        return try DOTCodeGenerator(cfgOnly: cfgOnly).generate(validated)
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
        let (validated, validator) = try validatePipelineInput(input, output: output, bindingArrayLengths: bindingArrayLengths)
        let entries = validated.module.entryPoints
        let entry = try selectComputeEntryPoint(entries, named: entryPoint)
        let specialized = try specializeOverrides(validated, entry: entry, constants: constants,
                                                   retainLinkedStages: false, validator: validator)
        let module = try specializeGlobalParameters(specialized, validator: validator)
        let compiled = try generator.generateCompute(module, onlyReachableBindings: entryPoint != nil || entries.count > 1)
        if webGPU, case .wgsl = input, compiled.storageBindings.contains(where: { !$0.memoryDecorations.isEmpty }) {
            throw ShaderFailure.unsupported("Browser WGSL does not support native storage memory attributes")
        }
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

    /// Compiles an MSL vertex entry that loads the supplied vertex streams directly.
    /// Pass the same layouts when creating a render pipeline with the returned artifact.
    public func compileMetalVertexStage(
        _ input: ShaderInput, entryPoint: String, vertexBuffers: [VertexBufferLayout],
        constants: [String: Double] = [:], bindingArrayLengths: [IRResourceBinding: UInt32] = [:]
    ) throws -> RenderStageArtifact {
        try compileRenderStage(input, entryPoint: entryPoint, to: .msl, constants: constants,
                               bindingArrayLengths: bindingArrayLengths, webGPU: false, metalVertexBuffers: vertexBuffers)
    }

    private func compileRenderStage(
        _ input: ShaderInput, entryPoint: String, to output: ShaderOutputFormat, constants: [String: Double],
        bindingArrayLengths: [IRResourceBinding: UInt32], webGPU: Bool, metalVertexBuffers: [VertexBufferLayout]? = nil
    ) throws -> RenderStageArtifact {
        try Task.checkCancellation()
        let module = try pipelineModule(input, output: output, entryPoint: entryPoint, constants: constants,
                                        bindingArrayLengths: bindingArrayLengths)
        let compiled = try RenderCodeGenerator(output, bindingArrayLengths: bindingArrayLengths, webGPU: webGPU,
                                               metalVertexBuffers: metalVertexBuffers)
            .generate(module, entryPoint: entryPoint)
        if webGPU, compiled.usesVertexWritableStorage {
            throw ShaderFailure.unsupported("Browser WebGPU does not support vertex writable storage")
        }
        if webGPU, case .wgsl = input, compiled.storageBindings.contains(where: { !$0.memoryDecorations.isEmpty }) {
            throw ShaderFailure.unsupported("Browser WGSL does not support native storage memory attributes")
        }
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
        let (module, validator) = try validatePipelineInput(input, output: output, bindingArrayLengths: bindingArrayLengths)
        let entries = module.module.entryPoints
        let entry = if let entryPoint {
            entries.first { $0.name.utf8.elementsEqual(entryPoint.utf8) }
        } else {
            entries.count == 1 ? entries.first : nil
        }
        let specialized = try specializeOverrides(module, entry: entry, constants: constants,
                                                  retainLinkedStages: output == .glsl, validator: validator)
        return try specializeGlobalParameters(specialized, validator: validator, lowerPointerSubviews: output == .spirv)
    }

    private func validatePipelineInput(
        _ input: ShaderInput, output: ShaderOutputFormat, bindingArrayLengths: [IRResourceBinding: UInt32]
    ) throws -> (ValidatedModule<IRModule>, ShaderSemanticValidator) {
        var capabilities = Self.compilationCapabilities
        if output != .msl { capabilities.insert(.shaderInt64AtomicAllOps) }
        let usesNativeWGSLProfile = switch output {
        case .msl, .hlsl, .glsl, .spirv: true
        case .wgsl, .dot: false
        }
        let validator = ShaderSemanticValidator(capabilities: capabilities, allowsVertexWritableStorage: true)
        let module = try parseAndValidate(input, validator: validator,
                                          allow1DTextureSampleLevelOffset: output == .hlsl || output == .glsl || output == .dot,
                                          allowMixedWorkgroupDimensionTypes: usesNativeWGSLProfile,
                                          allowLegacyNagaSubgroups: usesNativeWGSLProfile)
        if !bindingArrayLengths.isEmpty {
            try validateBindingArrayLengths(bindingArrayLengths, in: module.module)
        }
        return (module, validator)
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
