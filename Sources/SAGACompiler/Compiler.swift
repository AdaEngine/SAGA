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

public enum WGSLSourceProfile: Sendable, Equatable {
    /// Enables native WGSL syntax for MSL, HLSL, GLSL, and SPIR-V translation.
    case targetDefault
    /// Accepts Naga's mixed workgroup dimensions, implicit subgroups, and abstract numeric conversions.
    case nagaNative
}

public struct IndependentShaderCompiler: Sendable {
    private static let compilationCapabilities = ShaderCapabilities.referenceDefault.union([
        .storageTexture16BitNormFormats, .linearInterpolation, .shaderFloat16, .clipDistances, .primitiveIndex, .subgroup, .dualSourceBlending, .earlyDepthTest, .immediates,
    ]).union(.bindingArrays).union(.textureExternal).union([
        .textureAtomic, .textureInt64Atomic, .shaderInt64, .shaderInt64AtomicMinMax, .shaderFloat32Atomic, .subgroupBarrier, .shaderBarycentrics,
        .multiview, .shaderInt16, .perVertex, .drawIndex, .memoryDecorationCoherent, .memoryDecorationVolatile, .debugPrintf,
    ])

    let nagaFloatRounding: Bool
    private let wgslSourceProfile: WGSLSourceProfile

    public init() {
        self.init(wgslSourceProfile: .targetDefault)
    }

    /// Allows abstract values just outside a float's finite range when Naga rounds them to finite values.
    public init(nagaFloatRounding: Bool) {
        self.nagaFloatRounding = nagaFloatRounding
        self.wgslSourceProfile = .targetDefault
    }

    /// Selects a WGSL source dialect for compilation. WebGPU compilation and standalone validation use their default rules.
    public init(wgslSourceProfile: WGSLSourceProfile) {
        self.nagaFloatRounding = false
        self.wgslSourceProfile = wgslSourceProfile
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
        let (validated, validator) = try validatePipelineInput(input, output: output,
            bindingArrayLengths: bindingArrayLengths, webGPU: webGPU)
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

    /// Compiles a native task entry point after validation and override specialization.
    public func compileTaskStage(
        _ input: ShaderInput, entryPoint: String, to output: ShaderOutputFormat,
        constants: [String: Double] = [:], bindingArrayLengths: [IRResourceBinding: UInt32] = [:]
    ) throws -> ShaderArtifact {
        try compileTaskStage(input, entryPoint: entryPoint, to: output, constants: constants,
                             bindingArrayLengths: bindingArrayLengths, hlslTaskDispatchLimits: nil)
    }

    /// Optionally guards HLSL mesh dispatches using limits supplied by the target device.
    public func compileTaskStage(
        _ input: ShaderInput, entryPoint: String, to output: ShaderOutputFormat,
        constants: [String: Double] = [:], bindingArrayLengths: [IRResourceBinding: UInt32] = [:],
        hlslTaskDispatchLimits: HLSLTaskDispatchLimits?
    ) throws -> ShaderArtifact {
        try compileTaskStage(input, entryPoint: entryPoint, to: output, constants: constants,
                             bindingArrayLengths: bindingArrayLengths, hlslLimits: hlslTaskDispatchLimits, glslLimits: nil)
    }

    /// Optionally guards GLSL mesh dispatches using limits supplied by the target device.
    public func compileTaskStage(
        _ input: ShaderInput, entryPoint: String, to output: ShaderOutputFormat,
        constants: [String: Double] = [:], bindingArrayLengths: [IRResourceBinding: UInt32] = [:],
        glslTaskDispatchLimits: GLSLTaskDispatchLimits?
    ) throws -> ShaderArtifact {
        try compileTaskStage(input, entryPoint: entryPoint, to: output, constants: constants,
                             bindingArrayLengths: bindingArrayLengths, hlslLimits: nil, glslLimits: glslTaskDispatchLimits)
    }

    private func compileTaskStage(
        _ input: ShaderInput, entryPoint: String, to output: ShaderOutputFormat,
        constants: [String: Double], bindingArrayLengths: [IRResourceBinding: UInt32],
        hlslLimits: HLSLTaskDispatchLimits?, glslLimits: GLSLTaskDispatchLimits?
    ) throws -> ShaderArtifact {
        guard output == .msl || output == .hlsl || output == .glsl else { throw ShaderFailure.unsupported("Task shader target") }
        guard (hlslLimits == nil || output == .hlsl), (glslLimits == nil || output == .glsl) else {
            throw ShaderFailure.unsupported("Task dispatch limits do not match shader target")
        }
        let validator = ShaderSemanticValidator(capabilities: Self.compilationCapabilities.union(.meshShader),
                                                allowsVertexWritableStorage: true)
        let validated = try parseAndValidate(input, validator: validator, allow1DTextureSampleLevelOffset: output == .hlsl || output == .glsl,
                                             allowMixedWorkgroupDimensionTypes: true,
                                             allowLegacyNagaSubgroups: true, useNagaFloatRounding: wgslSourceProfile == .nagaNative,
                                             useNagaIntegerConcretization: wgslSourceProfile == .nagaNative)
        if !bindingArrayLengths.isEmpty { try validateBindingArrayLengths(bindingArrayLengths, in: validated.module) }
        guard let entry = validated.module.entryPoints.first(where: { $0.name == entryPoint && $0.stage == .task }) else {
            throw ShaderFailure.unsupported("Unknown task entry point")
        }
        let specialized = try specializeOverrides(validated, entry: entry, constants: constants,
                                                   retainLinkedStages: output == .glsl, validator: validator)
        let module = try specializeGlobalParameters(specialized, validator: validator)
        if output == .msl, let selected = module.module.entryPoints.first(where: { $0.name == entryPoint && $0.stage == .task }) {
            _ = try meshPayload(selected.taskPayload, in: module.module)
        }
        let artifact = try output == .glsl
            ? TaskGLSLCodeGenerator(bindingArrayLengths: bindingArrayLengths, dispatchLimits: glslLimits).generate(module, entryPoint: entryPoint)
            : TaskCodeGenerator(format: output, bindingArrayLengths: bindingArrayLengths, hlslTaskDispatchLimits: hlslLimits).generate(module, entryPoint: entryPoint)
        try artifact.validateContainer(for: output)
        return artifact
    }

    public func compileReflectedTaskStage(
        _ input: ShaderInput, entryPoint: String, to output: ShaderOutputFormat, constants: [String: Double] = [:],
        bindingArrayLengths: [IRResourceBinding: UInt32] = [:]
    ) throws -> TaskStageArtifact {
        guard output == .msl else { throw ShaderFailure.unsupported("Reflected task shader target") }
        let validator = ShaderSemanticValidator(capabilities: Self.compilationCapabilities.union(.meshShader),
                                                allowsVertexWritableStorage: true)
        let validated = try parseAndValidate(input, validator: validator, allow1DTextureSampleLevelOffset: output == .hlsl,
                                             allowMixedWorkgroupDimensionTypes: true,
                                             allowLegacyNagaSubgroups: true, useNagaFloatRounding: wgslSourceProfile == .nagaNative,
                                             useNagaIntegerConcretization: wgslSourceProfile == .nagaNative)
        if !bindingArrayLengths.isEmpty { try validateBindingArrayLengths(bindingArrayLengths, in: validated.module) }
        guard let entry = validated.module.entryPoints.first(where: { $0.name == entryPoint && $0.stage == .task }) else {
            throw ShaderFailure.unsupported("Unknown task entry point")
        }
        let specialized = try specializeOverrides(validated, entry: entry, constants: constants,
                                                   retainLinkedStages: false, validator: validator)
        let module = try specializeGlobalParameters(specialized, validator: validator)
        let generated = try TaskCodeGenerator(format: output, bindingArrayLengths: bindingArrayLengths).generateReflected(module, entryPoint: entryPoint)
        try generated.artifact.validateContainer(for: output)
        guard let selected = module.module.entryPoints.first(where: { $0.name == entryPoint && $0.stage == .task }) else { throw ShaderFailure.unsupported("Task metadata") }
        let numeric = try meshNumericCapabilities(module: module.module, entry: selected)
        return TaskStageArtifact(artifact: generated.artifact, sourceEntryPoint: entryPoint, workgroupSize: selected.workgroupSize, workgroupStorageByteSize: try meshWorkgroupStorageByteSize(module: module.module, entry: selected), payload: try meshPayload(selected.taskPayload, in: module.module), uniformBindings: generated.uniforms, storageBindings: generated.buffers, sampledTextures: generated.textures, samplerBindings: generated.samplers, storageTextures: generated.storageTextures, requiresSubgroups: generated.requiresSubgroups, requiresShaderF16: numeric.f16, requiresShaderInt16: numeric.int16, requiresShaderInt64: numeric.int64, requiresShaderInt64AtomicMinMax: numeric.atomicMinMax, requiresShaderInt64AtomicAllOps: numeric.atomicAllOps)
    }

    /// Compiles a native mesh entry point after validation and override specialization.
    public func compileMeshStage(
        _ input: ShaderInput, entryPoint: String, to output: ShaderOutputFormat,
        constants: [String: Double] = [:], bindingArrayLengths: [IRResourceBinding: UInt32] = [:]
    ) throws -> ShaderArtifact {
        guard output == .msl || output == .hlsl || output == .glsl else { throw ShaderFailure.unsupported("Mesh shader target") }
        let validator = ShaderSemanticValidator(capabilities: Self.compilationCapabilities.union([.meshShader, .meshShaderPointTopology]),
                                                allowsVertexWritableStorage: true)
        let validated = try parseAndValidate(input, validator: validator, allow1DTextureSampleLevelOffset: output == .hlsl || output == .glsl,
                                             allowMixedWorkgroupDimensionTypes: true,
                                             allowLegacyNagaSubgroups: true, useNagaFloatRounding: wgslSourceProfile == .nagaNative,
                                             useNagaIntegerConcretization: wgslSourceProfile == .nagaNative)
        if !bindingArrayLengths.isEmpty { try validateBindingArrayLengths(bindingArrayLengths, in: validated.module) }
        guard let entry = validated.module.entryPoints.first(where: { $0.name == entryPoint && $0.stage == .mesh }) else {
            throw ShaderFailure.unsupported("Unknown mesh entry point")
        }
        let specialized = try specializeOverrides(validated, entry: entry, constants: constants,
                                                   retainLinkedStages: output == .glsl, validator: validator)
        let module = try specializeGlobalParameters(specialized, validator: validator)
        if output == .msl, let selected = module.module.entryPoints.first(where: { $0.name == entryPoint && $0.stage == .mesh }) {
            _ = try meshPayload(selected.taskPayload, in: module.module)
        }
        let artifact: ShaderArtifact
        switch output {
        case .msl: artifact = try MeshMetalCodeGenerator().generate(module, entryPoint: entryPoint, bindingArrayLengths: bindingArrayLengths)
        case .hlsl: artifact = try MeshHLSLCodeGenerator(bindingArrayLengths: bindingArrayLengths).generate(module, entryPoint: entryPoint)
        case .glsl: artifact = try MeshGLSLCodeGenerator(bindingArrayLengths: bindingArrayLengths).generate(module, entryPoint: entryPoint)
        default: throw ShaderFailure.unsupported("Mesh shader target")
        }
        try artifact.validateContainer(for: output)
        return artifact
    }

    public func compileReflectedMeshStage(
        _ input: ShaderInput, entryPoint: String, to output: ShaderOutputFormat, constants: [String: Double] = [:],
        bindingArrayLengths: [IRResourceBinding: UInt32] = [:]
    ) throws -> MeshStageArtifact {
        guard output == .msl else { throw ShaderFailure.unsupported("Reflected mesh shader target") }
        let validator = ShaderSemanticValidator(capabilities: Self.compilationCapabilities.union([.meshShader, .meshShaderPointTopology]),
                                                allowsVertexWritableStorage: true)
        let validated = try parseAndValidate(input, validator: validator, allow1DTextureSampleLevelOffset: output == .hlsl,
                                             allowMixedWorkgroupDimensionTypes: true,
                                             allowLegacyNagaSubgroups: true, useNagaFloatRounding: wgslSourceProfile == .nagaNative,
                                             useNagaIntegerConcretization: wgslSourceProfile == .nagaNative)
        if !bindingArrayLengths.isEmpty { try validateBindingArrayLengths(bindingArrayLengths, in: validated.module) }
        guard let entry = validated.module.entryPoints.first(where: { $0.name == entryPoint && $0.stage == .mesh }) else {
            throw ShaderFailure.unsupported("Unknown mesh entry point")
        }
        let specialized = try specializeOverrides(validated, entry: entry, constants: constants,
                                                   retainLinkedStages: false, validator: validator)
        let module = try specializeGlobalParameters(specialized, validator: validator)
        let generated = try MeshMetalCodeGenerator().generateReflected(module, entryPoint: entryPoint, bindingArrayLengths: bindingArrayLengths)
        try generated.artifact.validateContainer(for: output)
        guard let selected = module.module.entryPoints.first(where: { $0.name == entryPoint && $0.stage == .mesh }), let info = selected.meshInfo else { throw ShaderFailure.unsupported("Mesh metadata") }
        let numeric = try meshNumericCapabilities(module: module.module, entry: selected)
        return MeshStageArtifact(artifact: generated.artifact, sourceEntryPoint: entryPoint, workgroupSize: selected.workgroupSize, workgroupStorageByteSize: try meshWorkgroupStorageByteSize(module: module.module, entry: selected), payload: try meshPayload(selected.taskPayload, in: module.module), topology: info.topology, maxVertices: info.maxVertices, maxPrimitives: info.maxPrimitives, vertexVaryings: try meshVaryings(info.vertexOutputType, in: module.module), primitiveVaryings: try meshVaryings(info.primitiveOutputType, in: module.module), uniformBindings: generated.uniforms, storageBindings: generated.buffers, sampledTextures: generated.textures, samplerBindings: generated.samplers, storageTextures: generated.storageTextures, requiresSubgroups: generated.requiresSubgroups, writesPrimitiveIndex: try meshWritesPrimitiveIndex(info.primitiveOutputType, in: module.module), requiresShaderF16: numeric.f16, requiresShaderInt16: numeric.int16, requiresShaderInt64: numeric.int64, requiresShaderInt64AtomicMinMax: numeric.atomicMinMax, requiresShaderInt64AtomicAllOps: numeric.atomicAllOps)
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

    /// Selects the pre-rasterization family whose resources are linked with a GLSL render stage.
    public func compileRenderStage(
        _ input: ShaderInput, entryPoint: String, to output: ShaderOutputFormat,
        glslPipelineFamily: GLSLRenderPipelineFamily, constants: [String: Double] = [:],
        bindingArrayLengths: [IRResourceBinding: UInt32] = [:]
    ) throws -> RenderStageArtifact {
        guard output == .glsl else {
            throw ShaderFailure.unsupported("GLSL pipeline family requires GLSL output")
        }
        return try compileRenderStage(input, entryPoint: entryPoint, to: output, constants: constants,
                                      bindingArrayLengths: bindingArrayLengths, webGPU: false,
                                      glslPipelineFamily: glslPipelineFamily)
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
        bindingArrayLengths: [IRResourceBinding: UInt32], webGPU: Bool, metalVertexBuffers: [VertexBufferLayout]? = nil,
        glslPipelineFamily: GLSLRenderPipelineFamily? = nil
    ) throws -> RenderStageArtifact {
        try Task.checkCancellation()
        let module = try pipelineModule(input, output: output, entryPoint: entryPoint, constants: constants,
                                        bindingArrayLengths: bindingArrayLengths, webGPU: webGPU,
                                        glslPipelineFamily: glslPipelineFamily)
        let compiled = try RenderCodeGenerator(output, bindingArrayLengths: bindingArrayLengths, webGPU: webGPU,
                                               metalVertexBuffers: metalVertexBuffers, glslPipelineFamily: glslPipelineFamily)
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
        bindingArrayLengths: [IRResourceBinding: UInt32], webGPU: Bool = false,
        glslPipelineFamily: GLSLRenderPipelineFamily? = nil
    ) throws -> ValidatedModule<IRModule> {
        let (module, validator) = try validatePipelineInput(input, output: output,
            bindingArrayLengths: bindingArrayLengths, webGPU: webGPU)
        let entries = module.module.entryPoints
        let entry = if let entryPoint {
            entries.first { $0.name.utf8.elementsEqual(entryPoint.utf8) }
        } else {
            entries.count == 1 ? entries.first : nil
        }
        if glslPipelineFamily == .mesh, entry?.stage == .vertex {
            throw ShaderFailure.unsupported("GLSL mesh pipeline family cannot compile a vertex stage")
        }
        if output == .glsl, glslPipelineFamily == nil, entry?.stage == .fragment,
           entries.contains(where: { $0.stage == .vertex }), entries.contains(where: { $0.stage == .mesh }) {
            throw ShaderFailure.unsupported("GLSL fragment stage requires an unambiguous linked pipeline family")
        }
        let specialized = try specializeOverrides(module, entry: entry, constants: constants,
                                                  retainLinkedStages: output == .glsl, validator: validator,
                                                  glslPipelineFamily: glslPipelineFamily)
        return try specializeGlobalParameters(specialized, validator: validator, lowerPointerSubviews: output == .spirv)
    }

    private func validatePipelineInput(
        _ input: ShaderInput, output: ShaderOutputFormat, bindingArrayLengths: [IRResourceBinding: UInt32],
        webGPU: Bool = false
    ) throws -> (ValidatedModule<IRModule>, ShaderSemanticValidator) {
        var capabilities = Self.compilationCapabilities
        if output != .msl { capabilities.insert(.shaderInt64AtomicAllOps) }
        if [.hlsl, .glsl, .spirv, .dot].contains(output) { capabilities.insert(.float64) }
        if !webGPU { capabilities.formUnion([.cooperativeMatrix, .meshShader]) }
        if !webGPU && [.msl, .hlsl, .glsl].contains(output) { capabilities.insert(.rayQuery) }
        if !webGPU && output == .glsl { capabilities.insert(.rayHitVertexPosition) }
        let usesNagaSource = !webGPU && wgslSourceProfile == .nagaNative
        let nativeTarget = switch output {
        case .msl, .hlsl, .glsl, .spirv: true
        case .wgsl, .dot: false
        }
        let validator = ShaderSemanticValidator(capabilities: capabilities, allowsVertexWritableStorage: true)
        let module = try parseAndValidate(input, validator: validator,
                                          allow1DTextureSampleLevelOffset: output == .hlsl || output == .glsl || output == .spirv || output == .dot,
                                          allowMixedWorkgroupDimensionTypes: nativeTarget || usesNagaSource,
                                          allowLegacyNagaSubgroups: nativeTarget || usesNagaSource,
                                          useNagaFloatRounding: usesNagaSource,
                                          useNagaIntegerConcretization: usesNagaSource)
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
