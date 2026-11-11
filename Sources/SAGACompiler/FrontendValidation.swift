internal import SAGAFrontends
public import SAGAShaderIR
public import SAGAValidation

extension IndependentShaderCompiler {
  /// Produces a witness through the built-in semantic checker, never from parser success alone.
  /// Unsupported syntax/rules fail closed; this API does not claim full Naga conformance or emit code.
  public func parseAndValidate(
    _ input: ShaderInput, capabilities: ShaderCapabilities = .referenceDefault,
    limits: ShaderLimits = .default
  ) throws -> ValidatedModule<IRModule> {
    let validator = ShaderSemanticValidator(capabilities: capabilities, limits: limits)
    return try parseAndValidate(input, validator: validator, allow1DTextureSampleLevelOffset: false)
  }

  func parseAndValidate(
    _ input: ShaderInput, validator: ShaderSemanticValidator,
    allow1DTextureSampleLevelOffset: Bool, allowMixedWorkgroupDimensionTypes: Bool = false,
    allowLegacyNagaSubgroups: Bool = false, useNagaFloatRounding: Bool = false,
    useNagaIntegerConcretization: Bool = false
  ) throws -> ValidatedModule<IRModule> {
    try Task.checkCancellation()
    let module: IRModule
    switch input {
    case .wgsl:
      let options = WGSLFrontendOptions(capabilities: validator.capabilities,
        allow1DTextureSampleLevelOffset: allow1DTextureSampleLevelOffset,
        nagaFloatRounding: nagaFloatRounding || useNagaFloatRounding,
        nagaIntegerConcretization: useNagaIntegerConcretization,
        allowMixedWorkgroupDimensionTypes: allowMixedWorkgroupDimensionTypes,
        allowLegacyNagaSubgroups: allowLegacyNagaSubgroups)
      module = try WGSLFrontend(options: options, limits: validator.limits).parse(input)
    case .glsl(_, let stage, let defines):
      module = try GLSLFrontend(
        options: .init(stage: IRStage(stage), defines: defines), limits: validator.limits
      ).parse(input)
    case .spirv:
      module = try SPIRVFrontend(options: spirvOptions, limits: validator.limits).parse(input)
    }
    let checked = try ValidatedModule.checking(module, with: validator)
    try Task.checkCancellation()
    return checked
  }
}
