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
    try Task.checkCancellation()
    let module: IRModule
    switch input {
    case .wgsl:
      module = try WGSLFrontend(options: .init(capabilities: capabilities), limits: limits).parse(
        input)
    case .glsl(_, let stage, let defines):
      module = try GLSLFrontend(
        options: .init(stage: IRStage(stage), defines: defines), limits: limits
      ).parse(input)
    case .spirv:
      module = try SPIRVFrontend(limits: limits).parse(input)
    }
    let checked = try ValidatedModule.checking(
      module, with: ShaderSemanticValidator(capabilities: capabilities, limits: limits))
    try Task.checkCancellation()
    return checked
  }
}
