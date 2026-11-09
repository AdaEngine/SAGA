import SAGAShaderIR

/// Implementors are trusted compiler components, not validators supplied by untrusted shader input.
public protocol IRValidator: Sendable {
    associatedtype Module: ShaderIRModule
    func validate(_ module: Module) throws
    func validateReportingDiagnostics(_ module: Module) throws -> [ShaderDiagnostic]
    /// Returns the validated form used by code generation.
    func validateAndNormalize(_ module: Module) throws -> (module: Module, diagnostics: [ShaderDiagnostic])
}

public extension IRValidator {
    func validateReportingDiagnostics(_ module: Module) throws -> [ShaderDiagnostic] {
        try validate(module)
        return []
    }

    func validateAndNormalize(_ module: Module) throws -> (module: Module, diagnostics: [ShaderDiagnostic]) {
        (module, try validateReportingDiagnostics(module))
    }
}

public struct ValidatedModule<Module: ShaderIRModule>: Sendable {
    public let module: Module
    public let diagnostics: [ShaderDiagnostic]
    private init(_ module: Module, diagnostics: [ShaderDiagnostic]) {
        self.module = module
        self.diagnostics = diagnostics
    }

    public static func checking<V: IRValidator>(_ module: Module, with validator: V) throws -> Self where V.Module == Module {
        let checked = try validator.validateAndNormalize(module)
        return Self(checked.module, diagnostics: checked.diagnostics)
    }
}
