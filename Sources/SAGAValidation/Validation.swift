import SAGAShaderIR

/// Implementors are trusted compiler components, not validators supplied by untrusted shader input.
public protocol IRValidator: Sendable {
    associatedtype Module: ShaderIRModule
    func validate(_ module: Module) throws
    func validateReportingDiagnostics(_ module: Module) throws -> [ShaderDiagnostic]
}

public extension IRValidator {
    func validateReportingDiagnostics(_ module: Module) throws -> [ShaderDiagnostic] {
        try validate(module)
        return []
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
        Self(module, diagnostics: try validator.validateReportingDiagnostics(module))
    }
}
