import SAGAShaderIR

/// Implementors are trusted compiler components, not validators supplied by untrusted shader input.
public protocol IRValidator: Sendable {
    associatedtype Module: ShaderIRModule
    func validate(_ module: Module) throws
}

public struct ValidatedModule<Module: ShaderIRModule>: Sendable {
    public let module: Module
    private init(_ module: Module) { self.module = module }

    public static func checking<V: IRValidator>(_ module: Module, with validator: V) throws -> Self where V.Module == Module {
        try validator.validate(module)
        return Self(module)
    }
}
