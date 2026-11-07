import SAGAShaderIR
import SAGAValidation

public protocol ShaderCodeGenerator: Sendable {
    associatedtype Module: ShaderIRModule
    var outputFormat: ShaderOutputFormat { get }
    func generate(_ module: ValidatedModule<Module>) throws -> ShaderArtifact
}
