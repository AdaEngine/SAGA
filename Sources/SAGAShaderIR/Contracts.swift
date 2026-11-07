public enum ShaderInputFormat: String, Sendable, CaseIterable { case wgsl, glsl, spirv }
public enum ShaderOutputFormat: String, Sendable, CaseIterable { case wgsl, glsl, spirv, msl, hlsl, dot }
public enum ShaderStage: String, Sendable { case vertex, fragment, compute }

public enum ShaderInput: Sendable, Equatable {
    case wgsl(String)
    case glsl(String, stage: ShaderStage, defines: [String: String])
    case spirv([UInt32])
    public var format: ShaderInputFormat {
        switch self { case .wgsl: .wgsl; case .glsl: .glsl; case .spirv: .spirv }
    }
}

public struct SourceSpan: Sendable, Equatable {
    public let startUTF8: Int
    public let endUTF8: Int
    public init(startUTF8: Int, endUTF8: Int) throws {
        guard startUTF8 >= 0, endUTF8 >= startUTF8 else { throw ShaderFailure.invalidSpan }
        self.startUTF8 = startUTF8; self.endUTF8 = endUTF8
    }
}

public struct ShaderDiagnostic: Sendable, Equatable {
    public enum Severity: Sendable { case warning, error }
    public let severity: Severity
    public let message: String
    public let code: ShaderDiagnosticCode
    public let location: ShaderLocation?
    public var span: SourceSpan? {
        if case .source(let span) = location { return span }
        return nil
    }
    public init(severity: Severity, message: String, span: SourceSpan? = nil) {
        self.init(severity: severity, code: .unspecified, message: message, location: span.map(ShaderLocation.source))
    }
    public init(severity: Severity, code: ShaderDiagnosticCode, message: String, location: ShaderLocation? = nil) {
        self.severity = severity; self.code = code; self.message = message; self.location = location
    }
}

public enum ShaderFailure: Error, Sendable, Equatable {
    case unsupported(String)
    case diagnostics([ShaderDiagnostic])
    case invalidHandle
    case invalidBudget
    case budgetExceeded(ShaderBudgetResource)
    case invalidSpan
    case incompatibleArtifact
    case incompatibleSchema(UInt32)
}

/// Raw module values must be immutable and value-semantic. Conformance is not semantic validation.
public protocol ShaderIRModule: Sendable { var schemaVersion: UInt32 { get } }

public protocol ShaderTransform: Sendable {
    associatedtype Module: ShaderIRModule
    func apply(to module: Module) throws -> Module
}

public protocol IRInterchange: Sendable {
    associatedtype Module: ShaderIRModule
    var formatIdentifier: String { get }
    func encode(_ module: Module) throws -> [UInt8]
    func decode(_ bytes: [UInt8]) throws -> Module
}

public enum ShaderArtifact: Sendable, Equatable {
    case text(format: ShaderOutputFormat, source: String)
    case spirv([UInt32])
    public func validateContainer(for format: ShaderOutputFormat) throws {
        switch self {
        case .text(let actual, _) where actual == format && actual != .spirv: return
        case .spirv where format == .spirv: return
        default: throw ShaderFailure.incompatibleArtifact
        }
    }
}
