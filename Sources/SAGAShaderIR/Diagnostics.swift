/// Byte offsets into a binary container. Deliberately distinct from UTF-8 source spans.
public struct BinaryByteSpan: Sendable, Equatable, Hashable {
    public let startByte: Int
    public let endByte: Int
    public init(startByte: Int, endByte: Int) throws {
        guard startByte >= 0, endByte >= startByte else { throw ShaderFailure.invalidSpan }
        self.startByte = startByte; self.endByte = endByte
    }
}

/// Half-open word indices into the original SPIR-V container, including its header.
public struct BinaryWordSpan: Sendable, Equatable, Hashable {
    public let startWord: Int
    public let endWord: Int
    public init(startWord: Int, endWord: Int) throws {
        guard startWord >= 0, endWord >= startWord else { throw ShaderFailure.invalidSpan }
        self.startWord = startWord; self.endWord = endWord
    }
    public func validate(wordCount: Int) throws {
        guard wordCount >= 0, endWord <= wordCount else { throw ShaderFailure.invalidSpan }
    }
    /// Byte offsets, NOT UTF-8 text coordinates; rejects multiplication overflow.
    public func byteSpan() throws -> BinaryByteSpan {
        guard endWord <= Int.max / 4 else { throw ShaderFailure.invalidSpan }
        return try BinaryByteSpan(startByte: startWord * 4, endByte: endWord * 4)
    }
}

public enum ShaderLocation: Sendable, Equatable {
    case source(SourceSpan)
    case binary(BinaryWordSpan)
}

public enum ShaderDiagnosticCategory: String, Sendable, CaseIterable {
    case unspecified, lexical, syntax, nameResolution, type, semantic, capability, binary, resourceLimit
}

/// Stable machine-readable identifiers, independent of human-readable messages.
public enum ShaderDiagnosticCode: String, Sendable, CaseIterable {
    case unspecified = "SAGA0000"
    case invalidToken = "SAGA1001"
    case unexpectedToken = "SAGA2001"
    case unresolvedName = "SAGA3001"
    case duplicateName = "SAGA3002"
    case typeMismatch = "SAGA4001"
    case invalidType = "SAGA4002"
    case invalidControlFlow = "SAGA5001"
    case invalidAddressSpace = "SAGA5002"
    case nonUniformControlFlow = "SAGA5003"
    case invalidBinding = "SAGA5004"
    case invalidReference = "SAGA5005"
    case unknownDiagnosticRule = "SAGA5006"
    case unusedResult = "SAGA5007"
    case unsupportedCapability = "SAGA6001"
    case unsupportedConstruct = "SAGA6002"
    case invalidBinary = "SAGA7001"
    case budgetExceeded = "SAGA8001"
    public var category: ShaderDiagnosticCategory {
        switch self {
        case .unspecified: .unspecified
        case .invalidToken: .lexical
        case .unexpectedToken: .syntax
        case .unresolvedName, .duplicateName: .nameResolution
        case .typeMismatch, .invalidType: .type
        case .invalidControlFlow, .invalidAddressSpace, .nonUniformControlFlow, .invalidBinding, .invalidReference,
             .unknownDiagnosticRule, .unusedResult: .semantic
        case .unsupportedCapability, .unsupportedConstruct: .capability
        case .invalidBinary: .binary
        case .budgetExceeded: .resourceLimit
        }
    }
}

extension SourceSpan {
    /// Check scalar boundaries (not grapheme boundaries) and source bounds without offset arithmetic traps.
    public func validate(in source: String) throws {
        let bytes = source.utf8
        guard let start = bytes.index(bytes.startIndex, offsetBy: startUTF8, limitedBy: bytes.endIndex),
              let end = bytes.index(bytes.startIndex, offsetBy: endUTF8, limitedBy: bytes.endIndex),
              start == bytes.endIndex || bytes[start] & 0xc0 != 0x80,
              end == bytes.endIndex || bytes[end] & 0xc0 != 0x80 else { throw ShaderFailure.invalidSpan }
    }
}
