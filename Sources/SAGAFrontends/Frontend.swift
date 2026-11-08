import SAGAShaderIR

public protocol ShaderFrontend: Sendable {
    associatedtype Module: ShaderIRModule
    var inputFormat: ShaderInputFormat { get }
    func parse(_ input: ShaderInput) throws -> Module
}

/// New consumers implement the budget-bearing requirement; legacy test frontends remain source compatible.
/// No default discards the budget. Implementations charge tokens, nodes, expansion work and nesting.
public protocol BoundedShaderFrontend: ShaderFrontend {
    var limits: ShaderLimits { get }
    func parse(_ input: ShaderInput, budget: inout ShaderBudget) throws -> Module
}

extension BoundedShaderFrontend {
    public func parse(_ input: ShaderInput) throws -> Module {
        var budget = ShaderBudget(limits: limits)
        try budget.checkInput(input)
        guard input.format == inputFormat else { throw ShaderFailure.unsupported("Frontend input format mismatch") }
        let module = try parse(input, budget: &budget)
        try budget.checkCancellation()
        return module
    }
}

public struct WGSLFrontendOptions: Sendable, Equatable {
    public let parseDocComments: Bool
    public let capabilities: ShaderCapabilities
    package let allow1DTextureSampleLevelOffset: Bool
    package let nagaFloatRounding: Bool
    public init(parseDocComments: Bool = false, capabilities: ShaderCapabilities = .allKnown) {
        self.init(parseDocComments: parseDocComments, capabilities: capabilities, allow1DTextureSampleLevelOffset: false)
    }

    package init(
        parseDocComments: Bool = false, capabilities: ShaderCapabilities, allow1DTextureSampleLevelOffset: Bool,
        nagaFloatRounding: Bool = false
    ) {
        self.parseDocComments = parseDocComments
        self.capabilities = capabilities
        self.allow1DTextureSampleLevelOffset = allow1DTextureSampleLevelOffset
        self.nagaFloatRounding = nagaFloatRounding
    }
}

public struct GLSLFrontendOptions: Sendable, Equatable {
    public let stage: IRStage
    public let defines: [String: String]
    public init(stage: IRStage, defines: [String: String] = [:]) { self.stage = stage; self.defines = defines }
}

public struct SPIRVFrontendOptions: Sendable, Equatable {
    public let adjustCoordinateSpace: Bool
    public let strictCapabilities: Bool
    public init(adjustCoordinateSpace: Bool = true, strictCapabilities: Bool = true) {
        self.adjustCoordinateSpace = adjustCoordinateSpace; self.strictCapabilities = strictCapabilities
    }
}
