public enum RenderColorLoadAction: Sendable, Equatable {
    case load
    case clear(RenderClearColor)

    package var clearColor: RenderClearColor? {
        if case .clear(let color) = self { return color }
        return nil
    }
}

public struct RenderColorWriteMask: OptionSet, Sendable {
    public let rawValue: UInt32
    public init(rawValue: UInt32) { self.rawValue = rawValue }

    public static let red = Self(rawValue: 1 << 0)
    public static let green = Self(rawValue: 1 << 1)
    public static let blue = Self(rawValue: 1 << 2)
    public static let alpha = Self(rawValue: 1 << 3)
    public static let all: Self = [.red, .green, .blue, .alpha]
}

public enum RenderBlendOperation: UInt32, Sendable, Equatable {
    case add
    case subtract
    case reverseSubtract
    case min
    case max
}

public enum RenderBlendFactor: UInt32, Sendable, Equatable {
    case zero
    case one
    case sourceColor
    case oneMinusSourceColor
    case destinationColor
    case oneMinusDestinationColor
    case sourceAlpha
    case oneMinusSourceAlpha
    case destinationAlpha
    case oneMinusDestinationAlpha
    case sourceAlphaSaturated
    case constant
    case oneMinusConstant
    case source1Color
    case oneMinusSource1Color
    case source1Alpha
    case oneMinusSource1Alpha

    package var usesSourceAlpha: Bool {
        switch self {
        case .sourceAlpha, .oneMinusSourceAlpha, .sourceAlphaSaturated, .source1Alpha, .oneMinusSource1Alpha: true
        default: false
        }
    }

    package var usesSecondSource: Bool {
        switch self {
        case .source1Color, .oneMinusSource1Color, .source1Alpha, .oneMinusSource1Alpha: true
        default: false
        }
    }
}

public struct RenderBlendComponent: Sendable, Equatable {
    public let operation: RenderBlendOperation
    public let sourceFactor: RenderBlendFactor
    public let destinationFactor: RenderBlendFactor

    public init(operation: RenderBlendOperation = .add, sourceFactor: RenderBlendFactor = .one,
                destinationFactor: RenderBlendFactor = .zero) {
        self.operation = operation
        self.sourceFactor = sourceFactor
        self.destinationFactor = destinationFactor
    }

    package var hasValidFactors: Bool {
        switch operation {
        case .min, .max: return sourceFactor == .one && destinationFactor == .one
        default: return true
        }
    }
}

public struct RenderBlendState: Sendable, Equatable {
    public let color: RenderBlendComponent
    public let alpha: RenderBlendComponent

    public init(color: RenderBlendComponent = .init(), alpha: RenderBlendComponent = .init()) {
        self.color = color
        self.alpha = alpha
    }
}

public struct RenderColorState: Sendable, Equatable {
    public let blend: RenderBlendState?
    public let writeMask: RenderColorWriteMask

    public init(blend: RenderBlendState? = nil, writeMask: RenderColorWriteMask = .all) {
        self.blend = blend
        self.writeMask = writeMask
    }

    package var usesSecondBlendSource: Bool {
        guard let blend else { return false }
        return [blend.color.sourceFactor, blend.color.destinationFactor, blend.alpha.sourceFactor, blend.alpha.destinationFactor]
            .contains(where: \.usesSecondSource)
    }
}

public struct RenderColorTarget: Sendable, Equatable {
    public let format: TextureFormat
    public let colorState: RenderColorState

    public init(format: TextureFormat, colorState: RenderColorState = .init()) {
        self.format = format
        self.colorState = colorState
    }
}

public struct RenderColorAttachment: Sendable, Equatable {
    public let texture: ResourceID
    public let mipLevel: UInt32
    public let loadAction: RenderColorLoadAction
    public let resolve: RenderResolveAttachment?
    /// Required for a 3D color attachment and relative to the selected mip.
    public let depthSlice: UInt32?

    public init(texture: ResourceID, mipLevel: UInt32 = 0, loadAction: RenderColorLoadAction = .load,
                resolve: RenderResolveAttachment? = nil, depthSlice: UInt32? = nil) {
        self.texture = texture
        self.mipLevel = mipLevel
        self.loadAction = loadAction
        self.resolve = resolve
        self.depthSlice = depthSlice
    }
}
