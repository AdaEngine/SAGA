public enum SamplerFilter: UInt32, Sendable, Equatable {
    case nearest
    case linear
}

public enum SamplerAddressMode: UInt32, Sendable, Equatable {
    case clampToEdge
    case `repeat`
    case mirrorRepeat
    /// Texels outside the texture read as the sampler's border color; needs `GPUFeature.addressModeClampToBorder`.
    case clampToBorder
}

/// The color returned for texels outside the texture with `SamplerAddressMode.clampToBorder`.
public enum SamplerBorderColor: UInt32, Sendable, Equatable {
    case transparentBlack
    case opaqueBlack
    case opaqueWhite
}

/// A sampler for normalized coordinates, with an optional depth comparison.
public struct SamplerDescriptor: Sendable, Equatable {
    public let addressModeU: SamplerAddressMode
    public let addressModeV: SamplerAddressMode
    public let addressModeW: SamplerAddressMode
    public let magFilter: SamplerFilter
    public let minFilter: SamplerFilter
    public let mipmapFilter: SamplerFilter
    public let lodMinClamp: Float
    public let lodMaxClamp: Float
    public let compare: CompareFunction?
    /// Requested anisotropy; the backend clamps it to its supported maximum.
    public let maxAnisotropy: UInt16
    /// Required exactly when an address mode is `.clampToBorder`.
    public let borderColor: SamplerBorderColor?

    public init(addressModeU: SamplerAddressMode = .clampToEdge,
                addressModeV: SamplerAddressMode = .clampToEdge,
                addressModeW: SamplerAddressMode = .clampToEdge,
                minFilter: SamplerFilter = .nearest, magFilter: SamplerFilter = .nearest,
                mipmapFilter: SamplerFilter = .nearest, lodMinClamp: Float = 0, lodMaxClamp: Float = 32,
                compare: CompareFunction? = nil, maxAnisotropy: UInt16 = 1, borderColor: SamplerBorderColor? = nil) {
        self.addressModeU = addressModeU
        self.addressModeV = addressModeV
        self.addressModeW = addressModeW
        self.minFilter = minFilter
        self.magFilter = magFilter
        self.mipmapFilter = mipmapFilter
        self.lodMinClamp = lodMinClamp
        self.lodMaxClamp = lodMaxClamp
        self.compare = compare
        self.maxAnisotropy = maxAnisotropy
        self.borderColor = borderColor
    }

    /// True when any axis clamps to the border color.
    public var usesBorder: Bool {
        [addressModeU, addressModeV, addressModeW].contains(.clampToBorder)
    }

    public func validate() throws {
        guard lodMinClamp.isFinite, lodMaxClamp.isFinite, lodMinClamp >= 0, lodMinClamp <= lodMaxClamp else {
            throw GPUError.validation("Invalid sampler LOD clamp range")
        }
        guard usesBorder == (borderColor != nil) else {
            throw GPUError.validation("A sampler needs a border color exactly when an address mode clamps to the border")
        }
        guard maxAnisotropy >= 1 else {
            throw GPUError.validation("Sampler maxAnisotropy must be at least 1")
        }
        guard maxAnisotropy == 1 || (minFilter == .linear && magFilter == .linear && mipmapFilter == .linear) else {
            throw GPUError.validation("Anisotropic samplers require linear filters")
        }
    }
}
