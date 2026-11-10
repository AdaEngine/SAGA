public enum SamplerFilter: UInt32, Sendable, Equatable {
    case nearest
    case linear
}

public enum SamplerAddressMode: UInt32, Sendable, Equatable {
    case clampToEdge
    case `repeat`
    case mirrorRepeat
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

    public init(addressModeU: SamplerAddressMode = .clampToEdge,
                addressModeV: SamplerAddressMode = .clampToEdge,
                addressModeW: SamplerAddressMode = .clampToEdge,
                minFilter: SamplerFilter = .nearest, magFilter: SamplerFilter = .nearest,
                mipmapFilter: SamplerFilter = .nearest, lodMinClamp: Float = 0, lodMaxClamp: Float = 32,
                compare: CompareFunction? = nil, maxAnisotropy: UInt16 = 1) {
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
    }

    public func validate() throws {
        guard lodMinClamp.isFinite, lodMaxClamp.isFinite, lodMinClamp >= 0, lodMinClamp <= lodMaxClamp else {
            throw GPUError.validation("Invalid sampler LOD clamp range")
        }
        guard maxAnisotropy >= 1 else {
            throw GPUError.validation("Sampler maxAnisotropy must be at least 1")
        }
        guard maxAnisotropy == 1 || (minFilter == .linear && magFilter == .linear && mipmapFilter == .linear) else {
            throw GPUError.validation("Anisotropic samplers require linear filters")
        }
    }
}
