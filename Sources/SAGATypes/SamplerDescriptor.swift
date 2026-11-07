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
    public let magFilter: SamplerFilter
    public let minFilter: SamplerFilter
    public let mipmapFilter: SamplerFilter
    public let compare: CompareFunction?

    public init(addressModeU: SamplerAddressMode = .clampToEdge,
                addressModeV: SamplerAddressMode = .clampToEdge,
                minFilter: SamplerFilter = .nearest, magFilter: SamplerFilter = .nearest,
                mipmapFilter: SamplerFilter = .nearest, compare: CompareFunction? = nil) {
        self.addressModeU = addressModeU
        self.addressModeV = addressModeV
        self.minFilter = minFilter
        self.magFilter = magFilter
        self.mipmapFilter = mipmapFilter
        self.compare = compare
    }
}
