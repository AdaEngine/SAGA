public enum TextureFormat: UInt32, Sendable, Hashable {
    case rgba8Unorm = 1
    case bgra8Unorm = 2
    case r32Uint = 3
    case depth32Float = 4
    case rgba8Uint = 5
    case rgba8Sint = 6
    case depth24PlusStencil8 = 7
    case rgba8Snorm = 8
    case rgba16Uint = 9
    case rgba16Sint = 10
    case rgba16Float = 11
    case r32Sint = 12
    case r32Float = 13
    case rg32Uint = 14
    case rg32Sint = 15
    case rg32Float = 16
    case rgba32Uint = 17
    case rgba32Sint = 18
    case rgba32Float = 19
    case bc1RGBAUnorm = 20
    case bc1RGBAUnormSrgb = 21
    case bc2RGBAUnorm = 22
    case bc2RGBAUnormSrgb = 23
    case bc3RGBAUnorm = 24
    case bc3RGBAUnormSrgb = 25
    case bc4RUnorm = 26
    case bc4RSnorm = 27
    case bc5RGUnorm = 28
    case bc5RGSnorm = 29
    case bc6HRGBUfloat = 30
    case bc6HRGBFloat = 31
    case bc7RGBAUnorm = 32
    case bc7RGBAUnormSrgb = 33
    case etc2RGB8Unorm = 34
    case etc2RGB8UnormSrgb = 35
    case etc2RGB8A1Unorm = 36
    case etc2RGB8A1UnormSrgb = 37
    case etc2RGBA8Unorm = 38
    case etc2RGBA8UnormSrgb = 39
    case eacR11Unorm = 40
    case eacR11Snorm = 41
    case eacRG11Unorm = 42
    case eacRG11Snorm = 43
    case astc4x4Unorm = 44
    case astc4x4UnormSrgb = 45
    case astc5x4Unorm = 46
    case astc5x4UnormSrgb = 47
    case astc5x5Unorm = 48
    case astc5x5UnormSrgb = 49
    case astc6x5Unorm = 50
    case astc6x5UnormSrgb = 51
    case astc6x6Unorm = 52
    case astc6x6UnormSrgb = 53
    case astc8x5Unorm = 54
    case astc8x5UnormSrgb = 55
    case astc8x6Unorm = 56
    case astc8x6UnormSrgb = 57
    case astc8x8Unorm = 58
    case astc8x8UnormSrgb = 59
    case astc10x5Unorm = 60
    case astc10x5UnormSrgb = 61
    case astc10x6Unorm = 62
    case astc10x6UnormSrgb = 63
    case astc10x8Unorm = 64
    case astc10x8UnormSrgb = 65
    case astc10x10Unorm = 66
    case astc10x10UnormSrgb = 67
    case astc12x10Unorm = 68
    case astc12x10UnormSrgb = 69
    case astc12x12Unorm = 70
    case astc12x12UnormSrgb = 71
    case astc4x4HDR = 72
    case astc5x4HDR = 73
    case astc5x5HDR = 74
    case astc6x5HDR = 75
    case astc6x6HDR = 76
    case astc8x5HDR = 77
    case astc8x6HDR = 78
    case astc8x8HDR = 79
    case astc10x5HDR = 80
    case astc10x6HDR = 81
    case astc10x8HDR = 82
    case astc10x10HDR = 83
    case astc12x10HDR = 84
    case astc12x12HDR = 85
    case r16Float = 86
    case rg16Float = 87
    case r16Uint = 88
    case r16Sint = 89
    case rg16Uint = 90
    case rg16Sint = 91
    case r16Unorm = 92
    case rg16Unorm = 93
    case rgba16Unorm = 94
    case r16Snorm = 95
    case rg16Snorm = 96
    case rgba16Snorm = 97
    case rgb10a2Unorm = 98
    case rgb10a2Uint = 99
    case rg11b10Ufloat = 100
    case rgba8UnormSrgb = 101
    case bgra8UnormSrgb = 102
    case rgb9e5Ufloat = 103
    case r8Unorm = 104
    case r8Snorm = 105
    case r8Uint = 106
    case r8Sint = 107
    case rg8Unorm = 108
    case rg8Snorm = 109
    case rg8Uint = 110
    case rg8Sint = 111
    case depth16Unorm = 112
    case depth24Plus = 113
    case depth32FloatStencil8 = 114
    case stencil8 = 115
    case r64Uint = 116

    package var hasStencil: Bool { self == .depth24PlusStencil8 || self == .depth32FloatStencil8 || self == .stencil8 }
    package var hasDepth: Bool {
        self == .depth16Unorm || self == .depth24Plus || self == .depth32Float ||
            self == .depth24PlusStencil8 || self == .depth32FloatStencil8
    }
    package var isDepthStencil: Bool { hasDepth || hasStencil }

    package func supports(aspect: TextureViewAspect) -> Bool {
        switch aspect {
        case .all: true
        case .depthOnly: hasDepth
        case .stencilOnly: hasStencil
        }
    }

    package var compression: (feature: GPUFeature, block: TextureCopyBlock)? {
        switch self {
        case .bc1RGBAUnorm, .bc1RGBAUnormSrgb, .bc4RUnorm, .bc4RSnorm:
            (.textureCompressionBC, .init(width: 4, height: 4, bytes: 8))
        case .bc2RGBAUnorm, .bc2RGBAUnormSrgb, .bc3RGBAUnorm, .bc3RGBAUnormSrgb,
             .bc5RGUnorm, .bc5RGSnorm, .bc6HRGBUfloat, .bc6HRGBFloat, .bc7RGBAUnorm, .bc7RGBAUnormSrgb:
            (.textureCompressionBC, .init(width: 4, height: 4, bytes: 16))
        case .etc2RGB8Unorm, .etc2RGB8UnormSrgb, .etc2RGB8A1Unorm, .etc2RGB8A1UnormSrgb, .eacR11Unorm, .eacR11Snorm:
            (.textureCompressionETC2, .init(width: 4, height: 4, bytes: 8))
        case .etc2RGBA8Unorm, .etc2RGBA8UnormSrgb, .eacRG11Unorm, .eacRG11Snorm:
            (.textureCompressionETC2, .init(width: 4, height: 4, bytes: 16))
        default:
            astcBlock.map { (astcUsesHDR ? .textureCompressionASTCHDR : .textureCompressionASTC, $0) }
        }
    }

    private var astcBlock: TextureCopyBlock? {
        switch self {
        case .astc4x4Unorm, .astc4x4UnormSrgb, .astc4x4HDR: .init(width: 4, height: 4, bytes: 16)
        case .astc5x4Unorm, .astc5x4UnormSrgb, .astc5x4HDR: .init(width: 5, height: 4, bytes: 16)
        case .astc5x5Unorm, .astc5x5UnormSrgb, .astc5x5HDR: .init(width: 5, height: 5, bytes: 16)
        case .astc6x5Unorm, .astc6x5UnormSrgb, .astc6x5HDR: .init(width: 6, height: 5, bytes: 16)
        case .astc6x6Unorm, .astc6x6UnormSrgb, .astc6x6HDR: .init(width: 6, height: 6, bytes: 16)
        case .astc8x5Unorm, .astc8x5UnormSrgb, .astc8x5HDR: .init(width: 8, height: 5, bytes: 16)
        case .astc8x6Unorm, .astc8x6UnormSrgb, .astc8x6HDR: .init(width: 8, height: 6, bytes: 16)
        case .astc8x8Unorm, .astc8x8UnormSrgb, .astc8x8HDR: .init(width: 8, height: 8, bytes: 16)
        case .astc10x5Unorm, .astc10x5UnormSrgb, .astc10x5HDR: .init(width: 10, height: 5, bytes: 16)
        case .astc10x6Unorm, .astc10x6UnormSrgb, .astc10x6HDR: .init(width: 10, height: 6, bytes: 16)
        case .astc10x8Unorm, .astc10x8UnormSrgb, .astc10x8HDR: .init(width: 10, height: 8, bytes: 16)
        case .astc10x10Unorm, .astc10x10UnormSrgb, .astc10x10HDR: .init(width: 10, height: 10, bytes: 16)
        case .astc12x10Unorm, .astc12x10UnormSrgb, .astc12x10HDR: .init(width: 12, height: 10, bytes: 16)
        case .astc12x12Unorm, .astc12x12UnormSrgb, .astc12x12HDR: .init(width: 12, height: 12, bytes: 16)
        default: nil
        }
    }

    private var astcUsesHDR: Bool {
        switch self {
        case .astc4x4HDR, .astc5x4HDR, .astc5x5HDR, .astc6x5HDR,
             .astc6x6HDR, .astc8x5HDR, .astc8x6HDR, .astc8x8HDR,
             .astc10x5HDR, .astc10x6HDR, .astc10x8HDR, .astc10x10HDR,
             .astc12x10HDR, .astc12x12HDR: true
        default: false
        }
    }

    package var copyCompatibleFormat: Self {
        switch self {
        case .rgba8UnormSrgb: .rgba8Unorm
        case .bgra8UnormSrgb: .bgra8Unorm
        case .bc1RGBAUnormSrgb: .bc1RGBAUnorm
        case .bc2RGBAUnormSrgb: .bc2RGBAUnorm
        case .bc3RGBAUnormSrgb: .bc3RGBAUnorm
        case .bc7RGBAUnormSrgb: .bc7RGBAUnorm
        case .etc2RGB8UnormSrgb: .etc2RGB8Unorm
        case .etc2RGB8A1UnormSrgb: .etc2RGB8A1Unorm
        case .etc2RGBA8UnormSrgb: .etc2RGBA8Unorm
        case .astc4x4UnormSrgb: .astc4x4Unorm
        case .astc5x4UnormSrgb: .astc5x4Unorm
        case .astc5x5UnormSrgb: .astc5x5Unorm
        case .astc6x5UnormSrgb: .astc6x5Unorm
        case .astc6x6UnormSrgb: .astc6x6Unorm
        case .astc8x5UnormSrgb: .astc8x5Unorm
        case .astc8x6UnormSrgb: .astc8x6Unorm
        case .astc8x8UnormSrgb: .astc8x8Unorm
        case .astc10x5UnormSrgb: .astc10x5Unorm
        case .astc10x6UnormSrgb: .astc10x6Unorm
        case .astc10x8UnormSrgb: .astc10x8Unorm
        case .astc10x10UnormSrgb: .astc10x10Unorm
        case .astc12x10UnormSrgb: .astc12x10Unorm
        case .astc12x12UnormSrgb: .astc12x12Unorm
        default: self
        }
    }

    package func copyBlock(aspect: TextureViewAspect) -> TextureCopyBlock? {
        guard supports(aspect: aspect) else { return nil }
        if let compression { return compression.block }
        if hasStencil {
            switch aspect {
            case .stencilOnly: return .init(bytes: 1)
            case .all where !hasDepth: return .init(bytes: 1)
            case .depthOnly where self == .depth32FloatStencil8: return .init(bytes: 4)
            default: return nil
            }
        }
        switch self {
        case .depth24Plus: return nil
        case .depth16Unorm: return .init(bytes: 2)
        case .r8Unorm, .r8Snorm, .r8Uint, .r8Sint: return .init(bytes: 1)
        case .rg8Unorm, .rg8Snorm, .rg8Uint, .rg8Sint,
             .r16Float, .r16Uint, .r16Sint, .r16Unorm, .r16Snorm: return .init(bytes: 2)
        case .rgba16Uint, .rgba16Sint, .rgba16Float, .rgba16Unorm, .rgba16Snorm,
             .rg32Uint, .rg32Sint, .rg32Float, .r64Uint: return .init(bytes: 8)
        case .rgba32Uint, .rgba32Sint, .rgba32Float: return .init(bytes: 16)
        default: return .init(bytes: 4)
        }
    }

    package var requiresNarrowInteger16Storage: Bool {
        self == .r16Uint || self == .r16Sint || self == .rg16Uint || self == .rg16Sint
    }
    package var isInteger16Color: Bool { requiresNarrowInteger16Storage || self == .rgba16Uint || self == .rgba16Sint }
    package var isFloat16Color: Bool { self == .r16Float || self == .rg16Float || self == .rgba16Float }
    package var isFloat32Color: Bool { self == .r32Float || self == .rg32Float || self == .rgba32Float }
    package var isNormalized16Color: Bool {
        switch self {
        case .r16Unorm, .r16Snorm, .rg16Unorm, .rg16Snorm, .rgba16Unorm, .rgba16Snorm: true
        default: false
        }
    }
    package var isPackedColor: Bool { isPackedFloatColor || self == .rgb10a2Uint }
    package var isPackedFloatColor: Bool { self == .rgb10a2Unorm || self == .rg11b10Ufloat }
    package var isNarrow8BitColor: Bool {
        switch self {
        case .r8Unorm, .r8Snorm, .r8Uint, .r8Sint, .rg8Unorm, .rg8Snorm, .rg8Uint, .rg8Sint: true
        default: false
        }
    }
    package var isSnorm8Color: Bool { self == .r8Snorm || self == .rg8Snorm || self == .rgba8Snorm }
    package var isNarrow8UnormColor: Bool { self == .r8Unorm || self == .rg8Unorm }
    package var isNormalizedColor: Bool { self == .rgba8Unorm || self == .bgra8Unorm }
    package var isSRGB8Color: Bool { self == .rgba8UnormSrgb || self == .bgra8UnormSrgb }

    package var hdrRenderFeature: GPUFeature? {
        switch self {
        case .rg11b10Ufloat: .rg11b10UfloatRenderable
        case .rgb9e5Ufloat: .rgb9e5Renderable
        default: nil
        }
    }

    package var isColorAttachment: Bool {
        isNormalizedColor || isSRGB8Color || isNormalized16Color || isFloat16Color || isFloat32Color ||
            isInteger16Color || isPackedColor || isNarrow8BitColor || isSnorm8Color ||
            self == .rgb9e5Ufloat || self == .rgba8Uint || self == .rgba8Sint
    }
    package func supportsColorResolve(features: Set<GPUFeature>) -> Bool {
        if let feature = hdrRenderFeature { return features.contains(feature) }
        return isNormalizedColor || isSRGB8Color || isFloat16Color || isNarrow8UnormColor || self == .rgb10a2Unorm ||
            (isSnorm8Color && features.contains(.snorm8Renderable)) ||
            (isNormalized16Color && features.contains(.normalized16Resolve))
    }
    package func supportsMultisampledAttachment(in capabilities: DeviceCapabilities?) -> Bool {
        isDepthStencil || (isColorAttachment && (!isFloat32Color || capabilities?.multisampledFloat32Formats.contains(self) == true))
    }

    package var colorAttachmentComponentCount: UInt8 {
        switch self {
        case .r8Unorm, .r8Snorm, .r8Uint, .r8Sint, .r16Float, .r32Float, .r16Uint, .r16Sint, .r16Unorm, .r16Snorm: 1
        case .rg8Unorm, .rg8Snorm, .rg8Uint, .rg8Sint, .rg16Float, .rg32Float, .rg16Uint, .rg16Sint, .rg16Unorm, .rg16Snorm: 2
        case .rg11b10Ufloat, .rgb9e5Ufloat: 3
        default: 4
        }
    }

    package var colorAttachmentByteCost: Int {
        switch self {
        case .r8Unorm, .r8Snorm, .r8Uint, .r8Sint: 1
        case .rg8Unorm, .rg8Snorm, .rg8Uint, .rg8Sint, .r16Float, .r16Uint, .r16Sint, .r16Unorm, .r16Snorm: 2
        case .rgba32Float: 16
        case .rgba8Unorm, .bgra8Unorm, .rg32Float, .rgba16Float, .rgba16Uint, .rgba16Sint, .rgba16Unorm, .rgba16Snorm,
             .rgb10a2Unorm, .rgb10a2Uint, .rgba8UnormSrgb, .bgra8UnormSrgb, .rgba8Snorm: 8
        default: 4
        }
    }

    package var colorAttachmentAlignment: Int {
        if isFloat32Color || isPackedColor || self == .rgb9e5Ufloat { return 4 }
        return isFloat16Color || isInteger16Color || isNormalized16Color ? 2 : 1
    }

    package var supportsStorage: Bool {
        switch self {
        case .depth16Unorm, .depth24Plus, .depth32Float, .depth24PlusStencil8, .depth32FloatStencil8, .stencil8,
             .rgba8UnormSrgb, .bgra8UnormSrgb, .rgb9e5Ufloat: false
        default: compression == nil
        }
    }

    package var sampleType: TextureSampleType {
        switch self {
        case .depth16Unorm, .depth24Plus, .depth32Float, .depth24PlusStencil8, .depth32FloatStencil8: .depth
        case .stencil8, .r8Uint, .rg8Uint, .r16Uint, .rg16Uint, .r32Uint, .r64Uint, .rgba8Uint, .rgba16Uint,
             .rg32Uint, .rgba32Uint, .rgb10a2Uint: .uint
        case .r8Sint, .rg8Sint, .r16Sint, .rg16Sint, .r32Sint, .rgba8Sint, .rgba16Sint, .rg32Sint, .rgba32Sint: .sint
        default: .float
        }
    }
}

package struct TextureCopyBlock: Sendable {
    package let width: UInt32
    package let height: UInt32
    package let bytes: UInt32

    init(width: UInt32 = 1, height: UInt32 = 1, bytes: UInt32) {
        self.width = width
        self.height = height
        self.bytes = bytes
    }

    package func rowBytes(width: UInt32) -> UInt64 {
        UInt64(width / self.width) * UInt64(bytes)
    }
}

package enum TextureSampleType: UInt32, Sendable {
    case float = 0
    case depth = 1
    case sint = 2
    case uint = 3
}

public struct TextureUsage: OptionSet, Sendable {
    public let rawValue: UInt32
    public init(rawValue: UInt32) { self.rawValue = rawValue }
    public static let copySource = Self(rawValue: 1 << 0)
    public static let copyDestination = Self(rawValue: 1 << 1)
    public static let textureBinding = Self(rawValue: 1 << 2)
    public static let storageBinding = Self(rawValue: 1 << 3)
    public static let renderAttachment = Self(rawValue: 1 << 4)
    public static let storageAtomic = Self(rawValue: 1 << 5)
}

public enum TextureDimension: UInt32, Sendable, Equatable {
    case d2
    case d3
    case d1
}

public struct TextureDescriptor: Sendable, Equatable {
    public let width: UInt32
    public let height: UInt32
    public let format: TextureFormat
    public let usage: TextureUsage
    public let mipLevelCount: UInt32
    public let sampleCount: UInt32
    public let arrayLayerCount: UInt32
    public let dimension: TextureDimension
    public let depth: UInt32

    public init(width: UInt32, height: UInt32, format: TextureFormat, usage: TextureUsage, mipLevelCount: UInt32 = 1,
                sampleCount: UInt32 = 1, arrayLayerCount: UInt32 = 1, dimension: TextureDimension = .d2, depth: UInt32 = 1) {
        self.width = width
        self.height = height
        self.format = format
        self.usage = usage
        self.mipLevelCount = mipLevelCount
        self.sampleCount = sampleCount
        self.arrayLayerCount = arrayLayerCount
        self.dimension = dimension
        self.depth = depth
    }

    private var maximumMipLevelCount: UInt32 {
        UInt32(UInt32.bitWidth - max(width, height, depth).leadingZeroBitCount)
    }

    package func mipExtent(_ level: UInt32) -> (width: UInt32, height: UInt32)? {
        guard width > 0, height > 0, level < mipLevelCount, level < maximumMipLevelCount else { return nil }
        return (max(1, width >> level), max(1, height >> level))
    }

    package func mipPhysicalExtent(_ level: UInt32) -> (width: UInt32, height: UInt32)? {
        guard let extent = mipExtent(level) else { return nil }
        guard let block = format.compression?.block else { return extent }
        func roundUp(_ extent: UInt32, multiple: UInt32) -> UInt32? {
            let axis = UInt64(multiple)
            return UInt32(exactly: (UInt64(extent) + axis - 1) / axis * axis)
        }
        guard let width = roundUp(extent.width, multiple: block.width),
              let height = roundUp(extent.height, multiple: block.height) else { return nil }
        return (width, height)
    }

    package func mipDepth(_ level: UInt32) -> UInt32 { max(1, depth >> level) }

    public func validate(against capabilities: DeviceCapabilities) throws {
        if format == .depth32FloatStencil8 { try capabilities.require([.depth32FloatStencil8]) }
        if format.isNormalized16Color { try capabilities.require([.normalized16Textures]) }
        if usage.contains(.renderAttachment) {
            if let feature = format.hdrRenderFeature { try capabilities.require([feature]) }
            if format.isSnorm8Color { try capabilities.require([.snorm8Renderable]) }
        }
        if let compression = format.compression {
            try capabilities.require([compression.feature])
            guard dimension != .d1, width % compression.block.width == 0, height % compression.block.height == 0,
                  sampleCount == 1,
                  usage.intersection([.storageBinding, .renderAttachment]).isEmpty else {
                throw GPUError.validation("Invalid compressed texture extent, sample count or usage")
            }
            if dimension == .d3 {
                switch compression.feature {
                case .textureCompressionBC:
                    try capabilities.require([.textureCompressionBCSliced3D])
                case .textureCompressionASTC, .textureCompressionASTCHDR:
                    try capabilities.require([.textureCompressionASTCSliced3D])
                default:
                    throw GPUError.validation("Format does not support 3D compressed textures")
                }
            }
        }
        if dimension == .d1 {
            guard height == 1, depth == 1, arrayLayerCount == 1, sampleCount == 1, mipLevelCount == 1,
                  !format.isDepthStencil, !usage.contains(.renderAttachment) else {
                throw GPUError.validation("Invalid 1D texture extent, mip levels, format or usage")
            }
        } else if dimension == .d3 {
            guard (1...2048).contains(width), (1...2048).contains(height), (1...2048).contains(depth),
                  arrayLayerCount == 1, sampleCount == 1, !format.isDepthStencil,
                  !usage.contains(.renderAttachment) || format.isColorAttachment else {
                throw GPUError.validation("Invalid 3D texture extent, format or usage")
            }
        } else if depth != 1 {
            throw GPUError.validation("2D textures must have depth one")
        }
        guard width > 0, height > 0, width <= capabilities.maxTextureDimension2D,
              height <= capabilities.maxTextureDimension2D else {
            throw GPUError.validation("Invalid 2D texture extent")
        }
        guard (1...256).contains(arrayLayerCount) else {
            throw GPUError.validation("Invalid 2D texture array layer count")
        }
        guard mipLevelCount > 0, mipLevelCount <= maximumMipLevelCount else {
            throw GPUError.validation("Invalid texture mip level count")
        }
        guard !usage.isEmpty, usage.subtracting([.copySource, .copyDestination, .textureBinding, .storageBinding,
                                                  .renderAttachment, .storageAtomic]).isEmpty else {
            throw GPUError.validation("Invalid texture usage")
        }
        if format == .r64Uint { try capabilities.require([.textureInt64Atomic]) }
        if usage.contains(.storageAtomic) {
            try capabilities.require([format == .r64Uint ? .textureInt64Atomic : .textureAtomic])
            guard sampleCount == 1, format == .r32Uint || format == .r32Sint || format == .r64Uint else {
                throw GPUError.validation("Atomic storage requires a single-sampled R32Uint, R32Sint or R64Uint texture")
            }
        }
        guard sampleCount == 1 || sampleCount == 4 else {
            throw GPUError.validation("Texture sample count must be 1 or 4")
        }
        if sampleCount > 1 {
            guard mipLevelCount == 1, arrayLayerCount == 1, usage.contains(.renderAttachment), !usage.contains(.storageBinding),
                  format.supportsMultisampledAttachment(in: capabilities) else {
                throw GPUError.validation("Invalid multisampled texture format, mip levels or usage")
            }
        }
        if usage.contains(.storageBinding) {
            guard format.supportsStorage else { throw GPUError.validation("Invalid format for a storage texture") }
            if format == .r16Float || format == .rg16Float {
                try capabilities.require([.halfFloatStorageTextures])
            }
            if format.requiresNarrowInteger16Storage {
                try capabilities.require([.narrowInteger16StorageTextures])
            }
            if format.isPackedColor { try capabilities.require([.packedColorStorageTextures]) }
            if format == .bgra8Unorm { try capabilities.require([.bgra8UnormStorage]) }
            if format.isNarrow8BitColor { try capabilities.require([.narrow8BitStorageTextures]) }
        }
        if format.supportsStorage, !format.isColorAttachment,
           !usage.intersection([.textureBinding, .renderAttachment]).isEmpty {
            throw GPUError.validation("\(format) supports storage bindings and copies in this profile")
        }
    }
}

public enum TextureViewAspect: UInt32, Sendable, Equatable {
    case all
    case depthOnly
    case stencilOnly
}

public enum TextureViewDimension: UInt32, Sendable, Equatable {
    case d2
    case d2Array
    case cube
    case cubeArray
    case d3
    case d1
}

/// A texture view whose level zero is the selected base mip.
public struct TextureViewDescriptor: Sendable, Equatable {
    public let baseMipLevel: UInt32
    /// When omitted, includes every mip level from the base through the end of the texture.
    public let mipLevelCount: UInt32?
    public let aspect: TextureViewAspect
    public let dimension: TextureViewDimension?
    public let baseArrayLayer: UInt32
    public let arrayLayerCount: UInt32?

    public init(baseMipLevel: UInt32 = 0, mipLevelCount: UInt32? = nil, aspect: TextureViewAspect = .all,
                dimension: TextureViewDimension? = nil, baseArrayLayer: UInt32 = 0, arrayLayerCount: UInt32? = nil) {
        self.baseMipLevel = baseMipLevel
        self.mipLevelCount = mipLevelCount
        self.aspect = aspect
        self.dimension = dimension
        self.baseArrayLayer = baseArrayLayer
        self.arrayLayerCount = arrayLayerCount
    }

    package func resolvedDimension(in texture: TextureDescriptor) -> TextureViewDimension {
        if let dimension { return dimension }
        switch texture.dimension {
        case .d1: return .d1
        case .d3: return .d3
        case .d2: return texture.arrayLayerCount > 1 ? .d2Array : .d2
        }
    }

    package func arrayLayers(in texture: TextureDescriptor) throws -> Range<UInt32> {
        guard baseArrayLayer < texture.arrayLayerCount else {
            throw GPUError.validation("Invalid texture view base array layer")
        }
        let dimension = resolvedDimension(in: texture)
        switch (texture.dimension, dimension) {
        case (.d1, .d1), (.d3, .d3), (.d2, .d2), (.d2, .d2Array), (.d2, .cube), (.d2, .cubeArray): break
        default: throw GPUError.validation("Texture view dimension does not match the texture")
        }
        let remaining = texture.arrayLayerCount - baseArrayLayer
        let count = arrayLayerCount ?? (dimension == .d2 ? 1 : dimension == .cube ? 6 : remaining)
        guard count > 0, count <= remaining, texture.sampleCount == 1 || dimension == .d2 else {
            throw GPUError.validation("Invalid texture view dimension or array layer count")
        }
        switch dimension {
        case .d2 where count != 1:
            throw GPUError.validation("Invalid texture view dimension or array layer count")
        case .cube, .cubeArray:
            guard texture.width == texture.height, dimension == .cube ? count == 6 : count % 6 == 0 else {
                throw GPUError.validation("Cube views require square faces in groups of six")
            }
        default: break
        }
        return baseArrayLayer..<(baseArrayLayer + count)
    }

    package func mipLevels(in texture: TextureDescriptor) throws -> Range<UInt32> {
        guard texture.format.supports(aspect: aspect) else {
            throw GPUError.validation("Texture view aspect does not match the texture format")
        }
        guard baseMipLevel < texture.mipLevelCount else {
            throw GPUError.validation("Invalid texture view base mip level")
        }
        let remaining = texture.mipLevelCount - baseMipLevel
        let count = mipLevelCount ?? remaining
        guard count > 0, count <= remaining, texture.mipExtent(baseMipLevel + count - 1) != nil else {
            throw GPUError.validation("Invalid texture view mip level count")
        }
        return baseMipLevel..<(baseMipLevel + count)
    }
}

package struct TextureViewResource: Sendable {
    package let texture: ResourceID
    package let levels: Range<UInt32>
    package let aspect: TextureViewAspect
    package let layers: Range<UInt32>
    package let dimension: TextureViewDimension

    package init(texture: ResourceID, levels: Range<UInt32>, aspect: TextureViewAspect = .all,
                 layers: Range<UInt32> = 0..<1, dimension: TextureViewDimension = .d2) {
        self.texture = texture
        self.levels = levels
        self.aspect = aspect
        self.layers = layers
        self.dimension = dimension
    }

    package var descriptor: TextureViewDescriptor {
        .init(baseMipLevel: levels.lowerBound, mipLevelCount: levels.upperBound - levels.lowerBound,
              aspect: aspect, dimension: dimension, baseArrayLayer: layers.lowerBound,
              arrayLayerCount: layers.upperBound - layers.lowerBound)
    }

    package func overlaps(_ other: Self) -> Bool {
        texture == other.texture && levels.overlaps(other.levels) && layers.overlaps(other.layers) &&
            (aspect == .all || other.aspect == .all || aspect == other.aspect)
    }
}

public struct TextureCopyRegion: Sendable, Equatable {
    public let x: UInt32
    public let y: UInt32
    public let width: UInt32
    public let height: UInt32
    public let mipLevel: UInt32
    public let arrayLayer: UInt32
    public let z: UInt32
    public let depth: UInt32

    public init(x: UInt32 = 0, y: UInt32 = 0, width: UInt32, height: UInt32, mipLevel: UInt32 = 0,
                arrayLayer: UInt32 = 0, z: UInt32 = 0, depth: UInt32 = 1) {
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.mipLevel = mipLevel
        self.arrayLayer = arrayLayer
        self.z = z
        self.depth = depth
    }

    package var isEmpty: Bool { width == 0 || height == 0 || depth == 0 }

    package func planes(in texture: TextureDescriptor) -> Range<UInt32> {
        texture.dimension == .d3 ? z..<(z + depth) : arrayLayer..<(arrayLayer + depth)
    }

    package func fits(in texture: TextureDescriptor, allowsArrayLayerSpan: Bool = false) -> Bool {
        guard let extent = texture.mipPhysicalExtent(mipLevel), x <= extent.width, y <= extent.height else { return false }
        if let block = texture.format.compression?.block {
            guard x % block.width == 0, y % block.height == 0,
                  width % block.width == 0, height % block.height == 0 else { return false }
        }
        if texture.dimension == .d3 {
            let mipDepth = texture.mipDepth(mipLevel)
            guard arrayLayer == 0, texture.arrayLayerCount == 1, z <= mipDepth, depth <= mipDepth - z else { return false }
        } else {
            guard z == 0, arrayLayer <= texture.arrayLayerCount,
                  depth <= texture.arrayLayerCount - arrayLayer, allowsArrayLayerSpan || depth == 1 else { return false }
        }
        return width <= extent.width - x && height <= extent.height - y
    }

    package func covers(_ texture: TextureDescriptor) -> Bool {
        guard let extent = texture.mipPhysicalExtent(mipLevel) else { return false }
        return x == 0 && y == 0 && width == extent.width && height == extent.height
    }
}

public struct TextureCopyOrigin: Sendable, Equatable {
    public let x: UInt32
    public let y: UInt32
    public let mipLevel: UInt32
    public let arrayLayer: UInt32
    public let z: UInt32

    public init(x: UInt32 = 0, y: UInt32 = 0, mipLevel: UInt32 = 0, arrayLayer: UInt32 = 0, z: UInt32 = 0) {
        self.x = x
        self.y = y
        self.mipLevel = mipLevel
        self.arrayLayer = arrayLayer
        self.z = z
    }

    package func region(width: UInt32, height: UInt32, depth: UInt32 = 1) -> TextureCopyRegion {
        .init(x: x, y: y, width: width, height: height, mipLevel: mipLevel, arrayLayer: arrayLayer, z: z, depth: depth)
    }
}

public struct TextureBufferLayout: Sendable, Equatable {
    public let offset: UInt64
    /// Required for multiple rows. When present, it is a multiple of 256 bytes.
    public let bytesPerRow: UInt32?

    public init(offset: UInt64 = 0, bytesPerRow: UInt32? = nil) {
        self.offset = offset
        self.bytesPerRow = bytesPerRow
    }

    package func rowStride(width: UInt32, block: TextureCopyBlock) -> UInt64 {
        bytesPerRow.map(UInt64.init) ?? block.rowBytes(width: width)
    }

    package func byteCount(width: UInt32, height: UInt32, block: TextureCopyBlock, depth: UInt32 = 1) throws -> UInt64 {
        let rowBytes = block.rowBytes(width: width)
        let blockRows = height / block.height
        if let bytesPerRow {
            guard bytesPerRow % 256 == 0, UInt64(bytesPerRow) >= rowBytes else {
                throw GPUError.validation("Invalid texture copy row stride")
            }
        } else if blockRows > 1 || depth > 1 {
            throw GPUError.validation("Texture copies with multiple rows or planes require a row stride")
        }
        if blockRows == 0 || depth == 0 { return 0 }
        let rows = UInt64(blockRows) * UInt64(depth) - 1
        let (padding, overflow) = rows.multipliedReportingOverflow(by: rowStride(width: width, block: block))
        guard !overflow, padding <= UInt64.max - rowBytes else { throw GPUError.validation("Texture copy byte count overflows") }
        return padding + rowBytes
    }
}

/// Layout of CPU bytes used by a texture upload. Offsets and row strides are in bytes.
public struct TextureDataLayout: Sendable, Equatable {
    public let offset: UInt64
    public let bytesPerRow: UInt32?
    public let rowsPerImage: UInt32?

    public init(offset: UInt64 = 0, bytesPerRow: UInt32? = nil, rowsPerImage: UInt32? = nil) {
        self.offset = offset
        self.bytesPerRow = bytesPerRow
        self.rowsPerImage = rowsPerImage
    }

    package func footprint(
        region: TextureCopyRegion, block: TextureCopyBlock, dataSize: UInt64
    ) throws -> (rowBytes: UInt64, rowStride: UInt64, imageStride: UInt64) {
        let rowBytes = block.rowBytes(width: region.width)
        let blockRows = region.height / block.height
        let stride = bytesPerRow.map(UInt64.init) ?? rowBytes
        guard stride >= rowBytes,
              bytesPerRow != nil || (blockRows <= 1 && region.depth <= 1),
              rowsPerImage != nil || region.depth <= 1,
              rowsPerImage.map({ $0 >= blockRows }) ?? true else {
            throw GPUError.validation("Invalid texture data row or image stride")
        }
        var imageStride: UInt64 = 0
        var required: UInt64 = 0
        if region.depth > 1, let rowsPerImage {
            // Both layout strides are explicitly present and 32-bit for multiple images.
            imageStride = stride * UInt64(rowsPerImage)
            let (size, overflow) = imageStride.multipliedReportingOverflow(by: UInt64(region.depth - 1))
            guard !overflow else { throw GPUError.validation("Texture upload footprint overflows") }
            required = size
        }
        if region.depth > 0 && blockRows > 0 {
            let (padding, rowOverflow) = stride.multipliedReportingOverflow(by: UInt64(blockRows - 1))
            let (size, imageOverflow) = required.addingReportingOverflow(padding)
            let (end, lastRowOverflow) = size.addingReportingOverflow(rowBytes)
            guard !rowOverflow, !imageOverflow, !lastRowOverflow else {
                throw GPUError.validation("Texture upload footprint overflows")
            }
            required = end
        }
        guard offset <= dataSize, required <= dataSize - offset else {
            throw GPUError.validation("Texture upload exceeds the source bytes")
        }
        return (rowBytes, stride, imageStride)
    }
}

public struct RenderClearColor: Sendable, Equatable {
    public let red: Double
    public let green: Double
    public let blue: Double
    public let alpha: Double

    public init(red: Double, green: Double, blue: Double, alpha: Double) {
        self.red = red
        self.green = green
        self.blue = blue
        self.alpha = alpha
    }

    public static let transparentBlack = Self(red: 0, green: 0, blue: 0, alpha: 0)

    package var channels: [Double] { [red, green, blue, alpha] }

    package func isValid(for format: TextureFormat) -> Bool {
        switch format.sampleType {
        case .sint: return channels.allSatisfy { (Double(Int32.min)...Double(Int32.max)).contains($0) }
        case .uint: return channels.allSatisfy { (0...Double(UInt32.max)).contains($0) }
        default: return channels.allSatisfy(\.isFinite)
        }
    }
}

public struct RenderPrimitiveState: Sendable, Equatable {
    public enum Topology: UInt32, Sendable, Equatable {
        case pointList
        case lineList
        case lineStrip
        case triangleList
        case triangleStrip

        package var isStrip: Bool { self == .lineStrip || self == .triangleStrip }
    }

    public enum FrontFace: UInt32, Sendable, Equatable {
        case counterClockwise
        case clockwise
    }

    public enum CullMode: UInt32, Sendable, Equatable {
        case none
        case front
        case back
    }

    public let topology: Topology
    public let stripIndexFormat: IndexFormat?
    public let frontFace: FrontFace
    public let cullMode: CullMode
    /// Disables clipping against the near and far planes when the device supports depth clip control.
    public let unclippedDepth: Bool

    public init(topology: Topology = .triangleList, stripIndexFormat: IndexFormat? = nil,
                frontFace: FrontFace = .counterClockwise, cullMode: CullMode = .none, unclippedDepth: Bool = false) {
        self.topology = topology
        self.stripIndexFormat = stripIndexFormat
        self.frontFace = frontFace
        self.cullMode = cullMode
        self.unclippedDepth = unclippedDepth
    }
}

public enum CompareFunction: UInt32, Sendable, Equatable {
    case never, less, equal, lessEqual, greater, notEqual, greaterEqual, always
}

public struct RenderDepthBiasState: Sendable, Equatable {
    public let constant: Int32
    public let slopeScale: Float
    public let clamp: Float

    public init(constant: Int32 = 0, slopeScale: Float = 0, clamp: Float = 0) {
        self.constant = constant
        self.slopeScale = slopeScale
        self.clamp = clamp
    }
}

public struct RenderDepthState: Sendable, Equatable {
    public let format: TextureFormat
    public let compare: CompareFunction
    public let writeEnabled: Bool
    public let bias: RenderDepthBiasState
    public let stencil: RenderStencilState?

    public init(format: TextureFormat = .depth32Float, compare: CompareFunction = .less, writeEnabled: Bool = true,
                bias: RenderDepthBiasState = .init(), stencil: RenderStencilState? = nil) {
        self.format = format
        self.compare = compare
        self.writeEnabled = writeEnabled
        self.bias = bias
        self.stencil = stencil
    }
}

/// A single-sample color destination for a multisampled render pass.
public struct RenderResolveAttachment: Sendable, Equatable {
    public let texture: ResourceID
    /// Destination mip. Use zero when binding a single-level view.
    public let mipLevel: UInt32

    public init(texture: ResourceID, mipLevel: UInt32 = 0) {
        self.texture = texture
        self.mipLevel = mipLevel
    }
}

public struct RenderDepthAttachment: Sendable, Equatable {
    public enum LoadAction: Sendable, Equatable {
        case load
        case readOnly
        case clear(Double)
        /// Omit the depth aspect when the attachment format is stencil-only.
        case unused
    }

    public let texture: ResourceID
    public let mipLevel: UInt32
    public let loadAction: LoadAction
    public let stencil: RenderStencilAttachmentState?

    public init(texture: ResourceID, mipLevel: UInt32 = 0, loadAction: LoadAction = .clear(1),
                stencil: RenderStencilAttachmentState? = nil) {
        self.texture = texture
        self.mipLevel = mipLevel
        self.loadAction = loadAction
        self.stencil = stencil
    }
}

public struct RenderPipeline: Sendable, Equatable {
    package struct Uniform: Sendable, Equatable {
        package let slot: ComputeBindingSlot
        package let minimumByteSize: UInt64
        package let vertexIndex: UInt32?
        package let fragmentIndex: UInt32?
        package let visibility: UInt32

        package init(slot: ComputeBindingSlot, minimumByteSize: UInt64, vertexIndex: UInt32?,
                     fragmentIndex: UInt32?, visibility: UInt32) {
            self.slot = slot
            self.minimumByteSize = minimumByteSize
            self.vertexIndex = vertexIndex
            self.fragmentIndex = fragmentIndex
            self.visibility = visibility
        }
    }

    package struct HandleBinding: Sendable, Equatable {
        package let slot: ComputeBindingSlot
        package let vertexIndex: UInt32?
        package let fragmentIndex: UInt32?
        package let visibility: UInt32

        package init(slot: ComputeBindingSlot, vertexIndex: UInt32?, fragmentIndex: UInt32?, visibility: UInt32) {
            self.slot = slot
            self.vertexIndex = vertexIndex
            self.fragmentIndex = fragmentIndex
            self.visibility = visibility
        }
    }

    package struct SamplerBinding: Sendable, Equatable {
        package let handle: HandleBinding
        package let requirement: SamplerRequirement

        package init(handle: HandleBinding, requirement: SamplerRequirement) {
            self.handle = handle
            self.requirement = requirement
        }
    }

    package struct SampledTextureBinding: Sendable, Equatable {
        package let handle: HandleBinding
        package let requirement: SampledTextureRequirement

        package init(handle: HandleBinding, requirement: SampledTextureRequirement) {
            self.handle = handle
            self.requirement = requirement
        }
    }

    package struct StorageBufferBinding: Sendable, Equatable {
        package let handle: HandleBinding
        package let minimumByteSize: UInt64
        package let isReadOnly: Bool

        package init(handle: HandleBinding, minimumByteSize: UInt64, isReadOnly: Bool) {
            self.handle = handle
            self.minimumByteSize = minimumByteSize
            self.isReadOnly = isReadOnly
        }
    }

    package struct StorageTextureBinding: Sendable, Equatable {
        package let handle: HandleBinding
        package let requirement: StorageTextureRequirement

        package init(handle: HandleBinding, requirement: StorageTextureRequirement) {
            self.handle = handle
            self.requirement = requirement
        }
    }

    public let id: ResourceID
    /// The shared, word-aligned immediate data range for both shader stages.
    public let immediateByteSize: UInt64
    public let colorFormat: TextureFormat
    public let writesColor: Bool
    public let colorState: RenderColorState
    public let additionalColorTargets: [UInt32: RenderColorTarget]
    package let additionalColorOutputs: Set<UInt32>

    package var colorTargets: [UInt32: RenderColorTarget] {
        var targets = additionalColorTargets
        targets[0] = .init(format: colorFormat, colorState: colorState)
        return targets
    }

    package func colorWriteMask(at location: UInt32) -> RenderColorWriteMask {
        if location == 0 { return writesColor ? colorState.writeMask : [] }
        return additionalColorOutputs.contains(location) ? additionalColorTargets[location]?.colorState.writeMask ?? [] : []
    }
    public let depthState: RenderDepthState?
    public let sampleCount: UInt32
    public let sampleMask: UInt32
    public let alphaToCoverageEnabled: Bool
    public let primitive: RenderPrimitiveState
    public let vertexBuffers: [VertexBufferLayout]
    package let uniforms: [Uniform]
    package let storageBuffers: [StorageBufferBinding]
    package let sampledTextures: [SampledTextureBinding]
    package let externalTextureBindings: [HandleBinding]
    package let storageTextures: [StorageTextureBinding]
    package let samplerBindings: [SamplerBinding]
    package let webBindingIndices: [ComputeBindingSlot: UInt32]
    /// Explicit and shader-inferred nearest-only sampler slots.
    public var nonFilteringSamplers: Set<ComputeBindingSlot> {
        Set(samplerBindings.filter { $0.requirement.requiresNonFiltering }.map { $0.handle.slot })
    }

    package init(id: ResourceID, colorFormat: TextureFormat, vertexBuffers: [VertexBufferLayout] = [],
                 depthState: RenderDepthState? = nil, sampleCount: UInt32 = 1, sampleMask: UInt32 = .max,
                 alphaToCoverageEnabled: Bool = false, writesColor: Bool = true,
                 primitive: RenderPrimitiveState = .init(), colorState: RenderColorState = .init(),
                 additionalColorTargets: [UInt32: RenderColorTarget] = [:], additionalColorOutputs: Set<UInt32> = [],
                 uniforms: [Uniform] = [], storageBuffers: [StorageBufferBinding] = [], sampledTextures: [SampledTextureBinding] = [],
                 storageTextures: [StorageTextureBinding] = [], samplerBindings: [SamplerBinding] = [],
                 externalTextureBindings: [HandleBinding] = [],
                 webBindingIndices: [ComputeBindingSlot: UInt32] = [:], immediateByteSize: UInt64 = 0) {
        self.id = id
        self.immediateByteSize = immediateByteSize
        self.colorFormat = colorFormat
        self.writesColor = writesColor
        self.colorState = colorState
        self.additionalColorTargets = additionalColorTargets
        self.additionalColorOutputs = additionalColorOutputs
        self.depthState = depthState
        self.sampleCount = sampleCount
        self.sampleMask = sampleMask
        self.alphaToCoverageEnabled = alphaToCoverageEnabled
        self.primitive = primitive
        self.vertexBuffers = vertexBuffers
        self.uniforms = uniforms
        self.storageBuffers = storageBuffers
        self.sampledTextures = sampledTextures
        self.externalTextureBindings = externalTextureBindings
        self.storageTextures = storageTextures
        self.samplerBindings = samplerBindings
        self.webBindingIndices = webBindingIndices
    }
}
