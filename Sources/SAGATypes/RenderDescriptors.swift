public enum TextureFormat: UInt32, Sendable, Equatable {
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

    package var hasStencil: Bool { self == .depth24PlusStencil8 }
    package var hasDepth: Bool { self == .depth32Float || hasStencil }

    package func supports(aspect: TextureViewAspect) -> Bool {
        switch aspect {
        case .all: true
        case .depthOnly: hasDepth
        case .stencilOnly: hasStencil
        }
    }

    package func copyTexelBytes(aspect: TextureViewAspect) -> UInt32? {
        guard supports(aspect: aspect) else { return nil }
        if hasStencil { return aspect == .stencilOnly ? 1 : nil }
        switch self {
        case .rgba16Uint, .rgba16Sint, .rgba16Float, .rg32Uint, .rg32Sint, .rg32Float: return 8
        case .rgba32Uint, .rgba32Sint, .rgba32Float: return 16
        default: return 4
        }
    }

    package var isNormalizedColor: Bool { self == .rgba8Unorm || self == .bgra8Unorm }
    package var isColorAttachment: Bool { isNormalizedColor || self == .rgba8Uint || self == .rgba8Sint }

    package var supportsStorage: Bool {
        switch self {
        case .bgra8Unorm, .depth32Float, .depth24PlusStencil8: false
        default: true
        }
    }

    package var sampleType: TextureSampleType {
        switch self {
        case .rgba8Unorm, .bgra8Unorm, .rgba8Snorm, .rgba16Float, .r32Float, .rg32Float, .rgba32Float: .float
        case .depth32Float, .depth24PlusStencil8: .depth
        case .r32Uint, .rgba8Uint, .rgba16Uint, .rg32Uint, .rgba32Uint: .uint
        case .r32Sint, .rgba8Sint, .rgba16Sint, .rg32Sint, .rgba32Sint: .sint
        }
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
}

public enum TextureDimension: UInt32, Sendable, Equatable {
    case d2
    case d3
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

    package func mipDepth(_ level: UInt32) -> UInt32 { max(1, depth >> level) }

    public func validate(against capabilities: DeviceCapabilities) throws {
        if dimension == .d3 {
            guard (1...2048).contains(width), (1...2048).contains(height), (1...2048).contains(depth),
                  arrayLayerCount == 1, sampleCount == 1, !format.hasDepth,
                  usage.intersection([.renderAttachment, .storageBinding]).isEmpty else {
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
        guard !usage.isEmpty, usage.subtracting([.copySource, .copyDestination, .textureBinding, .storageBinding, .renderAttachment]).isEmpty else {
            throw GPUError.validation("Invalid texture usage")
        }
        guard sampleCount == 1 || sampleCount == 4 else {
            throw GPUError.validation("Texture sample count must be 1 or 4")
        }
        if sampleCount > 1 {
            guard mipLevelCount == 1, arrayLayerCount == 1, usage.contains(.renderAttachment), !usage.contains(.storageBinding),
                  format.isColorAttachment || format.hasDepth else {
                throw GPUError.validation("Invalid multisampled texture format, mip levels or usage")
            }
        }
        guard !usage.contains(.storageBinding) || format.supportsStorage else {
            throw GPUError.validation("Invalid format for a storage texture")
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
        dimension ?? (texture.dimension == .d3 ? .d3 : texture.arrayLayerCount > 1 ? .d2Array : .d2)
    }

    package func arrayLayers(in texture: TextureDescriptor) throws -> Range<UInt32> {
        guard baseArrayLayer < texture.arrayLayerCount else {
            throw GPUError.validation("Invalid texture view base array layer")
        }
        let dimension = resolvedDimension(in: texture)
        guard (dimension == .d3) == (texture.dimension == .d3) else {
            throw GPUError.validation("Texture view dimension does not match the texture")
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
        texture.dimension == .d3 ? z..<(z + depth) : arrayLayer..<(arrayLayer + 1)
    }

    package func fits(in texture: TextureDescriptor) -> Bool {
        guard let extent = texture.mipExtent(mipLevel), x <= extent.width, y <= extent.height,
              arrayLayer < texture.arrayLayerCount else { return false }
        if texture.dimension == .d3 {
            let mipDepth = texture.mipDepth(mipLevel)
            guard z <= mipDepth, depth <= mipDepth - z else { return false }
        } else if z != 0 || depth != 1 {
            return false
        }
        return width <= extent.width - x && height <= extent.height - y
    }

    package func covers(_ texture: TextureDescriptor) -> Bool {
        guard let extent = texture.mipExtent(mipLevel) else { return false }
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

    package func rowStride(width: UInt32, texelBytes: UInt32 = 4) -> UInt64 {
        bytesPerRow.map(UInt64.init) ?? UInt64(width) * UInt64(texelBytes)
    }

    package func byteCount(width: UInt32, height: UInt32, texelBytes: UInt32 = 4, depth: UInt32 = 1) throws -> UInt64 {
        let rowBytes = UInt64(width) * UInt64(texelBytes)
        if let bytesPerRow {
            guard bytesPerRow % 256 == 0, UInt64(bytesPerRow) >= rowBytes else {
                throw GPUError.validation("Invalid texture copy row stride")
            }
        } else if height > 1 || depth > 1 {
            throw GPUError.validation("Texture copies with multiple rows or planes require a row stride")
        }
        if height == 0 || depth == 0 { return 0 }
        let rows = UInt64(height) * UInt64(depth) - 1
        let (padding, overflow) = rows.multipliedReportingOverflow(by: rowStride(width: width, texelBytes: texelBytes))
        guard !overflow, padding <= UInt64.max - rowBytes else { throw GPUError.validation("Texture copy byte count overflows") }
        return padding + rowBytes
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

    public init(topology: Topology = .triangleList, stripIndexFormat: IndexFormat? = nil,
                frontFace: FrontFace = .counterClockwise, cullMode: CullMode = .none) {
        self.topology = topology
        self.stripIndexFormat = stripIndexFormat
        self.frontFace = frontFace
        self.cullMode = cullMode
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

    package struct StorageTextureBinding: Sendable, Equatable {
        package let handle: HandleBinding
        package let requirement: StorageTextureRequirement

        package init(handle: HandleBinding, requirement: StorageTextureRequirement) {
            self.handle = handle
            self.requirement = requirement
        }
    }

    public let id: ResourceID
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
    package let sampledTextures: [SampledTextureBinding]
    package let storageTextures: [StorageTextureBinding]
    package let samplerBindings: [SamplerBinding]

    package init(id: ResourceID, colorFormat: TextureFormat, vertexBuffers: [VertexBufferLayout] = [],
                 depthState: RenderDepthState? = nil, sampleCount: UInt32 = 1, sampleMask: UInt32 = .max,
                 alphaToCoverageEnabled: Bool = false, writesColor: Bool = true,
                 primitive: RenderPrimitiveState = .init(), colorState: RenderColorState = .init(),
                 additionalColorTargets: [UInt32: RenderColorTarget] = [:], additionalColorOutputs: Set<UInt32> = [],
                 uniforms: [Uniform] = [], sampledTextures: [SampledTextureBinding] = [],
                 storageTextures: [StorageTextureBinding] = [], samplerBindings: [SamplerBinding] = []) {
        self.id = id
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
        self.sampledTextures = sampledTextures
        self.storageTextures = storageTextures
        self.samplerBindings = samplerBindings
    }
}
