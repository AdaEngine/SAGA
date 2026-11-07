public enum TextureFormat: UInt32, Sendable, Equatable {
    case rgba8Unorm = 1
    case bgra8Unorm = 2
}

public struct TextureUsage: OptionSet, Sendable {
    public let rawValue: UInt32
    public init(rawValue: UInt32) { self.rawValue = rawValue }
    public static let copySource = Self(rawValue: 1 << 0)
    public static let renderAttachment = Self(rawValue: 1 << 4)
}

public struct TextureDescriptor: Sendable, Equatable {
    public let width: UInt32
    public let height: UInt32
    public let format: TextureFormat
    public let usage: TextureUsage

    public init(width: UInt32, height: UInt32, format: TextureFormat, usage: TextureUsage) {
        self.width = width
        self.height = height
        self.format = format
        self.usage = usage
    }

    public func validate(against capabilities: DeviceCapabilities) throws {
        guard width > 0, height > 0, width <= capabilities.maxTextureDimension2D,
              height <= capabilities.maxTextureDimension2D else {
            throw GPUError.validation("Invalid 2D texture extent")
        }
        guard !usage.isEmpty, usage.subtracting([.copySource, .renderAttachment]).isEmpty else {
            throw GPUError.validation("Invalid texture usage")
        }
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

    package var isFinite: Bool { red.isFinite && green.isFinite && blue.isFinite && alpha.isFinite }
}

public struct RenderPipeline: Sendable, Equatable {
    public let id: ResourceID
    public let colorFormat: TextureFormat
    public let vertexBuffers: [VertexBufferLayout]

    package init(id: ResourceID, colorFormat: TextureFormat, vertexBuffers: [VertexBufferLayout] = []) {
        self.id = id
        self.colorFormat = colorFormat
        self.vertexBuffers = vertexBuffers
    }
}
