public enum GPUError: Error, Sendable, Equatable {
    case adapterUnavailable
    case timeout
    case validation(String)
    case outOfMemory
    case deviceLost(String)
    case unsupported(String)
    case backendFailure(String)
}

public enum GPUFeature: String, Sendable, Hashable, CaseIterable {
    case shaderF16, timestampQuery, storageTextureRead, storageTextureReadWrite, uniformBufferStandardLayout
    case textureCompressionBC, textureCompressionETC2, textureCompressionASTC, textureCompressionASTCHDR
    case textureCompressionBCSliced3D, textureCompressionASTCSliced3D
    case shaderEarlyDepthTest, float32Filterable, float32Blendable
    case halfFloatStorageTextures, r16FloatStorageReadWrite
    case narrowInteger16StorageTextures, integer16StorageReadWrite
    case normalized16Textures, normalized16Filterable, normalized16Resolve
    case packedColorStorageTextures, rg11b10UfloatRenderable, rgb9e5Renderable
    case narrow8BitStorageTextures, storage8BitReadWrite, snorm8Renderable
    case depth32FloatStencil8
    case textureAtomic, textureInt64Atomic
    case shaderInt64, shaderInt64AtomicMinMax, shaderInt64AtomicAllOps
    case linearIndexing
    case bgra8UnormStorage, wideRGBAStorageReadWrite
    case indirectFirstInstance, clipDistances, primitiveIndex, depthClipControl, subgroups, subgroupID, dualSourceBlending
    case subgroupSizeControl, subgroupUniformity, shaderBarycentrics
}

public struct DeviceCapabilities: Sendable, Equatable {
    public let maxBufferSize: UInt64
    public let maxTextureDimension2D: UInt32
    public let maxImmediateSize: UInt32
    /// Maximum rounded workgroup storage for one compute pipeline entry point.
    public let maxComputeWorkgroupStorageSize: UInt32
    /// Advertised subgroup-size limits when the backend provides them.
    public let subgroupSizeRange: ClosedRange<UInt32>?
    public let features: Set<GPUFeature>
    /// Float32 formats that the backend supports as 4× color attachments.
    public let multisampledFloat32Formats: Set<TextureFormat>
    /// Multiply a difference between two timestamp query results by this value to obtain nanoseconds.
    public let timestampPeriodNanoseconds: Double

    public init(maxBufferSize: UInt64, features: Set<GPUFeature> = [], maxTextureDimension2D: UInt32 = 8192,
                timestampPeriodNanoseconds: Double = 1, multisampledFloat32Formats: Set<TextureFormat> = [],
                maxImmediateSize: UInt32 = 0, subgroupSizeRange: ClosedRange<UInt32>? = nil,
                maxComputeWorkgroupStorageSize: UInt32 = 16 * 1024) throws {
        guard maxBufferSize > 0 else { throw GPUError.validation("maxBufferSize must be positive") }
        guard maxTextureDimension2D > 0 else { throw GPUError.validation("maxTextureDimension2D must be positive") }
        guard maxImmediateSize.isMultiple(of: 4) else { throw GPUError.validation("maxImmediateSize must be word-aligned") }
        if let subgroupSizeRange, subgroupSizeRange.lowerBound == 0 {
            throw GPUError.validation("Subgroup sizes must be positive")
        }
        guard timestampPeriodNanoseconds.isFinite, timestampPeriodNanoseconds > 0 else {
            throw GPUError.validation("Timestamp period must be finite and positive")
        }
        guard multisampledFloat32Formats.allSatisfy(\.isFloat32Color) else {
            throw GPUError.validation("Float32 multisampling capabilities require float32 color formats")
        }
        self.maxBufferSize = maxBufferSize
        self.maxTextureDimension2D = maxTextureDimension2D
        self.maxImmediateSize = maxImmediateSize
        self.maxComputeWorkgroupStorageSize = maxComputeWorkgroupStorageSize
        self.subgroupSizeRange = subgroupSizeRange
        self.features = features
        self.multisampledFloat32Formats = multisampledFloat32Formats
        self.timestampPeriodNanoseconds = timestampPeriodNanoseconds
    }

    public func require(_ requested: Set<GPUFeature>) throws {
        guard requested.isSubset(of: features) else { throw GPUError.validation("Feature not advertised") }
    }

    package func requireShaderInt64Atomics(minMax: Bool, allOps: Bool) throws {
        if allOps {
            try require([.shaderInt64AtomicAllOps])
        } else if minMax && !features.contains(.shaderInt64AtomicAllOps) {
            try require([.shaderInt64AtomicMinMax])
        }
    }

    package func requireImmediateSize(_ byteSize: UInt64) throws {
        guard byteSize <= UInt64(maxImmediateSize) else { throw GPUError.validation("Immediate data exceeds the device limit") }
    }

    package func requireWorkgroupStorageSize(_ byteSize: UInt64) throws {
        guard byteSize <= UInt64(maxComputeWorkgroupStorageSize) else {
            throw GPUError.validation("Workgroup storage exceeds the device limit")
        }
    }

    package func requireSubgroupSize(_ size: UInt32?) throws {
        guard let size else { return }
        try require([.subgroupSizeControl])
        if let subgroupSizeRange, !subgroupSizeRange.contains(size) {
            throw GPUError.validation("Requested subgroup size is outside the device range")
        }
    }
}

public struct BufferUsage: OptionSet, Sendable, Hashable {
    public let rawValue: UInt32
    public init(rawValue: UInt32) { self.rawValue = rawValue }
    public static let mapRead = Self(rawValue: 1 << 0)
    public static let mapWrite = Self(rawValue: 1 << 1)
    public static let copySource = Self(rawValue: 1 << 2)
    public static let copyDestination = Self(rawValue: 1 << 3)
    public static let index = Self(rawValue: 1 << 4)
    public static let vertex = Self(rawValue: 1 << 5)
    public static let uniform = Self(rawValue: 1 << 6)
    public static let storage = Self(rawValue: 1 << 7)
    public static let indirect = Self(rawValue: 1 << 8)
    public static let queryResolve = Self(rawValue: 1 << 9)
    public static let all: Self = [.mapRead, .mapWrite, .copySource, .copyDestination,
                                  .index, .vertex, .uniform, .storage, .indirect, .queryResolve]
}

public struct BufferDescriptor: Sendable, Equatable {
    public let label: String
    public let size: UInt64
    public let usage: BufferUsage
    public let mappedAtCreation: Bool

    public init(label: String = "", size: UInt64, usage: BufferUsage, mappedAtCreation: Bool = false) {
        self.label = label
        self.size = size
        self.usage = usage
        self.mappedAtCreation = mappedAtCreation
    }

    public func validate(against capabilities: DeviceCapabilities) throws {
        guard !usage.isEmpty, usage.subtracting(.all).isEmpty else {
            throw GPUError.validation("Buffer usage must contain only known nonzero flags")
        }
        if usage.contains(.mapRead), !usage.subtracting([.mapRead, .copyDestination]).isEmpty {
            throw GPUError.validation("MAP_READ permits only COPY_DST")
        }
        if usage.contains(.mapWrite), !usage.subtracting([.mapWrite, .copySource]).isEmpty {
            throw GPUError.validation("MAP_WRITE permits only COPY_SRC")
        }
        guard size <= capabilities.maxBufferSize else { throw GPUError.validation("Buffer exceeds maxBufferSize") }
        guard !mappedAtCreation || size % 4 == 0 else { throw GPUError.validation("Mapped size must be aligned to four bytes") }
    }
}

public struct ResourceID: Hashable, Sendable {
    public let device: UInt64
    public let serial: UInt64
    public init(device: UInt64, serial: UInt64) { self.device = device; self.serial = serial }
}

public struct Submission: Sendable, Equatable {
    public let sequence: UInt64
    public let resources: [ResourceID]
    public init(sequence: UInt64, resources: [ResourceID]) { self.sequence = sequence; self.resources = resources }
}

public enum ErrorFilter: Sendable {
    case validation, outOfMemory, internalError
    public func matches(_ error: GPUError) -> Bool {
        switch (self, error) {
        case (.validation, .validation), (.outOfMemory, .outOfMemory), (.internalError, .backendFailure): true
        default: false
        }
    }
}
