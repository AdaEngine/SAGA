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
    case shaderF16, timestampQuery, storageTextureRead, storageTextureReadWrite
}

public struct DeviceCapabilities: Sendable, Equatable {
    public let maxBufferSize: UInt64
    public let maxTextureDimension2D: UInt32
    public let features: Set<GPUFeature>

    public init(maxBufferSize: UInt64, features: Set<GPUFeature> = [], maxTextureDimension2D: UInt32 = 8192) throws {
        guard maxBufferSize > 0 else { throw GPUError.validation("maxBufferSize must be positive") }
        guard maxTextureDimension2D > 0 else { throw GPUError.validation("maxTextureDimension2D must be positive") }
        self.maxBufferSize = maxBufferSize
        self.maxTextureDimension2D = maxTextureDimension2D
        self.features = features
    }

    public func require(_ requested: Set<GPUFeature>) throws {
        guard requested.isSubset(of: features) else { throw GPUError.validation("Feature not advertised") }
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
