public enum IndexFormat: UInt32, Sendable, Equatable {
    case uint16 = 2
    case uint32 = 4

    public var byteSize: UInt32 { rawValue }
}

public struct IndexBufferBinding: Sendable, Equatable {
    public let buffer: ResourceID
    public let format: IndexFormat
    public let offset: UInt64
    public let size: UInt64?

    public init(buffer: ResourceID, format: IndexFormat, offset: UInt64 = 0, size: UInt64? = nil) {
        self.buffer = buffer
        self.format = format
        self.offset = offset
        self.size = size
    }
}

public struct IndexedDraw: Sendable, Equatable {
    public let indexCount: UInt32
    public let firstIndex: UInt32
    public let baseVertex: Int32

    public init(indexCount: UInt32, firstIndex: UInt32 = 0, baseVertex: Int32 = 0) {
        self.indexCount = indexCount
        self.firstIndex = firstIndex
        self.baseVertex = baseVertex
    }
}
