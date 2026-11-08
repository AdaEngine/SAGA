public enum BufferMapMode: UInt32, Sendable, Equatable {
    case read = 1
    case write = 2
}

public enum BufferMapState: Sendable, Equatable {
    case unmapped
    case pending
    case mapped
}

/// Identifies a single mapping of a buffer. Unmapping invalidates the handle.
public struct BufferMapping: Sendable, Equatable {
    public let buffer: ResourceID
    public let mode: BufferMapMode
    public let offset: UInt64
    public let size: UInt64
    package let generation: UInt64

    package init(buffer: ResourceID, mode: BufferMapMode, offset: UInt64, size: UInt64, generation: UInt64) {
        self.buffer = buffer
        self.mode = mode
        self.offset = offset
        self.size = size
        self.generation = generation
    }

    package func accessSize(offset: UInt64, size: UInt64?) throws -> UInt64 {
        let end = self.offset + self.size
        guard offset >= self.offset, offset <= end else { throw GPUError.validation("Mapped access is outside the range") }
        let length = size ?? (end - offset)
        guard length <= end - offset, Int(exactly: length) != nil else {
            throw GPUError.validation("Mapped access is outside the range")
        }
        return length
    }
}
