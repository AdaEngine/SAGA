#if os(Windows)
import SAGATypes
import SAGABackend

// Buffer mapping. Buffers live in default-heap memory, so a read mapping holds a snapshot copied back from the GPU
// and a write mapping collects the written ranges and uploads them when the mapping ends.

extension D3D12Device: BufferMappingBackendDevice {
    struct Mapping {
        var mode: BufferMapMode
        var snapshot: [UInt8] = []
        var writes: [(offset: UInt64, bytes: [UInt8])] = []
    }

    public func beginBufferMapping(id: ResourceID, mode: BufferMapMode, offset: UInt64, size: UInt64) async throws {
        try requireActive()
        guard let buffer = buffers[id], mappings[id] == nil, offset <= buffer.descriptor.size,
              size <= buffer.descriptor.size - offset else {
            throw GPUError.validation("Invalid Direct3D 12 buffer mapping")
        }
        var mapping = Mapping(mode: mode)
        if mode == .read {
            let length = offset + size
            if length > 0 {
                let words = try await runCommands(Submission(sequence: 0, resources: [id]),
                                                  commands: [.read(buffer: id, byteCount: length)], validate: false)
                mapping.snapshot = words[0].flatMap { word in
                    [UInt8(truncatingIfNeeded: word), UInt8(truncatingIfNeeded: word >> 8),
                     UInt8(truncatingIfNeeded: word >> 16), UInt8(truncatingIfNeeded: word >> 24)]
                }
            }
        }
        mappings[id] = mapping
    }

    public func readMappedBytes(id: ResourceID, offset: UInt64, size: UInt64) async throws -> [UInt8] {
        try requireActive()
        guard let mapping = mappings[id], mapping.mode == .read, offset + size <= UInt64(mapping.snapshot.count) else {
            throw GPUError.validation("Invalid Direct3D 12 mapped read")
        }
        return Array(mapping.snapshot[Int(offset)..<Int(offset + size)])
    }

    public func writeMappedBytes(id: ResourceID, offset: UInt64, bytes: [UInt8]) async throws {
        try requireActive()
        guard let buffer = buffers[id], var mapping = mappings[id], mapping.mode == .write,
              offset <= buffer.descriptor.size, UInt64(bytes.count) <= buffer.descriptor.size - offset else {
            throw GPUError.validation("Invalid Direct3D 12 mapped write")
        }
        mapping.writes.append((offset, bytes))
        mappings[id] = mapping
    }

    public func endBufferMapping(id: ResourceID) async throws {
        guard let mapping = mappings.removeValue(forKey: id) else { return }
        guard mapping.mode == .write, !mapping.writes.isEmpty else { return }
        let commands = mapping.writes.map {
            BufferCommand.writeBytes(buffer: id, offset: $0.offset, bytes: $0.bytes, dataOffset: 0, byteCount: nil)
        }
        _ = try await runCommands(Submission(sequence: 0, resources: [id]), commands: commands, validate: false)
    }
}
#endif
