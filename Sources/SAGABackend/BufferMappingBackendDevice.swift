import SAGATypes

/// Trusted mapping SPI. The device serializes accesses per buffer and waits for prior GPU use.
/// A failed begin must leave the buffer unmapped. End is idempotent and publishes mapped writes.
/// A buffer created mapped is ready for a write mapping when allocation completes.
public protocol BufferMappingBackendDevice: BackendDevice {
    func beginBufferMapping(id: ResourceID, mode: BufferMapMode, offset: UInt64, size: UInt64) async throws
    func readMappedBytes(id: ResourceID, offset: UInt64, size: UInt64) async throws -> [UInt8]
    func writeMappedBytes(id: ResourceID, offset: UInt64, bytes: [UInt8]) async throws
    func endBufferMapping(id: ResourceID) async throws
}
