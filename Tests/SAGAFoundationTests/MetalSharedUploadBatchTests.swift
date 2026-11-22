#if canImport(Metal)
import SAGACore
import SAGAMetal
import SAGATypes
import Testing

@Test
func metalSharedUploadBatchPreservesOverlappingWriteOrder() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let first = try await device.createBuffer(.init(size: 8, usage: [.copyDestination, .copySource]))
        let second = try await device.createBuffer(.init(size: 4, usage: [.copyDestination, .copySource]))
        let result = try await device.createBuffer(.init(size: 12, usage: [.copyDestination, .mapRead]))
        let uploads = try await device.submitCommands([
            .writeBytes(buffer: first, offset: 0, bytes: [1, 2, 3, 4, 5, 6, 7, 8]),
            .writeBytes(buffer: second, offset: 0, bytes: [13, 14, 15, 16]),
            .writeBytes(buffer: first, offset: 4, bytes: [0, 9, 10, 11, 12, 0], dataOffset: 1, byteCount: 4),
        ])
        let read = try await device.submitCommands([
            .copy(source: first, destination: result, byteCount: 8),
            .copy(source: second, destination: result, destinationOffset: 8, byteCount: 4),
            .read(buffer: result, byteCount: 12),
        ])
        #expect(try await read.wait() == [[0x04030201, 0x0C0B0A09, 0x100F0E0D]])
        _ = try await uploads.wait()
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
