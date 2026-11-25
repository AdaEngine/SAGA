#if os(Windows) || os(Linux) || os(Android)
import SAGACore
import SAGATypes
@testable import SAGAVulkan
import Testing

private extension VulkanDevice {
    func mappedBufferHandle(_ id: ResourceID) -> UInt64? {
        buffers[id]?.native.handle
    }
}

@Test(.enabled(if: VulkanBackend.requestAdapter() != nil), .gpuExclusive)
func vulkanUploadOnlyJobsWriteMappedMemoryInOrder() async throws {
    let backend = try VulkanBackend.makeDevice()
    let device = try await Device.make(backend: backend)
    do {
        let source = try await device.createBuffer(.init(size: 8, usage: [.copySource, .copyDestination]))
        let snapshot = try await device.createBuffer(.init(size: 8, usage: [.copyDestination, .mapRead]))
        _ = try await device.submitCommands([
            .writeBytes(buffer: source, offset: 0, bytes: [1, 0, 0, 0, 2, 0, 0, 0]),
            .writeBytes(buffer: source, offset: 4, bytes: [3, 0, 0, 0]),
        ]).wait()
        #expect(await backend.idleFences.isEmpty)

        // Interleaved writes and copies still run through the GPU and preserve each snapshot.
        let result = try await device.submitCommands([
            .copy(source: source, destination: snapshot),
            .read(buffer: snapshot, byteCount: 8),
            .writeBytes(buffer: source, offset: 0, bytes: [4, 0, 0, 0]),
            .copy(source: source, destination: snapshot),
            .read(buffer: snapshot, byteCount: 8),
        ]).wait()
        #expect(result == [[1, 3], [4, 3]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}

@Test(.enabled(if: VulkanBackend.requestAdapter() != nil), .gpuExclusive)
func vulkanReusesRetiredBufferStorageAndClearsItForTheNewOwner() async throws {
    let backend = try VulkanBackend.makeDevice()
    let device = try await Device.make(backend: backend)
    do {
        let descriptor = BufferDescriptor(size: 8, usage: [.copySource, .copyDestination])
        let old = try await device.createBuffer(descriptor)
        let handle = try #require(await backend.mappedBufferHandle(old))
        _ = try await device.writeBuffer(old, bytes: [1, 2, 3, 4, 5, 6, 7, 8]).wait()
        try await device.destroyBuffer(old)

        let fresh = try await device.createBuffer(descriptor)
        #expect(old != fresh)
        #expect(await backend.mappedBufferHandle(fresh) == handle)
        let readback = try await device.createBuffer(.init(size: 8, usage: [.copyDestination, .mapRead]))
        let bytes = try await device.submitCommands([
            .copy(source: fresh, destination: readback), .read(buffer: readback, byteCount: 8),
        ]).wait()
        #expect(bytes == [[0, 0]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
