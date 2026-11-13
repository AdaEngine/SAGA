import Foundation
import Testing
import SAGATypes
import SAGABackend
import SAGACore
@testable import SAGAVulkan

// These tests execute on a real Vulkan driver. Without one they are reported as skipped, never as passed.
private let vulkanAvailable = VulkanBackend.requestAdapter() != nil

private final class NativePassProbe: @unchecked Sendable {
    private let lock = NSLock()
    private var calls = 0
    private var handles: (commandBuffer: UInt, device: UInt)?

    func record(commandBuffer: UInt, device: UInt) {
        lock.lock(); defer { lock.unlock() }
        calls += 1
        handles = (commandBuffer, device)
    }

    var snapshot: (calls: Int, commandBuffer: UInt, device: UInt) {
        lock.lock(); defer { lock.unlock() }
        return (calls, handles?.commandBuffer ?? 0, handles?.device ?? 0)
    }
}

private typealias FillBuffer = @convention(c) (UInt, UInt64, UInt64, UInt64, UInt32) -> Void

@Test(.enabled(if: vulkanAvailable), .gpuExclusive) func nativePassRecordsInOrderAndItsEffectsAreVisibleToLaterCommands() async throws {
    let backend = try VulkanBackend.makeDevice()
    let device = try await Device.make(backend: backend)
    do {
        let target = try await device.createBuffer(.init(size: 16, usage: [.copyDestination, .copySource]))
        let readback = try await device.createBuffer(.init(size: 16, usage: [.copyDestination, .mapRead]))
        let probe = NativePassProbe()
        let pass = NativePass(label: "fill", uses: [.init(target, .write)]) { context in
            guard let vulkan = context as? VulkanNativePassContext else { throw GPUError.validation("Expected a Vulkan context") }
            probe.record(commandBuffer: vulkan.commandBuffer, device: vulkan.device)
            let fill = try #require(vulkan.deviceProcedure("vkCmdFillBuffer"))
            let call = unsafeBitCast(fill, to: FillBuffer.self)
            call(vulkan.commandBuffer, try vulkan.buffer(target), 0, 16, 7)
        }
        let ticket = try await device.submitCommands([
            .write(buffer: target, offset: 0, words: [1, 2, 3, 4]),
            .native(pass),
            .copy(source: target, destination: readback, byteCount: 16),
            .read(buffer: readback, byteCount: 16),
        ])
        #expect(try await ticket.wait() == [[7, 7, 7, 7]])
        let seen = probe.snapshot
        #expect(seen.calls == 1)
        #expect(seen.commandBuffer != 0 && seen.device != 0)
        await device.close()
    } catch { await device.close(); throw error }
}

@Test(.enabled(if: vulkanAvailable), .gpuExclusive) func nativePassWithAnUnknownResourceIsRejectedAndTheQueueRemainsUsable() async throws {
    let backend = try VulkanBackend.makeDevice()
    let device = try await Device.make(backend: backend)
    do {
        let buffer = try await device.createBuffer(.init(size: 16, usage: [.copyDestination, .mapRead]))
        let probe = NativePassProbe()
        let stranger = ResourceID(device: 0, serial: 999_999)
        let pass = NativePass(label: "invalid", uses: [.init(stranger, .write)]) { context in
            probe.record(commandBuffer: 0, device: 0)
        }
        await #expect(throws: GPUError.self) {
            _ = try await device.submitCommands([.native(pass)])
        }
        #expect(probe.snapshot.calls == 0)
        let ticket = try await device.submitCommands([.write(buffer: buffer, offset: 0, words: [9]), .read(buffer: buffer, byteCount: 4)])
        #expect(try await ticket.wait() == [[9]])
        await device.close()
    } catch { await device.close(); throw error }
}
