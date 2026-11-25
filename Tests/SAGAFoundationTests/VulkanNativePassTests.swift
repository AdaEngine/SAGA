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

@Test func latestSurfaceFrameUsesOnlySupportedModesAndImageCounts() {
    var info = VkSwapchainCreateInfoKHR()
    let fifo = VKPresent.preferredMode([2], fifoLatestReadyEnabled: true)
    let mailbox = VKPresent.preferredMode([2, 1], fifoLatestReadyEnabled: true)
    info.configurePresentation(mode: fifo, minimumImages: 1, maximumImages: 0)
    #expect(info.presentMode == 2 && info.minImageCount == 2)
    info.configurePresentation(mode: mailbox, minimumImages: 2, maximumImages: 0)
    #expect(info.presentMode == 1 && info.minImageCount == 3)
    info.configurePresentation(mode: mailbox, minimumImages: 1, maximumImages: 2)
    #expect(info.presentMode == 1 && info.minImageCount == 2)
    info.configurePresentation(mode: VKPresent.preferredMode([], fifoLatestReadyEnabled: true), minimumImages: 4, maximumImages: 5)
    #expect(info.presentMode == 2 && info.minImageCount == 4)
}

@Test func fifoLatestReadyTakesPriorityOverMailboxWhenAvailable() {
    // VkPresentModeKHR::VK_PRESENT_MODE_FIFO_LATEST_READY_KHR.
    let latestReady: UInt32 = 1_000_361_000
    var info = VkSwapchainCreateInfoKHR()
    let supported = [2, 1, latestReady]
    let preferred = VKPresent.preferredMode(supported, fifoLatestReadyEnabled: true)
    info.configurePresentation(mode: preferred, minimumImages: 2, maximumImages: 0)
    #expect(info.presentMode == latestReady && info.minImageCount == 3)
    info.configurePresentation(mode: preferred, minimumImages: 1, maximumImages: 2)
    #expect(info.presentMode == latestReady && info.minImageCount == 2)
    #expect(VKPresent.preferredMode(supported, fifoLatestReadyEnabled: false) == VKPresent.mailbox)
}

@Test func fifoLatestReadyUsesTheAvailableExtensionName() {
    let khr = "VK_KHR_present_mode_fifo_latest_ready"
    let ext = "VK_EXT_present_mode_fifo_latest_ready"
    #expect(VKPresent.fifoLatestReadyExtension(in: [khr, ext]) == khr)
    #expect(VKPresent.fifoLatestReadyExtension(in: [ext]) == ext)
    #expect(VKPresent.fifoLatestReadyExtension(in: ["VK_KHR_swapchain"]) == nil)
}
