import Testing
@testable import SAGAVulkan

@Suite("Vulkan extension requests", .serialized)
struct VulkanExtensionRequestTests {
    @Test func acceptsOnlyAvailableExtensionsInRequestOrderWithoutDuplicates() {
        let accepted = VulkanExtensionRequest.accepted(
            ["VK_A", "VK_MISSING", "VK_B", "VK_A"], available: ["VK_B", "VK_A"], alreadyEnabled: []
        )
        #expect(accepted.enabled == ["VK_A", "VK_B"])
        #expect(accepted.unavailable == ["VK_MISSING"])
    }

    @Test func skipsExtensionsTheDeviceAlreadyEnables() {
        let accepted = VulkanExtensionRequest.accepted(["VK_A", "VK_B"], available: ["VK_A", "VK_B"], alreadyEnabled: ["VK_A"])
        #expect(accepted.enabled == ["VK_B"])
        #expect(accepted.unavailable.isEmpty)
    }

    @Test func nothingRequestedChangesNothing() {
        VulkanExtensionRequest.set(instance: [], device: [])
        let accepted = VulkanExtensionRequest.accepted([], available: ["VK_A"], alreadyEnabled: [])
        #expect(accepted.enabled.isEmpty && accepted.unavailable.isEmpty)
        #expect(VulkanExtensionRequest.requestedInstance.isEmpty && VulkanExtensionRequest.requestedDevice.isEmpty)
    }

    @Test func requestsAreStoredUntilReplaced() {
        VulkanExtensionRequest.set(instance: ["VK_I"], device: ["VK_D"])
        #expect(VulkanExtensionRequest.requestedInstance == ["VK_I"] && VulkanExtensionRequest.requestedDevice == ["VK_D"])
        VulkanExtensionRequest.set(instance: [], device: [])
    }
}
