// The subset of the Vulkan C ABI used by SAGAVulkan, declared in Swift so the package needs neither
// Vulkan headers nor an SDK to build. Structure layouts follow the Vulkan 1.1 specification.
//
// Dispatchable handles are pointer-sized (`UInt`), non-dispatchable handles are 64-bit (`UInt64`).

typealias VkResult = Int32

enum VK {
    static let success: VkResult = 0
    static let notReady: VkResult = 1
    static let timeout: VkResult = 2
    static let outOfHostMemory: VkResult = -1
    static let outOfDeviceMemory: VkResult = -2
    static let deviceLost: VkResult = -4

    static func version(_ major: UInt32, _ minor: UInt32, _ patch: UInt32 = 0) -> UInt32 {
        major << 22 | minor << 12 | patch
    }

    enum StructureType {
        static let applicationInfo: UInt32 = 0
        static let instanceCreateInfo: UInt32 = 1
        static let deviceQueueCreateInfo: UInt32 = 2
        static let deviceCreateInfo: UInt32 = 3
        static let submitInfo: UInt32 = 4
        static let memoryAllocateInfo: UInt32 = 5
        static let fenceCreateInfo: UInt32 = 8
        static let bufferCreateInfo: UInt32 = 12
        static let shaderModuleCreateInfo: UInt32 = 16
        static let pipelineShaderStageCreateInfo: UInt32 = 18
        static let computePipelineCreateInfo: UInt32 = 29
        static let pipelineLayoutCreateInfo: UInt32 = 30
        static let descriptorSetLayoutCreateInfo: UInt32 = 32
        static let descriptorPoolCreateInfo: UInt32 = 33
        static let descriptorSetAllocateInfo: UInt32 = 34
        static let writeDescriptorSet: UInt32 = 35
        static let commandPoolCreateInfo: UInt32 = 39
        static let commandBufferAllocateInfo: UInt32 = 40
        static let commandBufferBeginInfo: UInt32 = 42
        static let memoryBarrier: UInt32 = 46
    }

    enum BufferUsage {
        static let transferSource: UInt32 = 0x1
        static let transferDestination: UInt32 = 0x2
        static let uniformBuffer: UInt32 = 0x10
        static let storageBuffer: UInt32 = 0x20
        static let indirectBuffer: UInt32 = 0x100
        static let indexBuffer: UInt32 = 0x40
        static let vertexBuffer: UInt32 = 0x80
    }

    enum MemoryProperty {
        static let deviceLocal: UInt32 = 0x1
        static let hostVisible: UInt32 = 0x2
        static let hostCoherent: UInt32 = 0x4
        static let hostCached: UInt32 = 0x8
    }

    enum DescriptorType {
        static let sampler: UInt32 = 0
        static let sampledImage: UInt32 = 2
        static let storageImage: UInt32 = 3
        static let uniformBuffer: UInt32 = 6
        static let storageBuffer: UInt32 = 7
    }

    enum PhysicalDeviceType: UInt32 {
        case other = 0, integratedGPU, discreteGPU, virtualGPU, cpu
    }

    static let queueCompute: UInt32 = 0x2
    static let queueTransfer: UInt32 = 0x4
    static let shaderStageCompute: UInt32 = 0x20
    static let pipelineBindPointCompute: UInt32 = 1
    static let commandBufferLevelPrimary: UInt32 = 0
    static let commandBufferUsageOneTimeSubmit: UInt32 = 0x1
    static let pipelineStageTransfer: UInt32 = 0x1000
    static let pipelineStageComputeShader: UInt32 = 0x800
    static let pipelineStageHost: UInt32 = 0x4000
    static let pipelineStageAllCommands: UInt32 = 0x10000
    static let accessHostRead: UInt32 = 0x2000
    static let accessMemoryRead: UInt32 = 0x8000
    static let accessMemoryWrite: UInt32 = 0x10000
    static let sharingModeExclusive: UInt32 = 0
    static let wholeSize: UInt64 = .max
}

struct VkApplicationInfo {
    var sType: UInt32 = VK.StructureType.applicationInfo
    var pNext: UnsafeRawPointer?
    var pApplicationName: UnsafePointer<CChar>?
    var applicationVersion: UInt32 = 0
    var pEngineName: UnsafePointer<CChar>?
    var engineVersion: UInt32 = 0
    var apiVersion: UInt32 = 0
}

struct VkInstanceCreateInfo {
    var sType: UInt32 = VK.StructureType.instanceCreateInfo
    var pNext: UnsafeRawPointer?
    var flags: UInt32 = 0
    var pApplicationInfo: UnsafeRawPointer?
    var enabledLayerCount: UInt32 = 0
    var ppEnabledLayerNames: UnsafeRawPointer?
    var enabledExtensionCount: UInt32 = 0
    var ppEnabledExtensionNames: UnsafeRawPointer?
}

struct VkDeviceQueueCreateInfo {
    var sType: UInt32 = VK.StructureType.deviceQueueCreateInfo
    var pNext: UnsafeRawPointer?
    var flags: UInt32 = 0
    var queueFamilyIndex: UInt32 = 0
    var queueCount: UInt32 = 0
    var pQueuePriorities: UnsafePointer<Float>?
}

struct VkDeviceCreateInfo {
    var sType: UInt32 = VK.StructureType.deviceCreateInfo
    var pNext: UnsafeRawPointer?
    var flags: UInt32 = 0
    var queueCreateInfoCount: UInt32 = 0
    var pQueueCreateInfos: UnsafeRawPointer?
    var enabledLayerCount: UInt32 = 0
    var ppEnabledLayerNames: UnsafeRawPointer?
    var enabledExtensionCount: UInt32 = 0
    var ppEnabledExtensionNames: UnsafeRawPointer?
    var pEnabledFeatures: UnsafeRawPointer?
}

struct VkBufferCreateInfo {
    var sType: UInt32 = VK.StructureType.bufferCreateInfo
    var pNext: UnsafeRawPointer?
    var flags: UInt32 = 0
    var size: UInt64 = 0
    var usage: UInt32 = 0
    var sharingMode: UInt32 = VK.sharingModeExclusive
    var queueFamilyIndexCount: UInt32 = 0
    var pQueueFamilyIndices: UnsafePointer<UInt32>?
}

struct VkMemoryRequirements {
    var size: UInt64 = 0
    var alignment: UInt64 = 0
    var memoryTypeBits: UInt32 = 0
}

struct VkMemoryAllocateInfo {
    var sType: UInt32 = VK.StructureType.memoryAllocateInfo
    var pNext: UnsafeRawPointer?
    var allocationSize: UInt64 = 0
    var memoryTypeIndex: UInt32 = 0
}

struct VkShaderModuleCreateInfo {
    var sType: UInt32 = VK.StructureType.shaderModuleCreateInfo
    var pNext: UnsafeRawPointer?
    var flags: UInt32 = 0
    var codeSize: Int = 0
    var pCode: UnsafePointer<UInt32>?
}

struct VkDescriptorSetLayoutBinding {
    var binding: UInt32 = 0
    var descriptorType: UInt32 = 0
    var descriptorCount: UInt32 = 1
    var stageFlags: UInt32 = VK.shaderStageCompute
    var pImmutableSamplers: UnsafeRawPointer?
}

struct VkDescriptorSetLayoutCreateInfo {
    var sType: UInt32 = VK.StructureType.descriptorSetLayoutCreateInfo
    var pNext: UnsafeRawPointer?
    var flags: UInt32 = 0
    var bindingCount: UInt32 = 0
    var pBindings: UnsafePointer<VkDescriptorSetLayoutBinding>?
}

struct VkPipelineLayoutCreateInfo {
    var sType: UInt32 = VK.StructureType.pipelineLayoutCreateInfo
    var pNext: UnsafeRawPointer?
    var flags: UInt32 = 0
    var setLayoutCount: UInt32 = 0
    var pSetLayouts: UnsafePointer<UInt64>?
    var pushConstantRangeCount: UInt32 = 0
    var pPushConstantRanges: UnsafeRawPointer?
}

struct VkPipelineShaderStageCreateInfo {
    var sType: UInt32 = VK.StructureType.pipelineShaderStageCreateInfo
    var pNext: UnsafeRawPointer?
    var flags: UInt32 = 0
    var stage: UInt32 = VK.shaderStageCompute
    var module: UInt64 = 0
    var pName: UnsafePointer<CChar>?
    var pSpecializationInfo: UnsafeRawPointer?
}

struct VkComputePipelineCreateInfo {
    var sType: UInt32 = VK.StructureType.computePipelineCreateInfo
    var pNext: UnsafeRawPointer?
    var flags: UInt32 = 0
    var stage = VkPipelineShaderStageCreateInfo()
    var layout: UInt64 = 0
    var basePipelineHandle: UInt64 = 0
    var basePipelineIndex: Int32 = -1
}

struct VkDescriptorPoolSize {
    var type: UInt32 = 0
    var descriptorCount: UInt32 = 0
}

struct VkDescriptorPoolCreateInfo {
    var sType: UInt32 = VK.StructureType.descriptorPoolCreateInfo
    var pNext: UnsafeRawPointer?
    var flags: UInt32 = 0
    var maxSets: UInt32 = 0
    var poolSizeCount: UInt32 = 0
    var pPoolSizes: UnsafePointer<VkDescriptorPoolSize>?
}

struct VkDescriptorSetAllocateInfo {
    var sType: UInt32 = VK.StructureType.descriptorSetAllocateInfo
    var pNext: UnsafeRawPointer?
    var descriptorPool: UInt64 = 0
    var descriptorSetCount: UInt32 = 0
    var pSetLayouts: UnsafePointer<UInt64>?
}

struct VkDescriptorBufferInfo {
    var buffer: UInt64 = 0
    var offset: UInt64 = 0
    var range: UInt64 = 0
}

struct VkWriteDescriptorSet {
    var sType: UInt32 = VK.StructureType.writeDescriptorSet
    var pNext: UnsafeRawPointer?
    var dstSet: UInt64 = 0
    var dstBinding: UInt32 = 0
    var dstArrayElement: UInt32 = 0
    var descriptorCount: UInt32 = 1
    var descriptorType: UInt32 = 0
    var pImageInfo: UnsafeRawPointer?
    var pBufferInfo: UnsafePointer<VkDescriptorBufferInfo>?
    var pTexelBufferView: UnsafeRawPointer?
}

struct VkCommandPoolCreateInfo {
    var sType: UInt32 = VK.StructureType.commandPoolCreateInfo
    var pNext: UnsafeRawPointer?
    var flags: UInt32 = 0
    var queueFamilyIndex: UInt32 = 0
}

struct VkCommandBufferAllocateInfo {
    var sType: UInt32 = VK.StructureType.commandBufferAllocateInfo
    var pNext: UnsafeRawPointer?
    var commandPool: UInt64 = 0
    var level: UInt32 = VK.commandBufferLevelPrimary
    var commandBufferCount: UInt32 = 1
}

struct VkCommandBufferBeginInfo {
    var sType: UInt32 = VK.StructureType.commandBufferBeginInfo
    var pNext: UnsafeRawPointer?
    var flags: UInt32 = VK.commandBufferUsageOneTimeSubmit
    var pInheritanceInfo: UnsafeRawPointer?
}

struct VkBufferCopy {
    var srcOffset: UInt64 = 0
    var dstOffset: UInt64 = 0
    var size: UInt64 = 0
}

struct VkMemoryBarrier {
    var sType: UInt32 = VK.StructureType.memoryBarrier
    var pNext: UnsafeRawPointer?
    var srcAccessMask: UInt32 = 0
    var dstAccessMask: UInt32 = 0
}

struct VkSubmitInfo {
    var sType: UInt32 = VK.StructureType.submitInfo
    var pNext: UnsafeRawPointer?
    var waitSemaphoreCount: UInt32 = 0
    var pWaitSemaphores: UnsafeRawPointer?
    var pWaitDstStageMask: UnsafeRawPointer?
    var commandBufferCount: UInt32 = 0
    var pCommandBuffers: UnsafePointer<UInt>?
    var signalSemaphoreCount: UInt32 = 0
    var pSignalSemaphores: UnsafeRawPointer?
}

struct VkFenceCreateInfo {
    var sType: UInt32 = VK.StructureType.fenceCreateInfo
    var pNext: UnsafeRawPointer?
    var flags: UInt32 = 0
}

struct VkQueueFamilyProperties {
    var queueFlags: UInt32 = 0
    var queueCount: UInt32 = 0
    var timestampValidBits: UInt32 = 0
    var minImageTransferGranularityWidth: UInt32 = 0
    var minImageTransferGranularityHeight: UInt32 = 0
    var minImageTransferGranularityDepth: UInt32 = 0
}

struct VkPushConstantRange {
    var stageFlags: UInt32 = VK.shaderStageCompute
    var offset: UInt32 = 0
    var size: UInt32 = 0
}
