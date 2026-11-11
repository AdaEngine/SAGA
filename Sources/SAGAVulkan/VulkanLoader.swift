import SAGATypes

#if os(Windows)
import WinSDK
#elseif canImport(Glibc)
import Glibc
#elseif canImport(Musl)
import Musl
#elseif canImport(Android)
import Android
#elseif canImport(Darwin)
import Darwin
#endif

/// The system Vulkan loader. The library stays loaded for the life of the process because
/// instances and devices may outlive any single owner.
struct VulkanLibrary: Sendable {
    private let handle: UInt

    static func open() -> VulkanLibrary? {
        #if os(Windows)
        guard let module = LoadLibraryA("vulkan-1.dll") else { return nil }
        return VulkanLibrary(handle: UInt(bitPattern: module))
        #else
        #if os(Android)
        let names = ["libvulkan.so"]
        #elseif canImport(Darwin)
        let names = ["libvulkan.1.dylib", "libvulkan.dylib", "libMoltenVK.dylib"]
        #else
        let names = ["libvulkan.so.1", "libvulkan.so"]
        #endif
        for name in names {
            if let module = dlopen(name, RTLD_NOW | RTLD_LOCAL) {
                return VulkanLibrary(handle: UInt(bitPattern: module))
            }
        }
        return nil
        #endif
    }

    func symbol(_ name: String) -> UnsafeRawPointer? {
        #if os(Windows)
        guard let module = HMODULE(bitPattern: handle), let address = GetProcAddress(module, name) else { return nil }
        return unsafeBitCast(address, to: UnsafeRawPointer.self)
        #else
        guard let module = UnsafeMutableRawPointer(bitPattern: handle), let address = dlsym(module, name) else { return nil }
        return UnsafeRawPointer(address)
        #endif
    }
}

typealias GetInstanceProcAddr = @convention(c) (UInt, UnsafePointer<CChar>) -> UnsafeRawPointer?
typealias CreateInstance = @convention(c) (UnsafeRawPointer, UnsafeRawPointer?, UnsafeMutablePointer<UInt>) -> VkResult

/// Instance and device entry points, resolved once through `vkGetInstanceProcAddr`.
struct VulkanFunctions: Sendable {
    let destroyInstance: @convention(c) (UInt, UnsafeRawPointer?) -> Void
    let enumeratePhysicalDevices: @convention(c) (UInt, UnsafeMutablePointer<UInt32>, UnsafeMutablePointer<UInt>?) -> VkResult
    let getPhysicalDeviceProperties: @convention(c) (UInt, UnsafeMutableRawPointer) -> Void
    let getPhysicalDeviceMemoryProperties: @convention(c) (UInt, UnsafeMutableRawPointer) -> Void
    let getPhysicalDeviceFormatProperties: @convention(c) (UInt, UInt32, UnsafeMutableRawPointer) -> Void
    let getPhysicalDeviceFeatures2: @convention(c) (UInt, UnsafeMutableRawPointer) -> Void
    let getPhysicalDeviceProperties2: @convention(c) (UInt, UnsafeMutableRawPointer) -> Void
    let getPhysicalDeviceQueueFamilyProperties:
        @convention(c) (UInt, UnsafeMutablePointer<UInt32>, UnsafeMutableRawPointer?) -> Void
    let createDevice: @convention(c) (UInt, UnsafeRawPointer, UnsafeRawPointer?, UnsafeMutablePointer<UInt>) -> VkResult
    let destroyDevice: @convention(c) (UInt, UnsafeRawPointer?) -> Void
    let getDeviceQueue: @convention(c) (UInt, UInt32, UInt32, UnsafeMutablePointer<UInt>) -> Void
    let deviceWaitIdle: @convention(c) (UInt) -> VkResult

    let createBuffer: @convention(c) (UInt, UnsafeRawPointer, UnsafeRawPointer?, UnsafeMutablePointer<UInt64>) -> VkResult
    let destroyBuffer: @convention(c) (UInt, UInt64, UnsafeRawPointer?) -> Void
    let getBufferMemoryRequirements: @convention(c) (UInt, UInt64, UnsafeMutableRawPointer) -> Void
    let allocateMemory: @convention(c) (UInt, UnsafeRawPointer, UnsafeRawPointer?, UnsafeMutablePointer<UInt64>) -> VkResult
    let freeMemory: @convention(c) (UInt, UInt64, UnsafeRawPointer?) -> Void
    let bindBufferMemory: @convention(c) (UInt, UInt64, UInt64, UInt64) -> VkResult
    let mapMemory: @convention(c) (UInt, UInt64, UInt64, UInt64, UInt32, UnsafeMutablePointer<UnsafeMutableRawPointer?>) -> VkResult
    let unmapMemory: @convention(c) (UInt, UInt64) -> Void

    let createShaderModule: @convention(c) (UInt, UnsafeRawPointer, UnsafeRawPointer?, UnsafeMutablePointer<UInt64>) -> VkResult
    let destroyShaderModule: @convention(c) (UInt, UInt64, UnsafeRawPointer?) -> Void
    let createDescriptorSetLayout: @convention(c) (UInt, UnsafeRawPointer, UnsafeRawPointer?, UnsafeMutablePointer<UInt64>) -> VkResult
    let destroyDescriptorSetLayout: @convention(c) (UInt, UInt64, UnsafeRawPointer?) -> Void
    let createPipelineLayout: @convention(c) (UInt, UnsafeRawPointer, UnsafeRawPointer?, UnsafeMutablePointer<UInt64>) -> VkResult
    let destroyPipelineLayout: @convention(c) (UInt, UInt64, UnsafeRawPointer?) -> Void
    let createComputePipelines:
        @convention(c) (UInt, UInt64, UInt32, UnsafeRawPointer, UnsafeRawPointer?, UnsafeMutablePointer<UInt64>) -> VkResult
    let destroyPipeline: @convention(c) (UInt, UInt64, UnsafeRawPointer?) -> Void
    let createDescriptorPool: @convention(c) (UInt, UnsafeRawPointer, UnsafeRawPointer?, UnsafeMutablePointer<UInt64>) -> VkResult
    let destroyDescriptorPool: @convention(c) (UInt, UInt64, UnsafeRawPointer?) -> Void
    let allocateDescriptorSets: @convention(c) (UInt, UnsafeRawPointer, UnsafeMutablePointer<UInt64>) -> VkResult
    let updateDescriptorSets: @convention(c) (UInt, UInt32, UnsafeRawPointer?, UInt32, UnsafeRawPointer?) -> Void

    let createCommandPool: @convention(c) (UInt, UnsafeRawPointer, UnsafeRawPointer?, UnsafeMutablePointer<UInt64>) -> VkResult
    let destroyCommandPool: @convention(c) (UInt, UInt64, UnsafeRawPointer?) -> Void
    let allocateCommandBuffers: @convention(c) (UInt, UnsafeRawPointer, UnsafeMutablePointer<UInt>) -> VkResult
    let beginCommandBuffer: @convention(c) (UInt, UnsafeRawPointer) -> VkResult
    let endCommandBuffer: @convention(c) (UInt) -> VkResult
    let cmdBindPipeline: @convention(c) (UInt, UInt32, UInt64) -> Void
    let cmdBindDescriptorSets:
        @convention(c) (UInt, UInt32, UInt64, UInt32, UInt32, UnsafePointer<UInt64>, UInt32, UnsafePointer<UInt32>?) -> Void
    let cmdDispatch: @convention(c) (UInt, UInt32, UInt32, UInt32) -> Void
    let cmdDispatchIndirect: @convention(c) (UInt, UInt64, UInt64) -> Void
    let cmdPushConstants: @convention(c) (UInt, UInt64, UInt32, UInt32, UInt32, UnsafeRawPointer) -> Void
    let cmdCopyBuffer: @convention(c) (UInt, UInt64, UInt64, UInt32, UnsafeRawPointer) -> Void
    let cmdFillBuffer: @convention(c) (UInt, UInt64, UInt64, UInt64, UInt32) -> Void
    let cmdPipelineBarrier:
        @convention(c) (UInt, UInt32, UInt32, UInt32, UInt32, UnsafeRawPointer?, UInt32, UnsafeRawPointer?, UInt32, UnsafeRawPointer?) -> Void

    let queueSubmit: @convention(c) (UInt, UInt32, UnsafeRawPointer, UInt64) -> VkResult
    let createFence: @convention(c) (UInt, UnsafeRawPointer, UnsafeRawPointer?, UnsafeMutablePointer<UInt64>) -> VkResult
    let destroyFence: @convention(c) (UInt, UInt64, UnsafeRawPointer?) -> Void
    let getFenceStatus: @convention(c) (UInt, UInt64) -> VkResult
    let waitForFences: @convention(c) (UInt, UInt32, UnsafePointer<UInt64>, UInt32, UInt64) -> VkResult
    let resetFences: @convention(c) (UInt, UInt32, UnsafePointer<UInt64>) -> VkResult
    let resetCommandPool: @convention(c) (UInt, UInt64, UInt32) -> VkResult

    init(instance: UInt, getInstanceProcAddr: GetInstanceProcAddr) throws {
        func load<T>(_ name: String, as type: T.Type = T.self) throws -> T {
            guard let address = getInstanceProcAddr(instance, name) else {
                throw GPUError.unsupported("Vulkan loader does not export \(name)")
            }
            return unsafeBitCast(address, to: T.self)
        }
        destroyInstance = try load("vkDestroyInstance")
        enumeratePhysicalDevices = try load("vkEnumeratePhysicalDevices")
        getPhysicalDeviceProperties = try load("vkGetPhysicalDeviceProperties")
        getPhysicalDeviceMemoryProperties = try load("vkGetPhysicalDeviceMemoryProperties")
        getPhysicalDeviceFormatProperties = try load("vkGetPhysicalDeviceFormatProperties")
        getPhysicalDeviceFeatures2 = try load("vkGetPhysicalDeviceFeatures2")
        getPhysicalDeviceProperties2 = try load("vkGetPhysicalDeviceProperties2")
        getPhysicalDeviceQueueFamilyProperties = try load("vkGetPhysicalDeviceQueueFamilyProperties")
        createDevice = try load("vkCreateDevice")
        destroyDevice = try load("vkDestroyDevice")
        getDeviceQueue = try load("vkGetDeviceQueue")
        deviceWaitIdle = try load("vkDeviceWaitIdle")
        createBuffer = try load("vkCreateBuffer")
        destroyBuffer = try load("vkDestroyBuffer")
        getBufferMemoryRequirements = try load("vkGetBufferMemoryRequirements")
        allocateMemory = try load("vkAllocateMemory")
        freeMemory = try load("vkFreeMemory")
        bindBufferMemory = try load("vkBindBufferMemory")
        mapMemory = try load("vkMapMemory")
        unmapMemory = try load("vkUnmapMemory")
        createShaderModule = try load("vkCreateShaderModule")
        destroyShaderModule = try load("vkDestroyShaderModule")
        createDescriptorSetLayout = try load("vkCreateDescriptorSetLayout")
        destroyDescriptorSetLayout = try load("vkDestroyDescriptorSetLayout")
        createPipelineLayout = try load("vkCreatePipelineLayout")
        destroyPipelineLayout = try load("vkDestroyPipelineLayout")
        createComputePipelines = try load("vkCreateComputePipelines")
        destroyPipeline = try load("vkDestroyPipeline")
        createDescriptorPool = try load("vkCreateDescriptorPool")
        destroyDescriptorPool = try load("vkDestroyDescriptorPool")
        allocateDescriptorSets = try load("vkAllocateDescriptorSets")
        updateDescriptorSets = try load("vkUpdateDescriptorSets")
        createCommandPool = try load("vkCreateCommandPool")
        destroyCommandPool = try load("vkDestroyCommandPool")
        allocateCommandBuffers = try load("vkAllocateCommandBuffers")
        beginCommandBuffer = try load("vkBeginCommandBuffer")
        endCommandBuffer = try load("vkEndCommandBuffer")
        cmdBindPipeline = try load("vkCmdBindPipeline")
        cmdBindDescriptorSets = try load("vkCmdBindDescriptorSets")
        cmdDispatch = try load("vkCmdDispatch")
        cmdDispatchIndirect = try load("vkCmdDispatchIndirect")
        cmdPushConstants = try load("vkCmdPushConstants")
        cmdCopyBuffer = try load("vkCmdCopyBuffer")
        cmdFillBuffer = try load("vkCmdFillBuffer")
        cmdPipelineBarrier = try load("vkCmdPipelineBarrier")
        queueSubmit = try load("vkQueueSubmit")
        createFence = try load("vkCreateFence")
        destroyFence = try load("vkDestroyFence")
        getFenceStatus = try load("vkGetFenceStatus")
        waitForFences = try load("vkWaitForFences")
        resetFences = try load("vkResetFences")
        resetCommandPool = try load("vkResetCommandPool")
    }
}

/// Maps Vulkan result codes onto the shared error type.
func vulkanCheck(_ result: VkResult, _ operation: String) throws {
    switch result {
    case VK.success: return
    case VK.outOfHostMemory, VK.outOfDeviceMemory: throw GPUError.outOfMemory
    case VK.deviceLost: throw GPUError.deviceLost("Vulkan device lost during \(operation)")
    default: throw GPUError.backendFailure("Vulkan \(operation) failed with result \(result)")
    }
}
