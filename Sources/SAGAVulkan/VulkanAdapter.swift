import SAGATypes

public enum VulkanAdapterKind: Sendable, Equatable {
    case discrete, integrated, virtual, other
    /// Software Vulkan implementations such as lavapipe. They are real Vulkan drivers, ranked last.
    case software

    fileprivate var rank: Int {
        switch self {
        case .discrete: 0
        case .integrated: 1
        case .virtual: 2
        case .other: 3
        case .software: 4
        }
    }
}

/// Limits and memory layout read once from the physical device.
struct VulkanPhysicalLimits: Sendable {
    var maxStorageBufferRange: UInt32
    var maxUniformBufferRange: UInt32
    var maxBoundDescriptorSets: UInt32
    var maxPushConstantsSize: UInt32
    var maxImageDimension2D: UInt32
    var maxFragmentOutputAttachments: UInt32
    var maxVertexInputAttributes: UInt32
    var maxVertexInputBindings: UInt32
    var maxComputeSharedMemorySize: UInt32
    var maxComputeWorkGroupCount: (UInt32, UInt32, UInt32)
    var maxComputeWorkGroupInvocations: UInt32
    var maxComputeWorkGroupSize: (UInt32, UInt32, UInt32)
    var memoryTypeFlags: [UInt32]
    var memoryTypeHeaps: [UInt32]
    var heapSizes: [UInt64]
    var heapIsDeviceLocal: [Bool]
    /// Depth-stencil formats usable as attachments, sampled images and transfer endpoints.
    var depthStencilFormats: Set<UInt32>
    var timestampPeriod: Float
    var timestampsSupported: Bool
    var featureSupport: VulkanFeatureSupport
    /// 32-bit float color formats (r32, rg32, rgba32) can be sampled with linear filtering.
    var float32Filterable = false
    /// 32-bit float color formats can be blended when used as color attachments.
    var float32Blendable = false
}

/// Owns the Vulkan instance. Devices and adapters retain it, so the instance outlives every handle.
final class VulkanInstance: Sendable {
    let functions: VulkanFunctions
    let handle: UInt

    private init(functions: VulkanFunctions, handle: UInt) {
        self.functions = functions
        self.handle = handle
    }

    deinit { functions.destroyInstance(handle, nil) }

    static func create() throws -> VulkanInstance {
        guard let library = VulkanLibrary.open(), let address = library.symbol("vkGetInstanceProcAddr") else {
            throw GPUError.adapterUnavailable
        }
        let getInstanceProcAddr = unsafeBitCast(address, to: GetInstanceProcAddr.self)
        guard let createAddress = getInstanceProcAddr(0, "vkCreateInstance") else { throw GPUError.adapterUnavailable }
        let createInstance = unsafeBitCast(createAddress, to: CreateInstance.self)
        var application = VkApplicationInfo()
        application.apiVersion = VK.version(1, 1)
        // Surface extensions are optional: enable whichever the loader offers so a window can be attached later.
        var extensions: [String] = []
        typealias Enumerate = @convention(c) (UnsafePointer<CChar>?, UnsafeMutablePointer<UInt32>, UnsafeMutableRawPointer?) -> VkResult
        if let enumerateAddress = getInstanceProcAddr(0, "vkEnumerateInstanceExtensionProperties") {
            let enumerate = unsafeBitCast(enumerateAddress, to: Enumerate.self)
            var count: UInt32 = 0
            if enumerate(nil, &count, nil) == VK.success, count > 0 {
                var raw = [UInt8](repeating: 0, count: Int(count) * 260)
                raw.withUnsafeMutableBytes { _ = enumerate(nil, &count, $0.baseAddress) }
                let available = Set((0..<Int(count)).map { index in
                    String(decoding: raw[(index * 260)..<(index * 260 + 256)].prefix { $0 != 0 }, as: UTF8.self)
                })
                let wanted = [VKPresent.surfaceExtension, "VK_KHR_win32_surface", "VK_KHR_xlib_surface", "VK_KHR_wayland_surface", "VK_KHR_android_surface"]
                extensions = wanted.filter(available.contains)
                if !extensions.contains(VKPresent.surfaceExtension) { extensions = [] }
            }
        }
        var handle: UInt = 0
        let result: VkResult = withCStringArray(extensions) { names in
            withUnsafePointer(to: &application) { applicationPointer in
                var info = VkInstanceCreateInfo()
                info.pApplicationInfo = UnsafeRawPointer(applicationPointer)
                info.enabledExtensionCount = UInt32(extensions.count)
                info.ppEnabledExtensionNames = UnsafeRawPointer(names)
                return withUnsafePointer(to: &info) { createInstance(UnsafeRawPointer($0), nil, &handle) }
            }
        }
        // VK_ERROR_INCOMPATIBLE_DRIVER, VK_ERROR_INITIALIZATION_FAILED: no usable Vulkan 1.1 driver.
        if result == -9 || result == -3 { throw GPUError.adapterUnavailable }
        try vulkanCheck(result, "instance creation")
        return VulkanInstance(functions: try VulkanFunctions(instance: handle, getInstanceProcAddr: getInstanceProcAddr),
                              handle: handle)
    }
}

public struct VulkanAdapter: Sendable {
    public let name: String
    public let kind: VulkanAdapterKind
    /// Vulkan API version supported by the driver as `(major, minor)`.
    public let apiVersion: (major: UInt32, minor: UInt32)
    let instance: VulkanInstance
    let physical: UInt
    let queueFamily: UInt32
    let limits: VulkanPhysicalLimits
}

public enum VulkanBackend {
    /// Returns the best Vulkan 1.1 adapter with a compute queue, or nil when no driver is installed.
    public static func requestAdapter() -> VulkanAdapter? {
        (try? requestAdapters())?.first
    }

    /// Every usable adapter, best first. Throws only when the loader exists but enumeration fails.
    public static func requestAdapters() throws -> [VulkanAdapter] {
        let instance: VulkanInstance
        do { instance = try VulkanInstance.create() } catch GPUError.adapterUnavailable { return [] }
        let f = instance.functions
        var count: UInt32 = 0
        try enumerationCheck(f.enumeratePhysicalDevices(instance.handle, &count, nil))
        var devices = [UInt](repeating: 0, count: Int(count))
        try enumerationCheck(devices.withUnsafeMutableBufferPointer {
            f.enumeratePhysicalDevices(instance.handle, &count, $0.baseAddress)
        })
        let adapters = devices.prefix(Int(count)).compactMap { adapter(for: $0, instance: instance) }
        return adapters.sorted { $0.kind.rank < $1.kind.rank }
    }

    public static func makeDevice(timeout: Duration = .seconds(10)) throws -> VulkanDevice {
        try makeDevice(adapter: requestAdapter(), timeout: timeout)
    }

    public static func makeDevice(adapter: VulkanAdapter?, timeout: Duration = .seconds(10)) throws -> VulkanDevice {
        guard let adapter else { throw GPUError.adapterUnavailable }
        return try VulkanDevice(adapter: adapter, timeout: timeout)
    }

    /// Creates a device that can attach native windows after creating GPU resources.
    public static func makePresentableDevice(adapter: VulkanAdapter?, timeout: Duration = .seconds(10)) throws -> VulkanDevice {
        guard let adapter else { throw GPUError.adapterUnavailable }
        return try VulkanDevice(adapter: adapter, timeout: timeout, enablePresentation: true)
    }

    /// Creates a device that can present to `surface`. Throws `GPUError.unsupported` when the driver cannot.
    public static func makeDevice(
        surface: VulkanSurfaceTarget, adapter: VulkanAdapter? = requestAdapter(), timeout: Duration = .seconds(10)
    ) throws -> VulkanDevice {
        guard let adapter else { throw GPUError.adapterUnavailable }
        return try VulkanDevice(adapter: adapter, timeout: timeout, surface: surface)
    }

    private static func enumerationCheck(_ result: VkResult) throws {
        // VK_INCOMPLETE only means the device count changed between calls.
        if result == 5 { return }
        try vulkanCheck(result, "physical device enumeration")
    }

    private static func adapter(for physical: UInt, instance: VulkanInstance) -> VulkanAdapter? {
        let f = instance.functions
        // VkPhysicalDeviceProperties is 824 bytes; read fixed offsets from the Vulkan 1.0 layout.
        var properties = [UInt8](repeating: 0, count: 1024)
        properties.withUnsafeMutableBytes { f.getPhysicalDeviceProperties(physical, $0.baseAddress!) }
        let apiVersion = properties.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 0, as: UInt32.self) }
        let type = properties.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 16, as: UInt32.self) }
        guard apiVersion >= VK.version(1, 1) else { return nil }
        let nameBytes = properties[20..<(20 + 256)].prefix { $0 != 0 }
        let name = String(decoding: nameBytes, as: UTF8.self)
        func limit(_ offset: Int) -> UInt32 {
            properties.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: limitsBase + offset, as: UInt32.self) }
        }

        var familyCount: UInt32 = 0
        f.getPhysicalDeviceQueueFamilyProperties(physical, &familyCount, nil)
        var families = [VkQueueFamilyProperties](repeating: .init(), count: Int(familyCount))
        families.withUnsafeMutableBufferPointer {
            f.getPhysicalDeviceQueueFamilyProperties(physical, &familyCount, UnsafeMutableRawPointer($0.baseAddress))
        }
        guard let family = families.firstIndex(where: { $0.queueFlags & VK.queueCompute != 0 && $0.queueCount > 0 }) else {
            return nil
        }

        // VkPhysicalDeviceMemoryProperties: count, 32 x {flags, heap}, heapCount, 16 x {size, flags}.
        var memory = [UInt8](repeating: 0, count: 520)
        memory.withUnsafeMutableBytes { f.getPhysicalDeviceMemoryProperties(physical, $0.baseAddress!) }
        func word(_ offset: Int) -> UInt32 { memory.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: offset, as: UInt32.self) } }
        let typeCount = Int(min(word(0), 32))
        let heapCount = Int(min(word(260), 16))
        let heapSizes = (0..<heapCount).map { index in
            memory.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 264 + index * 16, as: UInt64.self) }
        }
        let heapLocal = (0..<heapCount).map { word(264 + $0 * 16 + 8) & VK.MemoryProperty.deviceLocal != 0 }
        var depthFormats: Set<UInt32> = []
        for candidate: UInt32 in [124, 126, 127, 129, 130] {
            var properties = [UInt32](repeating: 0, count: 3)
            properties.withUnsafeMutableBytes { f.getPhysicalDeviceFormatProperties(physical, candidate, $0.baseAddress!) }
            // optimalTilingFeatures: depth-stencil attachment, sampled image, transfer source and destination.
            let required: UInt32 = 0x200 | 0x1 | 0x4000 | 0x8000
            if properties[1] & required == required { depthFormats.insert(candidate) }
        }
        // optimalTilingFeatures bit 0x1000 is linear sampling and 0x80 is blending, for R32, RG32 and RGBA32 float.
        var float32Filterable = true
        var float32Blendable = true
        for candidate: UInt32 in [100, 103, 109] {
            var properties = [UInt32](repeating: 0, count: 3)
            properties.withUnsafeMutableBytes { f.getPhysicalDeviceFormatProperties(physical, candidate, $0.baseAddress!) }
            if properties[1] & 0x1000 == 0 { float32Filterable = false }
            if properties[1] & 0x80 == 0 { float32Blendable = false }
        }
        let limits = VulkanPhysicalLimits(
            maxStorageBufferRange: limit(28), maxUniformBufferRange: limit(24), maxBoundDescriptorSets: limit(64), maxPushConstantsSize: limit(32), maxImageDimension2D: limit(4), maxFragmentOutputAttachments: limit(204), maxVertexInputAttributes: limit(128), maxVertexInputBindings: limit(132),
            maxComputeSharedMemorySize: limit(216),
            maxComputeWorkGroupCount: (limit(220), limit(224), limit(228)),
            maxComputeWorkGroupInvocations: limit(232),
            maxComputeWorkGroupSize: (limit(236), limit(240), limit(244)),
            memoryTypeFlags: (0..<typeCount).map { word(4 + $0 * 8) },
            memoryTypeHeaps: (0..<typeCount).map { word(4 + $0 * 8 + 4) },
            heapSizes: heapSizes, heapIsDeviceLocal: heapLocal, depthStencilFormats: depthFormats,
            timestampPeriod: Float(bitPattern: limit(424)),
            timestampsSupported: limit(420) != 0 && families[family].timestampValidBits > 0,
            featureSupport: vulkanQueryFeatures(physical: physical, instance: instance),
            float32Filterable: float32Filterable, float32Blendable: float32Blendable)
        let kind: VulkanAdapterKind = switch VK.PhysicalDeviceType(rawValue: type) {
        case .discreteGPU: .discrete
        case .integratedGPU: .integrated
        case .virtualGPU: .virtual
        case .cpu: .software
        default: .other
        }
        return VulkanAdapter(name: name, kind: kind, apiVersion: (apiVersion >> 22, (apiVersion >> 12) & 0x3ff),
                             instance: instance, physical: physical, queueFamily: UInt32(family), limits: limits)
    }

    /// Offset of `VkPhysicalDeviceLimits` inside `VkPhysicalDeviceProperties`.
    private static let limitsBase = 296
}
