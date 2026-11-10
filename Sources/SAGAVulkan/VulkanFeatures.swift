import SAGATypes

// Optional device features. The backend enables every feature it can use when the driver offers it, and
// advertises the matching `GPUFeature` only when the shader path for it is implemented.

/// What the physical device supports, read once through `vkGetPhysicalDeviceFeatures2`.
struct VulkanFeatureSupport: Sendable {
    /// `VkPhysicalDeviceFeatures` flags, in declaration order.
    var core = [UInt32](repeating: 0, count: 55)
    var float16 = false
    var int8 = false
    var storage16 = false
    var multiview = false
    var maxMultiviewViewCount: UInt32 = 0
    var deviceExtensions: Set<String> = []
    var subgroupSize: UInt32 = 0
    var subgroupStages: UInt32 = 0
    var subgroupOperations: UInt32 = 0

    /// Core feature indices this backend enables when present.
    static let enabledCoreFeatures = [
        10,  // drawIndirectFirstInstance
        12,  // depthBiasClamp
        19,  // samplerAnisotropy
        33,  // shaderUniformBufferArrayDynamicIndexing
        34,  // shaderSampledImageArrayDynamicIndexing
        35,  // shaderStorageBufferArrayDynamicIndexing
        36,  // shaderStorageImageArrayDynamicIndexing
        25,  // vertexPipelineStoresAndAtomics
        26,  // fragmentStoresAndAtomics
        29,  // shaderStorageImageExtendedFormats
        37,  // shaderClipDistance
        40,  // shaderInt64
        41,  // shaderInt16
        20,  // textureCompressionETC2
        21,  // textureCompressionASTC_LDR
        22,  // textureCompressionBC
    ]

    static let float16Extension = "VK_KHR_shader_float16_int8"

    /// Every operation class WebGPU subgroups need, in compute and fragment shaders.
    var supportsSubgroups: Bool {
        let required: UInt32 = 0x1 | 0x2 | 0x4 | 0x8 | 0x10 | 0x20 | 0x80
        return subgroupSize >= 4 && subgroupOperations & required == required && subgroupStages & 0x30 == 0x30
    }

    var canEnableFloat16: Bool { float16 && storage16 && deviceExtensions.contains(Self.float16Extension) }
}

/// The three chained feature structures as one raw allocation, so pNext links stay valid for the call.
final class VulkanFeatureChain {
    private let memory: UnsafeMutableRawPointer
    private static let size = 256 + 32 + 32 + 32

    /// Offsets: features2 at 0 (16 header bytes + 220 feature bytes), multiview at 256, float16/int8 at 288,
    /// 16-bit storage at 320. They are chained in that order.
    init() {
        memory = UnsafeMutableRawPointer.allocate(byteCount: Self.size, alignment: 8)
        memory.initializeMemory(as: UInt8.self, repeating: 0, count: Self.size)
        memory.storeBytes(of: 1000059000, as: UInt32.self)
        memory.storeBytes(of: UInt(bitPattern: memory + 256), toByteOffset: 8, as: UInt.self)
        (memory + 256).storeBytes(of: 1000053001, as: UInt32.self)
        (memory + 256).storeBytes(of: UInt(bitPattern: memory + 288), toByteOffset: 8, as: UInt.self)
        (memory + 288).storeBytes(of: 1000082000, as: UInt32.self)
        (memory + 288).storeBytes(of: UInt(bitPattern: memory + 320), toByteOffset: 8, as: UInt.self)
        (memory + 320).storeBytes(of: 1000083000, as: UInt32.self)
    }

    deinit { memory.deallocate() }

    var features2: UnsafeMutableRawPointer { memory }

    func setCore(_ index: Int, _ value: UInt32) { memory.storeBytes(of: value, toByteOffset: 16 + index * 4, as: UInt32.self) }
    func core(_ index: Int) -> UInt32 { memory.load(fromByteOffset: 16 + index * 4, as: UInt32.self) }

    var float16: UInt32 {
        get { (memory + 288).load(fromByteOffset: 16, as: UInt32.self) }
        set { (memory + 288).storeBytes(of: newValue, toByteOffset: 16, as: UInt32.self) }
    }
    var int8: UInt32 {
        get { (memory + 288).load(fromByteOffset: 20, as: UInt32.self) }
        set { (memory + 288).storeBytes(of: newValue, toByteOffset: 20, as: UInt32.self) }
    }
    /// storageBuffer16BitAccess and uniformAndStorageBuffer16BitAccess.
    var storage16: UInt32 {
        get { (memory + 320).load(fromByteOffset: 16, as: UInt32.self) }
        set {
            (memory + 320).storeBytes(of: newValue, toByteOffset: 16, as: UInt32.self)
            (memory + 320).storeBytes(of: newValue, toByteOffset: 20, as: UInt32.self)
        }
    }
    var storage16Supported: Bool { (memory + 320).load(fromByteOffset: 16, as: UInt32.self) != 0 }

    var multiview: UInt32 {
        get { (memory + 256).load(fromByteOffset: 16, as: UInt32.self) }
        set { (memory + 256).storeBytes(of: newValue, toByteOffset: 16, as: UInt32.self) }
    }
    var multiviewSupported: Bool { multiview != 0 }

    /// Detaches the float16 and 16-bit storage structures, for drivers that lack the extension.
    func unlinkFloat16() {
        (memory + 256).storeBytes(of: UInt(0), toByteOffset: 8, as: UInt.self)
    }
}

func vulkanQueryFeatures(physical: UInt, instance: VulkanInstance) -> VulkanFeatureSupport {
    var support = VulkanFeatureSupport()
    support.deviceExtensions = vulkanDeviceExtensions(physical: physical, instance: instance)
    let chain = VulkanFeatureChain()
    instance.functions.getPhysicalDeviceFeatures2(physical, chain.features2)
    support.core = (0..<55).map { chain.core($0) }
    support.float16 = chain.float16 != 0
    support.int8 = chain.int8 != 0
    support.storage16 = chain.storage16Supported
    support.multiview = chain.multiviewSupported
    // VkPhysicalDeviceProperties2 (16-byte header, 824-byte properties) chained to the subgroup properties.
    let properties = UnsafeMutableRawPointer.allocate(byteCount: 16 + 824 + 40 + 24, alignment: 8)
    defer { properties.deallocate() }
    properties.initializeMemory(as: UInt8.self, repeating: 0, count: 16 + 824 + 40 + 24)
    properties.storeBytes(of: 1000059001, as: UInt32.self)
    let subgroup = properties + 16 + 824
    properties.storeBytes(of: UInt(bitPattern: subgroup), toByteOffset: 8, as: UInt.self)
    subgroup.storeBytes(of: 1000094000, as: UInt32.self)
    let multiviewProperties = subgroup + 40
    subgroup.storeBytes(of: UInt(bitPattern: multiviewProperties), toByteOffset: 8, as: UInt.self)
    multiviewProperties.storeBytes(of: 1000053002, as: UInt32.self)
    instance.functions.getPhysicalDeviceProperties2(physical, properties)
    support.subgroupSize = subgroup.load(fromByteOffset: 16, as: UInt32.self)
    support.subgroupStages = subgroup.load(fromByteOffset: 20, as: UInt32.self)
    support.subgroupOperations = subgroup.load(fromByteOffset: 24, as: UInt32.self)
    support.maxMultiviewViewCount = multiviewProperties.load(fromByteOffset: 16, as: UInt32.self)
    return support
}

func vulkanDeviceExtensions(physical: UInt, instance: VulkanInstance) -> Set<String> {
    guard let library = VulkanLibrary.open(), let address = library.symbol("vkGetInstanceProcAddr") else { return [] }
    let getInstanceProcAddr = unsafeBitCast(address, to: GetInstanceProcAddr.self)
    typealias Enumerate = @convention(c) (UInt, UnsafePointer<CChar>?, UnsafeMutablePointer<UInt32>, UnsafeMutableRawPointer?) -> VkResult
    guard let enumerateAddress = getInstanceProcAddr(instance.handle, "vkEnumerateDeviceExtensionProperties") else { return [] }
    let enumerate = unsafeBitCast(enumerateAddress, to: Enumerate.self)
    var count: UInt32 = 0
    guard enumerate(physical, nil, &count, nil) == VK.success, count > 0 else { return [] }
    var raw = [UInt8](repeating: 0, count: Int(count) * 260)
    raw.withUnsafeMutableBytes { _ = enumerate(physical, nil, &count, $0.baseAddress) }
    return Set((0..<Int(count)).map { index in
        String(decoding: raw[(index * 260)..<(index * 260 + 256)].prefix { $0 != 0 }, as: UTF8.self)
    })
}
