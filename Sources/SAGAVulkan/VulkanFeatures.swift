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
    var meshShader = false
    var taskShader = false
    var meshLimits = VulkanMeshLimits()
    var cooperativeMatrix = false
    var rayQuery = false
    var accelerationStructure = false
    var bufferDeviceAddress = false
    var rayPositionFetch = false
    var memoryModel = false
    var memoryModelDeviceScope = false
    var cooperativeProperties: [CooperativeMatrixProperties] = []
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

    static let cooperativeExtension = "VK_KHR_cooperative_matrix"
    static let memoryModelExtension = "VK_KHR_vulkan_memory_model"

    var canEnableCooperativeMatrix: Bool {
        cooperativeMatrix && memoryModel && memoryModelDeviceScope && !cooperativeProperties.isEmpty
            && deviceExtensions.contains(Self.cooperativeExtension)
    }

    static let meshExtension = "VK_EXT_mesh_shader"
    static let rayQueryExtensions = ["VK_KHR_ray_query", "VK_KHR_acceleration_structure", "VK_KHR_deferred_host_operations",
                                     "VK_KHR_buffer_device_address", "VK_KHR_spirv_1_4", "VK_KHR_shader_float_controls",
                                     "VK_EXT_descriptor_indexing", "VK_KHR_maintenance3"]

    static let positionFetchExtension = "VK_KHR_ray_tracing_position_fetch"

    /// Vertex positions of ray hits need the extension and feature on top of ray queries.
    var canEnablePositionFetch: Bool { canEnableRayQuery && rayPositionFetch && deviceExtensions.contains(Self.positionFetchExtension) }

    /// Ray queries need the three features and every extension that carries them.
    var canEnableRayQuery: Bool {
        rayQuery && accelerationStructure && bufferDeviceAddress && Self.rayQueryExtensions.allSatisfy(deviceExtensions.contains)
    }

    /// Mesh shaders need the extension, the feature, and SPIR-V 1.4 through its core version or extension.
    var canEnableMeshShaders: Bool { meshShader && deviceExtensions.contains(Self.meshExtension) }

    var canEnableFloat16: Bool { float16 && storage16 && deviceExtensions.contains(Self.float16Extension) }
}

/// The three chained feature structures as one raw allocation, so pNext links stay valid for the call.
final class VulkanFeatureChain {
    private let memory: UnsafeMutableRawPointer
    private static let size = 600

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
        (memory + 320).storeBytes(of: UInt(bitPattern: memory + 352), toByteOffset: 8, as: UInt.self)
        (memory + 352).storeBytes(of: 1000506000, as: UInt32.self)  // cooperative matrix features
        (memory + 352).storeBytes(of: UInt(bitPattern: memory + 384), toByteOffset: 8, as: UInt.self)
        (memory + 384).storeBytes(of: 1000211000, as: UInt32.self)  // Vulkan memory model features
        (memory + 384).storeBytes(of: UInt(bitPattern: memory + 416), toByteOffset: 8, as: UInt.self)
        (memory + 416).storeBytes(of: 1000328000, as: UInt32.self)  // mesh shader features
        (memory + 456).storeBytes(of: 1000257000, as: UInt32.self)  // buffer device address features
        (memory + 456).storeBytes(of: UInt(bitPattern: memory + 488), toByteOffset: 8, as: UInt.self)
        (memory + 488).storeBytes(of: 1000150013, as: UInt32.self)  // acceleration structure features
        (memory + 488).storeBytes(of: UInt(bitPattern: memory + 528), toByteOffset: 8, as: UInt.self)
        (memory + 528).storeBytes(of: 1000348013, as: UInt32.self)  // ray query features
        (memory + 528).storeBytes(of: UInt(bitPattern: memory + 552), toByteOffset: 8, as: UInt.self)
        (memory + 552).storeBytes(of: 1000481000, as: UInt32.self)  // ray tracing position fetch features
        // Mesh features link onward to the ray structures so a full query sees every one.
        (memory + 416).storeBytes(of: UInt(bitPattern: memory + 456), toByteOffset: 8, as: UInt.self)
    }

    var bufferDeviceAddress: UInt32 {
        get { (memory + 456).load(fromByteOffset: 16, as: UInt32.self) }
        set { (memory + 456).storeBytes(of: newValue, toByteOffset: 16, as: UInt32.self) }
    }
    var accelerationStructure: UInt32 {
        get { (memory + 488).load(fromByteOffset: 16, as: UInt32.self) }
        set { (memory + 488).storeBytes(of: newValue, toByteOffset: 16, as: UInt32.self) }
    }
    var rayPositionFetch: UInt32 {
        get { (memory + 552).load(fromByteOffset: 16, as: UInt32.self) }
        set { (memory + 552).storeBytes(of: newValue, toByteOffset: 16, as: UInt32.self) }
    }
    var rayQuery: UInt32 {
        get { (memory + 528).load(fromByteOffset: 16, as: UInt32.self) }
        set { (memory + 528).storeBytes(of: newValue, toByteOffset: 16, as: UInt32.self) }
    }

    /// taskShader at 16 and meshShader at 20 of the mesh shader feature structure.
    var meshShader: UInt32 {
        get { (memory + 416).load(fromByteOffset: 20, as: UInt32.self) }
        set { (memory + 416).storeBytes(of: newValue, toByteOffset: 20, as: UInt32.self) }
    }
    var taskShader: UInt32 {
        get { (memory + 416).load(fromByteOffset: 16, as: UInt32.self) }
        set { (memory + 416).storeBytes(of: newValue, toByteOffset: 16, as: UInt32.self) }
    }

    var cooperativeMatrix: UInt32 {
        get { (memory + 352).load(fromByteOffset: 16, as: UInt32.self) }
        set { (memory + 352).storeBytes(of: newValue, toByteOffset: 16, as: UInt32.self) }
    }
    var vulkanMemoryModel: UInt32 {
        get { (memory + 384).load(fromByteOffset: 16, as: UInt32.self) }
        set {
            (memory + 384).storeBytes(of: newValue, toByteOffset: 16, as: UInt32.self)
            // Device-scope visibility chains are needed for subgroup-scope cooperative loads and stores.
            (memory + 384).storeBytes(of: newValue, toByteOffset: 20, as: UInt32.self)
        }
    }
    var memoryModelDeviceScope: UInt32 { (memory + 384).load(fromByteOffset: 20, as: UInt32.self) }

    /// Chains only the optional structures that will be enabled.
    func relink(float16: Bool, cooperative: Bool, mesh: Bool = false, rayQuery: Bool = false, positionFetch: Bool = false) {
        var order: [Int] = [256]
        if float16 { order += [288, 320] }
        if cooperative { order += [352, 384] }
        if mesh { order.append(416) }
        if rayQuery { order += [456, 488, 528] }
        if positionFetch { order.append(552) }
        memory.storeBytes(of: UInt(bitPattern: memory + 256), toByteOffset: 8, as: UInt.self)
        for (index, offset) in order.enumerated() {
            let next = index + 1 < order.count ? UInt(bitPattern: memory + order[index + 1]) : 0
            (memory + offset).storeBytes(of: next, toByteOffset: 8, as: UInt.self)
        }
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
    support.meshShader = chain.meshShader != 0
    support.taskShader = chain.taskShader != 0
    support.cooperativeMatrix = chain.cooperativeMatrix != 0
    support.bufferDeviceAddress = chain.bufferDeviceAddress != 0
    support.accelerationStructure = chain.accelerationStructure != 0
    support.rayQuery = chain.rayQuery != 0
    support.rayPositionFetch = chain.rayPositionFetch != 0
    support.memoryModel = chain.vulkanMemoryModel != 0
    support.memoryModelDeviceScope = chain.memoryModelDeviceScope != 0
    if support.cooperativeMatrix { support.cooperativeProperties = vulkanCooperativeProperties(physical: physical, instance: instance) }
    // VkPhysicalDeviceProperties2 (16-byte header, 824-byte properties) chained to the subgroup properties.
    let properties = UnsafeMutableRawPointer.allocate(byteCount: 16 + 824 + 40 + 24 + 160, alignment: 8)
    defer { properties.deallocate() }
    properties.initializeMemory(as: UInt8.self, repeating: 0, count: 16 + 824 + 40 + 24 + 160)
    properties.storeBytes(of: 1000059001, as: UInt32.self)
    let subgroup = properties + 16 + 824
    properties.storeBytes(of: UInt(bitPattern: subgroup), toByteOffset: 8, as: UInt.self)
    subgroup.storeBytes(of: 1000094000, as: UInt32.self)
    let multiviewProperties = subgroup + 40
    subgroup.storeBytes(of: UInt(bitPattern: multiviewProperties), toByteOffset: 8, as: UInt.self)
    multiviewProperties.storeBytes(of: 1000053002, as: UInt32.self)
    let meshProperties = multiviewProperties + 24
    multiviewProperties.storeBytes(of: UInt(bitPattern: meshProperties), toByteOffset: 8, as: UInt.self)
    meshProperties.storeBytes(of: 1000328001, as: UInt32.self)
    instance.functions.getPhysicalDeviceProperties2(physical, properties)
    support.subgroupSize = subgroup.load(fromByteOffset: 16, as: UInt32.self)
    support.subgroupStages = subgroup.load(fromByteOffset: 20, as: UInt32.self)
    support.subgroupOperations = subgroup.load(fromByteOffset: 24, as: UInt32.self)
    support.maxMultiviewViewCount = multiviewProperties.load(fromByteOffset: 16, as: UInt32.self)
    func meshWord(_ offset: Int) -> UInt32 { meshProperties.load(fromByteOffset: offset, as: UInt32.self) }
    support.meshLimits = VulkanMeshLimits(
        taskTotal: meshWord(16), taskCount: meshWord(20), taskInvocations: meshWord(32), taskSize: meshWord(36),
        taskPayload: meshWord(48), meshTotal: meshWord(60), meshCount: meshWord(64), meshInvocations: meshWord(76),
        meshSize: meshWord(80), outputVertices: meshWord(112), outputPrimitives: meshWord(116))
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

/// Subgroup-scope cooperative matrix shapes with half or single precision operands.
func vulkanCooperativeProperties(physical: UInt, instance: VulkanInstance) -> [CooperativeMatrixProperties] {
    guard let library = VulkanLibrary.open(), let address = library.symbol("vkGetInstanceProcAddr") else { return [] }
    let getInstanceProcAddr = unsafeBitCast(address, to: GetInstanceProcAddr.self)
    typealias Query = @convention(c) (UInt, UnsafeMutablePointer<UInt32>, UnsafeMutableRawPointer?) -> VkResult
    guard let queryAddress = getInstanceProcAddr(instance.handle, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR") else { return [] }
    let query = unsafeBitCast(queryAddress, to: Query.self)
    var count: UInt32 = 0
    guard query(physical, &count, nil) == VK.success, count > 0 else { return [] }
    // VkCooperativeMatrixPropertiesKHR is 56 bytes: sType, pNext, M, N, K, A, B, C, result, saturating, scope.
    let stride = 56
    let raw = UnsafeMutableRawPointer.allocate(byteCount: Int(count) * stride, alignment: 8)
    defer { raw.deallocate() }
    raw.initializeMemory(as: UInt8.self, repeating: 0, count: Int(count) * stride)
    for index in 0..<Int(count) { (raw + index * stride).storeBytes(of: 1000506001, as: UInt32.self) }
    guard query(physical, &count, raw) == VK.success else { return [] }
    func scalar(_ type: UInt32) -> CooperativeMatrixScalar? { type == 0 ? .float16 : (type == 1 ? .float32 : nil) }
    var result: [CooperativeMatrixProperties] = []
    for index in 0..<Int(count) {
        let base = raw + index * stride
        func word(_ offset: Int) -> UInt32 { base.load(fromByteOffset: offset, as: UInt32.self) }
        guard word(48) == 3, word(28) == word(32), word(36) == word(40),  // subgroup scope; A and B match; C is the result type
              let input = scalar(word(28)), let accumulator = scalar(word(36)) else { continue }
        let properties = CooperativeMatrixProperties(mSize: word(16), nSize: word(20), kSize: word(24), inputType: input,
                                                     accumulatorType: accumulator, saturatingAccumulation: word(44) != 0)
        if !result.contains(properties) { result.append(properties) }
    }
    return result
}

/// Limits from VkPhysicalDeviceMeshShaderPropertiesEXT.
struct VulkanMeshLimits: Sendable {
    var taskTotal: UInt32 = 0
    var taskCount: UInt32 = 0
    var taskInvocations: UInt32 = 0
    var taskSize: UInt32 = 0
    var taskPayload: UInt32 = 0
    var meshTotal: UInt32 = 0
    var meshCount: UInt32 = 0
    var meshInvocations: UInt32 = 0
    var meshSize: UInt32 = 0
    var outputVertices: UInt32 = 0
    var outputPrimitives: UInt32 = 0
}
