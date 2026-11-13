import SAGATypes

/// The Vulkan objects of the submission a ``NativePass`` body is recorded into.
///
/// Every handle is valid only while the body runs. SAGA keeps all images in `VK_IMAGE_LAYOUT_GENERAL`, so a body can use
/// them as they are; it must leave each resource in that layout. SAGA places a full memory barrier before and after the
/// body, so earlier commands are visible to it and its writes are visible to later commands.
public struct VulkanNativePassContext: NativePassContext {
    /// A texture of the pass, with a view over all of its mip levels and array layers.
    public struct Image: Sendable {
        public let image: UInt64
        public let view: UInt64
        /// A `VkFormat` value.
        public let format: UInt32
        public let width: UInt32
        public let height: UInt32
        public let mipLevels: UInt32
        public let arrayLayers: UInt32
        /// A `VkImageAspectFlags` mask covering the whole view.
        public let aspectMask: UInt32
        /// Always `VK_IMAGE_LAYOUT_GENERAL`.
        public let layout: UInt32
    }

    public let instance: UInt
    public let physicalDevice: UInt
    public let device: UInt
    public let queue: UInt
    public let queueFamilyIndex: UInt32
    /// The command buffer the pass records into. It is in the recording state.
    public let commandBuffer: UInt

    let buffers: [ResourceID: UInt64]
    let images: [ResourceID: Image]
    private let instanceProcedureLoader: @Sendable (UInt, String) -> UnsafeRawPointer?

    init(
        instance: UInt, physicalDevice: UInt, device: UInt, queue: UInt, queueFamilyIndex: UInt32, commandBuffer: UInt,
        buffers: [ResourceID: UInt64], images: [ResourceID: Image],
        procedureLoader: @escaping @Sendable (UInt, String) -> UnsafeRawPointer?
    ) {
        self.instance = instance
        self.physicalDevice = physicalDevice
        self.device = device
        self.queue = queue
        self.queueFamilyIndex = queueFamilyIndex
        self.commandBuffer = commandBuffer
        self.buffers = buffers
        self.images = images
        self.instanceProcedureLoader = procedureLoader
    }

    /// The `VkBuffer` of a buffer named in the pass's uses.
    public func buffer(_ id: ResourceID) throws -> UInt64 {
        guard let handle = buffers[id] else { throw GPUError.validation("The native pass did not name this buffer") }
        return handle
    }

    /// The image and view of a texture named in the pass's uses.
    public func image(_ id: ResourceID) throws -> Image {
        guard let image = images[id] else { throw GPUError.validation("The native pass did not name this texture") }
        return image
    }

    /// `vkGetInstanceProcAddr` resolved against this instance, for libraries that load Vulkan themselves.
    public func instanceProcedure(_ name: String) -> UnsafeRawPointer? {
        instanceProcedureLoader(instance, name)
    }

    /// A device-level function, resolved through `vkGetDeviceProcAddr` of this device.
    public func deviceProcedure(_ name: String) -> UnsafeRawPointer? {
        typealias GetDeviceProcAddr = @convention(c) (UInt, UnsafePointer<CChar>) -> UnsafeRawPointer?
        guard let loader = instanceProcedureLoader(instance, "vkGetDeviceProcAddr") else { return nil }
        let getDeviceProcAddr = unsafeBitCast(loader, to: GetDeviceProcAddr.self)
        return name.withCString { getDeviceProcAddr(device, $0) }
    }
}

extension VulkanInstance {
    /// `vkGetInstanceProcAddr` for this instance.
    func procedure(_ handle: UInt, _ name: String) -> UnsafeRawPointer? {
        guard let library = VulkanLibrary.open(), let address = library.symbol("vkGetInstanceProcAddr") else { return nil }
        let getProc = unsafeBitCast(address, to: GetInstanceProcAddr.self)
        return name.withCString { getProc(handle, $0) }
    }
}

extension VulkanDevice {
    /// Everything a native pass needs, resolved while the submission is planned.
    struct NativePlan {
        var pass: NativePass
        var buffers: [ResourceID: UInt64]
        var textures: [ResourceID: Texture]
    }

    func planNative(_ pass: NativePass) throws -> NativePlan {
        var bufferHandles: [ResourceID: UInt64] = [:]
        var textureHandles: [ResourceID: Texture] = [:]
        for use in pass.uses {
            if let buffer = buffers[use.resource] {
                bufferHandles[use.resource] = buffer.native.handle
            } else if let texture = texture(for: use.resource) {
                guard texture.descriptor.dimension == .d2 else {
                    throw GPUError.unsupported("Vulkan native passes support 2D textures only")
                }
                textureHandles[use.resource] = texture
            } else {
                throw GPUError.validation("A native pass names an unknown resource")
            }
        }
        return NativePlan(pass: pass, buffers: bufferHandles, textures: textureHandles)
    }

    /// Creates the views of the pass's textures (released with the batch) and runs its body.
    func recordNative(_ plan: NativePlan, commandBuffer: UInt, batch: inout PendingBatch) throws {
        let g = native.g, device = native.device
        var images: [ResourceID: VulkanNativePassContext.Image] = [:]
        for (id, texture) in plan.textures {
            let descriptor = texture.descriptor
            var info = VkImageViewCreateInfo()
            info.image = texture.image
            info.format = texture.format.vk
            info.viewType = descriptor.arrayLayerCount > 1 ? 5 : VK.Image.viewType2D
            let aspect = texture.format.hasDepth ? VK.Depth.aspectDepth : texture.format.aspect
            info.subresourceRange = VkImageSubresourceRange(
                aspectMask: aspect, baseMipLevel: 0, levelCount: descriptor.mipLevelCount,
                baseArrayLayer: 0, layerCount: descriptor.arrayLayerCount)
            var view: UInt64 = 0
            try vulkanCheck(withUnsafePointer(to: &info) { g.createImageView(device, UnsafeRawPointer($0), nil, &view) },
                            "native pass image view creation")
            batch.views.append(view)
            images[id] = .init(
                image: texture.image, view: view, format: texture.format.vk, width: descriptor.width, height: descriptor.height,
                mipLevels: descriptor.mipLevelCount, arrayLayers: descriptor.arrayLayerCount, aspectMask: aspect,
                layout: VK.Image.layoutGeneral)
        }
        let instance = native.instance
        let context = VulkanNativePassContext(
            instance: instance.handle, physicalDevice: native.physical, device: device, queue: native.queue,
            queueFamilyIndex: native.queueFamily, commandBuffer: commandBuffer, buffers: plan.buffers, images: images,
            procedureLoader: { handle, name in instance.procedure(handle, name) })
        try plan.pass.record(into: context)
    }
}
