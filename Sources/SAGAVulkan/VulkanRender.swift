import SAGATypes
import SAGABackend
import SAGAShaderIR
import SAGACompiler

// Offscreen rendering and texture transfer for the Vulkan backend.
//
// Every image lives in VK_IMAGE_LAYOUT_GENERAL for its whole life. GENERAL is valid for transfer,
// sampling, storage and attachment use, so no per-command layout tracking is needed.

struct VulkanFormatInfo {
    let vk: UInt32
    let bytes: UInt32
    var aspect: UInt32 = VK.Image.aspectColor
    /// Texels per block along each axis; `bytes` is the size of one block.
    var blockWidth: UInt32 = 1
    var blockHeight: UInt32 = 1
    /// True for any depth or stencil format.
    var isDepth: Bool { aspect != VK.Image.aspectColor }
    var hasDepth: Bool { aspect & VK.Depth.aspectDepth != 0 }
    var hasStencil: Bool { aspect & VK.Depth.aspectStencil != 0 }
    /// Aspect a shader samples: depth when present, otherwise the stencil or color aspect.
    var sampledAspect: UInt32 { hasDepth ? VK.Depth.aspectDepth : aspect }
}

func vulkanFormat(_ format: TextureFormat) -> VulkanFormatInfo? {
    switch format {
    case .r8Unorm: .init(vk: 9, bytes: 1)
    case .rg8Unorm: .init(vk: 16, bytes: 2)
    case .rgba8Unorm: .init(vk: 37, bytes: 4)
    case .rgba8Snorm: .init(vk: 38, bytes: 4)
    case .rgba8Uint: .init(vk: 41, bytes: 4)
    case .rgba8Sint: .init(vk: 42, bytes: 4)
    case .rgba8UnormSrgb: .init(vk: 43, bytes: 4)
    case .bgra8Unorm: .init(vk: 44, bytes: 4)
    case .bgra8UnormSrgb: .init(vk: 50, bytes: 4)
    case .rgba16Uint: .init(vk: 95, bytes: 8)
    case .rgba16Sint: .init(vk: 96, bytes: 8)
    case .rgba16Float: .init(vk: 97, bytes: 8)
    case .r32Uint: .init(vk: 98, bytes: 4)
    case .r32Sint: .init(vk: 99, bytes: 4)
    case .r32Float: .init(vk: 100, bytes: 4)
    case .rg32Uint: .init(vk: 101, bytes: 8)
    case .rg32Sint: .init(vk: 102, bytes: 8)
    case .rg32Float: .init(vk: 103, bytes: 8)
    case .rgba32Uint: .init(vk: 107, bytes: 16)
    case .rgba32Sint: .init(vk: 108, bytes: 16)
    case .rgba32Float: .init(vk: 109, bytes: 16)
    default: nil
    }
}

/// The `VK_FORMAT_ASTC_*_SFLOAT_BLOCK` value of an ASTC HDR format. The 14 formats are numbered from 1000066000 in
/// the same block-size order as `TextureFormat.astc4x4HDR` through `astc12x12HDR`.
func vulkanASTCHDRFormat(_ format: TextureFormat) -> UInt32? {
    (72...85).contains(format.rawValue) ? 1_000_066_000 + format.rawValue - 72 : nil
}

extension VulkanDevice {
    /// Resolves a texture format, mapping depth-stencil formats to what this device supports.
    func format(_ format: TextureFormat) -> VulkanFormatInfo? {
        let supported = native.limits.depthStencilFormats
        func pick(_ candidates: [(UInt32, UInt32, UInt32)]) -> VulkanFormatInfo? {
            candidates.first { supported.contains($0.0) }.map { .init(vk: $0.0, bytes: $0.1, aspect: $0.2) }
        }
        let depth = VK.Depth.aspectDepth, stencil = VK.Depth.aspectStencil
        if let compression = format.compression, (20...71).contains(format.rawValue) {
            let support = native.limits.featureSupport
            let supported: Bool = switch compression.feature {
            case .textureCompressionBC: support.core[22] != 0
            case .textureCompressionETC2: support.core[20] != 0
            default: support.core[21] != 0
            }
            guard supported else { return nil }
            return .init(vk: format.rawValue + 113, bytes: compression.block.bytes, blockWidth: compression.block.width,
                         blockHeight: compression.block.height)
        }
        if let compression = format.compression, let number = vulkanASTCHDRFormat(format) {
            guard capabilities.features.contains(.textureCompressionASTCHDR) else { return nil }
            return .init(vk: number, bytes: compression.block.bytes, blockWidth: compression.block.width,
                         blockHeight: compression.block.height)
        }
        switch format {
        case .depth16Unorm: return pick([(124, 2, depth)])
        case .depth32Float, .depth24Plus: return pick([(126, 4, depth)])
        case .stencil8: return pick([(127, 1, stencil)])
        case .depth24PlusStencil8: return pick([(129, 4, depth | stencil), (130, 4, depth | stencil)])
        case .depth32FloatStencil8: return pick([(130, 4, depth | stencil)])
        default: return vulkanFormat(format)
        }
    }
}

func vulkanVertexFormat(_ format: VertexFormat) -> UInt32? {
    switch format {
    case .float32: 100
    case .float32x2: 103
    case .float32x3: 106
    case .float32x4: 109
    case .uint32: 98
    case .uint32x2: 101
    case .uint32x3: 104
    case .uint32x4: 107
    case .sint32: 99
    case .sint32x2: 102
    case .sint32x3: 105
    case .sint32x4: 108
    case .float16: 76
    case .float16x2: 83
    case .float16x4: 97
    case .uint8: 13
    case .uint8x2: 20
    case .uint8x4: 41
    case .sint8: 14
    case .sint8x2: 21
    case .sint8x4: 42
    case .unorm8: 9
    case .unorm8x2: 16
    case .unorm8x4: 37
    case .snorm8: 10
    case .snorm8x2: 17
    case .snorm8x4: 38
    case .uint16: 74
    case .uint16x2: 81
    case .uint16x4: 95
    case .sint16: 75
    case .sint16x2: 82
    case .sint16x4: 96
    case .unorm16: 70
    case .unorm16x2: 77
    case .unorm16x4: 91
    case .snorm16: 71
    case .snorm16x2: 78
    case .snorm16x4: 92
    case .unorm8x4BGRA: 44
    case .unorm1010102, .snorm1010102: nil
    }
}

extension VulkanNativeDevice {
    var g: VulkanGraphicsFunctions { graphics }

    /// Picks a memory type allowed by `typeBits`, preferring device-local memory.
    func deviceMemoryType(typeBits: UInt32) -> UInt32? {
        var fallback: UInt32?
        for (index, flags) in limits.memoryTypeFlags.enumerated() where typeBits & (1 << UInt32(index)) != 0 {
            if flags & VK.MemoryProperty.deviceLocal != 0 { return UInt32(index) }
            fallback = fallback ?? UInt32(index)
        }
        return fallback
    }
}

extension VulkanDevice: RenderBackendDevice {
    struct Texture {
        var descriptor: TextureDescriptor
        var format: VulkanFormatInfo
        var image: UInt64
        var memory: UInt64
        /// Set when the value stands for a texture view: first mip, mip count and aspect.
        var baseMip: UInt32 = 0
        var viewMips: UInt32?
        var viewAspect: TextureViewAspect = .all
        var baseLayer: UInt32 = 0
        var viewLayers: UInt32?
        var viewDimension: TextureViewDimension?
    }

    struct RenderPipelineState {
        var model: RenderPipeline
        /// Every shader module of the pipeline: vertex and fragment, or task, mesh and fragment.
        var modules: [UInt64]
        var setLayouts: [UInt64]
        var layout: UInt64
        var pipeline: UInt64
        var renderPass: UInt64
        var descriptorTypes: [ComputeBindingSlot: UInt32]
    }

    enum BufferReference {
        case staging
        case handle(UInt64)
    }

    struct ImageCopyPlan {
        var toImage: Bool
        var image: UInt64
        var buffer: BufferReference
        var region: VkBufferImageCopy
    }

    struct DepthTarget {
        var image: UInt64
        var format: UInt32
        var mipLevel: UInt32
        var extent: (width: UInt32, height: UInt32)
        var loadOp: UInt32
        var clearValue: Float
        var storeOp: UInt32
        var aspect: UInt32 = VK.Depth.aspectDepth
        var stencilLoadOp: UInt32 = 2
        var stencilStoreOp: UInt32 = 1
        var stencilClear: UInt32 = 0
        var stencilReference: UInt32?
        var layer: UInt32 = 0
    }

    struct RenderPlan {
        var pipeline: RenderPipelineState
        var image: UInt64
        var format: UInt32
        var mipLevel: UInt32
        var extent: (width: UInt32, height: UInt32)
        var clear: (UInt32, UInt32, UInt32, UInt32)?
        var writes: [DescriptorWrite]
        var vertexBuffers: [(buffer: UInt64, offset: UInt64)]
        var vertexCount: UInt32
        var firstVertex: UInt32
        var instanceCount: UInt32
        var firstInstance: UInt32
        var viewport: VkViewport
        var scissor: VkRect2D
        var blendConstant: [Float]
        var immediates: [UInt32]
        var indexed: (buffer: UInt64, offset: UInt64, type: UInt32, count: UInt32, first: UInt32, baseVertex: Int32)?
        var depth: DepthTarget?
        var indirect: (source: UInt64, offset: UInt64, slot: Int)?
        var indirectArguments: (buffer: UInt64, offset: UInt64)?
        var occlusion: (pool: UInt64, index: UInt32)?
        var resolve: (image: UInt64, mipLevel: UInt32, layer: UInt32)?
        var colorStore: UInt32 = 0
        var samples: UInt32 = 1
        var layer: UInt32 = 0
        var layerCount: UInt32 = 1
        var viewMask: UInt32?
        /// Mesh task group counts; the plan then ends in vkCmdDrawMeshTasksEXT instead of a vertex draw.
        var meshGroups: (UInt32, UInt32, UInt32)?
        var meshUsesTask = false
    }

    // MARK: Textures

    /// Resolves a texture or texture view id to the texture it covers.
    func texture(for id: ResourceID) -> Texture? {
        if let texture = textures[id] { return texture }
        guard let view = textureViews[id], var texture = textures[view.texture] else { return nil }
        texture.baseMip = view.levels.lowerBound
        texture.viewMips = UInt32(view.levels.count)
        texture.viewAspect = view.aspect
        texture.baseLayer = view.layers.lowerBound
        texture.viewLayers = UInt32(view.layers.count)
        texture.viewDimension = view.dimension
        return texture
    }

    public func allocateTexture(id: ResourceID, descriptor: TextureDescriptor) async throws {
        try requireActive()
        try descriptor.validate(against: capabilities)
        guard textures[id] == nil, buffers[id] == nil else { throw GPUError.validation("Duplicate Vulkan resource") }
        guard descriptor.dimension == .d2 || (descriptor.dimension != .d2 && descriptor.sampleCount == 1),
              descriptor.sampleCount == 1 || descriptor.sampleCount == 4 else {
            throw GPUError.unsupported("Vulkan textures are limited to 2D and 3D images with one or four samples")
        }
        guard let format = self.format(descriptor.format) else {
            throw GPUError.unsupported("Vulkan does not implement texture format \(descriptor.format)")
        }
        let multisampled = descriptor.sampleCount > 1
        guard !multisampled || descriptor.usage == .renderAttachment else {
            throw GPUError.unsupported("Vulkan multisampled textures must be render attachments only")
        }
        guard !descriptor.usage.contains(.storageAtomic) else {
            throw GPUError.unsupported("Vulkan texture atomics are not implemented")
        }
        var usage: UInt32 = descriptor.sampleCount > 1 ? 0 : VK.Image.usageTransferSource | VK.Image.usageTransferDestination
        if descriptor.usage.contains(.renderAttachment) {
            usage |= format.isDepth ? VK.Depth.usageAttachment : VK.Image.usageColorAttachment
        }
        if descriptor.usage.contains(.textureBinding) { usage |= VK.Image.usageSampled }
        if descriptor.usage.contains(.storageBinding) {
            // Only these formats are guaranteed to support storage use without optional device features.
            switch descriptor.format {
            case .rgba8Unorm, .rgba8Snorm, .rgba8Uint, .rgba8Sint, .rgba16Uint, .rgba16Sint, .rgba16Float,
                 .r32Uint, .r32Sint, .r32Float, .rg32Uint, .rg32Sint, .rg32Float, .rgba32Uint, .rgba32Sint, .rgba32Float:
                usage |= VK.Image.usageStorage
            default:
                throw GPUError.unsupported("Vulkan storage textures require a core storage format")
            }
        }
        let f = native.f, g = native.g, device = native.device
        var info = VkImageCreateInfo()
        info.format = format.vk
        info.extent = VkExtent3D(width: descriptor.width, height: descriptor.height,
                                 depth: descriptor.dimension == .d3 ? descriptor.depth : 1)
        if descriptor.dimension == .d3 { info.imageType = 2 }
        if descriptor.dimension == .d1 { info.imageType = 0 }
        if descriptor.dimension == .d3, descriptor.usage.contains(.renderAttachment) { info.flags |= 0x20 }  // 2D-array compatible
        if descriptor.dimension == .d2, descriptor.width == descriptor.height, descriptor.arrayLayerCount >= 6,
           descriptor.arrayLayerCount % 6 == 0 {
            info.flags = 0x10  // cube compatible
        }
        info.mipLevels = descriptor.mipLevelCount
        info.arrayLayers = descriptor.arrayLayerCount
        info.usage = usage
        info.samples = descriptor.sampleCount
        var image: UInt64 = 0
        try vulkanCheck(withUnsafePointer(to: &info) { g.createImage(device, UnsafeRawPointer($0), nil, &image) }, "image creation")
        var requirements = VkMemoryRequirements()
        g.getImageMemoryRequirements(device, image, &requirements)
        guard let type = native.deviceMemoryType(typeBits: requirements.memoryTypeBits) else {
            g.destroyImage(device, image, nil)
            throw GPUError.unsupported("Vulkan device has no memory for this image")
        }
        var allocation = VkMemoryAllocateInfo()
        allocation.allocationSize = requirements.size
        allocation.memoryTypeIndex = type
        var memory: UInt64 = 0
        let allocated = withUnsafePointer(to: &allocation) { f.allocateMemory(device, UnsafeRawPointer($0), nil, &memory) }
        guard allocated == VK.success else {
            g.destroyImage(device, image, nil)
            try vulkanCheck(allocated, "image memory allocation")
            throw GPUError.outOfMemory
        }
        let bound = g.bindImageMemory(device, image, memory, 0)
        guard bound == VK.success else {
            g.destroyImage(device, image, nil)
            f.freeMemory(device, memory, nil)
            try vulkanCheck(bound, "image memory binding")
            return
        }
        // Zero-initialize: move to GENERAL and clear every subresource.
        let range = VkImageSubresourceRange(aspectMask: format.aspect, levelCount: descriptor.mipLevelCount,
                                            layerCount: descriptor.arrayLayerCount)
        var initView: UInt64 = 0, initPass: UInt64 = 0, initFramebuffer: UInt64 = 0
        var initScratch: VulkanHostBuffer?
        let blockCompressed = format.blockWidth > 1 || format.blockHeight > 1
        func releaseInit() {
            if initFramebuffer != 0 { g.destroyFramebuffer(device, initFramebuffer, nil) }
            if initPass != 0 { g.destroyRenderPass(device, initPass, nil) }
            if initView != 0 { g.destroyImageView(device, initView, nil) }
            if let initScratch { native.destroy(initScratch) }
        }
        var zeroRegions: [VkBufferImageCopy] = []
        if blockCompressed {
            // Compressed images cannot be cleared; copy zero blocks into every mip and layer instead.
            var offset: UInt64 = 0
            for mip in 0..<descriptor.mipLevelCount {
                guard let size = descriptor.mipExtent(mip) else { continue }
                var region = VkBufferImageCopy()
                region.bufferOffset = offset
                region.mipLevel = mip
                region.layerCount = descriptor.arrayLayerCount
                region.imageExtent = VkExtent3D(width: size.width, height: size.height, depth: 1)
                zeroRegions.append(region)
                let blocks = UInt64((size.width + format.blockWidth - 1) / format.blockWidth)
                    * UInt64((size.height + format.blockHeight - 1) / format.blockHeight)
                offset += blocks * UInt64(format.bytes) * UInt64(descriptor.arrayLayerCount)
            }
            do { initScratch = try native.makeHostBuffer(size: max(offset, 4)) } catch {
                g.destroyImage(device, image, nil)
                f.freeMemory(device, memory, nil)
                throw error
            }
        }
        let scratchForCopy: UInt64? = initScratch?.handle
        let regionsForCopy = zeroRegions
        do {
            try await runOneShot { commandBuffer in
                var barrier = VkImageMemoryBarrier()
                barrier.dstAccessMask = VK.accessMemoryRead | VK.accessMemoryWrite
                barrier.oldLayout = VK.Image.layoutUndefined
                barrier.newLayout = VK.Image.layoutGeneral
                barrier.image = image
                barrier.subresourceRange = range
                withUnsafePointer(to: &barrier) {
                    f.cmdPipelineBarrier(commandBuffer, VK.Stage.pipelineTopOfPipe, VK.pipelineStageAllCommands, 0,
                                         0, nil, 0, nil, 1, UnsafeRawPointer($0))
                }
                var rangeCopy = range
                if let scratch = scratchForCopy {
                    regionsForCopy.withUnsafeBufferPointer {
                        g.cmdCopyBufferToImage(commandBuffer, scratch, image, VK.Image.layoutGeneral,
                                               UInt32($0.count), UnsafeRawPointer($0.baseAddress!))
                    }
                } else if multisampled {
                    // Clear commands do not accept multisampled images; clear through a render pass instead.
                    var viewInfo = VkImageViewCreateInfo()
                    viewInfo.image = image
                    viewInfo.format = format.vk
                    viewInfo.subresourceRange = VkImageSubresourceRange(aspectMask: format.aspect)
                    try vulkanCheck(withUnsafePointer(to: &viewInfo) { g.createImageView(device, UnsafeRawPointer($0), nil, &initView) },
                                    "image view creation")
                    initPass = format.isDepth
                        ? try makeDepthOnlyRenderPass(format: format.vk, samples: descriptor.sampleCount)
                        : try makeRenderPass(format: format.vk, load: 1, samples: descriptor.sampleCount)
                    var attachments = [initView]
                    try attachments.withUnsafeMutableBufferPointer { pointer in
                        var framebufferInfo = VkFramebufferCreateInfo()
                        framebufferInfo.renderPass = initPass
                        framebufferInfo.attachmentCount = 1
                        framebufferInfo.pAttachments = UnsafePointer(pointer.baseAddress)
                        framebufferInfo.width = descriptor.width
                        framebufferInfo.height = descriptor.height
                        try vulkanCheck(withUnsafePointer(to: &framebufferInfo) {
                            g.createFramebuffer(device, UnsafeRawPointer($0), nil, &initFramebuffer)
                        }, "framebuffer creation")
                    }
                    var clearValue = VkClearColorValue()
                    var begin = VkRenderPassBeginInfo()
                    begin.renderPass = initPass
                    begin.framebuffer = initFramebuffer
                    begin.areaWidth = descriptor.width
                    begin.areaHeight = descriptor.height
                    begin.clearValueCount = 1
                    withUnsafePointer(to: &clearValue) { clearPointer in
                        begin.pClearValues = UnsafeRawPointer(clearPointer)
                        withUnsafePointer(to: &begin) { g.cmdBeginRenderPass(commandBuffer, UnsafeRawPointer($0), VK.subpassContentsInline) }
                    }
                    g.cmdEndRenderPass(commandBuffer)
                } else if format.isDepth {
                    var clear = VkClearDepthStencilValue()
                    g.cmdClearDepthStencilImage(commandBuffer, image, VK.Image.layoutGeneral, &clear, 1, &rangeCopy)
                } else {
                    var clear = VkClearColorValue()
                    g.cmdClearColorImage(commandBuffer, image, VK.Image.layoutGeneral, &clear, 1, &rangeCopy)
                }
            }
        } catch {
            if case GPUError.timeout = error { return }  // quarantined by runOneShot; the device is terminal
            releaseInit()
            g.destroyImage(device, image, nil)
            f.freeMemory(device, memory, nil)
            throw error
        }
        releaseInit()
        textures[id] = Texture(descriptor: descriptor, format: format, image: image, memory: memory)
    }

    public func destroyTexture(id: ResourceID) async {
        guard let texture = textures.removeValue(forKey: id) else { return }
        native.g.destroyImage(native.device, texture.image, nil)
        native.f.freeMemory(native.device, texture.memory, nil)
    }

    /// Records and runs a command buffer to completion.
    func runOneShot(waiting waitSemaphore: UInt64 = 0, signaling signalSemaphore: UInt64 = 0,
                    _ record: (UInt) throws -> Void) async throws {
        let f = native.f, device = native.device
        var pool: UInt64 = 0
        var fence: UInt64 = 0
        var poolInfo = VkCommandPoolCreateInfo()
        poolInfo.queueFamilyIndex = native.queueFamily
        try vulkanCheck(withUnsafePointer(to: &poolInfo) { f.createCommandPool(device, UnsafeRawPointer($0), nil, &pool) },
                        "command pool creation")
        func cleanup() {
            if fence != 0 { f.destroyFence(device, fence, nil) }
            f.destroyCommandPool(device, pool, nil)
        }
        var commandBuffer: UInt = 0
        do {
            var allocate = VkCommandBufferAllocateInfo()
            allocate.commandPool = pool
            try vulkanCheck(withUnsafePointer(to: &allocate) { f.allocateCommandBuffers(device, UnsafeRawPointer($0), &commandBuffer) },
                            "command buffer allocation")
            var begin = VkCommandBufferBeginInfo()
            try vulkanCheck(withUnsafePointer(to: &begin) { f.beginCommandBuffer(commandBuffer, UnsafeRawPointer($0)) },
                            "command recording")
            try record(commandBuffer)
            try vulkanCheck(f.endCommandBuffer(commandBuffer), "command recording")
            var fenceInfo = VkFenceCreateInfo()
            try vulkanCheck(withUnsafePointer(to: &fenceInfo) { f.createFence(device, UnsafeRawPointer($0), nil, &fence) },
                            "fence creation")
            var submit = VkSubmitInfo()
            submit.commandBufferCount = 1
            let queue = native.queue
            let fenceHandle = fence
            var waitHandle = waitSemaphore, signalHandle = signalSemaphore
            var waitStage = VKPresent.stageTransfer
            if waitSemaphore != 0 {
                submit.waitSemaphoreCount = 1
            }
            if signalSemaphore != 0 { submit.signalSemaphoreCount = 1 }
            try withUnsafePointer(to: &commandBuffer) { pointer in
                submit.pCommandBuffers = pointer
                try withUnsafePointer(to: &waitHandle) { waitPointer in
                    try withUnsafePointer(to: &signalHandle) { signalPointer in
                        if waitSemaphore != 0 { submit.pWaitSemaphores = UnsafeRawPointer(waitPointer) }
                        if signalSemaphore != 0 { submit.pSignalSemaphores = UnsafeRawPointer(signalPointer) }
                        try withUnsafePointer(to: &waitStage) { stagePointer in
                            if waitSemaphore != 0 { submit.pWaitDstStageMask = UnsafeRawPointer(stagePointer) }
                            try vulkanCheck(withUnsafePointer(to: &submit) { f.queueSubmit(queue, 1, UnsafeRawPointer($0), fenceHandle) },
                                            "queue submission")
                        }
                    }
                }
            }
        } catch {
            cleanup()
            throw error
        }
        do {
            try await waitForFence(fence)
        } catch {
            terminal = true  // the GPU may still own the pool; leak it
            throw error
        }
        cleanup()
    }

    // MARK: Render pipelines

    /// One shader stage of a graphics pipeline.
    struct GraphicsStage {
        var stage: UInt32
        var words: [UInt32]
        var entryPoint: String
    }

    public func allocateRenderPipeline(_ pipeline: RenderPipeline, vertex: RenderStageArtifact,
                                       fragment: RenderStageArtifact) async throws {
        try requireActive()
        guard case .spirv(let vertexWords) = vertex.artifact, case .spirv(let fragmentWords) = fragment.artifact else {
            throw GPUError.validation("Vulkan requires SPIR-V render stages")
        }
        try buildGraphicsPipeline(pipeline, stages: [
            GraphicsStage(stage: VK.Stage.vertex, words: vertexWords, entryPoint: vertex.entryPoint),
            GraphicsStage(stage: VK.Stage.fragment, words: fragmentWords, entryPoint: fragment.entryPoint)], meshStages: false)
    }

    /// Builds a graphics pipeline from vertex and fragment stages, or from mesh (and optional task) and fragment stages.
    func buildGraphicsPipeline(_ pipeline: RenderPipeline, stages stageSpecs: [GraphicsStage], meshStages: Bool) throws {
        try requireActive()
        guard renderPipelines[pipeline.id] == nil else { throw GPUError.validation("Duplicate Vulkan render pipeline") }
        guard let colorFormat = pipeline.colorFormat, let format = self.format(colorFormat) else {
            throw GPUError.unsupported("Vulkan render pipelines require a supported color format")
        }
        var depthFormat: VulkanFormatInfo?
        if let depthState = pipeline.depthState {
            guard let info = self.format(depthState.format), info.isDepth, depthState.bias.clamp == 0 else {
                throw GPUError.unsupported("Vulkan device lacks this depth format, or the depth bias clamp is requested")
            }
            depthFormat = info
        }
        guard pipeline.sampleCount == 1 || (pipeline.sampleCount == 4),
                            pipeline.additionalColorTargets.isEmpty, !pipeline.primitive.unclippedDepth,
              pipeline.storageTextures.isEmpty,
              !pipeline.colorState.usesSecondBlendSource else {
            throw GPUError.unsupported("Vulkan render pipelines support one single-sampled color target with buffer bindings only")
        }
        guard pipeline.immediateByteSize <= UInt64(capabilities.maxImmediateSize) else {
            throw GPUError.validation("Immediate data exceeds the Vulkan push constant limit")
        }
        var types: [ComputeBindingSlot: UInt32] = [:]
        for uniform in pipeline.uniforms { types[uniform.slot] = VK.DescriptorType.uniformBuffer }
        for storage in pipeline.storageBuffers { types[storage.handle.slot] = VK.DescriptorType.storageBuffer }
        types.merge(try vulkanHandleDescriptorTypes(sampled: pipeline.sampledTextures.map { $0.requirement }, storage: [], samplers: pipeline.samplerBindings.map { $0.requirement })) { first, _ in first }
        types.merge(vulkanExternalDescriptorTypes(pipeline.externalTextureBindings.map(\.slot))) { first, _ in first }
        guard pipeline.accelerationStructureBindings.isEmpty || native.ray != nil else {
            throw GPUError.unsupported("This Vulkan device does not support ray queries")
        }
        for binding in pipeline.accelerationStructureBindings { types[binding.slot] = Self.accelerationDescriptorType }
        guard types.keys.allSatisfy({ $0.group < min(native.limits.maxBoundDescriptorSets, 4) }) else {
            throw GPUError.unsupported("Vulkan binding arrays are not implemented")
        }
        let f = native.f, g = native.g, device = native.device
        var modules: [UInt64] = []
        var layout: UInt64 = 0, renderPass: UInt64 = 0, handle: UInt64 = 0
        var setLayouts: [UInt64] = []
        func cleanup() {
            if handle != 0 { f.destroyPipeline(device, handle, nil) }
            if renderPass != 0 { g.destroyRenderPass(device, renderPass, nil) }
            if layout != 0 { f.destroyPipelineLayout(device, layout, nil) }
            setLayouts.forEach { f.destroyDescriptorSetLayout(device, $0, nil) }
            modules.forEach { f.destroyShaderModule(device, $0, nil) }
        }
        func makeModule(_ words: [UInt32]) throws -> UInt64 {
            var module: UInt64 = 0
            try words.withUnsafeBufferPointer { code in
                var info = VkShaderModuleCreateInfo()
                info.codeSize = code.count * 4
                info.pCode = code.baseAddress
                try vulkanCheck(withUnsafePointer(to: &info) { f.createShaderModule(device, UnsafeRawPointer($0), nil, &module) },
                                "shader module creation")
            }
            return module
        }
        do {
            for spec in stageSpecs { modules.append(try makeModule(spec.words)) }
            let stages = stageSpecs.reduce(0) { $0 | $1.stage }
            let setCount = (types.keys.map(\.group).max()).map { Int($0) + 1 } ?? 0
            for group in 0..<setCount {
                let bindings = vulkanLayoutBindings(types: types, group: group, stages: stages)
                var setLayout: UInt64 = 0
                try bindings.withUnsafeBufferPointer { pointer in
                    var info = VkDescriptorSetLayoutCreateInfo()
                    info.bindingCount = UInt32(pointer.count)
                    info.pBindings = pointer.baseAddress
                    try vulkanCheck(withUnsafePointer(to: &info) {
                        f.createDescriptorSetLayout(device, UnsafeRawPointer($0), nil, &setLayout)
                    }, "descriptor set layout creation")
                }
                setLayouts.append(setLayout)
            }
            var pushRange = VkPushConstantRange(stageFlags: stages, size: UInt32(pipeline.immediateByteSize))
            try setLayouts.withUnsafeBufferPointer { pointer in
                var info = VkPipelineLayoutCreateInfo()
                info.setLayoutCount = UInt32(pointer.count)
                info.pSetLayouts = pointer.baseAddress
                try withUnsafePointer(to: &pushRange) { range in
                    if range.pointee.size > 0 {
                        info.pushConstantRangeCount = 1
                        info.pPushConstantRanges = UnsafeRawPointer(range)
                    }
                    try vulkanCheck(withUnsafePointer(to: &info) {
                        f.createPipelineLayout(device, UnsafeRawPointer($0), nil, &layout)
                    }, "pipeline layout creation")
                }
            }
            // Multisampled pipelines always resolve, so every compatible pass carries a resolve attachment.
            renderPass = try makeRenderPass(format: format.vk, load: 0, depthFormat: depthFormat?.vk,
                                            samples: pipeline.sampleCount, resolve: pipeline.sampleCount > 1,
                                            viewMask: pipeline.multiviewMask)

            var bindingDescriptions: [VkVertexInputBindingDescription] = []
            var attributeDescriptions: [VkVertexInputAttributeDescription] = []
            for (index, buffer) in pipeline.vertexBuffers.enumerated() {
                bindingDescriptions.append(.init(binding: UInt32(index), stride: buffer.stride,
                                                 inputRate: buffer.stepMode == .instance ? 1 : 0))
                for attribute in buffer.attributes {
                    guard let attributeFormat = vulkanVertexFormat(attribute.format) else {
                        throw GPUError.unsupported("Vulkan does not implement vertex format \(attribute.format)")
                    }
                    attributeDescriptions.append(.init(location: attribute.location, binding: UInt32(index),
                                                       format: attributeFormat, offset: attribute.offset))
                }
            }
            try withCStringArray(stageSpecs.map(\.entryPoint)) { entryNames in
                try ({
                    var stageInfos = [VkPipelineShaderStageCreateInfo](repeating: VkPipelineShaderStageCreateInfo(), count: stageSpecs.count)
                    for (index, spec) in stageSpecs.enumerated() {
                        stageInfos[index].stage = spec.stage
                        stageInfos[index].module = modules[index]
                        stageInfos[index].pName = entryNames![index]
                    }
                    var topology: UInt32
                    switch pipeline.primitive.topology {
                    case .pointList: topology = 0
                    case .lineList: topology = 1
                    case .lineStrip: topology = 2
                    case .triangleList: topology = 3
                    case .triangleStrip: topology = 4
                    }
                    var inputAssembly = VkPipelineInputAssemblyStateCreateInfo(topology: topology)
                    var viewport = VkPipelineViewportStateCreateInfo()
                    var raster = VkPipelineRasterizationStateCreateInfo()
                    raster.cullMode = switch pipeline.primitive.cullMode { case .none: 0; case .front: 1; case .back: 2 }
                    raster.frontFace = pipeline.primitive.frontFace == .clockwise ? 1 : 0
                    raster.polygonMode = switch pipeline.primitive.polygonMode { case .fill: 0; case .line: 1; case .point: 2 }
                    var depthStencil = VkPipelineDepthStencilStateCreateInfo()
                    if let depthState = pipeline.depthState {
                        depthStencil.depthTestEnable = depthFormat?.hasDepth == true ? 1 : 0
                        if let stencil = depthState.stencil {
                            func face(_ state: RenderStencilFaceState) -> VkStencilOpState {
                                VkStencilOpState(failOp: vulkanStencilOp(state.fail), passOp: vulkanStencilOp(state.pass),
                                                 depthFailOp: vulkanStencilOp(state.depthFail), compareOp: state.compare.rawValue,
                                                 compareMask: stencil.readMask, writeMask: stencil.writeMask)
                            }
                            depthStencil.stencilTestEnable = 1
                            depthStencil.front = face(stencil.front)
                            depthStencil.back = face(stencil.back)
                        }
                        depthStencil.depthWriteEnable = depthState.writeEnabled ? 1 : 0
                        depthStencil.depthCompareOp = depthState.compare.rawValue
                        if depthState.bias.constant != 0 || depthState.bias.slopeScale != 0 {
                            raster.depthBiasEnable = 1
                            raster.depthBiasConstantFactor = Float(depthState.bias.constant)
                            raster.depthBiasSlopeFactor = depthState.bias.slopeScale
                        }
                    }
                    let depthPointer = UnsafeMutablePointer<VkPipelineDepthStencilStateCreateInfo>.allocate(capacity: 1)
                    depthPointer.initialize(to: depthStencil)
                    defer { depthPointer.deallocate() }
                    var multisample = VkPipelineMultisampleStateCreateInfo()
                    multisample.rasterizationSamples = pipeline.sampleCount
                    multisample.alphaToCoverageEnable = pipeline.alphaToCoverageEnabled ? 1 : 0
                    let maskPointer = UnsafeMutablePointer<UInt32>.allocate(capacity: 1)
                    maskPointer.initialize(to: pipeline.sampleMask)
                    defer { maskPointer.deallocate() }
                    multisample.pSampleMask = UnsafeRawPointer(maskPointer)
                    var attachment = VkPipelineColorBlendAttachmentState()
                    attachment.colorWriteMask = pipeline.colorState.writeMask.rawValue
                    if let blend = pipeline.colorState.blend {
                        attachment.blendEnable = 1
                        attachment.srcColorBlendFactor = vulkanBlendFactor(blend.color.sourceFactor)
                        attachment.dstColorBlendFactor = vulkanBlendFactor(blend.color.destinationFactor)
                        attachment.colorBlendOp = blend.color.operation.rawValue
                        attachment.srcAlphaBlendFactor = vulkanBlendFactor(blend.alpha.sourceFactor)
                        attachment.dstAlphaBlendFactor = vulkanBlendFactor(blend.alpha.destinationFactor)
                        attachment.alphaBlendOp = blend.alpha.operation.rawValue
                    }
                    var blendState = VkPipelineColorBlendStateCreateInfo()
                    blendState.attachmentCount = 1
                    var dynamicStates: [UInt32] = [VK.dynamicStateViewport, VK.dynamicStateScissor, VK.dynamicStateBlendConstants]
                    if pipeline.depthState?.stencil != nil { dynamicStates.append(VK.dynamicStateStencilReference) }
                    try bindingDescriptions.withUnsafeBufferPointer { bindingPointer in
                        try attributeDescriptions.withUnsafeBufferPointer { attributePointer in
                            try dynamicStates.withUnsafeBufferPointer { dynamicPointer in
                                try withUnsafePointer(to: &attachment) { attachmentPointer in
                                    blendState.pAttachments = UnsafeRawPointer(attachmentPointer)
                                    var vertexInput = VkPipelineVertexInputStateCreateInfo()
                                    vertexInput.vertexBindingDescriptionCount = UInt32(bindingPointer.count)
                                    vertexInput.pVertexBindingDescriptions = bindingPointer.baseAddress.map(UnsafeRawPointer.init)
                                    vertexInput.vertexAttributeDescriptionCount = UInt32(attributePointer.count)
                                    vertexInput.pVertexAttributeDescriptions = attributePointer.baseAddress.map(UnsafeRawPointer.init)
                                    var dynamic = VkPipelineDynamicStateCreateInfo()
                                    dynamic.dynamicStateCount = UInt32(dynamicPointer.count)
                                    dynamic.pDynamicStates = dynamicPointer.baseAddress
                                    try stageInfos.withUnsafeBufferPointer { stagePointer in
                                        try withUnsafePointer(to: &vertexInput) { vi in
                                        try withUnsafePointer(to: &inputAssembly) { ia in
                                        try withUnsafePointer(to: &viewport) { vp in
                                        try withUnsafePointer(to: &raster) { rs in
                                        try withUnsafePointer(to: &multisample) { ms in
                                        try withUnsafePointer(to: &blendState) { cb in
                                        try withUnsafePointer(to: &dynamic) { dy in
                                            var info = VkGraphicsPipelineCreateInfo()
                                            info.stageCount = UInt32(stageInfos.count)
                                            info.pStages = UnsafeRawPointer(stagePointer.baseAddress)
                                            // Mesh pipelines take no vertex input or input assembly.
                                            info.pVertexInputState = meshStages ? nil : UnsafeRawPointer(vi)
                                            info.pInputAssemblyState = meshStages ? nil : UnsafeRawPointer(ia)
                                            info.pViewportState = UnsafeRawPointer(vp)
                                            info.pRasterizationState = UnsafeRawPointer(rs)
                                            info.pMultisampleState = UnsafeRawPointer(ms)
                                            info.pColorBlendState = UnsafeRawPointer(cb)
                                            info.pDepthStencilState = pipeline.depthState == nil ? nil : UnsafeRawPointer(depthPointer)
                                            info.pDynamicState = UnsafeRawPointer(dy)
                                            info.layout = layout
                                            info.renderPass = renderPass
                                            try vulkanCheck(withUnsafePointer(to: &info) {
                                                g.createGraphicsPipelines(device, 0, 1, UnsafeRawPointer($0), nil, &handle)
                                            }, "graphics pipeline creation")
                                        }}}}}}}
                                    }
                                }
                            }
                        }
                    }
                } as () throws -> Void)()
            }
        } catch {
            cleanup()
            throw error
        }
        renderPipelines[pipeline.id] = RenderPipelineState(
            model: pipeline, modules: modules, setLayouts: setLayouts,
            layout: layout, pipeline: handle, renderPass: renderPass, descriptorTypes: types)
    }

    public func destroyRenderPipeline(id: ResourceID) async {
        guard let state = renderPipelines.removeValue(forKey: id) else { return }
        let f = native.f
        f.destroyPipeline(native.device, state.pipeline, nil)
        native.g.destroyRenderPass(native.device, state.renderPass, nil)
        f.destroyPipelineLayout(native.device, state.layout, nil)
        state.setLayouts.forEach { f.destroyDescriptorSetLayout(native.device, $0, nil) }
        state.modules.forEach { f.destroyShaderModule(native.device, $0, nil) }
    }

    /// A single-subpass render pass with one color attachment that stays in GENERAL layout.
    func makeRenderPass(format: UInt32, load: UInt32, depthFormat: UInt32? = nil, depthLoad: UInt32 = 0,
                        depthStore: UInt32 = 0, stencilLoad: UInt32 = 2, stencilStore: UInt32 = 1,
                        samples: UInt32 = 1, resolve: Bool = false, colorStore: UInt32 = 0,
                        viewMask: UInt32? = nil) throws -> UInt64 {
        var attachments = [VkAttachmentDescription()]
        attachments[0].format = format
        attachments[0].loadOp = load
        attachments[0].storeOp = colorStore
        attachments[0].samples = samples
        if let depthFormat {
            var depth = VkAttachmentDescription()
            depth.format = depthFormat
            depth.loadOp = depthLoad
            depth.storeOp = depthStore
            depth.stencilLoadOp = stencilLoad
            depth.stencilStoreOp = stencilStore
            depth.samples = samples
            attachments.append(depth)
        }
        var resolveReference = VkAttachmentReference(attachment: UInt32(attachments.count))
        if resolve {
            var target = VkAttachmentDescription()
            target.format = format
            target.loadOp = 2
            attachments.append(target)
        }
        var reference = VkAttachmentReference()
        var depthReference = VkAttachmentReference(attachment: 1)
        var renderPass: UInt64 = 0
        let g = native.g, device = native.device
        try withUnsafePointer(to: &reference) { referencePointer in
            var subpass = VkSubpassDescription()
            subpass.colorAttachmentCount = 1
            subpass.pColorAttachments = UnsafeRawPointer(referencePointer)
            try withUnsafePointer(to: &resolveReference) { resolvePointer in
            if resolve { subpass.pResolveAttachments = UnsafeRawPointer(resolvePointer) }
            try withUnsafePointer(to: &depthReference) { depthPointer in
            if depthFormat != nil { subpass.pDepthStencilAttachment = UnsafeRawPointer(depthPointer) }
            try attachments.withUnsafeBufferPointer { attachmentPointer in
                try withUnsafePointer(to: &subpass) { subpassPointer in
                    var info = VkRenderPassCreateInfo()
                    info.attachmentCount = UInt32(attachmentPointer.count)
                    info.pAttachments = UnsafeRawPointer(attachmentPointer.baseAddress)
                    info.subpassCount = 1
                    info.pSubpasses = UnsafeRawPointer(subpassPointer)
                    var mask = viewMask ?? 0
                    var multiview = VkRenderPassMultiviewCreateInfo()
                    try withUnsafePointer(to: &mask) { maskPointer in
                        multiview.subpassCount = 1
                        multiview.pViewMasks = maskPointer
                        try withUnsafePointer(to: &multiview) { multiviewPointer in
                            if viewMask != nil { info.pNext = UnsafeRawPointer(multiviewPointer) }
                            try vulkanCheck(withUnsafePointer(to: &info) {
                                g.createRenderPass(device, UnsafeRawPointer($0), nil, &renderPass)
                            }, "render pass creation")
                        }
                    }
                }
            }
            }
            }
        }
        return renderPass
    }

    // MARK: Command planning

    func planRender(_ command: BufferCommand, totals: inout DescriptorTotals) throws -> RenderPlan {
        guard case .render(let pipeline, let target, let colorLoad, let vertexCount, let vertexBuffers, let firstVertex,
                           let instances, let uniforms, let sampledTextures, let samplers, let mipLevel, let depthSlice,
                           let depth, let resolve, let blendConstant, let additionalColors, let raster, let storageBuffers,
                           let immediates, let colorStore) = command,
              let state = renderPipelines[pipeline.id], state.model == pipeline else {
            throw GPUError.validation("Unknown Vulkan render pipeline")
        }
        guard additionalColors.isEmpty else {
            throw GPUError.unsupported("Vulkan render commands support one color target")
        }
        guard pipeline.sampleCount > 1 || resolve == nil else {
            throw GPUError.validation("Single-sampled rendering has no resolve target")
        }
        guard colorStore == .store || resolve != nil else {
            throw GPUError.unsupported("Vulkan discards the color target only when it resolves")
        }
        var resolveTarget: (image: UInt64, mipLevel: UInt32, layer: UInt32)?
        if let resolve {
            guard let destination = self.texture(for: resolve.texture), destination.descriptor.sampleCount == 1,
                  destination.descriptor.format == pipeline.colorFormat,
                  destination.descriptor.usage.contains(.renderAttachment),
                  let size = destination.descriptor.mipExtent(resolve.mipLevel + destination.baseMip),
                  let source = target.flatMap({ self.texture(for: $0) }), let sourceSize = source.descriptor.mipExtent(mipLevel + source.baseMip),
                  size.width == sourceSize.width, size.height == sourceSize.height else {
                throw GPUError.validation("Invalid Vulkan resolve target")
            }
            resolveTarget = (destination.image, resolve.mipLevel + destination.baseMip, destination.baseLayer)
        }
        var depthTarget: DepthTarget?
        if let depth {
            guard let depthState = pipeline.depthState, let attachment = self.texture(for: depth.texture),
                  attachment.descriptor.format == depthState.format, attachment.format.isDepth,
                  attachment.descriptor.usage.contains(.renderAttachment),
                  attachment.descriptor.sampleCount == pipeline.sampleCount,
                  attachment.format.hasStencil == (depth.stencil != nil),
                  let size = attachment.descriptor.mipExtent(depth.mipLevel + attachment.baseMip) else {
                throw GPUError.validation("Invalid Vulkan depth attachment")
            }
            var loadOp: UInt32 = 2
            var storeOp: UInt32 = 1
            var clearValue: Float = 0
            if attachment.format.hasDepth {
                switch depth.loadAction {
                case .load, .readOnly: loadOp = 0; storeOp = depth.storeAction == .discard ? 1 : 0
                case .clear(let value): loadOp = 1; clearValue = Float(value); storeOp = depth.storeAction == .discard ? 1 : 0
                case .unused: break
                }
            }
            var target = DepthTarget(image: attachment.image, format: attachment.format.vk, mipLevel: depth.mipLevel + attachment.baseMip,
                                     extent: (size.width, size.height), loadOp: loadOp, clearValue: clearValue,
                                     storeOp: storeOp, aspect: attachment.format.aspect, layer: attachment.baseLayer)
            if let stencil = depth.stencil {
                switch stencil.loadAction {
                case .load, .readOnly: target.stencilLoadOp = 0
                case .clear(let value): target.stencilLoadOp = 1; target.stencilClear = value
                }
                target.stencilStoreOp = stencil.storeAction == .discard ? 1 : 0
                target.stencilReference = stencil.reference
            }
            depthTarget = target
        } else if pipeline.depthState != nil {
            throw GPUError.validation("Vulkan render pipeline requires a depth attachment")
        }
        guard let target, let texture = self.texture(for: target), texture.descriptor.format == pipeline.colorFormat,
              texture.descriptor.sampleCount == pipeline.sampleCount, texture.descriptor.usage.contains(.renderAttachment), let extent = texture.descriptor.mipExtent(mipLevel + texture.baseMip) else {
            throw GPUError.validation("Invalid Vulkan render target")
        }
        guard UInt64(immediates.count) * 4 == (pipeline.immediateByteSize + 3) / 4 * 4 else {
            throw GPUError.validation("Immediate data size does not match the render pipeline")
        }
        var merged = storageBuffers
        merged.merge(uniforms) { first, _ in first }
        let writes = try planBindings(
            types: state.descriptorTypes,
            buffers: pipeline.uniforms.map { ($0.slot, BufferUsage.uniform) }
                + pipeline.storageBuffers.map { ($0.handle.slot, BufferUsage.storage) },
            suppliedBuffers: merged, sampledTextures: pipeline.sampledTextures.map { $0.handle.slot },
            storageTextures: [], samplerSlots: pipeline.samplerBindings.map { $0.handle.slot },
            textureResources: sampledTextures, samplerResources: samplers,
            externalSlots: pipeline.externalTextureBindings.map(\.slot),
            accelerationSlots: pipeline.accelerationStructureBindings.map(\.slot), accelerationResources: sampledTextures, totals: &totals)
        var vertexTargets: [(buffer: UInt64, offset: UInt64)?] = Array(repeating: nil, count: pipeline.vertexBuffers.count)
        for (index, binding) in vertexBuffers.enumerated() {
            let slot = Int(binding.slot ?? UInt32(index))
            guard slot < vertexTargets.count, vertexTargets[slot] == nil, let buffer = buffers[binding.buffer],
                  binding.offset % 4 == 0, binding.offset <= buffer.descriptor.size else {
                throw GPUError.validation("Invalid Vulkan vertex buffer binding")
            }
            vertexTargets[slot] = (buffer.native.handle, binding.offset)
        }
        guard vertexTargets.allSatisfy({ $0 != nil }) else { throw GPUError.validation("Missing Vulkan vertex buffer binding") }
        var clear: (UInt32, UInt32, UInt32, UInt32)?
        if case .clear(let color) = colorLoad {
            guard color.isValid(for: pipeline.colorFormat ?? .rgba8Unorm) else { throw GPUError.validation("Invalid clear color") }
            switch texture.descriptor.format.sampleType {
            case .uint:
                clear = (UInt32(color.red), UInt32(color.green), UInt32(color.blue), UInt32(color.alpha))
            case .sint:
                clear = (UInt32(bitPattern: Int32(color.red)), UInt32(bitPattern: Int32(color.green)),
                         UInt32(bitPattern: Int32(color.blue)), UInt32(bitPattern: Int32(color.alpha)))
            default:
                clear = (Float(color.red).bitPattern, Float(color.green).bitPattern, Float(color.blue).bitPattern,
                         Float(color.alpha).bitPattern)
            }
        }
        if let depthTarget {
            guard depthTarget.extent == (extent.width, extent.height) else {
                throw GPUError.validation("Depth attachment extent must match the color target")
            }
        }
        var viewport = VkViewport(x: 0, y: 0, width: Float(extent.width), height: Float(extent.height))
        if let v = raster.viewport {
            viewport = VkViewport(x: Float(v.x), y: Float(v.y), width: Float(v.width), height: Float(v.height),
                                  minDepth: Float(v.minDepth), maxDepth: Float(v.maxDepth))
        }
        var scissor = VkRect2D(width: extent.width, height: extent.height)
        if let s = raster.scissor { scissor = VkRect2D(x: Int32(s.x), y: Int32(s.y), width: s.width, height: s.height) }
        return RenderPlan(
            pipeline: state, image: texture.image, format: texture.format.vk, mipLevel: mipLevel + texture.baseMip,
            extent: (extent.width, extent.height), clear: clear, writes: writes,
            vertexBuffers: vertexTargets.map { $0! }, vertexCount: vertexCount, firstVertex: firstVertex,
            instanceCount: instances.count, firstInstance: instances.first, viewport: viewport, scissor: scissor,
            blendConstant: [Float(blendConstant.red), Float(blendConstant.green), Float(blendConstant.blue),
                            Float(blendConstant.alpha)],
            immediates: immediates, indexed: nil, depth: depthTarget, resolve: resolveTarget,
            colorStore: colorStore == .store ? 0 : 1, samples: pipeline.sampleCount, layer: texture.baseLayer + (depthSlice ?? 0),
            layerCount: pipeline.multiviewMask == nil ? 1 : (texture.viewLayers ?? texture.descriptor.arrayLayerCount),
            viewMask: pipeline.multiviewMask)
    }

    /// Vulkan aspect mask for a copy, or nil when the aspect is not copyable for this format.
    func copyAspect(_ aspect: TextureViewAspect, _ texture: Texture) -> UInt32? {
        let format = texture.format
        switch aspect {
        case .all: return format.hasDepth && format.hasStencil ? nil : format.aspect
        case .depthOnly: return format.hasDepth ? VK.Depth.aspectDepth : nil
        case .stencilOnly: return format.hasStencil ? VK.Depth.aspectStencil : nil
        }
    }

    /// Builds the buffer/image copy for a texture transfer command, or nil when the command is not one.
    func planImageCopy(_ command: BufferCommand, staged: inout [UInt8]) throws -> ImageCopyPlan? {
        func texture(_ id: ResourceID) throws -> Texture {
            guard let texture = textures[id] else { throw GPUError.validation("Unknown Vulkan texture") }
            return texture
        }
        func region(_ texture: Texture, aspect: UInt32, x: UInt32, y: UInt32, width: UInt32, height: UInt32, mip: UInt32, layer: UInt32,
                    layers: UInt32, z: UInt32 = 0, offset: UInt64, stride: UInt64?, rows: UInt32?) throws -> VkBufferImageCopy {
            let bytes = aspect == VK.Depth.aspectStencil ? 1 : UInt64(texture.format.bytes)
            var copy = VkBufferImageCopy()
            copy.bufferOffset = offset
            if let stride {
                guard stride % bytes == 0 else { throw GPUError.unsupported("Vulkan texture row stride must be a texel multiple") }
                copy.bufferRowLength = UInt32(stride / bytes) * texture.format.blockWidth
            }
            copy.bufferImageHeight = (rows ?? 0) * texture.format.blockHeight
            copy.aspectMask = aspect
            copy.mipLevel = mip
            if texture.descriptor.dimension == .d3 {
                copy.imageOffset = VkOffset3D(x: Int32(x), y: Int32(y), z: Int32(z))
                copy.imageExtent = VkExtent3D(width: width, height: height, depth: layers)
            } else {
                copy.baseArrayLayer = layer
                copy.layerCount = layers
                copy.imageOffset = VkOffset3D(x: Int32(x), y: Int32(y), z: 0)
                copy.imageExtent = VkExtent3D(width: width, height: height, depth: 1)
            }
            return copy
        }
        switch command {
        case .copyTextureToBuffer(let source, let destination, let bytesPerRow, let mip, let aspect, let offset, let layer,
                                  let z, let depth, let rows):
            let texture = try texture(source)
            guard let mask = copyAspect(aspect, texture), depth >= 1, let extent = texture.descriptor.mipExtent(mip),
                  let buffer = buffers[destination] else { throw GPUError.unsupported("Unsupported Vulkan texture copy") }
            return ImageCopyPlan(toImage: false, image: texture.image, buffer: .handle(buffer.native.handle),
                                 region: try region(texture, aspect: mask, x: 0, y: 0, width: extent.width, height: extent.height, mip: mip,
                                                    layer: layer, layers: depth, z: z, offset: offset, stride: UInt64(bytesPerRow), rows: rows))
        case .copyTextureRegionToBuffer(let source, let destination, let layout, let r, let aspect):
            let texture = try texture(source)
            guard let mask = copyAspect(aspect, texture), let buffer = buffers[destination] else { throw GPUError.unsupported("Unsupported Vulkan texture copy") }
            return ImageCopyPlan(toImage: false, image: texture.image, buffer: .handle(buffer.native.handle),
                                 region: try region(texture, aspect: mask, x: r.x, y: r.y, width: r.width, height: r.height, mip: r.mipLevel,
                                                    layer: r.arrayLayer, layers: r.depth, z: r.z, offset: layout.offset,
                                                    stride: layout.bytesPerRow.map(UInt64.init), rows: layout.rowsPerImage))
        case .copyBufferToTexture(let source, let layout, let destination, let r, let aspect):
            let texture = try texture(destination)
            guard let mask = copyAspect(aspect, texture), let buffer = buffers[source] else { throw GPUError.unsupported("Unsupported Vulkan texture copy") }
            return ImageCopyPlan(toImage: true, image: texture.image, buffer: .handle(buffer.native.handle),
                                 region: try region(texture, aspect: mask, x: r.x, y: r.y, width: r.width, height: r.height, mip: r.mipLevel,
                                                    layer: r.arrayLayer, layers: r.depth, z: r.z, offset: layout.offset,
                                                    stride: layout.bytesPerRow.map(UInt64.init), rows: layout.rowsPerImage))
        case .writeTexture(let destination, let data, let layout, let r, let aspect):
            let texture = try texture(destination)
            guard let mask = copyAspect(aspect, texture) else { throw GPUError.unsupported("Unsupported Vulkan texture upload") }
            while staged.count % 4 != 0 { staged.append(0) }
            let start = UInt64(staged.count)
            staged.append(contentsOf: data)
            let plan = ImageCopyPlan(
                toImage: true, image: texture.image, buffer: .staging,
                region: try region(texture, aspect: mask, x: r.x, y: r.y, width: r.width, height: r.height, mip: r.mipLevel,
                                   layer: r.arrayLayer, layers: r.depth, z: r.z, offset: start + layout.offset,
                                   stride: layout.bytesPerRow.map(UInt64.init), rows: layout.rowsPerImage))
            // Buffer offsets for image copies must be a multiple of the texel size and four.
            guard plan.region.bufferOffset % UInt64(max(4, texture.format.bytes)) == 0 else {
                throw GPUError.unsupported("Vulkan texture uploads require a texel-aligned data offset")
            }
            return plan
        default:
            return nil
        }
    }
}

func vulkanBlendFactor(_ factor: RenderBlendFactor) -> UInt32 {
    switch factor {
    case .zero: 0
    case .one: 1
    case .sourceColor: 2
    case .oneMinusSourceColor: 3
    case .destinationColor: 4
    case .oneMinusDestinationColor: 5
    case .sourceAlpha: 6
    case .oneMinusSourceAlpha: 7
    case .destinationAlpha: 8
    case .oneMinusDestinationAlpha: 9
    case .sourceAlphaSaturated: 14
    case .constant: 10
    case .oneMinusConstant: 11
    case .source1Color: 15
    case .oneMinusSource1Color: 16
    case .source1Alpha: 17
    case .oneMinusSource1Alpha: 18
    }
}

extension VulkanDevice {
    func recordRender(_ plan: RenderPlan, commandBuffer: UInt, batch: inout PendingBatch) throws {
        let g = native.g, f = native.f, device = native.device
        var viewInfo = VkImageViewCreateInfo()
        viewInfo.image = plan.image
        if plan.viewMask != nil { viewInfo.viewType = 5 }  // 2D array
        viewInfo.format = plan.format
        viewInfo.subresourceRange = VkImageSubresourceRange(baseMipLevel: plan.mipLevel, baseArrayLayer: plan.layer,
                                                                  layerCount: plan.layerCount)
        var view: UInt64 = 0
        try vulkanCheck(withUnsafePointer(to: &viewInfo) { g.createImageView(device, UnsafeRawPointer($0), nil, &view) },
                        "image view creation")
        batch.views.append(view)
        let renderPass = try makeRenderPass(format: plan.format, load: plan.clear == nil ? 0 : 1,
                                            depthFormat: plan.depth?.format, depthLoad: plan.depth?.loadOp ?? 0,
                                            depthStore: plan.depth?.storeOp ?? 0,
                                            stencilLoad: plan.depth?.stencilLoadOp ?? 2,
                                            stencilStore: plan.depth?.stencilStoreOp ?? 1, samples: plan.samples,
                                            resolve: plan.resolve != nil, colorStore: plan.colorStore,
                                            viewMask: plan.viewMask)
        batch.renderPasses.append(renderPass)
        var framebuffer: UInt64 = 0
        var attachments = [view]
        if let depth = plan.depth {
            var depthInfo = VkImageViewCreateInfo()
            depthInfo.image = depth.image
            depthInfo.format = depth.format
            if plan.viewMask != nil { depthInfo.viewType = 5 }
            depthInfo.subresourceRange = VkImageSubresourceRange(aspectMask: depth.aspect, baseMipLevel: depth.mipLevel, baseArrayLayer: depth.layer,
                                                               layerCount: plan.layerCount)
            var depthView: UInt64 = 0
            try vulkanCheck(withUnsafePointer(to: &depthInfo) { g.createImageView(device, UnsafeRawPointer($0), nil, &depthView) },
                            "image view creation")
            batch.views.append(depthView)
            attachments.append(depthView)
        }
        if let resolve = plan.resolve {
            var resolveInfo = VkImageViewCreateInfo()
            resolveInfo.image = resolve.image
            resolveInfo.format = plan.format
            if plan.viewMask != nil { resolveInfo.viewType = 5 }
            resolveInfo.subresourceRange = VkImageSubresourceRange(baseMipLevel: resolve.mipLevel, baseArrayLayer: resolve.layer,
                                                                    layerCount: plan.layerCount)
            var resolveView: UInt64 = 0
            try vulkanCheck(withUnsafePointer(to: &resolveInfo) { g.createImageView(device, UnsafeRawPointer($0), nil, &resolveView) },
                            "image view creation")
            batch.views.append(resolveView)
            attachments.append(resolveView)
        }
        try attachments.withUnsafeMutableBufferPointer { pointer in
            var info = VkFramebufferCreateInfo()
            info.renderPass = renderPass
            info.attachmentCount = UInt32(pointer.count)
            info.pAttachments = UnsafePointer(pointer.baseAddress)
            info.width = plan.extent.width
            info.height = plan.extent.height
            try vulkanCheck(withUnsafePointer(to: &info) { g.createFramebuffer(device, UnsafeRawPointer($0), nil, &framebuffer) },
                            "framebuffer creation")
        }
        batch.framebuffers.append(framebuffer)

        var clearValues = [VkClearColorValue(words: plan.clear ?? (0, 0, 0, 0))]
        if let depth = plan.depth {
            clearValues.append(VkClearColorValue(words: (depth.clearValue.bitPattern, depth.stencilClear, 0, 0)))
        }
        var begin = VkRenderPassBeginInfo()
        begin.renderPass = renderPass
        begin.framebuffer = framebuffer
        begin.areaWidth = plan.extent.width
        begin.areaHeight = plan.extent.height
        begin.clearValueCount = plan.depth != nil ? 2 : (plan.clear == nil ? 0 : 1)
        clearValues.withUnsafeBufferPointer { clearPointer in
            begin.pClearValues = UnsafeRawPointer(clearPointer.baseAddress)
            withUnsafePointer(to: &begin) { g.cmdBeginRenderPass(commandBuffer, UnsafeRawPointer($0), VK.subpassContentsInline) }
        }
        f.cmdBindPipeline(commandBuffer, VK.pipelineBindPointGraphics, plan.pipeline.pipeline)
        var viewport = plan.viewport
        var scissor = plan.scissor
        if let reference = plan.depth?.stencilReference { g.cmdSetStencilReference(commandBuffer, 3, reference) }
        g.cmdSetViewport(commandBuffer, 0, 1, &viewport)
        g.cmdSetScissor(commandBuffer, 0, 1, &scissor)
        plan.blendConstant.withUnsafeBufferPointer { g.cmdSetBlendConstants(commandBuffer, $0.baseAddress!) }
        try bindDescriptors(commandBuffer: commandBuffer, bindPoint: VK.pipelineBindPointGraphics, layout: plan.pipeline.layout,
                            setLayouts: plan.pipeline.setLayouts, writes: plan.writes, batch: &batch)
        if !plan.immediates.isEmpty {
            plan.immediates.withUnsafeBytes {
                f.cmdPushConstants(commandBuffer, plan.pipeline.layout, VK.Stage.vertex | VK.Stage.fragment, 0,
                                   UInt32($0.count), $0.baseAddress!)
            }
        }
        if !plan.vertexBuffers.isEmpty {
            let handles = plan.vertexBuffers.map(\.buffer)
            let offsets = plan.vertexBuffers.map(\.offset)
            handles.withUnsafeBufferPointer { h in
                offsets.withUnsafeBufferPointer { o in
                    g.cmdBindVertexBuffers(commandBuffer, 0, UInt32(h.count), h.baseAddress!, o.baseAddress!)
                }
            }
        }
        if let occlusion = plan.occlusion { g.cmdBeginQuery(commandBuffer, occlusion.pool, occlusion.index, 0) }
        if plan.meshGroups != nil, let arguments = plan.indirectArguments, let drawIndirect = native.drawMeshTasksIndirect {
            drawIndirect(commandBuffer, arguments.buffer, arguments.offset, 1, 12)
        } else if let groups = plan.meshGroups, let drawMeshTasks = native.drawMeshTasks {
            drawMeshTasks(commandBuffer, groups.0, groups.1, groups.2)
        } else if let arguments = plan.indirectArguments {
            if let indexed = plan.indexed {
                g.cmdBindIndexBuffer(commandBuffer, indexed.buffer, indexed.offset, indexed.type)
                g.cmdDrawIndexedIndirect(commandBuffer, arguments.buffer, arguments.offset, 1, 20)
            } else {
                g.cmdDrawIndirect(commandBuffer, arguments.buffer, arguments.offset, 1, 16)
            }
        } else if plan.instanceCount > 0 {
            if let indexed = plan.indexed {
                g.cmdBindIndexBuffer(commandBuffer, indexed.buffer, indexed.offset, indexed.type)
                if indexed.count > 0 {
                    g.cmdDrawIndexed(commandBuffer, indexed.count, plan.instanceCount, indexed.first, indexed.baseVertex,
                                     plan.firstInstance)
                }
            } else if plan.vertexCount > 0 {
                g.cmdDraw(commandBuffer, plan.vertexCount, plan.instanceCount, plan.firstVertex, plan.firstInstance)
            }
        }
        if let occlusion = plan.occlusion { g.cmdEndQuery(commandBuffer, occlusion.pool, occlusion.index) }
        g.cmdEndRenderPass(commandBuffer)
    }
}

extension VulkanDevice {
    func destroyTextureNow(id: ResourceID) {
        guard let texture = textures.removeValue(forKey: id) else { return }
        native.g.destroyImage(native.device, texture.image, nil)
        native.f.freeMemory(native.device, texture.memory, nil)
    }

    func destroyRenderPipelineNow(id: ResourceID) {
        guard let state = renderPipelines.removeValue(forKey: id) else { return }
        let f = native.f
        f.destroyPipeline(native.device, state.pipeline, nil)
        native.g.destroyRenderPass(native.device, state.renderPass, nil)
        f.destroyPipelineLayout(native.device, state.layout, nil)
        state.setLayouts.forEach { f.destroyDescriptorSetLayout(native.device, $0, nil) }
        state.modules.forEach { f.destroyShaderModule(native.device, $0, nil) }
    }
}

extension VulkanDevice {
    /// Plans `renderIndexed` by reusing the ordinary render planner, then attaching the index buffer.
    func planIndexedRender(_ command: BufferCommand, totals: inout DescriptorTotals) throws -> RenderPlan {
        guard case .renderIndexed(let pipeline, let target, let colorLoad, let indexBuffer, let draw, let vertexBuffers,
                                  let uniforms, let textures, let samplers, let mip, let slice, let depth, let resolve,
                                  let blend, let additional, let raster, let storage, let immediates, let store) = command,
              let buffer = buffers[indexBuffer.buffer], indexBuffer.offset % UInt64(indexBuffer.format.byteSize) == 0,
              indexBuffer.offset <= buffer.descriptor.size else {
            throw GPUError.validation("Invalid Vulkan index buffer")
        }
        let equivalent = BufferCommand.render(
            pipeline: pipeline, target: target, colorLoad: colorLoad, vertexCount: 0, vertexBuffers: vertexBuffers,
            firstVertex: 0, instances: draw.instances, uniforms: uniforms, textures: textures, samplers: samplers,
            targetMipLevel: mip, targetDepthSlice: slice, depth: depth, resolve: resolve, blendConstant: blend,
            additionalColors: additional, raster: raster, storageBuffers: storage, immediates: immediates,
            colorStore: store)
        var plan = try planRender(equivalent, totals: &totals)
        plan.indexed = (buffer.native.handle, indexBuffer.offset, indexBuffer.format == .uint16 ? 0 : 1,
                        draw.indexCount, draw.firstIndex, draw.baseVertex)
        return plan
    }
}

extension VulkanDevice {
    /// Plans `renderIndirect`: the draw arguments stay on the GPU and are validated into scratch memory before the pass.
    func planIndirectRender(_ command: BufferCommand, slot: Int, totals: inout DescriptorTotals) throws -> RenderPlan {
        guard case .renderIndirect(let pipeline, let target, let colorLoad, let arguments, let offset, let index, let vertexBuffers,
                                   let uniforms, let textures, let samplers, let mip, let slice, let depth, let resolve,
                                   let blend, let additional, let raster, let storage, let immediates, let store) = command,
              let source = buffers[arguments], offset % 4 == 0 else {
            throw GPUError.validation("Invalid Vulkan indirect draw arguments")
        }
        let equivalent = BufferCommand.render(
            pipeline: pipeline, target: target, colorLoad: colorLoad, vertexCount: 0, vertexBuffers: vertexBuffers,
            firstVertex: 0, instances: .init(), uniforms: uniforms, textures: textures, samplers: samplers,
            targetMipLevel: mip, targetDepthSlice: slice, depth: depth, resolve: resolve, blendConstant: blend,
            additionalColors: additional, raster: raster, storageBuffers: storage, immediates: immediates,
            colorStore: store)
        var plan = try planRender(equivalent, totals: &totals)
        let size: UInt64 = index == nil ? 16 : 20
        guard offset <= source.descriptor.size, source.descriptor.size - offset >= size else {
            throw GPUError.validation("Indirect draw arguments exceed the buffer")
        }
        if let index {
            guard let buffer = buffers[index.buffer], index.offset % UInt64(index.format.byteSize) == 0,
                  index.offset <= buffer.descriptor.size else {
                throw GPUError.validation("Invalid Vulkan index buffer")
            }
            plan.indexed = (buffer.native.handle, index.offset, index.format == .uint16 ? 0 : 1, 0, 0, 0)
        }
        plan.indirect = (source.native.handle, offset, slot)
        return plan
    }

    /// Copies GPU-authored draw arguments and zeroes them when `firstInstance` is set, because that needs an optional feature.
    func drawValidatorPipeline() throws -> Pipeline {
        let shader = try IndependentShaderCompiler().compileCompute(.wgsl("""
            @group(0) @binding(0) var<storage, read> source: array<u32, 5>;
            @group(0) @binding(1) var<storage, read_write> destination: array<u32, 5>;
            @group(0) @binding(2) var<storage, read> limits: array<u32, 2>;
            @compute @workgroup_size(1)
            fn validate_draw() {
                let invalid = source[limits[0]] != 0u;
                for (var i = 0u; i < 5u; i = i + 1u) {
                    destination[i] = select(source[i], 0u, invalid);
                }
            }
            """), to: .spirv)
        guard case .spirv(let words) = shader.artifact else { throw GPUError.backendFailure("Draw validator is not SPIR-V") }
        let model = try ComputePipeline(id: ResourceID(device: 0, serial: 0), shader: shader)
        return try makePipeline(model, words: words, entryPoint: shader.entryPoint)
    }
}

func vulkanStencilOp(_ operation: RenderStencilOperation) -> UInt32 {
    switch operation {
    case .keep: 0
    case .zero: 1
    case .replace: 2
    case .incrementClamp: 3
    case .decrementClamp: 4
    case .invert: 5
    case .incrementWrap: 6
    case .decrementWrap: 7
    }
}

struct TextureViewState {
    var texture: ResourceID
    var levels: Range<UInt32>
    var aspect: TextureViewAspect
    var layers: Range<UInt32> = 0..<1
    var dimension: TextureViewDimension = .d2
}

extension VulkanDevice: TextureViewBackendDevice {
    public func allocateTextureView(id: ResourceID, texture: ResourceID, mipLevels: Range<UInt32>) async throws {
        try await allocateTextureView(id: id, texture: texture, mipLevels: mipLevels, aspect: .all)
    }

    public func allocateTextureView(id: ResourceID, texture: ResourceID, descriptor: TextureViewDescriptor) async throws {
        guard let base = textures[texture] else { throw GPUError.validation("Invalid Vulkan texture view") }
        let count = descriptor.mipLevelCount ?? (base.descriptor.mipLevelCount - min(descriptor.baseMipLevel, base.descriptor.mipLevelCount))
        guard count > 0, count <= UInt32.max - descriptor.baseMipLevel else {
            throw GPUError.validation("Invalid Vulkan texture view")
        }
        let dimension = descriptor.resolvedDimension(in: base.descriptor)
        try await allocateTextureView(id: id, texture: texture, mipLevels: descriptor.baseMipLevel..<(descriptor.baseMipLevel + count),
                                      aspect: descriptor.aspect, layers: try descriptor.arrayLayers(in: base.descriptor),
                                      dimension: dimension)
    }

    public func allocateTextureView(id: ResourceID, texture: ResourceID, mipLevels: Range<UInt32>,
                                    aspect: TextureViewAspect) async throws {
        try await allocateTextureView(id: id, texture: texture, mipLevels: mipLevels, aspect: aspect, layers: nil, dimension: nil)
    }

    func allocateTextureView(id: ResourceID, texture: ResourceID, mipLevels: Range<UInt32>, aspect: TextureViewAspect,
                             layers: Range<UInt32>?, dimension: TextureViewDimension?) async throws {
        try requireActive()
        guard textures[id] == nil, buffers[id] == nil, textureViews[id] == nil else {
            throw GPUError.validation("Duplicate Vulkan resource")
        }
        guard let base = textures[texture], mipLevels.upperBound <= base.descriptor.mipLevelCount, !mipLevels.isEmpty else {
            throw GPUError.validation("Invalid Vulkan texture view")
        }
        guard aspect == .all || (aspect == .depthOnly && base.format.hasDepth) || (aspect == .stencilOnly && base.format.hasStencil) else {
            throw GPUError.validation("Texture view aspect does not exist in the texture format")
        }
        let defaultDimension: TextureViewDimension = base.descriptor.dimension == .d3 ? .d3
            : (base.descriptor.dimension == .d1 ? .d1 : (base.descriptor.arrayLayerCount > 1 ? .d2Array : .d2))
        textureViews[id] = TextureViewState(
            texture: texture, levels: mipLevels, aspect: aspect,
            layers: layers ?? (base.descriptor.dimension == .d3 ? 0..<1 : 0..<base.descriptor.arrayLayerCount),
            dimension: dimension ?? defaultDimension)
    }

    public func destroyTextureView(id: ResourceID) async { textureViews[id] = nil }

    func imageAspect(_ texture: Texture) -> UInt32 {
        switch texture.viewAspect {
        case .all: texture.format.sampledAspect
        case .depthOnly: VK.Depth.aspectDepth
        case .stencilOnly: VK.Depth.aspectStencil
        }
    }
}

extension VulkanDevice {
    /// Vulkan image view type for a texture or view bound to a shader.
    func viewType(_ texture: Texture) -> UInt32 {
        let dimension: TextureViewDimension = texture.viewDimension
            ?? (texture.descriptor.dimension == .d3 ? .d3 : (texture.descriptor.dimension == .d1 ? .d1 : (texture.descriptor.arrayLayerCount > 1 ? .d2Array : .d2)))
        switch dimension {
        case .d1: return 0
        case .d2: return 1
        case .d3: return 2
        case .cube: return 3
        case .d2Array: return 5
        case .cubeArray: return 6
        }
    }
}

extension VulkanDevice: BufferMappingBackendDevice {
    public func beginBufferMapping(id: ResourceID, mode: BufferMapMode, offset: UInt64, size: UInt64) async throws {
        try requireActive()
        guard let buffer = buffers[id], offset <= buffer.descriptor.size, size <= buffer.descriptor.size - offset,
              mappings[id] == nil else {
            throw GPUError.validation("Invalid Vulkan buffer mapping")
        }
        mappings[id] = mode
    }

    public func readMappedBytes(id: ResourceID, offset: UInt64, size: UInt64) async throws -> [UInt8] {
        try requireActive()
        guard let buffer = buffers[id], mappings[id] == .read, offset <= buffer.descriptor.size,
              size <= buffer.descriptor.size - offset else {
            throw GPUError.validation("Invalid Vulkan mapped read")
        }
        let base = buffer.native.mapped + Int(offset)
        return [UInt8](UnsafeRawBufferPointer(start: base, count: Int(size)))
    }

    public func writeMappedBytes(id: ResourceID, offset: UInt64, bytes: [UInt8]) async throws {
        try requireActive()
        guard let buffer = buffers[id], mappings[id] != .read, offset <= buffer.descriptor.size,
              UInt64(bytes.count) <= buffer.descriptor.size - offset else {
            throw GPUError.validation("Invalid Vulkan mapped write")
        }
        bytes.withUnsafeBytes { (buffer.native.mapped + Int(offset)).copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
    }

    public func endBufferMapping(id: ResourceID) async throws { mappings[id] = nil }
}

extension VulkanDevice {
    /// A render pass with a single depth-stencil attachment that clears on load, used to zero multisampled depth images.
    func makeDepthOnlyRenderPass(format: UInt32, samples: UInt32) throws -> UInt64 {
        var attachment = VkAttachmentDescription()
        attachment.format = format
        attachment.samples = samples
        attachment.loadOp = 1
        attachment.stencilLoadOp = 1
        attachment.stencilStoreOp = 0
        var reference = VkAttachmentReference()
        var renderPass: UInt64 = 0
        let g = native.g, device = native.device
        try withUnsafePointer(to: &reference) { referencePointer in
            var subpass = VkSubpassDescription()
            subpass.pDepthStencilAttachment = UnsafeRawPointer(referencePointer)
            try withUnsafePointer(to: &attachment) { attachmentPointer in
                try withUnsafePointer(to: &subpass) { subpassPointer in
                    var info = VkRenderPassCreateInfo()
                    info.attachmentCount = 1
                    info.pAttachments = UnsafeRawPointer(attachmentPointer)
                    info.subpassCount = 1
                    info.pSubpasses = UnsafeRawPointer(subpassPointer)
                    try vulkanCheck(withUnsafePointer(to: &info) {
                        g.createRenderPass(device, UnsafeRawPointer($0), nil, &renderPass)
                    }, "render pass creation")
                }
            }
        }
        return renderPass
    }
}
