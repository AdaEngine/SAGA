import SAGATypes
import SAGABackend
import SAGAShaderIR
import SAGACompiler

/// Owns the Vulkan logical device and its single compute-capable queue. Every public Vulkan object
/// in `VulkanDevice` is destroyed through this handle's functions, and the device outlives them.
final class VulkanNativeDevice: Sendable {
    let instance: VulkanInstance
    let limits: VulkanPhysicalLimits
    let device: UInt
    let queue: UInt
    let queueFamily: UInt32

    var f: VulkanFunctions { instance.functions }
    let graphics: VulkanGraphicsFunctions

    init(adapter: VulkanAdapter, swapchain: Bool = false) throws {
        let support = adapter.limits.featureSupport
        if swapchain, !support.deviceExtensions.contains(VKPresent.swapchainExtension) {
            throw GPUError.unsupported("The Vulkan device does not support presentation")
        }
        var deviceExtensions = swapchain ? [VKPresent.swapchainExtension] : []
        let chain = VulkanFeatureChain()
        chain.setCore(0, 1)  // robustBufferAccess: out-of-range buffer fetches return zero or are discarded
        for index in VulkanFeatureSupport.enabledCoreFeatures where support.core[index] != 0 { chain.setCore(index, 1) }
        if support.multiview { chain.multiview = 1 }
        if support.canEnableFloat16 {
            chain.float16 = 1
            chain.storage16 = 1
            if support.int8 { chain.int8 = 1 }
            deviceExtensions.append(VulkanFeatureSupport.float16Extension)
        } else {
            chain.unlinkFloat16()
        }
        instance = adapter.instance
        graphics = try VulkanGraphicsFunctions(instance: adapter.instance)
        limits = adapter.limits
        queueFamily = adapter.queueFamily
        var priority: Float = 1
        var handle: UInt = 0
        let result: VkResult = withUnsafePointer(to: &priority) { priorityPointer in
            var queueInfo = VkDeviceQueueCreateInfo()
            queueInfo.queueFamilyIndex = adapter.queueFamily
            queueInfo.queueCount = 1
            queueInfo.pQueuePriorities = priorityPointer
            return withUnsafePointer(to: &queueInfo) { queuePointer in
                var info = VkDeviceCreateInfo()
                info.pNext = UnsafeRawPointer(chain.features2)
                info.queueCreateInfoCount = 1
                info.pQueueCreateInfos = UnsafeRawPointer(queuePointer)
                return withCStringArray(deviceExtensions) { names in
                    info.enabledExtensionCount = UInt32(deviceExtensions.count)
                    info.ppEnabledExtensionNames = UnsafeRawPointer(names)
                    return withUnsafePointer(to: &info) {
                        adapter.instance.functions.createDevice(adapter.physical, UnsafeRawPointer($0), nil, &handle)
                    }
                }
            }
        }
        withExtendedLifetime(chain) {}
        try vulkanCheck(result, "device creation")
        device = handle
        var queueHandle: UInt = 0
        adapter.instance.functions.getDeviceQueue(handle, adapter.queueFamily, 0, &queueHandle)
        queue = queueHandle
    }

    deinit {
        _ = f.deviceWaitIdle(device)
        f.destroyDevice(device, nil)
    }

    /// Picks a host-visible coherent memory type allowed by `typeBits`, preferring device-local memory.
    func hostMemoryType(typeBits: UInt32) -> UInt32? {
        let wanted = VK.MemoryProperty.hostVisible | VK.MemoryProperty.hostCoherent
        var fallback: UInt32?
        for (index, flags) in limits.memoryTypeFlags.enumerated()
        where typeBits & (1 << UInt32(index)) != 0 && flags & wanted == wanted {
            if flags & VK.MemoryProperty.deviceLocal != 0 { return UInt32(index) }
            fallback = fallback ?? UInt32(index)
        }
        return fallback
    }
}

/// A host-visible, persistently mapped, zero-initialized buffer.
struct VulkanHostBuffer {
    let handle: UInt64
    let memory: UInt64
    let size: UInt64
    let mapped: UnsafeMutableRawPointer
}

extension VulkanNativeDevice {
    func makeHostBuffer(size requested: UInt64) throws -> VulkanHostBuffer {
        let size = max(4, (requested + 3) & ~3)
        var info = VkBufferCreateInfo()
        info.size = size
        info.usage = VK.BufferUsage.transferSource | VK.BufferUsage.transferDestination | VK.BufferUsage.storageBuffer |
            VK.BufferUsage.uniformBuffer | VK.BufferUsage.indirectBuffer
        var buffer: UInt64 = 0
        try vulkanCheck(withUnsafePointer(to: &info) { f.createBuffer(device, UnsafeRawPointer($0), nil, &buffer) }, "buffer creation")
        var requirements = VkMemoryRequirements()
        f.getBufferMemoryRequirements(device, buffer, &requirements)
        guard let type = hostMemoryType(typeBits: requirements.memoryTypeBits) else {
            f.destroyBuffer(device, buffer, nil)
            throw GPUError.unsupported("Vulkan device has no host-visible coherent memory")
        }
        var allocation = VkMemoryAllocateInfo()
        allocation.allocationSize = requirements.size
        allocation.memoryTypeIndex = type
        var memory: UInt64 = 0
        let allocated = withUnsafePointer(to: &allocation) { f.allocateMemory(device, UnsafeRawPointer($0), nil, &memory) }
        guard allocated == VK.success else {
            f.destroyBuffer(device, buffer, nil)
            try vulkanCheck(allocated, "memory allocation")
            throw GPUError.outOfMemory
        }
        var mapped: UnsafeMutableRawPointer?
        let bound = f.bindBufferMemory(device, buffer, memory, 0)
        let mappedResult = bound == VK.success ? f.mapMemory(device, memory, 0, VK.wholeSize, 0, &mapped) : bound
        guard mappedResult == VK.success, let mapped else {
            f.freeMemory(device, memory, nil)
            f.destroyBuffer(device, buffer, nil)
            try vulkanCheck(mappedResult, "memory mapping")
            throw GPUError.backendFailure("Vulkan returned no mapping")
        }
        mapped.initializeMemory(as: UInt8.self, repeating: 0, count: Int(requirements.size))
        return VulkanHostBuffer(handle: buffer, memory: memory, size: size, mapped: mapped)
    }

    func destroy(_ buffer: VulkanHostBuffer) {
        f.unmapMemory(device, buffer.memory)
        f.destroyBuffer(device, buffer.handle, nil)
        f.freeMemory(device, buffer.memory, nil)
    }
}

/// Vulkan runtime backend for Windows, Linux and Android: SPIR-V compute pipelines over storage and
/// uniform buffers, direct dispatch, buffer copies, clears and readback.
///
/// Buffers use host-visible coherent memory in this slice. Textures, samplers, render passes,
/// indirect dispatch, immediates, queries and presentation are rejected with `GPUError.unsupported`.
public actor VulkanDevice: ComputeBackendDevice {
    public nonisolated let capabilities: DeviceCapabilities
    public nonisolated let adapterName: String
    public nonisolated let adapterKind: VulkanAdapterKind
    public nonisolated let supportsPresentation: Bool

    struct Buffer {
        var descriptor: BufferDescriptor
        var native: VulkanHostBuffer
    }

    struct Pipeline {
        var model: ComputePipeline
        var module: UInt64
        var setLayouts: [UInt64]
        var layout: UInt64
        var pipeline: UInt64
        /// Descriptor type of every buffer binding, indexed by set then binding.
        var descriptorTypes: [ComputeBindingSlot: UInt32]
    }

    let native: VulkanNativeDevice
    let adapter: VulkanAdapter
    var presenter: VulkanPresenter?
    let timeout: Duration
    var buffers: [ResourceID: Buffer] = [:]
    var mappings: [ResourceID: BufferMapMode] = [:]
    var pipelines: [ResourceID: Pipeline] = [:]
    var textures: [ResourceID: Texture] = [:]
    var renderPipelines: [ResourceID: RenderPipelineState] = [:]
    var samplers: [ResourceID: SamplerState] = [:]
    var textureViews: [ResourceID: TextureViewState] = [:]
    var querySets: [ResourceID: VulkanQuerySetState] = [:]
    var terminal = false
    /// Native objects of timed-out batches stay alive: the GPU may still be using them.
    var quarantine: [PendingBatch] = []

    init(adapter: VulkanAdapter, timeout: Duration, surface: VulkanSurfaceTarget? = nil) throws {
        native = try VulkanNativeDevice(adapter: adapter, swapchain: surface != nil)
        self.adapter = adapter
        supportsPresentation = surface != nil
        if let surface { presenter = try VulkanPresenter(adapter: adapter, target: surface) }
        self.timeout = timeout
        adapterName = adapter.name
        adapterKind = adapter.kind
        let limits = adapter.limits
        let largestHeap = limits.heapSizes.enumerated().filter { limits.heapIsDeviceLocal[$0.offset] }.map(\.element).max()
            ?? limits.heapSizes.max() ?? 0
        let (cx, cy, cz) = limits.maxComputeWorkGroupCount
        let (sx, sy, sz) = limits.maxComputeWorkGroupSize
        var features: Set<GPUFeature> = []
        if limits.timestampsSupported { features.insert(.timestampQuery) }
        let support = limits.featureSupport
        if support.core[41] != 0 { features.insert(.shaderInt16) }
        if support.core[40] != 0 { features.insert(.shaderInt64) }
        if support.canEnableFloat16 { features.insert(.shaderF16) }
        if support.supportsSubgroups { features.insert(.subgroups) }
        let multiviewViews = support.multiview ? min(support.maxMultiviewViewCount, 32) : 0
        if multiviewViews > 1 { features.formUnion([.multiview, .selectiveMultiview]) }
        if support.core[22] != 0 { features.insert(.textureCompressionBC) }
        if support.core[20] != 0 { features.insert(.textureCompressionETC2) }
        if support.core[21] != 0 { features.insert(.textureCompressionASTC) }
        if limits.depthStencilFormats.contains(130) { features.insert(.depth32FloatStencil8) }
        capabilities = try DeviceCapabilities(
            maxBufferSize: min(UInt64(limits.maxStorageBufferRange), largestHeap / 2), features: features,
            maxTextureDimension2D: limits.maxImageDimension2D,
            timestampPeriodNanoseconds: limits.timestampsSupported ? Double(limits.timestampPeriod) : 1,
            maxImmediateSize: min(limits.maxPushConstantsSize, 256) & ~3,
            subgroupSizeRange: limits.featureSupport.supportsSubgroups
                ? limits.featureSupport.subgroupSize...limits.featureSupport.subgroupSize : nil,
            maxComputeWorkgroupStorageSize: limits.maxComputeSharedMemorySize,
            maxComputeInvocationsPerWorkgroup: limits.maxComputeWorkGroupInvocations,
            maxComputeWorkgroupSize: DispatchWorkgroups(x: sx, y: sy, z: sz),
            maxComputeWorkgroupsPerDimension: min(cx, cy, cz),
            maxVertexAttributes: min(limits.maxVertexInputAttributes, 16),
            maxVertexBuffers: min(limits.maxVertexInputBindings, 8),
            maxColorAttachments: min(limits.maxFragmentOutputAttachments, 8),
            maxMultiviewViewCount: multiviewViews > 1 ? multiviewViews : 0)
    }

    func requireActive() throws {
        guard !terminal else { throw GPUError.deviceLost("Vulkan device is terminal after a command failure or timeout") }
    }

    public func resourceCountsForTesting() -> (buffers: Int, pipelines: Int) { (buffers.count, pipelines.count) }

    // MARK: Buffers

    public func allocateBuffer(id: ResourceID, descriptor: BufferDescriptor) throws {
        try requireActive()
        try descriptor.validate(against: capabilities)
        guard buffers[id] == nil else { throw GPUError.validation("Duplicate Vulkan buffer") }
        buffers[id] = Buffer(descriptor: descriptor, native: try native.makeHostBuffer(size: descriptor.size))
    }

    public func destroyBuffer(id: ResourceID) {
        if let buffer = buffers.removeValue(forKey: id) { native.destroy(buffer.native) }
    }

    public func close(owner: UInt64) {
        if let presenter { presenter.destroy(native: native); self.presenter = nil }
        for id in buffers.keys where id.device == owner { destroyBuffer(id: id) }
        for id in pipelines.keys where id.device == owner { destroyComputePipeline(id: id) }
        for id in renderPipelines.keys where id.device == owner { destroyRenderPipelineNow(id: id) }
        for id in textures.keys where id.device == owner { destroyTextureNow(id: id) }
        for id in samplers.keys where id.device == owner { destroySamplerNow(id: id) }
        for id in querySets.keys where id.device == owner { destroyQuerySetNow(id: id) }
    }

    // MARK: Compute pipelines

    public func allocateComputePipeline(_ pipeline: ComputePipeline, shader: ComputeShaderArtifact) throws {
        try requireActive()
        guard case .spirv(let words) = shader.artifact, !words.isEmpty else {
            throw GPUError.validation("Vulkan requires a compiled SPIR-V compute shader")
        }
        let reflected = try ComputePipeline(id: pipeline.id, shader: shader, nonFilteringSamplers: pipeline.nonFilteringSamplers)
        guard pipeline == reflected else {
            throw GPUError.validation("Vulkan compute pipeline layout does not match compiled shader")
        }
        guard pipelines[pipeline.id] == nil else { throw GPUError.validation("Duplicate Vulkan compute pipeline") }
        guard
              pipeline.externalTextureBindings.isEmpty, pipeline.accelerationStructureBindings.isEmpty else {
            throw GPUError.unsupported("Vulkan compute pipelines do not support external textures or acceleration structures")
        }
        pipelines[pipeline.id] = try makePipeline(pipeline, words: words, entryPoint: shader.entryPoint)
    }

    func makePipeline(_ pipeline: ComputePipeline, words: [UInt32], entryPoint: String) throws -> Pipeline {
        guard pipeline.immediateByteSize <= UInt64(capabilities.maxImmediateSize) else {
            throw GPUError.validation("Immediate data exceeds the Vulkan push constant limit")
        }
        var types: [ComputeBindingSlot: UInt32] = [:]
        for binding in pipeline.bufferBindings {
            guard binding.slot.group < min(native.limits.maxBoundDescriptorSets, ComputeShaderArtifact.portableBindGroupCount) else {
                throw GPUError.validation("Bind group exceeds the Vulkan device limit")
            }
            types[binding.slot] = binding.usage == .uniform ? VK.DescriptorType.uniformBuffer : VK.DescriptorType.storageBuffer
        }
        types.merge(try vulkanHandleDescriptorTypes(sampled: pipeline.textureBindings, storage: pipeline.storageTextureBindings, samplers: pipeline.samplerBindings)) { first, _ in first }
        let setCount = (types.keys.map(\.group).max()).map { Int($0) + 1 } ?? 0
        let f = native.f
        let device = native.device

        var module: UInt64 = 0
        var layouts: [UInt64] = []
        var pipelineLayout: UInt64 = 0
        var handle: UInt64 = 0
        func cleanup() {
            if handle != 0 { f.destroyPipeline(device, handle, nil) }
            if pipelineLayout != 0 { f.destroyPipelineLayout(device, pipelineLayout, nil) }
            layouts.forEach { f.destroyDescriptorSetLayout(device, $0, nil) }
            if module != 0 { f.destroyShaderModule(device, module, nil) }
        }
        do {
            try words.withUnsafeBufferPointer { code in
                var info = VkShaderModuleCreateInfo()
                info.codeSize = code.count * 4
                info.pCode = code.baseAddress
                try vulkanCheck(withUnsafePointer(to: &info) { f.createShaderModule(device, UnsafeRawPointer($0), nil, &module) },
                                "shader module creation")
            }
            for group in 0..<setCount {
                let bindings = vulkanLayoutBindings(types: types, group: group, stages: VK.shaderStageCompute)
                var layout: UInt64 = 0
                try bindings.withUnsafeBufferPointer { pointer in
                    var info = VkDescriptorSetLayoutCreateInfo()
                    info.bindingCount = UInt32(pointer.count)
                    info.pBindings = pointer.baseAddress
                    try vulkanCheck(withUnsafePointer(to: &info) {
                        f.createDescriptorSetLayout(device, UnsafeRawPointer($0), nil, &layout)
                    }, "descriptor set layout creation")
                }
                layouts.append(layout)
            }
            var pushRange = VkPushConstantRange(size: UInt32(pipeline.immediateByteSize))
            try layouts.withUnsafeBufferPointer { pointer in
                var info = VkPipelineLayoutCreateInfo()
                info.setLayoutCount = UInt32(pointer.count)
                info.pSetLayouts = pointer.baseAddress
                try withUnsafePointer(to: &pushRange) { range in
                    if range.pointee.size > 0 {
                        info.pushConstantRangeCount = 1
                        info.pPushConstantRanges = UnsafeRawPointer(range)
                    }
                    try vulkanCheck(withUnsafePointer(to: &info) {
                        f.createPipelineLayout(device, UnsafeRawPointer($0), nil, &pipelineLayout)
                    }, "pipeline layout creation")
                }
            }
            try entryPoint.withCString { name in
                var info = VkComputePipelineCreateInfo()
                info.stage.module = module
                info.stage.pName = name
                info.layout = pipelineLayout
                try vulkanCheck(withUnsafePointer(to: &info) {
                    f.createComputePipelines(device, 0, 1, UnsafeRawPointer($0), nil, &handle)
                }, "compute pipeline creation")
            }
        } catch {
            cleanup()
            throw error
        }
        return Pipeline(model: pipeline, module: module, setLayouts: layouts, layout: pipelineLayout,
                                          pipeline: handle, descriptorTypes: types)
    }

    public func destroyComputePipeline(id: ResourceID) {
        guard let pipeline = pipelines.removeValue(forKey: id) else { return }
        let f = native.f
        f.destroyPipeline(native.device, pipeline.pipeline, nil)
        f.destroyPipelineLayout(native.device, pipeline.layout, nil)
        pipeline.setLayouts.forEach { f.destroyDescriptorSetLayout(native.device, $0, nil) }
        f.destroyShaderModule(native.device, pipeline.module, nil)
    }

    // MARK: Queue

    public func submit(_ submission: Submission) async throws {
        _ = try await submitCommands(submission, commands: [])
    }

    enum Operation {
        case upload(destination: UInt64, offset: UInt64, stagingOffset: UInt64, count: UInt64)
        case copy(source: UInt64, sourceOffset: UInt64, destination: UInt64, destinationOffset: UInt64, count: UInt64)
        case fill(buffer: UInt64, offset: UInt64, count: UInt64)
        case read(source: UInt64, readbackOffset: UInt64, count: UInt64)
        case dispatch(DispatchPlan)
        case imageCopy(ImageCopyPlan)
        case render(RenderPlan)
        case resetQuery(pool: UInt64, index: UInt32)
        case writeTimestamp(pool: UInt64, index: UInt32, end: Bool)
        case resolveQueries(QueryResolve)
    }

    struct DispatchPlan {
        var pipeline: Pipeline
        var writes: [DescriptorWrite]
        var groups: DispatchWorkgroups
        var immediates: [UInt32]
        /// GPU-authored counts: source buffer, byte offset, and the scratch slot that holds validated counts.
        var indirect: (source: UInt64, offset: UInt64, slot: Int)?
    }

    struct DescriptorWrite {
        var set: Int
        var binding: UInt32
        var type: UInt32
        var buffer: UInt64
        var offset: UInt64
        var range: UInt64
        var image: UInt64 = 0
        var imageFormat: UInt32 = 0
        var imageAspect: UInt32 = VK.Image.aspectColor
        var baseMip: UInt32 = 0
        var viewType: UInt32 = VK.Image.viewType2D
        var baseLayer: UInt32 = 0
        var layerCount: UInt32 = 1
        var mipLevels: UInt32 = 1
        var sampler: UInt64 = 0
        var arrayElement: UInt32 = 0
    }

    /// Native objects of one submitted batch. They are released only after the GPU retires the fence.
    struct PendingBatch {
        var staging: VulkanHostBuffer?
        var readback: VulkanHostBuffer?
        var scratch: VulkanHostBuffer?
        var views: [UInt64] = []
        var renderPasses: [UInt64] = []
        var framebuffers: [UInt64] = []
        var commandPool: UInt64 = 0
        var descriptorPool: UInt64 = 0
        var fence: UInt64 = 0
    }

    public func submitCommands(_ submission: Submission, commands: [BufferCommand]) async throws -> [[UInt32]] {
        try requireActive()
        let owner = submission.resources.first?.device ??
            commands.lazy.compactMap { $0.recordedPass.command.computeDispatch?.pipeline.id.device }.first ?? 0
        let descriptors = buffers.mapValues(\.descriptor)
        let pinned = Set(submission.resources)
        let textureDescriptors = textures.mapValues(\.descriptor)
        for id in submission.resources
        where buffers[id] == nil && pipelines[id] == nil && textures[id] == nil && renderPipelines[id] == nil && samplers[id] == nil && textureViews[id] == nil && querySets[id] == nil {
            throw GPUError.validation("Unknown Vulkan resource")
        }

        var operations: [Operation] = []
        var staged: [UInt8] = []
        var readSizes: [UInt64] = []
        var readbackSize: UInt64 = 0
        var descriptorTotals = DescriptorTotals()
        var indirectLimits: [(linear: UInt32, axis: UInt32)] = []
        var validator: Pipeline?
        var drawValidator: Pipeline?

        for recorded in commands {
            guard recorded.resources.allSatisfy(pinned.contains) else {
                throw GPUError.validation("Unpinned command resource")
            }
            try recorded.validate(device: owner, descriptors: descriptors, textures: textureDescriptors,
                                  samplerDescriptors: samplers.mapValues { $0.descriptor }, textureViews: textureViews.mapValues { .init(texture: $0.texture, levels: $0.levels, aspect: $0.aspect, layers: $0.layers, dimension: $0.dimension) },
                                  maximumViewportDimension: capabilities.maxTextureDimension2D,
                                  maxVertexBuffers: capabilities.maxVertexBuffers,
                                  maxBindGroupsPlusVertexBuffers: capabilities.maxBindGroupsPlusVertexBuffers,
                                  querySets: querySets.mapValues { $0.descriptor },
                                  maxMultiviewViewCount: capabilities.maxMultiviewViewCount,
                                  features: capabilities.features,
                                  maxComputeWorkgroupsPerDimension: capabilities.maxComputeWorkgroupsPerDimension)
            let (command, timestamps, occlusion) = recorded.recordedPass
            let measurement = timestamps == nil && occlusion == nil ? nil
                : try planMeasurement(timestamps: timestamps, occlusion: occlusion)
            let operationsBefore = operations.count
            defer {
                if let measurement, operations.count == operationsBefore + 1 {
                    var pass = operations[operationsBefore]
                    if case .render(var plan) = pass {
                        plan.occlusion = measurement.occlusion
                        pass = .render(plan)
                    }
                    var wrapped = measurement.measured.resets.map { Operation.resetQuery(pool: $0.pool, index: $0.index) }
                    if let begin = measurement.measured.begin { wrapped.append(.writeTimestamp(pool: begin.pool, index: begin.index, end: false)) }
                    wrapped.append(pass)
                    if let end = measurement.measured.end { wrapped.append(.writeTimestamp(pool: end.pool, index: end.index, end: true)) }
                    operations.replaceSubrange(operationsBefore..., with: wrapped)
                }
            }
            switch command {
            case .resolveQuerySet(let set, let first, let count, let destination, let offset):
                operations.append(.resolveQueries(try planResolve(querySet: set, first: first, count: count,
                                                                  destination: destination, offset: offset)))
            case .write(let id, let offset, let words):
                let count = UInt64(words.count) * 4
                guard count > 0 else { continue }
                let stagingOffset = UInt64(staged.count)
                for word in words { staged.append(contentsOf: [UInt8(truncatingIfNeeded: word), UInt8(truncatingIfNeeded: word >> 8),
                                                              UInt8(truncatingIfNeeded: word >> 16), UInt8(truncatingIfNeeded: word >> 24)]) }
                operations.append(.upload(destination: try handle(id), offset: offset, stagingOffset: stagingOffset, count: count))
            case .writeBytes(let id, let offset, let bytes, let dataOffset, let byteCount):
                let slice = try BufferCommand.uploadBytes(bytes, dataOffset: dataOffset, byteCount: byteCount)
                guard !slice.isEmpty else { continue }
                while staged.count % 4 != 0 { staged.append(0) }
                let stagingOffset = UInt64(staged.count)
                staged.append(contentsOf: slice)
                operations.append(.upload(destination: try handle(id), offset: offset, stagingOffset: stagingOffset,
                                          count: UInt64(slice.count)))
            case .copy(let source, let sourceOffset, let destination, let destinationOffset, let byteCount):
                guard let sourceBuffer = buffers[source] else { throw GPUError.validation("Unknown Vulkan buffer") }
                let count = byteCount ?? (sourceBuffer.descriptor.size - sourceOffset)
                guard count > 0 else { continue }
                operations.append(.copy(source: sourceBuffer.native.handle, sourceOffset: sourceOffset,
                                        destination: try handle(destination), destinationOffset: destinationOffset, count: count))
            case .clearBuffer(let id, let offset, let byteCount):
                guard let buffer = buffers[id] else { throw GPUError.validation("Unknown Vulkan buffer") }
                let count = byteCount ?? (buffer.descriptor.size - offset)
                guard count > 0 else { continue }
                operations.append(.fill(buffer: buffer.native.handle, offset: offset, count: count))
            case .read(let id, let byteCount):
                readSizes.append(byteCount)
                operations.append(.read(source: try handle(id), readbackOffset: readbackSize, count: byteCount))
                readbackSize += (byteCount + 3) & ~3
            case .render:
                let plan = try planRender(command, totals: &descriptorTotals)
                descriptorTotals.sets += UInt32(plan.pipeline.setLayouts.count)
                operations.append(.render(plan))
            case .renderIndexed:
                let plan = try planIndexedRender(command, totals: &descriptorTotals)
                descriptorTotals.sets += UInt32(plan.pipeline.setLayouts.count)
                operations.append(.render(plan))
            case .renderIndirect:
                let plan = try planIndirectRender(command, slot: indirectLimits.count, totals: &descriptorTotals)
                drawValidator = try drawValidator ?? drawValidatorPipeline()
                descriptorTotals.sets += UInt32(plan.pipeline.setLayouts.count) + 1
                descriptorTotals.add(VK.DescriptorType.storageBuffer, 3)
                indirectLimits.append((plan.indexed == nil ? 3 : 4, 0))
                operations.append(.render(plan))
            case .copyTextureToBuffer, .copyTextureRegionToBuffer, .copyBufferToTexture, .writeTexture:
                guard let plan = try planImageCopy(command, staged: &staged) else {
                    throw GPUError.unsupported("Vulkan does not implement this command yet")
                }
                operations.append(.imageCopy(plan))
            case .dispatchCompute, .dispatchComputeRanges, .dispatchComputeIndirect:
                guard let dispatch = command.computeDispatch,
                      let pipeline = pipelines[dispatch.pipeline.id], pipeline.model == dispatch.pipeline else {
                    throw GPUError.validation("Unknown Vulkan compute pipeline")
                }
                guard UInt64(dispatch.immediates.count) * 4 == (pipeline.model.immediateByteSize + 3) / 4 * 4 else {
                    throw GPUError.validation("Immediate data size does not match the compute pipeline")
                }
                let writes = try planBindings(
                    types: pipeline.descriptorTypes,
                    buffers: pipeline.model.bufferBindings.map { ($0.slot, $0.usage) }, suppliedBuffers: dispatch.bindings,
                    sampledTextures: pipeline.model.textureBindings.map(\.slot),
                    storageTextures: pipeline.model.storageTextureBindings.map(\.slot),
                    samplerSlots: pipeline.model.samplerBindings.map(\.slot),
                    textureResources: dispatch.textures, samplerResources: dispatch.samplers, totals: &descriptorTotals)
                descriptorTotals.sets += UInt32(pipeline.setLayouts.count)
                switch dispatch.source {
                case .direct(let groups):
                    operations.append(.dispatch(DispatchPlan(pipeline: pipeline, writes: writes, groups: groups,
                                                             immediates: dispatch.immediates, indirect: nil)))
                case .indirect(let source, let offset):
                    let validatorPipeline = try validator ?? dispatchValidatorPipeline()
                    validator = validatorPipeline
                    descriptorTotals.sets += 1
                    descriptorTotals.add(VK.DescriptorType.storageBuffer, 3)
                    operations.append(.dispatch(DispatchPlan(
                        pipeline: pipeline, writes: writes, groups: DispatchWorkgroups(x: 0), immediates: dispatch.immediates,
                        indirect: (try handle(source), offset, indirectLimits.count))))
                    indirectLimits.append((pipeline.model.maximumLinearWorkgroupCount ?? 0,
                                           capabilities.maxComputeWorkgroupsPerDimension))
                }
            default:
                throw GPUError.unsupported("Vulkan does not implement this command yet")
            }
        }
        guard !operations.isEmpty else { return [] }

        var batch = PendingBatch()
        let f = native.f
        let device = native.device
        func release(_ batch: PendingBatch) {
            if batch.fence != 0 { f.destroyFence(device, batch.fence, nil) }
            if batch.descriptorPool != 0 { f.destroyDescriptorPool(device, batch.descriptorPool, nil) }
            if batch.commandPool != 0 { f.destroyCommandPool(device, batch.commandPool, nil) }
            if let staging = batch.staging { native.destroy(staging) }
            if let readback = batch.readback { native.destroy(readback) }
            if let scratch = batch.scratch { native.destroy(scratch) }
            batch.framebuffers.forEach { native.g.destroyFramebuffer(device, $0, nil) }
            batch.renderPasses.forEach { native.g.destroyRenderPass(device, $0, nil) }
            batch.views.forEach { native.g.destroyImageView(device, $0, nil) }
        }
        do {
            if !staged.isEmpty {
                let staging = try native.makeHostBuffer(size: UInt64(staged.count))
                batch.staging = staging
                staged.withUnsafeBytes { staging.mapped.copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
            }
            if readbackSize > 0 { batch.readback = try native.makeHostBuffer(size: readbackSize) }
            if !indirectLimits.isEmpty {
                let scratch = try native.makeHostBuffer(size: UInt64(indirectLimits.count) * Self.scratchStride)
                batch.scratch = scratch
                for (slot, limit) in indirectLimits.enumerated() {
                    let base = scratch.mapped + slot * Int(Self.scratchStride) + 512
                    base.storeBytes(of: limit.linear, as: UInt32.self)
                    (base + 4).storeBytes(of: limit.axis, as: UInt32.self)
                }
            }

            var poolInfo = VkCommandPoolCreateInfo()
            poolInfo.queueFamilyIndex = native.queueFamily
            try vulkanCheck(withUnsafePointer(to: &poolInfo) {
                f.createCommandPool(device, UnsafeRawPointer($0), nil, &batch.commandPool)
            }, "command pool creation")
            var allocateInfo = VkCommandBufferAllocateInfo()
            allocateInfo.commandPool = batch.commandPool
            var commandBuffer: UInt = 0
            try vulkanCheck(withUnsafePointer(to: &allocateInfo) {
                f.allocateCommandBuffers(device, UnsafeRawPointer($0), &commandBuffer)
            }, "command buffer allocation")

            let sizes = descriptorTotals.counts.filter { $0.value > 0 }.map { VkDescriptorPoolSize(type: $0.key, descriptorCount: $0.value) }
            if descriptorTotals.sets > 0, !sizes.isEmpty {
                try sizes.withUnsafeBufferPointer { pointer in
                    var info = VkDescriptorPoolCreateInfo()
                    info.maxSets = descriptorTotals.sets
                    info.poolSizeCount = UInt32(pointer.count)
                    info.pPoolSizes = pointer.baseAddress
                    try vulkanCheck(withUnsafePointer(to: &info) {
                        f.createDescriptorPool(device, UnsafeRawPointer($0), nil, &batch.descriptorPool)
                    }, "descriptor pool creation")
                }
            }

            var begin = VkCommandBufferBeginInfo()
            try vulkanCheck(withUnsafePointer(to: &begin) { f.beginCommandBuffer(commandBuffer, UnsafeRawPointer($0)) }, "command recording")
            func barrier(sourceStages: UInt32 = VK.pipelineStageAllCommands, destinationStages: UInt32 = VK.pipelineStageAllCommands,
                         destinationAccess: UInt32 = VK.accessMemoryRead | VK.accessMemoryWrite) {
                var memory = VkMemoryBarrier()
                memory.srcAccessMask = VK.accessMemoryWrite
                memory.dstAccessMask = destinationAccess
                withUnsafePointer(to: &memory) {
                    f.cmdPipelineBarrier(commandBuffer, sourceStages, destinationStages, 0, 1, UnsafeRawPointer($0), 0, nil, 0, nil)
                }
            }
            func copy(_ source: UInt64, _ sourceOffset: UInt64, _ destination: UInt64, _ destinationOffset: UInt64, _ count: UInt64) {
                var region = VkBufferCopy(srcOffset: sourceOffset, dstOffset: destinationOffset, size: count)
                f.cmdCopyBuffer(commandBuffer, source, destination, 1, &region)
            }
            for operation in operations {
                switch operation {
                case .upload(let destination, let offset, let stagingOffset, let count):
                    copy(batch.staging!.handle, stagingOffset, destination, offset, count)
                case .copy(let source, let sourceOffset, let destination, let destinationOffset, let count):
                    copy(source, sourceOffset, destination, destinationOffset, count)
                case .fill(let buffer, let offset, let count):
                    f.cmdFillBuffer(commandBuffer, buffer, offset, count, 0)
                case .read(let source, let readbackOffset, let count):
                    copy(source, 0, batch.readback!.handle, readbackOffset, count)
                case .imageCopy(let plan):
                    let bufferHandle: UInt64
                    switch plan.buffer {
                    case .staging: bufferHandle = batch.staging!.handle
                    case .handle(let handle): bufferHandle = handle
                    }
                    var region = plan.region
                    if plan.toImage {
                        native.g.cmdCopyBufferToImage(commandBuffer, bufferHandle, plan.image, VK.Image.layoutGeneral, 1, &region)
                    } else {
                        native.g.cmdCopyImageToBuffer(commandBuffer, plan.image, VK.Image.layoutGeneral, bufferHandle, 1, &region)
                    }
                case .resetQuery(let pool, let index):
                    native.g.cmdResetQueryPool(commandBuffer, pool, index, 1)
                case .writeTimestamp(let pool, let index, let end):
                    native.g.cmdWriteTimestamp(commandBuffer, end ? VulkanQuery.bottomOfPipe : VulkanQuery.topOfPipe, pool, index)
                case .resolveQueries(let resolve):
                    f.cmdFillBuffer(commandBuffer, resolve.destination, resolve.destinationOffset, UInt64(resolve.count) * 8, 0)
                    barrier()
                    for index in resolve.written {
                        native.g.cmdCopyQueryPoolResults(
                            commandBuffer, resolve.pool, index, 1, resolve.destination,
                            resolve.destinationOffset + UInt64(index - resolve.first) * 8, 8, VulkanQuery.results64Bit)
                    }
                case .render(var plan):
                    if let indirect = plan.indirect, let drawValidator, let scratch = batch.scratch {
                        let base = UInt64(indirect.slot) * Self.scratchStride
                        copy(indirect.source, indirect.offset, scratch.handle, base, plan.indexed == nil ? 16 : 20)
                        barrier()
                        try bindDescriptors(commandBuffer: commandBuffer, bindPoint: VK.pipelineBindPointCompute,
                                            layout: drawValidator.layout, setLayouts: drawValidator.setLayouts, writes: [
                            .init(set: 0, binding: 0, type: VK.DescriptorType.storageBuffer, buffer: scratch.handle, offset: base, range: 20),
                            .init(set: 0, binding: 1, type: VK.DescriptorType.storageBuffer, buffer: scratch.handle, offset: base + 256, range: 20),
                            .init(set: 0, binding: 2, type: VK.DescriptorType.storageBuffer, buffer: scratch.handle, offset: base + 512, range: 8),
                        ], batch: &batch)
                        f.cmdBindPipeline(commandBuffer, VK.pipelineBindPointCompute, drawValidator.pipeline)
                        f.cmdDispatch(commandBuffer, 1, 1, 1)
                        barrier()
                        plan.indirectArguments = (scratch.handle, base + 256)
                    }
                    try recordRender(plan, commandBuffer: commandBuffer, batch: &batch)
                case .dispatch(let plan):
                    func bind(_ pipeline: Pipeline, _ writeList: [DescriptorWrite]) throws {
                        try bindDescriptors(commandBuffer: commandBuffer, bindPoint: VK.pipelineBindPointCompute,
                                            layout: pipeline.layout, setLayouts: pipeline.setLayouts, writes: writeList,
                                            batch: &batch)
                        f.cmdBindPipeline(commandBuffer, VK.pipelineBindPointCompute, pipeline.pipeline)
                    }
                    var indirectOffset: UInt64 = 0
                    if let indirect = plan.indirect, let validator, let scratch = batch.scratch {
                        let base = UInt64(indirect.slot) * Self.scratchStride
                        copy(indirect.source, indirect.offset, scratch.handle, base, 12)
                        barrier()
                        try bind(validator, [
                            .init(set: 0, binding: 0, type: VK.DescriptorType.storageBuffer, buffer: scratch.handle, offset: base, range: 12),
                            .init(set: 0, binding: 1, type: VK.DescriptorType.storageBuffer, buffer: scratch.handle, offset: base + 256, range: 12),
                            .init(set: 0, binding: 2, type: VK.DescriptorType.storageBuffer, buffer: scratch.handle, offset: base + 512, range: 8),
                        ])
                        f.cmdDispatch(commandBuffer, 1, 1, 1)
                        barrier()
                        indirectOffset = base + 256
                    }
                    try bind(plan.pipeline, plan.writes)
                    if !plan.immediates.isEmpty {
                        plan.immediates.withUnsafeBytes {
                            f.cmdPushConstants(commandBuffer, plan.pipeline.layout, VK.shaderStageCompute, 0,
                                               UInt32($0.count), $0.baseAddress!)
                        }
                    }
                    if plan.indirect != nil, let scratch = batch.scratch {
                        f.cmdDispatchIndirect(commandBuffer, scratch.handle, indirectOffset)
                    } else {
                        f.cmdDispatch(commandBuffer, plan.groups.x, plan.groups.y, plan.groups.z)
                    }
                }
                barrier()
            }
            barrier(destinationStages: VK.pipelineStageHost, destinationAccess: VK.accessHostRead)
            try vulkanCheck(f.endCommandBuffer(commandBuffer), "command recording")

            var fenceInfo = VkFenceCreateInfo()
            try vulkanCheck(withUnsafePointer(to: &fenceInfo) {
                f.createFence(device, UnsafeRawPointer($0), nil, &batch.fence)
            }, "fence creation")
            var submit = VkSubmitInfo()
            submit.commandBufferCount = 1
            let queue = native.queue
            let fence = batch.fence
            try withUnsafePointer(to: &commandBuffer) { buffer in
                submit.pCommandBuffers = buffer
                try vulkanCheck(withUnsafePointer(to: &submit) { f.queueSubmit(queue, 1, UnsafeRawPointer($0), fence) },
                                "queue submission")
            }
        } catch {
            if case GPUError.deviceLost = error { terminal = true }
            release(batch)
            throw error
        }

        do {
            try await waitForFence(batch.fence)
        } catch {
            terminal = true
            // The GPU may still own these objects; leaking them is safer than destroying them.
            quarantine.append(batch)
            throw error
        }
        var results: [[UInt32]] = []
        if let readback = batch.readback {
            for (operation, size) in zip(operations.compactMap { operation -> UInt64? in
                if case .read(_, let offset, _) = operation { return offset }
                return nil
            }, readSizes) {
                let words = (0..<Int(size / 4)).map {
                    readback.mapped.loadUnaligned(fromByteOffset: Int(operation) + $0 * 4, as: UInt32.self)
                }
                results.append(words)
            }
        }
        release(batch)
        return results
    }

    /// Per indirect dispatch: raw counts at 0, validated counts at 256, limits at 512.
    private static let scratchStride: UInt64 = 768

    /// Zeroes GPU-authored workgroup counts that exceed device limits, as the Metal backend does.
    private func dispatchValidatorPipeline() throws -> Pipeline {
        let shader = try IndependentShaderCompiler().compileCompute(.wgsl("""
            @group(0) @binding(0) var<storage, read> source: array<u32, 3>;
            @group(0) @binding(1) var<storage, read_write> destination: array<u32, 3>;
            @group(0) @binding(2) var<storage, read> limits: array<u32, 2>;
            @compute @workgroup_size(1)
            fn validate_dispatch() {
                let x = source[0];
                let y = source[1];
                let z = source[2];
                var invalid = x > limits[1] || y > limits[1] || z > limits[1] || x == 0u || y == 0u || z == 0u;
                if (!invalid && limits[0] != 0u) { invalid = z > limits[0] / x / y; }
                if (invalid) {
                    destination[0] = 0u; destination[1] = 0u; destination[2] = 0u;
                } else {
                    destination[0] = x; destination[1] = y; destination[2] = z;
                }
            }
            """), to: .spirv)
        guard case .spirv(let words) = shader.artifact else { throw GPUError.backendFailure("Dispatch validator is not SPIR-V") }
        let model = try ComputePipeline(id: ResourceID(device: 0, serial: 0), shader: shader)
        return try makePipeline(model, words: words, entryPoint: shader.entryPoint)
    }

    func handle(_ id: ResourceID) throws -> UInt64 {
        guard let buffer = buffers[id] else { throw GPUError.validation("Unknown Vulkan buffer") }
        return buffer.native.handle
    }

    /// Polls the fence without blocking a cooperative thread. Cancelling the caller does not stop the wait,
    /// because native resources must outlive the GPU work.
    func waitForFence(_ fence: UInt64) async throws {
        let clock = ContinuousClock()
        let deadline = clock.now + timeout
        var delay = Duration.microseconds(50)
        while true {
            let status = native.f.getFenceStatus(native.device, fence)
            if status == VK.success { return }
            guard status == VK.notReady else { try vulkanCheck(status, "fence wait"); return }
            guard clock.now < deadline else { throw GPUError.timeout }
            try? await Task.sleep(for: delay)
            delay = min(delay * 2, .milliseconds(2))
        }
    }
}
