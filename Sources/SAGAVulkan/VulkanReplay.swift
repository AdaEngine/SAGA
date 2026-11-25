import SAGATypes

extension VulkanDevice {
    struct ImageViewKey: Hashable {
        let image: UInt64
        let format: UInt32
        let type: UInt32
        let range: VkImageSubresourceRange
    }

    /// All batch views use default component swizzles and live until the batch fence retires.
    func batchImageView(_ info: VkImageViewCreateInfo, batch: inout PendingBatch) throws -> UInt64 {
        let key = ImageViewKey(image: info.image, format: info.format, type: info.viewType, range: info.subresourceRange)
        if let view = batch.views[key] { return view }
        var view: UInt64 = 0
        try vulkanCheck(withUnsafePointer(to: info) {
            native.g.createImageView(native.device, UnsafeRawPointer($0), nil, &view)
        }, "image view creation")
        batch.views[key] = view
        return view
    }

    /// A single retired recording per backend. Its arenas keep the identity of the immutable list alive.
    struct DispatchRecording {
        let list: CommandList
        var batch: PendingBatch
        let resources: Set<ResourceID>

        func matches(_ other: CommandList) -> Bool {
            sameStorage(list.records, other.records)
                && sameStorage(list.pipelines, other.pipelines)
                && sameStorage(list.bindings, other.bindings)
                && sameStorage(list.immediates, other.immediates)
        }

        private func sameStorage<Element>(_ lhs: [Element], _ rhs: [Element]) -> Bool {
            guard lhs.count == rhs.count else { return false }
            if lhs.isEmpty { return true }
            return lhs.withUnsafeBufferPointer { left in
                rhs.withUnsafeBufferPointer { right in left.baseAddress == right.baseAddress }
            }
        }
    }

    /// While the fence is pending, other submissions record independently. Native retirement may
    /// remove a referenced handle from its lookup table but must leave it alive until the fence signals.
    struct PendingDispatchRecording {
        let resources: Set<ResourceID>
        var invalidated = false
        var retiredBuffers: [VulkanHostBuffer] = []
        var retiredPipelines: [Pipeline] = []
    }

    func invalidateDispatchRecording(using id: ResourceID) {
        if idleDispatchRecording?.resources.contains(id) == true { discardIdleDispatchRecording() }
        if pendingDispatchRecording?.resources.contains(id) == true { pendingDispatchRecording?.invalidated = true }
    }

    func discardIdleDispatchRecording() {
        guard let recording = idleDispatchRecording else { return }
        idleDispatchRecording = nil
        releaseBatch(recording.batch)
    }

    func finishDispatchRecording(_ recording: DispatchRecording) {
        guard let pending = pendingDispatchRecording else {
            releaseBatch(recording.batch)
            return
        }
        pendingDispatchRecording = nil
        if pending.invalidated || terminal {
            releaseBatch(recording.batch)
        } else {
            if let previous = idleDispatchRecording { releaseBatch(previous.batch, recycle: true) }
            idleDispatchRecording = recording
        }
        pending.retiredBuffers.forEach { native.destroy($0) }
        pending.retiredPipelines.forEach { destroyHelper($0) }
    }

    func replayDispatchRecording(_ recording: DispatchRecording) async throws {
        var recording = recording
        do {
            try submitBatch(&recording.batch)
        } catch {
            if case GPUError.deviceLost = error { terminal = true }
            pendingDispatchRecording = nil
            releaseBatch(recording.batch)
            throw error
        }
        try await waitForBatch(recording.batch)
        finishDispatchRecording(recording)
    }

    func acquireCommandBuffer(batch: inout PendingBatch) throws -> UInt {
        let f = native.f, device = native.device
        if let idle = idleCommandPools.popLast() {
            batch.commandPool = idle.pool
            batch.commandBuffer = idle.buffer
            try vulkanCheck(f.resetCommandPool(device, idle.pool, 0), "command pool reset")
        } else {
            var poolInfo = VkCommandPoolCreateInfo()
            poolInfo.queueFamilyIndex = native.queueFamily
            try vulkanCheck(withUnsafePointer(to: &poolInfo) {
                f.createCommandPool(device, UnsafeRawPointer($0), nil, &batch.commandPool)
            }, "command pool creation")
            var allocateInfo = VkCommandBufferAllocateInfo()
            allocateInfo.commandPool = batch.commandPool
            try vulkanCheck(withUnsafePointer(to: &allocateInfo) {
                f.allocateCommandBuffers(device, UnsafeRawPointer($0), &batch.commandBuffer)
            }, "command buffer allocation")
        }
        return batch.commandBuffer
    }

    func submitBatch(_ batch: inout PendingBatch, waiting waitSemaphore: UInt64 = 0, signaling signalSemaphore: UInt64 = 0) throws {
        let f = native.f
        let device = native.device
        if batch.fence == 0, let fence = idleFences.popLast() { batch.fence = fence }
        if batch.fence != 0 {
            var fence = batch.fence
            try vulkanCheck(f.resetFences(device, 1, &fence), "fence reset")
        } else {
            var info = VkFenceCreateInfo()
            try vulkanCheck(withUnsafePointer(to: &info) {
                f.createFence(device, UnsafeRawPointer($0), nil, &batch.fence)
            }, "fence creation")
        }
        let fence = batch.fence
        var commandBuffer = batch.commandBuffer
        var submit = VkSubmitInfo()
        submit.commandBufferCount = 1
        try withUnsafePointer(to: &commandBuffer) { buffer in
            submit.pCommandBuffers = buffer
            func submitToQueue() throws {
                try vulkanCheck(withUnsafePointer(to: &submit) {
                    f.queueSubmit(native.queue, 1, UnsafeRawPointer($0), fence)
                }, "queue submission")
            }
            if waitSemaphore == 0, signalSemaphore == 0 {
                try submitToQueue()
                return
            }
            try withUnsafePointer(to: waitSemaphore) { wait in
                try withUnsafePointer(to: signalSemaphore) { signal in
                    try withUnsafePointer(to: VKPresent.stageTransfer) { stage in
                        if waitSemaphore != 0 {
                            submit.waitSemaphoreCount = 1
                            submit.pWaitSemaphores = UnsafeRawPointer(wait)
                            submit.pWaitDstStageMask = UnsafeRawPointer(stage)
                        }
                        if signalSemaphore != 0 {
                            submit.signalSemaphoreCount = 1
                            submit.pSignalSemaphores = UnsafeRawPointer(signal)
                        }
                        try submitToQueue()
                    }
                }
            }
        }
    }

    func waitForBatch(_ batch: PendingBatch) async throws {
        do {
            try await waitForFence(batch.fence)
        } catch {
            terminal = true
            discardIdleDispatchRecording()
            // The GPU may still own the batch and the pending recording's retired dependencies.
            quarantine.append(batch)
            throw error
        }
    }

    /// Records initialization work and keeps its command pool alive through retirement.
    func runOneShot(_ record: (UInt) throws -> Void) async throws {
        var batch = PendingBatch()
        do {
            let commandBuffer = try acquireCommandBuffer(batch: &batch)
            var begin = VkCommandBufferBeginInfo()
            try vulkanCheck(withUnsafePointer(to: &begin) {
                native.f.beginCommandBuffer(commandBuffer, UnsafeRawPointer($0))
            }, "command recording")
            try record(commandBuffer)
            try vulkanCheck(native.f.endCommandBuffer(commandBuffer), "command recording")
            try submitBatch(&batch)
        } catch {
            if case GPUError.deviceLost = error { terminal = true }
            releaseBatch(batch)
            throw error
        }
        try await waitForBatch(batch)
        releaseBatch(batch, recycle: true)
    }

    /// `recycle` is only for batches whose fence signalled: their pool and fence go back to the idle lists.
    func releaseBatch(_ batch: PendingBatch, recycle: Bool = false) {
        let f = native.f
        let device = native.device
        if batch.fence != 0 {
            if recycle { idleFences.append(batch.fence) } else { f.destroyFence(device, batch.fence, nil) }
        }
        if batch.descriptorPool != 0 { f.destroyDescriptorPool(device, batch.descriptorPool, nil) }
        if batch.commandPool != 0 {
            if recycle { idleCommandPools.append((batch.commandPool, batch.commandBuffer)) } else { f.destroyCommandPool(device, batch.commandPool, nil) }
        }
        if let staging = batch.staging { if recycle { recycleHostBuffer(staging, readback: false) } else { native.destroy(staging) } }
        if let readback = batch.readback { if recycle { recycleHostBuffer(readback, readback: true) } else { native.destroy(readback) } }
        if let scratch = batch.scratch { native.destroy(scratch) }
        batch.owned.forEach { native.destroy($0) }
        batch.framebuffers.forEach { native.g.destroyFramebuffer(device, $0, nil) }
        batch.renderPasses.forEach { native.g.destroyRenderPass(device, $0, nil) }
        batch.views.values.forEach { native.g.destroyImageView(device, $0, nil) }
    }
}
