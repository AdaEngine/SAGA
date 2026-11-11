import SAGATypes

extension VulkanDevice {
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

    func submitBatch(_ batch: inout PendingBatch) throws {
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
            try vulkanCheck(withUnsafePointer(to: &submit) {
                f.queueSubmit(native.queue, 1, UnsafeRawPointer($0), fence)
            }, "queue submission")
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
        batch.views.forEach { native.g.destroyImageView(device, $0, nil) }
    }
}
