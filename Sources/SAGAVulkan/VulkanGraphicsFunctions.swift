import SAGATypes

/// Image, render pass and graphics pipeline entry points, resolved separately from the compute set.
struct VulkanGraphicsFunctions: Sendable {
    let createImage: @convention(c) (UInt, UnsafeRawPointer, UnsafeRawPointer?, UnsafeMutablePointer<UInt64>) -> VkResult
    let destroyImage: @convention(c) (UInt, UInt64, UnsafeRawPointer?) -> Void
    let getImageMemoryRequirements: @convention(c) (UInt, UInt64, UnsafeMutableRawPointer) -> Void
    let bindImageMemory: @convention(c) (UInt, UInt64, UInt64, UInt64) -> VkResult
    let createImageView: @convention(c) (UInt, UnsafeRawPointer, UnsafeRawPointer?, UnsafeMutablePointer<UInt64>) -> VkResult
    let destroyImageView: @convention(c) (UInt, UInt64, UnsafeRawPointer?) -> Void
    let createRenderPass: @convention(c) (UInt, UnsafeRawPointer, UnsafeRawPointer?, UnsafeMutablePointer<UInt64>) -> VkResult
    let destroyRenderPass: @convention(c) (UInt, UInt64, UnsafeRawPointer?) -> Void
    let createFramebuffer: @convention(c) (UInt, UnsafeRawPointer, UnsafeRawPointer?, UnsafeMutablePointer<UInt64>) -> VkResult
    let destroyFramebuffer: @convention(c) (UInt, UInt64, UnsafeRawPointer?) -> Void
    let createGraphicsPipelines:
        @convention(c) (UInt, UInt64, UInt32, UnsafeRawPointer, UnsafeRawPointer?, UnsafeMutablePointer<UInt64>) -> VkResult
    let cmdBeginRenderPass: @convention(c) (UInt, UnsafeRawPointer, UInt32) -> Void
    let cmdEndRenderPass: @convention(c) (UInt) -> Void
    let cmdBindVertexBuffers: @convention(c) (UInt, UInt32, UInt32, UnsafePointer<UInt64>, UnsafePointer<UInt64>) -> Void
    let cmdDraw: @convention(c) (UInt, UInt32, UInt32, UInt32, UInt32) -> Void
    let cmdDrawIndexed: @convention(c) (UInt, UInt32, UInt32, UInt32, Int32, UInt32) -> Void
    let cmdBindIndexBuffer: @convention(c) (UInt, UInt64, UInt64, UInt32) -> Void
    let cmdSetStencilReference: @convention(c) (UInt, UInt32, UInt32) -> Void
    let createQueryPool: @convention(c) (UInt, UnsafeRawPointer, UnsafeRawPointer?, UnsafeMutablePointer<UInt64>) -> VkResult
    let destroyQueryPool: @convention(c) (UInt, UInt64, UnsafeRawPointer?) -> Void
    let cmdResetQueryPool: @convention(c) (UInt, UInt64, UInt32, UInt32) -> Void
    let cmdWriteTimestamp: @convention(c) (UInt, UInt32, UInt64, UInt32) -> Void
    let cmdBeginQuery: @convention(c) (UInt, UInt64, UInt32, UInt32) -> Void
    let cmdEndQuery: @convention(c) (UInt, UInt64, UInt32) -> Void
    let cmdCopyQueryPoolResults: @convention(c) (UInt, UInt64, UInt32, UInt32, UInt64, UInt64, UInt64, UInt32) -> Void
    let cmdDrawIndirect: @convention(c) (UInt, UInt64, UInt64, UInt32, UInt32) -> Void
    let cmdDrawIndexedIndirect: @convention(c) (UInt, UInt64, UInt64, UInt32, UInt32) -> Void
    let cmdSetViewport: @convention(c) (UInt, UInt32, UInt32, UnsafeRawPointer) -> Void
    let cmdSetScissor: @convention(c) (UInt, UInt32, UInt32, UnsafeRawPointer) -> Void
    let cmdSetBlendConstants: @convention(c) (UInt, UnsafePointer<Float>) -> Void
    let cmdClearColorImage: @convention(c) (UInt, UInt64, UInt32, UnsafeRawPointer, UInt32, UnsafeRawPointer) -> Void
    let cmdCopyBufferToImage: @convention(c) (UInt, UInt64, UInt64, UInt32, UInt32, UnsafeRawPointer) -> Void
    let createSampler: @convention(c) (UInt, UnsafeRawPointer, UnsafeRawPointer?, UnsafeMutablePointer<UInt64>) -> VkResult
    let destroySampler: @convention(c) (UInt, UInt64, UnsafeRawPointer?) -> Void
    let cmdCopyImageToBuffer: @convention(c) (UInt, UInt64, UInt32, UInt64, UInt32, UnsafeRawPointer) -> Void
    let cmdClearDepthStencilImage: @convention(c) (UInt, UInt64, UInt32, UnsafeRawPointer, UInt32, UnsafeRawPointer) -> Void

    init(instance: VulkanInstance) throws {
        guard let library = VulkanLibrary.open(), let address = library.symbol("vkGetInstanceProcAddr") else {
            throw GPUError.adapterUnavailable
        }
        let getInstanceProcAddr = unsafeBitCast(address, to: GetInstanceProcAddr.self)
        let handle = instance.handle
        func load<T>(_ name: String, as type: T.Type = T.self) throws -> T {
            guard let address = getInstanceProcAddr(handle, name) else {
                throw GPUError.unsupported("Vulkan loader does not export \(name)")
            }
            return unsafeBitCast(address, to: T.self)
        }
        createSampler = try load("vkCreateSampler")
        destroySampler = try load("vkDestroySampler")
        createImage = try load("vkCreateImage")
        destroyImage = try load("vkDestroyImage")
        getImageMemoryRequirements = try load("vkGetImageMemoryRequirements")
        bindImageMemory = try load("vkBindImageMemory")
        createImageView = try load("vkCreateImageView")
        destroyImageView = try load("vkDestroyImageView")
        createRenderPass = try load("vkCreateRenderPass")
        destroyRenderPass = try load("vkDestroyRenderPass")
        createFramebuffer = try load("vkCreateFramebuffer")
        destroyFramebuffer = try load("vkDestroyFramebuffer")
        createGraphicsPipelines = try load("vkCreateGraphicsPipelines")
        cmdBeginRenderPass = try load("vkCmdBeginRenderPass")
        cmdEndRenderPass = try load("vkCmdEndRenderPass")
        cmdBindVertexBuffers = try load("vkCmdBindVertexBuffers")
        cmdDraw = try load("vkCmdDraw")
        cmdDrawIndexed = try load("vkCmdDrawIndexed")
        cmdBindIndexBuffer = try load("vkCmdBindIndexBuffer")
        cmdSetStencilReference = try load("vkCmdSetStencilReference")
        createQueryPool = try load("vkCreateQueryPool")
        destroyQueryPool = try load("vkDestroyQueryPool")
        cmdResetQueryPool = try load("vkCmdResetQueryPool")
        cmdWriteTimestamp = try load("vkCmdWriteTimestamp")
        cmdBeginQuery = try load("vkCmdBeginQuery")
        cmdEndQuery = try load("vkCmdEndQuery")
        cmdCopyQueryPoolResults = try load("vkCmdCopyQueryPoolResults")
        cmdDrawIndirect = try load("vkCmdDrawIndirect")
        cmdDrawIndexedIndirect = try load("vkCmdDrawIndexedIndirect")
        cmdSetViewport = try load("vkCmdSetViewport")
        cmdSetScissor = try load("vkCmdSetScissor")
        cmdSetBlendConstants = try load("vkCmdSetBlendConstants")
        cmdClearColorImage = try load("vkCmdClearColorImage")
        cmdCopyBufferToImage = try load("vkCmdCopyBufferToImage")
        cmdCopyImageToBuffer = try load("vkCmdCopyImageToBuffer")
        cmdClearDepthStencilImage = try load("vkCmdClearDepthStencilImage")
    }
}
