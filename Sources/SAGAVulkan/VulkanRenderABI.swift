// Image, render pass and graphics pipeline declarations used by the Vulkan render path.

extension VK {
    enum Image {
        static let layoutUndefined: UInt32 = 0
        static let layoutGeneral: UInt32 = 1
        static let usageTransferSource: UInt32 = 0x1
        static let usageTransferDestination: UInt32 = 0x2
        static let usageSampled: UInt32 = 0x4
        static let usageStorage: UInt32 = 0x8
        static let usageColorAttachment: UInt32 = 0x10
        static let aspectColor: UInt32 = 0x1
        static let type2D: UInt32 = 1
        static let viewType2D: UInt32 = 1
        static let tilingOptimal: UInt32 = 0
        static let sampleCount1: UInt32 = 1
    }

    enum Stage {
        static let vertex: UInt32 = 0x1
        static let fragment: UInt32 = 0x10
        static let pipelineTopOfPipe: UInt32 = 0x1
        static let pipelineColorAttachmentOutput: UInt32 = 0x400
        static let pipelineTransfer: UInt32 = 0x1000
    }

    static let structureImageCreateInfo: UInt32 = 14
    static let structureImageViewCreateInfo: UInt32 = 15
    static let structureGraphicsPipelineCreateInfo: UInt32 = 28
    static let structureVertexInputState: UInt32 = 19
    static let structureInputAssemblyState: UInt32 = 20
    static let structureViewportState: UInt32 = 22
    static let structureRasterizationState: UInt32 = 23
    static let structureMultisampleState: UInt32 = 24
    static let structureColorBlendState: UInt32 = 26
    static let structureDynamicState: UInt32 = 27
    static let structureFramebufferCreateInfo: UInt32 = 37
    static let structureRenderPassCreateInfo: UInt32 = 38
    static let structureRenderPassBeginInfo: UInt32 = 43
    static let structureImageMemoryBarrier: UInt32 = 45
    static let pipelineBindPointGraphics: UInt32 = 0
    static let subpassContentsInline: UInt32 = 0
    static let dynamicStateViewport: UInt32 = 0
    static let dynamicStateScissor: UInt32 = 1
    static let dynamicStateBlendConstants: UInt32 = 4
    static let dynamicStateStencilReference: UInt32 = 8
}

struct VkExtent3D {
    var width: UInt32 = 0
    var height: UInt32 = 0
    var depth: UInt32 = 1
}

struct VkOffset3D {
    var x: Int32 = 0
    var y: Int32 = 0
    var z: Int32 = 0
}

struct VkImageCreateInfo {
    var sType: UInt32 = VK.structureImageCreateInfo
    var pNext: UnsafeRawPointer?
    var flags: UInt32 = 0
    var imageType: UInt32 = VK.Image.type2D
    var format: UInt32 = 0
    var extent = VkExtent3D()
    var mipLevels: UInt32 = 1
    var arrayLayers: UInt32 = 1
    var samples: UInt32 = VK.Image.sampleCount1
    var tiling: UInt32 = VK.Image.tilingOptimal
    var usage: UInt32 = 0
    var sharingMode: UInt32 = VK.sharingModeExclusive
    var queueFamilyIndexCount: UInt32 = 0
    var pQueueFamilyIndices: UnsafePointer<UInt32>?
    var initialLayout: UInt32 = VK.Image.layoutUndefined
}

struct VkImageSubresourceRange {
    var aspectMask: UInt32 = VK.Image.aspectColor
    var baseMipLevel: UInt32 = 0
    var levelCount: UInt32 = 1
    var baseArrayLayer: UInt32 = 0
    var layerCount: UInt32 = 1
}

struct VkImageViewCreateInfo {
    var sType: UInt32 = VK.structureImageViewCreateInfo
    var pNext: UnsafeRawPointer?
    var flags: UInt32 = 0
    var image: UInt64 = 0
    var viewType: UInt32 = VK.Image.viewType2D
    var format: UInt32 = 0
    var componentR: UInt32 = 0
    var componentG: UInt32 = 0
    var componentB: UInt32 = 0
    var componentA: UInt32 = 0
    var subresourceRange = VkImageSubresourceRange()
}

struct VkImageMemoryBarrier {
    var sType: UInt32 = VK.structureImageMemoryBarrier
    var pNext: UnsafeRawPointer?
    var srcAccessMask: UInt32 = 0
    var dstAccessMask: UInt32 = 0
    var oldLayout: UInt32 = 0
    var newLayout: UInt32 = 0
    var srcQueueFamilyIndex: UInt32 = UInt32.max
    var dstQueueFamilyIndex: UInt32 = UInt32.max
    var image: UInt64 = 0
    var subresourceRange = VkImageSubresourceRange()
}

struct VkClearColorValue {
    var words: (UInt32, UInt32, UInt32, UInt32) = (0, 0, 0, 0)
}

struct VkBufferImageCopy {
    var bufferOffset: UInt64 = 0
    var bufferRowLength: UInt32 = 0
    var bufferImageHeight: UInt32 = 0
    var aspectMask: UInt32 = VK.Image.aspectColor
    var mipLevel: UInt32 = 0
    var baseArrayLayer: UInt32 = 0
    var layerCount: UInt32 = 1
    var imageOffset = VkOffset3D()
    var imageExtent = VkExtent3D()
}

struct VkAttachmentDescription {
    var flags: UInt32 = 0
    var format: UInt32 = 0
    var samples: UInt32 = VK.Image.sampleCount1
    var loadOp: UInt32 = 0
    var storeOp: UInt32 = 0
    var stencilLoadOp: UInt32 = 2
    var stencilStoreOp: UInt32 = 1
    var initialLayout: UInt32 = VK.Image.layoutGeneral
    var finalLayout: UInt32 = VK.Image.layoutGeneral
}

struct VkAttachmentReference {
    var attachment: UInt32 = 0
    var layout: UInt32 = VK.Image.layoutGeneral
}

struct VkSubpassDescription {
    var flags: UInt32 = 0
    var pipelineBindPoint: UInt32 = VK.pipelineBindPointGraphics
    var inputAttachmentCount: UInt32 = 0
    var pInputAttachments: UnsafeRawPointer?
    var colorAttachmentCount: UInt32 = 0
    var pColorAttachments: UnsafeRawPointer?
    var pResolveAttachments: UnsafeRawPointer?
    var pDepthStencilAttachment: UnsafeRawPointer?
    var preserveAttachmentCount: UInt32 = 0
    var pPreserveAttachments: UnsafeRawPointer?
}

struct VkRenderPassCreateInfo {
    var sType: UInt32 = VK.structureRenderPassCreateInfo
    var pNext: UnsafeRawPointer?
    var flags: UInt32 = 0
    var attachmentCount: UInt32 = 0
    var pAttachments: UnsafeRawPointer?
    var subpassCount: UInt32 = 0
    var pSubpasses: UnsafeRawPointer?
    var dependencyCount: UInt32 = 0
    var pDependencies: UnsafeRawPointer?
}

struct VkFramebufferCreateInfo {
    var sType: UInt32 = VK.structureFramebufferCreateInfo
    var pNext: UnsafeRawPointer?
    var flags: UInt32 = 0
    var renderPass: UInt64 = 0
    var attachmentCount: UInt32 = 0
    var pAttachments: UnsafePointer<UInt64>?
    var width: UInt32 = 0
    var height: UInt32 = 0
    var layers: UInt32 = 1
}

struct VkRenderPassBeginInfo {
    var sType: UInt32 = VK.structureRenderPassBeginInfo
    var pNext: UnsafeRawPointer?
    var renderPass: UInt64 = 0
    var framebuffer: UInt64 = 0
    var areaX: Int32 = 0
    var areaY: Int32 = 0
    var areaWidth: UInt32 = 0
    var areaHeight: UInt32 = 0
    var clearValueCount: UInt32 = 0
    var pClearValues: UnsafeRawPointer?
}

struct VkVertexInputBindingDescription {
    var binding: UInt32 = 0
    var stride: UInt32 = 0
    var inputRate: UInt32 = 0
}

struct VkVertexInputAttributeDescription {
    var location: UInt32 = 0
    var binding: UInt32 = 0
    var format: UInt32 = 0
    var offset: UInt32 = 0
}

struct VkPipelineVertexInputStateCreateInfo {
    var sType: UInt32 = VK.structureVertexInputState
    var pNext: UnsafeRawPointer?
    var flags: UInt32 = 0
    var vertexBindingDescriptionCount: UInt32 = 0
    var pVertexBindingDescriptions: UnsafeRawPointer?
    var vertexAttributeDescriptionCount: UInt32 = 0
    var pVertexAttributeDescriptions: UnsafeRawPointer?
}

struct VkPipelineInputAssemblyStateCreateInfo {
    var sType: UInt32 = VK.structureInputAssemblyState
    var pNext: UnsafeRawPointer?
    var flags: UInt32 = 0
    var topology: UInt32 = 3
    var primitiveRestartEnable: UInt32 = 0
}

struct VkPipelineViewportStateCreateInfo {
    var sType: UInt32 = VK.structureViewportState
    var pNext: UnsafeRawPointer?
    var flags: UInt32 = 0
    var viewportCount: UInt32 = 1
    var pViewports: UnsafeRawPointer?
    var scissorCount: UInt32 = 1
    var pScissors: UnsafeRawPointer?
}

struct VkPipelineRasterizationStateCreateInfo {
    var sType: UInt32 = VK.structureRasterizationState
    var pNext: UnsafeRawPointer?
    var flags: UInt32 = 0
    var depthClampEnable: UInt32 = 0
    var rasterizerDiscardEnable: UInt32 = 0
    var polygonMode: UInt32 = 0
    var cullMode: UInt32 = 0
    var frontFace: UInt32 = 0
    var depthBiasEnable: UInt32 = 0
    var depthBiasConstantFactor: Float = 0
    var depthBiasClamp: Float = 0
    var depthBiasSlopeFactor: Float = 0
    var lineWidth: Float = 1
}

struct VkPipelineMultisampleStateCreateInfo {
    var sType: UInt32 = VK.structureMultisampleState
    var pNext: UnsafeRawPointer?
    var flags: UInt32 = 0
    var rasterizationSamples: UInt32 = VK.Image.sampleCount1
    var sampleShadingEnable: UInt32 = 0
    var minSampleShading: Float = 0
    var pSampleMask: UnsafeRawPointer?
    var alphaToCoverageEnable: UInt32 = 0
    var alphaToOneEnable: UInt32 = 0
}

struct VkPipelineColorBlendAttachmentState {
    var blendEnable: UInt32 = 0
    var srcColorBlendFactor: UInt32 = 1
    var dstColorBlendFactor: UInt32 = 0
    var colorBlendOp: UInt32 = 0
    var srcAlphaBlendFactor: UInt32 = 1
    var dstAlphaBlendFactor: UInt32 = 0
    var alphaBlendOp: UInt32 = 0
    var colorWriteMask: UInt32 = 0xF
}

struct VkPipelineColorBlendStateCreateInfo {
    var sType: UInt32 = VK.structureColorBlendState
    var pNext: UnsafeRawPointer?
    var flags: UInt32 = 0
    var logicOpEnable: UInt32 = 0
    var logicOp: UInt32 = 0
    var attachmentCount: UInt32 = 0
    var pAttachments: UnsafeRawPointer?
    var blendConstant0: Float = 0
    var blendConstant1: Float = 0
    var blendConstant2: Float = 0
    var blendConstant3: Float = 0
}

struct VkPipelineDynamicStateCreateInfo {
    var sType: UInt32 = VK.structureDynamicState
    var pNext: UnsafeRawPointer?
    var flags: UInt32 = 0
    var dynamicStateCount: UInt32 = 0
    var pDynamicStates: UnsafePointer<UInt32>?
}

struct VkGraphicsPipelineCreateInfo {
    var sType: UInt32 = VK.structureGraphicsPipelineCreateInfo
    var pNext: UnsafeRawPointer?
    var flags: UInt32 = 0
    var stageCount: UInt32 = 0
    var pStages: UnsafeRawPointer?
    var pVertexInputState: UnsafeRawPointer?
    var pInputAssemblyState: UnsafeRawPointer?
    var pTessellationState: UnsafeRawPointer?
    var pViewportState: UnsafeRawPointer?
    var pRasterizationState: UnsafeRawPointer?
    var pMultisampleState: UnsafeRawPointer?
    var pDepthStencilState: UnsafeRawPointer?
    var pColorBlendState: UnsafeRawPointer?
    var pDynamicState: UnsafeRawPointer?
    var layout: UInt64 = 0
    var renderPass: UInt64 = 0
    var subpass: UInt32 = 0
    var basePipelineHandle: UInt64 = 0
    var basePipelineIndex: Int32 = -1
}

struct VkViewport {
    var x: Float = 0
    var y: Float = 0
    var width: Float = 0
    var height: Float = 0
    var minDepth: Float = 0
    var maxDepth: Float = 1
}

struct VkRect2D {
    var x: Int32 = 0
    var y: Int32 = 0
    var width: UInt32 = 0
    var height: UInt32 = 0
}

extension VK {
    enum Depth {
        static let usageAttachment: UInt32 = 0x20
        static let aspectDepth: UInt32 = 0x2
        static let aspectStencil: UInt32 = 0x4
        static let structureDepthStencilState: UInt32 = 25
        static let layoutAttachment: UInt32 = 1  // images stay in GENERAL
    }
}

struct VkStencilOpState {
    var failOp: UInt32 = 0
    var passOp: UInt32 = 0
    var depthFailOp: UInt32 = 0
    var compareOp: UInt32 = 7
    var compareMask: UInt32 = 0
    var writeMask: UInt32 = 0
    var reference: UInt32 = 0
}

struct VkPipelineDepthStencilStateCreateInfo {
    var sType: UInt32 = VK.Depth.structureDepthStencilState
    var pNext: UnsafeRawPointer?
    var flags: UInt32 = 0
    var depthTestEnable: UInt32 = 0
    var depthWriteEnable: UInt32 = 0
    var depthCompareOp: UInt32 = 7
    var depthBoundsTestEnable: UInt32 = 0
    var stencilTestEnable: UInt32 = 0
    var front = VkStencilOpState()
    var back = VkStencilOpState()
    var minDepthBounds: Float = 0
    var maxDepthBounds: Float = 1
}

struct VkClearDepthStencilValue {
    var depth: Float = 0
    var stencil: UInt32 = 0
}

struct VkRenderPassMultiviewCreateInfo {
    var sType: UInt32 = 1000053000
    var pNext: UnsafeRawPointer?
    var subpassCount: UInt32 = 0
    var pViewMasks: UnsafePointer<UInt32>?
    var dependencyCount: UInt32 = 0
    var pViewOffsets: UnsafePointer<Int32>?
    var correlationMaskCount: UInt32 = 0
    var pCorrelationMasks: UnsafePointer<UInt32>?
}
