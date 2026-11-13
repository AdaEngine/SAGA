#if os(WASI) && compiler(>=6.3)
import SAGAWebHost
import SAGAWebReactor

@_expose(wasm, "saga_run_job")
func runJob(_ pointer: UnsafeMutableRawPointer) { sagaWebRunJob(pointer) }

@_expose(wasm, "saga_reply")
func reply(_ id: UInt32, _ status: UInt32, _ count: UInt32) { sagaWebReply(id, status, count) }

@_expose(wasm, "saga_start")
func start(_ maxBufferSize: UInt64,
           _ maxTextureDimension2D: UInt32,
           _ presentation: UInt32,
           _ readableStorage: UInt32,
           _ shaderF16: UInt32,
           _ uniformBufferStandardLayout: UInt32,
           _ indirectFirstInstance: UInt32,
           _ clipDistances: UInt32,
           _ primitiveIndex: UInt32,
           _ depthClipControl: UInt32,
           _ timestampQuery: UInt32,
           _ subgroups: UInt32,
           _ subgroupID: UInt32,
           _ dualSourceBlending: UInt32,
           _ textureCompressionBC: UInt32,
           _ textureCompressionETC2: UInt32,
           _ textureCompressionASTC: UInt32,
           _ textureCompressionBCSliced3D: UInt32,
           _ textureCompressionASTCSliced3D: UInt32,
           _ float32Filterable: UInt32,
           _ float32Blendable: UInt32,
           _ multisampledFloat32Mask: UInt32,
           _ textureFormatsTier1: UInt32,
           _ textureFormatsTier2: UInt32,
           _ rg11b10UfloatRenderable: UInt32,
           _ depth32FloatStencil8: UInt32,
           _ bgra8UnormStorage: UInt32,
           _ firstVideoFrame: UInt64,
           _ secondVideoFrame: UInt64,
           _ closedVideoFrame: UInt64,
           _ maxImmediateSize: UInt32,
           _ linearIndexing: UInt32,
           _ subgroupSizeControl: UInt32,
           _ subgroupMinSize: UInt32,
           _ subgroupMaxSize: UInt32,
           _ subgroupUniformity: UInt32,
           _ maxComputeWorkgroupStorageSize: UInt32,
           _ maxComputeInvocationsPerWorkgroup: UInt32,
           _ maxComputeWorkgroupSizeX: UInt32,
           _ maxComputeWorkgroupSizeY: UInt32,
           _ maxComputeWorkgroupSizeZ: UInt32,
           _ maxComputeWorkgroupsPerDimension: UInt32,
           _ maxVertexAttributes: UInt32,
           _ maxVertexBuffers: UInt32,
           _ maxVertexBufferArrayStride: UInt32,
           _ vertexFormatSnorm1010102: UInt32,
           _ maxInterStageShaderVariables: UInt32,
           _ maxColorAttachments: UInt32,
           _ maxColorAttachmentBytesPerSample: UInt32,
           _ maxBindGroupsPlusVertexBuffers: UInt32) {
    var p = WebHostParameters()
    p.maxBufferSize = maxBufferSize
    p.maxTextureDimension2D = maxTextureDimension2D
    p.presentation = presentation
    p.readableStorage = readableStorage
    p.shaderF16 = shaderF16
    p.uniformBufferStandardLayout = uniformBufferStandardLayout
    p.indirectFirstInstance = indirectFirstInstance
    p.clipDistances = clipDistances
    p.primitiveIndex = primitiveIndex
    p.depthClipControl = depthClipControl
    p.timestampQuery = timestampQuery
    p.subgroups = subgroups
    p.subgroupID = subgroupID
    p.dualSourceBlending = dualSourceBlending
    p.textureCompressionBC = textureCompressionBC
    p.textureCompressionETC2 = textureCompressionETC2
    p.textureCompressionASTC = textureCompressionASTC
    p.textureCompressionBCSliced3D = textureCompressionBCSliced3D
    p.textureCompressionASTCSliced3D = textureCompressionASTCSliced3D
    p.float32Filterable = float32Filterable
    p.float32Blendable = float32Blendable
    p.multisampledFloat32Mask = multisampledFloat32Mask
    p.textureFormatsTier1 = textureFormatsTier1
    p.textureFormatsTier2 = textureFormatsTier2
    p.rg11b10UfloatRenderable = rg11b10UfloatRenderable
    p.depth32FloatStencil8 = depth32FloatStencil8
    p.bgra8UnormStorage = bgra8UnormStorage
    p.maxImmediateSize = maxImmediateSize
    p.linearIndexing = linearIndexing
    p.subgroupSizeControl = subgroupSizeControl
    p.subgroupMinSize = subgroupMinSize
    p.subgroupMaxSize = subgroupMaxSize
    p.subgroupUniformity = subgroupUniformity
    p.maxComputeWorkgroupStorageSize = maxComputeWorkgroupStorageSize
    p.maxComputeInvocationsPerWorkgroup = maxComputeInvocationsPerWorkgroup
    p.maxComputeWorkgroupSizeX = maxComputeWorkgroupSizeX
    p.maxComputeWorkgroupSizeY = maxComputeWorkgroupSizeY
    p.maxComputeWorkgroupSizeZ = maxComputeWorkgroupSizeZ
    p.maxComputeWorkgroupsPerDimension = maxComputeWorkgroupsPerDimension
    p.maxVertexAttributes = maxVertexAttributes
    p.maxVertexBuffers = maxVertexBuffers
    p.maxVertexBufferArrayStride = maxVertexBufferArrayStride
    p.vertexFormatSnorm1010102 = vertexFormatSnorm1010102
    p.maxInterStageShaderVariables = maxInterStageShaderVariables
    p.maxColorAttachments = maxColorAttachments
    p.maxColorAttachmentBytesPerSample = maxColorAttachmentBytesPerSample
    p.maxBindGroupsPlusVertexBuffers = maxBindGroupsPlusVertexBuffers
    sagaWebStart(p) { capabilities, presentation, transport in
        try await runWebDiagnostic(capabilities: capabilities, supportsPresentation: presentation, videoFrames: (firstVideoFrame, secondVideoFrame, closedVideoFrame),
                                   transport: transport)
    }
}
#endif
