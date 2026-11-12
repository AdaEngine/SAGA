import SAGATypes

/// Feature flags and limits reported by the JavaScript WebGPU host.
public struct WebHostParameters: Sendable {
    public var maxBufferSize: UInt64 = 0
    public var maxTextureDimension2D: UInt32 = 0
    public var presentation: UInt32 = 0
    public var readableStorage: UInt32 = 0
    public var shaderF16: UInt32 = 0
    public var uniformBufferStandardLayout: UInt32 = 0
    public var indirectFirstInstance: UInt32 = 0
    public var clipDistances: UInt32 = 0
    public var primitiveIndex: UInt32 = 0
    public var depthClipControl: UInt32 = 0
    public var timestampQuery: UInt32 = 0
    public var subgroups: UInt32 = 0
    public var subgroupID: UInt32 = 0
    public var dualSourceBlending: UInt32 = 0
    public var textureCompressionBC: UInt32 = 0
    public var textureCompressionETC2: UInt32 = 0
    public var textureCompressionASTC: UInt32 = 0
    public var textureCompressionBCSliced3D: UInt32 = 0
    public var textureCompressionASTCSliced3D: UInt32 = 0
    public var float32Filterable: UInt32 = 0
    public var float32Blendable: UInt32 = 0
    public var multisampledFloat32Mask: UInt32 = 0
    public var textureFormatsTier1: UInt32 = 0
    public var textureFormatsTier2: UInt32 = 0
    public var rg11b10UfloatRenderable: UInt32 = 0
    public var depth32FloatStencil8: UInt32 = 0
    public var bgra8UnormStorage: UInt32 = 0
    public var maxImmediateSize: UInt32 = 0
    public var linearIndexing: UInt32 = 0
    public var subgroupSizeControl: UInt32 = 0
    public var subgroupMinSize: UInt32 = 0
    public var subgroupMaxSize: UInt32 = 0
    public var subgroupUniformity: UInt32 = 0
    public var maxComputeWorkgroupStorageSize: UInt32 = 0
    public var maxComputeInvocationsPerWorkgroup: UInt32 = 0
    public var maxComputeWorkgroupSizeX: UInt32 = 0
    public var maxComputeWorkgroupSizeY: UInt32 = 0
    public var maxComputeWorkgroupSizeZ: UInt32 = 0
    public var maxComputeWorkgroupsPerDimension: UInt32 = 0
    public var maxVertexAttributes: UInt32 = 0
    public var maxVertexBuffers: UInt32 = 0
    public var maxVertexBufferArrayStride: UInt32 = 0
    public var vertexFormatSnorm1010102: UInt32 = 0
    public var maxInterStageShaderVariables: UInt32 = 0
    public var maxColorAttachments: UInt32 = 0
    public var maxColorAttachmentBytesPerSample: UInt32 = 0
    public var maxBindGroupsPlusVertexBuffers: UInt32 = 0

    public init() {}
}

/// Validates the limits and maps the host feature flags to SAGA capabilities.
public func makeWebDeviceCapabilities(_ parameters: WebHostParameters) throws -> DeviceCapabilities {
    let maxBufferSize = parameters.maxBufferSize
    let maxTextureDimension2D = parameters.maxTextureDimension2D
    let readableStorage = parameters.readableStorage
    let shaderF16 = parameters.shaderF16
    let uniformBufferStandardLayout = parameters.uniformBufferStandardLayout
    let indirectFirstInstance = parameters.indirectFirstInstance
    let clipDistances = parameters.clipDistances
    let primitiveIndex = parameters.primitiveIndex
    let depthClipControl = parameters.depthClipControl
    let timestampQuery = parameters.timestampQuery
    let subgroups = parameters.subgroups
    let subgroupID = parameters.subgroupID
    let dualSourceBlending = parameters.dualSourceBlending
    let textureCompressionBC = parameters.textureCompressionBC
    let textureCompressionETC2 = parameters.textureCompressionETC2
    let textureCompressionASTC = parameters.textureCompressionASTC
    let textureCompressionBCSliced3D = parameters.textureCompressionBCSliced3D
    let textureCompressionASTCSliced3D = parameters.textureCompressionASTCSliced3D
    let float32Filterable = parameters.float32Filterable
    let float32Blendable = parameters.float32Blendable
    let multisampledFloat32Mask = parameters.multisampledFloat32Mask
    let textureFormatsTier1 = parameters.textureFormatsTier1
    let textureFormatsTier2 = parameters.textureFormatsTier2
    let rg11b10UfloatRenderable = parameters.rg11b10UfloatRenderable
    let depth32FloatStencil8 = parameters.depth32FloatStencil8
    let bgra8UnormStorage = parameters.bgra8UnormStorage
    let maxImmediateSize = parameters.maxImmediateSize
    let linearIndexing = parameters.linearIndexing
    let subgroupSizeControl = parameters.subgroupSizeControl
    let subgroupMinSize = parameters.subgroupMinSize
    let subgroupMaxSize = parameters.subgroupMaxSize
    let subgroupUniformity = parameters.subgroupUniformity
    let maxComputeWorkgroupStorageSize = parameters.maxComputeWorkgroupStorageSize
    let maxComputeInvocationsPerWorkgroup = parameters.maxComputeInvocationsPerWorkgroup
    let maxComputeWorkgroupSizeX = parameters.maxComputeWorkgroupSizeX
    let maxComputeWorkgroupSizeY = parameters.maxComputeWorkgroupSizeY
    let maxComputeWorkgroupSizeZ = parameters.maxComputeWorkgroupSizeZ
    let maxComputeWorkgroupsPerDimension = parameters.maxComputeWorkgroupsPerDimension
    let maxVertexAttributes = parameters.maxVertexAttributes
    let maxVertexBuffers = parameters.maxVertexBuffers
    let maxVertexBufferArrayStride = parameters.maxVertexBufferArrayStride
    let vertexFormatSnorm1010102 = parameters.vertexFormatSnorm1010102
    let maxInterStageShaderVariables = parameters.maxInterStageShaderVariables
    let maxColorAttachments = parameters.maxColorAttachments
    let maxColorAttachmentBytesPerSample = parameters.maxColorAttachmentBytesPerSample
    let maxBindGroupsPlusVertexBuffers = parameters.maxBindGroupsPlusVertexBuffers
    var features: Set<GPUFeature> = readableStorage == 0 ? [] : [.storageTextureRead, .storageTextureReadWrite]
    if shaderF16 != 0 { features.insert(.shaderF16) }
    if uniformBufferStandardLayout != 0 { features.insert(.uniformBufferStandardLayout) }
    if indirectFirstInstance != 0 { features.insert(.indirectFirstInstance) }
    if clipDistances != 0 { features.insert(.clipDistances) }
    if primitiveIndex != 0 { features.insert(.primitiveIndex) }
    if depthClipControl != 0 { features.insert(.depthClipControl) }
    if timestampQuery != 0 { features.insert(.timestampQuery) }
    if subgroups != 0 { features.insert(.subgroups) }
    if subgroupSizeControl != 0 { features.insert(.subgroupSizeControl) }
    if subgroupID != 0 { features.insert(.subgroupID) }
    if subgroupUniformity != 0 { features.insert(.subgroupUniformity) }
    if linearIndexing != 0 { features.insert(.linearIndexing) }
    if vertexFormatSnorm1010102 != 0 { features.insert(.vertexFormatSnorm1010102) }
    if dualSourceBlending != 0 { features.insert(.dualSourceBlending) }
    if textureCompressionBC != 0 { features.insert(.textureCompressionBC) }
    if textureCompressionETC2 != 0 { features.insert(.textureCompressionETC2) }
    if textureCompressionASTC != 0 { features.insert(.textureCompressionASTC) }
    if textureCompressionBCSliced3D != 0 { features.insert(.textureCompressionBCSliced3D) }
    if textureCompressionASTCSliced3D != 0 { features.insert(.textureCompressionASTCSliced3D) }
    if float32Filterable != 0 { features.insert(.float32Filterable) }
    if float32Blendable != 0 { features.insert(.float32Blendable) }
    if textureFormatsTier1 != 0 {
        features.formUnion([.halfFloatStorageTextures, .narrowInteger16StorageTextures, .normalized16Textures,
                            .packedColorStorageTextures, .narrow8BitStorageTextures, .snorm8Renderable])
    }
    if rg11b10UfloatRenderable != 0 { features.insert(.rg11b10UfloatRenderable) }
    if depth32FloatStencil8 != 0 { features.insert(.depth32FloatStencil8) }
    if bgra8UnormStorage != 0 { features.insert(.bgra8UnormStorage) }
    if textureFormatsTier2 != 0 { features.formUnion([.r16FloatStorageReadWrite, .integer16StorageReadWrite, .storage8BitReadWrite, .wideRGBAStorageReadWrite]) }
    let float32Formats: [TextureFormat] = [.r32Float, .rg32Float, .rgba32Float]
    let multisampledFloat32Formats = Set<TextureFormat>(float32Formats.enumerated().compactMap { index, format in
        multisampledFloat32Mask & (1 << index) == 0 ? nil : format
    })
    return try DeviceCapabilities(maxBufferSize: maxBufferSize, features: features,
                                               maxTextureDimension2D: maxTextureDimension2D,
                                               multisampledFloat32Formats: multisampledFloat32Formats,
                                               maxImmediateSize: maxImmediateSize,
                                               subgroupSizeRange: subgroupMinSize > 0 && subgroupMaxSize >= subgroupMinSize
                                                   ? subgroupMinSize...subgroupMaxSize : nil,
                                               maxComputeWorkgroupStorageSize: maxComputeWorkgroupStorageSize,
                                               maxComputeInvocationsPerWorkgroup: maxComputeInvocationsPerWorkgroup,
                                               maxComputeWorkgroupSize: .init(x: maxComputeWorkgroupSizeX,
                                                   y: maxComputeWorkgroupSizeY, z: maxComputeWorkgroupSizeZ),
                                               maxComputeWorkgroupsPerDimension: maxComputeWorkgroupsPerDimension,
                                               maxVertexAttributes: maxVertexAttributes, maxVertexBuffers: maxVertexBuffers,
                                               maxBindGroupsPlusVertexBuffers: maxBindGroupsPlusVertexBuffers,
                                               maxVertexBufferArrayStride: maxVertexBufferArrayStride,
                                               maxInterStageShaderVariables: maxInterStageShaderVariables,
                                               maxColorAttachments: maxColorAttachments,
                                               maxColorAttachmentBytesPerSample: maxColorAttachmentBytesPerSample)
}
