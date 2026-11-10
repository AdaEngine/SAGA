/// A device-owned mesh render pipeline with color targets.
public struct MeshPipeline: Sendable, Equatable {
    public let id: ResourceID
    package let hasTaskShader: Bool
    public let primitive: RenderPrimitiveState
    public let depthState: RenderDepthState?
    public let sampleMask: UInt32
    public let alphaToCoverageEnabled: Bool
    public let sampleCount: UInt32
    public let colorFormats: [UInt32: TextureFormat]
    public var colorFormat: TextureFormat { colorFormats[0]! }
    package let fragmentBufferBindings: [ComputePipeline.BufferBinding]
    package let fragmentTextures: [SampledTextureRequirement]
    package let fragmentStorageTextures: [StorageTextureRequirement]
    package let fragmentSamplers: [SamplerRequirement]
    package let taskBufferBindings: [ComputePipeline.BufferBinding]
    package let meshBufferBindings: [ComputePipeline.BufferBinding]

    package let taskTextures: [SampledTextureRequirement]
    package let meshTextures: [SampledTextureRequirement]
    package let taskStorageTextures: [StorageTextureRequirement]
    package let meshStorageTextures: [StorageTextureRequirement]
    package let taskSamplers: [SamplerRequirement]
    package let meshSamplers: [SamplerRequirement]

    package init(id: ResourceID, colorFormat: TextureFormat, sampleCount: UInt32 = 1, sampleMask: UInt32 = .max, alphaToCoverageEnabled: Bool = false, depthState: RenderDepthState? = nil, primitive: RenderPrimitiveState = .init(), hasTaskShader: Bool = false,
                 fragmentBufferBindings: [ComputePipeline.BufferBinding] = [], fragmentTextures: [SampledTextureRequirement] = [], fragmentStorageTextures: [StorageTextureRequirement] = [], fragmentSamplers: [SamplerRequirement] = [],
                 taskBufferBindings: [ComputePipeline.BufferBinding] = [],
                 meshBufferBindings: [ComputePipeline.BufferBinding] = [], taskTextures: [SampledTextureRequirement] = [], meshTextures: [SampledTextureRequirement] = [], taskSamplers: [SamplerRequirement] = [], meshSamplers: [SamplerRequirement] = [], taskStorageTextures: [StorageTextureRequirement] = [], meshStorageTextures: [StorageTextureRequirement] = []) {
        self.id = id
        self.hasTaskShader = hasTaskShader
        self.primitive = primitive
        self.sampleMask = sampleMask
        self.alphaToCoverageEnabled = alphaToCoverageEnabled
        self.depthState = depthState
        self.sampleCount = sampleCount
        self.colorFormats = [0: colorFormat]
        self.fragmentBufferBindings = fragmentBufferBindings; self.fragmentTextures = fragmentTextures; self.fragmentStorageTextures = fragmentStorageTextures; self.fragmentSamplers = fragmentSamplers
        self.taskBufferBindings = taskBufferBindings
        self.meshBufferBindings = meshBufferBindings
        self.taskTextures = taskTextures; self.meshTextures = meshTextures
        self.taskSamplers = taskSamplers; self.meshSamplers = meshSamplers
        self.taskStorageTextures = taskStorageTextures; self.meshStorageTextures = meshStorageTextures
    }
    package init(id: ResourceID, colorFormats: [UInt32: TextureFormat], sampleCount: UInt32 = 1, sampleMask: UInt32 = .max, alphaToCoverageEnabled: Bool = false, depthState: RenderDepthState? = nil, primitive: RenderPrimitiveState = .init(), hasTaskShader: Bool = false,
                 fragmentBufferBindings: [ComputePipeline.BufferBinding] = [], fragmentTextures: [SampledTextureRequirement] = [], fragmentStorageTextures: [StorageTextureRequirement] = [], fragmentSamplers: [SamplerRequirement] = [],
                 taskBufferBindings: [ComputePipeline.BufferBinding] = [], meshBufferBindings: [ComputePipeline.BufferBinding] = [],
                 taskTextures: [SampledTextureRequirement] = [], meshTextures: [SampledTextureRequirement] = [],
                 taskSamplers: [SamplerRequirement] = [], meshSamplers: [SamplerRequirement] = [], taskStorageTextures: [StorageTextureRequirement] = [], meshStorageTextures: [StorageTextureRequirement] = []) {
        self.hasTaskShader = hasTaskShader
        self.id = id; self.primitive = primitive; self.colorFormats = colorFormats; self.sampleCount = sampleCount; self.sampleMask = sampleMask; self.alphaToCoverageEnabled = alphaToCoverageEnabled; self.depthState = depthState
        self.fragmentBufferBindings = fragmentBufferBindings; self.fragmentTextures = fragmentTextures; self.fragmentStorageTextures = fragmentStorageTextures; self.fragmentSamplers = fragmentSamplers
        self.taskBufferBindings = taskBufferBindings; self.meshBufferBindings = meshBufferBindings
        self.taskTextures = taskTextures; self.meshTextures = meshTextures
        self.taskSamplers = taskSamplers; self.meshSamplers = meshSamplers
        self.taskStorageTextures = taskStorageTextures; self.meshStorageTextures = meshStorageTextures
    }

}
