/// Source and entry point for one selected vertex or fragment stage.
public struct RenderStageArtifact: Sendable, Equatable {
    package static let metalPointTopologyMacro = "SAGA_POINT_TOPOLOGY"
    package static let metalPipelineSampleMaskMacro = "SAGA_PIPELINE_SAMPLE_MASK"

    public struct VertexInput: Sendable, Equatable {
        public let location: UInt32
        public let componentCount: UInt8
        public let scalar: IRScalar

        package init(location: UInt32, componentCount: UInt8, scalar: IRScalar) {
            self.location = location
            self.componentCount = componentCount
            self.scalar = scalar
        }
    }

    public struct Varying: Sendable, Equatable {
        public let location: UInt32
        public let componentCount: UInt8
        public let scalar: IRScalar
        public let interpolation: IRInterpolation
        public let sampling: IRSampling

        package init(location: UInt32, componentCount: UInt8, scalar: IRScalar,
                     interpolation: IRInterpolation, sampling: IRSampling) {
            self.location = location
            self.componentCount = componentCount
            self.scalar = scalar
            self.interpolation = interpolation
            self.sampling = sampling
        }
    }

    public let artifact: ShaderArtifact
    public let stage: ShaderStage
    public let sourceEntryPoint: String
    public let entryPoint: String
    /// DXC target profile for HLSL artifacts.
    public let hlslProfile: String?
    /// Vertex outputs or fragment inputs, ordered by shader location.
    public let varyings: [Varying]
    /// Vertex attributes ordered by shader location.
    public let vertexInputs: [VertexInput]
    /// Scalar kinds of the color outputs, keyed by fragment location.
    public let colorOutputKinds: [UInt32: IRScalarKind]
    public var colorOutputKind: IRScalarKind? { colorOutputKinds[0] }
    public let writesDepth: Bool
    /// Metal fragment buffer containing the viewport's minimum and maximum depth.
    public static let metalViewportDepthBufferIndex = 0
    public let writesSampleMask: Bool
    public let uniformBindings: [ComputeShaderArtifact.UniformBinding]
    public let sampledTextures: [ComputeShaderArtifact.SampledTextureBinding]
    public let storageTextures: [ComputeShaderArtifact.StorageTextureBinding]
    public let samplerBindings: [ComputeShaderArtifact.SamplerBinding]
    public let glslTextureSamplerBindings: [ComputeShaderArtifact.GLSLTextureSamplerBinding]

    package init(
        artifact: ShaderArtifact, stage: ShaderStage, sourceEntryPoint: String, entryPoint: String,
        varyings: [Varying] = [], vertexInputs: [VertexInput] = [], hlslProfile: String? = nil,
        uniformBindings: [ComputeShaderArtifact.UniformBinding] = [],
        sampledTextures: [ComputeShaderArtifact.SampledTextureBinding] = [],
        storageTextures: [ComputeShaderArtifact.StorageTextureBinding] = [],
        samplerBindings: [ComputeShaderArtifact.SamplerBinding] = [],
        glslTextureSamplerBindings: [ComputeShaderArtifact.GLSLTextureSamplerBinding] = [],
        colorOutputKinds: [UInt32: IRScalarKind] = [:], writesDepth: Bool = false, writesSampleMask: Bool = false
    ) {
        self.artifact = artifact
        self.stage = stage
        self.sourceEntryPoint = sourceEntryPoint
        self.entryPoint = entryPoint
        self.hlslProfile = hlslProfile
        self.varyings = varyings
        self.vertexInputs = vertexInputs
        self.colorOutputKinds = colorOutputKinds
        self.writesDepth = writesDepth
        self.writesSampleMask = writesSampleMask
        self.uniformBindings = uniformBindings
        self.sampledTextures = sampledTextures
        self.storageTextures = storageTextures
        self.samplerBindings = samplerBindings
        self.glslTextureSamplerBindings = glslTextureSamplerBindings
    }
}
