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
    /// Component counts of the color outputs, keyed by fragment location.
    public let colorOutputComponentCounts: [UInt32: UInt8]
    public var colorOutputKind: IRScalarKind? { colorOutputKinds[0] }
    public let writesDepth: Bool
    public let earlyDepthTest: IREarlyDepthTest?
    /// Metal fragment buffer containing the viewport's minimum and maximum depth.
    public static let metalViewportDepthBufferIndex = 0
    package static let metalStorageBufferBase = ComputeShaderArtifact.uniformBindingBase + ComputeShaderArtifact.uniformBindingCapacity
    package static let metalStorageLengthBufferIndex = metalStorageBufferBase + ComputeShaderArtifact.storageBindingCapacity
    package static let metalTextureAtomicGuardSlot = metalStorageLengthBufferIndex + 1
    // Slot zero can be a vertex stream and is also reserved for fragment viewport depth.
    package static let metalVertexImmediateBufferIndex = metalTextureAtomicGuardSlot
    package static let metalFragmentImmediateBufferIndex: UInt32 = 1
    /// Metal fragment buffer containing the zero-valued driver workaround guard.
    public let metalTextureAtomicGuardBufferIndex: UInt32?
    public let writesSampleMask: Bool
    /// Whether the vertex stage requires invariant position calculations.
    public let invariantPosition: Bool
    /// Number of user-defined clipping planes written by the vertex stage.
    public let clipDistanceCount: Int
    /// Whether the fragment stage declares the primitive index input.
    public let readsPrimitiveIndex: Bool
    /// Whether the fragment stage receives native barycentric coordinates.
    public let requiresBarycentrics: Bool
    /// Fragment builtins that consume a slot in the inter-stage input budget.
    package let fragmentInputBuiltinSlots: Int
    public let requiresShaderF16: Bool
    public let requiresShaderInt64: Bool
    public let requiresShaderInt64AtomicMinMax: Bool
    public let requiresShaderInt64AtomicAllOps: Bool
    public let requiresSubgroups: Bool
    public let requiresSubgroupUniformity: Bool
    public let dualSourceComponentCount: UInt8?
    public var usesDualSourceBlending: Bool { dualSourceComponentCount != nil }
    /// The emitted WGSL requires the uniform_buffer_standard_layout language feature.
    public let requiresUniformBufferStandardLayout: Bool
    public let uniformBindings: [ComputeShaderArtifact.UniformBinding]
    public let storageBindings: [ComputeShaderArtifact.StorageBinding]
    /// MSL buffer index containing runtime array lengths in storage binding order.
    public let metalRuntimeLengthBuffer: UInt32?
    public let immediateLayout: ShaderImmediateLayout?
    public let sampledTextures: [ComputeShaderArtifact.SampledTextureBinding]
    public let externalTextures: [IRResourceBinding]
    public let externalTextureLayouts: [ComputeShaderArtifact.ExternalTextureLayout]
    public let storageTextures: [ComputeShaderArtifact.StorageTextureBinding]
    public let samplerBindings: [ComputeShaderArtifact.SamplerBinding]
    public let glslTextureSamplerBindings: [ComputeShaderArtifact.GLSLTextureSamplerBinding]

    package init(
        artifact: ShaderArtifact, stage: ShaderStage, sourceEntryPoint: String, entryPoint: String,
        varyings: [Varying] = [], vertexInputs: [VertexInput] = [], hlslProfile: String? = nil,
        uniformBindings: [ComputeShaderArtifact.UniformBinding] = [],
        storageBindings: [ComputeShaderArtifact.StorageBinding] = [], metalRuntimeLengthBuffer: UInt32? = nil,
        sampledTextures: [ComputeShaderArtifact.SampledTextureBinding] = [],
        storageTextures: [ComputeShaderArtifact.StorageTextureBinding] = [],
        externalTextures: [IRResourceBinding] = [],
        externalTextureLayouts: [ComputeShaderArtifact.ExternalTextureLayout] = [],
        samplerBindings: [ComputeShaderArtifact.SamplerBinding] = [],
        glslTextureSamplerBindings: [ComputeShaderArtifact.GLSLTextureSamplerBinding] = [],
        colorOutputKinds: [UInt32: IRScalarKind] = [:], colorOutputComponentCounts: [UInt32: UInt8] = [:],
        writesDepth: Bool = false, writesSampleMask: Bool = false,
        invariantPosition: Bool = false, clipDistanceCount: Int = 0, readsPrimitiveIndex: Bool = false,
        requiresBarycentrics: Bool = false, fragmentInputBuiltinSlots: Int = 0,
        requiresShaderF16: Bool = false,
        requiresUniformBufferStandardLayout: Bool = false, requiresSubgroups: Bool = false,
        dualSourceComponentCount: UInt8? = nil, earlyDepthTest: IREarlyDepthTest? = nil,
        metalTextureAtomicGuardBufferIndex: UInt32? = nil, requiresShaderInt64AtomicMinMax: Bool = false,
        requiresShaderInt64AtomicAllOps: Bool = false, requiresShaderInt64: Bool = false,
        immediateLayout: ShaderImmediateLayout? = nil,
        requiresSubgroupUniformity: Bool = false
    ) {
        self.artifact = artifact
        self.stage = stage
        self.sourceEntryPoint = sourceEntryPoint
        self.entryPoint = entryPoint
        self.hlslProfile = hlslProfile
        self.varyings = varyings
        self.vertexInputs = vertexInputs
        self.colorOutputKinds = colorOutputKinds
        self.colorOutputComponentCounts = colorOutputComponentCounts
        self.writesDepth = writesDepth
        self.earlyDepthTest = earlyDepthTest
        self.metalTextureAtomicGuardBufferIndex = metalTextureAtomicGuardBufferIndex
        self.writesSampleMask = writesSampleMask
        self.invariantPosition = invariantPosition
        self.clipDistanceCount = clipDistanceCount
        self.readsPrimitiveIndex = readsPrimitiveIndex
        self.requiresBarycentrics = requiresBarycentrics
        self.fragmentInputBuiltinSlots = fragmentInputBuiltinSlots
        self.requiresShaderF16 = requiresShaderF16
        self.requiresShaderInt64 = requiresShaderInt64
        self.requiresShaderInt64AtomicMinMax = requiresShaderInt64AtomicMinMax
        self.requiresShaderInt64AtomicAllOps = requiresShaderInt64AtomicAllOps
        self.requiresSubgroups = requiresSubgroups
        self.requiresSubgroupUniformity = requiresSubgroupUniformity
        self.dualSourceComponentCount = dualSourceComponentCount
        self.requiresUniformBufferStandardLayout = requiresUniformBufferStandardLayout
        self.uniformBindings = uniformBindings
        self.storageBindings = storageBindings
        self.metalRuntimeLengthBuffer = metalRuntimeLengthBuffer
        self.immediateLayout = immediateLayout
        self.sampledTextures = sampledTextures
        self.externalTextures = externalTextures
        self.externalTextureLayouts = externalTextureLayouts
        self.storageTextures = storageTextures
        self.samplerBindings = samplerBindings
        self.glslTextureSamplerBindings = glslTextureSamplerBindings
    }
}
