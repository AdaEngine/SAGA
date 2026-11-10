/// Source and entry point for one selected vertex or fragment stage.
public struct RenderStageArtifact: Sendable, Equatable {
    package static let metalPointTopologyMacro = "SAGA_POINT_TOPOLOGY"
    package static let metalPipelineSampleMaskMacro = "SAGA_PIPELINE_SAMPLE_MASK"
    package static let metalMultiviewMaskMacro = "SAGA_MULTIVIEW_MASK"

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
        /// Whether the fragment input is emitted once per mesh primitive.
        public let perPrimitive: Bool

        package init(location: UInt32, componentCount: UInt8, scalar: IRScalar,
                     interpolation: IRInterpolation, sampling: IRSampling, perPrimitive: Bool = false) {
            self.location = location
            self.componentCount = componentCount
            self.scalar = scalar
            self.interpolation = interpolation
            self.sampling = sampling
            self.perPrimitive = perPrimitive
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
    public var requiresPerVertex: Bool { stage == .fragment && varyings.contains { $0.interpolation == .perVertex } }
    package var usesVertexWritableStorage: Bool {
        stage == .vertex && (storageBindings.contains { !$0.isReadOnly } || storageTextures.contains { $0.access.contains(.store) })
    }
    /// Vertex attributes ordered by shader location.
    public let vertexInputs: [VertexInput]
    package let metalVertexPullingLayout: [UInt32]?
    /// The MSL entry point loads its vertex streams directly.
    public var usesMetalVertexPulling: Bool { metalVertexPullingLayout != nil }
    /// Scalar kinds of the color outputs, keyed by fragment location.
    public let colorOutputKinds: [UInt32: IRScalarKind]
    /// Component counts of the color outputs, keyed by fragment location.
    public let colorOutputComponentCounts: [UInt32: UInt8]
    public var colorOutputKind: IRScalarKind? { colorOutputKinds[0] }
    public let writesDepth: Bool
    public let earlyDepthTest: IREarlyDepthTest?
    /// Metal fragment buffer containing the viewport's minimum and maximum depth.
    public static let metalViewportDepthBufferIndex = 0
    package static let metalVertexAttributeCapacity: UInt32 = 31
    package static let nativeColorAttachmentCapacity: UInt32 = 8
    /// Metal vertex buffer index for the UInt64 byte lengths of bound streams, after their offsets.
    public static let metalVertexPullingSizeBufferIndex: UInt32 = 8
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
    /// Whether the stage declares the native multiview index input.
    public let readsViewIndex: Bool
    /// Fragment builtins that consume a slot in the inter-stage input budget.
    package let fragmentInputBuiltinSlots: Int
    public let requiresShaderF16: Bool
    public let requiresShaderInt64: Bool
    public let requiresShaderInt16: Bool
    public let requiresDebugPrintf: Bool
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
        requiresBarycentrics: Bool = false, readsViewIndex: Bool = false, fragmentInputBuiltinSlots: Int = 0,
        requiresShaderF16: Bool = false,
        requiresUniformBufferStandardLayout: Bool = false, requiresSubgroups: Bool = false,
        dualSourceComponentCount: UInt8? = nil, earlyDepthTest: IREarlyDepthTest? = nil,
        metalTextureAtomicGuardBufferIndex: UInt32? = nil, requiresShaderInt64AtomicMinMax: Bool = false,
        requiresShaderInt64AtomicAllOps: Bool = false, requiresShaderInt64: Bool = false, requiresShaderInt16: Bool = false,
        requiresDebugPrintf: Bool = false,
        immediateLayout: ShaderImmediateLayout? = nil,
        requiresSubgroupUniformity: Bool = false, metalVertexPullingLayout: [UInt32]? = nil
    ) {
        self.artifact = artifact
        self.stage = stage
        self.sourceEntryPoint = sourceEntryPoint
        self.entryPoint = entryPoint
        self.hlslProfile = hlslProfile
        self.varyings = varyings
        self.vertexInputs = vertexInputs
        self.metalVertexPullingLayout = metalVertexPullingLayout
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
        self.readsViewIndex = readsViewIndex
        self.fragmentInputBuiltinSlots = fragmentInputBuiltinSlots
        self.requiresShaderF16 = requiresShaderF16
        self.requiresShaderInt64 = requiresShaderInt64
        self.requiresShaderInt16 = requiresShaderInt16
        self.requiresDebugPrintf = requiresDebugPrintf
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

extension RenderStageArtifact {
    /// Replaces the emitted code with a SPIR-V module whose entry point keeps its source name. Interface and
    /// resource reflection come from a text writer over the same IR; text-only fields are cleared.
    package func withSPIRV(_ words: [UInt32]) -> RenderStageArtifact {
        RenderStageArtifact(
            artifact: .spirv(words), stage: stage, sourceEntryPoint: sourceEntryPoint, entryPoint: sourceEntryPoint,
            varyings: varyings, vertexInputs: vertexInputs, hlslProfile: nil, uniformBindings: uniformBindings,
            storageBindings: storageBindings, metalRuntimeLengthBuffer: nil, sampledTextures: sampledTextures,
            storageTextures: storageTextures, externalTextures: externalTextures,
            externalTextureLayouts: externalTextureLayouts, samplerBindings: samplerBindings,
            glslTextureSamplerBindings: glslTextureSamplerBindings, colorOutputKinds: colorOutputKinds,
            colorOutputComponentCounts: colorOutputComponentCounts, writesDepth: writesDepth,
            writesSampleMask: writesSampleMask, invariantPosition: invariantPosition, clipDistanceCount: clipDistanceCount,
            readsPrimitiveIndex: readsPrimitiveIndex, requiresBarycentrics: requiresBarycentrics,
            readsViewIndex: readsViewIndex, fragmentInputBuiltinSlots: fragmentInputBuiltinSlots,
            requiresShaderF16: requiresShaderF16, requiresUniformBufferStandardLayout: requiresUniformBufferStandardLayout,
            requiresSubgroups: requiresSubgroups, dualSourceComponentCount: dualSourceComponentCount,
            earlyDepthTest: earlyDepthTest, metalTextureAtomicGuardBufferIndex: nil,
            requiresShaderInt64AtomicMinMax: requiresShaderInt64AtomicMinMax,
            requiresShaderInt64AtomicAllOps: requiresShaderInt64AtomicAllOps, requiresShaderInt64: requiresShaderInt64,
            requiresShaderInt16: requiresShaderInt16, requiresDebugPrintf: requiresDebugPrintf,
            immediateLayout: immediateLayout, requiresSubgroupUniformity: requiresSubgroupUniformity,
            metalVertexPullingLayout: nil)
    }
}
