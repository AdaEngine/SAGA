/// A payload layout whose scalar fields have the same byte offsets in the native stage ABI.
public struct MeshPayloadLayout: Sendable, Equatable {
    public struct Member: Sendable, Equatable {
        public let offset: UInt32
        public let scalar: IRScalar
        package init(offset: UInt32, scalar: IRScalar) { self.offset = offset; self.scalar = scalar }
    }
    public struct Field: Sendable, Equatable {
        public let offset: UInt32
        public let shape: Shape
        package init(offset: UInt32, shape: Shape) { self.offset = offset; self.shape = shape }
    }
    public indirect enum Shape: Sendable, Equatable {
        case scalar(IRScalar)
        case vector(scalar: IRScalar, count: UInt8)
        case array(element: Shape, count: UInt32, stride: UInt32)
        case structure([Field])
    }
    public let size: UInt32
    /// Top-level scalar fields retained for compatibility; use shape for the complete interface.
    public let members: [Member]
    public let shape: Shape
    package init(size: UInt32, members: [Member], shape: Shape? = nil) {
        self.size = size; self.members = members
        self.shape = shape ?? .structure(members.map { .init(offset: $0.offset, shape: .scalar($0.scalar)) })
    }
}

/// Native code and validated interface for an object/task shader.
public struct TaskStageArtifact: Sendable, Equatable {
    package static let metalMeshGridLimitMacro = "SAGA_MESH_GRID_LIMIT"
    public let artifact: ShaderArtifact
    public let sourceEntryPoint: String
    public let entryPoint: String
    public let workgroupSize: IRWorkgroupSize
    /// Estimated Metal threadgroup allocation in bytes, with each emitted variable
    /// rounded to 16 bytes. The object-data payload is accounted for separately;
    /// native compilation may add its own overhead.
    public let workgroupStorageByteSize: UInt64
    public let payload: MeshPayloadLayout?
    public let uniformBindings: [ComputeShaderArtifact.UniformBinding]
    public let storageBindings: [ComputeShaderArtifact.StorageBinding]
    public let sampledTextures: [ComputeShaderArtifact.SampledTextureBinding]
    public let samplerBindings: [ComputeShaderArtifact.SamplerBinding]
    public let storageTextures: [ComputeShaderArtifact.StorageTextureBinding]
    public let requiresSubgroups: Bool
    public let requiresShaderF16: Bool
    public let requiresShaderInt16: Bool
    public let requiresShaderInt64: Bool
    public let requiresShaderInt64AtomicMinMax: Bool
    public let requiresShaderInt64AtomicAllOps: Bool
    package init(artifact: ShaderArtifact, sourceEntryPoint: String, workgroupSize: IRWorkgroupSize, workgroupStorageByteSize: UInt64 = 0, payload: MeshPayloadLayout?, uniformBindings: [ComputeShaderArtifact.UniformBinding] = [], storageBindings: [ComputeShaderArtifact.StorageBinding] = [], sampledTextures: [ComputeShaderArtifact.SampledTextureBinding] = [], samplerBindings: [ComputeShaderArtifact.SamplerBinding] = [], storageTextures: [ComputeShaderArtifact.StorageTextureBinding] = [], requiresSubgroups: Bool = false, requiresShaderF16: Bool = false, requiresShaderInt16: Bool = false, requiresShaderInt64: Bool = false, requiresShaderInt64AtomicMinMax: Bool = false, requiresShaderInt64AtomicAllOps: Bool = false) {
        self.artifact = artifact; self.sourceEntryPoint = sourceEntryPoint; self.entryPoint = "saga_main"
        self.workgroupSize = workgroupSize; self.workgroupStorageByteSize = workgroupStorageByteSize; self.payload = payload
        self.uniformBindings = uniformBindings; self.storageBindings = storageBindings
        self.sampledTextures = sampledTextures; self.samplerBindings = samplerBindings; self.storageTextures = storageTextures; self.requiresSubgroups = requiresSubgroups
        self.requiresShaderF16 = requiresShaderF16; self.requiresShaderInt16 = requiresShaderInt16; self.requiresShaderInt64 = requiresShaderInt64
        self.requiresShaderInt64AtomicMinMax = requiresShaderInt64AtomicMinMax; self.requiresShaderInt64AtomicAllOps = requiresShaderInt64AtomicAllOps
    }
}

/// Native code and validated interface for a mesh shader.
public struct MeshStageArtifact: Sendable, Equatable {
    public struct Varying: Sendable, Equatable {
        public let location: UInt32
        public let scalar: IRScalar
        public let componentCount: UInt8
        public let interpolation: IRInterpolation?
        public let sampling: IRSampling?
        package init(location: UInt32, scalar: IRScalar, componentCount: UInt8, interpolation: IRInterpolation? = nil, sampling: IRSampling? = nil) {
            self.location = location; self.scalar = scalar; self.componentCount = componentCount
            self.interpolation = interpolation; self.sampling = sampling
        }
    }
    public let artifact: ShaderArtifact
    public let sourceEntryPoint: String
    public let entryPoint: String
    public let workgroupSize: IRWorkgroupSize
    /// Estimated Metal threadgroup allocation in bytes, with each emitted variable
    /// rounded to 16 bytes. Includes the materialized mesh output, not object data;
    /// native compilation may add its own overhead.
    public let workgroupStorageByteSize: UInt64
    public let payload: MeshPayloadLayout?
    public let topology: IRMeshOutputTopology
    public let maxVertices: UInt32
    public let maxPrimitives: UInt32
    public let vertexVaryings: [Varying]
    public let primitiveVaryings: [Varying]
    public let writesPrimitiveIndex: Bool
    public let uniformBindings: [ComputeShaderArtifact.UniformBinding]
    public let storageBindings: [ComputeShaderArtifact.StorageBinding]
    public let sampledTextures: [ComputeShaderArtifact.SampledTextureBinding]
    public let samplerBindings: [ComputeShaderArtifact.SamplerBinding]
    public let storageTextures: [ComputeShaderArtifact.StorageTextureBinding]
    public let requiresSubgroups: Bool
    public let requiresShaderF16: Bool
    public let requiresShaderInt16: Bool
    public let requiresShaderInt64: Bool
    public let requiresShaderInt64AtomicMinMax: Bool
    public let requiresShaderInt64AtomicAllOps: Bool
    package init(artifact: ShaderArtifact, sourceEntryPoint: String, workgroupSize: IRWorkgroupSize, workgroupStorageByteSize: UInt64 = 0, payload: MeshPayloadLayout?, topology: IRMeshOutputTopology, maxVertices: UInt32, maxPrimitives: UInt32, vertexVaryings: [Varying], primitiveVaryings: [Varying], uniformBindings: [ComputeShaderArtifact.UniformBinding] = [], storageBindings: [ComputeShaderArtifact.StorageBinding] = [], sampledTextures: [ComputeShaderArtifact.SampledTextureBinding] = [], samplerBindings: [ComputeShaderArtifact.SamplerBinding] = [], storageTextures: [ComputeShaderArtifact.StorageTextureBinding] = [], requiresSubgroups: Bool = false, writesPrimitiveIndex: Bool = false, requiresShaderF16: Bool = false, requiresShaderInt16: Bool = false, requiresShaderInt64: Bool = false, requiresShaderInt64AtomicMinMax: Bool = false, requiresShaderInt64AtomicAllOps: Bool = false) {
        self.artifact = artifact; self.sourceEntryPoint = sourceEntryPoint; self.entryPoint = "saga_main"
        self.workgroupSize = workgroupSize; self.workgroupStorageByteSize = workgroupStorageByteSize; self.payload = payload; self.topology = topology
        self.maxVertices = maxVertices; self.maxPrimitives = maxPrimitives
        self.vertexVaryings = vertexVaryings; self.primitiveVaryings = primitiveVaryings; self.writesPrimitiveIndex = writesPrimitiveIndex
        self.uniformBindings = uniformBindings; self.storageBindings = storageBindings
        self.sampledTextures = sampledTextures; self.samplerBindings = samplerBindings; self.storageTextures = storageTextures; self.requiresSubgroups = requiresSubgroups
        self.requiresShaderF16 = requiresShaderF16; self.requiresShaderInt16 = requiresShaderInt16; self.requiresShaderInt64 = requiresShaderInt64
        self.requiresShaderInt64AtomicMinMax = requiresShaderInt64AtomicMinMax; self.requiresShaderInt64AtomicAllOps = requiresShaderInt64AtomicAllOps
    }
}
