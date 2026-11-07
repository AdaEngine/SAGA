/// A compiled compute shader and the layout required by its emitted entry point.
public struct ComputeShaderArtifact: Sendable, Equatable {
    package static let storageBindingCapacity: UInt32 = 8
    package static let sampledTextureBindingCapacity: UInt32 = 16
    package static let storageTextureBindingCapacity: UInt32 = 4
    package static let samplerBindingCapacity: UInt32 = 16
    package static let uniformBindingCapacity: UInt32 = 12
    package static let uniformBindingBase = storageBindingCapacity + 1
    package static let portableBindGroupCount: UInt32 = 4

    public struct StorageBinding: Sendable, Equatable {
        public let binding: IRResourceBinding
        /// Emitted buffer index for Metal or binding point for OpenGL GLSL.
        public let flatBindingIndex: UInt32?
        /// The HLSL binding uses a raw UAV with byte offsets.
        public let hlslUsesByteAddressBuffer: Bool
        public let minimumByteSize: UInt64
        public let isReadOnly: Bool
        public let runtimeArrayStride: UInt32?
        /// Byte offset of the runtime array from the start of the binding.
        public let runtimeArrayOffset: UInt32

        package init(
            binding: IRResourceBinding, flatBindingIndex: UInt32?, minimumByteSize: UInt64, isReadOnly: Bool,
            runtimeArrayStride: UInt32? = nil, runtimeArrayOffset: UInt32 = 0,
            hlslUsesByteAddressBuffer: Bool = false
        ) {
            self.binding = binding
            self.flatBindingIndex = flatBindingIndex
            self.hlslUsesByteAddressBuffer = hlslUsesByteAddressBuffer
            self.minimumByteSize = minimumByteSize
            self.isReadOnly = isReadOnly
            self.runtimeArrayStride = runtimeArrayStride
            self.runtimeArrayOffset = runtimeArrayOffset
        }
    }

    public struct UniformBinding: Sendable, Equatable {
        public let binding: IRResourceBinding
        public let flatBindingIndex: UInt32?
        public let minimumByteSize: UInt64

        package init(binding: IRResourceBinding, flatBindingIndex: UInt32?, minimumByteSize: UInt64) {
            self.binding = binding
            self.flatBindingIndex = flatBindingIndex
            self.minimumByteSize = minimumByteSize
        }
    }

    public struct SampledTextureBinding: Sendable, Equatable {
        public let binding: IRResourceBinding
        public let isDepth: Bool
        public let isMultisampled: Bool
        public let isArrayed: Bool
        public let dimension: IRImageDimension
        public let sampleKind: IRScalarKind
        /// Metal texture index or OpenGL GLSL texture unit.
        public let flatBindingIndex: UInt32?

        package init(binding: IRResourceBinding, flatBindingIndex: UInt32?, isDepth: Bool = false,
                     isMultisampled: Bool = false, isArrayed: Bool = false, sampleKind: IRScalarKind = .float,
                     dimension: IRImageDimension = .d2) {
            self.binding = binding
            self.isDepth = isDepth
            self.isMultisampled = isMultisampled
            self.isArrayed = isArrayed
            self.sampleKind = sampleKind
            self.dimension = dimension
            self.flatBindingIndex = flatBindingIndex
        }
    }

    public struct StorageTextureBinding: Sendable, Equatable {
        public let binding: IRResourceBinding
        public let format: IRStorageFormat
        public let access: IRStorageAccess
        public let isArrayed: Bool
        /// Metal texture index or OpenGL image unit.
        public let flatBindingIndex: UInt32?

        package init(binding: IRResourceBinding, format: IRStorageFormat, access: IRStorageAccess = .store,
                     flatBindingIndex: UInt32?, isArrayed: Bool = false) {
            self.binding = binding
            self.format = format
            self.access = access
            self.isArrayed = isArrayed
            self.flatBindingIndex = flatBindingIndex
        }
    }

    public struct SamplerBinding: Sendable, Equatable {
        public let binding: IRResourceBinding
        public let isComparison: Bool
        /// True when a depth texture uses this sampler without a comparison.
        public let requiresNonFiltering: Bool
        /// Metal sampler index.
        public let flatBindingIndex: UInt32?

        package init(binding: IRResourceBinding, flatBindingIndex: UInt32?, isComparison: Bool = false,
                     requiresNonFiltering: Bool = false) {
            self.binding = binding
            self.isComparison = isComparison
            self.requiresNonFiltering = requiresNonFiltering
            self.flatBindingIndex = flatBindingIndex
        }
    }

    /// An OpenGL combined sampler uniform requires the texture and sampler at the same unit.
    public struct GLSLTextureSamplerBinding: Sendable, Equatable {
        public let texture: IRResourceBinding
        public let sampler: IRResourceBinding
        public let isComparison: Bool
        public let flatBindingIndex: UInt32

        package init(texture: IRResourceBinding, sampler: IRResourceBinding, flatBindingIndex: UInt32,
                     isComparison: Bool = false) {
            self.texture = texture
            self.sampler = sampler
            self.isComparison = isComparison
            self.flatBindingIndex = flatBindingIndex
        }
    }

    public let artifact: ShaderArtifact
    public let sourceEntryPoint: String
    public let entryPoint: String
    public let workgroupSize: IRWorkgroupSize
    /// Total workgroup memory in bytes, with each variable rounded up to 16 bytes.
    public let workgroupStorageByteSize: UInt64
    public let storageBindings: [StorageBinding]
    public let uniformBindings: [UniformBinding]
    public let sampledTextures: [SampledTextureBinding]
    public let storageTextures: [StorageTextureBinding]
    public let usesTextureBarrier: Bool
    public let samplerBindings: [SamplerBinding]
    public let glslTextureSamplerBindings: [GLSLTextureSamplerBinding]
    /// MSL buffer index for u32 element counts, in runtime storage binding order.
    public let metalRuntimeLengthBuffer: UInt32?
    /// HLSL constant-buffer register and space for the dispatch dimensions (x, y, z, padding).
    public let hlslWorkgroupCountBinding: IRResourceBinding?

    package init(
        artifact: ShaderArtifact, sourceEntryPoint: String, entryPoint: String,
        workgroupSize: IRWorkgroupSize, storageBindings: [StorageBinding], uniformBindings: [UniformBinding] = [],
        sampledTextures: [SampledTextureBinding] = [], storageTextures: [StorageTextureBinding] = [],
        samplerBindings: [SamplerBinding] = [],
        glslTextureSamplerBindings: [GLSLTextureSamplerBinding] = [], workgroupStorageByteSize: UInt64 = 0,
        metalRuntimeLengthBuffer: UInt32? = nil,
        hlslWorkgroupCountBinding: IRResourceBinding? = nil, usesTextureBarrier: Bool = false
    ) {
        self.artifact = artifact
        self.sourceEntryPoint = sourceEntryPoint
        self.entryPoint = entryPoint
        self.workgroupSize = workgroupSize
        self.workgroupStorageByteSize = workgroupStorageByteSize
        self.storageBindings = storageBindings
        self.uniformBindings = uniformBindings
        self.sampledTextures = sampledTextures
        self.storageTextures = storageTextures
        self.usesTextureBarrier = usesTextureBarrier
        self.samplerBindings = samplerBindings
        self.glslTextureSamplerBindings = glslTextureSamplerBindings
        self.metalRuntimeLengthBuffer = metalRuntimeLengthBuffer
        self.hlslWorkgroupCountBinding = hlslWorkgroupCountBinding
    }
}
