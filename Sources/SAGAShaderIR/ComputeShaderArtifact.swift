/// A compiled compute shader and the layout required by its emitted entry point.
public struct ComputeShaderArtifact: Sendable, Equatable {
    package static let storageBindingCapacity: UInt32 = 8
    package static let sampledTextureBindingCapacity: UInt32 = 16
    package static let storageTextureBindingCapacity: UInt32 = 4
    package static let samplerBindingCapacity: UInt32 = 16
    package static let uniformBindingCapacity: UInt32 = 12
    package static let uniformBindingBase = storageBindingCapacity + 1
    package static let metalImmediateBufferIndex = uniformBindingBase + uniformBindingCapacity
    package static let portableBindGroupCount: UInt32 = 4
    package static let portableBindingIndexLimit: UInt32 = 1_000

    public struct BindingArrayLayout: Sendable, Equatable {
        public let count: UInt32
        /// First HLSL register in the logical binding's space. Metal uses the reflected flat binding index.
        public let hlslRegisterBase: UInt32?
        /// First physical binding in the logical group after WebGPU lowering.
        public let webBindingBase: UInt32?

        package init(count: UInt32, hlslRegisterBase: UInt32?, webBindingBase: UInt32? = nil) {
            self.count = count
            self.hlslRegisterBase = hlslRegisterBase
            self.webBindingBase = webBindingBase
        }
    }

    public struct ExternalTextureLayout: Sendable, Equatable {
        public static let parameterByteSize = 208
        public let binding: IRResourceBinding
        public let planeBindingIndices: [UInt32]
        public let parametersBindingIndex: UInt32

        package init(binding: IRResourceBinding, planeBindingIndices: [UInt32], parametersBindingIndex: UInt32) {
            self.binding = binding
            self.planeBindingIndices = planeBindingIndices
            self.parametersBindingIndex = parametersBindingIndex
        }
    }

    public struct StorageBinding: Sendable, Equatable {
        public let binding: IRResourceBinding
        /// Emitted buffer index for Metal or binding point for OpenGL GLSL.
        public let flatBindingIndex: UInt32?
        /// The HLSL binding uses a raw byte-address buffer.
        public let hlslUsesByteAddressBuffer: Bool
        public let minimumByteSize: UInt64
        public let isReadOnly: Bool
        public let runtimeArrayStride: UInt32?
        /// Byte offset of the runtime array from the start of the binding.
        public let runtimeArrayOffset: UInt32
        public let bindingArray: BindingArrayLayout?

        package init(
            binding: IRResourceBinding, flatBindingIndex: UInt32?, minimumByteSize: UInt64, isReadOnly: Bool,
            runtimeArrayStride: UInt32? = nil, runtimeArrayOffset: UInt32 = 0,
            hlslUsesByteAddressBuffer: Bool = false, bindingArray: BindingArrayLayout? = nil
        ) {
            self.binding = binding
            self.flatBindingIndex = flatBindingIndex
            self.hlslUsesByteAddressBuffer = hlslUsesByteAddressBuffer
            self.minimumByteSize = minimumByteSize
            self.isReadOnly = isReadOnly
            self.runtimeArrayStride = runtimeArrayStride
            self.runtimeArrayOffset = runtimeArrayOffset
            self.bindingArray = bindingArray
        }
    }

    public struct UniformBinding: Sendable, Equatable {
        public let binding: IRResourceBinding
        public let flatBindingIndex: UInt32?
        public let minimumByteSize: UInt64
        public let bindingArray: BindingArrayLayout?

        package init(binding: IRResourceBinding, flatBindingIndex: UInt32?, minimumByteSize: UInt64,
                     bindingArray: BindingArrayLayout? = nil) {
            self.binding = binding
            self.flatBindingIndex = flatBindingIndex
            self.minimumByteSize = minimumByteSize
            self.bindingArray = bindingArray
        }
    }

    public struct SampledTextureBinding: Sendable, Equatable {
        public let binding: IRResourceBinding
        public let isDepth: Bool
        public let isMultisampled: Bool
        public let isArrayed: Bool
        public let dimension: IRImageDimension
        public let sampleKind: IRScalarKind
        /// Sampler bindings used by sampling operations on this float texture.
        public let samplingSamplers: Set<IRResourceBinding>
        /// The inferred layout requires filterability unless every paired sampler is non-filtering.
        public var requiresFiltering: Bool { !samplingSamplers.isEmpty }
        /// Metal texture index or OpenGL GLSL texture unit.
        public let flatBindingIndex: UInt32?
        public let bindingArray: BindingArrayLayout?

        package init(binding: IRResourceBinding, flatBindingIndex: UInt32?, isDepth: Bool = false,
                     isMultisampled: Bool = false, isArrayed: Bool = false, sampleKind: IRScalarKind = .float,
                     dimension: IRImageDimension = .d2, samplingSamplers: Set<IRResourceBinding> = [],
                     bindingArray: BindingArrayLayout? = nil) {
            self.binding = binding
            self.isDepth = isDepth
            self.isMultisampled = isMultisampled
            self.isArrayed = isArrayed
            self.sampleKind = sampleKind
            self.samplingSamplers = samplingSamplers
            self.dimension = dimension
            self.flatBindingIndex = flatBindingIndex
            self.bindingArray = bindingArray
        }
    }

    public struct StorageTextureBinding: Sendable, Equatable {
        public let binding: IRResourceBinding
        public let format: IRStorageFormat
        public let access: IRStorageAccess
        public let isArrayed: Bool
        public let dimension: IRImageDimension
        /// Metal texture index or OpenGL image unit.
        public let flatBindingIndex: UInt32?
        public let bindingArray: BindingArrayLayout?

        package init(binding: IRResourceBinding, format: IRStorageFormat, access: IRStorageAccess = .store,
                     flatBindingIndex: UInt32?, isArrayed: Bool = false, dimension: IRImageDimension = .d2,
                     bindingArray: BindingArrayLayout? = nil) {
            self.binding = binding
            self.format = format
            self.access = access
            self.isArrayed = isArrayed
            self.dimension = dimension
            self.flatBindingIndex = flatBindingIndex
            self.bindingArray = bindingArray
        }
    }

    public struct SamplerBinding: Sendable, Equatable {
        public let binding: IRResourceBinding
        public let isComparison: Bool
        /// True when a depth texture uses this sampler without a comparison.
        public let requiresNonFiltering: Bool
        /// Metal sampler index.
        public let flatBindingIndex: UInt32?
        public let bindingArray: BindingArrayLayout?

        package init(binding: IRResourceBinding, flatBindingIndex: UInt32?, isComparison: Bool = false,
                     requiresNonFiltering: Bool = false, bindingArray: BindingArrayLayout? = nil) {
            self.binding = binding
            self.isComparison = isComparison
            self.requiresNonFiltering = requiresNonFiltering
            self.bindingArray = bindingArray
            self.flatBindingIndex = flatBindingIndex
        }
    }

    /// An OpenGL combined sampler uniform requires the texture and sampler at the same unit.
    public struct GLSLTextureSamplerBinding: Sendable, Equatable {
        public let texture: IRResourceBinding
        public let sampler: IRResourceBinding
        public let textureArrayElement: UInt32?
        public let samplerArrayElement: UInt32?
        public let externalTexturePlane: UInt32?
        public let isComparison: Bool
        public let flatBindingIndex: UInt32

        package init(texture: IRResourceBinding, sampler: IRResourceBinding, flatBindingIndex: UInt32,
                     isComparison: Bool = false, textureArrayElement: UInt32? = nil, samplerArrayElement: UInt32? = nil,
                     externalTexturePlane: UInt32? = nil) {
            self.texture = texture
            self.sampler = sampler
            self.textureArrayElement = textureArrayElement
            self.samplerArrayElement = samplerArrayElement
            self.externalTexturePlane = externalTexturePlane
            self.isComparison = isComparison
            self.flatBindingIndex = flatBindingIndex
        }
    }

    public let artifact: ShaderArtifact
    public let sourceEntryPoint: String
    public let entryPoint: String
    public let workgroupSize: IRWorkgroupSize
    /// DXC target profile for HLSL artifacts.
    public let hlslProfile: String?
    /// Total workgroup memory in bytes, with each variable rounded up to 16 bytes.
    public let workgroupStorageByteSize: UInt64
    public let storageBindings: [StorageBinding]
    public let uniformBindings: [UniformBinding]
    public let sampledTextures: [SampledTextureBinding]
    public let externalTextures: [IRResourceBinding]
    public let externalTextureLayouts: [ExternalTextureLayout]
    public let storageTextures: [StorageTextureBinding]
    public let usesTextureBarrier: Bool
    public let requiresShaderF16: Bool
    public let requiresShaderInt64: Bool
    public let requiresShaderInt64AtomicMinMax: Bool
    public let requiresShaderInt64AtomicAllOps: Bool
    public let requiresSubgroups: Bool
    public let requiresSubgroupUniformity: Bool
    /// The native compute pipeline must run with this subgroup size when present.
    public let requiredSubgroupSize: UInt32?
    /// Whether the stage reads subgroup IDs or the number of subgroups.
    public let requiresSubgroupID: Bool
    public let usesGlobalInvocationIndex: Bool
    public let usesWorkgroupIndex: Bool
    public var requiresLinearIndexing: Bool { usesGlobalInvocationIndex || usesWorkgroupIndex }
    /// The emitted WGSL requires the uniform_buffer_standard_layout language feature.
    public let requiresUniformBufferStandardLayout: Bool
    public let samplerBindings: [SamplerBinding]
    public let glslTextureSamplerBindings: [GLSLTextureSamplerBinding]
    /// MSL buffer index for u32 element counts, in runtime storage binding order.
    public let metalRuntimeLengthBuffer: UInt32?
    public let immediateLayout: ShaderImmediateLayout?
    /// HLSL constant-buffer register and space for the dispatch dimensions (x, y, z, padding).
    public let hlslWorkgroupCountBinding: IRResourceBinding?

    package init(
        artifact: ShaderArtifact, sourceEntryPoint: String, entryPoint: String,
        workgroupSize: IRWorkgroupSize, storageBindings: [StorageBinding], uniformBindings: [UniformBinding] = [],
        sampledTextures: [SampledTextureBinding] = [], storageTextures: [StorageTextureBinding] = [],
        externalTextures: [IRResourceBinding] = [],
        externalTextureLayouts: [ExternalTextureLayout] = [],
        samplerBindings: [SamplerBinding] = [],
        glslTextureSamplerBindings: [GLSLTextureSamplerBinding] = [], workgroupStorageByteSize: UInt64 = 0,
        metalRuntimeLengthBuffer: UInt32? = nil,
        hlslWorkgroupCountBinding: IRResourceBinding? = nil, usesTextureBarrier: Bool = false, requiresShaderF16: Bool = false,
        requiresUniformBufferStandardLayout: Bool = false, requiresSubgroups: Bool = false, requiresSubgroupID: Bool = false,
        hlslProfile: String? = nil, requiresShaderInt64AtomicMinMax: Bool = false,
        requiresShaderInt64AtomicAllOps: Bool = false, requiresShaderInt64: Bool = false,
        immediateLayout: ShaderImmediateLayout? = nil, usesGlobalInvocationIndex: Bool = false,
        usesWorkgroupIndex: Bool = false, requiredSubgroupSize: UInt32? = nil,
        requiresSubgroupUniformity: Bool = false
    ) {
        self.artifact = artifact
        self.sourceEntryPoint = sourceEntryPoint
        self.entryPoint = entryPoint
        self.workgroupSize = workgroupSize
        self.hlslProfile = hlslProfile
        self.workgroupStorageByteSize = workgroupStorageByteSize
        self.storageBindings = storageBindings
        self.uniformBindings = uniformBindings
        self.sampledTextures = sampledTextures
        self.externalTextures = externalTextures
        self.externalTextureLayouts = externalTextureLayouts
        self.storageTextures = storageTextures
        self.usesTextureBarrier = usesTextureBarrier
        self.requiresShaderF16 = requiresShaderF16
        self.requiresShaderInt64 = requiresShaderInt64
        self.requiresShaderInt64AtomicMinMax = requiresShaderInt64AtomicMinMax
        self.requiresShaderInt64AtomicAllOps = requiresShaderInt64AtomicAllOps
        self.requiresSubgroups = requiresSubgroups
        self.requiresSubgroupUniformity = requiresSubgroupUniformity
        self.requiredSubgroupSize = requiredSubgroupSize
        self.requiresSubgroupID = requiresSubgroupID
        self.usesGlobalInvocationIndex = usesGlobalInvocationIndex
        self.usesWorkgroupIndex = usesWorkgroupIndex
        self.requiresUniformBufferStandardLayout = requiresUniformBufferStandardLayout
        self.samplerBindings = samplerBindings
        self.glslTextureSamplerBindings = glslTextureSamplerBindings
        self.metalRuntimeLengthBuffer = metalRuntimeLengthBuffer
        self.immediateLayout = immediateLayout
        self.hlslWorkgroupCountBinding = hlslWorkgroupCountBinding
    }
}
