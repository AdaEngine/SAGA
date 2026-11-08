/// Capability names and bit positions mapped to pinned naga/src/valid/mod.rs.
/// Presence is configuration only: it does not mean SAGA has implemented that feature.
public struct ShaderCapabilities: OptionSet, Sendable, Equatable {
    public let rawValue: UInt64
    public init(rawValue: UInt64) { self.rawValue = rawValue }
    public static let immediates = Self(rawValue: 1 << 0)
    public static let float64 = Self(rawValue: 1 << 1)
    public static let primitiveIndex = Self(rawValue: 1 << 2)
    public static let textureAndSamplerBindingArray = Self(rawValue: 1 << 3)
    public static let bufferBindingArray = Self(rawValue: 1 << 4)
    public static let storageTextureBindingArray = Self(rawValue: 1 << 5)
    public static let storageBufferBindingArray = Self(rawValue: 1 << 6)
    public static let clipDistances = Self(rawValue: 1 << 7)
    public static let cullDistance = Self(rawValue: 1 << 8)
    public static let storageTexture16BitNormFormats = Self(rawValue: 1 << 9)
    public static let multiview = Self(rawValue: 1 << 10)
    public static let earlyDepthTest = Self(rawValue: 1 << 11)
    public static let multisampledShading = Self(rawValue: 1 << 12)
    public static let rayQuery = Self(rawValue: 1 << 13)
    public static let dualSourceBlending = Self(rawValue: 1 << 14)
    public static let cubeArrayTextures = Self(rawValue: 1 << 15)
    public static let shaderInt64 = Self(rawValue: 1 << 16)
    public static let subgroup = Self(rawValue: 1 << 17)
    public static let subgroupBarrier = Self(rawValue: 1 << 18)
    public static let subgroupVertexStage = Self(rawValue: 1 << 19)
    public static let shaderInt64AtomicMinMax = Self(rawValue: 1 << 20)
    public static let shaderInt64AtomicAllOps = Self(rawValue: 1 << 21)
    public static let shaderFloat32Atomic = Self(rawValue: 1 << 22)
    public static let textureAtomic = Self(rawValue: 1 << 23)
    public static let textureInt64Atomic = Self(rawValue: 1 << 24)
    public static let rayHitVertexPosition = Self(rawValue: 1 << 25)
    public static let shaderFloat16 = Self(rawValue: 1 << 26)
    public static let textureExternal = Self(rawValue: 1 << 27)
    public static let shaderFloat16InFloat32 = Self(rawValue: 1 << 28)
    public static let shaderBarycentrics = Self(rawValue: 1 << 29)
    public static let meshShader = Self(rawValue: 1 << 30)
    public static let meshShaderPointTopology = Self(rawValue: 1 << 31)
    public static let textureAndSamplerBindingArrayNonUniformIndexing = Self(rawValue: 1 << 32)
    public static let bufferBindingArrayNonUniformIndexing = Self(rawValue: 1 << 33)
    public static let storageTextureBindingArrayNonUniformIndexing = Self(rawValue: 1 << 34)
    public static let storageBufferBindingArrayNonUniformIndexing = Self(rawValue: 1 << 35)
    public static let cooperativeMatrix = Self(rawValue: 1 << 36)
    public static let perVertex = Self(rawValue: 1 << 37)
    public static let rayTracingPipeline = Self(rawValue: 1 << 38)
    public static let drawIndex = Self(rawValue: 1 << 39)
    public static let accelerationStructureBindingArray = Self(rawValue: 1 << 40)
    public static let memoryDecorationCoherent = Self(rawValue: 1 << 41)
    public static let memoryDecorationVolatile = Self(rawValue: 1 << 42)
    public static let shaderInt16 = Self(rawValue: 1 << 43)
    public static let linearInterpolation = Self(rawValue: 1 << 44)
    public static let debugPrintf = Self(rawValue: 1 << 45)
    public static let allKnown = Self(rawValue: (1 << 46) - 1)
    public static let referenceDefault: Self = [.multisampledShading, .cubeArrayTextures]

    package static let bindingArrays: Self = [
        .textureAndSamplerBindingArray, .bufferBindingArray,
        .storageTextureBindingArray, .storageBufferBindingArray,
        .textureAndSamplerBindingArrayNonUniformIndexing, .bufferBindingArrayNonUniformIndexing,
        .storageTextureBindingArrayNonUniformIndexing, .storageBufferBindingArrayNonUniformIndexing,
        .accelerationStructureBindingArray,
    ]
}

extension IRStage {
    public init(_ stage: ShaderStage) {
        switch stage { case .vertex: self = .vertex; case .fragment: self = .fragment; case .compute: self = .compute }
    }
}
