/// Source and entry point for one selected vertex or fragment stage.
public struct RenderStageArtifact: Sendable, Equatable {
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
    /// Vertex outputs or fragment inputs, ordered by shader location.
    public let varyings: [Varying]
    /// Vertex attributes ordered by shader location.
    public let vertexInputs: [VertexInput]

    package init(
        artifact: ShaderArtifact, stage: ShaderStage, sourceEntryPoint: String, entryPoint: String,
        varyings: [Varying] = [], vertexInputs: [VertexInput] = []
    ) {
        self.artifact = artifact
        self.stage = stage
        self.sourceEntryPoint = sourceEntryPoint
        self.entryPoint = entryPoint
        self.varyings = varyings
        self.vertexInputs = vertexInputs
    }
}
