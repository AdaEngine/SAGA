/// A compiled compute shader and the layout required by its emitted entry point.
public struct ComputeShaderArtifact: Sendable, Equatable {
    public struct StorageBinding: Sendable, Equatable {
        public let binding: IRResourceBinding
        public let minimumByteSize: UInt64
        public let isReadOnly: Bool

        package init(binding: IRResourceBinding, minimumByteSize: UInt64, isReadOnly: Bool) {
            self.binding = binding
            self.minimumByteSize = minimumByteSize
            self.isReadOnly = isReadOnly
        }
    }

    public let artifact: ShaderArtifact
    public let sourceEntryPoint: String
    public let entryPoint: String
    public let workgroupSize: IRWorkgroupSize
    public let storageBindings: [StorageBinding]

    package init(
        artifact: ShaderArtifact, sourceEntryPoint: String, entryPoint: String,
        workgroupSize: IRWorkgroupSize, storageBindings: [StorageBinding]
    ) {
        self.artifact = artifact
        self.sourceEntryPoint = sourceEntryPoint
        self.entryPoint = entryPoint
        self.workgroupSize = workgroupSize
        self.storageBindings = storageBindings
    }
}
