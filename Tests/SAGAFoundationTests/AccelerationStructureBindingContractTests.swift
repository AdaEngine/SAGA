import Testing
import SAGABackend
import SAGAShaderIR
import SAGATypes

@Test func computePipelineReflectsAccelerationStructureBinding() throws {
    let shader = ComputeShaderArtifact(
        artifact: .text(format: .msl, source: "kernel void main0() {}"), sourceEntryPoint: "main", entryPoint: "main0",
        workgroupSize: .init(x: 1, y: 1, z: 1), storageBindings: [],
        accelerationStructures: [.init(binding: .init(group: 0, binding: 1), flatBindingIndex: 0)]
    )
    let pipeline = try ComputePipeline(id: .init(device: 1, serial: 1), shader: shader)
    #expect(pipeline.accelerationStructureBindings == [.init(binding: 1)])
}
