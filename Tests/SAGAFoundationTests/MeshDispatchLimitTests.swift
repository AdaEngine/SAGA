import Testing
import SAGATypes

@Test func meshDrawUsesStageSpecificDispatchLimits() throws {
    let target = ResourceID(device: 1, serial: 2)
    let texture = TextureDescriptor(width: 4, height: 4, format: .rgba8Unorm, usage: [.renderAttachment])
    let mesh = MeshPipeline(id: .init(device: 1, serial: 1), colorFormat: .rgba8Unorm)
    let task = MeshPipeline(id: .init(device: 1, serial: 3), colorFormat: .rgba8Unorm, hasTaskShader: true)
    func check(_ pipeline: MeshPipeline, _ groups: DispatchWorkgroups) throws {
        try BufferCommand.drawMesh(pipeline: pipeline, target: target, colorLoad: .load, workgroups: groups)
            .validate(device: 1, descriptors: [:], textures: [target: texture],
                      maxTaskWorkgroupTotalCount: 24, maxTaskWorkgroupsPerDimension: 8,
                      maxMeshWorkgroupTotalCount: 6, maxMeshWorkgroupsPerDimension: 3)
    }
    try check(mesh, .init(x: 3, y: 2, z: 1))
    try check(task, .init(x: 4, y: 3, z: 2))
    #expect(throws: GPUError.self) { try check(mesh, .init(x: 4)) }
    #expect(throws: GPUError.self) { try check(mesh, .init(x: 3, y: 3)) }
    #expect(throws: GPUError.self) { try check(task, .init(x: 9)) }
    #expect(throws: GPUError.self) { try check(task, .init(x: 5, y: 5)) }
    try check(task, .init(x: 8, y: 8, z: 0))
    #expect(throws: GPUError.self) { try check(task, .init(x: 9, y: 0)) }
}

@Test func meshDrawTotalCountSaturatesWithoutOverflow() throws {
    let target = ResourceID(device: 1, serial: 2)
    let texture = TextureDescriptor(width: 4, height: 4, format: .rgba8Unorm, usage: [.renderAttachment])
    let pipeline = MeshPipeline(id: .init(device: 1, serial: 1), colorFormat: .rgba8Unorm)
    let command = BufferCommand.drawMesh(pipeline: pipeline, target: target, colorLoad: .load,
                                         workgroups: .init(x: .max, y: .max, z: .max))
    #expect(throws: GPUError.self) {
        try command.validate(device: 1, descriptors: [:], textures: [target: texture],
                             maxMeshWorkgroupTotalCount: .max - 1, maxMeshWorkgroupsPerDimension: .max)
    }
}
