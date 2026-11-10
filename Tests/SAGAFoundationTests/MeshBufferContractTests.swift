import Testing
import SAGATypes
import SAGABackend
import SAGAShaderIR

@Test func meshDrawValidatesStageBuffersAndWritableAliases() throws {
    let uniform = ComputeBindingSlot(binding: 0), storage = ComputeBindingSlot(binding: 1)
    let pipeline = MeshPipeline(id: .init(device: 1, serial: 1), colorFormat: .rgba8Unorm,
        taskBufferBindings: [.init(slot: uniform, minimumByteSize: 16, isReadOnly: true, usage: .uniform)],
        meshBufferBindings: [.init(slot: storage, minimumByteSize: 32, isReadOnly: false)])
    let target = ResourceID(device: 1, serial: 2), buffer = ResourceID(device: 1, serial: 3)
    let separate = ResourceID(device: 1, serial: 4)
    let texture = TextureDescriptor(width: 4, height: 4, format: .rgba8Unorm, usage: [.renderAttachment])
    let descriptor = BufferDescriptor(size: 512, usage: [.uniform, .storage])
    func draw(_ task: [ComputeBindingSlot: StorageBufferBinding], _ mesh: [ComputeBindingSlot: StorageBufferBinding]) -> BufferCommand {
        .drawMesh(pipeline: pipeline, target: target, colorLoad: .load, workgroups: .init(x: 1), taskBuffers: task, meshBuffers: mesh)
    }
    let descriptors = [buffer: descriptor, separate: descriptor]
    let command = draw([uniform: .init(buffer: buffer, offset: 0, size: 16)], [storage: .init(buffer: separate, offset: 256, size: 32)])
    #expect(command.resources.contains(buffer))
    try command.validate(device: 1, descriptors: descriptors, textures: [target: texture])
    #expect(throws: GPUError.self) {
        try draw([:], [storage: .init(buffer: separate, offset: 256, size: 32)]).validate(device: 1, descriptors: descriptors, textures: [target: texture])
    }
    #expect(throws: GPUError.self) {
        try draw([uniform: .init(buffer: buffer, size: 16)], [storage: .init(buffer: buffer, size: 32)]).validate(device: 1, descriptors: descriptors, textures: [target: texture])
    }
    #expect(throws: GPUError.self) {
        try draw([uniform: .init(buffer: buffer, size: 8)], [storage: .init(buffer: separate, offset: 256, size: 32)]).validate(device: 1, descriptors: descriptors, textures: [target: texture])
    }
}

@Test func meshRuntimeArrayRequiresCompleteFirstElement() throws {
    let reflected = ComputeShaderArtifact.StorageBinding(binding: .init(group: 0, binding: 1), flatBindingIndex: 0,
        minimumByteSize: 4, isReadOnly: false, runtimeArrayStride: 16, runtimeArrayOffset: 32)
    let requirements = try meshBufferRequirements(uniforms: [], storage: [reflected])
    let pipeline = MeshPipeline(id: .init(device: 1, serial: 1), colorFormat: .rgba8Unorm, meshBufferBindings: requirements)
    let target = ResourceID(device: 1, serial: 2), buffer = ResourceID(device: 1, serial: 3)
    let texture = TextureDescriptor(width: 4, height: 4, format: .rgba8Unorm, usage: [.renderAttachment])
    let descriptor = BufferDescriptor(size: 256, usage: .storage)
    func command(_ size: UInt64) -> BufferCommand {
        .drawMesh(pipeline: pipeline, target: target, colorLoad: .load, workgroups: .init(x: 1),
                  meshBuffers: [.init(binding: 1): .init(buffer: buffer, size: size)])
    }
    #expect(throws: GPUError.self) { try command(44).validate(device: 1, descriptors: [buffer: descriptor], textures: [target: texture]) }
    try command(48).validate(device: 1, descriptors: [buffer: descriptor], textures: [target: texture])
}

@Test func meshTextureBindingsValidatePerStageAndPinResources() throws {
    let tex = ComputeBindingSlot(binding: 0), samp = ComputeBindingSlot(binding: 1)
    let requirement = SampledTextureRequirement(slot: tex, sampleType: .float)
    let samplerRequirement = SamplerRequirement(slot: samp, isComparison: false)
    let pipeline = MeshPipeline(id: .init(device: 1, serial: 1), colorFormat: .rgba8Unorm,
        taskTextures: [requirement], meshTextures: [requirement], taskSamplers: [samplerRequirement], meshSamplers: [samplerRequirement])
    let target = ResourceID(device: 1, serial: 2), image = ResourceID(device: 1, serial: 3)
    let sampler = ResourceID(device: 1, serial: 4), wrongDevice = ResourceID(device: 2, serial: 3)
    let textures = [target: TextureDescriptor(width: 1, height: 1, format: .rgba8Unorm, usage: [.renderAttachment, .textureBinding]),
                    image: TextureDescriptor(width: 1, height: 1, format: .rgba8Unorm, usage: .textureBinding)]
    func draw(_ task: ResourceID?, _ mesh: ResourceID?, _ taskSampler: ResourceID?, _ meshSampler: ResourceID?) -> BufferCommand {
        .drawMesh(pipeline: pipeline, target: target, colorLoad: .load, workgroups: .init(x: 1),
                  taskTextures: task.map { [tex: $0] } ?? [:], meshTextures: mesh.map { [tex: $0] } ?? [:],
                  taskSamplers: taskSampler.map { [samp: $0] } ?? [:], meshSamplers: meshSampler.map { [samp: $0] } ?? [:])
    }
    let command = draw(image, image, sampler, sampler)
    #expect(command.resources.contains(image)); #expect(command.resources.contains(sampler))
    try command.validate(device: 1, descriptors: [:], textures: textures, samplers: [sampler])
    #expect(throws: GPUError.self) { try draw(nil, image, sampler, sampler).validate(device: 1, descriptors: [:], textures: textures, samplers: [sampler]) }
    #expect(throws: GPUError.self) { try draw(image, wrongDevice, sampler, sampler).validate(device: 1, descriptors: [:], textures: textures, samplers: [sampler]) }
    #expect(throws: GPUError.self) { try draw(target, image, sampler, sampler).validate(device: 1, descriptors: [:], textures: textures, samplers: [sampler]) }
    #expect(throws: GPUError.self) { try draw(image, image, sampler, nil).validate(device: 1, descriptors: [:], textures: textures, samplers: [sampler]) }
    #expect(throws: GPUError.self) { try command.validate(device: 1, descriptors: [:], textures: textures, comparisonSamplers: [sampler]) }
}
