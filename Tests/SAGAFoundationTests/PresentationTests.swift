import Testing
import SAGATypes
import SAGACore

@Test func presentationSharesQueueOrderAndPinsItsSource() async throws {
    let releaseFrame = Gate()
    let backend = RenderTestBackend(presentationGate: releaseFrame)
    let device = try await Device.make(backend: backend)
    let source = try await device.createTexture(.init(width: 8, height: 8, format: .rgba8Unorm, usage: .copySource))
    let wrongUsage = try await device.createTexture(.init(width: 8, height: 8, format: .rgba8Unorm,
                                                         usage: .renderAttachment))
    await #expect(throws: GPUError.self) { try await device.present(wrongUsage) }
    let previous = try await device.submitCommands([])
    await backend.started.wait()
    let frame = try await device.present(source)
    #expect(frame.sequence == previous.sequence + 1)
    #expect(await backend.presented.isEmpty)
    try await device.destroyTexture(source)
    #expect(await backend.textures.contains(source))
    await backend.release.release()
    await backend.presentStarted.wait()
    #expect(await backend.presented == [source])
    #expect(await backend.textures.contains(source))
    await releaseFrame.release()
    try await frame.wait()
    #expect(!(await backend.textures.contains(source)))
    await device.close()
}

@Test func presentationRequiresAnExplicitSurface() async throws {
    let device = try await Device.make(backend: RenderTestBackend())
    let source = try await device.createTexture(.init(width: 8, height: 8, format: .rgba8Unorm, usage: .copySource))
    await #expect(throws: GPUError.unsupported("Backend does not have a presentation surface")) {
        try await device.present(source)
    }
    await device.close()
}
