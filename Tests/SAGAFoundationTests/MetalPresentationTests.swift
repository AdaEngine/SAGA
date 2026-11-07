import Testing
import SAGACore
import SAGATypes
import SAGAMetal

#if canImport(Metal) && canImport(QuartzCore)
import QuartzCore

@MainActor
@Test func metalPresentationSurfaceHasOneBackendOwner() async throws {
    let surface = MetalPresentationSurface(layer: CAMetalLayer())
    let backend = try MetalBackend.makeDevice(surface: surface)
    let device = try await Device.make(backend: backend)
    #expect(device.supportsPresentation)
    #expect(throws: GPUError.self) { try MetalBackend.makeDevice(surface: surface) }
    await device.close()

    let headless = try await Device.make(backend: MetalBackend.makeDevice())
    #expect(!headless.supportsPresentation)
    await headless.close()
}
#endif
