import Testing
import SAGATypes

@Test func rayQueryIsAnExplicitDeviceFeature() throws {
    let generic = try DeviceCapabilities(maxBufferSize: 1024)
    #expect(!generic.features.contains(.rayQuery))
    let supported = try DeviceCapabilities(maxBufferSize: 1024, features: [.rayQuery])
    #expect(supported.features.contains(.rayQuery))
}

#if canImport(Metal)
import Metal
import SAGAMetal

@Test @MainActor func metalRayQueryFeatureMatchesNativeSupport() throws {
    guard let native = MTLCreateSystemDefaultDevice() else { return }
    let backend = try MetalBackend.makeDevice()
    #expect(backend.capabilities.features.contains(.rayQuery) == native.supportsRaytracing)
}
#endif
