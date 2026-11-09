import SAGACore
import SAGAMetal
import SAGATypes
import Testing

@Test func validatesAdvertisedCooperativeMatrixProperties() throws {
    let configuration = CooperativeMatrixProperties(
        mSize: 8, nSize: 16, kSize: 8, inputType: .float16, accumulatorType: .float32)
    let capabilities = try DeviceCapabilities(
        maxBufferSize: 4096, features: [.cooperativeMatrix], cooperativeMatrixProperties: [configuration])
    #expect(capabilities.cooperativeMatrixProperties == [configuration])
    #expect(throws: GPUError.self) {
        try DeviceCapabilities(maxBufferSize: 4096, cooperativeMatrixProperties: [configuration])
    }
    #expect(throws: GPUError.self) {
        try DeviceCapabilities(maxBufferSize: 4096, features: [.cooperativeMatrix],
                               cooperativeMatrixProperties: [configuration, configuration])
    }
    #expect(throws: GPUError.self) {
        try DeviceCapabilities(maxBufferSize: 4096, features: [.cooperativeMatrix],
            cooperativeMatrixProperties: [.init(mSize: 0, nSize: 8, kSize: 8,
                                                 inputType: .float32, accumulatorType: .float32)])
    }
}

#if canImport(Metal)
@Test func metalAdvertisesCooperativeMatrixConfigurations() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    if device.capabilities.features.contains(.cooperativeMatrix) {
        let properties = device.capabilities.cooperativeMatrixProperties
        #expect(properties.count == 24)
        #expect(properties.contains(.init(mSize: 16, nSize: 8, kSize: 16,
                                          inputType: .float16, accumulatorType: .float32)))
        #expect(!properties.contains(.init(mSize: 8, nSize: 8, kSize: 8,
                                           inputType: .float32, accumulatorType: .float16)))
    } else {
        #expect(device.capabilities.cooperativeMatrixProperties.isEmpty)
    }
    await device.close()
}
#endif
