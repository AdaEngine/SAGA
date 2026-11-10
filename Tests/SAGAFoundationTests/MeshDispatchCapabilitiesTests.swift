import SAGATypes
import Testing
#if canImport(Metal)
import Metal
import SAGAMetal
#endif

@Test func meshDispatchCapabilitiesHaveCompatibleDefaultsAndOverrides() throws {
    let unsupported = try DeviceCapabilities(maxBufferSize: 1024)
    #expect(unsupported.maxTaskWorkgroupTotalCount == 0)
    #expect(unsupported.maxTaskWorkgroupsPerDimension == 0)
    #expect(unsupported.maxMeshWorkgroupTotalCount == 0)
    #expect(unsupported.maxMeshWorkgroupsPerDimension == 0)
    let mock = try DeviceCapabilities(maxBufferSize: 1024, features: [.meshShaders])
    #expect(mock.maxTaskWorkgroupTotalCount == 1024)
    #expect(mock.maxTaskWorkgroupsPerDimension == 1024)
    #expect(mock.maxMeshWorkgroupTotalCount == 1024)
    #expect(mock.maxMeshWorkgroupsPerDimension == 1024)
    let custom = try DeviceCapabilities(maxBufferSize: 1024, features: [.meshShaders],
        maxTaskWorkgroupTotalCount: 30, maxTaskWorkgroupsPerDimension: 10,
        maxMeshWorkgroupTotalCount: 60, maxMeshWorkgroupsPerDimension: 20)
    #expect(custom.maxTaskWorkgroupTotalCount == 30)
    #expect(custom.maxTaskWorkgroupsPerDimension == 10)
    #expect(custom.maxMeshWorkgroupTotalCount == 60)
    #expect(custom.maxMeshWorkgroupsPerDimension == 20)
    #expect(throws: GPUError.self) {
        try DeviceCapabilities(maxBufferSize: 1024, features: [.meshShaders], maxTaskWorkgroupTotalCount: 0)
    }
}

#if canImport(Metal)
@Test func metalMeshDispatchCapabilitiesMatchGPUFamily() throws {
    guard let native = MTLCreateSystemDefaultDevice() else { return }
    let capabilities = try MetalBackend.makeDevice().capabilities
    guard capabilities.features.contains(.meshShaders) else {
        #expect(capabilities.maxTaskWorkgroupTotalCount == 0)
        #expect(capabilities.maxMeshWorkgroupTotalCount == 0)
        return
    }
    var metal4 = false
    if #available(macOS 26, iOS 26, tvOS 26, visionOS 26, *) {
        metal4 = native.supportsFamily(.metal4)
    }
    let task: UInt32 = metal4 || native.supportsFamily(.apple7) ? .max : 1024
    let mesh: UInt32
    if #available(macOS 26, iOS 26, tvOS 26, visionOS 26, *), native.supportsFamily(.apple10) {
        mesh = 1 << 22
    } else if native.supportsFamily(.apple9) {
        mesh = 1 << 20
    } else {
        mesh = 1024
    }
    #expect(capabilities.maxTaskWorkgroupTotalCount == task)
    #expect(capabilities.maxTaskWorkgroupsPerDimension == task)
    #expect(capabilities.maxMeshWorkgroupTotalCount == mesh)
    #expect(capabilities.maxMeshWorkgroupsPerDimension == mesh)
}
#endif
