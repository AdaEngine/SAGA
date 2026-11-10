import SAGATypes
import Testing
#if canImport(Metal)
import Metal
import SAGAMetal
#endif

@Test func meshStageCapabilitiesExposeDefaultsAndOverrides() throws {
    let unsupported = try DeviceCapabilities(maxBufferSize: 1024)
    #expect(unsupported.maxTaskInvocationsPerWorkgroup == 0)
    #expect(unsupported.maxMeshInvocationsPerDimension == 0)
    #expect(unsupported.maxTaskPayloadSize == 0)
    #expect(unsupported.maxMeshOutputVertices == 256)
    #expect(unsupported.maxMeshOutputPrimitives == 256)
    let mesh = try DeviceCapabilities(maxBufferSize: 1024, features: [.meshShaders])
    #expect(mesh.maxTaskInvocationsPerWorkgroup == 1024)
    #expect(mesh.maxTaskInvocationsPerDimension == 1024)
    #expect(mesh.maxMeshInvocationsPerWorkgroup == 1024)
    #expect(mesh.maxMeshInvocationsPerDimension == 1024)
    #expect(mesh.maxTaskPayloadSize == 16352)
    let custom = try DeviceCapabilities(maxBufferSize: 1024, features: [.meshShaders],
        maxTaskInvocationsPerWorkgroup: 32, maxTaskInvocationsPerDimension: 16,
        maxMeshInvocationsPerWorkgroup: 64, maxMeshInvocationsPerDimension: 8,
        maxTaskPayloadSize: 128, maxMeshOutputVertices: 10, maxMeshOutputPrimitives: 20)
    #expect(custom.maxTaskInvocationsPerWorkgroup == 32)
    #expect(custom.maxTaskInvocationsPerDimension == 16)
    #expect(custom.maxMeshInvocationsPerWorkgroup == 64)
    #expect(custom.maxMeshInvocationsPerDimension == 8)
    #expect(custom.maxTaskPayloadSize == 128)
    #expect(custom.maxMeshOutputVertices == 10)
    #expect(custom.maxMeshOutputPrimitives == 20)
    #expect(throws: GPUError.self) {
        try DeviceCapabilities(maxBufferSize: 1024, features: [.meshShaders], maxTaskInvocationsPerWorkgroup: 0)
    }
    #expect(throws: GPUError.self) {
        try DeviceCapabilities(maxBufferSize: 1024, maxTaskPayloadSize: 1)
    }
}

#if canImport(Metal)
@Test func metalMeshStageCapabilitiesMatchPinnedAdapter() throws {
    guard MTLCreateSystemDefaultDevice() != nil else { return }
    let caps = try MetalBackend.makeDevice().capabilities
    let supported = caps.features.contains(.meshShaders)
    #expect(caps.maxTaskInvocationsPerWorkgroup == (supported ? 1024 : 0))
    #expect(caps.maxTaskInvocationsPerDimension == (supported ? 1024 : 0))
    #expect(caps.maxMeshInvocationsPerWorkgroup == (supported ? 1024 : 0))
    #expect(caps.maxMeshInvocationsPerDimension == (supported ? 1024 : 0))
    #expect(caps.maxTaskPayloadSize == (supported ? 16352 : 0))
    #expect(caps.maxMeshOutputVertices == 256)
    #expect(caps.maxMeshOutputPrimitives == 256)
}
#endif
