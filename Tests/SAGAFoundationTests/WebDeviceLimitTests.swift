import Testing
import SAGATypes
import SAGAWebHost

@Test func webSampledTextureDeviceLimitIsRequiredAndPreserved() throws {
    var p = WebHostParameters()
    p.maxBufferSize = 1024
    p.maxTextureDimension2D = 1024
    p.maxComputeInvocationsPerWorkgroup = 256
    p.maxComputeWorkgroupSizeX = 256
    p.maxComputeWorkgroupSizeY = 256
    p.maxComputeWorkgroupSizeZ = 64
    p.maxComputeWorkgroupsPerDimension = 65535
    #expect(throws: GPUError.self) { try makeWebDeviceCapabilities(p) }
    p.maxSampledTexturesPerShaderStage = 16
    #expect(try makeWebDeviceCapabilities(p).maxSampledTexturesPerShaderStage == 16)
    p.maxSampledTexturesPerShaderStage = 32
    #expect(try makeWebDeviceCapabilities(p).maxSampledTexturesPerShaderStage == 32)
}
