import Testing
import SAGATypes

@Test func samplerBorderColorIsRequiredExactlyWithBorderAddressing() throws {
    try SamplerDescriptor(addressModeU: .clampToBorder, borderColor: .opaqueBlack).validate()
    try SamplerDescriptor().validate()
    #expect(throws: GPUError.self) { try SamplerDescriptor(addressModeV: .clampToBorder).validate() }
    #expect(throws: GPUError.self) { try SamplerDescriptor(borderColor: .opaqueWhite).validate() }
    #expect(SamplerDescriptor(addressModeW: .clampToBorder, borderColor: .transparentBlack).usesBorder)
    #expect(!SamplerDescriptor().usesBorder)
}
