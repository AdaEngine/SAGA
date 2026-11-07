import Testing
import SAGATypes

@Test func rejectsUnknownAndIncompatibleBufferUsage() throws {
    let caps = try DeviceCapabilities(maxBufferSize: 1024)
    #expect(throws: GPUError.self) { try BufferDescriptor(size: 4, usage: []).validate(against: caps) }
    #expect(throws: GPUError.self) { try BufferDescriptor(size: 4, usage: .init(rawValue: 1 << 31)).validate(against: caps) }
    #expect(throws: GPUError.self) { try BufferDescriptor(size: 4, usage: [.mapRead, .storage]).validate(against: caps) }
    #expect(throws: GPUError.self) { try BufferDescriptor(size: 4, usage: [.mapWrite, .copyDestination]).validate(against: caps) }
}

@Test func validatesLimitsAndMappingAlignmentWithoutOverflow() throws {
    #expect(throws: GPUError.self) { try DeviceCapabilities(maxBufferSize: 0) }
    let caps = try DeviceCapabilities(maxBufferSize: 1024)
    #expect(throws: GPUError.self) { try BufferDescriptor(size: UInt64.max, usage: .storage).validate(against: caps) }
    #expect(throws: GPUError.self) { try BufferDescriptor(size: 3, usage: .storage, mappedAtCreation: true).validate(against: caps) }
    try BufferDescriptor(size: 0, usage: .copySource).validate(against: caps)
    try BufferDescriptor(size: 1024, usage: [.mapRead, .copyDestination]).validate(against: caps)
    try BufferDescriptor(size: 4, usage: [.mapWrite, .copySource]).validate(against: caps)
}

@Test func featuresMustBeAdvertised() throws {
    let caps = try DeviceCapabilities(maxBufferSize: 1024, features: [.shaderF16])
    try caps.require([.shaderF16])
    #expect(throws: GPUError.self) { try caps.require([.timestampQuery]) }
}
