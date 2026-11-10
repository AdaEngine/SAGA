import Testing
import SAGATypes
import SAGACore
import SAGAMetal

@Test func bufferRangeCommandsValidateByteAlignmentBoundsAndUsage() throws {
    let source = ResourceID(device: 1, serial: 1)
    let destination = ResourceID(device: 1, serial: 2)
    let storage = ResourceID(device: 1, serial: 3)
    let descriptors: [ResourceID: BufferDescriptor] = [
        source: .init(size: 20, usage: [.copySource, .copyDestination]),
        destination: .init(size: 24, usage: .copyDestination),
        storage: .init(size: 24, usage: .storage),
    ]
    let valid: [BufferCommand] = [
        .copy(source: source, sourceOffset: 4, destination: destination, destinationOffset: 8),
        .copy(source: source, sourceOffset: 20, destination: destination, destinationOffset: 24, byteCount: 0),
        .clearBuffer(buffer: destination, offset: 24),
        .clearBuffer(buffer: destination, offset: 4, byteCount: 12),
        .writeBytes(buffer: destination, offset: 8, bytes: [99, 1, 2, 3, 4], dataOffset: 1, byteCount: 4),
        .writeBytes(buffer: destination, offset: 24, bytes: [], byteCount: 0),
        .write(buffer: destination, offset: 24, words: []),
    ]
    for command in valid {
        try command.validate(device: 1, descriptors: descriptors)
    }
    let invalid: [BufferCommand] = [
        .copy(source: source, sourceOffset: 2, destination: destination, byteCount: 4),
        .copy(source: source, destination: destination, destinationOffset: 2, byteCount: 4),
        .copy(source: source, sourceOffset: .max - 3, destination: destination, byteCount: 8),
        .copy(source: source, sourceOffset: 16, destination: destination, byteCount: 8),
        .copy(source: source, sourceOffset: 4, destination: destination, destinationOffset: 12),
        .copy(source: source, destination: source, destinationOffset: 8, byteCount: 4),
        .copy(source: source, destination: storage, byteCount: 0),
        .clearBuffer(buffer: destination, offset: 2, byteCount: 0),
        .clearBuffer(buffer: destination, offset: 20, byteCount: 8),
        .clearBuffer(buffer: destination, offset: .max),
        .clearBuffer(buffer: storage, byteCount: 0),
        .writeBytes(buffer: destination, offset: 0, bytes: [1, 2, 3, 4], dataOffset: 1),
        .writeBytes(buffer: destination, offset: 0, bytes: [1, 2, 3, 4], dataOffset: 1, byteCount: 4),
        .writeBytes(buffer: destination, offset: 0, bytes: [], dataOffset: .max, byteCount: 0),
        .writeBytes(buffer: destination, offset: 24, bytes: [1, 2, 3, 4]),
        .writeBytes(buffer: storage, offset: 0, bytes: []),
        .write(buffer: storage, offset: 0, words: []),
        .write(buffer: destination, offset: 2, words: []),
        .write(buffer: destination, offset: 28, words: []),
    ]
    for command in invalid {
        #expect(throws: GPUError.self) { try command.validate(device: 1, descriptors: descriptors) }
    }
}

#if canImport(Metal)
@Test func byteUploadRangeCopiesAndClearsPreserveNeighborsOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    let source = try await device.createBuffer(.init(size: 20, usage: [.copySource, .copyDestination]))
    let destination = try await device.createBuffer(.init(size: 24, usage: [.copySource, .copyDestination]))
    let readback = try await device.createBuffer(.init(size: 24, usage: [.mapRead, .copyDestination]))
    let payload = [UInt8.max] + Array(UInt8(1)...UInt8(16)) + [UInt8.max]
    let upload = try await device.writeBuffer(source, offset: 4, bytes: payload, dataOffset: 1, byteCount: 16)
    let operations = try await device.submitCommands([
        .writeBytes(buffer: destination, offset: 0, bytes: Array(repeating: 0xee, count: 24)),
        .copy(source: source, sourceOffset: 8, destination: destination, destinationOffset: 4, byteCount: 8),
        .clearBuffer(buffer: destination, offset: 8, byteCount: 4),
        .copy(source: source, sourceOffset: 16, destination: destination, destinationOffset: 16),
        .clearBuffer(buffer: destination, offset: 20),
        .copy(source: source, sourceOffset: 20, destination: destination, destinationOffset: 24, byteCount: 0),
        .clearBuffer(buffer: destination, offset: 24),
        .writeBytes(buffer: destination, offset: 24, bytes: []),
        .write(buffer: destination, offset: 24, words: []),
        .copy(source: destination, destination: readback),
    ])
    let mapping = try await device.mapBuffer(readback, mode: .read)
    let bytes = try await device.readMappedBytes(mapping)
    #expect(bytes == [
        0xee, 0xee, 0xee, 0xee, 5, 6, 7, 8, 0, 0, 0, 0,
        0xee, 0xee, 0xee, 0xee, 13, 14, 15, 16, 0, 0, 0, 0,
    ])
    #expect(upload.sequence < operations.sequence)
    try await device.unmapBuffer(readback)
    await device.close()
}
#endif
