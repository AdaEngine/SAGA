import Testing
@testable import SAGATypes

@Test func triangleGeometryChecksVertexRangesAndOverflow() throws {
    let geometry = AccelerationStructureTriangleGeometry(vertexBuffer: .init(device: 1, serial: 1), vertexOffset: 4,
        vertexStride: 16, vertexCount: 3)
    #expect(try geometry.vertexRange(bufferSize: 48) == 4..<48)
    #expect(throws: GPUError.self) { try geometry.vertexRange(bufferSize: 47) }
    let overflow = AccelerationStructureTriangleGeometry(vertexBuffer: .init(device: 1, serial: 1),
        vertexOffset: UInt64.max - 3, vertexStride: 16, vertexCount: 3)
    #expect(throws: GPUError.self) { try overflow.vertexRange(bufferSize: UInt64.max) }
}

@Test func triangleGeometryRejectsInvalidLayout() {
    for geometry in [
        AccelerationStructureTriangleGeometry(vertexBuffer: .init(device: 1, serial: 1), vertexOffset: 2, vertexStride: 12, vertexCount: 3),
        AccelerationStructureTriangleGeometry(vertexBuffer: .init(device: 1, serial: 1), vertexStride: 8, vertexCount: 3),
        AccelerationStructureTriangleGeometry(vertexBuffer: .init(device: 1, serial: 1), vertexStride: 14, vertexCount: 3),
        AccelerationStructureTriangleGeometry(vertexBuffer: .init(device: 1, serial: 1), vertexStride: 12, vertexCount: 2)
    ] {
        #expect(throws: GPUError.self) { try geometry.vertexRange(bufferSize: 100) }
    }
}

@Test func indexedTriangleGeometryChecksRangesAndCounts() throws {
    let vertex = ResourceID(device: 1, serial: 1)
    let index = ResourceID(device: 1, serial: 2)
    let geometry = AccelerationStructureTriangleGeometry(vertexBuffer: vertex, vertexStride: 12, vertexCount: 4,
        indexBuffer: .init(buffer: index, format: .uint16, offset: 2), indexCount: 6)
    #expect(try geometry.vertexRange(bufferSize: 48) == 0..<48)
    #expect(try geometry.indexRange(bufferSize: 14) == 2..<14)
    #expect(geometry.triangleCount == 2)
    let command = BufferCommand.buildAccelerationStructure(destination: .init(device: 1, serial: 3), geometry: geometry)
    #expect(command.resources == [.init(device: 1, serial: 3), vertex, index])
    #expect(throws: GPUError.self) { try geometry.indexRange(bufferSize: 13) }
    let invalid = AccelerationStructureTriangleGeometry(vertexBuffer: vertex, vertexStride: 12, vertexCount: 4,
        indexBuffer: .init(buffer: index, format: .uint32, offset: 2), indexCount: 3)
    #expect(throws: GPUError.self) { try invalid.indexRange(bufferSize: 100) }
}
