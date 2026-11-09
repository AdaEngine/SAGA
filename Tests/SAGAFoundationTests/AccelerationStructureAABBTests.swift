import Testing
@testable import SAGATypes

@Test func aabbGeometryValidatesLayoutAndBounds() throws {
    let id = ResourceID(device: 1, serial: 1)
    let geometry = AccelerationStructureAABBGeometry(buffer: id, offset: 4, stride: 32, count: 2)
    #expect(try geometry.range(bufferSize: 60) == 4..<60)
    #expect(throws: GPUError.self) { try geometry.range(bufferSize: 59) }
    for invalid in [AccelerationStructureAABBGeometry(buffer: id, offset: 2, stride: 24, count: 1),
                    AccelerationStructureAABBGeometry(buffer: id, stride: 20, count: 1),
                    AccelerationStructureAABBGeometry(buffer: id, stride: 24, count: 0),
                    AccelerationStructureAABBGeometry(buffer: id, offset: UInt64.max - 3, stride: 24, count: 1)] {
        #expect(throws: GPUError.self) { try invalid.range(bufferSize: UInt64.max) }
    }
}
