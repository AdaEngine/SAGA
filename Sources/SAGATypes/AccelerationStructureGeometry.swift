/// Triangle positions used to build a bottom-level acceleration structure.
/// Positions are three consecutive 32-bit floating-point components. Indexed builds
/// replace any triangle containing an out-of-range index with a degenerate triangle.
/// The index buffer must have index usage and is read when GPU work executes.
public struct AccelerationStructureTriangleGeometry: Sendable, Equatable {
    public let vertexBuffer: ResourceID
    public let vertexOffset: UInt64
    public let vertexStride: UInt64
    public let vertexCount: UInt32
    public let indexBuffer: IndexBufferBinding?
    public let indexCount: UInt32
    public var triangleCount: UInt32 { (indexBuffer == nil ? vertexCount : indexCount) / 3 }

    public init(vertexBuffer: ResourceID, vertexOffset: UInt64 = 0, vertexStride: UInt64, vertexCount: UInt32,
                indexBuffer: IndexBufferBinding? = nil, indexCount: UInt32 = 0) {
        self.vertexBuffer = vertexBuffer
        self.vertexOffset = vertexOffset
        self.vertexStride = vertexStride
        self.vertexCount = vertexCount
        self.indexBuffer = indexBuffer
        self.indexCount = indexCount
    }

    /// Checks the position layout and returns the bytes accessed by the build.
    package func vertexRange(bufferSize: UInt64) throws -> Range<UInt64> {
        guard vertexOffset.isMultiple(of: 4), vertexStride >= 12, vertexStride.isMultiple(of: 4),
              vertexCount > 0, (indexBuffer != nil ? indexCount > 0 && indexCount.isMultiple(of: 3) : indexCount == 0 && vertexCount.isMultiple(of: 3)) else {
            throw GPUError.validation("Invalid acceleration structure triangle layout")
        }
        let (span, spanOverflow) = vertexStride.multipliedReportingOverflow(by: UInt64(vertexCount - 1))
        let (last, offsetOverflow) = vertexOffset.addingReportingOverflow(span)
        let (end, endOverflow) = last.addingReportingOverflow(12)
        guard !spanOverflow, !offsetOverflow, !endOverflow, end <= bufferSize else {
            throw GPUError.validation("Acceleration structure vertex range exceeds the buffer")
        }
        return vertexOffset..<end
    }
    package func indexRange(bufferSize: UInt64) throws -> Range<UInt64> {
        guard let indexBuffer, indexCount > 0, indexCount.isMultiple(of: 3),
              indexBuffer.offset.isMultiple(of: UInt64(indexBuffer.format.byteSize)),
              indexBuffer.offset <= bufferSize else {
            throw GPUError.validation("Invalid acceleration structure index layout")
        }
        let length = UInt64(indexCount) * UInt64(indexBuffer.format.byteSize)
        let available = indexBuffer.size ?? (bufferSize - indexBuffer.offset)
        guard available <= bufferSize - indexBuffer.offset, length <= available else {
            throw GPUError.validation("Acceleration structure index range exceeds the buffer")
        }
        return indexBuffer.offset..<(indexBuffer.offset + length)
    }
}
