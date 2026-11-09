/// Axis-aligned bounding boxes in a buffer, each encoded as six consecutive
/// 32-bit floats: minimum xyz followed by maximum xyz.
public struct AccelerationStructureAABBGeometry: Sendable, Equatable {
    public let buffer: ResourceID
    public let offset: UInt64
    public let stride: UInt64
    public let count: UInt32

    public init(buffer: ResourceID, offset: UInt64 = 0, stride: UInt64 = 24, count: UInt32) {
        self.buffer = buffer
        self.offset = offset
        self.stride = stride
        self.count = count
    }

    package func range(bufferSize: UInt64) throws -> Range<UInt64> {
        guard offset.isMultiple(of: 4), stride >= 24, stride.isMultiple(of: 4), count > 0 else {
            throw GPUError.validation("Invalid acceleration structure AABB layout")
        }
        let (span, spanOverflow) = stride.multipliedReportingOverflow(by: UInt64(count - 1))
        let (last, offsetOverflow) = offset.addingReportingOverflow(span)
        let (end, endOverflow) = last.addingReportingOverflow(24)
        guard !spanOverflow, !offsetOverflow, !endOverflow, end <= bufferSize else {
            throw GPUError.validation("Acceleration structure AABB range exceeds the buffer")
        }
        return offset..<end
    }
}
