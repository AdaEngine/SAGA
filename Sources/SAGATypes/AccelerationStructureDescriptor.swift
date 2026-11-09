/// Capacity reserved for an acceleration structure.
public struct AccelerationStructureDescriptor: Sendable, Equatable {
    public enum Capacity: Sendable, Equatable {
        case triangles(UInt32)
        case instances(UInt32)
    }
    public let capacity: Capacity
    /// Zero for a top-level structure.
    public var maxTriangleCount: UInt32 { if case .triangles(let count) = capacity { count } else { 0 } }
    public init(maxTriangleCount: UInt32) { capacity = .triangles(maxTriangleCount) }
    public init(maxInstanceCount: UInt32) { capacity = .instances(maxInstanceCount) }
}
