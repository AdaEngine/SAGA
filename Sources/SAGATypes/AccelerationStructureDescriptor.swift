/// Capacity reserved for an acceleration structure.
public struct AccelerationStructureDescriptor: Sendable, Equatable {
    public enum Capacity: Sendable, Equatable {
        case triangles(UInt32)
        /// Axis-aligned boxes for procedural geometry; rays report a candidate hit that the shader confirms.
        case boxes(UInt32)
        case instances(UInt32)
    }
    public let capacity: Capacity
    /// Zero for a top-level structure.
    public var maxTriangleCount: UInt32 { if case .triangles(let count) = capacity { count } else { 0 } }
    public init(maxTriangleCount: UInt32) { capacity = .triangles(maxTriangleCount) }
    public init(maxInstanceCount: UInt32) { capacity = .instances(maxInstanceCount) }
    public init(maxBoxCount: UInt32) { capacity = .boxes(maxBoxCount) }
}

extension AccelerationStructureDescriptor.Capacity {
    /// True for structures that hold geometry and can be instanced by a top-level structure.
    package var isBottomLevel: Bool {
        switch self {
        case .triangles, .boxes: true
        case .instances: false
        }
    }
}
