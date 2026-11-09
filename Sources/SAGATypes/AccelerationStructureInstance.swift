/// A bottom-level structure placed in a top-level structure. The transform is a
/// row-major affine 3-by-4 matrix, with translation in its last column.
public struct AccelerationStructureInstance: Sendable, Equatable {
    public let accelerationStructure: ResourceID
    public let transform: [Float]
    public let mask: UInt8
    public let instanceID: UInt32

    public init(accelerationStructure: ResourceID,
                transform: [Float] = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
                mask: UInt8 = 0xff, instanceID: UInt32 = 0) {
        self.accelerationStructure = accelerationStructure
        self.transform = transform
        self.mask = mask
        self.instanceID = instanceID
    }
}
