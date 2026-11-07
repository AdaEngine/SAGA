public struct ComputeBindingSlot: Hashable, Sendable {
    public let group: UInt32
    public let binding: UInt32

    public init(group: UInt32 = 0, binding: UInt32) {
        self.group = group
        self.binding = binding
    }
}

public struct DispatchWorkgroups: Equatable, Sendable {
    public let x: UInt32
    public let y: UInt32
    public let z: UInt32

    public init(x: UInt32, y: UInt32 = 1, z: UInt32 = 1) {
        self.x = x
        self.y = y
        self.z = z
    }

    package var isEmpty: Bool { x == 0 || y == 0 || z == 0 }
}

public struct ComputePipeline: Equatable, Sendable {
    package struct StorageBinding: Equatable, Sendable {
        let slot: ComputeBindingSlot
        let minimumByteSize: UInt64
        let isReadOnly: Bool

        package init(slot: ComputeBindingSlot, minimumByteSize: UInt64, isReadOnly: Bool) {
            self.slot = slot
            self.minimumByteSize = minimumByteSize
            self.isReadOnly = isReadOnly
        }
    }

    public let id: ResourceID
    let storageBindings: [StorageBinding]

    package init(id: ResourceID, storageBindings: [StorageBinding]) {
        self.id = id
        self.storageBindings = storageBindings
    }
}
