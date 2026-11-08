public enum QueryType: UInt32, Sendable, Equatable {
    case timestamp
    case occlusion
}

public struct QuerySetDescriptor: Sendable, Equatable {
    public let label: String
    public let type: QueryType
    public let count: UInt32

    public init(label: String = "", type: QueryType, count: UInt32) {
        self.label = label
        self.type = type
        self.count = count
    }

    package func validate(against capabilities: DeviceCapabilities) throws {
        switch type {
        case .timestamp: try capabilities.require([.timestampQuery])
        case .occlusion: break
        }
        guard (1...4096).contains(count) else {
            throw GPUError.validation("Query set count must be between 1 and 4096")
        }
    }
}

public struct PassOcclusionQuery: Sendable, Equatable {
    public let querySet: ResourceID
    public let queryIndex: UInt32

    public init(querySet: ResourceID, queryIndex: UInt32) {
        self.querySet = querySet
        self.queryIndex = queryIndex
    }
}

public struct PassTimestampWrites: Sendable, Equatable {
    public let querySet: ResourceID
    public let beginningOfPassWriteIndex: UInt32?
    public let endOfPassWriteIndex: UInt32?

    public init(querySet: ResourceID, beginningOfPassWriteIndex: UInt32? = nil, endOfPassWriteIndex: UInt32? = nil) {
        self.querySet = querySet
        self.beginningOfPassWriteIndex = beginningOfPassWriteIndex
        self.endOfPassWriteIndex = endOfPassWriteIndex
    }

    package var indices: [UInt32] { [beginningOfPassWriteIndex, endOfPassWriteIndex].compactMap { $0 } }
}
