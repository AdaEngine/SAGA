public enum VertexDataType: Sendable, Equatable {
    case float, uint, sint
}

public enum VertexFormat: UInt32, Sendable, Equatable {
    case float32 = 1
    case float32x2 = 2
    case float32x3 = 3
    case float32x4 = 4
    case uint32 = 5
    case uint32x2 = 6
    case uint32x3 = 7
    case uint32x4 = 8
    case sint32 = 9
    case sint32x2 = 10
    case sint32x3 = 11
    case sint32x4 = 12
    case float16 = 13
    case float16x2 = 14
    case float16x4 = 15

    public var componentCount: UInt32 {
        switch self {
        case .float16: 1
        case .float16x2: 2
        case .float16x4: 4
        default: (rawValue - 1) % 4 + 1
        }
    }
    public var byteSize: UInt32 {
        switch self {
        case .float16, .float16x2, .float16x4: componentCount * 2
        default: componentCount * 4
        }
    }
    public var dataType: VertexDataType {
        switch self {
        case .float32, .float32x2, .float32x3, .float32x4, .float16, .float16x2, .float16x4: .float
        case .uint32, .uint32x2, .uint32x3, .uint32x4: .uint
        case .sint32, .sint32x2, .sint32x3, .sint32x4: .sint
        }
    }
}

public struct VertexAttribute: Sendable, Equatable {
    public let location: UInt32
    public let offset: UInt32
    public let format: VertexFormat

    public init(location: UInt32, offset: UInt32, format: VertexFormat) {
        self.location = location
        self.offset = offset
        self.format = format
    }
}

public enum VertexStepMode: UInt32, Sendable, Equatable {
    case vertex = 0
    case instance = 1
}

public struct VertexBufferLayout: Sendable, Equatable {
    public let stride: UInt32
    public let stepMode: VertexStepMode
    public let attributes: [VertexAttribute]

    public init(stride: UInt32, stepMode: VertexStepMode = .vertex, attributes: [VertexAttribute]) {
        self.stride = stride
        self.stepMode = stepMode
        self.attributes = attributes
    }

    package var attributeEnd: UInt64 {
        attributes.map { UInt64($0.offset) + UInt64($0.format.byteSize) }.max() ?? 0
    }

    package static func validate(_ layouts: [Self]) throws -> [UInt32: VertexFormat] {
        guard layouts.count <= 8 else { throw GPUError.validation("Too many vertex buffer slots") }
        var locations: [UInt32: VertexFormat] = [:]
        for layout in layouts {
            let limit = layout.stride == 0 ? 2048 : layout.stride
            guard !layout.attributes.isEmpty, layout.attributes.count <= 16 - locations.count,
                  layout.stride <= 2048, layout.stride % 4 == 0,
                  layout.attributeEnd <= UInt64(limit) else {
                throw GPUError.validation("Invalid vertex buffer stride or attribute range")
            }
            for attribute in layout.attributes {
                let alignment = min(4, attribute.format.byteSize)
                guard attribute.location < 16, attribute.offset % alignment == 0,
                      locations.updateValue(attribute.format, forKey: attribute.location) == nil else {
                    throw GPUError.validation("Invalid, duplicate or unaligned vertex attribute")
                }
            }
        }
        return locations
    }
}

public struct DrawInstances: Sendable, Equatable {
    public let count: UInt32
    public let first: UInt32

    public init(count: UInt32 = 1, first: UInt32 = 0) {
        self.count = count
        self.first = first
    }

    package var end: UInt64 { UInt64(first) + UInt64(count) }
}

public struct VertexBufferBinding: Sendable, Equatable {
    public let buffer: ResourceID
    public let offset: UInt64

    public init(buffer: ResourceID, offset: UInt64 = 0) {
        self.buffer = buffer
        self.offset = offset
    }
}
