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

    case uint8 = 16
    case uint8x2 = 17
    case uint8x4 = 18
    case sint8 = 19
    case sint8x2 = 20
    case sint8x4 = 21
    case unorm8 = 22
    case unorm8x2 = 23
    case unorm8x4 = 24
    case snorm8 = 25
    case snorm8x2 = 26
    case snorm8x4 = 27
    case uint16 = 28
    case uint16x2 = 29
    case uint16x4 = 30
    case sint16 = 31
    case sint16x2 = 32
    case sint16x4 = 33
    case unorm16 = 34
    case unorm16x2 = 35
    case unorm16x4 = 36
    case snorm16 = 37
    case snorm16x2 = 38
    case snorm16x4 = 39
    case unorm1010102 = 40
    case unorm8x4BGRA = 41
    case snorm1010102 = 42

    public var componentCount: UInt32 {
        switch self {
        case .float32, .uint32, .sint32, .float16,
             .uint8, .sint8, .unorm8, .snorm8, .uint16, .sint16, .unorm16, .snorm16: 1
        case .float32x2, .uint32x2, .sint32x2, .float16x2,
             .uint8x2, .sint8x2, .unorm8x2, .snorm8x2, .uint16x2, .sint16x2, .unorm16x2, .snorm16x2: 2
        case .float32x3, .uint32x3, .sint32x3: 3
        case .float32x4, .uint32x4, .sint32x4, .float16x4,
             .uint8x4, .sint8x4, .unorm8x4, .snorm8x4, .uint16x4, .sint16x4, .unorm16x4, .snorm16x4,
             .unorm1010102, .unorm8x4BGRA, .snorm1010102: 4
        }
    }

    public var byteSize: UInt32 {
        switch self {
        case .uint8, .uint8x2, .uint8x4, .sint8, .sint8x2, .sint8x4,
             .unorm8, .unorm8x2, .unorm8x4, .snorm8, .snorm8x2, .snorm8x4:
            componentCount
        case .float16, .float16x2, .float16x4,
             .uint16, .uint16x2, .uint16x4, .sint16, .sint16x2, .sint16x4,
             .unorm16, .unorm16x2, .unorm16x4, .snorm16, .snorm16x2, .snorm16x4:
            componentCount * 2
        case .unorm1010102, .unorm8x4BGRA, .snorm1010102:
            4
        default:
            componentCount * 4
        }
    }

    public var dataType: VertexDataType {
        switch self {
        case .uint8, .uint8x2, .uint8x4, .uint16, .uint16x2, .uint16x4,
             .uint32, .uint32x2, .uint32x3, .uint32x4: .uint
        case .sint8, .sint8x2, .sint8x4, .sint16, .sint16x2, .sint16x4,
             .sint32, .sint32x2, .sint32x3, .sint32x4: .sint
        default: .float
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

    package static func validate(_ layouts: [Self], against capabilities: DeviceCapabilities) throws -> [UInt32: VertexFormat] {
        let locations = try validate(layouts, maxBuffers: capabilities.maxVertexBuffers,
            maxAttributes: capabilities.maxVertexAttributes, maxStride: capabilities.maxVertexBufferArrayStride)
        if locations.values.contains(.snorm1010102) { try capabilities.require([.vertexFormatSnorm1010102]) }
        return locations
    }

    package static func validate(
        _ layouts: [Self], maxBuffers: UInt32, maxAttributes: UInt32, maxStride: UInt32
    ) throws -> [UInt32: VertexFormat] {
        guard UInt64(layouts.count) <= UInt64(maxBuffers) else {
            throw GPUError.validation("Too many vertex buffer slots")
        }
        var locations: [UInt32: VertexFormat] = [:]
        for layout in layouts {
            let limit = layout.stride == 0 ? maxStride : layout.stride
            let totalAttributes = UInt64(locations.count) + UInt64(layout.attributes.count)
            guard totalAttributes <= UInt64(maxAttributes),
                  layout.stride <= maxStride, layout.stride % 4 == 0,
                  layout.attributeEnd <= UInt64(limit) else {
                throw GPUError.validation("Invalid vertex buffer stride or attribute range")
            }
            for attribute in layout.attributes {
                let alignment = min(4, attribute.format.byteSize)
                guard attribute.location < maxAttributes, attribute.offset % alignment == 0,
                      locations.updateValue(attribute.format, forKey: attribute.location) == nil else {
                    throw GPUError.validation("Invalid, duplicate or unaligned vertex attribute")
                }
            }
        }
        return locations
    }

    package static func metalPullingSignature(_ layouts: [Self]) -> [UInt32] {
        layouts.flatMap { layout in
            [layout.stride, layout.stepMode.rawValue, UInt32(layout.attributes.count)] +
                layout.attributes.sorted { $0.location < $1.location }.flatMap {
                    [$0.location, $0.offset, $0.format.rawValue]
                }
        }
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
    /// Pipeline slot; nil uses the binding's position in the array.
    public let slot: UInt32?

    public init(buffer: ResourceID, offset: UInt64 = 0, slot: UInt32? = nil) {
        self.buffer = buffer
        self.offset = offset
        self.slot = slot
    }
}
