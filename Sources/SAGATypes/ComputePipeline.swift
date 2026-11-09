public struct ComputeBindingSlot: Hashable, Sendable {
    public let group: UInt32
    public let binding: UInt32
    /// Zero-based element of a binding array. Scalar bindings use nil.
    public let arrayElement: UInt32?

    public init(group: UInt32 = 0, binding: UInt32) {
        self.init(group: group, binding: binding, arrayElement: nil)
    }

    public init(group: UInt32 = 0, binding: UInt32, arrayElement: UInt32?) {
        self.group = group
        self.binding = binding
        self.arrayElement = arrayElement
    }

    package var baseSlot: Self { .init(group: group, binding: binding) }
}

public struct DispatchWorkgroups: Equatable, Sendable {
    package static let defaultMaximumDimension: UInt32 = 65_535

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

public struct StorageBufferBinding: Equatable, Sendable {
    public let buffer: ResourceID
    /// Byte offset, aligned to 256 bytes in the supported compute profile.
    public let offset: UInt64
    /// Byte count, or the rest of the buffer when omitted.
    public let size: UInt64?

    public init(buffer: ResourceID, offset: UInt64 = 0, size: UInt64? = nil) {
        self.buffer = buffer
        self.offset = offset
        self.size = size
    }

    package func range(in descriptor: BufferDescriptor, usage: BufferUsage = .storage) throws -> Range<UInt64> {
        guard descriptor.usage.contains(usage), offset % 256 == 0, offset <= descriptor.size else {
            throw GPUError.validation("Invalid compute buffer usage or offset")
        }
        let remaining = descriptor.size - offset
        let length = size ?? remaining
        guard length > 0, length <= remaining,
              usage == .uniform ? length <= 65_536 : length % 4 == 0 else {
            throw GPUError.validation("Invalid compute buffer binding size")
        }
        return offset..<(offset + length)
    }
}

public typealias ComputeBufferBinding = StorageBufferBinding

package struct SamplerRequirement: Equatable, Sendable {
    package let slot: ComputeBindingSlot
    package let isComparison: Bool
    package let requiresNonFiltering: Bool

    package init(slot: ComputeBindingSlot, isComparison: Bool, requiresNonFiltering: Bool = false) {
        self.slot = slot
        self.isComparison = isComparison
        self.requiresNonFiltering = requiresNonFiltering
    }
}

package struct SampledTextureRequirement: Equatable, Sendable {
    package let slot: ComputeBindingSlot
    package let sampleType: TextureSampleType
    package let isMultisampled: Bool
    package let dimension: TextureViewDimension
    package let requiresFiltering: Bool
    // The browser protocol uses 4 for unfilterable-float.
    package var webSampleType: UInt32 { sampleType == .float && !requiresFiltering ? 4 : sampleType.rawValue }

    package init(slot: ComputeBindingSlot, sampleType: TextureSampleType, isMultisampled: Bool = false,
                 dimension: TextureViewDimension = .d2, requiresFiltering: Bool = false) {
        self.slot = slot
        self.sampleType = sampleType
        self.isMultisampled = isMultisampled
        self.dimension = dimension
        self.requiresFiltering = requiresFiltering
    }
}

package enum StorageTextureAccess: UInt32, Sendable {
    case readOnly = 1
    case writeOnly = 2
    case readWrite = 3
    case atomic = 7
}

package struct StorageTextureRequirement: Equatable, Sendable {
    package let slot: ComputeBindingSlot
    package let format: TextureFormat
    package let access: StorageTextureAccess
    package let dimension: TextureViewDimension
    package var writable: Bool { access != .readOnly }

    package init(slot: ComputeBindingSlot, format: TextureFormat, access: StorageTextureAccess,
                 dimension: TextureViewDimension = .d2) {
        self.slot = slot
        self.format = format
        self.access = access
        self.dimension = dimension
    }
}

public struct ComputePipeline: Equatable, Sendable {
    package struct BufferBinding: Equatable, Sendable {
        package let slot: ComputeBindingSlot
        package let minimumByteSize: UInt64
        package let isReadOnly: Bool
        package let usage: BufferUsage

        package init(slot: ComputeBindingSlot, minimumByteSize: UInt64, isReadOnly: Bool, usage: BufferUsage = .storage) {
            self.slot = slot
            self.minimumByteSize = minimumByteSize
            self.isReadOnly = isReadOnly
            self.usage = usage
        }
    }

    public let id: ResourceID
    public let immediateByteSize: UInt64
    /// Maximum total workgroup count when the entry point uses a linear index.
    public let maximumLinearWorkgroupCount: UInt32?
    public let requiredSubgroupSize: UInt32?
    package let bufferBindings: [BufferBinding]

    package let textureBindings: [SampledTextureRequirement]
    package let externalTextureBindings: [ComputeBindingSlot]
    package let storageTextureBindings: [StorageTextureRequirement]
    package let samplerBindings: [SamplerRequirement]
    package let webBindingIndices: [ComputeBindingSlot: UInt32]
    /// Explicit and shader-inferred nearest-only sampler slots.
    public var nonFilteringSamplers: Set<ComputeBindingSlot> {
        Set(samplerBindings.filter(\.requiresNonFiltering).map(\.slot))
    }

    package init(id: ResourceID, storageBindings: [BufferBinding], uniformBindings: [BufferBinding] = [],
                 textureBindings: [SampledTextureRequirement] = [], storageTextureBindings: [StorageTextureRequirement] = [],
                 samplerBindings: [SamplerRequirement] = [], externalTextureBindings: [ComputeBindingSlot] = [],
                 webBindingIndices: [ComputeBindingSlot: UInt32] = [:], immediateByteSize: UInt64 = 0,
                 maximumLinearWorkgroupCount: UInt32? = nil, requiredSubgroupSize: UInt32? = nil) {
        self.id = id
        self.immediateByteSize = immediateByteSize
        self.maximumLinearWorkgroupCount = maximumLinearWorkgroupCount
        self.requiredSubgroupSize = requiredSubgroupSize
        self.bufferBindings = storageBindings + uniformBindings
        self.textureBindings = textureBindings
        self.externalTextureBindings = externalTextureBindings
        self.storageTextureBindings = storageTextureBindings
        self.samplerBindings = samplerBindings
        self.webBindingIndices = webBindingIndices
    }
}
