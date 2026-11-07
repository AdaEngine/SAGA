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

    package init(slot: ComputeBindingSlot, sampleType: TextureSampleType, isMultisampled: Bool = false,
                 dimension: TextureViewDimension = .d2) {
        self.slot = slot
        self.sampleType = sampleType
        self.isMultisampled = isMultisampled
        self.dimension = dimension
    }
}

package enum StorageTextureAccess: UInt32, Sendable {
    case readOnly = 1
    case writeOnly = 2
    case readWrite = 3
}

package struct StorageTextureRequirement: Equatable, Sendable {
    package let slot: ComputeBindingSlot
    package let format: TextureFormat
    package let access: StorageTextureAccess
    package let isArrayed: Bool
    package var writable: Bool { access != .readOnly }

    package init(slot: ComputeBindingSlot, format: TextureFormat, access: StorageTextureAccess, isArrayed: Bool = false) {
        self.slot = slot
        self.format = format
        self.access = access
        self.isArrayed = isArrayed
    }
}

public struct ComputePipeline: Equatable, Sendable {
    package struct BufferBinding: Equatable, Sendable {
        let slot: ComputeBindingSlot
        let minimumByteSize: UInt64
        let isReadOnly: Bool
        let usage: BufferUsage

        package init(slot: ComputeBindingSlot, minimumByteSize: UInt64, isReadOnly: Bool, usage: BufferUsage = .storage) {
            self.slot = slot
            self.minimumByteSize = minimumByteSize
            self.isReadOnly = isReadOnly
            self.usage = usage
        }
    }

    public let id: ResourceID
    let bufferBindings: [BufferBinding]

    let textureBindings: [SampledTextureRequirement]
    let storageTextureBindings: [StorageTextureRequirement]
    let samplerBindings: [SamplerRequirement]

    package init(id: ResourceID, storageBindings: [BufferBinding], uniformBindings: [BufferBinding] = [],
                 textureBindings: [SampledTextureRequirement] = [], storageTextureBindings: [StorageTextureRequirement] = [],
                 samplerBindings: [SamplerRequirement] = []) {
        self.id = id
        self.bufferBindings = storageBindings + uniformBindings
        self.textureBindings = textureBindings
        self.storageTextureBindings = storageTextureBindings
        self.samplerBindings = samplerBindings
    }
}
