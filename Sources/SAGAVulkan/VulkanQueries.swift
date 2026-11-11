import SAGATypes
import SAGABackend

// Timestamp and occlusion queries.
//
// Every query is reset immediately before the pass that writes it, outside any render pass. Results that were
// never written resolve to zero, because reading an unavailable Vulkan query is undefined.

struct VkQueryPoolCreateInfo {
    var sType: UInt32 = 11
    var pNext: UnsafeRawPointer?
    var flags: UInt32 = 0
    var queryType: UInt32 = 0
    var queryCount: UInt32 = 0
    var pipelineStatistics: UInt32 = 0
}

enum VulkanQuery {
    static let occlusion: UInt32 = 0
    static let timestamp: UInt32 = 2
    /// 64-bit results; the wait flag guarantees a finished query is read even when the driver resolves it late.
    static let results64Bit: UInt32 = 0x1 | 0x2
    static let topOfPipe: UInt32 = 0x1
    static let bottomOfPipe: UInt32 = 0x2000
}

struct VulkanQuerySetState {
    var descriptor: QuerySetDescriptor
    var pool: UInt64
    /// Queries that some submitted pass has written.
    var written: Set<UInt32> = []
}

/// Query work wrapped around one pass: resets before it, timestamps around it.
struct MeasuredPass {
    var resets: [(pool: UInt64, index: UInt32)] = []
    var begin: (pool: UInt64, index: UInt32)?
    var end: (pool: UInt64, index: UInt32)?
}

struct QueryResolve {
    var pool: UInt64
    var written: [UInt32]
    var first: UInt32
    var count: UInt32
    var destination: UInt64
    var destinationOffset: UInt64
}

extension VulkanDevice: QuerySetBackendDevice {
    public func allocateQuerySet(id: ResourceID, descriptor: QuerySetDescriptor) async throws {
        try requireActive()
        try descriptor.validate(against: capabilities)
        guard querySets[id] == nil else { throw GPUError.validation("Duplicate Vulkan query set") }
        var info = VkQueryPoolCreateInfo()
        info.queryType = descriptor.type == .timestamp ? VulkanQuery.timestamp : VulkanQuery.occlusion
        info.queryCount = descriptor.count
        var pool: UInt64 = 0
        let g = native.g, device = native.device
        try vulkanCheck(withUnsafePointer(to: &info) { g.createQueryPool(device, UnsafeRawPointer($0), nil, &pool) },
                        "query pool creation")
        querySets[id] = VulkanQuerySetState(descriptor: descriptor, pool: pool)
    }

    public func destroyQuerySet(id: ResourceID) async { destroyQuerySetNow(id: id) }

    func destroyQuerySetNow(id: ResourceID) {
        guard let state = querySets.removeValue(forKey: id) else { return }
        native.g.destroyQueryPool(native.device, state.pool, nil)
    }

    /// Plans the query side of a pass and marks the written queries.
    func planMeasurement(timestamps: PassTimestampWrites?, occlusion: PassOcclusionQuery?)
        throws -> (measured: MeasuredPass, occlusion: (pool: UInt64, index: UInt32)?) {
        var measured = MeasuredPass()
        var occlusionTarget: (pool: UInt64, index: UInt32)?
        if let timestamps {
            guard var state = querySets[timestamps.querySet], state.descriptor.type == .timestamp else {
                throw GPUError.validation("Unknown Vulkan timestamp query set")
            }
            if let index = timestamps.beginningOfPassWriteIndex {
                measured.resets.append((state.pool, index))
                measured.begin = (state.pool, index)
                state.written.insert(index)
            }
            if let index = timestamps.endOfPassWriteIndex {
                measured.resets.append((state.pool, index))
                measured.end = (state.pool, index)
                state.written.insert(index)
            }
            querySets[timestamps.querySet] = state
        }
        if let occlusion {
            guard var state = querySets[occlusion.querySet], state.descriptor.type == .occlusion else {
                throw GPUError.validation("Unknown Vulkan occlusion query set")
            }
            measured.resets.append((state.pool, occlusion.queryIndex))
            occlusionTarget = (state.pool, occlusion.queryIndex)
            state.written.insert(occlusion.queryIndex)
            querySets[occlusion.querySet] = state
        }
        return (measured, occlusionTarget)
    }

    func planResolve(querySet: ResourceID, first: UInt32, count: UInt32, destination: ResourceID,
                     offset: UInt64) throws -> QueryResolve {
        guard let state = querySets[querySet], let buffer = buffers[destination] else {
            throw GPUError.validation("Unknown Vulkan query resolve resource")
        }
        let written = state.written.filter { $0 >= first && $0 < first + count }.sorted()
        return QueryResolve(pool: state.pool, written: written, first: first, count: count,
                            destination: buffer.native.handle, destinationOffset: offset)
    }
}
