#if os(Windows)
import SAGATypes
import SAGABackend

// Timestamp and occlusion queries. Queries that no submitted pass has written resolve to zero, because resolving an
// unwritten Direct3D 12 query is undefined.

struct D3D12QuerySet {
    var descriptor: QuerySetDescriptor
    var heap: COMObject
    var written: Set<UInt32> = []
}

extension D3D12Device: QuerySetBackendDevice {
    public func allocateQuerySet(id: ResourceID, descriptor: QuerySetDescriptor) async throws {
        try requireActive()
        try descriptor.validate(against: capabilities)
        guard querySets[id] == nil else { throw GPUError.validation("Duplicate Direct3D 12 query set") }
        typealias CreateQueryHeap = @convention(c) (UnsafeMutableRawPointer, UnsafeRawPointer, UnsafeRawPointer,
                                                    UnsafeMutablePointer<UnsafeMutableRawPointer?>) -> HRESULT
        var description: (UInt32, UInt32, UInt32) = (descriptor.type == .timestamp ? 1 : 0, descriptor.count, 0)
        var iid = IID.queryHeap
        var object: UnsafeMutableRawPointer?
        try d3dCheck(device.slot(39, CreateQueryHeap.self)(device.pointer, &description, &iid, &object), "query heap creation")
        querySets[id] = D3D12QuerySet(descriptor: descriptor, heap: COMObject(object!))
    }

    public func destroyQuerySet(id: ResourceID) async { querySets[id] = nil }

    /// Query work around one pass: timestamp writes before and after it, and an occlusion query that brackets it.
    struct Measurement {
        var begin: (heap: COMObject, index: UInt32)?
        var end: (heap: COMObject, index: UInt32)?
        var occlusion: (heap: COMObject, index: UInt32)?
    }

    func planMeasurement(timestamps: PassTimestampWrites?, occlusion: PassOcclusionQuery?) throws -> Measurement {
        var measurement = Measurement()
        if let timestamps {
            guard var state = querySets[timestamps.querySet], state.descriptor.type == .timestamp else {
                throw GPUError.validation("Unknown Direct3D 12 timestamp query set")
            }
            if let index = timestamps.beginningOfPassWriteIndex { measurement.begin = (state.heap, index); state.written.insert(index) }
            if let index = timestamps.endOfPassWriteIndex { measurement.end = (state.heap, index); state.written.insert(index) }
            querySets[timestamps.querySet] = state
        }
        if let occlusion {
            guard var state = querySets[occlusion.querySet], state.descriptor.type == .occlusion else {
                throw GPUError.validation("Unknown Direct3D 12 occlusion query set")
            }
            measurement.occlusion = (state.heap, occlusion.queryIndex)
            state.written.insert(occlusion.queryIndex)
            querySets[occlusion.querySet] = state
        }
        return measurement
    }

    /// Resolves the written queries of a range into a buffer, zero-filling the rest.
    func recordResolve(querySet: ResourceID, first: UInt32, count: UInt32, destination: ResourceID, offset: UInt64,
                       list: COMObject, transition: (ResourceID, UInt32) -> Void, retained: inout Retained) throws {
        guard let state = querySets[querySet], let buffer = buffers[destination] else {
            throw GPUError.validation("Unknown Direct3D 12 query resolve resource")
        }
        typealias CopyRegion = @convention(c) (UnsafeMutableRawPointer, UnsafeMutableRawPointer, UInt64, UnsafeMutableRawPointer, UInt64, UInt64) -> Void
        typealias ResolveData = @convention(c) (UnsafeMutableRawPointer, UnsafeMutableRawPointer, UInt32, UInt32, UInt32,
                                                UnsafeMutableRawPointer, UInt64) -> Void
        let zeros = try makeBuffer(size: UInt64(count) * 8, heap: D3D12.heapUpload, state: D3D12.stateGenericRead)
        retained.objects.append(zeros)
        transition(destination, D3D12.stateCopyDestination)
        list.slot(15, CopyRegion.self)(list.pointer, buffer.resource.pointer, offset, zeros.pointer, 0, UInt64(count) * 8)
        let type: UInt32 = state.descriptor.type == .timestamp ? 2 : 0
        for index in state.written.filter({ $0 >= first && $0 < first + count }).sorted() {
            list.slot(54, ResolveData.self)(list.pointer, state.heap.pointer, type, index, 1, buffer.resource.pointer,
                                            offset + UInt64(index - first) * 8)
        }
    }
}
#endif
