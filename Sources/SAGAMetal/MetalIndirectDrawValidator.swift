#if canImport(Metal)
import Metal
import SAGATypes

/// Checks GPU-authored arguments before Metal can use them for vertex or index fetch.
struct MetalIndirectDrawValidator {
    static let maximumCapacity: UInt64 = UInt64(UInt32.max) + 1

    private let prepare: any MTLComputePipelineState
    private let scan: any MTLComputePipelineState

    init(device: any MTLDevice) throws {
        let source = """
        #include <metal_stdlib>
        using namespace metal;

        bool fits(uint first, uint count, uint last) {
            return count != 0u && first <= last && count - 1u <= last - first;
        }

        kernel void prepare_draw(device const uint* source [[buffer(0)]],
                                 device uint* args [[buffer(1)]],
                                 constant uint4& bounds [[buffer(2)]]) {
            bool indexed = (bounds.z & 1u) != 0u;
            uint count = source[0];
            uint instances = source[1];
            uint first = source[2];
            uint baseVertex = indexed ? source[3] : 0u;
            uint firstInstance = indexed ? source[4] : source[3];
            bool valid = fits(first, count, bounds.x) && fits(firstInstance, instances, bounds.y);
            args[0] = valid ? count : 0u;
            args[1] = valid ? instances : 0u;
            args[2] = valid ? first : 0u;
            args[3] = valid ? (indexed ? baseVertex : firstInstance) : 0u;
            if (indexed) {
                args[4] = valid ? firstInstance : 0u;
                // The scan updates the native draw count atomically and uses this immutable count.
                args[5] = valid ? count : 0u;
            }
        }

        kernel void scan_indices(device const uint* indices [[buffer(0)]],
                                 device uint* args [[buffer(1)]],
                                 constant uint4& bounds [[buffer(2)]],
                                 uint i [[thread_position_in_grid]]) {
            bool shortIndex = (bounds.w & 1u) != 0u;
            bool restart = (bounds.w & 2u) != 0u;
            device const ushort* shorts = reinterpret_cast<device const ushort*>(indices) + bounds.y;
            uint count = args[5];
            uint base = args[3];
            while (i < count) {
                uint at = args[2] + i;
                uint index = shortIndex ? uint(shorts[at]) : indices[at];
                if (!restart || index != (shortIndex ? 0xffffu : 0xffffffffu)) {
                    uint adjusted = index + base;
                    bool overflow = as_type<int>(base) < 0 ? adjusted > index : adjusted < index;
                    if (overflow || adjusted > bounds.x) {
                        atomic_store_explicit(reinterpret_cast<device atomic_uint*>(args), 0u, memory_order_relaxed);
                        return;
                    }
                }
                if (count - i <= bounds.z) { break; }
                i += bounds.z;
            }
        }
        """
        do {
            let library = try device.makeLibrary(source: source, options: nil)
            guard let prepare = library.makeFunction(name: "prepare_draw"),
                  let scan = library.makeFunction(name: "scan_indices") else {
                throw GPUError.backendFailure("Metal indirect draw validator entry point missing")
            }
            self.prepare = try device.makeComputePipelineState(function: prepare)
            self.scan = try device.makeComputePipelineState(function: scan)
        } catch {
            throw GPUError.backendFailure("Metal indirect draw validator compilation: \(error)")
        }
    }

    func encode(
        on commands: any MTLCommandBuffer, device: any MTLDevice, source: any MTLBuffer, offset: Int,
        vertices: UInt64, instances: UInt64,
        indices: (binding: IndexBufferBinding, buffer: any MTLBuffer, count: UInt64)?,
        scanIndices: Bool, primitiveRestart: Bool
    ) throws -> any MTLBuffer {
        let vertexOrIndexCount = min(indices?.count ?? vertices, Self.maximumCapacity)
        var bounds = SIMD4<UInt32>(UInt32(vertexOrIndexCount - 1), UInt32(instances - 1), indices == nil ? 0 : 1, 0)
        guard let arguments = device.makeBuffer(length: indices == nil ? 16 : 24, options: .storageModePrivate),
              let encoder = commands.makeComputeCommandEncoder() else { throw GPUError.outOfMemory }
        encoder.setComputePipelineState(prepare)
        encoder.setBuffer(source, offset: offset, index: 0)
        encoder.setBuffer(arguments, offset: 0, index: 1)
        encoder.setBytes(&bounds, length: MemoryLayout.size(ofValue: bounds), index: 2)
        let one = MTLSize(width: 1, height: 1, depth: 1)
        encoder.dispatchThreadgroups(one, threadsPerThreadgroup: one)
        encoder.endEncoding()

        if let indices, scanIndices {
            guard let encoder = commands.makeComputeCommandEncoder() else { throw GPUError.outOfMemory }
            let width = min(64, scan.maxTotalThreadsPerThreadgroup)
            let groups = Int(min(16, (vertexOrIndexCount + UInt64(width) - 1) / UInt64(width)))
            let offset = Int(indices.binding.offset)
            let remainder = UInt32(offset % 4) / indices.binding.format.byteSize
            let formatFlag: UInt32 = indices.binding.format == .uint16 ? 1 : 0
            let restartFlag: UInt32 = primitiveRestart ? 2 : 0
            var scanBounds = SIMD4<UInt32>(UInt32(vertices - 1), remainder, UInt32(width * groups), formatFlag | restartFlag)
            encoder.setComputePipelineState(scan)
            encoder.setBuffer(indices.buffer, offset: offset & ~3, index: 0)
            encoder.setBuffer(arguments, offset: 0, index: 1)
            encoder.setBytes(&scanBounds, length: MemoryLayout.size(ofValue: scanBounds), index: 2)
            encoder.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
            encoder.endEncoding()
        }
        return arguments
    }
}
#endif
