#if canImport(Metal)
import Metal
import SAGATypes

/// Cancels an indexed draw if any fetched vertex lies outside a bound vertex buffer.
struct MetalIndexValidator {
    private let uint16: any MTLComputePipelineState
    private let uint32: any MTLComputePipelineState

    init(device: any MTLDevice) throws {
        let source = """
        #include <metal_stdlib>
        using namespace metal;
        template<typename Index>
        void validate_index(device const Index* indices, device atomic_uint* drawCount,
                            constant uint4& bounds, uint restart, uint i) {
            if (i >= bounds.x) { return; }
            uint index = uint(indices[i + bounds.y]);
            if (restart != 0u && index == uint(Index(-1))) { return; }
            uint adjusted = index + bounds.w;
            bool overflow = as_type<int>(bounds.w) < 0 ? adjusted > index : adjusted < index;
            if (overflow || adjusted > bounds.z) {
                atomic_store_explicit(drawCount, 0u, memory_order_relaxed);
            }
        }
        kernel void validate_index16(device const ushort* indices [[buffer(0)]],
                                     device atomic_uint* drawCount [[buffer(1)]],
                                     constant uint4& bounds [[buffer(2)]],
                                     constant uint& restart [[buffer(3)]],
                                     uint i [[thread_position_in_grid]]) {
            validate_index(indices, drawCount, bounds, restart, i);
        }
        kernel void validate_index32(device const uint* indices [[buffer(0)]],
                                     device atomic_uint* drawCount [[buffer(1)]],
                                     constant uint4& bounds [[buffer(2)]],
                                     constant uint& restart [[buffer(3)]],
                                     uint i [[thread_position_in_grid]]) {
            validate_index(indices, drawCount, bounds, restart, i);
        }
        """
        do {
            let library = try device.makeLibrary(source: source, options: nil)
            guard let short = library.makeFunction(name: "validate_index16"),
                  let word = library.makeFunction(name: "validate_index32") else {
                throw GPUError.backendFailure("Metal index validator entry point missing")
            }
            uint16 = try device.makeComputePipelineState(function: short)
            uint32 = try device.makeComputePipelineState(function: word)
        } catch {
            throw GPUError.backendFailure("Metal index validator compilation: \(error)")
        }
    }

    func encode(
        on commands: any MTLCommandBuffer, device: any MTLDevice,
        indices: any MTLBuffer, format: IndexFormat, byteOffset: Int,
        draw: IndexedDraw, maximumVertex: UInt32, primitiveRestart: Bool
    ) throws -> any MTLBuffer {
        var args = MTLDrawIndexedPrimitivesIndirectArguments(
            indexCount: draw.indexCount, instanceCount: draw.instances.count, indexStart: 0,
            baseVertex: draw.baseVertex, baseInstance: draw.instances.first)
        guard let indirect = device.makeBuffer(bytes: &args, length: MemoryLayout.size(ofValue: args),
                                              options: .storageModeShared),
              let encoder = commands.makeComputeCommandEncoder() else { throw GPUError.outOfMemory }
        let pipeline = format == .uint16 ? uint16 : uint32
        let alignedOffset = byteOffset & ~3
        var bounds = SIMD4<UInt32>(draw.indexCount, UInt32(byteOffset - alignedOffset) / format.byteSize,
                                   maximumVertex, UInt32(bitPattern: draw.baseVertex))
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(indices, offset: alignedOffset, index: 0)
        // The first word in the native indirect structure is its index count.
        encoder.setBuffer(indirect, offset: 0, index: 1)
        encoder.setBytes(&bounds, length: MemoryLayout.size(ofValue: bounds), index: 2)
        var restart: UInt32 = primitiveRestart ? 1 : 0
        encoder.setBytes(&restart, length: MemoryLayout.size(ofValue: restart), index: 3)
        let width = min(64, pipeline.maxTotalThreadsPerThreadgroup)
        let groups = (Int(draw.indexCount) + width - 1) / width
        encoder.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        encoder.endEncoding()
        return indirect
    }
}
#endif
