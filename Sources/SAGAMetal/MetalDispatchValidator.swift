#if canImport(Metal)
import Metal
import SAGATypes

/// Applies the device workgroup limit after any earlier GPU writes to the argument buffer.
struct MetalDispatchValidator {
    private let pipeline: any MTLComputePipelineState

    init(device: any MTLDevice) throws {
        let source = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void validate_dispatch(device const packed_uint3& source [[buffer(0)]],
                                      device packed_uint3& destination [[buffer(1)]],
                                      constant uint2& linear_limit [[buffer(2)]],
                                      constant uint& axis_limit [[buffer(3)]]) {
            uint3 groups(source);
            bool invalid = any(groups > axis_limit) || any(groups == 0u);
            if (!invalid && linear_limit.y != 0u) {
                invalid = groups.z > linear_limit.x / groups.x / groups.y;
            }
            destination = invalid ? packed_uint3(0u) : packed_uint3(groups);
        }
        """
        do {
            let library = try device.makeLibrary(source: source, options: nil)
            guard let function = library.makeFunction(name: "validate_dispatch") else {
                throw GPUError.backendFailure("Metal dispatch validator entry point missing")
            }
            pipeline = try device.makeComputePipelineState(function: function)
        } catch {
            throw GPUError.backendFailure("Metal dispatch validator compilation: \(error)")
        }
    }

    func encode(
        on commands: any MTLCommandBuffer, device: any MTLDevice, source: any MTLBuffer, offset: Int,
        maxWorkgroupsPerDimension: UInt32, maximumLinearWorkgroupCount: UInt32?
    ) throws -> any MTLBuffer {
        guard let arguments = device.makeBuffer(length: MemoryLayout<MTLDispatchThreadgroupsIndirectArguments>.size,
                                                options: .storageModePrivate),
              let encoder = commands.makeComputeCommandEncoder() else { throw GPUError.outOfMemory }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(source, offset: offset, index: 0)
        encoder.setBuffer(arguments, offset: 0, index: 1)
        var limit = SIMD2<UInt32>(maximumLinearWorkgroupCount ?? 0, maximumLinearWorkgroupCount == nil ? 0 : 1)
        encoder.setBytes(&limit, length: MemoryLayout<SIMD2<UInt32>>.size, index: 2)
        var axisLimit = maxWorkgroupsPerDimension
        encoder.setBytes(&axisLimit, length: MemoryLayout<UInt32>.size, index: 3)
        let one = MTLSize(width: 1, height: 1, depth: 1)
        encoder.dispatchThreadgroups(one, threadsPerThreadgroup: one)
        encoder.endEncoding()
        return arguments
    }
}
#endif
