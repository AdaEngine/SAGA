#if canImport(Metal)
import Metal
import SAGATypes

/// Applies the device workgroup limit after any earlier GPU writes to the argument buffer.
struct MetalDispatchValidator {
    private let pipeline: any MTLComputePipelineState

    init(device: any MTLDevice, maxWorkgroupsPerDimension: UInt32) throws {
        let source = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void validate_dispatch(device const packed_uint3& source [[buffer(0)]],
                                      device packed_uint3& destination [[buffer(1)]],
                                      constant uint& linear_limit [[buffer(2)]]) {
            uint3 groups(source);
            bool invalid = any(groups > \(maxWorkgroupsPerDimension)u) || any(groups == 0u);
            if (!invalid && linear_limit != 0u) {
                invalid = groups.z > linear_limit / groups.x / groups.y;
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
        maximumLinearWorkgroupCount: UInt32?
    ) throws -> any MTLBuffer {
        guard let arguments = device.makeBuffer(length: MemoryLayout<MTLDispatchThreadgroupsIndirectArguments>.size,
                                                options: .storageModePrivate),
              let encoder = commands.makeComputeCommandEncoder() else { throw GPUError.outOfMemory }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(source, offset: offset, index: 0)
        encoder.setBuffer(arguments, offset: 0, index: 1)
        var limit = maximumLinearWorkgroupCount ?? 0
        encoder.setBytes(&limit, length: MemoryLayout<UInt32>.size, index: 2)
        let one = MTLSize(width: 1, height: 1, depth: 1)
        encoder.dispatchThreadgroups(one, threadsPerThreadgroup: one)
        encoder.endEncoding()
        return arguments
    }
}
#endif
