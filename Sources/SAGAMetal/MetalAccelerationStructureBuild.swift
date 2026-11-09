#if canImport(Metal)
import Metal
import SAGATypes

/// Owns the scratch allocation until the encoded build retires.
struct MetalAccelerationStructureBuild {
    let structure: any MTLAccelerationStructure
    var scratch: (any MTLBuffer)?
    var instanceBuffer: (any MTLBuffer)? = nil
    var indexScratch: (any MTLBuffer)? = nil
    var dependencies: [MetalAccelerationStructureBuild] = []

    /// The native structures remain live with the TLAS; build inputs only until retirement.
    var persistent: Self {
        Self(structure: structure, scratch: nil, dependencies: dependencies.map(\.persistent))
    }

    static func encode(device: any MTLDevice, commandBuffer: any MTLCommandBuffer,
                       vertexBuffer: any MTLBuffer, vertexOffset: Int, vertexStride: Int,
                       vertexCount: Int,
                       indices: ((any MTLBuffer), IndexBufferBinding, UInt32)? = nil) throws -> Self {
        guard device.supportsRaytracing else { throw GPUError.unsupported("Metal ray tracing is unavailable") }
        let triangles = MTLAccelerationStructureTriangleGeometryDescriptor()
        triangles.vertexBuffer = vertexBuffer
        triangles.vertexBufferOffset = vertexOffset
        triangles.vertexStride = vertexStride
        var indexScratch: (any MTLBuffer)?
        if let (source, binding, count) = indices {
            // Copy and sanitize into private storage so later GPU writes cannot change the build input.
            let shader = """
            #include <metal_stdlib>
            using namespace metal;
            kernel void sanitize(device const ushort* shorts [[buffer(0)]],
                                 device uint* output [[buffer(1)]],
                                 constant uint4& params [[buffer(2)]],
                                 uint triangle [[thread_position_in_grid]]) {
                if (triangle >= params.x) return;
                uint i = triangle * 3u;
                device const uint* words = reinterpret_cast<device const uint*>(shorts);
                uint a = params.z == 2u ? uint(shorts[params.y + i]) : words[i];
                uint b = params.z == 2u ? uint(shorts[params.y + i + 1u]) : words[i + 1u];
                uint c = params.z == 2u ? uint(shorts[params.y + i + 2u]) : words[i + 2u];
                bool valid = a < params.w && b < params.w && c < params.w;
                output[i] = valid ? a : 0u;
                output[i + 1u] = valid ? b : 0u;
                output[i + 2u] = valid ? c : 0u;
            }
            """
            let library = try device.makeLibrary(source: shader, options: nil)
            guard let function = library.makeFunction(name: "sanitize"),
                  let scratch = device.makeBuffer(length: Int(count) * 4, options: .storageModePrivate),
                  let encoder = commandBuffer.makeComputeCommandEncoder() else { throw GPUError.outOfMemory }
            let pipeline = try device.makeComputePipelineState(function: function)
            let offset = Int(binding.offset)
            var params = SIMD4<UInt32>(count / 3, UInt32((offset % 4) / 2), binding.format.byteSize, UInt32(vertexCount))
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(source, offset: offset & ~3, index: 0)
            encoder.setBuffer(scratch, offset: 0, index: 1)
            encoder.setBytes(&params, length: MemoryLayout.size(ofValue: params), index: 2)
            let width = min(64, pipeline.maxTotalThreadsPerThreadgroup)
            encoder.dispatchThreads(MTLSize(width: Int(count / 3), height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
            encoder.endEncoding()
            indexScratch = scratch
            triangles.indexBuffer = scratch
            triangles.indexType = .uint32
            triangles.triangleCount = Int(count / 3)
        } else {
            triangles.triangleCount = vertexCount / 3
        }
        let descriptor = MTLPrimitiveAccelerationStructureDescriptor()
        descriptor.geometryDescriptors = [triangles]
        let sizes = device.accelerationStructureSizes(descriptor: descriptor)
        guard let structure = device.makeAccelerationStructure(size: sizes.accelerationStructureSize),
              let scratch = device.makeBuffer(length: max(1, sizes.buildScratchBufferSize), options: .storageModePrivate),
              let encoder = commandBuffer.makeAccelerationStructureCommandEncoder() else { throw GPUError.outOfMemory }
        encoder.build(accelerationStructure: structure, descriptor: descriptor, scratchBuffer: scratch, scratchBufferOffset: 0)
        encoder.endEncoding()
        return Self(structure: structure, scratch: scratch, indexScratch: indexScratch)
    }
    static func encodeTopLevel(device: any MTLDevice, commandBuffer: any MTLCommandBuffer,
                               instances: [(AccelerationStructureInstance, MetalAccelerationStructureBuild)]) throws -> Self {
        guard device.supportsRaytracing else { throw GPUError.unsupported("Metal ray tracing is unavailable") }
        var records = [MTLAccelerationStructureUserIDInstanceDescriptor]()
        for (index, pair) in instances.enumerated() {
            let instance = pair.0
            var record = MTLAccelerationStructureUserIDInstanceDescriptor()
            // Metal stores four packed columns of three floats; the public matrix is row-major 3x4.
            withUnsafeMutableBytes(of: &record.transformationMatrix) { bytes in
                for column in 0..<4 {
                    for row in 0..<3 {
                        bytes.storeBytes(of: instance.transform[row * 4 + column],
                                         toByteOffset: (column * 3 + row) * MemoryLayout<Float>.size, as: Float.self)
                    }
                }
            }
            record.mask = UInt32(instance.mask)
            record.accelerationStructureIndex = UInt32(index)
            record.userID = instance.instanceID
            records.append(record)
        }
        let length = max(1, records.count * MemoryLayout<MTLAccelerationStructureUserIDInstanceDescriptor>.stride)
        guard let instanceBuffer = device.makeBuffer(length: length, options: .storageModeShared) else {
            throw GPUError.outOfMemory
        }
        records.withUnsafeBytes { bytes in
            if let base = bytes.baseAddress { instanceBuffer.contents().copyMemory(from: base, byteCount: bytes.count) }
        }
        let descriptor = MTLInstanceAccelerationStructureDescriptor()
        descriptor.instanceDescriptorType = .userID
        descriptor.instanceDescriptorBuffer = instanceBuffer
        descriptor.instanceCount = records.count
        descriptor.instancedAccelerationStructures = instances.map { $0.1.structure }
        let sizes = device.accelerationStructureSizes(descriptor: descriptor)
        guard let structure = device.makeAccelerationStructure(size: sizes.accelerationStructureSize),
              let scratch = device.makeBuffer(length: max(1, sizes.buildScratchBufferSize), options: .storageModePrivate),
              let encoder = commandBuffer.makeAccelerationStructureCommandEncoder() else { throw GPUError.outOfMemory }
        encoder.build(accelerationStructure: structure, descriptor: descriptor, scratchBuffer: scratch, scratchBufferOffset: 0)
        encoder.endEncoding()
        return Self(structure: structure, scratch: scratch, instanceBuffer: instanceBuffer,
                    dependencies: instances.map { $0.1 })
    }

}
#endif
