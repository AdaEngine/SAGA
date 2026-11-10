import SAGATypes
import SAGABackend
import SAGAShaderIR
import SAGACompiler

// Acceleration structures (VK_KHR_acceleration_structure) for ray queries. Structures live in host-visible buffers with
// device addresses; builds read vertex, index and instance data by address and write into the structure's buffer.

struct VulkanRayFunctions: Sendable {
    let create: @convention(c) (UInt, UnsafeRawPointer, UnsafeRawPointer?, UnsafeMutablePointer<UInt64>) -> VkResult
    let destroy: @convention(c) (UInt, UInt64, UnsafeRawPointer?) -> Void
    let buildSizes: @convention(c) (UInt, UInt32, UnsafeRawPointer, UnsafePointer<UInt32>, UnsafeMutableRawPointer) -> Void
    let cmdBuild: @convention(c) (UInt, UInt32, UnsafeRawPointer, UnsafePointer<UnsafeRawPointer?>) -> Void
    let structureAddress: @convention(c) (UInt, UnsafeRawPointer) -> UInt64
    let bufferAddress: @convention(c) (UInt, UnsafeRawPointer) -> UInt64

    init?(instance: VulkanInstance) {
        guard let library = VulkanLibrary.open(), let address = library.symbol("vkGetInstanceProcAddr") else { return nil }
        let getProc = unsafeBitCast(address, to: GetInstanceProcAddr.self)
        let handle = instance.handle
        func load<T>(_ name: String, as type: T.Type = T.self) -> T? {
            getProc(handle, name).map { unsafeBitCast($0, to: T.self) }
        }
        guard let create: @convention(c) (UInt, UnsafeRawPointer, UnsafeRawPointer?, UnsafeMutablePointer<UInt64>) -> VkResult = load("vkCreateAccelerationStructureKHR"),
              let destroy: @convention(c) (UInt, UInt64, UnsafeRawPointer?) -> Void = load("vkDestroyAccelerationStructureKHR"),
              let buildSizes: @convention(c) (UInt, UInt32, UnsafeRawPointer, UnsafePointer<UInt32>, UnsafeMutableRawPointer) -> Void = load("vkGetAccelerationStructureBuildSizesKHR"),
              let cmdBuild: @convention(c) (UInt, UInt32, UnsafeRawPointer, UnsafePointer<UnsafeRawPointer?>) -> Void = load("vkCmdBuildAccelerationStructuresKHR"),
              let structureAddress: @convention(c) (UInt, UnsafeRawPointer) -> UInt64 = load("vkGetAccelerationStructureDeviceAddressKHR"),
              let bufferAddress: @convention(c) (UInt, UnsafeRawPointer) -> UInt64 = load("vkGetBufferDeviceAddressKHR") else { return nil }
        (self.create, self.destroy, self.buildSizes, self.cmdBuild, self.structureAddress, self.bufferAddress) =
            (create, destroy, buildSizes, cmdBuild, structureAddress, bufferAddress)
    }
}

/// `VkWriteDescriptorSetAccelerationStructureKHR`.
struct VkWriteDescriptorSetAccelerationStructure {
    var sType: UInt32 = 1000150007
    var pNext: UnsafeRawPointer?
    var accelerationStructureCount: UInt32 = 1
    var pAccelerationStructures: UnsafeRawPointer?
}

extension VulkanDevice {
    static let accelerationDescriptorType: UInt32 = 1000150000

    struct AccelerationStructureState {
        var descriptor: AccelerationStructureDescriptor
        var handle: UInt64
        var buffer: VulkanHostBuffer
        var address: UInt64
    }

    /// What one structure build reads.
    enum BuildInput {
        case triangles(vertexAddress: UInt64, stride: UInt64, maxVertex: UInt32, indexAddress: UInt64?)
        case instances(address: UInt64)
        case boxes(address: UInt64, stride: UInt64)
    }

    /// A 96-byte `VkAccelerationStructureGeometryKHR`.
    private func geometryWords(_ input: BuildInput) -> [UInt64] {
        var words = [UInt64](repeating: 0, count: 12)
        words[0] = 1000150006
        switch input {
        case .triangles(let vertexAddress, let stride, let maxVertex, let indexAddress):
            words[2] = 0  // TRIANGLES
            words[3] = 1000150005  // VkAccelerationStructureGeometryTrianglesDataKHR
            words[5] = 106  // R32G32B32_SFLOAT
            words[6] = vertexAddress
            words[7] = stride
            words[8] = UInt64(maxVertex) | UInt64(indexAddress == nil ? 1_000_165_000 : 1) << 32  // indexType: NONE or UINT32
            words[9] = indexAddress ?? 0
        case .boxes(let address, let stride):
            words[2] = 1  // AABBS
            words[3] = 1000150003  // VkAccelerationStructureGeometryAabbsDataKHR
            words[5] = address
            words[6] = stride
        case .instances(let address):
            words[2] = 2  // INSTANCES
            words[3] = 1000150004  // VkAccelerationStructureGeometryInstancesDataKHR
            words[6] = address
        }
        words[11] = 0  // not opaque, so candidates are reported and ray flags can cull by opacity like the Metal backend
        return words
    }

    /// A 80-byte `VkAccelerationStructureBuildGeometryInfoKHR` for one geometry.
    private func buildInfoWords(top: Bool, geometry: UnsafeRawPointer, destination: UInt64, scratch: UInt64) -> [UInt64] {
        var words = [UInt64](repeating: 0, count: 10)
        words[0] = 1000150000
        // Type, PREFER_FAST_TRACE and, with position fetch, ALLOW_DATA_ACCESS.
        words[2] = (top ? 0 : 1) | UInt64(native.positionFetch ? 0x14 : 0x4) << 32
        words[3] = 0  // BUILD
        words[5] = destination
        words[6] = 1  // geometryCount
        words[7] = UInt64(UInt(bitPattern: geometry))
        words[9] = scratch
        return words
    }

    func buildSizes(top: Bool, input: BuildInput, primitives: UInt32) throws -> (structure: UInt64, scratch: UInt64) {
        guard let ray = native.ray else { throw GPUError.unsupported("This Vulkan device does not support ray queries") }
        var geometry = geometryWords(input)
        var sizes = [UInt64](repeating: 0, count: 5)
        sizes[0] = 1000150020
        return geometry.withUnsafeMutableBytes { geometryBytes in
            let info = buildInfoWords(top: top, geometry: UnsafeRawPointer(geometryBytes.baseAddress!), destination: 0, scratch: 0)
            var count = primitives
            sizes.withUnsafeMutableBytes { sizeBytes in
                withUnsafePointer(to: &count) { counts in
                    info.withUnsafeBytes {
                        ray.buildSizes(native.device, 1, UnsafeRawPointer($0.baseAddress!), counts, sizeBytes.baseAddress!)
                    }
                }
            }
            return (sizes[2], sizes[4])
        }
    }

    func deviceAddress(of buffer: UInt64) throws -> UInt64 {
        guard let ray = native.ray else { throw GPUError.unsupported("This Vulkan device does not support ray queries") }
        var info: [UInt64] = [1000244001, 0, buffer]
        return info.withUnsafeMutableBytes { ray.bufferAddress(native.device, UnsafeRawPointer($0.baseAddress!)) }
    }

    // MARK: Allocation

    public func allocateAccelerationStructure(id: ResourceID, descriptor: AccelerationStructureDescriptor) async throws {
        try requireActive()
        guard let ray = native.ray else { throw GPUError.unsupported("This Vulkan device does not support ray queries") }
        guard accelerationStructures[id] == nil else { throw GPUError.validation("Duplicate Vulkan acceleration structure") }
        let top: Bool, primitives: UInt32, input: BuildInput
        switch descriptor.capacity {
        case .triangles(let count):
            (top, primitives) = (false, count)
            input = .triangles(vertexAddress: 0, stride: 12, maxVertex: max(count * 3, 3), indexAddress: nil)
        case .boxes(let count):
            (top, primitives) = (false, count)
            input = .boxes(address: 0, stride: 24)
        case .instances(let count):
            (top, primitives) = (true, count)
            input = .instances(address: 0)
        }
        let sizes = try buildSizes(top: top, input: input, primitives: max(1, primitives))
        let buffer = try native.makeHostBuffer(size: sizes.structure + 256)
        var info = [UInt64](repeating: 0, count: 8)
        info[0] = 1000150017
        info[3] = buffer.handle
        info[5] = sizes.structure
        info[6] = top ? 0 : 1
        var handle: UInt64 = 0
        let result: VkResult = info.withUnsafeBytes { ray.create(native.device, UnsafeRawPointer($0.baseAddress!), nil, &handle) }
        guard result == VK.success else {
            native.destroy(buffer)
            try vulkanCheck(result, "acceleration structure creation")
            return
        }
        var addressInfo: [UInt64] = [1000150002, 0, handle]
        let address = addressInfo.withUnsafeMutableBytes { ray.structureAddress(native.device, UnsafeRawPointer($0.baseAddress!)) }
        accelerationStructures[id] = AccelerationStructureState(descriptor: descriptor, handle: handle, buffer: buffer, address: address)
    }

    public func destroyAccelerationStructure(id: ResourceID) async { destroyAccelerationStructureNow(id: id) }

    func destroyAccelerationStructureNow(id: ResourceID) {
        guard let state = accelerationStructures.removeValue(forKey: id), let ray = native.ray else { return }
        ray.destroy(native.device, state.handle, nil)
        native.destroy(state.buffer)
    }

    // MARK: Builds

    /// Index data copied into a private buffer and sanitized: a triangle with an out-of-range index becomes degenerate.
    private func sanitizerPipeline() throws -> Pipeline {
        if let cachedIndexSanitizer { return cachedIndexSanitizer }
        let shader = try IndependentShaderCompiler().compileCompute(.wgsl("""
            struct Params { triangles: u32, firstHalf: u32, size: u32, vertices: u32, }
            @group(0) @binding(0) var<storage, read> source: array<u32>;
            @group(0) @binding(1) var<storage, read_write> output: array<u32>;
            @group(0) @binding(2) var<storage, read> params: Params;
            fn fetch(index: u32) -> u32 {
                if (params.size == 2u) {
                    let half = params.firstHalf + index;
                    return (source[half / 2u] >> ((half & 1u) * 16u)) & 0xffffu;
                }
                return source[index];
            }
            @compute @workgroup_size(64)
            fn sanitize(@builtin(global_invocation_id) id: vec3u) {
                let triangle = id.x;
                if (triangle >= params.triangles) { return; }
                let i = triangle * 3u;
                let a = fetch(i);
                let b = fetch(i + 1u);
                let c = fetch(i + 2u);
                let valid = a < params.vertices && b < params.vertices && c < params.vertices;
                output[i] = select(0u, a, valid);
                output[i + 1u] = select(0u, b, valid);
                output[i + 2u] = select(0u, c, valid);
            }
            """), to: .spirv)
        guard case .spirv(let words) = shader.artifact else { throw GPUError.backendFailure("Index sanitizer is not SPIR-V") }
        let model = try ComputePipeline(id: ResourceID(device: 0, serial: 0), shader: shader)
        let pipeline = try makePipeline(model, words: words, entryPoint: shader.entryPoint)
        cachedIndexSanitizer = pipeline
        return pipeline
    }

    /// Records the structure build for one `buildAccelerationStructure` or `buildTopLevelAccelerationStructure` command.
    func recordStructureBuild(
        _ command: BufferCommand, commandBuffer: UInt, batch: inout PendingBatch, barrier: () -> Void
    ) throws {
        guard let ray = native.ray else { throw GPUError.unsupported("This Vulkan device does not support ray queries") }
        let f = native.f
        switch command {
        case .buildAccelerationStructure(let destination, let geometry):
            guard let target = accelerationStructures[destination], let vertexBuffer = buffers[geometry.vertexBuffer] else {
                throw GPUError.validation("Unknown Vulkan acceleration structure build input")
            }
            _ = try geometry.vertexRange(bufferSize: vertexBuffer.descriptor.size)
            let vertexAddress = try deviceAddress(of: vertexBuffer.native.handle) + geometry.vertexOffset
            var indexAddress: UInt64?
            if let binding = geometry.indexBuffer {
                guard let indexBuffer = buffers[binding.buffer] else { throw GPUError.validation("Unknown Vulkan index buffer") }
                _ = try geometry.indexRange(bufferSize: indexBuffer.descriptor.size)
                let triangles = geometry.indexCount / 3
                let width = UInt64(binding.format.byteSize)
                let firstWord = binding.offset & ~3
                let copyBytes = ((binding.offset - firstWord) + UInt64(geometry.indexCount) * width + 3) & ~3
                let staged = try native.makeHostBuffer(size: max(4, copyBytes))
                let sanitized = try native.makeHostBuffer(size: UInt64(geometry.indexCount) * 4)
                let parameters = try native.makeHostBuffer(size: 16)
                batch.owned += [staged, sanitized, parameters]
                parameters.mapped.storeBytes(of: triangles, as: UInt32.self)
                (parameters.mapped + 4).storeBytes(of: UInt32((binding.offset - firstWord) / 2), as: UInt32.self)
                (parameters.mapped + 8).storeBytes(of: UInt32(width), as: UInt32.self)
                (parameters.mapped + 12).storeBytes(of: geometry.vertexCount, as: UInt32.self)
                var region = VkBufferCopy(srcOffset: firstWord, dstOffset: 0, size: copyBytes)
                f.cmdCopyBuffer(commandBuffer, indexBuffer.native.handle, staged.handle, 1, &region)
                barrier()
                let pipeline = try sanitizerPipeline()
                try bindDescriptors(commandBuffer: commandBuffer, bindPoint: VK.pipelineBindPointCompute, layout: pipeline.layout,
                                    setLayouts: pipeline.setLayouts, writes: [
                    .init(set: 0, binding: 0, type: VK.DescriptorType.storageBuffer, buffer: staged.handle, offset: 0, range: staged.size),
                    .init(set: 0, binding: 1, type: VK.DescriptorType.storageBuffer, buffer: sanitized.handle, offset: 0, range: sanitized.size),
                    .init(set: 0, binding: 2, type: VK.DescriptorType.storageBuffer, buffer: parameters.handle, offset: 0, range: 16),
                ], batch: &batch)
                f.cmdBindPipeline(commandBuffer, VK.pipelineBindPointCompute, pipeline.pipeline)
                f.cmdDispatch(commandBuffer, (triangles + 63) / 64, 1, 1)
                barrier()
                indexAddress = try deviceAddress(of: sanitized.handle)
            }
            try buildStructure(top: false, input: .triangles(vertexAddress: vertexAddress, stride: geometry.vertexStride,
                                                             maxVertex: geometry.vertexCount - 1, indexAddress: indexAddress),
                               primitives: geometry.triangleCount, target: target, commandBuffer: commandBuffer, batch: &batch, ray: ray)
        case .buildBoxAccelerationStructure(let destination, let geometry):
            guard let target = accelerationStructures[destination], let buffer = buffers[geometry.buffer] else {
                throw GPUError.validation("Unknown Vulkan acceleration structure build input")
            }
            _ = try geometry.range(bufferSize: buffer.descriptor.size)
            let address = try deviceAddress(of: buffer.native.handle) + geometry.offset
            try buildStructure(top: false, input: .boxes(address: address, stride: geometry.stride), primitives: geometry.count,
                               target: target, commandBuffer: commandBuffer, batch: &batch, ray: ray)
        case .buildTopLevelAccelerationStructure(let destination, let instances):
            guard let target = accelerationStructures[destination] else {
                throw GPUError.validation("Unknown Vulkan acceleration structure build input")
            }
            let records = try native.makeHostBuffer(size: UInt64(max(1, instances.count)) * 64)
            batch.owned.append(records)
            for (index, instance) in instances.enumerated() {
                guard let bottom = accelerationStructures[instance.accelerationStructure], instance.transform.count == 12 else {
                    throw GPUError.validation("Invalid Vulkan top-level acceleration structure instance")
                }
                let base = records.mapped + index * 64
                for (offset, value) in instance.transform.enumerated() { (base + offset * 4).storeBytes(of: value, as: Float.self) }
                (base + 48).storeBytes(of: (instance.instanceID & 0xff_ffff) | UInt32(instance.mask) << 24, as: UInt32.self)
                (base + 52).storeBytes(of: UInt32(0), as: UInt32.self)
                (base + 56).storeBytes(of: bottom.address, as: UInt64.self)
            }
            try buildStructure(top: true, input: .instances(address: try deviceAddress(of: records.handle)),
                               primitives: UInt32(instances.count), target: target, commandBuffer: commandBuffer, batch: &batch, ray: ray)
        default:
            throw GPUError.unsupported("Vulkan does not implement this acceleration structure command")
        }
        barrier()
    }

    private func buildStructure(
        top: Bool, input: BuildInput, primitives: UInt32, target: AccelerationStructureState, commandBuffer: UInt,
        batch: inout PendingBatch, ray: VulkanRayFunctions
    ) throws {
        let sizes = try buildSizes(top: top, input: input, primitives: max(1, primitives))
        let scratch = try native.makeHostBuffer(size: sizes.scratch + 512)
        batch.owned.append(scratch)
        // Scratch addresses must be aligned; the allocation is oversized so the aligned address stays inside it.
        let scratchAddress = (try deviceAddress(of: scratch.handle) + 255) & ~255
        var geometry = geometryWords(input)
        geometry.withUnsafeMutableBytes { geometryBytes in
            let info = buildInfoWords(top: top, geometry: UnsafeRawPointer(geometryBytes.baseAddress!), destination: target.handle,
                                      scratch: scratchAddress)
            let range: [UInt32] = [primitives, 0, 0, 0]
            range.withUnsafeBytes { rangeBytes in
                var pointer: UnsafeRawPointer? = UnsafeRawPointer(rangeBytes.baseAddress!)
                info.withUnsafeBytes {
                    ray.cmdBuild(commandBuffer, 1, UnsafeRawPointer($0.baseAddress!), &pointer)
                }
            }
        }
    }
}
