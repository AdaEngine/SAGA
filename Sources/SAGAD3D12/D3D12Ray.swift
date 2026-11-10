#if os(Windows)
import WinSDK
import SAGATypes
import SAGABackend

// DXR acceleration structures for HLSL ray queries. A structure is a default-heap buffer in the acceleration structure
// state; builds go through ID3D12GraphicsCommandList4 and compute shaders read the structure through a root SRV.

extension D3D12Device: ComputeAccelerationStructureBackendDevice {
    struct AccelerationStructureState {
        var descriptor: AccelerationStructureDescriptor
        var resource: COMObject
        var address: UInt64
    }

    static let stateAccelerationStructure: UInt32 = 0x400000

    /// Ray queries need DXR tier 1.1, read from `D3D12_FEATURE_D3D12_OPTIONS5`.
    nonisolated static func supportsRayQueries(_ device: COMObject) -> Bool {
        typealias CheckSupport = @convention(c) (UnsafeMutableRawPointer, UInt32, UnsafeMutableRawPointer, UInt32) -> HRESULT
        var options: [UInt32] = [0, 0, 0]  // SRVOnlyTiledResourcesTier, RenderPassesTier, RaytracingTier
        let result = options.withUnsafeMutableBytes {
            device.slot(13, CheckSupport.self)(device.pointer, 27, $0.baseAddress!, 12)
        }
        return result >= 0 && options[2] >= 11
    }

    private func queryInterface(_ object: COMObject, _ iid: String) throws -> COMObject {
        typealias QueryInterface = @convention(c) (UnsafeMutableRawPointer, UnsafeRawPointer, UnsafeMutablePointer<UnsafeMutableRawPointer?>) -> HRESULT
        var guid = D3DGUID(iid)
        var result: UnsafeMutableRawPointer?
        try d3dCheck(object.slot(0, QueryInterface.self)(object.pointer, &guid, &result), "interface query")
        return COMObject(result!)
    }

    /// A raytracing geometry description array (56 bytes per triangle geometry) for one input.
    private func triangleGeometry(vertexAddress: UInt64, stride: UInt64, vertexCount: UInt32, indexAddress: UInt64?,
                                  indexCount: UInt32) -> [UInt64] {
        var words = [UInt64](repeating: 0, count: 7)
        words[0] = 0  // TRIANGLES, flags NONE (not opaque, like the Metal backend)
        // Triangles: Transform3x4 (words[1]), IndexFormat | VertexFormat, IndexCount | VertexCount, IndexBuffer, then vertex buffer.
        words[2] = UInt64(indexAddress == nil ? 0 : 42) | UInt64(6) << 32  // R32_UINT or none; R32G32B32_FLOAT
        words[3] = UInt64(indexCount) | UInt64(vertexCount) << 32
        words[4] = indexAddress ?? 0
        words[5] = vertexAddress
        words[6] = stride
        return words
    }

    /// The prebuild sizes (structure, scratch) for either kind of input.
    private func prebuild(top: Bool, count: UInt32, geometry: [UInt64]?) throws -> (structure: UInt64, scratch: UInt64) {
        let device5 = try queryInterface(device, "8b4f173b-2fea-4b80-8f58-4307191ab95d")  // ID3D12Device5
        typealias Prebuild = @convention(c) (UnsafeMutableRawPointer, UnsafeRawPointer, UnsafeMutableRawPointer) -> Void
        var info = [UInt64](repeating: 0, count: 3)
        var geometryWords = geometry ?? [0]
        geometryWords.withUnsafeMutableBytes { geometryBytes in
            // D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS: type, flags, count, layout, pointer.
            let inputs: [UInt64] = [(top ? 0 : 1) | UInt64(0x4) << 32, UInt64(count) | 0 << 32,
                                    top ? 0 : UInt64(UInt(bitPattern: geometryBytes.baseAddress))]
            inputs.withUnsafeBytes { inputBytes in
                info.withUnsafeMutableBytes {
                    device5.slot(63, Prebuild.self)(device5.pointer, inputBytes.baseAddress!, $0.baseAddress!)
                }
            }
        }
        return (info[0], info[1])
    }

    public func allocateAccelerationStructure(id: ResourceID, descriptor: AccelerationStructureDescriptor) async throws {
        try requireActive()
        guard capabilities.features.contains(.rayQuery) else {
            throw GPUError.unsupported("This Direct3D 12 device does not support ray queries")
        }
        guard accelerationStructures[id] == nil else { throw GPUError.validation("Duplicate Direct3D 12 acceleration structure") }
        let sizes: (structure: UInt64, scratch: UInt64)
        switch descriptor.capacity {
        case .triangles(let count):
            sizes = try prebuild(top: false, count: 1, geometry: triangleGeometry(
                vertexAddress: 0, stride: 12, vertexCount: max(3, count * 3), indexAddress: nil, indexCount: 0))
        case .instances(let count):
            sizes = try prebuild(top: true, count: max(1, count), geometry: nil)
        }
        let resource = try makeBuffer(size: (sizes.structure + 255) & ~255, heap: D3D12.heapDefault,
                                      state: Self.stateAccelerationStructure, flags: D3D12.flagAllowUnorderedAccess)
        accelerationStructures[id] = AccelerationStructureState(descriptor: descriptor, resource: resource, address: gpuAddress(resource))
    }

    public func destroyAccelerationStructure(id: ResourceID) async { accelerationStructures[id] = nil }

    /// Sanitizes indices on the GPU: a triangle with an out-of-range index becomes degenerate.
    private func indexSanitizerPipeline() throws -> Pipeline {
        if let indexSanitizer { return indexSanitizer }
        if compiler == nil { compiler = try D3D12ShaderCompiler(directories: compilerDirectories) }
        let hlsl = """
            ByteAddressBuffer source : register(t0);
            RWByteAddressBuffer destination : register(u0);
            cbuffer Parameters : register(b0) { uint triangles; uint firstHalf; uint size; uint vertices; };
            uint fetch(uint index) {
                if (size == 2u) {
                    uint half_index = firstHalf + index;
                    return (source.Load((half_index / 2u) * 4u) >> ((half_index & 1u) * 16u)) & 0xffffu;
                }
                return source.Load(index * 4u);
            }
            [numthreads(64, 1, 1)]
            void sanitize(uint3 id : SV_DispatchThreadID) {
                if (id.x >= triangles) { return; }
                uint i = id.x * 3u;
                uint a = fetch(i), b = fetch(i + 1u), c = fetch(i + 2u);
                bool valid = a < vertices && b < vertices && c < vertices;
                destination.Store(i * 4u, valid ? a : 0u);
                destination.Store((i + 1u) * 4u, valid ? b : 0u);
                destination.Store((i + 2u) * 4u, valid ? c : 0u);
            }
            """
        let dxil = try compiler!.compile(hlsl, entryPoint: "sanitize", profile: "cs_6_0")
        var parameters = [
            D3D12RootParameter(parameterType: D3D12.rootParameterSRV, shaderRegister: 0, registerSpace: 0),
            D3D12RootParameter(parameterType: D3D12.rootParameterUAV, shaderRegister: 0, registerSpace: 0),
            D3D12RootParameter(parameterType: D3D12.rootParameterConstants, shaderRegister: 0, registerSpace: 0, unionTail: 4),
        ]
        let (serialized, blob, errorBlob) = Self.serialize(library, parameters: &parameters)
        withExtendedLifetime(errorBlob) {}
        try d3dCheck(serialized, "root signature serialization")
        let root = try makeRootSignature(blob!)
        typealias CreatePipeline = @convention(c) (UnsafeMutableRawPointer, UnsafeRawPointer, UnsafeRawPointer, UnsafeMutablePointer<UnsafeMutableRawPointer?>) -> HRESULT
        var stateIID = IID.pipelineState
        let (created, state) = Self.createState(device, root: root, dxil: dxil, stateIID: &stateIID, create: device.slot(11, CreatePipeline.self))
        try d3dCheck(created, "index sanitizer creation")
        let model = ComputePipeline(id: ResourceID(device: 0, serial: 0), storageBindings: [], uniformBindings: [])
        let pipeline = Pipeline(model: model, rootSignature: root, state: state!)
        indexSanitizer = pipeline
        return pipeline
    }

    /// Records one structure build. `transition` moves build inputs to the shader-resource state.
    func recordStructureBuild(
        _ command: BufferCommand, list: COMObject, transition: (ResourceID, UInt32) -> Void, retained: inout Retained
    ) throws {
        let list4 = try queryInterface(list, "8754318e-d3a9-4541-98cf-645b50dc4874")  // ID3D12GraphicsCommandList4
        typealias Barrier = @convention(c) (UnsafeMutableRawPointer, UInt32, UnsafeRawPointer) -> Void
        typealias Build = @convention(c) (UnsafeMutableRawPointer, UnsafeRawPointer, UInt32, UnsafeRawPointer?) -> Void
        func manual(_ resource: COMObject, _ before: UInt32, _ after: UInt32) {
            var barrier = D3D12ResourceBarrier(type: D3D12.barrierTransition, resource: resource.pointer,
                                               subresource: 0xFFFF_FFFF, stateBefore: before, stateAfter: after)
            list.slot(26, Barrier.self)(list.pointer, 1, &barrier)
        }
        func uav(_ resource: COMObject) {
            var barrier = D3D12ResourceBarrier(type: D3D12.barrierUAV, resource: resource.pointer)
            list.slot(26, Barrier.self)(list.pointer, 1, &barrier)
        }
        func build(top: Bool, count: UInt32, inputPointer: UInt64, target: AccelerationStructureState, scratchSize: UInt64) throws {
            let scratch = try makeBuffer(size: scratchSize + 256, heap: D3D12.heapDefault, state: D3D12.stateCommon,
                                         flags: D3D12.flagAllowUnorderedAccess)
            retained.objects.append(scratch)
            // D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC: destination, inputs (24 bytes), source, scratch.
            let description: [UInt64] = [target.address, (top ? 0 : 1) | UInt64(0x4) << 32, UInt64(count), inputPointer, 0, gpuAddress(scratch)]
            description.withUnsafeBytes { list4.slot(72, Build.self)(list4.pointer, $0.baseAddress!, 0, nil) }
            uav(target.resource)
        }
        switch command {
        case .buildAccelerationStructure(let destination, let geometry):
            guard let target = accelerationStructures[destination], let vertexBuffer = buffers[geometry.vertexBuffer] else {
                throw GPUError.validation("Unknown Direct3D 12 acceleration structure build input")
            }
            _ = try geometry.vertexRange(bufferSize: vertexBuffer.descriptor.size)
            transition(geometry.vertexBuffer, D3D12.stateNonPixelShaderResource)
            var indexAddress: UInt64?
            var indexCount: UInt32 = 0
            if let binding = geometry.indexBuffer {
                guard let indexBuffer = buffers[binding.buffer] else { throw GPUError.validation("Unknown Direct3D 12 index buffer") }
                _ = try geometry.indexRange(bufferSize: indexBuffer.descriptor.size)
                let width = UInt64(binding.format.byteSize)
                let firstWord = binding.offset & ~3
                let sanitized = try makeBuffer(size: UInt64(geometry.indexCount) * 4, heap: D3D12.heapDefault,
                                               state: D3D12.stateCommon, flags: D3D12.flagAllowUnorderedAccess)
                retained.objects.append(sanitized)
                let helper = try indexSanitizerPipeline()
                transition(binding.buffer, D3D12.stateNonPixelShaderResource)
                manual(sanitized, D3D12.stateCommon, D3D12.stateUnorderedAccess)
                typealias SetRoot = @convention(c) (UnsafeMutableRawPointer, UnsafeMutableRawPointer) -> Void
                typealias SetPipeline = @convention(c) (UnsafeMutableRawPointer, UnsafeMutableRawPointer) -> Void
                typealias SetAddress = @convention(c) (UnsafeMutableRawPointer, UInt32, UInt64) -> Void
                typealias SetConstants = @convention(c) (UnsafeMutableRawPointer, UInt32, UInt32, UnsafeRawPointer, UInt32) -> Void
                typealias Dispatch = @convention(c) (UnsafeMutableRawPointer, UInt32, UInt32, UInt32) -> Void
                list.slot(29, SetRoot.self)(list.pointer, helper.rootSignature.pointer)
                list.slot(25, SetPipeline.self)(list.pointer, helper.state.pointer)
                list.slot(39, SetAddress.self)(list.pointer, 0, indexBuffer.address + firstWord)
                list.slot(41, SetAddress.self)(list.pointer, 1, gpuAddress(sanitized))
                let triangles = geometry.indexCount / 3
                let constants: [UInt32] = [triangles, UInt32((binding.offset - firstWord) / 2), UInt32(width), geometry.vertexCount]
                constants.withUnsafeBytes { list.slot(35, SetConstants.self)(list.pointer, 2, 4, $0.baseAddress!, 0) }
                list.slot(14, Dispatch.self)(list.pointer, (triangles + 63) / 64, 1, 1)
                uav(sanitized)
                manual(sanitized, D3D12.stateUnorderedAccess, D3D12.stateNonPixelShaderResource)
                indexAddress = gpuAddress(sanitized)
                indexCount = geometry.indexCount
            }
            var geometryWords = triangleGeometry(vertexAddress: vertexBuffer.address + geometry.vertexOffset, stride: geometry.vertexStride,
                                                 vertexCount: geometry.vertexCount, indexAddress: indexAddress, indexCount: indexCount)
            let sizes = try prebuild(top: false, count: 1, geometry: geometryWords)
            try geometryWords.withUnsafeMutableBytes { bytes in
                try build(top: false, count: 1, inputPointer: UInt64(UInt(bitPattern: bytes.baseAddress)), target: target,
                          scratchSize: sizes.scratch)
            }
        case .buildTopLevelAccelerationStructure(let destination, let instances):
            guard let target = accelerationStructures[destination] else {
                throw GPUError.validation("Unknown Direct3D 12 acceleration structure build input")
            }
            let records = try makeBuffer(size: UInt64(max(1, instances.count)) * 64, heap: D3D12.heapUpload, state: D3D12.stateGenericRead)
            retained.objects.append(records)
            var bytes = [UInt8](repeating: 0, count: max(1, instances.count) * 64)
            for (index, instance) in instances.enumerated() {
                guard let bottom = accelerationStructures[instance.accelerationStructure], instance.transform.count == 12 else {
                    throw GPUError.validation("Invalid Direct3D 12 top-level acceleration structure instance")
                }
                func put32(_ offset: Int, _ value: UInt32) {
                    withUnsafeBytes(of: value) { for (i, byte) in $0.enumerated() { bytes[index * 64 + offset + i] = byte } }
                }
                for (offset, value) in instance.transform.enumerated() { put32(offset * 4, value.bitPattern) }
                put32(48, (instance.instanceID & 0xff_ffff) | UInt32(instance.mask) << 24)
                put32(52, 0)
                withUnsafeBytes(of: bottom.address) { for (i, byte) in $0.enumerated() { bytes[index * 64 + 56 + i] = byte } }
            }
            try map(records, count: bytes.count) { $0.copyMemory(from: bytes, byteCount: bytes.count) }
            let sizes = try prebuild(top: true, count: UInt32(max(1, instances.count)), geometry: nil)
            try build(top: true, count: UInt32(instances.count), inputPointer: gpuAddress(records), target: target, scratchSize: sizes.scratch)
        default:
            throw GPUError.unsupported("Direct3D 12 does not implement this acceleration structure command")
        }
    }
}
#endif
