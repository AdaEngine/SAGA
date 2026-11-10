#if os(Windows)
import WinSDK
import SAGATypes
import SAGABackend
import SAGAShaderIR

// Mesh pipelines (ID3D12Device2 pipeline-state streams, amplification and mesh shaders) and DispatchMesh draws. A mesh
// pipeline reuses the render pipeline bookkeeping: the merged binding model gets the same root signature layout, and a
// mesh draw is recorded like a render command that ends in DispatchMesh.

extension D3D12Device: MeshBackendDevice {
    public func allocateMeshPipeline(
        _ pipeline: MeshPipeline, task: TaskStageArtifact?, mesh: MeshStageArtifact, fragment: RenderStageArtifact
    ) async throws {
        try requireActive()
        guard capabilities.features.contains(.meshShaders) else {
            throw GPUError.unsupported("This Direct3D 12 device does not support mesh shaders")
        }
        guard renderPipelines[pipeline.id] == nil else { throw GPUError.validation("Duplicate Direct3D 12 mesh pipeline") }
        guard case .text(.hlsl, let meshSource) = mesh.artifact, case .text(.hlsl, let fragmentSource) = fragment.artifact,
              let fragmentProfile = fragment.hlslProfile else {
            throw GPUError.validation("Direct3D 12 requires HLSL mesh and fragment stages")
        }
        var taskSource: String?
        if let task {
            guard pipeline.hasTaskShader, case .text(.hlsl, let source) = task.artifact else {
                throw GPUError.validation("Direct3D 12 requires an HLSL task stage that matches the pipeline description")
            }
            taskSource = source
        } else if pipeline.hasTaskShader {
            throw GPUError.validation("The mesh pipeline description expects a task stage")
        }
        guard pipeline.colorFormats.count == 1, let colorFormat = pipeline.colorFormats[0], let target = d3d12Format(colorFormat),
              pipeline.fragmentStorageTextures.isEmpty, pipeline.meshStorageTextures.isEmpty, pipeline.taskStorageTextures.isEmpty,
              !pipeline.alphaToCoverageEnabled, !pipeline.primitive.unclippedDepth else {
            throw GPUError.unsupported("Direct3D 12 mesh pipelines support one color target and no storage textures yet")
        }
        var depthFormat: UInt32 = 0
        if let depthState = pipeline.depthState {
            guard let formats = d3d12DepthFormats(depthState.format) else {
                throw GPUError.unsupported("Direct3D 12 does not implement depth format \(depthState.format)")
            }
            depthFormat = formats.dsv
        }
        let surrogate = meshRenderSurrogate(pipeline, colorFormat: colorFormat)
        guard surrogate.uniforms.allSatisfy({ $0.slot.arrayElement == nil }),
              surrogate.storageBuffers.allSatisfy({ $0.handle.slot.arrayElement == nil }),
              surrogate.sampledTextures.allSatisfy({ $0.handle.slot.arrayElement == nil && !$0.requirement.isMultisampled }),
              surrogate.samplerBindings.allSatisfy({ $0.handle.slot.arrayElement == nil }) else {
            throw GPUError.unsupported("Direct3D 12 mesh pipelines do not support binding arrays or multisampled textures yet")
        }
        if compiler == nil { compiler = try D3D12ShaderCompiler(directories: compilerDirectories) }
        let meshCode = try compiler!.compile(meshSource, entryPoint: "saga_main", profile: "ms_6_5")
        let taskCode = try taskSource.map { try compiler!.compile($0, entryPoint: "saga_main", profile: "as_6_5") }
        let fragmentCode = try compiler!.compile(fragmentSource, entryPoint: fragment.entryPoint, profile: fragmentProfile)

        // Same root parameter order as render pipelines: uniforms, storage buffers, then the view and sampler tables.
        var parameters: [D3D12RootParameter] = []
        for uniform in surrogate.uniforms {
            parameters.append(D3D12RootParameter(parameterType: D3D12.rootParameterCBV, shaderRegister: uniform.slot.binding,
                                                 registerSpace: uniform.slot.group))
        }
        var rootStorage: [(binding: RenderPipeline.StorageBufferBinding, root: Int)] = []
        for storage in surrogate.storageBuffers {
            rootStorage.append((storage, parameters.count))
            parameters.append(D3D12RootParameter(parameterType: storage.isReadOnly ? D3D12.rootParameterSRV : D3D12.rootParameterUAV,
                                                 shaderRegister: storage.handle.slot.binding, registerSpace: storage.handle.slot.group))
        }
        func ordered(_ slots: [ComputeBindingSlot]) -> [ComputeBindingSlot] {
            slots.sorted { ($0.group, $0.binding) < ($1.group, $1.binding) }
        }
        func ranges(_ slots: [ComputeBindingSlot], type: UInt32) -> [D3D12DescriptorRange] {
            ordered(slots).enumerated().map {
                D3D12DescriptorRange(rangeType: type, baseShaderRegister: $0.element.binding, registerSpace: $0.element.group,
                                     offsetInDescriptors: UInt32($0.offset))
            }
        }
        let viewSlots = surrogate.sampledTextures.map(\.handle.slot)
        let samplerSlots = surrogate.samplerBindings.map(\.handle.slot)
        let viewRanges = ranges(viewSlots, type: 0), samplerRanges = ranges(samplerSlots, type: 3)
        let viewMemory = UnsafeMutablePointer<D3D12DescriptorRange>.allocate(capacity: max(1, viewRanges.count))
        let samplerMemory = UnsafeMutablePointer<D3D12DescriptorRange>.allocate(capacity: max(1, samplerRanges.count))
        defer { viewMemory.deallocate(); samplerMemory.deallocate() }
        var viewRoot: Int?, samplerRoot: Int?
        if !viewRanges.isEmpty {
            viewMemory.initialize(from: viewRanges, count: viewRanges.count)
            viewRoot = parameters.count
            parameters.append(D3D12RootParameter(parameterType: 0, shaderRegister: UInt32(viewRanges.count),
                                                 unionTail: UInt64(UInt(bitPattern: viewMemory))))
        }
        if !samplerRanges.isEmpty {
            samplerMemory.initialize(from: samplerRanges, count: samplerRanges.count)
            samplerRoot = parameters.count
            parameters.append(D3D12RootParameter(parameterType: 0, shaderRegister: UInt32(samplerRanges.count),
                                                 unionTail: UInt64(UInt(bitPattern: samplerMemory))))
        }
        let (serialized, blob, errorBlob) = Self.serialize(library, parameters: &parameters, flags: 0)
        withExtendedLifetime(errorBlob) {}
        try d3dCheck(serialized, "root signature serialization")
        let root = try makeRootSignature(blob!)

        // The legacy description is only a scratch area: its blend, rasterizer, depth and format bytes feed the stream.
        let size = 656
        let desc = UnsafeMutableRawPointer.allocate(byteCount: size, alignment: 8)
        defer { desc.deallocate() }
        desc.initializeMemory(as: UInt8.self, repeating: 0, count: size)
        Self.writeFixedFunctionState(desc, pipeline: surrogate, target: target, depthFormat: depthFormat)
        let fixed = Array(UnsafeRawBufferPointer(start: desc, count: size))

        var stream: [UInt8] = []
        func align8() { while stream.count % 8 != 0 { stream.append(0) } }
        func append32(_ value: UInt32) { withUnsafeBytes(of: value) { stream += $0 } }
        func append64(_ value: UInt64) { withUnsafeBytes(of: value) { stream += $0 } }
        func subobject(_ type: UInt32, innerAlignment: Int, _ write: () -> Void) {
            align8()
            append32(type)
            while stream.count % innerAlignment != 0 { stream.append(0) }
            write()
            align8()
        }
        func shader(_ type: UInt32, _ bytes: UnsafeRawBufferPointer) {
            subobject(type, innerAlignment: 8) {
                append64(UInt64(UInt(bitPattern: bytes.baseAddress)))
                append64(UInt64(bytes.count))
            }
        }
        let state: COMObject = try fragmentCode.withUnsafeBytes { fragmentBytes in
            try meshCode.withUnsafeBytes { meshBytes in
                try (taskCode ?? []).withUnsafeBytes { taskBytes in
                    subobject(0, innerAlignment: 8) { append64(UInt64(UInt(bitPattern: root.pointer))) }
                    if taskCode != nil { shader(24, taskBytes) }
                    shader(25, meshBytes)
                    shader(2, fragmentBytes)
                    subobject(8, innerAlignment: 4) { stream += fixed[120..<448] }  // blend
                    subobject(9, innerAlignment: 4) { append32(surrogate.sampleMask) }
                    subobject(10, innerAlignment: 4) { stream += fixed[452..<496] }  // rasterizer
                    subobject(11, innerAlignment: 4) { stream += fixed[496..<548] }  // depth and stencil
                    subobject(15, innerAlignment: 4) {
                        append32(target)
                        for _ in 1..<8 { append32(0) }
                        append32(1)
                    }
                    subobject(16, innerAlignment: 4) { append32(depthFormat) }
                    subobject(17, innerAlignment: 4) { append32(surrogate.sampleCount); append32(0) }
                    return try stream.withUnsafeBytes { bytes in
                        var description: [UInt64] = [UInt64(bytes.count), UInt64(UInt(bitPattern: bytes.baseAddress))]
                        return try createStreamState(&description)
                    }
                }
            }
        }
        renderPipelines[pipeline.id] = D3D12RenderPipelineState(
            model: surrogate, rootSignature: root, state: state, topology: 0, immediateRoot: nil, immediateWords: 0,
            viewRoot: viewRoot, samplerRoot: samplerRoot, viewLayout: ordered(viewSlots).map { ($0, false) },
            samplerLayout: ordered(samplerSlots), bufferViewLayout: [], rootStorage: rootStorage, drawParametersRoot: nil)
        meshPipelines[pipeline.id] = pipeline.hasTaskShader
    }

    public func destroyMeshPipeline(id: ResourceID) async {
        meshPipelines[id] = nil
        renderPipelines[id] = nil
    }

    /// Creates a pipeline state from a stream description through `ID3D12Device2::CreatePipelineState`.
    private func createStreamState(_ description: inout [UInt64]) throws -> COMObject {
        typealias QueryInterface = @convention(c) (UnsafeMutableRawPointer, UnsafeRawPointer, UnsafeMutablePointer<UnsafeMutableRawPointer?>) -> HRESULT
        typealias CreateState = @convention(c) (UnsafeMutableRawPointer, UnsafeRawPointer, UnsafeRawPointer,
                                                UnsafeMutablePointer<UnsafeMutableRawPointer?>) -> HRESULT
        var deviceIID = D3DGUID("30baa41e-b15b-475c-a0bb-1af5c5b64328")  // ID3D12Device2
        var deviceObject: UnsafeMutableRawPointer?
        try d3dCheck(device.slot(0, QueryInterface.self)(device.pointer, &deviceIID, &deviceObject), "device interface")
        let device2 = COMObject(deviceObject!)
        var iid = IID.pipelineState
        var object: UnsafeMutableRawPointer?
        try description.withUnsafeBytes {
            try d3dCheck(device2.slot(47, CreateState.self)(device2.pointer, $0.baseAddress!, &iid, &object), "mesh pipeline creation")
        }
        return COMObject(object!)
    }

    /// Mesh shader support, read once from `D3D12_FEATURE_D3D12_OPTIONS7`.
    nonisolated static func supportsMeshShaders(_ device: COMObject) -> Bool {
        typealias CheckSupport = @convention(c) (UnsafeMutableRawPointer, UInt32, UnsafeMutableRawPointer, UInt32) -> HRESULT
        var options: [UInt32] = [0, 0]  // MeshShaderTier, SamplerFeedbackTier
        let result = options.withUnsafeMutableBytes {
            device.slot(13, CheckSupport.self)(device.pointer, 32, $0.baseAddress!, 8)
        }
        return result >= 0 && options[0] >= 10
    }

    /// Plans `drawMesh` as the equivalent render command; the caller dispatches the mesh groups.
    func equivalentRender(
        for command: BufferCommand
    ) throws -> (BufferCommand, (UInt32, UInt32, UInt32), (buffer: ResourceID, offset: UInt64)?) {
        guard case .drawMesh(let pipeline, let target, let colorLoad, let workgroups, let taskBuffers, let meshBuffers,
                             let taskTextures, let meshTextures, let taskSamplers, let meshSamplers, let additional, let colorStore,
                             let resolve, let depth, let indirect, let indirectOffset, let fragmentBuffers, let fragmentTextures, let fragmentSamplers,
                             let raster) = command, let state = renderPipelines[pipeline.id], meshPipelines[pipeline.id] != nil else {
            throw GPUError.validation("Unknown Direct3D 12 mesh pipeline")
        }
        let buffers = taskBuffers.merging(meshBuffers) { first, _ in first }.merging(fragmentBuffers) { first, _ in first }
        let equivalent = BufferCommand.render(
            pipeline: state.model, target: target, colorLoad: colorLoad, vertexCount: 0, vertexBuffers: [], firstVertex: 0,
            instances: .init(), uniforms: buffers,
            textures: taskTextures.merging(meshTextures) { first, _ in first }.merging(fragmentTextures) { first, _ in first },
            samplers: taskSamplers.merging(meshSamplers) { first, _ in first }.merging(fragmentSamplers) { first, _ in first },
            targetMipLevel: 0, targetDepthSlice: nil, depth: depth, resolve: resolve, blendConstant: .transparentBlack,
            additionalColors: additional, raster: raster, storageBuffers: buffers, immediates: [], colorStore: colorStore)
        return (equivalent, (workgroups.x, workgroups.y, workgroups.z), indirect.map { ($0, indirectOffset) })
    }

    /// Executes validated GPU-authored group counts through a `DispatchMesh` command signature.
    func executeIndirectMesh(list: COMObject, arguments: COMObject) throws {
        typealias ExecuteIndirect = @convention(c) (UnsafeMutableRawPointer, UnsafeMutableRawPointer, UInt32, UnsafeMutableRawPointer, UInt64,
                                                    UnsafeMutableRawPointer?, UInt64) -> Void
        let signature = try meshSignature()
        list.slot(59, ExecuteIndirect.self)(list.pointer, signature.pointer, 1, arguments.pointer, 0, nil, 0)
    }

    /// Records `DispatchMesh` through `ID3D12GraphicsCommandList6`.
    func dispatchMesh(list: COMObject, groups: (UInt32, UInt32, UInt32)) throws {
        typealias QueryInterface = @convention(c) (UnsafeMutableRawPointer, UnsafeRawPointer, UnsafeMutablePointer<UnsafeMutableRawPointer?>) -> HRESULT
        typealias DispatchMeshCall = @convention(c) (UnsafeMutableRawPointer, UInt32, UInt32, UInt32) -> Void
        var iid = D3DGUID("c3827890-e548-4cfa-96cf-5689a9370f80")  // ID3D12GraphicsCommandList6
        var object: UnsafeMutableRawPointer?
        try d3dCheck(list.slot(0, QueryInterface.self)(list.pointer, &iid, &object), "command list interface")
        let list6 = COMObject(object!)
        guard groups.0 > 0, groups.1 > 0, groups.2 > 0 else { return }
        list6.slot(79, DispatchMeshCall.self)(list6.pointer, groups.0, groups.1, groups.2)
    }
}
#endif
