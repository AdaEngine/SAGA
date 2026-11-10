import SAGATypes
import SAGABackend
import SAGAShaderIR

// Mesh pipelines (VK_EXT_mesh_shader). A mesh pipeline reuses the ordinary graphics pipeline builder with mesh and
// fragment stages and no vertex input, and a mesh draw is planned like a render command that ends in
// vkCmdDrawMeshTasksEXT.

extension VulkanDevice: MeshBackendDevice {
    public func allocateMeshPipeline(
        _ pipeline: MeshPipeline, task: TaskStageArtifact?, mesh: MeshStageArtifact, fragment: RenderStageArtifact
    ) async throws {
        try requireActive()
        guard capabilities.features.contains(.meshShaders), native.drawMeshTasks != nil else {
            throw GPUError.unsupported("This Vulkan device does not support mesh shaders")
        }
        guard case .spirv(let meshWords) = mesh.artifact, case .spirv(let fragmentWords) = fragment.artifact else {
            throw GPUError.validation("Vulkan requires SPIR-V mesh and fragment stages")
        }
        guard pipeline.colorFormats.count == 1, let colorFormat = pipeline.colorFormats[0],
              pipeline.fragmentStorageTextures.isEmpty, pipeline.meshStorageTextures.isEmpty,
              !pipeline.hasTaskShader, task == nil else {
            throw GPUError.unsupported("Vulkan mesh pipelines support one color target, no task stage and no storage textures yet")
        }
        // Merge the per-stage binding lists into the graphics pipeline model the ordinary draw path understands.
        var uniforms: [ComputeBindingSlot: RenderPipeline.Uniform] = [:]
        var storage: [ComputeBindingSlot: RenderPipeline.StorageBufferBinding] = [:]
        func add(_ bindings: [ComputePipeline.BufferBinding], visibility: UInt32) {
            for binding in bindings {
                let handle = RenderPipeline.HandleBinding(slot: binding.slot, vertexIndex: nil, fragmentIndex: nil, visibility: visibility)
                if binding.usage == .uniform {
                    let previous = uniforms[binding.slot]
                    uniforms[binding.slot] = .init(slot: binding.slot, minimumByteSize: max(previous?.minimumByteSize ?? 0, binding.minimumByteSize),
                                                   vertexIndex: nil, fragmentIndex: nil, visibility: (previous?.visibility ?? 0) | visibility)
                } else {
                    storage[binding.slot] = .init(handle: handle, minimumByteSize: max(storage[binding.slot]?.minimumByteSize ?? 0, binding.minimumByteSize),
                                                  isReadOnly: binding.isReadOnly && (storage[binding.slot]?.isReadOnly ?? true))
                }
            }
        }
        add(pipeline.meshBufferBindings, visibility: 1)
        add(pipeline.fragmentBufferBindings, visibility: 2)
        var textures: [ComputeBindingSlot: RenderPipeline.SampledTextureBinding] = [:]
        for requirement in pipeline.meshTextures + pipeline.fragmentTextures {
            textures[requirement.slot] = .init(handle: .init(slot: requirement.slot, vertexIndex: nil, fragmentIndex: nil, visibility: 3),
                                               requirement: requirement)
        }
        var samplers: [ComputeBindingSlot: RenderPipeline.SamplerBinding] = [:]
        for requirement in pipeline.meshSamplers + pipeline.fragmentSamplers {
            samplers[requirement.slot] = .init(handle: .init(slot: requirement.slot, vertexIndex: nil, fragmentIndex: nil, visibility: 3),
                                               requirement: requirement)
        }
        func ordered<T>(_ values: [ComputeBindingSlot: T]) -> [T] {
            values.sorted { ($0.key.group, $0.key.binding, $0.key.arrayElement ?? 0) < ($1.key.group, $1.key.binding, $1.key.arrayElement ?? 0) }
                .map(\.value)
        }
        let surrogate = RenderPipeline(
            id: pipeline.id, colorFormat: colorFormat, depthState: pipeline.depthState, sampleCount: pipeline.sampleCount,
            sampleMask: pipeline.sampleMask, alphaToCoverageEnabled: pipeline.alphaToCoverageEnabled,
            primitive: pipeline.primitive, uniforms: ordered(uniforms), storageBuffers: ordered(storage),
            sampledTextures: ordered(textures), samplerBindings: ordered(samplers))
        try buildGraphicsPipeline(surrogate, stages: [
            GraphicsStage(stage: VK.Stage.mesh, words: meshWords, entryPoint: mesh.sourceEntryPoint),
            GraphicsStage(stage: VK.Stage.fragment, words: fragmentWords, entryPoint: fragment.entryPoint)], meshStages: true)
    }

    public func destroyMeshPipeline(id: ResourceID) async { destroyRenderPipelineNow(id: id) }

    /// Plans `drawMesh` as an equivalent render command and records the mesh task dimensions.
    func planMeshDraw(_ command: BufferCommand, slot: Int, totals: inout DescriptorTotals) throws -> RenderPlan {
        guard case .drawMesh(let pipeline, let target, let colorLoad, let workgroups, let taskBuffers, let meshBuffers,
                             _, let meshTextures, _, let meshSamplers, let additional, let colorStore, let resolve, let depth,
                             let indirect, let indirectOffset, let fragmentBuffers, let fragmentTextures, let fragmentSamplers, let raster) = command,
              let state = renderPipelines[pipeline.id] else {
            throw GPUError.validation("Unknown Vulkan mesh pipeline")
        }
        guard taskBuffers.isEmpty else {
            throw GPUError.unsupported("Vulkan mesh draws do not support task stages yet")
        }
        let buffers = meshBuffers.merging(fragmentBuffers) { first, _ in first }
        let equivalent = BufferCommand.render(
            pipeline: state.model, target: target, colorLoad: colorLoad, vertexCount: 0, vertexBuffers: [], firstVertex: 0,
            instances: .init(), uniforms: buffers, textures: meshTextures.merging(fragmentTextures) { first, _ in first },
            samplers: meshSamplers.merging(fragmentSamplers) { first, _ in first }, targetMipLevel: 0, targetDepthSlice: nil,
            depth: depth, resolve: resolve, blendConstant: .transparentBlack, additionalColors: additional, raster: raster,
            storageBuffers: buffers, immediates: [], colorStore: colorStore)
        var plan = try planRender(equivalent, totals: &totals)
        plan.meshGroups = (workgroups.x, workgroups.y, workgroups.z)
        if let indirect {
            guard let source = self.buffers[indirect], indirectOffset % 4 == 0, indirectOffset <= source.descriptor.size,
                  source.descriptor.size - indirectOffset >= 12 else {
                throw GPUError.validation("Invalid Vulkan indirect mesh draw arguments")
            }
            plan.indirect = (source.native.handle, indirectOffset, slot)
        }
        return plan
    }
}
