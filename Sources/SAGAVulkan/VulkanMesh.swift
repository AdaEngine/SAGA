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
        var taskWords: [UInt32]?
        if let task {
            guard pipeline.hasTaskShader, case .spirv(let words) = task.artifact else {
                throw GPUError.validation("Vulkan requires a SPIR-V task stage that matches the pipeline description")
            }
            taskWords = words
        } else if pipeline.hasTaskShader {
            throw GPUError.validation("The mesh pipeline description expects a task stage")
        }
        guard pipeline.colorFormats.count == 1, let colorFormat = pipeline.colorFormats[0],
              pipeline.fragmentStorageTextures.isEmpty, pipeline.meshStorageTextures.isEmpty,
              pipeline.taskStorageTextures.isEmpty else {
            throw GPUError.unsupported("Vulkan mesh pipelines support one color target and no storage textures yet")
        }
        let surrogate = meshRenderSurrogate(pipeline, colorFormat: colorFormat)
        var stages: [GraphicsStage] = []
        if let task, let taskWords {
            stages.append(GraphicsStage(stage: VK.Stage.task, words: taskWords, entryPoint: task.sourceEntryPoint))
        }
        stages.append(GraphicsStage(stage: VK.Stage.mesh, words: meshWords, entryPoint: mesh.sourceEntryPoint))
        stages.append(GraphicsStage(stage: VK.Stage.fragment, words: fragmentWords, entryPoint: fragment.entryPoint))
        try buildGraphicsPipeline(surrogate, stages: stages, meshStages: true)
        meshPipelinesWithTask[pipeline.id] = taskWords != nil
    }

    public func destroyMeshPipeline(id: ResourceID) async {
        meshPipelinesWithTask[id] = nil
        destroyRenderPipelineNow(id: id)
    }

    /// Plans `drawMesh` as an equivalent render command and records the mesh task dimensions.
    func planMeshDraw(_ command: BufferCommand, slot: Int, totals: inout DescriptorTotals) throws -> RenderPlan {
        guard case .drawMesh(let pipeline, let target, let colorLoad, let workgroups, let taskBuffers, let meshBuffers,
                             let taskTextures, let meshTextures, let taskSamplers, let meshSamplers, let additional, let colorStore, let resolve, let depth,
                             let indirect, let indirectOffset, let fragmentBuffers, let fragmentTextures, let fragmentSamplers, let raster) = command,
              let state = renderPipelines[pipeline.id] else {
            throw GPUError.validation("Unknown Vulkan mesh pipeline")
        }
        let buffers = taskBuffers.merging(meshBuffers) { first, _ in first }.merging(fragmentBuffers) { first, _ in first }
        let equivalent = BufferCommand.render(
            pipeline: state.model, target: target, colorLoad: colorLoad, vertexCount: 0, vertexBuffers: [], firstVertex: 0,
            instances: .init(), uniforms: buffers, textures: taskTextures.merging(meshTextures) { first, _ in first }.merging(fragmentTextures) { first, _ in first },
            samplers: taskSamplers.merging(meshSamplers) { first, _ in first }.merging(fragmentSamplers) { first, _ in first }, targetMipLevel: 0, targetDepthSlice: nil,
            depth: depth, resolve: resolve, blendConstant: .transparentBlack, additionalColors: additional, raster: raster,
            storageBuffers: buffers, immediates: [], colorStore: colorStore)
        var plan = try planRender(equivalent, totals: &totals)
        plan.meshGroups = (workgroups.x, workgroups.y, workgroups.z)
        plan.meshUsesTask = meshPipelinesWithTask[pipeline.id] ?? false
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
