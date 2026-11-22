import Foundation
import SAGABackend
import SAGAShaderIR
import SAGATypes
#if canImport(Metal)
import Metal
#endif

extension MetalDevice {
    public func allocateComputePipeline(_ pipeline: ComputePipeline, shader: ComputeShaderArtifact) throws {
        try requireActive()
        #if canImport(Metal)
        guard case .text(.msl, let source) = shader.artifact else {
            throw GPUError.validation("Metal requires a compiled MSL compute shader")
        }
        let coherentStorage = shader.storageBindings.contains { $0.memoryDecorations.contains(.coherent) }
        if coherentStorage { try capabilities.require([.memoryDecorationCoherent]) }
        if shader.requiresDebugPrintf { try capabilities.require([.shaderDebugPrintf]) }
        try capabilities.require(shader.storageTextureFeatures)
        if shader.requiresShaderInt16 { try capabilities.require([.shaderInt16]) }
        if shader.requiresShaderInt64 { try capabilities.require([.shaderInt64]) }
        try capabilities.requireShaderInt64Atomics(minMax: shader.requiresShaderInt64AtomicMinMax,
                                                  allOps: shader.requiresShaderInt64AtomicAllOps)
        if shader.requiresSubgroups { try capabilities.require([.subgroups]) }
        if shader.requiresCooperativeMatrix { try capabilities.require([.cooperativeMatrix]) }
        if shader.requiresSubgroupUniformity { try capabilities.require([.subgroupUniformity]) }
        try capabilities.requireSubgroupSize(shader.requiredSubgroupSize)
        if shader.requiresSubgroupID { try capabilities.require([.subgroupID]) }
        if shader.requiresLinearIndexing { try capabilities.require([.linearIndexing]) }
        let reflected = try ComputePipeline(id: pipeline.id, shader: shader,
                                            nonFilteringSamplers: pipeline.nonFilteringSamplers)
        guard pipeline == reflected else {
            throw GPUError.validation("Metal compute pipeline layout does not match compiled shader")
        }
        guard computePipelines[pipeline.id] == nil else { throw GPUError.validation("Duplicate Metal compute pipeline") }
        try capabilities.requireImmediateSize(pipeline.immediateByteSize)
        if let immediate = shader.immediateLayout,
           immediate.metalBufferIndex != ComputeShaderArtifact.metalImmediateBufferIndex {
            throw GPUError.validation("Invalid Metal compute immediate slot")
        }
        var bufferIndices: [ComputeBindingSlot: Int] = [:]
        var occupied = Set<UInt32>()
        func addBuffers(_ slots: [ShaderBindingIndex], range: Range<UInt32>) throws {
            for (slot, index) in slots {
                guard let index, range.contains(index), occupied.insert(index).inserted, bufferIndices[slot] == nil else {
                    throw GPUError.validation("Invalid Metal compute buffer binding layout")
                }
                bufferIndices[slot] = Int(index)
            }
        }
        for binding in shader.storageBindings {
            try addBuffers(binding.indexedSlots(), range: 0..<ComputeShaderArtifact.storageBindingCapacity)
        }
        let uniformBase = ComputeShaderArtifact.uniformBindingBase
        let uniformLimit = uniformBase + ComputeShaderArtifact.uniformBindingCapacity
        for binding in shader.uniformBindings {
            try addBuffers(binding.indexedSlots(), range: uniformBase..<uniformLimit)
        }
        var accelerationIndices: [ComputeBindingSlot: Int] = [:]
        for binding in shader.accelerationStructures {
            let slot = ComputeBindingSlot(group: binding.binding.group, binding: binding.binding.binding)
            guard binding.flatBindingIndex < ComputeShaderArtifact.storageBindingCapacity,
                  occupied.insert(binding.flatBindingIndex).inserted, bufferIndices[slot] == nil,
                  accelerationIndices[slot] == nil else {
                throw GPUError.validation("Invalid Metal acceleration structure binding layout")
            }
            accelerationIndices[slot] = Int(binding.flatBindingIndex)
        }

        func addResource(
            _ binding: IRResourceBinding, array: ComputeShaderArtifact.BindingArrayLayout?, index: UInt32?,
            range: Range<UInt32>, occupied: inout Set<UInt32>, slots: inout [ComputeBindingSlot: Int]
        ) throws {
            guard let index, range.contains(index), (array?.count ?? 1) <= range.upperBound - index else {
                throw GPUError.validation("Invalid Metal resource index range")
            }
            let elements = try binding.computeSlots(array: array, limit: range.upperBound - range.lowerBound)
            for (offset, element) in elements.enumerated() {
                let native = index + UInt32(offset)
                guard occupied.insert(native).inserted, slots[element] == nil, bufferIndices[element] == nil,
                      accelerationIndices[element] == nil else {
                    throw GPUError.validation("Overlapping Metal resource bindings")
                }
                slots[element] = Int(native)
            }
        }

        var textureIndices: [ComputeBindingSlot: Int] = [:]
        var occupiedTextures = Set<UInt32>()
        for binding in shader.sampledTextures {
            try addResource(binding.binding, array: binding.bindingArray, index: binding.flatBindingIndex,
                            range: 0..<ComputeShaderArtifact.sampledTextureBindingCapacity,
                            occupied: &occupiedTextures, slots: &textureIndices)
        }

        let storageBase = ComputeShaderArtifact.sampledTextureBindingCapacity
        let storageLimit = storageBase + ComputeShaderArtifact.storageTextureBindingCapacity
        for binding in shader.storageTextures {
            try addResource(binding.binding, array: binding.bindingArray, index: binding.flatBindingIndex,
                            range: storageBase..<storageLimit, occupied: &occupiedTextures, slots: &textureIndices)
        }

        var samplerIndices: [ComputeBindingSlot: Int] = [:]
        var occupiedSamplers = Set<UInt32>()
        for binding in shader.samplerBindings {
            try addResource(binding.binding, array: binding.bindingArray, index: binding.flatBindingIndex,
                            range: 0..<ComputeShaderArtifact.samplerBindingCapacity,
                            occupied: &occupiedSamplers, slots: &samplerIndices)
        }
        guard samplerIndices.keys.allSatisfy({ textureIndices[$0] == nil }) else {
            throw GPUError.validation("Overlapping Metal texture and sampler bindings")
        }
        try validateExternalTextureLayouts(shader.externalTextureLayouts, bindings: shader.externalTextures,
                                           textureIndices: occupiedTextures, uniformIndices: occupied)

        let size = shader.workgroupSize
        let threads = MTLSize(width: Int(size.x), height: Int(size.y), depth: Int(size.z))
        try capabilities.requireComputeWorkgroupSize(x: size.x, y: size.y, z: size.z)
        try capabilities.requireWorkgroupStorageSize(shader.workgroupStorageByteSize)
        let state: any MTLComputePipelineState
        do {
            let options = coherentStorage || shader.requiresDebugPrintf ? MTLCompileOptions() : nil
            if #available(macOS 15, iOS 18, tvOS 18, visionOS 2, *) {
                options?.languageVersion = .version3_2
                options?.enableLogging = shader.requiresDebugPrintf
            }
            let library = try native.makeLibrary(source: source, options: options)
            guard let function = library.makeFunction(name: shader.entryPoint) else {
                throw GPUError.backendFailure("Compiled Metal entry point missing")
            }
            state = try native.makeComputePipelineState(function: function)
        } catch let error as GPUError {
            throw error
        } catch {
            throw GPUError.backendFailure("Compute MSL compilation: \(error)")
        }
        // Native compilation can add scratch space, including validation metadata.
        try capabilities.requireWorkgroupStorageSize(UInt64(state.staticThreadgroupMemoryLength))
        if let required = shader.requiredSubgroupSize, state.threadExecutionWidth != Int(required) {
            throw GPUError.validation("Requested subgroup size does not match the Metal pipeline SIMD width")
        }
        guard threads.width * threads.height * threads.depth <= state.maxTotalThreadsPerThreadgroup else {
            throw GPUError.validation("Workgroup size exceeds Metal pipeline limits")
        }
        let runtimeLengths = try runtimeLengthRequirements(shader.storageBindings)
        guard runtimeLengths.isEmpty == (shader.metalRuntimeLengthBuffer == nil),
              runtimeLengths.allSatisfy({ $0.1 > 0 }),
              shader.metalRuntimeLengthBuffer == nil ||
                  shader.metalRuntimeLengthBuffer == ComputeShaderArtifact.storageBindingCapacity
        else { throw GPUError.validation("Invalid Metal runtime array layout") }
        computePipelines[pipeline.id] = Pipeline(state: state, threadsPerThreadgroup: threads,
                                                bufferIndices: bufferIndices, accelerationIndices: accelerationIndices, textureIndices: textureIndices,
                                                samplerIndices: samplerIndices, runtimeLengths: runtimeLengths,
                                                lengthBufferIndex: shader.metalRuntimeLengthBuffer,
                                                immediateBufferIndex: shader.immediateLayout?.metalBufferIndex,
                                                externalTextures: shader.externalTextureLayouts)
        #else
        throw GPUError.adapterUnavailable
        #endif
    }

    public func destroyComputePipeline(id: ResourceID) {
        #if canImport(Metal)
        computePipelines.removeValue(forKey: id)
        #endif
    }

    public func allocateRenderPipeline(
        _ pipeline: RenderPipeline, vertex: RenderStageArtifact, fragment: RenderStageArtifact
    ) throws {
        try allocateRenderPipeline(pipeline, vertex: vertex, fragment: Optional(fragment))
    }

    public func allocateRenderPipeline(
        _ pipeline: RenderPipeline, vertex: RenderStageArtifact, fragment: RenderStageArtifact?
    ) throws {
        try requireActive()
        #if canImport(Metal)
        guard renderPipelines[pipeline.id] == nil, vertex.stage == .vertex,
              case .text(.msl, let vertexSource) = vertex.artifact else {
            throw GPUError.validation("Metal requires compiled MSL vertex and fragment stages")
        }
        let fragmentSource: String?
        if let fragment {
            guard fragment.stage == .fragment, case .text(.msl, let source) = fragment.artifact else {
                throw GPUError.validation("Metal requires compiled MSL vertex and fragment stages")
            }
            fragmentSource = source
        } else {
            fragmentSource = nil
        }
        try RenderPipeline.validateColorTargetLimits(pipeline.colorTargets, fragment: fragment, against: capabilities)
        try capabilities.validateInterstageInterfaces(vertex: vertex, fragment: fragment, topology: pipeline.primitive.topology,
                                                     multiviewMask: pipeline.multiviewMask)
        _ = try VertexBufferLayout.validate(pipeline.vertexBuffers, against: capabilities)
        try pipeline.validateBindGroupsPlusVertexBuffers(against: capabilities)
        if let pulling = vertex.metalVertexPullingLayout,
           pulling != VertexBufferLayout.metalPullingSignature(pipeline.vertexBuffers) {
            throw GPUError.validation("Vertex buffers do not match the Metal pulling layout")
        }
        if fragment?.readsPrimitiveIndex == true { try capabilities.require([.primitiveIndex]) }
        if fragment?.requiresBarycentrics == true { try capabilities.require([.shaderBarycentrics]) }
        guard vertex.immediateLayout.map({ $0.metalBufferIndex == RenderStageArtifact.metalVertexImmediateBufferIndex }) ?? true,
              fragment?.immediateLayout.map({ $0.metalBufferIndex == RenderStageArtifact.metalFragmentImmediateBufferIndex }) ?? true else {
            throw GPUError.validation("Invalid Metal render immediate slot")
        }
        if fragment?.earlyDepthTest != nil { try capabilities.require([.shaderEarlyDepthTest]) }
        if fragment?.requiresSubgroups == true || vertex.requiresSubgroups { try capabilities.require([.subgroups]) }
        if vertex.requiresSubgroupUniformity || fragment?.requiresSubgroupUniformity == true {
            try capabilities.require([.subgroupUniformity])
        }
        if vertex.requiresShaderInt16 || fragment?.requiresShaderInt16 == true { try capabilities.require([.shaderInt16]) }
        if vertex.requiresShaderInt64 || fragment?.requiresShaderInt64 == true { try capabilities.require([.shaderInt64]) }
        try capabilities.requireShaderInt64Atomics(
            minMax: vertex.requiresShaderInt64AtomicMinMax || fragment?.requiresShaderInt64AtomicMinMax == true,
            allOps: vertex.requiresShaderInt64AtomicAllOps || fragment?.requiresShaderInt64AtomicAllOps == true)
        if fragment?.usesDualSourceBlending == true || pipeline.colorTargets.values.contains(where: { $0.colorState.usesSecondBlendSource }) {
            try capabilities.require([.dualSourceBlending])
        }
        if pipeline.primitive.unclippedDepth { try capabilities.require([.depthClipControl]) }
        let vertexCoherentStorage = vertex.storageBindings.contains { $0.memoryDecorations.contains(.coherent) }
        let fragmentCoherentStorage = fragment?.storageBindings.contains { $0.memoryDecorations.contains(.coherent) } == true
        if vertexCoherentStorage || fragmentCoherentStorage { try capabilities.require([.memoryDecorationCoherent]) }
        if vertex.requiresDebugPrintf || fragment?.requiresDebugPrintf == true { try capabilities.require([.shaderDebugPrintf]) }
        let features = vertex.storageTextures.requiredStorageTextureFeatures
            .union((fragment?.storageTextures ?? []).requiredStorageTextureFeatures)
        try capabilities.require(features)
        let reflected = try RenderPipeline(id: pipeline.id, colorFormat: pipeline.colorFormat,
                                           vertexBuffers: pipeline.vertexBuffers, depthState: pipeline.depthState,
                                           sampleCount: pipeline.sampleCount, sampleMask: pipeline.sampleMask,
                                           alphaToCoverageEnabled: pipeline.alphaToCoverageEnabled, primitive: pipeline.primitive,
                                           colorState: pipeline.colorState, additionalColorTargets: pipeline.additionalColorTargets,
                                           vertex: vertex, fragment: fragment,
                                           nonFilteringSamplers: pipeline.nonFilteringSamplers, capabilities: capabilities,
                                           multiviewMask: pipeline.multiviewMask)
        guard pipeline == reflected else {
            throw GPUError.validation("Metal render pipeline layout does not match compiled shaders")
        }
        let needsAtomicGuard = fragment?.storageTextures.contains { $0.access.contains(.atomic) } == true
        let expectedGuard = needsAtomicGuard ? RenderStageArtifact.metalTextureAtomicGuardSlot : nil
        guard vertex.metalTextureAtomicGuardBufferIndex == nil,
              fragment?.metalTextureAtomicGuardBufferIndex == expectedGuard else {
            throw GPUError.validation("Invalid Metal fragment texture atomic guard layout")
        }
        let uniformBase = ComputeShaderArtifact.uniformBindingBase
        let uniformLimit = uniformBase + ComputeShaderArtifact.uniformBindingCapacity
        for stage in [vertex, fragment].compactMap({ $0 }) {
            let uniformSlots = try stage.uniformBindings.flatMap { try $0.indexedSlots() }
            let indices = uniformSlots.compactMap(\.index)
            guard indices.count == uniformSlots.count, Set(indices).count == indices.count,
                  indices.allSatisfy({ (uniformBase..<uniformLimit).contains($0) && Int($0) >= pipeline.vertexBuffers.count })
            else { throw GPUError.validation("Invalid Metal render uniform layout") }
            let storageBufferSlots = try stage.storageBindings.flatMap { try $0.indexedSlots() }
            let storageBufferIndices = storageBufferSlots.compactMap(\.index)
            let bufferBase = RenderStageArtifact.metalStorageBufferBase
            let bufferLimit = RenderStageArtifact.metalStorageLengthBufferIndex
            let lengths = try runtimeLengthRequirements(stage.storageBindings)
            let expectedLengthBuffer = lengths.isEmpty ? nil : bufferLimit
            guard storageBufferIndices.count == storageBufferSlots.count,
                  Set(storageBufferIndices).count == storageBufferIndices.count,
                  storageBufferIndices.allSatisfy({ (bufferBase..<bufferLimit).contains($0) }),
                  lengths.allSatisfy({ $0.1 > 0 }), stage.metalRuntimeLengthBuffer == expectedLengthBuffer else {
                throw GPUError.validation("Invalid Metal render storage buffer layout")
            }
            let textureSlots = try stage.sampledTextures.flatMap { try $0.indexedSlots() }
            let textureIndices = textureSlots.compactMap(\.index)
            guard textureIndices.count == textureSlots.count, Set(textureIndices).count == textureIndices.count,
                  textureIndices.allSatisfy({ $0 < ComputeShaderArtifact.sampledTextureBindingCapacity })
            else { throw GPUError.validation("Invalid Metal render sampled texture layout") }
            let storageBase = ComputeShaderArtifact.sampledTextureBindingCapacity
            let storageLimit = storageBase + ComputeShaderArtifact.storageTextureBindingCapacity
            let storageSlots = try stage.storageTextures.flatMap { try $0.indexedSlots() }
            let storageIndices = storageSlots.compactMap(\.index)
            guard storageIndices.count == storageSlots.count, Set(storageIndices).count == storageIndices.count,
                  storageIndices.allSatisfy({ (storageBase..<storageLimit).contains($0) })
            else { throw GPUError.validation("Invalid Metal render storage texture layout") }
            try validateExternalTextureLayouts(stage.externalTextureLayouts, bindings: stage.externalTextures,
                                               textureIndices: Set(textureIndices + storageIndices), uniformIndices: Set(indices))
            let samplerSlots = try stage.samplerBindings.flatMap { try $0.indexedSlots() }
            let samplerIndices = samplerSlots.compactMap(\.index)
            guard samplerIndices.count == samplerSlots.count, Set(samplerIndices).count == samplerIndices.count,
                  samplerIndices.allSatisfy({ $0 < ComputeShaderArtifact.samplerBindingCapacity })
            else { throw GPUError.validation("Invalid Metal render sampler layout") }
        }
        do {
            var multiviewMacros: [String: NSObject] = [:]
            if let mask = pipeline.multiviewMask {
                guard mask.nonzeroBitCount == 1 || native.supportsVertexAmplificationCount(mask.nonzeroBitCount) else {
                    throw GPUError.unsupported("Metal does not support the requested vertex amplification count")
                }
                multiviewMacros[RenderStageArtifact.metalMultiviewMaskMacro] = NSNumber(value: mask)
            }
            var vertexMacros = multiviewMacros
            if pipeline.primitive.topology == .pointList {
                vertexMacros[RenderStageArtifact.metalPointTopologyMacro] = NSNumber(value: 1)
            }
            let vertexOptions: MTLCompileOptions?
            if !vertexMacros.isEmpty || vertex.invariantPosition || vertexCoherentStorage || vertex.requiresDebugPrintf {
                let options = MTLCompileOptions()
                options.preserveInvariance = vertex.invariantPosition
                options.preprocessorMacros = vertexMacros
                if vertexCoherentStorage || vertex.requiresDebugPrintf, #available(macOS 15, iOS 18, tvOS 18, visionOS 2, *) {
                    options.languageVersion = .version3_2
                    options.enableLogging = vertex.requiresDebugPrintf
                }
                vertexOptions = options
            } else {
                vertexOptions = nil
            }
            let vs = try native.makeLibrary(source: vertexSource, options: vertexOptions)
            let activeSamples = (UInt32(1) << pipeline.sampleCount) - 1
            let sampleMask = pipeline.sampleMask & activeSamples
            guard let vertexFunction = vs.makeFunction(name: vertex.entryPoint) else {
                throw GPUError.backendFailure("Compiled Metal render entry point missing")
            }
            let fragmentFunction: (any MTLFunction)?
            if let fragment, let fragmentSource {
                var fragmentMacros = multiviewMacros
                if sampleMask != activeSamples {
                    fragmentMacros[RenderStageArtifact.metalPipelineSampleMaskMacro] = NSNumber(value: sampleMask)
                }
                let fragmentOptions: MTLCompileOptions?
                if fragmentMacros.isEmpty && !fragment.requiresPerVertex && !fragmentCoherentStorage && !fragment.requiresDebugPrintf {
                    fragmentOptions = nil
                } else {
                    let options = MTLCompileOptions()
                    options.preprocessorMacros = fragmentMacros
                    if fragmentCoherentStorage || fragment.requiresDebugPrintf, #available(macOS 15, iOS 18, tvOS 18, visionOS 2, *) {
                        options.languageVersion = .version3_2
                        options.enableLogging = fragment.requiresDebugPrintf
                    }
                    if fragment.requiresPerVertex {
                        guard #available(macOS 26, iOS 26, tvOS 26, visionOS 26, *) else {
                            throw GPUError.unsupported("Metal per-vertex fragment inputs require Metal 4")
                        }
                        options.languageVersion = .version4_0
                    }
                    fragmentOptions = options
                }
                let fs = try native.makeLibrary(source: fragmentSource, options: fragmentOptions)
                guard let function = fs.makeFunction(name: fragment.entryPoint) else {
                    throw GPUError.backendFailure("Compiled Metal render entry point missing")
                }
                fragmentFunction = function
            } else if sampleMask != activeSamples {
                // Metal applies programmable sample coverage through a fragment output.
                let source = """
                    struct SagaCoverage { uint mask [[sample_mask]]; };
                    fragment SagaCoverage saga_coverage() { return { \(sampleMask)u }; }
                    """
                let library = try native.makeLibrary(source: source, options: nil)
                guard let function = library.makeFunction(name: "saga_coverage") else {
                    throw GPUError.backendFailure("Compiled Metal coverage entry point missing")
                }
                fragmentFunction = function
            } else {
                fragmentFunction = nil
            }
            let descriptor = MTLRenderPipelineDescriptor()
            let samples = Int(pipeline.sampleCount)
            guard native.supportsTextureSampleCount(samples) else {
                throw GPUError.unsupported("Metal does not support the render pipeline sample count")
            }
            descriptor.rasterSampleCount = samples
            descriptor.maxVertexAmplificationCount = pipeline.multiviewMask?.nonzeroBitCount ?? 1
            descriptor.isAlphaToCoverageEnabled = pipeline.alphaToCoverageEnabled
            descriptor.inputPrimitiveTopology = pipeline.primitive.topology.metal.topologyClass
            descriptor.vertexFunction = vertexFunction
            descriptor.fragmentFunction = fragmentFunction
            for (location, target) in pipeline.colorTargets {
                let color = descriptor.colorAttachments[Int(location)]!
                color.pixelFormat = target.format.metal
                color.writeMask = pipeline.colorWriteMask(at: location).metal
                if let blend = target.colorState.blend {
                    color.isBlendingEnabled = true
                    color.rgbBlendOperation = blend.color.operation.metal
                    color.sourceRGBBlendFactor = blend.color.sourceFactor.metal
                    color.destinationRGBBlendFactor = blend.color.destinationFactor.metal
                    color.alphaBlendOperation = blend.alpha.operation.metal
                    color.sourceAlphaBlendFactor = blend.alpha.sourceFactor.metal
                    color.destinationAlphaBlendFactor = blend.alpha.destinationFactor.metal
                }
            }
            let depth: (any MTLDepthStencilState)?
            if let state = pipeline.depthState {
                if state.format.hasDepth { descriptor.depthAttachmentPixelFormat = state.format.metal }
                let depthDescriptor = MTLDepthStencilDescriptor()
                depthDescriptor.depthCompareFunction = state.compare.metal
                depthDescriptor.isDepthWriteEnabled = state.writeEnabled
                if state.format.hasStencil {
                    descriptor.stencilAttachmentPixelFormat = state.format.metal
                    let stencil = state.stencil ?? .init()
                    let writeMask = stencil.writes(cullMode: pipeline.primitive.cullMode) ? stencil.writeMask : 0
                    depthDescriptor.frontFaceStencil = stencil.front.metal(readMask: stencil.readMask, writeMask: writeMask)
                    depthDescriptor.backFaceStencil = stencil.back.metal(readMask: stencil.readMask, writeMask: writeMask)
                }
                guard let depthState = native.makeDepthStencilState(descriptor: depthDescriptor) else {
                    throw GPUError.backendFailure("Metal could not create a depth state")
                }
                depth = depthState
            } else {
                depth = nil
            }
            if !vertex.usesMetalVertexPulling, !pipeline.vertexBuffers.isEmpty {
                let vertexDescriptor = MTLVertexDescriptor()
                for (slot, layout) in pipeline.vertexBuffers.enumerated() where !layout.attributes.isEmpty {
                    let buffer = vertexDescriptor.layouts[slot]!
                    buffer.stride = layout.stride == 0 ? Int(layout.attributeEnd) : Int(layout.stride)
                    if layout.stride == 0 {
                        buffer.stepFunction = .constant
                    } else {
                        buffer.stepFunction = layout.stepMode == .instance ? .perInstance : .perVertex
                    }
                    buffer.stepRate = layout.stride == 0 ? 0 : 1
                    for attribute in layout.attributes {
                        let destination = vertexDescriptor.attributes[Int(attribute.location)]!
                        destination.format = attribute.format.metal
                        destination.bufferIndex = slot
                        destination.offset = Int(attribute.offset)
                    }
                }
                descriptor.vertexDescriptor = vertexDescriptor
            }
            let state = try native.makeRenderPipelineState(descriptor: descriptor)
            renderPipelines[pipeline.id] = RenderState(pipeline: state, depth: depth, writesDepth: fragment?.writesDepth == true,
                                                       usesVertexPulling: vertex.usesMetalVertexPulling,
                                                       vertexRuntimeLengths: try runtimeLengthRequirements(vertex.storageBindings),
                                                       fragmentRuntimeLengths: try runtimeLengthRequirements(fragment?.storageBindings ?? []),
                                                       vertexExternalTextures: vertex.externalTextureLayouts,
                                                       fragmentExternalTextures: fragment?.externalTextureLayouts ?? [],
                                                       fragmentTextureAtomicGuardBufferIndex: fragment?.metalTextureAtomicGuardBufferIndex,
                                                       vertexImmediateBufferIndex: vertex.immediateLayout?.metalBufferIndex,
                                                       fragmentImmediateBufferIndex: fragment?.immediateLayout?.metalBufferIndex,
                                                       metalRenderTargetLayer: vertex.metalRenderTargetLayer)
        } catch let error as GPUError {
            throw error
        } catch {
            throw GPUError.backendFailure("Render MSL compilation: \(error)")
        }
        #else
        throw GPUError.adapterUnavailable
        #endif
    }

    public func allocateMeshPipeline(
        _ pipeline: MeshPipeline, task: TaskStageArtifact?, mesh: MeshStageArtifact, fragment: RenderStageArtifact
    ) throws {
        try requireActive()
        #if canImport(Metal)
        guard #available(macOS 13, iOS 16, *), capabilities.features.contains(.meshShaders) else {
            throw GPUError.unsupported("Metal mesh shaders are unavailable")
        }
        let threadgroupMemoryLimit = UInt64(native.maxThreadgroupMemoryLength)
        guard mesh.workgroupStorageByteSize <= threadgroupMemoryLimit,
              (task?.workgroupStorageByteSize ?? 0) <= threadgroupMemoryLimit else {
            throw GPUError.validation("Metal mesh or object threadgroup storage exceeds the device limit")
        }
        if mesh.requiresSubgroups || task?.requiresSubgroups == true { try capabilities.require([.subgroups]) }
        if mesh.requiresShaderF16 || task?.requiresShaderF16 == true { try capabilities.require([.shaderF16]) }
        if mesh.requiresShaderInt16 || task?.requiresShaderInt16 == true { try capabilities.require([.shaderInt16]) }
        if mesh.requiresShaderInt64 || task?.requiresShaderInt64 == true || fragment.requiresShaderInt64 {
            try capabilities.require([.shaderInt64])
        }
        try capabilities.requireShaderInt64Atomics(
            minMax: mesh.requiresShaderInt64AtomicMinMax || task?.requiresShaderInt64AtomicMinMax == true || fragment.requiresShaderInt64AtomicMinMax,
            allOps: mesh.requiresShaderInt64AtomicAllOps || task?.requiresShaderInt64AtomicAllOps == true || fragment.requiresShaderInt64AtomicAllOps)
        guard meshPipelines[pipeline.id] == nil, case .text(.msl, let meshSource) = mesh.artifact,
              fragment.stage == .fragment, case .text(.msl, let fragmentSource) = fragment.artifact else {
            throw GPUError.validation("Metal requires compiled MSL mesh and fragment stages")
        }
        do {
            func compileOptions(_ textures: [ComputeShaderArtifact.StorageTextureBinding]) throws -> MTLCompileOptions {
                let options = MTLCompileOptions()
                if textures.contains(where: { $0.access.contains(.atomic) }) {
                    guard #available(macOS 14, iOS 17, *) else {
                        throw GPUError.unsupported("Metal texture atomics require Metal 3.1")
                    }
                    options.languageVersion = .version3_1
                } else {
                    options.languageVersion = .version3_0
                }
                return options
            }
            func function(_ source: String, _ name: String, textures: [ComputeShaderArtifact.StorageTextureBinding],
                          meshGridLimit: UInt32? = nil) throws -> any MTLFunction {
                let options = try compileOptions(textures)
                if let meshGridLimit {
                    options.preprocessorMacros = [TaskStageArtifact.metalMeshGridLimitMacro: NSNumber(value: meshGridLimit)]
                }
                let library = try native.makeLibrary(source: source, options: options)
                guard let function = library.makeFunction(name: name) else {
                    throw GPUError.backendFailure("Compiled Metal mesh entry point missing")
                }
                return function
            }
            let descriptor = MTLMeshRenderPipelineDescriptor()
            descriptor.rasterSampleCount = Int(pipeline.sampleCount)
            descriptor.meshFunction = try function(meshSource, mesh.entryPoint, textures: mesh.storageTextures)
            let activeSamples = (UInt32(1) << pipeline.sampleCount) - 1
            let sampleMask = pipeline.sampleMask & activeSamples
            let fragmentOptions = try compileOptions(fragment.storageTextures)
            if sampleMask != activeSamples {
                fragmentOptions.preprocessorMacros = [RenderStageArtifact.metalPipelineSampleMaskMacro: NSNumber(value: sampleMask)]
            }
            let fragmentLibrary = try native.makeLibrary(source: fragmentSource, options: fragmentOptions)
            guard let fragmentFunction = fragmentLibrary.makeFunction(name: fragment.entryPoint) else {
                throw GPUError.backendFailure("Compiled Metal mesh fragment entry point missing")
            }
            descriptor.fragmentFunction = fragmentFunction
            descriptor.isAlphaToCoverageEnabled = pipeline.alphaToCoverageEnabled
            if let task {
                guard case .text(.msl, let source) = task.artifact else {
                    throw GPUError.validation("Metal requires compiled MSL task stage")
                }
                // Metal's pinned task-output total and per-axis limits are equal.
                let meshGridLimit = min(capabilities.maxMeshWorkgroupsPerDimension, capabilities.maxMeshWorkgroupTotalCount)
                descriptor.objectFunction = try function(source, task.entryPoint, textures: task.storageTextures,
                                                         meshGridLimit: meshGridLimit)
            }
            for (location, format) in pipeline.colorFormats { descriptor.colorAttachments[Int(location)].pixelFormat = format.metal }
            let depth: (any MTLDepthStencilState)?
            if let state = pipeline.depthState {
                if state.format.hasDepth { descriptor.depthAttachmentPixelFormat = state.format.metal }
                let depthDescriptor = MTLDepthStencilDescriptor()
                depthDescriptor.depthCompareFunction = state.compare.metal
                depthDescriptor.isDepthWriteEnabled = state.writeEnabled
                if state.format.hasStencil {
                    descriptor.stencilAttachmentPixelFormat = state.format.metal
                    let stencil = state.stencil ?? .init()
                    let writeMask = stencil.writes(cullMode: pipeline.primitive.cullMode) ? stencil.writeMask : 0
                    depthDescriptor.frontFaceStencil = stencil.front.metal(readMask: stencil.readMask, writeMask: writeMask)
                    depthDescriptor.backFaceStencil = stencil.back.metal(readMask: stencil.readMask, writeMask: writeMask)
                }
                guard let value = native.makeDepthStencilState(descriptor: depthDescriptor) else { throw GPUError.backendFailure("Metal could not create a depth state") }
                depth = value
            } else { depth = nil }
            let (state, _) = try native.makeRenderPipelineState(descriptor: descriptor, options: [])
            func validateThreads(_ value: IRWorkgroupSize, maximum: Int) throws {
                let dimensions = [UInt64(value.x), UInt64(value.y), UInt64(value.z)]
                let product = dimensions.reduce(UInt64(1)) { partial, next in
                    partial.multipliedReportingOverflow(by: next).overflow ? UInt64.max : partial * next
                }
                guard dimensions.allSatisfy({ $0 > 0 }), product <= UInt64(maximum) else {
                    throw GPUError.validation("Metal mesh threadgroup exceeds the pipeline limit")
                }
            }
            try validateThreads(mesh.workgroupSize, maximum: state.maxTotalThreadsPerMeshThreadgroup)
            if let task { try validateThreads(task.workgroupSize, maximum: state.maxTotalThreadsPerObjectThreadgroup) }
            func size(_ value: IRWorkgroupSize) -> MTLSize {
                MTLSize(width: Int(value.x), height: Int(value.y), depth: Int(value.z))
            }
            func slots(_ uniforms: [ComputeShaderArtifact.UniformBinding],
                       _ storage: [ComputeShaderArtifact.StorageBinding]) throws -> ([(ComputeBindingSlot, Int)], [(ComputeBindingSlot, Int)]) {
                func mapped(_ values: [ShaderBindingIndex]) throws -> [(ComputeBindingSlot, Int)] {
                    try values.map { slot, index in
                        guard let index else { throw GPUError.validation("Missing Metal buffer index") }
                        return (slot, Int(index))
                    }
                }
                return (try uniforms.flatMap { try mapped($0.indexedSlots()) },
                        try storage.flatMap { try mapped($0.indexedSlots()) })
            }
            let objectSlots = try slots(task?.uniformBindings ?? [], task?.storageBindings ?? [])
            let meshSlots = try slots(mesh.uniformBindings, mesh.storageBindings)
            let fragmentSlots = try slots(fragment.uniformBindings, fragment.storageBindings)
            func resourceSlots(_ binding: IRResourceBinding, array: ComputeShaderArtifact.BindingArrayLayout?,
                               index: UInt32?, limit: UInt32) throws -> [(ComputeBindingSlot, Int)] {
                try binding.indexedSlots(array: array, limit: limit, first: index).map { slot, index in
                    guard let index else { throw GPUError.validation("Missing Metal resource index") }
                    return (slot, Int(index))
                }
            }
            func textureSlots(_ values: [ComputeShaderArtifact.SampledTextureBinding]) throws -> [(ComputeBindingSlot, Int)] {
                try values.flatMap { try resourceSlots($0.binding, array: $0.bindingArray, index: $0.flatBindingIndex,
                                                        limit: ComputeShaderArtifact.sampledTextureBindingCapacity) }
            }
            func samplerSlots(_ values: [ComputeShaderArtifact.SamplerBinding]) throws -> [(ComputeBindingSlot, Int)] {
                try values.flatMap { try resourceSlots($0.binding, array: $0.bindingArray, index: $0.flatBindingIndex,
                                                        limit: ComputeShaderArtifact.samplerBindingCapacity) }
            }
            let objectTextureSlots = try textureSlots(task?.sampledTextures ?? [])
            let meshTextureSlots = try textureSlots(mesh.sampledTextures)
            func storageTextureSlots(_ values: [ComputeShaderArtifact.StorageTextureBinding]) throws -> [(ComputeBindingSlot, Int)] {
                try values.flatMap { try resourceSlots($0.binding, array: $0.bindingArray, index: $0.flatBindingIndex,
                                                        limit: ComputeShaderArtifact.storageTextureBindingCapacity) }
            }
            let objectStorageTextureSlots = try storageTextureSlots(task?.storageTextures ?? [])
            let meshStorageTextureSlots = try storageTextureSlots(mesh.storageTextures)
            let objectSamplerSlots = try samplerSlots(task?.samplerBindings ?? [])
            let meshSamplerSlots = try samplerSlots(mesh.samplerBindings)
            // Object and mesh stages use the compute-stage Metal binding ABI. In
            // particular, buffer 8 is reserved for runtime array lengths.
            func validateMeshStageLayout(
                uniforms: [ComputeShaderArtifact.UniformBinding], storage: [ComputeShaderArtifact.StorageBinding],
                textures: [ComputeShaderArtifact.SampledTextureBinding], storageTextures: [ComputeShaderArtifact.StorageTextureBinding],
                samplers: [ComputeShaderArtifact.SamplerBinding], slots: ([(ComputeBindingSlot, Int)], [(ComputeBindingSlot, Int)]),
                textureSlots: [(ComputeBindingSlot, Int)], storageTextureSlots: [(ComputeBindingSlot, Int)],
                samplerSlots: [(ComputeBindingSlot, Int)]
            ) throws {
                let u = slots.0.map(\.1), s = slots.1.map(\.1)
                let t = textureSlots.map(\.1), st = storageTextureSlots.map(\.1), sm = samplerSlots.map(\.1)
                guard Set(u).count == u.count, u.allSatisfy({ (Int(ComputeShaderArtifact.uniformBindingBase)..<Int(ComputeShaderArtifact.uniformBindingBase + ComputeShaderArtifact.uniformBindingCapacity)).contains($0) }),
                      Set(s).count == s.count, s.allSatisfy({ (0..<Int(ComputeShaderArtifact.storageBindingCapacity)).contains($0) }),
                      Set(t + st).count == t.count + st.count,
                      t.allSatisfy({ (0..<Int(ComputeShaderArtifact.sampledTextureBindingCapacity)).contains($0) }),
                      st.allSatisfy({ (Int(ComputeShaderArtifact.sampledTextureBindingCapacity)..<Int(ComputeShaderArtifact.sampledTextureBindingCapacity + ComputeShaderArtifact.storageTextureBindingCapacity)).contains($0) }),
                      Set(sm).count == sm.count, sm.allSatisfy({ (0..<Int(ComputeShaderArtifact.samplerBindingCapacity)).contains($0) }) else {
                    throw GPUError.validation("Invalid Metal object or mesh resource layout")
                }
            }
            try validateMeshStageLayout(uniforms: task?.uniformBindings ?? [], storage: task?.storageBindings ?? [],
                textures: task?.sampledTextures ?? [], storageTextures: task?.storageTextures ?? [], samplers: task?.samplerBindings ?? [],
                slots: objectSlots, textureSlots: objectTextureSlots, storageTextureSlots: objectStorageTextureSlots, samplerSlots: objectSamplerSlots)
            try validateMeshStageLayout(uniforms: mesh.uniformBindings, storage: mesh.storageBindings,
                textures: mesh.sampledTextures, storageTextures: mesh.storageTextures, samplers: mesh.samplerBindings,
                slots: meshSlots, textureSlots: meshTextureSlots, storageTextureSlots: meshStorageTextureSlots, samplerSlots: meshSamplerSlots)
            let fragmentSampledTextureSlots = try textureSlots(fragment.sampledTextures)
            let fragmentStorageTextureSlots = try storageTextureSlots(fragment.storageTextures)
            let fragmentTextureSlots = fragmentSampledTextureSlots + fragmentStorageTextureSlots
            let fragmentSamplerSlots = try samplerSlots(fragment.samplerBindings)
            let fragmentLengths = try runtimeLengthRequirements(fragment.storageBindings)
            let uniformIndices = fragmentSlots.0.map(\.1)
            let storageIndices = fragmentSlots.1.map(\.1)
            let textureIndices = fragmentTextureSlots.map(\.1)
            let samplerIndices = fragmentSamplerSlots.map(\.1)
            guard Set(uniformIndices).count == uniformIndices.count,
                  uniformIndices.allSatisfy({ (Int(ComputeShaderArtifact.uniformBindingBase)..<Int(ComputeShaderArtifact.uniformBindingBase + ComputeShaderArtifact.uniformBindingCapacity)).contains($0) }),
                  Set(storageIndices).count == storageIndices.count,
                  storageIndices.allSatisfy({ (Int(RenderStageArtifact.metalStorageBufferBase)..<Int(RenderStageArtifact.metalStorageLengthBufferIndex)).contains($0) }),
                  Set(textureIndices).count == textureIndices.count,
                  fragmentSampledTextureSlots.allSatisfy({ $0.1 >= 0 && $0.1 < Int(ComputeShaderArtifact.sampledTextureBindingCapacity) }),
                  fragmentStorageTextureSlots.allSatisfy({ $0.1 >= Int(ComputeShaderArtifact.sampledTextureBindingCapacity) && $0.1 < Int(ComputeShaderArtifact.sampledTextureBindingCapacity + ComputeShaderArtifact.storageTextureBindingCapacity) }),
                  Set(samplerIndices).count == samplerIndices.count, samplerIndices.allSatisfy({ $0 >= 0 && $0 < Int(ComputeShaderArtifact.samplerBindingCapacity) }),
                  fragmentLengths.allSatisfy({ $0.1 > 0 }),
                  fragment.metalRuntimeLengthBuffer == (fragmentLengths.isEmpty ? nil : RenderStageArtifact.metalStorageLengthBufferIndex),
                  fragment.metalTextureAtomicGuardBufferIndex == (fragment.storageTextures.contains { $0.access.contains(.atomic) } ? RenderStageArtifact.metalTextureAtomicGuardSlot : nil) else {
                throw GPUError.validation("Invalid Metal mesh fragment resource layout")
            }
            let objectLengths = try runtimeLengthRequirements(task?.storageBindings ?? [])
            let meshLengths = try runtimeLengthRequirements(mesh.storageBindings)
            guard (objectLengths + meshLengths).allSatisfy({ $0.1 > 0 }) else {
                throw GPUError.validation("Invalid Metal mesh runtime array stride")
            }
            meshPipelines[pipeline.id] = MeshState(pipeline: state, depth: depth, writesDepth: fragment.writesDepth, objectThreads: task.map { size($0.workgroupSize) }
                                                   ?? MTLSize(width: 1, height: 1, depth: 1),
                                                   meshThreads: size(mesh.workgroupSize),
                                                   objectUniforms: objectSlots.0, objectStorage: objectSlots.1,
                                                   meshUniforms: meshSlots.0, meshStorage: meshSlots.1,
                                                   objectTextures: objectTextureSlots + objectStorageTextureSlots, meshTextures: meshTextureSlots + meshStorageTextureSlots, objectSamplers: objectSamplerSlots, meshSamplers: meshSamplerSlots,
                                                   fragmentBuffers: fragmentSlots.0 + fragmentSlots.1, fragmentTextures: fragmentTextureSlots, fragmentSamplers: fragmentSamplerSlots, fragmentLengths: fragmentLengths,
                                                   fragmentTextureAtomicGuardBufferIndex: fragment.metalTextureAtomicGuardBufferIndex,
                                                   objectLengths: objectLengths, meshLengths: meshLengths)
        } catch let error as GPUError { throw error }
        catch { throw GPUError.backendFailure("Mesh MSL compilation: \(error)") }
        #else
        throw GPUError.adapterUnavailable
        #endif
    }

    public func destroyMeshPipeline(id: ResourceID) {
        #if canImport(Metal)
        meshPipelines.removeValue(forKey: id)
        #endif
    }

    public func destroyRenderPipeline(id: ResourceID) {
        #if canImport(Metal)
        renderPipelines.removeValue(forKey: id)
        #endif
    }

    public func makeDiagnosticPipeline(owner: UInt64) throws -> ResourceID {
        try requireActive()
        #if canImport(Metal)
        guard owner > 0, nextPipeline < UInt64.max else { throw GPUError.validation("Invalid pipeline owner/serial") }
        if let existing = pipelines.keys.first(where: { $0.device == owner }) { return existing }
        let source = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void saga_diagnostic(device const uint* input [[buffer(0)]],
                                    device uint* output [[buffer(1)]],
                                    constant uint& count [[buffer(2)]],
                                    uint i [[thread_position_in_grid]]) {
            if (i < count) { output[i] = input[i] * 3u + 7u; }
        }
        """
        do {
            let pipeline: any MTLComputePipelineState
            if let cachedDiagnosticPipeline { pipeline = cachedDiagnosticPipeline }
            else {
                let library = try native.makeLibrary(source: source, options: nil)
                guard let function = library.makeFunction(name: "saga_diagnostic") else {
                    throw GPUError.backendFailure("Diagnostic entry point missing")
                }
                pipeline = try native.makeComputePipelineState(function: function)
                cachedDiagnosticPipeline = pipeline
            }
            nextPipeline += 1
            let id = ResourceID(device: owner, serial: nextPipeline)
            pipelines[id] = pipeline
            return id
        } catch { throw GPUError.backendFailure("Diagnostic MSL compilation: \(error)") }
        #else
        throw GPUError.adapterUnavailable
        #endif
    }
}
