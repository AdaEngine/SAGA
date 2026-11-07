#if os(WASI) && compiler(>=6.3)
import SAGATypes
import SAGABackend
import SAGAShaderIR
import SAGAWebGPU

private struct Packet {
    var bytes: [UInt8] = []
    init(_ opcode: UInt32) { add(opcode) }
    mutating func add<T: FixedWidthInteger>(_ value: T) {
        withUnsafeBytes(of: value.littleEndian) { bytes.append(contentsOf: $0) }
    }
    mutating func add(_ id: ResourceID) { add(id.device); add(id.serial) }
    mutating func add(_ text: String) {
        add(UInt32(text.utf8.count))
        bytes.append(contentsOf: text.utf8)
    }
    mutating func addRender(
        _ pipeline: RenderPipeline, target: ResourceID, targetMipLevel: UInt32, colorLoad: RenderColorLoadAction,
        count: UInt32, vertexBuffers: [VertexBufferBinding], firstVertex: UInt32, instances: DrawInstances,
        uniforms: [ComputeBindingSlot: ComputeBufferBinding], textures: [ComputeBindingSlot: ResourceID],
        samplers: [ComputeBindingSlot: ResourceID], depth: RenderDepthAttachment?, resolve: RenderResolveAttachment?,
        blendConstant: RenderClearColor, additionalColors: [UInt32: RenderColorAttachment], raster: RenderRasterState
    ) {
        add(pipeline.id); add(count)
        add(firstVertex); add(instances.count); add(instances.first)
        for component in blendConstant.channels { add(component.bitPattern) }
        add(UInt32(raster.viewport == nil ? 0 : 1))
        if let v = raster.viewport {
            for value in [v.x, v.y, v.width, v.height, v.minDepth, v.maxDepth] { add(Double(value).bitPattern) }
        }
        add(UInt32(raster.scissor == nil ? 0 : 1))
        if let s = raster.scissor { add(s.x); add(s.y); add(s.width); add(s.height) }
        add(UInt32(vertexBuffers.count))
        for binding in vertexBuffers { add(binding.buffer); add(binding.offset) }
        addBindings(uniforms)
        addResourceIDs(textures)
        addResourceIDs(samplers)
        add(UInt32(depth == nil ? 0 : 1))
        if let depth {
            add(depth.texture); add(depth.mipLevel)
            switch depth.loadAction {
            case .load: add(UInt32(0))
            case .readOnly: add(UInt32(2))
            case .clear(let value): add(UInt32(1)); add(value.bitPattern)
            }
            add(UInt32(depth.stencil == nil ? 0 : 1))
            if let stencil = depth.stencil {
                add(stencil.reference)
                switch stencil.loadAction {
                case .load: add(UInt32(0))
                case .readOnly: add(UInt32(2))
                case .clear(let value): add(UInt32(1)); add(value)
                }
            }
        }
        var colors = additionalColors
        colors[0] = .init(texture: target, mipLevel: targetMipLevel, loadAction: colorLoad, resolve: resolve)
        add(UInt32(colors.count))
        for (location, color) in colors.sorted(by: { $0.key < $1.key }) {
            add(location); add(color.texture); add(color.mipLevel)
            let clear = color.loadAction.clearColor
            add(UInt32(clear == nil ? 0 : 1))
            if let clear {
                for component in clear.channels { add(component.bitPattern) }
            }
            add(UInt32(color.resolve == nil ? 0 : 1))
            if let resolve = color.resolve { add(resolve.texture); add(resolve.mipLevel) }
        }
    }
    mutating func addResourceIDs(_ bindings: [ComputeBindingSlot: ResourceID]) {
        add(UInt32(bindings.count))
        for (slot, id) in bindings { add(slot.group); add(slot.binding); add(id) }
    }
    mutating func addBindings(_ bindings: [ComputeBindingSlot: ComputeBufferBinding]) {
        add(UInt32(bindings.count))
        for (slot, binding) in bindings {
            add(slot.group); add(slot.binding); add(binding.buffer); add(binding.offset)
            add(UInt32(binding.size == nil ? 0 : 1))
            if let size = binding.size { add(size) }
        }
    }
    func send() async throws -> [UInt8] { try await BrowserRPC.shared.send(bytes) }
}

struct WebGPUHost: BrowserGPUPresentationHost, SamplerBackendDevice, TextureViewBackendDevice {
    let capabilities: DeviceCapabilities
    let supportsPresentation: Bool

    func allocateBuffer(id: ResourceID, descriptor: BufferDescriptor) async throws {
        var packet = Packet(1)
        packet.add(id); packet.add(descriptor.size); packet.add(descriptor.usage.rawValue)
        _ = try await packet.send()
    }
    func destroyBuffer(id: ResourceID) async {
        var packet = Packet(2)
        packet.add(id)
        _ = try? await packet.send()
    }
    func allocateComputePipeline(_ pipeline: ComputePipeline, shader: ComputeShaderArtifact) async throws {
        guard case .text(.wgsl, let source) = shader.artifact else { throw GPUError.validation("Expected WGSL") }
        var packet = Packet(3)
        packet.add(pipeline.id); packet.add(shader.entryPoint); packet.add(source)
        packet.add(UInt32(shader.storageBindings.count + shader.uniformBindings.count +
                          shader.sampledTextures.count + shader.storageTextures.count + shader.samplerBindings.count))
        for binding in shader.storageBindings {
            packet.add(binding.binding.group); packet.add(binding.binding.binding)
            packet.add(binding.minimumByteSize); packet.add(UInt32(binding.isReadOnly ? 1 : 0))
        }
        for binding in shader.uniformBindings {
            packet.add(binding.binding.group); packet.add(binding.binding.binding)
            packet.add(binding.minimumByteSize); packet.add(UInt32(2))
        }
        for binding in shader.sampledTextures {
            packet.add(binding.binding.group); packet.add(binding.binding.binding)
            packet.add(UInt64(0)); packet.add(UInt32(3))
            packet.add(binding.requirement.sampleType.rawValue); packet.add(UInt32(binding.isMultisampled ? 1 : 0))
            packet.add(binding.requirement.dimension.rawValue)
        }
        for binding in shader.storageTextures {
            let format = try binding.textureFormat()
            packet.add(binding.binding.group); packet.add(binding.binding.binding)
            packet.add(UInt64(0)); packet.add(UInt32(5))
            packet.add(format.rawValue); packet.add(UInt32(binding.access.rawValue))
            packet.add(UInt32(binding.isArrayed ? 1 : 0))
        }
        for binding in shader.samplerBindings {
            packet.add(binding.binding.group); packet.add(binding.binding.binding)
            packet.add(UInt64(0)); packet.add(UInt32(4))
            packet.add(binding.requirement.webBindingType)
        }
        _ = try await packet.send()
    }
    func destroyComputePipeline(id: ResourceID) async {
        var packet = Packet(4)
        packet.add(id)
        _ = try? await packet.send()
    }
    func allocateSampler(id: ResourceID, descriptor: SamplerDescriptor) async throws {
        var packet = Packet(13)
        packet.add(id)
        packet.add(descriptor.addressModeU.rawValue); packet.add(descriptor.addressModeV.rawValue)
        packet.add(descriptor.minFilter.rawValue); packet.add(descriptor.magFilter.rawValue)
        packet.add(descriptor.mipmapFilter.rawValue)
        packet.add(descriptor.compare.map { $0.rawValue + 1 } ?? 0)
        _ = try await packet.send()
    }
    func destroySampler(id: ResourceID) async {
        var packet = Packet(14)
        packet.add(id)
        _ = try? await packet.send()
    }
    func allocateTexture(id: ResourceID, descriptor: TextureDescriptor) async throws {
        var packet = Packet(8)
        packet.add(id); packet.add(descriptor.width); packet.add(descriptor.height)
        packet.add(descriptor.format.rawValue); packet.add(descriptor.usage.rawValue); packet.add(descriptor.mipLevelCount)
        packet.add(descriptor.sampleCount)
        packet.add(descriptor.dimension == .d3 ? descriptor.depth : descriptor.arrayLayerCount)
        packet.add(descriptor.dimension.rawValue)
        _ = try await packet.send()
    }
    func destroyTexture(id: ResourceID) async {
        var packet = Packet(9)
        packet.add(id)
        _ = try? await packet.send()
    }
    func allocateTextureView(id: ResourceID, texture: ResourceID, mipLevels: Range<UInt32>) async throws {
        try await allocateTextureView(id: id, texture: texture, mipLevels: mipLevels, aspect: .all)
    }
    func allocateTextureView(id: ResourceID, texture: ResourceID, mipLevels: Range<UInt32>,
                             aspect: TextureViewAspect) async throws {
        try await allocateTextureView(id: id, texture: texture, descriptor: .init(
            baseMipLevel: mipLevels.lowerBound, mipLevelCount: mipLevels.upperBound - mipLevels.lowerBound, aspect: aspect))
    }
    func allocateTextureView(id: ResourceID, texture: ResourceID, descriptor: TextureViewDescriptor) async throws {
        var packet = Packet(15)
        packet.add(id); packet.add(texture); packet.add(descriptor.baseMipLevel)
        packet.add(UInt32(descriptor.mipLevelCount == nil ? 0 : 1))
        if let count = descriptor.mipLevelCount { packet.add(count) }
        packet.add(descriptor.aspect.rawValue)
        packet.add(descriptor.dimension.map { $0.rawValue + 1 } ?? 0)
        packet.add(descriptor.baseArrayLayer)
        packet.add(UInt32(descriptor.arrayLayerCount == nil ? 0 : 1))
        if let count = descriptor.arrayLayerCount { packet.add(count) }
        _ = try await packet.send()
    }
    func destroyTextureView(id: ResourceID) async {
        var packet = Packet(16)
        packet.add(id)
        _ = try? await packet.send()
    }
    func allocateRenderPipeline(
        _ pipeline: RenderPipeline, vertex: RenderStageArtifact, fragment: RenderStageArtifact
    ) async throws {
        guard case .text(.wgsl, let vertexSource) = vertex.artifact,
              case .text(.wgsl, let fragmentSource) = fragment.artifact else {
            throw GPUError.validation("Expected WGSL render stages")
        }
        var packet = Packet(10)
        packet.add(pipeline.id); packet.add(pipeline.sampleCount)
        packet.add(pipeline.sampleMask); packet.add(UInt32(pipeline.alphaToCoverageEnabled ? 1 : 0))
        let targets = pipeline.colorTargets
        packet.add(UInt32(targets.count))
        for (location, target) in targets.sorted(by: { $0.key < $1.key }) {
            packet.add(location); packet.add(target.format.rawValue)
            packet.add(pipeline.colorWriteMask(at: location).rawValue)
            packet.add(UInt32(target.colorState.blend == nil ? 0 : 1))
            if let blend = target.colorState.blend {
                for component in [blend.color, blend.alpha] {
                    packet.add(component.operation.rawValue)
                    packet.add(component.sourceFactor.rawValue)
                    packet.add(component.destinationFactor.rawValue)
                }
            }
        }
        packet.add(pipeline.primitive.frontFace.rawValue); packet.add(pipeline.primitive.cullMode.rawValue)
        packet.add(pipeline.primitive.topology.rawValue); packet.add(pipeline.primitive.stripIndexFormat?.rawValue ?? 0)
        packet.add(vertex.entryPoint); packet.add(vertexSource)
        packet.add(fragment.entryPoint); packet.add(fragmentSource)
        packet.add(UInt32(pipeline.vertexBuffers.count))
        for layout in pipeline.vertexBuffers {
            packet.add(layout.stride)
            packet.add(layout.stepMode.rawValue)
            packet.add(UInt32(layout.attributes.count))
            for attribute in layout.attributes {
                packet.add(attribute.location); packet.add(attribute.offset); packet.add(attribute.format.rawValue)
            }
        }
        packet.add(UInt32(pipeline.uniforms.count))
        for uniform in pipeline.uniforms {
            packet.add(uniform.slot.group); packet.add(uniform.slot.binding)
            packet.add(uniform.minimumByteSize); packet.add(uniform.visibility)
        }
        packet.add(UInt32(pipeline.sampledTextures.count))
        for texture in pipeline.sampledTextures {
            packet.add(texture.handle.slot.group); packet.add(texture.handle.slot.binding); packet.add(texture.handle.visibility)
            packet.add(texture.requirement.sampleType.rawValue)
            packet.add(UInt32(texture.requirement.isMultisampled ? 1 : 0))
            packet.add(texture.requirement.dimension.rawValue)
        }
        packet.add(UInt32(pipeline.storageTextures.count))
        for texture in pipeline.storageTextures {
            packet.add(texture.handle.slot.group); packet.add(texture.handle.slot.binding); packet.add(texture.handle.visibility)
            packet.add(texture.requirement.format.rawValue); packet.add(texture.requirement.access.rawValue)
            packet.add(UInt32(texture.requirement.isArrayed ? 1 : 0))
        }
        packet.add(UInt32(pipeline.samplerBindings.count))
        for sampler in pipeline.samplerBindings {
            packet.add(sampler.handle.slot.group); packet.add(sampler.handle.slot.binding); packet.add(sampler.handle.visibility)
            packet.add(sampler.requirement.webBindingType)
        }
        packet.add(UInt32(pipeline.depthState == nil ? 0 : 1))
        if let state = pipeline.depthState {
            packet.add(state.format.rawValue); packet.add(state.compare.rawValue)
            packet.add(UInt32(state.writeEnabled ? 1 : 0))
            packet.add(UInt32(bitPattern: state.bias.constant))
            packet.add(Double(state.bias.slopeScale).bitPattern)
            packet.add(Double(state.bias.clamp).bitPattern)
            packet.add(UInt32(state.stencil == nil ? 0 : 1))
            if let stencil = state.stencil {
                packet.add(stencil.readMask); packet.add(stencil.writeMask)
                for face in [stencil.front, stencil.back] {
                    packet.add(face.compare.rawValue); packet.add(face.fail.rawValue)
                    packet.add(face.depthFail.rawValue); packet.add(face.pass.rawValue)
                }
            }
        }
        _ = try await packet.send()
    }
    func destroyRenderPipeline(id: ResourceID) async {
        var packet = Packet(11)
        packet.add(id)
        _ = try? await packet.send()
    }
    func close(owner: UInt64) async {
        var packet = Packet(5)
        packet.add(owner)
        _ = try? await packet.send()
    }
    func present(_ submission: Submission, source: ResourceID) async throws {
        guard supportsPresentation, submission.resources.contains(source) else {
            throw GPUError.validation("Missing browser surface or unpinned presentation source")
        }
        var packet = Packet(12)
        packet.add(source)
        _ = try await packet.send()
    }
    func submit(_ submission: Submission) async throws {
        _ = try await Packet(7).send()
    }
    func submitCommands(_ submission: Submission, commands: [BufferCommand]) async throws -> [[UInt32]] {
        var packet = Packet(6)
        packet.add(UInt32(commands.count))
        for command in commands {
            switch command {
            case .write(let id, let offset, let words):
                let headerSize = 32 // opcode, resource identity, offset and word count
                guard words.count <= (BrowserRPC.packetLimit - packet.bytes.count - headerSize) / 4 else {
                    throw GPUError.validation("Browser upload exceeds its packet limit")
                }
                packet.add(UInt32(1)); packet.add(id); packet.add(offset)
                packet.add(UInt32(words.count))
                for word in words { packet.add(word) }
            case .dispatchCompute, .dispatchComputeRanges:
                let (pipeline, bindings, groups, textures, samplers) = command.computeDispatch!
                packet.add(UInt32(2)); packet.add(pipeline.id)
                packet.add(groups.x); packet.add(groups.y); packet.add(groups.z)
                packet.addBindings(bindings)
                packet.addResourceIDs(textures)
                packet.addResourceIDs(samplers)
            case .copy(let source, let destination, let size):
                packet.add(UInt32(3)); packet.add(source); packet.add(destination); packet.add(size)
            case .read(let id, let size):
                packet.add(UInt32(4)); packet.add(id); packet.add(size)
            case .dispatch:
                throw GPUError.unsupported("Metal diagnostic dispatch is unavailable on WebGPU")
            case .render(let pipeline, let target, let colorLoad, let vertexCount, let vertexBuffers, let firstVertex, let instances, let uniforms, let textures, let samplers, let targetMipLevel, let depth, let resolve, let blendConstant, let additionalColors, let raster):
                packet.add(UInt32(5))
                packet.addRender(pipeline, target: target, targetMipLevel: targetMipLevel, colorLoad: colorLoad, count: vertexCount, vertexBuffers: vertexBuffers,
                                 firstVertex: firstVertex, instances: instances, uniforms: uniforms,
                                 textures: textures, samplers: samplers, depth: depth, resolve: resolve,
                                 blendConstant: blendConstant, additionalColors: additionalColors, raster: raster)
            case .renderIndexed(let pipeline, let target, let colorLoad, let index, let draw, let vertexBuffers, let uniforms, let textures, let samplers, let targetMipLevel, let depth, let resolve, let blendConstant, let additionalColors, let raster):
                packet.add(UInt32(7))
                packet.addRender(pipeline, target: target, targetMipLevel: targetMipLevel, colorLoad: colorLoad, count: draw.indexCount, vertexBuffers: vertexBuffers,
                                 firstVertex: 0, instances: draw.instances, uniforms: uniforms,
                                 textures: textures, samplers: samplers, depth: depth, resolve: resolve,
                                 blendConstant: blendConstant, additionalColors: additionalColors, raster: raster)
                packet.add(index.buffer); packet.add(index.format.rawValue); packet.add(index.offset)
                packet.add(UInt32(index.size == nil ? 0 : 1))
                if let size = index.size { packet.add(size) }
                packet.add(draw.firstIndex); packet.add(draw.baseVertex)
            case .copyTextureToBuffer(let source, let destination, let bytesPerRow, let mipLevel, let aspect, let offset,
                                      let layer, let z, let depth):
                packet.add(UInt32(6)); packet.add(source); packet.add(destination); packet.add(bytesPerRow); packet.add(mipLevel)
                packet.add(aspect.rawValue); packet.add(offset); packet.add(layer + z); packet.add(depth)
            case .copyTextureToTexture(let source, let destination, let region, let origin):
                packet.add(UInt32(9)); packet.add(source); packet.add(destination)
                packet.add(region.x); packet.add(region.y); packet.add(region.arrayLayer + region.z); packet.add(region.mipLevel)
                packet.add(origin.x); packet.add(origin.y); packet.add(origin.arrayLayer + origin.z); packet.add(origin.mipLevel)
                packet.add(region.width); packet.add(region.height); packet.add(region.depth)
            case .copyBufferToTexture(let source, let layout, let destination, let region, let aspect):
                packet.add(UInt32(8)); packet.add(source); packet.add(destination); packet.add(layout.offset)
                packet.add(UInt32(layout.bytesPerRow == nil ? 0 : 1))
                if let stride = layout.bytesPerRow { packet.add(stride) }
                packet.add(region.x); packet.add(region.y); packet.add(region.arrayLayer + region.z)
                packet.add(region.width); packet.add(region.height); packet.add(region.depth)
                packet.add(region.mipLevel); packet.add(aspect.rawValue)
            }
            guard packet.bytes.count <= BrowserRPC.packetLimit else {
                throw GPUError.validation("Browser submission packet exceeds its limit")
            }
        }
        let result = try await packet.send()
        var offset = 0
        func word() throws -> UInt32 {
            guard offset + 4 <= result.count else { throw GPUError.backendFailure("Truncated browser readback") }
            defer { offset += 4 }
            return result.withUnsafeBytes {
                UInt32(littleEndian: $0.loadUnaligned(fromByteOffset: offset, as: UInt32.self))
            }
        }
        let count = try word()
        var reads: [[UInt32]] = []
        for _ in 0..<count {
            let length = try word()
            guard length <= (result.count - offset) / 4 else {
                throw GPUError.backendFailure("Invalid browser readback length")
            }
            reads.append(try (0..<length).map { _ in try word() })
        }
        guard offset == result.count else { throw GPUError.backendFailure("Trailing browser readback bytes") }
        return reads
    }
}

private extension SamplerRequirement {
    var webBindingType: UInt32 {
        if isComparison { return 1 }
        return requiresNonFiltering ? 2 : 0
    }
}
#endif
