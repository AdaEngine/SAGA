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
    mutating func add(_ index: IndexBufferBinding) {
        add(index.buffer); add(index.format.rawValue); add(index.offset)
        add(UInt32(index.size == nil ? 0 : 1))
        if let size = index.size { add(size) }
    }
    mutating func add(_ text: String) {
        add(UInt32(text.utf8.count))
        bytes.append(contentsOf: text.utf8)
    }
    mutating func addRender(
        _ pipeline: RenderPipeline, target: ResourceID?, targetMipLevel: UInt32, targetDepthSlice: UInt32?,
        colorLoad: RenderColorLoadAction, colorStore: RenderStoreAction,
        count: UInt32? = nil, vertexBuffers: [VertexBufferBinding], firstVertex: UInt32 = 0, instances: DrawInstances = .init(),
        uniforms: [ComputeBindingSlot: ComputeBufferBinding], storageBuffers: [ComputeBindingSlot: ComputeBufferBinding],
        textures: [ComputeBindingSlot: ResourceID],
        samplers: [ComputeBindingSlot: ResourceID], depth: RenderDepthAttachment?, resolve: RenderResolveAttachment?,
        blendConstant: RenderClearColor, additionalColors: [UInt32: RenderColorAttachment], raster: RenderRasterState,
        immediates: [UInt32]
    ) {
        add(pipeline.id)
        add(UInt32(immediates.count))
        for word in immediates { add(word) }
        if let count {
            add(count); add(firstVertex); add(instances.count); add(instances.first)
        }
        for component in blendConstant.channels { add(component.bitPattern) }
        add(UInt32(raster.viewport == nil ? 0 : 1))
        if let v = raster.viewport {
            for value in [v.x, v.y, v.width, v.height, v.minDepth, v.maxDepth] { add(Double(value).bitPattern) }
        }
        add(UInt32(raster.scissor == nil ? 0 : 1))
        if let s = raster.scissor { add(s.x); add(s.y); add(s.width); add(s.height) }
        add(UInt32(vertexBuffers.count))
        for (index, binding) in vertexBuffers.enumerated() {
            add(binding.slot ?? UInt32(index)); add(binding.buffer); add(binding.offset)
        }
        addBindings(uniforms.merging(storageBuffers) { uniform, _ in uniform }, remapped: pipeline.webBindingIndices)
        addResourceIDs(textures, remapped: pipeline.webBindingIndices)
        addResourceIDs(samplers, remapped: pipeline.webBindingIndices)
        add(UInt32(depth == nil ? 0 : 1))
        if let depth {
            add(depth.texture); add(depth.mipLevel)
            switch depth.loadAction {
            case .load: add(UInt32(0))
            case .readOnly: add(UInt32(2))
            case .unused: add(UInt32(3))
            case .clear(let value): add(UInt32(1)); add(value.bitPattern)
            }
            add(depth.storeAction.rawValue)
            add(UInt32(depth.stencil == nil ? 0 : 1))
            if let stencil = depth.stencil {
                add(stencil.reference)
                switch stencil.loadAction {
                case .load: add(UInt32(0))
                case .readOnly: add(UInt32(2))
                case .clear(let value): add(UInt32(1)); add(value)
                }
                add(stencil.storeAction.rawValue)
            }
        }
        var colors = additionalColors
        if let target {
            colors[0] = .init(texture: target, mipLevel: targetMipLevel, loadAction: colorLoad, resolve: resolve,
                              depthSlice: targetDepthSlice, storeAction: colorStore)
        }
        add(UInt32(colors.count))
        for (location, color) in colors.sorted(by: { $0.key < $1.key }) {
            add(location); add(color.texture); add(color.mipLevel)
            add(UInt32(color.depthSlice == nil ? 0 : 1))
            if let depthSlice = color.depthSlice { add(depthSlice) }
            let clear = color.loadAction.clearColor
            add(UInt32(clear == nil ? 0 : 1))
            if let clear {
                for component in clear.channels { add(component.bitPattern) }
            }
            add(color.storeAction.rawValue)
            add(UInt32(color.resolve == nil ? 0 : 1))
            if let resolve = color.resolve { add(resolve.texture); add(resolve.mipLevel) }
        }
    }
    mutating func add(_ slot: ComputeBindingSlot, remapped: [ComputeBindingSlot: UInt32]) {
        add(slot.group); add(remapped[slot] ?? slot.binding)
    }
    mutating func addResourceIDs(_ bindings: [ComputeBindingSlot: ResourceID], remapped: [ComputeBindingSlot: UInt32]) {
        add(UInt32(bindings.count))
        for (slot, id) in bindings { add(slot, remapped: remapped); add(id) }
    }
    mutating func addBindings(_ bindings: [ComputeBindingSlot: ComputeBufferBinding], remapped: [ComputeBindingSlot: UInt32]) {
        add(UInt32(bindings.count))
        for (slot, binding) in bindings {
            add(slot, remapped: remapped); add(binding.buffer); add(binding.offset)
            add(UInt32(binding.size == nil ? 0 : 1))
            if let size = binding.size { add(size) }
        }
    }
    func send() async throws -> [UInt8] { try await BrowserRPC.shared.send(bytes) }
}

struct WebGPUHost: BrowserGPUPresentationHost, SamplerBackendDevice, TextureViewBackendDevice, QuerySetBackendDevice,
                   BufferMappingBackendDevice, ExternalTextureBackendDevice {
    let capabilities: DeviceCapabilities
    let supportsPresentation: Bool

    func allocateBuffer(id: ResourceID, descriptor: BufferDescriptor) async throws {
        var packet = Packet(1)
        packet.add(id); packet.add(descriptor.size); packet.add(descriptor.usage.rawValue)
        packet.add(UInt32(descriptor.mappedAtCreation ? 1 : 0))
        _ = try await packet.send()
    }
    func destroyBuffer(id: ResourceID) async {
        var packet = Packet(2)
        packet.add(id)
        _ = try? await packet.send()
    }
    func allocateExternalTexture(id: ResourceID, descriptor: ExternalTextureDescriptor) async throws {
        var packet = Packet(26)
        packet.add(id); packet.add(descriptor.videoFrame); packet.add(descriptor.colorSpace.rawValue)
        _ = try await packet.send()
    }
    func destroyExternalTexture(id: ResourceID) async {
        var packet = Packet(27)
        packet.add(id)
        _ = try? await packet.send()
    }
    func beginBufferMapping(id: ResourceID, mode: BufferMapMode, offset: UInt64, size: UInt64) async throws {
        var packet = Packet(19)
        packet.add(id); packet.add(mode.rawValue); packet.add(offset); packet.add(size)
        _ = try await packet.send()
    }
    func readMappedBytes(id: ResourceID, offset: UInt64, size: UInt64) async throws -> [UInt8] {
        guard let count = Int(exactly: size) else { throw GPUError.validation("Mapped readback is too large") }
        var result: [UInt8] = []
        result.reserveCapacity(count)
        while result.count < count {
            let length = min(count - result.count, BrowserRPC.packetLimit - 32)
            var packet = Packet(20)
            packet.add(id); packet.add(offset + UInt64(result.count)); packet.add(UInt32(length))
            let reply = try await packet.send()
            guard reply.count == length else { throw GPUError.backendFailure("Truncated browser mapped readback") }
            result.append(contentsOf: reply)
        }
        return result
    }
    func writeMappedBytes(id: ResourceID, offset: UInt64, bytes: [UInt8]) async throws {
        var start = 0
        while start < bytes.count {
            let length = min(bytes.count - start, BrowserRPC.packetLimit - 32)
            var packet = Packet(21)
            packet.add(id); packet.add(offset + UInt64(start)); packet.add(UInt32(length))
            packet.bytes.append(contentsOf: bytes[start..<(start + length)])
            _ = try await packet.send()
            start += length
        }
    }
    func endBufferMapping(id: ResourceID) async throws {
        var packet = Packet(22)
        packet.add(id)
        _ = try await packet.send()
    }
    func allocateQuerySet(id: ResourceID, descriptor: QuerySetDescriptor) async throws {
        try descriptor.validate(against: capabilities)
        var packet = Packet(17)
        packet.add(id); packet.add(descriptor.type.rawValue); packet.add(descriptor.count); packet.add(descriptor.label)
        _ = try await packet.send()
    }
    func destroyQuerySet(id: ResourceID) async {
        var packet = Packet(18)
        packet.add(id)
        _ = try? await packet.send()
    }
    func allocateComputePipeline(_ pipeline: ComputePipeline, shader: ComputeShaderArtifact) async throws {
        guard case .text(.wgsl, let source) = shader.artifact else { throw GPUError.validation("Expected WGSL") }
        var packet = Packet(3)
        packet.add(pipeline.id); packet.add(shader.entryPoint); packet.add(source)
        packet.add(UInt32(pipeline.immediateByteSize))
        packet.add(UInt32(pipeline.bufferBindings.count +
                          pipeline.textureBindings.count + pipeline.storageTextureBindings.count + pipeline.samplerBindings.count +
                          pipeline.externalTextureBindings.count))
        for binding in pipeline.bufferBindings {
            packet.add(binding.slot, remapped: pipeline.webBindingIndices)
            packet.add(binding.minimumByteSize)
            packet.add(UInt32(binding.usage == .uniform ? 2 : binding.isReadOnly ? 1 : 0))
        }
        for binding in pipeline.textureBindings {
            packet.add(binding.slot, remapped: pipeline.webBindingIndices)
            packet.add(UInt64(0)); packet.add(UInt32(3))
            packet.add(binding.webSampleType); packet.add(UInt32(binding.isMultisampled ? 1 : 0))
            packet.add(binding.dimension.rawValue)
        }
        for requirement in pipeline.storageTextureBindings {
            packet.add(requirement.slot, remapped: pipeline.webBindingIndices)
            packet.add(UInt64(0)); packet.add(UInt32(5))
            packet.add(requirement.format.rawValue); packet.add(requirement.access.rawValue)
            packet.add(requirement.dimension.rawValue)
        }
        for binding in pipeline.samplerBindings {
            packet.add(binding.slot, remapped: pipeline.webBindingIndices)
            packet.add(UInt64(0)); packet.add(UInt32(4))
            packet.add(binding.webBindingType)
        }
        for slot in pipeline.externalTextureBindings {
            packet.add(slot, remapped: pipeline.webBindingIndices)
            packet.add(UInt64(0)); packet.add(UInt32(6))
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
        packet.add(descriptor.addressModeW.rawValue)
        packet.add(descriptor.minFilter.rawValue); packet.add(descriptor.magFilter.rawValue)
        packet.add(descriptor.mipmapFilter.rawValue)
        packet.add(Double(descriptor.lodMinClamp).bitPattern); packet.add(Double(descriptor.lodMaxClamp).bitPattern)
        packet.add(descriptor.compare.map { $0.rawValue + 1 } ?? 0)
        packet.add(UInt32(descriptor.maxAnisotropy))
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
        try await allocateRenderPipeline(pipeline, vertex: vertex, fragment: Optional(fragment))
    }

    func allocateRenderPipeline(
        _ pipeline: RenderPipeline, vertex: RenderStageArtifact, fragment: RenderStageArtifact?
    ) async throws {
        guard case .text(.wgsl, let vertexSource) = vertex.artifact else {
            throw GPUError.validation("Expected WGSL render stages")
        }
        var packet = Packet(10)
        packet.add(pipeline.id); packet.add(UInt32(pipeline.immediateByteSize)); packet.add(pipeline.sampleCount)
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
        packet.add(UInt32(pipeline.primitive.unclippedDepth ? 1 : 0))
        packet.add(vertex.entryPoint); packet.add(vertexSource)
        packet.add(UInt32(fragment == nil ? 0 : 1))
        if let fragment {
            guard case .text(.wgsl, let source) = fragment.artifact else {
                throw GPUError.validation("Expected WGSL render stages")
            }
            packet.add(fragment.entryPoint); packet.add(source)
        }
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
            packet.add(uniform.slot, remapped: pipeline.webBindingIndices)
            packet.add(uniform.minimumByteSize); packet.add(uniform.visibility)
        }
        packet.add(UInt32(pipeline.storageBuffers.count))
        for buffer in pipeline.storageBuffers {
            packet.add(buffer.handle.slot, remapped: pipeline.webBindingIndices)
            packet.add(buffer.minimumByteSize); packet.add(buffer.handle.visibility)
            packet.add(UInt32(buffer.isReadOnly ? 1 : 0))
        }
        packet.add(UInt32(pipeline.sampledTextures.count))
        for texture in pipeline.sampledTextures {
            packet.add(texture.handle.slot, remapped: pipeline.webBindingIndices); packet.add(texture.handle.visibility)
            packet.add(texture.requirement.webSampleType)
            packet.add(UInt32(texture.requirement.isMultisampled ? 1 : 0))
            packet.add(texture.requirement.dimension.rawValue)
        }
        packet.add(UInt32(pipeline.storageTextures.count))
        for texture in pipeline.storageTextures {
            packet.add(texture.handle.slot, remapped: pipeline.webBindingIndices); packet.add(texture.handle.visibility)
            packet.add(texture.requirement.format.rawValue); packet.add(texture.requirement.access.rawValue)
            packet.add(texture.requirement.dimension.rawValue)
        }
        packet.add(UInt32(pipeline.samplerBindings.count))
        for sampler in pipeline.samplerBindings {
            packet.add(sampler.handle.slot, remapped: pipeline.webBindingIndices); packet.add(sampler.handle.visibility)
            packet.add(sampler.requirement.webBindingType)
        }
        packet.add(UInt32(pipeline.externalTextureBindings.count))
        for external in pipeline.externalTextureBindings {
            packet.add(external.slot, remapped: pipeline.webBindingIndices); packet.add(external.visibility)
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
        for recorded in commands {
            let (command, timestamps, occlusion) = recorded.recordedPass
            if let timestamps {
                packet.add(UInt32(12)); packet.add(timestamps.querySet)
                packet.add(timestamps.beginningOfPassWriteIndex ?? .max)
                packet.add(timestamps.endOfPassWriteIndex ?? .max)
            }
            if let occlusion {
                packet.add(UInt32(14)); packet.add(occlusion.querySet); packet.add(occlusion.queryIndex)
            }
            switch command {
            case .buildAccelerationStructure, .buildTopLevelAccelerationStructure:
                throw GPUError.unsupported("Browser acceleration structures are not supported")
            case .timestamped, .occlusionQueried:
                throw GPUError.validation("Invalid query pass nesting")
            case .resolveQuerySet(let set, let first, let count, let destination, let offset):
                packet.add(UInt32(13)); packet.add(set); packet.add(first); packet.add(count)
                packet.add(destination); packet.add(offset)
            case .write(let id, let offset, let words):
                packet.add(UInt32(1)); packet.add(id); packet.add(offset)
                packet.add(UInt32(words.count))
                for word in words { packet.add(word) }
            case .writeBytes(let id, let offset, let bytes, let dataOffset, let byteCount):
                let source = try BufferCommand.uploadBytes(bytes, dataOffset: dataOffset, byteCount: byteCount)
                packet.add(UInt32(15)); packet.add(id); packet.add(offset)
                packet.add(UInt32(source.count))
                packet.bytes.append(contentsOf: source)
            case .clearBuffer(let id, let offset, let size):
                packet.add(UInt32(16)); packet.add(id); packet.add(offset)
                packet.add(UInt32(size == nil ? 0 : 1))
                if let size { packet.add(size) }
            case .dispatchCompute, .dispatchComputeRanges, .dispatchComputeIndirect:
                let (pipeline, bindings, source, textures, samplers, immediates, _) = command.computeDispatch!
                switch source {
                case .direct(let groups):
                    packet.add(UInt32(2)); packet.add(pipeline.id)
                    packet.add(groups.x); packet.add(groups.y); packet.add(groups.z)
                case .indirect(let buffer, let offset):
                    packet.add(UInt32(10)); packet.add(pipeline.id)
                    packet.add(buffer); packet.add(offset)
                }
                packet.addBindings(bindings, remapped: pipeline.webBindingIndices)
                packet.addResourceIDs(textures, remapped: pipeline.webBindingIndices)
                packet.addResourceIDs(samplers, remapped: pipeline.webBindingIndices)
                packet.add(UInt32(immediates.count))
                for word in immediates { packet.add(word) }
            case .copy(let source, let sourceOffset, let destination, let destinationOffset, let size):
                packet.add(UInt32(3)); packet.add(source); packet.add(sourceOffset)
                packet.add(destination); packet.add(destinationOffset)
                packet.add(UInt32(size == nil ? 0 : 1))
                if let size { packet.add(size) }
            case .read(let id, let size):
                packet.add(UInt32(4)); packet.add(id); packet.add(size)
            case .dispatch:
                throw GPUError.unsupported("Metal diagnostic dispatch is unavailable on WebGPU")
            case .render(let pipeline, let target, let colorLoad, let vertexCount, let vertexBuffers, let firstVertex, let instances, let uniforms, let textures, let samplers, let targetMipLevel, let targetDepthSlice, let depth, let resolve, let blendConstant, let additionalColors, let raster, let storageBuffers, let immediates, let colorStore):
                packet.add(UInt32(5))
                packet.addRender(pipeline, target: target, targetMipLevel: targetMipLevel, targetDepthSlice: targetDepthSlice,
                                 colorLoad: colorLoad, colorStore: colorStore, count: vertexCount, vertexBuffers: vertexBuffers,
                                 firstVertex: firstVertex, instances: instances, uniforms: uniforms, storageBuffers: storageBuffers,
                                 textures: textures, samplers: samplers, depth: depth, resolve: resolve,
                                 blendConstant: blendConstant, additionalColors: additionalColors, raster: raster, immediates: immediates)
            case .renderIndexed(let pipeline, let target, let colorLoad, let index, let draw, let vertexBuffers, let uniforms, let textures, let samplers, let targetMipLevel, let targetDepthSlice, let depth, let resolve, let blendConstant, let additionalColors, let raster, let storageBuffers, let immediates, let colorStore):
                packet.add(UInt32(7))
                packet.addRender(pipeline, target: target, targetMipLevel: targetMipLevel, targetDepthSlice: targetDepthSlice,
                                 colorLoad: colorLoad, colorStore: colorStore, count: draw.indexCount, vertexBuffers: vertexBuffers,
                                 firstVertex: 0, instances: draw.instances, uniforms: uniforms, storageBuffers: storageBuffers,
                                 textures: textures, samplers: samplers, depth: depth, resolve: resolve,
                                 blendConstant: blendConstant, additionalColors: additionalColors, raster: raster, immediates: immediates)
                packet.add(index)
                packet.add(draw.firstIndex); packet.add(draw.baseVertex)
            case .drawMesh:
                throw GPUError.unsupported("WebGPU host does not implement mesh draws")
            case .renderIndirect(let pipeline, let target, let colorLoad, let arguments, let offset, let index, let vertexBuffers, let uniforms, let textures, let samplers, let targetMipLevel, let targetDepthSlice, let depth, let resolve, let blendConstant, let additionalColors, let raster, let storageBuffers, let immediates, let colorStore):
                packet.add(UInt32(11)); packet.add(arguments); packet.add(offset)
                packet.add(UInt32(index == nil ? 0 : 1))
                packet.addRender(pipeline, target: target, targetMipLevel: targetMipLevel, targetDepthSlice: targetDepthSlice,
                                 colorLoad: colorLoad, colorStore: colorStore, vertexBuffers: vertexBuffers, uniforms: uniforms, storageBuffers: storageBuffers,
                                 textures: textures, samplers: samplers, depth: depth, resolve: resolve,
                                 blendConstant: blendConstant, additionalColors: additionalColors, raster: raster, immediates: immediates)
                if let index { packet.add(index) }
            case .copyTextureToBuffer(let source, let destination, let bytesPerRow, let mipLevel, let aspect, let offset,
                                      let layer, let z, let depth, let rowsPerImage):
                packet.add(UInt32(6)); packet.add(source); packet.add(destination); packet.add(bytesPerRow); packet.add(mipLevel)
                packet.add(aspect.rawValue); packet.add(offset); packet.add(layer + z); packet.add(depth)
                packet.add(UInt32(rowsPerImage == nil ? 0 : 1))
                if let rowsPerImage { packet.add(rowsPerImage) }
            case .copyTextureRegionToBuffer(let source, let destination, let layout, let region, let aspect):
                packet.add(UInt32(18)); packet.add(source); packet.add(destination); packet.add(layout.offset)
                packet.add(UInt32(layout.bytesPerRow == nil ? 0 : 1))
                if let stride = layout.bytesPerRow { packet.add(stride) }
                packet.add(UInt32(layout.rowsPerImage == nil ? 0 : 1))
                if let rows = layout.rowsPerImage { packet.add(rows) }
                packet.add(region.x); packet.add(region.y); packet.add(region.arrayLayer + region.z)
                packet.add(region.width); packet.add(region.height); packet.add(region.depth)
                packet.add(region.mipLevel); packet.add(aspect.rawValue)
            case .copyTextureToTexture(let source, let destination, let region, let origin):
                packet.add(UInt32(9)); packet.add(source); packet.add(destination)
                packet.add(region.x); packet.add(region.y); packet.add(region.arrayLayer + region.z); packet.add(region.mipLevel)
                packet.add(origin.x); packet.add(origin.y); packet.add(origin.arrayLayer + origin.z); packet.add(origin.mipLevel)
                packet.add(region.width); packet.add(region.height); packet.add(region.depth)
            case .copyBufferToTexture(let source, let layout, let destination, let region, let aspect):
                packet.add(UInt32(8)); packet.add(source); packet.add(destination); packet.add(layout.offset)
                packet.add(UInt32(layout.bytesPerRow == nil ? 0 : 1))
                if let stride = layout.bytesPerRow { packet.add(stride) }
                packet.add(UInt32(layout.rowsPerImage == nil ? 0 : 1))
                if let rows = layout.rowsPerImage { packet.add(rows) }
                packet.add(region.x); packet.add(region.y); packet.add(region.arrayLayer + region.z)
                packet.add(region.width); packet.add(region.height); packet.add(region.depth)
                packet.add(region.mipLevel); packet.add(aspect.rawValue)
            case .writeTexture(let destination, let data, let layout, let region, let aspect):
                let source = data.dropFirst(region.isEmpty ? data.count : Int(layout.offset))
                packet.add(UInt32(17)); packet.add(destination); packet.add(region.mipLevel); packet.add(aspect.rawValue)
                packet.add(region.x); packet.add(region.y); packet.add(region.arrayLayer + region.z)
                packet.add(region.width); packet.add(region.height); packet.add(region.depth)
                packet.add(UInt32(layout.bytesPerRow == nil ? 0 : 1))
                if let stride = layout.bytesPerRow { packet.add(stride) }
                packet.add(UInt32(layout.rowsPerImage == nil ? 0 : 1))
                if let rows = layout.rowsPerImage { packet.add(rows) }
                packet.add(UInt32(source.count))
                packet.bytes.append(contentsOf: source)
            }
        }
        let result = try await sendSubmission(packet, submission: submission)
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

    private func sendSubmission(_ packet: Packet, submission: Submission) async throws -> [UInt8] {
        if packet.bytes.count <= BrowserRPC.packetLimit { return try await packet.send() }
        guard let owner = submission.resources.first?.device else {
            throw GPUError.validation("A staged browser submission requires a resource owner")
        }
        let key = ResourceID(device: owner, serial: submission.sequence)
        let chunkCapacity = BrowserRPC.packetLimit - 40
        do {
            var start = 0
            while start < packet.bytes.count {
                let length = min(chunkCapacity, packet.bytes.count - start)
                var chunk = Packet(23)
                chunk.add(key); chunk.add(UInt64(packet.bytes.count)); chunk.add(UInt64(start)); chunk.add(UInt32(length))
                chunk.bytes.append(contentsOf: packet.bytes[start..<(start + length)])
                _ = try await chunk.send()
                start += length
            }
            var execute = Packet(24)
            execute.add(key)
            return try await execute.send()
        } catch {
            var discard = Packet(25)
            discard.add(key)
            _ = try? await discard.send()
            throw error
        }
    }
}

private extension SamplerRequirement {
    var webBindingType: UInt32 {
        if isComparison { return 1 }
        return requiresNonFiltering ? 2 : 0
    }
}
#endif
