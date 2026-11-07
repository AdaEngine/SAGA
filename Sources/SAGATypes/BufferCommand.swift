/// Initial queue commands. The diagnostic dispatch has a fixed two-buffer ABI.
public enum BufferCommand: Sendable, Equatable {
    case write(buffer: ResourceID, offset: UInt64, words: [UInt32])
    case dispatch(pipeline: ResourceID, input: ResourceID, output: ResourceID, count: UInt32)
    case dispatchCompute(pipeline: ComputePipeline, bindings: [ComputeBindingSlot: ResourceID], workgroups: DispatchWorkgroups)
    case render(pipeline: RenderPipeline, target: ResourceID, clear: RenderClearColor,
                vertexCount: UInt32, vertexBuffers: [VertexBufferBinding] = [])
    case renderIndexed(pipeline: RenderPipeline, target: ResourceID, clear: RenderClearColor,
                       indexBuffer: IndexBufferBinding, draw: IndexedDraw, vertexBuffers: [VertexBufferBinding] = [])
    case copyTextureToBuffer(source: ResourceID, destination: ResourceID, bytesPerRow: UInt32)
    case copy(source: ResourceID, destination: ResourceID, byteCount: UInt64)
    case read(buffer: ResourceID, byteCount: UInt64)

    public var resources: [ResourceID] {
        switch self {
        case .write(let id, _, _), .read(let id, _): [id]
        case .render(_, let target, _, _, let bindings): [target] + bindings.map(\.buffer)
        case .renderIndexed(_, let target, _, let index, _, let bindings):
            [target, index.buffer] + bindings.map(\.buffer)
        case .dispatch(_, let input, let output, _): [input, output]
        case .dispatchCompute(_, let bindings, _): Array(bindings.values)
        case .copy(let source, let destination, _), .copyTextureToBuffer(let source, let destination, _):
            [source, destination]
        }
    }

    public func validate(
        device: UInt64, descriptors: [ResourceID: BufferDescriptor],
        textures: [ResourceID: TextureDescriptor] = [:]
    ) throws {
        func require(_ id: ResourceID, _ usage: BufferUsage, _ size: UInt64, offset: UInt64 = 0) throws {
            guard id.device == device, let descriptor = descriptors[id], descriptor.usage.contains(usage),
                  size > 0, size % 4 == 0, offset % 4 == 0,
                  offset <= descriptor.size, size <= descriptor.size - offset else {
                throw GPUError.validation("Command has invalid resource, usage, size or alignment")
            }
        }
        func requireRender(
            _ pipeline: RenderPipeline, _ target: ResourceID, _ clear: RenderClearColor,
            _ bindings: [VertexBufferBinding], vertexCount: UInt32
        ) throws {
            guard pipeline.id.device == device, target.device == device,
                  let texture = textures[target], texture.usage.contains(.renderAttachment),
                  texture.format == pipeline.colorFormat, clear.isFinite else {
                throw GPUError.validation("Invalid render pipeline, color attachment or clear value")
            }
            guard bindings.count == pipeline.vertexBuffers.count else {
                throw GPUError.validation("Vertex buffers do not match pipeline slots")
            }
            for (binding, layout) in zip(bindings, pipeline.vertexBuffers) {
                let required = vertexCount == 0 ? 0 :
                    UInt64(vertexCount - 1) * UInt64(layout.stride) + layout.attributeEnd
                guard binding.buffer.device == device, let buffer = descriptors[binding.buffer],
                      buffer.usage.contains(.vertex), binding.offset % 4 == 0,
                      binding.offset <= buffer.size, required <= buffer.size - binding.offset else {
                    throw GPUError.validation("Invalid vertex buffer, usage, offset or fetch range")
                }
            }
        }
        switch self {
        case .write(let id, let offset, let words):
            try require(id, .copyDestination, UInt64(words.count) * 4, offset: offset)
        case .dispatch(let pipeline, let input, let output, let count):
            guard pipeline.device == device, pipeline.serial > 0, input != output else {
                throw GPUError.validation("Invalid pipeline identity or aliased storage buffers")
            }
            try require(input, .storage, UInt64(count) * 4)
            try require(output, .storage, UInt64(count) * 4)
        case .dispatchCompute(let pipeline, let bindings, let workgroups):
            guard pipeline.id.device == device else { throw GPUError.validation("Foreign compute pipeline") }
            guard workgroups.x <= 65_535, workgroups.y <= 65_535, workgroups.z <= 65_535 else {
                throw GPUError.validation("Workgroup count exceeds the supported limit")
            }
            guard bindings.count == pipeline.storageBindings.count else {
                throw GPUError.validation("Compute bindings do not match the pipeline layout")
            }
            var used: [ResourceID: Bool] = [:]
            for requirement in pipeline.storageBindings {
                guard let buffer = bindings[requirement.slot] else {
                    throw GPUError.validation("Compute bindings do not match the pipeline layout")
                }
                try require(buffer, .storage, requirement.minimumByteSize)
                if let previousReadOnly = used[buffer], !previousReadOnly || !requirement.isReadOnly {
                    throw GPUError.validation("Aliased writable compute buffers")
                }
                used[buffer] = requirement.isReadOnly
            }
        case .copy(let source, let destination, let size):
            guard source != destination else { throw GPUError.validation("Aliased copy") }
            try require(source, .copySource, size)
            try require(destination, .copyDestination, size)
        case .render(let pipeline, let target, let clear, let count, let bindings):
            try requireRender(pipeline, target, clear, bindings, vertexCount: count)
        case .renderIndexed(let pipeline, let target, let clear, let index, let draw, let bindings):
            try requireRender(pipeline, target, clear, bindings, vertexCount: 0)
            let stride = UInt64(index.format.byteSize)
            guard index.buffer.device == device, let buffer = descriptors[index.buffer],
                  buffer.usage.contains(.index), index.offset % stride == 0, index.offset <= buffer.size else {
                throw GPUError.validation("Invalid index buffer, usage or offset")
            }
            let remaining = buffer.size - index.offset
            let size = index.size ?? remaining
            let end = UInt64(draw.firstIndex) + UInt64(draw.indexCount)
            guard size <= remaining, end <= size / stride else {
                throw GPUError.validation("Indexed draw exceeds index buffer range")
            }
        case .copyTextureToBuffer(let source, let destination, let bytesPerRow):
            guard source.device == device, let texture = textures[source], texture.usage.contains(.copySource),
                  texture.width > 0, texture.height > 0 else {
                throw GPUError.validation("Invalid texture copy source")
            }
            let rowBytes = UInt64(texture.width) * 4
            guard UInt64(bytesPerRow) >= rowBytes, bytesPerRow % 256 == 0 else {
                throw GPUError.validation("Invalid texture copy row stride")
            }
            let size = UInt64(texture.height - 1) * UInt64(bytesPerRow) + rowBytes
            try require(destination, .copyDestination, size)
        case .read(let id, let size):
            try require(id, .mapRead, size)
        }
    }
}
