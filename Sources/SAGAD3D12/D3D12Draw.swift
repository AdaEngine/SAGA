#if os(Windows)
import WinSDK
import SAGATypes
import SAGABackend

extension D3D12Device {
    /// Records one `render` or `renderIndexed` command into the command list.
    func recordDraw(
        _ command: BufferCommand, list: COMObject, transition: (ResourceID, UInt32) -> Void,
        arena: inout D3D12DescriptorArena?, heapsSet: inout Bool, renderTargets: inout D3D12RenderTargetArena?,
        meshGroups: (UInt32, UInt32, UInt32)? = nil, meshIndirect: COMObject? = nil
    ) throws {
        let pipeline: RenderPipeline, target: ResourceID?, colorLoad: RenderColorLoadAction
        let vertexBuffers: [VertexBufferBinding], uniforms: [ComputeBindingSlot: ComputeBufferBinding]
        let sampledTextures: [ComputeBindingSlot: ResourceID], samplerBindings: [ComputeBindingSlot: ResourceID]
        let mipLevel: UInt32, depthSlice: UInt32?, depth: RenderDepthAttachment?, resolve: RenderResolveAttachment?
        let blendConstant: RenderClearColor, additional: [UInt32: RenderColorAttachment], raster: RenderRasterState
        let storageBuffers: [ComputeBindingSlot: ComputeBufferBinding], immediates: [UInt32], colorStore: RenderStoreAction
        var vertexCount: UInt32 = 0, firstVertex: UInt32 = 0, instances = DrawInstances()
        var indexed: (binding: IndexBufferBinding, draw: IndexedDraw)?
        var indirect: (buffer: ResourceID, offset: UInt64, index: IndexBufferBinding?)?
        switch command {
        case .renderIndirect(let p, let t, let load, let arguments, let argumentOffset, let index, let vertices, let uniformBuffers,
                             let textures, let samplers, let mip, let slice, let depthAttachment, let resolveTarget, let blend,
                             let additionalColors, let rasterState, let storage, let immediateWords, let store):
            (pipeline, target, colorLoad, vertexBuffers, uniforms) = (p, t, load, vertices, uniformBuffers)
            (sampledTextures, samplerBindings, mipLevel, depthSlice) = (textures, samplers, mip, slice)
            (depth, resolve, blendConstant, additional, raster) = (depthAttachment, resolveTarget, blend, additionalColors, rasterState)
            (storageBuffers, immediates, colorStore) = (storage, immediateWords, store)
            indirect = (arguments, argumentOffset, index)
        case .render(let p, let t, let load, let count, let vertices, let first, let drawInstances, let uniformBuffers,
                     let textures, let samplers, let mip, let slice, let depthAttachment, let resolveTarget, let blend,
                     let additionalColors, let rasterState, let storage, let immediateWords, let store):
            (pipeline, target, colorLoad, vertexBuffers, uniforms) = (p, t, load, vertices, uniformBuffers)
            (sampledTextures, samplerBindings, mipLevel, depthSlice) = (textures, samplers, mip, slice)
            (depth, resolve, blendConstant, additional, raster) = (depthAttachment, resolveTarget, blend, additionalColors, rasterState)
            (storageBuffers, immediates, colorStore) = (storage, immediateWords, store)
            vertexCount = count; firstVertex = first; instances = drawInstances
        case .renderIndexed(let p, let t, let load, let index, let draw, let vertices, let uniformBuffers, let textures,
                            let samplers, let mip, let slice, let depthAttachment, let resolveTarget, let blend,
                            let additionalColors, let rasterState, let storage, let immediateWords, let store):
            (pipeline, target, colorLoad, vertexBuffers, uniforms) = (p, t, load, vertices, uniformBuffers)
            (sampledTextures, samplerBindings, mipLevel, depthSlice) = (textures, samplers, mip, slice)
            (depth, resolve, blendConstant, additional, raster) = (depthAttachment, resolveTarget, blend, additionalColors, rasterState)
            (storageBuffers, immediates, colorStore) = (storage, immediateWords, store)
            indexed = (index, draw)
        default:
            throw GPUError.unsupported("Direct3D 12 does not implement this render command")
        }
        guard let state = renderPipelines[pipeline.id], state.model == pipeline else {
            throw GPUError.validation("Unknown Direct3D 12 render pipeline")
        }
        guard additional.isEmpty, colorStore == .store || resolve != nil else {
            throw GPUError.unsupported("Direct3D 12 render commands support one color target, discarded only when it resolves")
        }
        guard let target, let texture = self.texture(for: target), texture.descriptor.format == pipeline.colorFormat,
              texture.descriptor.sampleCount == pipeline.sampleCount,
              texture.descriptor.usage.contains(.renderAttachment), let extent = texture.descriptor.mipExtent(mipLevel + texture.baseMip) else {
            throw GPUError.validation("Invalid Direct3D 12 render target")
        }
        guard (pipeline.depthState == nil) == (depth == nil) else {
            throw GPUError.validation("Direct3D 12 render pipeline and depth attachment do not match")
        }
        guard UInt64(immediates.count) * 4 == (pipeline.immediateByteSize + 3) / 4 * 4 else {
            throw GPUError.validation("Immediate data size does not match the render pipeline")
        }

        typealias SetRoot = @convention(c) (UnsafeMutableRawPointer, UnsafeMutableRawPointer) -> Void
        typealias SetState = @convention(c) (UnsafeMutableRawPointer, UnsafeMutableRawPointer) -> Void
        typealias SetAddress = @convention(c) (UnsafeMutableRawPointer, UInt32, UInt64) -> Void
        typealias SetConstants = @convention(c) (UnsafeMutableRawPointer, UInt32, UInt32, UnsafeRawPointer, UInt32) -> Void
        typealias SetTable = @convention(c) (UnsafeMutableRawPointer, UInt32, UInt64) -> Void

        // Render target.
        transition(target, D3D12.stateRenderTarget)
        if renderTargets == nil { renderTargets = try D3D12RenderTargetArena(device: device) }
        let handle = try renderTargets!.allocate()
        typealias CreateRTV = @convention(c) (UnsafeMutableRawPointer, UnsafeMutableRawPointer, UnsafeRawPointer?, UInt) -> Void
        var viewWords = [UInt32](repeating: 0, count: 8)
        viewWords[0] = texture.viewFormat
        let is3D = texture.descriptor.dimension == .d3
        let isArray = texture.descriptor.arrayLayerCount > 1
        viewWords[1] = pipeline.sampleCount > 1 ? 6 : (is3D ? 8 : (isArray ? 5 : 4))
        viewWords[2] = mipLevel + texture.baseMip
        if is3D { viewWords[3] = depthSlice ?? 0; viewWords[4] = 1 }
        else if texture.layerCount != nil && pipeline.sampleCount == 1 { viewWords[1] = 5; viewWords[3] = texture.baseLayer; viewWords[4] = 1 }
        else if isArray { viewWords[3] = 0; viewWords[4] = texture.descriptor.arrayLayerCount }
        viewWords.withUnsafeBytes { device.slot(20, CreateRTV.self)(device.pointer, texture.resource.pointer, $0.baseAddress, handle) }
        if case .clear(let color) = colorLoad {
            typealias Clear = @convention(c) (UnsafeMutableRawPointer, UInt, UnsafeRawPointer, UInt32, UnsafeRawPointer?) -> Void
            let components = [Float(color.red), Float(color.green), Float(color.blue), Float(color.alpha)]
            components.withUnsafeBytes { list.slot(48, Clear.self)(list.pointer, handle, $0.baseAddress!, 0, nil) }
        }
        // Depth-stencil attachment.
        var depthHandle: [UInt] = []
        var stencilReference: UInt32?
        if let depth, let depthState = pipeline.depthState {
            guard let attachment = textures[depth.texture], let dsvFormat = attachment.dsvFormat,
                  attachment.descriptor.format == depthState.format, attachment.descriptor.sampleCount == pipeline.sampleCount,
                  attachment.descriptor.usage.contains(.renderAttachment),
                  let depthExtent = attachment.descriptor.mipExtent(depth.mipLevel), depthExtent == extent else {
                throw GPUError.validation("Invalid Direct3D 12 depth attachment")
            }
            let readOnly = depth.loadAction == .readOnly && (depth.stencil == nil || depth.stencil?.loadAction == .readOnly)
            transition(depth.texture, readOnly ? D3D12.stateDepthRead : D3D12.stateDepthWrite)
            let dsv = try renderTargets!.allocateDepth()
            typealias CreateDSV = @convention(c) (UnsafeMutableRawPointer, UnsafeMutableRawPointer, UnsafeRawPointer, UInt) -> Void
            var flags: UInt32 = 0
            if depth.loadAction == .readOnly { flags |= 1 }
            if depth.stencil?.loadAction == .readOnly { flags |= 2 }
            let words: [UInt32] = [dsvFormat, pipeline.sampleCount > 1 ? 5 : 3, flags, depth.mipLevel]
            words.withUnsafeBytes { device.slot(21, CreateDSV.self)(device.pointer, attachment.resource.pointer, $0.baseAddress!, dsv) }
            var clearFlags: UInt32 = 0
            var clearDepth: Float = 1
            var clearStencil: UInt8 = 0
            if case .clear(let value) = depth.loadAction { clearFlags |= 1; clearDepth = Float(value) }
            if case .clear(let value)? = depth.stencil?.loadAction { clearFlags |= 2; clearStencil = UInt8(truncatingIfNeeded: value) }
            if clearFlags != 0 {
                typealias ClearDepth = @convention(c) (UnsafeMutableRawPointer, UInt, UInt32, Float, UInt8, UInt32, UnsafeRawPointer?) -> Void
                list.slot(47, ClearDepth.self)(list.pointer, dsv, clearFlags, clearDepth, clearStencil, 0, nil)
            }
            depthHandle = [dsv]
            stencilReference = depth.stencil?.reference
        }
        typealias SetTargets = @convention(c) (UnsafeMutableRawPointer, UInt32, UnsafeRawPointer, Bool, UnsafeRawPointer?) -> Void
        let handles: [UInt] = [handle]
        handles.withUnsafeBytes { targets in
            depthHandle.withUnsafeBytes { list.slot(46, SetTargets.self)(list.pointer, 1, targets.baseAddress!, false, depthHandle.isEmpty ? nil : $0.baseAddress) }
        }
        if let stencilReference {
            typealias SetStencil = @convention(c) (UnsafeMutableRawPointer, UInt32) -> Void
            list.slot(24, SetStencil.self)(list.pointer, stencilReference)
        }

        // Viewport, scissor, blend constant, topology.
        var viewport: [Float] = [0, 0, Float(extent.width), Float(extent.height), 0, 1]
        if let v = raster.viewport { viewport = [v.x, v.y, v.width, v.height, v.minDepth, v.maxDepth] }
        var scissor: [Int32] = [0, 0, Int32(extent.width), Int32(extent.height)]
        if let s = raster.scissor { scissor = [Int32(s.x), Int32(s.y), Int32(s.x + s.width), Int32(s.y + s.height)] }
        typealias SetViewports = @convention(c) (UnsafeMutableRawPointer, UInt32, UnsafeRawPointer) -> Void
        viewport.withUnsafeBytes { list.slot(21, SetViewports.self)(list.pointer, 1, $0.baseAddress!) }
        scissor.withUnsafeBytes { list.slot(22, SetViewports.self)(list.pointer, 1, $0.baseAddress!) }
        let blend = [Float(blendConstant.red), Float(blendConstant.green), Float(blendConstant.blue), Float(blendConstant.alpha)]
        typealias SetBlend = @convention(c) (UnsafeMutableRawPointer, UnsafeRawPointer) -> Void
        blend.withUnsafeBytes { list.slot(23, SetBlend.self)(list.pointer, $0.baseAddress!) }
        typealias SetTopology = @convention(c) (UnsafeMutableRawPointer, UInt32) -> Void
        list.slot(20, SetTopology.self)(list.pointer, state.topology)

        // Pipeline and bindings.
        list.slot(30, SetRoot.self)(list.pointer, state.rootSignature.pointer)
        list.slot(25, SetState.self)(list.pointer, state.state.pointer)
        var rootIndex = 0
        for uniform in pipeline.uniforms {
            guard let supplied = uniforms[uniform.slot], let buffer = buffers[supplied.buffer] else {
                throw GPUError.validation("Missing Direct3D 12 uniform binding")
            }
            let range = try supplied.range(in: buffer.descriptor, usage: .uniform)
            transition(supplied.buffer, D3D12.stateConstantBuffer)
            list.slot(38, SetAddress.self)(list.pointer, UInt32(rootIndex), buffer.address + range.lowerBound)
            rootIndex += 1
        }
        for (storage, root) in state.rootStorage {
            rootIndex = root
            guard let supplied = storageBuffers[storage.handle.slot], let buffer = buffers[supplied.buffer] else {
                throw GPUError.validation("Missing Direct3D 12 storage binding")
            }
            let range = try supplied.range(in: buffer.descriptor, usage: .storage)
            if storage.isReadOnly {
                transition(supplied.buffer, D3D12.stateNonPixelShaderResource | D3D12.statePixelShaderResource)
                list.slot(40, SetAddress.self)(list.pointer, UInt32(rootIndex), buffer.address + range.lowerBound)
            } else {
                transition(supplied.buffer, D3D12.stateUnorderedAccess)
                list.slot(42, SetAddress.self)(list.pointer, UInt32(rootIndex), buffer.address + range.lowerBound)
            }
            rootIndex += 1
        }
        if let root = state.immediateRoot {
            var words = immediates
            if words.count < state.immediateWords { words += [UInt32](repeating: 0, count: state.immediateWords - words.count) }
            words.withUnsafeBytes { list.slot(36, SetConstants.self)(list.pointer, UInt32(root), UInt32(state.immediateWords), $0.baseAddress!, 0) }
        }
        if let root = state.drawParametersRoot {
            let first = indexed.map { UInt32(bitPattern: $0.draw.baseVertex) } ?? firstVertex
            let firstInstance = indexed?.draw.instances.first ?? instances.first
            let words: [UInt32] = [first, firstInstance]
            words.withUnsafeBytes { list.slot(36, SetConstants.self)(list.pointer, UInt32(root), 2, $0.baseAddress!, 0) }
        }
        for entry in state.accelerationRoots {
            guard let id = sampledTextures[entry.slot], let structure = accelerationStructures[id] else {
                throw GPUError.validation("Missing Direct3D 12 acceleration structure binding")
            }
            list.slot(40, SetAddress.self)(list.pointer, UInt32(entry.root), structure.address)
        }
        for entry in state.externalLayouts {
            let slot = ComputeBindingSlot(group: entry.layout.binding.group, binding: entry.layout.binding.binding)
            guard let id = sampledTextures[slot], let external = externalTextures[id] else {
                throw GPUError.validation("Missing Direct3D 12 external texture binding")
            }
            list.slot(38, SetAddress.self)(list.pointer, UInt32(entry.root), external.address)
        }
        if state.viewRoot != nil || state.samplerRoot != nil {
            if arena == nil { arena = try D3D12DescriptorArena(device: device) }
            let heaps = arena!
            if !heapsSet {
                typealias SetHeaps = @convention(c) (UnsafeMutableRawPointer, UInt32, UnsafeRawPointer) -> Void
                var heapPointers: [UnsafeMutableRawPointer?] = [heaps.viewHeap.pointer, heaps.samplerHeap.pointer]
                list.slot(28, SetHeaps.self)(list.pointer, 2, &heapPointers)
                heapsSet = true
            }
            if let root = state.viewRoot {
                let base = try heaps.allocateViews(state.viewLayout.count + state.bufferViewLayout.count + 3 * state.externalLayouts.count)
                for (offset, entry) in state.bufferViewLayout.enumerated() {
                    guard let supplied = storageBuffers[entry.slot], let buffer = buffers[supplied.buffer] else {
                        throw GPUError.validation("Missing Direct3D 12 buffer array binding")
                    }
                    let range = try supplied.range(in: buffer.descriptor, usage: .storage)
                    transition(supplied.buffer, entry.readOnly ? D3D12.stateNonPixelShaderResource | D3D12.statePixelShaderResource
                               : D3D12.stateUnorderedAccess)
                    createRawBufferView(buffer.resource, offset: range.lowerBound, size: UInt64(range.count), storage: !entry.readOnly,
                                        at: base.cpu + UInt(state.viewLayout.count + offset) * heaps.viewIncrement)
                }
                for (index, entry) in state.viewLayout.enumerated() {
                    guard let id = sampledTextures[entry.slot], let bound = self.texture(for: id),
                          bound.descriptor.usage.contains(entry.storage ? .storageBinding : .textureBinding) else {
                        throw GPUError.validation("Missing or invalid Direct3D 12 texture binding")
                    }
                    transition(id, entry.storage ? D3D12.stateUnorderedAccess
                               : D3D12.stateNonPixelShaderResource | D3D12.statePixelShaderResource)
                    createView(bound, storage: entry.storage, at: base.cpu + UInt(index) * heaps.viewIncrement)
                }
                let externalBase = state.viewLayout.count + state.bufferViewLayout.count
                for (index, entry) in state.externalLayouts.enumerated() {
                    let slot = ComputeBindingSlot(group: entry.layout.binding.group, binding: entry.layout.binding.binding)
                    guard let id = sampledTextures[slot], let external = externalTextures[id] else {
                        throw GPUError.validation("Missing Direct3D 12 external texture binding")
                    }
                    for plane in 0..<3 {
                        let planeID = external.planes[plane < external.planes.count ? plane : 0]
                        guard let planeTexture = self.texture(for: planeID) else {
                            throw GPUError.validation("Unknown Direct3D 12 external texture plane")
                        }
                        transition(planeID, D3D12.stateNonPixelShaderResource | D3D12.statePixelShaderResource)
                        createView(planeTexture, storage: false,
                                   at: base.cpu + UInt(externalBase + 3 * index + plane) * heaps.viewIncrement)
                    }
                }
                list.slot(32, SetTable.self)(list.pointer, UInt32(root), base.gpu)
            }
            if let root = state.samplerRoot {
                let base = try heaps.allocateSamplers(state.samplerLayout.count)
                for (index, slot) in state.samplerLayout.enumerated() {
                    guard let id = samplerBindings[slot], let sampler = samplers[id] else {
                        throw GPUError.validation("Missing Direct3D 12 sampler binding")
                    }
                    createSampler(sampler, at: base.cpu + UInt(index) * heaps.samplerIncrement)
                }
                list.slot(32, SetTable.self)(list.pointer, UInt32(root), base.gpu)
            }
        }

        // Vertex and index buffers.
        var views = [UInt64](repeating: 0, count: pipeline.vertexBuffers.count * 2)  // address, then size and stride
        var bound = [Bool](repeating: false, count: pipeline.vertexBuffers.count)
        for (index, binding) in vertexBuffers.enumerated() {
            let slot = Int(binding.slot ?? UInt32(index))
            guard slot < bound.count, !bound[slot], let buffer = buffers[binding.buffer], binding.offset <= buffer.descriptor.size else {
                throw GPUError.validation("Invalid Direct3D 12 vertex buffer binding")
            }
            transition(binding.buffer, D3D12.stateVertexAndConstantBuffer)
            bound[slot] = true
            views[slot * 2] = buffer.address + binding.offset
            let stride = UInt64(pipeline.vertexBuffers[slot].stride)
            views[slot * 2 + 1] = UInt64(buffer.descriptor.size - binding.offset) | stride << 32
        }
        guard bound.allSatisfy({ $0 }) else { throw GPUError.validation("Missing Direct3D 12 vertex buffer binding") }
        if !views.isEmpty {
            typealias SetVertexBuffers = @convention(c) (UnsafeMutableRawPointer, UInt32, UInt32, UnsafeRawPointer) -> Void
            views.withUnsafeBytes { list.slot(44, SetVertexBuffers.self)(list.pointer, 0, UInt32(bound.count), $0.baseAddress!) }
        }
        if let meshIndirect {
            try executeIndirectMesh(list: list, arguments: meshIndirect)
        } else if let meshGroups {
            try dispatchMesh(list: list, groups: meshGroups)
        } else if let indirect {
            guard state.drawParametersRoot == nil else {
                throw GPUError.unsupported("Direct3D 12 indirect draws need a shader model 6.8 validator for this vertex shader")
            }
            guard let source = buffers[indirect.buffer] else { throw GPUError.validation("Unknown Direct3D 12 indirect buffer") }
            if let index = indirect.index {
                guard let indices = buffers[index.buffer], index.offset <= indices.descriptor.size else {
                    throw GPUError.validation("Invalid Direct3D 12 index buffer")
                }
                transition(index.buffer, D3D12.stateIndexBuffer)
                let view: [UInt64] = [indices.address + index.offset,
                                      UInt64(indices.descriptor.size - index.offset) | UInt64(index.format == .uint16 ? 57 : 42) << 32]
                typealias SetIndexBuffer = @convention(c) (UnsafeMutableRawPointer, UnsafeRawPointer) -> Void
                view.withUnsafeBytes { list.slot(43, SetIndexBuffer.self)(list.pointer, $0.baseAddress!) }
            }
            transition(indirect.buffer, D3D12.stateIndirectArgument)
            typealias ExecuteIndirect = @convention(c) (UnsafeMutableRawPointer, UnsafeMutableRawPointer, UInt32, UnsafeMutableRawPointer, UInt64,
                                                        UnsafeMutableRawPointer?, UInt64) -> Void
            let signature = try drawSignature(indexed: indirect.index != nil)
            list.slot(59, ExecuteIndirect.self)(list.pointer, signature.pointer, 1, source.resource.pointer, indirect.offset, nil, 0)
        } else if let indexed {
            guard let buffer = buffers[indexed.binding.buffer], indexed.binding.offset <= buffer.descriptor.size else {
                throw GPUError.validation("Invalid Direct3D 12 index buffer")
            }
            transition(indexed.binding.buffer, D3D12.stateIndexBuffer)
            let view: [UInt64] = [buffer.address + indexed.binding.offset,
                                  UInt64(buffer.descriptor.size - indexed.binding.offset) | UInt64(indexed.binding.format == .uint16 ? 57 : 42) << 32]
            typealias SetIndexBuffer = @convention(c) (UnsafeMutableRawPointer, UnsafeRawPointer) -> Void
            view.withUnsafeBytes { list.slot(43, SetIndexBuffer.self)(list.pointer, $0.baseAddress!) }
            if indexed.draw.indexCount > 0, indexed.draw.instances.count > 0 {
                typealias DrawIndexed = @convention(c) (UnsafeMutableRawPointer, UInt32, UInt32, UInt32, Int32, UInt32) -> Void
                list.slot(13, DrawIndexed.self)(list.pointer, indexed.draw.indexCount, indexed.draw.instances.count,
                                                indexed.draw.firstIndex, indexed.draw.baseVertex, indexed.draw.instances.first)
            }
        } else if vertexCount > 0, instances.count > 0 {
            typealias Draw = @convention(c) (UnsafeMutableRawPointer, UInt32, UInt32, UInt32, UInt32) -> Void
            list.slot(12, Draw.self)(list.pointer, vertexCount, instances.count, firstVertex, instances.first)
        }
        if let resolve {
            guard let destination = textures[resolve.texture], destination.descriptor.sampleCount == 1,
                  destination.descriptor.format == pipeline.colorFormat,
                  destination.descriptor.usage.contains(.renderAttachment),
                  let destinationExtent = destination.descriptor.mipExtent(resolve.mipLevel),
                  destinationExtent == extent else {
                throw GPUError.validation("Invalid Direct3D 12 resolve target")
            }
            transition(target, D3D12.stateResolveSource)
            transition(resolve.texture, D3D12.stateResolveDestination)
            typealias Resolve = @convention(c) (UnsafeMutableRawPointer, UnsafeMutableRawPointer, UInt32, UnsafeMutableRawPointer, UInt32, UInt32) -> Void
            list.slot(19, Resolve.self)(list.pointer, destination.resource.pointer, resolve.mipLevel, texture.resource.pointer, 0,
                                        texture.viewFormat)
        }
    }
}
#endif
