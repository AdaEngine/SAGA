#if os(Windows)
import WinSDK
import SAGATypes
import SAGABackend

/// Native graphics state lasts until the list is reset or another command changes it.
struct D3D12DrawState {
    var colorCount: Int?
    var hasDepth = false
    var stencilReference: UInt32?
    var viewport: RenderViewport?
    var scissor: RenderScissor?
    var blend: RenderClearColor?
    var topology: UInt32?
    var pipeline: ResourceID?
}

extension D3D12Device {
    /// Records one `render` or `renderIndexed` command into the command list.
    func recordDraw(
        _ command: BufferCommand, list: COMObject, transition: (ResourceID, UInt32) -> Void,
        arena: D3D12DescriptorArena, renderTargets: inout D3D12RenderTargetArena?, graphics: inout D3D12DrawState,
        meshGroups: (UInt32, UInt32, UInt32)? = nil, meshIndirect: COMObject? = nil, multiCount: IndirectDrawCount? = nil
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
        guard Set(additional.keys) == Set(pipeline.additionalColorTargets.keys),
              (target == nil) == (pipeline.colorFormat == nil) else {
            throw GPUError.validation("Color attachments do not match the Direct3D 12 pipeline")
        }
        guard colorStore == .store || resolve != nil,
              additional.values.allSatisfy({ $0.storeAction == .store && $0.resolve == nil }) else {
            throw GPUError.unsupported("Direct3D 12 supports resolving and discarding only the primary color target")
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

        if renderTargets == nil { renderTargets = try D3D12RenderTargetArena(device: device) }
        renderTargets?.beginDraw()
        var extent: (width: UInt32, height: UInt32)?
        var handles: [UInt] = []
        var changedAttachments = false
        func bindColor(_ attachment: RenderColorAttachment, format: TextureFormat) throws {
            let target = attachment.texture
            guard let texture = self.texture(for: target), texture.descriptor.format == format,
                  texture.descriptor.sampleCount == pipeline.sampleCount, texture.descriptor.usage.contains(.renderAttachment),
                  let size = texture.descriptor.mipExtent(attachment.mipLevel + texture.baseMip),
                  extent.map({ $0 == size }) ?? true else {
                throw GPUError.validation("Invalid Direct3D 12 render target")
            }
            extent = size
            transition(target, D3D12.stateRenderTarget)
            let (handle, needsWrite) = try renderTargets!.allocateColor(
                .init(texture: target, mip: attachment.mipLevel, sliceOrFlags: attachment.depthSlice ?? 0)
            )
            if needsWrite {
                changedAttachments = true
                typealias CreateRTV = @convention(c) (UnsafeMutableRawPointer, UnsafeMutableRawPointer, UnsafeRawPointer?, UInt) -> Void
                var viewWords = [UInt32](repeating: 0, count: 8)
                viewWords[0] = texture.viewFormat
                let is3D = texture.descriptor.dimension == .d3
                let isArray = texture.descriptor.arrayLayerCount > 1
                viewWords[1] = pipeline.sampleCount > 1 ? 6 : (is3D ? 8 : (isArray ? 5 : 4))
                viewWords[2] = attachment.mipLevel + texture.baseMip
                if is3D { viewWords[3] = attachment.depthSlice ?? 0; viewWords[4] = 1 }
                else if isArray && pipeline.sampleCount > 1 {
                    viewWords[1] = 7; viewWords[2] = texture.baseLayer; viewWords[3] = texture.layerCount ?? texture.descriptor.arrayLayerCount
                } else if isArray {
                    viewWords[3] = texture.baseLayer; viewWords[4] = texture.layerCount ?? texture.descriptor.arrayLayerCount
                }
                viewWords.withUnsafeBytes { device.slot(20, CreateRTV.self)(device.pointer, texture.resource.pointer, $0.baseAddress, handle) }
            }
            if case .clear(let color) = attachment.loadAction {
                typealias Clear = @convention(c) (UnsafeMutableRawPointer, UInt, UnsafeRawPointer, UInt32, UnsafeRawPointer?) -> Void
                let components = [Float(color.red), Float(color.green), Float(color.blue), Float(color.alpha)]
                components.withUnsafeBytes { list.slot(48, Clear.self)(list.pointer, handle, $0.baseAddress!, 0, nil) }
            }
            handles.append(handle)
        }
        if let target, let format = pipeline.colorFormat {
            try bindColor(.init(texture: target, mipLevel: mipLevel, loadAction: colorLoad, depthSlice: depthSlice), format: format)
        }
        for location in additional.keys.sorted() {
            guard let attachment = additional[location], let format = pipeline.additionalColorTargets[location]?.format else {
                throw GPUError.validation("Missing Direct3D 12 render target")
            }
            try bindColor(attachment, format: format)
        }
        // Depth-stencil attachment.
        var depthHandle: [UInt] = []
        var stencilReference: UInt32?
        if let depth, let depthState = pipeline.depthState {
            guard let attachment = texture(for: depth.texture), let dsvFormat = attachment.dsvFormat,
                  attachment.descriptor.format == depthState.format, attachment.descriptor.sampleCount == pipeline.sampleCount,
                  attachment.descriptor.usage.contains(.renderAttachment),
                  let depthExtent = attachment.descriptor.mipExtent(depth.mipLevel + attachment.baseMip),
                  extent.map({ $0 == depthExtent }) ?? true else {
                throw GPUError.validation("Invalid Direct3D 12 depth attachment")
            }
            extent = depthExtent
            let readOnly = depth.loadAction == .readOnly && (depth.stencil == nil || depth.stencil?.loadAction == .readOnly)
            transition(depth.texture, readOnly ? D3D12.stateDepthRead : D3D12.stateDepthWrite)
            var flags: UInt32 = 0
            if depth.loadAction == .readOnly { flags |= 1 }
            if depth.stencil?.loadAction == .readOnly { flags |= 2 }
            let (dsv, needsWrite) = try renderTargets!.allocateDepth(.init(texture: depth.texture, mip: depth.mipLevel, sliceOrFlags: flags))
            if needsWrite {
                changedAttachments = true
                typealias CreateDSV = @convention(c) (UnsafeMutableRawPointer, UnsafeMutableRawPointer, UnsafeRawPointer, UInt) -> Void
                let array = attachment.descriptor.arrayLayerCount > 1
                let mip = attachment.baseMip + depth.mipLevel
                let layers = attachment.layerCount ?? attachment.descriptor.arrayLayerCount
                let words: [UInt32] = switch (pipeline.sampleCount > 1, array) {
                case (true, true): [dsvFormat, 6, flags, attachment.baseLayer, layers, 0]
                case (true, false): [dsvFormat, 5, flags, 0, 0, 0]
                case (false, true): [dsvFormat, 4, flags, mip, attachment.baseLayer, layers]
                case (false, false): [dsvFormat, 3, flags, mip, 0, 0]
                }
                words.withUnsafeBytes { device.slot(21, CreateDSV.self)(device.pointer, attachment.resource.pointer, $0.baseAddress!, dsv) }
            }
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
        guard let extent else { throw GPUError.validation("Direct3D 12 draw has no attachments") }
        typealias SetTargets = @convention(c) (UnsafeMutableRawPointer, UInt32, UnsafeRawPointer?, Bool, UnsafeRawPointer?) -> Void
        let hasDepth = !depthHandle.isEmpty
        if changedAttachments || graphics.colorCount != handles.count || graphics.hasDepth != hasDepth {
            handles.withUnsafeBytes { targets in
                depthHandle.withUnsafeBytes {
                    list.slot(46, SetTargets.self)(list.pointer, UInt32(handles.count), handles.isEmpty ? nil : targets.baseAddress,
                                                  false, depthHandle.isEmpty ? nil : $0.baseAddress)
                }
            }
            graphics.colorCount = handles.count
            graphics.hasDepth = hasDepth
        }
        if let stencilReference, graphics.stencilReference != stencilReference {
            typealias SetStencil = @convention(c) (UnsafeMutableRawPointer, UInt32) -> Void
            list.slot(24, SetStencil.self)(list.pointer, stencilReference)
            graphics.stencilReference = stencilReference
        }

        // Viewport, scissor, blend constant, topology.
        let viewport = raster.viewport ?? RenderViewport(x: 0, y: 0, width: Float(extent.width), height: Float(extent.height))
        let scissor = raster.scissor ?? RenderScissor(x: 0, y: 0, width: extent.width, height: extent.height)
        typealias SetViewports = @convention(c) (UnsafeMutableRawPointer, UInt32, UnsafeRawPointer) -> Void
        if graphics.viewport != viewport {
            let values = [viewport.x, viewport.y, viewport.width, viewport.height, viewport.minDepth, viewport.maxDepth]
            values.withUnsafeBytes { list.slot(21, SetViewports.self)(list.pointer, 1, $0.baseAddress!) }
            graphics.viewport = viewport
        }
        if graphics.scissor != scissor {
            let values = [Int32(scissor.x), Int32(scissor.y), Int32(scissor.x + scissor.width), Int32(scissor.y + scissor.height)]
            values.withUnsafeBytes { list.slot(22, SetViewports.self)(list.pointer, 1, $0.baseAddress!) }
            graphics.scissor = scissor
        }
        if graphics.blend != blendConstant {
            let blend = [Float(blendConstant.red), Float(blendConstant.green), Float(blendConstant.blue), Float(blendConstant.alpha)]
            typealias SetBlend = @convention(c) (UnsafeMutableRawPointer, UnsafeRawPointer) -> Void
            blend.withUnsafeBytes { list.slot(23, SetBlend.self)(list.pointer, $0.baseAddress!) }
            graphics.blend = blendConstant
        }
        if graphics.topology != state.topology {
            typealias SetTopology = @convention(c) (UnsafeMutableRawPointer, UInt32) -> Void
            list.slot(20, SetTopology.self)(list.pointer, state.topology)
            graphics.topology = state.topology
        }

        // Pipeline and bindings.
        if graphics.pipeline != pipeline.id {
            list.slot(30, SetRoot.self)(list.pointer, state.rootSignature.pointer)
            list.slot(25, SetState.self)(list.pointer, state.state.pointer)
            graphics.pipeline = pipeline.id
        }
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
            let viewCount = state.viewLayout.count + state.bufferViewLayout.count + 3 * state.externalLayouts.count
            let heaps = try arena.prepare(device: device, list: list, viewCount: viewCount, samplerCount: state.samplerLayout.count)
            if let root = state.viewRoot {
                let sampledKey: [ResourceID]? = state.externalLayouts.isEmpty && state.bufferViewLayout.isEmpty
                    && !state.viewLayout.contains(where: { $0.storage }) ? state.viewLayout.compactMap { sampledTextures[$0.slot] } : nil
                let cachedTable = sampledKey.flatMap { heaps.sampledTables[$0] }
                let base = try cachedTable.map { (cpu: UInt(0), gpu: $0) } ?? heaps.allocateViews(viewCount)
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
                    if cachedTable == nil {
                        createView(bound, storage: entry.storage, at: base.cpu + UInt(index) * heaps.viewIncrement)
                    }
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
                if let sampledKey, cachedTable == nil {
                    heaps.sampledTables[sampledKey] = base.gpu
                }
                list.slot(32, SetTable.self)(list.pointer, UInt32(root), base.gpu)
            }
            if let root = state.samplerRoot {
                let key = try state.samplerLayout.map { slot in
                    guard let id = samplerBindings[slot] else {
                        throw GPUError.validation("Missing Direct3D 12 sampler binding")
                    }
                    return id
                }
                let table: UInt64
                if let cached = heaps.samplerTables[key] {
                    table = cached
                } else {
                    let base = try heaps.allocateSamplers(key.count)
                    for (index, id) in key.enumerated() {
                        guard let sampler = samplers[id] else {
                            throw GPUError.validation("Missing Direct3D 12 sampler binding")
                        }
                        createSampler(sampler, at: base.cpu + UInt(index) * heaps.samplerIncrement)
                    }
                    table = base.gpu
                    heaps.samplerTables[key] = table
                }
                list.slot(32, SetTable.self)(list.pointer, UInt32(root), table)
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
            // A multi-draw command runs up to its maximum count; a GPU count word, when given, lowers it further.
            var countResource: COMObject?
            var countOffset: UInt64 = 0
            if case .fromBuffer(let id, let offset, _)? = multiCount {
                guard let counts = buffers[id] else { throw GPUError.validation("Unknown Direct3D 12 multi-draw count buffer") }
                transition(id, D3D12.stateIndirectArgument)
                countResource = counts.resource
                countOffset = offset
            }
            list.slot(59, ExecuteIndirect.self)(list.pointer, signature.pointer, multiCount?.maximum ?? 1, source.resource.pointer,
                                                indirect.offset, countResource?.pointer, countOffset)
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
            guard let target, let texture = texture(for: target),
                  let destination = textures[resolve.texture], destination.descriptor.sampleCount == 1,
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
