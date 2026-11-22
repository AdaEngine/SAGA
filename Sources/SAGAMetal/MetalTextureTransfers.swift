import Foundation
import SAGABackend
import SAGAShaderIR
import SAGATypes
#if canImport(Metal)
import Metal

extension MetalDevice {
    func encodeUpload(
        _ bytes: UnsafeRawBufferPointer, to destination: any MTLBuffer, offset: Int, on commands: any MTLCommandBuffer
    ) throws -> any MTLBuffer {
        guard let staging = native.makeBuffer(length: bytes.count, options: .storageModeShared),
              let encoder = commands.makeBlitCommandEncoder() else { throw GPUError.outOfMemory }
        staging.contents().copyMemory(from: bytes.baseAddress!, byteCount: bytes.count)
        encoder.copy(from: staging, sourceOffset: 0, to: destination, destinationOffset: offset, size: bytes.count)
        encoder.endEncoding()
        return staging
    }

    func applySharedUploads(_ commands: [BufferCommand], resources: [ResourceID]) throws -> Bool {
        guard let first = commands.first, case .writeBytes = first else { return false }
        let pinned = Set(resources)
        var uploads: [(buffer: any MTLBuffer, offset: Int, bytes: ArraySlice<UInt8>)] = []
        uploads.reserveCapacity(commands.count)
        for command in commands {
            guard case let .writeBytes(id, offset, bytes, dataOffset, byteCount) = command,
                  pinned.contains(id), let buffer = buffers[id], buffer.mapping == nil,
                  buffer.native.storageMode == .shared else { return false }
            try command.validate(device: id.device, descriptors: [id: buffer.descriptor])
            let source = try BufferCommand.uploadBytes(bytes, dataOffset: dataOffset, byteCount: byteCount)
            uploads.append((buffer.native, Int(offset), source))
        }
        if Self.traceGrouping && uploads.count > 1 { print("SAGA-UPLOAD\t\(uploads.count)") }
        for upload in uploads {
            upload.bytes.withUnsafeBytes { bytes in
                if let address = bytes.baseAddress, !bytes.isEmpty {
                    upload.buffer.contents().advanced(by: upload.offset).copyMemory(from: address, byteCount: bytes.count)
                }
            }
        }
        return true
    }

    func initializeBoundTextures(
        _ bindings: [ComputeBindingSlot: ResourceID], on commands: any MTLCommandBuffer,
        initialized: inout [ResourceID: Set<TextureSubresource>]
    ) throws {
        for id in Set(bindings.values.flatMap { externalTextures[$0]?.planes ?? [$0] }) {
            try initializeTextureSubresources(id, on: commands, initialized: &initialized)
        }
    }

    func initializeTextureSubresources(
        _ id: ResourceID, on commands: any MTLCommandBuffer,
        initialized: inout [ResourceID: Set<TextureSubresource>]
    ) throws {
        let view = textureViews[id]?.resource
        let parent = view?.texture ?? id
        let descriptor = textures[parent]!.descriptor
        let levels = view?.levels ?? 0..<descriptor.mipLevelCount
        for level in levels {
            let layers = descriptor.dimension == .d3 ? 0..<descriptor.mipDepth(level) :
                (view?.layers ?? 0..<descriptor.arrayLayerCount)
            for layer in layers {
                try initializeTexture(parent, level: level, layer: layer, on: commands, initialized: &initialized)
            }
        }
    }

    func stageTextureUpload(
        _ data: [UInt8], layout: TextureDataLayout, region: TextureCopyRegion, block: TextureCopyBlock
    ) throws -> (any MTLBuffer, TextureBufferLayout) {
        let footprint = try layout.footprint(region: region, block: block, dataSize: UInt64(data.count))
        let rowBytes = Int(footprint.rowBytes)
        let stride = (rowBytes + 3) & ~3
        let blockRows = Int(region.height / block.height)
        let (imageStride, imageOverflow) = stride.multipliedReportingOverflow(by: blockRows)
        let (size, sizeOverflow) = imageStride.multipliedReportingOverflow(by: Int(region.depth))
        guard !imageOverflow, !sizeOverflow, let bytesPerRow = UInt32(exactly: stride),
              let staging = native.makeBuffer(length: size, options: .storageModeShared) else {
            throw GPUError.outOfMemory
        }
        data.withUnsafeBytes { source in
            let input = source.baseAddress!.advanced(by: Int(layout.offset))
            let output = staging.contents()
            for image in 0..<Int(region.depth) {
                let imageStart = input.advanced(by: image * Int(footprint.imageStride))
                for row in 0..<blockRows {
                    let from = imageStart.advanced(by: row * Int(footprint.rowStride))
                    let to = output.advanced(by: image * imageStride + row * stride)
                    to.copyMemory(from: from, byteCount: rowBytes)
                }
            }
        }
        return (staging, .init(bytesPerRow: bytesPerRow))
    }

    func encodeTextureUpload(
        _ source: any MTLBuffer, layout: TextureBufferLayout, destination: ResourceID, region: TextureCopyRegion,
        aspect: TextureViewAspect, on commands: any MTLCommandBuffer,
        initialized: inout [ResourceID: Set<TextureSubresource>]
    ) throws {
        let texture = textures[destination]!
        try prepareTextureWrite(destination, region: region, aspect: aspect, on: commands, initialized: &initialized)
        guard let encoder = commands.makeBlitCommandEncoder() else { throw GPUError.outOfMemory }
        let block = texture.descriptor.format.copyBlock(aspect: aspect)!
        let stride = Int(layout.rowStride(width: region.width, block: block))
        let imageStride = region.depth > 1 ? stride * Int(layout.rowsPerImage ?? (region.height / block.height)) : 0
        let metalRowStride = region.depth == 1 && region.height == block.height
            ? Int(block.rowBytes(width: region.width)) : stride
        let volume = texture.descriptor.dimension == .d3
        // Metal buffer blits take the logical mip extent even when they transfer a complete compression block.
        let extent = texture.descriptor.mipExtent(region.mipLevel)!
        let size = MTLSize(width: Int(min(region.width, extent.width - region.x)),
                           height: Int(min(region.height, extent.height - region.y)),
                           depth: volume ? Int(region.depth) : 1)
        let origin = MTLOrigin(x: Int(region.x), y: Int(region.y), z: Int(region.z))
        for layer in 0..<(volume ? 1 : Int(region.depth)) {
            encoder.copy(from: source, sourceOffset: Int(layout.offset) + layer * imageStride,
                         sourceBytesPerRow: metalRowStride, sourceBytesPerImage: volume && region.depth > 1 ? imageStride : 0,
                         sourceSize: size, to: texture.native, destinationSlice: Int(region.arrayLayer) + layer,
                         destinationLevel: Int(region.mipLevel), destinationOrigin: origin,
                         options: aspect == .stencilOnly && texture.descriptor.format.hasDepth ? .stencilFromDepthStencil : [])
        }
        encoder.endEncoding()
    }

    func prepareTextureWrite(
        _ id: ResourceID, region: TextureCopyRegion, aspect: TextureViewAspect, on commands: any MTLCommandBuffer,
        initialized: inout [ResourceID: Set<TextureSubresource>]
    ) throws {
        let descriptor = textures[id]!.descriptor
        let fullPlane = (aspect != .stencilOnly || !descriptor.format.hasDepth) && region.covers(descriptor)
        for plane in region.planes(in: descriptor) {
            if fullPlane {
                initialized[id, default: []].insert(.init(level: region.mipLevel, layer: plane))
            } else {
                try initializeTexture(id, level: region.mipLevel, layer: plane, on: commands, initialized: &initialized)
            }
        }
    }

    func clearsTextureWithBlit(_ descriptor: TextureDescriptor) -> Bool {
        if descriptor.dimension == .d1 || descriptor.format.compression != nil { return true }
        if descriptor.format.isSnorm8Color && !capabilities.features.contains(.snorm8Renderable) { return true }
        guard let feature = descriptor.format.hdrRenderFeature else { return false }
        return !capabilities.features.contains(feature)
    }

    func initializeTexture(
        _ id: ResourceID, level: UInt32, layer: UInt32 = 0, on commands: any MTLCommandBuffer,
        initialized: inout [ResourceID: Set<TextureSubresource>]
    ) throws {
        let texture = textures[id]!
        // Only clear the subresource that could still expose uninitialized Metal memory.
        let subresource = TextureSubresource(level: level, layer: layer)
        guard !texture.initializedSubresources.contains(subresource),
              initialized[id, default: []].insert(subresource).inserted else { return }
        if clearsTextureWithBlit(texture.descriptor) {
            let logical = texture.descriptor.mipExtent(level)!
            let physical = texture.descriptor.mipPhysicalExtent(level)!
            let block = texture.descriptor.format.copyBlock(aspect: .all)!
            let rowBytes = Int(block.rowBytes(width: physical.width))
            let size = rowBytes * Int(physical.height / block.height)
            let fillSize = (size + 3) & ~3
            guard let zeroes = native.makeBuffer(length: fillSize, options: .storageModePrivate),
                  let encoder = commands.makeBlitCommandEncoder() else { throw GPUError.outOfMemory }
            encoder.fill(buffer: zeroes, range: 0..<fillSize, value: 0)
            let volume = texture.descriptor.dimension == .d3
            let origin = MTLOrigin(x: 0, y: 0, z: volume ? Int(layer) : 0)
            encoder.copy(from: zeroes, sourceOffset: 0, sourceBytesPerRow: rowBytes, sourceBytesPerImage: volume ? size : 0,
                         sourceSize: MTLSize(width: Int(logical.width), height: Int(logical.height), depth: 1),
                         to: texture.native, destinationSlice: volume ? 0 : Int(layer), destinationLevel: Int(level),
                         destinationOrigin: origin)
            encoder.endEncoding()
        } else if texture.descriptor.format.isDepthStencil {
            let pass = MTLRenderPassDescriptor()
            setDepthAttachment(pass, texture: texture.native, level: level, layer: layer,
                               loadAction: texture.descriptor.format.hasDepth ? .clear(0) : .unused,
                               stencilLoadAction: texture.descriptor.format.hasStencil ? .clear(0) : nil)
            guard let encoder = commands.makeRenderCommandEncoder(descriptor: pass) else { throw GPUError.outOfMemory }
            encoder.endEncoding()
        } else {
            let clear = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
            try renderEncoder(commands, target: texture.native, level: level, layer: layer, clear: clear).endEncoding()
        }
    }
}
#endif
