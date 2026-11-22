import Foundation
import SAGABackend
import SAGAShaderIR
import SAGATypes
#if canImport(Metal)
import Metal
#endif

extension MetalDevice {
    public func allocateAccelerationStructure(id: ResourceID, descriptor: AccelerationStructureDescriptor) throws {
        try requireActive()
        #if canImport(Metal)
        guard native.supportsRaytracing else { throw GPUError.unsupported("Metal ray tracing is unavailable") }
        guard accelerationStructures[id] == nil else {
            throw GPUError.validation("Invalid or duplicate Metal acceleration structure")
        }
        accelerationStructures[id] = AccelerationStructure(descriptor: descriptor, build: nil)
        #else
        throw GPUError.adapterUnavailable
        #endif
    }

    public func destroyAccelerationStructure(id: ResourceID) {
        #if canImport(Metal)
        accelerationStructures.removeValue(forKey: id)
        #endif
    }

    public func allocateBuffer(id: ResourceID, descriptor: BufferDescriptor) throws {
        try requireActive()
        try descriptor.validate(against: capabilities)
        #if canImport(Metal)
        guard buffers[id] == nil, textures[id] == nil, samplers[id] == nil, textureViews[id] == nil, let length = Int(exactly: descriptor.size) else {
            throw GPUError.validation("Duplicate buffer or unrepresentable size")
        }
        guard let buffer = native.makeBuffer(length: max(1, length), options: .storageModeShared) else {
            throw GPUError.outOfMemory
        }
        buffer.contents().initializeMemory(as: UInt8.self, repeating: 0, count: max(1, length))
        buffer.label = descriptor.label
        let mapping = descriptor.mappedAtCreation ?
            BufferMapping(buffer: id, mode: .write, offset: 0, size: descriptor.size, generation: 0) : nil
        buffers[id] = Buffer(descriptor: descriptor, native: buffer, mapping: mapping)
        #else
        throw GPUError.adapterUnavailable
        #endif
    }

    public func destroyBuffer(id: ResourceID) {
        #if canImport(Metal)
        buffers.removeValue(forKey: id)
        #endif
    }

    public func beginBufferMapping(id: ResourceID, mode: BufferMapMode, offset: UInt64, size: UInt64) throws {
        try requireActive()
        #if canImport(Metal)
        let usage: BufferUsage = mode == .read ? .mapRead : .mapWrite
        guard let buffer = buffers[id], buffer.mapping == nil, buffer.descriptor.usage.contains(usage),
              offset % 8 == 0, size % 4 == 0, offset <= buffer.descriptor.size, size <= buffer.descriptor.size - offset else {
            throw GPUError.validation("Invalid Metal buffer mapping")
        }
        buffers[id]?.mapping = .init(buffer: id, mode: mode, offset: offset, size: size, generation: 0)
        #else
        throw GPUError.adapterUnavailable
        #endif
    }

    public func readMappedBytes(id: ResourceID, offset: UInt64, size: UInt64) throws -> [UInt8] {
        try requireActive()
        #if canImport(Metal)
        guard let buffer = buffers[id], let mapping = buffer.mapping else {
            throw GPUError.validation("Metal buffer has no active mapping")
        }
        let length = try mapping.accessSize(offset: offset, size: size)
        let source = buffer.native.contents().advanced(by: Int(offset))
        return Array(UnsafeRawBufferPointer(start: source, count: Int(length)))
        #else
        throw GPUError.adapterUnavailable
        #endif
    }

    public func writeMappedBytes(id: ResourceID, offset: UInt64, bytes: [UInt8]) throws {
        try requireActive()
        #if canImport(Metal)
        guard let buffer = buffers[id], let mapping = buffer.mapping, mapping.mode == .write else {
            throw GPUError.validation("Metal buffer has no writable mapping")
        }
        _ = try mapping.accessSize(offset: offset, size: UInt64(bytes.count))
        if !bytes.isEmpty {
            bytes.withUnsafeBytes { source in
                buffer.native.contents().advanced(by: Int(offset))
                    .copyMemory(from: source.baseAddress!, byteCount: source.count)
            }
        }
        #else
        throw GPUError.adapterUnavailable
        #endif
    }

    public func endBufferMapping(id: ResourceID) {
        #if canImport(Metal)
        buffers[id]?.mapping = nil
        #endif
    }

    public func allocateQuerySet(id: ResourceID, descriptor: QuerySetDescriptor) throws {
        try requireActive()
        #if canImport(Metal)
        try descriptor.validate(against: capabilities)
        guard querySets[id] == nil else {
            throw GPUError.validation("Invalid Metal query set allocation")
        }
        let storage: QuerySet.Storage
        switch descriptor.type {
        case .timestamp:
            guard let timestampCounters else { throw GPUError.unsupported("Metal timestamp counters are unavailable") }
            let sampleDescriptor = MTLCounterSampleBufferDescriptor()
            sampleDescriptor.label = descriptor.label
            sampleDescriptor.counterSet = timestampCounters
            sampleDescriptor.sampleCount = Int(descriptor.count)
            sampleDescriptor.storageMode = .shared
            storage = .timestamp(try native.makeCounterSampleBuffer(descriptor: sampleDescriptor))
        case .occlusion:
            guard let visibility = native.makeBuffer(length: Int(descriptor.count) * 8, options: .storageModePrivate)
            else { throw GPUError.outOfMemory }
            visibility.label = descriptor.label
            storage = .occlusion(visibility)
        }
        querySets[id] = QuerySet(descriptor: descriptor, storage: storage)
        #else
        throw GPUError.adapterUnavailable
        #endif
    }

    public func destroyQuerySet(id: ResourceID) {
        #if canImport(Metal)
        querySets.removeValue(forKey: id)
        #endif
    }

    public func close(owner: UInt64) {
        #if canImport(Metal)
        accelerationStructures = accelerationStructures.filter { $0.key.device != owner }
        buffers = buffers.filter { $0.key.device != owner }
        querySets = querySets.filter { $0.key.device != owner }
        textureViews = textureViews.filter { $0.key.device != owner }
        textures = textures.filter { $0.key.device != owner }
        externalTextures = externalTextures.filter { $0.key.device != owner }
        samplers = samplers.filter { $0.key.device != owner }
        pipelines = pipelines.filter { $0.key.device != owner }
        computePipelines = computePipelines.filter { $0.key.device != owner }
        renderPipelines = renderPipelines.filter { $0.key.device != owner }
        meshPipelines = meshPipelines.filter { $0.key.device != owner }
        #endif
    }

    public func allocatePlanarExternalTexture(id: ResourceID, descriptor: PlanarExternalTextureDescriptor) throws {
        try requireActive()
        #if canImport(Metal)
        guard externalTextures[id] == nil, descriptor.planes.count == descriptor.format.planeComponents.count else {
            throw GPUError.validation("Invalid Metal external texture descriptor")
        }
        var first: (width: UInt32, height: UInt32)?
        for (index, plane) in descriptor.planes.enumerated() {
            let view = textureViews[plane]?.resource
            guard plane.device == id.device, let texture = textures[view?.texture ?? plane] else {
                throw GPUError.validation("Unknown Metal external texture plane")
            }
            let extent = try descriptor.planeExtent(at: index, texture: texture.descriptor, view: view, capabilities: capabilities)
            if index == 0 { first = extent }
        }
        guard let first else { throw GPUError.validation("External texture requires a first plane") }
        let words = try descriptor.parameterWords(planeWidth: first.width, planeHeight: first.height)
        externalTextures[id] = .init(planes: descriptor.planes, parameters: words)
        #else
        throw GPUError.adapterUnavailable
        #endif
    }

    public func destroyExternalTexture(id: ResourceID) {
        #if canImport(Metal)
        externalTextures.removeValue(forKey: id)
        #endif
    }

    public func allocateTexture(id: ResourceID, descriptor: TextureDescriptor) throws {
        try requireActive()
        try descriptor.validate(against: capabilities)
        #if canImport(Metal)
        guard textures[id] == nil, buffers[id] == nil, samplers[id] == nil, textureViews[id] == nil else {
            throw GPUError.validation("Duplicate Metal resource")
        }
        let metal = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: descriptor.format.metal, width: Int(descriptor.width),
            height: Int(descriptor.height), mipmapped: descriptor.mipLevelCount > 1)
        let samples = Int(descriptor.sampleCount)
        guard native.supportsTextureSampleCount(samples) else {
            throw GPUError.unsupported("Metal does not support the texture sample count")
        }
        if descriptor.dimension == .d1 {
            metal.textureType = .type1D
        } else if descriptor.dimension == .d3 {
            metal.textureType = .type3D
            metal.depth = Int(descriptor.depth)
        } else if samples > 1 {
            metal.textureType = descriptor.arrayLayerCount > 1 ? .type2DMultisampleArray : .type2DMultisample
            metal.arrayLength = Int(descriptor.arrayLayerCount)
        } else if descriptor.arrayLayerCount > 1 {
            metal.textureType = .type2DArray
            metal.arrayLength = Int(descriptor.arrayLayerCount)
        }
        metal.sampleCount = samples
        metal.mipmapLevelCount = Int(descriptor.mipLevelCount)
        metal.storageMode = .private
        metal.usage = clearsTextureWithBlit(descriptor) ? [] : .renderTarget
        if descriptor.usage.contains(.textureBinding) ||
            (supportsPresentation && descriptor.format.isNormalizedColor && descriptor.usage.contains(.copySource)) {
            metal.usage.insert(.shaderRead)
        }
        if descriptor.format.hasStencil || descriptor.format.compression != nil ||
            ((descriptor.format.isNormalizedColor || descriptor.format.isSRGB8Color) && descriptor.usage.contains(.copyDestination)) {
            metal.usage.insert(.pixelFormatView)
        }
        if descriptor.usage.contains(.storageBinding) { metal.usage.formUnion([.shaderRead, .shaderWrite]) }
        if descriptor.usage.contains(.storageAtomic) {
            if #available(macOS 14, iOS 17, *) {
                metal.usage.formUnion([.shaderRead, .shaderWrite, .shaderAtomic])
            }
        }
        guard let texture = native.makeTexture(descriptor: metal) else { throw GPUError.outOfMemory }
        textures[id] = Texture(descriptor: descriptor, native: texture)
        #else
        throw GPUError.adapterUnavailable
        #endif
    }

    public func destroyTexture(id: ResourceID) {
        #if canImport(Metal)
        textures.removeValue(forKey: id)
        #endif
    }

    public func allocateTextureView(id: ResourceID, texture: ResourceID, mipLevels: Range<UInt32>) throws {
        try allocateTextureView(id: id, texture: texture, mipLevels: mipLevels, aspect: .all)
    }

    public func allocateTextureView(id: ResourceID, texture: ResourceID, mipLevels: Range<UInt32>,
                                    aspect: TextureViewAspect) throws {
        try allocateTextureView(id: id, texture: texture, descriptor: .init(
            baseMipLevel: mipLevels.lowerBound, mipLevelCount: mipLevels.upperBound - mipLevels.lowerBound, aspect: aspect))
    }

    public func allocateTextureView(id: ResourceID, texture: ResourceID, descriptor: TextureViewDescriptor) throws {
        try requireActive()
        #if canImport(Metal)
        guard textureViews[id] == nil, buffers[id] == nil, textures[id] == nil, samplers[id] == nil,
              id.device == texture.device, let parent = textures[texture] else {
            throw GPUError.validation("Invalid Metal texture view")
        }
        let mipLevels = try descriptor.mipLevels(in: parent.descriptor)
        let layers = try descriptor.arrayLayers(in: parent.descriptor)
        let dimension = descriptor.resolvedDimension(in: parent.descriptor)
        let aspect = descriptor.aspect
        let view: any MTLTexture
        let multisampled = parent.descriptor.sampleCount > 1
        let format: MTLPixelFormat = aspect == .stencilOnly && parent.descriptor.format.hasDepth ? .x32_stencil8 : parent.native.pixelFormat
        if ((multisampled && parent.descriptor.arrayLayerCount == 1) || dimension == .d1) && format == parent.native.pixelFormat {
            // These textures have one mip and can share the handle unless reinterpreted.
            view = parent.native
        } else {
            let levels = Int(mipLevels.lowerBound)..<Int(mipLevels.upperBound)
            let slices = Int(layers.lowerBound)..<Int(layers.upperBound)
            let type: MTLTextureType = switch dimension {
            case .d2: multisampled ? .type2DMultisample : .type2D
            case .d2Array: multisampled ? .type2DMultisampleArray : .type2DArray
            case .cube: .typeCube
            case .cubeArray: .typeCubeArray
            case .d3: .type3D
            case .d1: .type1D
            }
            guard let nativeView = parent.native.makeTextureView(
                pixelFormat: format, textureType: type, levels: levels, slices: slices) else {
                throw GPUError.backendFailure("Metal could not create a texture view")
            }
            view = nativeView
        }
        textureViews[id] = .init(resource: .init(texture: texture, levels: mipLevels, aspect: aspect,
                                                layers: layers, dimension: dimension), native: view)
        #else
        throw GPUError.adapterUnavailable
        #endif
    }

    public func destroyTextureView(id: ResourceID) {
        #if canImport(Metal)
        textureViews.removeValue(forKey: id)
        #endif
    }

    public func allocateSampler(id: ResourceID, descriptor: SamplerDescriptor) throws {
        try requireActive()
        #if canImport(Metal)
        guard samplers[id] == nil, buffers[id] == nil, textures[id] == nil, textureViews[id] == nil else {
            throw GPUError.validation("Duplicate Metal resource")
        }
        func addressMode(_ mode: SamplerAddressMode) -> MTLSamplerAddressMode {
            switch mode {
            case .clampToEdge: .clampToEdge
            case .clampToBorder: .clampToEdge  // never reached: Metal does not advertise addressModeClampToBorder
            case .repeat: .repeat
            case .mirrorRepeat: .mirrorRepeat
            }
        }
        let metal = MTLSamplerDescriptor()
        metal.sAddressMode = addressMode(descriptor.addressModeU)
        metal.tAddressMode = addressMode(descriptor.addressModeV)
        metal.rAddressMode = addressMode(descriptor.addressModeW)
        metal.minFilter = descriptor.minFilter == .linear ? .linear : .nearest
        metal.magFilter = descriptor.magFilter == .linear ? .linear : .nearest
        metal.mipFilter = descriptor.mipmapFilter == .linear ? .linear : .nearest
        metal.maxAnisotropy = Int(min(descriptor.maxAnisotropy, 16))
        metal.lodMinClamp = descriptor.lodMinClamp
        metal.lodMaxClamp = descriptor.lodMaxClamp
        if let compare = descriptor.compare { metal.compareFunction = compare.metal }
        guard let sampler = native.makeSamplerState(descriptor: metal) else { throw GPUError.outOfMemory }
        samplers[id] = .init(native: sampler, descriptor: descriptor)
        #else
        throw GPUError.adapterUnavailable
        #endif
    }

    public func destroySampler(id: ResourceID) {
        #if canImport(Metal)
        samplers.removeValue(forKey: id)
        #endif
    }
}
