import SAGATypes
import SAGABackend
import SAGAShaderIR
#if canImport(Metal)
import Metal
#endif
#if canImport(QuartzCore)
import QuartzCore
#endif

public struct MetalAdapter: Sendable {
    public let name: String
    #if canImport(Metal)
    fileprivate let native: any MTLDevice
    fileprivate init(_ native: any MTLDevice) { self.native = native; name = native.name }
    #endif
}

public enum MetalBackend {
    public static func requestAdapter() -> MetalAdapter? {
        #if canImport(Metal)
        MTLCreateSystemDefaultDevice().map(MetalAdapter.init)
        #else
        nil
        #endif
    }

    public static func makeDevice(timeout: Duration = .seconds(10)) throws -> MetalDevice {
        try makeDevice(adapter: requestAdapter(), timeout: timeout)
    }

    public static func makeDevice(adapter: MetalAdapter?, timeout: Duration = .seconds(10)) throws -> MetalDevice {
        guard let adapter else { throw GPUError.adapterUnavailable }
        return try MetalDevice(adapter: adapter, timeout: timeout, surface: nil)
    }

    #if canImport(Metal) && canImport(QuartzCore)
    @MainActor
    public static func makeDevice(
        surface: MetalPresentationSurface, adapter: MetalAdapter? = requestAdapter(),
        timeout: Duration = .seconds(10)
    ) throws -> MetalDevice {
        guard let adapter else { throw GPUError.adapterUnavailable }
        let backend = try MetalDevice(adapter: adapter, timeout: timeout, surface: surface)
        try surface.attach(to: adapter.native)
        return backend
    }
    #endif
}

@MainActor
public final class MetalDevice: ComputeBackendDevice, PresentationBackendDevice, SamplerBackendDevice, TextureViewBackendDevice {
    public nonisolated let capabilities: DeviceCapabilities
    public nonisolated let adapterName: String
    public nonisolated let supportsPresentation: Bool
    private let timeout: Duration
    private var terminal = false
    #if canImport(Metal)
    private let native: any MTLDevice
    private let queue: any MTLCommandQueue
    private let surface: MetalPresentationSurface?
    private struct Buffer {
        let descriptor: BufferDescriptor
        let native: any MTLBuffer
    }
    private struct Batch {
        let commandBuffer: any MTLCommandBuffer
        let buffers: [any MTLBuffer]
        let pipelines: [any MTLComputePipelineState]
        let textures: [any MTLTexture]
        let samplers: [any MTLSamplerState]
        let renderPipelines: [RenderState]
        let drawable: (any MTLDrawable)?
    }
    private struct RenderState {
        let pipeline: any MTLRenderPipelineState
        let depth: (any MTLDepthStencilState)?
        let writesDepth: Bool
    }

    private struct TextureSubresource: Hashable {
        let level: UInt32
        let layer: UInt32
    }

    private struct Texture {
        let descriptor: TextureDescriptor
        let native: any MTLTexture
        var initializedSubresources: Set<TextureSubresource> = []
    }
    private struct Sampler {
        let native: any MTLSamplerState
        let descriptor: SamplerDescriptor
    }

    private struct TextureView {
        let resource: TextureViewResource
        let native: any MTLTexture
    }
    private struct Pipeline {
        let state: any MTLComputePipelineState
        let threadsPerThreadgroup: MTLSize
        let bufferIndices: [ComputeBindingSlot: Int]
        let textureIndices: [ComputeBindingSlot: Int]
        let samplerIndices: [ComputeBindingSlot: Int]
        let runtimeLengths: [(slot: ComputeBindingSlot, stride: UInt32, offset: UInt32)]
        let lengthBufferIndex: UInt32?
    }
    private var buffers: [ResourceID: Buffer] = [:]
    private var pipelines: [ResourceID: any MTLComputePipelineState] = [:]
    private var computePipelines: [ResourceID: Pipeline] = [:]
    private var textures: [ResourceID: Texture] = [:]
    private var samplers: [ResourceID: Sampler] = [:]
    private var textureViews: [ResourceID: TextureView] = [:]
    private var renderPipelines: [ResourceID: RenderState] = [:]
    private var surfacePipeline: (any MTLRenderPipelineState)?
    private var presenting = false
    private var indexValidator: MetalIndexValidator?
    private var cachedDiagnosticPipeline: (any MTLComputePipelineState)?
    private var retirementHook: (@Sendable () async -> Void)?
    private var retirementWaiters: [CheckedContinuation<Void, Never>] = []
    private var nextPipeline: UInt64 = 0
    private var nextBatch: UInt64 = 0
    private var inFlight: [UInt64: Batch] = [:]
    #endif

    nonisolated fileprivate init(adapter: MetalAdapter, timeout: Duration, surface: MetalPresentationSurface?) throws {
        guard timeout > .zero else { throw GPUError.validation("Timeout must be positive") }
        self.timeout = timeout
        adapterName = adapter.name
        supportsPresentation = surface != nil
        #if canImport(Metal)
        native = adapter.native
        self.surface = surface
        guard let queue = adapter.native.makeCommandQueue() else { throw GPUError.outOfMemory }
        self.queue = queue
        var features: Set<GPUFeature> = [.storageTextureRead]
        if adapter.native.readWriteTextureSupport != .tierNone { features.insert(.storageTextureReadWrite) }
        capabilities = try DeviceCapabilities(maxBufferSize: UInt64(adapter.native.maxBufferLength), features: features)
        #else
        throw GPUError.adapterUnavailable
        #endif
    }

    private func requireActive() throws {
        guard !terminal else { throw GPUError.deviceLost("Metal device is terminal after command failure/timeout") }
    }

    public func allocateBuffer(id: ResourceID, descriptor: BufferDescriptor) throws {
        try requireActive()
        try descriptor.validate(against: capabilities)
        guard !descriptor.mappedAtCreation else { throw GPUError.unsupported("Mapped-at-creation API not implemented") }
        #if canImport(Metal)
        guard buffers[id] == nil, textures[id] == nil, samplers[id] == nil, textureViews[id] == nil, let length = Int(exactly: descriptor.size) else {
            throw GPUError.validation("Duplicate buffer or unrepresentable size")
        }
        guard let buffer = native.makeBuffer(length: max(1, length), options: .storageModeShared) else {
            throw GPUError.outOfMemory
        }
        buffer.contents().initializeMemory(as: UInt8.self, repeating: 0, count: max(1, length))
        buffer.label = descriptor.label
        buffers[id] = Buffer(descriptor: descriptor, native: buffer)
        #else
        throw GPUError.adapterUnavailable
        #endif
    }

    public func destroyBuffer(id: ResourceID) {
        #if canImport(Metal)
        buffers.removeValue(forKey: id)
        #endif
    }

    public func close(owner: UInt64) {
        #if canImport(Metal)
        buffers = buffers.filter { $0.key.device != owner }
        textureViews = textureViews.filter { $0.key.device != owner }
        textures = textures.filter { $0.key.device != owner }
        samplers = samplers.filter { $0.key.device != owner }
        pipelines = pipelines.filter { $0.key.device != owner }
        computePipelines = computePipelines.filter { $0.key.device != owner }
        renderPipelines = renderPipelines.filter { $0.key.device != owner }
        #endif
    }

    public func allocateComputePipeline(_ pipeline: ComputePipeline, shader: ComputeShaderArtifact) throws {
        try requireActive()
        #if canImport(Metal)
        guard case .text(.msl, let source) = shader.artifact else {
            throw GPUError.validation("Metal requires a compiled MSL compute shader")
        }
        try capabilities.require(shader.storageTextureFeatures)
        let reflected = try ComputePipeline(id: pipeline.id, shader: shader)
        guard pipeline == reflected else {
            throw GPUError.validation("Metal compute pipeline layout does not match compiled shader")
        }
        guard computePipelines[pipeline.id] == nil else { throw GPUError.validation("Duplicate Metal compute pipeline") }
        var bufferIndices: [ComputeBindingSlot: Int] = [:]
        var occupied = Set<UInt32>()
        for binding in shader.storageBindings {
            let slot = ComputeBindingSlot(group: binding.binding.group, binding: binding.binding.binding)
            guard let index = binding.flatBindingIndex, index < ComputeShaderArtifact.storageBindingCapacity,
                  occupied.insert(index).inserted, bufferIndices[slot] == nil
            else { throw GPUError.validation("Invalid Metal compute binding layout") }
            bufferIndices[slot] = Int(index)
        }
        let uniformBase = ComputeShaderArtifact.uniformBindingBase
        let uniformLimit = uniformBase + ComputeShaderArtifact.uniformBindingCapacity
        for binding in shader.uniformBindings {
            let slot = ComputeBindingSlot(group: binding.binding.group, binding: binding.binding.binding)
            guard let index = binding.flatBindingIndex, (uniformBase..<uniformLimit).contains(index),
                  occupied.insert(index).inserted, bufferIndices[slot] == nil
            else { throw GPUError.validation("Invalid Metal uniform binding layout") }
            bufferIndices[slot] = Int(index)
        }

        var textureIndices: [ComputeBindingSlot: Int] = [:]
        var occupiedTextures = Set<UInt32>()
        for binding in shader.sampledTextures {
            let slot = ComputeBindingSlot(group: binding.binding.group, binding: binding.binding.binding)
            guard let index = binding.flatBindingIndex, index < ComputeShaderArtifact.sampledTextureBindingCapacity,
                  occupiedTextures.insert(index).inserted, textureIndices[slot] == nil, bufferIndices[slot] == nil
            else { throw GPUError.validation("Invalid Metal sampled texture layout") }
            textureIndices[slot] = Int(index)
        }

        let storageBase = ComputeShaderArtifact.sampledTextureBindingCapacity
        let storageLimit = storageBase + ComputeShaderArtifact.storageTextureBindingCapacity
        for binding in shader.storageTextures {
            let slot = ComputeBindingSlot(group: binding.binding.group, binding: binding.binding.binding)
            guard let index = binding.flatBindingIndex,
                  (storageBase..<storageLimit).contains(index), occupiedTextures.insert(index).inserted,
                  textureIndices[slot] == nil, bufferIndices[slot] == nil
            else { throw GPUError.validation("Invalid Metal storage texture layout") }
            textureIndices[slot] = Int(index)
        }

        var samplerIndices: [ComputeBindingSlot: Int] = [:]
        var occupiedSamplers = Set<UInt32>()
        for binding in shader.samplerBindings {
            let slot = ComputeBindingSlot(group: binding.binding.group, binding: binding.binding.binding)
            guard let index = binding.flatBindingIndex, index < ComputeShaderArtifact.samplerBindingCapacity,
                  occupiedSamplers.insert(index).inserted, samplerIndices[slot] == nil,
                  bufferIndices[slot] == nil, textureIndices[slot] == nil
            else { throw GPUError.validation("Invalid Metal sampler layout") }
            samplerIndices[slot] = Int(index)
        }

        let size = shader.workgroupSize
        let threads = MTLSize(width: Int(size.x), height: Int(size.y), depth: Int(size.z))
        let limit = native.maxThreadsPerThreadgroup
        guard (1...limit.width).contains(threads.width),
              (1...limit.height).contains(threads.height),
              (1...limit.depth).contains(threads.depth)
        else { throw GPUError.validation("Workgroup dimensions exceed Metal device limits") }
        guard shader.workgroupStorageByteSize <= UInt64(native.maxThreadgroupMemoryLength) else {
            throw GPUError.validation("Workgroup storage exceeds Metal device limits")
        }
        let state: any MTLComputePipelineState
        do {
            let library = try native.makeLibrary(source: source, options: nil)
            guard let function = library.makeFunction(name: shader.entryPoint) else {
                throw GPUError.backendFailure("Compiled Metal entry point missing")
            }
            state = try native.makeComputePipelineState(function: function)
        } catch let error as GPUError {
            throw error
        } catch {
            throw GPUError.backendFailure("Compute MSL compilation: \(error)")
        }
        guard threads.width * threads.height * threads.depth <= state.maxTotalThreadsPerThreadgroup else {
            throw GPUError.validation("Workgroup size exceeds Metal pipeline limits")
        }
        let runtimeLengths = shader.storageBindings.compactMap { binding in
            binding.runtimeArrayStride.map {
                (ComputeBindingSlot(group: binding.binding.group, binding: binding.binding.binding),
                 $0, binding.runtimeArrayOffset)
            }
        }
        guard runtimeLengths.isEmpty == (shader.metalRuntimeLengthBuffer == nil),
              runtimeLengths.allSatisfy({ $0.1 > 0 }),
              shader.metalRuntimeLengthBuffer == nil ||
                  shader.metalRuntimeLengthBuffer == ComputeShaderArtifact.storageBindingCapacity
        else { throw GPUError.validation("Invalid Metal runtime array layout") }
        computePipelines[pipeline.id] = Pipeline(state: state, threadsPerThreadgroup: threads,
                                                bufferIndices: bufferIndices, textureIndices: textureIndices,
                                                samplerIndices: samplerIndices, runtimeLengths: runtimeLengths,
                                                lengthBufferIndex: shader.metalRuntimeLengthBuffer)
        #else
        throw GPUError.adapterUnavailable
        #endif
    }

    public func destroyComputePipeline(id: ResourceID) {
        #if canImport(Metal)
        computePipelines.removeValue(forKey: id)
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
        if descriptor.dimension == .d3 {
            metal.textureType = .type3D
            metal.depth = Int(descriptor.depth)
        } else if samples > 1 {
            metal.textureType = .type2DMultisample
        } else if descriptor.arrayLayerCount > 1 {
            metal.textureType = .type2DArray
            metal.arrayLength = Int(descriptor.arrayLayerCount)
        }
        metal.sampleCount = samples
        metal.mipmapLevelCount = Int(descriptor.mipLevelCount)
        metal.storageMode = .private
        metal.usage = .renderTarget
        if descriptor.usage.contains(.textureBinding) ||
            (supportsPresentation && descriptor.format.isNormalizedColor && descriptor.usage.contains(.copySource)) {
            metal.usage.insert(.shaderRead)
        }
        if descriptor.format.hasStencil { metal.usage.insert(.pixelFormatView) }
        if descriptor.usage.contains(.storageBinding) { metal.usage.formUnion([.shaderRead, .shaderWrite]) }
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
        let format: MTLPixelFormat = aspect == .stencilOnly ? .x32_stencil8 : parent.native.pixelFormat
        if multisampled && format == parent.native.pixelFormat {
            // Multisampled 2D textures have one mip and can share the handle unless reinterpreted.
            view = parent.native
        } else {
            let levels = Int(mipLevels.lowerBound)..<Int(mipLevels.upperBound)
            let slices = Int(layers.lowerBound)..<Int(layers.upperBound)
            let type: MTLTextureType = switch dimension {
            case .d2: multisampled ? .type2DMultisample : .type2D
            case .d2Array: .type2DArray
            case .cube: .typeCube
            case .cubeArray: .typeCubeArray
            case .d3: .type3D
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
            case .repeat: .repeat
            case .mirrorRepeat: .mirrorRepeat
            }
        }
        let metal = MTLSamplerDescriptor()
        metal.sAddressMode = addressMode(descriptor.addressModeU)
        metal.tAddressMode = addressMode(descriptor.addressModeV)
        metal.minFilter = descriptor.minFilter == .linear ? .linear : .nearest
        metal.magFilter = descriptor.magFilter == .linear ? .linear : .nearest
        metal.mipFilter = descriptor.mipmapFilter == .linear ? .linear : .nearest
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

    public func allocateRenderPipeline(
        _ pipeline: RenderPipeline, vertex: RenderStageArtifact, fragment: RenderStageArtifact
    ) throws {
        try requireActive()
        #if canImport(Metal)
        guard renderPipelines[pipeline.id] == nil, vertex.stage == .vertex, fragment.stage == .fragment,
              case .text(.msl, let vertexSource) = vertex.artifact,
              case .text(.msl, let fragmentSource) = fragment.artifact else {
            throw GPUError.validation("Metal requires compiled MSL vertex and fragment stages")
        }
        let features = vertex.storageTextures.requiredStorageTextureFeatures
            .union(fragment.storageTextures.requiredStorageTextureFeatures)
        try capabilities.require(features)
        let reflected = try RenderPipeline(id: pipeline.id, colorFormat: pipeline.colorFormat,
                                           vertexBuffers: pipeline.vertexBuffers, depthState: pipeline.depthState,
                                           sampleCount: pipeline.sampleCount, sampleMask: pipeline.sampleMask,
                                           alphaToCoverageEnabled: pipeline.alphaToCoverageEnabled, primitive: pipeline.primitive,
                                           colorState: pipeline.colorState, additionalColorTargets: pipeline.additionalColorTargets,
                                           vertex: vertex, fragment: fragment)
        guard pipeline == reflected else {
            throw GPUError.validation("Metal render pipeline layout does not match compiled shaders")
        }
        let uniformBase = ComputeShaderArtifact.uniformBindingBase
        let uniformLimit = uniformBase + ComputeShaderArtifact.uniformBindingCapacity
        for stage in [vertex, fragment] {
            let indices = stage.uniformBindings.compactMap(\.flatBindingIndex)
            guard indices.count == stage.uniformBindings.count, Set(indices).count == indices.count,
                  indices.allSatisfy({ (uniformBase..<uniformLimit).contains($0) && Int($0) >= pipeline.vertexBuffers.count })
            else { throw GPUError.validation("Invalid Metal render uniform layout") }
            let textureIndices = stage.sampledTextures.compactMap(\.flatBindingIndex)
            guard textureIndices.count == stage.sampledTextures.count, Set(textureIndices).count == textureIndices.count,
                  textureIndices.allSatisfy({ $0 < ComputeShaderArtifact.sampledTextureBindingCapacity })
            else { throw GPUError.validation("Invalid Metal render sampled texture layout") }
            let storageBase = ComputeShaderArtifact.sampledTextureBindingCapacity
            let storageLimit = storageBase + ComputeShaderArtifact.storageTextureBindingCapacity
            let storageIndices = stage.storageTextures.compactMap(\.flatBindingIndex)
            guard storageIndices.count == stage.storageTextures.count, Set(storageIndices).count == storageIndices.count,
                  storageIndices.allSatisfy({ (storageBase..<storageLimit).contains($0) })
            else { throw GPUError.validation("Invalid Metal render storage texture layout") }
            let samplerIndices = stage.samplerBindings.compactMap(\.flatBindingIndex)
            guard samplerIndices.count == stage.samplerBindings.count, Set(samplerIndices).count == samplerIndices.count,
                  samplerIndices.allSatisfy({ $0 < ComputeShaderArtifact.samplerBindingCapacity })
            else { throw GPUError.validation("Invalid Metal render sampler layout") }
        }
        do {
            let vertexOptions: MTLCompileOptions?
            if pipeline.primitive.topology == .pointList {
                let options = MTLCompileOptions()
                options.preprocessorMacros = [RenderStageArtifact.metalPointTopologyMacro: NSNumber(value: 1)]
                vertexOptions = options
            } else {
                vertexOptions = nil
            }
            let vs = try native.makeLibrary(source: vertexSource, options: vertexOptions)
            let activeSamples = (UInt32(1) << pipeline.sampleCount) - 1
            let sampleMask = pipeline.sampleMask & activeSamples
            let fragmentOptions: MTLCompileOptions?
            if sampleMask != activeSamples {
                let options = MTLCompileOptions()
                options.preprocessorMacros = [RenderStageArtifact.metalPipelineSampleMaskMacro: NSNumber(value: sampleMask)]
                fragmentOptions = options
            } else {
                fragmentOptions = nil
            }
            let fs = try native.makeLibrary(source: fragmentSource, options: fragmentOptions)
            guard let vertexFunction = vs.makeFunction(name: vertex.entryPoint),
                  let fragmentFunction = fs.makeFunction(name: fragment.entryPoint) else {
                throw GPUError.backendFailure("Compiled Metal render entry point missing")
            }
            let descriptor = MTLRenderPipelineDescriptor()
            let samples = Int(pipeline.sampleCount)
            guard native.supportsTextureSampleCount(samples) else {
                throw GPUError.unsupported("Metal does not support the render pipeline sample count")
            }
            descriptor.rasterSampleCount = samples
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
                descriptor.depthAttachmentPixelFormat = state.format.metal
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
            if !pipeline.vertexBuffers.isEmpty {
                let vertexDescriptor = MTLVertexDescriptor()
                for (slot, layout) in pipeline.vertexBuffers.enumerated() {
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
            renderPipelines[pipeline.id] = RenderState(pipeline: state, depth: depth, writesDepth: fragment.writesDepth)
        } catch let error as GPUError {
            throw error
        } catch {
            throw GPUError.backendFailure("Render MSL compilation: \(error)")
        }
        #else
        throw GPUError.adapterUnavailable
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

    public func submit(_ submission: Submission) async throws {
        _ = try await submitCommands(submission, commands: [])
    }

    public func present(_ submission: Submission, source: ResourceID) async throws {
        try requireActive()
        #if canImport(Metal) && canImport(QuartzCore)
        guard let surface else { throw GPUError.unsupported("Metal backend does not have a presentation surface") }
        guard !presenting else { throw GPUError.validation("Metal surface already has a pending frame") }
        guard submission.resources.contains(source), let texture = textures[source],
              texture.descriptor.format.isNormalizedColor, texture.descriptor.sampleCount == 1,
              texture.descriptor.dimension == .d2, texture.descriptor.arrayLayerCount == 1,
              texture.descriptor.usage.contains(.copySource) else {
            throw GPUError.validation("Unpinned or invalid Metal presentation source")
        }
        let descriptor = texture.descriptor
        presenting = true
        defer { presenting = false }
        let pipeline: any MTLRenderPipelineState
        if let surfacePipeline {
            pipeline = surfacePipeline
        } else {
            pipeline = try MetalSurfaceCopy.makePipeline(device: native)
            surfacePipeline = pipeline
        }
        let drawable = try surface.acquire(width: descriptor.width, height: descriptor.height)
        guard drawable.texture.width == Int(descriptor.width), drawable.texture.height == Int(descriptor.height) else {
            throw GPUError.backendFailure("Metal drawable size did not match the presentation source")
        }
        guard let commands = queue.makeCommandBuffer() else { throw GPUError.outOfMemory }
        let clear = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        var initialized: [ResourceID: Set<TextureSubresource>] = [:]
        try initializeTexture(source, level: 0, on: commands, initialized: &initialized)
        let encoder = try renderEncoder(commands, target: drawable.texture, clear: clear)
        encoder.setRenderPipelineState(pipeline)
        encoder.setFragmentTexture(texture.native, index: 0)
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        encoder.endEncoding()
        commands.present(drawable)
        try await commit(Batch(commandBuffer: commands, buffers: [], pipelines: [], textures: [texture.native],
                               samplers: [], renderPipelines: [.init(pipeline: pipeline, depth: nil, writesDepth: false)], drawable: drawable), initialized: initialized)
        #else
        throw GPUError.unsupported("Metal presentation is unavailable")
        #endif
    }

    public func submitCommands(_ submission: Submission, commands: [BufferCommand]) async throws -> [[UInt32]] {
        try requireActive()
        #if canImport(Metal)
        let owner = submission.resources.first?.device ??
            commands.lazy.compactMap { $0.computeDispatch?.pipeline.id.device }.first ?? 0
        let descriptors = buffers.mapValues(\.descriptor)
        let textureDescriptors = textures.mapValues(\.descriptor)
        let samplerDescriptors = samplers.mapValues(\.descriptor)
        let views = textureViews.mapValues(\.resource)
        for id in submission.resources {
            guard buffers[id] != nil || textures[id] != nil || views[id] != nil || samplers[id] != nil else {
                throw GPUError.validation("Unknown Metal resource")
            }
            if let view = views[id], !submission.resources.contains(view.texture) {
                throw GPUError.validation("Unpinned texture view parent")
            }
        }
        for command in commands {
            guard command.resources.allSatisfy({ submission.resources.contains($0) }) else {
                throw GPUError.validation("Unpinned command resource")
            }
            try command.validate(device: owner, descriptors: descriptors, textures: textureDescriptors,
                                 samplerDescriptors: samplerDescriptors, textureViews: views,
                                 maximumViewportDimension: capabilities.maxTextureDimension2D)
            if case .dispatch(let id, _, _, _) = command, pipelines[id] == nil {
                throw GPUError.validation("Unknown Metal diagnostic pipeline")
            }
            if let dispatch = command.computeDispatch, computePipelines[dispatch.pipeline.id] == nil {
                throw GPUError.validation("Unknown Metal compute pipeline")
            }
            switch command {
            case .render(let pipeline, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _),
                 .renderIndexed(let pipeline, _, _, _, _, _, _, _, _, _, _, _, _, _, _):
                guard renderPipelines[pipeline.id] != nil else {
                    throw GPUError.validation("Unknown Metal render pipeline")
                }
            default: break
            }
        }
        guard let commandBuffer = queue.makeCommandBuffer() else { throw GPUError.outOfMemory }
        var retained = submission.resources.compactMap { buffers[$0]?.native }
        var retainedPipelines: [any MTLComputePipelineState] = []
        var retainedRenderPipelines: [RenderState] = []
        var initialized: [ResourceID: Set<TextureSubresource>] = [:]
        var reads: [(any MTLBuffer, Int)] = []
        for command in commands {
            switch command {
            case .write(let id, let offset, let words):
                let length = words.count * 4
                guard let staging = native.makeBuffer(length: length, options: .storageModeShared),
                      let encoder = commandBuffer.makeBlitCommandEncoder() else { throw GPUError.outOfMemory }
                words.withUnsafeBytes { staging.contents().copyMemory(from: $0.baseAddress!, byteCount: length) }
                encoder.copy(from: staging, sourceOffset: 0, to: buffers[id]!.native,
                             destinationOffset: Int(offset), size: length)
                encoder.endEncoding()
                retained.append(staging)
            case .dispatch(let id, let input, let output, let count):
                let pipeline = pipelines[id]!
                guard let encoder = commandBuffer.makeComputeCommandEncoder() else { throw GPUError.outOfMemory }
                encoder.setComputePipelineState(pipeline)
                encoder.setBuffer(buffers[input]!.native, offset: 0, index: 0)
                encoder.setBuffer(buffers[output]!.native, offset: 0, index: 1)
                var count = count
                encoder.setBytes(&count, length: 4, index: 2)
                let width = min(64, pipeline.maxTotalThreadsPerThreadgroup)
                let groups = (Int(count) + width - 1) / width
                encoder.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
                encoder.endEncoding()
                retainedPipelines.append(pipeline)
            case .dispatchCompute, .dispatchComputeRanges:
                let (pipeline, bindings, workgroups, boundTextures, samplerBindings) = command.computeDispatch!
                if workgroups.isEmpty { continue }
                let compute = computePipelines[pipeline.id]!
                let lengths = try compute.runtimeLengths.map { slot, stride, offset in
                    let binding = bindings[slot]!
                    let range = try binding.range(in: descriptors[binding.buffer]!)
                    let size = range.upperBound - range.lowerBound
                    guard size >= UInt64(offset) + UInt64(stride),
                          let count = UInt32(exactly: (size - UInt64(offset)) / UInt64(stride)) else {
                        throw GPUError.validation("Invalid Metal runtime array length")
                    }
                    return count
                }
                try initializeBoundTextures(boundTextures, on: commandBuffer, initialized: &initialized)
                guard let encoder = commandBuffer.makeComputeCommandEncoder() else { throw GPUError.outOfMemory }
                encoder.setComputePipelineState(compute.state)
                for (slot, index) in compute.textureIndices {
                    encoder.setTexture(nativeTexture(boundTextures[slot]!), index: index)
                }
                for (slot, index) in compute.samplerIndices {
                    encoder.setSamplerState(samplers[samplerBindings[slot]!]!.native, index: index)
                }
                for (slot, index) in compute.bufferIndices {
                    let binding = bindings[slot]!
                    encoder.setBuffer(buffers[binding.buffer]!.native, offset: Int(binding.offset),
                                      index: index)
                }
                if let index = compute.lengthBufferIndex {
                    lengths.withUnsafeBytes {
                        encoder.setBytes($0.baseAddress!, length: $0.count, index: Int(index))
                    }
                }
                encoder.dispatchThreadgroups(
                    MTLSize(width: Int(workgroups.x), height: Int(workgroups.y), depth: Int(workgroups.z)),
                    threadsPerThreadgroup: compute.threadsPerThreadgroup)
                encoder.endEncoding()
                retainedPipelines.append(compute.state)
            case .copy(let source, let destination, let size):
                guard let encoder = commandBuffer.makeBlitCommandEncoder() else { throw GPUError.outOfMemory }
                encoder.copy(from: buffers[source]!.native, sourceOffset: 0,
                             to: buffers[destination]!.native, destinationOffset: 0, size: Int(size))
                encoder.endEncoding()
            case .render(let pipeline, let target, let colorLoad, let vertexCount, let vertexBuffers, let firstVertex, let instances, let uniforms, let boundTextures, let samplerBindings, let targetMipLevel, let depth, let resolve, let blendConstant, let additionalColors, let raster):
                let state = renderPipelines[pipeline.id]!
                let drawEnabled = vertexCount > 0 && instances.count > 0 && !raster.hasEmptyScissor
                if drawEnabled {
                    try initializeBoundTextures(boundTextures, on: commandBuffer, initialized: &initialized)
                }
                try initializeLoadedAttachments(target: target, level: targetMipLevel, colorLoad: colorLoad,
                                                depth: depth, additionalColors: additionalColors,
                                                on: commandBuffer, initialized: &initialized)
                let clear = colorLoad.clearColor.map { MTLClearColor(red: $0.red, green: $0.green, blue: $0.blue, alpha: $0.alpha) }
                let encoder = try renderEncoder(
                    commandBuffer, target: nativeTexture(target), level: targetMipLevel,
                    clear: clear, depth: depth, resolve: resolve, additionalColors: additionalColors)
                if drawEnabled {
                    bindRender(encoder, state: state, pipeline: pipeline, vertexBuffers: vertexBuffers,
                               uniforms: uniforms, boundTextures: boundTextures, samplerBindings: samplerBindings,
                               blendConstant: blendConstant, raster: raster)
                    if let stencil = depth?.stencil { encoder.setStencilReferenceValue(stencil.reference & 0xff) }
                    encoder.drawPrimitives(type: pipeline.primitive.topology.metal.type, vertexStart: Int(firstVertex), vertexCount: Int(vertexCount),
                                           instanceCount: Int(instances.count), baseInstance: Int(instances.first))
                }
                encoder.endEncoding()
                retainedRenderPipelines.append(state)
                markRenderedAttachments(target: target, level: targetMipLevel, depth: depth, resolve: resolve,
                                        additionalColors: additionalColors, initialized: &initialized)
            case .renderIndexed(let pipeline, let target, let colorLoad, let index, let draw, let vertexBuffers, let uniforms, let boundTextures, let samplerBindings, let targetMipLevel, let depth, let resolve, let blendConstant, let additionalColors, let raster):
                let state = renderPipelines[pipeline.id]!
                let indexBuffer = buffers[index.buffer]!.native
                let byteOffset = Int(index.offset + UInt64(draw.firstIndex) * UInt64(index.format.byteSize))
                var indirect: (any MTLBuffer)?
                var drawEnabled = draw.indexCount > 0 && draw.instances.count > 0 && !raster.hasEmptyScissor
                if drawEnabled && pipeline.vertexBuffers.contains(where: { $0.stepMode == .vertex }) {
                    if let maximumVertex = maximumVertexIndex(pipeline.vertexBuffers, bindings: vertexBuffers) {
                        if indexValidator == nil { indexValidator = try MetalIndexValidator(device: native) }
                        let args = try indexValidator!.encode(
                            on: commandBuffer, device: native, indices: indexBuffer, format: index.format,
                            byteOffset: byteOffset, draw: draw, maximumVertex: maximumVertex,
                            primitiveRestart: pipeline.primitive.topology.isStrip)
                        indirect = args
                        retained.append(args)
                    } else {
                        drawEnabled = false
                    }
                }
                if drawEnabled {
                    try initializeBoundTextures(boundTextures, on: commandBuffer, initialized: &initialized)
                }
                try initializeLoadedAttachments(target: target, level: targetMipLevel, colorLoad: colorLoad,
                                                depth: depth, additionalColors: additionalColors,
                                                on: commandBuffer, initialized: &initialized)
                let clear = colorLoad.clearColor.map { MTLClearColor(red: $0.red, green: $0.green, blue: $0.blue, alpha: $0.alpha) }
                let encoder = try renderEncoder(
                    commandBuffer, target: nativeTexture(target), level: targetMipLevel,
                    clear: clear, depth: depth, resolve: resolve, additionalColors: additionalColors)
                if drawEnabled {
                    bindRender(encoder, state: state, pipeline: pipeline, vertexBuffers: vertexBuffers,
                               uniforms: uniforms, boundTextures: boundTextures, samplerBindings: samplerBindings,
                               blendConstant: blendConstant, raster: raster)
                    if let stencil = depth?.stencil { encoder.setStencilReferenceValue(stencil.reference & 0xff) }
                    if let indirect {
                        encoder.drawIndexedPrimitives(type: pipeline.primitive.topology.metal.type, indexType: index.format.metal,
                                                      indexBuffer: indexBuffer, indexBufferOffset: byteOffset,
                                                      indirectBuffer: indirect, indirectBufferOffset: 0)
                    } else {
                        encoder.drawIndexedPrimitives(type: pipeline.primitive.topology.metal.type, indexCount: Int(draw.indexCount),
                                                      indexType: index.format.metal, indexBuffer: indexBuffer,
                                                      indexBufferOffset: byteOffset, instanceCount: Int(draw.instances.count),
                                                      baseVertex: Int(draw.baseVertex), baseInstance: Int(draw.instances.first))
                    }
                }
                encoder.endEncoding()
                retainedRenderPipelines.append(state)
                markRenderedAttachments(target: target, level: targetMipLevel, depth: depth, resolve: resolve,
                                        additionalColors: additionalColors, initialized: &initialized)
            case .copyTextureToBuffer(let source, let destination, let bytesPerRow, let mipLevel, let aspect, let offset,
                                      let layer, let z, let depth):
                let texture = textures[source]!
                let extent = texture.descriptor.mipExtent(mipLevel)!
                let region = TextureCopyRegion(width: extent.width, height: extent.height, mipLevel: mipLevel,
                                               arrayLayer: layer, z: z, depth: depth)
                for plane in region.planes(in: texture.descriptor) {
                    try initializeTexture(source, level: mipLevel, layer: plane, on: commandBuffer, initialized: &initialized)
                }
                guard let encoder = commandBuffer.makeBlitCommandEncoder() else { throw GPUError.outOfMemory }
                encoder.copy(from: texture.native, sourceSlice: Int(layer), sourceLevel: Int(mipLevel),
                             sourceOrigin: MTLOrigin(x: 0, y: 0, z: Int(z)),
                             sourceSize: MTLSize(width: Int(extent.width), height: Int(extent.height), depth: Int(depth)),
                             to: buffers[destination]!.native, destinationOffset: Int(offset),
                             destinationBytesPerRow: Int(bytesPerRow),
                             destinationBytesPerImage: texture.descriptor.dimension == .d3 ? Int(bytesPerRow) * Int(extent.height) : 0,
                             options: aspect == .stencilOnly ? .stencilFromDepthStencil : [])
                encoder.endEncoding()
            case .copyBufferToTexture(let source, let layout, let destination, let region, let aspect):
                if region.isEmpty { continue }
                let texture = textures[destination]!
                try prepareTextureWrite(destination, region: region, aspect: aspect, on: commandBuffer, initialized: &initialized)
                guard let encoder = commandBuffer.makeBlitCommandEncoder() else { throw GPUError.outOfMemory }
                let texelBytes = texture.descriptor.format.copyTexelBytes(aspect: aspect)!
                let stride = Int(layout.rowStride(width: region.width, texelBytes: texelBytes))
                encoder.copy(from: buffers[source]!.native, sourceOffset: Int(layout.offset),
                             sourceBytesPerRow: stride,
                             sourceBytesPerImage: texture.descriptor.dimension == .d3 ? stride * Int(region.height) : 0,
                             sourceSize: MTLSize(width: Int(region.width), height: Int(region.height), depth: Int(region.depth)),
                             to: texture.native, destinationSlice: Int(region.arrayLayer), destinationLevel: Int(region.mipLevel),
                             destinationOrigin: MTLOrigin(x: Int(region.x), y: Int(region.y), z: Int(region.z)),
                             options: aspect == .stencilOnly ? .stencilFromDepthStencil : [])
                encoder.endEncoding()
            case .copyTextureToTexture(let source, let destination, let region, let origin):
                if region.isEmpty { continue }
                let sourceTexture = textures[source]!
                for plane in region.planes(in: sourceTexture.descriptor) {
                    try initializeTexture(source, level: region.mipLevel, layer: plane, on: commandBuffer, initialized: &initialized)
                }
                let target = textures[destination]!
                let destinationRegion = origin.region(width: region.width, height: region.height, depth: region.depth)
                try prepareTextureWrite(destination, region: destinationRegion, aspect: .all, on: commandBuffer, initialized: &initialized)
                guard let encoder = commandBuffer.makeBlitCommandEncoder() else { throw GPUError.outOfMemory }
                encoder.copy(from: sourceTexture.native, sourceSlice: Int(region.arrayLayer), sourceLevel: Int(region.mipLevel),
                             sourceOrigin: MTLOrigin(x: Int(region.x), y: Int(region.y), z: Int(region.z)),
                             sourceSize: MTLSize(width: Int(region.width), height: Int(region.height), depth: Int(region.depth)),
                             to: target.native, destinationSlice: Int(origin.arrayLayer), destinationLevel: Int(origin.mipLevel),
                             destinationOrigin: MTLOrigin(x: Int(origin.x), y: Int(origin.y), z: Int(origin.z)))
                encoder.endEncoding()
            case .read(let id, let size):
                guard let snapshot = native.makeBuffer(length: Int(size), options: .storageModeShared),
                      let encoder = commandBuffer.makeBlitCommandEncoder() else { throw GPUError.outOfMemory }
                encoder.copy(from: buffers[id]!.native, sourceOffset: 0, to: snapshot,
                             destinationOffset: 0, size: Int(size))
                encoder.endEncoding()
                retained.append(snapshot)
                reads.append((snapshot, Int(size / 4)))
            }
        }
        try await commit(Batch(
            commandBuffer: commandBuffer, buffers: retained, pipelines: retainedPipelines,
            textures: submission.resources.compactMap { textures[$0]?.native ?? textureViews[$0]?.native },
            samplers: submission.resources.compactMap { samplers[$0]?.native }, renderPipelines: retainedRenderPipelines, drawable: nil), initialized: initialized)
        return reads.map { buffer, count in
            Array(UnsafeBufferPointer(start: buffer.contents().assumingMemoryBound(to: UInt32.self), count: count))
        }
        #else
        throw GPUError.adapterUnavailable
        #endif
    }

    #if canImport(Metal)
    private func nativeTexture(_ id: ResourceID) -> any MTLTexture {
        textureViews[id]?.native ?? textures[id]!.native
    }

    private func markRenderedTexture(_ id: ResourceID, level: UInt32, initialized: inout [ResourceID: Set<TextureSubresource>]) {
        let view = textureViews[id]?.resource
        initialized[view?.texture ?? id, default: []].insert(.init(level: view?.levels.lowerBound ?? level, layer: view?.layers.lowerBound ?? 0))
    }

    private func markRenderedAttachments(
        target: ResourceID, level: UInt32, depth: RenderDepthAttachment?, resolve: RenderResolveAttachment?,
        additionalColors: [UInt32: RenderColorAttachment], initialized: inout [ResourceID: Set<TextureSubresource>]
    ) {
        markRenderedTexture(target, level: level, initialized: &initialized)
        if let resolve { markRenderedTexture(resolve.texture, level: resolve.mipLevel, initialized: &initialized) }
        for color in additionalColors.values {
            markRenderedTexture(color.texture, level: color.mipLevel, initialized: &initialized)
            if let resolve = color.resolve {
                markRenderedTexture(resolve.texture, level: resolve.mipLevel, initialized: &initialized)
            }
        }
        if let depth { markRenderedTexture(depth.texture, level: depth.mipLevel, initialized: &initialized) }
    }

    private func initializeLoadedAttachments(
        target: ResourceID, level: UInt32, colorLoad: RenderColorLoadAction,
        depth: RenderDepthAttachment?, additionalColors: [UInt32: RenderColorAttachment], on commands: any MTLCommandBuffer,
        initialized: inout [ResourceID: Set<TextureSubresource>]
    ) throws {
        func initialize(_ id: ResourceID, level: UInt32) throws {
            let view = textureViews[id]?.resource
            try initializeTexture(view?.texture ?? id, level: view?.levels.lowerBound ?? level,
                                  layer: view?.layers.lowerBound ?? 0, on: commands, initialized: &initialized)
        }
        if case .load = colorLoad { try initialize(target, level: level) }
        for attachment in additionalColors.values where attachment.loadAction == .load {
            try initialize(attachment.texture, level: attachment.mipLevel)
        }
        if let depth {
            let loadsDepth = depth.loadAction == .load || depth.loadAction == .readOnly
            let loadsStencil = depth.stencil?.loadAction == .load || depth.stencil?.loadAction == .readOnly
            if loadsDepth || loadsStencil { try initialize(depth.texture, level: depth.mipLevel) }
        }
    }

    private func initializeBoundTextures(
        _ bindings: [ComputeBindingSlot: ResourceID], on commands: any MTLCommandBuffer,
        initialized: inout [ResourceID: Set<TextureSubresource>]
    ) throws {
        for id in Set(bindings.values) {
            let view = textureViews[id]?.resource
            let parent = view?.texture ?? id
            let levels = view?.levels ?? 0..<textures[parent]!.descriptor.mipLevelCount
            let descriptor = textures[parent]!.descriptor
            for level in levels {
                let layers = descriptor.dimension == .d3 ? 0..<descriptor.mipDepth(level) :
                    (view?.layers ?? 0..<descriptor.arrayLayerCount)
                for layer in layers {
                    try initializeTexture(parent, level: level, layer: layer, on: commands, initialized: &initialized)
                }
            }
        }
    }

    private func prepareTextureWrite(
        _ id: ResourceID, region: TextureCopyRegion, aspect: TextureViewAspect, on commands: any MTLCommandBuffer,
        initialized: inout [ResourceID: Set<TextureSubresource>]
    ) throws {
        let descriptor = textures[id]!.descriptor
        let fullPlane = aspect != .stencilOnly && region.covers(descriptor)
        for plane in region.planes(in: descriptor) {
            if fullPlane {
                initialized[id, default: []].insert(.init(level: region.mipLevel, layer: plane))
            } else {
                try initializeTexture(id, level: region.mipLevel, layer: plane, on: commands, initialized: &initialized)
            }
        }
    }

    private func initializeTexture(
        _ id: ResourceID, level: UInt32, layer: UInt32 = 0, on commands: any MTLCommandBuffer,
        initialized: inout [ResourceID: Set<TextureSubresource>]
    ) throws {
        let texture = textures[id]!
        // Only clear the subresource that could still expose uninitialized Metal memory.
        let subresource = TextureSubresource(level: level, layer: layer)
        guard !texture.initializedSubresources.contains(subresource),
              initialized[id, default: []].insert(subresource).inserted else { return }
        if texture.descriptor.format.hasDepth {
            let pass = MTLRenderPassDescriptor()
            setDepthAttachment(pass, texture: texture.native, level: level, layer: layer, loadAction: .clear(0),
                               stencilLoadAction: texture.descriptor.format.hasStencil ? .clear(0) : nil)
            guard let encoder = commands.makeRenderCommandEncoder(descriptor: pass) else { throw GPUError.outOfMemory }
            encoder.endEncoding()
        } else {
            let clear = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
            try renderEncoder(commands, target: texture.native, level: level, layer: layer, clear: clear).endEncoding()
        }
    }

    private func commit(_ batch: Batch, initialized: [ResourceID: Set<TextureSubresource>]) async throws {
        guard nextBatch < UInt64.max else { throw GPUError.outOfMemory }
        nextBatch += 1
        let serial = nextBatch
        let completion = BackendCompletion()
        inFlight[serial] = batch
        let retirementHook = retirementHook
        batch.commandBuffer.addCompletedHandler { [self] buffer in
            let result: Result<Void, GPUError> = buffer.status == .completed ? .success(()) :
                .failure(.deviceLost("Metal command failed: \(String(describing: buffer.error))"))
            Task {
                if let retirementHook { await retirementHook() }
                await retire(serial, completion: completion, result: result)
            }
        }
        batch.commandBuffer.commit()
        for (id, levels) in initialized { textures[id]?.initializedSubresources.formUnion(levels) }
        do { try await completion.wait(timeout: timeout) }
        catch { terminal = true; throw error }
    }

    private func bindRender(
        _ encoder: any MTLRenderCommandEncoder, state: RenderState, pipeline: RenderPipeline,
        vertexBuffers: [VertexBufferBinding], uniforms: [ComputeBindingSlot: ComputeBufferBinding],
        boundTextures: [ComputeBindingSlot: ResourceID], samplerBindings: [ComputeBindingSlot: ResourceID],
        blendConstant: RenderClearColor, raster: RenderRasterState
    ) {
        encoder.setRenderPipelineState(state.pipeline)
        if let viewport = raster.viewport {
            encoder.setViewport(MTLViewport(originX: Double(viewport.x), originY: Double(viewport.y),
                                            width: Double(viewport.width), height: Double(viewport.height),
                                            znear: Double(viewport.minDepth), zfar: Double(viewport.maxDepth)))
        }
        if let scissor = raster.scissor {
            encoder.setScissorRect(MTLScissorRect(x: Int(scissor.x), y: Int(scissor.y),
                                                 width: Int(scissor.width), height: Int(scissor.height)))
        }
        if pipeline.colorState.blend != nil || pipeline.additionalColorTargets.values.contains(where: { $0.colorState.blend != nil }) {
            encoder.setBlendColor(red: Float(blendConstant.red), green: Float(blendConstant.green),
                                  blue: Float(blendConstant.blue), alpha: Float(blendConstant.alpha))
        }
        encoder.setFrontFacing(pipeline.primitive.frontFace == .clockwise ? .clockwise : .counterClockwise)
        switch pipeline.primitive.cullMode {
        case .none: encoder.setCullMode(.none)
        case .front: encoder.setCullMode(.front)
        case .back: encoder.setCullMode(.back)
        }
        if let depth = state.depth { encoder.setDepthStencilState(depth) }
        if let bias = pipeline.depthState?.bias, bias != .init() {
            encoder.setDepthBias(Float(bias.constant), slopeScale: bias.slopeScale, clamp: bias.clamp)
        }
        if state.writesDepth {
            var range = SIMD2<Float>(raster.viewport?.minDepth ?? 0, raster.viewport?.maxDepth ?? 1)
            encoder.setFragmentBytes(&range, length: MemoryLayout.size(ofValue: range),
                                     index: RenderStageArtifact.metalViewportDepthBufferIndex)
        }
        for (slot, binding) in vertexBuffers.enumerated() {
            encoder.setVertexBuffer(buffers[binding.buffer]!.native, offset: Int(binding.offset), index: slot)
        }
        for entry in pipeline.uniforms {
            let binding = uniforms[entry.slot]!
            let buffer = buffers[binding.buffer]!.native
            if let index = entry.vertexIndex {
                encoder.setVertexBuffer(buffer, offset: Int(binding.offset), index: Int(index))
            }
            if let index = entry.fragmentIndex {
                encoder.setFragmentBuffer(buffer, offset: Int(binding.offset), index: Int(index))
            }
        }
        for entry in pipeline.sampledTextures.map(\.handle) + pipeline.storageTextures.map(\.handle) {
            let texture = nativeTexture(boundTextures[entry.slot]!)
            if let index = entry.vertexIndex { encoder.setVertexTexture(texture, index: Int(index)) }
            if let index = entry.fragmentIndex { encoder.setFragmentTexture(texture, index: Int(index)) }
        }
        for binding in pipeline.samplerBindings {
            let entry = binding.handle
            let sampler = samplers[samplerBindings[entry.slot]!]!.native
            if let index = entry.vertexIndex { encoder.setVertexSamplerState(sampler, index: Int(index)) }
            if let index = entry.fragmentIndex { encoder.setFragmentSamplerState(sampler, index: Int(index)) }
        }
    }

    private func maximumVertexIndex(_ layouts: [VertexBufferLayout], bindings: [VertexBufferBinding]) -> UInt32? {
        var maximum = UInt64(UInt32.max)
        for (layout, binding) in zip(layouts, bindings) where layout.stepMode == .vertex {
            let available = buffers[binding.buffer]!.descriptor.size - binding.offset
            guard available >= layout.attributeEnd else { return nil }
            if layout.stride > 0 {
                maximum = min(maximum, (available - layout.attributeEnd) / UInt64(layout.stride))
            }
        }
        return UInt32(maximum)
    }

    private func renderEncoder(
        _ commands: any MTLCommandBuffer, target: any MTLTexture, level: UInt32 = 0, layer: UInt32 = 0, clear: MTLClearColor?,
        depth: RenderDepthAttachment? = nil, resolve: RenderResolveAttachment? = nil,
        additionalColors: [UInt32: RenderColorAttachment] = [:]
    ) throws -> any MTLRenderCommandEncoder {
        let pass = MTLRenderPassDescriptor()
        func setColor(_ location: Int, texture: any MTLTexture, level: UInt32, layer: UInt32 = 0, clear: MTLClearColor?,
                      resolve: RenderResolveAttachment?) {
            let attachment = pass.colorAttachments[location]!
            attachment.texture = texture
            attachment.level = Int(level)
            if texture.textureType == .type3D {
                attachment.depthPlane = Int(layer)
            } else {
                attachment.slice = Int(layer)
            }
            attachment.loadAction = clear == nil ? .load : .clear
            attachment.storeAction = .store
            if let clear { attachment.clearColor = clear }
            if let resolve {
                attachment.resolveTexture = nativeTexture(resolve.texture)
                attachment.resolveLevel = Int(resolve.mipLevel)
                attachment.storeAction = .storeAndMultisampleResolve
            }
        }
        setColor(0, texture: target, level: level, layer: layer, clear: clear, resolve: resolve)
        for (location, attachment) in additionalColors {
            let clear = attachment.loadAction.clearColor.map {
                MTLClearColor(red: $0.red, green: $0.green, blue: $0.blue, alpha: $0.alpha)
            }
            setColor(Int(location), texture: nativeTexture(attachment.texture), level: attachment.mipLevel,
                     clear: clear, resolve: attachment.resolve)
        }
        if let depth {
            setDepthAttachment(pass, texture: nativeTexture(depth.texture), level: depth.mipLevel,
                               loadAction: depth.loadAction, stencilLoadAction: depth.stencil?.loadAction)
        }
        guard let encoder = commands.makeRenderCommandEncoder(descriptor: pass) else { throw GPUError.outOfMemory }
        return encoder
    }

    private func setDepthAttachment(
        _ pass: MTLRenderPassDescriptor, texture: any MTLTexture, level: UInt32, layer: UInt32 = 0,
        loadAction: RenderDepthAttachment.LoadAction, stencilLoadAction: RenderStencilAttachmentState.LoadAction?
    ) {
        let attachment = pass.depthAttachment!
        attachment.texture = texture
        attachment.level = Int(level)
        attachment.slice = Int(layer)
        attachment.storeAction = .store
        switch loadAction {
        case .load, .readOnly:
            attachment.loadAction = .load
        case .clear(let depth):
            attachment.loadAction = .clear
            attachment.clearDepth = depth
        }
        if let stencilLoadAction {
            let stencil = pass.stencilAttachment!
            stencil.texture = texture
            stencil.level = Int(level)
            stencil.slice = Int(layer)
            stencil.storeAction = .store
            switch stencilLoadAction {
            case .load, .readOnly: stencil.loadAction = .load
            case .clear(let value):
                stencil.loadAction = .clear
                stencil.clearStencil = value & 0xff
            }
        }
    }

    private func retire(_ serial: UInt64, completion: BackendCompletion, result: Result<Void, GPUError>) async {
        inFlight.removeValue(forKey: serial)
        if case .failure = result { terminal = true }
        await completion.finish(result)
        if inFlight.isEmpty {
            let waiters = retirementWaiters; retirementWaiters.removeAll()
            for waiter in waiters { waiter.resume() }
        }
    }

    // Internal deterministic fault-injection seam, never used by the production diagnostic.
    func setRetirementHookForTesting(_ hook: @escaping @Sendable () async -> Void) { retirementHook = hook }
    func resourceCountsForTesting() -> (buffers: Int, pipelines: Int, batches: Int, retainedBuffers: Int) {
        (buffers.count, pipelines.count + computePipelines.count, inFlight.count,
         inFlight.values.reduce(0) { $0 + $1.buffers.count })
    }
    func waitForRetirementForTesting() async {
        if !inFlight.isEmpty { await withCheckedContinuation { retirementWaiters.append($0) } }
    }
    #endif
}

#if canImport(Metal)
private extension RenderColorWriteMask {
    var metal: MTLColorWriteMask {
        var mask: MTLColorWriteMask = []
        if contains(.red) { mask.insert(.red) }
        if contains(.green) { mask.insert(.green) }
        if contains(.blue) { mask.insert(.blue) }
        if contains(.alpha) { mask.insert(.alpha) }
        return mask
    }
}

private extension RenderBlendOperation {
    var metal: MTLBlendOperation {
        switch self {
        case .add: .add
        case .subtract: .subtract
        case .reverseSubtract: .reverseSubtract
        case .min: .min
        case .max: .max
        }
    }
}

private extension RenderBlendFactor {
    var metal: MTLBlendFactor {
        switch self {
        case .zero: .zero
        case .one: .one
        case .sourceColor: .sourceColor
        case .oneMinusSourceColor: .oneMinusSourceColor
        case .destinationColor: .destinationColor
        case .oneMinusDestinationColor: .oneMinusDestinationColor
        case .sourceAlpha: .sourceAlpha
        case .oneMinusSourceAlpha: .oneMinusSourceAlpha
        case .destinationAlpha: .destinationAlpha
        case .oneMinusDestinationAlpha: .oneMinusDestinationAlpha
        case .sourceAlphaSaturated: .sourceAlphaSaturated
        case .constant: .blendColor
        case .oneMinusConstant: .oneMinusBlendColor
        }
    }
}

private extension RenderPrimitiveState.Topology {
    var metal: (topologyClass: MTLPrimitiveTopologyClass, type: MTLPrimitiveType) {
        switch self {
        case .pointList: (.point, .point)
        case .lineList: (.line, .line)
        case .lineStrip: (.line, .lineStrip)
        case .triangleList: (.triangle, .triangle)
        case .triangleStrip: (.triangle, .triangleStrip)
        }
    }
}

private extension IndexFormat {
    var metal: MTLIndexType { self == .uint16 ? .uint16 : .uint32 }
}

private extension VertexFormat {
    var metal: MTLVertexFormat {
        switch self {
        case .float32: .float
        case .float32x2: .float2
        case .float32x3: .float3
        case .float32x4: .float4
        case .uint32: .uint
        case .uint32x2: .uint2
        case .uint32x3: .uint3
        case .uint32x4: .uint4
        case .sint32: .int
        case .sint32x2: .int2
        case .sint32x3: .int3
        case .sint32x4: .int4
        }
    }
}

private extension RenderStencilFaceState {
    func metal(readMask: UInt32, writeMask: UInt32) -> MTLStencilDescriptor {
        let descriptor = MTLStencilDescriptor()
        descriptor.stencilCompareFunction = compare.metal
        descriptor.stencilFailureOperation = fail.metal
        descriptor.depthFailureOperation = depthFail.metal
        descriptor.depthStencilPassOperation = pass.metal
        descriptor.readMask = readMask & 0xff
        descriptor.writeMask = writeMask & 0xff
        return descriptor
    }
}

private extension RenderStencilOperation {
    var metal: MTLStencilOperation {
        switch self {
        case .keep: .keep
        case .zero: .zero
        case .replace: .replace
        case .invert: .invert
        case .incrementClamp: .incrementClamp
        case .decrementClamp: .decrementClamp
        case .incrementWrap: .incrementWrap
        case .decrementWrap: .decrementWrap
        }
    }
}

private extension CompareFunction {
    var metal: MTLCompareFunction {
        switch self {
        case .never: .never
        case .less: .less
        case .equal: .equal
        case .lessEqual: .lessEqual
        case .greater: .greater
        case .notEqual: .notEqual
        case .greaterEqual: .greaterEqual
        case .always: .always
        }
    }
}

private extension TextureFormat {
    var metal: MTLPixelFormat {
        switch self {
        case .rgba8Unorm: .rgba8Unorm
        case .bgra8Unorm: .bgra8Unorm
        case .r32Uint: .r32Uint
        case .depth32Float: .depth32Float
        case .depth24PlusStencil8: .depth32Float_stencil8
        case .rgba8Uint: .rgba8Uint
        case .rgba8Sint: .rgba8Sint
        case .rgba8Snorm: .rgba8Snorm
        case .rgba16Uint: .rgba16Uint
        case .rgba16Sint: .rgba16Sint
        case .rgba16Float: .rgba16Float
        case .r32Sint: .r32Sint
        case .r32Float: .r32Float
        case .rg32Uint: .rg32Uint
        case .rg32Sint: .rg32Sint
        case .rg32Float: .rg32Float
        case .rgba32Uint: .rgba32Uint
        case .rgba32Sint: .rgba32Sint
        case .rgba32Float: .rgba32Float
        }
    }
}
#endif
