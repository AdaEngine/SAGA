#if os(Windows)
import WinSDK
import SAGATypes
import SAGABackend
import SAGAShaderIR

// Textures, samplers and descriptor heaps.

func d3d12Format(_ format: TextureFormat) -> UInt32? {
    switch format {
    case .r8Unorm: 61
    case .rg8Unorm: 49
    case .rgba8Unorm: 28
    case .rgba8UnormSrgb: 29
    case .rgba8Snorm: 31
    case .rgba8Uint: 30
    case .rgba8Sint: 32
    case .bgra8Unorm: 87
    case .bgra8UnormSrgb: 91
    case .rgba16Uint: 12
    case .rgba16Sint: 14
    case .rgba16Float: 10
    case .r32Uint: 42
    case .r32Sint: 43
    case .r32Float: 41
    case .rg32Uint: 17
    case .rg32Sint: 18
    case .rg32Float: 16
    case .rgba32Uint: 3
    case .rgba32Sint: 4
    case .rgba32Float: 2
    default: nil
    }
}

/// Typeless resource, depth-stencil view and shader view formats for depth-stencil textures.
func d3d12DepthFormats(_ format: TextureFormat) -> (typeless: UInt32, dsv: UInt32, srv: UInt32)? {
    switch format {
    case .depth16Unorm: (53, 55, 56)
    case .depth32Float, .depth24Plus: (39, 40, 41)
    case .depth24PlusStencil8, .stencil8: (44, 45, 46)
    case .depth32FloatStencil8: (19, 20, 21)
    default: nil
    }
}

func d3d12BytesPerTexel(_ format: TextureFormat) -> UInt32 {
    switch format {
    case .depth16Unorm: return 2
    case .depth32Float, .depth24Plus: return 4
    default: break
    }
    return switch format {
    case .r8Unorm: 1
    case .rg8Unorm: 2
    case .rgba16Uint, .rgba16Sint, .rgba16Float, .rg32Uint, .rg32Sint, .rg32Float: 8
    case .rgba32Uint, .rgba32Sint, .rgba32Float: 16
    default: 4
    }
}

extension D3D12 {
    static let resourceDimensionTexture2D: UInt32 = 3
    static let resourceDimensionTexture3D: UInt32 = 4
    static let flagAllowRenderTarget: UInt32 = 0x1
    static let flagAllowDepthStencil: UInt32 = 0x2
    static let stateDepthWrite: UInt32 = 0x10
    static let stateDepthRead: UInt32 = 0x20
    static let stateResolveDestination: UInt32 = 0x1000
    static let stateResolveSource: UInt32 = 0x2000
    static let stateRenderTarget: UInt32 = 0x4
    static let descriptorHeapCBVSRVUAV: UInt32 = 0
    static let descriptorHeapSampler: UInt32 = 1
    static let descriptorHeapFlagShaderVisible: UInt32 = 1
    static let defaultComponentMapping: UInt32 = 0x1688
}

struct D3D12TextureCopyLocation {
    var resource: UnsafeMutableRawPointer?
    var type: UInt32 = 0
    var offset: UInt64 = 0
    var format: UInt32 = 0
    var width: UInt32 = 0
    var height: UInt32 = 0
    var depth: UInt32 = 0
    var rowPitch: UInt32 = 0
}

struct D3D12Box {
    var left: UInt32 = 0
    var top: UInt32 = 0
    var front: UInt32 = 0
    var right: UInt32 = 0
    var bottom: UInt32 = 0
    var back: UInt32 = 1
}

/// A shader-visible descriptor heap pair that hands out consecutive slots for one submission.
final class D3D12DescriptorArena: @unchecked Sendable {
    let viewHeap: COMObject
    let samplerHeap: COMObject
    let viewIncrement: UInt
    let samplerIncrement: UInt
    let viewCPU: UInt
    let viewGPU: UInt64
    let samplerCPU: UInt
    let samplerGPU: UInt64
    private let viewCapacity: Int
    private let samplerCapacity: Int
    private var viewsUsed = 0
    private var samplersUsed = 0

    init(device: COMObject, viewCapacity: Int = 4096, samplerCapacity: Int = 512) throws {
        typealias CreateHeap = @convention(c) (UnsafeMutableRawPointer, UnsafeRawPointer, UnsafeRawPointer,
                                               UnsafeMutablePointer<UnsafeMutableRawPointer?>) -> HRESULT
        typealias Increment = @convention(c) (UnsafeMutableRawPointer, UInt32) -> UInt32
        typealias HeapStart = @convention(c) (UnsafeMutableRawPointer, UnsafeMutableRawPointer) -> UnsafeMutableRawPointer
        func make(_ type: UInt32, _ count: Int) throws -> COMObject {
            var description: (UInt32, UInt32, UInt32, UInt32) = (type, UInt32(count), D3D12.descriptorHeapFlagShaderVisible, 0)
            var iid = IID.descriptorHeap
            var object: UnsafeMutableRawPointer?
            try d3dCheck(device.slot(14, CreateHeap.self)(device.pointer, &description, &iid, &object), "descriptor heap creation")
            return COMObject(object!)
        }
        viewHeap = try make(D3D12.descriptorHeapCBVSRVUAV, viewCapacity)
        samplerHeap = try make(D3D12.descriptorHeapSampler, samplerCapacity)
        self.viewCapacity = viewCapacity
        self.samplerCapacity = samplerCapacity
        viewIncrement = UInt(device.slot(15, Increment.self)(device.pointer, D3D12.descriptorHeapCBVSRVUAV))
        samplerIncrement = UInt(device.slot(15, Increment.self)(device.pointer, D3D12.descriptorHeapSampler))
        func start(_ heap: COMObject, gpu: Bool) -> UInt64 {
            var handle: UInt64 = 0
            _ = heap.slot(gpu ? 10 : 9, HeapStart.self)(heap.pointer, &handle)
            return handle
        }
        viewCPU = UInt(start(viewHeap, gpu: false))
        viewGPU = start(viewHeap, gpu: true)
        samplerCPU = UInt(start(samplerHeap, gpu: false))
        samplerGPU = start(samplerHeap, gpu: true)
    }

    func allocateViews(_ count: Int) throws -> (cpu: UInt, gpu: UInt64) {
        guard viewsUsed + count <= viewCapacity else { throw GPUError.outOfMemory }
        defer { viewsUsed += count }
        return (viewCPU + UInt(viewsUsed) * viewIncrement, viewGPU + UInt64(viewsUsed) * UInt64(viewIncrement))
    }

    func allocateSamplers(_ count: Int) throws -> (cpu: UInt, gpu: UInt64) {
        guard samplersUsed + count <= samplerCapacity else { throw GPUError.outOfMemory }
        defer { samplersUsed += count }
        return (samplerCPU + UInt(samplersUsed) * samplerIncrement, samplerGPU + UInt64(samplersUsed) * UInt64(samplerIncrement))
    }
}

extension D3D12Device: RenderBackendDevice, SamplerBackendDevice {
    struct Texture {
        var descriptor: TextureDescriptor
        var format: UInt32
        var resource: COMObject
        /// Format used by shader resource views, copies and render target views.
        var viewFormat: UInt32
        var dsvFormat: UInt32?
        /// Set when the value stands for a texture view.
        var baseMip: UInt32 = 0
        var mipCount: UInt32?
        var baseLayer: UInt32 = 0
        var layerCount: UInt32?
        var viewDimension: TextureViewDimension?
        var viewAspect: TextureViewAspect = .all
    }

    struct TextureViewState {
        var texture: ResourceID
        var levels: Range<UInt32>
        var layers: Range<UInt32>
        var dimension: TextureViewDimension
        var aspect: TextureViewAspect = .all
    }

    /// Resolves a texture or texture view id to the texture it covers.
    func texture(for id: ResourceID) -> Texture? {
        if let texture = textures[id] { return texture }
        guard let view = textureViews[id], var texture = textures[view.texture] else { return nil }
        texture.baseMip = view.levels.lowerBound
        texture.mipCount = UInt32(view.levels.count)
        texture.baseLayer = view.layers.lowerBound
        texture.layerCount = UInt32(view.layers.count)
        texture.viewDimension = view.dimension
        texture.viewAspect = view.aspect
        return texture
    }

    // MARK: Textures

    public func allocateTexture(id: ResourceID, descriptor: TextureDescriptor) async throws {
        try requireActive()
        try descriptor.validate(against: capabilities)
        guard textures[id] == nil, buffers[id] == nil else { throw GPUError.validation("Duplicate Direct3D 12 resource") }
        let depthFormats = d3d12DepthFormats(descriptor.format)
        guard descriptor.dimension != .d1, descriptor.sampleCount == 1 || (descriptor.sampleCount == 4 && descriptor.dimension == .d2),
              let format = depthFormats?.typeless ?? d3d12Format(descriptor.format) else {
            throw GPUError.unsupported("Direct3D 12 textures are limited to 2D and 3D images in supported formats in this release")
        }
        if descriptor.sampleCount > 1 && descriptor.usage != .renderAttachment {
            throw GPUError.unsupported("Direct3D 12 multisampled textures must be render attachments")
        }
        var flags: UInt32 = 0
        if descriptor.usage.contains(.storageBinding) { flags |= D3D12.flagAllowUnorderedAccess }
        if descriptor.usage.contains(.renderAttachment) {
            flags |= depthFormats != nil ? D3D12.flagAllowDepthStencil : D3D12.flagAllowRenderTarget
        }
        typealias CreateCommitted = @convention(c) (
            UnsafeMutableRawPointer, UnsafeRawPointer, UInt32, UnsafeRawPointer, UInt32, UnsafeRawPointer?, UnsafeRawPointer,
            UnsafeMutablePointer<UnsafeMutableRawPointer?>) -> HRESULT
        var properties = D3D12HeapProperties(type: D3D12.heapDefault)
        var description = D3D12ResourceDesc()
        description.dimension = descriptor.dimension == .d3 ? D3D12.resourceDimensionTexture3D : D3D12.resourceDimensionTexture2D
        description.width = UInt64(descriptor.width)
        description.height = descriptor.height
        description.depthOrArraySize = UInt16(descriptor.dimension == .d3 ? descriptor.depth : descriptor.arrayLayerCount)
        description.mipLevels = UInt16(descriptor.mipLevelCount)
        description.format = format
        description.sampleCount = descriptor.sampleCount
        description.layout = 0
        description.flags = flags
        let created = Self.createCommitted(device, properties: &properties, description: &description,
                                           create: device.slot(27, CreateCommitted.self))
        D3D12DebugLayer.drain(device: device)
        try d3dCheck(created.0, "texture creation")
        textures[id] = Texture(descriptor: descriptor, format: format, resource: created.1!,
                               viewFormat: depthFormats?.srv ?? format, dsvFormat: depthFormats?.dsv)
    }

    public func destroyTexture(id: ResourceID) async { textures[id] = nil }

    nonisolated static func createCommitted(
        _ device: COMObject, properties: inout D3D12HeapProperties, description: inout D3D12ResourceDesc,
        create: @convention(c) (UnsafeMutableRawPointer, UnsafeRawPointer, UInt32, UnsafeRawPointer, UInt32, UnsafeRawPointer?,
                                UnsafeRawPointer, UnsafeMutablePointer<UnsafeMutableRawPointer?>) -> HRESULT
    ) -> (HRESULT, COMObject?) {
        var iid = IID.resource
        var object: UnsafeMutableRawPointer?
        let result = create(device.pointer, &properties, 0, &description, D3D12.stateCommon, nil, &iid, &object)
        return (result, object.map(COMObject.init))
    }

    public func allocateRenderPipeline(_ pipeline: RenderPipeline, vertex: RenderStageArtifact,
                                       fragment: RenderStageArtifact) async throws {
        try requireActive()
        guard renderPipelines[pipeline.id] == nil else { throw GPUError.validation("Duplicate Direct3D 12 render pipeline") }
        renderPipelines[pipeline.id] = try allocateRenderPipelineNative(pipeline, vertex: vertex, fragment: fragment)
    }

    public func destroyRenderPipeline(id: ResourceID) async { renderPipelines[id] = nil }

    // MARK: Samplers

    public func allocateSampler(id: ResourceID, descriptor: SamplerDescriptor) async throws {
        try requireActive()
        try descriptor.validate()
        guard samplers[id] == nil else { throw GPUError.validation("Duplicate Direct3D 12 sampler") }
        samplers[id] = descriptor
    }

    public func destroySampler(id: ResourceID) async { samplers[id] = nil }

    /// `D3D12_SAMPLER_DESC` bytes for a sampler descriptor.
    func samplerDescription(_ descriptor: SamplerDescriptor) -> [UInt32] {
        var filter = descriptor.minFilter.rawValue << 4 | descriptor.magFilter.rawValue << 2 | descriptor.mipmapFilter.rawValue
        if descriptor.compare != nil { filter |= 0x80 }
        func address(_ mode: SamplerAddressMode) -> UInt32 {
            switch mode { case .repeat: 1; case .mirrorRepeat: 2; case .clampToEdge: 3 }
        }
        return [filter, address(descriptor.addressModeU), address(descriptor.addressModeV), address(descriptor.addressModeW),
                Float(0).bitPattern, 1, (descriptor.compare?.rawValue ?? 7) + 1, 0, 0, 0, 0,
                descriptor.lodMinClamp.bitPattern, descriptor.lodMaxClamp.bitPattern]
    }
}
#endif

#if os(Windows)
extension D3D12Device: TextureViewBackendDevice {
    public func allocateTextureView(id: ResourceID, texture: ResourceID, mipLevels: Range<UInt32>) async throws {
        try await allocateTextureView(id: id, texture: texture, descriptor: .init(
            baseMipLevel: mipLevels.lowerBound, mipLevelCount: UInt32(mipLevels.count)))
    }

    public func allocateTextureView(id: ResourceID, texture: ResourceID, descriptor: TextureViewDescriptor) async throws {
        try requireActive()
        guard textures[id] == nil, buffers[id] == nil, textureViews[id] == nil, let base = textures[texture] else {
            throw GPUError.validation("Invalid Direct3D 12 texture view")
        }
        guard descriptor.aspect == .all || base.dsvFormat != nil else {
            throw GPUError.validation("Aspect views need a depth or stencil texture")
        }
        let count = descriptor.mipLevelCount ?? (base.descriptor.mipLevelCount - min(descriptor.baseMipLevel, base.descriptor.mipLevelCount))
        guard count > 0, descriptor.baseMipLevel + count <= base.descriptor.mipLevelCount else {
            throw GPUError.validation("Invalid Direct3D 12 texture view mip range")
        }
        let dimension = descriptor.resolvedDimension(in: base.descriptor)
        guard dimension != .d1 else { throw GPUError.unsupported("Direct3D 12 1D texture views are not implemented") }
        textureViews[id] = TextureViewState(
            texture: texture, levels: descriptor.baseMipLevel..<(descriptor.baseMipLevel + count),
            layers: try descriptor.arrayLayers(in: base.descriptor), dimension: dimension, aspect: descriptor.aspect)
    }

    public func destroyTextureView(id: ResourceID) async { textureViews[id] = nil }
}
#endif
