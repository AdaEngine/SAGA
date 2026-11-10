#if os(Windows)
import WinSDK
import SAGATypes
import SAGABackend

struct D3D12DescriptorRange {
    var rangeType: UInt32 = 0
    var descriptorCount: UInt32 = 1
    var baseShaderRegister: UInt32 = 0
    var registerSpace: UInt32 = 0
    var offsetInDescriptors: UInt32 = 0
}

/// One buffer/texture copy in Direct3D 12 terms.
struct D3D12ImageCopy {
    var toTexture: Bool
    var texture: ResourceID
    var mip: UInt32
    var layer: UInt32
    var planes: UInt32
    var x: UInt32 = 0
    var y: UInt32 = 0
    var z: UInt32 = 0
    var width: UInt32
    var height: UInt32
    var buffer: COMObject
    var bufferOffset: UInt64
    var rowPitch: UInt32
}

extension D3D12Device {
    func align(_ value: UInt64, _ alignment: UInt64) -> UInt64 { (value + alignment - 1) / alignment * alignment }

    /// Records the copy commands for one buffer/texture transfer.
    func recordImageCopy(_ copy: D3D12ImageCopy, list: COMObject, transition: (ResourceID, UInt32) -> Void) throws {
        guard let texture = textures[copy.texture] else { throw GPUError.validation("Unknown Direct3D 12 texture") }
        typealias CopyTexture = @convention(c) (UnsafeMutableRawPointer, UnsafeRawPointer, UInt32, UInt32, UInt32, UnsafeRawPointer,
                                                UnsafeRawPointer?) -> Void
        guard copy.bufferOffset % 512 == 0, copy.rowPitch % 256 == 0 else {
            throw GPUError.unsupported("Direct3D 12 texture copies need a 512-byte buffer offset and a 256-byte row pitch")
        }
        transition(copy.texture, copy.toTexture ? D3D12.stateCopyDestination : D3D12.stateCopySource)
        let is3D = texture.descriptor.dimension == .d3
        let mipCount = texture.descriptor.mipLevelCount
        let planeBytes = align(UInt64(copy.height) * UInt64(copy.rowPitch), 512)
        let layers = is3D ? 1 : copy.planes
        for plane in 0..<layers {
            let subresource = copy.mip + (copy.layer + plane) * mipCount
            var buffer = D3D12TextureCopyLocation(
                resource: copy.buffer.pointer, type: 1, offset: copy.bufferOffset + UInt64(plane) * planeBytes,
                format: texture.viewFormat, width: copy.width, height: copy.height, depth: is3D ? copy.planes : 1,
                rowPitch: copy.rowPitch)
            var image = D3D12TextureCopyLocation(resource: texture.resource.pointer, type: 0, offset: UInt64(subresource))
            let front = is3D ? copy.z : 0
            let depth = is3D ? copy.planes : 1
            if copy.toTexture {
                var box = D3D12Box(right: copy.width, bottom: copy.height, back: depth)
                list.slot(16, CopyTexture.self)(list.pointer, &image, copy.x, copy.y, front, &buffer, &box)
            } else {
                var box = D3D12Box(left: copy.x, top: copy.y, front: front, right: copy.x + copy.width,
                                   bottom: copy.y + copy.height, back: front + depth)
                list.slot(16, CopyTexture.self)(list.pointer, &buffer, 0, 0, 0, &image, &box)
            }
        }
    }

    func createView(_ texture: Texture, storage: Bool, at handle: UInt) {
        var words = [UInt32](repeating: 0, count: 16)
        let dimension: TextureViewDimension = texture.viewDimension
            ?? (texture.descriptor.dimension == .d3 ? .d3 : (texture.descriptor.arrayLayerCount > 1 ? .d2Array : .d2))
        let mips = texture.mipCount ?? (texture.descriptor.mipLevelCount - texture.baseMip)
        let layers = texture.layerCount ?? (texture.descriptor.dimension == .d3 ? 1 : texture.descriptor.arrayLayerCount)
        words[0] = texture.viewFormat
        // Stencil views read plane one of the combined depth-stencil formats.
        var stencilPlane = false
        if texture.viewAspect == .stencilOnly, let format = d3d12StencilViewFormat(texture.descriptor.format) {
            words[0] = format
            stencilPlane = true
        }
        if storage {
            typealias CreateUAV = @convention(c) (UnsafeMutableRawPointer, UnsafeMutableRawPointer, UnsafeMutableRawPointer?, UnsafeRawPointer, UInt) -> Void
            words[2] = texture.baseMip  // the union starts at offset 8 for storage views
            switch dimension {
            case .d3: words[1] = 8; words[4] = texture.descriptor.depth
            case .d2: words[1] = 4
            default: words[1] = 5; words[3] = texture.baseLayer; words[4] = layers
            }
            words.withUnsafeBytes { device.slot(19, CreateUAV.self)(device.pointer, texture.resource.pointer, nil, $0.baseAddress!, handle) }
        } else {
            typealias CreateSRV = @convention(c) (UnsafeMutableRawPointer, UnsafeMutableRawPointer, UnsafeRawPointer, UInt) -> Void
            words[2] = D3D12.defaultComponentMapping
            words[4] = texture.baseMip  // the SRV union starts at an 8-byte boundary
            words[5] = mips
            if stencilPlane && dimension == .d2 { words[6] = 1 }
            switch dimension {
            case .d3: words[1] = 8
            case .d2: words[1] = 4
            case .cube: words[1] = 9
            case .cubeArray: words[1] = 10; words[6] = texture.baseLayer; words[7] = layers / 6
            default: words[1] = 5; words[6] = texture.baseLayer; words[7] = layers
            }
            words.withUnsafeBytes { device.slot(18, CreateSRV.self)(device.pointer, texture.resource.pointer, $0.baseAddress!, handle) }
        }
    }

    /// A raw (byte-address) view of a buffer range, as used by binding-array elements.
    func createRawBufferView(_ resource: COMObject, offset: UInt64, size: UInt64, storage: Bool, at handle: UInt) {
        var words = [UInt32](repeating: 0, count: 16)
        words[0] = 39  // R32_TYPELESS
        words[1] = 1  // buffer
        let elements = UInt32(size / 4)
        if storage {
            typealias CreateUAV = @convention(c) (UnsafeMutableRawPointer, UnsafeMutableRawPointer, UnsafeMutableRawPointer?, UnsafeRawPointer, UInt) -> Void
            words[2] = UInt32(truncatingIfNeeded: offset / 4); words[3] = UInt32(truncatingIfNeeded: (offset / 4) >> 32)
            words[4] = elements
            words[8] = 1  // raw
            words.withUnsafeBytes { device.slot(19, CreateUAV.self)(device.pointer, resource.pointer, nil, $0.baseAddress!, handle) }
        } else {
            typealias CreateSRV = @convention(c) (UnsafeMutableRawPointer, UnsafeMutableRawPointer, UnsafeRawPointer, UInt) -> Void
            words[2] = D3D12.defaultComponentMapping
            words[4] = UInt32(truncatingIfNeeded: offset / 4); words[5] = UInt32(truncatingIfNeeded: (offset / 4) >> 32)
            words[6] = elements
            words[8] = 1  // raw
            words.withUnsafeBytes { device.slot(18, CreateSRV.self)(device.pointer, resource.pointer, $0.baseAddress!, handle) }
        }
    }

    func createSampler(_ descriptor: SamplerDescriptor, at handle: UInt) {
        typealias CreateSampler = @convention(c) (UnsafeMutableRawPointer, UnsafeRawPointer, UInt) -> Void
        let words = samplerDescription(descriptor)
        words.withUnsafeBytes { device.slot(22, CreateSampler.self)(device.pointer, $0.baseAddress!, handle) }
    }
}
#endif

#if os(Windows)
/// Shader view format for the stencil plane of a depth-stencil texture.
func d3d12StencilViewFormat(_ format: TextureFormat) -> UInt32? {
    switch format {
    case .depth24PlusStencil8, .stencil8: 47  // X24_TYPELESS_G8_UINT
    case .depth32FloatStencil8: 22  // X32_TYPELESS_G8X24_UINT
    default: nil
    }
}
#endif
