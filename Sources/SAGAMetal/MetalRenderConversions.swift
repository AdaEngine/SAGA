import SAGATypes
#if canImport(Metal)
import Metal
#endif

#if canImport(Metal)
extension RenderColorWriteMask {
    var metal: MTLColorWriteMask {
        var mask: MTLColorWriteMask = []
        if contains(.red) { mask.insert(.red) }
        if contains(.green) { mask.insert(.green) }
        if contains(.blue) { mask.insert(.blue) }
        if contains(.alpha) { mask.insert(.alpha) }
        return mask
    }
}

extension RenderBlendOperation {
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

extension RenderBlendFactor {
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
        case .source1Color: .source1Color
        case .oneMinusSource1Color: .oneMinusSource1Color
        case .source1Alpha: .source1Alpha
        case .oneMinusSource1Alpha: .oneMinusSource1Alpha
        }
    }
}

extension RenderPrimitiveState.Topology {
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

extension IndexFormat {
    var metal: MTLIndexType { self == .uint16 ? .uint16 : .uint32 }
}

extension VertexFormat {
    var metal: MTLVertexFormat {
        switch self {
        case .uint8: .uchar
        case .uint8x2: .uchar2
        case .uint8x4: .uchar4
        case .sint8: .char
        case .sint8x2: .char2
        case .sint8x4: .char4
        case .unorm8: .ucharNormalized
        case .unorm8x2: .uchar2Normalized
        case .unorm8x4: .uchar4Normalized
        case .snorm8: .charNormalized
        case .snorm8x2: .char2Normalized
        case .snorm8x4: .char4Normalized
        case .uint16: .ushort
        case .uint16x2: .ushort2
        case .uint16x4: .ushort4
        case .sint16: .short
        case .sint16x2: .short2
        case .sint16x4: .short4
        case .unorm16: .ushortNormalized
        case .unorm16x2: .ushort2Normalized
        case .unorm16x4: .ushort4Normalized
        case .snorm16: .shortNormalized
        case .snorm16x2: .short2Normalized
        case .snorm16x4: .short4Normalized
        case .unorm1010102: .uint1010102Normalized
        case .unorm8x4BGRA: .uchar4Normalized_bgra
        case .snorm1010102: .int1010102Normalized
        case .float32: .float
        case .float32x2: .float2
        case .float32x3: .float3
        case .float32x4: .float4
        case .float16: .half
        case .float16x2: .half2
        case .float16x4: .half4
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

extension RenderStencilFaceState {
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

extension RenderStencilOperation {
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

extension CompareFunction {
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

extension TextureFormat {
    var metal: MTLPixelFormat {
        switch self {
        case .rgba8Unorm: .rgba8Unorm
        case .bgra8Unorm: .bgra8Unorm
        case .r32Uint: .r32Uint
        case .r64Uint: .rg32Uint
        case .stencil8: .stencil8
        case .depth16Unorm: .depth16Unorm
        case .depth24Plus, .depth32Float: .depth32Float
        case .depth24PlusStencil8, .depth32FloatStencil8: .depth32Float_stencil8
        case .rgba8Uint: .rgba8Uint
        case .rgba8Sint: .rgba8Sint
        case .rgba8Snorm: .rgba8Snorm
        case .rgba16Uint: .rgba16Uint
        case .rgba16Sint: .rgba16Sint
        case .rgba16Float: .rgba16Float
        case .r8Unorm: .r8Unorm
        case .r8Snorm: .r8Snorm
        case .r8Uint: .r8Uint
        case .r8Sint: .r8Sint
        case .rg8Unorm: .rg8Unorm
        case .rg8Snorm: .rg8Snorm
        case .rg8Uint: .rg8Uint
        case .rg8Sint: .rg8Sint
        case .rgba8UnormSrgb: .rgba8Unorm_srgb
        case .bgra8UnormSrgb: .bgra8Unorm_srgb
        case .rgb9e5Ufloat: .rgb9e5Float
        case .rgb10a2Unorm: .rgb10a2Unorm
        case .rgb10a2Uint: .rgb10a2Uint
        case .rg11b10Ufloat: .rg11b10Float
        case .r16Unorm: .r16Unorm
        case .rg16Unorm: .rg16Unorm
        case .rgba16Unorm: .rgba16Unorm
        case .r16Snorm: .r16Snorm
        case .rg16Snorm: .rg16Snorm
        case .rgba16Snorm: .rgba16Snorm
        case .r16Uint: .r16Uint
        case .r16Sint: .r16Sint
        case .rg16Uint: .rg16Uint
        case .rg16Sint: .rg16Sint
        case .r16Float: .r16Float
        case .rg16Float: .rg16Float
        case .r32Sint: .r32Sint
        case .r32Float: .r32Float
        case .rg32Uint: .rg32Uint
        case .rg32Sint: .rg32Sint
        case .rg32Float: .rg32Float
        case .rgba32Uint: .rgba32Uint
        case .rgba32Sint: .rgba32Sint
        case .rgba32Float: .rgba32Float
        case .bc1RGBAUnorm: .bc1_rgba
        case .bc1RGBAUnormSrgb: .bc1_rgba_srgb
        case .bc2RGBAUnorm: .bc2_rgba
        case .bc2RGBAUnormSrgb: .bc2_rgba_srgb
        case .bc3RGBAUnorm: .bc3_rgba
        case .bc3RGBAUnormSrgb: .bc3_rgba_srgb
        case .bc4RUnorm: .bc4_rUnorm
        case .bc4RSnorm: .bc4_rSnorm
        case .bc5RGUnorm: .bc5_rgUnorm
        case .bc5RGSnorm: .bc5_rgSnorm
        case .bc6HRGBUfloat: .bc6H_rgbuFloat
        case .bc6HRGBFloat: .bc6H_rgbFloat
        case .bc7RGBAUnorm: .bc7_rgbaUnorm
        case .bc7RGBAUnormSrgb: .bc7_rgbaUnorm_srgb
        case .etc2RGB8Unorm: .etc2_rgb8
        case .etc2RGB8UnormSrgb: .etc2_rgb8_srgb
        case .etc2RGB8A1Unorm: .etc2_rgb8a1
        case .etc2RGB8A1UnormSrgb: .etc2_rgb8a1_srgb
        case .etc2RGBA8Unorm: .eac_rgba8
        case .etc2RGBA8UnormSrgb: .eac_rgba8_srgb
        case .eacR11Unorm: .eac_r11Unorm
        case .eacR11Snorm: .eac_r11Snorm
        case .eacRG11Unorm: .eac_rg11Unorm
        case .eacRG11Snorm: .eac_rg11Snorm
        case .astc4x4Unorm: .astc_4x4_ldr
        case .astc4x4UnormSrgb: .astc_4x4_srgb
        case .astc4x4HDR: .astc_4x4_hdr
        case .astc5x4Unorm: .astc_5x4_ldr
        case .astc5x4UnormSrgb: .astc_5x4_srgb
        case .astc5x4HDR: .astc_5x4_hdr
        case .astc5x5Unorm: .astc_5x5_ldr
        case .astc5x5UnormSrgb: .astc_5x5_srgb
        case .astc5x5HDR: .astc_5x5_hdr
        case .astc6x5Unorm: .astc_6x5_ldr
        case .astc6x5UnormSrgb: .astc_6x5_srgb
        case .astc6x5HDR: .astc_6x5_hdr
        case .astc6x6Unorm: .astc_6x6_ldr
        case .astc6x6UnormSrgb: .astc_6x6_srgb
        case .astc6x6HDR: .astc_6x6_hdr
        case .astc8x5Unorm: .astc_8x5_ldr
        case .astc8x5UnormSrgb: .astc_8x5_srgb
        case .astc8x5HDR: .astc_8x5_hdr
        case .astc8x6Unorm: .astc_8x6_ldr
        case .astc8x6UnormSrgb: .astc_8x6_srgb
        case .astc8x6HDR: .astc_8x6_hdr
        case .astc8x8Unorm: .astc_8x8_ldr
        case .astc8x8UnormSrgb: .astc_8x8_srgb
        case .astc8x8HDR: .astc_8x8_hdr
        case .astc10x5Unorm: .astc_10x5_ldr
        case .astc10x5UnormSrgb: .astc_10x5_srgb
        case .astc10x5HDR: .astc_10x5_hdr
        case .astc10x6Unorm: .astc_10x6_ldr
        case .astc10x6UnormSrgb: .astc_10x6_srgb
        case .astc10x6HDR: .astc_10x6_hdr
        case .astc10x8Unorm: .astc_10x8_ldr
        case .astc10x8UnormSrgb: .astc_10x8_srgb
        case .astc10x8HDR: .astc_10x8_hdr
        case .astc10x10Unorm: .astc_10x10_ldr
        case .astc10x10UnormSrgb: .astc_10x10_srgb
        case .astc10x10HDR: .astc_10x10_hdr
        case .astc12x10Unorm: .astc_12x10_ldr
        case .astc12x10UnormSrgb: .astc_12x10_srgb
        case .astc12x10HDR: .astc_12x10_hdr
        case .astc12x12Unorm: .astc_12x12_ldr
        case .astc12x12UnormSrgb: .astc_12x12_srgb
        case .astc12x12HDR: .astc_12x12_hdr
        }
    }
}
#endif
