import SAGAShaderIR

func packedHalfBitcast(
    _ source: String, input: IRBitcastShape, output: IRBitcastShape,
    words: Int, resultType: String, format: ShaderOutputFormat
) -> String {
    let components = ["x", "y", "z", "w"]
    let converted = (0..<words).map { index -> String in
        if input.scalar == .f16 {
            let bits: String
            if format == .hlsl {
                let low = "\(source).\(components[2 * index])"
                let high = "\(source).\(components[2 * index + 1])"
                bits = "(uint(asuint16(\(low))) | (uint(asuint16(\(high))) << 16u))"
            } else {
                let pair = words == 1 ? source : "\(source).\(index == 0 ? "xy" : "zw")"
                bits = "packFloat2x16(\(pair))"
            }
            switch (format, output.scalar) {
            case (.hlsl, .i32): return "asint(\(bits))"
            case (.hlsl, .f32): return "asfloat(\(bits))"
            case (.glsl, .i32): return "int(\(bits))"
            case (.glsl, .f32): return "uintBitsToFloat(\(bits))"
            default: return bits
            }
        }

        let word = words == 1 ? source : "\(source).\(components[index])"
        let bits: String
        switch (format, input.scalar) {
        case (_, .u32): bits = word
        case (.hlsl, _): bits = "asuint(\(word))"
        case (.glsl, .f32): bits = "floatBitsToUint(\(word))"
        default: bits = "uint(\(word))"
        }
        if format == .hlsl {
            return "half2(asfloat16(uint16_t(\(bits))), asfloat16(uint16_t((\(bits)) >> 16u)))"
        }
        return "unpackFloat2x16(\(bits))"
    }
    return words == 1 ? converted[0] : "\(resultType)(\(converted.joined(separator: ", ")))"
}
