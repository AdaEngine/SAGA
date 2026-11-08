// Binary16 conversions use binary64 significand bits to avoid an intermediate rounding to f32.
package enum Binary16 {
    package static func encode(_ value: Double) -> UInt16 {
        let bits = value.bitPattern
        let sign = UInt16((bits >> 48) & 0x8000)
        let exponent = Int((bits >> 52) & 0x7ff) - 1023
        guard exponent >= -25 else { return sign }
        guard exponent <= 15 else { return sign | 0x7c00 }
        let significand = (bits & 0x000f_ffff_ffff_ffff) | 0x0010_0000_0000_0000
        let shift = exponent >= -14 ? 42 : 28 - exponent
        let quotient = significand >> shift
        let remainder = significand & ((1 << shift) - 1)
        let halfway: UInt64 = 1 << (shift - 1)
        let rounded = quotient + (remainder > halfway || (remainder == halfway && quotient & 1 != 0) ? 1 : 0)
        let exponentBase = exponent >= -14 ? UInt16(exponent + 14) << 10 : 0
        return sign | (exponentBase + UInt16(rounded))
    }

    package static func decode(_ bits: UInt16) -> Float {
        let word = UInt32(bits)
        let sign = (word & 0x8000) << 16
        let exponent = (word >> 10) & 31
        let fraction = word & 1023
        if exponent == 0 {
            let magnitude = Float(fraction) / 16777216
            return Float(bitPattern: sign | magnitude.bitPattern)
        }
        let exponent32 = exponent == 31 ? 255 : exponent + 112
        return Float(bitPattern: sign | (exponent32 << 23) | (fraction << 13))
    }
}
