import SAGAShaderIR

struct Float64IntegerBounds {
    let lower: Double
    let upper: Double

    init(target: IRScalar) throws {
        switch target {
        case .i16: (lower, upper) = (Double(Int16.min), Double(Int16.max))
        case .u16: (lower, upper) = (0, Double(UInt16.max))
        case .i32: (lower, upper) = (Double(Int32.min), Double(Int32.max))
        case .u32: (lower, upper) = (0, Double(UInt32.max))
        // Rounding the 64-bit maxima to Double gives 2^63 and 2^64 respectively.
        case .i64: (lower, upper) = (Double(Int64.min), Double(Int64.max).nextDown)
        case .u64: (lower, upper) = (0, Double(UInt64.max).nextDown)
        default: throw ShaderFailure.invalidHandle
        }
    }
}
