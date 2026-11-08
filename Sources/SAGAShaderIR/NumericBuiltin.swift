package struct IRNumericBuiltin: Sendable {
    package enum Domain: Sendable {
        case ordered, signed, floating, binary32
    }

    package let arity: Int
    package let domain: Domain
    package let scalarLastArgument: Bool

    fileprivate init(_ arity: Int, _ domain: Domain, scalarLastArgument: Bool = false) {
        self.arity = arity
        self.domain = domain
        self.scalarLastArgument = scalarLastArgument
    }

    package func accepts(_ scalar: IRScalar) -> Bool {
        if domain == .binary32 { return scalar == .f32 }
        if scalar.kind == .float || scalar.kind == .abstractFloat { return true }
        if domain == .floating { return false }
        if scalar.kind == .sint || scalar.kind == .abstractInt { return true }
        return domain == .ordered && scalar.kind == .uint
    }
}

extension IRMathFunction {
    package init?(wgslName: String) {
        if wgslName == "smoothstep" { self = .smoothStep }
        else if wgslName == Self.smoothStep.rawValue || wgslName == Self.inverse.rawValue { return nil }
        else { self.init(rawValue: wgslName) }
    }

    package var wgslName: String { self == .smoothStep ? "smoothstep" : rawValue }

    package var numericBuiltin: IRNumericBuiltin? {
        switch self {
        case .abs: .init(1, .ordered)
        case .sign: .init(1, .signed)
        case .min, .max: .init(2, .ordered)
        case .clamp: .init(3, .ordered)
        case .ceil, .floor, .round, .fract, .trunc, .sqrt, .inverseSqrt, .saturate: .init(1, .floating)
        case .sin, .cos, .tan, .asin, .acos, .atan, .sinh, .cosh, .tanh, .asinh, .acosh, .atanh,
             .exp, .exp2, .log, .log2, .radians, .degrees: .init(1, .floating)
        case .atan2, .pow, .step: .init(2, .floating)
        case .fma, .smoothStep: .init(3, .floating)
        case .quantizeToF16: .init(1, .binary32)
        case .mix: .init(3, .floating, scalarLastArgument: true)
        default: nil
        }
    }
}
