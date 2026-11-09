extension IRMathFunction {
    package var decomposesFloat: Bool { self == .modf || self == .frexp }

    package func decompositionResult(size: IRVectorSize?, scalar: IRScalar) -> IRPredeclaredType {
        self == .modf ? .modfResult(size: size, scalar: scalar) : .frexpResult(size: size, scalar: scalar)
    }

    package var decompositionSecondMember: String { self == .frexp ? "exp" : "whole" }

    package func decompositionSecondScalar(_ scalar: IRScalar) -> IRScalar {
        guard self == .frexp else { return scalar }
        return scalar == .abstractFloat ? .abstractInt : .i32
    }
}

extension IRPredeclaredType {
    package var decomposition: (function: IRMathFunction, size: IRVectorSize?, scalar: IRScalar)? {
        switch self {
        case .modfResult(let size, let scalar): (.modf, size, scalar)
        case .frexpResult(let size, let scalar): (.frexp, size, scalar)
        case .atomicCompareExchangeWeakResult: nil
        }
    }
}

extension IRMathFunction {
    package func decompositionMemberStride(size: IRVectorSize?, scalar: IRScalar) -> UInt32 {
        let alignment: UInt32 = size == .three ? 4 : UInt32(size?.rawValue ?? 1)
        return alignment * UInt32(Swift.max(scalar.width, decompositionSecondScalar(scalar).width))
    }
}

extension IRTypeKind {
    package var concreteFloatShape: (size: IRVectorSize?, scalar: IRScalar)? {
        switch self {
        case .scalar(let lane) where lane == .f32 || lane == .f16: (nil, lane)
        case .vector(let size, let lane) where lane == .f32 || lane == .f16: (size, lane)
        default: nil
        }
    }

    package var ldexpExponentType: IRTypeKind? {
        guard let shape = IRBitcastShape(self), [.f16, .f32, .f64].contains(shape.scalar) else { return nil }
        return shape.size.map { .vector(size: $0, scalar: .i32) } ?? .scalar(.i32)
    }

    package var usesFloat16: Bool {
        switch self {
        case .scalar(.f16), .vector(_, .f16), .matrix(_, _, .f16), .cooperativeMatrix(_, _, .f16, _): true
        default: false
        }
    }
}
