package struct IRPackingBuiltin: Sendable {
    package enum Encoding: Sendable {
        case snorm, unorm, binary16, sint, uint, sintClamp, uintClamp

        package var scalar: IRScalar {
            switch self {
            case .snorm, .unorm, .binary16: .f32
            case .sint, .sintClamp: .i32
            case .uint, .uintClamp: .u32
            }
        }

        package var isSignedIntegerField: Bool {
            self == .snorm || self == .sint || self == .sintClamp
        }
    }

    package let encoding: Encoding
    package let size: IRVectorSize
    package let packs: Bool

    package var bitsPerLane: Int { 32 / Int(size.rawValue) }
    package var mask: UInt32 { (1 << bitsPerLane) - 1 }
    package var normalizedScale: UInt32 { encoding == .snorm ? mask >> 1 : mask }
    package var argumentType: IRTypeKind { packs ? vectorType : .scalar(.u32) }
    package var resultType: IRTypeKind { packs ? .scalar(.u32) : vectorType }

    private var vectorType: IRTypeKind { .vector(size: size, scalar: encoding.scalar) }
}

extension IRMathFunction {
    package var packingBuiltin: IRPackingBuiltin? {
        switch self {
        case .pack4x8snorm: .init(encoding: .snorm, size: .four, packs: true)
        case .pack4x8unorm: .init(encoding: .unorm, size: .four, packs: true)
        case .pack2x16snorm: .init(encoding: .snorm, size: .two, packs: true)
        case .pack2x16unorm: .init(encoding: .unorm, size: .two, packs: true)
        case .pack2x16float: .init(encoding: .binary16, size: .two, packs: true)
        case .pack4xI8: .init(encoding: .sint, size: .four, packs: true)
        case .pack4xU8: .init(encoding: .uint, size: .four, packs: true)
        case .pack4xI8Clamp: .init(encoding: .sintClamp, size: .four, packs: true)
        case .pack4xU8Clamp: .init(encoding: .uintClamp, size: .four, packs: true)
        case .unpack4x8snorm: .init(encoding: .snorm, size: .four, packs: false)
        case .unpack4x8unorm: .init(encoding: .unorm, size: .four, packs: false)
        case .unpack2x16snorm: .init(encoding: .snorm, size: .two, packs: false)
        case .unpack2x16unorm: .init(encoding: .unorm, size: .two, packs: false)
        case .unpack2x16float: .init(encoding: .binary16, size: .two, packs: false)
        case .unpack4xI8: .init(encoding: .sint, size: .four, packs: false)
        case .unpack4xU8: .init(encoding: .uint, size: .four, packs: false)
        default: nil
        }
    }
}
