import SAGAShaderIR

struct IntegerShaderHelper: Hashable {
    enum Operation: String {
        case divide = "div"
        case remainder = "rem"
        case shiftLeft = "shl"
        case shiftRight = "shr"

        var symbol: String {
            switch self {
            case .divide: "/"
            case .remainder: "%"
            case .shiftLeft: "<<"
            case .shiftRight: ">>"
            }
        }

        var isShift: Bool { self == .shiftLeft || self == .shiftRight }
    }

    let operation: Operation
    let scalar: IRScalar

    init?(operation: IRBinaryOperator, scalar: IRScalar) {
        guard [.i32, .u32, .i64, .u64].contains(scalar) else { return nil }
        switch operation {
        case .divide: self.operation = .divide
        case .modulo: self.operation = .remainder
        case .shiftLeft: self.operation = .shiftLeft
        case .shiftRight: self.operation = .shiftRight
        default: return nil
        }
        self.scalar = scalar
    }

    var name: String { "saga_\(integerLaneSuffix(scalar))_\(operation.rawValue)" }

    func lines(in format: ShaderOutputFormat) throws -> [String] {
        let type = try shaderVectorType(1, scalar: scalar, in: format)
        let countType = try shaderVectorType(1, scalar: .u32, in: format)
        if format == .wgsl {
            let rhs = operation.isShift ? countType : type
            return ["fn \(name)(a: \(type), b: \(rhs)) -> \(type) { return a \(operation.symbol) b; }"]
        }
        let wide = scalar.width == 8
        let unsigned = try shaderVectorType(1, scalar: wide ? .u64 : .u32, in: format)
        let zero = wide ? unsigned64Literal(0, in: format) : "0u"
        let one = wide ? unsigned64Literal(1, in: format) : "1u"
        let signBit = Int(scalar.width) * 8 - 1
        func bits(_ value: String, signed: Bool) throws -> String {
            try integerLaneBits(value, signed: signed, scalar: scalar, in: format)
        }
        if operation.isShift {
            guard operation == .shiftRight, scalar.kind == .sint else { return [] }
            let shifted = "((bits ^ sign_mask) >> (b & \(signBit)u)) ^ sign_mask"
            return [
                "\(type) \(name)(\(type) a, \(countType) b) {",
                "    \(unsigned) bits = \(try bits("a", signed: false));",
                "    \(unsigned) sign_mask = \(zero) - (bits >> \(signBit)u);",
                "    return \(try bits(shifted, signed: true));",
                "}",
            ]
        }
        if scalar.kind == .uint {
            return [
                "\(type) \(name)(\(type) a, \(type) b) {",
                "    \(type) divisor = b;",
                "    if (divisor == \(zero)) { divisor = \(one); }",
                "    return a \(operation.symbol) divisor;",
                "}",
            ]
        }
        // Unsigned magnitudes handle the most negative value without overflow.
        let negative = operation == .divide ? "(a < 0) != (b < 0)" : "a < 0"
        return [
            "\(type) \(name)(\(type) a, \(type) b) {",
            "    \(unsigned) dividend = \(try bits("a", signed: false));",
            "    \(unsigned) divisor = \(try bits("b", signed: false));",
            "    if (a < 0) { dividend = \(zero) - dividend; }",
            "    if (b < 0) { divisor = \(zero) - divisor; }",
            "    if (divisor == \(zero)) { divisor = \(one); }",
            "    \(unsigned) bits = dividend \(operation.symbol) divisor;",
            "    if (\(negative)) { bits = \(zero) - bits; }",
            "    return \(try bits("bits", signed: true));",
            "}",
        ]
    }
}

private func integerLaneSuffix(_ scalar: IRScalar) -> String {
    "\(scalar.kind == .sint ? "i" : "u")\(Int(scalar.width) * 8)"
}

private func integerLaneBits(_ value: String, signed: Bool, scalar: IRScalar, in format: ShaderOutputFormat) throws -> String {
    if scalar.width == 8 { return try wideIntegerBits(value, signed: signed, in: format) }
    return integerBits(value, signed: signed, in: format)
}

func integerBits(_ value: String, signed: Bool, width: UInt8? = nil, in format: ShaderOutputFormat) -> String {
    let type = signed ? "int" : "uint"
    let native = width.map { type + String($0) } ?? type
    let scalar = signed ? "i32" : "u32"
    switch format {
    case .msl: return "as_type<\(native)>(\(value))"
    case .hlsl: return "as\(type)(\(value))"
    case .wgsl:
        let wgsl = width.map { "vec\($0)<\(scalar)>" } ?? scalar
        return "\(wgsl)(\(value))"
    default:
        let glsl = width.map { "\(signed ? "ivec" : "uvec")\($0)" } ?? type
        return "\(glsl)(\(value))"
    }
}

func wideIntegerBits(_ value: String, signed: Bool, width: UInt8 = 1, in format: ShaderOutputFormat) throws -> String {
    let type = try shaderVectorType(width, scalar: signed ? .i64 : .u64, in: format)
    switch format {
    case .msl: return "as_type<\(type)>(\(value))"
    case .wgsl: return "bitcast<\(type)>(\(value))"
    default: return "\(type)(\(value))"
    }
}

func integer16Bits(_ value: String, signed: Bool, width: UInt8 = 1, in format: ShaderOutputFormat) throws -> String {
    let unsigned = try shaderVectorType(width, scalar: .u16, in: format)
    let narrowed = "\(unsigned)(\(value))"
    guard signed else { return narrowed }
    let type = try shaderVectorType(width, scalar: .i16, in: format)
    switch format {
    case .msl: return "as_type<\(type)>(\(narrowed))"
    case .wgsl: return "bitcast<\(type)>(\(narrowed))"
    default: return "\(type)(\(narrowed))"
    }
}

private func integer16Binary(
    _ op: IRBinaryOperator, scalar: IRScalar, left: String, right: String,
    in format: ShaderOutputFormat, helpers: inout Set<IntegerShaderHelper>
) throws -> String? {
    let signed = scalar == .i16
    let promoted = signed ? IRScalar.i32 : .u32
    if let helper = IntegerShaderHelper(operation: op, scalar: promoted) {
        if format == .wgsl { return "(\(left) \(helper.operation.symbol) \(right))" }
        // Promotion makes min(i16) / -1 representable and keeps native shifts inside the lane width.
        let type = try shaderVectorType(1, scalar: promoted, in: format)
        let rhs = helper.operation.isShift ? "(\(right) & 15u)" : "\(type)(\(right))"
        guard let value = try integerBinary(op, scalar: promoted, left: "\(type)(\(left))", right: rhs,
                                            in: format, helpers: &helpers) else { return nil }
        return try integer16Bits(value, signed: signed, in: format)
    }
    guard format != .wgsl, [.add, .subtract, .multiply, .and, .inclusiveOr, .exclusiveOr].contains(op) else {
        return nil
    }
    let symbol = try shaderBinarySymbol(op, type: .u32)
    // In C++ even ushort operands promote to signed int. Widen explicitly before multiplying.
    return try integer16Bits("uint(\(left)) \(symbol) uint(\(right))", signed: signed, in: format)
}

func wideIntegerBinary(
    _ op: IRBinaryOperator, scalar: IRScalar, width: UInt8 = 1, left: String, right: String,
    in format: ShaderOutputFormat
) throws -> String {
    if format == .glsl, width > 1 {
        let comparison: String? = switch op {
        case .equal: "equal"
        case .notEqual: "notEqual"
        case .less: "lessThan"
        case .lessEqual: "lessThanEqual"
        case .greater: "greaterThan"
        case .greaterEqual: "greaterThanEqual"
        default: nil
        }
        if let comparison { return "\(comparison)(\(left), \(right))" }
    }
    let symbol = try shaderBinarySymbol(op, type: scalar)
    if scalar == .i64, format != .wgsl, op == .add || op == .subtract || op == .multiply {
        let lhs = try wideIntegerBits(left, signed: false, width: width, in: format)
        let rhs = try wideIntegerBits(right, signed: false, width: width, in: format)
        return try wideIntegerBits("\(lhs) \(symbol) \(rhs)", signed: true, width: width, in: format)
    }
    return "(\(left) \(symbol) \(right))"
}

func integerBinary(
    _ op: IRBinaryOperator, scalar: IRScalar, left: String, right: String,
    in format: ShaderOutputFormat, helpers: inout Set<IntegerShaderHelper>
) throws -> String? {
    if scalar == .i16 || scalar == .u16 {
        return try integer16Binary(op, scalar: scalar, left: left, right: right, in: format, helpers: &helpers)
    }
    if let helper = IntegerShaderHelper(operation: op, scalar: scalar) {
        let signed = scalar.kind == .sint
        if format == .wgsl || !helper.operation.isShift || (op == .shiftRight && signed) {
            helpers.insert(helper)
            return "\(helper.name)(\(left), \(right))"
        }
        let bits = signed ? try integerLaneBits(left, signed: false, scalar: scalar, in: format) : left
        let mask = Int(scalar.width) * 8 - 1
        let shifted = "(\(bits) \(helper.operation.symbol) (\(right) & \(mask)u))"
        return signed ? try integerLaneBits(shifted, signed: true, scalar: scalar, in: format) : shifted
    }
    if scalar == .i32, format != .wgsl, op == .add || op == .subtract || op == .multiply {
        let symbol = try shaderBinarySymbol(op, type: scalar)
        let a = integerBits(left, signed: false, in: format)
        let b = integerBits(right, signed: false, in: format)
        return integerBits("\(a) \(symbol) \(b)", signed: true, in: format)
    }
    return nil
}

func signedNegation(_ value: String, in format: ShaderOutputFormat) -> String {
    if format == .wgsl { return "(-\(value))" }
    let bits = integerBits(value, signed: false, in: format)
    return integerBits("0u - \(bits)", signed: true, in: format)
}

func integerNegation(_ value: String, scalar: IRScalar, in format: ShaderOutputFormat) throws -> String {
    if scalar == .i32 { return signedNegation(value, in: format) }
    if format == .wgsl { return "(-\(value))" }
    if scalar == .i16 { return try integer16Bits("0u - uint(\(value))", signed: true, in: format) }
    let bits = try wideIntegerBits(value, signed: false, in: format)
    let zero = unsigned64Literal(0, in: format)
    return try wideIntegerBits("\(zero) - \(bits)", signed: true, in: format)
}

struct IntegerVectorHelper: Hashable {
    enum Operation: Hashable { case binary(IRBinaryOperator), negate }

    let operation: Operation
    let scalar: IRScalar
    let width: UInt8

    var name: String {
        let suffix: String
        switch operation {
        case .binary(let op): suffix = op.rawValue
        case .negate: suffix = "negate"
        }
        return "saga_vec\(width)_\(integerLaneSuffix(scalar))_\(suffix)"
    }

    func lines(in format: ShaderOutputFormat, helpers: inout Set<IntegerShaderHelper>) throws -> [String] {
        let left = try shaderVectorType(width, scalar: scalar, in: format)
        var right: String?
        var result = left
        if case .binary(let op) = operation {
            right = try shaderVectorType(width, scalar: op == .shiftLeft || op == .shiftRight ? .u32 : scalar, in: format)
            if [.equal, .notEqual, .less, .lessEqual, .greater, .greaterEqual].contains(op) {
                result = try shaderVectorType(width, scalar: .bool, in: format)
            }
        }
        var lanes: [String] = []
        for lane in "xyzw".prefix(Int(width)) {
            let a = "a.\(lane)"
            switch operation {
            case .negate: lanes.append(try integerNegation(a, scalar: scalar, in: format))
            case .binary(let op):
                let b = "b.\(lane)"
                if let lowered = try integerBinary(op, scalar: scalar, left: a, right: b, in: format, helpers: &helpers) {
                    lanes.append(lowered)
                } else {
                    let symbol = try shaderBinarySymbol(op, type: scalar)
                    lanes.append("\(a) \(symbol) \(b)")
                }
            }
        }
        let signature: String
        if format == .wgsl {
            signature = "fn \(name)(a: \(left)\(right.map { ", b: \($0)" } ?? "")) -> \(result)"
        } else {
            signature = "\(result) \(name)(\(left) a\(right.map { ", \($0) b" } ?? ""))"
        }
        return ["\(signature) { return \(result)(\(lanes.joined(separator: ", "))); }"]
    }
}

func integerPrelude(
    _ helpers: Set<IntegerShaderHelper>, vectors: Set<IntegerVectorHelper>, in format: ShaderOutputFormat
) throws -> [String] {
    var required = helpers
    let vectorLines = try vectors.sorted { $0.name < $1.name }.flatMap { try $0.lines(in: format, helpers: &required) }
    return try required.sorted { $0.name < $1.name }.flatMap { try $0.lines(in: format) } + vectorLines
}
