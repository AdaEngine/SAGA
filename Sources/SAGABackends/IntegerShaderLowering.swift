import SAGAShaderIR

enum IntegerShaderHelper: String, Hashable {
    case i32Div = "saga_i32_div"
    case i32Rem = "saga_i32_rem"
    case u32Div = "saga_u32_div"
    case u32Rem = "saga_u32_rem"
    case i32Left = "saga_i32_shl"
    case i32Right = "saga_i32_shr"
    case u32Left = "saga_u32_shl"
    case u32Right = "saga_u32_shr"

    init?(operation: IRBinaryOperator, scalar: IRScalar) {
        switch (operation, scalar) {
        case (.divide, .i32): self = .i32Div
        case (.modulo, .i32): self = .i32Rem
        case (.divide, .u32): self = .u32Div
        case (.modulo, .u32): self = .u32Rem
        case (.shiftLeft, .i32): self = .i32Left
        case (.shiftRight, .i32): self = .i32Right
        case (.shiftLeft, .u32): self = .u32Left
        case (.shiftRight, .u32): self = .u32Right
        default: return nil
        }
    }

    func lines(in format: ShaderOutputFormat) -> [String] {
        if format == .wgsl {
            let lhs: String
            let rhs: String
            let symbol: String
            switch self {
            case .i32Div: (lhs, rhs, symbol) = ("i32", "i32", "/")
            case .i32Rem: (lhs, rhs, symbol) = ("i32", "i32", "%")
            case .u32Div: (lhs, rhs, symbol) = ("u32", "u32", "/")
            case .u32Rem: (lhs, rhs, symbol) = ("u32", "u32", "%")
            case .i32Left: (lhs, rhs, symbol) = ("i32", "u32", "<<")
            case .i32Right: (lhs, rhs, symbol) = ("i32", "u32", ">>")
            case .u32Left: (lhs, rhs, symbol) = ("u32", "u32", "<<")
            case .u32Right: (lhs, rhs, symbol) = ("u32", "u32", ">>")
            }
            return ["fn \(rawValue)(a: \(lhs), b: \(rhs)) -> \(lhs) { return a \(symbol) b; }"]
        }
        switch self {
        case .u32Div, .u32Rem:
            let symbol = self == .u32Div ? "/" : "%"
            return [
                "uint \(rawValue)(uint a, uint b) {",
                "    uint divisor = b;",
                "    if (divisor == 0u) { divisor = 1u; }",
                "    return a \(symbol) divisor;",
                "}",
            ]
        case .i32Div, .i32Rem:
            // Magnitudes use unsigned arithmetic, including abs(INT_MIN) and INT_MIN / -1.
            let symbol = self == .i32Div ? "/" : "%"
            let negative = self == .i32Div ? "(a < 0) != (b < 0)" : "a < 0"
            return [
                "int \(rawValue)(int a, int b) {",
                "    uint dividend = \(integerBits("a", signed: false, in: format));",
                "    uint divisor = \(integerBits("b", signed: false, in: format));",
                "    if (a < 0) { dividend = 0u - dividend; }",
                "    if (b < 0) { divisor = 0u - divisor; }",
                "    if (divisor == 0u) { divisor = 1u; }",
                "    uint bits = dividend \(symbol) divisor;",
                "    if (\(negative)) { bits = 0u - bits; }",
                "    return \(integerBits("bits", signed: true, in: format));",
                "}",
            ]
        case .i32Right:
            return [
                "int \(rawValue)(int a, uint b) {",
                "    uint bits = \(integerBits("a", signed: false, in: format));",
                "    uint sign_mask = 0u - (bits >> 31u);",
                "    return \(integerBits("((bits ^ sign_mask) >> (b & 31u)) ^ sign_mask", signed: true, in: format));",
                "}",
            ]
        case .i32Left, .u32Left, .u32Right:
            return []
        }
    }
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

func integerBinary(
    _ op: IRBinaryOperator, scalar: IRScalar, left: String, right: String,
    in format: ShaderOutputFormat, helpers: inout Set<IntegerShaderHelper>
) throws -> String? {
    if let helper = IntegerShaderHelper(operation: op, scalar: scalar) {
        let shift = op == .shiftLeft || op == .shiftRight
        if format == .wgsl || !shift || helper == .i32Right {
            helpers.insert(helper)
            return "\(helper.rawValue)(\(left), \(right))"
        }
        let symbol = op == .shiftLeft ? "<<" : ">>"
        let bits = scalar == .i32 ? integerBits(left, signed: false, in: format) : left
        let shifted = "(\(bits) \(symbol) (\(right) & 31u))"
        return scalar == .i32 ? integerBits(shifted, signed: true, in: format) : shifted
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

struct IntegerVectorHelper: Hashable {
    enum Operation: Hashable { case binary(IRBinaryOperator), negate }

    let operation: Operation
    let signed: Bool
    let width: UInt8

    var name: String {
        let suffix: String
        switch operation {
        case .binary(let op): suffix = op.rawValue
        case .negate: suffix = "negate"
        }
        return "saga_vec\(width)_\(signed ? "i32" : "u32")_\(suffix)"
    }

    func lines(in format: ShaderOutputFormat, helpers: inout Set<IntegerShaderHelper>) throws -> [String] {
        let scalar: IRScalar = signed ? .i32 : .u32
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
            case .negate: lanes.append(signedNegation(a, in: format))
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
    return required.sorted { $0.rawValue < $1.rawValue }.flatMap { $0.lines(in: format) } + vectorLines
}
