import SAGAShaderIR

// Typed integer values and audited arithmetic, shared by the scalar constant tree.
struct WGSLIntegerConstant {
  enum Kind {
    case abstract, sint16, uint16, sint, uint, sint64, uint64

    var isUnsigned: Bool { self == .uint16 || self == .uint || self == .uint64 }
    var isNarrow: Bool { self == .sint16 || self == .uint16 }
    var isWide: Bool { self == .sint64 || self == .uint64 }
    var bitWidth: Int64 { Int64(scalar.width) * 8 }
    var signedMinimum: Int64 { Int64.min >> (64 - bitWidth) }
    var scalar: IRScalar {
      switch self {
      case .abstract: .abstractInt
      case .sint16: .i16
      case .uint16: .u16
      case .sint: .i32
      case .uint: .u32
      case .sint64: .i64
      case .uint64: .u64
      }
    }

    func truncating(_ value: Int64) -> Int64 {
      switch self {
      case .sint16: Int64(Int16(truncatingIfNeeded: value))
      case .uint16: Int64(UInt16(truncatingIfNeeded: value))
      case .sint: Int64(Int32(truncatingIfNeeded: value))
      case .uint: Int64(UInt32(truncatingIfNeeded: value))
      case .abstract, .sint64, .uint64: value
      }
    }
  }
  // Unsigned 64-bit values retain their full bit pattern.
  var value: Int64
  var kind: Kind
  var scalar: IRScalar { kind.scalar }
  var bits: UInt64 { UInt64(bitPattern: value) }
  var literal: IRLiteral {
    switch kind {
    case .abstract: .abstractInt(value)
    case .sint16: .i16(Int16(value))
    case .uint16: .u16(UInt16(value))
    case .sint: .i32(Int32(value))
    case .uint: .u32(UInt32(value))
    case .sint64: .i64(value)
    case .uint64: .u64(bits)
    }
  }
}

struct WGSLIntegerEvaluator {
  let parser: WGSLParser
  let tokens: [WGSLToken]
  let fallback: WGSLToken
  func checked(_ value: Int64, _ kind: WGSLIntegerConstant.Kind, at t: WGSLToken) throws
    -> WGSLIntegerConstant
  {
    let valid: Bool = switch kind {
    case .abstract, .sint64: true
    case .sint16: Int16(exactly: value) != nil
    case .uint16: UInt16(exactly: value) != nil
    case .sint: Int32(exactly: value) != nil
    case .uint: UInt32(exactly: value) != nil
    case .uint64: value >= 0
    }
    guard valid else {
      throw wgslError(.invalidType, "Integer constant overflow or unrepresentable conversion", t)
    }
    return .init(value: value, kind: kind)
  }
  func materialized(_ x: WGSLIntegerConstant, as kind: WGSLIntegerConstant.Kind, at t: WGSLToken) throws
    -> WGSLIntegerConstant
  {
    if x.kind == kind { return x }
    // The native Naga extension materializes abstract integer values by keeping their low 16 bits.
    if x.kind == .abstract, kind.isNarrow { return .init(value: kind.truncating(x.value), kind: kind) }
    return try checked(x.value, kind, at: t)
  }
  func compare(_ op: String, _ lhs: WGSLIntegerConstant, _ rhs: WGSLIntegerConstant) -> Bool {
    let equal = lhs.value == rhs.value
    let less = lhs.kind == .uint64 ? lhs.bits < rhs.bits : lhs.value < rhs.value
    return switch op {
    case "==": equal
    case "!=": !equal
    case "<": less
    case "<=": less || equal
    case ">": !less && !equal
    default: !less
    }
  }
  func binary(_ op: String, _ lhs: WGSLIntegerConstant, _ rhs: WGSLIntegerConstant, at t: WGSLToken)
    throws -> WGSLIntegerConstant
  {
    var a = lhs
    var b = rhs
    if op == "<<" || op == ">>" {
      guard b.kind == .abstract || b.kind == .uint else {
        throw wgslError(.typeMismatch, "Shift count requires u32 or abstract integer", t)
      }
      let width = a.kind.bitWidth
      guard b.value >= 0 && b.value < width else {
        throw wgslError(.invalidType, "Constant shift count outside bit width", t)
      }
      if a.kind == .uint64 {
        if op == ">>" { return .init(value: Int64(bitPattern: a.bits >> b.value), kind: .uint64) }
        guard a.bits <= UInt64.max >> b.value else {
          throw wgslError(.invalidType, "Constant shift overflow", t)
        }
        return .init(value: Int64(bitPattern: a.bits << b.value), kind: .uint64)
      }
      if op == ">>" { return try checked(a.value >> b.value, a.kind, at: t) }
      // WGSL const left shift requires no significant bits to be lost.
      var value = a.value
      for _ in 0..<Int(b.value) {
        try parser.budget.consume(.work)
        let (n, overflow) = value.multipliedReportingOverflow(by: 2)
        guard !overflow else { throw wgslError(.invalidType, "Constant shift overflow", t) }
        value = try checked(n, a.kind, at: t).value
      }
      return try checked(value, a.kind, at: t)
    }
    if a.kind != b.kind {
      if a.kind == .abstract {
        a = try materialized(a, as: b.kind, at: t)
      } else if b.kind == .abstract {
        b = try materialized(b, as: a.kind, at: t)
      } else {
        throw wgslError(.typeMismatch, "Mixed concrete integer types", t)
      }
    }
    if a.kind.isWide {
      let value: Int64
      switch op {
      case "+": value = a.value &+ b.value
      case "-": value = a.value &- b.value
      case "*": value = a.value &* b.value
      case "&": value = a.value & b.value
      case "|": value = a.value | b.value
      case "^": value = a.value ^ b.value
      case "/", "%":
        guard b.value != 0, a.kind == .uint64 || a.value != Int64.min || b.value != -1 else {
          throw wgslError(.invalidType, "Invalid constant division/remainder", t)
        }
        if a.kind == .uint64 {
          value = Int64(bitPattern: op == "/" ? a.bits / b.bits : a.bits % b.bits)
        } else {
          value = op == "/" ? a.value / b.value : a.value % b.value
        }
      default: throw wgslError(.unsupportedConstruct, "Integer operator not implemented", t)
      }
      return .init(value: value, kind: a.kind)
    }
    if a.kind != .abstract && ["+", "-", "*"].contains(op) {
      let bits = op == "+" ? a.value &+ b.value : op == "-" ? a.value &- b.value : a.value &* b.value
      return .init(value: a.kind.truncating(bits), kind: a.kind)
    }
    let n: Int64
    let overflow: Bool
    switch op {
    case "+": (n, overflow) = a.value.addingReportingOverflow(b.value)
    case "-": (n, overflow) = a.value.subtractingReportingOverflow(b.value)
    case "*": (n, overflow) = a.value.multipliedReportingOverflow(by: b.value)
    case "/", "%":
      guard b.value != 0, !(!a.kind.isUnsigned && a.value == a.kind.signedMinimum && b.value == -1)
      else { throw wgslError(.invalidType, "Invalid constant division/remainder", t) }
      n = op == "/" ? a.value / b.value : a.value % b.value
      overflow = false
    case "&":
      n = a.value & b.value
      overflow = false
    case "|":
      n = a.value | b.value
      overflow = false
    case "^":
      n = a.value ^ b.value
      overflow = false
    default: throw wgslError(.unsupportedConstruct, "Integer operator not implemented", t)
    }
    guard !overflow else { throw wgslError(.invalidType, "Integer constant overflow", t) }
    return try checked(n, a.kind, at: t)
  }
}

extension WGSLParser {
  func integerConstant(_ tokens: [WGSLToken], at t: WGSLToken) throws -> WGSLIntegerConstant {
    guard !tokens.isEmpty else {
      throw wgslError(.unexpectedToken, "Missing constant expression", t)
    }
    let result = try scalarConstant(tokens, at: t)
    guard case .integer(let integer) = result else {
      throw wgslError(
        .typeMismatch, "Attribute/count requires integer, not bool", tokens.first ?? t)
    }
    return integer
  }
  func attributeConstant(_ a: Attribute) throws -> WGSLIntegerConstant {
    guard a.expressions.count == 1, let expression = a.expressions.first else {
      throw wgslError(.unexpectedToken, "Expected one attribute argument", a.name)
    }
    return try integerConstant(expression, at: a.name)
  }
  func attributeValue(_ v: WGSLIntegerConstant, at t: WGSLToken) throws -> UInt32 {
    // An abstract integer is materialized as i32, per the attribute parameter rule.
    guard !v.kind.isWide, let value = UInt32(exactly: v.value),
      v.kind != .abstract || Int32(exactly: v.value) != nil
    else {
      throw wgslError(.invalidType, "Attribute integer must resolve to a non-negative 16- or 32-bit integer", t)
    }
    return value
  }
}
