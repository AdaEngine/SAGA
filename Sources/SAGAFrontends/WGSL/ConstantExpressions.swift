import SAGAShaderIR

// Typed integer values and audited arithmetic, shared by the scalar constant tree.
struct WGSLIntegerConstant {
  enum Kind { case abstract, sint, uint }
  var value: Int64
  var kind: Kind
  var scalar: IRScalar { kind == .abstract ? .abstractInt : kind == .sint ? .i32 : .u32 }
  var literal: IRLiteral {
    switch kind {
    case .abstract: .abstractInt(value)
    case .sint: .i32(Int32(value))
    case .uint: .u32(UInt32(value))
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
    let valid =
      kind == .abstract
      || (kind == .sint ? Int32(exactly: value) != nil : UInt32(exactly: value) != nil)
    guard valid else {
      throw wgslError(.invalidType, "Integer constant overflow or unrepresentable conversion", t)
    }
    return .init(value: value, kind: kind)
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
      let width = a.kind == .abstract ? 64 : 32
      guard b.value >= 0 && b.value < width else {
        throw wgslError(.invalidType, "Constant shift count outside bit width", t)
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
        a = try checked(a.value, b.kind, at: t)
      } else if b.kind == .abstract {
        b = try checked(b.value, a.kind, at: t)
      } else {
        throw wgslError(.typeMismatch, "Mixed concrete integer types", t)
      }
    }
    if a.kind != .abstract && ["+", "-", "*"].contains(op) {
      let x = UInt32(truncatingIfNeeded: a.value)
      let y = UInt32(truncatingIfNeeded: b.value)
      let bits: UInt32 = op == "+" ? x &+ y : op == "-" ? x &- y : x &* y
      return .init(
        value: a.kind == .uint ? Int64(bits) : Int64(Int32(bitPattern: bits)), kind: a.kind)
    }
    let n: Int64
    let overflow: Bool
    switch op {
    case "+": (n, overflow) = a.value.addingReportingOverflow(b.value)
    case "-": (n, overflow) = a.value.subtractingReportingOverflow(b.value)
    case "*": (n, overflow) = a.value.multipliedReportingOverflow(by: b.value)
    case "/", "%":
      guard b.value != 0, !(a.value == Int64.min && b.value == -1),
        !(a.kind == .sint && a.value == Int64(Int32.min) && b.value == -1)
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
    guard let value = UInt32(exactly: v.value),
      v.kind != .abstract || Int32(exactly: v.value) != nil
    else {
      throw wgslError(.invalidType, "Attribute integer must resolve to non-negative i32/u32", t)
    }
    return value
  }
}
