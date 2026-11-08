import SAGAShaderIR

enum WGSLFloatKind: Equatable { case abstract, f32, f16 }
struct WGSLFloatingConstant {
  let value: Double
  let kind: WGSLFloatKind
  var literal: IRLiteral {
    switch kind {
    case .abstract: .abstractFloat(value)
    case .f32: .f32(Float(value))
    case .f16: .f16Bits(Binary16.encode(value))
    }
  }

  static func rounded(_ value: Double, to kind: WGSLFloatKind, nagaFloatRounding: Bool) -> Self? {
    guard value.isFinite else { return nil }
    let result: Double
    switch kind {
    case .abstract:
      result = value
    case .f32:
      guard nagaFloatRounding || abs(value) <= Double(Float.greatestFiniteMagnitude) else { return nil }
      result = Double(Float(value))
    case .f16:
      guard nagaFloatRounding || abs(value) <= 65504 else { return nil }
      result = Double(Binary16.decode(Binary16.encode(value)))
    }
    return result.isFinite ? .init(value: result, kind: kind) : nil
  }
}

extension WGSLScalarEvaluator {
  func floating(
    _ value: Double, kind: WGSLFloatKind, at t: WGSLToken, nagaFloatRounding: Bool = false
  ) throws -> WGSLScalarConstant
  {
    guard value.isFinite else {
      throw wgslError(.invalidType, "Non-finite floating constant result", t)
    }
    guard let result = WGSLFloatingConstant.rounded(value, to: kind, nagaFloatRounding: nagaFloatRounding) else {
      throw wgslError(.invalidType, "Floating constant outside \(kind) finite range", t)
    }
    return .floating(result)
  }
  func floatValue(
    _ value: WGSLScalarConstant, kind: WGSLFloatKind, at t: WGSLToken, explicit: Bool = false
  ) throws -> WGSLScalarConstant {
    switch value {
    case .floating(let f):
      guard explicit || f.kind == kind || f.kind == .abstract else {
        throw wgslError(.typeMismatch, "Cannot automatically widen concrete float", t)
      }
      return try floating(f.value, kind: kind, at: t, nagaFloatRounding: f.kind == .abstract && parser.options.nagaFloatRounding)
    case .integer(let i):
      guard explicit || i.kind == .abstract else {
        throw wgslError(.typeMismatch, "Concrete integer cannot implicitly convert to float", t)
      }
      // Convert directly to Float, not via Double, to avoid integer double rounding.
      if kind == .f32 {
        let value = i.kind == .uint64 ? Float(i.bits) : Float(i.value)
        return .floating(.init(value: Double(value), kind: .f32))
      }
      return try floating(i.kind == .uint64 ? Double(i.bits) : Double(i.value), kind: kind, at: t,
                          nagaFloatRounding: i.kind == .abstract && parser.options.nagaFloatRounding)
    case .vector, .matrix, .array, .composite, .decomposition:
      throw wgslError(.typeMismatch, "Scalar float conversion requires scalar", t)
    case .boolean(let b):
      guard explicit else { throw wgslError(.typeMismatch, "Bool is not numeric", t) }
      return try floating(b ? 1 : 0, kind: kind, at: t)
    }
  }
  func floatingBinary(
    _ op: String, _ a: WGSLScalarConstant, _ b: WGSLScalarConstant, at t: WGSLToken
  ) throws -> WGSLScalarConstant {
    let commonType = try common(a.kind, b.kind, at: t)
    guard case .floating(let kind) = commonType,
      case .floating(let l) = try floatValue(a, kind: kind, at: t),
      case .floating(let r) = try floatValue(b, kind: kind, at: t)
    else { throw ShaderFailure.invalidHandle }
    switch op {
    case "==": return .boolean(l.value == r.value)
    case "!=": return .boolean(l.value != r.value)
    case "<": return .boolean(l.value < r.value)
    case "<=": return .boolean(l.value <= r.value)
    case ">": return .boolean(l.value > r.value)
    case ">=": return .boolean(l.value >= r.value)
    default: break
    }
    let result: Double
    if kind == .f32 {
      // Each operation executes in binary32 and rounds before the next AST node.
      let x = Float(l.value)
      let y = Float(r.value)
      let v: Float
      switch op {
      case "+": v = x + y
      case "-": v = x - y
      case "*": v = x * y
      case "/": v = x / y
      case "%": v = x.truncatingRemainder(dividingBy: y)
      default: throw wgslError(.typeMismatch, "Invalid floating operator", t)
      }
      result = Double(v)
    } else {
      switch op {
      case "+": result = l.value + r.value
      case "-": result = l.value - r.value
      case "*": result = l.value * r.value
      case "/": result = l.value / r.value
      case "%": result = l.value.truncatingRemainder(dividingBy: r.value)
      default: throw wgslError(.typeMismatch, "Invalid floating operator", t)
      }
    }
    return try floating(result, kind: kind, at: t)
  }
  func floatToInteger(_ f: WGSLFloatingConstant, target: WGSLIntegerConstant.Kind, at t: WGSLToken)
    throws -> WGSLScalarConstant
  {
    guard f.value.isFinite else {
      throw wgslError(.invalidType, "Non-finite integral conversion", t)
    }
    // Bounds are the greatest integers in range also representable by the SOURCE format.
    if target.isWide {
      let limit = target.isUnsigned ? Double(UInt64.max) : Double(Int64.max)
      let upper = f.kind == .f32 ? Double(Float(limit).nextDown) : limit.nextDown
      let lower = target.isUnsigned ? 0.0 : Double(Int64.min)
      let clipped = min(max(f.value, lower), upper).rounded(.towardZero)
      let value = target.isUnsigned ? Int64(bitPattern: UInt64(clipped)) : Int64(clipped)
      return .integer(.init(value: value, kind: target))
    }
    let low = target == .uint ? 0.0 : Double(Int32.min)
    let high: Double =
      target == .uint
      ? (f.kind == .f32 ? 4294967040.0 : 4294967295.0)
      : (f.kind == .f32 ? 2147483520.0 : 2147483647.0)
    let clipped = min(max(f.value, low), high).rounded(.towardZero)
    return .integer(.init(value: Int64(clipped), kind: target))
  }
  func converted(_ v: WGSLScalarConstant, to kind: WGSLConstKind, at t: WGSLToken) throws
    -> WGSLScalarConstant
  {
    if case .array(let count, let leaf) = kind, case .array(let array) = v {
      if v.kind == kind { return v }
      try limit()
      try parser.enter()
      defer { parser.nesting -= 1 }
      guard count == array.elements.count else {
        throw wgslError(.typeMismatch, "Array conversion shape mismatch", t)
      }
      try parser.budget.consume(.nodes, amount: count)
      let elements = try array.elements.map { try converted($0, to: leaf, at: t) }
      return .array(.init(elements: elements, elementKind: leaf))
    }
    if case .matrix(let c,let r,let leaf)=kind,case .matrix(let m)=v {
      guard m.columns.count == c,m.rows == r else {throw wgslError(.typeMismatch,"Matrix conversion shape",t)}
      var columns:[WGSLVectorConstant]=[]
      for column in m.columns {
        try parser.budget.consume(.work)
        guard case .vector(let converted)=try converted(.vector(column),to:.vector(r,leaf),at:t) else {throw ShaderFailure.invalidHandle}
        columns.append(converted)
      }
      return .matrix(.init(columns:columns,rows:r,element:leaf))
    }
    if case .vector(let count, let element) = kind, case .vector(let vector) = v {
      guard count == vector.elements.count else {
        throw wgslError(.typeMismatch, "Vector conversion shape", t)
      }
      var values: [WGSLScalarConstant] = []
      for x in vector.elements {
        try parser.budget.consume(.work)
        values.append(try scalarConvert(x, to: element, at: t, explicit: false))
      }
      return .vector(.init(elements: values, element: element))
    }
    if case .floating(let k) = kind { return try floatValue(v, kind: k, at: t) }
    if case .integer(let k) = kind, case .integer(let x) = v {
      return .integer(
        try WGSLIntegerEvaluator(parser: parser, tokens: [], fallback: t).materialized(x, as: k, at: t)
      )
    }
    if v.kind == kind { return v }
    throw wgslError(.typeMismatch, "Infeasible constant conversion", t)
  }
}

extension WGSLParser {
  func annotatedConstant(_ value: WGSLScalarConstant, type: WType, at t: WGSLToken) throws
    -> WGSLScalarConstant
  {
    switch types[type.index].kind {
    case .array(_, .constant, _), .structure:
      let target = try constantKind(type, at: t)
      let evaluator = WGSLScalarEvaluator(parser: self, tokens: [], fallback: t)
      guard try evaluator.arrayElementType(value.kind, target, at: t) == target else {
        throw wgslError(.typeMismatch, "Incompatible composite const annotation", t)
      }
      return try evaluator.converted(value, to: target, at: t)
    default: break
    }
    if case .matrix(let c,let r,let s)=types[type.index].kind {
      return try WGSLScalarEvaluator(parser:self,tokens:[],fallback:t).converted(value,to:.matrix(Int(c.rawValue),Int(r.rawValue),.scalar(s,at:t)),at:t)
    }
    if case .vector(let size, let scalar) = types[type.index].kind {
      return try WGSLScalarEvaluator(parser: self, tokens: [], fallback: t).converted(
        value, to: .vector(Int(size.rawValue), .scalar(scalar, at: t)), at: t)
    }
    guard case .scalar(let s) = types[type.index].kind else {
      throw wgslError(.unsupportedConstruct, "Aggregate const annotation unsupported", t)
    }
    let target = try WGSLConstKind.scalar(s, at: t)
    let e = WGSLScalarEvaluator(parser: self, tokens: [], fallback: t)
    if value.kind == target { return value }
    switch value.kind {
    case .integer(.abstract), .floating(.abstract): return try e.converted(value, to: target, at: t)
    default: throw wgslError(.typeMismatch, "Concrete const annotation mismatch", t)
    }
  }
}

extension WGSLScalarConstant {
  var isFloat: Bool {
    if case .floating = self { return true }
    return false
  }
}

extension WGSLScalarEvaluator {
  func floatingRoot(_ v: WGSLScalarConstant, result: WGSLConstKind, inverse: Bool, at t: WGSLToken)
    throws -> WGSLScalarConstant
  {
    if case .vector(let size, let element) = result, case .vector(let vector) = v {
      guard vector.elements.count == size else { throw ShaderFailure.invalidHandle }
      var xs: [WGSLScalarConstant] = []
      for x in vector.elements {
        try parser.budget.consume(.work)
        xs.append(try floatingRoot(x, result: element, inverse: inverse, at: t))
      }
      return .vector(.init(elements: xs, element: element))
    }
    guard case .floating(let kind) = result,
      case .floating(let f) = try floatValue(v, kind: kind, at: t)
    else { throw ShaderFailure.invalidHandle }
    let value: Double
    if kind == .f32 {
      let root = Float(f.value).squareRoot()
      value = Double(inverse ? Float(1) / root : root)
    } else {
      let root = f.value.squareRoot()
      value = inverse ? 1 / root : root
    }
    return try floating(value, kind: kind, at: t)
  }
}
