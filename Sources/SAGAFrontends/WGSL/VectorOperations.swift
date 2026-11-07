import SAGAShaderIR

extension WGSLScalarEvaluator {
  func shaped(_ leaf: WGSLConstKind, like shape: WGSLConstKind) -> WGSLConstKind {
    shape.lanes == 1 ? leaf : .vector(shape.lanes, leaf)
  }
  mutating func vectorBinaryType(_ op: String, _ lhs: Int, _ rhs: Int, at t: WGSLToken) throws
    -> WGSLConstKind
  {
    let a = nodes[lhs].kind
    let b = nodes[rhs].kind
    let width = max(a.lanes, b.lanes)
    let arithmetic = ["+", "-", "*", "/", "%"].contains(op)
    guard (2...4).contains(width),
      a.lanes == b.lanes || (arithmetic && (a.lanes == 1 || b.lanes == 1))
    else {
      throw wgslError(.typeMismatch, "Vector operand shapes do not match a WGSL overload", t)
    }
    let x = a.scalarLeaf
    let y = b.scalarLeaf
    if op == "<<" || op == ">>" {
      guard case .integer = x, case .integer(let k) = y, k != .sint else {
        throw wgslError(.typeMismatch, "Vector shift needs integer values and u32 counts", t)
      }
      if k == .abstract { conversions.append((rhs, .vector(width, .integer(.uint)), t)) }
      return a
    }
    let comparison = ["==", "!=", "<", "<=", ">", ">="].contains(op)
    let leaf: WGSLConstKind
    if x == .boolean || y == .boolean {
      guard x == y, ["==", "!=", "&", "|"].contains(op) else {
        throw wgslError(.typeMismatch, "No boolean vector overload for operator", t)
      }
      leaf = .boolean
    } else {
      leaf = try common(x, y, at: t)
      if ["&", "|", "^"].contains(op) {
        guard case .integer = leaf else {
          throw wgslError(.typeMismatch, "Bitwise vectors require integers", t)
        }
      } else {
        guard arithmetic || comparison else {
          throw wgslError(.typeMismatch, "Unsupported vector operator overload", t)
        }
      }
    }
    for arg in [lhs, rhs] where nodes[arg].kind.scalarLeaf != leaf {
      conversions.append((arg, shaped(leaf, like: nodes[arg].kind), t))
    }
    return .vector(width, comparison ? .boolean : leaf)
  }
  func scalarBinaryValue(
    _ op: String, _ a: WGSLScalarConstant, _ b: WGSLScalarConstant, at t: WGSLToken
  ) throws -> WGSLScalarConstant {
    let arithmetic = WGSLIntegerEvaluator(parser: parser, tokens: [], fallback: t)
    if case .boolean(let l) = a, case .boolean(let r) = b {
      switch op {
      case "&", "&&": return .boolean(l && r)
      case "|", "||": return .boolean(l || r)
      case "==": return .boolean(l == r)
      case "!=": return .boolean(l != r)
      default: throw wgslError(.typeMismatch, "Invalid boolean operation", t)
      }
    }
    if a.isFloat || b.isFloat { return try floatingBinary(op, a, b, at: t) }
    guard case .integer(var l) = a, case .integer(var r) = b else {
      throw wgslError(.typeMismatch, "Scalar numeric operands required", t)
    }
    if ["==", "!=", "<", "<=", ">", ">="].contains(op) {
      guard case .integer(let k) = try common(a.kind, b.kind, at: t) else {
        throw ShaderFailure.invalidHandle
      }
      l = try arithmetic.checked(l.value, k, at: t)
      r = try arithmetic.checked(r.value, k, at: t)
      switch op {
      case "==": return .boolean(l.value == r.value)
      case "!=": return .boolean(l.value != r.value)
      case "<": return .boolean(l.value < r.value)
      case "<=": return .boolean(l.value <= r.value)
      case ">": return .boolean(l.value > r.value)
      default: return .boolean(l.value >= r.value)
      }
    }
    return .integer(try arithmetic.binary(op, l, r, at: t))
  }
  func components(_ v: WGSLScalarConstant, width: Int, at t: WGSLToken) throws
    -> [WGSLScalarConstant]
  {
    if case .vector(let x) = v {
      guard x.elements.count == width else {
        throw wgslError(.typeMismatch, "Vector value shape", t)
      }
      return x.elements
    }
    return Array(repeating: v, count: width)
  }
  func vectorBinaryValue(
    _ op: String, _ a: WGSLScalarConstant, _ b: WGSLScalarConstant, result: WGSLConstKind,
    at t: WGSLToken
  ) throws -> WGSLScalarConstant {
    guard case .vector(let width, let leaf) = result else { throw ShaderFailure.invalidHandle }
    let left = try components(a, width: width, at: t)
    let right = try components(b, width: width, at: t)
    var values: [WGSLScalarConstant] = []
    for (x, y) in zip(left, right) {
      try parser.budget.consume(.work)
      values.append(try scalarBinaryValue(op, x, y, at: t))
    }
    return .vector(.init(elements: values, element: leaf))
  }
  mutating func selectType(_ args: [Int], at t: WGSLToken) throws -> WGSLConstKind {
    let a = nodes[args[0]].kind
    let b = nodes[args[1]].kind
    let c = nodes[args[2]].kind
    guard a.lanes == b.lanes else {
      throw wgslError(.typeMismatch, "Select value shapes must match", t)
    }
    guard c == .boolean || (a.lanes > 1 && c == .vector(a.lanes, .boolean)) else {
      throw wgslError(.typeMismatch, "Select condition must be bool or matching bool vector", t)
    }
    let leaf: WGSLConstKind
    if a.scalarLeaf == .boolean, b.scalarLeaf == .boolean {
      leaf = .boolean
    } else {
      leaf = try common(a.scalarLeaf, b.scalarLeaf, at: t)
    }
    let result = shaped(leaf, like: a)
    for arg in args.prefix(2) where nodes[arg].kind != result {
      conversions.append((arg, result, t))
    }
    return result
  }
  func selectedValue(
    _ a: WGSLScalarConstant, _ b: WGSLScalarConstant, _ c: WGSLScalarConstant,
    result: WGSLConstKind, at t: WGSLToken
  ) throws -> WGSLScalarConstant {
    let first = try converted(a, to: result, at: t)
    let second = try converted(b, to: result, at: t)
    if case .boolean(let choose) = c { return choose ? second : first }
    guard case .vector(let cond) = c, case .vector(let l) = first, case .vector(let r) = second,
      case .vector(let count, let leaf) = result, cond.elements.count == count
    else { throw ShaderFailure.invalidHandle }
    var values: [WGSLScalarConstant] = []
    for i in 0..<count {
      try parser.budget.consume(.work)
      guard case .boolean(let choose) = cond.elements[i] else { throw ShaderFailure.invalidHandle }
      values.append(choose ? r.elements[i] : l.elements[i])
    }
    return .vector(.init(elements: values, element: leaf))
  }
  mutating func primary() throws -> Int {
    var base = try prefix()
    while index < tokens.count {
      if token.text == "." {
        _ = try take()
        let name = try take()
        guard case .vector(let width, let leaf) = nodes[base].kind, name.kind == .word else {
          throw wgslError(.typeMismatch, "Swizzle requires vector and member name", name)
        }
        var indices: [Int] = []
        var alphabet: Int?
        for c in name.text.utf8 {
          try parser.budget.consume(.work)
          let xy: [UInt8] = [120, 121, 122, 119]
          let rgba: [UInt8] = [114, 103, 98, 97]
          let n: Int
          let family: Int
          if let i = xy.firstIndex(of: c) {
            n = i
            family = 0
          } else if let i = rgba.firstIndex(of: c) {
            n = i
            family = 1
          } else {
            throw wgslError(.invalidType, "Invalid swizzle alphabet", name)
          }
          guard alphabet == nil || alphabet == family else {
            throw wgslError(.invalidType, "Mixed swizzle alphabets", name)
          }
          alphabet = family
          guard n < width, indices.count < 4 else {
            throw wgslError(.invalidType, "Swizzle component/count outside vector bounds", name)
          }
          indices.append(n)
        }
        guard !indices.isEmpty else { throw wgslError(.invalidType, "Empty swizzle", name) }
        base = try append(
          .swizzle(base, indices), indices.count == 1 ? leaf : .vector(indices.count, leaf), name)
      } else if token.text == "[" {
        let at = try take()
        let leaf: WGSLConstKind
        switch nodes[base].kind {
        case .vector(_, let k): leaf = k
        case .matrix(_, let rows, let k): leaf = .vector(rows, k)
        default: throw wgslError(.typeMismatch, "Constant index requires vector or matrix", at)
        }
        let begin = index
        let ix = try parse()
        guard index < tokens.count, token.text == "]" else {
          throw wgslError(.unexpectedToken, "Unclosed constant index", at)
        }
        let end = index
        _ = try take()
        guard case .integer = nodes[ix].kind else {
          throw wgslError(.typeMismatch, "Vector index must be scalar integer", tokens[begin])
        }
        let loc = WGSLToken(
          kind: .symbol, text: "index", start: tokens[begin].start,
          end: tokens[max(begin, end - 1)].end)
        base = try append(.index(base, ix), leaf, loc)
      } else {
        break
      }
    }
    return base
  }
  mutating func checkedIndex(_ base: Int, _ ix: Int, at t: WGSLToken) throws -> Int {
    let count: Int
    switch nodes[base].kind {
    case .vector(let n, _): count = n
    case .matrix(let c, _, _): count = c
    default: throw ShaderFailure.invalidHandle
    }
    guard case .integer(let value) = try value(ix), value.value >= 0, value.value < Int64(count)
    else { throw wgslError(.invalidType, "Constant aggregate index out of bounds", t) }
    return Int(value.value)
  }
  mutating func vectorCreationChecks(_ n: WGSLConstNode, op: String, lhs: Int, rhs: Int) throws {
    let leaf = n.kind.scalarLeaf
    guard case .integer(let kind) = leaf else { return }
    let count = n.kind.lanes
    let rs = try components(value(rhs), width: count, at: n.token)
    var left: [WGSLScalarConstant]?
    let arithmetic = WGSLIntegerEvaluator(parser: parser, tokens: [], fallback: n.token)
    for i in 0..<count {
      try parser.budget.consume(.work)
      guard case .integer(let r) = rs[i] else { throw ShaderFailure.invalidHandle }
      if op == "/" || op == "%" {
        guard r.value != 0 else {
          throw wgslError(.invalidType, "Constant vector divisor is zero", n.token)
        }
        if r.value == -1, kind != .uint {
          if left == nil { left = try components(value(lhs), width: count, at: n.token) }
          guard case .integer(let l) = left![i] else { throw ShaderFailure.invalidHandle }
          let typed = try arithmetic.checked(l.value, kind, at: n.token)
          guard typed.value != (kind == .sint ? Int64(Int32.min) : Int64.min) else {
            throw wgslError(.invalidType, "Constant vector division overflow", n.token)
          }
        }
      } else {
        guard r.value >= 0, r.value < (kind == .abstract ? 64 : 32) else {
          throw wgslError(.invalidType, "Constant vector shift exceeds bit width", n.token)
        }
        if op == "<<" {
          if left == nil { left = try components(value(lhs), width: count, at: n.token) }
          guard case .integer(let l) = left![i] else { throw ShaderFailure.invalidHandle }
          _ = try arithmetic.binary(op, l, r, at: n.token)
        }
      }
    }
  }
}

// Only pure, already-lowered constant compose/access nodes may fold for contextual materialization.
// Mutable loads or effectful expressions never use this path.
extension WGSLParser {
  func abstractVectorLeaves(_ h: WExpr, at t: WGSLToken) throws -> [IRLiteral]? {
    guard nesting < min(budget.limits.depth, 24) else { throw ShaderFailure.budgetExceeded(.depth) }
    try enter()
    defer { nesting -= 1 }
    guard body.expressions.indices.contains(h.index) else { throw ShaderFailure.invalidHandle }
    switch body.expressions[h.index].kind {
    case .compose(_, let xs):
      var values: [IRLiteral] = []
      guard (2...4).contains(xs.count) else { return nil }
      for x in xs {
        try budget.consume(.work)
        guard let leaf = try abstractScalarLeaf(x, at: t) else { return nil }
        values.append(leaf)
      }
      return values
    case .accessIndex(let base, let i):
      return try abstractMatrixColumn(base, index: Int64(i), at:t)
    case .access(let base, let index):
      guard body.expressions.indices.contains(index.index) else {throw ShaderFailure.invalidHandle}
      let i:Int64
      switch body.expressions[index.index].kind {case .literal(.u32(let n)):i=Int64(n);case .literal(.i32(let n)):i=Int64(n);default:return nil}
      return try abstractMatrixColumn(base,index:i,at:t)
    case .swizzle(let n, let base, let p):
      guard let xs = try abstractVectorLeaves(base, at: t) else { return nil }
      let pattern: [IRSwizzleComponent] = [p.x, p.y, p.z, p.w]
      let order: [IRSwizzleComponent: Int] = [.x: 0, .y: 1, .z: 2, .w: 3]
      var values: [IRLiteral] = []
      for component in pattern.prefix(Int(n.rawValue)) {
        try budget.consume(.work)
        guard let i = order[component], xs.indices.contains(i) else {
          throw wgslError(.invalidType, "Constant swizzle out of bounds", t)
        }
        values.append(xs[i])
      }
      return values
    default: return nil
    }
  }
  func abstractMatrixColumn(_ base:WExpr,index:Int64,at t:WGSLToken) throws -> [IRLiteral]? {
    try budget.consume(.work)
    guard body.expressions.indices.contains(base.index) else {throw ShaderFailure.invalidHandle}
    let baseExpression=body.expressions[base.index]
    guard case .matrix(let width,_,_)=types[baseExpression.type.index].kind,
          case .compose(_,let columns)=baseExpression.kind else {return nil}
    guard index >= 0,index < Int64(width.rawValue),index < columns.count else {throw wgslError(.invalidType,"Constant matrix column out of bounds",t)}
    return try abstractVectorLeaves(columns[Int(index)],at:t)
  }
  func abstractScalarLeaf(_ h: WExpr, at t: WGSLToken) throws -> IRLiteral? {
    guard nesting < min(budget.limits.depth, 24) else { throw ShaderFailure.budgetExceeded(.depth) }
    try enter()
    defer { nesting -= 1 }
    guard body.expressions.indices.contains(h.index) else { throw ShaderFailure.invalidHandle }
    switch body.expressions[h.index].kind {
    case .literal(let literal): return literal
    case .accessIndex(let base, let i):
      guard let xs = try abstractVectorLeaves(base, at: t), i < xs.count else { return nil }
      return xs[Int(i)]
    case .access(let base, let ix):
      guard body.expressions.indices.contains(ix.index) else { throw ShaderFailure.invalidHandle }
      let i: Int64
      switch body.expressions[ix.index].kind {
      case .literal(.u32(let x)): i = Int64(x)
      case .literal(.i32(let x)): i = Int64(x)
      default: return nil
      }
      guard let xs = try abstractVectorLeaves(base, at: t), i >= 0, i < xs.count else {
        throw wgslError(.invalidType, "Constant vector index out of bounds", t)
      }
      return xs[Int(i)]
    default: return nil
    }
  }
}
