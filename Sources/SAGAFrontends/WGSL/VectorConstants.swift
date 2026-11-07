import SAGAShaderIR

// Only scalar leaves; no recursive aggregate allocation or unbounded component counts.
struct WGSLVectorConstant {
  let elements: [WGSLScalarConstant]
  let element: WGSLConstKind
}
extension WGSLConstKind {
  var scalarLeaf: WGSLConstKind {
    if case .vector(_, let s) = self { return s }
    return self
  }
  var lanes: Int {
    if case .vector(let n, _) = self { return n }
    return 1
  }
  static func scalar(_ scalar: IRScalar, at t: WGSLToken) throws -> WGSLConstKind {
    switch scalar {
    case .bool: return .boolean
    case .i32: return .integer(.sint)
    case .u32: return .integer(.uint)
    case .f32: return .floating(.f32)
    default: throw wgslError(.unsupportedConstruct, "Vector element scalar not implemented", t)
    }
  }
  func irScalar(at t: WGSLToken) throws -> IRScalar {
    switch self {
    case .boolean: return .bool
    case .integer(let k): return k == .abstract ? .abstractInt : k == .sint ? .i32 : .u32
    case .floating(let k): return k == .abstract ? .abstractFloat : .f32
    case .vector, .matrix: throw wgslError(.invalidType, "Nested vector elements forbidden", t)
    }
  }
}
extension WGSLScalarEvaluator {
  func scalarConvert(
    _ v: WGSLScalarConstant, to target: WGSLConstKind, at t: WGSLToken, explicit: Bool
  ) throws -> WGSLScalarConstant {
    if !explicit {
      guard canAuto(v.kind, to: target) else {
        throw wgslError(.typeMismatch, "No implicit scalar conversion", t)
      }
      return try converted(v, to: target, at: t)
    }
    switch target {
    case .boolean:
      switch v {
      case .boolean: return v
      case .integer(let x): return .boolean(x.value != 0)
      case .floating(let x): return .boolean(x.value != 0)
      case .vector, .matrix: throw wgslError(.typeMismatch, "Scalar conversion needs scalar", t)
      }
    case .floating(let k): return try floatValue(v, kind: k, at: t, explicit: true)
    case .integer(let k):
      switch v {
      case .boolean(let x): return .integer(.init(value: x ? 1 : 0, kind: k))
      case .floating(let x): return try floatToInteger(x, target: k, at: t)
      case .integer(let x):
        if x.kind == .abstract || x.kind == k { return try converted(v, to: target, at: t) }
        let bits = UInt32(truncatingIfNeeded: x.value)
        return .integer(
          .init(value: k == .uint ? Int64(bits) : Int64(Int32(bitPattern: bits)), kind: k))
      case .vector, .matrix: throw wgslError(.typeMismatch, "Scalar conversion needs scalar", t)
      }
    case .vector, .matrix: throw wgslError(.invalidType, "Vector scalar conversion target", t)
    }
  }
  func canAuto(_ from: WGSLConstKind, to: WGSLConstKind) -> Bool {
    if from == to { return true }
    if from == .integer(.abstract) {
      switch to {
      case .integer, .floating: return true
      default: return false
      }
    }
    return from == .floating(.abstract) && to == .floating(.f32)
  }
  func zero(_ k: WGSLConstKind, at t: WGSLToken) throws -> WGSLScalarConstant {
    switch k {
    case .boolean: return .boolean(false)
    case .integer(let i): return .integer(.init(value: 0, kind: i))
    case .floating(let f): return .floating(.init(value: 0, kind: f))
    case .vector, .matrix: throw wgslError(.invalidType, "Nested zero vector", t)
    }
  }
  mutating func vectorConstructor(_ t: WGSLToken, size: Int, target: WGSLConstKind?) throws -> Int {
    guard (2...4).contains(size), index < tokens.count, try take().text == "(" else {
      throw wgslError(.unexpectedToken, "Expected vector constructor call", t)
    }
    var args: [Int] = []
    while index < tokens.count, token.text != ")" {
      guard args.count < 4 else {
        throw wgslError(.typeMismatch, "Too many vector arguments", token)
      }
      args.append(try parse())
      if index < tokens.count, token.text == "," { _ = try take() } else { break }
    }
    guard index < tokens.count, try take().text == ")" else {
      throw wgslError(.unexpectedToken, "Unterminated vector constructor", t)
    }
    let conversion = args.count == 1 && nodes[args[0]].kind.lanes > 1
    let splat = args.count == 1 && nodes[args[0]].kind.lanes == 1
    let count = args.reduce(0) { $0 + nodes[$1].kind.lanes }
    guard args.isEmpty || splat || count == size else {
      throw wgslError(.typeMismatch, "Vector component count mismatch", t)
    }
    if conversion {
      guard nodes[args[0]].kind.lanes == size else {
        throw wgslError(.typeMismatch, "Vector conversion width mismatch", t)
      }
    }
    var inferred: WGSLConstKind = .integer(.abstract)
    if let first = args.first {
      inferred = nodes[first].kind.scalarLeaf
      for a in args.dropFirst() {
        let element = nodes[a].kind.scalarLeaf
        if inferred != element { inferred = try common(inferred, element, at: t) }
      }
    }
    let element = target ?? inferred
    guard element.lanes == 1 && !element.isMatrix else {
      throw wgslError(.invalidType, "Vector element must be scalar", t)
    }
    for a in args {
      let source = nodes[a].kind.scalarLeaf
      guard conversion || canAuto(source, to: element) else {
        throw wgslError(
          .typeMismatch, "Vector components require automatic scalar conversion", nodes[a].token)
      }
      if !conversion && source != element {
        let shape: WGSLConstKind =
          nodes[a].kind.lanes == 1 ? element : .vector(nodes[a].kind.lanes, element)
        conversions.append((a, shape, t))
      }
    }
    return try append(.vector(size, element, args, conversion), .vector(size, element), t)
  }
  mutating func vectorValue(
    _ size: Int, _ element: WGSLConstKind, _ args: [Int], _ explicit: Bool, at t: WGSLToken
  ) throws -> WGSLScalarConstant {
    var values: [WGSLScalarConstant] = []
    for a in args {
      try parser.budget.consume(.work)
      let v = try value(a)
      if case .vector(let vector) = v {
        values.append(contentsOf: vector.elements)
      } else {
        values.append(v)
      }
      guard values.count <= 4 else { throw wgslError(.typeMismatch, "Vector count exceeds4", t) }
    }
    if args.isEmpty {
      values = Array(repeating: try zero(element, at: t), count: size)
    } else if args.count == 1 && values.count == 1 {
      values = Array(repeating: values[0], count: size)
    }
    guard values.count == size else { throw wgslError(.typeMismatch, "Vector component count", t) }
    var result: [WGSLScalarConstant] = []
    for v in values {
      try parser.budget.consume(.work)
      result.append(try scalarConvert(v, to: element, at: t, explicit: explicit))
    }
    return .vector(.init(elements: result, element: element))
  }
}
extension WGSLParser {
  func vectorRuntimeValue(_ v: WGSLVectorConstant, to target: WType?, at t: WGSLToken) throws
    -> WGSLValue
  {
    let count = v.elements.count
    guard (2...4).contains(count), let size = IRVectorSize(rawValue: UInt8(count)) else {
      throw wgslError(.invalidType, "Vector runtime width", t)
    }
    let element: WGSLConstKind
    if let target {
      guard case .vector(let n, let s) = types[target.index].kind, n == size else {
        throw wgslError(.typeMismatch, "Vector runtime context shape", t)
      }
      element = try .scalar(s, at: t)
    } else {
      element = v.element
    }
    let evaluator = WGSLScalarEvaluator(parser: self, tokens: [], fallback: t)
    var components: [WExpr] = []
    for x in v.elements {
      try budget.consume(.work)
      let converted = try evaluator.scalarConvert(x, to: element, at: t, explicit: false)
      components.append(
        try expressionNode(
          .literal(converted.literal), type: intern(.scalar(element.irScalar(at: t))), at: t,
          emit: false
        ).handle)
    }
    let type = try target ?? intern(.vector(size: size, scalar: element.irScalar(at: t)))
    return try expressionNode(.compose(type: type, components: components), type: type, at: t)
  }
}
