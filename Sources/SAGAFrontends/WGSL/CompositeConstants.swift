import SAGAShaderIR

struct WGSLArrayConstant {
  let elements: [WGSLScalarConstant]
  let elementKind: WGSLConstKind
}

extension WGSLConstKind {
  var containsAbstract: Bool {
    switch self {
    case .integer(.abstract), .floating(.abstract): return true
    case .vector(_, let element), .matrix(_, _, let element), .array(_, let element):
      return element.containsAbstract
    default: return false
    }
  }

  var concretized: WGSLConstKind {
    switch self {
    case .integer(.abstract): return .integer(.sint)
    case .floating(.abstract): return .floating(.f32)
    case .vector(let n, let element): return .vector(n, element.concretized)
    case .matrix(let c, let r, let element): return .matrix(c, r, element.concretized)
    case .array(let n, let element): return .array(n, element.concretized)
    default: return self
    }
  }
}

extension WGSLScalarEvaluator {
  func arrayElementType(_ a: WGSLConstKind, _ b: WGSLConstKind, at t: WGSLToken) throws -> WGSLConstKind {
    try parser.budget.consume(.work)
    try limit()
    try parser.enter()
    defer { parser.nesting -= 1 }
    if a == b { return a }
    switch (a, b) {
    case (.array(let n, let x), .array(let m, let y)) where n == m:
      return try .array(n, arrayElementType(x, y, at: t))
    case (.vector(let n, let x), .vector(let m, let y)) where n == m:
      return try .vector(n, arrayElementType(x, y, at: t))
    case (.matrix(let c, let r, let x), .matrix(let d, let s, let y)) where c == d && r == s:
      return try .matrix(c, r, arrayElementType(x, y, at: t))
    default:
      if canAuto(a, to: b) { return b }
      if canAuto(b, to: a) { return a }
      throw wgslError(.typeMismatch, "Array elements must have a common type", t)
    }
  }

  // The expression tokens are a slice of the parser's tokens. Reuse the WGSL type
  // parser so nested templates, aliases and constant element counts follow one grammar.
  func arrayTemplate(at start: Int) throws -> (WType, Int) {
    let at = tokens[start]
    var low = 0, high = parser.tokens.count
    while low < high {
      try parser.budget.consume(.work)
      let middle = low + (high - low) / 2
      if parser.tokens[middle].start < at.start { low = middle + 1 } else { high = middle }
    }
    guard parser.tokens.indices.contains(low), parser.tokens[low].start == at.start else {
      throw ShaderFailure.invalidHandle
    }
    let saved = parser.cursor
    parser.cursor = low
    defer { parser.cursor = saved }
    let type = try parser.type()
    let end = start + (parser.cursor - low)
    guard tokens.indices.contains(end), tokens[end].text == "(" else {
      throw wgslError(.unexpectedToken, "Array template requires constructor arguments", at)
    }
    return (type, end)
  }

  mutating func compositeConstructor(_ t: WGSLToken, type: WType?) throws -> Int {
    guard index < tokens.count, try take().text == "(" else {
      throw wgslError(.unexpectedToken, "Expected composite constructor", t)
    }
    var args: [Int] = []
    while index < tokens.count, token.text != ")" {
      try parser.node()
      args.append(try parse())
      if index < tokens.count, token.text == "," { _ = try take() } else { break }
    }
    guard index < tokens.count, try take().text == ")" else {
      throw wgslError(.unexpectedToken, "Unterminated composite constructor", t)
    }
    let kind: WGSLConstKind
    if let type {
      kind = try parser.constantKind(type, at: t)
      if args.isEmpty {
        return try append(.value(parser.zeroConstant(kind, at: t)), kind, t)
      }
    } else {
      guard let first = args.first else {
        throw wgslError(.typeMismatch, "Inferred array requires at least one element", t)
      }
      var element = nodes[first].kind
      for argument in args.dropFirst() {
        element = try arrayElementType(element, nodes[argument].kind, at: t)
      }
      kind = .array(args.count, element)
    }
    switch kind {
    case .array(let count, let element) where args.count == count:
      for arg in args { try checkCompositeConversion(arg, to: element, at: t) }
      return try append(.array(element, args), kind, t)
    case .composite(let handle, let count) where args.count == count:
      guard case .structure(let members, _) = parser.types[handle.index].kind else {
        throw ShaderFailure.invalidHandle
      }
      var components: [Int] = []
      for (arg, member) in zip(args, members) {
        let target = try parser.constantKind(member.type, at: t)
        try checkCompositeConversion(arg, to: target, at: t)
        components.append(nodes[arg].kind == target ? arg : try append(.convert(arg), target, t))
      }
      return try append(.composite(handle, components), kind, t)
    default:
      throw wgslError(.typeMismatch, "Composite constructor arity or type mismatch", t)
    }
  }

  mutating func checkCompositeConversion(_ arg: Int, to target: WGSLConstKind, at t: WGSLToken) throws {
    let source = nodes[arg].kind
    try parser.budget.consume(.work)
    if source == target { return }
    guard try arrayElementType(source, target, at: t) == target else {
      throw wgslError(.typeMismatch, "Incompatible composite element type", t)
    }
    conversions.append((arg, target, t))
  }
}

extension WGSLParser {
  func constantKind(_ type: WType, at t: WGSLToken) throws -> WGSLConstKind {
    try enter()
    defer { nesting -= 1 }
    try budget.consume(.work)
    let kind = types[type.index].kind
    switch kind {
    case .array(let element, .constant(let count), _):
      return try .array(Int(count), constantKind(element, at: t))
    case .structure(let members, _):
      return .composite(type, members.count)
    default:
      return try WGSLScalarEvaluator(parser: self, tokens: [], fallback: t).numericConstKind(kind, at: t)
    }
  }

  func constantType(_ kind: WGSLConstKind, at t: WGSLToken) throws -> WType {
    try enter()
    defer { nesting -= 1 }
    try budget.consume(.work)
    switch kind {
    case .array(let count, let leaf):
      guard let count = UInt32(exactly: count), count > 0 else {
        throw wgslError(.invalidType, "Invalid array count", t)
      }
      let element = try constantType(leaf, at: t)
      let (alignment, size) = try layout(element)
      return try intern(.array(base: element, size: .constant(count), stride: aligned(size, to: alignment, at: t)))
    case .composite(let type, _): return type
    case .vector(let n, let leaf):
      guard let size = IRVectorSize(rawValue: UInt8(n)) else { throw ShaderFailure.invalidHandle }
      return try intern(.vector(size: size, scalar: leaf.irScalar(at: t)))
    case .matrix(let c, let r, let leaf):
      guard let columns = IRVectorSize(rawValue: UInt8(c)), let rows = IRVectorSize(rawValue: UInt8(r)) else {
        throw ShaderFailure.invalidHandle
      }
      return try intern(.matrix(columns: columns, rows: rows, scalar: leaf.irScalar(at: t)))
    default: return try intern(.scalar(kind.irScalar(at: t)))
    }
  }

  func zeroConstant(_ kind: WGSLConstKind, at t: WGSLToken) throws -> WGSLScalarConstant {
    try enter()
    defer { nesting -= 1 }
    try budget.consume(.work)
    switch kind {
    case .array(let count, let element):
      try budget.consume(.nodes, amount: count)
      let zero = try zeroConstant(element, at: t)
      return .array(.init(elements: Array(repeating: zero, count: count), elementKind: element))
    case .composite(let type, let count):
      guard case .structure(let members, _) = types[type.index].kind else { throw ShaderFailure.invalidHandle }
      try budget.consume(.nodes, amount: count)
      return .composite(type, try members.map { try zeroConstant(constantKind($0.type, at: t), at: t) })
    case .vector(let width, let leaf):
      let zero = try zeroConstant(leaf, at: t)
      return .vector(.init(elements: Array(repeating: zero, count: width), element: leaf))
    case .matrix(let columns, let rows, let leaf):
      let zero = try zeroConstant(leaf, at: t)
      let column = WGSLVectorConstant(elements: Array(repeating: zero, count: rows), element: leaf)
      return .matrix(.init(columns: Array(repeating: column, count: columns), rows: rows, element: leaf))
    default:
      return try WGSLScalarEvaluator(parser: self, tokens: [], fallback: t).zero(kind, at: t)
    }
  }

  func compositeRuntimeValue(_ value: WGSLScalarConstant, at t: WGSLToken) throws -> WGSLValue {
    try enter()
    defer { nesting -= 1 }
    let type = try constantType(value.kind, at: t)
    switch value {
    case .array(let array):
      return try compositeValue(array.elements, type: type, at: t)
    case .composite(_, let members):
      return try compositeValue(members, type: type, at: t)
    case .vector(let vector):
      return try vectorRuntimeValue(vector, to: type, at: t)
    case .matrix(let matrix):
      return try matrixRuntimeValue(matrix, to: type, at: t)
    default:
      return try expressionNode(.literal(value.literal), type: type, at: t, emit: false)
    }
  }

  private func compositeValue(_ values: [WGSLScalarConstant], type: WType, at t: WGSLToken) throws -> WGSLValue {
    try budget.consume(.work, amount: values.count)
    let components = try values.map { try compositeRuntimeValue($0, at: t).handle }
    return try expressionNode(.compose(type: type, components: components), type: type, at: t)
  }

  func inferredArray(_ t: WGSLToken) throws -> WGSLValue {
    let args = try arguments()
    guard let first = args.first else {
      throw wgslError(.typeMismatch, "Inferred array requires at least one element", t)
    }
    let evaluator = WGSLScalarEvaluator(parser: self, tokens: [], fallback: t)
    var element = try constantKind(first.type, at: t)
    for arg in args.dropFirst() {
      element = try evaluator.arrayElementType(element, constantKind(arg.type, at: t), at: t)
    }
    let type = try constantType(.array(args.count, element), at: t)
    return try construct(type, at: t, supplied: args)
  }
}
