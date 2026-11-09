import SAGAShaderIR

// A flat, typed tree separates whole-expression validation from lazy value evaluation.
// The existing audited integer arithmetic is reused, not source rewriting or runtime IR.
enum WGSLScalarConstant {
  case integer(WGSLIntegerConstant)
  case boolean(Bool)
  case floating(WGSLFloatingConstant)
  case vector(WGSLVectorConstant)
  case matrix(WGSLMatrixConstant)
  case array(WGSLArrayConstant)
  case composite(IRHandle<IRType>, [WGSLScalarConstant])
  indirect case decomposition(WGSLDecompositionConstant)
  var kind: WGSLConstKind {
    switch self {
    case .integer(let x): .integer(x.kind)
    case .boolean: .boolean
    case .floating(let f): .floating(f.kind)
    case .vector(let v): .vector(v.elements.count, v.element)
    case .matrix(let m): .matrix(m.columns.count, m.rows, m.element)
    case .array(let array): .array(array.elements.count, array.elementKind)
    case .composite(let type, let components): .composite(type, components.count)
    case .decomposition(let parts): .decomposition(parts.function, parts.fraction.kind)
    }
  }
  var literal: IRLiteral {
    get throws {
      switch self {
      case .integer(let x): return x.literal
      case .boolean(let x): return .bool(x)
      case .floating(let x): return x.literal
      case .vector, .matrix, .array, .composite, .decomposition:
        throw ShaderFailure.unsupported("Aggregate cannot be encoded as scalar literal")
      }
    }
  }
  var scalar: IRScalar {
    get throws { try kind.irScalar(at: WGSLToken(kind: .end, text: "", start: 0, end: 0)) }
  }
}

indirect enum WGSLConstKind: Equatable {
  case integer(WGSLIntegerConstant.Kind)
  case floating(WGSLFloatKind)
  case boolean
  case vector(Int, WGSLConstKind)
  case matrix(Int, Int, WGSLConstKind)
  case array(Int, WGSLConstKind)
  case composite(IRHandle<IRType>, Int)
  case decomposition(IRMathFunction, WGSLConstKind)
}
struct WGSLConstNode {
  enum Op {
    case value(WGSLScalarConstant)
    case vector(Int, WGSLConstKind, [Int], Bool)
    case matrix(Int, Int, WGSLConstKind, [Int], Bool)
    case array(WGSLConstKind, [Int])
    case composite(IRHandle<IRType>, [Int])
    case convert(Int)
    case swizzle(Int, [Int])
    case index(Int, Int)
    case unary(String, Int)
    case binary(String, Int, Int)
    case call(String, [Int])
    case bitcast(Int, WGSLConstKind)
    case integerBit(IRMathFunction, [Int], [WGSLConstKind])
    case packing(IRPackingBuiltin, Int)
    case geometric(IRMathFunction, [Int], [WGSLConstKind])
    case numeric(IRMathFunction, [Int])
    case decomposition(IRMathFunction, Int, WGSLConstKind)
    case decompositionMember(Int, fraction: Bool)
    case ldexp(Int, Int, WGSLConstKind)
  }
  let op: Op
  let kind: WGSLConstKind
  let token: WGSLToken
}

struct WGSLScalarEvaluator {
  static let constantFunctions = [
    "bool", "i16", "u16", "i32", "u32", "i64", "u64", "f64", "f32", "f16", "all", "any", "select",
  ]
  let parser: WGSLParser
  let tokens: [WGSLToken]
  let fallback: WGSLToken
  var index = 0
  var nodes: [WGSLConstNode] = []
  var cache: [Int: WGSLScalarConstant] = [:]
  var conversions: [(Int, WGSLConstKind, WGSLToken)] = []
  var token: WGSLToken { index < tokens.count ? tokens[index] : fallback }
  func limit() throws {
    guard parser.nesting < min(24, parser.budget.limits.depth) else {
      throw ShaderFailure.budgetExceeded(.depth)
    }
  }
  mutating func take() throws -> WGSLToken {
    try parser.budget.consume(.work)
    guard index < tokens.count else {
      throw wgslError(.unexpectedToken, "Incomplete constant expression", fallback)
    }
    defer { index += 1 }
    return tokens[index]
  }
  mutating func append(_ op: WGSLConstNode.Op, _ kind: WGSLConstKind, _ t: WGSLToken) throws -> Int
  {
    try parser.node()
    nodes.append(.init(op: op, kind: kind, token: t))
    return nodes.count - 1
  }
  func operation(at i: Int) -> (String, Int) {
    let t = tokens[i]
    if t.kind == .templateStart || t.kind == .templateEnd { return ("", 1) }
    if ["<", ">"].contains(t.text), i + 1 < tokens.count,
      tokens[i + 1].kind == .symbol, tokens[i + 1].text == t.text, t.end == tokens[i + 1].start
    {
      return (t.text + t.text, 2)
    }
    return (t.text, 1)
  }
  // Validate WGSL's non-associating groups at each explicit parenthesis/call-argument level.
  func grammar() throws {
    var frames: [[(String, WGSLToken)]] = [[]]
    var expectOperand = true
    func validate(_ ops: [(String, WGSLToken)]) throws {
      let logical = ops.filter { $0.0 == "&&" || $0.0 == "||" }
      if Set(logical.map(\.0)).count > 1 {
        throw wgslError(
          .unexpectedToken, "Mixed logical groups require parentheses", logical.last!.1)
      }
      var segments = [[(String, WGSLToken)]]()
      var part: [(String, WGSLToken)] = []
      for op in ops {
        if op.0 == "&&" || op.0 == "||" {
          segments.append(part)
          part = []
        } else {
          part.append(op)
        }
      }
      segments.append(part)
      for segment in segments {
        let relational = segment.filter { ["==", "!=", "<", "<=", ">", ">="].contains($0.0) }
        if relational.count > 1 {
          throw wgslError(.unexpectedToken, "Relational operators do not chain", relational[1].1)
        }
        let bits = segment.filter { ["&", "|", "^"].contains($0.0) }
        if !bits.isEmpty
          && (Set(bits.map(\.0)).count > 1 || bits.count != segment.count || !logical.isEmpty)
        {
          throw wgslError(
            .unexpectedToken, "Bitwise groups require explicit parentheses", bits[0].1)
        }
        var numericParts = [[(String, WGSLToken)]]()
        var n: [(String, WGSLToken)] = []
        for op in segment {
          if ["==", "!=", "<", "<=", ">", ">="].contains(op.0) {
            numericParts.append(n)
            n = []
          } else {
            n.append(op)
          }
        }
        numericParts.append(n)
        for p in numericParts {
          let shifts = p.filter { $0.0 == "<<" || $0.0 == ">>" }
          if !shifts.isEmpty && p.count != 1 {
            throw wgslError(.unexpectedToken, "Shift group requires parentheses", shifts[0].1)
          }
        }
      }
    }
    var i = 0
    while i < tokens.count {
      try parser.budget.consume(.work)
      let t = tokens[i]
      if expectOperand, t.text == "array", i + 1 < tokens.count, tokens[i + 1].kind == .templateStart {
        (_, i) = try arrayTemplate(at: i)
        expectOperand = false
        continue
      }
      if expectOperand, t.text == "bitcast", i + 1 < tokens.count, tokens[i + 1].kind == .templateStart,
        parser.declarationNames[wgslKey(t.text)] == nil, !(try parser.localConstantNameExists(t))
      {
        var angles = 1
        i += 2
        while i < tokens.count, angles > 0 {
          try parser.budget.consume(.work)
          if tokens[i].kind == .templateStart { angles += 1 }
          if tokens[i].kind == .templateEnd { angles -= 1 }
          i += 1
        }
        guard angles == 0 else { throw wgslError(.unexpectedToken, "Unclosed bitcast type", t) }
        expectOperand = false
        continue
      }
      if ["vec2", "vec3", "vec4"].contains(t.text) || Self.matrixNames[t.text] != nil,
        i + 3 < tokens.count, tokens[i + 1].kind == .templateStart,
        tokens[i + 3].kind == .templateEnd
      {
        i += 4
        expectOperand = false
        continue
      }
      if t.text == "(" || t.text == "[" {
        guard frames.count < min(24, parser.budget.limits.depth) else {
          throw ShaderFailure.budgetExceeded(.depth)
        }
        frames.append([])
        expectOperand = true
        i += 1
        continue
      }
      if t.text == ")" || t.text == "]" {
        guard frames.count > 1 else {
          throw wgslError(.unexpectedToken, "Unmatched parenthesis", t)
        }
        try validate(frames.removeLast())
        expectOperand = false
        i += 1
        continue
      }
      if t.text == "," {
        try validate(frames[frames.count - 1])
        frames[frames.count - 1] = []
        expectOperand = true
        i += 1
        continue
      }
      if expectOperand {
        if !["-", "~", "!"].contains(t.text) { expectOperand = false }
        i += 1
        continue
      }
      let (op, len) = operation(at: i)
      if Self.precedence[op] != nil {
        frames[frames.count - 1].append((op, t))
        expectOperand = true
      }
      i += len
    }
    guard frames.count == 1 else {
      throw wgslError(.unexpectedToken, "Unclosed parenthesis", fallback)
    }
    try validate(frames[0])
  }
  static let precedence: [String: Int] = [
    "||": 1, "&&": 2, "|": 3, "^": 4, "&": 5, "==": 6, "!=": 6, "<": 6, "<=": 6, ">": 6, ">=": 6,
    "<<": 7, ">>": 7, "+": 8, "-": 8, "*": 9, "/": 9, "%": 9,
  ]
  mutating func parse(_ minimum: Int = 0) throws -> Int {
    try limit()
    try parser.enter()
    defer { parser.nesting -= 1 }
    var lhs = try primary()
    while index < tokens.count {
      let (op, len) = operation(at: index)
      guard let precedence = Self.precedence[op], precedence >= minimum else { break }
      let at = try take()
      if len == 2 { _ = try take() }
      let rhs = try parse(precedence + 1)
      let a = nodes[lhs].kind
      let b = nodes[rhs].kind
      let type: WGSLConstKind
      if a.isMatrix || b.isMatrix {
        let result = try matrixBinaryType(op, lhs, rhs, at: at)
        lhs = try append(.binary(op, lhs, rhs), result, at)
        continue
      }
      if a.lanes > 1 || b.lanes > 1 {
        let result = try vectorBinaryType(op, lhs, rhs, at: at)
        lhs = try append(.binary(op, lhs, rhs), result, at)
        continue
      }
      if ["&&", "||"].contains(op) {
        guard a == .boolean, b == .boolean else {
          throw wgslError(.typeMismatch, "Logical operands require bool", at)
        }
        type = .boolean
      } else if ["==", "!=", "<", "<=", ">", ">="].contains(op) {
        if case .vector(let count, let element) = a, case .vector(let other, let rhsElement) = b {
          guard count == other else {
            throw wgslError(.typeMismatch, "Vector comparison width", at)
          }
          if element == .boolean || rhsElement == .boolean {
            guard element == rhsElement && ["==", "!="].contains(op) else {
              throw wgslError(.typeMismatch, "Vector bool comparison", at)
            }
          } else {
            _ = try common(element, rhsElement, at: at)
          }
          let merged = element == rhsElement ? element : try common(element, rhsElement, at: at)
          if element != merged { conversions.append((lhs, .vector(count, merged), at)) }
          if rhsElement != merged { conversions.append((rhs, .vector(count, merged), at)) }
          lhs = try append(.binary(op, lhs, rhs), .vector(count, .boolean), at)
          continue
        }
        if a == .boolean || b == .boolean {
          guard a == b, ["==", "!="].contains(op) else {
            throw wgslError(.typeMismatch, "Invalid boolean comparison", at)
          }
        } else {
          _ = try common(a, b, at: at)
        }
        type = .boolean
      } else if ["&", "|"].contains(op), a == .boolean, b == .boolean {
        type = .boolean
      } else if op == "<<" || op == ">>" {
        guard case .integer = a, case .integer(let k) = b, k == .abstract || k == .uint else {
          throw wgslError(.typeMismatch, "Invalid shift operand types", at)
        }
        type = a
      } else {
        type = try common(a, b, at: at)
      }
      if case .integer(let x) = a, case .integer(let y) = b {
        if op == "<<" || op == ">>" {
          if y == .abstract { conversions.append((rhs, .integer(.uint), at)) }
        } else if x != y {
          if x == .abstract {
            conversions.append((lhs, .integer(y), at))
          } else if y == .abstract {
            conversions.append((rhs, .integer(x), at))
          }
        }
      }
      let numericCommon: WGSLConstKind?
      if a != .boolean && b != .boolean && !["<<", ">>"].contains(op) {
        numericCommon = try common(a, b, at: at)
      } else {
        numericCommon = nil
      }
      if let numericCommon, case .floating = numericCommon {
        guard ["+", "-", "*", "/", "%", "==", "!=", "<", "<=", ">", ">="].contains(op) else {
          throw wgslError(.typeMismatch, "Float operator has no overload", at)
        }
        if a != numericCommon { conversions.append((lhs, numericCommon, at)) }
        if b != numericCommon { conversions.append((rhs, numericCommon, at)) }
      }
      lhs = try append(.binary(op, lhs, rhs), type, at)
    }
    return lhs
  }
  func common(_ a: WGSLConstKind, _ b: WGSLConstKind, at t: WGSLToken) throws -> WGSLConstKind {
    if case .floating(let x) = a {
      if b == .integer(.abstract) { return a }
      if case .floating(let y) = b {
        guard x == y || x == .abstract || y == .abstract else {
          throw wgslError(.typeMismatch, "Mixed concrete float types", t)
        }
        return .floating(x == .abstract ? y : x)
      }
    }
    if case .floating = b, a == .integer(.abstract) { return b }
    guard case .integer(let x) = a, case .integer(let y) = b else {
      throw wgslError(.typeMismatch, "Operator requires integers", t)
    }
    guard x == y || x == .abstract || y == .abstract else {
      throw wgslError(.typeMismatch, "Mixed concrete integer types", t)
    }
    return .integer(x == .abstract ? y : x)
  }
  mutating func prefix() throws -> Int {
    try limit()
    try parser.enter()
    defer { parser.nesting -= 1 }
    let t = try take()
    if t.text == "(" {
      let n = try parse()
      guard index < tokens.count, try take().text == ")" else {
        throw wgslError(.unexpectedToken, "Expected closing parenthesis", t)
      }
      return n
    }
    if ["!", "~", "-"].contains(t.text) {
      if t.text == "-", token.kind == .number, token.text.hasSuffix("li") {
        let number = try take()
        guard case .i64(let value) = try parser.literal(number, negativeInt64: true).0 else {
          throw ShaderFailure.invalidHandle
        }
        return try append(.value(.integer(.init(value: value, kind: .sint64))), .integer(.sint64), t)
      }
      let n = try primary()
      let kind = nodes[n].kind
      if t.text == "!" {
        guard kind.scalarLeaf == .boolean else {
          throw wgslError(.typeMismatch, "Logical not requires bool", t)
        }
      } else {
        if case .floating = kind.scalarLeaf {
          guard t.text == "-" else {
            throw wgslError(.typeMismatch, "Float complement is invalid", t)
          }
        } else {
          guard case .integer(let k) = kind.scalarLeaf, t.text != "-" || !k.isUnsigned else {
            throw wgslError(.typeMismatch, "Invalid integer unary operand", t)
          }
        }
      }
      return try append(.unary(t.text, n), kind, t)
    }
    if t.text == "true" || t.text == "false" {
      return try append(.value(.boolean(t.text == "true")), .boolean, t)
    }
    if t.kind == .number {
      let (l, _) = try parser.literal(t)
      let v: WGSLIntegerConstant
      switch l {
      case .abstractInt(let x): v = .init(value: x, kind: .abstract)
      case .i32(let x): v = .init(value: Int64(x), kind: .sint)
      case .u32(let x): v = .init(value: Int64(x), kind: .uint)
      case .i64(let x): v = .init(value: x, kind: .sint64)
      case .u64(let x): v = .init(value: Int64(bitPattern: x), kind: .uint64)
      case .abstractFloat(let x):
        return try append(
          .value(.floating(.init(value: x, kind: .abstract))), .floating(.abstract), t)
      case .f32(let x):
        return try append(
          .value(.floating(.init(value: Double(x), kind: .f32))), .floating(.f32), t)
      case .f64(let x):
        return try append(.value(.floating(.init(value: x, kind: .f64))), .floating(.f64), t)
      case .f16Bits(let bits):
        return try append(.value(.floating(.init(value: Double(Binary16.decode(bits)), kind: .f16))), .floating(.f16), t)
      default: throw wgslError(.unsupportedConstruct, "Numeric constant kind unsupported", t)
      }
      return try append(.value(.integer(v)), .integer(v.kind), t)
    }
    guard t.kind == .word else {
      throw wgslError(.unexpectedToken, "Expected scalar constant expression", t)
    }
    if t.text == "array", index < tokens.count, !(try parser.localConstantNameExists(t)) {
      if token.kind == .templateStart {
        let (type, end) = try arrayTemplate(at: index - 1)
        index = end
        return try compositeConstructor(t, type: type)
      }
      return try compositeConstructor(t, type: nil)
    }
    if t.text == "bitcast", index < tokens.count, token.kind == .templateStart,
      parser.declarationNames[wgslKey(t.text)] == nil, !(try parser.localConstantNameExists(t))
    {
      return try constantBitcast(at: t)
    }
    if let (columns, rows) = Self.matrixNames[t.text],
      parser.declarationNames[wgslKey(t.text)] == nil
    {
      guard !(try parser.localConstantNameExists(t)) else {
        throw wgslError(.typeMismatch, "Shadowed matrix type", t)
      }
      var element: WGSLConstKind?
      if index < tokens.count, token.kind == .templateStart {
        _ = try take()
        let name = try take()
        try parser.resolveTypeDeclarationIfIndexed(name)
        if let handle = parser.typeNames[wgslKey(name.text)],
           case .scalar(let scalar) = parser.types[handle.index].kind, scalar == .f16 || scalar == .f32 {
          element = try .scalar(scalar, at: name)
        } else if parser.declarationNames[wgslKey(name.text)] == nil, name.text == "f32" || name.text == "f16" {
          if name.text == "f16" { try parser.requireFloat16(at: name) }
          element = .floating(name.text == "f16" ? .f16 : .f32)
        } else {
          throw wgslError(.typeMismatch, "Matrix scalar must be f32 or f16", name)
        }
        guard index < tokens.count, try take().kind == .templateEnd else {
          throw wgslError(.unexpectedToken, "Unclosed matrix type", t)
        }
      }
      return try matrixConstructor(t, columns: columns, rows: rows, target: element)
    }
    if ["vec2", "vec3", "vec4"].contains(t.text), parser.declarationNames[wgslKey(t.text)] == nil {
      guard !(try parser.localConstantNameExists(t)) else {
        throw wgslError(.typeMismatch, "Shadowed vector type", t)
      }
      let size = t.text == "vec2" ? 2 : t.text == "vec3" ? 3 : 4
      var element: WGSLConstKind?
      if index < tokens.count, token.kind == .templateStart {
        _ = try take()
        let name = try take()
        try parser.resolveTypeDeclarationIfIndexed(name)
        let scalar: IRScalar
        if let alias = parser.typeNames[wgslKey(name.text)],
          case .scalar(let s) = parser.types[alias.index].kind
        {
          scalar = s
        } else if name.text == "bool" {
          scalar = .bool
        } else if name.text == "i16" || name.text == "u16" {
          try parser.requireInt16(at: name)
          scalar = name.text == "i16" ? .i16 : .u16
        } else if name.text == "i32" {
          scalar = .i32
        } else if name.text == "u32" {
          scalar = .u32
        } else if name.text == "i64" {
          scalar = .i64
        } else if name.text == "u64" {
          scalar = .u64
        } else if name.text == "f64" {
          try parser.requireFloat64(at: name)
          scalar = .f64
        } else if name.text == "f32" {
          scalar = .f32
        } else if name.text == "f16" {
          try parser.requireFloat16(at: name)
          scalar = .f16
        } else {
          throw wgslError(.unsupportedConstruct, "Vector scalar template type unsupported", name)
        }
        guard index < tokens.count, try take().kind == .templateEnd else {
          throw wgslError(.unexpectedToken, "Vector template closing bracket", t)
        }
        element = try .scalar(scalar, at: name)
      }
      return try vectorConstructor(t, size: size, target: element)
    }
    if index < tokens.count, token.text == "(" { return try constantCall(t) }
    let v = try parser.scalarConstantName(t)
    return try append(.value(v), v.kind, t)
  }
  private mutating func constantCall(_ t: WGSLToken) throws -> Int {
    guard !(try parser.localConstantNameExists(t)) else {
      throw wgslError(.typeMismatch, "Local value shadows constructor", t)
    }
    if let kind = parser.predeclaredType(t.text) {
      if kind.usesFloat16 { try parser.requireFloat16(at: t) }
      switch kind {
      case .vector(let n,let s):
        return try vectorConstructor(t,size:Int(n.rawValue),target:.scalar(s,at:t))
      case .matrix(let c,let r,let s):
        return try matrixConstructor(t,columns:Int(c.rawValue),rows:Int(r.rawValue),target:.scalar(s,at:t))
      default:throw ShaderFailure.invalidHandle
      }
    }
    var callName = t.text
    if let declaration = parser.declarationNames[wgslKey(t.text)] {
      guard ["alias", "struct"].contains(parser.declarations[declaration].kind) else {
        throw wgslError(.typeMismatch, "Module value is not a constructor", t)
      }
      try parser.resolveTypeDeclaration(t)
      if let handle = parser.typeNames[wgslKey(t.text)] {
        switch parser.types[handle.index].kind {
        case .array(_, .constant, _), .structure:
          return try compositeConstructor(t, type: handle)
        default: break
        }
      }
      if let handle = parser.typeNames[wgslKey(t.text)],
        case .matrix(let c, let r, let s) = parser.types[handle.index].kind
      {
        return try matrixConstructor(
          t, columns: Int(c.rawValue), rows: Int(r.rawValue), target: .scalar(s, at: t))
      }
      if let handle = parser.typeNames[wgslKey(t.text)],
        case .vector(let n, let s) = parser.types[handle.index].kind
      {
        return try vectorConstructor(t, size: Int(n.rawValue), target: .scalar(s, at: t))
      }
      guard let handle = parser.typeNames[wgslKey(t.text)],
        case .scalar(let scalar) = parser.types[handle.index].kind
      else {
        throw wgslError(.unsupportedConstruct, "Aggregate alias const constructor unsupported", t)
      }
      if scalar == .bool {
        callName = "bool"
      } else if scalar == .i16 {
        callName = "i16"
      } else if scalar == .u16 {
        callName = "u16"
      } else if scalar == .i32 {
        callName = "i32"
      } else if scalar == .u32 {
        callName = "u32"
      } else if scalar == .f64 {
        callName = "f64"
      } else if scalar == .f32 {
        callName = "f32"
      } else if scalar == .f16 {
        callName = "f16"
      } else {
        throw wgslError(.unsupportedConstruct, "Scalar alias kind unsupported", t)
      }
    }
    if callName == "f16" { try parser.requireFloat16(at: t) }
    if callName == "f64" { try parser.requireFloat64(at: t) }
    if callName == "i16" || callName == "u16" { try parser.requireInt16(at: t) }
    let function = IRMathFunction(wgslName: callName)
    let bitArity = function?.integerBitArity
    let packing = function?.packingBuiltin
    let geometryArity = function?.geometricArity
    let numeric = function?.numericBuiltin
    guard Self.constantFunctions.contains(callName) || bitArity != nil || packing != nil || geometryArity != nil ||
      numeric != nil || function?.decomposesFloat == true || function == .ldexp
    else { throw wgslError(.unsupportedConstruct, "Const function not implemented", t) }
    _ = try take()
    var args: [Int] = []
    let maxArguments = packing != nil || function?.decomposesFloat == true ? 1
      : bitArity ?? geometryArity ?? numeric?.arity ?? (function == .ldexp ? 2 : 3)
    while index < tokens.count, token.text != ")" {
      guard args.count < maxArguments else {
        throw wgslError(.typeMismatch, "Too many const arguments", t)
      }
      args.append(try parse())
      if index < tokens.count, token.text == "," { _ = try take() } else { break }
    }
    guard index < tokens.count, try take().text == ")" else {
      throw wgslError(.unexpectedToken, "Unterminated const call", t)
    }
    if let function, let bitArity {
      let signature = try integerBitArgumentKinds(args.map { nodes[$0].kind }, arity: bitArity, at: t)
      conversions += zip(args, signature).compactMap { node, kind in nodes[node].kind == kind ? nil : (node, kind, t) }
      return try append(.integerBit(function, args, signature), signature[0], t)
    }
    if let packing {
      guard let argument = args.first else { throw wgslError(.typeMismatch, "Packing builtin requires one operand", t) }
      let input = try numericConstKind(packing.argumentType, at: t)
      let actual = nodes[argument].kind
      guard actual.lanes == input.lanes, canAuto(actual.scalarLeaf, to: input.scalarLeaf) else {
        throw wgslError(.typeMismatch, "Packing builtin argument type mismatch", t)
      }
      if input != actual { conversions.append((argument, input, t)) }
      return try append(.packing(packing, argument), numericConstKind(packing.resultType, at: t), t)
    }
    if let function, geometryArity != nil { return try geometricConstCall(function, arguments: args, at: t) }
    if let function, function.decomposesFloat { return try decompositionCall(function, arguments: args, at: t) }
    if function == .ldexp { return try ldexpConstCall(args, at: t) }
    if let function, let numeric {
      let signature = try numericSignature(numeric, arguments: args.map { nodes[$0].kind }, materialize: false, at: t)
      for (argument, kind) in zip(args, signature.arguments) where nodes[argument].kind != kind {
        conversions.append((argument, kind, t))
      }
      return try append(.numeric(function, args), signature.result, t)
    }
    let needed = callName == "select" ? 3 : 1
    if args.isEmpty && ["bool", "i16", "u16", "i32", "u32", "i64", "u64", "f64", "f32", "f16"].contains(callName) {
      let zero: WGSLScalarConstant
      switch callName {
      case "bool": zero = .boolean(false)
      case "f64": zero = .floating(.init(value: 0, kind: .f64))
      case "f32": zero = .floating(.init(value: 0, kind: .f32))
      case "f16": zero = .floating(.init(value: 0, kind: .f16))
      case "i16": zero = .integer(.init(value: 0, kind: .sint16))
      case "u16": zero = .integer(.init(value: 0, kind: .uint16))
      case "i32": zero = .integer(.init(value: 0, kind: .sint))
      case "i64": zero = .integer(.init(value: 0, kind: .sint64))
      case "u64": zero = .integer(.init(value: 0, kind: .uint64))
      default: zero = .integer(.init(value: 0, kind: .uint))
      }
      return try append(.value(zero), zero.kind, t)
    }
    guard args.count == needed else {
      throw wgslError(.typeMismatch, "Constant function arity", t)
    }
    if callName == "all" || callName == "any" {
      guard nodes[args[0]].kind.scalarLeaf == .boolean else {
        throw wgslError(.typeMismatch, "Scalar all/any require bool", t)
      }
      return try append(.call(callName, args), .boolean, t)
    }
    if callName == "select" {
      let result = try selectType(args, at: t)
      return try append(.call(callName, args), result, t)
    }
    let kind: WGSLConstKind
    switch callName {
    case "f64": kind = .floating(.f64)
    case "f32": kind = .floating(.f32)
    case "f16": kind = .floating(.f16)
    case "bool": kind = .boolean
    case "i16": kind = .integer(.sint16)
    case "u16": kind = .integer(.uint16)
    case "i32": kind = .integer(.sint)
    case "u32": kind = .integer(.uint)
    case "i64": kind = .integer(.sint64)
    case "u64": kind = .integer(.uint64)
    default: throw ShaderFailure.invalidHandle
    }
    switch nodes[args[0]].kind {
    case .integer, .floating, .boolean: return try append(.call(callName, args), kind, t)
    case .vector, .matrix, .array, .composite, .decomposition:
      throw wgslError(.typeMismatch, "Scalar constructor requires one scalar", t)
    }
  }
  mutating func evaluate() throws -> WGSLScalarConstant {
    guard !tokens.isEmpty else {
      throw wgslError(.unexpectedToken, "Missing constant expression", fallback)
    }
    try grammar()
    let root = try parse()
    guard index == tokens.count else {
      throw wgslError(.unexpectedToken, "Unexpected constant expression token", token)
    }
    return try evaluateTyped(root)
  }

  mutating func evaluateTyped(_ root: Int) throws -> WGSLScalarConstant {
    for (node, target, at) in conversions {
      try parser.budget.consume(.work)
      _ = try converted(value(node), to: target, at: at)
    }
    // WGSL #const-expr explicitly requires shader-creation checks such as integer division
    // even in a skipped RHS. Names/types/syntax above always cover the complete tree.
    for i in nodes.indices {
      try parser.budget.consume(.work)
      let n = nodes[i]
      switch n.op {
      case .integerBit(_, let arguments, _) where arguments.count > 1:
        guard case .integer(let offset) = try value(arguments[arguments.count - 2]),
              case .integer(let count) = try value(arguments[arguments.count - 1]) else {
          throw ShaderFailure.invalidHandle
        }
        let scalar = try n.kind.scalarLeaf.irScalar(at: n.token)
        try validateIntegerBitRange(offset.value, count.value, width: Int64(scalar.width) * 8, at: n.token)
      case .index(let base, let ix): _ = try checkedIndex(base, ix, at: n.token)
      case .binary(let op, let lhs, let rhs)
      where n.kind.lanes > 1 && ["/", "%", "<<", ">>"].contains(op):
        try vectorCreationChecks(n, op: op, lhs: lhs, rhs: rhs)
      case .binary(let op, let lhs, let rhs)
      where (op == "/" || op == "%")
        && {
          if case .integer = n.kind { return true }
          return false
        }():
        guard case .integer(let denominator) = try value(rhs) else {
          throw ShaderFailure.invalidHandle
        }
        guard denominator.value != 0 else {
          throw wgslError(.invalidType, "Constant division/remainder by zero", n.token)
        }
        // Only the -1 divisor needs the signed numerator for the mandatory overflow check.
        if denominator.value == -1, case .integer(let kind) = nodes[lhs].kind, !kind.isUnsigned {
          guard case .integer(let numerator) = try value(lhs) else {
            throw ShaderFailure.invalidHandle
          }
          let minimum = kind.signedMinimum
          guard numerator.value != minimum else {
            throw wgslError(.invalidType, "Constant division/remainder overflow", n.token)
          }
        }
      case .binary(let op, let lhs, let rhs) where op == "<<" || op == ">>":
        guard case .integer(let count) = try value(rhs), case .integer(let kind) = nodes[lhs].kind
        else { throw ShaderFailure.invalidHandle }
        let width = kind.bitWidth
        guard count.value >= 0 && count.value < width else {
          throw wgslError(.invalidType, "Const shift count exceeds supported bit width", n.token)
        }
        if op == "<<" { _ = try value(i) }  // both constant operands determine left-shift overflow
      case .ldexp(_, let exponent, let kind):
        try validateLdexpExponent(converted(value(exponent), to: kind, at: n.token), mantissa: n.kind, at: n.token)
      case .numeric(.quantizeToF16, _):
        _ = try value(i)
      case .numeric(let function, let args) where function == .clamp || function == .smoothStep:
        let lowIndex = function == .clamp ? 1 : 0
        let low = try converted(value(args[lowIndex]), to: n.kind, at: n.token)
        let high = try converted(value(args[lowIndex + 1]), to: n.kind, at: n.token)
        try validateNumericBounds(function, low, high, at: n.token)
      default: break
      }
    }
    return try value(root)
  }
  mutating func value(_ i: Int) throws -> WGSLScalarConstant {
    try parser.budget.consume(.work)
    if let v = cache[i] { return v }
    try limit()
    try parser.enter()
    defer { parser.nesting -= 1 }
    let n = nodes[i]
    let arithmetic = WGSLIntegerEvaluator(parser: parser, tokens: [], fallback: n.token)
    let result: WGSLScalarConstant
    switch n.op {
    case .value(let v): result = v
    case .bitcast(let operand, let input):
      let value = try converted(value(operand), to: input, at: n.token)
      result = try bitcastConstant(value, to: n.kind, at: n.token)
    case .integerBit(let function, let arguments, let signature):
      let operands = try zip(arguments, signature).map { argument, kind in
        try converted(value(argument), to: kind, at: n.token)
      }
      result = try constantIntegerBitBuiltin(function, arguments: operands, at: n.token)
    case .packing(let builtin, let operand):
      let argument = try converted(value(operand), to: numericConstKind(builtin.argumentType, at: n.token), at: n.token)
      result = try packingConstant(builtin, argument: argument, at: n.token)
    case .geometric(let function, let arguments, let signature):
      let operands = try zip(arguments, signature).map { argument, kind in
        try converted(value(argument), to: kind, at: n.token)
      }
      result = try geometricConstant(function, arguments: operands, result: n.kind, at: n.token)
    case .numeric(let function, let arguments):
      result = try numericConstValue(function, arguments: arguments, kind: n.kind, at: n.token)
    case .decomposition(let function, let argument, let input):
      let value = try decompositionValue(function, argument: argument, input: input, at: n.token)
      if case .composite(let type, _) = n.kind, case .decomposition(let parts) = value {
        result = .composite(type, [parts.fraction, parts.second])
      } else {
        result = value
      }
    case .decompositionMember(let base, let fraction):
      result = try decompositionMember(base, fraction: fraction)
    case .ldexp(let mantissa, let exponent, let kind):
      result = try ldexpConstValue(mantissa, exponent, exponentKind: kind, result: n.kind, at: n.token)
    case .array(let element, let components):
      let elements = try components.map { try converted(value($0), to: element, at: n.token) }
      result = .array(.init(elements: elements, elementKind: element))
    case .composite(let type, let components):
      result = .composite(type, try components.map { try value($0) })
    case .convert(let operand):
      result = try converted(value(operand), to: n.kind, at: n.token)
    case .matrix(let c, let r, let k, let xs, let explicit):
      result = try matrixValue(c, r, k, xs, conversion: explicit, at: n.token)
    case .swizzle(let base, let indices):
      guard case .vector(let vector) = try value(base) else { throw ShaderFailure.invalidHandle }
      var values: [WGSLScalarConstant] = []
      for j in indices {
        try parser.budget.consume(.work)
        guard vector.elements.indices.contains(j) else { throw ShaderFailure.invalidHandle }
        values.append(vector.elements[j])
      }
      result =
        values.count == 1 ? values[0] : .vector(.init(elements: values, element: vector.element))
    case .index(let base, let ix):
      let j = try checkedIndex(base, ix, at: n.token)
      switch try value(base) {
      case .vector(let vector): result = vector.elements[j]
      case .matrix(let matrix): result = .vector(matrix.columns[j])
      case .array(let array): result = array.elements[j]
      case .composite(_, let components): result = components[j]
      default: throw ShaderFailure.invalidHandle
      }
    case .vector(let n, let k, let args, let explicit):
      result = try vectorValue(n, k, args, explicit, at: nodes[i].token)
    case .unary(let op, let x):
      let v = try value(x)
      if case .vector(let vector) = v {
        var values: [WGSLScalarConstant] = []
        for component in vector.elements {
          try parser.budget.consume(.work)
          let leaf = try append(.value(component), component.kind, n.token)
          let unary = try append(.unary(op, leaf), component.kind, n.token)
          values.append(try value(unary))
        }
        result = .vector(.init(elements: values, element: vector.element))
      } else if case .boolean(let b) = v {
        result = .boolean(!b)
      } else if case .integer(let a) = v {
        if a.kind != .abstract {
          result = .integer(.init(value: a.kind.truncating(op == "~" ? ~a.value : 0 &- a.value), kind: a.kind))
        } else if op == "~" {
          result = .integer(try arithmetic.checked(~a.value, a.kind, at: n.token))
        } else {
          let (v, overflow) = Int64.zero.subtractingReportingOverflow(a.value)
          guard !overflow else {
            throw wgslError(.invalidType, "Abstract negation overflow", n.token)
          }
          result = .integer(try arithmetic.checked(v, a.kind, at: n.token))
        }
      } else if case .floating(let f) = v {
        result = try floating(-f.value, kind: f.kind, at: n.token)
      } else {
        throw ShaderFailure.invalidHandle
      }
    case .binary(let op, let x, let y):
      let a = try value(x)
      if case .boolean(let b) = a, (op == "&&" && !b) || (op == "||" && b) {
        result = a
      } else {
        result = try binaryValue(op, a, value(y), node: n)
      }
    case .call(let name, let args):
      result = try callValue(name, arguments: args, node: n)
    }
    cache[i] = result
    return result
  }
  private func binaryValue(
    _ op: String, _ a: WGSLScalarConstant, _ b: WGSLScalarConstant, node n: WGSLConstNode
  ) throws -> WGSLScalarConstant {
    let arithmetic = WGSLIntegerEvaluator(parser: parser, tokens: [], fallback: n.token)
    let result: WGSLScalarConstant
    if a.kind.isMatrix || b.kind.isMatrix {
      result = try matrixBinaryValue(op, a, b, result: n.kind, at: n.token)
    } else if case .vector = n.kind {
      result = try vectorBinaryValue(op, a, b, result: n.kind, at: n.token)
    } else if case .boolean(let l) = a, case .boolean(let r) = b {
      switch op {
      case "&&", "&": result = .boolean(l && r)
      case "||", "|": result = .boolean(l || r)
      case "==": result = .boolean(l == r)
      case "!=": result = .boolean(l != r)
      default: throw wgslError(.typeMismatch, "Invalid boolean operation", n.token)
      }
    } else if a.isFloat || b.isFloat {
      result = try floatingBinary(op, a, b, at: n.token)
    } else if case .integer(var l) = a, case .integer(var r) = b {
      if ["==", "!=", "<", "<=", ">", ">="].contains(op) {
        let kind = try common(a.kind, b.kind, at: n.token)
        guard case .integer(let k) = kind else { throw ShaderFailure.invalidHandle }
        l = try arithmetic.materialized(l, as: k, at: n.token)
        r = try arithmetic.materialized(r, as: k, at: n.token)
        result = .boolean(arithmetic.compare(op, l, r))
      } else {
        result = .integer(try arithmetic.binary(op, l, r, at: n.token))
      }
    } else {
      throw wgslError(.typeMismatch, "Mismatched constant operand values", n.token)
    }
    return result
  }

  private mutating func callValue(_ name: String, arguments args: [Int], node n: WGSLConstNode) throws -> WGSLScalarConstant {
    let result: WGSLScalarConstant
    if name == "all" || name == "any" {
      let operand = try value(args[0])
      if case .vector(let vector) = operand {
        var reduction = name == "all"
        for component in vector.elements {
          try parser.budget.consume(.work)
          guard case .boolean(let b) = component else { throw ShaderFailure.invalidHandle }
          reduction = name == "all" ? reduction && b : reduction || b
        }
        result = .boolean(reduction)
      } else {
        result = operand
      }
      return result
    }
    if name == "select" {
      let a = try value(args[0])
      let b = try value(args[1])
      let c = try value(args[2])
      result = try selectedValue(a, b, c, result: n.kind, at: n.token)
      return result
    }
    if name == "bool" {
      let v = try value(args[0])
      switch v {
      case .boolean: result = v
      case .integer(let x): result = .boolean(x.value != 0)
      case .floating(let x): result = .boolean(x.value != 0)
      case .vector, .matrix, .array, .composite, .decomposition:
        throw wgslError(.typeMismatch, "Scalar bool constructor requires scalar", n.token)
      }
      return result
    }
    if case .floating(let kind) = n.kind {
      result = try floatValue(value(args[0]), kind: kind, at: n.token, explicit: true)
      return result
    }
    guard case .integer = n.kind else {
      throw ShaderFailure.invalidHandle
    }
    return try scalarConvert(value(args[0]), to: n.kind, at: n.token, explicit: true)
  }

}

extension WGSLParser {
  func localConstantNameExists(_ t: WGSLToken) throws -> Bool {
    for scope in body.scopes.reversed() {
      try budget.consume(.work)
      if scope[wgslKey(t.text)] != nil { return true }
    }
    return false
  }
  func scalarConstantName(_ t: WGSLToken) throws -> WGSLScalarConstant {
    let key = wgslKey(t.text)
    try budget.consume(.work, amount: t.text.utf8.count)
    for i in body.scopes.indices.reversed() {
      try budget.consume(.work)
      if body.scopes[i][key] != nil {
        try requireVisibleToContinuing(key, scope: i, at: t)
        guard body.constScopes.indices.contains(i), let v = body.constScopes[i][key] else {
          throw wgslError(.invalidType, "Runtime/let value cannot occur in const-expression", t)
        }
        return v
      }
    }
    try resolveConstantDeclaration(t)
    if let composite = compositeConstants[key] { return composite }
    if let parts = decompositionConstants[key] { return .decomposition(parts) }
    if let m = matrixConstants[key] { return .matrix(m) }
    if let v = vectorConstants[key] { return .vector(v) }
    if let b = booleanConstants[key] { return .boolean(b) }
    if let i = integerConstants[key] { return .integer(i) }
    if let f = floatConstants[key] { return .floating(f) }
    if let f = abstractFloatConstants[key] { return .floating(.init(value: f, kind: .abstract)) }
    if constantNames[key] != nil {
      throw wgslError(.unsupportedConstruct, "Float/aggregate constant expression unsupported", t)
    }
    throw wgslError(.unresolvedName, "Unknown constant name '\(t.text)'", t)
  }
  func scalarConstant(_ tokens: [WGSLToken], at t: WGSLToken) throws -> WGSLScalarConstant {
    var e = WGSLScalarEvaluator(parser: self, tokens: tokens, fallback: t)
    return try e.evaluate()
  }
  func knownConstantValue(_ range: Range<Int>, at name: WGSLToken) throws -> WGSLScalarConstant? {
    for index in range {
      try budget.consume(.work)
      let token = tokens[index]
      guard token.kind == .word, index == range.lowerBound || tokens[index - 1].text != "." else { continue }
      let key = wgslKey(token.text)
      var isLocalConst = false
      for scope in body.scopes.indices.reversed() {
        try budget.consume(.work)
        if body.scopes[scope][key] != nil {
          guard body.constScopes[scope][key] != nil else { return nil }
          isLocalConst = true
          break
        }
      }
      if isLocalConst { continue }
      if let declaration = declarationNames[key] {
        guard ["alias", "const", "struct"].contains(declarations[declaration].kind) else { return nil }
        continue
      }
      let math = IRMathFunction(wgslName: token.text)
      guard WGSLScalarEvaluator.constantFunctions.contains(token.text) || predeclaredType(token.text) != nil ||
            math?.integerBitArity != nil || math?.packingBuiltin != nil || math?.geometricArity != nil ||
            math?.numericBuiltin != nil || math?.decomposesFloat == true || math == .ldexp ||
            ["array", "bitcast", "vec2", "vec3", "vec4", "true", "false"].contains(token.text) else { return nil }
    }
    return try scalarConstant(Array(tokens[range]), at: name)
  }
  func constantExpressionTokens(at start: WGSLToken) throws -> [WGSLToken] {
    var result: [WGSLToken] = []
    while !isNext(";") {
      guard token.kind != .end, token.text != "}" else {
        throw wgslError(.unexpectedToken, "Unterminated constant expression", start)
      }
      try node()
      result.append(try take())
    }
    try expect(";")
    return result
  }
  func constAssertion(_ attrs: [String: Attribute] = [:]) throws {
    let start = try expect("const_assert")
    guard attrs.isEmpty else {
      throw wgslError(.unexpectedToken, "Assertions do not accept attributes", start)
    }
    let expression = try constantExpressionTokens(at: start)
    let result = try scalarConstant(expression, at: start)
    guard case .boolean(let accepted) = result else {
      throw wgslError(.typeMismatch, "const_assert requires bool", expression.first ?? start)
    }
    if !accepted {
      let first = expression.first ?? start
      let last = expression.last ?? start
      throw ShaderFailure.diagnostics([
        .init(
          severity: .error, code: .invalidType, message: "const_assert evaluated to false",
          location: .source(try SourceSpan(startUTF8: first.start, endUTF8: last.end)))
      ])
    }
  }
  func localConst(at start: WGSLToken) throws {
    let name = try ident()
    guard body.scopes.last?[wgslKey(name.text)] == nil else {
      throw wgslError(.duplicateName, "Duplicate local constant", name)
    }
    let annotation = try eat(":") ? try type() : nil
    try expect("=")
    let expression = try constantExpressionTokens(at: start)
    var value = try scalarConstant(expression, at: name)
    if let annotation { value = try annotatedConstant(value, type: annotation, at: name) }
    switch value {
    case .array, .composite:
      try define(name, compositeRuntimeValue(value, at: name))
      body.constScopes[body.constScopes.count - 1][wgslKey(name.text)] = value
      return
    default: break
    }
    if case .decomposition(let parts) = value {
      try define(name, decompositionRuntimeValue(parts, at: name))
      body.constScopes[body.constScopes.count - 1][wgslKey(name.text)] = value
      return
    }
    if case .matrix(let matrix)=value {
      let v=try matrixRuntimeValue(matrix,to:annotation,at:name)
      try define(name,v);body.constScopes[body.constScopes.count-1][wgslKey(name.text)]=value;return
    }
    if case .vector(let vector) = value {
      let v = try vectorRuntimeValue(vector, to: annotation, at: name)
      try define(name, v)
      body.constScopes[body.constScopes.count - 1][wgslKey(name.text)] = value
      return
    }
    let v = try expressionNode(
      .literal(value.literal), type: annotation ?? intern(.scalar(value.scalar)), at: name,
      emit: false)
    try define(name, v)
    body.constScopes[body.constScopes.count - 1][wgslKey(name.text)] = value
  }
}
