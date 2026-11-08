import SAGAShaderIR

private func bitcastShape(_ kind: IRTypeKind, at token: WGSLToken) throws -> IRBitcastShape {
  guard let shape = IRBitcastShape(kind) else {
    throw wgslError(.typeMismatch, "Bitcast requires a numeric scalar or vector", token)
  }
  return shape
}

private func bitcastInput(_ input: IRBitcastShape, target: IRBitcastShape, at token: WGSLToken) throws -> IRBitcastShape {
  let supported: [IRScalar] = [.i32, .u32, .i64, .u64, .f32, .f16]
  let concrete: IRBitcastShape
  if input.scalar == .abstractInt || input.scalar == .abstractFloat {
    let half = IRBitcastShape(size: input.size, scalar: .f16)
    if (target.scalar == .f16 && target.size == input.size) || half.packedHalfWords(to: target) != nil {
      concrete = half
    } else {
      let scalar: IRScalar = input.scalar == .abstractFloat ? .f32 :
        target.scalar.width == 8 || target.scalar == .u32 ? target.scalar : .i32
      concrete = .init(size: input.size, scalar: scalar)
    }
  } else {
    concrete = input
  }
  guard supported.contains(concrete.scalar), supported.contains(target.scalar), concrete.permits(target) else {
    throw wgslError(.typeMismatch, "Invalid bitcast lane types or shapes", token)
  }
  return concrete
}

extension WGSLParser {
  func bitcastExpression(at name: WGSLToken, start: Int) throws -> WGSLValue {
    try expect("<")
    let target = try type()
    try expect(">")
    try expect("(")
    let arguments = try arguments()
    guard arguments.count == 1 else { throw wgslError(.typeMismatch, "Bitcast requires one operand", name) }
    let value = arguments[0]
    let input = try bitcastShape(types[value.type.index].kind, at: name)
    let output = try bitcastShape(types[target.index].kind, at: name)
    let concrete = try bitcastInput(input, target: output, at: name)
    if concrete.scalar == .f16 { try requireFloat16(at: name) }

    if let folded = try knownConstantValue(start..<cursor, at: name) {
      if case .vector(let vector) = folded { return try vectorRuntimeValue(vector, to: target, at: name) }
      return try expressionNode(.literal(folded.literal), type: target, at: name, emit: false)
    }
    let operand = try coerce(value, to: intern(concrete.type), at: name)
    return try expressionNode(.bitcast(expression: operand.handle, type: target), type: target, at: name)
  }

}

extension WGSLScalarEvaluator {
  mutating func constantBitcast(at name: WGSLToken) throws -> Int {
    guard parser.declarationNames[wgslKey(name.text)] == nil, !(try parser.localConstantNameExists(name)) else {
      throw wgslError(.typeMismatch, "Bitcast is shadowed", name)
    }
    _ = try take() // <
    let type = try take()
    let kind: IRTypeKind
    if let size = ["vec2": IRVectorSize.two, "vec3": .three, "vec4": .four][type.text],
       parser.declarationNames[wgslKey(type.text)] == nil, index < tokens.count, token.kind == .templateStart {
      _ = try take()
      let leaf = try take()
      guard case .scalar(let scalar) = try bitcastNamedType(leaf), try take().kind == .templateEnd else {
        throw wgslError(.typeMismatch, "Invalid bitcast vector type", leaf)
      }
      kind = .vector(size: size, scalar: scalar)
    } else {
      kind = try bitcastNamedType(type)
    }
    guard try take().kind == .templateEnd, try take().text == "(" else {
      throw wgslError(.unexpectedToken, "Expected bitcast operand", name)
    }
    let operand = try parse()
    if index < tokens.count, token.text == "," { _ = try take() }
    guard try take().text == ")" else { throw wgslError(.typeMismatch, "Bitcast requires one operand", name) }
    let target = try bitcastShape(kind, at: type)
    let source = nodes[operand].kind
    let shape = try IRBitcastShape(size: IRVectorSize(rawValue: UInt8(source.lanes)),
                                   scalar: source.scalarLeaf.irScalar(at: name))
    let concrete = try bitcastInput(shape, target: target, at: name)
    if concrete.scalar == .f16 { try parser.requireFloat16(at: name) }
    let input = try numericConstKind(concrete.type, at: name)
    let result = try numericConstKind(target.type, at: name)
    if source != input { conversions.append((operand, input, name)) }
    return try append(.bitcast(operand, input), result, name)
  }

  private func bitcastNamedType(_ token: WGSLToken) throws -> IRTypeKind {
    try parser.resolveTypeDeclarationIfIndexed(token)
    if let type = parser.typeNames[wgslKey(token.text)] { return parser.types[type.index].kind }
    if let kind = parser.predeclaredType(token.text) {
      if kind.usesFloat16 { try parser.requireFloat16(at: token) }
      return kind
    }
    switch token.text {
    case "i32": return .scalar(.i32)
    case "u32": return .scalar(.u32)
    case "i64": return .scalar(.i64)
    case "u64": return .scalar(.u64)
    case "f32": return .scalar(.f32)
    case "f16": try parser.requireFloat16(at: token); return .scalar(.f16)
    case "bool": return .scalar(.bool)
    default: throw wgslError(.typeMismatch, "Invalid bitcast target type", token)
    }
  }

  func bitcastConstant(_ value: WGSLScalarConstant, to target: WGSLConstKind, at name: WGSLToken) throws -> WGSLScalarConstant {
    if value.kind == target { return value }
    if value.kind.scalarLeaf == .floating(.f16) || target.scalarLeaf == .floating(.f16) {
      return try packedHalfConstant(value, to: target, at: name)
    }
    if case .vector(let vector) = value, case .vector(_, let scalar) = target {
      let elements = try vector.elements.map { lane in
        try parser.budget.consume(.work)
        return try bitcastConstant(lane, to: scalar, at: name)
      }
      return .vector(.init(elements: elements, element: scalar))
    }
    if case .integer(let source) = value, source.kind.isWide,
       case .integer(let destination) = target, destination.isWide {
      return .integer(.init(value: source.value, kind: destination))
    }
    return try bitcastWordValue(bitcastWordBits(value), to: target, at: name)
  }

  private func bitcastWordBits(_ value: WGSLScalarConstant) throws -> UInt32 {
    switch try value.literal {
    case .i32(let x): return UInt32(bitPattern: x)
    case .u32(let x): return x
    case .f32(let x): return x.bitPattern
    default: throw ShaderFailure.invalidHandle
    }
  }

  private func bitcastWordValue(_ bits: UInt32, to target: WGSLConstKind, at name: WGSLToken) throws -> WGSLScalarConstant {
    switch target {
    case .integer(.sint): return .integer(.init(value: Int64(Int32(bitPattern: bits)), kind: .sint))
    case .integer(.uint): return .integer(.init(value: Int64(bits), kind: .uint))
    case .floating(.f32): return try floating(Double(Float(bitPattern: bits)), kind: .f32, at: name)
    default: throw ShaderFailure.invalidHandle
    }
  }

  private func packedHalfConstant(
    _ value: WGSLScalarConstant, to target: WGSLConstKind, at name: WGSLToken
  ) throws -> WGSLScalarConstant {
    if target.scalarLeaf == .floating(.f16) {
      let words: [WGSLScalarConstant]
      if case .vector(let vector) = value { words = vector.elements } else { words = [value] }
      let halves = try words.flatMap { word in
        let bits = try bitcastWordBits(word)
        return try [UInt16(truncatingIfNeeded: bits), UInt16(bits >> 16)].map { lane -> WGSLScalarConstant in
          try parser.budget.consume(.work)
          return try floating(Double(Binary16.decode(lane)), kind: .f16, at: name)
        }
      }
      return .vector(.init(elements: halves, element: .floating(.f16)))
    }
    guard case .vector(let vector) = value, vector.elements.count == target.lanes * 2 else {
      throw ShaderFailure.invalidHandle
    }
    let words = try (0..<target.lanes).map { index in
      try parser.budget.consume(.work)
      guard case .f16Bits(let low) = try vector.elements[index * 2].literal,
            case .f16Bits(let high) = try vector.elements[index * 2 + 1].literal else {
        throw ShaderFailure.invalidHandle
      }
      return try bitcastWordValue(UInt32(low) | (UInt32(high) << 16), to: target.scalarLeaf, at: name)
    }
    if target.lanes == 1 { return words[0] }
    return .vector(.init(elements: words, element: target.scalarLeaf))
  }
}
