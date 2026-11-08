import SAGAShaderIR

extension WGSLParser {
  func ldexpBuiltin(at name: WGSLToken) throws -> WGSLValue {
    guard !(try localConstantNameExists(name)), !isNext(")") else {
      throw wgslError(.typeMismatch, "ldexp requires a mantissa and exponent", name)
    }
    let start = cursor - 2
    let mantissa = try loaded(expression(), at: name)
    guard try eat(","), !isNext(")") else {
      throw wgslError(.typeMismatch, "ldexp requires a mantissa and exponent", name)
    }
    let exponentStart = cursor
    let exponent = try loaded(expression(), at: name)
    let exponentEnd = cursor
    _ = try eat(",")
    guard isNext(")") else { throw wgslError(.typeMismatch, "ldexp requires two operands", name) }
    try expect(")")

    if let folded = try knownConstantValue(start..<cursor, at: name) {
      if case .vector(let vector) = folded { return try vectorRuntimeValue(vector, to: nil, at: name) }
      return try expressionNode(.literal(folded.literal), type: intern(.scalar(folded.scalar)), at: name, emit: false)
    }

    let evaluator = WGSLScalarEvaluator(parser: self, tokens: [], fallback: name)
    let signature = try evaluator.ldexpSignature(
      mantissa: evaluator.numericConstKind(types[mantissa.type.index].kind, at: name),
      exponent: evaluator.numericConstKind(types[exponent.type.index].kind, at: name),
      materialize: true, at: name)
    let width = IRVectorSize(rawValue: UInt8(signature.mantissa.lanes))
    let scalar = try signature.mantissa.scalarLeaf.irScalar(at: name)
    let result: IRTypeKind = width.map { .vector(size: $0, scalar: scalar) } ?? .scalar(scalar)
    guard let exponentType = result.ldexpExponentType else { throw ShaderFailure.invalidHandle }

    if let constant = try knownConstantValue(exponentStart..<exponentEnd, at: name) {
      try evaluator.validateLdexpExponent(evaluator.converted(constant, to: signature.exponent, at: name),
                                          mantissa: signature.mantissa, at: name)
    }
    let type = try intern(result)
    let x = try coerce(mantissa, to: type, at: name)
    let y = try coerce(exponent, to: intern(exponentType), at: name)
    return try expressionNode(.math(function: .ldexp, argument: x.handle, argument1: y.handle,
                                    argument2: nil, argument3: nil), type: type, at: name)
  }
}

extension WGSLScalarEvaluator {
  func ldexpSignature(
    mantissa: WGSLConstKind, exponent: WGSLConstKind, materialize: Bool, at name: WGSLToken
  ) throws -> (mantissa: WGSLConstKind, exponent: WGSLConstKind) {
    let x = mantissa.scalarLeaf
    let y = exponent.scalarLeaf
    guard !mantissa.isMatrix, !exponent.isMatrix, mantissa.lanes == exponent.lanes,
          x == .integer(.abstract) || x == .floating(.abstract) || x == .floating(.f32) || x == .floating(.f16),
          y == .integer(.abstract) || y == .integer(.sint) else {
      throw wgslError(.typeMismatch, "ldexp requires float and signed integer operands of matching shape", name)
    }
    let concrete = materialize || x == .floating(.f32) || x == .floating(.f16) || y == .integer(.sint)
    let lane: WGSLConstKind = x == .floating(.f16) ? x : .floating(concrete ? .f32 : .abstract)
    return (shaped(lane, like: mantissa),
            shaped(.integer(concrete ? .sint : .abstract), like: exponent))
  }

  mutating func ldexpConstCall(_ arguments: [Int], at name: WGSLToken) throws -> Int {
    guard arguments.count == 2 else { throw wgslError(.typeMismatch, "ldexp requires two operands", name) }
    let x = arguments[0], y = arguments[1]
    let signature = try ldexpSignature(mantissa: nodes[x].kind, exponent: nodes[y].kind, materialize: false, at: name)
    if nodes[x].kind != signature.mantissa { conversions.append((x, signature.mantissa, name)) }
    if nodes[y].kind != signature.exponent { conversions.append((y, signature.exponent, name)) }
    return try append(.ldexp(x, y, signature.exponent), signature.mantissa, name)
  }

  func validateLdexpExponent(_ value: WGSLScalarConstant, mantissa: WGSLConstKind, at name: WGSLToken) throws {
    let maximum: Int64 = switch mantissa.scalarLeaf {
    case .floating(.f16): 16
    case .floating(.f32): 128
    default: 1024
    }
    for lane in try components(value, width: mantissa.lanes, at: name) {
      try parser.budget.consume(.work)
      guard case .integer(let exponent) = lane else { throw ShaderFailure.invalidHandle }
      guard exponent.value <= maximum else {
        throw wgslError(.invalidType, "Constant ldexp exponent exceeds the floating-point format limit", name)
      }
    }
  }

  mutating func ldexpConstValue(
    _ mantissa: Int, _ exponent: Int, exponentKind: WGSLConstKind, result: WGSLConstKind, at name: WGSLToken
  ) throws -> WGSLScalarConstant {
    let x = try components(converted(value(mantissa), to: result, at: name), width: result.lanes, at: name)
    let y = try components(converted(value(exponent), to: exponentKind, at: name), width: result.lanes, at: name)
    let scaled = try zip(x, y).map { mantissa, exponent -> WGSLScalarConstant in
      try parser.budget.consume(.work)
      guard case .floating(let number) = mantissa, case .integer(let power) = exponent else {
        throw ShaderFailure.invalidHandle
      }
      if number.value == 0 { return mantissa }
      // No binary64 number can survive scaling down by more than this. Bound the
      // exponent before adding it to the normalized input exponent.
      let shift = max(power.value, -2099)
      let value = Double(sign: number.value.sign, exponent: number.value.exponent + Int(shift),
                         significand: number.value.significand)
      return try floating(value, kind: number.kind, at: name)
    }
    if result.lanes == 1 { return scaled[0] }
    return .vector(.init(elements: scaled, element: result.scalarLeaf))
  }
}
