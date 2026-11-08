import SAGAShaderIR

extension WGSLParser {
  func packingCall(_ builtin: IRPackingBuiltin, function: IRMathFunction, at name: WGSLToken) throws -> WGSLValue {
    guard !(try localConstantNameExists(name)) else {
      throw wgslError(.typeMismatch, "Local value shadows packing builtin", name)
    }
    let start = cursor
    let argument = try expression()
    let end = cursor
    _ = try eat(",")
    guard isNext(")") else { throw wgslError(.typeMismatch, "Packing builtin requires one operand", name) }
    try expect(")")
    let operand = try coerce(argument, to: intern(builtin.argumentType), at: name)
    let result = try intern(builtin.resultType)
    if let value = try knownConstantValue(start..<end, at: name) {
      let evaluator = WGSLScalarEvaluator(parser: self, tokens: [], fallback: name)
      let type = try evaluator.numericConstKind(builtin.argumentType, at: name)
      let folded = try evaluator.packingConstant(builtin, argument: evaluator.converted(value, to: type, at: name), at: name)
      if case .vector(let vector) = folded { return try vectorRuntimeValue(vector, to: result, at: name) }
      return try expressionNode(.literal(folded.literal), type: result, at: name, emit: false)
    }
    return try expressionNode(.math(function: function, argument: operand.handle,
                                    argument1: nil, argument2: nil, argument3: nil), type: result, at: name)
  }
}

extension WGSLScalarEvaluator {
  func packingConstant(_ builtin: IRPackingBuiltin, argument: WGSLScalarConstant, at name: WGSLToken) throws -> WGSLScalarConstant {
    if builtin.packs {
      guard case .vector(let vector) = argument else { throw ShaderFailure.invalidHandle }
      var word: UInt32 = 0
      for (index, element) in vector.elements.enumerated() {
        try parser.budget.consume(.work)
        let bits: UInt32
        switch builtin.encoding {
        case .snorm, .unorm:
          guard case .floating(let value) = element else { throw ShaderFailure.invalidHandle }
          let lower = builtin.encoding == .snorm ? -1.0 : 0.0
          let scaled = min(1.0, max(lower, value.value)) * Double(builtin.normalizedScale)
          bits = UInt32(truncatingIfNeeded: Int64((scaled + 0.5).rounded(.down)))
        case .binary16:
          guard case .floating(let value) = element else { throw ShaderFailure.invalidHandle }
          guard abs(value.value) <= 65504 else {
            throw wgslError(.invalidType, "Constant pack2x16float exceeds the finite binary16 range", name)
          }
          bits = UInt32(Binary16.encode(value.value))
        case .sint, .uint, .sintClamp, .uintClamp:
          guard case .integer(let value) = element else { throw ShaderFailure.invalidHandle }
          let number: Int64
          switch builtin.encoding {
          case .sintClamp: number = min(127, max(-128, value.value))
          case .uintClamp: number = min(255, value.value)
          default: number = value.value
          }
          bits = UInt32(truncatingIfNeeded: number)
        }
        word |= (bits & builtin.mask) << (index * builtin.bitsPerLane)
      }
      return .integer(.init(value: Int64(word), kind: .uint))
    }

    guard case .integer(let word) = argument else { throw ShaderFailure.invalidHandle }
    let element = try WGSLConstKind.scalar(builtin.encoding.scalar, at: name)
    var values: [WGSLScalarConstant] = []
    for index in 0..<Int(builtin.size.rawValue) {
      try parser.budget.consume(.work)
      let field = (UInt32(word.value) >> (index * builtin.bitsPerLane)) & builtin.mask
      let sign: UInt32 = builtin.encoding.isSignedIntegerField ? 1 << (builtin.bitsPerLane - 1) : 0
      let number = Int64(Int32(bitPattern: (field ^ sign) &- sign))
      switch builtin.encoding {
      case .snorm, .unorm:
        let decoded = Double(number) / Double(builtin.normalizedScale)
        values.append(try floating(max(-1.0, decoded), kind: .f32, at: name))
      case .binary16:
        values.append(try floating(Double(Binary16.decode(UInt16(field))), kind: .f32, at: name))
      case .sint, .uint:
        values.append(.integer(.init(value: number, kind: builtin.encoding == .sint ? .sint : .uint)))
      case .sintClamp, .uintClamp: throw ShaderFailure.invalidHandle
      }
    }
    return .vector(.init(elements: values, element: element))
  }
}
