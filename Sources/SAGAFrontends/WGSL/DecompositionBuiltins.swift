import SAGAShaderIR

struct WGSLDecompositionConstant {
  let function: IRMathFunction
  let fraction: WGSLScalarConstant
  let second: WGSLScalarConstant
}

extension WGSLParser {
  func decompositionBuiltin(_ function: IRMathFunction, at name: WGSLToken) throws -> WGSLValue {
    let start = cursor - 2
    guard !(try localConstantNameExists(name)), !isNext(")") else {
      throw wgslError(.typeMismatch, "Float decomposition requires one operand", name)
    }
    let argument = try loaded(expression(), at: name)
    _ = try eat(",")
    guard isNext(")") else { throw wgslError(.typeMismatch, "Float decomposition requires one operand", name) }
    try expect(")")
    if case .decomposition(let constant)? = try knownConstantValue(start..<cursor, at: name) {
      return try decompositionRuntimeValue(constant, at: name)
    }

    guard let shape = types[argument.type.index].kind.concreteFloatShape else {
      throw wgslError(.typeMismatch, "Float decomposition requires a floating scalar or vector", name)
    }
    let type = try decompositionType(function, size: shape.size, scalar: shape.scalar, at: name)
    return try expressionNode(.math(function: function, argument: argument.handle, argument1: nil,
                                    argument2: nil, argument3: nil), type: type, at: name)
  }

  private func decompositionType(
    _ function: IRMathFunction, size: IRVectorSize?, scalar: IRScalar, at name: WGSLToken
  ) throws -> WType {
    let kind = function.decompositionResult(size: size, scalar: scalar)
    if let existing = predeclaredTypes.first(where: { $0.kind == kind }) { return existing.type }
    func memberType(_ lane: IRScalar) throws -> WType {
      try intern(size.map { .vector(size: $0, scalar: lane) } ?? .scalar(lane))
    }
    let stride = function.decompositionMemberStride(size: size, scalar: scalar)
    let members: [IRStructMember] = [
      .init(name: "fract", type: try memberType(scalar), offset: 0),
      .init(name: function.decompositionSecondMember,
            type: try memberType(function.decompositionSecondScalar(scalar)), offset: stride),
    ]
    let lane = scalar == .abstractFloat ? "abstract" : scalar == .f16 ? "f16" : "f32"
    let result = try intern(.structure(members: members, span: stride * 2),
                            name: "__\(function.rawValue)_result_\(size?.rawValue ?? 1)_\(lane)", at: name)
    predeclaredTypes.append(.init(kind: kind, type: result))
    return result
  }

  func decompositionRuntimeValue(
    _ constant: WGSLDecompositionConstant, at name: WGSLToken
  ) throws -> WGSLValue {
    let fraction = try decompositionComponent(constant.fraction, at: name)
    let second = try decompositionComponent(constant.second, at: name)
    let type = try decompositionType(constant.function, size: IRVectorSize(rawValue: UInt8(constant.fraction.kind.lanes)),
                                     scalar: constant.fraction.kind.scalarLeaf.irScalar(at: name), at: name)
    return try expressionNode(.compose(type: type, components: [fraction.handle, second.handle]), type: type, at: name)
  }

  private func decompositionComponent(_ constant: WGSLScalarConstant, at name: WGSLToken) throws -> WGSLValue {
    if case .vector(let vector) = constant { return try vectorRuntimeValue(vector, to: nil, at: name) }
    return try expressionNode(.literal(constant.literal), type: intern(.scalar(constant.scalar)), at: name, emit: false)
  }

  func materializeDecomposition(_ value: WGSLValue, at name: WGSLToken) throws -> WGSLValue? {
    guard let shape = predeclaredTypes.first(where: { $0.type == value.type })?.kind.decomposition,
          shape.scalar == .abstractFloat else { return nil }
    guard case .compose(_, let components) = body.expressions[value.handle.index].kind else {
      throw ShaderFailure.invalidHandle
    }
    let type = try decompositionType(shape.function, size: shape.size, scalar: .f32, at: name)
    guard case .structure(let members, _) = types[type.index].kind else { throw ShaderFailure.invalidHandle }
    let converted = try zip(components, members).map { handle, member in
      try coerce(.init(handle: handle, type: body.expressions[handle.index].type, pointer: false, writable: false),
                 to: member.type, at: name).handle
    }
    return try expressionNode(.compose(type: type, components: converted), type: type, at: name)
  }
}

extension WGSLScalarEvaluator {
  mutating func decompositionMember(_ base: Int, fraction: Bool) throws -> WGSLScalarConstant {
    guard case .decomposition(let parts) = try value(base) else { throw ShaderFailure.invalidHandle }
    return fraction ? parts.fraction : parts.second
  }

  mutating func decompositionCall(_ function: IRMathFunction, arguments: [Int], at name: WGSLToken) throws -> Int {
    guard arguments.count == 1, let argument = arguments.first else {
      throw wgslError(.typeMismatch, "Float decomposition requires one operand", name)
    }
    let kind = nodes[argument].kind
    let input: WGSLConstKind
    switch kind {
    case .floating, .vector(_, .floating): input = kind
    case .integer(.abstract), .vector(_, .integer(.abstract)):
      input = shaped(.floating(.abstract), like: kind)
      conversions.append((argument, input, name))
    default: throw wgslError(.typeMismatch, "Float decomposition requires a float or float vector", name)
    }
    return try append(.decomposition(function, argument, input), .decomposition(function, input), name)
  }

  mutating func decompositionValue(
    _ function: IRMathFunction, argument: Int, input: WGSLConstKind, at name: WGSLToken
  ) throws -> WGSLScalarConstant {
    let value = try converted(value(argument), to: input, at: name)
    let width = input.lanes
    let secondKind = input.decompositionSecondKind(function)
    var fractions: [WGSLScalarConstant] = []
    var second: [WGSLScalarConstant] = []
    for lane in try components(value, width: width, at: name) {
      try parser.budget.consume(.work)
      guard case .floating(let float) = lane else { throw ShaderFailure.invalidHandle }
      let number = float.value
      if function == .modf {
        let whole = number.rounded(.towardZero)
        fractions.append(try floating(number - whole, kind: float.kind, at: name))
        second.append(.floating(.init(value: whole, kind: float.kind)))
      } else {
        let fraction = number == 0 ? number : Double(sign: number.sign, exponent: -1, significand: number.significand)
        fractions.append(try floating(fraction, kind: float.kind, at: name))
        second.append(.integer(.init(value: number == 0 ? 0 : Int64(number.exponent) + 1,
                                     kind: float.kind == .abstract ? .abstract : .sint)))
      }
    }
    if width == 1 {
      return .decomposition(.init(function: function, fraction: fractions[0], second: second[0]))
    }
    return .decomposition(.init(function: function, fraction: .vector(.init(elements: fractions, element: input.scalarLeaf)),
                                second: .vector(.init(elements: second, element: secondKind.scalarLeaf))))
  }
}

extension WGSLConstKind {
  func decompositionSecondKind(_ function: IRMathFunction) -> WGSLConstKind {
    guard function == .frexp else { return self }
    let leaf: WGSLConstKind = .integer(scalarLeaf == .floating(.abstract) ? .abstract : .sint)
    return lanes == 1 ? leaf : .vector(lanes, leaf)
  }
}
