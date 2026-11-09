import SAGAShaderIR

extension WGSLParser {
  func numericBuiltin(_ function: IRMathFunction, builtin: IRNumericBuiltin, at name: WGSLToken) throws -> WGSLValue {
    guard !(try localConstantNameExists(name)) else { throw wgslError(.typeMismatch, "Local value shadows numeric builtin", name) }
    let callStart = cursor - 2
    var args: [WGSLValue] = []
    var ranges: [Range<Int>] = []
    while !isNext(")") {
      guard args.count < builtin.arity else { throw wgslError(.typeMismatch, "Numeric builtin arity", name) }
      let start = cursor
      args.append(try loaded(expression(), at: name))
      ranges.append(start..<cursor)
      if !(try eat(",")) { break }
    }
    try expect(")")
    let evaluator = WGSLScalarEvaluator(parser: self, tokens: [], fallback: name)
    let kinds = try args.map { try evaluator.numericConstKind(types[$0.type.index].kind, at: name) }
    let signature = try evaluator.numericSignature(builtin, arguments: kinds, materialize: true, at: name)
    if function == .clamp || function == .smoothStep {
      let lowIndex = function == .clamp ? 1 : 0
      if let lower = try knownConstantValue(ranges[lowIndex], at: name),
         let upper = try knownConstantValue(ranges[lowIndex + 1], at: name) {
        try evaluator.validateNumericBounds(function, evaluator.converted(lower, to: signature.result, at: name),
                                             evaluator.converted(upper, to: signature.result, at: name), at: name)
      }
    }
    if let folded = try knownConstantValue(callStart..<cursor, at: name) {
      if case .vector(let vector) = folded { return try vectorRuntimeValue(vector, to: nil, at: name) }
      return try expressionNode(.literal(folded.literal), type: intern(.scalar(folded.scalar)), at: name, emit: false)
    }
    let lane = try signature.result.scalarLeaf.irScalar(at: name)
    let parameterTypes = try signature.arguments.map { argument in
      try intern(IRVectorSize(rawValue: UInt8(argument.lanes)).map { .vector(size: $0, scalar: lane) } ?? .scalar(lane))
    }
    let handles = try zip(args, parameterTypes).map { argument, type in try coerce(argument, to: type, at: name).handle }
    let type = parameterTypes[0]
    return try expressionNode(.math(function: function, argument: handles[0],
                                    argument1: handles.count >= 2 ? handles[1] : nil,
                                    argument2: handles.count == 3 ? handles[2] : nil, argument3: nil),
                              type: type, at: name)
  }
}

extension WGSLScalarEvaluator {
  func numericSignature(_ builtin: IRNumericBuiltin, arguments: [WGSLConstKind], materialize: Bool, at name: WGSLToken)
    throws -> (result: WGSLConstKind, arguments: [WGSLConstKind])
  {
    guard arguments.count == builtin.arity, let first = arguments.first,
          arguments.enumerated().allSatisfy({ index, argument in
            !argument.isMatrix && (argument.lanes == first.lanes ||
              (builtin.scalarLastArgument && index == builtin.arity - 1 && argument.lanes == 1))
          }) else {
      throw wgslError(.typeMismatch, "Numeric builtin operand shape or arity mismatch", name)
    }
    let leaves = arguments.map(\.scalarLeaf)
    let target: WGSLConstKind
    if let concrete = leaves.first(where: { $0 != .integer(.abstract) && $0 != .floating(.abstract) }) {
      target = concrete
    } else if builtin.domain == .binary32 {
      target = .floating(.f32)
    } else if builtin.domain == .floating || leaves.contains(.floating(.abstract)) {
      target = .floating(materialize ? .f32 : .abstract)
    } else {
      target = .integer(materialize ? .sint : .abstract)
    }
    guard try builtin.accepts(target.irScalar(at: name)), leaves.allSatisfy({ canAuto($0, to: target) }) else {
      throw wgslError(.typeMismatch, "Numeric builtin scalar types disagree", name)
    }
    return (shaped(target, like: first), arguments.map { shaped(target, like: $0) })
  }

  func validateNumericBounds(
    _ function: IRMathFunction, _ lower: WGSLScalarConstant, _ upper: WGSLScalarConstant, at name: WGSLToken
  ) throws {
    let valid: Bool
    switch (lower, upper) {
    case (.vector(let left), .vector(let right)):
      guard left.elements.count == right.elements.count else { throw ShaderFailure.invalidHandle }
      for (a, b) in zip(left.elements, right.elements) { try validateNumericBounds(function, a, b, at: name) }
      return
    case (.floating(let left), .floating(let right)):
      valid = function == .smoothStep ? left.value != right.value : left.value <= right.value
    case (.integer(let left), .integer(let right)):
      valid = WGSLIntegerEvaluator(parser: parser, tokens: [], fallback: name).compare("<=", left, right)
    default: throw ShaderFailure.invalidHandle
    }
    if !valid {
      let reason = function == .smoothStep ? "Constant smoothstep edges must differ" : "Const clamp bounds out of order"
      throw wgslError(.invalidType, reason, name)
    }
  }

  mutating func numericConstValue(_ function: IRMathFunction, arguments: [Int], kind: WGSLConstKind, at name: WGSLToken)
    throws -> WGSLScalarConstant
  {
    let values = try arguments.map { argument -> WGSLScalarConstant in
      let target = shaped(kind.scalarLeaf, like: nodes[argument].kind)
      let scalarOrVector = try converted(value(argument), to: target, at: name)
      if target == kind { return scalarOrVector }
      return .vector(.init(elements: Array(repeating: scalarOrVector, count: kind.lanes), element: target))
    }
    return try evaluateNumeric(function, values: values, kind: kind, at: name)
  }

  private func evaluateNumeric(_ function: IRMathFunction, values: [WGSLScalarConstant], kind: WGSLConstKind, at name: WGSLToken)
    throws -> WGSLScalarConstant
  {
    if function == .sqrt || function == .inverseSqrt {
      return try floatingRoot(values[0], result: kind, inverse: function == .inverseSqrt, at: name)
    }
    if case .vector(let count, let element) = kind {
      let vectors = try values.map { value -> [WGSLScalarConstant] in
        guard case .vector(let vector) = value, vector.elements.count == count else { throw ShaderFailure.invalidHandle }
        return vector.elements
      }
      let elements = try (0..<count).map { lane in
        try parser.budget.consume(.work)
        return try evaluateNumeric(function, values: vectors.map { $0[lane] }, kind: element, at: name)
      }
      return .vector(.init(elements: elements, element: element))
    }
    if case .integer(let integerKind) = kind {
      let integers = try values.map { value -> WGSLIntegerConstant in
        guard case .integer(let integer) = value else { throw ShaderFailure.invalidHandle }
        return integer
      }
      let value = integers[0].value
      let arithmetic = WGSLIntegerEvaluator(parser: parser, tokens: [], fallback: name)
      func minValue(_ lhs: WGSLIntegerConstant, _ rhs: WGSLIntegerConstant) -> WGSLIntegerConstant {
        arithmetic.compare("<", lhs, rhs) ? lhs : rhs
      }
      func maxValue(_ lhs: WGSLIntegerConstant, _ rhs: WGSLIntegerConstant) -> WGSLIntegerConstant {
        arithmetic.compare(">", lhs, rhs) ? lhs : rhs
      }
      let result: Int64
      switch function {
      case .abs:
        if integerKind.isUnsigned || (integerKind != .abstract && value == integerKind.signedMinimum) {
          return .integer(integers[0])
        }
        guard value != Int64.min else { throw wgslError(.invalidType, "Abstract absolute value overflow", name) }
        result = abs(value)
      case .sign: result = value == 0 ? 0 : value < 0 ? -1 : 1
      case .min: return .integer(minValue(integers[0], integers[1]))
      case .max: return .integer(maxValue(integers[0], integers[1]))
      case .clamp: return .integer(minValue(maxValue(integers[0], integers[1]), integers[2]))
      default: throw ShaderFailure.invalidHandle
      }
      return .integer(.init(value: result, kind: integerKind))
    }
    guard case .floating(let floatKind) = kind else { throw ShaderFailure.invalidHandle }
    let floats = try values.map { value -> Double in
      guard case .floating(let float) = value else { throw ShaderFailure.invalidHandle }
      return float.value
    }
    let value = floats[0]
    let result: Double
    switch function {
    case .abs: result = abs(value)
    case .sign: result = value == 0 ? 0 : value < 0 ? -1 : 1
    case .min: result = min(value, floats[1])
    case .max: result = max(value, floats[1])
    case .clamp: result = min(max(value, floats[1]), floats[2])
    case .saturate: result = min(max(value, 0), 1)
    case .ceil: result = value.rounded(.up)
    case .floor: result = value.rounded(.down)
    case .trunc: result = value.rounded(.towardZero)
    case .round: result = value.rounded(.toNearestOrEven)
    case .fract: result = value - value.rounded(.down)
    case .quantizeToF16:
      guard abs(value) <= 65504 else {
        throw wgslError(.invalidType, "Constant quantizeToF16 exceeds the finite binary16 range", name)
      }
      result = Double(Binary16.decode(Binary16.encode(value)))
    case .fma:
      if floatKind == .f32 {
        result = Double(Float(floats[2]).addingProduct(Float(value), Float(floats[1])))
      } else {
        result = floats[2].addingProduct(value, floats[1])
      }
    case .mix: result = value * (1 - floats[2]) + floats[1] * floats[2]
    case .step: result = value <= floats[1] ? 1 : 0
    case .smoothStep:
      let t = min(max((floats[2] - value) / (floats[1] - value), 0), 1)
      result = t * t * (3 - 2 * t)
    default: result = try transcendentalConstant(function, values: floats)
    }
    return try floating(result, kind: floatKind, at: name)
  }
}
