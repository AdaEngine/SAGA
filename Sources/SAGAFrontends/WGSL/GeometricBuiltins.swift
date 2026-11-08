import SAGAShaderIR

private func geometricShape(_ type: IRTypeKind, at name: WGSLToken) throws -> (size: IRVectorSize?, scalar: IRScalar) {
  switch type {
  case .scalar(let scalar): return (nil, scalar)
  case .vector(let width, let scalar): return (width, scalar)
  default: throw wgslError(.typeMismatch, "Geometric builtin requires scalar or vector operands", name)
  }
}

private func geometricType(_ size: IRVectorSize?, _ scalar: IRScalar) -> IRTypeKind {
  size.map { .vector(size: $0, scalar: scalar) } ?? .scalar(scalar)
}

private func geometricSignature(
  _ function: IRMathFunction, types: [IRTypeKind], materialize: Bool, at name: WGSLToken
) throws -> (arguments: [IRTypeKind], result: IRTypeKind) {
  guard types.count == function.geometricArity, !types.isEmpty else {
    throw wgslError(.typeMismatch, "Geometric builtin arity mismatch", name)
  }
  if case .matrix(let columns, let rows, let scalar) = types[0] {
    let concrete = materialize && scalar == .abstractFloat ? .f32 : scalar
    let operand = IRTypeKind.matrix(columns: columns, rows: rows, scalar: concrete)
    guard let signature = function.geometricSignature(for: operand) else {
      throw wgslError(.typeMismatch, "Invalid matrix shape or scalar for \(function.rawValue)", name)
    }
    return signature
  }
  let shapes = try types.map { try geometricShape($0, at: name) }
  let packed = function == .dot4I8Packed || function == .dot4U8Packed
  let scalar: IRScalar
  if packed {
    scalar = .u32
  } else if let concrete = shapes.first(where: { $0.scalar != .abstractInt && $0.scalar != .abstractFloat }) {
    scalar = concrete.scalar
  } else if function != .dot || shapes.contains(where: { $0.scalar == .abstractFloat }) {
    scalar = materialize ? .f32 : .abstractFloat
  } else {
    scalar = materialize ? .i32 : .abstractInt
  }
  let primary = geometricType(shapes[0].size, scalar)
  guard let signature = function.geometricSignature(for: primary) else {
    throw wgslError(.typeMismatch, "Invalid geometric builtin operand type", name)
  }
  for (actual, expected) in zip(shapes, signature.arguments) {
    let target = try geometricShape(expected, at: name)
    let converts = actual.scalar == target.scalar || actual.scalar == .abstractInt ||
      (actual.scalar == .abstractFloat && [.f16, .f32].contains(target.scalar))
    guard actual.size == target.size, converts else {
      throw wgslError(.typeMismatch, "Geometric builtin operand types or dimensions disagree", name)
    }
  }
  return signature
}

extension WGSLParser {
  func geometricBuiltin(_ function: IRMathFunction, at name: WGSLToken) throws -> WGSLValue {
    guard !(try localConstantNameExists(name)) else {
      throw wgslError(.typeMismatch, "Local value shadows geometric builtin", name)
    }
    let start = cursor - 2
    let args = try arguments()
    let signature = try geometricSignature(function, types: args.map { types[$0.type.index].kind }, materialize: true, at: name)
    if let folded = try knownConstantValue(start..<cursor, at: name) {
      if case .matrix(let matrix) = folded { return try matrixRuntimeValue(matrix, to: nil, at: name) }
      if case .vector(let vector) = folded { return try vectorRuntimeValue(vector, to: nil, at: name) }
      return try expressionNode(.literal(folded.literal), type: intern(.scalar(folded.scalar)), at: name, emit: false)
    }
    let values = try zip(args, signature.arguments).map { arg, type in try coerce(arg, to: intern(type), at: name).handle }
    return try expressionNode(.math(function: function, argument: values[0],
                                    argument1: values.count > 1 ? values[1] : nil,
                                    argument2: values.count > 2 ? values[2] : nil, argument3: nil),
                              type: intern(signature.result), at: name)
  }
}

extension WGSLScalarEvaluator {
  mutating func geometricConstCall(_ function: IRMathFunction, arguments: [Int], at name: WGSLToken) throws -> Int {
    let types = try arguments.map { index -> IRTypeKind in
      let kind = nodes[index].kind
      if case .matrix(let columns, let rows, let element) = kind,
         let c = IRVectorSize(rawValue: UInt8(columns)), let r = IRVectorSize(rawValue: UInt8(rows)) {
        return try .matrix(columns: c, rows: r, scalar: element.irScalar(at: name))
      }
      let scalar = try kind.scalarLeaf.irScalar(at: name)
      return geometricType(IRVectorSize(rawValue: UInt8(kind.lanes)), scalar)
    }
    let signature = try geometricSignature(function, types: types, materialize: false, at: name)
    let inputs = try signature.arguments.map { try numericConstKind($0, at: name) }
    for (argument, kind) in zip(arguments, inputs) where nodes[argument].kind != kind {
      conversions.append((argument, kind, name))
    }
    return try append(.geometric(function, arguments, inputs), numericConstKind(signature.result, at: name), name)
  }

  func geometricConstant(
    _ function: IRMathFunction, arguments: [WGSLScalarConstant], result: WGSLConstKind, at name: WGSLToken
  ) throws -> WGSLScalarConstant {
    if case .matrix(let matrix) = arguments[0] {
      return try matrixBuiltinConstant(function, matrix: matrix, at: name)
    }
    if function == .dot4I8Packed || function == .dot4U8Packed {
      guard case .integer(let first) = arguments[0], case .integer(let second) = arguments[1] else {
        throw ShaderFailure.invalidHandle
      }
      let signed = function == .dot4I8Packed
      func lane(_ value: Int64, _ shift: Int) -> Int64 {
        let bits = UInt8(truncatingIfNeeded: value >> shift)
        return signed ? Int64(Int8(bitPattern: bits)) : Int64(bits)
      }
      var sum: Int64 = 0
      for shift in [0, 8, 16, 24] {
        try parser.budget.consume(.work)
        sum += lane(first.value, shift) * lane(second.value, shift)
      }
      return .integer(.init(value: sum, kind: signed ? .sint : .uint))
    }
    if case .integer(let kind) = result {
      guard case .vector(let first) = arguments[0], case .vector(let second) = arguments[1] else {
        throw ShaderFailure.invalidHandle
      }
      let arithmetic = WGSLIntegerEvaluator(parser: parser, tokens: [], fallback: name)
      var sum = WGSLIntegerConstant(value: 0, kind: kind)
      for (left, right) in zip(first.elements, second.elements) {
        try parser.budget.consume(.work)
        guard case .integer(let a) = left, case .integer(let b) = right else { throw ShaderFailure.invalidHandle }
        sum = try arithmetic.binary("+", sum, arithmetic.binary("*", a, b, at: name), at: name)
      }
      return .integer(sum)
    }
    guard case .floating(let kind) = result.scalarLeaf else { throw ShaderFailure.invalidHandle }
    let operands = try arguments.map { argument -> [Double] in
      let elements: [WGSLScalarConstant]
      if case .vector(let vector) = argument { elements = vector.elements } else { elements = [argument] }
      return try elements.map { element in
        guard case .floating(let value) = element else { throw ShaderFailure.invalidHandle }
        return value.value
      }
    }
    func rounded(_ value: Double) throws -> Double {
      try parser.budget.consume(.work)
      guard case .floating(let checked) = try floating(value, kind: kind, at: name) else {
        throw ShaderFailure.invalidHandle
      }
      return checked.value
    }
    func dot(_ left: [Double], _ right: [Double]) throws -> Double {
      var sum = 0.0
      for (a, b) in zip(left, right) { sum = try rounded(sum + rounded(a * b)) }
      return sum
    }
    func scaled(_ vector: [Double]) -> (lanes: [Double], factor: Double) {
      let factor = vector.map(abs).max() ?? 0
      return (factor == 0 ? vector : vector.map { $0 / factor }, factor)
    }
    func length(_ vector: [Double]) throws -> Double {
      if vector.count == 1 { return abs(vector[0]) }
      let (lanes, factor) = scaled(vector)
      return try rounded(dot(lanes, lanes).squareRoot() * factor)
    }
    let a = operands[0]
    let values: [Double]
    switch function {
    case .dot: values = [try dot(a, operands[1])]
    case .length: values = [try length(a)]
    case .distance:
      values = [try length(zip(a, operands[1]).map { try rounded($0 - $1) })]
    case .cross:
      let b = operands[1]
      values = try [0, 1, 2].map { lane in
        let i = (lane + 1) % 3, j = (lane + 2) % 3
        return try rounded(rounded(a[i] * b[j]) - rounded(a[j] * b[i]))
      }
    case .normalize:
      let (lanes, factor) = scaled(a)
      guard factor > 0 else { throw wgslError(.invalidType, "Cannot normalize a zero constant vector", name) }
      let magnitude = try rounded(dot(lanes, lanes).squareRoot())
      values = try lanes.map { try rounded($0 / magnitude) }
    case .faceForward:
      values = try dot(operands[1], operands[2]) < 0 ? a : a.map { -$0 }
    case .reflect:
      let b = operands[1]
      let factor = try rounded(2 * dot(b, a))
      values = try zip(a, b).map { try rounded($0 - rounded(factor * $1)) }
    case .refract:
      let b = operands[1]
      let eta = operands[2][0]
      let projection = try dot(b, a)
      let transverse = try rounded(1 - rounded(projection * projection))
      let k = try rounded(1 - rounded(rounded(eta * eta) * transverse))
      if k < 0 {
        values = Array(repeating: 0, count: a.count)
      } else {
        let factor = try rounded(rounded(eta * projection) + rounded(k.squareRoot()))
        values = try zip(a, b).map { try rounded(rounded(eta * $0) - rounded(factor * $1)) }
      }
    default: throw ShaderFailure.invalidHandle
    }
    if case .vector(_, let element) = result {
      return .vector(.init(elements: try values.map { try floating($0, kind: kind, at: name) }, element: element))
    }
    return try floating(values[0], kind: kind, at: name)
  }
}
