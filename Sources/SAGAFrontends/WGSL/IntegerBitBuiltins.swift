import SAGAShaderIR

func integerBitArgumentKinds(_ arguments: [WGSLConstKind], arity: Int, at name: WGSLToken) throws -> [WGSLConstKind] {
  guard arguments.count == arity, let first = arguments.first, case .integer(var scalar) = first.scalarLeaf else {
    throw wgslError(.typeMismatch, "Bit builtin requires integer operands and matching arity", name)
  }
  let valueCount = arity == 4 ? 2 : 1
  if valueCount == 2 {
    let second = arguments[1]
    guard first.lanes == second.lanes, case .integer(let other) = second.scalarLeaf,
          scalar == other || scalar == .abstract || other == .abstract else {
      throw wgslError(.typeMismatch, "Bit insertion values must have a common integer type", name)
    }
    if scalar == .abstract { scalar = other }
  }
  if scalar == .abstract { scalar = .sint }
  let leaf = WGSLConstKind.integer(scalar)
  let result = first.lanes == 1 ? leaf : .vector(first.lanes, leaf)
  let bounds = arguments.dropFirst(valueCount)
  guard bounds.allSatisfy({ $0 == .integer(.uint) || $0 == .integer(.abstract) }) else {
    throw wgslError(.typeMismatch, "Bit range requires scalar u32 offset and count", name)
  }
  return Array(repeating: result, count: valueCount) + bounds.map { _ in .integer(.uint) }
}

func validateIntegerBitRange(_ offset: Int64, _ count: Int64, width: Int64, at name: WGSLToken) throws {
  guard (0...width).contains(offset), (0...(width - offset)).contains(count) else {
    throw wgslError(.invalidType, "Constant bit range exceeds \(width) bits", name)
  }
}

extension WGSLParser {
  func integerBitBuiltin(_ function: IRMathFunction, arity: Int, at name: WGSLToken) throws -> WGSLValue {
    guard !(try localConstantNameExists(name)) else {
      throw wgslError(.typeMismatch, "Local value shadows bit builtin", name)
    }
    var arguments: [WGSLValue] = []
    var ranges: [Range<Int>] = []
    while !isNext(")") {
      guard arguments.count < arity else { throw wgslError(.typeMismatch, "Bit builtin arity", name) }
      let start = cursor
      arguments.append(try loaded(expression(), at: name))
      ranges.append(start..<cursor)
      if !(try eat(",")) { break }
    }
    try expect(")")
    let kinds = try arguments.map { argument -> WGSLConstKind in
      let scalar: IRScalar
      let size: IRVectorSize?
      switch types[argument.type.index].kind {
      case .scalar(let lane): scalar = lane; size = nil
      case .vector(let width, let lane): scalar = lane; size = width
      default: throw wgslError(.typeMismatch, "Bit builtin requires integer scalars or vectors", name)
      }
      let leaf: WGSLConstKind
      switch scalar {
      case .i16: leaf = .integer(.sint16)
      case .u16: leaf = .integer(.uint16)
      case .i32: leaf = .integer(.sint)
      case .u32: leaf = .integer(.uint)
      case .i64: leaf = .integer(.sint64)
      case .u64: leaf = .integer(.uint64)
      case .abstractInt: leaf = .integer(.abstract)
      default: throw wgslError(.typeMismatch, "Bit builtin requires integer scalars or vectors", name)
      }
      return size.map { .vector(Int($0.rawValue), leaf) } ?? leaf
    }
    let signature = try integerBitArgumentKinds(kinds, arity: arity, at: name)
    if arity > 1,
       case .integer(let offset)? = try knownConstantValue(ranges[arity - 2], at: name),
       case .integer(let count)? = try knownConstantValue(ranges[arity - 1], at: name) {
      let scalar = try signature[0].scalarLeaf.irScalar(at: name)
      try validateIntegerBitRange(offset.value, count.value, width: Int64(scalar.width) * 8, at: name)
    }
    for index in arguments.indices {
      let kind = signature[index]
      let scalar = try kind.scalarLeaf.irScalar(at: name)
      let type = IRVectorSize(rawValue: UInt8(kind.lanes)).map { IRTypeKind.vector(size: $0, scalar: scalar) } ?? .scalar(scalar)
      arguments[index] = try coerce(arguments[index], to: intern(type), at: name)
    }
    let handles = arguments.map(\.handle)
    return try expressionNode(.math(function: function, argument: handles[0],
                                    argument1: arity >= 2 ? handles[1] : nil, argument2: arity >= 3 ? handles[2] : nil,
                                    argument3: arity == 4 ? handles[3] : nil), type: arguments[0].type, at: name)
  }
}

extension WGSLScalarEvaluator {
  func constantIntegerBitBuiltin(_ function: IRMathFunction, arguments: [WGSLScalarConstant], at name: WGSLToken) throws -> WGSLScalarConstant {
    let arity = arguments.count
    let valueCount = arity == 4 ? 2 : 1
    let scalar = try arguments[0].kind.scalarLeaf.irScalar(at: name)
    let width = Int(scalar.width) * 8
    var offset = 0
    var count = 0
    if arity > 1 {
      guard case .integer(let start) = arguments[arity - 2], case .integer(let length) = arguments[arity - 1] else {
        throw ShaderFailure.invalidHandle
      }
      try validateIntegerBitRange(start.value, length.value, width: Int64(width), at: name)
      offset = Int(start.value)
      count = Int(length.value)
    }
    if case .vector(let vector) = arguments[0] {
      var elements: [WGSLScalarConstant] = []
      for index in vector.elements.indices {
        try parser.budget.consume(.work)
        var operands = [vector.elements[index]]
        if valueCount == 2 {
          guard case .vector(let inserted) = arguments[1] else { throw ShaderFailure.invalidHandle }
          operands.append(inserted.elements[index])
        }
        operands += arguments.dropFirst(valueCount)
        elements.append(try constantIntegerBitBuiltin(function, arguments: operands, at: name))
      }
      return .vector(.init(elements: elements, element: vector.element))
    }
    guard case .integer(let value) = arguments[0] else { throw ShaderFailure.invalidHandle }
    let laneMask = UInt64.max >> (64 - width)
    let bits = value.bits & laneMask
    let result: UInt64
    switch function {
    case .countLeadingZeros: result = UInt64(bits.leadingZeroBitCount - (64 - width))
    case .countTrailingZeros: result = UInt64(min(bits.trailingZeroBitCount, width))
    case .countOneBits: result = UInt64(bits.nonzeroBitCount)
    case .reverseBits:
      try parser.budget.consume(.work, amount: width)
      var reversed: UInt64 = 0
      for index in 0..<width { reversed = (reversed << 1) | ((bits >> index) & 1) }
      result = reversed
    case .firstLeadingBit:
      let search = scalar.kind == .sint && value.value < 0 ? bits ^ laneMask : bits
      result = UInt64(truncatingIfNeeded: 63 - search.leadingZeroBitCount)
    case .firstTrailingBit:
      result = bits == 0 ? .max : UInt64(bits.trailingZeroBitCount)
    case .extractBits:
      let mask = UInt64.max >> (64 - count)
      let extracted = (bits >> offset) & mask
      let sign: UInt64 = scalar.kind == .sint && count > 0 ? 1 << (count - 1) : 0
      result = (extracted ^ sign) &- sign
    case .insertBits:
      guard case .integer(let inserted) = arguments[1] else { throw ShaderFailure.invalidHandle }
      let mask = (UInt64.max >> (64 - count)) << offset
      result = (bits & ~mask) | ((inserted.bits << offset) & mask)
    default: throw ShaderFailure.invalidHandle
    }
    let number = value.kind.truncating(Int64(bitPattern: result))
    return .integer(.init(value: number, kind: value.kind))
  }
}
