import SAGAShaderIR

// Bounded frontend matrix value: at most4 columns of at most4 scalar leaves.
struct WGSLMatrixConstant {
  let columns: [WGSLVectorConstant]
  let rows: Int
  let element: WGSLConstKind
}
extension WGSLConstKind {
  var isMatrix: Bool {
    if case .matrix = self { return true }
    return false
  }
}
extension WGSLScalarEvaluator {
  static let matrixNames: [String: (Int, Int)] = [
    "mat2x2": (2, 2), "mat2x3": (2, 3), "mat2x4": (2, 4), "mat3x2": (3, 2), "mat3x3": (3, 3),
    "mat3x4": (3, 4), "mat4x2": (4, 2), "mat4x3": (4, 3), "mat4x4": (4, 4),
  ]

  func matrixBuiltinConstant(_ function: IRMathFunction, matrix: WGSLMatrixConstant, at name: WGSLToken) throws
    -> WGSLScalarConstant
  {
    if function == .transpose {
      let columns = try (0..<matrix.rows).map { row in
        try parser.budget.consume(.work, amount: matrix.columns.count)
        return WGSLVectorConstant(elements: matrix.columns.map { $0.elements[row] }, element: matrix.element)
      }
      return .matrix(.init(columns: columns, rows: matrix.columns.count, element: matrix.element))
    }
    guard function == .determinant, matrix.columns.count == matrix.rows,
          case .floating(let kind) = matrix.element else { throw ShaderFailure.invalidHandle }
    let values = try matrix.columns.map { column in
      try column.elements.map { element -> Double in
        guard case .floating(let scalar) = element else { throw ShaderFailure.invalidHandle }
        return scalar.value
      }
    }
    // At most four columns. Expanding along a column also handles singular matrices without division.
    func minor(column: Int, rows: [Int]) throws -> Double {
      try parser.budget.consume(.work)
      if rows.count == 1 { return values[column][rows[0]] }
      var sum = 0.0
      for (index, row) in rows.enumerated() {
        var remaining = rows
        remaining.remove(at: index)
        let term = try values[column][row] * minor(column: column + 1, rows: remaining)
        sum += index.isMultiple(of: 2) ? term : -term
      }
      return sum
    }
    return try floating(minor(column: 0, rows: Array(0..<matrix.rows)), kind: kind, at: name)
  }

  mutating func matrixConstructor(_ t: WGSLToken, columns: Int, rows: Int, target: WGSLConstKind?)
    throws -> Int
  {
    guard index < tokens.count, try take().text == "(" else {
      throw wgslError(.unexpectedToken, "Matrix constructor requires parentheses", t)
    }
    var args: [Int] = []
    while index < tokens.count, token.text != ")" {
      guard args.count < 16 else {
        throw wgslError(.typeMismatch, "Matrix argument count exceeds16", t)
      }
      args.append(try parse())
      if index < tokens.count, token.text == "," { _ = try take() } else { break }
    }
    guard index < tokens.count, try take().text == ")" else {
      throw wgslError(.unexpectedToken, "Unterminated matrix constructor", t)
    }
    let conversion = args.count == 1 && nodes[args[0]].kind.isMatrix
    var leaf: WGSLConstKind = target ?? .floating(.abstract)
    if conversion {
      guard case .matrix(let c, let r, let k) = nodes[args[0]].kind, c == columns, r == rows else {
        throw wgslError(.typeMismatch, "Matrix conversion shape mismatch", t)
      }
      leaf = target ?? k
    } else if !args.isEmpty {
      let columnForm = args.count == columns && args.allSatisfy { nodes[$0].kind.lanes == rows }
      let scalarForm =
        args.count == columns * rows
        && args.allSatisfy { nodes[$0].kind.lanes == 1 && !nodes[$0].kind.isMatrix }
      guard columnForm || scalarForm else {
        throw wgslError(
          .typeMismatch, "Matrix constructor requires exact columns or scalar component count", t)
      }
      if target == nil {
        leaf = nodes[args[0]].kind.scalarLeaf
        for a in args.dropFirst() {
          try parser.budget.consume(.work)
          leaf = try common(leaf, nodes[a].kind.scalarLeaf, at: t)
        }
        if leaf == .integer(.abstract) { leaf = .floating(.abstract) }
      }
      guard case .floating = leaf else {
        throw wgslError(.typeMismatch, "Matrix components must be floating point", t)
      }
      for a in args {
        guard canAuto(nodes[a].kind.scalarLeaf, to: leaf) else {
          throw wgslError(
            .typeMismatch, "Matrix component needs implicit floating conversion", nodes[a].token)
        }
        let expected = shaped(leaf, like: nodes[a].kind)
        if expected != nodes[a].kind { conversions.append((a, expected, t)) }
      }
    }
    guard case .floating = leaf else {
      throw wgslError(.typeMismatch, "Matrix scalar must be floating", t)
    }
    return try append(
      .matrix(columns, rows, leaf, args, conversion), .matrix(columns, rows, leaf), t)
  }
  mutating func matrixValue(
    _ c: Int, _ r: Int, _ leaf: WGSLConstKind, _ args: [Int], conversion: Bool, at t: WGSLToken
  ) throws -> WGSLScalarConstant {
    var values: [WGSLScalarConstant] = []
    for arg in args {
      try parser.budget.consume(.work)
      switch try value(arg) {
      case .matrix(let m):
        for column in m.columns {
          try parser.budget.consume(.work)
          values.append(contentsOf: column.elements)
        }
      case .vector(let v): values.append(contentsOf: v.elements)
      case let scalar: values.append(scalar)
      }
      guard values.count <= 16 else {
        throw wgslError(.invalidType, "Matrix component overflow", t)
      }
    }
    if args.isEmpty { values = Array(repeating: try zero(leaf, at: t), count: c * r) }
    guard values.count == c * r else {
      throw wgslError(.typeMismatch, "Matrix value count mismatch", t)
    }
    var columns: [WGSLVectorConstant] = []
    for i in 0..<c {
      var xs: [WGSLScalarConstant] = []
      for j in 0..<r {
        try parser.budget.consume(.work)
        xs.append(try scalarConvert(values[i * r + j], to: leaf, at: t, explicit: conversion))
      }
      columns.append(.init(elements: xs, element: leaf))
    }
    return .matrix(.init(columns: columns, rows: r, element: leaf))
  }
}

extension WGSLParser {
  func matrixRuntimeValue(_ matrix: WGSLMatrixConstant, to target: WType?, at t: WGSLToken) throws
    -> WGSLValue
  {
    guard let c = IRVectorSize(rawValue: UInt8(matrix.columns.count)),
      let r = IRVectorSize(rawValue: UInt8(matrix.rows))
    else { throw wgslError(.invalidType, "Matrix runtime shape", t) }
    let leaf: WGSLConstKind
    if let target {
      guard case .matrix(let cols, let rows, let scalar) = types[target.index].kind, cols == c,
        rows == r
      else { throw wgslError(.typeMismatch, "Matrix runtime context shape", t) }
      leaf = try .scalar(scalar, at: t)
    } else {
      leaf = matrix.element
    }
    var columns: [WExpr] = []
    let columnType = try intern(.vector(size: r, scalar: leaf.irScalar(at: t)))
    for column in matrix.columns {
      try budget.consume(.work)
      columns.append(
        try vectorRuntimeValue(column, to: leaf == .floating(.abstract) ? nil : columnType, at: t)
          .handle)
    }
    let matrixType =
      try target ?? intern(.matrix(columns: c, rows: r, scalar: leaf.irScalar(at: t)))
    return try expressionNode(
      .compose(type: matrixType, components: columns), type: matrixType, at: t)
  }
}
