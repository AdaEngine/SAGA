import SAGAShaderIR

extension WGSLConstKind {
  var numericLeaf: WGSLConstKind {
    if case .matrix(_, _, let s) = self { return s }
    return scalarLeaf
  }
  func replacingLeaf(_ s: WGSLConstKind) -> WGSLConstKind {
    switch self {
    case .matrix(let c, let r, _): return .matrix(c, r, s)
    case .vector(let n, _): return .vector(n, s)
    default: return s
    }
  }
}

extension WGSLScalarEvaluator {
  mutating func matrixBinaryType(_ op: String, _ lhs: Int, _ rhs: Int, at t: WGSLToken) throws -> WGSLConstKind {
    let a = nodes[lhs].kind
    let b = nodes[rhs].kind
    let leaf = try common(a.numericLeaf, b.numericLeaf, at: t)
    guard leaf == .floating(.abstract) || leaf == .floating(.f32),
      canAuto(a.numericLeaf, to: leaf), canAuto(b.numericLeaf, to: leaf)
    else { throw wgslError(.typeMismatch, "Matrix operators require a common floating scalar", t) }
    let shape: WGSLConstKind
    switch (op, a, b) {
    case ("+", .matrix(let c, let r, _), .matrix(let d, let s, _)),
         ("-", .matrix(let c, let r, _), .matrix(let d, let s, _)):
      guard c == d && r == s else { throw wgslError(.typeMismatch, "Matrix addition shape mismatch", t) }
      shape = a
    case ("*", .matrix(let c, let r, _), .matrix(let d, let s, _)) where c == s:
      shape = .matrix(d, r, leaf)
    case ("*", .matrix(let c, let r, _), .vector(let n, _)) where c == n:
      shape = .vector(r, leaf)
    case ("*", .vector(let n, _), .matrix(let c, let r, _)) where n == r:
      shape = .vector(c, leaf)
    case ("*", .matrix, .floating), ("*", .matrix, .integer): shape = a
    case ("*", .floating, .matrix), ("*", .integer, .matrix): shape = b
    default: throw wgslError(.typeMismatch, "No matrix overload for operator and operand shapes", t)
    }
    for arg in [lhs, rhs] {
      let target = nodes[arg].kind.replacingLeaf(leaf)
      if nodes[arg].kind != target { conversions.append((arg, target, t)) }
    }
    return shape.replacingLeaf(leaf)
  }

  func matrixBinaryValue(_ op: String, _ a: WGSLScalarConstant, _ b: WGSLScalarConstant,
                         result: WGSLConstKind, at t: WGSLToken) throws -> WGSLScalarConstant {
    let leaf = result.numericLeaf
    let left = try converted(a, to: a.kind.replacingLeaf(leaf), at: t)
    let right = try converted(b, to: b.kind.replacingLeaf(leaf), at: t)
    func product(_ xs: [WGSLScalarConstant], _ ys: [WGSLScalarConstant]) throws -> WGSLScalarConstant {
      guard xs.count == ys.count, (2...4).contains(xs.count) else { throw ShaderFailure.invalidHandle }
      var sum: WGSLScalarConstant?
      for (x, y) in zip(xs, ys) {
        try parser.budget.consume(.work)
        let term = try scalarBinaryValue("*", x, y, at: t)
        sum = try sum.map { try scalarBinaryValue("+", $0, term, at: t) } ?? term
      }
      guard let sum else { throw ShaderFailure.invalidHandle }
      return sum
    }
    switch (left, right) {
    case (.matrix(let m), .matrix(let n)):
      if op == "+" || op == "-" {
        var columns: [WGSLVectorConstant] = []
        for (x, y) in zip(m.columns, n.columns) {
          guard case .vector(let v) = try vectorBinaryValue(op, .vector(x), .vector(y), result: .vector(m.rows, leaf), at: t)
          else { throw ShaderFailure.invalidHandle }
          columns.append(v)
        }
        return .matrix(.init(columns: columns, rows: m.rows, element: leaf))
      }
      var columns: [WGSLVectorConstant] = []
      for column in n.columns {
        var xs: [WGSLScalarConstant] = []
        for row in 0..<m.rows {
          try parser.budget.consume(.work)
          xs.append(try product(m.columns.map { $0.elements[row] }, column.elements))
        }
        columns.append(.init(elements: xs, element: leaf))
      }
      return .matrix(.init(columns: columns, rows: m.rows, element: leaf))
    case (.matrix(let m), .vector(let v)):
      var xs: [WGSLScalarConstant] = []
      for row in 0..<m.rows {
        try parser.budget.consume(.work)
        xs.append(try product(m.columns.map { $0.elements[row] }, v.elements))
      }
      return .vector(.init(elements: xs, element: leaf))
    case (.vector(let v), .matrix(let m)):
      var xs: [WGSLScalarConstant] = []
      for column in m.columns {
        try parser.budget.consume(.work)
        xs.append(try product(v.elements, column.elements))
      }
      return .vector(.init(elements: xs, element: leaf))
    case (.matrix(let m), let scalar), (let scalar, .matrix(let m)):
      var columns: [WGSLVectorConstant] = []
      for column in m.columns {
        var xs: [WGSLScalarConstant] = []
        for value in column.elements {
          try parser.budget.consume(.work)
          xs.append(try scalarBinaryValue("*", value, scalar, at: t))
        }
        columns.append(.init(elements: xs, element: leaf))
      }
      return .matrix(.init(columns: columns, rows: m.rows, element: leaf))
    default: throw ShaderFailure.invalidHandle
    }
  }
}

extension WGSLParser {
  func matrixBinary(_ op: IRBinaryOperator, left: WGSLValue, right: WGSLValue, at t: WGSLToken) throws -> WGSLValue? {
    let a = types[left.type.index].kind
    let b = types[right.type.index].kind
    func hasMatrix(_ k: IRTypeKind) -> Bool { if case .matrix = k { return true }; return false }
    guard hasMatrix(a) || hasMatrix(b) else { return nil }
    func scalar(_ k: IRTypeKind) throws -> IRScalar {
      switch k {
      case .matrix(_, _, let s), .vector(_, let s), .scalar(let s): return s
      default: throw wgslError(.typeMismatch, "Matrix operand must be scalar/vector/matrix", t)
      }
    }
    let x = try scalar(a)
    let y = try scalar(b)
    let s: IRScalar = x == .f32 || y == .f32 ? .f32 : .abstractFloat
    guard [IRScalar.f32, .abstractFloat, .abstractInt].contains(x),
      [IRScalar.f32, .abstractFloat, .abstractInt].contains(y)
    else { throw wgslError(.typeMismatch, "Matrix arithmetic requires floating operands", t) }
    func replace(_ k: IRTypeKind) -> IRTypeKind {
      switch k {
      case .matrix(let c, let r, _): return .matrix(columns: c, rows: r, scalar: s)
      case .vector(let n, _): return .vector(size: n, scalar: s)
      default: return .scalar(s)
      }
    }
    let result: IRTypeKind
    switch (op, a, b) {
    case (.add, .matrix(let c, let r, _), .matrix(let d, let q, _)),
         (.subtract, .matrix(let c, let r, _), .matrix(let d, let q, _)):
      guard c == d && r == q else { throw wgslError(.typeMismatch, "Matrix addition shape mismatch", t) }
      result = replace(a)
    case (.multiply, .matrix(let c, let r, _), .matrix(let d, let q, _)) where c == q:
      result = .matrix(columns: d, rows: r, scalar: s)
    case (.multiply, .matrix(let c, let r, _), .vector(let n, _)) where c == n:
      result = .vector(size: r, scalar: s)
    case (.multiply, .vector(let n, _), .matrix(let c, let r, _)) where n == r:
      result = .vector(size: c, scalar: s)
    case (.multiply, .matrix, .scalar): result = replace(a)
    case (.multiply, .scalar, .matrix): result = replace(b)
    default: throw wgslError(.typeMismatch, "Invalid matrix arithmetic overload", t)
    }
    if let l = try pureCompositeConstant(left, at:t), let r = try pureCompositeConstant(right, at:t) {
      let evaluator = WGSLScalarEvaluator(parser:self,tokens:[],fallback:t)
      let kind: WGSLConstKind
      let leaf: WGSLConstKind = s == .f32 ? .floating(.f32) : .floating(.abstract)
      switch result {
      case .matrix(let c,let r,_):kind = .matrix(Int(c.rawValue),Int(r.rawValue),leaf)
      case .vector(let n,_):kind = .vector(Int(n.rawValue),leaf)
      default:throw ShaderFailure.invalidHandle
      }
      let value = try evaluator.matrixBinaryValue(op == .add ? "+" : op == .subtract ? "-" : "*",l,r,result:kind,at:t)
      switch value {
      case .matrix(let m):return try matrixRuntimeValue(m,to:nil,at:t)
      case .vector(let v):return try vectorRuntimeValue(v,to:nil,at:t)
      default:throw ShaderFailure.invalidHandle
      }
    }
    let l = try coerce(left, to: intern(replace(a)), at: t)
    let r = try coerce(right, to: intern(replace(b)), at: t)
    return try expressionNode(.binary(op: op, left: l.handle, right: r.handle), type: intern(result), at: t)
  }
}


extension WGSLParser {
  func inferredComposite(_ t:WGSLToken) throws -> WGSLValue {
    let args = try arguments()
    let matrix = WGSLScalarEvaluator.matrixNames[t.text]
    var leaf: IRScalar = matrix == nil ? .abstractInt : .abstractFloat
    var concrete: IRScalar?
    for arg in args {
      try budget.consume(.work)
      let scalar:IRScalar
      switch types[arg.type.index].kind {
      case .scalar(let s), .vector(_,let s), .matrix(_,_,let s):scalar = s
      default:throw wgslError(.typeMismatch,"Inferred constructor requires numeric/scalar components",t)
      }
      if scalar == .abstractFloat {leaf = .abstractFloat}
      else if scalar != .abstractInt {
        if let concrete, concrete != scalar {throw wgslError(.typeMismatch,"Inferred constructor concrete types disagree",t)}
        concrete = scalar
      }
    }
    leaf = concrete ?? leaf
    let kind:IRTypeKind
    if let (c,r) = matrix {
      guard leaf == .f32 || leaf == .abstractFloat else {throw wgslError(.typeMismatch,"Matrix scalar must be floating",t)}
      guard let columns = IRVectorSize(rawValue:UInt8(c)),let rows = IRVectorSize(rawValue:UInt8(r)) else {throw ShaderFailure.invalidHandle}
      kind = .matrix(columns:columns,rows:rows,scalar:leaf)
    } else {
      guard let digit = t.text.last?.wholeNumberValue,let width = IRVectorSize(rawValue:UInt8(digit)) else {throw ShaderFailure.invalidHandle}
      kind = .vector(size:width,scalar:leaf)
    }
    return try construct(intern(kind),at:t,supplied:args)
  }
  func pureCompositeConstant(_ v:WGSLValue,at t:WGSLToken) throws -> WGSLScalarConstant? {
    try budget.consume(.work)
    func scalar(_ l:IRLiteral) -> WGSLScalarConstant? {
      switch l {
      case .abstractInt(let n):return .integer(.init(value:n,kind:.abstract))
      case .i32(let n):return .integer(.init(value:Int64(n),kind:.sint))
      case .u32(let n):return .integer(.init(value:Int64(n),kind:.uint))
      case .abstractFloat(let x):return .floating(.init(value:x,kind:.abstract))
      case .f32(let x):return .floating(.init(value:Double(x),kind:.f32))
      case .bool(let x):return .boolean(x)
      default:return nil
      }
    }
    func vector(_ xs:[IRLiteral],_ s:IRScalar) throws -> WGSLVectorConstant? {
      var values:[WGSLScalarConstant] = []
      for x in xs {try budget.consume(.work);guard let value = scalar(x) else {return nil};values.append(value)}
      let kind:WGSLConstKind = s == .abstractFloat ? .floating(.abstract) : s == .abstractInt ? .integer(.abstract) : try .scalar(s,at:t)
      return .init(elements:values,element:kind)
    }
    switch types[v.type.index].kind {
    case .scalar:
      guard let x = try abstractScalarLeaf(v.handle,at:t) else {return nil}
      return scalar(x)
    case .vector(_,let s):
      guard let xs = try abstractVectorLeaves(v.handle,at:t),let x = try vector(xs,s) else {return nil}
      return .vector(x)
    case .matrix(let c,let r,let s):
      var columns:[WGSLVectorConstant] = []
      for i in 0..<Int(c.rawValue) {
        guard let xs = try abstractMatrixColumn(v.handle,index:Int64(i),at:t),let column = try vector(xs,s) else {return nil}
        columns.append(column)
      }
      return .matrix(.init(columns:columns,rows:Int(r.rawValue),element:s == .abstractFloat ? .floating(.abstract) : try .scalar(s,at:t)))
    default:return nil
    }
  }
}
