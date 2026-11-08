import SAGAShaderIR

extension WGSLParser {
  func overrideDeclaration(_ attrs: [String: Attribute]) throws {
    try checkAttributes(attrs, allowed: ["id"])
    let name = try ident()
    try reserve(name)
    let id: UInt16?
    if let attribute = attrs["id"] {
      guard let value = UInt16(exactly: try attributeUInt(attribute)) else {
        throw wgslError(.invalidBinding, "Override ID must be in 0...65535", attribute.name)
      }
      id = value
    } else {
      id = nil
    }
    let annotation = try eat(":") ? try type() : nil
    if let annotation {
      guard case .scalar = types[annotation.index].kind else {
        throw wgslError(.typeMismatch, "Override type must be scalar", name)
      }
    }
    let type: WType
    let initializer: WExpr?
    if try eat("=") {
      let tokens = try constantExpressionTokens(at: name)
      if try containsOverrideReference(tokens) {
        initializer = try lowerPipelineScalar(tokens, annotation: annotation, at: name)
        guard let initializer else { throw ShaderFailure.invalidHandle }
        type = globalExpressions[initializer.index].type
      } else {
        let initial = try scalarConstant(tokens, at: name)
        switch initial.kind {
        case .integer, .floating, .boolean: break
        default: throw wgslError(.typeMismatch, "Override initializer must be scalar", name)
        }
        let scalar = try initial.scalar
        let concrete: IRScalar = scalar == .abstractInt ? .i32 : scalar == .abstractFloat ? .f32 : scalar
        type = try annotation ?? intern(.scalar(concrete))
        let value = try annotatedConstant(initial, type: type, at: name)
        try node()
        initializer = try WExpr(index: globalExpressions.count)
        globalExpressions.append(.init(type: type, kind: .literal(try value.literal), location: name.location))
      }
    } else {
      guard let annotation else {
        throw wgslError(.typeMismatch, "Override without a default requires a type", name)
      }
      try expect(";")
      type = annotation
      initializer = nil
    }
    try node()
    overrideNames[wgslKey(name.text)] = try IRHandle(index: overrides.count)
    overrides.append(.init(name: name.text, id: id, type: type, initializer: initializer, location: name.location))
  }

  private func containsOverrideReference(_ expression: [WGSLToken], respectLocalScopes: Bool = false) throws -> Bool {
    for (index, token) in expression.enumerated() {
      try budget.consume(.work)
      if token.kind == .word, index == 0 || expression[index - 1].text != ".",
         let declaration = declarationNames[wgslKey(token.text)], declarations[declaration].kind == "override" {
        if respectLocalScopes, try localConstantNameExists(token) { continue }
        return true
      }
    }
    return false
  }

  func overrideArrayCount(_ expression: [WGSLToken], at token: WGSLToken) throws -> IRHandle<IROverride>? {
    guard try containsOverrideReference(expression, respectLocalScopes: true) else { return nil }
    let root = try lowerPipelineScalar(expression, annotation: nil, at: token)
    let value = globalExpressions[root.index]
    let kind = types[value.type.index].kind
    guard kind == .scalar(.i32) || kind == .scalar(.u32) else {
      throw wgslError(.typeMismatch, "Array override count must be i32 or u32", token)
    }
    if expression.count == 1, case .override(let named) = value.kind { return named }

    // Non-identifier counts have their own type identity, even when they reference one override.
    try node()
    let handle = try IRHandle<IROverride>(index: overrides.count)
    overrides.append(.init(type: value.type, initializer: root, location: token.location))
    return handle
  }

  func lowerPrivateInitializer(_ expression: [WGSLToken], annotation: WType?, at name: WGSLToken) throws -> WExpr {
    if try containsOverrideReference(expression) {
      return try lowerPipelineExpression(expression, annotation: annotation, scalarOnly: false, at: name)
    }
    var constant = try scalarConstant(expression, at: name)
    if let annotation { constant = try annotatedConstant(constant, type: annotation, at: name) }
    let saved = body
    body = WGSLFunctionBody()
    defer { body = saved }
    let initial: WGSLValue
    if case .decomposition(let value) = constant {
      initial = try decompositionRuntimeValue(value, at: name)
    } else {
      initial = try compositeRuntimeValue(constant, at: name)
    }
    let value = try annotation.map { try coerce(initial, to: $0, at: name) } ?? defaultMaterialize(initial, at: name)
    return try exportGlobalExpression(value, at: name)
  }

  private func lowerPipelineScalar(_ expression: [WGSLToken], annotation: WType?, at name: WGSLToken) throws -> WExpr {
    try lowerPipelineExpression(expression, annotation: annotation, scalarOnly: true, at: name)
  }

  private func lowerPipelineExpression(
    _ expression: [WGSLToken], annotation: WType?, scalarOnly: Bool, at name: WGSLToken
  ) throws -> WExpr {
    guard let start = expression.first?.start else { throw wgslError(.unexpectedToken, "Missing override expression", name) }
    var low = 0, high = tokens.count
    while low < high {
      try budget.consume(.work)
      let mid = low + (high - low) / 2
      if tokens[mid].start < start { low = mid + 1 } else { high = mid }
    }
    guard tokens.indices.contains(low), tokens[low].start == start else { throw ShaderFailure.invalidHandle }
    let end = low + expression.count
    let saved = (cursor, body, pipelineExpressionEnd)
    cursor = low
    body = WGSLFunctionBody()
    pipelineExpressionEnd = end
    defer { (cursor, body, pipelineExpressionEnd) = saved }
    let value = try self.expression()
    guard cursor == end else {
      throw wgslError(.unexpectedToken, "Unexpected token in override expression", token)
    }
    let root = try annotation.map { try coerce(value, to: $0, at: name) } ?? defaultMaterialize(value, at: name)
    if scalarOnly {
      guard case .scalar = types[root.type.index].kind else {
        throw wgslError(.typeMismatch, "Override initializer or dimension must be scalar", name)
      }
    }
    return try exportGlobalExpression(root, at: name)
  }

  private func exportGlobalExpression(_ root: WGSLValue, at name: WGSLToken) throws -> WExpr {
    var pending = [root.handle]
    var visited = Set(pending)
    while let handle = pending.popLast() {
      try budget.consume(.work)
      let value = body.expressions[handle.index]
      guard types[value.type.index].kind.isPipelineValueType, let operands = value.kind.pipelineOperands else {
        throw wgslError(.unsupportedConstruct, "Pipeline expression requires supported constant-evaluable operations", name)
      }
      for operand in operands where visited.insert(operand).inserted { pending.append(operand) }
    }
    var remap: [WExpr: WExpr] = [:]
    for handle in visited.sorted(by: { $0.index < $1.index }) {
      try node()
      let value = body.expressions[handle.index]
      guard let kind = try value.kind.remappingPipelineOperands({ operand in
        guard let mapped = remap[operand] else { throw ShaderFailure.invalidHandle }
        return mapped
      }) else { throw ShaderFailure.invalidHandle }
      remap[handle] = try WExpr(index: globalExpressions.count)
      globalExpressions.append(.init(type: value.type, kind: kind, location: value.location))
    }
    guard let result = remap[root.handle] else { throw ShaderFailure.invalidHandle }
    return result
  }

  func subgroupSizeConfiguration(_ attribute: Attribute) throws -> IRSubgroupSize {
    guard attribute.expressions.count == 1, let expression = attribute.expressions.first,
          let first = expression.first else {
      throw wgslError(.unexpectedToken, "subgroup_size requires one size", attribute.name)
    }
    if try containsOverrideReference(expression) {
      let root = try lowerPipelineScalar(expression, annotation: nil, at: first)
      let type = types[globalExpressions[root.index].type.index].kind
      guard type == .scalar(.i32) || type == .scalar(.u32) else {
        throw wgslError(.typeMismatch, "Subgroup size override must be i32 or u32", first)
      }
      return .overrideExpression(root)
    }
    let value = try attributeValue(integerConstant(expression, at: first), at: first)
    guard value.nonzeroBitCount == 1 else {
      throw wgslError(.invalidBinding, "Subgroup size must be a power of two", first)
    }
    return .fixed(value)
  }

  func workgroupConfiguration(_ attribute: Attribute) throws -> (IRWorkgroupSize, IRWorkgroupOverrides?) {
    var dimensions: [(value: WGSLIntegerConstant, expression: WExpr?)] = []
    for tokens in attribute.expressions {
      if try containsOverrideReference(tokens) {
        let root = try lowerPipelineScalar(tokens, annotation: nil, at: tokens[0])
        let kind = types[globalExpressions[root.index].type.index].kind
        guard kind == .scalar(.i32) || kind == .scalar(.u32) else {
          throw wgslError(.typeMismatch, "Workgroup override must be i32 or u32", tokens[0])
        }
        let value = WGSLIntegerConstant(value: 1, kind: kind == .scalar(.i32) ? .sint : .uint)
        dimensions.append((value, root))
      } else {
        dimensions.append((try integerConstant(tokens, at: attribute.name), nil))
      }
    }
    let concrete = Set(dimensions.map(\.value.kind).filter { $0 != .abstract }.map { $0 == .sint ? "i32" : "u32" })
    guard concrete.count <= 1 else {
      throw wgslError(.typeMismatch, "Workgroup dimensions must share i32/u32 type", attribute.name)
    }
    let context: WGSLIntegerConstant.Kind = concrete.first == "u32" ? .uint : .sint
    let evaluator = WGSLIntegerEvaluator(parser: self, tokens: [], fallback: attribute.name)
    var fixed = [UInt32](repeating: 1, count: 3)
    var expressions = [WExpr?](repeating: nil, count: 3)
    for (index, dimension) in dimensions.enumerated() {
      let token = attribute.expressions[index][0]
      let value = dimension.value
      let typed = value.kind == .abstract ? try evaluator.checked(value.value, context, at: token) : value
      fixed[index] = try attributeValue(typed, at: token)
      guard fixed[index] > 0 else { throw wgslError(.invalidBinding, "Zero workgroup dimension", token) }
      expressions[index] = dimension.expression
    }
    let size = IRWorkgroupSize(x: fixed[0], y: fixed[1], z: fixed[2])
    let deferred = expressions.contains { $0 != nil }
      ? IRWorkgroupOverrides(x: expressions[0], y: expressions[1], z: expressions[2]) : nil
    return (size, deferred)
  }
}
