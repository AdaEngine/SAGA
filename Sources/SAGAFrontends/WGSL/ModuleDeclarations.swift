import SAGAShaderIR

struct WGSLModuleDeclaration {
  let start: Int
  let end: Int
  let name: WGSLToken
  let kind: String
}

extension WGSLParser {
  // Index immutable token ranges; tokens and source locations are never reordered or rewritten.
  func indexModuleDeclarations() throws {
    var i = cursor
    func advance(_ index: inout Int) throws {
      try budget.consume(.work)
      guard index < tokens.count - 1 else {
        throw wgslError(.unexpectedToken, "Incomplete module declaration", tokens[index])
      }
      index += 1
    }
    while tokens[i].kind != .end {
      if tokens[i].text == ";" {
        try advance(&i)
        continue
      }
      let start = i
      while tokens[i].text == "@" {
        try advance(&i)
        try advance(&i)
        if tokens[i].text == "(" {
          var depth = 0
          repeat {
            if tokens[i].text == "(" {
              depth += 1
              guard depth <= min(budget.limits.depth, 128) else {
                throw ShaderFailure.budgetExceeded(.depth)
              }
            }
            if tokens[i].text == ")" { depth -= 1 }
            try advance(&i)
          } while depth > 0
        }
      }
      let kind = tokens[i].text
      if kind == "override" { hasOverrideDeclarations = true }
      guard ["const", "override", "alias", "struct", "var", "fn", "const_assert"].contains(kind) else {
        throw unsupported("module declaration/directive '\(kind)'", at: tokens[i])
      }
      try advance(&i)
      if kind == "var", tokens[i].kind == .templateStart {
        repeat { try advance(&i) } while tokens[i].kind != .templateEnd
        try advance(&i)
      }
      let name = kind == "const_assert" ? tokens[i-1] : tokens[i]
      guard kind == "const_assert" || name.kind == .word else {
        throw wgslError(.unexpectedToken, "Expected declaration name", name)
      }
      try budget.consume(.work, amount: name.text.utf8.count)
      let key = wgslKey(name.text)
      guard kind == "const_assert" || declarationNames[key] == nil else {
        throw wgslError(.duplicateName, "Duplicate module declaration '\(name.text)'", name)
      }
      var brace = 0
      if kind == "struct" || kind == "fn" {
        while tokens[i].text != "{" { try advance(&i) }
        repeat {
          if tokens[i].text == "{" {
            brace += 1
            guard brace <= min(budget.limits.depth, 128) else {
              throw ShaderFailure.budgetExceeded(.depth)
            }
          }
          if tokens[i].text == "}" { brace -= 1 }
          try advance(&i)
        } while brace > 0
      } else {
        while tokens[i].text != ";" { try advance(&i) }
        try advance(&i)
      }
      try node()
      if kind != "const_assert" { declarationNames[key] = declarations.count }
      declarations.append(.init(start: start, end: i, name: name, kind: kind))
      declarationStates.append(0)
      declarationEdges.append([])
    }
  }
  func resolveDeclaration(_ index: Int, at use: WGSLToken) throws {
    try budget.consume(.work)
    if let parent = declarationStack.last {
      try budget.consume(.work)
      if declarationEdges[parent].insert(index).inserted { try node() }
    }
    if declarationStates[index] == 2 { return }
    guard declarationStates[index] != 1 else {
      throw wgslError(.invalidReference, "Cyclic module declaration dependency", use)
    }
    // Separate, conservative recursive resolver cap, preserving the caller's lower limit.
    guard dependencyDepth < min(budget.limits.depth, 24) else {
      throw ShaderFailure.budgetExceeded(.depth)
    }
    dependencyDepth += 1
    let saved = cursor
    let savedBody = body
    let savedPipelineEnd = pipelineExpressionEnd
    pipelineExpressionEnd = nil
    declarationStack.append(index)
    defer {
      cursor = saved
      body = savedBody
      pipelineExpressionEnd = savedPipelineEnd
      dependencyDepth -= 1
      declarationStack.removeLast()
    }
    declarationStates[index] = 1
    let d = declarations[index]
    // Resolve const initializer edges before invoking the evaluator, rather than suspending
    // one recursive expression evaluator per declaration in a multi-hop graph.
    if d.kind == "const" {
      var initializer = false
      for i in d.start..<d.end {
        try budget.consume(.work)
        let token = tokens[i]
        if token.text == "=" {
          initializer = true
          continue
        }
        if initializer, token.kind == .word, i == d.start || tokens[i - 1].text != ".",
          let dependency = declarationNames[wgslKey(token.text)],
          declarations[dependency].kind == "const"
        {
          try resolveDeclaration(dependency, at: token)
        }
      }
    }
    cursor = d.start
    if d.kind != "fn" {body = WGSLFunctionBody()}
    try lowerModuleDeclaration()
    guard cursor == d.end else {
      throw wgslError(
        .unexpectedToken, "Declaration did not consume its indexed token range", token)
    }
    if let lines = sourceDocumentation.beforeToken[tokens[d.start].start] {
      try recordDeclarationDocumentation(d, lines: lines)
    }
    declarationStates[index] = 2
  }

  private func recordDeclarationDocumentation(_ declaration: WGSLModuleDeclaration, lines: [String]) throws {
    let name = wgslKey(declaration.name.text)
    switch declaration.kind {
    case "struct":
      guard let type = typeNames[name] else { throw ShaderFailure.invalidHandle }
      try recordDocumentation(lines, for: .type(type))
    case "fn":
      guard let index = functionNames[name] else { throw ShaderFailure.invalidHandle }
      if let entryIndex = entries.indices.last, entries[entryIndex].function.index == index {
        try recordDocumentation(lines, for: .entryPoint(UInt32(entryIndex)))
      } else {
        try recordDocumentation(lines, for: .function(.init(index: index)))
      }
    case "const":
      if let index = constantNames[name] { try recordDocumentation(lines, for: .constant(.init(index: index))) }
    case "var":
      guard let index = globalNames[name] else { throw ShaderFailure.invalidHandle }
      try recordDocumentation(lines, for: .globalVariable(.init(index: index)))
    default: break
    }
  }
  func resolveTypeDeclarationIfIndexed(_ name: WGSLToken) throws {
    if declarationNames[wgslKey(name.text)] != nil { try resolveTypeDeclaration(name) }
  }
  func resolveTypeDeclaration(_ name: WGSLToken) throws {
    try budget.consume(.work, amount: name.text.utf8.count)
    if let index = declarationNames[wgslKey(name.text)] {
      guard ["alias", "struct"].contains(declarations[index].kind) else {
        throw wgslError(.typeMismatch, "Name is not a type", name)
      }
      try resolveDeclaration(index, at: name)
    }
  }
  func resolveConstantDeclaration(_ name: WGSLToken) throws {
    try budget.consume(.work, amount: name.text.utf8.count)
    if let index = declarationNames[wgslKey(name.text)] {
      guard declarations[index].kind == "const" else {
        throw wgslError(.typeMismatch, "Name is not a creation-fixed constant", name)
      }
      try resolveDeclaration(index, at: name)
    }
  }
  func resolveValueDeclaration(_ name: WGSLToken) throws {
    try budget.consume(.work, amount: name.text.utf8.count)
    if let i = declarationNames[wgslKey(name.text)],
      ["const", "override", "var"].contains(declarations[i].kind)
    {
      try resolveDeclaration(i, at: name)
    }
  }
  func resolveFunctionDeclaration(_ name: WGSLToken) throws {
    for scope in body.scopes.reversed() {
      try budget.consume(.work)
      if scope[wgslKey(name.text)] != nil {
        throw wgslError(.typeMismatch, "Local value shadows callable name", name)
      }
    }
    try budget.consume(.work, amount: name.text.utf8.count)
    if let i = declarationNames[wgslKey(name.text)] {
      guard declarations[i].kind == "fn" else {
        throw wgslError(.typeMismatch, "Value/type cannot be called", name)
      }
      try resolveDeclaration(i, at: name)
    }
  }
  func lowerModuleDeclaration() throws {
    let attrs = try attributes()
    if isNext("const_assert") {try constAssertion(attrs);return}
    if try eat("fn") {
      try function(attrs)
      return
    }
    if try eat("struct") {
      guard attrs.isEmpty else { throw unsupported("struct attributes") }
      try structure()
      return
    }
    if try eat("alias") {
      guard attrs.isEmpty else { throw unsupported("alias attributes") }
      let n = try ident()
      try reserve(n)
      try expect("=")
      typeNames[wgslKey(n.text)] = try type()
      try expect(";")
      return
    }
    if try eat("var") {
      try global(attrs)
      return
    }
    if try eat("const") {
      try constant(attrs)
      return
    }
    if try eat("override") {
      try overrideDeclaration(attrs)
      return
    }
    throw unsupported("indexed declaration")
  }
}
