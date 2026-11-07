import SAGAShaderIR

// Frontend-local lowering boundary, not a permissive raw-IR validator or general transform.
// Pinned Naga lower/mod.rs compacts abstract intermediates before validation; compact/mod.rs
// deliberately does not root abstract named constants. Concrete named declarations remain.
extension WGSLParser {
  func compactAbstractIntermediates() throws {
    var typeMap: [WType?] = []
    var keptTypes: [IRType] = []
    for t in types {
      try budget.consume(.work)
      let abstract: Bool
      if case .matrix(_,_,let s) = t.kind {
        abstract = s.kind == .abstractFloat
      } else if case .vector(_,let s) = t.kind {
        abstract = s.kind == .abstractInt || s.kind == .abstractFloat
      } else if case .scalar(let s) = t.kind {
        abstract = s.kind == .abstractInt || s.kind == .abstractFloat
      } else {
        abstract = false
      }
      typeMap.append(abstract ? nil : try WType(index: keptTypes.count))
      if !abstract { keptTypes.append(t) }
    }
    // No abstract node or dangling abstract-dependent type is permitted to cross the boundary.
    func ty(_ h: WType) throws -> WType {
      try budget.consume(.work)
      guard typeMap.indices.contains(h.index), let new = typeMap[h.index] else {
        throw ShaderFailure.unsupported("Unmaterialized abstract type at WGSL lowering boundary")
      }
      return new
    }
    for i in keptTypes.indices {
      try budget.consume(.work)
      let t = keptTypes[i]
      let kind: IRTypeKind
      switch t.kind {
      case .pointer(let b, let s): kind = .pointer(base: try ty(b), space: s)
      case .array(let b, let n, let s): kind = .array(base: try ty(b), size: n, stride: s)
      case .structure(let ms, let span):
        kind = .structure(
          members: try ms.map { m in
            try budget.consume(.work)
            return .init(
              name: m.name, type: try ty(m.type), binding: m.binding, offset: m.offset,
              location: m.location)
          }, span: span)
      case .scalar, .vector, .matrix, .atomic, .image, .sampler: kind = t.kind
      default:
        throw ShaderFailure.unsupported(
          "WGSL materialization type remapping family not implemented")
      }
      keptTypes[i] = IRType(name: t.name, kind: kind, location: t.location)
    }
    // Global expressions are concrete literals in the currently supported const profile.
    for i in globalExpressions.indices {
      try budget.consume(.work)
      let e = globalExpressions[i]
      guard case .literal = e.kind else {
        throw ShaderFailure.unsupported("Nonliteral global initializer compaction not implemented")
      }
      globalExpressions[i] = .init(type: try ty(e.type), kind: e.kind, location: e.location)
    }
    for i in constants.indices {
      try budget.consume(.work)
      let c = constants[i]
      constants[i] = .init(
        name: c.name, type: try ty(c.type), initializer: c.initializer, location: c.location)
    }
    for i in globals.indices {
      try budget.consume(.work)
      let g = globals[i]
      globals[i] = .init(
        name: g.name, space: g.space, binding: g.binding, type: try ty(g.type),
        initializer: g.initializer, memoryDecorations: g.memoryDecorations, location: g.location)
    }
    for i in functions.indices {
      try budget.consume(.work)
      let f = functions[i]
      var exprMap: [WExpr?] = []
      var kept: [IRExpression] = []
      for e in f.expressions.values {
        try budget.consume(.work)
        guard typeMap.indices.contains(e.type.index) else { throw ShaderFailure.invalidHandle }
        let keep = typeMap[e.type.index] != nil
        exprMap.append(keep ? try WExpr(index: kept.count) : nil)
        if keep { kept.append(e) }
      }
      func ex(_ h: WExpr) throws -> WExpr {
        try budget.consume(.work)
        guard exprMap.indices.contains(h.index), let new = exprMap[h.index] else {
          throw ShaderFailure.unsupported(
            "Abstract value was not contextually materialized before runtime use")
        }
        return new
      }
      func optional(_ h: WExpr?) throws -> WExpr? { try h.map(ex) }
      for j in kept.indices {
        try budget.consume(.work)
        let e = kept[j]
        let k: IRExpressionKind
        switch e.kind {
        case .literal, .constant, .functionArgument, .globalVariable, .localVariable, .callResult:
          k = e.kind
        case .zeroValue(let t): k = .zeroValue(try ty(t))
        case .compose(let t, let xs): k = .compose(type: try ty(t), components: try xs.map(ex))
        case .access(let b, let x): k = .access(base: try ex(b), index: try ex(x))
        case .accessIndex(let b, let x): k = .accessIndex(base: try ex(b), index: x)
        case .splat(let n, let v): k = .splat(size: n, value: try ex(v))
        case .swizzle(let n, let v, let p): k = .swizzle(size: n, vector: try ex(v), pattern: p)
        case .load(let p): k = .load(pointer: try ex(p))
        case .atomicResult(let t, let comparison): k = .atomicResult(type: try ty(t), comparison: comparison)
        case .workgroupUniformLoadResult(let t): k = .workgroupUniformLoadResult(type: try ty(t))
        case .arrayLength(let p): k = .arrayLength(try ex(p))
        case .imageSample(let sample):
          let level: IRSampleLevel
          switch sample.level {
          case .auto, .zero: level = sample.level
          case .exact(let value): level = .exact(try ex(value))
          case .bias(let value): level = .bias(try ex(value))
          case .gradient(let x, let y): level = .gradient(x: try ex(x), y: try ex(y))
          }
          k = .imageSample(.init(image: try ex(sample.image), sampler: try ex(sample.sampler), gather: sample.gather,
                                 coordinate: try ex(sample.coordinate), arrayIndex: try optional(sample.arrayIndex),
                                 offset: try optional(sample.offset), level: level,
                                 depthReference: try optional(sample.depthReference), clampToEdge: sample.clampToEdge))
        case .imageLoad(let load):
          k = .imageLoad(.init(image: try ex(load.image), coordinate: try ex(load.coordinate),
                               arrayIndex: try optional(load.arrayIndex), sample: try optional(load.sample),
                               level: try optional(load.level)))
        case .imageQuery(let image, let query):
          if case .size(let level) = query {
            k = .imageQuery(image: try ex(image), query: .size(level: try optional(level)))
          } else {
            k = .imageQuery(image: try ex(image), query: query)
          }
        case .unary(let op, let v): k = .unary(op: op, expression: try ex(v))
        case .binary(let op, let a, let b): k = .binary(op: op, left: try ex(a), right: try ex(b))
        case .select(let c, let a, let b):
          k = .select(condition: try ex(c), accept: try ex(a), reject: try ex(b))
        case .derivative(let a, let c, let v):
          k = .derivative(axis: a, control: c, expression: try ex(v))
        case .math(let fn, let a, let b, let c, let d):
          k = .math(
            function: fn, argument: try ex(a), argument1: try optional(b),
            argument2: try optional(c), argument3: try optional(d))
        case .cast(let v, let t): k = .cast(expression: try ex(v), target: t)
        case .bitcast(let v, let t): k = .bitcast(expression: try ex(v), type: try ty(t))
        default: throw ShaderFailure.unsupported("WGSL expression remapping family not implemented")
        }
        kept[j] = .init(type: try ty(e.type), kind: k, location: e.location)
      }
      let statements = try f.statements.values.map { s -> IRStatement in
        try budget.consume(.work)
        let k: IRStatementKind
        switch s.kind {
        case .emit(let xs):
          // Abstract-only intermediate emissions have no effects; any concrete dependent
          // still pointing to one is rejected by ex above, not silently rewritten.
          var mapped: [WExpr] = []
          for x in xs {
            try budget.consume(.work)
            guard exprMap.indices.contains(x.index) else { throw ShaderFailure.invalidHandle }
            if let m = exprMap[x.index] { mapped.append(m) }
          }
          k = .emit(mapped)
        case .block, .break, .continue, .kill, .controlBarrier, .memoryBarrier: k = s.kind
        case .ifCondition(let c, let a, let b):
          k = .ifCondition(condition: try ex(c), accept: a, reject: b)
        case .switchValue(let v, let cs): k = .switchValue(selector: try ex(v), cases: cs)
        case .loop(let b, let c, let x): k = .loop(body: b, continuing: c, breakIf: try optional(x))
        case .returnValue(let x): k = .returnValue(try optional(x))
        case .store(let p, let v): k = .store(pointer: try ex(p), value: try ex(v))
        case .imageStore(let image, let coordinate, let arrayIndex, let value):
          k = .imageStore(image: try ex(image), coordinate: try ex(coordinate),
                          arrayIndex: try optional(arrayIndex), value: try ex(value))
        case .workgroupUniformLoad(let p, let r): k = .workgroupUniformLoad(pointer: try ex(p), result: try ex(r))
        case .atomic(let p, let operation, let v, let r):
          let mapped: IRAtomicFunction
          if case .exchange(let compare) = operation { mapped = .exchange(compare: try optional(compare)) }
          else { mapped = operation }
          k = .atomic(pointer: try ex(p), function: mapped, value: try ex(v), result: try optional(r))
        case .call(let t, let xs, let r):
          k = .call(function: t, arguments: try xs.map(ex), result: try optional(r))
        default: throw ShaderFailure.unsupported("WGSL statement remapping family not implemented")
        }
        return IRStatement(kind: k, location: s.location)
      }
      functions[i] = IRFunction(
        name: f.name,
        arguments: try f.arguments.map { a in
          try budget.consume(.work)
          return .init(name: a.name, type: try ty(a.type), binding: a.binding, location: a.location)
        }, result: try f.result.map { .init(type: try ty($0.type), binding: $0.binding) },
        localVariables: IRTable(
          try f.localVariables.values.map { v in
            try budget.consume(.work)
            return .init(
              name: v.name, type: try ty(v.type), initializer: try optional(v.initializer),
              location: v.location)
          }), expressions: IRTable(kept),
        namedExpressions: try f.namedExpressions.map {
          .init(expression: try ex($0.expression), name: $0.name)
        }, statements: IRTable(statements), blocks: f.blocks, body: f.body,
        diagnosticFilter: f.diagnosticFilter, location: f.location)
    }
    predeclaredTypes = try predeclaredTypes.map { .init(kind: $0.kind, type: try ty($0.type)) }
    types = keptTypes
  }
}
