import SAGAShaderIR

// Frontend-local lowering boundary, not a permissive raw-IR validator or general transform.
// Pinned Naga lower/mod.rs compacts abstract intermediates before validation; compact/mod.rs
// deliberately does not root abstract named constants. Concrete named declarations remain.
extension WGSLParser {
  func compactAbstractIntermediates() throws {
    var typeMap: [WType?] = []
    var keptTypes: [IRType] = []
    let abstractStructures = Set(predeclaredTypes.filter { $0.kind.decomposition?.scalar == .abstractFloat }.map(\.type.index))
    for (i, t) in types.enumerated() {
      try budget.consume(.work)
      let abstract: Bool
      if case .matrix(_,_,let s) = t.kind {
        abstract = s.kind == .abstractFloat
      } else if case .vector(_,let s) = t.kind {
        abstract = s.kind == .abstractInt || s.kind == .abstractFloat
      } else if case .scalar(let s) = t.kind {
        abstract = s.kind == .abstractInt || s.kind == .abstractFloat
      } else if case .array(let base, _, _) = t.kind {
        guard typeMap.indices.contains(base.index) else { throw ShaderFailure.invalidHandle }
        abstract = typeMap[base.index] == nil
      } else {
        abstract = abstractStructures.contains(i)
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
      case .bindingArray(let b, let n): kind = .bindingArray(base: try ty(b), size: n)
      case .structure(let ms, let span):
        kind = .structure(
          members: try ms.map { m in
            try budget.consume(.work)
            return .init(
              name: m.name, type: try ty(m.type), binding: m.binding, offset: m.offset,
              location: m.location)
          }, span: span)
      case .scalar, .vector, .matrix, .cooperativeMatrix, .atomic, .image, .sampler, .rayQuery, .accelerationStructure: kind = t.kind
      default:
        throw ShaderFailure.unsupported(
          "WGSL materialization type remapping family not implemented")
      }
      keptTypes[i] = IRType(name: t.name, kind: kind, location: t.location)
    }
    // Global expressions include concrete literals and typed override references.
    for i in globalExpressions.indices {
      try budget.consume(.work)
      let e = globalExpressions[i]
      let kind: IRExpressionKind
      switch e.kind {
      case .zeroValue(let type): kind = .zeroValue(try ty(type))
      case .compose(let type, let operands): kind = .compose(type: try ty(type), components: operands)
      case .bitcast(let operand, let type): kind = .bitcast(expression: operand, type: try ty(type))
      default:
        guard e.kind.pipelineOperands != nil else {
          throw ShaderFailure.unsupported("Global initializer compaction not implemented")
        }
        kind = e.kind
      }
      globalExpressions[i] = .init(type: try ty(e.type), kind: kind, location: e.location)
    }
    for i in constants.indices {
      try budget.consume(.work)
      let c = constants[i]
      constants[i] = .init(
        name: c.name, type: try ty(c.type), initializer: c.initializer, location: c.location)
    }
    for i in overrides.indices {
      try budget.consume(.work)
      let value = overrides[i]
      overrides[i] = .init(name: value.name, id: value.id, type: try ty(value.type),
                           initializer: value.initializer, location: value.location)
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
      func cooperativeData(_ data: IRCooperativeData) throws -> IRCooperativeData {
        .init(pointer: try ex(data.pointer), stride: try ex(data.stride), rowMajor: data.rowMajor)
      }
      for j in kept.indices {
        try budget.consume(.work)
        let e = kept[j]
        let k: IRExpressionKind
        switch e.kind {
        case .literal, .constant, .override, .functionArgument, .globalVariable, .localVariable, .callResult, .subgroupBallotResult, .rayQueryProceedResult:
          k = e.kind
        case .zeroValue(let t): k = .zeroValue(try ty(t))
        case .runtimeValue(let value): k = .runtimeValue(try ex(value))
        case .compose(let t, let xs): k = .compose(type: try ty(t), components: try xs.map(ex))
        case .access(let b, let x): k = .access(base: try ex(b), index: try ex(x))
        case .accessIndex(let b, let x): k = .accessIndex(base: try ex(b), index: x)
        case .splat(let n, let v): k = .splat(size: n, value: try ex(v))
        case .swizzle(let n, let v, let p): k = .swizzle(size: n, vector: try ex(v), pattern: p)
        case .load(let p): k = .load(pointer: try ex(p))
        case .atomicResult(let t, let comparison): k = .atomicResult(type: try ty(t), comparison: comparison)
        case .workgroupUniformLoadResult(let t): k = .workgroupUniformLoadResult(type: try ty(t))
        case .subgroupOperationResult(let t): k = .subgroupOperationResult(type: try ty(t))
        case .rayQueryGetIntersection(let query, let committed): k = .rayQueryGetIntersection(query: try ex(query), committed: committed)
        case .rayQueryVertexPositions(let query, let committed): k = .rayQueryVertexPositions(query: try ex(query), committed: committed)
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
        case .relational(let function, let operand):
          k = .relational(function: function, argument: try ex(operand))
        case .math(let fn, let a, let b, let c, let d):
          k = .math(
            function: fn, argument: try ex(a), argument1: try optional(b),
            argument2: try optional(c), argument3: try optional(d))
        case .cast(let v, let t): k = .cast(expression: try ex(v), target: t)
        case .bitcast(let v, let t): k = .bitcast(expression: try ex(v), type: try ty(t))
        case .cooperativeLoad(let columns, let rows, let role, let data):
          k = .cooperativeLoad(columns: columns, rows: rows, role: role, data: try cooperativeData(data))
        case .cooperativeMultiplyAdd(let a, let b, let c):
          k = .cooperativeMultiplyAdd(a: try ex(a), b: try ex(b), c: try ex(c))
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
        case .block, .break, .continue, .kill, .demoteToHelperInvocation, .controlBarrier, .memoryBarrier: k = s.kind
        case .ifCondition(let c, let a, let b):
          k = .ifCondition(condition: try ex(c), accept: a, reject: b)
        case .switchValue(let v, let cs): k = .switchValue(selector: try ex(v), cases: cs)
        case .loop(let b, let c, let x): k = .loop(body: b, continuing: c, breakIf: try optional(x))
        case .returnValue(let x): k = .returnValue(try optional(x))
        case .store(let p, let v): k = .store(pointer: try ex(p), value: try ex(v))
        case .rayQuery(let query, let function):
          let mapped: IRRayQueryFunction
          switch function {
          case .initialize(let acceleration, let descriptor): mapped = .initialize(accelerationStructure: try ex(acceleration), descriptor: try ex(descriptor))
          case .proceed(let result): mapped = .proceed(result: try ex(result))
          case .generateIntersection(let hit): mapped = .generateIntersection(hitT: try ex(hit))
          case .confirmIntersection, .terminate, .begin: mapped = function
          }
          k = .rayQuery(query: try ex(query), function: mapped)
        case .cooperativeStore(let target, let data):
          k = .cooperativeStore(target: try ex(target), data: try cooperativeData(data))
        case .imageStore(let image, let coordinate, let arrayIndex, let value):
          k = .imageStore(image: try ex(image), coordinate: try ex(coordinate),
                          arrayIndex: try optional(arrayIndex), value: try ex(value))
        case .imageAtomic(let image, let coordinate, let arrayIndex, let function, let value):
          k = .imageAtomic(image: try ex(image), coordinate: try ex(coordinate),
                           arrayIndex: try optional(arrayIndex), function: function, value: try ex(value))
        case .workgroupUniformLoad(let p, let r): k = .workgroupUniformLoad(pointer: try ex(p), result: try ex(r))
        case .subgroupGather(let mode, let argument, let result):
          k = .subgroupGather(mode: try mode.remappingIndex(ex), argument: try ex(argument), result: try ex(result))
        case .subgroupBallot(let result, let predicate):
          k = .subgroupBallot(result: try ex(result), predicate: try optional(predicate))
        case .subgroupCollectiveOperation(let operation, let collective, let argument, let result):
          k = .subgroupCollectiveOperation(op: operation, collective: collective, argument: try ex(argument), result: try ex(result))
        case .atomic(let p, let operation, let v, let r):
          let mapped: IRAtomicFunction
          if case .exchange(let compare, let strong) = operation {
            mapped = .exchange(compare: try optional(compare), strong: strong)
          }
          else { mapped = operation }
          k = .atomic(pointer: try ex(p), function: mapped, value: try ex(v), result: try optional(r))
        case .call(let t, let xs, let r):
          k = .call(function: t, arguments: try xs.map(ex), result: try optional(r))
        case .debugPrintf(let format, let arguments):
          k = .debugPrintf(format: format, arguments: try arguments.map(ex))
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
    rayDescriptorType = try rayDescriptorType.map(ty)
    rayIntersectionType = try rayIntersectionType.map(ty)
    rayVertexReturnType = try rayVertexReturnType.map(ty)
    predeclaredTypes = try predeclaredTypes.compactMap {
      guard !abstractStructures.contains($0.type.index) else { return nil }
      return .init(kind: $0.kind, type: try ty($0.type))
    }
    for index in documentation.indices {
      try budget.consume(.work)
      let record = documentation[index]
      let target: IRDocumentationTarget
      switch record.target {
      case .type(let handle): target = .type(try ty(handle))
      case .member(let handle, let member): target = .member(type: try ty(handle), index: member)
      default: continue
      }
      documentation[index] = .init(target: target, lines: record.lines)
    }
    types = keptTypes
  }
}
