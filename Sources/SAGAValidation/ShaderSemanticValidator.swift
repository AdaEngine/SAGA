import SAGAShaderIR

/// Fail-closed semantic validation. Unsupported rule families never issue a witness.
public struct ShaderSemanticValidator: IRValidator {
  public typealias Module = IRModule
  public let capabilities: ShaderCapabilities
  public let limits: ShaderLimits
  public init(capabilities: ShaderCapabilities = .referenceDefault, limits: ShaderLimits = .default)
  {
    self.capabilities = capabilities
    self.limits = limits
  }
  public func validate(_ module: IRModule) throws {
    var checker = SemanticChecker(
      module: module, capabilities: capabilities, budget: ShaderBudget(limits: limits))
    try checker.run()
  }
}

private struct TypeLayout {
  var alignment: UInt64
  var size: UInt64
  var sized: Bool = true
  var data: Bool = true
  var constructible: Bool = true
  var containsAtomic: Bool = false
  var host: Bool = true
  var uniformIssue: ShaderDiagnostic? = nil
}

private struct SemanticChecker {
  let module: IRModule
  let capabilities: ShaderCapabilities
  var budget: ShaderBudget
  var layouts: [TypeLayout] = []
  var functionStages: [Set<IRStage>] = []
  var functionUniform: [Bool] = []
  let allStages: Set<IRStage> = [.vertex, .fragment, .compute]

  func fail(_ code: ShaderDiagnosticCode, _ message: String, _ location: ShaderLocation? = nil)
    throws -> Never
  {
    throw ShaderFailure.diagnostics([
      .init(severity: .error, code: code, message: message, location: location)
    ])
  }
  func unsupported(_ message: String, _ loc: ShaderLocation? = nil) throws -> Never {
    try fail(.unsupportedConstruct, message, loc)
  }
  func need(
    _ condition: Bool, _ code: ShaderDiagnosticCode, _ message: String, _ loc: ShaderLocation? = nil
  ) throws {
    if !condition { try fail(code, message, loc) }
  }
  func cap(_ cap: ShaderCapabilities, _ loc: ShaderLocation? = nil) throws {
    try need(
      capabilities.contains(cap), .unsupportedCapability,
      "Required capability is not enabled: \(cap.rawValue)", loc)
  }
  func type(_ h: IRHandle<IRType>) throws -> IRTypeKind { try module.types.value(for: h).kind }
  mutating func same(_ a: IRHandle<IRType>, _ b: IRHandle<IRType>) throws -> Bool {
    try budget.consume(.work)
    if a == b {
      _ = try type(a)
      return true
    }
    // Nested handles are topologically ordered; compare recursively with explicit depth quota.
    try budget.enterDepth()
    defer { try? budget.leaveDepth() }
    switch (try type(a), try type(b)) {
    case (.pointer(let x, let xs), .pointer(let y, let ys)): return try xs == ys && same(x, y)
    case (.array(let x, let nx, let sx), .array(let y, let ny, let sy)):
      return try nx == ny && sx == sy && same(x, y)
    case (.structure, .structure): return false  // Nominal aggregate equivalence requires a dedicated interning pass.
    default: return try type(a) == type(b)
    }
  }
  func scalar(_ h: IRHandle<IRType>) throws -> IRScalar? {
    switch try type(h) {
    case .scalar(let s), .vector(_, let s), .matrix(_, _, let s): return s
    default: return nil
    }
  }
  func boolean(_ h: IRHandle<IRType>) throws -> Bool {
    switch try type(h) {
    case .scalar(let s): return s == .bool
    default: return false
    }
  }
  func integer(_ h: IRHandle<IRType>) throws -> Bool {
    guard let s = try scalar(h) else { return false }
    return s.kind == .sint || s.kind == .uint
  }
  func round(_ size: UInt64, _ alignment: UInt64) throws -> UInt64 {
    guard alignment > 0, size <= UInt64(Int32.max) else {
      try fail(.invalidType, "Layout exceeds maximum type size")
    }
    let result = ((size + alignment - 1) / alignment) * alignment
    try need(result <= UInt64(Int32.max), .invalidType, "Layout exceeds maximum type size")
    return result
  }
  mutating func text(_ value: String?) throws {
    guard let value else { return }
    for _ in value.utf8 {
      try budget.consume(.sourceBytes)
      try budget.consume(.work)
    }
  }
  mutating func run() throws {
    try budget.consume(.work)
    try need(
      capabilities.rawValue & ~ShaderCapabilities.allKnown.rawValue == 0, .unsupportedCapability,
      "Unknown capability bits")
    // Filter/interchange/special-type semantic rules are not silently ignored.
    guard module.diagnosticFilters.count == 0, module.diagnosticFilter == nil,
      module.specialTypes == IRSpecialTypes(), module.documentation.isEmpty
    else {
      try unsupported(
        "Special type, diagnostic-filter and documentation handle validation not implemented")
    }
    for (i, t) in module.types.values.enumerated() {
      try budget.consume(.nodes)
      try budget.consume(.work)
      try text(t.name)
      var checked = try checkType(t, index: i)
      checked.uniformIssue = try uniformConstraint(t)
      layouts.append(checked)
    }
    try expressions(module.globalExpressions, function: nil, currentFunction: nil)
    for c in module.constants.values {
      try text(c.name)
      try budget.consume(.nodes)
      _ = try type(c.type)
      let e = try module.globalExpressions.value(for: c.initializer)
      try need(
        try same(c.type, e.type), .typeMismatch, "Constant initializer type mismatch", c.location)
      try need(
        try constant(c.initializer), .typeMismatch,
        "Constant initializer is not a constant expression", c.location)
    }
    // Override evaluation, specialization and dependent layout need separate rule implementation.
    if module.overrides.count != 0 { try unsupported("Override validation is not implemented") }
    var bindings = Set<String>()
    for g in module.globalVariables.values {
      try text(g.name)
      try budget.consume(.nodes)
      try budget.consume(.work)
      let layout = try layout(g.type)
      try storeType(g.type, space: g.space, location: g.location)
      switch g.space {
      case .private, .workgroup:
        try need(
          g.binding == nil && layout.data && layout.sized, .invalidAddressSpace,
          "Invalid private/workgroup global", g.location)
        if g.space == .workgroup {
          try need(
            g.initializer == nil, .invalidAddressSpace,
            "Workgroup globals cannot have initializers", g.location)
        }
      case .uniform, .storage:
        try need(
          layout.host && g.binding != nil, .invalidAddressSpace,
          "Host-shareable bound type required", g.location)
        if g.space == .uniform {
          try need(layout.sized, .invalidType, "Uniform type must be sized", g.location)
        }
        if g.space == .uniform, let issue = layout.uniformIssue {
          try fail(issue.code, issue.message, issue.location ?? g.location)
        }
      default: try unsupported("Global address space \(g.space) not implemented", g.location)
      }
      if case .storage(let a) = g.space {
        try need(
          a.rawValue > 0 && a.rawValue & ~UInt8(7) == 0 && a.contains(.load), .invalidAddressSpace,
          "Invalid storage access", g.location)
      }
      try need(
        g.memoryDecorations.isEmpty, .unsupportedConstruct,
        "Memory decoration validation not implemented", g.location)
      if let b = g.binding {
        try need(
          bindings.insert("\(b.group):\(b.binding)").inserted, .invalidBinding,
          "Duplicate resource binding", g.location)
      }
      if let initExpr = g.initializer {
        try need(
          g.space == .private, .invalidAddressSpace, "Initializer only allowed for private globals",
          g.location)
        let expr = try module.globalExpressions.value(for: initExpr)
        try need(
          try same(g.type, expr.type), .typeMismatch, "Global initializer type mismatch", g.location
        )
        try need(
          try constant(initExpr), .typeMismatch, "Global initializer must be constant", g.location)
      }
    }
    for (i, f) in module.functions.values.enumerated() { try function(f, index: i) }
    var entryNames = Set<String>()
    var entryFunctions = Set<IRHandle<IRFunction>>()
    for ep in module.entryPoints {
      try text(ep.name)
      try budget.consume(.nodes)
      try budget.consume(.work)
      try need(
        entryNames.insert("\(ep.stage):\(ep.name)").inserted, .duplicateName,
        "Duplicate entrypoint", ep.location)
      try need(
        entryFunctions.insert(ep.function).inserted, .invalidReference,
        "Entrypoint function reuse not implemented", ep.location)
      let f = try module.functions.value(for: ep.function)
      try need(
        functionStages[ep.function.index].contains(ep.stage), .invalidControlFlow,
        "Operation incompatible with entrypoint stage", ep.location)
      guard ep.meshInfo == nil, ep.taskPayload == nil, ep.incomingRayPayload == nil,
        ep.workgroupOverrides == nil
      else {
        try unsupported("Extended entrypoint metadata/overrides not implemented", ep.location)
      }
      if ep.earlyDepthTest != nil {
        try cap(.earlyDepthTest, ep.location)
        try need(
          ep.stage == .fragment, .invalidControlFlow, "Early depth requires fragment", ep.location)
      }
      if ep.stage == .compute {
        let size = ep.workgroupSize
        try need(
          size.x > 0 && size.x <= 0x4000 && size.y > 0 && size.y <= 0x4000 && size.z > 0
            && size.z <= 0x4000, .invalidType, "Workgroup dimensions must be positive", ep.location)
        try need(
          f.result == nil, .invalidType, "Compute entrypoint cannot return a value", ep.location)
      }
      var locations = Set<UInt32>()
      var builtins = Set<String>()
      for a in f.arguments {
        try interface(
          a.type, binding: a.binding, stage: ep.stage, output: false, locations: &locations,
          builtins: &builtins, location: a.location)
      }
      if let r = f.result {
        locations = []
        builtins = []
        try interface(
          r.type, binding: r.binding, stage: ep.stage, output: true, locations: &locations,
          builtins: &builtins)
      }
      if ep.stage == .vertex {
        // Output builtin identities include members of an entrypoint result structure.
        try need(
          builtins.contains("position"), .invalidBinding,
          "Vertex entrypoint requires position output", ep.location)
      }
    }
    // Entry points must not be callable. Inspect even unreachable statements.
    for f in module.functions.values {
      for s in f.statements.values {
        try budget.consume(.work)
        if case .call(let target, _, _) = s.kind {
          try need(
            !entryFunctions.contains(target), .invalidReference, "Cannot call entrypoint",
            s.location)
        }
      }
    }
  }
  func layout(_ h: IRHandle<IRType>) throws -> TypeLayout {
    try need(layouts.indices.contains(h.index), .invalidReference, "Invalid type handle")
    return layouts[h.index]
  }
  mutating func storeType(_ t: IRHandle<IRType>, space: IRAddressSpace, location: ShaderLocation?)
    throws
  {
    try budget.consume(.work)
    let l = try layout(t)
    try need(
      l.data, .invalidAddressSpace,
      "Store type must be a plain scalar/vector/matrix/array/structure/atomic type", location)
    switch space {
    case .private, .function:
      try need(
        l.constructible && l.sized, .invalidAddressSpace,
        "Private/function store type must be constructible and sized", location)
    case .workgroup:
      try need(
        l.sized, .invalidAddressSpace, "Workgroup store type must have fixed footprint", location)
    case .uniform:
      try need(
        l.host && l.constructible && l.sized, .invalidAddressSpace,
        "Uniform store type must be host-shareable and constructible", location)
      if let issue = l.uniformIssue {
        try fail(issue.code, issue.message, issue.location ?? location)
      }
    case .storage(let access):
      try need(l.host, .invalidAddressSpace, "Storage store type must be host-shareable", location)
      try need(
        access.contains(.load) && access.rawValue & ~UInt8(7) == 0, .invalidAddressSpace,
        "Invalid storage access", location)
      if l.containsAtomic {
        try need(
          access.contains(.store), .invalidAddressSpace,
          "Atomic storage requires read-write access", location)
      }
    default: try unsupported("Store address space legality not implemented", location)
    }
  }

  mutating func scalarCheck(_ s: IRScalar, loc: ShaderLocation?) throws {
    switch s.kind {
    case .bool: try need(s.width == 1, .invalidType, "bool width must be 1", loc)
    case .float:
      try need([2, 4, 8].contains(s.width), .invalidType, "Invalid float width", loc)
      if s.width == 2 { try cap(.shaderFloat16, loc) }
      if s.width == 8 { try cap(.float64, loc) }
    case .sint, .uint:
      try need([2, 4, 8].contains(s.width), .invalidType, "Invalid integer width", loc)
      if s.width == 2 { try cap(.shaderInt16, loc) }
      if s.width == 8 { try cap(.shaderInt64, loc) }
    case .abstractInt, .abstractFloat:
      try unsupported("Abstract scalar validation requires concretization", loc)
    }
  }
  mutating func checkType(_ t: IRType, index: Int) throws -> TypeLayout {
    let loc = t.location
    func prior(_ h: IRHandle<IRType>) throws {
      try need(
        h.index < index, .invalidReference, "Type reference must precede use (cycles forbidden)",
        loc)
    }
    switch t.kind {
    case .scalar(let s):
      try scalarCheck(s, loc: loc)
      return TypeLayout(alignment: UInt64(s.width), size: UInt64(s.width), host: s.kind != .bool)
    case .vector(let n, let s):
      try scalarCheck(s, loc: loc)
      let lanes = UInt64(n.rawValue)
      return TypeLayout(
        alignment: (lanes == 3 ? 4 : lanes) * UInt64(s.width), size: lanes * UInt64(s.width),
        host: s.kind != .bool)
    case .matrix(let cols, let rows, let s):
      try scalarCheck(s, loc: loc)
      try need(s.kind == .float, .invalidType, "Matrix scalar must be floating point", loc)
      let row = UInt64(rows.rawValue)
      let align = (row == 3 ? 4 : row) * UInt64(s.width)
      return TypeLayout(
        alignment: align, size: try round(row * UInt64(s.width), align) * UInt64(cols.rawValue))
    case .atomic(let s):
      try scalarCheck(s, loc: loc)
      try need(
        (s.kind == .uint || s.kind == .sint) && s.width == 4, .unsupportedConstruct,
        "Only 32-bit integer atomic type rules implemented", loc)
      return TypeLayout(alignment: 4, size: 4, constructible: false, containsAtomic: true)
    case .pointer(let base, let space):
      try prior(base)
      let pointee = try layout(base)
      try storeType(base, space: space, location: loc)
      try need(pointee.data, .invalidType, "Pointer base must be a plain type", loc)
      switch space {
      case .function, .private, .workgroup, .uniform: break
      case .storage(let a):
        try need(
          a.rawValue > 0 && a.rawValue & ~UInt8(7) == 0, .invalidAddressSpace,
          "Invalid pointer access", loc)
      default: try unsupported("Pointer address space not implemented", loc)
      }
      return TypeLayout(alignment: 1, size: 0, data: false, constructible: false, host: false)
    case .array(let base, let count, let stride):
      try prior(base)
      let b = try layout(base)
      try need(
        b.data && b.sized && UInt64(stride) >= b.size && UInt64(stride) % b.alignment == 0,
        .invalidType, "Invalid array stride/base", loc)
      switch count {
      case .constant(let n):
        try need(
          n > 0 && UInt64(n) * UInt64(stride) <= UInt64(Int32.max), .invalidType,
          "Array size overflow/zero", loc)
        return TypeLayout(
          alignment: b.alignment, size: UInt64(n) * UInt64(stride), constructible: b.constructible,
          containsAtomic: b.containsAtomic, host: b.host)
      case .dynamic:
        return TypeLayout(
          alignment: b.alignment, size: UInt64(stride), sized: false, constructible: false,
          containsAtomic: b.containsAtomic, host: b.host)
      case .override: try unsupported("Override-sized arrays not implemented", loc)
      }
    case .structure(let members, let span):
      try need(!members.isEmpty, .invalidType, "Empty structure", loc)
      var end: UInt64 = 0
      var align: UInt64 = 1
      var sized = true
      var host = true
      var constructible = true
      var atomic = false
      for (j, m) in members.enumerated() {
        try text(m.name)
        try budget.consume(.work)
        try prior(m.type)
        let b = try layout(m.type)
        try need(
          b.data && UInt64(m.offset) >= end && UInt64(m.offset) % b.alignment == 0, .invalidType,
          "Invalid member offset/type", m.location ?? loc)
        if !b.sized {
          guard case .array(_, .dynamic, _) = try type(m.type), j == members.count - 1 else {
            try fail(
              .invalidType, "Only a direct runtime array may be the final unsized member",
              m.location ?? loc)
          }
        }
        end = UInt64(m.offset) + b.size
        align = max(align, b.alignment)
        sized = b.sized
        host = host && b.host
        constructible = constructible && b.constructible
        atomic = atomic || b.containsAtomic
      }
      try need(
        UInt64(span) >= end && UInt64(span) % align == 0 && UInt64(span) <= UInt64(Int32.max),
        .invalidType, "Invalid structure span", loc)
      return TypeLayout(
        alignment: align, size: UInt64(span), sized: sized, constructible: constructible && sized,
        containsAtomic: atomic, host: host)
    default: try unsupported("Type rule family not implemented", loc)
    }
  }
  // Cache uniform compatibility separately: an invalid uniform layout can still be valid storage.
  // Pinned WGSL address-space-layout-constraints; Naga valid/type.rs uniform_layout.
  mutating func uniformConstraint(_ t: IRType) throws -> ShaderDiagnostic? {
    func issue(_ message: String, _ location: ShaderLocation?) -> ShaderDiagnostic {
      .init(severity: .error, code: .invalidType, message: message, location: location)
    }
    switch t.kind {
    case .atomic:
      return issue("Atomic types cannot appear in uniform address space", t.location)
    case .array(let base, _, let stride):
      let child = try layout(base)
      if let inherited = child.uniformIssue { return inherited }
      if UInt64(stride) % max(16, child.alignment) != 0 {
        return issue(
          "Uniform array stride must be a multiple of the element alignment and 16", t.location)
      }
    case .structure(let members, _):
      var previousStructEnd: UInt64? = nil
      for member in members {
        try budget.consume(.work)
        let child = try layout(member.type)
        if let inherited = child.uniformIssue { return inherited }
        let childType = try type(member.type)
        let required: UInt64
        switch childType {
        case .array, .structure: required = max(16, child.alignment)
        default: required = child.alignment
        }
        if UInt64(member.offset) % required != 0 {
          return issue(
            "Uniform member offset violates required alignment", member.location ?? t.location)
        }
        if let previousStructEnd, UInt64(member.offset) < previousStructEnd {
          return issue(
            "Uniform member after a structure requires roundUp(16, SizeOf(struct)) spacing",
            member.location ?? t.location)
        }
        if case .structure = childType {
          previousStructEnd = UInt64(member.offset) + (try round(child.size, 16))
        } else {
          previousStructEnd = nil
        }
      }
    default: break
    }
    return nil
  }

  mutating func constant(_ h: IRHandle<IRExpression>) throws -> Bool {
    try budget.consume(.work)
    try budget.enterDepth()
    defer { try? budget.leaveDepth() }
    let e = try module.globalExpressions.value(for: h)
    switch e.kind {
    case .literal, .zeroValue: return true
    case .compose(_, let xs):
      for x in xs { if try !constant(x) { return false } }
      return true
    // Arithmetic constant evaluation requires overflow, domain and precision rules; reject until implemented.
    default: return false
    }
  }
  mutating func expressions(
    _ table: IRTable<IRExpression>, function: IRFunction?, currentFunction: Int?
  ) throws {
    for (i, e) in table.values.enumerated() {
      try budget.consume(.nodes)
      try budget.consume(.work)
      let declared = try type(e.type)
      if function == nil {
        switch e.kind {
        case .literal, .zeroValue, .compose, .constant: break
        default:
          try unsupported("Global expression constant evaluation not implemented", e.location)
        }
      }
      func operand(_ h: IRHandle<IRExpression>) throws -> IRExpression {
        try need(
          h.index < i, .invalidReference, "Expression references must precede use", e.location)
        return try table.value(for: h)
      }
      func match(_ expected: IRTypeKind) throws {
        try need(declared == expected, .typeMismatch, "Expression result type mismatch", e.location)
      }
      switch e.kind {
      case .literal(let l):
        let s: IRScalar
        switch l {
        case .u32: s = .u32
        case .i32: s = .i32
        case .bool: s = .bool
        case .f32(let v):
          try need(v.isFinite, .invalidType, "Non-finite literal", e.location)
          s = .f32
        case .f64(let v):
          try need(v.isFinite, .invalidType, "Non-finite literal", e.location)
          s = IRScalar(kind: .float, width: 8)
        case .f16Bits(let bits):
          try need(bits & 0x7c00 != 0x7c00, .invalidType, "Non-finite f16 literal", e.location)
          s = .f16
        case .u16: s = IRScalar(kind: .uint, width: 2)
        case .i16: s = IRScalar(kind: .sint, width: 2)
        case .u64: s = IRScalar(kind: .uint, width: 8)
        case .i64: s = IRScalar(kind: .sint, width: 8)
        case .abstractInt, .abstractFloat:
          try unsupported("Abstract literal concretization not implemented", e.location)
        }
        try match(.scalar(s))
      case .zeroValue(let t):
        try need(try same(e.type, t), .typeMismatch, "Zero value type mismatch", e.location)
        try need(
          try layout(t).constructible && layout(t).sized, .invalidType,
          "Zero type is not constructible",
          e.location)
      case .constant(let c):
        let c = try module.constants.value(for: c)
        try need(
          try same(e.type, c.type), .typeMismatch, "Constant reference type mismatch", e.location)
        // Constants can reference global expressions not yet checked by initializer pass, but not cycles.
        try need(
          c.initializer.index < module.globalExpressions.count, .invalidReference,
          "Invalid constant initializer", e.location)
        if function == nil {
          try need(
            c.initializer.index < i, .invalidReference, "Forward/cyclic constant expression",
            e.location)
        }
      case .functionArgument(let n):
        guard let f = function, let index = Int(exactly: n), f.arguments.indices.contains(index)
        else { try fail(.invalidReference, "Invalid function argument", e.location) }
        try need(
          try same(e.type, f.arguments[index].type), .typeMismatch,
          "Function argument type mismatch", e.location)
      case .localVariable(let v):
        guard let f = function else {
          try fail(.invalidReference, "Local reference outside function", e.location)
        }
        let v = try f.localVariables.value(for: v)
        guard case .pointer(let base, .function) = declared else {
          try fail(.typeMismatch, "Local variable expression must be function pointer", e.location)
        }
        try need(
          try same(base, v.type), .typeMismatch, "Local variable pointer base mismatch", e.location)
      case .globalVariable(let v):
        guard function != nil else {
          try fail(.invalidReference, "Global-variable address is not constant", e.location)
        }
        let v = try module.globalVariables.value(for: v)
        guard case .pointer(let base, let space) = declared else {
          try fail(.typeMismatch, "Global address must be pointer", e.location)
        }
        try need(
          space == v.space && (try same(base, v.type)), .typeMismatch,
          "Global pointer type mismatch", e.location)
      case .load(let p):
        let p = try operand(p)
        guard case .pointer(let base, let space) = try type(p.type) else {
          try fail(.typeMismatch, "Load requires pointer", e.location)
        }
        if case .storage(let a) = space {
          try need(a.contains(.load), .invalidAddressSpace, "Pointer cannot be loaded", e.location)
        }
        try need(try same(e.type, base), .typeMismatch, "Load type mismatch", e.location)
        try need(
          try layout(base).constructible, .typeMismatch,
          "Ordinary load requires constructible type", e.location)
      case .unary(let op, let x):
        let x = try operand(x)
        switch try type(x.type) {
        case .scalar, .vector: break
        default: try fail(.typeMismatch, "Unary operand must be scalar/vector", e.location)
        }
        try need(try same(x.type, e.type), .typeMismatch, "Unary result mismatch", e.location)
        guard let s = try scalar(x.type) else {
          try fail(.typeMismatch, "Unary requires numeric/bool", e.location)
        }
        switch op {
        case .negate:
          try need(
            s.kind == .sint || s.kind == .float, .typeMismatch, "Invalid negation", e.location)
        case .logicalNot:
          try need(s.kind == .bool, .typeMismatch, "Logical not requires bool", e.location)
        case .bitwiseNot:
          try need(
            s.kind == .sint || s.kind == .uint, .typeMismatch, "Bitwise not requires integer",
            e.location)
        }
      case .binary(let op, let lhs, let rhs):
        let l = try operand(lhs)
        let r = try operand(rhs)
        let leftKind = try type(l.type)
        let rightKind = try type(r.type)
        if op == .shiftLeft || op == .shiftRight {
          let valid: Bool
          switch (leftKind, rightKind) {
          case (.scalar(.i32), .scalar(.u32)), (.scalar(.u32), .scalar(.u32)): valid = true
          case (.vector(let a, .i32), .vector(let b, .u32)),
               (.vector(let a, .u32), .vector(let b, .u32)): valid = a == b
          default: valid = false
          }
          try need(
            valid, .typeMismatch, "Shifts require i32/u32 and a matching u32 count", e.location)
          try match(leftKind)
          break
        }
        func matrix(_ k: IRTypeKind) -> Bool { if case .matrix = k { return true }; return false }
        if matrix(leftKind) || matrix(rightKind) {
          let expected: IRTypeKind
          switch (op, leftKind, rightKind) {
          case (.add, .matrix(let c, let rows, let scalar), .matrix(let d, let q, let other)),
               (.subtract, .matrix(let c, let rows, let scalar), .matrix(let d, let q, let other)):
            try need(c == d && rows == q && scalar == other, .typeMismatch, "Matrix addition operand mismatch", e.location)
            expected = leftKind
          case (.multiply, .matrix(let c, let rows, let scalar), .matrix(let d, let q, let other)):
            try need(c == q && scalar == other, .typeMismatch, "Matrix product operand mismatch", e.location)
            expected = .matrix(columns: d, rows: rows, scalar: scalar)
          case (.multiply, .matrix(let c, let rows, let scalar), .vector(let n, let other)):
            try need(c == n && scalar == other, .typeMismatch, "Matrix-vector operand mismatch", e.location)
            expected = .vector(size: rows, scalar: scalar)
          case (.multiply, .vector(let n, let scalar), .matrix(let c, let rows, let other)):
            try need(n == rows && scalar == other, .typeMismatch, "Vector-matrix operand mismatch", e.location)
            expected = .vector(size: c, scalar: scalar)
          case (.multiply, .matrix(_, _, let scalar), .scalar(let other)):
            try need(scalar == other, .typeMismatch, "Matrix-scalar operand mismatch", e.location)
            expected = leftKind
          case (.multiply, .scalar(let scalar), .matrix(_, _, let other)):
            try need(scalar == other, .typeMismatch, "Scalar-matrix operand mismatch", e.location)
            expected = rightKind
          default: try fail(.typeMismatch, "Invalid matrix binary overload", e.location)
          }
          try match(expected)
        } else {
        try need(
          try same(l.type, r.type), .typeMismatch, "Binary operands must have matching types",
          e.location)
        guard let s = try scalar(l.type) else {
          try fail(.typeMismatch, "Binary operands must be numeric/bool", e.location)
        }
        switch op {
        case .equal, .notEqual, .less, .lessEqual, .greater, .greaterEqual:
          if op != .equal && op != .notEqual {
            try need(s.kind != .bool, .typeMismatch, "Boolean ordering is invalid", e.location)
          }
          switch try type(l.type) {
          case .scalar: try match(.scalar(.bool))
          case .vector(let n, _): try match(.vector(size: n, scalar: .bool))
          default: try unsupported("Aggregate comparisons not implemented", e.location)
          }
        default:
          try need(try same(e.type, l.type), .typeMismatch, "Binary result mismatch", e.location)
          switch op {
          case .logicalAnd, .logicalOr:
            try need(
              try boolean(l.type), .typeMismatch, "Logical operators require scalar bool",
              e.location)
          case .exclusiveOr:
            try need(
              s.kind == .sint || s.kind == .uint, .typeMismatch, "Xor requires integer operands",
              e.location)
          case .and, .inclusiveOr:
            try need(
              s.kind == .sint || s.kind == .uint || s.kind == .bool, .typeMismatch,
              "Invalid bitwise operand", e.location)
          default:
            try need(
              s.kind == .float || s.kind == .sint || s.kind == .uint, .typeMismatch,
              "Arithmetic requires numeric types", e.location)
            if case .matrix = try type(l.type) {
              try unsupported("Matrix arithmetic overload rules not implemented", e.location)
            }
          }
        }
        }
      case .select(let cond, let a, let b):
        let c = try operand(cond)
        let x = try operand(a)
        let y = try operand(b)
        let conditionKind = try type(c.type)
        let valueKind = try type(x.type)
        switch (conditionKind, valueKind) {
        case (.scalar(.bool), .scalar), (.scalar(.bool), .vector): break
        case (.vector(let n, .bool), .vector(let width, _)):
          try need(n == width, .typeMismatch, "Select condition vector shape mismatch", e.location)
        default:
          try fail(
            .typeMismatch,
            "Select requires scalar/vector values and scalar or matching vector bool", e.location)
        }
        try need(
          try same(x.type, y.type) && same(e.type, x.type), .typeMismatch,
          "Select value types mismatch", e.location)
      case .splat(let n, let value):
        let v = try operand(value)
        guard case .scalar(let s) = try type(v.type) else {
          try fail(.typeMismatch, "Splat needs scalar", e.location)
        }
        try match(.vector(size: n, scalar: s))
      case .compose(let t, let components):
        try need(
          try layout(t).constructible, .invalidType, "Constructor requires constructible type",
          e.location)
        try need(try same(e.type, t), .typeMismatch, "Constructor result type mismatch", e.location)
        var componentTypes: [IRHandle<IRType>] = []
        for c in components {
          try budget.consume(.work)
          componentTypes.append(try operand(c).type)
        }
        switch try type(t) {
        case .vector(let n, let s):
          var lanes = 0
          for c in componentTypes {
            try budget.consume(.work)
            switch try type(c) {
            case .scalar(let leaf) where leaf == s: lanes += 1
            case .vector(let count, let leaf) where leaf == s: lanes += Int(count.rawValue)
            default: try fail(.typeMismatch, "Vector compose component scalar mismatch", e.location)
            }
            try need(
              lanes <= Int(n.rawValue), .typeMismatch, "Too many vector compose components",
              e.location)
          }
          try need(
            lanes == Int(n.rawValue), .typeMismatch, "Vector compose component count", e.location)
        case .matrix(let columns, let rows, let scalar):
          try need(
            componentTypes.count == Int(columns.rawValue), .typeMismatch,
            "Matrix compose requires exactly one vector per column", e.location)
          for c in componentTypes {
            try budget.consume(.work)
            try need(
              try type(c) == .vector(size: rows, scalar: scalar), .typeMismatch,
              "Matrix compose column shape/scalar mismatch", e.location)
          }
        case .array(let base, .constant(let n), _):
          try need(
            UInt64(componentTypes.count) == UInt64(n), .typeMismatch, "Array constructor count",
            e.location)
          for c in componentTypes {
            try need(
              try same(c, base), .typeMismatch, "Array constructor component type", e.location)
          }
        case .structure(let members, _):
          try need(
            componentTypes.count == members.count, .typeMismatch, "Structure constructor count",
            e.location)
          for (c, m) in zip(componentTypes, members) {
            try budget.consume(.work)
            try need(try same(c, m.type), .typeMismatch, "Structure constructor type", e.location)
          }
        default: try unsupported("Constructor overload not implemented", e.location)
        }
      case .accessIndex(let base, let index):
        let b = try operand(base)
        try accessResult(baseType: b.type, index: index, result: e.type, loc: e.location)
      case .access(let base, let index):
        let b = try operand(base)
        let idx = try operand(index)
        try need(
          try type(idx.type) == .scalar(.u32) || type(idx.type) == .scalar(.i32), .typeMismatch,
          "Index must be scalar integer", e.location)
        var indexable = try type(b.type)
        if case .pointer(let pointee, _) = indexable { indexable = try type(pointee) }
        if case .structure = indexable {
          try fail(.typeMismatch, "Struct members require AccessIndex, not Access", e.location)
        }
        let knownIndex = try staticAccessIndex(index, in: table, location: e.location)
        try accessResult(baseType: b.type, index: knownIndex, result: e.type, loc: e.location)
      case .swizzle(let n, let vector, let pattern):
        let v = try operand(vector)
        guard case .vector(let width, let s) = try type(v.type) else {
          try fail(.typeMismatch, "Swizzle needs vector", e.location)
        }
        let map: [IRSwizzleComponent: Int] = [.x: 0, .y: 1, .z: 2, .w: 3]
        for c in [pattern.x, pattern.y, pattern.z, pattern.w].prefix(Int(n.rawValue)) {
          try need(
            map[c, default: 4] < Int(width.rawValue), .typeMismatch, "Invalid swizzle component",
            e.location)
        }
        try match(.vector(size: n, scalar: s))
      case .cast(let source, let target):
        let x = try operand(source)
        guard let input = try scalar(x.type) else {
          try fail(.typeMismatch, "Cast needs numeric scalar/vector", e.location)
        }
        try need(
          input.kind != .bool && target.kind != .bool, .typeMismatch, "Boolean cast is not numeric",
          e.location)
        switch try type(x.type) {
        case .scalar: try match(.scalar(target))
        case .vector(let size, _): try match(.vector(size: size, scalar: target))
        default: try unsupported("Matrix cast overload not implemented", e.location)
        }
      case .bitcast(let source, let target):
        let x = try operand(source)
        try need(try same(target, e.type), .typeMismatch, "Bitcast target mismatch", e.location)
        guard let a = try scalar(x.type), let b = try scalar(e.type), a.kind != .bool,
          b.kind != .bool, a.width == b.width
        else { try fail(.typeMismatch, "Bitcast width/type mismatch", e.location) }
        switch (try type(x.type), declared) {
        case (.scalar, .scalar): break
        case (.vector(let n, _), .vector(let m, _)):
          try need(n == m, .typeMismatch, "Bitcast shape mismatch", e.location)
        default: try unsupported("Packed/matrix bitcast overload not implemented", e.location)
        }
      case .math(let fn, let arg, let arg1, let arg2, let arg3):
        let a = try operand(arg)
        guard let scalar = try scalar(a.type) else {
          try fail(.typeMismatch, "Math needs numeric scalar/vector", e.location)
        }
        if case .matrix = try type(a.type) {
          try unsupported("Matrix builtin overload not implemented", e.location)
        }
        let count: Int
        let floatsOnly: Bool
        switch fn {
        case .min, .max:
          count = 2
          floatsOnly = false
        case .clamp:
          count = 3
          floatsOnly = false
        case .abs, .sign:
          count = 1
          floatsOnly = false
        case .ceil, .floor, .round, .fract, .trunc, .sqrt, .inverseSqrt, .sin, .cos, .tan, .asin,
          .acos, .atan, .exp, .exp2, .log, .log2, .saturate:
          count = 1
          floatsOnly = true
        case .pow, .atan2:
          count = 2
          floatsOnly = true
        default: try unsupported("Math builtin overload rules not implemented", e.location)
        }
        try need(
          scalar.kind != .bool && (!floatsOnly || scalar.kind == .float), .typeMismatch,
          "Math scalar kind mismatch", e.location)
        if fn == .abs || fn == .sign {
          try need(
            scalar.kind == .float || scalar.kind == .sint, .typeMismatch,
            "abs/sign require signed or float", e.location)
        }
        try need(
          arg3 == nil && ((count >= 2) == (arg1 != nil)) && ((count >= 3) == (arg2 != nil)),
          .typeMismatch, "Math argument count", e.location)
        for extra in [arg1, arg2].compactMap({ $0 }) {
          try need(
            try same(operand(extra).type, a.type), .typeMismatch, "Math argument type mismatch",
            e.location)
        }
        try need(try same(e.type, a.type), .typeMismatch, "Math result type mismatch", e.location)
      case .callResult(let target):
        guard let currentFunction, target.index < currentFunction else {
          try fail(.invalidReference, "Calls must target earlier functions", e.location)
        }
        let target = try module.functions.value(for: target)
        guard let r = target.result else {
          try fail(.typeMismatch, "Void call has no result", e.location)
        }
        try need(try same(e.type, r.type), .typeMismatch, "Call result type mismatch", e.location)
      case .derivative(_, _, let x):
        let x = try operand(x)
        switch try type(x.type) {
        case .scalar, .vector: break
        default: try fail(.typeMismatch, "Derivative operand must be scalar/vector", e.location)
        }
        guard let s = try scalar(x.type), s.width >= 4 else {
          try fail(.typeMismatch, "Derivative needs non-f16 float", e.location)
        }
        try need(
          s.kind == .float && (try same(x.type, e.type)), .typeMismatch, "Derivative type mismatch",
          e.location)
      default: try unsupported("Expression rule family is not implemented", e.location)
      }
    }
  }
  mutating func staticAccessIndex(
    _ index: IRHandle<IRExpression>, in expressions: IRTable<IRExpression>,
    location: ShaderLocation?
  ) throws -> UInt32? {
    try budget.consume(.work)
    var expression = try expressions.value(for: index)
    // Constant initializer references normally point to a literal; cap and track malicious cycles.
    var seen = Set<IRHandle<IRConstant>>()
    while case .constant(let handle) = expression.kind {
      try budget.consume(.work)
      try need(seen.insert(handle).inserted, .invalidReference, "Cyclic constant index", location)
      let constant = try module.constants.value(for: handle)
      expression = try module.globalExpressions.value(for: constant.initializer)
    }
    switch expression.kind {
    case .literal(.u32(let value)): return value
    case .literal(.i32(let value)):
      try need(value >= 0, .typeMismatch, "Constant index must not be negative", location)
      return UInt32(value)
    case .zeroValue: return 0
    default: return nil
    }
  }
  mutating func accessResult(
    baseType: IRHandle<IRType>, index: UInt32?, result: IRHandle<IRType>, loc: ShaderLocation?
  ) throws {
    let root = try type(baseType)
    let base: IRTypeKind
    let space: IRAddressSpace?
    if case .pointer(let t, let s) = root {
      base = try type(t)
      space = s
    } else {
      base = root
      space = nil
    }
    let child: IRTypeKind
    var childHandle: IRHandle<IRType>?
    switch base {
    case .array(let t, let n, _):
      if let index, case .constant(let count) = n {
        try need(index < count, .typeMismatch, "Index out of bounds", loc)
      }
      child = try type(t)
      childHandle = t
    case .structure(let ms, _):
      guard let index, let i = Int(exactly: index), ms.indices.contains(i) else {
        try fail(.typeMismatch, "Structure requires valid static member", loc)
      }
      child = try type(ms[i].type)
      childHandle = ms[i].type
    case .vector(let n, let s):
      if let index {
        try need(index < UInt32(n.rawValue), .typeMismatch, "Vector index out of bounds", loc)
      }
      child = .scalar(s)
    case .matrix(let c, let r, let s):
      if let index {
        try need(index < UInt32(c.rawValue), .typeMismatch, "Matrix index out of bounds", loc)
      }
      child = .vector(size: r, scalar: s)
    default: try fail(.typeMismatch, "Cannot index this type", loc)
    }
    if case .structure(let members, _) = child { try budget.consume(.work, amount: members.count) }
    if let space {
      guard case .pointer(let actual, let actualSpace) = try type(result) else {
        try fail(.typeMismatch, "Indexed pointer needs pointer result", loc)
      }
      try need(actualSpace == space, .typeMismatch, "Indexed pointer address space mismatch", loc)
      if let childHandle {
        try need(try same(actual, childHandle), .typeMismatch, "Indexed pointer type mismatch", loc)
      } else {
        try need(try type(actual) == child, .typeMismatch, "Indexed pointer type mismatch", loc)
      }
    } else if let childHandle {
      try need(try same(result, childHandle), .typeMismatch, "Indexed value type mismatch", loc)
    } else {
      try need(try type(result) == child, .typeMismatch, "Indexed value type mismatch", loc)
    }
  }
  mutating func function(_ f: IRFunction, index: Int) throws {
    try text(f.name)
    try budget.consume(.nodes)
    try budget.consume(.work)
    guard f.diagnosticFilter == nil else {
      try unsupported("Function diagnostic filters not implemented", f.location)
    }
    for a in f.arguments {
      try text(a.name)
      try budget.consume(.work)
      try need(
        try layout(a.type).constructible && layout(a.type).sized, .invalidType,
        "Function arguments must be sized data", a.location)
    }
    if let r = f.result {
      try need(
        try layout(r.type).constructible && layout(r.type).sized, .invalidType,
        "Function result must be sized data", f.location)
    }
    for v in f.localVariables.values {
      try text(v.name)
      try budget.consume(.nodes)
      try storeType(v.type, space: .function, location: v.location)
      try need(
        try layout(v.type).data && layout(v.type).sized, .invalidType, "Local must be sized data",
        v.location)
      if let x = v.initializer {
        let e = try f.expressions.value(for: x)
        try need(
          try same(v.type, e.type), .typeMismatch, "Local initializer type mismatch", v.location)
        switch e.kind {
        case .literal, .zeroValue: break
        default:
          try unsupported("Non-literal local initializer validation not implemented", v.location)
        }
      }
    }
    try expressions(f.expressions, function: f, currentFunction: index)
    var stages = allStages
    var requiresUniform = false
    for e in f.expressions.values {
      try budget.consume(.work)
      if case .derivative = e.kind {
        stages.formIntersection([.fragment])
        requiresUniform = true
      }
      if case .globalVariable(let g) = e.kind,
        try module.globalVariables.value(for: g).space == .workgroup
      {
        stages.formIntersection([.compute])
      }
    }
    for s in f.statements.values {
      try budget.consume(.work)
      switch s.kind {
      case .kill: stages.formIntersection([.fragment])
      case .controlBarrier, .memoryBarrier:
        stages.formIntersection([.compute])
        requiresUniform = true
      case .call(let target, _, _):
        try need(
          target.index < index, .invalidReference, "Recursive/forward function calls forbidden",
          s.location)
        stages.formIntersection(functionStages[target.index])
        requiresUniform = requiresUniform || functionUniform[target.index]
      default: break
      }
    }
    var available = Set<IRHandle<IRExpression>>()
    var uniform = [Bool]()
    for (i, e) in f.expressions.values.enumerated() {
      try budget.consume(.work)
      let h = try f.expressions.handle(at: i)
      switch e.kind {
      case .literal, .constant, .zeroValue:
        available.insert(h)
        uniform.append(true)
      case .functionArgument, .localVariable, .globalVariable:
        available.insert(h)
        uniform.append(false)
      default:
        var u = true
        for dep in dependencies(e.kind) {
          try budget.consume(.work)
          try need(dep.index < i, .invalidReference, "Forward dependency", e.location)
          u = u && uniform[dep.index]
        }
        if case .load = e.kind { u = false }
        if case .callResult = e.kind { u = false }
        uniform.append(u)
      }
    }
    var seenBlocks = Set<IRHandle<IRBlock>>()
    var seenStatements = Set<IRHandle<IRStatement>>()
    var results = Set<IRHandle<IRExpression>>()
    var divergentExit = false
    let returned = try block(
      f.body, f: f, available: available, uniform: uniform, controlUniform: true, active: [],
      seen: &seenBlocks, statements: &seenStatements, callResults: &results,
      divergentExit: &divergentExit, loopDepth: 0, inContinuing: false)
    if f.result != nil {
      try need(returned, .invalidControlFlow, "Not all function paths return a value", f.location)
    }
    try need(
      seenBlocks.count == f.blocks.count && seenStatements.count == f.statements.count,
      .invalidReference, "Orphan blocks/statements not accepted", f.location)
    for named in f.namedExpressions {
      try text(named.name)
      try budget.consume(.work)
      _ = try f.expressions.value(for: named.expression)
    }
    for (i, e) in f.expressions.values.enumerated() {
      if case .callResult = e.kind {
        try need(
          results.contains(try f.expressions.handle(at: i)), .invalidReference,
          "Call result has no matching call", e.location)
      }
    }
    functionStages.append(stages)
    functionUniform.append(requiresUniform)
  }
  func dependencies(_ k: IRExpressionKind) -> [IRHandle<IRExpression>] {
    switch k {
    case .unary(_, let x), .load(let x), .derivative(_, _, let x), .cast(let x, _),
      .bitcast(let x, _):
      return [x]
    case .binary(_, let a, let b), .access(let a, let b): return [a, b]
    case .select(let c, let a, let b): return [c, a, b]
    case .accessIndex(let b, _), .splat(_, let b), .swizzle(_, let b, _): return [b]
    case .compose(_, let xs): return xs
    case .math(_, let a, let b, let c, let d): return [a] + [b, c, d].compactMap({ $0 })
    default: return []
    }
  }
  mutating func block(
    _ handle: IRHandle<IRBlock>, f: IRFunction, available initial: Set<IRHandle<IRExpression>>,
    uniform: [Bool], controlUniform: Bool, active: Set<IRHandle<IRBlock>>,
    seen: inout Set<IRHandle<IRBlock>>, statements: inout Set<IRHandle<IRStatement>>,
    callResults: inout Set<IRHandle<IRExpression>>, divergentExit: inout Bool, loopDepth: Int,
    inContinuing: Bool, breakIf: IRHandle<IRExpression>? = nil
  ) throws -> Bool {
    try budget.enterDepth()
    defer { try? budget.leaveDepth() }
    try budget.consume(.work)
    try need(!active.contains(handle), .invalidReference, "Cyclic block graph")
    try need(
      seen.insert(handle).inserted, .invalidReference,
      "Shared blocks need explicit ownership; duplicate block rejected")
    try budget.consume(.work, amount: initial.count)
    var active = active
    active.insert(handle)
    let b = try f.blocks.value(for: handle)
    var available = initial
    var terminated = false
    var ctrl = controlUniform
    func use(_ h: IRHandle<IRExpression>, _ loc: ShaderLocation?) throws -> IRExpression {
      try need(
        available.contains(h), .invalidReference, "Expression is outside evaluation scope", loc)
      return try f.expressions.value(for: h)
    }
    for sh in b.statements {
      try budget.consume(.nodes)
      try budget.consume(.work)
      let s = try f.statements.value(for: sh)
      let loc = s.location
      try need(
        statements.insert(sh).inserted, .invalidReference, "Statement occurs more than once", loc)
      try need(!terminated, .invalidControlFlow, "Statement after unconditional terminator", loc)
      switch s.kind {
      case .emit(let xs):
        for x in xs {
          try budget.consume(.work)
          let e = try f.expressions.value(for: x)
          try need(!available.contains(x), .invalidReference, "Duplicate or invalid emit", loc)
          if case .callResult = e.kind {
            try fail(.invalidReference, "Call result is produced by call, not emit", loc)
          }
          for dep in dependencies(e.kind) { _ = try use(dep, loc) }
          if case .derivative = e.kind {
            try need(
              ctrl && !divergentExit, .nonUniformControlFlow,
              "Derivative in non-uniform control flow", e.location ?? loc)
          }
          available.insert(x)
        }
      case .returnValue(let x):
        try need(!inContinuing, .invalidControlFlow, "Return forbidden in continuing", loc)
        if let r = f.result, let x {
          try need(try same(use(x, loc).type, r.type), .typeMismatch, "Return type mismatch", loc)
        } else {
          try need(f.result == nil && x == nil, .typeMismatch, "Return value mismatch", loc)
        }
        if !ctrl { divergentExit = true }
        terminated = true
      case .store(let pointer, let value):
        let p = try use(pointer, loc)
        let v = try use(value, loc)
        guard case .pointer(let base, let space) = try type(p.type) else {
          try fail(.typeMismatch, "Store requires pointer", loc)
        }
        switch space {
        case .function, .private, .workgroup: break
        case .storage(let access):
          try need(access.contains(.store), .invalidAddressSpace, "Read-only store", loc)
        default: try fail(.invalidAddressSpace, "Read-only address space", loc)
        }
        try need(
          try layout(base).constructible, .typeMismatch,
          "Ordinary store requires constructible type", loc)
        try need(try same(base, v.type), .typeMismatch, "Store value type mismatch", loc)
      case .block(let child):
        terminated = try block(
          child, f: f, available: available, uniform: uniform, controlUniform: ctrl, active: active,
          seen: &seen, statements: &statements, callResults: &callResults,
          divergentExit: &divergentExit, loopDepth: loopDepth, inContinuing: inContinuing)
      case .ifCondition(let condition, let yes, let no):
        let c = try use(condition, loc)
        try need(try boolean(c.type), .typeMismatch, "If requires bool condition", loc)
        let nested = ctrl && uniform[condition.index]
        let a = try block(
          yes, f: f, available: available, uniform: uniform, controlUniform: nested, active: active,
          seen: &seen, statements: &statements, callResults: &callResults,
          divergentExit: &divergentExit, loopDepth: loopDepth, inContinuing: inContinuing)
        let b = try block(
          no, f: f, available: available, uniform: uniform, controlUniform: nested, active: active,
          seen: &seen, statements: &statements, callResults: &callResults,
          divergentExit: &divergentExit, loopDepth: loopDepth, inContinuing: inContinuing)
        terminated = a && b
        if !uniform[condition.index] && (a || b) { ctrl = false }
      case .call(let target, let args, let result):
        let callee = try module.functions.value(for: target)
        try need(
          args.count == callee.arguments.count, .typeMismatch, "Call argument count mismatch", loc)
        for (value, param) in zip(args, callee.arguments) {
          try budget.consume(.work)
          try need(
            try same(use(value, loc).type, param.type), .typeMismatch,
            "Call argument type mismatch", loc)
        }
        if functionUniform[target.index] {
          try need(
            ctrl && !divergentExit, .nonUniformControlFlow,
            "Uniformity-requiring call in divergent control flow", loc)
        }
        if let result, callee.result != nil {
          let e = try f.expressions.value(for: result)
          try need(
            e.kind == .callResult(target) && callResults.insert(result).inserted, .invalidReference,
            "Incorrect/duplicate call result", loc)
          available.insert(result)
        } else {
          try need(
            callee.result == nil && result == nil, .typeMismatch, "Missing/unexpected call result",
            loc)
        }
      case .controlBarrier(let barrier), .memoryBarrier(let barrier):
        try need(
          barrier.rawValue > 0 && barrier.rawValue & ~UInt8(15) == 0, .invalidType,
          "Invalid barrier flags", loc)
        if barrier.contains(.subgroup) { try cap([.subgroup, .subgroupBarrier], loc) }
        try need(
          ctrl && !divergentExit, .nonUniformControlFlow, "Barrier in divergent control flow", loc)
      case .loop(let body, let continuing, let breakIf):
        // No uniformity proof is inferred from iteration count or mutable conditions.
        _ = try block(
          body, f: f, available: available, uniform: uniform, controlUniform: false, active: active,
          seen: &seen, statements: &statements, callResults: &callResults,
          divergentExit: &divergentExit, loopDepth: loopDepth + 1, inContinuing: false)
        _ = try block(
          continuing, f: f, available: available, uniform: uniform, controlUniform: false,
          active: active, seen: &seen, statements: &statements, callResults: &callResults,
          divergentExit: &divergentExit, loopDepth: loopDepth + 1, inContinuing: true,
          breakIf: breakIf)
        ctrl = false
      case .kill:
        try need(!inContinuing, .invalidControlFlow, "Discard forbidden in continuing", loc)
        if !ctrl { divergentExit = true }
        terminated = true
      case .break, .continue:
        try need(loopDepth > 0 && !inContinuing, .invalidControlFlow, "Invalid loop jump", loc)
        if !ctrl { divergentExit = true }
        terminated = true
      default: try unsupported("Statement rule family not implemented", loc)
      }
    }
    if let breakIf {
      try need(
        try boolean(use(breakIf, b.location).type), .typeMismatch,
        "Loop break-if must be available bool", b.location)
    }
    return terminated
  }
  mutating func interface(
    _ t: IRHandle<IRType>, binding: IRBinding?, stage: IRStage, output: Bool,
    locations: inout Set<UInt32>, builtins: inout Set<String>, location: ShaderLocation? = nil,
    member: Bool = false
  ) throws {
    try budget.consume(.work)
    try budget.enterDepth()
    defer { try? budget.leaveDepth() }
    if case .structure(let members, _) = try type(t) {
      try need(
        binding == nil && !member, .invalidBinding,
        "Entrypoint struct cannot have a direct binding or nested struct member", location)
      for m in members {
        try interface(
          m.type, binding: m.binding, stage: stage, output: output,
          locations: &locations, builtins: &builtins, location: m.location ?? location, member: true
        )
      }
      return
    }
    guard let binding else {
      try fail(.invalidBinding, "Missing entrypoint interface binding", location)
    }
    do {
      switch binding {
      case .location(let n, let interp, let sampling, let blend, let primitive):
        try need(
          stage != .compute && locations.insert(n).inserted, .invalidBinding,
          "Invalid/duplicate location")
        try need(
          !primitive && blend == nil, .unsupportedConstruct,
          "Per-primitive/dual-source interface rules not implemented")
        let varying = (stage == .vertex && output) || (stage == .fragment && !output)
        try need(varying || (interp == nil && sampling == nil), .invalidBinding,
                 "Interpolation is restricted to vertex outputs and fragment inputs")
        guard let s = try scalar(t) else { try fail(.typeMismatch, "Location needs scalar/vector") }
        if case .matrix = try type(t) {
          try fail(.typeMismatch, "Matrix cannot have direct location")
        }
        try need(s.kind != .bool, .typeMismatch, "Boolean location not allowed")
        if s.kind == .uint || s.kind == .sint {
          try need(
            interp == .flat || (stage == .vertex && !output), .invalidBinding,
            "Integer interstage values need flat interpolation")
        }
        if sampling == .sample { try cap(.multisampledShading) }
        if interp == .linear { try cap(.linearInterpolation) }
      case .builtIn(let b):
        let key: String
        if case .position = b { key = "position" } else { key = String(describing: b) }
        try need(builtins.insert(key).inserted, .invalidBinding, "Duplicate builtin")
        let expected: IRTypeKind
        let stages: Set<IRStage>
        let isOutput: Bool
        switch b {
        case .localInvocationIndex:
          expected = .scalar(.u32)
          stages = [.compute]
          isOutput = false
        case .localInvocationId, .globalInvocationId, .workgroupId, .numWorkgroups:
          expected = .vector(size: .three, scalar: .u32)
          stages = [.compute]
          isOutput = false
        case .vertexIndex, .instanceIndex:
          expected = .scalar(.u32)
          stages = [.vertex]
          isOutput = false
        case .frontFacing:
          expected = .scalar(.bool)
          stages = [.fragment]
          isOutput = false
        case .position:
          expected = .vector(size: .four, scalar: .f32)
          stages = output ? [.vertex] : [.fragment]
          isOutput = output
        case .fragDepth:
          expected = .scalar(.f32)
          stages = [.fragment]
          isOutput = true
        case .sampleIndex:
          try cap(.multisampledShading)
          expected = .scalar(.u32)
          stages = [.fragment]
          isOutput = false
        case .sampleMask:
          expected = .scalar(.u32)
          stages = [.fragment]
          isOutput = output
        default: try unsupported("Builtin interface rule not implemented")
        }
        try need(
          stages.contains(stage) && isOutput == output, .invalidBinding,
          "Builtin incompatible with stage/direction")
        try need(try type(t) == expected, .typeMismatch, "Builtin type mismatch")
      }
    } catch ShaderFailure.diagnostics(let ds) {
      throw ShaderFailure.diagnostics(
        ds.map { d in
          ShaderDiagnostic(
            severity: d.severity, code: d.code, message: d.message, location: d.location ?? location
          )
        })
    }
  }
}
