import SAGAShaderIR

/// Fail-closed semantic validation. Unsupported rule families never issue a witness.
public struct ShaderSemanticValidator: IRValidator {
  public typealias Module = IRModule
  public let capabilities: ShaderCapabilities
  public let limits: ShaderLimits
  private let allowsVertexWritableStorage: Bool
  public init(capabilities: ShaderCapabilities = .referenceDefault, limits: ShaderLimits = .default)
  {
    self.init(capabilities: capabilities, limits: limits, allowsVertexWritableStorage: false)
  }
  package init(capabilities: ShaderCapabilities, limits: ShaderLimits = .default, allowsVertexWritableStorage: Bool) {
    self.capabilities = capabilities
    self.limits = limits
    self.allowsVertexWritableStorage = allowsVertexWritableStorage
  }
  public func validate(_ module: IRModule) throws {
    _ = try validateReportingDiagnostics(module)
  }
  public func validateReportingDiagnostics(_ module: IRModule) throws -> [ShaderDiagnostic] {
    try validateAndNormalize(module).diagnostics
  }

  public func validateAndNormalize(_ module: IRModule) throws -> (module: IRModule, diagnostics: [ShaderDiagnostic]) {
    let module = try recoverSPIRVWorkgroupLoads(in: module)
    var checker = SemanticChecker(
      module: module, capabilities: capabilities, allowsVertexWritableStorage: allowsVertexWritableStorage,
      budget: ShaderBudget(limits: limits))
    try checker.run()
    return (module, checker.diagnostics)
  }

  private func recoverSPIRVWorkgroupLoads(in module: IRModule) throws -> IRModule {
    var budget = ShaderBudget(limits: limits)
    var candidates = try SPIRVWorkgroupLoadRecovery.collect(in: module, budget: &budget)
    while !candidates.isEmpty {
      var checker = SemanticChecker(module: module, capabilities: capabilities,
                                    allowsVertexWritableStorage: allowsVertexWritableStorage, budget: budget)
      checker.synchronizedWorkgroupLoads = candidates.mapValues { Set($0.keys) }
      try checker.run()
      let proven = try checker.provenSynchronizedWorkgroupLoads()
      budget = checker.budget
      if proven == checker.synchronizedWorkgroupLoads {
        return try SPIRVWorkgroupLoadRecovery.rewrite(module, candidates: candidates, budget: &budget)
      }
      // A load can depend on another candidate that has just lost its proof.
      // Reanalyze after removing those candidates before changing the IR.
      for index in Array(candidates.keys) {
        guard let loads = candidates[index], let retained = proven[index] else {
          candidates[index] = nil
          continue
        }
        candidates[index] = loads.filter { retained.contains($0.key) }
      }
    }
    return module
  }

  package func nonUniformDescriptorAccesses(
    in module: IRModule
  ) throws -> [Int: Set<IRHandle<IRExpression>>] {
    var checker = SemanticChecker(
      module: module, capabilities: capabilities, allowsVertexWritableStorage: allowsVertexWritableStorage,
      budget: ShaderBudget(limits: limits))
    checker.collectDescriptorUniformity = true
    try checker.run()
    return checker.nonUniformDescriptorAccesses
  }
}

private struct TypeLayout {
  var alignment: UInt64
  var size: UInt64
  var sized: Bool = true
  var data: Bool = true
  var constructible: Bool = true
  var containsAtomic: Bool = false
  var containsWideAtomic: Bool = false
  var containsArray: Bool = false
  var host: Bool = true
  var uniformIssue: ShaderDiagnostic? = nil
  var pipelineSized: Bool = false
}

private struct StatementFlow: OptionSet {
  let rawValue: UInt8
  static let next = Self(rawValue: 1)
  static let exit = Self(rawValue: 2)
  static let breakTarget = Self(rawValue: 4)
  static let continueTarget = Self(rawValue: 8)
}

private struct UniformScopes: OptionSet {
  let rawValue: UInt8
  static let whole = Self(rawValue: 1)
  static let subgroup = Self(rawValue: 2)
  static let all: Self = [.whole, .subgroup]
}

// Each scope refers to a node in the current function's dependency graph.
// Zero is uniform, -1 is intrinsically varying, and other negative values are formal parameters.
private struct UniformValue {
  let whole: Int
  let subgroup: Int

  static let uniform = Self(whole: 0, subgroup: 0)
  static let varying = Self(whole: -1, subgroup: -1)

  static func argument(_ index: UInt32) -> Self {
    let node = -Int(index) - 2
    return Self(whole: node, subgroup: node)
  }

  static func known(_ scopes: UniformScopes) -> Self {
    Self(whole: scopes.contains(.whole) ? 0 : -1, subgroup: scopes.contains(.subgroup) ? 0 : -1)
  }

  subscript(scope: UniformScopes) -> Int { scope == .whole ? whole : subgroup }
}

private enum UniformityRule: Int, Hashable {
  case barrier, derivative, subgroup, cooperativeMatrix

  func scope(subgroupScope: UniformScopes) -> UniformScopes {
    switch self {
    case .barrier, .derivative: .whole
    case .subgroup: subgroupScope
    case .cooperativeMatrix: .subgroup
    }
  }
}

private struct UniformArgument: Hashable {
  let index: Int
  let rule: UniformityRule
}

private struct UniformReturn {
  var whole: Set<Int>?
  var subgroup: Set<Int>?
}

private enum LocalBreakTarget {
  case loop(Int)
  case selection([UniformValue]?)
}

private struct LoopUniformity {
  var breaks: [UniformValue]?
  var continues: [UniformValue]?
  var continueControl: UniformValue = .uniform
  var continueScope: Set<IRHandle<IRExpression>>?
}

private enum UniformityCheck {
  case rule(UniformityRule, IRDiagnosticSeverity, String, ShaderLocation?)
  case bindingIndex(ShaderCapabilities, ShaderLocation?, IRHandle<IRExpression>)
}

private struct FunctionUniformity {
  var computeEntry = false
  var merges: [(Int, Int)] = []
  var arguments: [UniformArgument: UniformityRequirement] = [:]
  var returns: UniformValue = .uniform
  var exits: UniformValue = .uniform
  var breakTargets: [LocalBreakTarget] = []
  var loops: [LoopUniformity] = []
  var pendingChecks: [(Int, UniformityCheck)] = []
  var controlRequirements = UniformityRequirements()
  var nonUniformDescriptorAccesses: Set<IRHandle<IRExpression>> = []
  var synchronizedLoads: [IRHandle<IRExpression>: Int] = [:]
  var synchronizedCalls: [(target: Int, arguments: [Int])] = []

  mutating func merge(_ left: Int, _ right: Int, budget: inout ShaderBudget) throws -> Int {
    if left == -1 || right == -1 { return -1 }
    if left == right || right == 0 { return left }
    if left == 0 { return right }
    return try addDependencies(left, right, budget: &budget)
  }

  private mutating func addDependencies(_ left: Int, _ right: Int, budget: inout ShaderBudget) throws -> Int {
    try budget.consume(.nodes)
    merges.append((left, right))
    return merges.count
  }

  mutating func loopHeader(_ entry: UniformValue, budget: inout ShaderBudget) throws -> UniformValue {
    try UniformValue(whole: addDependencies(entry.whole, 0, budget: &budget),
                     subgroup: addDependencies(entry.subgroup, 0, budget: &budget))
  }

  mutating func linkBackedge(_ header: UniformValue, to next: UniformValue) {
    merges[header.whole - 1].1 = next.whole
    merges[header.subgroup - 1].1 = next.subgroup
  }

  func parameters(of node: Int, budget: inout ShaderBudget) throws -> Set<Int>? {
    if node == -1 { return nil }
    if node == 0 { return [] }
    var parameters: Set<Int> = []
    var visited: Set<Int> = []
    var pending = [node]
    while let next = pending.popLast() {
      try budget.consume(.work)
      if next == -1 { return nil }
      if next < -1 {
        parameters.insert(-next - 2)
      } else if next > 0, visited.insert(next).inserted {
        let (left, right) = merges[next - 1]
        pending.append(left)
        pending.append(right)
      }
    }
    return parameters
  }
}

private struct SynchronizedWorkgroupProof {
  var loads: [IRHandle<IRExpression>: Set<Int>] = [:]
  var calls: [(target: Int, arguments: [Set<Int>?])] = []
}

private struct UniformityRequirement {
  var severity: IRDiagnosticSeverity = .off
  var location: ShaderLocation?

  mutating func require(_ severity: IRDiagnosticSeverity, at location: ShaderLocation?) {
    let order = IRDiagnosticSeverity.allCases
    if let old = order.firstIndex(of: self.severity), let new = order.firstIndex(of: severity), new > old {
      self.severity = severity
      self.location = location
    }
  }
}

private struct UniformityRequirements {
  var barrier = false
  var derivative = UniformityRequirement()
  var subgroup = UniformityRequirement()
  var cooperativeMatrix = UniformityRequirement()
  var arguments: [UniformArgument: UniformityRequirement] = [:]
  var returns = UniformReturn()
}

private struct DiagnosticLevels {
  var derivative: IRDiagnosticSeverity = .error
  var subgroup: IRDiagnosticSeverity = .error
}

private struct SemanticChecker {
  let module: IRModule
  let capabilities: ShaderCapabilities
  let allowsVertexWritableStorage: Bool
  var budget: ShaderBudget
  var layouts: [TypeLayout] = []
  var functionStages: [Set<IRStage>] = []
  var functionImmediates: [Set<IRHandle<IRGlobalVariable>>] = []
  var reusedResourceBindings: Set<IRResourceBinding> = []
  var functionResourceAliases: [Set<IRHandle<IRGlobalVariable>>] = []
  var functionUniform: [UniformityRequirements] = []
  var functionPointerWrites: [Int: Set<Int>] = [:]
  var uniformity = FunctionUniformity()
  var reachable = true
  var diagnosticLevels: [DiagnosticLevels] = []
  var diagnostics: [ShaderDiagnostic] = []
  var collectDescriptorUniformity = false
  var nonUniformDescriptorAccesses: [Int: Set<IRHandle<IRExpression>>] = [:]
  var synchronizedWorkgroupLoads: [Int: Set<IRHandle<IRExpression>>] = [:]
  var synchronizedWorkgroupProofs: [Int: SynchronizedWorkgroupProof] = [:]
  var activeSynchronizedLoads: Set<IRHandle<IRExpression>> = []
  var atomicCompareResults: [IRHandle<IRType>: IRScalar] = [:]
  let allStages: Set<IRStage> = [.vertex, .fragment, .compute]
  var subgroupScope: UniformScopes { module.requiresSubgroupUniformity ? .subgroup : .whole }

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
    case (.bindingArray(let x, let nx), .bindingArray(let y, let ny)):
      return try nx == ny && same(x, y)
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
  mutating func validateDocumentation() throws {
    for record in module.documentation {
      try budget.consume(.nodes)
      try budget.consume(.work)
      let validTarget: Bool
      switch record.target {
      case .module: validTarget = true
      case .type(let handle): validTarget = module.types[handle] != nil
      case .member(let handle, let index):
        if case .structure(let members, _) = module.types[handle]?.kind {
          validTarget = UInt64(index) < UInt64(members.count)
        } else {
          validTarget = false
        }
      case .function(let handle): validTarget = module.functions[handle] != nil
      case .entryPoint(let index): validTarget = UInt64(index) < UInt64(module.entryPoints.count)
      case .constant(let handle): validTarget = module.constants[handle] != nil
      case .globalVariable(let handle): validTarget = module.globalVariables[handle] != nil
      }
      try need(validTarget, .invalidReference, "Invalid documentation target")
      for line in record.lines {
        try budget.consume(.nodes)
        try budget.consume(.work)
        try text(line)
      }
    }
  }
  mutating func validateDiagnosticFilters() throws {
    for (index, filter) in module.diagnosticFilters.values.enumerated() {
      try budget.consume(.nodes)
      try budget.consume(.work)
      var inherited = DiagnosticLevels()
      if let parent = filter.parent {
        try need(parent.index < index, .invalidReference, "Diagnostic filter parent must precede the filter", filter.location)
        inherited = diagnosticLevels[parent.index]
      }
      switch filter.rule {
      case .derivativeUniformity:
        inherited.derivative = filter.severity
      case .subgroupUniformity:
        inherited.subgroup = filter.severity
      case .unknown(let name):
        try validateDiagnosticName(name, at: filter.location)
        try need(name != "derivative_uniformity" && name != "subgroup_uniformity",
                 .invalidType, "Standard diagnostic rule has invalid IR representation", filter.location)
        diagnostics.append(.init(severity: .warning, code: .unknownDiagnosticRule,
                                 message: "Unknown diagnostic rule '\(name)'", location: filter.location))
      case .user(let namespace, let name):
        try validateDiagnosticName(namespace, at: filter.location)
        try validateDiagnosticName(name, at: filter.location)
      }
      diagnosticLevels.append(inherited)
    }
    try validateDiagnosticScope(module.diagnosticFilter, parent: nil)
  }

  mutating func validateDiagnosticName(_ name: String, at location: ShaderLocation?) throws {
    try text(name)
    let scalars = name.unicodeScalars
    guard let first = scalars.first, first == "_" || first.properties.isXIDStart,
          name != "_", scalars.dropFirst().allSatisfy({ $0.properties.isXIDContinue }) else {
      try fail(.invalidType, "Invalid diagnostic rule identifier", location)
    }
  }

  mutating func validateDiagnosticScope(_ leaf: IRHandle<IRDiagnosticFilter>?, parent: IRHandle<IRDiagnosticFilter>?) throws {
    var cursor = leaf
    var seen = Set<IRDiagnosticRule>()
    while cursor != parent {
      try budget.consume(.work)
      guard let handle = cursor, diagnosticLevels.indices.contains(handle.index) else {
        try fail(.invalidReference, "Invalid diagnostic filter scope")
      }
      let filter = try module.diagnosticFilters.value(for: handle)
      try need(seen.insert(filter.rule).inserted, .duplicateName, "Duplicate diagnostic rule in one scope", filter.location)
      cursor = filter.parent
    }
  }

  func diagnosticLevels(for function: IRFunction) -> DiagnosticLevels {
    (function.diagnosticFilter ?? module.diagnosticFilter).map { diagnosticLevels[$0.index] } ?? .init()
  }

  mutating func reportUniformity(_ severity: IRDiagnosticSeverity, _ message: String, at location: ShaderLocation?) throws {
    switch severity {
    case .off: return
    case .error: try fail(.nonUniformControlFlow, message, location)
    case .warning, .info:
      try budget.consume(.nodes)
      diagnostics.append(.init(severity: severity == .warning ? .warning : .info, code: .nonUniformControlFlow,
                               message: message, location: location))
    }
  }

  mutating func mergeUniformity(_ left: UniformValue, _ right: UniformValue) throws -> UniformValue {
    let whole = try uniformity.merge(left.whole, right.whole, budget: &budget)
    let subgroup = left.whole == left.subgroup && right.whole == right.subgroup
      ? whole : try uniformity.merge(left.subgroup, right.subgroup, budget: &budget)
    return UniformValue(whole: whole, subgroup: subgroup)
  }

  mutating func requireUniformity(
    _ value: UniformValue, rule: UniformityRule, severity: IRDiagnosticSeverity,
    message: String, at location: ShaderLocation?
  ) throws {
    guard severity != .off else { return }
    let scope = rule.scope(subgroupScope: subgroupScope)
    try checkUniformity(value[scope], .rule(rule, severity, message, location))
  }

  mutating func checkUniformity(_ node: Int, _ check: UniformityCheck) throws {
    guard reachable else { return }
    // The graph is complete only after the enclosing loops have connected their backedges.
    if node > 0, !uniformity.loops.isEmpty {
      try budget.consume(.nodes)
      uniformity.pendingChecks.append((node, check))
      return
    }
    let parameters = try uniformity.parameters(of: node, budget: &budget)
    switch check {
    case .rule(let rule, let severity, let message, let location):
      guard let parameters else {
        try reportUniformity(severity, message, at: location)
        return
      }
      for index in parameters {
        try budget.consume(.work)
        uniformity.arguments[UniformArgument(index: index, rule: rule), default: .init()].require(severity, at: location)
      }
    case .bindingIndex(let capability, let location, let access):
      if parameters?.isEmpty != true {
        try cap(capability, location)
        if collectDescriptorUniformity { uniformity.nonUniformDescriptorAccesses.insert(access) }
      }
    }
  }

  mutating func requireUniformControl(
    _ control: UniformValue, rule: UniformityRule, severity: IRDiagnosticSeverity,
    message: String, at location: ShaderLocation?
  ) throws {
    guard reachable, severity != .off else { return }
    switch rule {
    case .barrier: uniformity.controlRequirements.barrier = true
    case .derivative: uniformity.controlRequirements.derivative.require(severity, at: location)
    case .subgroup: uniformity.controlRequirements.subgroup.require(severity, at: location)
    case .cooperativeMatrix: uniformity.controlRequirements.cooperativeMatrix.require(severity, at: location)
    }
    let active = try mergeUniformity(control, uniformity.exits)
    try requireUniformity(active, rule: rule, severity: severity, message: message, at: location)
  }

  mutating func returnDependency(
    _ parameters: Set<Int>?, arguments: [IRHandle<IRExpression>], uniform: [UniformValue], scope: UniformScopes
  ) throws -> Int {
    guard let parameters else { return -1 }
    var result = 0
    for parameter in parameters.sorted() {
      try budget.consume(.work)
      guard arguments.indices.contains(parameter), uniform.indices.contains(arguments[parameter].index) else { return -1 }
      result = try uniformity.merge(result, uniform[arguments[parameter].index][scope], budget: &budget)
    }
    return result
  }

  mutating func localReference(
    _ handle: IRHandle<IRExpression>, in function: IRFunction
  ) throws -> (index: Int, partial: Bool)? {
    var cursor = handle
    var partial = false
    while true {
      try budget.consume(.work)
      switch try function.expressions.value(for: cursor).kind {
      case .localVariable(let local): return (local.index, partial)
      case .access(let base, _), .accessIndex(let base, _):
        partial = true
        cursor = base
      case .runtimeValue(let base): cursor = base
      default: return nil
      }
    }
  }

  mutating func joinLocalUniformity(_ previous: [UniformValue]?, _ next: [UniformValue]) throws -> [UniformValue] {
    guard var merged = previous else { return next }
    try budget.consume(.work, amount: merged.count)
    for index in merged.indices {
      merged[index] = try mergeUniformity(merged[index], next[index])
    }
    return merged
  }

  mutating func recordLocalBreak(_ locals: [UniformValue]) throws {
    guard reachable, let target = uniformity.breakTargets.indices.last else { return }
    switch uniformity.breakTargets[target] {
    case .loop(let loop):
      let joined = try joinLocalUniformity(uniformity.loops[loop].breaks, locals)
      uniformity.loops[loop].breaks = joined
    case .selection(let previous):
      let joined = try joinLocalUniformity(previous, locals)
      uniformity.breakTargets[target] = .selection(joined)
    }
  }

  mutating func recordLocalContinue(
    _ locals: [UniformValue], control: UniformValue, available: Set<IRHandle<IRExpression>>
  ) throws {
    guard reachable, let loop = uniformity.loops.indices.last else { return }
    let joined = try joinLocalUniformity(uniformity.loops[loop].continues, locals)
    let continuing = try mergeUniformity(uniformity.loops[loop].continueControl, control)
    uniformity.loops[loop].continues = joined
    uniformity.loops[loop].continueControl = continuing
    if let previous = uniformity.loops[loop].continueScope {
      try budget.consume(.work, amount: min(previous.count, available.count))
      uniformity.loops[loop].continueScope = previous.intersection(available)
    } else {
      uniformity.loops[loop].continueScope = available
    }
  }

  func uniformComputeBuiltin(_ binding: IRBinding?) -> UniformValue {
    guard uniformity.computeEntry else { return .varying }
    switch binding {
    case .builtIn(.workgroupId), .builtIn(.numWorkgroups), .builtIn(.subgroupSize), .builtIn(.numSubgroups):
      return .uniform
    case .builtIn(.subgroupId): return .known(.subgroup)
    default: return .varying
    }
  }

  mutating func expressionUniformity(
    _ expression: IRExpression, handle: IRHandle<IRExpression>, in function: IRFunction,
    uniform: [UniformValue], locals: [UniformValue]
  ) throws -> UniformValue {
    var value = UniformValue.uniform
    for operand in expression.kind.operands {
      try budget.consume(.work)
      value = try mergeUniformity(value, uniform[operand.index])
    }
    switch expression.kind {
    case .arrayLength(let pointer):
      return try cooperativePointerUniformity(pointer, in: function, uniform: uniform)
    case .load(let pointer):
      let pointerType = try type(function.expressions.value(for: pointer).type)
      switch pointerType {
      case .pointer(_, .workgroup) where reachable && activeSynchronizedLoads.contains(handle):
        uniformity.synchronizedLoads[handle] = value.whole
        return value
      case .pointer(_, .function):
        guard let reference = try localReference(pointer, in: function) else { return .varying }
        return try mergeUniformity(value, locals[reference.index])
      case .pointer(_, .uniform), .pointer(_, .immediate): break
      case .pointer(_, .storage(let access)) where !access.contains(.store): break
      default: return .varying
      }
    case .access(let base, let index):
      if uniform[index.index].whole != 0,
         let capability = try nonUniformBindingArrayCapability(for: function.expressions.value(for: base).type) {
        try checkUniformity(uniform[index.index].whole, .bindingIndex(capability, expression.location, handle))
      }
    case .accessIndex(let base, let member):
      let argument = try function.expressions.value(for: base)
      if case .structure(let members, _) = try type(argument.type), members.indices.contains(Int(member)) {
        switch argument.kind {
        case .functionArgument where uniformity.computeEntry:
          return uniformComputeBuiltin(members[Int(member)].binding)
        case .compose(_, let components) where components.indices.contains(Int(member)):
          // An SSA struct can combine uniform builtins and varying lanes.
          return uniform[components[Int(member)].index]
        default: break
        }
      }
    case .imageLoad(let load):
      if try type(function.expressions.value(for: load.image).type).storageTexture?.access.contains(.store) == true {
        return .varying
      }
    default: break
    }
    return requiresImplicitDerivatives(expression.kind) ? .varying : value
  }

  mutating func cooperativePointerUniformity(
    _ handle: IRHandle<IRExpression>, in function: IRFunction, uniform: [UniformValue]
  ) throws -> UniformValue {
    try budget.enterDepth()
    defer { try? budget.leaveDepth() }
    let expression = try function.expressions.value(for: handle)
    switch expression.kind {
    case .globalVariable, .localVariable:
      return .uniform
    case .access(let base, let index):
      return try mergeUniformity(cooperativePointerUniformity(base, in: function, uniform: uniform), uniform[index.index])
    case .accessIndex(let base, _):
      return try cooperativePointerUniformity(base, in: function, uniform: uniform)
    default:
      return uniform[handle.index]
    }
  }

  mutating func subgroupResultUniformity(
    _ operands: [IRHandle<IRExpression>], uniform: [UniformValue], control: UniformValue
  ) throws -> UniformValue {
    guard module.requiresSubgroupUniformity else { return .varying }
    var subgroup = try uniformity.merge(control.subgroup, uniformity.exits.subgroup, budget: &budget)
    for operand in operands {
      subgroup = try uniformity.merge(subgroup, uniform[operand.index].subgroup, budget: &budget)
    }
    return UniformValue(whole: -1, subgroup: subgroup)
  }

  mutating func run() throws {
    try budget.consume(.work)
    try need(
      capabilities.rawValue & ~ShaderCapabilities.allKnown.rawValue == 0, .unsupportedCapability,
      "Unknown capability bits")
    guard module.specialTypes.predeclared.count <= 2 + 4 * (IRVectorSize.allCases.count + 1),
      module.specialTypes == IRSpecialTypes(predeclared: module.specialTypes.predeclared)
    else {
      try unsupported("Special type handle validation not implemented")
    }
    try validateDiagnosticFilters()
    try validateDocumentation()
    for (i, t) in module.types.values.enumerated() {
      try budget.consume(.nodes)
      try budget.consume(.work)
      try text(t.name)
      var checked = try checkType(t, index: i)
      if !module.uniformBufferStandardLayout {
        checked.uniformIssue = try uniformConstraint(t)
      }
      layouts.append(checked)
    }
    try validatePredeclaredTypes()
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
    try validateOverrides()
    var bindings: Set<IRResourceBinding> = []
    for g in module.globalVariables.values {
      try text(g.name)
      try budget.consume(.nodes)
      try budget.consume(.work)
      let resourceType = try bindingArrayElement(of: g)
      let layout = try layout(resourceType)
      if g.space != .handle { try storeType(resourceType, space: g.space, location: g.location) }
      switch g.space {
      case .handle:
        let kind = try type(resourceType)
        if kind.storageTexture?.format.isNormalized16Bit == true {
          try cap(.storageTexture16BitNormFormats, g.location)
        }
        try need(
          kind.isTextureOrSampler && g.binding != nil,
          .invalidBinding, "Texture or sampler requires a resource binding", g.location)
      case .private, .workgroup:
        try need(
          g.binding == nil && layout.data && (layout.sized || (g.space == .workgroup && layout.pipelineSized)), .invalidAddressSpace,
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
      case .immediate:
        try need(g.binding == nil, .invalidBinding, "Immediate data cannot have a resource binding", g.location)
      default: try unsupported("Global address space \(g.space) not implemented", g.location)
      }
      let allowedDecorations: IRMemoryDecorations
      if case .storage(let a) = g.space {
        allowedDecorations = [.coherent, .volatile]
        try need(
          a.rawValue > 0 && a.rawValue & ~UInt8(7) == 0 && a.contains(.load), .invalidAddressSpace,
          "Invalid storage access", g.location)
      } else { allowedDecorations = [] }
      try need(
        g.memoryDecorations.subtracting(allowedDecorations).isEmpty, .unsupportedConstruct,
        "Unsupported memory decoration for this address space", g.location)
      if let binding = g.binding, !bindings.insert(binding).inserted {
        reusedResourceBindings.insert(binding)
      }
      if let initExpr = g.initializer {
        try need(
          g.space == .private, .invalidAddressSpace, "Initializer only allowed for private globals",
          g.location)
        let expr = try module.globalExpressions.value(for: initExpr)
        try need(
          try same(g.type, expr.type), .typeMismatch, "Global initializer type mismatch", g.location
        )
        if try !constant(initExpr) {
          let graph = try module.pipelineGraph(initExpr, budget: &budget)
          var hasOverride = false
          for handle in graph {
            try budget.consume(.work)
            if case .override = try module.globalExpressions.value(for: handle).kind { hasOverride = true }
          }
          try need(hasOverride, .typeMismatch, "Global initializer must be constant or use overrides", g.location)
        }
      }
    }
    try budget.consume(.work, amount: module.entryPoints.count)
    let entryIndices = Set(module.entryPoints.map { $0.function.index })
    let computeEntries = Set(module.entryPoints.compactMap { $0.stage == .compute ? $0.function.index : nil })
    var hasPointerParameters = false
    for function in module.functions.values {
      try budget.consume(.work)
      for argument in function.arguments {
        try budget.consume(.work)
        if case .pointer = try type(argument.type) { hasPointerParameters = true }
      }
      if hasPointerParameters { break }
    }
    var pointerAccesses: [Int: ShaderPointerAccesses] = [:]
    for (i, f) in module.functions.values.enumerated() {
      try function(f, index: i, entryIndices: entryIndices, computeEntry: computeEntries.contains(i))
      if hasPointerParameters {
        let access = try ShaderPointerAccesses.checking(f, in: module, callees: pointerAccesses, budget: &budget)
        pointerAccesses[i] = access
        functionPointerWrites[i] = access.writtenParameters
      }
    }
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
      try need(functionImmediates[ep.function.index].count <= 1, .invalidBinding,
               "An entry point can use at most one immediate data variable", ep.location)
      let aliases = functionResourceAliases[ep.function.index]
      try budget.consume(.work, amount: aliases.count)
      var entryBindings: Set<IRResourceBinding> = []
      for handle in aliases.sorted(by: { $0.index < $1.index }) {
        let global = try module.globalVariables.value(for: handle)
        if let binding = global.binding, !entryBindings.insert(binding).inserted {
          try fail(.invalidBinding,
                   "Duplicate resource binding at group \(binding.group), binding \(binding.binding) in entry point '\(ep.name)'",
                   global.location)
        }
      }
      guard ep.meshInfo == nil, ep.taskPayload == nil, ep.incomingRayPayload == nil else {
        try unsupported("Extended entrypoint metadata not implemented", ep.location)
      }
      if let overrides = ep.workgroupOverrides {
        try need(ep.stage == .compute, .invalidBinding, "Workgroup overrides require a compute entry point", ep.location)
        try validateWorkgroupOverrides(overrides, at: ep.location)
      }
      if let subgroupSize = ep.subgroupSize {
        try cap(.subgroup, ep.location)
        try need(ep.stage == .compute, .invalidBinding, "Subgroup size requires a compute entry point", ep.location)
        switch subgroupSize {
        case .fixed(let size):
          try need(size.nonzeroBitCount == 1, .invalidBinding, "Subgroup size must be a power of two", ep.location)
        case .overrideExpression(let handle):
          try validateIntegerOverride(handle, subject: "Subgroup size")
        }
      }
      if ep.earlyDepthTest != nil {
        try cap(.earlyDepthTest, ep.location)
        try need(
          ep.stage == .fragment, .invalidControlFlow, "Early depth requires fragment", ep.location)
      }
      if ep.stage == .compute {
        let size = ep.workgroupSize
        try need(
          size.x > 0 && size.y > 0 && size.z > 0, .invalidType, "Workgroup dimensions must be positive", ep.location)
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
    if case .cooperativeMatrix = try type(t) {
      try need(space == .function || space == .private, .invalidAddressSpace,
               "Cooperative matrices require function or private memory", location)
    }
    try need(
      l.data, .invalidAddressSpace,
      "Store type must be a plain scalar/vector/matrix/array/structure/atomic type", location)
    switch space {
    case .private, .function:
      try need(
        l.constructible && l.sized, .invalidAddressSpace,
        "Private/function store type must be constructible and sized", location)
    case .workgroup:
      if l.containsWideAtomic { try cap(.shaderInt64AtomicAllOps, location) }
      try need(
        l.sized || l.pipelineSized, .invalidAddressSpace, "Workgroup store type must have fixed footprint", location)
    case .uniform:
      try need(
        l.host && l.constructible && l.sized, .invalidAddressSpace,
        "Uniform store type must be host-shareable and constructible", location)
      if let issue = l.uniformIssue {
        try fail(issue.code, issue.message, issue.location ?? location)
      }
    case .immediate:
      try cap(.immediates, location)
      try need(l.host && l.constructible && l.sized && !l.containsArray, .invalidAddressSpace,
               "Immediate data must be a fixed host-shareable constructible type without arrays", location)
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

  func bindingArrayElement(of global: IRGlobalVariable) throws -> IRHandle<IRType> {
    guard case .bindingArray(let element, _) = try type(global.type) else { return global.type }
    let capability: ShaderCapabilities
    switch global.space {
    case .handle:
      capability = try type(element).storageTexture == nil ? .textureAndSamplerBindingArray : .storageTextureBindingArray
    case .uniform: capability = .bufferBindingArray
    case .storage: capability = .storageBufferBindingArray
    default: try fail(.invalidAddressSpace, "Binding arrays require a resource address space", global.location)
    }
    try cap(capability, global.location)
    return element
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
    if t.kind.isExternalTexture { try cap(.textureExternal, loc) }
    if let storage = t.kind.storageTexture {
      if storage.scalar == .u64 { try cap(.shaderInt64, loc) }
      if storage.access.contains(.atomic) {
        try cap(storage.scalar == .u64 ? .textureInt64Atomic : .textureAtomic, loc)
      }
    }
    if t.kind.isTextureOrSampler {
      return TypeLayout(alignment: 1, size: 0, sized: false, data: false, constructible: false, host: false)
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
    case .cooperativeMatrix(let columns, let rows, let scalar, _):
      try cap(.cooperativeMatrix, loc)
      try need([.f16, .f32].contains(scalar), .invalidType, "Cooperative matrix scalar must be f16 or f32", loc)
      try scalarCheck(scalar, loc: loc)
      let width = UInt64(scalar.width)
      return TypeLayout(alignment: width, size: UInt64(columns.rawValue) * UInt64(rows.rawValue) * width, host: false)
    case .atomic(let s):
      try scalarCheck(s, loc: loc)
      if s == .i64 || s == .u64 {
        if !capabilities.contains(.shaderInt64AtomicAllOps) { try cap(.shaderInt64AtomicMinMax, loc) }
        return TypeLayout(alignment: 8, size: 8, constructible: false, containsAtomic: true, containsWideAtomic: true)
      }
      if s == .f32 {
        try cap(.shaderFloat32Atomic, loc)
      } else {
        try need(s == .i32 || s == .u32, .unsupportedConstruct, "Unsupported atomic scalar type", loc)
      }
      return TypeLayout(alignment: 4, size: 4, constructible: false, containsAtomic: true)
    case .pointer(let base, let space):
      try prior(base)
      if case .bindingArray(let element, _) = try type(base) {
        // Indexing the global address produces a regular pointer to the bound buffer.
        switch space {
        case .uniform, .storage: break
        default: try fail(.invalidAddressSpace, "Only buffer binding arrays use pointers", loc)
        }
        try storeType(element, space: space, location: loc)
        return TypeLayout(alignment: 1, size: 0, data: false, constructible: false, host: false)
      }
      let pointee = try layout(base)
      try storeType(base, space: space, location: loc)
      try need(pointee.data, .invalidType, "Pointer base must be a plain type", loc)
      switch space {
      case .function, .private, .workgroup, .uniform, .immediate: break
      case .storage(let a):
        try need(
          a.rawValue > 0 && a.rawValue & ~UInt8(7) == 0, .invalidAddressSpace,
          "Invalid pointer access", loc)
      default: try unsupported("Pointer address space not implemented", loc)
      }
      return TypeLayout(alignment: 1, size: 0, data: false, constructible: false, host: false)
    case .array(let base, let count, let stride):
      try prior(base)
      if case .cooperativeMatrix = try type(base) {
        try fail(.invalidType, "Cooperative matrices cannot be array elements", loc)
      }
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
          containsAtomic: b.containsAtomic, containsWideAtomic: b.containsWideAtomic, containsArray: true, host: b.host)
      case .dynamic:
        return TypeLayout(
          alignment: b.alignment, size: UInt64(stride), sized: false, constructible: false,
          containsAtomic: b.containsAtomic, containsWideAtomic: b.containsWideAtomic, containsArray: true, host: b.host)
      case .override(let handle):
        let countType = try type(module.overrides.value(for: handle).type)
        try need(isPipelineSizeType(countType), .typeMismatch, "Workgroup array override count must be a supported integer", loc)
        return TypeLayout(alignment: b.alignment, size: UInt64(stride), sized: false, constructible: false,
                          containsAtomic: b.containsAtomic, containsWideAtomic: b.containsWideAtomic,
                          containsArray: true, host: false, pipelineSized: true)
      }
    case .bindingArray(let base, let count):
      try prior(base)
      let element = try type(base)
      try need(!element.isExternalTexture, .invalidType, "External textures cannot be binding array elements", loc)
      switch element {
      case .structure: break
      default:
        try need(element.isTextureOrSampler, .invalidType, "Binding array elements must be textures, samplers or buffer structures", loc)
      }
      switch count {
      case .constant(let count):
        try need(count > 0, .invalidType, "Binding array count must be positive", loc)
      case .override(let handle):
        let countType = try type(module.overrides.value(for: handle).type)
        try need(isPipelineSizeType(countType), .typeMismatch, "Binding array override count must be a supported integer", loc)
      case .dynamic: break
      }
      return TypeLayout(alignment: 1, size: 0, sized: false, data: false, constructible: false, host: false)
    case .structure(let members, let span):
      try need(!members.isEmpty, .invalidType, "Empty structure", loc)
      try validateDualSourceMembers(members, at: loc)
      var end: UInt64 = 0
      var align: UInt64 = 1
      var sized = true
      var host = true
      var constructible = true
      var atomic = false
      var wideAtomic = false
      var containsArray = false
      for (j, m) in members.enumerated() {
        try text(m.name)
        try budget.consume(.work)
        try prior(m.type)
        if case .cooperativeMatrix = try type(m.type) {
          try fail(.invalidType, "Cooperative matrices cannot be structure members", m.location ?? loc)
        }
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
        wideAtomic = wideAtomic || b.containsWideAtomic
        containsArray = containsArray || b.containsArray
      }
      try need(
        UInt64(span) >= end && UInt64(span) % align == 0 && UInt64(span) <= UInt64(Int32.max),
        .invalidType, "Invalid structure span", loc)
      return TypeLayout(
        alignment: align, size: UInt64(span), sized: sized, constructible: constructible && sized,
        containsAtomic: atomic, containsWideAtomic: wideAtomic, containsArray: containsArray, host: host)
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

  mutating func validateOverrides() throws {
    var ids: Set<UInt16> = []
    var names: Set<[UInt8]> = []
    for (index, value) in module.overrides.values.enumerated() {
      try budget.consume(.nodes)
      try text(value.name)
      if let id = value.id {
        try need(ids.insert(id).inserted, .duplicateName, "Duplicate override ID", value.location)
      }
      if let name = value.name {
        try need(names.insert(Array(name.utf8)).inserted, .duplicateName, "Duplicate override name", value.location)
      }
      guard case .scalar(let scalar) = try type(value.type), [.bool, .i16, .u16, .i32, .u32, .f16, .f32].contains(scalar) else {
        try fail(.typeMismatch, "Override must have a concrete WGSL scalar type", value.location)
      }
      if value.name == nil && value.id == nil {
        try need(value.initializer != nil && isPipelineSizeType(.scalar(scalar)), .invalidReference,
                 "Anonymous array count override needs an integer initializer", value.location)
      }
      guard let initializer = value.initializer else { continue }
      let expression = try module.globalExpressions.value(for: initializer)
      try need(try same(expression.type, value.type), .typeMismatch, "Override initializer type mismatch", value.location)
      let graph = try module.pipelineGraph(initializer, budget: &budget)
      for handle in graph {
        try budget.consume(.work)
        if case .override(let dependency) = try module.globalExpressions.value(for: handle).kind {
          try need(dependency.index < index, .invalidReference, "Forward or cyclic override initializer", value.location)
        }
      }
    }
  }

  mutating func validateWorkgroupOverrides(_ overrides: IRWorkgroupOverrides, at location: ShaderLocation?) throws {
    let handles = [overrides.x, overrides.y, overrides.z].compactMap { $0 }
    try need(!handles.isEmpty, .invalidBinding, "Workgroup overrides must specify a dimension", location)
    for handle in handles {
      try validateIntegerOverride(handle, subject: "Workgroup")
    }
  }

  mutating func validateIntegerOverride(_ handle: IRHandle<IRExpression>, subject: String) throws {
    let expression = try module.globalExpressions.value(for: handle)
    let graph = try module.pipelineGraph(handle, budget: &budget)
    var hasOverride = false
    for dependency in graph {
      try budget.consume(.work)
      if case .override = try module.globalExpressions.value(for: dependency).kind { hasOverride = true }
    }
    try need(hasOverride, .invalidBinding, "\(subject) override expression must reference an override", expression.location)
    let kind = try type(expression.type)
    try need(isPipelineSizeType(kind), .typeMismatch,
             "\(subject) override must be a supported integer", expression.location)
  }

  private func isPipelineSizeType(_ type: IRTypeKind) -> Bool {
    switch type {
    case .scalar(.i16), .scalar(.u16), .scalar(.i32), .scalar(.u32): true
    default: false
    }
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
  func validateTextureSampleOffset(_ offset: IRExpression, width: IRVectorSize?, in table: IRTable<IRExpression>) throws {
    let offsetType = width.map { IRTypeKind.vector(size: $0, scalar: .i32) } ?? .scalar(.i32)
    try need(try type(offset.type) == offsetType, .typeMismatch,
             "Texture sample offset has the wrong dimension or scalar type", offset.location)
    let expression: IRExpression
    let expressions: IRTable<IRExpression>
    if case .constant(let handle) = offset.kind {
      let constant = try module.constants.value(for: handle)
      expressions = module.globalExpressions
      expression = try expressions.value(for: constant.initializer)
    } else {
      expression = offset
      expressions = table
    }
    func validateLane(_ lane: IRExpression) throws {
      if case .zeroValue = lane.kind, try type(lane.type) == .scalar(.i32) { return }
      guard case .literal(.i32(let value)) = lane.kind, (-8...7).contains(value) else {
        try fail(.invalidType, "Texture sample offset components must be constant integers between -8 and 7",
                 offset.location)
      }
    }
    guard let width else { return try validateLane(expression) }
    let components: [IRHandle<IRExpression>]
    switch expression.kind {
    case .zeroValue: return
    case .compose(_, let lanes) where lanes.count == Int(width.rawValue): components = lanes
    case .splat(let size, let lane) where size == width: components = [lane]
    default: try fail(.invalidType, "Texture sample offset must be constant", offset.location)
    }
    for component in components {
      try validateLane(expressions.value(for: component))
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
        case .literal, .zeroValue, .compose, .constant, .override: break
        default:
          guard declared.isPipelineValueType, e.kind.pipelineOperands != nil else {
            try unsupported("Global expression requires supported pipeline operations", e.location)
          }
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
      case .override(let handle):
        let value = try module.overrides.value(for: handle)
        try need(try same(e.type, value.type), .typeMismatch, "Override reference type mismatch", e.location)
      case .runtimeValue(let handle):
        let value = try operand(handle)
        try need(try layout(e.type).constructible && layout(e.type).sized, .typeMismatch,
                 "Runtime value snapshot requires a constructible, sized type", e.location)
        try need(try same(e.type, value.type), .typeMismatch, "Runtime value snapshot type mismatch", e.location)
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
        if v.space == .handle {
          try need(try same(e.type, v.type), .typeMismatch, "Global handle type mismatch", e.location)
          break
        }
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
        if let atomic = try atomicPointee(p.type) {
          if atomic.width == 8 { try cap(.shaderInt64AtomicAllOps, e.location) }
          try match(.scalar(atomic))
          break
        }
        if case .storage(let a) = space {
          try need(a.contains(.load), .invalidAddressSpace, "Pointer cannot be loaded", e.location)
        }
        try need(try same(e.type, base), .typeMismatch, "Load type mismatch", e.location)
        try need(
          try layout(base).constructible, .typeMismatch,
          "Ordinary load requires constructible type", e.location)
      case .imageSample(let sample):
        let comparison = sample.depthReference != nil
        let imageType = try type(operand(sample.image).type)
        let depth = imageType.isDepthTexture
        let volume = imageType.isVolumeTexture
        let width: IRVectorSize = imageType.isCubeTexture || volume ? .three : .two
        let coordinateType = imageType.isLineTexture ? IRTypeKind.scalar(.f32) : .vector(size: width, scalar: .f32)
        let sampledScalar = imageType.isExternalTexture ? IRScalar.f32 : imageType.sampledTextureScalar
        guard let scalar = sampledScalar, !imageType.isMultisampled2D,
              depth ? (sample.gather == nil || sample.gather == .x) : (!comparison && (scalar == .f32 || sample.gather != nil)),
              try type(operand(sample.sampler).type) == .sampler(comparison: comparison),
              try type(operand(sample.coordinate).type) == coordinateType,
              imageType.isArrayedTexture == (sample.arrayIndex != nil)
        else { try fail(.typeMismatch, "Invalid sampled texture operation", e.location) }
        try need(!imageType.isExternalTexture || sample.clampToEdge, .typeMismatch,
                 "External texture sampling requires base clamp", e.location)
        if let arrayIndex = sample.arrayIndex {
          try need(try [.scalar(.i32), .scalar(.u32)].contains(type(operand(arrayIndex).type)),
                   .typeMismatch, "Texture array index must be i32 or u32", e.location)
        }
        if sample.clampToEdge {
          try need((imageType == .sampledFloat2D || imageType.isExternalTexture) &&
                   sample.gather == nil && sample.offset == nil && sample.level == .zero,
                   .typeMismatch, "Clamped sampling requires a float texture at the base level without gather or offset",
                   e.location)
        }
        if let reference = sample.depthReference {
          try need(try type(operand(reference).type) == .scalar(.f32),
                   .typeMismatch, "Depth comparison reference must be f32", e.location)
        }
        if sample.gather != nil {
          try need(!volume && !imageType.isLineTexture && sample.level == .zero,
                   .typeMismatch, "Invalid texture gather dimension or level", e.location)
        }
        if let offset = sample.offset {
          try need(!imageType.isCubeTexture,
                   .typeMismatch, "Texture dimension does not support texel offsets", e.location)
          try validateTextureSampleOffset(operand(offset), width: imageType.isLineTexture ? nil : width, in: table)
        }
        let levelTypes: [IRTypeKind]
        if imageType.isLineTexture {
          switch sample.level {
          case .bias: try fail(.typeMismatch, "Invalid 1D texture sample level", e.location)
          default: break
          }
        }
        switch sample.level {
        case .gradient where depth && !comparison, .bias where depth:
          try fail(.typeMismatch, "Depth sampling does not support gradients or bias", e.location)
        case .exact where depth && !comparison:
          levelTypes = [.scalar(.i32), .scalar(.u32)]
        case .gradient:
          levelTypes = [coordinateType]
        default:
          levelTypes = [.scalar(.f32)]
        }
        for operandHandle in sample.level.operands {
          try need(try levelTypes.contains(type(operand(operandHandle).type)), .typeMismatch,
                   "Invalid texture sampling level or gradients", e.location)
        }
        try match(depth && sample.gather == nil ? .scalar(scalar) : .vector(size: .four, scalar: scalar))
      case .imageLoad(let load):
        let imageType = try type(operand(load.image).type)
        let coordinate = try type(operand(load.coordinate).type)
        guard [imageType.texelCoordinateType(.i32), imageType.texelCoordinateType(.u32)].contains(coordinate) else {
          try fail(.typeMismatch, "Invalid texture load coordinates", e.location)
        }
        try need(imageType.isArrayedTexture == (load.arrayIndex != nil), .typeMismatch,
                 "Texture load array index does not match the texture dimension", e.location)
        if let arrayIndex = load.arrayIndex {
          try need(try [.scalar(.i32), .scalar(.u32)].contains(type(operand(arrayIndex).type)),
                   .typeMismatch, "Texture array index must be i32 or u32", e.location)
        }
        if let storage = imageType.storageTexture {
          try need(storage.access.contains(.load) && load.level == nil && load.sample == nil, .typeMismatch,
                   "Storage texture load requires read access and the bound mip level", e.location)
          try match(.vector(size: .four, scalar: storage.scalar))
        } else if imageType.isExternalTexture {
          try need(load.level == nil && load.sample == nil, .typeMismatch,
                   "External texture load has no mip or sample index", e.location)
          try match(.vector(size: .four, scalar: .f32))
        } else {
          let multisampled = imageType.isMultisampled2D
          guard let scalar = imageType.sampledTextureScalar, !imageType.isCubeTexture,
                (multisampled ? load.level : load.sample) == nil,
                let index = multisampled ? load.sample : load.level,
                try [.scalar(.i32), .scalar(.u32)].contains(type(operand(index).type)) else {
            try fail(.typeMismatch, "Invalid sampled texture load", e.location)
          }
          try match(imageType.isDepthTexture ? .scalar(scalar) : .vector(size: .four, scalar: scalar))
        }
      case .imageQuery(let image, let query):
        let imageType = try type(operand(image).type)
        if imageType.storageTexture != nil || imageType.isExternalTexture {
          switch query {
          case .size(nil): try match(imageType.texelCoordinateType(.u32))
          case .numLayers where imageType.isArrayedTexture: try match(.scalar(.u32))
          default: try fail(.typeMismatch, "Invalid storage or external texture query", e.location)
          }
          break
        }
        try need(imageType.isSampledTexture, .typeMismatch, "Expected a sampled texture", e.location)
        switch query {
        case .size(let level):
          if let level {
            try need(try !imageType.isMultisampled2D && [.scalar(.i32), .scalar(.u32)].contains(type(operand(level).type)),
                     .typeMismatch, "Texture level must be i32 or u32", e.location)
          }
          try match(imageType.texelCoordinateType(.u32))
        case .numLevels where !imageType.isMultisampled2D, .numSamples where imageType.isMultisampled2D,
             .numLayers where imageType.isArrayedTexture:
          try match(.scalar(.u32))
        default: try fail(.typeMismatch, "Invalid sampled texture query", e.location)
        }
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
          case (.scalar(let scalar), .scalar(.u32)):
            valid = [.i16, .u16, .i32, .u32, .i64, .u64].contains(scalar)
          case (.vector(let a, let scalar), .vector(let b, .u32)):
            valid = a == b && [.i16, .u16, .i32, .u32, .i64, .u64].contains(scalar)
          default: valid = false
          }
          try need(
            valid, .typeMismatch, "Shifts require an integer and a matching u32 count", e.location)
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
          try fail(.typeMismatch, "Cast needs a scalar, vector or matrix", e.location)
        }
        if input.kind == .bool || target.kind == .bool {
          try need(input.isWGSLConversionScalar && target.isWGSLConversionScalar, .typeMismatch,
                   "Boolean conversion requires a supported scalar type", e.location)
        }
        switch try type(x.type) {
        case .scalar: try match(.scalar(target))
        case .vector(let size, _): try match(.vector(size: size, scalar: target))
        case .matrix(let columns, let rows, _) where [.f16, .f32, .f64].contains(input) && [.f16, .f32, .f64].contains(target):
          try match(.matrix(columns: columns, rows: rows, scalar: target))
        default: try fail(.typeMismatch, "Matrix cast needs a floating-point destination", e.location)
        }
      case .bitcast(let source, let target):
        let x = try operand(source)
        try need(try same(target, e.type), .typeMismatch, "Bitcast target mismatch", e.location)
        guard let a = try IRBitcastShape(type(x.type)), let b = IRBitcastShape(declared), a.permits(b) else {
          try fail(.typeMismatch, "Bitcast width/type mismatch", e.location)
        }
      case .arrayLength(let pointer):
        let value = try operand(pointer)
        guard case .pointer(let array, .storage) = try type(value.type),
              case .array(_, .dynamic, _) = try type(array) else {
          try fail(.typeMismatch, "arrayLength requires a pointer to a runtime storage array", e.location)
        }
        try match(.scalar(.u32))
      case .relational(let fn, let arg) where fn.isBooleanReduction:
        let operand = try operand(arg)
        try need(try fn.accepts(type(operand.type)), .typeMismatch,
                 "Boolean reduction requires a bool scalar or vector", e.location)
        try match(.scalar(.bool))
      case .math(let fn, let arg, let arg1, let arg2, let arg3):
        let a = try operand(arg)
        if fn == .ldexp {
          let mantissa = try type(a.type)
          guard let exponent = arg1, arg2 == nil, arg3 == nil,
                let expected = mantissa.ldexpExponentType, try type(operand(exponent).type) == expected else {
            try fail(.typeMismatch, "ldexp needs matching floating mantissa and i32 exponent shapes", e.location)
          }
          try match(mantissa)
          break
        }
        if fn.decomposesFloat {
          try validateDecompositionCall(fn, operand: a.type, result: e.type,
                                        hasExtraArguments: arg1 != nil || arg2 != nil || arg3 != nil, at: e.location)
          break
        }
        if let packing = fn.packingBuiltin {
          try need(try type(a.type) == packing.argumentType && arg1 == nil && arg2 == nil && arg3 == nil,
                   .typeMismatch, "Packing builtin argument type or arity mismatch", e.location)
          try match(packing.resultType)
          break
        }
        if let arity = fn.geometricArity {
          let supplied = [arg1, arg2, arg3]
          guard let signature = fn.geometricSignature(for: try type(a.type)),
                supplied.enumerated().allSatisfy({ ($0.offset < arity - 1) == ($0.element != nil) }) else {
            try fail(.typeMismatch, "Invalid geometric builtin operand or arity", e.location)
          }
          for (extra, expected) in zip(supplied.compactMap({ $0 }), signature.arguments.dropFirst()) {
            try need(try type(operand(extra).type) == expected, .typeMismatch, "Geometric builtin argument type mismatch", e.location)
          }
          try match(signature.result)
          break
        }
        guard let scalar = try scalar(a.type) else {
          try fail(.typeMismatch, "Math needs numeric scalar/vector", e.location)
        }
        if case .matrix = try type(a.type) {
          try unsupported("Matrix builtin overload not implemented", e.location)
        }
        if let arity = fn.integerBitArity {
          let arguments = [arg1, arg2, arg3]
          try need([.i16, .u16, .i32, .u32, .i64, .u64].contains(scalar) &&
                   arguments.enumerated().allSatisfy { ($0.offset < arity - 1) == ($0.element != nil) },
                   .typeMismatch, "Invalid integer bit builtin type or arity", e.location)
          if arity == 4, let inserted = arg1 {
            try need(try same(operand(inserted).type, a.type), .typeMismatch, "Bit insertion value type mismatch", e.location)
          }
          if arity > 1 {
            for bound in arguments[(arity - 3)..<(arity - 1)].compactMap({ $0 }) {
              try need(try type(operand(bound).type) == .scalar(.u32), .typeMismatch, "Bit range must be u32", e.location)
            }
          }
          try need(try same(e.type, a.type), .typeMismatch, "Bit builtin result type mismatch", e.location)
          break
        }
        if let numeric = fn.numericBuiltin {
          let operands = [arg1, arg2, arg3]
          try need(numeric.accepts(scalar) &&
                   operands.enumerated().allSatisfy({ ($0.offset < numeric.arity - 1) == ($0.element != nil) }),
                   .typeMismatch, "Numeric builtin scalar type or arity mismatch", e.location)
          for (index, operandHandle) in operands.compactMap({ $0 }).enumerated() {
            let operandType = try operand(operandHandle).type
            if numeric.scalarLastArgument, index == numeric.arity - 2, try type(operandType) == .scalar(scalar) {
              continue
            }
            try need(try same(operandType, a.type), .typeMismatch, "Numeric builtin operand type mismatch", e.location)
          }
          try need(try same(e.type, a.type), .typeMismatch, "Numeric builtin result type mismatch", e.location)
          break
        }
        try unsupported("Math builtin overload rules not implemented", e.location)
      case .callResult(let target):
        guard let currentFunction, target.index < currentFunction else {
          try fail(.invalidReference, "Calls must target earlier functions", e.location)
        }
        let target = try module.functions.value(for: target)
        guard let r = target.result else {
          try fail(.typeMismatch, "Void call has no result", e.location)
        }
        try need(try same(e.type, r.type), .typeMismatch, "Call result type mismatch", e.location)
      case .atomicResult(let resultType, let comparison):
        try need(currentFunction != nil, .invalidReference, "Atomic result requires a function", e.location)
        if comparison {
          try need(e.type == resultType && atomicCompareResults[resultType] != nil,
                   .typeMismatch, "Atomic comparison needs its predeclared result type", e.location)
        } else {
          let resultKind = try type(resultType)
          try need([IRTypeKind.scalar(.i32), .scalar(.u32), .scalar(.i64), .scalar(.u64), .scalar(.f32)].contains(resultKind),
                   .typeMismatch, "Atomic result must be a supported numeric scalar", e.location)
          try match(resultKind)
        }
      case .workgroupUniformLoadResult(let resultType):
        try need(currentFunction != nil && (try same(e.type, resultType)), .typeMismatch,
                 "Invalid workgroup uniform load result", e.location)
      case .subgroupOperationResult(let resultType):
        try need(currentFunction != nil && (try same(e.type, resultType)), .typeMismatch,
                 "Invalid subgroup operation result", e.location)
      case .subgroupBallotResult:
        try need(currentFunction != nil, .invalidReference, "Subgroup ballot requires a function", e.location)
        try match(.vector(size: .four, scalar: .u32))
      case .cooperativeLoad(let columns, let rows, let role, let data):
        try cap(.cooperativeMatrix, e.location)
        guard case .cooperativeMatrix(let tc, let tr, let scalar, let trl) = declared else {
          try fail(.typeMismatch, "coopLoad result must be a cooperative matrix", e.location)
        }
        try need(tc == columns && tr == rows && trl == role, .typeMismatch, "coopLoad result type mismatch", e.location)
        let pointer = try operand(data.pointer)
        let stride = try operand(data.stride)
        guard case .pointer(let pointee, let space) = try type(pointer.type),
              try type(pointee) == .scalar(scalar) else {
          try fail(.typeMismatch, "Cooperative matrix memory requires a matching scalar pointer", e.location)
        }
        switch space {
        case .storage(let access): try need(access.contains(.load), .invalidAddressSpace, "Cooperative load requires readable memory", e.location)
        case .function, .private, .workgroup, .uniform, .immediate: break
        default: try fail(.invalidAddressSpace, "Invalid cooperative matrix address space", e.location)
        }
        try need([.scalar(.i32), .scalar(.u32)].contains(type(stride.type)), .typeMismatch,
                 "Cooperative matrix stride must be i32 or u32", e.location)
        try need(scalar == .f16 || scalar == .f32, .typeMismatch, "Cooperative matrices require f16 or f32", e.location)
      case .cooperativeMultiplyAdd(let a, let b, let c):
        let aType = try type(operand(a).type), bType = try type(operand(b).type), cType = try type(operand(c).type)
        guard case .cooperativeMatrix(let ac, let ar, let ascalar, .a) = aType,
              case .cooperativeMatrix(let bc, let br, let bscalar, .b) = bType,
              case .cooperativeMatrix(let cc, let cr, let cscalar, .c) = cType else {
          try fail(.typeMismatch, "coopMultiplyAdd requires A, B, and C cooperative matrix roles", e.location)
        }
        try need(ac == br && ar == cr && bc == cc && ascalar == bscalar &&
                 (cscalar == ascalar || (ascalar == .f16 && cscalar == .f32)), .typeMismatch,
                 "Incompatible cooperative matrix dimensions or scalar types", e.location)
        try need(try type(e.type) == cType, .typeMismatch, "coopMultiplyAdd result must match C", e.location)
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
    case .array(let t, let n, _), .bindingArray(let t, let n):
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
  func nonUniformBindingArrayCapability(for handle: IRHandle<IRType>) throws -> ShaderCapabilities? {
    switch try type(handle) {
    case .bindingArray(let element, _):
      return try type(element).storageTexture == nil
        ? .textureAndSamplerBindingArrayNonUniformIndexing
        : .storageTextureBindingArrayNonUniformIndexing
    case .pointer(let base, let space):
      guard case .bindingArray = try type(base) else { return nil }
      switch space {
      case .uniform: return .bufferBindingArrayNonUniformIndexing
      case .storage: return .storageBufferBindingArrayNonUniformIndexing
      default: return nil
      }
    default: return nil
    }
  }

  mutating func function(_ f: IRFunction, index: Int, entryIndices: Set<Int>, computeEntry: Bool) throws {
    try text(f.name)
    try budget.consume(.nodes)
    try budget.consume(.work)
    if let filter = f.diagnosticFilter { try validateDiagnosticScope(filter, parent: module.diagnosticFilter) }
    for a in f.arguments {
      try text(a.name)
      try budget.consume(.work)
      let kind = try type(a.type)
      if case .pointer(let base, let space) = kind {
        if case .bindingArray = try type(base) {
          try fail(.invalidType, "Functions cannot take pointers to entire binding arrays", a.location)
        }
        try need(!entryIndices.contains(index) && (module.allowsPointerSubviews || space == .function || space == .private),
                 .invalidAddressSpace, "Shared pointer parameters require unrestricted_pointer_parameters", a.location)
      } else if kind.isTextureOrSampler {
        try need(!entryIndices.contains(index), .invalidType, "Entry points cannot take texture or sampler parameters", a.location)
      } else {
        try need(
          try layout(a.type).constructible && layout(a.type).sized, .invalidType,
          "Function arguments must be sized data", a.location)
      }
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
    uniformity = FunctionUniformity(computeEntry: computeEntry)
    activeSynchronizedLoads = synchronizedWorkgroupLoads[index] ?? []
    var stages = allStages
    var immediates: Set<IRHandle<IRGlobalVariable>> = []
    var resourceAliases: Set<IRHandle<IRGlobalVariable>> = []
    for e in f.expressions.values {
      try budget.consume(.work)
      switch e.kind {
      case .cooperativeLoad, .cooperativeMultiplyAdd: stages.formIntersection([.compute])
      default: break
      }
      if requiresImplicitDerivatives(e.kind) {
        stages.formIntersection([.fragment])
      }
      if case .globalVariable(let g) = e.kind {
        let global = try module.globalVariables.value(for: g)
        if let binding = global.binding, reusedResourceBindings.contains(binding) { resourceAliases.insert(g) }
        if global.space == .immediate, immediates.count < 2 { immediates.insert(g) }
        if global.space == .workgroup { stages.formIntersection([.compute]) }
        if !allowsVertexWritableStorage {
          if case .storage(let access) = global.space, access.contains(.store) {
            stages.formIntersection([.compute, .fragment])
          }
          if try type(bindingArrayElement(of: global)).storageTexture?.access.contains(.store) == true {
            stages.formIntersection([.compute, .fragment])
          }
        }
      }
    }
    for s in f.statements.values {
      try budget.consume(.work)
      switch s.kind {
      case .kill, .demoteToHelperInvocation: stages.formIntersection([.fragment])
      case .subgroupGather, .subgroupBallot, .subgroupCollectiveOperation:
        try cap(.subgroup, s.location)
        stages.formIntersection([.compute, .fragment])
      case .imageStore, .imageAtomic:
        if !allowsVertexWritableStorage { stages.formIntersection([.compute, .fragment]) }
      case .controlBarrier, .memoryBarrier, .workgroupUniformLoad:
        stages.formIntersection([.compute])
      case .cooperativeStore:
        stages.formIntersection([.compute])
      case .call(let target, _, _):
        try need(
          target.index < index, .invalidReference, "Recursive/forward function calls forbidden",
          s.location)
        try need(!entryIndices.contains(target.index), .invalidControlFlow, "Cannot call an entry point", s.location)
        stages.formIntersection(functionStages[target.index])
        let calleeAliases = functionResourceAliases[target.index]
        try budget.consume(.work, amount: calleeAliases.count)
        resourceAliases.formUnion(calleeAliases)
        for immediate in functionImmediates[target.index] where immediates.count < 2 {
          immediates.insert(immediate)
        }
      default: break
      }
    }
    var available = Set<IRHandle<IRExpression>>()
    let isEntry = entryIndices.contains(index)
    var uniform: [UniformValue] = []
    for (index, expression) in f.expressions.values.enumerated() {
      try budget.consume(.work)
      let handle = try f.expressions.handle(at: index)
      switch expression.kind {
      case .literal, .constant, .override, .zeroValue, .localVariable:
        available.insert(handle)
        uniform.append(.uniform)
      case .functionArgument(let argument):
        available.insert(handle)
        if try type(expression.type).isTextureOrSampler {
          uniform.append(.uniform)
        } else {
          uniform.append(isEntry ? uniformComputeBuiltin(f.arguments[Int(argument)].binding) : .argument(argument))
        }
      case .globalVariable(let global):
        available.insert(handle)
        let variable = try module.globalVariables.value(for: global)
        switch variable.space {
        case .workgroup, .handle, .uniform, .immediate: uniform.append(.uniform)
        case .storage(let access):
          uniform.append(access.contains(.store) || !variable.memoryDecorations.isEmpty ? .varying : .uniform)
        default: uniform.append(.varying)
        }
      default:
        uniform.append(.varying)
      }
    }
    var locals = Array(repeating: UniformValue.uniform, count: f.localVariables.count)
    var seenBlocks = Set<IRHandle<IRBlock>>()
    var seenStatements = Set<IRHandle<IRStatement>>()
    var results = Set<IRHandle<IRExpression>>()
    let result = try block(
      f.body, f: f, available: available, uniform: &uniform, locals: &locals, controlUniform: .uniform, active: [],
      seen: &seenBlocks, statements: &seenStatements, callResults: &results,
      loopDepth: 0, inContinuing: false)
    for (node, check) in uniformity.pendingChecks {
      try checkUniformity(node, check)
    }
    if !synchronizedWorkgroupLoads.isEmpty {
      var proof = SynchronizedWorkgroupProof()
      for (handle, node) in uniformity.synchronizedLoads {
        if let parameters = try uniformity.parameters(of: node, budget: &budget) {
          proof.loads[handle] = parameters
        }
      }
      for call in uniformity.synchronizedCalls {
        let arguments = try call.arguments.map { try uniformity.parameters(of: $0, budget: &budget) }
        proof.calls.append((call.target, arguments))
      }
      synchronizedWorkgroupProofs[index] = proof
    }
    if f.result != nil {
      try need(!result.flow.contains(.next), .invalidControlFlow, "Not all function paths return a value", f.location)
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
      switch e.kind {
      case .callResult, .atomicResult, .workgroupUniformLoadResult, .subgroupOperationResult, .subgroupBallotResult:
        try need(
          results.contains(try f.expressions.handle(at: i)), .invalidReference,
          "Statement result has no matching producer", e.location)
      default: break
      }
    }
    var requiresUniform = uniformity.controlRequirements
    requiresUniform.arguments = uniformity.arguments
    if f.result != nil {
      requiresUniform.returns = try .init(
        whole: uniformity.parameters(of: uniformity.returns.whole, budget: &budget),
        subgroup: uniformity.parameters(of: uniformity.returns.subgroup, budget: &budget))
    }
    functionStages.append(stages)
    functionImmediates.append(immediates)
    functionResourceAliases.append(resourceAliases)
    functionUniform.append(requiresUniform)
    if collectDescriptorUniformity, !uniformity.nonUniformDescriptorAccesses.isEmpty {
      nonUniformDescriptorAccesses[index] = uniformity.nonUniformDescriptorAccesses
    }
  }

  func atomicPointee(_ handle: IRHandle<IRType>) throws -> IRScalar? {
    guard case .pointer(let base, let space) = try type(handle),
          case .atomic(let scalar) = try type(base) else { return nil }
    switch space {
    case .workgroup: return scalar
    case .storage(let access) where access.contains([.load, .store]): return scalar
    default: try fail(.invalidAddressSpace, "Atomic operation requires workgroup or read-write storage")
    }
  }

  mutating func provenSynchronizedWorkgroupLoads() throws -> [Int: Set<IRHandle<IRExpression>>] {
    var uniformArguments: [Int: Set<Int>] = [:]
    for entry in module.entryPoints { uniformArguments[entry.function.index] = [] }
    var result: [Int: Set<IRHandle<IRExpression>>] = [:]
    // Each callee precedes its callers. A formal is uniform only at every reachable call site.
    for index in module.functions.values.indices.reversed() {
      try budget.consume(.work)
      guard let proof = synchronizedWorkgroupProofs[index], let formals = uniformArguments[index] else { continue }
      for (handle, dependencies) in proof.loads {
        try budget.consume(.work, amount: dependencies.count + 1)
        if dependencies.isSubset(of: formals) { result[index, default: []].insert(handle) }
      }
      for call in proof.calls {
        try budget.consume(.work)
        var incoming: Set<Int> = []
        for (argument, dependencies) in call.arguments.enumerated() {
          try budget.consume(.work, amount: (dependencies?.count ?? 0) + 1)
          if let dependencies, dependencies.isSubset(of: formals) { incoming.insert(argument) }
        }
        if let previous = uniformArguments[call.target] { incoming.formIntersection(previous) }
        uniformArguments[call.target] = incoming
      }
    }
    return result
  }
  mutating func validatePredeclaredTypes() throws {
    var handles: Set<IRHandle<IRType>> = []
    var kinds: [IRPredeclaredType] = []
    for entry in module.specialTypes.predeclared {
      try budget.consume(.work)
      try need(handles.insert(entry.type).inserted && !kinds.contains(entry.kind),
               .invalidType, "Duplicate predeclared type")
      kinds.append(entry.kind)
      if let shape = entry.kind.decomposition {
        try validateDecompositionType(entry.type, function: shape.function, size: shape.size, scalar: shape.scalar)
        continue
      }
      guard case .atomicCompareExchangeWeakResult(let scalar) = entry.kind else {
        try unsupported("Predeclared type validation not implemented")
      }
      let width = UInt32(scalar.width)
      guard [.i32, .u32, .i64, .u64].contains(scalar),
            case .structure(let members, let span) = try type(entry.type), span == width * 2, members.count == 2,
            members[0].name == "old_value", members[0].offset == 0, members[0].binding == nil,
            members[1].name == "exchanged", members[1].offset == width, members[1].binding == nil,
            try type(members[0].type) == .scalar(scalar), try type(members[1].type) == .scalar(.bool)
      else { try fail(.invalidType, "Invalid predeclared atomic comparison result") }
      atomicCompareResults[entry.type] = scalar
    }
  }
  func validateDecompositionType(
    _ handle: IRHandle<IRType>, function: IRMathFunction, size: IRVectorSize?, scalar: IRScalar
  ) throws {
    func shape(_ lane: IRScalar) -> IRTypeKind {
      size.map { .vector(size: $0, scalar: lane) } ?? .scalar(lane)
    }
    let stride = function.decompositionMemberStride(size: size, scalar: scalar)
    guard scalar == .f32 || scalar == .f16,
          case .structure(let members, let span) = try type(handle), members.count == 2, span == 2 * stride,
          members[0].name == "fract", members[0].offset == 0, members[0].binding == nil,
          members[1].name == function.decompositionSecondMember, members[1].offset == stride, members[1].binding == nil,
          try type(members[0].type) == shape(scalar),
          try type(members[1].type) == shape(function.decompositionSecondScalar(scalar))
    else { try fail(.invalidType, "Invalid predeclared float decomposition result") }
  }

  func validateDecompositionCall(
    _ function: IRMathFunction, operand: IRHandle<IRType>, result: IRHandle<IRType>,
    hasExtraArguments: Bool, at location: ShaderLocation?
  ) throws {
    guard let shape = try type(operand).concreteFloatShape else {
      try fail(.typeMismatch, "Float decomposition needs a floating scalar or vector", location)
    }
    let kind = function.decompositionResult(size: shape.size, scalar: shape.scalar)
    try need(!hasExtraArguments && module.specialTypes.predeclared.contains(.init(kind: kind, type: result)),
             .typeMismatch, "Float decomposition result or arity mismatch", location)
  }

  func requiresImplicitDerivatives(_ kind: IRExpressionKind) -> Bool {
    switch kind {
    case .derivative: return true
    case .imageSample(let sample):
      switch sample.level {
      case .auto, .bias: return true
      default: return false
      }
    default: return false
    }
  }

  mutating func validateSubgroupStatement(
    _ kind: IRStatementKind, function: IRFunction, available: inout Set<IRHandle<IRExpression>>,
    uniform: inout [UniformValue], control: UniformValue, severity: IRDiagnosticSeverity,
    producedResults: inout Set<IRHandle<IRExpression>>, location: ShaderLocation?
  ) throws {
    func use(_ handle: IRHandle<IRExpression>, _ loc: ShaderLocation?) throws -> IRExpression {
      try need(available.contains(handle), .invalidReference, "Expression is outside evaluation scope", loc)
      return try function.expressions.value(for: handle)
    }
    let loc = location
    switch kind {
    case .subgroupBallot(let result, let predicate):
      if let predicate {
        try need(try type(use(predicate, loc).type) == .scalar(.bool), .typeMismatch, "Invalid subgroup ballot predicate", loc)
      }
      let produced = try function.expressions.value(for: result)
      guard case .subgroupBallotResult = produced.kind, producedResults.insert(result).inserted else {
        try fail(.invalidReference, "Incorrect or duplicate subgroup ballot result", loc)
      }
      uniform[result.index] = try subgroupResultUniformity(predicate.map { [$0] } ?? [], uniform: uniform, control: control)
      available.insert(result)
    case .subgroupCollectiveOperation(let operation, let collective, let argument, let result):
      let value = try use(argument, loc)
      try need(try operation.accepts(type(value.type), collective: collective), .typeMismatch,
               "Invalid subgroup collective operand", loc)
      let produced = try function.expressions.value(for: result)
      guard case .subgroupOperationResult(let resultType) = produced.kind,
            try same(value.type, resultType), producedResults.insert(result).inserted else {
        try fail(.invalidReference, "Incorrect or duplicate subgroup collective result", loc)
      }
      if collective == .reduce {
        uniform[result.index] = try subgroupResultUniformity([argument], uniform: uniform, control: control)
      }
      available.insert(result)
    case .subgroupGather(let mode, let argument, let result):
      let value = try use(argument, loc)
      switch try type(value.type) {
      case .scalar(let scalar), .vector(_, let scalar):
        try need([.u32, .i32, .f32, .f16].contains(scalar), .typeMismatch, "Invalid subgroup gather operand", loc)
      default:
        try fail(.typeMismatch, "Invalid subgroup gather operand", loc)
      }
      if let index = mode.index {
        let operand = try use(index, loc)
        let indexType = try type(operand.type)
        let indexLimit: UInt32
        switch mode {
        case .quadBroadcast, .quadBroadcastDynamic: indexLimit = 4
        default: indexLimit = 128
        }
        switch mode {
        case .quadBroadcastDynamic:
          try need(indexType == .scalar(.u32), .typeMismatch, "Invalid dynamic quad selector type", loc)
        case .shuffle:
          try need([.scalar(.i32), .scalar(.u32)].contains(indexType), .typeMismatch, "Invalid shuffle index type", loc)
        case .broadcast, .quadBroadcast:
          switch operand.kind {
          case .literal(.u32), .literal(.i32): break
          default:
            try fail(.invalidType, "Broadcast requires a constant index", loc)
          }
          try need([.scalar(.i32), .scalar(.u32)].contains(indexType), .typeMismatch, "Invalid broadcast index type", loc)
        default:
          try need(indexType == .scalar(.u32), .typeMismatch, "Invalid relative shuffle selector type", loc)
          try requireUniformity(uniform[index.index], rule: .subgroup, severity: severity,
                               message: "Relative shuffle selector must be uniform", at: operand.location ?? loc)
        }
        switch operand.kind {
        case .literal(.u32(let value)): try need(value < indexLimit, .invalidType, "Subgroup selector is out of range", loc)
        case .literal(.i32(let value)): try need((0..<Int32(indexLimit)).contains(value), .invalidType,
                                                "Subgroup selector is out of range", loc)
        default: break
        }
      }
      let produced = try function.expressions.value(for: result)
      guard case .subgroupOperationResult(let resultType) = produced.kind,
            try same(value.type, resultType), producedResults.insert(result).inserted else {
        try fail(.invalidReference, "Incorrect or duplicate subgroup gather result", loc)
      }
      switch mode {
      case .broadcastFirst:
        uniform[result.index] = try subgroupResultUniformity([argument], uniform: uniform, control: control)
      case .broadcast(let index):
        uniform[result.index] = try subgroupResultUniformity([argument, index], uniform: uniform, control: control)
      default: break
      }
      available.insert(result)
    default:
      throw ShaderFailure.invalidHandle
    }
  }

  // Keep call-specific state out of the recursive block frame.
  mutating func validateCall(
    _ target: IRHandle<IRFunction>, arguments: [IRHandle<IRExpression>], result: IRHandle<IRExpression>?,
    function: IRFunction, available: inout Set<IRHandle<IRExpression>>, producedResults: inout Set<IRHandle<IRExpression>>,
    uniform: inout [UniformValue], locals: inout [UniformValue], uniformControl: UniformValue, at location: ShaderLocation?
  ) throws {
    let callee = try module.functions.value(for: target)
    try need(arguments.count == callee.arguments.count, .typeMismatch, "Call argument count mismatch", location)
    for (value, parameter) in zip(arguments, callee.arguments) {
      try budget.consume(.work)
      try need(available.contains(value), .invalidReference, "Expression is outside evaluation scope", location)
      let operand = try function.expressions.value(for: value)
      let matches: Bool
      switch (try type(operand.type), try type(parameter.type)) {
      case (.image(let dimension, let arrayed, .storage(let format, let access)),
            .image(let expectedDimension, let expectedArrayed, .storage(let expectedFormat, let expectedAccess))):
        matches = dimension == expectedDimension && arrayed == expectedArrayed && format == expectedFormat
          && access.contains(expectedAccess)
      default:
        matches = try same(operand.type, parameter.type)
      }
      try need(matches, .typeMismatch, "Call argument type mismatch", location)
    }
    let requirements = functionUniform[target.index]
    if reachable, !synchronizedWorkgroupLoads.isEmpty {
      try budget.consume(.nodes, amount: arguments.count + 1)
      uniformity.synchronizedCalls.append((target.index, arguments.map { uniform[$0.index].whole }))
    }
    if requirements.barrier {
      try requireUniformControl(uniformControl, rule: .barrier, severity: .error,
                                message: "Uniformity-requiring call in divergent control flow", at: location)
    }
    try requireUniformControl(uniformControl, rule: .derivative, severity: requirements.derivative.severity,
                              message: "Uniformity-requiring call in divergent control flow",
                              at: requirements.derivative.location ?? location)
    try requireUniformControl(uniformControl, rule: .subgroup, severity: requirements.subgroup.severity,
                              message: "Subgroup call in divergent control flow", at: requirements.subgroup.location ?? location)
    try requireUniformControl(uniformControl, rule: .cooperativeMatrix, severity: requirements.cooperativeMatrix.severity,
                              message: "Cooperative matrix call in divergent control flow",
                              at: requirements.cooperativeMatrix.location ?? location)
    for (parameter, requirement) in requirements.arguments.sorted(by: {
      ($0.key.index, $0.key.rule.rawValue) < ($1.key.index, $1.key.rule.rawValue)
    }) {
      try budget.consume(.work)
      let argument = arguments[parameter.index]
      let operand = try function.expressions.value(for: argument)
      let value: UniformValue
      if parameter.rule == .cooperativeMatrix, case .pointer = try type(operand.type) {
        value = try cooperativePointerUniformity(argument, in: function, uniform: uniform)
      } else {
        value = uniform[argument.index]
      }
      try requireUniformity(value, rule: parameter.rule, severity: requirement.severity,
                            message: "Uniformity-requiring call needs a uniform argument", at: operand.location ?? location)
    }
    if let result, callee.result != nil {
      let expression = try function.expressions.value(for: result)
      try need(expression.kind == .callResult(target) && producedResults.insert(result).inserted,
               .invalidReference, "Incorrect/duplicate call result", location)
      let summary = requirements.returns
      let value = try UniformValue(
        whole: returnDependency(summary.whole, arguments: arguments, uniform: uniform, scope: .whole),
        subgroup: returnDependency(summary.subgroup, arguments: arguments, uniform: uniform, scope: .subgroup))
      let active = try mergeUniformity(uniformControl, uniformity.exits)
      uniform[result.index] = try mergeUniformity(active, value)
      available.insert(result)
    } else {
      try need(callee.result == nil && result == nil, .typeMismatch, "Missing/unexpected call result", location)
    }
    for parameter in functionPointerWrites[target.index, default: []].sorted() {
      try budget.consume(.work)
      if let reference = try localReference(arguments[parameter], in: function) {
        locals[reference.index] = .varying
      }
    }
  }

  // Keep leaf validation out of the recursive block frame.
  mutating func validateLeafStatement(
    _ s: IRStatement, f: IRFunction, available: inout Set<IRHandle<IRExpression>>,
    uniform: inout [UniformValue], locals: inout [UniformValue], ctrl: UniformValue,
    callResults: inout Set<IRHandle<IRExpression>>,
    loopDepth: Int, inContinuing: Bool, switchDepth: Int
  ) throws -> StatementFlow {
    let loc = s.location
    let severities = diagnosticLevels(for: f)
    var step: StatementFlow = .next
    func use(_ handle: IRHandle<IRExpression>, _ location: ShaderLocation?) throws -> IRExpression {
      try need(available.contains(handle), .invalidReference, "Expression is outside evaluation scope", location)
      return try f.expressions.value(for: handle)
    }
    switch s.kind {
    case .emit(let xs):
      for x in xs {
        try budget.consume(.work)
        let e = try f.expressions.value(for: x)
        try need(!available.contains(x), .invalidReference, "Duplicate or invalid emit", loc)
        switch e.kind {
        case .callResult, .atomicResult, .workgroupUniformLoadResult, .subgroupOperationResult, .subgroupBallotResult:
          try fail(.invalidReference, "Statement result cannot be produced by emit", loc)
        default: break
        }
        for dep in e.kind.operands { _ = try use(dep, loc) }
        switch e.kind {
        case .cooperativeLoad(_, _, _, let data):
          try requireUniformControl(ctrl, rule: .cooperativeMatrix, severity: .error,
                                    message: "Cooperative matrices require uniform control flow", at: e.location ?? loc)
          let address = try cooperativePointerUniformity(data.pointer, in: f, uniform: uniform)
          for operand in [address, uniform[data.stride.index]] {
            try requireUniformity(operand, rule: .cooperativeMatrix, severity: .error,
                                  message: "Cooperative matrix address and stride must be uniform", at: e.location ?? loc)
          }
        case .cooperativeMultiplyAdd:
          try requireUniformControl(ctrl, rule: .cooperativeMatrix, severity: .error,
                                    message: "Cooperative matrices require uniform control flow", at: e.location ?? loc)
        default: break
        }
        if requiresImplicitDerivatives(e.kind) {
          try requireUniformControl(ctrl, rule: .derivative, severity: severities.derivative,
                                    message: "Operation requiring derivatives in non-uniform control flow", at: e.location ?? loc)
        }
        uniform[x.index] = try expressionUniformity(e, handle: x, in: f, uniform: uniform, locals: locals)
        available.insert(x)
      }
    case .returnValue(let x):
      try need(!inContinuing, .invalidControlFlow, "Return forbidden in continuing", loc)
      if let r = f.result, let x {
        try need(try same(use(x, loc).type, r.type), .typeMismatch, "Return type mismatch", loc)
        if reachable {
          let active = try mergeUniformity(ctrl, uniformity.exits)
          let result = try mergeUniformity(active, uniform[x.index])
          uniformity.returns = try mergeUniformity(uniformity.returns, result)
        }
      } else {
        try need(f.result == nil && x == nil, .typeMismatch, "Return value mismatch", loc)
      }
      if reachable { uniformity.exits = try mergeUniformity(uniformity.exits, ctrl) }
      step = .exit
    case .imageStore(let image, let coordinate, let arrayIndex, let value),
         .imageAtomic(let image, let coordinate, let arrayIndex, _, let value):
      let coordinateType = try type(use(coordinate, loc).type)
      let imageExpression = try use(image, loc)
      let imageType = try type(imageExpression.type)
      guard let storage = imageType.storageTexture, storage.access.contains(.store),
            imageType.isArrayedTexture == (arrayIndex != nil),
            [imageType.texelCoordinateType(.i32), imageType.texelCoordinateType(.u32)].contains(coordinateType) else {
        try fail(.typeMismatch, "Invalid storage texture store", loc)
      }
      let valueType: IRTypeKind
      if case .imageAtomic(_, _, _, let operation, _) = s.kind {
        try cap(storage.scalar == .u64 ? .textureInt64Atomic : .textureAtomic, loc)
        try need(storage.access.contains(.atomic), .typeMismatch, "Texture atomic operation requires atomic access", loc)
        switch operation {
        case .min, .max: break
        case .add, .and, .inclusiveOr, .exclusiveOr:
          try need(storage.scalar != .u64, .typeMismatch, "R64Uint atomics support min and max", loc)
        default: try fail(.typeMismatch, "Invalid texture atomic operation", loc)
        }
        let root: IRExpressionKind
        switch imageExpression.kind {
        case .access(let base, _), .accessIndex(let base, _): root = try use(base, loc).kind
        default: root = imageExpression.kind
        }
        switch (root, imageExpression.kind) {
        case (.globalVariable, _), (_, .functionArgument): break
        default: try fail(.invalidReference, "Texture atomics require a texture binding or parameter", loc)
        }
        valueType = .scalar(storage.scalar)
      } else {
        valueType = .vector(size: .four, scalar: storage.scalar)
      }
      try need(try type(use(value, loc).type) == valueType, .typeMismatch, "Invalid storage texture value", loc)
      if let arrayIndex {
        try need(try [.scalar(.i32), .scalar(.u32)].contains(type(use(arrayIndex, loc).type)),
                 .typeMismatch, "Texture array index must be i32 or u32", loc)
      }
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
      if let atomic = try atomicPointee(p.type) {
        if atomic.width == 8 { try cap(.shaderInt64AtomicAllOps, loc) }
        try need(try type(v.type) == .scalar(atomic), .typeMismatch, "Atomic store value type mismatch", loc)
      } else {
        try need(
          try layout(base).constructible, .typeMismatch,
          "Ordinary store requires constructible type", loc)
        try need(try same(base, v.type), .typeMismatch, "Store value type mismatch", loc)
      }
      if space == .function, let reference = try localReference(pointer, in: f) {
        let active = try mergeUniformity(ctrl, uniformity.exits)
        let assignment = try mergeUniformity(mergeUniformity(active, uniform[pointer.index]), uniform[value.index])
        locals[reference.index] = reference.partial ? try mergeUniformity(locals[reference.index], assignment) : assignment
      }
    case .atomic(let pointer, let operation, let value, let result):
      let p = try use(pointer, loc)
      let v = try use(value, loc)
      guard let atomic = try atomicPointee(p.type) else {
        try fail(.typeMismatch, "Atomic operation requires a pointer to an atomic integer", loc)
      }
      try need(try type(v.type) == .scalar(atomic), .typeMismatch, "Atomic operand type mismatch", loc)
      if operation == .addFloat32Bits {
        try need(atomic == .u32, .typeMismatch, "Float32 bitwise atomic add requires atomic<u32>", loc)
        guard case .pointer(_, .storage) = try type(p.type) else {
          try fail(.invalidAddressSpace, "Float atomic read-modify-write requires storage memory", loc)
        }
        try cap(.shaderFloat32Atomic, loc)
      }
      if operation == .minWithOppositeSignedness || operation == .maxWithOppositeSignedness {
        try need(atomic.kind == .sint || atomic.kind == .uint, .typeMismatch,
                 "An explicit atomic comparison order requires an integer", loc)
      }
      if atomic == .f32 {
        guard case .pointer(_, .storage) = try type(p.type) else {
          try fail(.invalidAddressSpace, "Float atomic read-modify-write requires storage memory", loc)
        }
        switch operation {
        case .add, .subtract, .exchange(compare: nil, strong: false): break
        default: try fail(.typeMismatch, "Float atomics support add, subtract and exchange", loc)
        }
      }
      if atomic.width == 8, !capabilities.contains(.shaderInt64AtomicAllOps) {
        if (operation == .min || operation == .max), result == nil,
           case .pointer(_, .storage) = try type(p.type) {
          try cap(.shaderInt64AtomicMinMax, loc)
        } else {
          try cap(.shaderInt64AtomicAllOps, loc)
        }
      }
      let compares: Bool
      if case .exchange(compare: nil, strong: true) = operation {
        try fail(.invalidReference, "Strong compare-exchange requires a comparator", loc)
      }
      if case .exchange(let compare?, _) = operation {
        let expected = try use(compare, loc)
        try need(try type(expected.type) == .scalar(atomic), .typeMismatch, "Atomic comparison type mismatch", loc)
        compares = true
      } else { compares = false }
      if let result {
        let e = try f.expressions.value(for: result)
        let matches = try compares ? atomicCompareResults[e.type] == atomic : type(e.type) == .scalar(atomic)
        guard case .atomicResult(_, let comparison) = e.kind, comparison == compares, matches,
              callResults.insert(result).inserted else {
          try fail(.invalidReference, "Incorrect or duplicate atomic result", loc)
        }
        available.insert(result)
      }
    case .subgroupBallot, .subgroupCollectiveOperation, .subgroupGather:
      try requireUniformControl(ctrl, rule: .subgroup, severity: severities.subgroup,
                                message: "Subgroup operations require uniform control flow", at: loc)
      try validateSubgroupStatement(s.kind, function: f, available: &available, uniform: &uniform,
                                    control: ctrl, severity: severities.subgroup, producedResults: &callResults, location: loc)
    case .cooperativeStore(let target, let data):
      try requireUniformControl(ctrl, rule: .cooperativeMatrix, severity: .error,
                                message: "Cooperative matrix operations require subgroup-uniform control flow", at: loc)
      let targetExpression = try use(target, loc)
      guard case .cooperativeMatrix(_, _, let scalar, _) = try type(targetExpression.type) else {
        try fail(.typeMismatch, "coopStore requires a cooperative matrix value", loc)
      }
      let pointer = try use(data.pointer, loc), stride = try use(data.stride, loc)
      let address = try cooperativePointerUniformity(data.pointer, in: f, uniform: uniform)
      for operand in [address, uniform[data.stride.index]] {
        try requireUniformity(operand, rule: .cooperativeMatrix, severity: .error,
                              message: "Cooperative matrix address and stride must be uniform", at: loc)
      }
      guard case .pointer(let pointee, let space) = try type(pointer.type),
            try type(pointee) == .scalar(scalar) else {
        try fail(.typeMismatch, "Cooperative matrix memory requires a matching scalar pointer", loc)
      }
      switch space {
      case .storage(let access): try need(access.contains(.store), .invalidAddressSpace, "Cooperative store requires writable memory", loc)
      case .function, .private, .workgroup: break
      default: try fail(.invalidAddressSpace, "Invalid cooperative matrix address space", loc)
      }
      try need([.scalar(.i32), .scalar(.u32)].contains(type(stride.type)), .typeMismatch,
               "Cooperative matrix stride must be i32 or u32", loc)
    case .workgroupUniformLoad(let pointer, let result):
      let p = try use(pointer, loc)
      guard case .pointer(let base, .workgroup) = try type(p.type) else {
        try fail(.typeMismatch, "Uniform load requires a workgroup pointer", loc)
      }
      try requireUniformControl(ctrl, rule: .barrier, severity: .error,
                                message: "Workgroup uniform load requires uniform control flow", at: loc)
      try requireUniformity(uniform[pointer.index], rule: .barrier, severity: .error,
                            message: "Workgroup uniform load requires a uniform address", at: loc)
      let e = try f.expressions.value(for: result)
      guard case .workgroupUniformLoadResult = e.kind, callResults.insert(result).inserted else {
        try fail(.invalidReference, "Incorrect or duplicate workgroup uniform load result", loc)
      }
      if let atomic = try atomicPointee(p.type) {
        try need(try type(e.type) == .scalar(atomic), .typeMismatch, "Atomic uniform load result type mismatch", loc)
      } else {
        try need(try layout(base).constructible && same(base, e.type), .typeMismatch,
                 "Uniform load requires a constructible pointee and matching result", loc)
      }
      uniform[result.index] = .uniform
      available.insert(result)
    case .call(let target, let args, let result):
      try validateCall(target, arguments: args, result: result, function: f, available: &available,
                       producedResults: &callResults, uniform: &uniform, locals: &locals, uniformControl: ctrl, at: loc)
    case .debugPrintf(let format, let arguments):
      try cap(.debugPrintf, loc)
      try text(format)
      let expected = try DebugPrintfFormat.arguments(in: format, location: loc, budget: &budget)
      try need(expected.count == arguments.count, .typeMismatch, "debugPrintf format argument count mismatch", loc)
      for (specifier, handle) in zip(expected, arguments) {
        try budget.consume(.work)
        let argument = try use(handle, loc)
        try need(try specifier.accepts(type(argument.type)), .typeMismatch, "debugPrintf argument type mismatch", argument.location ?? loc)
      }
    case .controlBarrier(let barrier), .memoryBarrier(let barrier):
      try need(
        barrier.rawValue > 0 && barrier.rawValue & ~UInt8(15) == 0, .invalidType,
        "Invalid barrier flags", loc)
      if barrier.contains(.subgroup) { try cap([.subgroup, .subgroupBarrier], loc) }
      try requireUniformControl(ctrl, rule: .barrier, severity: .error, message: "Barrier in divergent control flow", at: loc)
    case .kill:
      try need(!inContinuing, .invalidControlFlow, "Discard forbidden in continuing", loc)
      if reachable { uniformity.exits = try mergeUniformity(uniformity.exits, ctrl) }
      step = .exit
    case .demoteToHelperInvocation:
      // Demoted invocations can continue computing derivatives after reconvergence.
      break
    case .break:
      try need((loopDepth > 0 || switchDepth > 0) && !inContinuing, .invalidControlFlow, "Invalid break target", loc)
      try recordLocalBreak(locals)
      step = .breakTarget
    case .continue:
      try need(loopDepth > 0 && !inContinuing, .invalidControlFlow, "Invalid loop jump", loc)
      try recordLocalContinue(locals, control: ctrl, available: available)
      step = .continueTarget
    default: try unsupported("Statement rule family not implemented", loc)
    }
    return step
  }

  mutating func validateLoop(
    _ body: IRHandle<IRBlock>, continuing: IRHandle<IRBlock>, breakIf: IRHandle<IRExpression>?,
    f: IRFunction, available: Set<IRHandle<IRExpression>>, uniform: inout [UniformValue],
    locals: inout [UniformValue], control: UniformValue, active: Set<IRHandle<IRBlock>>,
    seen: inout Set<IRHandle<IRBlock>>, statements: inout Set<IRHandle<IRStatement>>,
    callResults: inout Set<IRHandle<IRExpression>>, loopDepth: Int, switchDepth: Int
  ) throws -> StatementFlow {
    // Reserve header values before walking the body so each load sees both entry and backedge values.
    let headerControl = try uniformity.loopHeader(control, budget: &budget)
    try budget.consume(.work, amount: locals.count)
    var headerLocals: [UniformValue] = []
    for value in locals {
      headerLocals.append(try uniformity.loopHeader(value, budget: &budget))
    }
    uniformity.breakTargets.append(.loop(uniformity.loops.count))
    uniformity.loops.append(LoopUniformity())
    var bodyLocals = headerLocals
    let bodyResult = try block(
      body, f: f, available: available, uniform: &uniform, locals: &bodyLocals, controlUniform: headerControl, active: active,
      seen: &seen, statements: &statements, callResults: &callResults,
      loopDepth: loopDepth, inContinuing: false, switchDepth: switchDepth)
    let entersContinuing = bodyResult.flow.contains(.next) || bodyResult.flow.contains(.continueTarget)
    let loop = uniformity.loops[uniformity.loops.count - 1]
    var continuingLocals = loop.continues ?? bodyLocals
    if bodyResult.flow.contains(.next) {
      continuingLocals = try joinLocalUniformity(loop.continues, bodyLocals)
    }
    let continuingControl = try mergeUniformity(
      mergeUniformity(headerControl, bodyResult.trailingUniformity), loop.continueControl)
    // The continuing block inherits the body's outer scope. A continue can bypass an emit.
    var continuingScope = bodyResult.available
    if let continueScope = loop.continueScope {
      try budget.consume(.work, amount: min(continuingScope.count, continueScope.count))
      continuingScope.formIntersection(continueScope)
    }
    let continuingResult = try block(
      continuing, f: f, available: continuingScope, uniform: &uniform, locals: &continuingLocals,
      controlUniform: continuingControl, active: active, seen: &seen, statements: &statements,
      callResults: &callResults, loopDepth: loopDepth, inContinuing: true, breakIf: breakIf, switchDepth: switchDepth,
      reachableFromEntry: entersContinuing)
    var exits = loop.breaks
    if entersContinuing {
      let backedge = try mergeUniformity(continuingControl, continuingResult.trailingUniformity)
      uniformity.linkBackedge(headerControl, to: backedge)
      for index in headerLocals.indices {
        try budget.consume(.work)
        uniformity.linkBackedge(headerLocals[index], to: continuingLocals[index])
      }
      if breakIf != nil {
        exits = try joinLocalUniformity(exits, continuingLocals)
      }
    }
    _ = uniformity.breakTargets.removeLast()
    _ = uniformity.loops.removeLast()
    if let exits { locals = exits }
    var flow = bodyResult.flow.intersection(.exit)
    if bodyResult.flow.contains(.breakTarget) || (entersContinuing && breakIf != nil) {
      flow.insert(.next)
    }
    return flow.isEmpty ? .next : flow
  }

  mutating func validateSwitch(
    _ cases: [IRSwitchCase], selector: UniformValue, type selectionType: IRTypeKind,
    f: IRFunction, available: Set<IRHandle<IRExpression>>, uniform: inout [UniformValue],
    locals: inout [UniformValue], control: UniformValue, active: Set<IRHandle<IRBlock>>,
    seen: inout Set<IRHandle<IRBlock>>, statements: inout Set<IRHandle<IRStatement>>,
    callResults: inout Set<IRHandle<IRExpression>>, loopDepth: Int, inContinuing: Bool,
    switchDepth: Int, location: ShaderLocation?
  ) throws -> (flow: StatementFlow, trailing: UniformValue) {
    try need(selectionType == .scalar(.i32) || selectionType == .scalar(.u32),
             .typeMismatch, "Switch selector must be i32 or u32", location)
    var labels = Set<IRSwitchValue>()
    var branches: StatementFlow = []
    var nextLocals: [UniformValue]?
    var fallthroughLocals: [UniformValue]?
    var fallthroughControl = UniformValue.uniform
    var trailing = selector
    let entryControl = try mergeUniformity(control, selector)
    uniformity.breakTargets.append(.selection(nil))
    var index = 0
    while index < cases.count {
      let group = cases[index]
      repeat {
        let entry = cases[index]
        try budget.consume(.work)
        try need(entry.fallThrough == group.fallThrough, .invalidControlFlow,
                 "Grouped labels must share the fall-through behavior", location)
        try need(labels.insert(entry.value).inserted, .duplicateName, "Duplicate switch selector", location)
        switch entry.value {
        case .i32: try need(selectionType == .scalar(.i32), .typeMismatch, "Switch case type mismatch", location)
        case .u32: try need(selectionType == .scalar(.u32), .typeMismatch, "Switch case type mismatch", location)
        case .default: break
        }
        index += 1
      } while index < cases.count && cases[index].body == group.body
      try need(index < cases.count || !group.fallThrough, .invalidControlFlow,
               "The final switch case cannot fall through", location)
      let nested = try mergeUniformity(entryControl, fallthroughControl)
      var caseLocals = try joinLocalUniformity(fallthroughLocals, locals)
      let branch = try block(
        group.body, f: f, available: available, uniform: &uniform, locals: &caseLocals,
        controlUniform: nested, active: active, seen: &seen, statements: &statements,
        callResults: &callResults, loopDepth: loopDepth, inContinuing: inContinuing, switchDepth: switchDepth + 1)
      if branch.flow.contains(.exit) || branch.flow.contains(.continueTarget) {
        trailing = try mergeUniformity(trailing, branch.trailingUniformity)
      }
      if group.fallThrough && branch.flow.contains(.next) {
        branches.formUnion(branch.flow.subtracting(.next))
        fallthroughLocals = caseLocals
        fallthroughControl = try mergeUniformity(fallthroughControl, branch.trailingUniformity)
      } else {
        branches.formUnion(branch.flow)
        fallthroughLocals = nil
        fallthroughControl = .uniform
        if branch.flow.contains(.next) { nextLocals = try joinLocalUniformity(nextLocals, caseLocals) }
      }
    }
    if case .selection(let breakLocals) = uniformity.breakTargets.removeLast(), let breakLocals {
      nextLocals = try joinLocalUniformity(nextLocals, breakLocals)
    }
    if let nextLocals { locals = nextLocals }
    try need(labels.contains(.default), .invalidControlFlow, "Switch requires default", location)
    if branches.remove(.breakTarget) != nil { branches.insert(.next) }
    return (branches, branches == .next ? .uniform : trailing)
  }

  mutating func block(
    _ handle: IRHandle<IRBlock>, f: IRFunction, available initial: Set<IRHandle<IRExpression>>,
    uniform: inout [UniformValue], locals: inout [UniformValue], controlUniform: UniformValue, active: Set<IRHandle<IRBlock>>,
    seen: inout Set<IRHandle<IRBlock>>, statements: inout Set<IRHandle<IRStatement>>,
    callResults: inout Set<IRHandle<IRExpression>>, loopDepth: Int,
    inContinuing: Bool, breakIf: IRHandle<IRExpression>? = nil, switchDepth: Int = 0,
    reachableFromEntry: Bool = true
  ) throws -> (flow: StatementFlow, trailingUniformity: UniformValue, available: Set<IRHandle<IRExpression>>) {
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
    var flow: StatementFlow = .next
    // Keep local control dependencies separate so a branch can drop its condition at reconvergence.
    var trailingUniformity = UniformValue.uniform
    let parentReachable = reachable
    reachable = reachable && reachableFromEntry
    defer { reachable = parentReachable }
    var lastReachableState: (locals: [UniformValue], trailing: UniformValue)?
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
      if reachable && !flow.contains(.next) {
        // Keep validating lexical scope and types without changing the live path.
        try budget.consume(.work, amount: locals.count)
        lastReachableState = (locals, trailingUniformity)
        reachable = false
      }
      let ctrl = try mergeUniformity(controlUniform, trailingUniformity)
      var step: StatementFlow = .next
      switch s.kind {
      case .block(let child):
        let result = try block(
          child, f: f, available: available, uniform: &uniform, locals: &locals, controlUniform: ctrl, active: active,
          seen: &seen, statements: &statements, callResults: &callResults,
          loopDepth: loopDepth, inContinuing: inContinuing, switchDepth: switchDepth)
        step = result.flow
        trailingUniformity = try mergeUniformity(trailingUniformity, result.trailingUniformity)
      case .ifCondition(let condition, let yes, let no):
        let c = try use(condition, loc)
        try need(try boolean(c.type), .typeMismatch, "If requires bool condition", loc)
        let nested = try mergeUniformity(ctrl, uniform[condition.index])
        var yesLocals = locals
        var noLocals = locals
        let a = try block(
          yes, f: f, available: available, uniform: &uniform, locals: &yesLocals, controlUniform: nested, active: active,
          seen: &seen, statements: &statements, callResults: &callResults,
          loopDepth: loopDepth, inContinuing: inContinuing, switchDepth: switchDepth)
        let b = try block(
          no, f: f, available: available, uniform: &uniform, locals: &noLocals, controlUniform: nested, active: active,
          seen: &seen, statements: &statements, callResults: &callResults,
          loopDepth: loopDepth, inContinuing: inContinuing, switchDepth: switchDepth)
        step = a.flow.union(b.flow)
        if a.flow.contains(.next) {
          locals = b.flow.contains(.next) ? try joinLocalUniformity(yesLocals, noLocals) : yesLocals
        } else if b.flow.contains(.next) {
          locals = noLocals
        }
        if step != .next {
          trailingUniformity = try mergeUniformity(trailingUniformity, uniform[condition.index])
          trailingUniformity = try mergeUniformity(trailingUniformity, a.trailingUniformity)
          trailingUniformity = try mergeUniformity(trailingUniformity, b.trailingUniformity)
        }
      case .switchValue(let selector, let cases):
        let selection = try use(selector, loc)
        let result = try validateSwitch(
          cases, selector: uniform[selector.index], type: type(selection.type), f: f, available: available,
          uniform: &uniform, locals: &locals, control: ctrl, active: active,
          seen: &seen, statements: &statements, callResults: &callResults,
          loopDepth: loopDepth, inContinuing: inContinuing, switchDepth: switchDepth, location: loc)
        step = result.flow
        trailingUniformity = try mergeUniformity(trailingUniformity, result.trailing)
      case .loop(let body, let continuing, let breakIf):
        step = try validateLoop(body, continuing: continuing, breakIf: breakIf, f: f, available: available,
                                uniform: &uniform, locals: &locals, control: ctrl, active: active,
                                seen: &seen, statements: &statements, callResults: &callResults,
                                loopDepth: loopDepth + 1, switchDepth: switchDepth)
      default:
        step = try validateLeafStatement(s, f: f, available: &available, uniform: &uniform, locals: &locals, ctrl: ctrl,
                                         callResults: &callResults, loopDepth: loopDepth, inContinuing: inContinuing,
                                         switchDepth: switchDepth)
      }
      if flow.contains(.next) {
        flow.remove(.next)
        flow.formUnion(step)
      }
    }
    if let breakIf {
      try need(
        try boolean(use(breakIf, b.location).type), .typeMismatch,
        "Loop break-if must be available bool", b.location)
      trailingUniformity = try mergeUniformity(trailingUniformity, uniform[breakIf.index])
    }
    if let lastReachableState {
      locals = lastReachableState.locals
      trailingUniformity = lastReachableState.trailing
    }
    return (flow, trailingUniformity, available)
  }
  private mutating func validateDualSourceMembers(_ members: [IRStructMember], at location: ShaderLocation?) throws {
    try budget.consume(.work, amount: members.count)
    guard members.contains(where: {
      if case .location(_, _, _, .some, _) = $0.binding { return true }
      return false
    }) else { return }
    try cap(.dualSourceBlending, location)
    var colorTypes: [UInt32: IRHandle<IRType>] = [:]
    for member in members {
      try budget.consume(.work)
      guard case .location(let locationIndex, _, _, let blend, let perPrimitive) = member.binding else { continue }
      guard locationIndex == 0, let blend, blend <= 1, !perPrimitive,
            colorTypes.updateValue(member.type, forKey: blend) == nil else {
        try fail(.invalidBinding, "Dual-source outputs require distinct sources 0 and 1 at location 0", member.location)
      }
    }
    guard let first = colorTypes[0], let second = colorTypes[1], try type(first) == type(second) else {
      try fail(.invalidBinding, "Dual-source outputs require two sources with the same type", location)
    }
    switch try type(first) {
    case .scalar(let value), .vector(_, let value):
      try need(value.kind != .bool, .typeMismatch, "Dual-source outputs require numeric scalar or vector types", location)
    default:
      try fail(.typeMismatch, "Dual-source outputs require numeric scalar or vector types", location)
    }
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
        try need(!primitive, .unsupportedConstruct, "Per-primitive interface rules not implemented")
        if let blend {
          try need(stage == .fragment && output && member && n == 0 && blend <= 1,
                   .invalidBinding, "Blend source must be a fragment output member at location 0")
        } else {
          try need(stage != .compute && locations.insert(n).inserted, .invalidBinding, "Invalid/duplicate location")
        }
        if interp == .perVertex {
          try need(stage == .fragment && !output, .invalidBinding,
                   "Per-vertex interpolation requires a fragment input")
          try cap(.perVertex)
          try need(sampling == nil || sampling == .center, .invalidBinding,
                   "Per-vertex interpolation only allows center sampling")
          guard case .array(let element, .constant(3), _) = try type(t),
                let shape = IRBitcastShape(try type(element)), shape.scalar.kind != .bool else {
            try fail(.typeMismatch, "Per-vertex interpolation requires an array of three numeric scalars or vectors")
          }
          return
        }
        let varying = (stage == .vertex && output) || (stage == .fragment && !output)
        try need(varying || (interp == nil && sampling == nil), .invalidBinding,
                 "Interpolation is restricted to vertex outputs and fragment inputs")
        guard let s = try scalar(t) else { try fail(.typeMismatch, "Location needs scalar/vector") }
        if case .matrix = try type(t) {
          try fail(.typeMismatch, "Matrix cannot have direct location")
        }
        try need(s.kind != .bool, .typeMismatch, "Boolean location not allowed")
        if varying && (s.kind == .uint || s.kind == .sint) {
          try need(
            interp == .flat, .invalidBinding,
            "Integer interstage values need flat interpolation")
        }
        if sampling == .sample { try cap(.multisampledShading) }
        if interp == .linear { try cap(.linearInterpolation) }
      case .builtIn(let b):
        let key: String
        switch b {
        case .position: key = "position"
        case .barycentric: key = "barycentric"
        default: key = String(describing: b)
        }
        try need(builtins.insert(key).inserted, .invalidBinding, "Duplicate builtin")
        let expected: IRTypeKind
        let stages: Set<IRStage>
        let isOutput: Bool
        switch b {
        case .localInvocationIndex, .globalInvocationIndex, .workgroupIndex:
          expected = .scalar(.u32)
          stages = [.compute]
          isOutput = false
        case .localInvocationId, .globalInvocationId, .workgroupId, .numWorkgroups:
          expected = .vector(size: .three, scalar: .u32)
          stages = [.compute]
          isOutput = false
        case .vertexIndex, .instanceIndex, .baseInstance:
          expected = .scalar(.u32)
          stages = [.vertex]
          isOutput = false
        case .drawIndex:
          try cap(.drawIndex)
          expected = .scalar(.u32)
          stages = [.vertex]
          isOutput = false
        case .viewIndex:
          try cap(.multiview)
          expected = .scalar(.u32)
          stages = [.vertex, .fragment]
          isOutput = false
        case .frontFacing:
          expected = .scalar(.bool)
          stages = [.fragment]
          isOutput = false
        case .primitiveIndex:
          try cap(.primitiveIndex)
          expected = .scalar(.u32)
          stages = [.fragment]
          isOutput = false
        case .barycentric:
          try cap(.shaderBarycentrics)
          expected = .vector(size: .three, scalar: .f32)
          stages = [.fragment]
          isOutput = false
        case .subgroupSize, .subgroupInvocationId:
          try cap(.subgroup)
          expected = .scalar(.u32)
          stages = [.compute, .fragment]
          isOutput = false
        case .subgroupId, .numSubgroups:
          try cap(.subgroup)
          expected = .scalar(.u32)
          stages = [.compute]
          isOutput = false
        case .position:
          expected = .vector(size: .four, scalar: .f32)
          stages = output ? [.vertex] : [.fragment]
          isOutput = output
        case .clipDistances:
          try cap(.clipDistances)
          try need(stage == .vertex && output, .invalidBinding, "Clip distances require a vertex output")
          guard case .array(let base, .constant(let count), _) = try type(t),
                (1...8).contains(count), try type(base) == .scalar(.f32) else {
            try fail(.typeMismatch, "Clip distances require array<f32, N> with 1 to 8 elements")
          }
          return
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
