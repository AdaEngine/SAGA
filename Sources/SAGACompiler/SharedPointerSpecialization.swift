internal import SAGAShaderIR
internal import SAGAValidation

func specializeGlobalParameters(
    _ validated: ValidatedModule<IRModule>, capabilities: ShaderCapabilities
) throws -> ValidatedModule<IRModule> {
    let module = validated.module
    var budget = ShaderBudget()
    for function in module.functions.values {
        try budget.consume(.work)
        for argument in function.arguments {
            try budget.consume(.work)
            if try module.types.value(for: argument.type).kind.requiresGlobalSpecialization {
                var specializer = GlobalParameterSpecializer(module: module, budget: budget)
                let specialized = try specializer.specialize()
                return try ValidatedModule.checking(specialized, with: ShaderSemanticValidator(capabilities: capabilities))
            }
        }
    }
    return validated
}

private extension IRTypeKind {
    var requiresGlobalSpecialization: Bool {
        if isTextureOrSampler { return true }
        guard case .pointer(_, let space) = self else { return false }
        switch space {
        case .workgroup, .storage, .uniform, .immediate: return true
        default: return false
        }
    }
}

private struct GlobalParameterPath: Hashable {
    struct Step: Hashable {
        enum Index: Hashable {
            case fixed(UInt32)
            case dynamic(IRHandle<IRType>)
        }

        let type: IRHandle<IRType>
        let index: Index
    }

    // A texture or sampler binds directly to the global. Shared pointers can also bind subviews.
    let global: IRHandle<IRGlobalVariable>
    let rootType: IRHandle<IRType>
    var steps: [Step]
}

private struct GlobalParameterBinding {
    var path: GlobalParameterPath
    var indices: [SharedExpression]
}

private struct GlobalParameterSignature {
    var arguments: [IRFunctionArgument] = []
    var remappedSlots: [Int: UInt32] = [:]
    var boundParameters: [Int: GlobalParameterBinding] = [:]
    var addresses: [Int: IRExpressionKind] = [:]
    var prefix: [IRExpression] = []
    var emitted: [SharedExpression] = []

    mutating func bind(_ path: GlobalParameterPath, slot: Int, budget: inout ShaderBudget) throws {
        var address = IRExpressionKind.globalVariable(path.global)
        var type = path.rootType
        var indices: [SharedExpression] = []
        for step in path.steps {
            try budget.consume(.nodes)
            let base = try SharedExpression(index: prefix.count)
            prefix.append(.init(type: type, kind: address))
            switch address {
            case .access, .accessIndex: emitted.append(base)
            default: break
            }
            switch step.index {
            case .fixed(let index):
                address = .accessIndex(base: base, index: index)
            case .dynamic(let indexType):
                try budget.consume(.nodes, amount: 2)
                let index = try SharedExpression(index: prefix.count)
                prefix.append(.init(type: indexType, kind: .functionArgument(UInt32(arguments.count))))
                arguments.append(.init(type: indexType))
                indices.append(index)
                address = .access(base: base, index: index)
            }
            type = step.type
        }
        addresses[slot] = address
        boundParameters[slot] = .init(path: path, indices: indices)
    }
}

private struct GlobalParameterSpecializer {
    struct Variant: Hashable {
        let function: IRHandle<IRFunction>
        let paths: [GlobalParameterPath?]
    }

    let module: IRModule
    var budget: ShaderBudget
    private var variants: [Variant: IRHandle<IRFunction>] = [:]
    private var functions: [IRFunction] = []

    init(module: IRModule, budget: ShaderBudget) {
        self.module = module
        self.budget = budget
    }

    mutating func specialize() throws -> IRModule {
        var entries: [IREntryPoint] = []
        for entry in module.entryPoints {
            try budget.consume(.work)
            let function = try module.functions.value(for: entry.function)
            let key = Variant(function: entry.function, paths: Array(repeating: nil, count: function.arguments.count))
            let handle = try specialize(key)
            entries.append(.init(name: entry.name, stage: entry.stage, function: handle,
                                 earlyDepthTest: entry.earlyDepthTest, workgroupSize: entry.workgroupSize,
                                 workgroupOverrides: entry.workgroupOverrides, subgroupSize: entry.subgroupSize, meshInfo: entry.meshInfo,
                                 taskPayload: entry.taskPayload, incomingRayPayload: entry.incomingRayPayload,
                                 location: entry.location))
        }
        return .init(types: module.types, specialTypes: module.specialTypes, constants: module.constants,
                     overrides: module.overrides, globalVariables: module.globalVariables,
                     globalExpressions: module.globalExpressions, functions: .init(functions), entryPoints: entries,
                     diagnosticFilters: module.diagnosticFilters, diagnosticFilter: module.diagnosticFilter,
                     documentation: module.documentation, allowsPointerSubviews: module.allowsPointerSubviews,
                     uniformBufferStandardLayout: module.uniformBufferStandardLayout,
                     requiresSubgroupUniformity: module.requiresSubgroupUniformity)
    }

    private mutating func specialize(_ key: Variant) throws -> IRHandle<IRFunction> {
        if let existing = variants[key] { return existing }
        try budget.enterDepth()
        defer { try? budget.leaveDepth() }
        try budget.consume(.nodes)
        let source = try module.functions.value(for: key.function)
        var signature = GlobalParameterSignature()
        for (slot, argument) in source.arguments.enumerated() {
            try budget.consume(.work)
            if let path = key.paths[slot] {
                try signature.bind(path, slot: slot, budget: &budget)
            } else {
                signature.remappedSlots[slot] = UInt32(signature.arguments.count)
                signature.arguments.append(argument)
            }
        }
        let offset = signature.prefix.count
        func mapped(_ handle: SharedExpression) throws -> SharedExpression {
            try SharedExpression(index: handle.index + offset)
        }

        var callResults: [SharedExpression: IRHandle<IRFunction>] = [:]
        var statements: [IRStatement] = []
        for statement in source.statements.values {
            try budget.consume(.nodes)
            let kind: IRStatementKind
            if case .call(let target, let arguments, let result) = statement.kind {
                let parameters = try module.functions.value(for: target).arguments
                var paths: [GlobalParameterPath?] = []
                var lowered: [SharedExpression] = []
                for (argument, parameter) in zip(arguments, parameters) {
                    try budget.consume(.work)
                    if try module.types.value(for: parameter.type).kind.requiresGlobalSpecialization {
                        let binding = try bind(argument, in: source, signature: signature, map: mapped)
                        try budget.consume(.work, amount: binding.path.steps.count + binding.indices.count)
                        paths.append(binding.path)
                        lowered.append(contentsOf: binding.indices)
                    } else {
                        paths.append(nil)
                        lowered.append(try mapped(argument))
                    }
                }
                let callee = try specialize(.init(function: target, paths: paths))
                if let result { callResults[result] = callee }
                kind = try .call(function: callee, arguments: lowered, result: result.map(mapped))
            } else {
                kind = try statement.kind.remappingSharedPointerOperands(mapped)
            }
            statements.append(.init(kind: kind, location: statement.location))
        }

        var expressions = signature.prefix
        var prologue = signature.emitted
        for (index, expression) in source.expressions.values.enumerated() {
            try budget.consume(.nodes)
            let kind: IRExpressionKind
            switch expression.kind {
            case .functionArgument(let argument):
                if let address = signature.addresses[Int(argument)] {
                    kind = address
                    switch address {
                    case .access, .accessIndex: prologue.append(try SharedExpression(index: expressions.count))
                    default: break
                    }
                } else {
                    guard let slot = signature.remappedSlots[Int(argument)] else { throw ShaderFailure.invalidHandle }
                    kind = .functionArgument(slot)
                }
            case .callResult:
                guard let target = callResults[try SharedExpression(index: index)] else { throw ShaderFailure.invalidHandle }
                kind = .callResult(target)
            default:
                kind = try expression.kind.remappingSharedPointerOperands(mapped)
            }
            expressions.append(.init(type: expression.type, kind: kind, location: expression.location))
        }
        var blocks = source.blocks.values
        if !prologue.isEmpty {
            let handle = try IRHandle<IRStatement>(index: statements.count)
            statements.append(.init(kind: .emit(prologue)))
            let root = try source.blocks.value(for: source.body)
            blocks[source.body.index] = .init(statements: [handle] + root.statements, location: root.location)
        }
        let locals = try source.localVariables.values.map { local in
            try budget.consume(.nodes)
            return try IRLocalVariable(name: local.name, type: local.type, initializer: local.initializer.map(mapped),
                                       location: local.location)
        }
        let names = try source.namedExpressions.map { named in
            try budget.consume(.nodes)
            return try IRNamedExpression(expression: mapped(named.expression), name: named.name)
        }
        let handle = try IRHandle<IRFunction>(index: functions.count)
        functions.append(.init(name: source.name, arguments: signature.arguments, result: source.result,
                               localVariables: .init(locals), expressions: .init(expressions), namedExpressions: names,
                               statements: .init(statements), blocks: .init(blocks), body: source.body,
                               diagnosticFilter: source.diagnosticFilter, location: source.location))
        variants[key] = handle
        return handle
    }

    private mutating func bind(
        _ argument: SharedExpression, in function: IRFunction, signature: GlobalParameterSignature,
        map: (SharedExpression) throws -> SharedExpression
    ) throws -> GlobalParameterBinding {
        var cursor = argument
        var steps: [GlobalParameterPath.Step] = []
        var indices: [SharedExpression] = []
        while true {
            try budget.consume(.work)
            let expression = try function.expressions.value(for: cursor)
            switch expression.kind {
            case .access(let base, let index):
                let type = try function.expressions.value(for: index).type
                steps.append(.init(type: expression.type, index: .dynamic(type)))
                indices.append(try map(index))
                cursor = base
            case .accessIndex(let base, let index):
                steps.append(.init(type: expression.type, index: .fixed(index)))
                cursor = base
            case .globalVariable(let global):
                return .init(path: .init(global: global, rootType: expression.type, steps: Array(steps.reversed())),
                             indices: Array(indices.reversed()))
            case .functionArgument(let slot):
                guard var binding = signature.boundParameters[Int(slot)] else { throw ShaderFailure.invalidHandle }
                binding.path.steps.append(contentsOf: steps.reversed())
                binding.indices.append(contentsOf: indices.reversed())
                return binding
            default:
                throw ShaderFailure.unsupported("Resource parameter must derive from a global variable")
            }
        }
    }
}
