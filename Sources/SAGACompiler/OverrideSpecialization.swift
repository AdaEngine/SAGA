internal import SAGAFrontends
internal import SAGAShaderIR
internal import SAGAValidation

func specializeOverrides(
    _ validated: ValidatedModule<IRModule>, entry: IREntryPoint?, constants: [String: Double],
    retainLinkedStages: Bool, validator: ShaderSemanticValidator
) throws -> ValidatedModule<IRModule> {
    let module = validated.module
    let isolateCompute = entry?.stage == .compute && module.entryPoints.count > 1
    guard isolateCompute || module.overrides.count != 0 || !constants.isEmpty else { return validated }
    guard let entry else {
        throw ShaderFailure.unsupported("Override specialization requires a selected entry point")
    }
    var specializer = OverrideSpecializer(module: module, budget: ShaderBudget())
    let specialized = try specializer.specialize(entry, constants: constants, retainLinkedStages: retainLinkedStages)
    return try ValidatedModule.checking(specialized, with: validator)
}

private struct OverrideSpecializer {
    let module: IRModule
    var budget: ShaderBudget
    var values: [IRHandle<IROverride>: IRLiteral] = [:]

    mutating func specialize(
        _ entry: IREntryPoint, constants: [String: Double], retainLinkedStages: Bool
    ) throws -> IRModule {
        var keys: [[UInt8]: IRHandle<IROverride>] = [:]
        for (index, declaration) in module.overrides.values.enumerated() {
            try budget.consume(.work)
            guard let key = declaration.id.map(String.init) ?? declaration.name else { continue }
            try budget.consume(.work, amount: key.utf8.count)
            keys[Array(key.utf8)] = try IRHandle(index: index)
        }
        try budget.consume(.nodes, amount: constants.count)
        for (key, value) in constants {
            try budget.consume(.work, amount: key.utf8.count + 1)
            guard let handle = keys[Array(key.utf8)] else {
                throw failure(.unresolvedName, "Unknown pipeline constant '\(key)'")
            }
            values[handle] = try literal(value, for: module.overrides.value(for: handle))
        }
        let live = try reachableFunctions(from: [entry.function])
        // Retain the other render entry points for shared GLSL resource layout.
        let retainedEntries = retainLinkedStages && entry.stage != .compute
            ? module.entryPoints.filter { $0.stage != .compute } : [entry]
        let retained = retainedEntries.count == 1 ? live : try reachableFunctions(from: retainedEntries.map(\.function))
        let used = try reachableOverrides(entry, functions: live, linkedFunctions: retained)
        // Validation requires dependencies to precede users in the override table.
        for handle in used.overrides.sorted(by: { $0.index < $1.index }) {
            try budget.consume(.work)
            guard values[handle] == nil else { continue }
            let value = try module.overrides.value(for: handle)
            guard let initializer = value.initializer else { throw ShaderFailure.invalidHandle }
            let literal = try evaluateWGSLPipelineScalar(initializer, in: module, overrides: values, budget: &budget)
            values[handle] = literal
        }

        let selectedSize = try workgroupSize(entry)
        let selectedSubgroupSize = try subgroupSize(entry)
        let types = try specializeArrayTypes(used.arrays)
        let globals = try stripPipelineGlobals()
        var arena = PipelineValueArena(types: types.values, constants: module.constants.values, expressions: globals)
        let variables = try specializePrivateInitializers(used.privateGlobals, arena: &arena)

        let order = retained.sorted(by: { $0.index < $1.index })
        var remap: [IRHandle<IRFunction>: IRHandle<IRFunction>] = [:]
        for (index, handle) in order.enumerated() {
            try budget.consume(.work)
            remap[handle] = try IRHandle(index: index)
        }
        func mapped(_ handle: IRHandle<IRFunction>) throws -> IRHandle<IRFunction> {
            guard let result = remap[handle] else { throw ShaderFailure.invalidHandle }
            return result
        }
        let functions = try order.map { handle -> IRFunction in
            try budget.consume(.work)
            let function = try module.functions.value(for: handle)
            let active = live.contains(handle)
            let body = active
                ? try specializePipelineBody(function, in: module, overrides: values, budget: &budget) : [:]
            let replacements = try arena.replacements(body, in: function, budget: &budget)
            let statements = try function.statements.values.map { statement -> IRStatement in
                try budget.consume(.work)
                switch statement.kind {
                case .call(let target, let arguments, let result):
                    return .init(kind: .call(function: try mapped(target), arguments: arguments, result: result),
                                 location: statement.location)
                case .emit(let handles) where !replacements.isEmpty:
                    try budget.consume(.work, amount: handles.count)
                    return .init(kind: .emit(handles.filter { replacements[$0] == nil }), location: statement.location)
                default:
                    return statement
                }
            }
            return try .init(name: function.name, arguments: function.arguments, result: function.result,
                             localVariables: function.localVariables,
                             expressions: expressions(function.expressions, functionRemap: remap,
                                                      replacements: replacements, useOverrideValues: active),
                             namedExpressions: function.namedExpressions, statements: .init(statements),
                             blocks: function.blocks, body: function.body, diagnosticFilter: function.diagnosticFilter,
                             location: function.location)
        }
        let entries = try retainedEntries.map { candidate -> IREntryPoint in
            try budget.consume(.work)
            return .init(name: candidate.name, stage: candidate.stage, function: try mapped(candidate.function),
                         earlyDepthTest: candidate.earlyDepthTest,
                         workgroupSize: candidate.function == entry.function ? selectedSize : candidate.workgroupSize,
                         subgroupSize: candidate.function == entry.function ? selectedSubgroupSize : candidate.subgroupSize,
                         meshInfo: candidate.meshInfo, taskPayload: candidate.taskPayload,
                         incomingRayPayload: candidate.incomingRayPayload, location: candidate.location)
        }
        return .init(types: .init(arena.types), specialTypes: module.specialTypes, constants: .init(arena.constants),
                     globalVariables: variables, globalExpressions: .init(arena.expressions),
                     functions: .init(functions), entryPoints: entries, diagnosticFilters: module.diagnosticFilters,
                     diagnosticFilter: module.diagnosticFilter, documentation: module.documentation,
                     allowsPointerSubviews: module.allowsPointerSubviews,
                     uniformBufferStandardLayout: module.uniformBufferStandardLayout,
                     requiresSubgroupUniformity: module.requiresSubgroupUniformity)
    }

    private mutating func reachableFunctions(
        from roots: [IRHandle<IRFunction>]
    ) throws -> Set<IRHandle<IRFunction>> {
        var pending = roots
        var visited = Set(roots)
        while let handle = pending.popLast() {
            try budget.consume(.work)
            let function = try module.functions.value(for: handle)
            for statement in function.statements.values {
                try budget.consume(.work)
                if case .call(let target, _, _) = statement.kind, visited.insert(target).inserted {
                    pending.append(target)
                }
            }
        }
        return visited
    }

    private mutating func reachableOverrides(
        _ entry: IREntryPoint, functions: Set<IRHandle<IRFunction>>, linkedFunctions: Set<IRHandle<IRFunction>>
    ) throws -> (overrides: Set<IRHandle<IROverride>>, arrays: Set<IRHandle<IRType>>,
                 privateGlobals: Set<IRHandle<IRGlobalVariable>>) {
        var used: Set<IRHandle<IROverride>> = []
        var arrays: Set<IRHandle<IRType>> = []
        var visitedGlobals: Set<IRHandle<IRGlobalVariable>> = []
        var privateGlobals: Set<IRHandle<IRGlobalVariable>> = []
        for handle in linkedFunctions {
            try budget.consume(.work)
            let function = try module.functions.value(for: handle)
            let active = functions.contains(handle)
            for expression in function.expressions.values {
                try budget.consume(.work)
                switch expression.kind {
                case .override(let value) where active:
                    used.insert(value)
                case .globalVariable(let variable):
                    let global = try module.globalVariables.value(for: variable)
                    let type = try module.types.value(for: global.type).kind
                    if case .bindingArray(_, .override(let count)) = type {
                        arrays.insert(global.type)
                        used.insert(count)
                    }
                    guard active, visitedGlobals.insert(variable).inserted else { continue }
                    if global.space == .private, let initializer = global.initializer {
                        let dependencies = try overrideDependencies(of: initializer)
                        if !dependencies.isEmpty {
                            privateGlobals.insert(variable)
                            used.formUnion(dependencies)
                        }
                    }
                    if case .array(_, .override(let count), _) = type {
                        arrays.insert(global.type)
                        used.insert(count)
                    }
                default:
                    break
                }
            }
        }
        if let dimensions = entry.workgroupOverrides {
            for handle in [dimensions.x, dimensions.y, dimensions.z].compactMap({ $0 }) {
                used.formUnion(try overrideDependencies(of: handle))
            }
        }
        if case .overrideExpression(let handle)? = entry.subgroupSize {
            used.formUnion(try overrideDependencies(of: handle))
        }
        var pending = Array(used)
        // A substituted declaration still contributes its static dependencies to the shader.
        while let handle = pending.popLast() {
            try budget.consume(.work)
            let value = try module.overrides.value(for: handle)
            if let initializer = value.initializer {
                for dependency in try overrideDependencies(of: initializer) where used.insert(dependency).inserted {
                    pending.append(dependency)
                }
            } else if values[handle] == nil {
                let key = value.id.map(String.init) ?? value.name ?? ""
                throw failure(.invalidReference, "Missing required pipeline constant '\(key)'", value.location)
            }
        }
        return (used, arrays, privateGlobals)
    }

    private mutating func specializePrivateInitializers(
        _ globals: Set<IRHandle<IRGlobalVariable>>, arena: inout PipelineValueArena
    ) throws -> IRTable<IRGlobalVariable> {
        var variables = module.globalVariables.values
        for handle in globals.sorted(by: { $0.index < $1.index }) {
            try budget.consume(.work)
            let global = try module.globalVariables.value(for: handle)
            guard let root = global.initializer else { throw ShaderFailure.invalidHandle }
            let value = try evaluateWGSLPipelineValue(root, in: module, overrides: values, budget: &budget)
            let initializer = try arena.compose(value, type: global.type, budget: &budget)
            variables[handle.index] = .init(name: global.name, space: global.space, binding: global.binding,
                                            type: global.type, initializer: initializer,
                                            memoryDecorations: global.memoryDecorations, location: global.location)
        }
        return .init(variables)
    }

    private mutating func overrideDependencies(of root: IRHandle<IRExpression>) throws -> Set<IRHandle<IROverride>> {
        let graph = try module.pipelineGraph(root, budget: &budget)
        var dependencies: Set<IRHandle<IROverride>> = []
        for handle in graph {
            try budget.consume(.work)
            if case .override(let value) = try module.globalExpressions.value(for: handle).kind {
                dependencies.insert(value)
            }
        }
        return dependencies
    }

    private mutating func workgroupSize(_ entry: IREntryPoint) throws -> IRWorkgroupSize {
        let fixed = entry.workgroupSize
        guard let dimensions = entry.workgroupOverrides else { return fixed }
        func dimension(_ handle: IRHandle<IRExpression>?, fallback: UInt32) throws -> UInt32 {
            guard let handle else { return fallback }
            let expression = try module.globalExpressions.value(for: handle)
            let literal = try evaluateWGSLPipelineScalar(handle, in: module, overrides: values, budget: &budget)
            guard let size = unsignedSize(literal), size > 0 else {
                throw failure(.invalidBinding, "Specialized workgroup dimension must be positive", expression.location)
            }
            return size
        }
        return try .init(x: dimension(dimensions.x, fallback: fixed.x),
                         y: dimension(dimensions.y, fallback: fixed.y),
                         z: dimension(dimensions.z, fallback: fixed.z))
    }

    private mutating func subgroupSize(_ entry: IREntryPoint) throws -> IRSubgroupSize? {
        guard case .overrideExpression(let handle)? = entry.subgroupSize else { return entry.subgroupSize }
        let literal = try evaluateWGSLPipelineScalar(handle, in: module, overrides: values, budget: &budget)
        guard let size = unsignedSize(literal), size.nonzeroBitCount == 1 else {
            let location = try module.globalExpressions.value(for: handle).location
            throw failure(.invalidBinding, "Specialized subgroup size must be a power of two", location)
        }
        return .fixed(size)
    }

    private mutating func specializeArrayTypes(_ live: Set<IRHandle<IRType>>) throws -> IRTable<IRType> {
        try .init(module.types.values.enumerated().map { index, type in
            try budget.consume(.work)
            let handle: IRHandle<IROverride>
            switch type.kind {
            case .array(_, .override(let value), _), .bindingArray(_, .override(let value)): handle = value
            default: return type
            }
            let count: UInt32
            if live.contains(try IRHandle(index: index)) {
                guard let size = values[handle].flatMap(unsignedSize), size > 0 else {
                    let value = try module.overrides.value(for: handle)
                    throw failure(.invalidType, "Specialized array count must be positive", value.location)
                }
                count = size
            } else {
                // No retained function accesses this type.
                count = 1
            }
            let kind: IRTypeKind
            switch type.kind {
            case .array(let base, _, let stride): kind = .array(base: base, size: .constant(count), stride: stride)
            case .bindingArray(let base, _): kind = .bindingArray(base: base, size: .constant(count))
            default: throw ShaderFailure.invalidHandle
            }
            return .init(name: type.name, kind: kind, location: type.location)
        })
    }

    private mutating func stripPipelineGlobals() throws -> [IRExpression] {
        var stages: [IRPipelineEvaluationStage] = []
        return try module.globalExpressions.values.map { expression in
            try budget.consume(.work)
            let type = try module.types.value(for: expression.type).kind
            let stage = expression.kind.pipelineEvaluationStage(type: type) { stages[$0.index] }
            stages.append(stage)
            // Only declarations and entry-point metadata reference these expressions, and they
            // have already been resolved. Keep their slots without revalidating skipped defaults.
            guard stage == .pipeline else { return expression }
            return .init(type: expression.type, kind: .zeroValue(expression.type), location: expression.location)
        }
    }

    private mutating func expressions(
        _ table: IRTable<IRExpression>, functionRemap: [IRHandle<IRFunction>: IRHandle<IRFunction>],
        replacements: [IRHandle<IRExpression>: IRExpressionKind], useOverrideValues: Bool
    ) throws -> IRTable<IRExpression> {
        try .init(table.values.enumerated().map { index, expression in
            try budget.consume(.work)
            if !replacements.isEmpty, let kind = replacements[try IRHandle(index: index)] {
                return .init(type: expression.type, kind: kind, location: expression.location)
            }
            if case .callResult(let target) = expression.kind {
                guard let mapped = functionRemap[target] else { throw ShaderFailure.invalidHandle }
                return .init(type: expression.type, kind: .callResult(mapped), location: expression.location)
            }
            guard case .override(let handle) = expression.kind else { return expression }
            // Unused declarations remain well-typed for validation and are outside the generated entry point.
            let value = try (useOverrideValues ? values[handle] : nil) ?? zero(expression.type)
            return .init(type: expression.type, kind: .literal(value), location: expression.location)
        })
    }

    private func zero(_ type: IRHandle<IRType>) throws -> IRLiteral {
        switch try module.types.value(for: type).kind {
        case .scalar(.bool): return .bool(false)
        case .scalar(.i16): return .i16(0)
        case .scalar(.u16): return .u16(0)
        case .scalar(.i32): return .i32(0)
        case .scalar(.u32): return .u32(0)
        case .scalar(.f32): return .f32(0)
        case .scalar(.f16): return .f16Bits(0)
        default: throw ShaderFailure.invalidHandle
        }
    }

    private func literal(_ number: Double, for declaration: IROverride) throws -> IRLiteral {
        let invalid = failure(.invalidType, "Pipeline constant is out of range for its scalar type", declaration.location)
        guard number.isFinite else { throw invalid }
        switch try module.types.value(for: declaration.type).kind {
        case .scalar(.bool): return .bool(number != 0)
        case .scalar(.i16):
            guard let value = Int16(exactly: number.rounded(.towardZero)) else { throw invalid }
            return .i16(value)
        case .scalar(.u16):
            guard let value = UInt16(exactly: number.rounded(.towardZero)) else { throw invalid }
            return .u16(value)
        case .scalar(.i32):
            guard let value = Int32(exactly: number.rounded(.towardZero)) else { throw invalid }
            return .i32(value)
        case .scalar(.u32):
            guard let value = UInt32(exactly: number.rounded(.towardZero)) else { throw invalid }
            return .u32(value)
        case .scalar(.f32):
            let value = Float(number)
            guard value.isFinite else { throw invalid }
            return .f32(value)
        case .scalar(.f16):
            let value = Binary16.encode(Double(Float(number)))
            guard value & 0x7c00 != 0x7c00 else { throw invalid }
            return .f16Bits(value)
        default: throw ShaderFailure.invalidHandle
        }
    }

    private func unsignedSize(_ literal: IRLiteral) -> UInt32? {
        switch literal {
        case .i16(let value): UInt32(exactly: value)
        case .u16(let value): UInt32(value)
        case .i32(let value): UInt32(exactly: value)
        case .u32(let value): value
        default: nil
        }
    }

    private func failure(_ code: ShaderDiagnosticCode, _ message: String, _ location: ShaderLocation? = nil) -> ShaderFailure {
        .diagnostics([.init(severity: .error, code: code, message: message, location: location)])
    }
}
