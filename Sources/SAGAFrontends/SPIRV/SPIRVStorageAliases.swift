import SAGAShaderIR

enum SPVStorageAliasDecoration {
    static let restrict: UInt32 = 19
    static let aliased: UInt32 = 20
}

private struct SPVStorageAliasTypePair: Hashable {
    let canonical: IRHandle<IRType>
    let alias: IRHandle<IRType>
}

private enum SPVStorageAliasLayout {
    case unchanged
    case mixedFloatWords
}

extension SPVDecoder {
    func canonicalStorageGlobal(_ id: IRHandle<IRGlobalVariable>) -> IRHandle<IRGlobalVariable> {
        storageAliases[id] ?? id
    }

    func validateStorageAliasDecoration(_ id: UInt32, _ decorations: [UInt32: [UInt32]], at instruction: SPVInstruction) throws {
        guard let (global, _) = globalMap[id], case .storage = globals[global.index].space else {
            throw instruction.fail("Restrict and Aliased are only implemented for storage-buffer variables", code: .unsupportedConstruct)
        }
        guard decorations[SPVStorageAliasDecoration.restrict] == nil || decorations[SPVStorageAliasDecoration.aliased] == nil else {
            throw instruction.fail("A storage-buffer variable cannot have both Restrict and Aliased")
        }
    }

    mutating func prepareStorageAliases(_: SPVInstruction) throws {
        typealias Global = IRHandle<IRGlobalVariable>
        var restricted: Set<Global> = []
        for (id, (global, _)) in globalMap {
            try charge()
            if decorations[id]?[SPVStorageAliasDecoration.restrict] != nil {
                restricted.insert(global)
            }
        }

        // Incompatible views can be valid when used by different entry points.
        // Keep them distinct and let entry-point validation check simultaneous binding use.
        var representatives: [IRResourceBinding: [Global]] = [:]
        var wordLayouts: [SPVStorageAliasTypePair: IRHandle<IRType>] = [:]
        for index in globals.indices {
            try charge()
            let view = globals[index]
            guard let binding = view.binding, case .storage(let viewAccess) = view.space,
                  view.initializer == nil else { continue }
            let handle = try Global(index: index)
            guard !restricted.contains(handle) else { continue }

            for canonical in representatives[binding] ?? [] {
                try charge()
                let primary = globals[canonical.index]
                guard case .storage(let primaryAccess) = primary.space,
                      primary.memoryDecorations == view.memoryDecorations,
                      let layout = try storageAliasLayout(primary.type, view.type) else { continue }
                // The physical binding supports every view's access. Original decorations
                // still govern each SPIR-V pointer independently.
                let access = primaryAccess.union(viewAccess)
                let type = if layout == .mixedFloatWords {
                    try storageAliasWordType(.init(canonical: primary.type, alias: view.type), cached: &wordLayouts)
                } else { primary.type }
                if access != primaryAccess || type != primary.type {
                    globals[canonical.index] = .init(name: primary.name, space: .storage(access: access), binding: primary.binding,
                        type: type, initializer: primary.initializer, memoryDecorations: primary.memoryDecorations, location: primary.location)
                }
                storageAliases[handle] = canonical
                break
            }
            if storageAliases[handle] == nil {
                representatives[binding, default: []].append(handle)
            }
        }
    }

    private mutating func storageAliasLayout(
        _ canonical: IRHandle<IRType>, _ alias: IRHandle<IRType>
    ) throws -> SPVStorageAliasLayout? {
        var layout = SPVStorageAliasLayout.unchanged
        var checked: Set<SPVStorageAliasTypePair> = []
        var pending = [SPVStorageAliasTypePair(canonical: canonical, alias: alias)]
        while let pair = pending.popLast() {
            try charge()
            guard pair.canonical != pair.alias, checked.insert(pair).inserted else { continue }
            switch (types[pair.canonical.index].kind, types[pair.alias.index].kind) {
            case (.scalar(let left), .scalar(let right)):
                guard compatibleStorageAliasScalar(left, right) else { return nil }
                if isMixedFloatWord(left, right) { layout = .mixedFloatWords }
            case (.vector(let leftWidth, let left), .vector(let rightWidth, let right)):
                guard leftWidth == rightWidth, compatibleStorageAliasScalar(left, right) else { return nil }
            case (.matrix(let leftColumns, let leftRows, let left), .matrix(let rightColumns, let rightRows, let right)):
                guard leftColumns == rightColumns, leftRows == rightRows, left == right else { return nil }
            case (.array(let left, let leftSize, let leftStride), .array(let right, let rightSize, let rightStride)):
                guard leftSize == rightSize, leftStride == rightStride else { return nil }
                pending.append(.init(canonical: left, alias: right))
            case (.structure(let left, let leftSpan), .structure(let right, let rightSpan)):
                guard leftSpan == rightSpan, left.count == right.count else { return nil }
                try charge(left.count)
                for (leftField, rightField) in zip(left, right) {
                    guard leftField.offset == rightField.offset, leftField.binding == rightField.binding else { return nil }
                    pending.append(.init(canonical: leftField.type, alias: rightField.type))
                }
            default:
                return nil
            }
        }
        return layout
    }

    private func isMixedFloatWord(_ left: IRScalar, _ right: IRScalar) -> Bool {
        left.width == 4 && (left == .f32) != (right == .f32)
    }

    private mutating func storageAliasWordType(
        _ pair: SPVStorageAliasTypePair, cached: inout [SPVStorageAliasTypePair: IRHandle<IRType>]
    ) throws -> IRHandle<IRType> {
        try charge()
        if pair.canonical == pair.alias { return pair.canonical }
        if let type = cached[pair] { return type }
        try budget.enterDepth()
        defer { try? budget.leaveDepth() }
        let original = types[pair.canonical.index].kind
        let normalized: IRTypeKind
        switch (original, types[pair.alias.index].kind) {
        case (.scalar(let left), .scalar(let right)) where isMixedFloatWord(left, right):
            // Float and integer atomics share one word, with the arithmetic selected by the opcode.
            normalized = .scalar(.u32)
        case (.array(let left, let size, let stride), .array(let right, _, _)):
            let element = try storageAliasWordType(.init(canonical: left, alias: right), cached: &cached)
            normalized = .array(base: element, size: size, stride: stride)
        case (.structure(let left, let span), .structure(let right, _)):
            var fields = left
            for (index, field) in left.enumerated() {
                let type = try storageAliasWordType(.init(canonical: field.type, alias: right[index].type), cached: &cached)
                if type != field.type {
                    fields[index] = .init(name: field.name, type: type, binding: field.binding, offset: field.offset, location: field.location)
                }
            }
            normalized = .structure(members: fields, span: span)
        default:
            normalized = original
        }
        let type = normalized == original ? pair.canonical : try internType(normalized)
        cached[pair] = type
        return type
    }

    private func compatibleStorageAliasScalar(_ left: IRScalar, _ right: IRScalar) -> Bool {
        guard left.width == right.width, [.sint, .uint, .float].contains(left.kind),
              [.sint, .uint, .float].contains(right.kind) else { return false }
        // Shared bitcast writers support integer reinterpretation and f16/f32 bit patterns.
        return left == right || left.width != 8 || (left.kind != .float && right.kind != .float)
    }

    private func storageAliasFailure(
        _ message: String, code: ShaderDiagnosticCode, global: IRGlobalVariable, at instruction: SPVInstruction
    ) -> ShaderFailure {
        .diagnostics([.init(severity: .error, code: code, message: message, location: global.location ?? instruction.location)])
    }

    mutating func compactStorageAliases(in functions: inout [IRFunction], at instruction: SPVInstruction) throws {
        guard !storageAliases.isEmpty else { return }
        try charge(globals.count)
        var mapping: [IRHandle<IRGlobalVariable>] = []
        var retained: [IRGlobalVariable] = []
        for (index, global) in globals.enumerated() {
            let original = try IRHandle<IRGlobalVariable>(index: index)
            let canonical = canonicalStorageGlobal(original)
            if canonical == original {
                mapping.append(try IRHandle(index: retained.count))
                retained.append(global)
            } else {
                guard mapping.indices.contains(canonical.index) else { throw instruction.fail("Invalid storage alias representative") }
                mapping.append(mapping[canonical.index])
            }
        }

        globalExpressions = try remappingStorageGlobals(globalExpressions, mapping: mapping, at: instruction)
        for index in functions.indices {
            try charge()
            let function = functions[index]
            let expressions = try remappingStorageGlobals(function.expressions.values, mapping: mapping, at: instruction)
            functions[index] = .init(name: function.name, arguments: function.arguments, result: function.result,
                localVariables: function.localVariables, expressions: .init(expressions), namedExpressions: function.namedExpressions,
                statements: function.statements, blocks: function.blocks, body: function.body,
                diagnosticFilter: function.diagnosticFilter, location: function.location)
        }
        globals = retained
    }

    private mutating func remappingStorageGlobals(
        _ expressions: [IRExpression], mapping: [IRHandle<IRGlobalVariable>], at instruction: SPVInstruction
    ) throws -> [IRExpression] {
        try charge(expressions.count)
        var result = expressions
        for (index, expression) in expressions.enumerated() {
            guard case .globalVariable(let original) = expression.kind else { continue }
            guard mapping.indices.contains(original.index) else { throw instruction.fail("Invalid global reference during alias compaction") }
            let canonical = canonicalStorageGlobal(original)
            if canonical != original {
                let primary = globals[canonical.index]
                guard case .pointer(let pointee, let space) = types[expression.type.index].kind,
                      pointee == primary.type, space == primary.space else {
                    throw storageAliasFailure("Storage alias address was not normalized before compaction",
                                              code: .unsupportedConstruct, global: globals[original.index], at: instruction)
                }
            }
            let mapped = mapping[original.index]
            if mapped != original {
                result[index] = .init(type: expression.type, kind: .globalVariable(mapped), location: expression.location)
            }
        }
        return result
    }
}
