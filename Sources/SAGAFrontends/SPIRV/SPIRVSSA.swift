import SAGAShaderIR

struct SPVValueDefinition {
    let block: UInt32
    let value: SPVValue
}

struct SPVDominancePair: Hashable {
    let definition: UInt32
    let use: UInt32
}

enum SPVAccessIndex {
    case constant(UInt32)
    case dynamic(SPVValue)
}

struct SPVAccessStep {
    let type: IRHandle<IRType>
    let index: SPVAccessIndex
}

struct SPVAccessPath {
    let base: SPVValue
    let steps: [SPVAccessStep]
}

extension SPVFunctionLowerer {
    mutating func defineValue(_ id: UInt32, _ value: SPVValue) {
        values[id] = value
        definitions[id] = .init(block: currentBlock, value: value)
    }

    mutating func reload(
        _ definition: SPVValueDefinition, in use: UInt32,
        statements: inout [IRHandle<IRStatement>], at instruction: SPVInstruction
    ) throws -> SPVValue {
        try proveDominance(of: definition.block, at: use, instruction)
        return try restore(definition.value, statements: &statements, at: instruction)
    }

    mutating func project(
        _ base: IRHandle<IRExpression>, through step: SPVAccessStep,
        statements: inout [IRHandle<IRStatement>], at instruction: SPVInstruction
    ) throws -> IRHandle<IRExpression> {
        let access: IRExpressionKind
        switch step.index {
        case .constant(let index): access = .accessIndex(base: base, index: index)
        case .dynamic(let index): access = .access(base: base, index: index.expression)
        }
        let expression = try expr(step.type, access, instruction)
        statements.append(try stmt(.emit([expression]), instruction))
        return expression
    }

    private mutating func restore(
        _ value: SPVValue, statements: inout [IRHandle<IRStatement>], at instruction: SPVInstruction
    ) throws -> SPVValue {
        guard let anchor = value.anchor else { return value }
        if let path = accessPaths[value.expression] {
            try budget.enterDepth()
            defer { try? budget.leaveDepth() }
            // Rebuild the path with the original index SSA snapshots, including those
            // used by a computed base. Only the index values need function storage.
            var expression = try restore(path.base, statements: &statements, at: instruction).expression
            for step in path.steps {
                let projection: SPVAccessStep
                if case .dynamic(let index) = step.index {
                    let captured = try restore(index, statements: &statements, at: instruction)
                    projection = .init(type: step.type, index: .dynamic(captured))
                } else {
                    projection = step
                }
                expression = try project(expression, through: projection, statements: &statements, at: instruction)
            }
            accessPaths[expression] = path
            return .init(expression: expression, type: value.type, anchor: statements.last)
        }
        let type = try module.type(value.type, instruction)
        let slot: IRHandle<IRExpression>
        if let existing = spillSlots[value.expression] {
            slot = existing
        } else {
            guard let pointer = module.localPointerTypes[type] else {
                throw instruction.fail("Pointer escapes its structured expression scope", code: .unsupportedConstruct)
            }
            try node()
            let local = try IRHandle<IRLocalVariable>(index: locals.count)
            let location = self.statements[anchor.index].location
            locals.append(.init(type: type, location: location))
            slot = try expr(pointer, .localVariable(local), instruction)
            let store = try stmt(.store(pointer: slot, value: value.expression), instruction)
            spillSlots[value.expression] = slot
            spillWrites[anchor, default: []].append(store)
        }
        let expression = try expr(type, .load(pointer: slot), instruction)
        let read = try stmt(.emit([expression]), instruction)
        statements.append(read)
        return .init(expression: expression, type: value.type, anchor: read)
    }

    private mutating func proveDominance(
        of definition: UInt32, at use: UInt32, _ instruction: SPVInstruction
    ) throws {
        if definition == use || definition == entryLabel { return }
        let pair = SPVDominancePair(definition: definition, use: use)
        if provenDominance.contains(pair) { return }
        // For an out-of-scope value, search for a path from the use back to entry
        // that bypasses the definition. Phi reads pass the incoming edge's source here.
        var pending = [use]
        var seen: Set<UInt32> = [use]
        var foundDefinition = false
        while let label = pending.popLast() {
            try work()
            if label == definition {
                foundDefinition = true
                continue
            }
            guard label != entryLabel else { throw instruction.fail("SSA definition does not dominate its use") }
            for predecessor in predecessors[label, default: []] {
                try work()
                if seen.insert(predecessor).inserted { pending.append(predecessor) }
            }
        }
        guard foundDefinition else { throw instruction.fail("SSA definition does not reach its use") }
        try node()
        provenDominance.insert(pair)
    }

    mutating func insertSpillWrites() throws {
        guard !spillWrites.isEmpty else { return }
        for index in irBlocks.indices {
            let original = irBlocks[index]
            try work(original.statements.count + 1)
            var expanded: [IRHandle<IRStatement>] = []
            var start = 0
            for (offset, anchor) in original.statements.enumerated() {
                guard let writes = spillWrites.removeValue(forKey: anchor) else { continue }
                expanded.append(contentsOf: original.statements[start...offset])
                expanded.append(contentsOf: writes)
                start = offset + 1
            }
            if start > 0 {
                expanded.append(contentsOf: original.statements[start...])
                irBlocks[index] = .init(statements: expanded, location: original.location)
            }
        }
        guard spillWrites.isEmpty else { throw record.header.fail("Missing SSA spill insertion point") }
    }
}
