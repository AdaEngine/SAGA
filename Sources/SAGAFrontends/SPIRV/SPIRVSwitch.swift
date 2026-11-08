import SAGAShaderIR

struct SPVSwitchBoundary {
    let entry: UInt32
    let merge: UInt32
    let targets: Set<UInt32>
    let depth: Int
    let loopDepth: Int
    var successor: UInt32?
}

extension SPVFunctionLowerer {
    func switchSelectorType(_ instruction: SPVInstruction) throws -> IRScalar {
        let id = instruction.a[0]
        guard let type = module.functionValueTypes[id] ?? module.constantMap[id]?.type ?? module.globalMap[id]?.1 else {
            throw instruction.fail("Undefined OpSwitch selector")
        }
        let scalar = try module.scalar(type, instruction)
        guard scalar.kind == .sint || scalar.kind == .uint else {
            throw instruction.fail("OpSwitch requires an integer scalar")
        }
        guard scalar.width == 4 else {
            throw instruction.fail("Only 32-bit switch selectors are supported", code: .unsupportedConstruct)
        }
        return scalar
    }

    mutating func regionDestination(
        _ target: UInt32, from source: UInt32, stop: UInt32?, depth: Int,
        statements: inout [IRHandle<IRStatement>], at instruction: SPVInstruction
    ) throws -> SPVRegionRoute {
        if target == stop { return .scopeExit }
        if let route = try loopDestination(target, from: source, depth: depth, statements: &statements, at: instruction) {
            return route
        }
        guard let boundary = switchBoundaries.last else { return .walk(target) }
        if target == boundary.merge {
            guard loopBoundaries.count == boundary.loopDepth else {
                throw instruction.fail("Switch break crosses a loop")
            }
            statements.append(try stmt(.break, instruction))
            return .jump
        }
        if boundary.targets.contains(target) {
            guard target != boundary.entry, depth == boundary.depth, loopBoundaries.count == boundary.loopDepth else {
                throw instruction.fail("Nested or cyclic switch case transfer", code: .unsupportedConstruct)
            }
            if let successor = boundary.successor, successor != target {
                throw instruction.fail("Switch case has more than one fall-through target")
            }
            switchBoundaries[switchBoundaries.count - 1].successor = target
            return .caseFallthrough
        }
        return .walk(target)
    }

    mutating func lowerSwitch(
        from header: UInt32, merge: UInt32, instruction: SPVInstruction, depth: Int
    ) throws -> [IRHandle<IRStatement>] {
        guard depth <= budget.limits.depth else { throw ShaderFailure.budgetExceeded(.depth) }
        var statements: [IRHandle<IRStatement>] = []
        let selector = try value(instruction.a[0], instruction, &statements, from: header)
        let scalar = try module.scalar(selector.type, instruction)
        let defaultTarget = instruction.a[1]
        var targets: [UInt32] = []
        var explicitPosition: [UInt32: Int] = [:]
        var labels: [UInt32: [IRSwitchValue]] = [:]
        var literals: Set<UInt32> = []
        for index in stride(from: 2, to: instruction.a.count, by: 2) {
            try work()
            let literal = instruction.a[index], target = instruction.a[index + 1]
            guard literals.insert(literal).inserted else { throw instruction.fail("Duplicate switch literal") }
            if targets.last != target {
                guard explicitPosition[target] == nil else { throw instruction.fail("Nonconsecutive switch target labels") }
                explicitPosition[target] = targets.count
                targets.append(target)
            }
            let label = scalar.kind == .uint ? IRSwitchValue.u32(literal) : .i32(Int32(bitPattern: literal))
            labels[target, default: []].append(label)
        }
        if labels[defaultTarget] == nil { targets.append(defaultTarget) }
        labels[defaultTarget, default: []].append(.default)

        let targetSet = Set(targets)
        let saved = values
        var bodies: [UInt32: IRHandle<IRBlock>] = [:]
        var successors: [UInt32: UInt32] = [:]
        var incoming: Set<UInt32> = []
        var dispatch: [IRSwitchCase] = []
        var needsDispatch = false
        for target in targets {
            try work()
            values = saved
            var entry = try edge(from: header, to: target, instruction)
            // A case with both dispatch and fall-through predecessors gets its dispatch stores
            // in a separate selection. Falling through must preserve the other edge's stores.
            if target != merge, loopBoundaries.last?.owns(target) != true,
               predecessors[target, default: []].count > 1, !entry.isEmpty {
                let body = try block(entry, instruction)
                for label in labels[target, default: []] {
                    try node()
                    dispatch.append(.init(value: label, body: body))
                }
                needsDispatch = true
                entry = []
            }
            if target != merge {
                switchBoundaries.append(.init(entry: target, merge: merge, targets: targetSet,
                                              depth: depth, loopDepth: loopBoundaries.count))
                let body: IRHandle<IRBlock>?
                if try loopDestination(target, from: header, depth: depth, statements: &entry, at: instruction) != nil {
                    body = nil
                } else {
                    body = try region(target, stop: merge, depth: depth)
                }
                if let successor = switchBoundaries.removeLast().successor {
                    guard incoming.insert(successor).inserted else {
                        throw instruction.fail("Switch case has multiple fall-through predecessors")
                    }
                    successors[target] = successor
                }
                if let body {
                    if entry.isEmpty {
                        bodies[target] = body
                        continue
                    }
                    entry.append(try stmt(.block(body), instruction))
                }
            }
            bodies[target] = try block(entry, instruction)
        }

        var cases: [IRSwitchCase] = []
        var emitted: Set<UInt32> = []
        for root in targets where !incoming.contains(root) {
            try work()
            var target: UInt32? = root
            var previousExplicit: Int?
            while let id = target {
                try work()
                guard emitted.insert(id).inserted, let body = bodies[id] else {
                    throw instruction.fail("Invalid switch case chain")
                }
                if let position = explicitPosition[id] {
                    if let previousExplicit, position != previousExplicit + 1 {
                        throw instruction.fail("Switch fall-through skips an explicit case")
                    }
                    previousExplicit = position
                }
                target = successors[id]
                for label in labels[id, default: []] {
                    try node()
                    cases.append(.init(value: label, body: body, fallThrough: target != nil))
                }
            }
        }
        guard emitted.count == targets.count else { throw instruction.fail("Cyclic switch fall-through") }
        values = saved
        if needsDispatch {
            if !dispatch.contains(where: { $0.value == .default }) {
                dispatch.append(.init(value: .default, body: try block([], instruction)))
            }
            statements.append(try stmt(.switchValue(selector: selector.expression, cases: dispatch), instruction))
        }
        statements.append(try stmt(.switchValue(selector: selector.expression, cases: cases), instruction))
        return statements
    }
}
