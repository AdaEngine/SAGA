import SAGAShaderIR

enum SPVRegionRoute {
    case walk(UInt32)
    case scopeExit
    case caseFallthrough
    case jump

    var destination: UInt32? {
        if case .walk(let label) = self { return label }
        return nil
    }
}

struct SPVLoopBoundary {
    let header: UInt32
    let continuing: UInt32
    let merge: UInt32
    let switchDepth: Int
    let depth: Int
    var inContinuing = false
    var backedge: UInt32?
    var breakIf: IRHandle<IRExpression>?
    var exitFlag: IRHandle<IRExpression>?
    var relays = 0

    func owns(_ label: UInt32) -> Bool {
        label == header || label == continuing || label == merge
    }
}

extension SPVFunctionLowerer {
    mutating func lowerLoop(_ id: UInt32, depth: Int) throws -> IRStatementKind {
        guard let header = blocks[id], let merge = header.merge, let continuing = header.continuing else {
            throw record.header.fail("Missing loop header or targets")
        }
        if continuing == id, let terminator = header.terminator {
            let targets = terminator.op == 249 ? terminator.a[...] : terminator.a[1...2]
            guard targets.allSatisfy({ $0 == id || $0 == merge }) else {
                throw terminator.fail("Single-block loop branches outside its header and merge")
            }
        }
        let outer = values
        loopBoundaries.append(.init(header: id, continuing: continuing, merge: merge,
                                    switchDepth: switchBoundaries.count, depth: depth))
        let body = try region(id, stop: nil, depth: depth, loopHeader: id)
        let continuation: IRHandle<IRBlock>
        if continuing == id {
            continuation = try block([], header.label)
        } else {
            loopBoundaries[loopBoundaries.count - 1].inContinuing = true
            continuation = try region(continuing, stop: nil, depth: depth)
        }
        let loop = loopBoundaries.removeLast()
        guard loop.backedge != nil else { throw header.label.fail("Loop has no backedge through its continue target") }
        if let flag = loop.exitFlag, let bool = module.controlBoolType {
            let clear = try expr(bool, .literal(.bool(false)), header.label)
            let reset = try stmt(.store(pointer: flag, value: clear), header.label)
            let original = irBlocks[body.index]
            try work(original.statements.count)
            irBlocks[body.index] = .init(statements: [reset] + original.statements, location: original.location)
        }
        values = outer
        return .loop(body: body, continuing: continuation, breakIf: loop.breakIf)
    }

    mutating func loopDestination(
        _ target: UInt32, from source: UInt32, depth: Int, statements: inout [IRHandle<IRStatement>],
        at instruction: SPVInstruction
    ) throws -> SPVRegionRoute? {
        guard let index = loopBoundaries.indices.last else { return nil }
        let loop = loopBoundaries[index]
        if target == loop.header {
            if loop.inContinuing {
                guard depth == loop.depth else {
                    throw instruction.fail("Nested backedge cannot end the continue construct", code: .unsupportedConstruct)
                }
                try recordBackedge(from: source, at: instruction)
                return .scopeExit
            }
            guard loop.header == loop.continuing, source == loop.header else {
                throw instruction.fail("Loop backedge bypasses its continue construct")
            }
            try recordBackedge(from: source, at: instruction)
            statements.append(try stmt(.continue, instruction))
            return .jump
        }
        if target == loop.continuing {
            guard !loop.inContinuing else { throw instruction.fail("Continue construct branches to its own entry") }
            statements.append(try stmt(.continue, instruction))
            return .jump
        }
        if target == loop.merge {
            guard !loop.inContinuing else { throw instruction.fail("Only the backedge block may exit continuing") }
            if switchBoundaries.count > loop.switchDepth {
                let flag = try loopExitFlag(at: instruction)
                guard let bool = module.controlBoolType else { throw ShaderFailure.invalidHandle }
                let set = try expr(bool, .literal(.bool(true)), instruction)
                statements.append(try stmt(.store(pointer: flag, value: set), instruction))
                loopBoundaries[index].relays += 1
            }
            statements.append(try stmt(.break, instruction))
            return .jump
        }
        for outer in loopBoundaries.dropLast() {
            try work()
            if outer.owns(target) { throw instruction.fail("Branch skips the innermost loop") }
        }
        return nil
    }

    private mutating func recordBackedge(from source: UInt32, at instruction: SPVInstruction) throws {
        let index = loopBoundaries.count - 1
        if let previous = loopBoundaries[index].backedge, previous != source {
            throw instruction.fail("Loop has more than one backedge block")
        }
        loopBoundaries[index].backedge = source
    }

    private mutating func loopExitFlag(at instruction: SPVInstruction) throws -> IRHandle<IRExpression> {
        let index = loopBoundaries.count - 1
        if let flag = loopBoundaries[index].exitFlag { return flag }
        guard let bool = module.controlBoolType, let pointer = module.localPointerTypes[bool] else {
            throw ShaderFailure.invalidHandle
        }
        try node()
        let local = try IRHandle<IRLocalVariable>(index: locals.count)
        locals.append(.init(type: bool, location: instruction.location))
        let flag = try expr(pointer, .localVariable(local), instruction)
        loopBoundaries[index].exitFlag = flag
        return flag
    }

    mutating func checkLoopExit(at instruction: SPVInstruction) throws -> [IRHandle<IRStatement>] {
        guard let flag = loopBoundaries.last?.exitFlag, let bool = module.controlBoolType else {
            throw ShaderFailure.invalidHandle
        }
        // Break one switch at a time until the loop is the nearest break target.
        let condition = try expr(bool, .load(pointer: flag), instruction)
        let accept = try block([stmt(.break, instruction)], instruction)
        let reject = try block([], instruction)
        return try [stmt(.emit([condition]), instruction),
                    stmt(.ifCondition(condition: condition, accept: accept, reject: reject), instruction)]
    }

    mutating func lowerUnmergedBranch(
        from source: UInt32, condition: SPVValue, stop: UInt32?, depth: Int,
        statements: inout [IRHandle<IRStatement>], at instruction: SPVInstruction
    ) throws -> UInt32? {
        let yes = instruction.a[1], no = instruction.a[2]
        if yes == no {
            statements += try edge(from: source, to: yes, instruction)
            return try regionDestination(yes, from: source, stop: stop, depth: depth,
                                         statements: &statements, at: instruction).destination
        }
        if let loop = loopBoundaries.last, loop.inContinuing,
           (yes == loop.header && no == loop.merge) || (no == loop.header && yes == loop.merge) {
            guard depth == loop.depth else {
                throw instruction.fail("Nested backedge cannot end the continue construct", code: .unsupportedConstruct)
            }
            try recordBackedge(from: source, at: instruction)
            let accept = try block(edge(from: source, to: yes, instruction), instruction)
            let reject = try block(edge(from: source, to: no, instruction), instruction)
            statements.append(try stmt(.ifCondition(condition: condition.expression, accept: accept, reject: reject), instruction))
            let exit: IRHandle<IRExpression>
            if yes == loop.merge {
                exit = condition.expression
            } else {
                let bool = try module.type(condition.type, instruction)
                exit = try expr(bool, .unary(op: .logicalNot, expression: condition.expression), instruction)
                statements.append(try stmt(.emit([exit]), instruction))
            }
            loopBoundaries[loopBoundaries.count - 1].breakIf = exit
            return nil
        }

        var accept = try edge(from: source, to: yes, instruction)
        var reject = try edge(from: source, to: no, instruction)
        let acceptRoute = try regionDestination(yes, from: source, stop: stop, depth: depth, statements: &accept, at: instruction)
        let rejectRoute = try regionDestination(no, from: source, stop: stop, depth: depth, statements: &reject, at: instruction)
        let acceptTarget = acceptRoute.destination, rejectTarget = rejectRoute.destination
        let next: UInt32?
        let tail: [IRHandle<IRStatement>]
        switch (acceptRoute, rejectRoute) {
        case (.walk(let label), .jump):
            next = label
            tail = accept
            accept = []
        case (.jump, .walk(let label)):
            next = label
            tail = reject
            reject = []
        case (.walk(let label), .scopeExit) where no == stop:
            let saved = values
            accept.append(try stmt(.block(region(label, stop: stop, depth: depth + 1)), instruction))
            values = saved
            next = nil
            tail = []
        case (.scopeExit, .walk(let label)) where yes == stop:
            let saved = values
            reject.append(try stmt(.block(region(label, stop: stop, depth: depth + 1)), instruction))
            values = saved
            next = nil
            tail = []
        case (.caseFallthrough, .scopeExit) where stop != nil && stop == switchBoundaries.last?.merge:
            reject.append(try stmt(.break, instruction))
            next = nil
            tail = []
        case (.scopeExit, .caseFallthrough) where stop != nil && stop == switchBoundaries.last?.merge:
            accept.append(try stmt(.break, instruction))
            next = nil
            tail = []
        case (.scopeExit, .caseFallthrough), (.caseFallthrough, .scopeExit):
            throw instruction.fail("Conditional fall-through crosses its region", code: .unsupportedConstruct)
        default:
            guard acceptTarget == nil, rejectTarget == nil else {
                throw instruction.fail("Conditional needs a merge or a structured exit", code: .unsupportedConstruct)
            }
            next = nil
            tail = []
        }
        let acceptBlock = try block(accept, instruction)
        let rejectBlock = try block(reject, instruction)
        statements.append(try stmt(.ifCondition(condition: condition.expression, accept: acceptBlock, reject: rejectBlock), instruction))
        statements += tail
        return next
    }
}
