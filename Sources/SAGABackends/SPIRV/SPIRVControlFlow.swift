import SAGAShaderIR

private struct SPIRVSwitchCaseBlock {
    let body: IRHandle<IRBlock>
    let label: UInt32
    let fallThrough: Bool
}

extension SPIRVBinaryWriter {
    mutating func writeLoop(
        body: IRHandle<IRBlock>, continuing: IRHandle<IRBlock>, breakIf: IRHandle<IRExpression>?,
        in state: inout SPIRVFunctionState, targets _: SPIRVBranchTargets
    ) throws -> Bool {
        let outerScope = state.expressionScope
        let header = try allocateID()
        let bodyLabel = try allocateID()
        let continueLabel = try allocateID()
        let merge = try allocateID()
        try emit(249, [header]) // OpBranch
        try emit(248, [header]) // OpLabel
        try emit(246, [merge, continueLabel, 0]) // OpLoopMerge
        try emit(249, [bodyLabel])

        try emit(248, [bodyLabel])
        let bodyTargets = SPIRVBranchTargets(breakTarget: merge, continueTarget: continueLabel)
        let bodyTerminates = try writeBlock(body, in: &state, targets: bodyTargets)
        if !bodyTerminates { try emit(249, [continueLabel]) }
        let entersContinuing = !bodyTerminates || state.explicitBranchTargets.contains(continueLabel)

        try emit(248, [continueLabel])
        if entersContinuing {
            // The continuing block can reference values emitted in the loop body's outer scope.
            let continuingTargets = SPIRVBranchTargets(breakTarget: nil, continueTarget: nil)
            if try writeBlock(continuing, in: &state, targets: continuingTargets) {
                throw unsupported("cannot lower a continuing block without a backedge")
            }
        }
        if entersContinuing, let breakIf {
            let condition = try value(breakIf, in: &state, emitted: false)
            try emit(250, [condition, merge, header]) // OpBranchConditional
        } else {
            // The loop requires one backedge even when every body path exits before continuing.
            try emit(249, [header])
        }

        try restoreExpressions(to: outerScope, in: &state)
        try emit(248, [merge])
        if !state.explicitBranchTargets.contains(merge), !(entersContinuing && breakIf != nil) {
            try emit(255) // OpUnreachable
            return true
        }
        return false
    }

    mutating func writeSwitch(
        selector: IRHandle<IRExpression>, cases: [IRSwitchCase],
        in state: inout SPIRVFunctionState, targets: SPIRVBranchTargets
    ) throws -> Bool {
        let selectorID = try value(selector, in: &state, emitted: false)
        let outerScope = state.expressionScope
        let merge = try allocateID()
        var defaultLabel: UInt32?
        var groups: [SPIRVSwitchCaseBlock] = []
        var literalTargets: [UInt32] = []
        for entry in cases {
            try charge()
            let label: UInt32
            if let previous = groups.last, previous.body == entry.body {
                label = previous.label
            } else {
                label = try allocateID()
                groups.append(.init(body: entry.body, label: label, fallThrough: entry.fallThrough))
            }
            switch entry.value {
            case .i32(let value): literalTargets += [UInt32(bitPattern: value), label]
            case .u32(let value): literalTargets += [value, label]
            case .default: defaultLabel = label
            }
        }
        guard let defaultLabel else { throw unsupported("requires a switch default case") }
        try emit(247, [merge, 0]) // OpSelectionMerge
        try emit(251, [selectorID, defaultLabel] + literalTargets) // OpSwitch

        let caseTargets = SPIRVBranchTargets(breakTarget: merge, continueTarget: targets.continueTarget)
        var reachesMerge = false
        for (index, group) in groups.enumerated() {
            try charge()
            try restoreExpressions(to: outerScope, in: &state)
            try emit(248, [group.label])
            if try !writeBlock(group.body, in: &state, targets: caseTargets) {
                if group.fallThrough {
                    guard index + 1 < groups.count else { throw unsupported("cannot fall through the final switch case") }
                    try emit(249, [groups[index + 1].label])
                } else {
                    try emit(249, [merge])
                    reachesMerge = true
                }
            }
        }
        try restoreExpressions(to: outerScope, in: &state)
        try emit(248, [merge])
        if !reachesMerge, !state.explicitBranchTargets.contains(merge) {
            try emit(255)
            return true
        }
        return false
    }
}
