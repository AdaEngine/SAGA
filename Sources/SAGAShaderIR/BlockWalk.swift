extension IRFunction {
    /// Bounded iterative structural walk, NOT semantic validation or a witness.
    /// Rejects dangling block/statement indices and cycles reachable from body.
    /// Re-visits shared blocks per occurrence, charging work; no recursive Swift values or calls.
    public func walkBlocks(budget: inout ShaderBudget, visit: (IRBlock) throws -> Void) throws {
        var stack: [(IRHandle<IRBlock>, Int, Bool)] = [(body, 1, false)]
        var active: Set<IRHandle<IRBlock>> = []
        while let (handle, depth, leaving) = stack.popLast() {
            try budget.consume(.work)
            if leaving { active.remove(handle); continue }
            guard depth <= budget.limits.depth else { throw ShaderFailure.budgetExceeded(.depth) }
            guard active.insert(handle).inserted else {
                throw ShaderFailure.diagnostics([.init(severity: .error, code: .invalidReference, message: "Cyclic block graph")])
            }
            let block = try blocks.value(for: handle)
            try visit(block)
            stack.append((handle, depth, true))
            // Reverse traversal preserves source order. Charge before pushing each child.
            for statementHandle in block.statements.reversed() {
                try budget.consume(.work)
                let statement = try statements.value(for: statementHandle)
                let children: [IRHandle<IRBlock>]
                switch statement.kind {
                case .block(let child): children = [child]
                case .ifCondition(_, let accept, let reject): children = [accept, reject]
                case .switchValue(_, let cases):
                    // Do not materialize an unbounded child list before checking the quota.
                    for switchCase in cases.reversed() {
                        try budget.consume(.work)
                        stack.append((switchCase.body, depth + 1, false))
                    }
                    continue
                case .loop(let body, let continuing, _): children = [body, continuing]
                default: children = []
                }
                for child in children.reversed() {
                    try budget.consume(.work)
                    stack.append((child, depth + 1, false))
                }
            }
        }
    }
}
