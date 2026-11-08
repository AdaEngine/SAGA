import SAGAShaderIR

package struct ShaderPointerAccesses {
    private enum Root: Hashable {
        case local(IRHandle<IRLocalVariable>)
        case global(IRHandle<IRGlobalVariable>)
        case parameter(Int)
    }

    private struct Mode: OptionSet {
        let rawValue: UInt8
        static let read = Self(rawValue: 1)
        static let write = Self(rawValue: 2)
    }

    private var effects: [Root: Mode] = [:]

    package var writtenParameters: Set<Int> {
        Set(effects.compactMap { root, mode in
            if case .parameter(let slot) = root, mode.contains(.write) { return slot }
            return nil
        })
    }

    package static func checking(
        _ function: IRFunction, in module: IRModule, callees: [Int: Self], budget: inout ShaderBudget
    ) throws -> Self {
        var result = Self()
        var roots: [Root?] = []
        for expression in function.expressions.values {
            try budget.consume(.work)
            let root: Root?
            switch expression.kind {
            case .localVariable(let handle): root = .local(handle)
            case .globalVariable(let handle): root = .global(handle)
            case .functionArgument(let slot):
                if case .pointer = try module.types.value(for: expression.type).kind {
                    root = .parameter(Int(slot))
                } else {
                    root = nil
                }
            case .access(let parent, _), .accessIndex(let parent, _), .runtimeValue(let parent):
                root = roots[parent.index]
            default: root = nil
            }
            roots.append(root)
        }
        func record(_ root: Root?, _ mode: Mode) throws {
            guard let root, !mode.isEmpty else { return }
            if case .local = root { return }
            try budget.consume(.work)
            if result.effects[root] == nil { try budget.consume(.nodes) }
            result.effects[root, default: []].formUnion(mode)
        }
        for expression in function.expressions.values {
            try budget.consume(.work)
            if case .load(let pointer) = expression.kind { try record(roots[pointer.index], .read) }
        }
        for statement in function.statements.values {
            try budget.consume(.work)
            switch statement.kind {
            case .store(let pointer, _):
                try record(roots[pointer.index], .write)
            case .atomic(let pointer, _, _, _):
                try record(roots[pointer.index], [.read, .write])
            case .workgroupUniformLoad(let pointer, _):
                try record(roots[pointer.index], .read)
            case .call(let target, let arguments, _):
                guard let callee = callees[target.index] else { throw ShaderFailure.invalidHandle }
                let signature = try module.functions.value(for: target).arguments
                var aliases: [Root: Mode] = [:]
                for (slot, argument) in arguments.enumerated() {
                    try budget.consume(.work)
                    let parameter = signature[slot]
                    guard case .pointer = try module.types.value(for: parameter.type).kind else { continue }
                    let expression = try function.expressions.value(for: argument)
                    switch expression.kind {
                    case .localVariable, .globalVariable, .functionArgument:
                        break
                    case .access, .accessIndex:
                        guard module.allowsPointerSubviews else {
                            throw invalid("Pointer argument to a subview requires unrestricted_pointer_parameters",
                                          at: statement.location)
                        }
                    default:
                        throw invalid("Pointer argument must address a whole variable or parameter", at: statement.location)
                    }
                    guard let root = roots[argument.index] else { throw ShaderFailure.invalidHandle }
                    let access = callee.effects[.parameter(slot), default: []]
                    if let previous = aliases[root], access.contains(.write) || previous.contains(.write) {
                        throw invalid("Aliased pointer arguments include a write", at: statement.location)
                    }
                    aliases[root, default: []].formUnion(access)
                    if case .global = root, let global = callee.effects[root],
                       !access.isEmpty, access.contains(.write) || global.contains(.write) {
                        throw invalid("Pointer argument aliases a global accessed by the callee", at: statement.location)
                    }
                    try record(root, access)
                }
                for (root, access) in callee.effects {
                    try budget.consume(.work)
                    if case .global = root { try record(root, access) }
                }
            default: break
            }
        }
        return result
    }

    private static func invalid(_ message: String, at location: ShaderLocation?) -> ShaderFailure {
        .diagnostics([.init(severity: .error, code: .invalidReference, message: message, location: location)])
    }
}
