import SAGAShaderIR

private struct DOTFlowTarget {
    let id: String
    var label: String? = nil
}

private struct DOTBlockFlow {
    let block: IRHandle<IRBlock>
    let next: DOTFlowTarget
    var breakTarget: String? = nil
    var continueTarget: String? = nil
}

extension DOTGraphWriter {
    mutating func writeControlFlow(_ function: IRFunction, functionIndex: Int) throws {
        let prefix = "f\(functionIndex)"
        let exit = "\(prefix)_exit"
        func blockID(_ handle: IRHandle<IRBlock>) -> String { "\(prefix)_b\(handle.index)" }
        func statementID(_ handle: IRHandle<IRStatement>) -> String { "\(prefix)_s\(handle.index)" }

        try node(exit, label: "exit", shape: "oval")
        try edge(prefix, blockID(function.body), label: "entry")
        var pending = [DOTBlockFlow(block: function.body, next: DOTFlowTarget(id: exit))]
        while let flow = pending.popLast() {
            try charge()
            let block = try function.blocks.value(for: flow.block)
            let id = blockID(flow.block)
            try node(id, label: "block \(flow.block.index)", shape: "box")
            if let first = block.statements.first {
                try edge(id, statementID(first))
            } else {
                try edge(id, flow.next.id, label: flow.next.label)
            }

            var children: [DOTBlockFlow] = []
            for (index, handle) in block.statements.enumerated() {
                try charge()
                let statement = try function.statements.value(for: handle).kind
                let id = statementID(handle)
                let next = index + 1 < block.statements.count
                    ? DOTFlowTarget(id: statementID(block.statements[index + 1])) : flow.next
                let (label, shape) = dotStatementLabel(statement)
                try node(id, label: label, shape: shape)
                if !cfgOnly { try writeDOTStatementData(statement, id: id, prefix: prefix) }

                switch statement {
                case .block(let child):
                    try edge(id, blockID(child))
                    children.append(.init(block: child, next: next,
                                          breakTarget: flow.breakTarget, continueTarget: flow.continueTarget))
                case .ifCondition(_, let accept, let reject):
                    try edge(id, blockID(accept), label: "true")
                    try edge(id, blockID(reject), label: "false")
                    children.append(.init(block: accept, next: next,
                                          breakTarget: flow.breakTarget, continueTarget: flow.continueTarget))
                    children.append(.init(block: reject, next: next,
                                          breakTarget: flow.breakTarget, continueTarget: flow.continueTarget))
                case .switchValue(_, let cases):
                    for (caseIndex, entry) in cases.enumerated() {
                        try charge()
                        let label = switch entry.value {
                        case .i32(let value): "case \(value)"
                        case .u32(let value): "case \(value)u"
                        case .default: "default"
                        }
                        try edge(id, blockID(entry.body), label: label)
                        let caseNext: DOTFlowTarget
                        if entry.fallThrough, caseIndex + 1 < cases.count {
                            caseNext = .init(id: blockID(cases[caseIndex + 1].body), label: "fallthrough")
                        } else {
                            caseNext = next
                        }
                        children.append(.init(block: entry.body, next: caseNext,
                                              breakTarget: next.id, continueTarget: flow.continueTarget))
                    }
                case .loop(let body, let continuing, let breakIf):
                    let bodyID = blockID(body)
                    let continuingID = blockID(continuing)
                    let afterContinuing: DOTFlowTarget
                    try edge(id, bodyID, label: "body")
                    if let breakIf {
                        let condition = "\(id)_break_if"
                        try node(condition, label: "break if", shape: "diamond")
                        try edge(condition, next.id, label: "true")
                        try edge(condition, bodyID, label: "false")
                        if !cfgOnly {
                            try edge(condition, "\(prefix)_e\(breakIf.index)", label: "condition", style: "dotted")
                        }
                        afterContinuing = .init(id: condition)
                    } else {
                        afterContinuing = .init(id: bodyID, label: "back")
                    }
                    children.append(.init(block: body, next: .init(id: continuingID, label: "continuing"),
                                          breakTarget: next.id, continueTarget: continuingID))
                    children.append(.init(block: continuing, next: afterContinuing,
                                          breakTarget: next.id, continueTarget: continuingID))
                case .break:
                    guard let target = flow.breakTarget else { throw ShaderFailure.invalidHandle }
                    try edge(id, target, label: "break")
                case .continue:
                    guard let target = flow.continueTarget else { throw ShaderFailure.invalidHandle }
                    try edge(id, target, label: "continue")
                case .returnValue:
                    try edge(id, exit, label: "return")
                case .kill:
                    try edge(id, exit, label: "kill")
                default:
                    try edge(id, next.id, label: next.label)
                }
            }
            pending.append(contentsOf: children.reversed())
        }
    }

    private mutating func writeDOTStatementData(
        _ statement: IRStatementKind, id: String, prefix: String
    ) throws {
        func operand(_ expression: IRHandle<IRExpression>?, _ label: String) throws {
            guard let expression else { return }
            try edge(id, "\(prefix)_e\(expression.index)", label: label, style: "dotted")
        }
        switch statement {
        case .emit(let expressions):
            for expression in expressions { try operand(expression, "emit") }
        case .ifCondition(let condition, _, _):
            try operand(condition, "condition")
        case .switchValue(let selector, _):
            try operand(selector, "selector")
        case .returnValue(let value):
            try operand(value, "value")
        case .store(let pointer, let value):
            try operand(pointer, "pointer")
            try operand(value, "value")
        case .imageStore(let image, let coordinate, let layer, let value):
            try operand(image, "image")
            try operand(coordinate, "coordinate")
            try operand(layer, "array index")
            try operand(value, "value")
        case .atomic(let pointer, let function, let value, let result):
            try operand(pointer, "pointer")
            try operand(value, "value")
            if case .exchange(let compare, _) = function { try operand(compare, "compare") }
            try operand(result, "result")
        case .imageAtomic(let image, let coordinate, let layer, let function, let value):
            try operand(image, "image")
            try operand(coordinate, "coordinate")
            try operand(layer, "array index")
            try operand(value, "value")
            if case .exchange(let compare, _) = function { try operand(compare, "compare") }
        case .workgroupUniformLoad(let pointer, let result):
            try operand(pointer, "pointer")
            try operand(result, "result")
        case .call(let function, let arguments, let result):
            try edge(id, "f\(function.index)", label: "call", style: "dashed")
            for (index, argument) in arguments.enumerated() { try operand(argument, "argument \(index)") }
            try operand(result, "result")
        case .rayQuery(let query, let function):
            try operand(query, "query")
            switch function {
            case .initialize(let accelerationStructure, let descriptor):
                try operand(accelerationStructure, "acceleration structure")
                try operand(descriptor, "descriptor")
            case .proceed(let result): try operand(result, "result")
            case .generateIntersection(let distance): try operand(distance, "hit distance")
            case .confirmIntersection, .terminate, .begin: break
            }
        case .traceRay(let accelerationStructure, let descriptor, let payload):
            try operand(accelerationStructure, "acceleration structure")
            try operand(descriptor, "descriptor")
            try operand(payload, "payload")
        case .subgroupBallot(let result, let predicate):
            try operand(predicate, "predicate")
            try operand(result, "result")
        case .subgroupGather(let mode, let argument, let result):
            try operand(argument, "argument")
            try operand(mode.index, "index")
            try operand(result, "result")
        case .subgroupCollectiveOperation(_, _, let argument, let result):
            try operand(argument, "argument")
            try operand(result, "result")
        case .cooperativeStore(let target, let data):
            try operand(target, "target")
            try operand(data.pointer, "pointer")
            try operand(data.stride, "stride")
        case .debugPrintf(_, let arguments):
            for (index, argument) in arguments.enumerated() { try operand(argument, "argument \(index)") }
        case .block, .loop, .break, .continue, .kill, .demoteToHelperInvocation, .controlBarrier, .memoryBarrier:
            break
        }
    }
}

private func dotStatementLabel(_ statement: IRStatementKind) -> (String, String) {
    switch statement {
    case .emit: ("emit", "box")
    case .block: ("block", "box")
    case .ifCondition: ("if", "diamond")
    case .switchValue: ("switch", "diamond")
    case .loop: ("loop", "box")
    case .break: ("break", "box")
    case .continue: ("continue", "box")
    case .returnValue: ("return", "box")
    case .kill: ("kill", "box")
    case .demoteToHelperInvocation: ("demote to helper invocation", "box")
    case .controlBarrier: ("control barrier", "box")
    case .memoryBarrier: ("memory barrier", "box")
    case .store: ("store", "box")
    case .imageStore: ("image store", "box")
    case .atomic: ("atomic", "box")
    case .imageAtomic: ("image atomic", "box")
    case .workgroupUniformLoad: ("workgroup uniform load", "box")
    case .call(let function, _, _): ("call f\(function.index)", "box")
    case .rayQuery: ("ray query", "box")
    case .traceRay: ("trace ray", "box")
    case .subgroupBallot: ("subgroup ballot", "box")
    case .subgroupGather: ("subgroup gather", "box")
    case .subgroupCollectiveOperation(let operation, let collective, _, _):
        ("subgroup \(collective.rawValue) \(operation.rawValue)", "box")
    case .cooperativeStore: ("cooperative store", "box")
    case .debugPrintf: ("debug printf", "box")
    }
}
