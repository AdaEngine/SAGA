internal import SAGAShaderIR

typealias SharedExpression = IRHandle<IRExpression>

extension IRExpressionKind {
    func remappingSharedPointerOperands(_ map: (SharedExpression) throws -> SharedExpression) throws -> Self {
        if let kind = try remappingPipelineOperands(map) { return kind }
        func optional(_ value: SharedExpression?) throws -> SharedExpression? { try value.map(map) }
        switch self {
        case .globalVariable, .localVariable, .atomicResult, .workgroupUniformLoadResult,
             .subgroupBallotResult, .subgroupOperationResult:
            return self
        case .relational(let operation, let argument):
            return try .relational(function: operation, argument: map(argument))
        case .math(let operation, let a, let b, let c, let d):
            return try .math(function: operation, argument: map(a), argument1: optional(b),
                             argument2: optional(c), argument3: optional(d))
        case .load(let pointer): return try .load(pointer: map(pointer))
        case .runtimeValue(let value): return try .runtimeValue(map(value))
        case .arrayLength(let pointer): return try .arrayLength(map(pointer))
        case .derivative(let axis, let control, let operand):
            return try .derivative(axis: axis, control: control, expression: map(operand))
        case .imageSample(let sample):
            let level: IRSampleLevel
            switch sample.level {
            case .auto, .zero: level = sample.level
            case .exact(let value): level = try .exact(map(value))
            case .bias(let value): level = try .bias(map(value))
            case .gradient(let x, let y): level = try .gradient(x: map(x), y: map(y))
            }
            return try .imageSample(.init(image: map(sample.image), sampler: map(sample.sampler),
                                          gather: sample.gather, coordinate: map(sample.coordinate),
                                          arrayIndex: optional(sample.arrayIndex), offset: optional(sample.offset),
                                          level: level, depthReference: optional(sample.depthReference),
                                          clampToEdge: sample.clampToEdge))
        case .imageLoad(let load):
            return try .imageLoad(.init(image: map(load.image), coordinate: map(load.coordinate),
                                        arrayIndex: optional(load.arrayIndex), sample: optional(load.sample),
                                        level: optional(load.level)))
        case .imageQuery(let image, let query):
            if case .size(let level) = query {
                return try .imageQuery(image: map(image), query: .size(level: optional(level)))
            }
            return try .imageQuery(image: map(image), query: query)
        default:
            throw ShaderFailure.unsupported("Expression remapping for shared pointer specialization")
        }
    }
}

extension IRStatementKind {
    func remappingSharedPointerOperands(_ map: (SharedExpression) throws -> SharedExpression) throws -> Self {
        func optional(_ value: SharedExpression?) throws -> SharedExpression? { try value.map(map) }
        switch self {
        case .block, .break, .continue, .kill, .demoteToHelperInvocation, .controlBarrier, .memoryBarrier:
            return self
        case .emit(let expressions): return try .emit(expressions.map(map))
        case .ifCondition(let condition, let accept, let reject):
            return try .ifCondition(condition: map(condition), accept: accept, reject: reject)
        case .switchValue(let selector, let cases):
            return try .switchValue(selector: map(selector), cases: cases)
        case .loop(let body, let continuing, let breakIf):
            return try .loop(body: body, continuing: continuing, breakIf: optional(breakIf))
        case .returnValue(let value): return try .returnValue(optional(value))
        case .store(let pointer, let value): return try .store(pointer: map(pointer), value: map(value))
        case .imageStore(let image, let coordinate, let arrayIndex, let value):
            return try .imageStore(image: map(image), coordinate: map(coordinate),
                                   arrayIndex: optional(arrayIndex), value: map(value))
        case .imageAtomic(let image, let coordinate, let arrayIndex, let function, let value):
            return try .imageAtomic(image: map(image), coordinate: map(coordinate), arrayIndex: optional(arrayIndex),
                                    function: function, value: map(value))
        case .subgroupGather(let mode, let argument, let result):
            return try .subgroupGather(mode: mode.remappingIndex(map), argument: map(argument), result: map(result))
        case .subgroupBallot(let result, let predicate):
            return try .subgroupBallot(result: map(result), predicate: optional(predicate))
        case .subgroupCollectiveOperation(let operation, let collective, let argument, let result):
            return try .subgroupCollectiveOperation(op: operation, collective: collective, argument: map(argument), result: map(result))
        case .atomic(let pointer, let operation, let value, let result):
            let function: IRAtomicFunction
            if case .exchange(let comparison, let strong) = operation {
                function = try .exchange(compare: optional(comparison), strong: strong)
            } else {
                function = operation
            }
            return try .atomic(pointer: map(pointer), function: function, value: map(value), result: optional(result))
        case .workgroupUniformLoad(let pointer, let result):
            return try .workgroupUniformLoad(pointer: map(pointer), result: map(result))
        case .debugPrintf(let format, let arguments):
            return try .debugPrintf(format: format, arguments: arguments.map(map))
        default:
            throw ShaderFailure.unsupported("Statement remapping for shared pointer specialization")
        }
    }
}
