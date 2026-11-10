import SAGAShaderIR

struct SPIRVBranchTargets {
    var breakTarget: UInt32? = nil
    var continueTarget: UInt32? = nil
}

private enum SPIRVBufferBounds {
    case outside
    case check(UInt32)
}

private struct SPIRVBufferProjection {
    let base: UInt32
    let index: UInt32
    let pointerType: UInt32
    let bounds: SPIRVBufferBounds
    let nonUniform: Bool
    var ownBounds: SPIRVBufferBounds? = nil
}

struct SPIRVFunctionState {
    let function: IRFunction
    let renderInterface: SPIRVRenderInterfacePlan?
    let nonUniformDescriptors: Set<IRHandle<IRExpression>>
    var arguments: [UInt32] = []
    var locals: [UInt32] = []
    var arrayValueSpills: [IRHandle<IRType>: UInt32] = [:]
    var storageImageAddresses: [UInt32: UInt32] = [:]
    private(set) var values: [IRHandle<IRExpression>: UInt32] = [:]
    private var expressionOrder: [IRHandle<IRExpression>] = []
    var explicitBranchTargets: Set<UInt32> = []
    /// Label that early returns of a mesh entry point branch to, leaving the single-iteration body loop.
    var meshExit: UInt32?
    var meshHeader: UInt32?
    var meshContinue: UInt32?
    /// Function variable that receives the dispatch size a task entry returns.
    var taskResult: UInt32?
    var layoutDescriptorCounts: [UInt32: UInt32] = [:]
    /// Auxiliary state variables of each ray query local, by local index.
    var rayQueries: [Int: SPIRVRayQueryLocal] = [:]
    /// External texture values by the id of their first loaded plane.
    var externalTextures: [UInt32: SPIRVExternalValue] = [:]
    var bufferArrays: [UInt32: SPIRVBufferArrayGlobal] = [:]
    var bufferElements: [UInt32: SPIRVBufferArrayGlobal] = [:]
    fileprivate var bufferProjections: [UInt32: SPIRVBufferProjection] = [:]

    init(function: IRFunction, renderInterface: SPIRVRenderInterfacePlan?, nonUniformDescriptors: Set<IRHandle<IRExpression>>) {
        self.function = function
        self.renderInterface = renderInterface
        self.nonUniformDescriptors = nonUniformDescriptors
    }

    var expressionScope: Int { expressionOrder.count }

    mutating func bind(_ handle: IRHandle<IRExpression>, to id: UInt32) {
        if values.updateValue(id, forKey: handle) == nil { expressionOrder.append(handle) }
    }

    mutating func restoreExpressions(to scope: Int) {
        for handle in expressionOrder[scope...] { values.removeValue(forKey: handle) }
        expressionOrder.removeLast(expressionOrder.count - scope)
    }
}

extension SPIRVBinaryWriter {
    mutating func writeFunction(
        _ function: IRFunction, id: UInt32, entryPlan: SPIRVEntryPlan?, nonUniformDescriptors: Set<IRHandle<IRExpression>>
    ) throws {
        var state = SPIRVFunctionState(function: function, renderInterface: entryPlan?.renderInterface,
                                       nonUniformDescriptors: nonUniformDescriptors)
        let result = if entryPlan == nil, let value = function.result { try typeID(value.type) } else { try voidTypeID() }
        var parameters: [UInt32] = []
        if entryPlan == nil {
            for argument in function.arguments {
                try charge()
                switch try module.types.value(for: argument.type).kind {
                case .pointer(_, .function), .pointer(_, .private): break
                case .pointer, .valuePointer: throw unsupported("requires function or private pointer parameters")
                default: break
                }
                parameters.append(try typeID(argument.type))
            }
        }
        let signature = try functionTypeID(result: result, parameters: parameters)
        try debugName(id, function.name)
        try emit(54, [result, id, 0, signature]) // OpFunction
        for (index, type) in parameters.enumerated() {
            let argument = try allocateID()
            try debugName(argument, function.arguments[index].name)
            try emit(55, [type, argument]) // OpFunctionParameter
            state.arguments.append(argument)
        }
        try emit(248, [allocateID()]) // OpLabel
        for (localIndex, local) in function.localVariables.values.enumerated() {
            let type = try typeID(local.type)
            let pointer = try pointerTypeID(storageClass: 7, pointee: type)
            let id = try allocateID()
            if case .rayQuery = try module.types.value(for: local.type).kind {
                try debugName(id, local.name)
                try emit(59, [pointer, id, 7]) // OpVariable Function, without an initializer
                state.locals.append(id)
                state.rayQueries[localIndex] = try declareRayQueryAuxiliaries()
                continue
            }
            let initializer = try local.initializer.map { try value($0, in: &state) } ?? zeroID(typeID: type)
            try debugName(id, local.name)
            try emit(59, [pointer, id, 7, initializer]) // OpVariable Function
            state.locals.append(id)
        }
        try declareArrayValueSpills(in: &state)
        var meshLoopVariable: UInt32 = 0
        if case .task(let plan)? = entryPlan {
            let pointer = try pointerTypeID(storageClass: 7, pointee: plan.vectorType)
            let result = try allocateID()
            try emit(59, [pointer, result, 7]) // OpVariable Function
            state.taskResult = result
        }
        if case .mesh? = entryPlan {
            // Variables must lead the first block, so the copy-loop counter is declared before any loads.
            let u32 = try scalarTypeID(.u32)
            let pointer = try pointerTypeID(storageClass: 7, pointee: u32)
            meshLoopVariable = try allocateID()
            try emit(59, [pointer, meshLoopVariable, 7]) // OpVariable Function
        }
        switch entryPlan {
        case .compute(let plan): state.arguments = try emitComputeInputs(plan)
        case .vertex(let plan), .fragment(let plan): state.arguments = try emitRenderInputs(plan)
        case .mesh(let plan): state.arguments = try emitComputeInputs(plan.compute)
        case .task(let plan): state.arguments = try emitComputeInputs(plan.compute)
        case nil: break
        }
        var meshPlan: SPIRVMeshPlan?
        var taskPlan: SPIRVTaskPlan?
        if case .task(let plan)? = entryPlan { taskPlan = plan }
        if case .mesh(let plan)? = entryPlan { meshPlan = plan }
        if meshPlan != nil || taskPlan != nil {
            // The body runs once inside a loop so that `return` can leave it for the epilogue.
            let header = try allocateID(), start = try allocateID(), continueLabel = try allocateID(), exit = try allocateID()
            try emit(249, [header])
            try emit(248, [header])
            try emit(246, [exit, continueLabel, 0]) // OpLoopMerge
            try emit(249, [start])
            try emit(248, [start])
            state.meshExit = exit
            state.meshHeader = header
            state.meshContinue = continueLabel
        }
        // Wrapper projections are implicit global references in IR. Give them a common
        // dominating block even when the first use occurs on a conditional loop path.
        for (index, expression) in function.expressions.values.enumerated() {
            try charge()
            if case .globalVariable = expression.kind { _ = try value(IRHandle(index: index), in: &state) }
        }
        if meshPlan != nil || taskPlan != nil, let exit = state.meshExit, let header = state.meshHeader,
           let continueLabel = state.meshContinue {
            if try !writeBlock(function.body, in: &state) { try emit(249, [exit]) }
            try emit(248, [continueLabel])
            try emit(250, [try booleanFalseID(), header, exit]) // Back-edge that is never taken.
            try emit(248, [exit])
            if let meshPlan {
                try writeMeshEpilogue(meshPlan, loopVariable: meshLoopVariable)
                try emit(253) // OpReturn
            } else if let taskPlan, let result = state.taskResult {
                try writeTaskEpilogue(taskPlan, resultVariable: result) // Ends the block with OpEmitMeshTasksEXT.
            }
        } else if try !writeBlock(function.body, in: &state) {
            guard function.result == nil else { throw unsupported("requires a return value for this function") }
            try emit(253) // OpReturn
        }
        try emit(56) // OpFunctionEnd
    }

    mutating func writeBlock(
        _ block: IRHandle<IRBlock>, in state: inout SPIRVFunctionState, targets: SPIRVBranchTargets = .init()
    ) throws -> Bool {
        try budget.enterDepth()
        defer { try? budget.leaveDepth() }
        for handle in try state.function.blocks.value(for: block).statements {
            try charge()
            switch try state.function.statements.value(for: handle).kind {
            case .emit(let expressions):
                for expression in expressions { _ = try value(expression, in: &state, emitted: true) }
            case .store(let pointer, let source):
                let address = try value(pointer, in: &state)
                var data = try value(source, in: &state)
                if let buffer = state.bufferElements[address] {
                    let sourceType = try state.function.expressions.value(for: source).type
                    data = try repackBufferRecord(data, from: typeID(sourceType), to: buffer.blockType, element: buffer.element)
                }
                if let type = try workgroupAggregatePointee(pointer, in: state) {
                    data = try convertWorkgroupValue(data, type: type, toWorkgroup: true)
                }
                if let atomic = try atomicPointee(pointer, in: state) {
                    try writeAtomicStore(address: address, data: data, scope: atomic.scope, in: state)
                } else {
                    try guardedBufferEffect(address: address, in: state) { writer, pointer in
                        try writer.emit(62, [pointer, data]) // OpStore
                    }
                }
            case .cooperativeStore(let target, let data):
                let object = try value(target, in: &state)
                let resultType = try state.function.expressions.value(for: target).type
                guard case .cooperativeMatrix(let columns, let rows, _, _) = try module.types.value(for: resultType).kind else {
                    throw unsupported("requires a cooperative matrix store value")
                }
                let address = try value(data.pointer, in: &state)
                let stride = try value(data.stride, in: &state)
                let (positiveStride, safeStride) = try cooperativeStoreStride(stride, handle: data.stride, in: state)
                let layout = try unsignedConstantID(data.rowMajor ? 0 : 1)
                let fullBounds = try cooperativeFootprintCondition(pointer: data.pointer, address: address,
                    strideHandle: data.stride, stride: stride, majorCount: data.rowMajor ? UInt32(rows.rawValue) : UInt32(columns.rawValue),
                    minorCount: data.rowMajor ? UInt32(columns.rawValue) : UInt32(rows.rawValue), in: &state)
                let storeCondition: UInt32
                if let fullBounds {
                    storeCondition = try rawValueInstruction(167, type: scalarTypeID(.bool),
                                                              operands: [fullBounds, positiveStride])
                } else { storeCondition = positiveStride }
                try effectWhen(storeCondition) { writer in
                        let pointer = try writer.materializeBufferPointer(address, in: state)
                        try writer.requireCooperativeMatrix()
                        try writer.emit(4458, [pointer, object, layout, safeStride])
                }
            case .atomic(let pointer, let operation, let operand, let result):
                try writeAtomic(pointer: pointer, operation: operation, operand: operand, result: result, in: &state)
            case .controlBarrier(let flags):
                try writeControlBarrier(flags)
            case .debugPrintf(let format, let arguments):
                let operands = try arguments.map { try value($0, in: &state) }
                try writeDebugPrintf(format: format, arguments: operands)
            case .subgroupBallot(let result, let predicate):
                try writeSubgroupBallot(predicate: predicate, result: result, in: &state)
            case .subgroupGather(let mode, let argument, let result):
                try writeSubgroupGather(mode, argument: argument, result: result, in: &state)
            case .subgroupCollectiveOperation(let operation, let collective, let argument, let result):
                try writeSubgroupCollective(operation, collective, argument: argument, result: result, in: &state)
            case .workgroupUniformLoad(let pointer, let result):
                let address = try value(pointer, in: &state)
                let safeAddress = try uniformWorkgroupAddress(address, in: state)
                let type = try typeID(state.function.expressions.value(for: result).type)
                let aggregate = try workgroupAggregatePointee(pointer, in: state)
                try writeControlBarrier(.workgroup)
                let loaded: UInt32
                if let atomic = try atomicPointee(pointer, in: state) {
                    loaded = try readAtomic(address: safeAddress, type: type, scope: atomic.scope, in: state)
                } else {
                    let pointee = try aggregate.map { try workgroupTypeID($0) } ?? type
                    loaded = try rawValueInstruction(61, type: pointee, operands: [safeAddress]) // OpLoad
                }
                try writeControlBarrier(.workgroup)
                let converted = try aggregate.map { try convertWorkgroupValue(loaded, type: $0, toWorkgroup: false) } ?? loaded
                let value: UInt32
                switch state.bufferProjections[address]?.bounds {
                case .check(let valid)?: value = try valueWhen(valid, type: type) { _ in converted }
                case .outside?: value = try zeroID(typeID: type)
                case nil: value = converted
                }
                state.bind(result, to: value)
            case .imageStore(let image, let coordinate, let layer, let source):
                try writeImageStore(image: image, coordinate: coordinate, layer: layer, source: source, in: &state)
            case .imageAtomic(let image, let coordinate, let layer, let operation, let source):
                try writeImageAtomic(image: image, coordinate: coordinate, layer: layer, operation: operation,
                                     source: source, in: &state)
            case .block(let child):
                let scope = state.expressionScope
                let terminated = try writeBlock(child, in: &state, targets: targets)
                try restoreExpressions(to: scope, in: &state)
                if terminated { return true }
            case .ifCondition(let condition, let accept, let reject):
                if try writeSelection(condition: condition, accept: accept, reject: reject, in: &state, targets: targets) {
                    return true
                }
            case .switchValue(let selector, let cases):
                if try writeSwitch(selector: selector, cases: cases, in: &state, targets: targets) { return true }
            case .loop(let body, let continuing, let breakIf):
                if try writeLoop(body: body, continuing: continuing, breakIf: breakIf, in: &state, targets: targets) { return true }
            case .break:
                guard let target = targets.breakTarget else { throw ShaderFailure.invalidHandle }
                state.explicitBranchTargets.insert(target)
                try emit(249, [target]) // OpBranch
                return true
            case .continue:
                guard let target = targets.continueTarget else { throw ShaderFailure.invalidHandle }
                state.explicitBranchTargets.insert(target)
                try emit(249, [target])
                return true
            case .call(let function, let arguments, let expression):
                let callee = try module.functions.value(for: function)
                guard arguments.count == callee.arguments.count else { throw ShaderFailure.invalidHandle }
                let result = try callee.result.map { try typeID($0.type) } ?? voidTypeID()
                let id = try allocateID()
                var operands = [result, id, functionIDs[function.index]]
                for (argument, parameter) in zip(arguments, callee.arguments) {
                    if case .pointer = try module.types.value(for: parameter.type).kind {
                        // Logical addressing permits memory object declarations at calls.
                        // An access chain into a Function or Private object is a derived pointer.
                        switch try state.function.expressions.value(for: argument).kind {
                        case .localVariable, .globalVariable, .functionArgument: break
                        default: throw unsupported("requires a whole variable or parameter for pointer call operands")
                        }
                    }
                    operands.append(try value(argument, in: &state))
                }
                try emit(57, operands) // OpFunctionCall
                if let expression { state.bind(expression, to: id) }
            case .returnValue(let expression):
                if let expression {
                    let result = try value(expression, in: &state)
                    if let variable = state.taskResult, let exit = state.meshExit {
                        guard targets.breakTarget == nil else { throw unsupported("does not lower returns inside task loops") }
                        try emit(62, [variable, result]) // OpStore
                        state.explicitBranchTargets.insert(exit)
                        try emit(249, [exit])
                    } else if let plan = state.renderInterface {
                        try emitRenderOutput(result, plan: plan)
                        try emit(253) // Entry point returns through Output variables.
                    } else {
                        try emit(254, [result])
                    }
                } else if let exit = state.meshExit {
                    guard targets.breakTarget == nil else { throw unsupported("does not lower returns inside mesh loops") }
                    state.explicitBranchTargets.insert(exit)
                    try emit(249, [exit])
                } else {
                    try emit(253)
                }
                return true
            case .rayQuery(let query, let function):
                try writeRayQueryStatement(query: query, function: function, in: &state)
            case .kill:
                try emit(252) // OpKill
                return true
            case .demoteToHelperInvocation:
                try requireHelperDemotion()
                try emit(5380) // OpDemoteToHelperInvocation
            default: throw unsupported("does not lower this statement")
            }
        }
        return false
    }

    private mutating func writeSelection(
        condition: IRHandle<IRExpression>, accept: IRHandle<IRBlock>, reject: IRHandle<IRBlock>,
        in state: inout SPIRVFunctionState, targets: SPIRVBranchTargets
    ) throws -> Bool {
        let condition = try value(condition, in: &state)
        let yes = try allocateID()
        let no = try allocateID()
        let merge = try allocateID()
        try emit(247, [merge, 0]) // OpSelectionMerge
        try emit(250, [condition, yes, no]) // OpBranchConditional

        let scope = state.expressionScope
        try emit(248, [yes])
        let acceptTerminates = try writeBlock(accept, in: &state, targets: targets)
        if !acceptTerminates { try emit(249, [merge]) }
        try restoreExpressions(to: scope, in: &state)

        try emit(248, [no])
        let rejectTerminates = try writeBlock(reject, in: &state, targets: targets)
        if !rejectTerminates { try emit(249, [merge]) }
        try restoreExpressions(to: scope, in: &state)

        try emit(248, [merge])
        let terminated = acceptTerminates && rejectTerminates
        if terminated { try emit(255) } // OpUnreachable
        return terminated
    }

    mutating func restoreExpressions(to scope: Int, in state: inout SPIRVFunctionState) throws {
        try charge(state.expressionScope - scope)
        state.restoreExpressions(to: scope)
    }

    mutating func value(
        _ handle: IRHandle<IRExpression>, in state: inout SPIRVFunctionState, emitted: Bool = false
    ) throws -> UInt32 {
        try charge()
        if let id = state.values[handle] { return id }
        let expression = try state.function.expressions.value(for: handle)
        let id: UInt32
        switch expression.kind {
        case .literal(let literal): id = try literalID(literal)
        case .constant(let constant): id = try constantID(constant)
        case .zeroValue(let type): id = try zeroID(typeID: typeID(type))
        case .functionArgument(let index):
            guard state.arguments.indices.contains(Int(index)) else { throw ShaderFailure.invalidHandle }
            id = state.arguments[Int(index)]
        case .localVariable(let local):
            guard state.locals.indices.contains(local.index) else { throw ShaderFailure.invalidHandle }
            id = state.locals[local.index]
        case .globalVariable(let global):
            let declaration = try module.globalVariables.value(for: global)
            if declaration.space == .handle {
                let kind = try module.types.value(for: expression.type).kind
                if kind.isExternalTexture {
                    id = try loadExternalTexture(global, in: &state)
                    state.bind(handle, to: id)
                    return id
                }
                let binding = try resourceGlobal(global)
                if case .bindingArray = kind {
                    id = binding.id
                    if let count = binding.layoutResolvedCount { state.layoutDescriptorCounts[id] = count }
                } else {
                    id = try allocateID()
                    try emit(61, [binding.valueType, id, binding.id]) // OpLoad
                    if kind.storageTexture != nil { state.storageImageAddresses[id] = binding.id }
                }
            } else if case .bindingArray = try module.types.value(for: declaration.type).kind {
                let binding = try bufferArrayGlobal(global)
                id = binding.id
                state.bufferArrays[id] = binding
            } else if declaration.space == .workgroup || declaration.space == .taskPayload {
                id = try workgroupGlobal(global)
            } else if declaration.space == .private {
                id = try privateGlobal(global)
            } else {
                let binding = try bufferGlobal(global)
                if let member = binding.wrapperMember {
                    let index = try unsignedConstantID(member)
                    id = try valueInstruction(65, type: expression.type, operands: [binding.id, index])
                } else {
                    id = binding.id
                }
            }
        default:
            guard emitted else { throw unsupported("found a value used before its emission") }
            id = try writeExpression(expression, handle: handle, in: &state)
        }
        state.bind(handle, to: id)
        return id
    }

    mutating func valueInstruction(_ opcode: UInt32, type: IRHandle<IRType>, operands: [UInt32]) throws -> UInt32 {
        try rawValueInstruction(opcode, type: typeID(type), operands: operands)
    }

    mutating func rawValueInstruction(_ opcode: UInt32, type: UInt32, operands: [UInt32]) throws -> UInt32 {
        let id = try allocateID()
        try emit(opcode, [type, id] + operands)
        return id
    }

    private mutating func writeExpression(
        _ expression: IRExpression, handle: IRHandle<IRExpression>, in state: inout SPIRVFunctionState
    ) throws -> UInt32 {
        let type = expression.type
        switch expression.kind {
        case .cooperativeLoad(let columns, let rows, _, let data):
            let address = try value(data.pointer, in: &state)
            let stride = try value(data.stride, in: &state)
            let layout = try unsignedConstantID(data.rowMajor ? 0 : 1)
            let resultType = try typeID(type)
            let fullBounds = try cooperativeFootprintCondition(pointer: data.pointer, address: address,
                strideHandle: data.stride, stride: stride,
                majorCount: data.rowMajor ? UInt32(rows.rawValue) : UInt32(columns.rawValue),
                minorCount: data.rowMajor ? UInt32(columns.rawValue) : UInt32(rows.rawValue), in: &state)
            if let fullBounds {
                return try valueWhen(fullBounds, type: resultType) { writer in
                    let pointer = try writer.materializeBufferPointer(address, in: state)
                    try writer.requireCooperativeMatrix()
                    let memory = try writer.cooperativeMemoryOperand(pointer: data.pointer, layout: layout, stride: stride, in: state)
                    return try writer.rawValueInstruction(4457, type: resultType,
                        operands: [pointer] + memory) // OpCooperativeMatrixLoadKHR
                }
            }
            return try guardedBufferRead(address: address, type: resultType, in: state) { writer, pointer in
                try writer.requireCooperativeMatrix()
                let memory = try writer.cooperativeMemoryOperand(pointer: data.pointer, layout: layout, stride: stride, in: state)
                return try writer.rawValueInstruction(4457, type: resultType, operands: [pointer] + memory)
            }
        case .cooperativeMultiplyAdd(let a, let b, let c):
            let aID = try value(a, in: &state), bID = try value(b, in: &state), cID = try value(c, in: &state)
            try requireCooperativeMatrix()
            return try rawValueInstruction(4459, type: typeID(type), operands: [aID, bID, cID])
        case .imageSample(let sample):
            return try writeImageSample(sample, type: type, in: &state)
        case .imageLoad(let load):
            return try writeImageLoad(load, type: type, in: &state)
        case .imageQuery(let image, let query):
            return try writeImageQuery(image: image, query: query, type: type, in: &state)
        case .runtimeValue(let operand):
            return try value(operand, in: &state)
        case .load(let pointer):
            let address = try value(pointer, in: &state)
            let resultType = try typeID(type)
            if let atomic = try atomicPointee(pointer, in: state) {
                return try readAtomic(address: address, type: resultType, scope: atomic.scope, in: state)
            }
            let aggregate = try workgroupAggregatePointee(pointer, in: state)
            let buffer = state.bufferElements[address]
            return try guardedBufferRead(address: address, type: resultType, in: state) { writer, pointer in
                if let aggregate {
                    let raw = try writer.rawValueInstruction(61, type: writer.workgroupTypeID(aggregate), operands: [pointer])
                    return try writer.convertWorkgroupValue(raw, type: aggregate, toWorkgroup: false)
                }
                guard let buffer else {
                    return try writer.rawValueInstruction(61, type: resultType, operands: [pointer])
                }
                let record = try writer.rawValueInstruction(61, type: buffer.blockType, operands: [pointer])
                return try writer.repackBufferRecord(record, from: buffer.blockType, to: resultType, element: buffer.element)
            }
        case .rayQueryGetIntersection(let query, let committed):
            return try writeRayQueryIntersection(query: query, committed: committed, type: type, in: &state)
        case .rayQueryVertexPositions(let query, let committed):
            return try writeRayQueryVertexPositions(query: query, committed: committed, type: type, in: &state)
        case .arrayLength(let pointer):
            return try writeRuntimeArrayLength(pointer, resultType: typeID(type), in: &state)
        case .accessIndex(let base, let index):
            let baseID = try value(base, in: &state)
            if let buffer = state.bufferArrays[baseID] {
                return try bufferArrayElement(buffer, index: unsignedConstantID(index),
                                              signed: false, varying: false, in: &state)
            }
            if case .bindingArray = try valueKind(base, in: state) {
                let bounded = state.layoutDescriptorCounts[baseID].map { min(index, $0 - 1) } ?? index
                return try descriptorArrayAccess(baseID, index: unsignedConstantID(bounded), type: type, in: &state)
            }
            if try isPointer(type) {
                let position = try unsignedConstantID(index)
                let bounds = try pointerAccessBounds(base, index: position, in: &state)
                return try dataAccessChain(type: type, base: baseID, index: position, bounds: bounds, in: &state)
            }
            return try valueInstruction(81, type: type, operands: [baseID, index]) // OpCompositeExtract
        case .access(let base, let index):
            let baseID = try value(base, in: &state)
            let rawIndex = try value(index, in: &state)
            let indexKind = try valueKind(index, in: state)
            let signed = indexKind == .scalar(.i32) || indexKind == .scalar(.i16)
            let indexID: UInt32 = switch indexKind {
            case .scalar(.i16): try rawValueInstruction(114, type: scalarTypeID(.i32), operands: [rawIndex])
            case .scalar(.u16): try rawValueInstruction(113, type: scalarTypeID(.u32), operands: [rawIndex])
            default: rawIndex
            }
            let operands = [baseID, indexID]
            if let buffer = state.bufferArrays[operands[0]] {
                return try bufferArrayElement(buffer, index: operands[1], signed: signed,
                                              varying: state.nonUniformDescriptors.contains(handle), in: &state)
            }
            if case .bindingArray = try valueKind(base, in: state) {
                let offset: UInt32
                if let count = state.layoutDescriptorCounts[operands[0]] {
                    offset = try boundedLayoutIndex(operands[1], signed: signed, count: count)
                } else {
                    offset = operands[1]
                }
                let constant = constantIDs.contains(offset)
                let varying = !constant && state.nonUniformDescriptors.contains(handle)
                if !constant {
                    let storageImage = try module.types.value(for: type).kind.storageTexture != nil
                    if varying { try requireNonUniformDescriptors(storageImage ? 5309 : 5307) }
                    else { try requireCapability(storageImage ? 31 : 29) } // StorageImage / SampledImageArrayDynamicIndexing
                }
                return try descriptorArrayAccess(operands[0], index: offset, type: type, nonUniform: varying, in: &state)
            }
            if try isPointer(type) {
                let position = try unsignedAccessIndex(operands[1], signed: signed)
                let bounds = try pointerAccessBounds(base, index: position, in: &state)
                return try dataAccessChain(type: type, base: operands[0], index: position, bounds: bounds, in: &state)
            }
            switch try valueKind(base, in: state) {
            case .vector: return try valueInstruction(77, type: type, operands: operands) // OpVectorExtractDynamic
            case .matrix(let columns, let rows, _):
                return try writeMatrixColumn(operands[0], index: operands[1], signed: signed,
                                             columns: columns, rows: rows, type: type)
            case .array(_, .constant(let count), _):
                let baseType = try state.function.expressions.value(for: base).type
                guard let spill = state.arrayValueSpills[baseType] else { throw ShaderFailure.invalidHandle }
                return try writeArrayValueAccess(operands[0], index: operands[1], signed: signed,
                                                 count: count, spill: spill, type: type)
            default: throw unsupported("does not lower dynamic indexing for this value type")
            }
        case .compose(_, let components):
            if components.count == 1, case .vector = try module.types.value(for: type).kind {
                let component = components[0]
                let componentType = try state.function.expressions.value(for: component).type
                if try typeID(componentType) == typeID(type) {
                    // A whole vector already has the result type. OpCompositeConstruct
                    // requires at least two constituents for a vector result.
                    return try value(component, in: &state)
                }
            }
            var operands: [UInt32] = []
            for component in components { operands.append(try value(component, in: &state)) }
            return try valueInstruction(80, type: type, operands: operands) // OpCompositeConstruct
        case .splat(let size, let operand):
            let id = try value(operand, in: &state)
            return try valueInstruction(80, type: type, operands: Array(repeating: id, count: Int(size.rawValue)))
        case .swizzle(let size, let vector, let pattern):
            let id = try value(vector, in: &state)
            let components = [pattern.x, pattern.y, pattern.z, pattern.w].prefix(Int(size.rawValue))
            let indices = components.map { component -> UInt32 in
                switch component { case .x: 0; case .y: 1; case .z: 2; case .w: 3 }
            }
            return try valueInstruction(79, type: type, operands: [id, id] + indices) // OpVectorShuffle
        case .bitcast(let operand, _):
            let id = try value(operand, in: &state)
            if try valueKind(operand, in: state) == module.types.value(for: type).kind { return id }
            return try valueInstruction(124, type: type, operands: [id])
        case .cast(let operand, let target):
            let id = try value(operand, in: &state)
            if case .matrix(let columns, let rows, let scalar) = try valueKind(operand, in: state) {
                if scalar == target { return id }
                return try writeMatrixFloatCast(id, columns: columns, rows: rows, source: scalar, target: target, type: type)
            }
            let source = try scalarKind(operand, in: state)
            if source == target { return id }
            if source == .bool, let shape = IRBitcastShape(try module.types.value(for: type).kind) {
                let unit: IRLiteral = switch target {
                case .i16: .i16(1)
                case .u16: .u16(1)
                case .i32: .i32(1)
                case .u32: .u32(1)
                case .i64: .i64(1)
                case .u64: .u64(1)
                case .f16: .f16Bits(0x3c00)
                case .f32: .f32(1)
                case .f64: .f64(1)
                default: throw unsupported("does not lower this boolean conversion")
                }
                let result = try typeID(type)
                let one = try splatLiteralID(unit, type: result, size: shape.size)
                return try rawValueInstruction(169, type: result, operands: [id, one, zeroID(typeID: result)]) // OpSelect
            }
            if target == .bool {
                let operandType = try state.function.expressions.value(for: operand).type
                let zero = try zeroID(typeID: typeID(operandType))
                let opcode: UInt32 = source.kind == .float ? 183 : 171 // OpFUnordNotEqual or OpINotEqual
                return try valueInstruction(opcode, type: type, operands: [id, zero])
            }
            if source == .f64, target.kind == .sint || target.kind == .uint,
               let shape = IRBitcastShape(try valueKind(operand, in: state)) {
                return try writeSaturatingFloat64Conversion(id, source: shape, target: target, type: type)
            }
            if [.f16, .f32].contains(source), [.i32, .u32, .i64, .u64].contains(target),
               let shape = IRBitcastShape(try valueKind(operand, in: state)) {
                return try writeSaturatingFloatConversion(id, source: shape, target: target, type: type)
            }
            if [.sint, .uint].contains(source.kind), [.sint, .uint].contains(target.kind) {
                if source.width == target.width {
                    return try valueInstruction(124, type: type, operands: [id]) // OpBitcast
                }
                let signed = source.width < target.width ? source.kind == .sint : target.kind == .sint
                if signed { return try valueInstruction(114, type: type, operands: [id]) } // OpSConvert
                if target.kind == .uint { return try valueInstruction(113, type: type, operands: [id]) } // OpUConvert

                // OpUConvert requires an unsigned result even when WGSL widens to a signed type.
                guard let shape = IRBitcastShape(try valueKind(operand, in: state)) else { throw ShaderFailure.invalidHandle }
                let unsigned = try scalarOrVectorTypeID(.init(kind: .uint, width: target.width), size: shape.size)
                let widened = try rawValueInstruction(113, type: unsigned, operands: [id])
                return try valueInstruction(124, type: type, operands: [widened])
            }
            let opcode: UInt32 = switch (source, target) {
            case (.i16, .f32), (.i32, .f32), (.i32, .f16), (.i64, .f32): 111
            case (.i16, .f64), (.i32, .f64), (.i64, .f64): 111
            case (.u16, .f32), (.u32, .f32), (.u32, .f16), (.u64, .f32): 112
            case (.u16, .f64), (.u32, .f64), (.u64, .f64): 112
            case (.f16, .f32), (.f32, .f16), (.f64, .f32), (.f32, .f64), (.f64, .f16), (.f16, .f64): 115 // OpFConvert
            default: throw unsupported("does not lower this numeric conversion")
            }
            return try valueInstruction(opcode, type: type, operands: [id])
        case .unary(let operation, let operand):
            let scalar = try scalarKind(operand, in: state)
            let opcode: UInt32 = switch operation {
            case .negate: scalar.kind == .float ? 127 : 126
            case .logicalNot: 168
            case .bitwiseNot: 200
            }
            return try valueInstruction(opcode, type: type, operands: [value(operand, in: &state)])
        case .binary(let operation, let left, let right) where operation == .shiftLeft || operation == .shiftRight:
            guard let shape = IRBitcastShape(try valueKind(left, in: state)),
                  shape.scalar.kind == .sint || shape.scalar.kind == .uint else {
                throw unsupported("requires an integer shift")
            }
            let base = try value(left, in: &state)
            let count = try value(right, in: &state)
            return try writeIntegerShift(operation, base: base, count: count, shape: shape, type: type)
        case .binary(let operation, let left, let right):
            let kind = try valueKind(left, in: state)
            let rightKind = try valueKind(right, in: state)
            if kind.isSPIRVMatrix || rightKind.isSPIRVMatrix {
                let lhs = try value(left, in: &state)
                let rhs = try value(right, in: &state)
                return try writeMatrixBinary(operation, lhs: (lhs, kind), rhs: (rhs, rightKind), type: type)
            }
            guard kind == rightKind else {
                throw unsupported("does not lower mixed-shape arithmetic")
            }
            if operation == .divide || operation == .modulo, let shape = IRBitcastShape(kind),
               shape.scalar.kind == .sint || shape.scalar.kind == .uint {
                let lhs = try value(left, in: &state)
                let rhs = try value(right, in: &state)
                return try writeIntegerQuotient(lhs, rhs, shape: shape, remainder: operation == .modulo)
            }
            let opcode = try binaryOpcode(operation, scalar: scalarKind(left, in: state))
            return try valueInstruction(opcode, type: type, operands: [value(left, in: &state), value(right, in: &state)])
        case .math(.transpose, let operand, nil, nil, nil):
            return try valueInstruction(84, type: type, operands: [value(operand, in: &state)]) // OpTranspose
        case .math(.dot, let left, let right?, nil, nil):
            return try writeDot(left, right, type: type, in: &state)
        case .math(let builtin, let left, let right?, nil, nil) where builtin == .dot4I8Packed || builtin == .dot4U8Packed:
            let lhs = try value(left, in: &state)
            let rhs = try value(right, in: &state)
            return try writePackedDot(lhs, rhs, signed: builtin == .dot4I8Packed, type: typeID(type))
        case .math(let builtin, let operand, nil, nil, nil) where builtin.packingBuiltin != nil:
            guard let packing = builtin.packingBuiltin else { throw ShaderFailure.invalidHandle }
            let id = try value(operand, in: &state)
            return try writePacking(packing, operand: id, type: type)
        case .math(.ldexp, let operand, let exponent?, nil, nil):
            let operands = try [value(operand, in: &state), value(exponent, in: &state)]
            return try glslMathInstruction(53, type: typeID(type), operands: operands)
        case .math(let builtin, let operand, nil, nil, nil) where builtin.decomposesFloat:
            let id = try value(operand, in: &state)
            return try writeDecomposition(builtin, operand: id, type: type)
        case .math(let builtin, let operand, let second, let third, nil) where builtin.numericBuiltin != nil:
            return try writeNumericBuiltin(builtin, operands: [operand] + [second, third].compactMap { $0 },
                                           type: type, in: &state)
        case .math(let builtin, let operand, nil, nil, nil) where builtin == .determinant || builtin == .inverse:
            let instruction: UInt32 = builtin == .determinant ? 33 : 34
            return try valueInstruction(12, type: type,
                                        operands: [glslInstructionSetID(), instruction, value(operand, in: &state)])
        case .math(let builtin, let operand, let second, let third, nil) where builtin.geometricArity != nil:
            return try writeGeometry(builtin, operands: [operand] + [second, third].compactMap { $0 },
                                      type: type, in: &state)
        case .math(let builtin, let operand, let second, let third, let fourth) where builtin.integerBitArity != nil:
            guard let shape = IRBitcastShape(try valueKind(operand, in: state)),
                  shape.scalar.kind == .sint || shape.scalar.kind == .uint else {
                throw unsupported("requires an integer bit operation")
            }
            var arguments = [try value(operand, in: &state)]
            for argument in [second, third, fourth].compactMap({ $0 }) {
                arguments.append(try value(argument, in: &state))
            }
            return try writeIntegerBitBuiltin(builtin, arguments: arguments, shape: shape, type: type)
        case .derivative(let axis, let control, let operand):
            let base: UInt32 = switch axis {
            case .x: 207 // OpDPdx
            case .y: 208 // OpDPdy
            case .width: 209 // OpFwidth
            }
            let offset: UInt32 = switch control {
            case .none: 0
            case .fine: 3
            case .coarse: 6
            }
            if offset != 0 { try requireCapability(51) } // DerivativeControl
            return try valueInstruction(base + offset, type: type, operands: [value(operand, in: &state)])
        case .relational(let reduction, let operand) where reduction.isBooleanReduction:
            let id = try value(operand, in: &state)
            if try valueKind(operand, in: state) == .scalar(.bool) { return id }
            return try valueInstruction(reduction == .any ? 154 : 155, type: type, operands: [id])
        case .select(let condition, let accept, let reject):
            var test = try value(condition, in: &state)
            switch try module.types.value(for: type).kind {
            case .scalar: break
            case .vector(let size, _):
                if try valueKind(condition, in: state) == .scalar(.bool) {
                    let vector = try vectorTypeID(size: size, scalar: .bool)
                    test = try rawValueInstruction(80, type: vector, operands: Array(repeating: test, count: Int(size.rawValue)))
                }
            default:
                throw unsupported("does not lower composite selects")
            }
            return try valueInstruction(169, type: type, operands: [test, value(accept, in: &state), value(reject, in: &state)])
        default: throw unsupported("does not lower this expression")
        }
    }

    private mutating func descriptorArrayAccess(
        _ base: UInt32, index: UInt32, type: IRHandle<IRType>, nonUniform: Bool = false,
        in state: inout SPIRVFunctionState
    ) throws -> UInt32 {
        let element = try typeID(type)
        let pointer = try pointerTypeID(storageClass: 0, pointee: element)
        let address = try allocateID()
        try emit(65, [pointer, address, base, index]) // OpAccessChain
        let resource = try allocateID()
        try emit(61, [element, resource, address]) // OpLoad
        if try module.types.value(for: type).kind.storageTexture != nil {
            state.storageImageAddresses[resource] = address
        }
        if nonUniform {
            try markNonUniformResource(address)
            try markNonUniformResource(resource)
        }
        return resource
    }

    private mutating func bufferArrayElement(
        _ buffer: SPIRVBufferArrayGlobal, index: UInt32, signed: Bool, varying: Bool, in state: inout SPIRVFunctionState
    ) throws -> UInt32 {
        let dynamic = !constantIDs.contains(index)
        let storage = buffer.storageClass == 12
        let nonUniform = dynamic && varying
        if dynamic {
            if nonUniform { try requireNonUniformDescriptors(storage ? 5308 : 5306) }
            else { try requireCapability(storage ? 30 : 28) } // StorageBuffer / UniformBufferArrayDynamicIndexing
        }
        let pointer = try pointerTypeID(storageClass: buffer.storageClass, pointee: buffer.blockType)
        let bounds = try buffer.layoutResolvedCount.flatMap { try indexBounds(index, signed: signed, count: $0) }
        let element: UInt32
        if let bounds {
            element = try reserveBufferProjection(.init(base: buffer.id, index: index, pointerType: pointer,
                                                        bounds: bounds, nonUniform: nonUniform), in: &state)
        } else {
            element = try rawValueInstruction(65, type: pointer, operands: [buffer.id, index])
            if nonUniform { try markNonUniformResource(element) }
        }
        state.bufferElements[element] = buffer
        return element
    }

    private mutating func indexBounds(_ index: UInt32, signed: Bool, count: UInt32) throws -> SPIRVBufferBounds? {
        if let bits = constantWords[index] { return bits < count ? nil : .outside }
        let unsigned = try unsignedAccessIndex(index, signed: signed)
        let limit = try unsignedConstantID(count)
        return try .check(rawValueInstruction(176, type: scalarTypeID(.bool), operands: [unsigned, limit]))
    }

    private mutating func cooperativeFootprintCondition(
        pointer: IRHandle<IRExpression>, address: UInt32, strideHandle: IRHandle<IRExpression>, stride: UInt32,
        majorCount: UInt32, minorCount: UInt32, in state: inout SPIRVFunctionState
    ) throws -> UInt32? {
        let base: IRHandle<IRExpression>
        let indexExpression: IRHandle<IRExpression>?
        let fixedIndex: UInt32?
        switch try state.function.expressions.value(for: pointer).kind {
        case .access(let array, let index):
            base = array
            indexExpression = index
            fixedIndex = nil
        case .accessIndex(let array, let index):
            base = array
            indexExpression = nil
            fixedIndex = index
        default:
            throw unsupported("cooperative matrix footprint requires an indexed array")
        }
        guard case .pointer(let arrayType, _) = try module.types.value(for: state.function.expressions.value(for: base).type).kind,
              case .array(_, let arraySize, _) = try module.types.value(for: arrayType).kind else {
            throw unsupported("cooperative matrix footprint requires an indexed array")
        }
        let length: UInt32
        switch arraySize {
        case .constant(let count): length = try unsignedConstantID(count)
        case .dynamic: length = try writeRuntimeArrayLength(base, resultType: scalarTypeID(.u32), in: &state)
        case .override: throw unsupported("cooperative matrix footprint with unspecialized array size")
        }

        let zero = try unsignedConstantID(0), one = try unsignedConstantID(1)
        let bool = try scalarTypeID(.bool)
        let rawIndex: UInt32
        let signedIndex: Bool
        if let indexExpression {
            rawIndex = try value(indexExpression, in: &state)
            signedIndex = try module.types.value(for: state.function.expressions.value(for: indexExpression).type).kind == .scalar(.i32)
        } else if let fixedIndex {
            rawIndex = try unsignedConstantID(fixedIndex)
            signedIndex = false
        } else { throw ShaderFailure.invalidHandle }
        let index = try state.bufferProjections[address]?.index ?? unsignedAccessIndex(rawIndex, signed: signedIndex)
        let indexNonNegative: UInt32
        if signedIndex {
            indexNonNegative = try rawValueInstruction(175, type: bool,
                operands: [rawIndex, literalID(.i32(0))]) // OpSGreaterThanEqual
        } else { indexNonNegative = try literalID(.bool(true)) }
        let indexBeforeEnd = try rawValueInstruction(176, type: bool, operands: [index, length])
        let safeIndex = try rawValueInstruction(169, type: scalarTypeID(.u32), operands: [indexBeforeEnd, index, length])
        let remaining = try rawValueInstruction(130, type: scalarTypeID(.u32), operands: [length, safeIndex])
        let strideType = try module.types.value(for: state.function.expressions.value(for: strideHandle).type).kind
        let signedStride: Bool
        if case .scalar(.i32) = strideType { signedStride = true } else { signedStride = false }
        let strideNonNegative: UInt32
        let unsignedStride: UInt32
        if signedStride {
            let signedZero = try literalID(.i32(0))
            strideNonNegative = try rawValueInstruction(175, type: bool, operands: [stride, signedZero]) // OpSGreaterThanEqual
            unsignedStride = try rawValueInstruction(124, type: scalarTypeID(.u32), operands: [stride]) // OpBitcast
        } else {
            strideNonNegative = try literalID(.bool(true))
            unsignedStride = stride
        }
        let strideIsZero = try rawValueInstruction(170, type: bool, operands: [unsignedStride, zero])
        let positiveStride = try rawValueInstruction(172, type: bool, operands: [unsignedStride, zero])
        let safeStride = try rawValueInstruction(169, type: scalarTypeID(.u32), operands: [positiveStride, unsignedStride, one])
        let minor = try unsignedConstantID(minorCount)
        let hasFirstRow = try rawValueInstruction(174, type: bool, operands: [remaining, minor]) // UGreaterThanEqual
        let safeTail = try rawValueInstruction(130, type: scalarTypeID(.u32),
            operands: [remaining, try rawValueInstruction(169, type: scalarTypeID(.u32), operands: [hasFirstRow, minor, zero])])
        let quotient = try rawValueInstruction(134, type: scalarTypeID(.u32), operands: [safeTail, safeStride]) // OpUDiv
        let lastMajor = try unsignedConstantID(majorCount - 1)
        let rowsFit = try rawValueInstruction(174, type: bool, operands: [quotient, lastMajor])
        let zeroStrideFits = try rawValueInstruction(166, type: bool, operands: [strideIsZero, hasFirstRow])
        let positiveStrideFits = try rawValueInstruction(166, type: bool, operands: [positiveStride, rowsFit])
        let extentFits = try rawValueInstruction(166, type: bool, operands: [zeroStrideFits, positiveStrideFits])
        var valid = try rawValueInstruction(167, type: bool, operands: [indexBeforeEnd, extentFits])
        valid = try rawValueInstruction(167, type: bool, operands: [valid, indexNonNegative])
        valid = try rawValueInstruction(167, type: bool, operands: [valid, strideNonNegative])
        valid = try rawValueInstruction(167, type: bool, operands: [valid, hasFirstRow])
        switch state.bufferProjections[address]?.bounds {
        case .outside?: return try literalID(.bool(false))
        case .check(let inherited)?: valid = try rawValueInstruction(167, type: bool, operands: [valid, inherited])
        case nil: break
        }
        return valid
    }

    private func cooperativeMemoryOperand(
        pointer: IRHandle<IRExpression>, layout: UInt32, stride: UInt32, in state: SPIRVFunctionState
    ) throws -> [UInt32] {
        guard case .pointer(_, let space) = try module.types.value(for: state.function.expressions.value(for: pointer).type).kind else {
            return [layout, stride]
        }
        switch space {
        case .storage, .workgroup, .taskPayload, .uniform:
            return [layout, stride, 0x20] // NonPrivatePointerKHR
        default:
            return [layout, stride]
        }
    }

    private mutating func cooperativeStoreStride(
        _ stride: UInt32, handle: IRHandle<IRExpression>, in state: SPIRVFunctionState
    ) throws -> (condition: UInt32, safe: UInt32) {
        let bool = try scalarTypeID(.bool), uint = try scalarTypeID(.u32), one = try unsignedConstantID(1)
        let kind = try module.types.value(for: state.function.expressions.value(for: handle).type).kind
        if let bits = constantWords[stride] {
            let positive: Bool
            switch kind {
            case .scalar(.i32): positive = Int32(bitPattern: bits) > 0
            case .scalar(.u32): positive = bits > 0
            default: throw unsupported("requires a 32-bit integer cooperative store stride")
            }
            let safe: UInt32
            if positive { safe = try unsignedConstantID(bits) } else { safe = one }
            return (try literalID(.bool(positive)), safe)
        }
        let zero: UInt32, condition: UInt32, unsignedStride: UInt32
        switch kind {
        case .scalar(.i32):
            zero = try literalID(.i32(0))
            condition = try rawValueInstruction(173, type: bool, operands: [stride, zero]) // OpSGreaterThan
            unsignedStride = try rawValueInstruction(124, type: uint, operands: [stride]) // OpBitcast
        case .scalar(.u32):
            zero = try unsignedConstantID(0)
            condition = try rawValueInstruction(172, type: bool, operands: [stride, zero]) // OpUGreaterThan
            unsignedStride = stride
        default:
            throw unsupported("requires a 32-bit integer cooperative store stride")
        }
        let safe = try rawValueInstruction(169, type: uint, operands: [condition, unsignedStride, one]) // OpSelect
        return (condition, safe)
    }

    private mutating func unsignedAccessIndex(_ index: UInt32, signed: Bool) throws -> UInt32 {
        guard signed else { return index }
        if let bits = constantWords[index] { return try unsignedConstantID(bits) }
        return try rawValueInstruction(124, type: scalarTypeID(.u32), operands: [index])
    }

    private mutating func pointerAccessBounds(
        _ base: IRHandle<IRExpression>, index: UInt32, in state: inout SPIRVFunctionState
    ) throws -> SPIRVBufferBounds? {
        switch try valueKind(base, in: state) {
        case .pointer(let type, let space):
            switch try module.types.value(for: type).kind {
            case .array(_, .constant(let count), _):
                return try indexBounds(index, signed: false, count: count)
            case .array(_, .dynamic, _):
                guard case .storage = space else { throw unsupported("runtime array indexing requires storage") }
                let length = try writeRuntimeArrayLength(base, resultType: scalarTypeID(.u32), in: &state)
                let bool = try scalarTypeID(.bool)
                let inRange = try rawValueInstruction(176, type: bool, operands: [index, length])
                // OpAccessChain treats even unsigned 32-bit indices as signed.
                let positive = try rawValueInstruction(176, type: bool, operands: [index, unsignedConstantID(0x8000_0000)])
                return try .check(rawValueInstruction(167, type: bool, operands: [inRange, positive]))
            case .matrix(let columns, _, _), .vector(let columns, _):
                return try indexBounds(index, signed: false, count: UInt32(columns.rawValue))
            default: return nil
            }
        case .valuePointer(let size?, _, _):
            return try indexBounds(index, signed: false, count: UInt32(size.rawValue))
        default: return nil
        }
    }

    private mutating func dataAccessChain(
        type: IRHandle<IRType>, base: UInt32, index: UInt32, bounds: SPIRVBufferBounds? = nil,
        in state: inout SPIRVFunctionState
    ) throws -> UInt32 {
        let projection = state.bufferProjections[base]
        let combined: SPIRVBufferBounds?
        switch (projection?.bounds, bounds) {
        case (.outside?, _), (_, .outside?): combined = .outside
        case (.check(let ancestor)?, .check(let current)?):
            combined = try .check(rawValueInstruction(167, type: scalarTypeID(.bool), operands: [ancestor, current]))
        case (let bound?, nil), (nil, let bound?): combined = bound
        case (nil, nil): combined = nil
        }
        if let combined {
            return try reserveBufferProjection(.init(base: base, index: index, pointerType: typeID(type),
                bounds: combined, nonUniform: projection?.nonUniform ?? false, ownBounds: bounds), in: &state)
        }
        let pointer = try valueInstruction(65, type: type, operands: [base, index])
        if nonUniformResourceIDs.contains(base) { try markNonUniformResource(pointer) }
        return pointer
    }

    private mutating func uniformWorkgroupAddress(_ address: UInt32, in state: SPIRVFunctionState) throws -> UInt32 {
        guard let projection = state.bufferProjections[address] else { return address }
        try budget.enterDepth()
        defer { try? budget.leaveDepth() }
        let base = try uniformWorkgroupAddress(projection.base, in: state)
        let index: UInt32
        // A valid fallback keeps the barriers adjacent to the load. Mask its result afterward.
        switch projection.ownBounds {
        case .check(let valid)?:
            index = try rawValueInstruction(169, type: scalarTypeID(.u32),
                                            operands: [valid, projection.index, unsignedConstantID(0)])
        case .outside?: index = try unsignedConstantID(0)
        case nil: index = projection.index
        }
        return try rawValueInstruction(65, type: projection.pointerType, operands: [base, index])
    }

    private mutating func reserveBufferProjection(
        _ projection: SPIRVBufferProjection, in state: inout SPIRVFunctionState
    ) throws -> UInt32 {
        // Keep an internal ID for the path. Each use emits its own OpAccessChain inside the bounds check.
        let id = try allocateID()
        state.bufferProjections[id] = projection
        return id
    }

    private mutating func materializeBufferPointer(_ address: UInt32, in state: SPIRVFunctionState) throws -> UInt32 {
        guard let projection = state.bufferProjections[address] else { return address }
        try budget.enterDepth()
        defer { try? budget.leaveDepth() }
        let base = try materializeBufferPointer(projection.base, in: state)
        let pointer = try rawValueInstruction(65, type: projection.pointerType, operands: [base, projection.index])
        if projection.nonUniform { try markNonUniformResource(pointer) }
        return pointer
    }

    mutating func guardedBufferRead(
        address: UInt32, type: UInt32, in state: SPIRVFunctionState,
        _ read: (inout SPIRVBinaryWriter, UInt32) throws -> UInt32
    ) throws -> UInt32 {
        guard let projection = state.bufferProjections[address] else { return try read(&self, address) }
        guard case .check(let condition) = projection.bounds else { return try zeroID(typeID: type) }
        return try valueWhen(condition, type: type) { writer in
            let pointer = try writer.materializeBufferPointer(address, in: state)
            return try read(&writer, pointer)
        }
    }

    mutating func valueWhen(
        _ condition: UInt32, type: UInt32, _ produce: (inout SPIRVBinaryWriter) throws -> UInt32
    ) throws -> UInt32 {
        let zero = try zeroID(typeID: type)
        let valid = try allocateID()
        let invalid = try allocateID()
        let merge = try allocateID()
        try emit(247, [merge, 0]) // OpSelectionMerge
        try emit(250, [condition, valid, invalid]) // OpBranchConditional
        try emit(248, [valid])
        let result = try produce(&self)
        guard let predecessor = currentBlock else { throw ShaderFailure.invalidHandle }
        try emit(249, [merge])
        try emit(248, [invalid])
        try emit(249, [merge])
        try emit(248, [merge])
        return try rawValueInstruction(245, type: type, operands: [result, predecessor, zero, invalid]) // OpPhi
    }

    mutating func guardedBufferEffect(
        address: UInt32, in state: SPIRVFunctionState,
        _ effect: (inout SPIRVBinaryWriter, UInt32) throws -> Void
    ) throws {
        guard let projection = state.bufferProjections[address] else {
            try effect(&self, address)
            return
        }
        guard case .check(let condition) = projection.bounds else { return }
        try effectWhen(condition) { writer in
            let pointer = try writer.materializeBufferPointer(address, in: state)
            try effect(&writer, pointer)
        }
    }

    mutating func effectWhen(
        _ condition: UInt32, _ effect: (inout SPIRVBinaryWriter) throws -> Void
    ) throws {
        let valid = try allocateID()
        let merge = try allocateID()
        try emit(247, [merge, 0]) // OpSelectionMerge
        try emit(250, [condition, valid, merge]) // OpBranchConditional
        try emit(248, [valid])
        try effect(&self)
        try emit(249, [merge])
        try emit(248, [merge])
    }

    private mutating func repackBufferRecord(
        _ value: UInt32, from source: UInt32, to target: UInt32, element: IRHandle<IRType>
    ) throws -> UInt32 {
        guard source != target else { return value }
        guard case .structure(let fields, _) = try module.types.value(for: element).kind else {
            throw ShaderFailure.invalidHandle
        }
        var components: [UInt32] = []
        for (index, field) in fields.enumerated() {
            components.append(try valueInstruction(81, type: field.type, operands: [value, UInt32(index)]))
        }
        return try rawValueInstruction(80, type: target, operands: components)
    }

    private mutating func writeRuntimeArrayLength(
        _ pointer: IRHandle<IRExpression>, resultType: UInt32, in state: inout SPIRVFunctionState
    ) throws -> UInt32 {
        let operands = try arrayLengthOperands(pointer, in: &state)
        return try guardedBufferRead(address: operands[0], type: resultType, in: state) { writer, structure in
            try writer.rawValueInstruction(68, type: resultType, operands: [structure, operands[1]]) // OpArrayLength
        }
    }

    private mutating func arrayLengthOperands(
        _ pointer: IRHandle<IRExpression>, in state: inout SPIRVFunctionState
    ) throws -> [UInt32] {
        try budget.enterDepth()
        defer { try? budget.leaveDepth() }
        switch try state.function.expressions.value(for: pointer).kind {
        case .accessIndex(let structure, let member):
            return try [value(structure, in: &state), member]
        case .globalVariable(let global):
            let binding = try bufferGlobal(global)
            guard let member = binding.wrapperMember else { throw unsupported("requires a runtime array member for arrayLength") }
            return [binding.id, member]
        case .runtimeValue(let source):
            return try arrayLengthOperands(source, in: &state)
        default: throw unsupported("requires an addressable runtime array for arrayLength")
        }
    }

    private mutating func boundedLayoutIndex(_ index: UInt32, signed: Bool, count: UInt32) throws -> UInt32 {
        if count == 1 { return try unsignedConstantID(0) }
        let limit = count - 1
        if let bits = constantWords[index] {
            return try unsignedConstantID(signed && Int32(bitPattern: bits) < 0 ? 0 : min(bits, limit))
        }

        let u32 = try scalarTypeID(.u32)
        let bool = try scalarTypeID(.bool)
        let upper = try unsignedConstantID(limit)
        let unsigned = signed ? try rawValueInstruction(124, type: u32, operands: [index]) : index
        let above = try rawValueInstruction(172, type: bool, operands: [unsigned, upper]) // OpUGreaterThan
        let bounded = try rawValueInstruction(169, type: u32, operands: [above, upper, unsigned]) // OpSelect
        guard signed else { return bounded }
        let signedZero = try literalID(.i32(0))
        let unsignedZero = try unsignedConstantID(0)
        let negative = try rawValueInstruction(177, type: bool, operands: [index, signedZero]) // OpSLessThan
        return try rawValueInstruction(169, type: u32, operands: [negative, unsignedZero, bounded])
    }

    private func isPointer(_ type: IRHandle<IRType>) throws -> Bool {
        switch try module.types.value(for: type).kind {
        case .pointer, .valuePointer: true
        default: false
        }
    }

    func valueKind(_ operand: IRHandle<IRExpression>, in state: SPIRVFunctionState) throws -> IRTypeKind {
        try module.types.value(for: state.function.expressions.value(for: operand).type).kind
    }

    private func scalarKind(_ operand: IRHandle<IRExpression>, in state: SPIRVFunctionState) throws -> IRScalar {
        switch try valueKind(operand, in: state) {
        case .scalar(let scalar), .vector(_, let scalar): return scalar
        default: throw unsupported("requires scalar or vector arithmetic")
        }
    }

    private func binaryOpcode(_ operation: IRBinaryOperator, scalar: IRScalar) throws -> UInt32 {
        let float = scalar.kind == .float
        let bool = scalar == .bool
        let signed = scalar.kind == .sint
        switch operation {
        case .add: return float ? 129 : 128
        case .subtract: return float ? 131 : 130
        case .multiply: return float ? 133 : 132
        case .divide where float: return 136
        case .modulo where float: return 140
        case .equal: return float ? 180 : bool ? 164 : 170
        case .notEqual: return float ? 183 : bool ? 165 : 171
        case .greater: return float ? 186 : signed ? 173 : 172
        case .greaterEqual: return float ? 190 : signed ? 175 : 174
        case .less: return float ? 184 : signed ? 177 : 176
        case .lessEqual: return float ? 188 : signed ? 179 : 178
        case .logicalOr: return 166
        case .logicalAnd: return 167
        case .inclusiveOr: return bool ? 166 : 197
        case .exclusiveOr: return bool ? 165 : 198
        case .and: return bool ? 167 : 199
        default: throw unsupported("does not lower this binary operator")
        }
    }
}
