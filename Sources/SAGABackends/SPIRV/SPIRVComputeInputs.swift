import SAGAShaderIR

struct SPIRVComputeInputPlan {
    fileprivate struct Input {
        let variable: UInt32
        let type: UInt32
        let builtin: UInt32
    }

    fileprivate enum Argument {
        case input(Int)
        case structure(type: UInt32, inputs: Range<Int>)
    }

    let interfaceIDs: [UInt32]
    fileprivate let inputs: [Input]
    fileprivate let arguments: [Argument]
    fileprivate let workgroup: [IRHandle<IRGlobalVariable>]
    fileprivate let initializationIndex: Int?

    /// Input variable that holds the local invocation index, when the plan loads it.
    var laneVariable: UInt32? { initializationIndex.map { inputs[$0].variable } }
}

extension SPIRVBinaryWriter {
    mutating func prepareComputeInputs(
        _ function: IRFunction, workgroup: [IRHandle<IRGlobalVariable>]
    ) throws -> SPIRVComputeInputPlan {
        try charge(function.arguments.count)
        var inputs: [SPIRVComputeInputPlan.Input] = []
        var arguments: [SPIRVComputeInputPlan.Argument] = []
        arguments.reserveCapacity(function.arguments.count)
        for argument in function.arguments {
            let start = inputs.count
            if case .structure(let members, _) = try module.types.value(for: argument.type).kind {
                guard argument.binding == nil else { throw unsupported("requires builtin bindings on input structure members") }
                try charge(members.count)
                for member in members {
                    inputs.append(try declareComputeInput(member.type, binding: member.binding, name: member.name))
                }
                let structure = try typeID(argument.type)
                arguments.append(.structure(type: structure, inputs: start..<inputs.count))
            } else {
                inputs.append(try declareComputeInput(argument.type, binding: argument.binding, name: argument.name))
                arguments.append(.input(start))
            }
        }
        let initializationIndex: Int?
        if workgroup.isEmpty {
            initializationIndex = nil
        } else if let existing = inputs.firstIndex(where: { $0.builtin == 29 }) {
            initializationIndex = existing
        } else {
            initializationIndex = inputs.count
            inputs.append(try declareComputeInput(scalarTypeID(.u32), builtin: 29, name: nil))
        }
        return .init(interfaceIDs: inputs.map(\.variable), inputs: inputs, arguments: arguments,
                     workgroup: workgroup, initializationIndex: initializationIndex)
    }

    mutating func emitComputeInputs(_ plan: SPIRVComputeInputPlan) throws -> [UInt32] {
        try charge(plan.inputs.count + plan.arguments.count)
        let values = try plan.inputs.map { input in
            let id = try allocateID()
            try emit(61, [input.type, id, input.variable]) // OpLoad
            return id
        }
        if let index = plan.initializationIndex {
            try writeWorkgroupInitialization(plan.workgroup, lane: values[index])
        }
        return try plan.arguments.map { argument in
            switch argument {
            case .input(let index):
                return values[index]
            case .structure(let type, let range):
                let id = try allocateID()
                try emit(80, [type, id] + values[range]) // OpCompositeConstruct
                return id
            }
        }
    }

    private mutating func declareComputeInput(
        _ type: IRHandle<IRType>, binding: IRBinding?, name: String?
    ) throws -> SPIRVComputeInputPlan.Input {
        guard case .builtIn(let builtin) = binding else {
            throw unsupported("requires builtin bindings on compute inputs")
        }
        let number: UInt32
        switch builtin {
        case .numWorkgroups: number = 24
        case .workgroupId: number = 26
        case .localInvocationId: number = 27
        case .globalInvocationId: number = 28
        case .localInvocationIndex: number = 29
        case .subgroupSize: number = 36
        case .numSubgroups: number = 38
        case .subgroupId: number = 40
        case .subgroupInvocationId: number = 41
        default: throw unsupported("does not lower compute builtin \(builtin)")
        }
        let subgroup = number >= 36
        if subgroup { try requireCapability(61) } // GroupNonUniform
        let expected: IRTypeKind = subgroup || builtin == .localInvocationIndex
            ? .scalar(.u32) : .vector(size: .three, scalar: .u32)
        guard try module.types.value(for: type).kind == expected else {
            throw unsupported("requires the declared SPIR-V type for compute builtin \(builtin)")
        }
        return try declareComputeInput(typeID(type), builtin: number, name: name)
    }

    private mutating func declareComputeInput(
        _ valueType: UInt32, builtin: UInt32, name: String?
    ) throws -> SPIRVComputeInputPlan.Input {
        let pointer = try pointerTypeID(storageClass: 1, pointee: valueType)
        let variable = try allocateID()
        try declare(59, [pointer, variable, 1]) // OpVariable Input
        try decorate(71, [variable, 11, builtin]) // BuiltIn
        try debugName(variable, name)
        return .init(variable: variable, type: valueType, builtin: builtin)
    }
}
