import SAGAShaderIR

extension SPIRVBinaryWriter {
    mutating func writeControlBarrier(_ flags: IRBarrier) throws {
        guard !flags.isEmpty, flags.subtracting([.workgroup, .storage, .texture]).isEmpty else {
            throw unsupported("requires a workgroup control barrier")
        }
        let scope = try unsignedConstantID(2)
        var semantics: UInt32 = 0x8 // AcquireRelease
        if flags.contains(.storage) { semantics |= 0x40 } // UniformMemory
        if flags.contains(.workgroup) { semantics |= 0x100 } // WorkgroupMemory
        if flags.contains(.texture) { semantics |= 0x800 } // ImageMemory
        try emit(224, [scope, scope, unsignedConstantID(semantics)]) // OpControlBarrier
    }

    mutating func writeWorkgroupInitialization(_ globals: [IRHandle<IRGlobalVariable>], lane: UInt32) throws {
        let initialize = try allocateID()
        let merge = try allocateID()
        let zero = try unsignedConstantID(0)
        let first = try rawValueInstruction(170, type: scalarTypeID(.bool), operands: [lane, zero]) // OpIEqual
        try emit(247, [merge, 0]) // OpSelectionMerge
        try emit(250, [first, initialize, merge]) // OpBranchConditional
        try emit(248, [initialize]) // OpLabel
        for handle in globals {
            let type = try module.globalVariables.value(for: handle).type
            let pointer = try workgroupGlobal(handle)
            try initializeWorkgroupValue(type, at: pointer)
        }
        try emit(249, [merge]) // OpBranch
        try emit(248, [merge])
        try writeControlBarrier(.workgroup)
    }

    private mutating func initializeWorkgroupValue(_ type: IRHandle<IRType>, at pointer: UInt32) throws {
        try charge()
        try budget.enterDepth()
        defer { try? budget.leaveDepth() }
        switch try module.types.value(for: type).kind {
        case .structure(let members, _):
            for (index, member) in members.enumerated() {
                let field = try workgroupProjection(pointer, index: unsignedConstantID(UInt32(index)), type: member.type)
                try initializeWorkgroupValue(member.type, at: field)
            }
        case .array(let element, .constant(let count), _):
            try initializeWorkgroupArray(element, count: count, at: pointer)
        case .atomic(let scalar):
            guard [.i32, .u32, .f32, .i64, .u64].contains(scalar) else {
                throw unsupported("requires a supported workgroup atomic")
            }
            if scalar.width == 8 { try requireCapability(12) } // Int64Atomics
            try emitAtomicStore(pointer, data: zeroID(typeID: typeID(type)), scope: 2)
        default:
            try emit(62, [pointer, zeroID(typeID: typeID(type))]) // OpStore
        }
    }

    mutating func workgroupProjection(_ base: UInt32, index: UInt32, type: IRHandle<IRType>) throws -> UInt32 {
        let payload = taskPayloadIDs.contains(base)
        let pointer = try pointerTypeID(storageClass: payload ? 5402 : 4, pointee: workgroupTypeID(type))
        let result = try rawValueInstruction(65, type: pointer, operands: [base, index]) // OpAccessChain
        if payload { taskPayloadIDs.insert(result) }
        return result
    }

    func workgroupAggregatePointee(
        _ pointer: IRHandle<IRExpression>, in state: SPIRVFunctionState
    ) throws -> IRHandle<IRType>? {
        guard case .pointer(let base, let space) = try valueKind(pointer, in: state),
              space == .workgroup || space == .taskPayload else { return nil }
        switch try module.types.value(for: base).kind {
        case .array, .structure: return base
        default: return nil
        }
    }

    mutating func convertWorkgroupValue(
        _ value: UInt32, type: IRHandle<IRType>, toWorkgroup: Bool
    ) throws -> UInt32 {
        try budget.enterDepth()
        defer { try? budget.leaveDepth() }
        let members: [IRHandle<IRType>]
        switch try module.types.value(for: type).kind {
        case .structure(let fields, _):
            try charge(fields.count)
            members = fields.map(\.type)
        case .array(let element, .constant(let count), _):
            try charge(Int(count))
            members = Array(repeating: element, count: Int(count))
        default: return value
        }
        var components: [UInt32] = []
        components.reserveCapacity(members.count)
        for (index, member) in members.enumerated() {
            let source = try toWorkgroup ? typeID(member) : workgroupTypeID(member)
            let child = try rawValueInstruction(81, type: source, operands: [value, UInt32(index)])
            components.append(try convertWorkgroupValue(child, type: member, toWorkgroup: toWorkgroup))
        }
        let target = try toWorkgroup ? workgroupTypeID(type) : typeID(type)
        return try rawValueInstruction(80, type: target, operands: components)
    }

    private mutating func initializeWorkgroupArray(_ element: IRHandle<IRType>, count: UInt32, at pointer: UInt32) throws {
        // A structured loop keeps initializer code size independent of the array length.
        let start = try allocateID()
        let header = try allocateID()
        let body = try allocateID()
        let next = try allocateID()
        let merge = try allocateID()
        let index = try allocateID()
        let incremented = try allocateID()
        let u32 = try scalarTypeID(.u32)
        let zero = try unsignedConstantID(0)
        let one = try unsignedConstantID(1)
        let length = try unsignedConstantID(count)
        try emit(249, [start])
        try emit(248, [start])
        try emit(249, [header])
        try emit(248, [header])
        try emit(245, [u32, index, zero, start, incremented, next]) // OpPhi
        try emit(246, [merge, next, 0]) // OpLoopMerge
        try emit(249, [body])
        try emit(248, [body])
        let address = try workgroupProjection(pointer, index: index, type: element)
        try initializeWorkgroupValue(element, at: address)
        try emit(249, [next])
        try emit(248, [next])
        try emit(128, [u32, incremented, index, one]) // OpIAdd
        let more = try rawValueInstruction(176, type: scalarTypeID(.bool), operands: [incremented, length]) // OpULessThan
        try emit(250, [more, header, merge])
        try emit(248, [merge])
    }
}
