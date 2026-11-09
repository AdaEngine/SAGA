import SAGAShaderIR

extension SPIRVBinaryWriter {
    mutating func atomicPointee(
        _ pointer: IRHandle<IRExpression>, in state: SPIRVFunctionState
    ) throws -> (scalar: IRScalar, scope: UInt32)? {
        guard case .pointer(let base, let space) = try valueKind(pointer, in: state),
              case .atomic(let scalar) = try module.types.value(for: base).kind else { return nil }
        guard [.i32, .u32, .f32, .i64, .u64].contains(scalar) else {
            throw unsupported("requires a supported 32-bit or 64-bit atomic")
        }
        if scalar.width == 8 { try requireCapability(12) } // Int64Atomics
        switch space {
        case .storage:
            markStorageDeviceScopeAtomic()
            return (scalar, 1) // Device scope; VulkanMemoryModelDeviceScope is added if needed.
        case .workgroup: return (scalar, 2)
        default: throw unsupported("requires a storage or workgroup atomic")
        }
    }

    mutating func readAtomic(address: UInt32, type: UInt32, scope: UInt32, in state: SPIRVFunctionState) throws -> UInt32 {
        let scope = try unsignedConstantID(scope)
        let semantics = try unsignedConstantID(0) // Relaxed
        return try guardedBufferRead(address: address, type: type, in: state) { writer, pointer in
            try writer.rawValueInstruction(227, type: type, operands: [pointer, scope, semantics]) // OpAtomicLoad
        }
    }

    mutating func writeAtomicStore(address: UInt32, data: UInt32, scope: UInt32, in state: SPIRVFunctionState) throws {
        try guardedBufferEffect(address: address, in: state) { writer, pointer in
            try writer.emitAtomicStore(pointer, data: data, scope: scope)
        }
    }

    mutating func emitAtomicStore(_ pointer: UInt32, data: UInt32, scope: UInt32) throws {
        try emit(228, [pointer, unsignedConstantID(scope), unsignedConstantID(0), data]) // OpAtomicStore
    }

    mutating func writeAtomic(
        pointer: IRHandle<IRExpression>, operation: IRAtomicFunction, operand: IRHandle<IRExpression>,
        result: IRHandle<IRExpression>?, in state: inout SPIRVFunctionState
    ) throws {
        guard let atomic = try atomicPointee(pointer, in: state) else { throw ShaderFailure.invalidHandle }
        let address = try value(pointer, in: &state)
        let data = try value(operand, in: &state)
        let scalarType = try scalarTypeID(atomic.scalar)
        let scope = try unsignedConstantID(atomic.scope)
        let semantics = try unsignedConstantID(0) // Relaxed
        let resultType = try result.map { try typeID(state.function.expressions.value(for: $0).type) } ?? scalarType
        let perform: (inout SPIRVBinaryWriter, UInt32) throws -> UInt32
        if operation == .addFloat32Bits {
            guard atomic.scalar == .u32, atomic.scope == 1 else {
                throw unsupported("requires a storage u32 atomic for float bit addition")
            }
            perform = { writer, pointer in
                try writer.addFloatAtomicBits(pointer: pointer, data: data, scope: scope, semantics: semantics)
            }
        } else {
            perform = try atomicOperation(operation, scalar: atomic.scalar, storage: atomic.scope == 1,
                                          data: data, scope: scope, semantics: semantics, resultType: resultType, in: &state)
        }
        if let result {
            let id = try guardedBufferRead(address: address, type: resultType, in: state, perform)
            state.bind(result, to: id)
        } else {
            try guardedBufferEffect(address: address, in: state) { writer, pointer in
                _ = try perform(&writer, pointer)
            }
        }
    }

    private mutating func atomicOperation(
        _ operation: IRAtomicFunction, scalar: IRScalar, storage: Bool, data: UInt32, scope: UInt32,
        semantics: UInt32, resultType: UInt32, in state: inout SPIRVFunctionState
    ) throws -> (inout SPIRVBinaryWriter, UInt32) throws -> UInt32 {
        let scalarType = try scalarTypeID(scalar)
        let opcode: UInt32
        let compare: UInt32?
        let replacement: UInt32
        switch operation {
        case .exchange(let comparator?, _):
            guard scalar != .f32 else { throw unsupported("requires an integer atomic compare-exchange") }
            opcode = 230 // OpAtomicCompareExchange
            compare = try value(comparator, in: &state)
            replacement = data
        default:
            compare = nil
            if scalar == .f32 {
                guard storage else { throw unsupported("requires storage for float atomic arithmetic and exchange") }
                switch operation {
                case .add, .subtract:
                    opcode = 6035 // OpAtomicFAddEXT
                    replacement = try operation == .subtract
                        ? rawValueInstruction(127, type: scalarType, operands: [data]) : data // OpFNegate
                case .exchange(compare: nil, strong: false):
                    opcode = 229 // OpAtomicExchange
                    replacement = data
                default: throw unsupported("does not lower this float atomic operation")
                }
            } else {
                opcode = try atomicOpcode(operation, signed: scalar.kind == .sint)
                replacement = data
            }
        }
        let operands = compare.map { [scope, semantics, semantics, replacement, $0] } ?? [scope, semantics, replacement]
        return { writer, pointer in
            if opcode == 6035 { try writer.requireFloatAtomicAdd() }
            let previous = try writer.rawValueInstruction(opcode, type: scalarType, operands: [pointer] + operands)
            guard let compare, resultType != scalarType else { return previous }
            // Strong compare-exchange is a valid implementation of WGSL's weak operation.
            let exchanged = try writer.rawValueInstruction(170, type: writer.scalarTypeID(.bool), operands: [previous, compare])
            return try writer.rawValueInstruction(80, type: resultType, operands: [previous, exchanged])
        }
    }

    private mutating func addFloatAtomicBits(
        pointer: UInt32, data: UInt32, scope: UInt32, semantics: UInt32
    ) throws -> UInt32 {
        // A mixed float/integer view uses one physical u32 binding. Compare its bits on retry,
        // including negative zero and NaN, and use float arithmetic only for the new value.
        guard let origin = currentBlock else { throw ShaderFailure.invalidHandle }
        let u32 = try scalarTypeID(.u32)
        let f32 = try scalarTypeID(.f32)
        let delta = try rawValueInstruction(124, type: f32, operands: [data])
        let initial = try rawValueInstruction(227, type: u32, operands: [pointer, scope, semantics])
        let header = try allocateID()
        let body = try allocateID()
        let next = try allocateID()
        let merge = try allocateID()
        let assumed = try allocateID()
        let previous = try allocateID()
        try emit(249, [header])
        try emit(248, [header])
        try emit(245, [u32, assumed, initial, origin, previous, next]) // OpPhi
        try emit(246, [merge, next, 0]) // OpLoopMerge
        try emit(249, [body])
        try emit(248, [body])
        let old = try rawValueInstruction(124, type: f32, operands: [assumed])
        let sum = try rawValueInstruction(129, type: f32, operands: [old, delta])
        let bits = try rawValueInstruction(124, type: u32, operands: [sum])
        try emit(230, [u32, previous, pointer, scope, semantics, semantics, bits, assumed])
        let success = try rawValueInstruction(170, type: scalarTypeID(.bool), operands: [previous, assumed])
        try emit(249, [next])
        try emit(248, [next])
        try emit(250, [success, merge, header])
        try emit(248, [merge])
        return assumed
    }

    func atomicOpcode(_ operation: IRAtomicFunction, signed: Bool) throws -> UInt32 {
        switch operation {
        case .exchange(compare: nil, strong: false): 229
        case .add: 234
        case .subtract: 235
        case .min: signed ? 236 : 237
        case .minWithOppositeSignedness: signed ? 237 : 236
        case .max: signed ? 238 : 239
        case .maxWithOppositeSignedness: signed ? 239 : 238
        case .and: 240
        case .inclusiveOr: 241
        case .exclusiveOr: 242
        default: throw unsupported("does not lower this atomic operation")
        }
    }
}
