import SAGAShaderIR

// Ray queries for SPIR-V (SPV_KHR_ray_query). Every query local gets a state word and the ray's tMax, mirroring the GLSL
// writer: bit 0 means initialized, bit 1 a pending candidate hit, bit 2 a finished query.

struct SPIRVRayQueryLocal {
    let state: UInt32
    let tMax: UInt32
}

extension SPIRVBinaryWriter {
    mutating func requireRayQuery() {
        requiresSpirv14 = true
        capabilities.insert(4472)  // RayQueryKHR
        extensions.insert("SPV_KHR_ray_query")
    }

    mutating func requireRayPositionFetch() {
        capabilities.insert(5391)  // RayQueryPositionFetchKHR
        extensions.insert("SPV_KHR_ray_tracing_position_fetch")
    }

    /// The three vertex positions of the pending or committed triangle hit, zero when there is none.
    mutating func writeRayQueryVertexPositions(
        query: IRHandle<IRExpression>, committed: Bool, type: IRHandle<IRType>, in state: inout SPIRVFunctionState
    ) throws -> UInt32 {
        requireRayQuery()
        requireRayPositionFetch()
        let (variable, aux) = try rayQueryLocal(query, in: state)
        guard case .array(_, .constant(3), _) = try module.types.value(for: type).kind else {
            throw unsupported("requires the ray vertex return layout")
        }
        let u32 = try scalarTypeID(.u32), bool = try scalarTypeID(.bool)
        let array = try typeID(type)
        let condition = committed ? try and(rayQueryFlag(aux, mask: 4), rayQueryFlag(aux, mask: 2)) : try candidatePending(aux)
        let choice = try unsignedConstantID(committed ? 1 : 0)
        return try valueWhen(condition, type: array) { writer in
            let hitType = try writer.rawValueInstruction(4479, type: u32, operands: [variable, choice])
            let isTriangle = try writer.rawValueInstruction(170, type: bool, operands: [hitType, writer.unsignedConstantID(committed ? 1 : 0)])
            return try writer.valueWhen(isTriangle, type: array) { writer in
                try writer.rawValueInstruction(5340, type: array, operands: [variable, choice])
            }
        }
    }

    /// Declares the state variables that accompany one query local; they must lead the function's first block.
    mutating func declareRayQueryAuxiliaries() throws -> SPIRVRayQueryLocal {
        let u32 = try scalarTypeID(.u32), f32 = try scalarTypeID(.f32)
        let statePointer = try pointerTypeID(storageClass: 7, pointee: u32)
        let maxPointer = try pointerTypeID(storageClass: 7, pointee: f32)
        let state = try allocateID(), tMax = try allocateID()
        try emit(59, [statePointer, state, 7, unsignedConstantID(0)])
        try emit(59, [maxPointer, tMax, 7, literalID(.f32(0))])
        return SPIRVRayQueryLocal(state: state, tMax: tMax)
    }

    private func rayQueryLocal(
        _ query: IRHandle<IRExpression>, in state: SPIRVFunctionState
    ) throws -> (variable: UInt32, aux: SPIRVRayQueryLocal) {
        guard case .localVariable(let local) = try state.function.expressions.value(for: query).kind,
              let aux = state.rayQueries[local.index], state.locals.indices.contains(local.index) else {
            throw unsupported("requires a ray query local variable")
        }
        return (state.locals[local.index], aux)
    }

    private mutating func rayQueryFlag(_ aux: SPIRVRayQueryLocal, mask: UInt32) throws -> UInt32 {
        let u32 = try scalarTypeID(.u32)
        let value = try rawValueInstruction(61, type: u32, operands: [aux.state])
        let bits = try rawValueInstruction(199, type: u32, operands: [value, unsignedConstantID(mask)])  // OpBitwiseAnd
        return try rawValueInstruction(171, type: scalarTypeID(.bool), operands: [bits, unsignedConstantID(0)])  // != 0
    }

    private mutating func and(_ a: UInt32, _ b: UInt32) throws -> UInt32 {
        try rawValueInstruction(167, type: scalarTypeID(.bool), operands: [a, b])  // OpLogicalAnd
    }

    private mutating func not(_ a: UInt32) throws -> UInt32 {
        try rawValueInstruction(168, type: scalarTypeID(.bool), operands: [a])  // OpLogicalNot
    }

    /// `(state & 2) != 0 && (state & 4) == 0`: a candidate hit is pending.
    private mutating func candidatePending(_ aux: SPIRVRayQueryLocal) throws -> UInt32 {
        try and(rayQueryFlag(aux, mask: 2), not(rayQueryFlag(aux, mask: 4)))
    }

    private mutating func setStateBits(_ aux: SPIRVRayQueryLocal, _ bits: UInt32) throws {
        let u32 = try scalarTypeID(.u32)
        let value = try rawValueInstruction(61, type: u32, operands: [aux.state])
        let updated = try rawValueInstruction(197, type: u32, operands: [value, unsignedConstantID(bits)])  // OpBitwiseOr
        try emit(62, [aux.state, updated])
    }

    private mutating func intersectionType(_ query: UInt32, committed: Bool) throws -> UInt32 {
        try rawValueInstruction(4479, type: scalarTypeID(.u32), operands: [query, unsignedConstantID(committed ? 1 : 0)])
    }

    mutating func writeRayQueryStatement(
        query: IRHandle<IRExpression>, function: IRRayQueryFunction, in state: inout SPIRVFunctionState
    ) throws {
        requireRayQuery()
        let (variable, aux) = try rayQueryLocal(query, in: state)
        let u32 = try scalarTypeID(.u32), f32 = try scalarTypeID(.f32), bool = try scalarTypeID(.bool)
        switch function {
        case .begin:
            try emit(62, [aux.state, unsignedConstantID(0)])
        case .initialize(let acceleration, let descriptor):
            guard let descriptorType = module.specialTypes.rayDescriptor,
                  case .structure(let members, _) = try module.types.value(for: descriptorType).kind, members.count == 6 else {
                throw unsupported("requires the ray descriptor layout")
            }
            let structure = try value(descriptor, in: &state)
            let structureAcceleration = try value(acceleration, in: &state)
            var fields: [UInt32] = []
            for (index, member) in members.enumerated() {
                fields.append(try rawValueInstruction(81, type: typeID(member.type), operands: [structure, UInt32(index)]))
            }
            let (flags, cull, tMin, tMax, origin, direction) = (fields[0], fields[1], fields[2], fields[3], fields[4], fields[5])
            try emit(62, [aux.state, unsignedConstantID(0)])
            // Reject non-finite vectors, NaN or negative times, inverted ranges and incompatible flag groups.
            let vec3Bool = try vectorTypeID(size: .three, scalar: .bool)
            func finite(_ vector: UInt32) throws -> UInt32 {
                let nan = try rawValueInstruction(154, type: bool, operands: [rawValueInstruction(156, type: vec3Bool, operands: [vector])])
                let inf = try rawValueInstruction(154, type: bool, operands: [rawValueInstruction(157, type: vec3Bool, operands: [vector])])
                return try and(not(nan), not(inf))
            }
            var valid = try and(finite(origin), finite(direction))
            valid = try and(valid, not(rawValueInstruction(156, type: bool, operands: [tMin])))
            valid = try and(valid, not(rawValueInstruction(156, type: bool, operands: [tMax])))
            valid = try and(valid, rawValueInstruction(190, type: bool, operands: [tMin, literalID(.f32(0))]))  // FOrdGreaterThanEqual
            valid = try and(valid, rawValueInstruction(188, type: bool, operands: [tMin, tMax]))  // FOrdLessThanEqual
            for mask: UInt32 in [195, 304, 768] {
                let masked = try rawValueInstruction(199, type: u32, operands: [flags, unsignedConstantID(mask)])
                let count = try rawValueInstruction(205, type: u32, operands: [masked])  // OpBitCount
                valid = try and(valid, rawValueInstruction(176, type: bool, operands: [count, unsignedConstantID(2)]))  // ULessThan
            }
            try effectWhen(valid) { writer in
                try writer.emit(4473, [variable, structureAcceleration, flags, cull, origin, tMin, direction, tMax])
                try writer.emit(62, [aux.tMax, tMax])
                try writer.emit(62, [aux.state, writer.unsignedConstantID(1)])
            }
        case .proceed(let result):
            let ready = try and(rayQueryFlag(aux, mask: 1), not(rayQueryFlag(aux, mask: 4)))
            let proceeded = try valueWhen(ready, type: bool) { writer in
                let more = try writer.rawValueInstruction(4477, type: bool, operands: [variable])
                let bits = try writer.rawValueInstruction(169, type: u32, operands: [more, writer.unsignedConstantID(2), writer.unsignedConstantID(6)])
                let current = try writer.rawValueInstruction(61, type: u32, operands: [aux.state])
                try writer.emit(62, [aux.state, writer.rawValueInstruction(197, type: u32, operands: [current, bits])])
                return more
            }
            state.bind(result, to: proceeded)
        case .terminate:
            try effectWhen(candidatePending(aux)) { writer in
                try writer.emit(4474, [variable])
                try writer.setStateBits(aux, 4)
            }
        case .confirmIntersection:
            let pending = try candidatePending(aux)
            try effectWhen(pending) { writer in
                let type = try writer.intersectionType(variable, committed: false)
                let triangle = try writer.rawValueInstruction(170, type: bool, operands: [type, writer.unsignedConstantID(0)])
                try writer.effectWhen(triangle) { try $0.emit(4476, [variable]) }
            }
        case .generateIntersection(let hit):
            let hitT = try value(hit, in: &state)
            let pending = try candidatePending(aux)
            try effectWhen(pending) { writer in
                let candidate = try writer.intersectionType(variable, committed: false)
                let box = try writer.rawValueInstruction(170, type: bool, operands: [candidate, writer.unsignedConstantID(1)])
                try writer.effectWhen(box) { writer in
                    let committed = try writer.intersectionType(variable, committed: true)
                    let hasHit = try writer.rawValueInstruction(171, type: bool, operands: [committed, writer.unsignedConstantID(0)])
                    let committedT = try writer.valueWhen(hasHit, type: f32) { writer in
                        try writer.rawValueInstruction(6018, type: f32, operands: [variable, writer.unsignedConstantID(1)])
                    }
                    let storedMax = try writer.rawValueInstruction(61, type: f32, operands: [aux.tMax])
                    let limit = try writer.rawValueInstruction(169, type: f32, operands: [hasHit, committedT, storedMax])
                    let tMin = try writer.rawValueInstruction(6016, type: f32, operands: [variable])
                    let inside = try writer.and(writer.rawValueInstruction(188, type: bool, operands: [tMin, hitT]),
                                                writer.rawValueInstruction(188, type: bool, operands: [hitT, limit]))
                    try writer.effectWhen(inside) { try $0.emit(4475, [variable, hitT]) }
                }
            }
        }
    }

    /// The intersection record of the query, `RayIntersection` in the IR, filled the way the GLSL writer fills it.
    mutating func writeRayQueryIntersection(
        query: IRHandle<IRExpression>, committed: Bool, type: IRHandle<IRType>, in state: inout SPIRVFunctionState
    ) throws -> UInt32 {
        requireRayQuery()
        let (variable, aux) = try rayQueryLocal(query, in: state)
        guard case .structure(let members, _) = try module.types.value(for: type).kind, members.count == 11 else {
            throw unsupported("requires the ray intersection layout")
        }
        let u32 = try scalarTypeID(.u32), f32 = try scalarTypeID(.f32), bool = try scalarTypeID(.bool)
        let structure = try typeID(type)
        let memberTypes = try members.map { try typeID($0.type) }
        let condition = committed ? try and(rayQueryFlag(aux, mask: 4), rayQueryFlag(aux, mask: 2)) : try candidatePending(aux)
        let one = try unsignedConstantID(committed ? 1 : 0)
        return try valueWhen(condition, type: structure) { writer in
            let hitType = try writer.rawValueInstruction(4479, type: u32, operands: [variable, one])
            // Candidate: 0 triangle, 1 box. Committed: 0 none, 1 triangle, 2 generated.
            let triangleValue = try writer.unsignedConstantID(committed ? 1 : 0)
            let isTriangle = try writer.rawValueInstruction(170, type: bool, operands: [hitType, triangleValue])
            let hasHit = committed ? try writer.rawValueInstruction(171, type: bool, operands: [hitType, writer.unsignedConstantID(0)])
                : try writer.literalID(.bool(true))
            return try writer.valueWhen(hasHit, type: structure) { writer in
                func get(_ writer: inout SPIRVBinaryWriter, _ opcode: UInt32, _ index: Int) throws -> UInt32 {
                    try writer.rawValueInstruction(opcode, type: memberTypes[index], operands: [variable, one])
                }
                let kind = try writer.rawValueInstruction(169, type: u32, operands: [
                    isTriangle, writer.unsignedConstantID(1), writer.unsignedConstantID(committed ? 2 : 3)])
                let timeKnown = committed ? try writer.literalID(.bool(true)) : isTriangle
                let distance = try writer.valueWhen(timeKnown, type: f32) { try get(&$0, 6018, 1) }
                let barycentrics = try writer.valueWhen(isTriangle, type: memberTypes[7]) { try get(&$0, 6024, 7) }
                let frontFace = try writer.valueWhen(isTriangle, type: bool) { try get(&$0, 6025, 8) }
                // Instance, geometry and primitive ids are the same for a committed hit and a candidate triangle or box.
                var fields: [UInt32] = [kind, distance]
                for (opcode, index) in [(6019, 2), (6020, 3), (6021, 4), (6022, 5), (6023, 6)] {
                    fields.append(try get(&writer, UInt32(opcode), index))
                }
                fields += [barycentrics, frontFace, try get(&writer, 6031, 9), try get(&writer, 6032, 10)]
                return try writer.rawValueInstruction(80, type: structure, operands: fields)
            }
        }
    }
}
