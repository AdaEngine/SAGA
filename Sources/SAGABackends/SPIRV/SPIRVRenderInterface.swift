import SAGAShaderIR

struct SPIRVRenderInterfacePlan {
    fileprivate struct Variable {
        let id: UInt32
        let type: UInt32
        let flipPositionY: Bool
        let sampleMaskArray: UInt32?
        let writesDepth: Bool
    }

    fileprivate enum Argument {
        case input(Int)
        case structure(type: UInt32, inputs: Range<Int>)
    }

    let interfaceIDs: [UInt32]
    fileprivate let inputs: [Variable]
    fileprivate let arguments: [Argument]
    fileprivate let outputs: [Variable]
    fileprivate let returnsStructure: Bool
    var writesDepth: Bool { outputs.contains(where: \.writesDepth) }
}

extension SPIRVBinaryWriter {
    mutating func prepareRenderInterface(
        _ function: IRFunction, stage: IRStage
    ) throws -> SPIRVRenderInterfacePlan {
        guard stage == .vertex || stage == .fragment else {
            throw unsupported("requires a vertex or fragment render stage")
        }
        try charge(function.arguments.count)
        var inputs: [SPIRVRenderInterfacePlan.Variable] = []
        var arguments: [SPIRVRenderInterfacePlan.Argument] = []
        for argument in function.arguments {
            let start = inputs.count
            if case .structure(let members, _) = try module.types.value(for: argument.type).kind {
                guard argument.binding == nil else { throw unsupported("requires bindings on input structure members") }
                try charge(members.count)
                for member in members {
                    inputs.append(try declareRenderVariable(member.type, binding: member.binding,
                                                            name: member.name, stage: stage, input: true))
                }
                arguments.append(.structure(type: try typeID(argument.type), inputs: start..<inputs.count))
            } else {
                inputs.append(try declareRenderVariable(argument.type, binding: argument.binding,
                                                        name: argument.name, stage: stage, input: true))
                arguments.append(.input(start))
            }
        }

        var outputs: [SPIRVRenderInterfacePlan.Variable] = []
        var returnsStructure = false
        if let result = function.result {
            if case .structure(let members, _) = try module.types.value(for: result.type).kind {
                guard result.binding == nil else { throw unsupported("requires bindings on output structure members") }
                try charge(members.count)
                for member in members {
                    outputs.append(try declareRenderVariable(member.type, binding: member.binding,
                                                             name: member.name, stage: stage, input: false))
                }
                returnsStructure = true
            } else {
                outputs.append(try declareRenderVariable(result.type, binding: result.binding,
                                                         name: nil, stage: stage, input: false))
            }
        }
        return .init(interfaceIDs: inputs.map(\.id) + outputs.map(\.id), inputs: inputs, arguments: arguments,
                     outputs: outputs, returnsStructure: returnsStructure)
    }

    mutating func emitRenderInputs(_ plan: SPIRVRenderInterfacePlan) throws -> [UInt32] {
        try charge(plan.inputs.count + plan.arguments.count)
        let values = try plan.inputs.map { input in
            let loaded = try renderInterfaceValue(61, type: input.sampleMaskArray ?? input.type, operands: [input.id]) // OpLoad
            guard input.sampleMaskArray != nil else { return loaded }
            let word = try renderInterfaceValue(81, type: scalarTypeID(.i32), operands: [loaded, 0])
            return try renderInterfaceValue(124, type: input.type, operands: [word]) // OpBitcast
        }
        return try plan.arguments.map { argument in
            switch argument {
            case .input(let index):
                return values[index]
            case .structure(let type, let range):
                return try renderInterfaceValue(80, type: type, operands: Array(values[range])) // OpCompositeConstruct
            }
        }
    }

    mutating func emitRenderOutput(_ value: UInt32, plan: SPIRVRenderInterfacePlan) throws {
        for (index, output) in plan.outputs.enumerated() {
            try charge()
            var field = value
            if plan.returnsStructure {
                field = try renderInterfaceValue(81, type: output.type, operands: [value, UInt32(index)])
            }
            if output.flipPositionY {
                // Convert to SPIR-V clip coordinates, reversing the importer's vertex Y adjustment.
                let scalar = try scalarTypeID(.f32)
                let y = try renderInterfaceValue(81, type: scalar, operands: [field, 1])
                let flipped = try renderInterfaceValue(127, type: scalar, operands: [y]) // OpFNegate
                field = try renderInterfaceValue(82, type: output.type, operands: [flipped, field, 1]) // OpCompositeInsert
            }
            if let array = output.sampleMaskArray {
                let word = try renderInterfaceValue(124, type: scalarTypeID(.i32), operands: [field])
                field = try renderInterfaceValue(80, type: array, operands: [word])
            }
            try emit(62, [output.id, field]) // OpStore
        }
    }

    private mutating func renderInterfaceValue(_ opcode: UInt32, type: UInt32, operands: [UInt32]) throws -> UInt32 {
        let id = try allocateID()
        try emit(opcode, [type, id] + operands)
        return id
    }

    private mutating func declareRenderVariable(
        _ type: IRHandle<IRType>, binding: IRBinding?, name: String?, stage: IRStage, input: Bool
    ) throws -> SPIRVRenderInterfacePlan.Variable {
        guard let binding else { throw unsupported("requires a binding for every render interface field") }
        let kind = try module.types.value(for: type).kind
        let decorations: [[UInt32]]
        var flipPositionY = false
        var sampleMaskArray: UInt32?
        switch binding {
        case .location(let index, .some(.perVertex), nil, nil, false):
            guard stage == .fragment, input,
                  case .array(let element, .constant(3), _) = kind,
                  let shape = IRBitcastShape(try module.types.value(for: element).kind),
                  [.f16, .f32, .i32, .u32].contains(shape.scalar) else {
                throw unsupported("requires an array of three supported per-vertex fragment input values")
            }
            if shape.scalar == .f16 { try requireCapability(4436) } // StorageInputOutput16
            try requireFragmentBarycentrics()
            decorations = [[30, index], [5285]] // Location, PerVertexKHR
        case .location(let index, let interpolation, let sampling, let blendSource, false):
            guard let shape = IRBitcastShape(kind), [.f16, .f32, .i32, .u32].contains(shape.scalar) else {
                throw unsupported("requires a supported numeric scalar or vector render location")
            }
            if shape.scalar == .f16 { try requireCapability(4436) } // StorageInputOutput16
            let varying = stage == .vertex ? !input : input
            var attributes: [[UInt32]] = [[30, index]] // Location
            if let blendSource {
                guard stage == .fragment, !input, index == 0, blendSource <= 1 else {
                    throw unsupported("requires a dual-source fragment output at location zero")
                }
                attributes.append([32, blendSource]) // Index
            }
            if varying {
                switch interpolation ?? .perspective {
                case .perspective:
                    guard shape.scalar.kind == .float else { throw unsupported("requires flat integer varyings") }
                case .linear:
                    guard shape.scalar.kind == .float else { throw unsupported("requires flat integer varyings") }
                    attributes.append([13]) // NoPerspective
                case .flat: attributes.append([14]) // Flat
                case .perVertex: throw unsupported("does not lower per-vertex interpolation")
                }
                if interpolation == .flat {
                    guard sampling == nil || sampling == .first || sampling == .either else {
                        throw unsupported("does not lower this flat interpolation sampling")
                    }
                } else {
                    switch sampling {
                    case nil, .center: break
                    case .centroid: attributes.append([16])
                    case .sample:
                        try requireCapability(35) // SampleRateShading
                        attributes.append([17]) // Sample
                    default: throw unsupported("does not lower this interpolation sampling")
                    }
                }
            } else if interpolation != nil || sampling != nil {
                throw unsupported("requires interpolation only on interstage varyings")
            }
            decorations = attributes
        case .builtIn(.vertexIndex) where stage == .vertex && input && kind == .scalar(.u32):
            decorations = [[11, 42]]
        case .builtIn(.instanceIndex) where stage == .vertex && input && kind == .scalar(.u32):
            decorations = [[11, 43]]
        case .builtIn(.drawIndex) where stage == .vertex && input && kind == .scalar(.u32):
            try requireCapability(4427) // DrawParameters is core in SPIR-V 1.3.
            decorations = [[11, 4426]] // DrawIndex
        case .builtIn(.viewIndex) where input && kind == .scalar(.u32):
            try requireCapability(4439) // MultiView is core in SPIR-V 1.3.
            decorations = stage == .fragment ? [[11, 4440], [14]] : [[11, 4440]] // ViewIndex and Flat
        case .builtIn(.position(let invariant)) where stage == .vertex && !input && kind == .vector(size: .four, scalar: .f32):
            decorations = invariant ? [[11, 0], [18]] : [[11, 0]]
            flipPositionY = true
        case .builtIn(.clipDistances) where stage == .vertex && !input:
            guard case .array(let element, .constant(let count), _) = kind, (1...8).contains(count),
                  try module.types.value(for: element).kind == .scalar(.f32) else {
                throw unsupported("requires one to eight f32 clip distances")
            }
            try requireCapability(32) // ClipDistance
            decorations = [[11, 3]]
        case .builtIn(.position(invariant: false)) where stage == .fragment && input && kind == .vector(size: .four, scalar: .f32):
            decorations = [[11, 15]] // FragCoord
        case .builtIn(.frontFacing) where stage == .fragment && input && kind == .scalar(.bool):
            decorations = [[11, 17]]
        case .builtIn(.primitiveIndex) where stage == .fragment && input && kind == .scalar(.u32):
            try requireCapability(2) // Geometry enables PrimitiveId.
            decorations = [[11, 7], [14]] // PrimitiveId and Flat
        case .builtIn(.barycentric(let perspective)) where stage == .fragment && input &&
            kind == .vector(size: .three, scalar: .f32):
            try requireFragmentBarycentrics()
            decorations = [[11, perspective ? 5286 : 5287]]
        case .builtIn(.sampleIndex) where stage == .fragment && input && kind == .scalar(.u32):
            try requireCapability(35) // SampleRateShading
            decorations = [[11, 18], [14]] // SampleId and Flat
        case .builtIn(.sampleMask) where stage == .fragment && kind == .scalar(.u32):
            decorations = input ? [[11, 20], [14]] : [[11, 20]] // SampleMask
            sampleMaskArray = try sampleMaskArrayTypeID()
        case .builtIn(.fragDepth) where stage == .fragment && !input && kind == .scalar(.f32):
            decorations = [[11, 22]] // FragDepth
        case .builtIn(let builtin) where stage == .fragment && input && kind == .scalar(.u32) &&
            (builtin == .subgroupSize || builtin == .subgroupInvocationId):
            try requireCapability(61) // GroupNonUniform
            decorations = [[11, builtin == .subgroupSize ? 36 : 41], [14]] // BuiltIn and Flat
        default:
            throw unsupported("does not lower this render interface binding")
        }
        let valueType = try typeID(type)
        let storageClass: UInt32 = input ? 1 : 3 // Input or Output
        let pointer = try pointerTypeID(storageClass: storageClass, pointee: sampleMaskArray ?? valueType)
        let id = try allocateID()
        try declare(59, [pointer, id, storageClass])
        for attributes in decorations { try decorate(71, [id] + attributes) }
        try debugName(id, name)
        return .init(id: id, type: valueType, flipPositionY: flipPositionY, sampleMaskArray: sampleMaskArray,
                     writesDepth: binding == .builtIn(.fragDepth))
    }
}
