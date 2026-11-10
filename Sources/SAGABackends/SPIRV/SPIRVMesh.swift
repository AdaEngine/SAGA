import SAGAShaderIR

// Mesh stage lowering for VK_EXT_mesh_shader (SPV_EXT_mesh_shader).
//
// The IR holds the mesh outputs in a workgroup-space structure, exactly like the GLSL writer's `saga_mesh_output`.
// The user body runs against that structure inside a single-iteration loop (so early returns can leave it), and an
// epilogue clamps the counts, calls OpSetMeshOutputsEXT and copies vertices and primitives into the stage outputs.

struct SPIRVMeshOutputVariable {
    /// Index of the member of the vertex or primitive structure this variable receives.
    let member: Int
    let variable: UInt32
    /// SPIR-V type of one element of the array.
    let element: UInt32
    let memberType: IRHandle<IRType>
}

struct SPIRVMeshPlan {
    let compute: SPIRVComputeInputPlan
    let info: IRMeshStageInfo
    let output: IRHandle<IRGlobalVariable>
    let verticesMember: Int
    let primitivesMember: Int
    let vertexCountMember: Int
    let primitiveCountMember: Int
    let vertexType: IRHandle<IRType>
    let primitiveType: IRHandle<IRType>
    let workgroupSize: IRWorkgroupSize
    /// `gl_MeshVerticesEXT`: an array of blocks whose first member is the position.
    let positionMember: Int?
    let meshVertices: UInt32?
    let vertexElement: UInt32?
    let indexVariable: UInt32
    let indexType: UInt32
    let indexMember: Int
    /// Indices per primitive: three for triangles, two for lines, one for points.
    let indexComponents: Int
    let vertexVaryings: [SPIRVMeshOutputVariable]
    let primitiveVaryings: [SPIRVMeshOutputVariable]
    let interfaceIDs: [UInt32]
}

extension SPIRVBinaryWriter {
    mutating func prepareMeshEntry(
        _ entry: IREntryPoint, function: IRFunction, id: UInt32
    ) throws -> SPIRVMeshPlan {
        guard let info = entry.meshInfo, function.result == nil, entry.workgroupOverrides == nil, entry.subgroupSize == nil,
              info.maxVerticesOverride == nil, info.maxPrimitivesOverride == nil else {
            throw unsupported("requires specialized mesh entry points")
        }
        // Topology picks the execution mode, the index builtin and how many indices each primitive has.
        let topology: (mode: UInt32, builtin: UInt32, binding: IRBuiltIn, components: Int)
        switch info.topology {
        case .triangles: topology = (5298, 5296, .triangleIndices, 3)  // OutputTrianglesEXT, PrimitiveTriangleIndicesEXT
        case .lines: topology = (5269, 5295, .lineIndices, 2)  // OutputLinesEXT, PrimitiveLineIndicesEXT
        case .points: topology = (27, 5294, .pointIndex, 1)  // OutputPoints, PrimitivePointIndicesEXT
        }
        let output = try module.globalVariables.value(for: info.outputVariable)
        guard case .structure(let outputMembers, _) = try module.types.value(for: output.type).kind else {
            throw unsupported("requires a mesh output structure")
        }
        func member(_ builtin: IRBuiltIn) throws -> Int {
            guard let index = outputMembers.firstIndex(where: { $0.binding == .builtIn(builtin) }) else {
                throw unsupported("requires the \(builtin) mesh output member")
            }
            return index
        }
        let verticesMember = try member(.vertices), primitivesMember = try member(.primitives)
        let vertexCountMember = try member(.vertexCount), primitiveCountMember = try member(.primitiveCount)
        guard case .structure(let vertexMembers, _) = try module.types.value(for: info.vertexOutputType).kind,
              case .structure(let primitiveMembers, _) = try module.types.value(for: info.primitiveOutputType).kind else {
            throw unsupported("requires mesh vertex and primitive structures")
        }

        requiredMeshCapabilities()
        let size = entry.workgroupSize
        executionModes += try instructionWords(16, [id, 17, size.x, size.y, size.z])  // LocalSize
        executionModes += try instructionWords(16, [id, 26, info.maxVertices])  // OutputVertices
        executionModes += try instructionWords(16, [id, 5270, info.maxPrimitives])  // OutputPrimitivesEXT
        executionModes += try instructionWords(16, [id, topology.mode])

        var workgroup: [IRHandle<IRGlobalVariable>] = []
        let reachable = try ShaderFunctions(module: module, entry: entry.function).reachableGlobals
        try charge(reachable.count)
        for index in reachable.sorted() where module.globalVariables.values[index].space == .workgroup {
            workgroup.append(try IRHandle(index: index))
        }
        if !workgroup.contains(info.outputVariable) { workgroup.append(info.outputVariable) }
        let compute = try prepareComputeInputs(function, workgroup: workgroup)

        func arrayType(_ element: UInt32, count: UInt32) throws -> UInt32 {
            let length = try unsignedConstantID(count)
            let id = try allocateID()
            try declare(28, [id, element, length])  // OpTypeArray
            return id
        }
        func outputVariable(_ pointee: UInt32) throws -> UInt32 {
            let pointer = try pointerTypeID(storageClass: 3, pointee: pointee)
            let id = try allocateID()
            try declare(59, [pointer, id, 3])  // OpVariable Output
            return id
        }
        var interfaceIDs = compute.interfaceIDs
        var positionMember: Int?
        var meshVertices: UInt32?
        var vertexElement: UInt32?
        var vertexVaryings: [SPIRVMeshOutputVariable] = []
        for (index, field) in vertexMembers.enumerated() {
            switch field.binding {
            case .builtIn(.position):
                guard positionMember == nil else { throw unsupported("requires a single mesh position output") }
                let vector = try typeID(field.type)
                let block = try allocateID()
                try declare(30, [block, vector])  // OpTypeStruct
                try decorate(72, [block, 0, 11, 0])  // OpMemberDecorate BuiltIn Position
                try decorate(71, [block, 2])  // Block
                let blocks = try arrayType(block, count: info.maxVertices)
                let variable = try outputVariable(blocks)
                interfaceIDs.append(variable)
                positionMember = index
                meshVertices = variable
                vertexElement = block
            case .location(let location, let interpolation, _, nil, false):
                let element = try typeID(field.type)
                let variable = try outputVariable(arrayType(element, count: info.maxVertices))
                try decorate(71, [variable, 30, location])  // Location
                switch interpolation {
                case .flat: try decorate(71, [variable, 14])  // Flat
                case .linear: try decorate(71, [variable, 13])  // NoPerspective
                default: break
                }
                interfaceIDs.append(variable)
                vertexVaryings.append(.init(member: index, variable: variable, element: element, memberType: field.type))
            default:
                throw unsupported("lowers only position and location mesh vertex outputs")
            }
        }
        var indexVariable: UInt32?
        var indexType: UInt32 = 0
        var indexMember = 0
        var primitiveVaryings: [SPIRVMeshOutputVariable] = []
        for (index, field) in primitiveMembers.enumerated() {
            switch field.binding {
            case .builtIn(let builtin) where builtin == topology.binding:
                guard indexVariable == nil else { throw unsupported("requires a single index output") }
                let element = try typeID(field.type)
                let variable = try outputVariable(arrayType(element, count: info.maxPrimitives))
                try decorate(71, [variable, 11, topology.builtin])  // BuiltIn Primitive*IndicesEXT
                interfaceIDs.append(variable)
                indexVariable = variable
                indexType = element
                indexMember = index
            case .location(let location, nil, nil, nil, _):
                let element = try typeID(field.type)
                let variable = try outputVariable(arrayType(element, count: info.maxPrimitives))
                try decorate(71, [variable, 30, location])
                try decorate(71, [variable, 5271])  // PerPrimitiveEXT
                interfaceIDs.append(variable)
                primitiveVaryings.append(.init(member: index, variable: variable, element: element, memberType: field.type))
            default:
                throw unsupported("lowers only triangle index and location mesh primitive outputs")
            }
        }
        guard let indexVariable else { throw unsupported("requires an index output") }
        return SPIRVMeshPlan(
            compute: compute, info: info, output: info.outputVariable, verticesMember: verticesMember,
            primitivesMember: primitivesMember, vertexCountMember: vertexCountMember,
            primitiveCountMember: primitiveCountMember, vertexType: info.vertexOutputType,
            primitiveType: info.primitiveOutputType, workgroupSize: size, positionMember: positionMember,
            meshVertices: meshVertices, vertexElement: vertexElement, indexVariable: indexVariable, indexType: indexType,
            indexMember: indexMember, indexComponents: topology.components, vertexVaryings: vertexVaryings,
            primitiveVaryings: primitiveVaryings, interfaceIDs: interfaceIDs)
    }

    /// Emits the code that runs after the user body: counts, `OpSetMeshOutputsEXT` and the output copies.
    mutating func writeMeshEpilogue(_ plan: SPIRVMeshPlan, loopVariable: UInt32) throws {
        let u32 = try scalarTypeID(.u32)
        let bool = try scalarTypeID(.bool)
        let zero = try unsignedConstantID(0)
        let one = try unsignedConstantID(1)
        try writeControlBarrier(.workgroup)
        let output = try workgroupGlobal(plan.output)
        let outputType = try module.globalVariables.value(for: plan.output).type
        guard case .structure(let members, _) = try module.types.value(for: outputType).kind else {
            throw unsupported("requires a mesh output structure")
        }
        func loadCount(_ member: Int) throws -> UInt32 {
            let type = try workgroupTypeID(members[member].type)
            let pointer = try workgroupProjection(output, index: unsignedConstantID(UInt32(member)), type: members[member].type)
            return try rawValueInstruction(61, type: type, operands: [pointer])  // OpLoad
        }
        func clamp(_ value: UInt32, to limit: UInt32) throws -> UInt32 {
            let maximum = try unsignedConstantID(limit)
            let below = try rawValueInstruction(176, type: bool, operands: [value, maximum])  // OpULessThan
            return try rawValueInstruction(169, type: u32, operands: [below, value, maximum])  // OpSelect
        }
        let vertexCount = try clamp(loadCount(plan.vertexCountMember), to: plan.info.maxVertices)
        let noVertices = try rawValueInstruction(170, type: bool, operands: [vertexCount, zero])  // OpIEqual
        let primitiveCount = try rawValueInstruction(
            169, type: u32, operands: [noVertices, zero, clamp(loadCount(plan.primitiveCountMember), to: plan.info.maxPrimitives)])
        try emit(5295, [vertexCount, primitiveCount])  // OpSetMeshOutputsEXT

        guard let laneInput = plan.compute.laneVariable else { throw unsupported("requires the local invocation index") }
        let lane = try rawValueInstruction(61, type: u32, operands: [laneInput])
        let size = plan.workgroupSize
        let stride = try unsignedConstantID(size.x * size.y * size.z)

        /// A strided loop `for (i = lane; i < count; i += stride)` around `body(i)`.
        func loop(count: UInt32, _ body: (UInt32) throws -> Void) throws {
            let header = try allocateID(), condition = try allocateID(), bodyLabel = try allocateID()
            let continueLabel = try allocateID(), merge = try allocateID()
            try emit(62, [loopVariable, lane])  // OpStore
            try emit(249, [header])
            try emit(248, [header])
            try emit(246, [merge, continueLabel, 0])  // OpLoopMerge
            try emit(249, [condition])
            try emit(248, [condition])
            let index = try rawValueInstruction(61, type: u32, operands: [loopVariable])
            let more = try rawValueInstruction(176, type: bool, operands: [index, count])
            try emit(250, [more, bodyLabel, merge])  // OpBranchConditional
            try emit(248, [bodyLabel])
            try body(index)
            try emit(249, [continueLabel])
            try emit(248, [continueLabel])
            let current = try rawValueInstruction(61, type: u32, operands: [loopVariable])
            let next = try rawValueInstruction(128, type: u32, operands: [current, stride])  // OpIAdd
            try emit(62, [loopVariable, next])
            try emit(249, [header])
            try emit(248, [merge])
        }
        func element(_ variable: UInt32, _ index: UInt32, type: UInt32) throws -> UInt32 {
            let pointer = try pointerTypeID(storageClass: 3, pointee: type)
            return try rawValueInstruction(65, type: pointer, operands: [variable, index])  // OpAccessChain
        }
        func source(arrayMember: Int, elementType: IRHandle<IRType>, index: UInt32, field: Int,
                    fieldType: IRHandle<IRType>) throws -> UInt32 {
            let arrayType = members[arrayMember].type
            let array = try workgroupProjection(output, index: unsignedConstantID(UInt32(arrayMember)), type: arrayType)
            let item = try workgroupProjection(array, index: index, type: elementType)
            let value = try workgroupProjection(item, index: unsignedConstantID(UInt32(field)), type: fieldType)
            return try rawValueInstruction(61, type: workgroupTypeID(fieldType), operands: [value])
        }
        try loop(count: vertexCount) { index in
            if let position = plan.positionMember, let meshVertices = plan.meshVertices, let block = plan.vertexElement {
                let fieldType = try vertexFieldType(plan, position)
                let value = try source(arrayMember: plan.verticesMember, elementType: plan.vertexType, index: index,
                                       field: position, fieldType: fieldType)
                let vector = try typeID(fieldType)
                let pointer = try pointerTypeID(storageClass: 3, pointee: vector)
                let target = try rawValueInstruction(65, type: pointer, operands: [meshVertices, index, zero])
                _ = block
                // Vulkan clip space has +Y down; negate Y like the vertex stage so output matches WebGPU.
                let scalar = try scalarTypeID(.f32)
                let y = try rawValueInstruction(81, type: scalar, operands: [value, 1])  // OpCompositeExtract
                let flipped = try rawValueInstruction(127, type: scalar, operands: [y])  // OpFNegate
                let adjusted = try rawValueInstruction(82, type: vector, operands: [flipped, value, 1])  // OpCompositeInsert
                try emit(62, [target, adjusted])
            }
            for varying in plan.vertexVaryings {
                let value = try source(arrayMember: plan.verticesMember, elementType: plan.vertexType, index: index,
                                       field: varying.member, fieldType: varying.memberType)
                try emit(62, [element(varying.variable, index, type: varying.element), value])
            }
        }
        let lastVertex = try rawValueInstruction(130, type: u32, operands: [vertexCount, one])  // OpISub
        try loop(count: primitiveCount) { index in
            let triangle = try source(arrayMember: plan.primitivesMember, elementType: plan.primitiveType, index: index,
                                      field: plan.indexMember, fieldType: try primitiveFieldType(plan, plan.indexMember))
            // Keep indices inside the vertices that were actually produced.
            func clampIndex(_ value: UInt32) throws -> UInt32 {
                let inside = try rawValueInstruction(176, type: bool, operands: [value, lastVertex])
                return try rawValueInstruction(169, type: u32, operands: [inside, value, lastVertex])
            }
            let rebuilt: UInt32
            if plan.indexComponents == 1 {
                rebuilt = try clampIndex(triangle)
            } else {
                var clamped: [UInt32] = []
                for component in 0..<plan.indexComponents {
                    let value = try rawValueInstruction(81, type: u32, operands: [triangle, UInt32(component)])  // OpCompositeExtract
                    clamped.append(try clampIndex(value))
                }
                rebuilt = try rawValueInstruction(80, type: plan.indexType, operands: clamped)  // OpCompositeConstruct
            }
            try emit(62, [element(plan.indexVariable, index, type: plan.indexType), rebuilt])
            for varying in plan.primitiveVaryings {
                let value = try source(arrayMember: plan.primitivesMember, elementType: plan.primitiveType, index: index,
                                       field: varying.member, fieldType: varying.memberType)
                try emit(62, [element(varying.variable, index, type: varying.element), value])
            }
        }
    }

    private func vertexFieldType(_ plan: SPIRVMeshPlan, _ member: Int) throws -> IRHandle<IRType> {
        guard case .structure(let members, _) = try module.types.value(for: plan.vertexType).kind else {
            throw unsupported("requires a mesh vertex structure")
        }
        return members[member].type
    }

    private func primitiveFieldType(_ plan: SPIRVMeshPlan, _ member: Int) throws -> IRHandle<IRType> {
        guard case .structure(let members, _) = try module.types.value(for: plan.primitiveType).kind else {
            throw unsupported("requires a mesh primitive structure")
        }
        return members[member].type
    }

    mutating func requiredMeshCapabilities() {
        requiresSpirv14 = true
        capabilities.insert(5283)  // MeshShadingEXT
        extensions.insert("SPV_EXT_mesh_shader")
    }
}

extension SPIRVBinaryWriter {
    mutating func booleanFalseID() throws -> UInt32 { try literalID(.bool(false)) }
}
