import SAGAShaderIR

// Task stage lowering for VK_EXT_mesh_shader (SPV_EXT_mesh_shader).
//
// The IR entry returns the mesh dispatch size as a `vec3<u32>` and writes the task payload, a `TaskPayloadWorkgroupEXT`
// variable the mesh stage reads. The body runs once inside a loop (so `return value` can leave it), then the size is
// clamped to the portable Vulkan minimum limits and emitted with OpEmitMeshTasksEXT.

struct SPIRVTaskPlan {
    let compute: SPIRVComputeInputPlan
    let payload: IRHandle<IRGlobalVariable>
    let vectorType: UInt32
    var interfaceIDs: [UInt32] { compute.interfaceIDs }
}

extension SPIRVBinaryWriter {
    /// Dispatch limits every Vulkan device with mesh shaders guarantees: per dimension and total workgroup count.
    private static var taskDispatchLimits: (perDimension: UInt32, total: UInt32) { (65_535, 4_194_304) }

    mutating func prepareTaskEntry(_ entry: IREntryPoint, function: IRFunction, id: UInt32) throws -> SPIRVTaskPlan {
        guard let payload = entry.taskPayload, function.result != nil, entry.workgroupOverrides == nil,
              entry.subgroupSize == nil else {
            throw unsupported("requires task entry points that return a dispatch size")
        }
        requiredMeshCapabilities()
        let size = entry.workgroupSize
        executionModes += try instructionWords(16, [id, 17, size.x, size.y, size.z])  // LocalSize
        var workgroup: [IRHandle<IRGlobalVariable>] = []
        let reachable = try ShaderFunctions(module: module, entry: entry.function).reachableGlobals
        try charge(reachable.count)
        for index in reachable.sorted() where module.globalVariables.values[index].space == .workgroup {
            workgroup.append(try IRHandle(index: index))
        }
        // The first invocation zero-initializes the payload, so the plan loads the local invocation index.
        workgroup.append(payload)
        let compute = try prepareComputeInputs(function, workgroup: workgroup)
        return SPIRVTaskPlan(compute: compute, payload: payload, vectorType: try vectorTypeID(size: .three, scalar: .u32))
    }

    /// Emits the dispatch-size clamp and OpEmitMeshTasksEXT after the body; `resultVariable` holds the returned vector.
    mutating func writeTaskEpilogue(_ plan: SPIRVTaskPlan, resultVariable: UInt32) throws {
        let u32 = try scalarTypeID(.u32)
        let bool = try scalarTypeID(.bool)
        let one = try unsignedConstantID(1)
        let limits = Self.taskDispatchLimits
        let maximum = try unsignedConstantID(limits.perDimension)
        let total = try unsignedConstantID(limits.total)
        let size = try rawValueInstruction(61, type: plan.vectorType, operands: [resultVariable])  // OpLoad
        var parts: [UInt32] = []
        for index in 0..<3 { parts.append(try rawValueInstruction(81, type: u32, operands: [size, UInt32(index)])) }  // OpCompositeExtract
        // A dispatch beyond the limits is dropped instead of reaching the driver.
        var invalid = try rawValueInstruction(172, type: bool, operands: [parts[0], maximum])  // OpUGreaterThan
        for part in parts.dropFirst() {
            let over = try rawValueInstruction(172, type: bool, operands: [part, maximum])
            invalid = try rawValueInstruction(166, type: bool, operands: [invalid, over])  // OpLogicalOr
        }
        // x * y * z <= total, checked by division so the product cannot wrap: a <= total / b  iff  a * b <= total.
        func divisor(_ value: UInt32) throws -> UInt32 {
            let isZero = try rawValueInstruction(176, type: bool, operands: [value, one])  // OpULessThan
            return try rawValueInstruction(169, type: u32, operands: [isZero, one, value])  // OpSelect
        }
        let xyLimit = try rawValueInstruction(134, type: u32, operands: [total, divisor(parts[1])])  // OpUDiv
        let xyOver = try rawValueInstruction(172, type: bool, operands: [parts[0], xyLimit])
        invalid = try rawValueInstruction(166, type: bool, operands: [invalid, xyOver])
        let xy = try rawValueInstruction(132, type: u32, operands: [parts[0], parts[1]])  // OpIMul
        let zLimit = try rawValueInstruction(134, type: u32, operands: [total, divisor(parts[2])])
        let xyzOver = try rawValueInstruction(172, type: bool, operands: [xy, zLimit])
        invalid = try rawValueInstruction(166, type: bool, operands: [invalid, xyzOver])
        let zero = try unsignedConstantID(0)
        let counts = try parts.map { try rawValueInstruction(169, type: u32, operands: [invalid, zero, $0]) }
        let payload = try workgroupGlobal(plan.payload)
        try emit(5294, counts + [payload])  // OpEmitMeshTasksEXT, a block terminator
    }
}
