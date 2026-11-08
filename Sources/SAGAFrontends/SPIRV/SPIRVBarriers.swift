import SAGAShaderIR

extension SPVFunctionLowerer {
    mutating func lowerBlockInstructions(
        _ instructions: [SPVInstruction], into statements: inout [IRHandle<IRStatement>], block: UInt32
    ) throws {
        var index = 0
        while index < instructions.count {
            let instruction = instructions[index]
            if instruction.op == 225 {
                try work(2)
                try instruction.count(2)
                guard index + 1 < instructions.count, instructions[index + 1].op == 224 else {
                    throw instruction.fail("Standalone SPIR-V memory fences are not implemented", code: .unsupportedConstruct)
                }
                // The workgroup rendezvous immediately follows the fence. Keep the memory
                // kinds of both instructions without adding a new execution rendezvous.
                let fence = try workgroupMemoryFlags(instruction.a[0], semantics: instruction.a[1], at: instruction)
                let control = instructions[index + 1]
                let flags = try controlBarrierFlags(control)
                guard !flags.contains(.subgroup) else {
                    throw instruction.fail("A workgroup memory fence requires a workgroup rendezvous", code: .unsupportedConstruct)
                }
                statements.append(try stmt(.controlBarrier(fence.union(flags)), control))
                index += 2
            } else {
                try operation(instruction, &statements, blockID: block)
                index += 1
            }
        }
    }

    func controlBarrierFlags(_ instruction: SPVInstruction) throws -> IRBarrier {
        try instruction.count(3)
        let execution = module.constantMap[instruction.a[0]]?.nonnegativeInteger
        if execution == 2 {
            return try workgroupMemoryFlags(instruction.a[1], semantics: instruction.a[2], at: instruction)
        }
        guard execution == 3, module.constantMap[instruction.a[1]]?.nonnegativeInteger == 3,
              let bits = module.constantMap[instruction.a[2]]?.nonnegativeInteger,
              bits & 0x8 != 0, bits & 0x9c0 != 0, bits & ~UInt32(0xdc8) == 0 else {
            throw instruction.fail("Unsupported barrier execution scope or subgroup memory semantics", code: .unsupportedConstruct)
        }
        // GLSL subgroupBarrier includes AtomicCounterMemory. The frontend has no atomic-counter address space.
        return memoryFlags(bits).union(.subgroup)
    }

    private func workgroupMemoryFlags(_ scope: UInt32, semantics: UInt32, at instruction: SPVInstruction) throws -> IRBarrier {
        guard module.constantMap[scope]?.nonnegativeInteger == 2,
              let bits = module.constantMap[semantics]?.nonnegativeInteger,
              bits & 0x8 != 0, bits & ~UInt32(0x948) == 0 else {
            throw instruction.fail("Unsupported workgroup acquire-release memory semantics", code: .unsupportedConstruct)
        }
        let flags = memoryFlags(bits)
        guard !flags.isEmpty else {
            throw instruction.fail("SPIR-V barrier needs a supported memory kind", code: .unsupportedConstruct)
        }
        return flags
    }

    private func memoryFlags(_ bits: UInt32) -> IRBarrier {
        var flags: IRBarrier = []
        if bits & 0x40 != 0 { flags.insert(.storage) }
        if bits & 0x100 != 0 { flags.insert(.workgroup) }
        if bits & 0x800 != 0 { flags.insert(.texture) }
        return flags
    }
}
