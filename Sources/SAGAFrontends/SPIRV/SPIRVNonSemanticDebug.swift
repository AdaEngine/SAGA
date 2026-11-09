import SAGAShaderIR

enum SPVNonSemanticDebug {
    static let extensionName = "SPV_KHR_non_semantic_info"
    static let instructionSet = "NonSemantic.Shader.DebugInfo.100"
    static let printfInstructionSet = "NonSemantic.DebugPrintf"
    static let coreVersion: UInt32 = 0x00010600
}

extension SPVDecoder {
    mutating func importNonSemanticDebug(_ instruction: SPVInstruction) throws {
        try requireNonSemanticInstructions(at: instruction)
        try define(instruction.a[0], instruction)
        nonSemanticDebugSets.insert(instruction.a[0])
    }

    func requireNonSemanticInstructions(at instruction: SPVInstruction) throws {
        guard version >= SPVNonSemanticDebug.coreVersion || extensions.contains(SPVNonSemanticDebug.extensionName) else {
            throw instruction.fail("Non-semantic instructions require SPIR-V 1.6 or SPV_KHR_non_semantic_info",
                                   code: .unsupportedCapability)
        }
    }

    mutating func skipNonSemanticDebug(_ instruction: SPVInstruction) throws -> Bool {
        let operands = instruction.a
        guard instruction.op == 12, operands.count >= 3, nonSemanticDebugSets.contains(operands[2]) else { return false }
        try instruction.minimum(4)
        guard voids.contains(operands[0]) else { throw instruction.fail("Shader debug instruction must return OpTypeVoid") }
        // The extension allows only IDs after the instruction number. Ordinary OpExtInst
        // does not allow forward references in NonSemantic.Shader.DebugInfo.100.
        for id in operands.dropFirst(4) {
            try charge()
            guard definitions.contains(id) else { throw instruction.fail("Undefined or forward shader debug operand %\(id)") }
        }
        try define(operands[1], instruction)
        nonSemanticDebugResults.insert(operands[1])
        return true
    }
}
