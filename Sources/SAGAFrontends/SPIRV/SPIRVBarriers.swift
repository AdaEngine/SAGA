import SAGAShaderIR

private enum SPVBarrierScope: UInt32 {
    case device = 1
    case workgroup = 2
    case subgroup = 3
}

private enum SPVBarrierSemantics {
    static let acquireRelease: UInt32 = 0x8
    static let uniformMemory: UInt32 = 0x40
    static let subgroupMemory: UInt32 = 0x80
    static let workgroupMemory: UInt32 = 0x100
    static let atomicCounterMemory: UInt32 = 0x400
    static let imageMemory: UInt32 = 0x800
    static let memoryKinds = uniformMemory | workgroupMemory | imageMemory
}

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
                guard !module.usesVulkanMemoryModel,
                      index + 1 < instructions.count, instructions[index + 1].op == 224 else {
                    throw instruction.fail("Standalone SPIR-V memory fences are not implemented", code: .unsupportedConstruct)
                }
                // The adjacent legacy fence has no intervening memory operation.
                // Retain both memory masks in the existing execution rendezvous.
                let control = instructions[index + 1]
                let flags = try controlBarrierFlags(control)
                let execution: SPVBarrierScope = flags.contains(.subgroup) ? .subgroup : .workgroup
                let fence = try barrierMemoryFlags(
                    instruction.a[0], semantics: instruction.a[1], execution: execution, at: instruction)
                if execution == .subgroup, !module.usesSimpleMemoryModel || fence != .subgroup {
                    throw instruction.fail("Unsupported subgroup memory fence", code: .unsupportedConstruct)
                }
                statements.append(try stmt(.controlBarrier(fence.union(flags)), control))
                index += 2
            } else {
                try operation(instruction, &statements, blockID: block)
                index += 1
            }
        }
    }

    mutating func controlBarrierFlags(_ instruction: SPVInstruction) throws -> IRBarrier {
        try instruction.count(3)
        let scope = try barrierConstant(instruction.a[0], at: instruction)
        try requireVulkanDeviceScopeCapability(scope, at: instruction)
        guard let execution = SPVBarrierScope(rawValue: scope), execution == .workgroup || execution == .subgroup,
              execution != .subgroup || !module.usesVulkanMemoryModel else {
            throw instruction.fail("Unsupported barrier execution scope", code: .unsupportedConstruct)
        }
        return try barrierMemoryFlags(
            instruction.a[1], semantics: instruction.a[2], execution: execution, at: instruction)
    }

    private mutating func barrierMemoryFlags(
        _ scopeID: UInt32, semantics: UInt32, execution: SPVBarrierScope, at instruction: SPVInstruction
    ) throws -> IRBarrier {
        let scope = try barrierConstant(scopeID, at: instruction)
        try requireVulkanDeviceScopeCapability(scope, at: instruction)
        let bits = try barrierConstant(semantics, at: instruction)
        let acquireRelease = SPVBarrierSemantics.acquireRelease
        var allowed = acquireRelease | SPVBarrierSemantics.memoryKinds
        if execution == .subgroup {
            // Legacy GLSL subgroupBarrier can include AtomicCounterMemory. Atomic counters have no IR address space.
            allowed |= SPVBarrierSemantics.subgroupMemory | SPVBarrierSemantics.atomicCounterMemory
        }
        guard bits & acquireRelease != 0, bits & ~allowed == 0 else {
            throw instruction.fail("Unsupported acquire-release barrier memory semantics", code: .unsupportedConstruct)
        }

        var flags: IRBarrier = []
        if bits & SPVBarrierSemantics.uniformMemory != 0 { flags.insert(.storage) }
        if bits & SPVBarrierSemantics.workgroupMemory != 0 { flags.insert(.workgroup) }
        if bits & SPVBarrierSemantics.imageMemory != 0 { flags.insert(.texture) }
        if execution == .subgroup, !flags.isEmpty || bits & SPVBarrierSemantics.subgroupMemory != 0 {
            flags.insert(.subgroup)
        }
        guard !flags.isEmpty else {
            throw instruction.fail("SPIR-V barrier needs a supported memory kind", code: .unsupportedConstruct)
        }

        if scope == execution.rawValue { return flags }
        if module.usesSimpleMemoryModel, execution == .workgroup, scope == SPVBarrierScope.device.rawValue {
            // Workgroup memory cannot be shared across workgroups. Device scope for the other
            // selected kinds is only unobservable if those resources are absent.
            try work(module.globals.count)
            let observable = module.globals.contains { global in
                switch global.space {
                case .uniform, .storage: flags.contains(.storage)
                case .handle: flags.contains(.texture)
                default: false
                }
            }
            if !observable { return flags }
        }
        throw instruction.fail("Unsupported barrier memory scope", code: .unsupportedConstruct)
    }

    private func barrierConstant(_ id: UInt32, at instruction: SPVInstruction) throws -> UInt32 {
        guard let value = module.constantMap[id]?.nonnegativeInteger else {
            throw instruction.fail("Barrier operand must be a nonnegative 32-bit integer constant")
        }
        return value
    }

    private func requireVulkanDeviceScopeCapability(_ scope: UInt32, at instruction: SPVInstruction) throws {
        if module.usesVulkanMemoryModel, scope == SPVBarrierScope.device.rawValue,
           !module.capabilities.contains(SPVVulkanMemoryModel.deviceScopeCapability) {
            throw instruction.fail("Device barrier scope needs VulkanMemoryModelDeviceScope", code: .unsupportedCapability)
        }
    }
}
