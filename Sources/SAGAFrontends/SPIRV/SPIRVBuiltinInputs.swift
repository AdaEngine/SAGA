import SAGAShaderIR

func spirvForwardedBuiltin(_ number: UInt32) -> IRBuiltIn? {
    switch number {
    case 24: .numWorkgroups
    case 26: .workgroupId
    case 27: .localInvocationId
    case 28: .globalInvocationId
    case 29: .localInvocationIndex
    case 36: .subgroupSize
    case 38: .numSubgroups
    case 40: .subgroupId
    case 41: .subgroupInvocationId
    default: nil
    }
}

extension SPVFunctionLowerer {
    mutating func builtinArgument(
        _ global: IRHandle<IRGlobalVariable>, at site: SPVInstruction
    ) throws -> IRHandle<IRExpression> {
        if let existing = builtinArguments[global] { return existing }
        let type = module.globals[global.index].type
        let index = UInt32(parameters.count)
        try node()
        parameters.append(.init(type: type, location: site.location))
        let argument = try expr(type, .functionArgument(index), site)
        builtinInputs.append(global)
        builtinArguments[global] = argument
        return argument
    }

    mutating func builtinInputPointer(
        _ global: IRHandle<IRGlobalVariable>, rawType: UInt32, at site: SPVInstruction
    ) throws -> SPVValue {
        if let existing = builtinLocals[global] { return existing }
        // Keep the source pointer read-only and thread the immutable value through
        // calls. The local copy preserves the argument's uniformity at each load.
        let type = module.globals[global.index].type
        guard let pointer = module.localPointerTypes[type] else { throw site.fail("Missing builtin input pointer type") }
        let argument = try builtinArgument(global, at: site)
        let local = try IRHandle<IRLocalVariable>(index: locals.count)
        try node()
        locals.append(.init(type: type, location: site.location))
        let expression = try expr(pointer, .localVariable(local), site)
        builtinSetup.append(try stmt(.store(pointer: expression, value: argument), site))
        let value = SPVValue(expression: expression, type: rawType)
        builtinLocals[global] = value
        return value
    }
}
