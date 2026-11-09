import SAGAShaderIR

enum SPVDerivative {
    static let controlCapability: UInt32 = 51
}

extension SPVFunctionLowerer {
    mutating func lowerDerivative(_ i: SPVInstruction, _ statements: inout [IRHandle<IRStatement>]) throws {
        try i.count(3)
        let axis: IRDerivativeAxis
        let control: IRDerivativeControl
        switch i.op {
        case 207: (axis, control) = (.x, .none)
        case 208: (axis, control) = (.y, .none)
        case 209: (axis, control) = (.width, .none)
        case 210: (axis, control) = (.x, .fine)
        case 211: (axis, control) = (.y, .fine)
        case 212: (axis, control) = (.width, .fine)
        case 213: (axis, control) = (.x, .coarse)
        case 214: (axis, control) = (.y, .coarse)
        case 215: (axis, control) = (.width, .coarse)
        default: throw i.fail("Invalid derivative opcode")
        }
        if control != .none, !module.capabilities.contains(SPVDerivative.controlCapability) {
            throw i.fail("Fine and coarse derivatives require DerivativeControl", code: .unsupportedCapability)
        }
        let operand = try value(i.a[2], i, &statements)
        guard operand.type == i.a[0], try module.kind(operand.type, i).concreteFloatShape?.scalar == .f32 else {
            throw i.fail("Derivative operand and result must share a 32-bit floating scalar or vector type")
        }
        try result(i, .derivative(axis: axis, control: control, expression: operand.expression), &statements)
    }
}
