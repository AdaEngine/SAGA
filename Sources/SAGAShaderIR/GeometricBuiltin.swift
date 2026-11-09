extension IRMathFunction {
    package var geometricArity: Int? {
        switch self {
        case .length, .normalize, .transpose, .determinant, .inverse: 1
        case .dot, .dot4I8Packed, .dot4U8Packed, .cross, .distance, .reflect: 2
        case .faceForward, .refract: 3
        default: nil
        }
    }

    package func geometricSignature(for operand: IRTypeKind) -> (arguments: [IRTypeKind], result: IRTypeKind)? {
        guard let arity = geometricArity else { return nil }
        if case .matrix(let columns, let rows, let scalar) = operand {
            guard [.f16, .f32, .f64, .abstractFloat].contains(scalar) else { return nil }
            switch self {
            case .transpose: return ([operand], .matrix(columns: rows, rows: columns, scalar: scalar))
            case .determinant where columns == rows: return ([operand], .scalar(scalar))
            case .inverse where columns == rows: return ([operand], operand)
            default: return nil
            }
        }
        let scalar: IRScalar
        let size: IRVectorSize?
        switch operand {
        case .scalar(let lane): scalar = lane; size = nil
        case .vector(let width, let lane): scalar = lane; size = width
        default: return nil
        }
        let scalarType = IRTypeKind.scalar(scalar)
        let floats = [.f16, .f32, .f64, .abstractFloat].contains(scalar)
        switch self {
        case .dot4I8Packed, .dot4U8Packed:
            guard operand == .scalar(.u32) else { return nil }
            return ([operand, operand], .scalar(self == .dot4I8Packed ? .i32 : .u32))
        case .dot:
            guard size != nil, floats || [.i16, .u16, .i32, .u32, .i64, .u64, .abstractInt].contains(scalar) else { return nil }
            return ([operand, operand], scalarType)
        case .length, .distance:
            guard floats else { return nil }
            return (Array(repeating: operand, count: arity), scalarType)
        case .cross:
            guard floats, size == .three else { return nil }
        case .normalize, .faceForward, .reflect, .refract:
            guard floats, size != nil else { return nil }
        default: return nil
        }
        let inputs = self == .refract ? [operand, operand, scalarType] : Array(repeating: operand, count: arity)
        return (inputs, operand)
    }
}
