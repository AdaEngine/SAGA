import SAGAShaderIR

func geometricStatements(
    _ function: IRMathFunction, arguments: [String], operand: IRTypeKind, resultType: String,
    named name: String, format: ShaderOutputFormat
) throws -> [String] {
    let isDouble: Bool = switch operand {
    case .scalar(.f64), .vector(_, .f64), .matrix(_, _, .f64): true
    default: false
    }
    if isDouble {
        guard format == .glsl else { throw ShaderFailure.unsupported("Double-precision geometric builtin target") }
        let scalar = operand == .scalar(.f64)
        let expression = floatingGeometryCall(function, arguments: arguments, scalar: scalar, format: format)
        return ["\(resultType) \(name) = \(expression);"]
    }
    if function == .inverse, case .matrix(let columns, let rows, let scalar) = operand, columns == rows {
        return try inverseMatrixStatements(arguments[0], size: columns, scalar: scalar,
                                           resultType: resultType, named: name, format: format)
    }
    if let shape = operand.concreteFloatShape, shape.scalar == .f16,
       [.length, .distance, .normalize].contains(function) {
        // Squaring large or small half inputs in half precision can lose the norm.
        let wideType = floatType(shape.size?.rawValue ?? 1, in: format)
        let wideArguments = arguments.map { "\(wideType)(\($0))" }
        let wide = floatingGeometryCall(function, arguments: wideArguments, scalar: shape.size == nil, format: format)
        let value = format == .hlsl ? "(\(resultType))(\(wide))" : "\(resultType)(\(wide))"
        let binding = format == .wgsl ? "let \(name)" : "\(resultType) \(name)"
        return ["\(binding) = \(value);"]
    }
    if format == .wgsl {
        return ["let \(name) = \(function.rawValue)(\(arguments.joined(separator: ", ")));"]
    }
    if format == .hlsl && function == .faceForward {
        // HLSL's intrinsic multiplies by -sign(dot(...)), which returns zero
        // at a zero dot product. WGSL selects the negated normal in that case.
        let normal = "\(name)_face_normal"
        let incidence = "\(name)_face_incidence"
        let reference = "\(name)_face_reference"
        return [
            "\(resultType) \(normal) = \(arguments[0]);",
            "\(resultType) \(incidence) = \(arguments[1]);",
            "\(resultType) \(reference) = \(arguments[2]);",
            "\(resultType) \(name) = dot(\(reference), \(incidence)) < 0.0f ? \(normal) : -\(normal);",
        ]
    }
    let expression: String
    var statements: [String] = []
    switch (function, operand) {
    case (.dot, .vector(let size, let scalar)) where scalar.kind == .sint || scalar.kind == .uint:
        let wide = scalar.width == 8
        let type = try shaderVectorType(size.rawValue, scalar: wide ? .u64 : .u32, in: format)
        let left = "\(name)_left", right = "\(name)_right"
        func bits(_ argument: String) throws -> String {
            if scalar.width == 2 {
                return "\(type)(\(try integer16Bits(argument, signed: false, width: size.rawValue, in: format)))"
            }
            guard scalar.kind == .sint else { return argument }
            return wide ? try wideIntegerBits(argument, signed: false, width: size.rawValue, in: format) :
                integerBits(argument, signed: false, width: size.rawValue, in: format)
        }
        statements = ["\(type) \(left) = \(try bits(arguments[0]));", "\(type) \(right) = \(try bits(arguments[1]));"]
        // Unsigned accumulation preserves modular arithmetic for signed dot products.
        let sum = "xyzw".prefix(Int(size.rawValue)).map { "\(left).\($0) * \(right).\($0)" }.joined(separator: " + ")
        switch scalar {
        case .i64: expression = try wideIntegerBits(sum, signed: true, in: format)
        case .i32: expression = integerBits(sum, signed: true, in: format)
        case .i16, .u16: expression = try integer16Bits(sum, signed: scalar == .i16, in: format)
        default: expression = sum
        }
    case (.dot4I8Packed, _), (.dot4U8Packed, _):
        let left = "\(name)_left", right = "\(name)_right"
        statements = ["uint \(left) = \(arguments[0]);", "uint \(right) = \(arguments[1]);"]
        func lane(_ value: String, shift: Int) -> String {
            let field = "((\(value) >> \(shift)u) & 255u)"
            return function == .dot4I8Packed ? integerBits("(\(field) ^ 128u) - 128u", signed: true, in: format) : field
        }
        expression = [0, 8, 16, 24].map { shift in
            "\(lane(left, shift: shift)) * \(lane(right, shift: shift))"
        }.joined(separator: " + ")
    default:
        let scalar: Bool
        if case .scalar = operand { scalar = true } else { scalar = false }
        expression = floatingGeometryCall(function, arguments: arguments, scalar: scalar, format: format)
    }
    statements.append("\(resultType) \(name) = \(expression);")
    return statements
}

private func floatingGeometryCall(
    _ function: IRMathFunction, arguments: [String], scalar: Bool, format: ShaderOutputFormat
) -> String {
    if scalar && format != .wgsl {
        if function == .length { return "abs(\(arguments[0]))" }
        if function == .distance { return "abs((\(arguments[0])) - (\(arguments[1])))" }
    }
    let operation = function == .faceForward && format != .wgsl ? "faceforward" : function.rawValue
    return "\(operation)(\(arguments.joined(separator: ", ")))"
}
