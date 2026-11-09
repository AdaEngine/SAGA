import SAGAShaderIR

extension SPVFunctionLowerer {
    mutating func projectedImageOperands(
        _ coordinate: SPVValue, position: IRHandle<IRExpression>, depthReference: IRHandle<IRExpression>?,
        image: IRTypeKind, at instruction: SPVInstruction, _ statements: inout [IRHandle<IRStatement>]
    ) throws -> (position: IRHandle<IRExpression>, depthReference: IRHandle<IRExpression>?) {
        guard !image.isArrayedTexture, !image.isCubeTexture, !image.isMultisampled2D else {
            throw instruction.fail("Projective sampling requires a non-arrayed 1D, 2D or 3D image", code: .typeMismatch)
        }
        let positionType = expressions[position.index].type
        let spatialWidth: IRVectorSize?
        switch module.types[positionType.index].kind {
        case .scalar(.f32): spatialWidth = nil
        case .vector(let width, .f32): spatialWidth = width
        default: throw instruction.fail("Projective sampling requires f32 coordinates", code: .typeMismatch)
        }
        // SPIR-V places q immediately after the spatial lanes, before any unused lanes.
        let qIndex = spatialWidth?.rawValue ?? 1
        guard case .vector(let inputWidth, .f32) = try module.kind(coordinate.type, instruction),
              inputWidth.rawValue > qIndex else {
            throw instruction.fail("Projective image coordinates must include the q divisor", code: .typeMismatch)
        }
        let scalar = try internLoweredType(.scalar(.f32))
        let q = try emitValue(scalar, .accessIndex(base: coordinate.expression, index: UInt32(qIndex)),
                              &statements, instruction)
        let divisor: IRHandle<IRExpression>
        if let spatialWidth {
            divisor = try emitValue(positionType, .splat(size: spatialWidth, value: q), &statements, instruction)
        } else {
            divisor = q
        }
        let projected = try emitValue(positionType, .binary(op: .divide, left: position, right: divisor),
                                      &statements, instruction)
        let reference: IRHandle<IRExpression>?
        if let depthReference {
            reference = try emitValue(scalar, .binary(op: .divide, left: depthReference, right: q), &statements, instruction)
        } else {
            reference = nil
        }
        return (projected, reference)
    }
}
