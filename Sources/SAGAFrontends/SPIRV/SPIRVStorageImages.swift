import SAGAShaderIR

enum SPVImageQuery {
    static let capability: UInt32 = 50
}

extension SPVFunctionLowerer {
    mutating func lowerStorageImageAccess(_ instruction: SPVInstruction, _ statements: inout [IRHandle<IRStatement>]) throws {
        let read = instruction.op == 98
        let required = read ? 4 : 3
        let operands = instruction.a
        try instruction.minimum(required)
        let mask = try imageOperandsWithoutMemoryHints(instruction, at: required)
        guard operands.count <= required + 1 && mask == 0 else {
            throw instruction.fail("Storage image operands are not implemented", code: .unsupportedConstruct)
        }
        let image = try imageObject(operands[read ? 2 : 0], at: instruction, &statements)
        guard let texture = image.kind.storageTexture, texture.access.contains(read ? .load : .store) else {
            throw instruction.fail("Storage image operation violates its access decoration", code: .typeMismatch)
        }
        let coordinate = try value(operands[read ? 3 : 1], instruction, &statements)
        let (position, layer) = try imageCoordinates(coordinate, image: image.kind, at: instruction, &statements)
        let fullKind = IRTypeKind.vector(size: .four, scalar: texture.scalar)
        if read {
            let load = IRExpressionKind.imageLoad(.init(image: image.expression, coordinate: position, arrayIndex: layer))
            let requested = try module.kind(operands[0], instruction)
            guard imageComponentCount(requested, scalar: texture.scalar) != nil else {
                throw instruction.fail("Image read result must match the image scalar", code: .typeMismatch)
            }
            if requested == fullKind {
                try result(instruction, load, &statements)
            } else {
                let texel = try emitValue(internLoweredType(fullKind), load, &statements, instruction)
                let extracted: IRExpressionKind = switch requested {
                case .vector(let width, _): .swizzle(size: width, vector: texel, pattern: .init(x: .x, y: .y, z: .z, w: .w))
                default: .accessIndex(base: texel, index: 0)
                }
                try result(instruction, extracted, &statements)
            }
        } else {
            let value = try value(operands[2], instruction, &statements)
            let kind = try module.kind(value.type, instruction)
            guard let count = imageComponentCount(kind, scalar: texture.scalar), count >= texture.components else {
                throw instruction.fail("Image write texel must match the format's scalar and component count", code: .typeMismatch)
            }
            let texel: IRHandle<IRExpression>
            if count == 4 {
                texel = value.expression
            } else {
                let padded: IRExpressionKind
                if count == 1 {
                    padded = .splat(size: .four, value: value.expression)
                } else {
                    let last: IRSwizzleComponent = count == 2 ? .y : .z
                    padded = .swizzle(size: .four, vector: value.expression, pattern: .init(x: .x, y: .y, z: last, w: last))
                }
                texel = try emitValue(internLoweredType(fullKind), padded, &statements, instruction)
            }
            statements.append(try stmt(.imageStore(image: image.expression, coordinate: position, arrayIndex: layer, value: texel), instruction))
        }
    }

    mutating func lowerImageQuery(_ instruction: SPVInstruction, _ statements: inout [IRHandle<IRStatement>]) throws {
        let opcode = instruction.op
        try instruction.count(opcode == 103 ? 4 : 3)
        guard module.capabilities.contains(SPVImageQuery.capability) else {
            throw instruction.fail("Image query requires ImageQuery", code: .unsupportedCapability)
        }
        let image = try imageObject(instruction.a[2], at: instruction, &statements)
        let mipmapped = image.kind.isSampledTexture && !image.kind.isMultisampled2D
        let query: IRImageQuery
        switch opcode {
        case 103 where mipmapped: query = .size(level: try imageIndexOperand(instruction.a[3], at: instruction, &statements))
        case 104 where !mipmapped: query = .size(level: nil)
        case 106 where mipmapped: query = .numLevels
        case 107 where image.kind.isMultisampled2D: query = .numSamples
        default: throw instruction.fail("Image query does not match the image class", code: .typeMismatch)
        }
        guard case .size = query else {
            try imageQueryResult(.imageQuery(image: image.expression, query: query), unsigned: .scalar(.u32),
                                 signed: .scalar(.i32), at: instruction, &statements)
            return
        }
        let unsigned: IRTypeKind = image.kind.isArrayedTexture ? .vector(size: .three, scalar: .u32) : image.kind.texelCoordinateType(.u32)
        let signed: IRTypeKind = image.kind.isArrayedTexture ? .vector(size: .three, scalar: .i32) : image.kind.texelCoordinateType(.i32)
        let size = IRExpressionKind.imageQuery(image: image.expression, query: query)
        if image.kind.isArrayedTexture {
            let dimensions = try emitValue(internLoweredType(image.kind.texelCoordinateType(.u32)), size, &statements, instruction)
            let layers = try emitValue(internLoweredType(.scalar(.u32)), .imageQuery(image: image.expression, query: .numLayers),
                                       &statements, instruction)
            let composed = IRExpressionKind.compose(type: try internLoweredType(unsigned), components: [dimensions, layers])
            try imageQueryResult(composed, unsigned: unsigned, signed: signed, at: instruction, &statements)
        } else {
            try imageQueryResult(size, unsigned: unsigned, signed: signed, at: instruction, &statements)
        }
    }

    private mutating func imageQueryResult(
        _ kind: IRExpressionKind, unsigned: IRTypeKind, signed: IRTypeKind,
        at instruction: SPVInstruction, _ statements: inout [IRHandle<IRStatement>]
    ) throws {
        let requested = try module.kind(instruction.a[0], instruction)
        guard requested == unsigned || requested == signed else {
            throw instruction.fail("Image query result must match the expected integer shape", code: .typeMismatch)
        }
        if requested == signed {
            let dimensions = try emitValue(internLoweredType(unsigned), kind, &statements, instruction)
            try result(instruction, .bitcast(expression: dimensions, type: module.type(instruction.a[0], instruction)), &statements)
        } else {
            try result(instruction, kind, &statements)
        }
    }

    mutating func imageIndexOperand(
        _ id: UInt32, at instruction: SPVInstruction, _ statements: inout [IRHandle<IRStatement>]
    ) throws -> IRHandle<IRExpression> {
        let index = try value(id, instruction, &statements)
        guard try [.scalar(.i32), .scalar(.u32)].contains(module.kind(index.type, instruction)) else {
            throw instruction.fail("Image mip or sample index must be a 32-bit integer", code: .typeMismatch)
        }
        return index.expression
    }

    mutating func imageObject(
        _ id: UInt32, at instruction: SPVInstruction, _ statements: inout [IRHandle<IRStatement>]
    ) throws -> (expression: IRHandle<IRExpression>, kind: IRTypeKind, anchor: IRHandle<IRStatement>?) {
        let image = try value(id, instruction, &statements)
        let kind = module.types[expressions[image.expression.index].type.index].kind
        guard module.pointers[image.type] == nil, kind.storageTexture != nil || kind.isSampledTexture else {
            throw instruction.fail("Expected an image object", code: .typeMismatch)
        }
        return (image.expression, kind, image.anchor)
    }

    mutating func imageCoordinates(
        _ coordinate: SPVValue, image: IRTypeKind, normalized: Bool = false,
        at instruction: SPVInstruction, _ statements: inout [IRHandle<IRStatement>]
    ) throws -> (IRHandle<IRExpression>, IRHandle<IRExpression>?) {
        let kind = try module.kind(coordinate.type, instruction)
        let scalar: IRScalar
        switch kind {
        case .scalar(let lane), .vector(_, let lane): scalar = lane
        default: throw instruction.fail("Image coordinates must be a scalar or vector", code: .typeMismatch)
        }
        let spatialWidth: IRVectorSize = image.isCubeTexture || image.isVolumeTexture ? .three : .two
        let dimensions = image.isLineTexture ? 1 : Int(spatialWidth.rawValue)
        let permitted: [IRScalar] = normalized ? [.f32] : [.i32, .u32]
        guard permitted.contains(scalar), let count = imageComponentCount(kind, scalar: scalar),
              count >= dimensions + (image.isArrayedTexture ? 1 : 0) else {
            throw instruction.fail("Image coordinate does not match the image dimensions", code: .typeMismatch)
        }
        let expected: IRTypeKind = image.isLineTexture ? .scalar(scalar) : .vector(size: spatialWidth, scalar: scalar)
        let position: IRHandle<IRExpression>
        if kind == expected {
            position = coordinate.expression
        } else {
            let prefix: IRExpressionKind = switch expected {
            case .vector(let size, _): .swizzle(size: size, vector: coordinate.expression, pattern: .init(x: .x, y: .y, z: .z, w: .w))
            default: .accessIndex(base: coordinate.expression, index: 0)
            }
            position = try emitValue(internLoweredType(expected), prefix, &statements, instruction)
        }
        guard image.isArrayedTexture else { return (position, nil) }
        let scalarType = try internLoweredType(.scalar(scalar))
        var layer = try emitValue(scalarType, .accessIndex(base: coordinate.expression, index: UInt32(dimensions)),
                                  &statements, instruction)
        if normalized {
            // Vulkan floating array coordinates are rounded to nearest-even before clamping to the view.
            let rounded = try emitValue(scalarType, .math(function: .round, argument: layer, argument1: nil, argument2: nil, argument3: nil),
                                        &statements, instruction)
            layer = try emitValue(internLoweredType(.scalar(.i32)), .cast(expression: rounded, target: .i32), &statements, instruction)
        }
        return (position, layer)
    }

    private func imageComponentCount(_ kind: IRTypeKind, scalar: IRScalar) -> Int? {
        switch kind {
        case .scalar(let lane) where lane == scalar: 1
        case .vector(let size, let lane) where lane == scalar: Int(size.rawValue)
        default: nil
        }
    }
}
