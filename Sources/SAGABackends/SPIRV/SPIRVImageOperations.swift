import SAGAShaderIR

extension SPIRVBinaryWriter {
    mutating func writeImageSample(
        _ sample: IRImageSample, type: IRHandle<IRType>, in state: inout SPIRVFunctionState
    ) throws -> UInt32 {
        let kind = try valueKind(sample.image, in: state)
        if kind.isExternalTexture {
            let image = try value(sample.image, in: &state)
            guard let external = state.externalTextures[image], sample.clampToEdge, sample.gather == nil, sample.offset == nil,
                  sample.arrayIndex == nil, sample.depthReference == nil else {
                throw unsupported("does not lower this external texture operation")
            }
            return try externalSample(external, sampler: value(sample.sampler, in: &state), coordinate: value(sample.coordinate, in: &state))
        }
        guard kind.isSampledTexture, !sample.clampToEdge else {
            throw unsupported("does not lower this sampled texture operation")
        }
        let imageType = try state.function.expressions.value(for: sample.image).type
        let sampled = try sampledImageTypeID(image: typeID(imageType))
        let image = try value(sample.image, in: &state)
        let sampler = try value(sample.sampler, in: &state)
        let pair = try imageInstruction(86, type: sampled, operands: [image, sampler]) // OpSampledImage
        if nonUniformResourceIDs.contains(image) || nonUniformResourceIDs.contains(sampler) {
            try markNonUniformResource(pair)
        }
        let coordinate = try packedImageCoordinate(sample.coordinate, layer: sample.arrayIndex, in: &state)
        var operands = [pair, coordinate]
        let comparison = sample.depthReference != nil
        if let reference = sample.depthReference { operands.append(try value(reference, in: &state)) }

        let opcode: UInt32
        var mask: UInt32 = 0
        var extras: [UInt32] = []
        if let component = sample.gather {
            opcode = comparison ? 97 : 96
            if !comparison {
                guard let index = IRSwizzleComponent.allCases.firstIndex(of: component) else { throw ShaderFailure.invalidHandle }
                operands.append(try unsignedConstantID(UInt32(index)))
            }
        } else {
            switch sample.level {
            case .auto:
                opcode = comparison ? 89 : 87
            case .zero:
                opcode = comparison ? 90 : 88
                mask = 2 // Lod
                extras = try [literalID(.f32(0))]
            case .exact(let level):
                opcode = comparison ? 90 : 88
                mask = 2
                let id = try value(level, in: &state)
                if kind.isDepthTexture && !comparison {
                    extras = try [integerImageValueAsFloat(level, id: id, in: state)]
                } else {
                    extras = [id]
                }
            case .bias(let bias):
                opcode = 87
                mask = 1 // Bias
                extras = try [value(bias, in: &state)]
            case .gradient(let x, let y):
                opcode = comparison ? 90 : 88
                mask = 4 // Grad
                extras = try [value(x, in: &state), value(y, in: &state)]
            }
        }
        if let offset = sample.offset {
            mask |= 8 // ConstOffset
            extras.append(try imageOffsetConstant(offset, in: state))
        }
        if mask != 0 { operands += [mask] + extras }
        return try imageTexelInstruction(opcode, type: type, operands: operands,
                                         expandDepth: kind.isDepthTexture && !comparison && sample.gather == nil)
    }

    mutating func writeImageLoad(
        _ load: IRImageLoad, type: IRHandle<IRType>, in state: inout SPIRVFunctionState
    ) throws -> UInt32 {
        let kind = try valueKind(load.image, in: state)
        let image = try value(load.image, in: &state)
        if kind.isExternalTexture {
            guard let external = state.externalTextures[image], load.arrayIndex == nil, load.sample == nil, load.level == nil else {
                throw unsupported("does not lower this external texture operation")
            }
            var pixel = try value(load.coordinate, in: &state)
            if case .vector(_, let scalar) = try valueKind(load.coordinate, in: state), scalar == .i32 {
                pixel = try rawValueInstruction(124, type: vectorTypeID(size: .two, scalar: .u32), operands: [pixel]) // OpBitcast
            }
            return try externalLoad(external, pixel: pixel)
        }
        let storage = kind.storageTexture != nil
        guard storage || kind.isSampledTexture else { throw unsupported("requires a sampled or storage image for texture load") }
        let position = try texelCoordinates(load.coordinate, layer: load.arrayIndex, in: &state)
        let level = try load.level.map { try unsignedTexelOperand($0, in: &state) }
        let sample = try load.sample.map { try unsignedTexelOperand($0, in: &state) }
        if !storage, level == nil, sample == nil {
            throw unsupported("requires a mip or sample operand for sampled texture load")
        }
        var operands = [image, position.id]
        if let sample { operands += [64, sample] } // Sample
        else if let level { operands += [2, level] } // Lod
        let valid = try texelAccessCondition(image: image, position: position, level: level, sample: sample)
        let result = try typeID(type)
        return try valueWhen(valid, type: result) { writer in
            if storage { return try writer.rawValueInstruction(98, type: result, operands: operands) } // OpImageRead
            return try writer.imageTexelInstruction(95, type: type, operands: operands, expandDepth: kind.isDepthTexture)
        }
    }

    mutating func writeImageStore(
        image: IRHandle<IRExpression>, coordinate: IRHandle<IRExpression>, layer: IRHandle<IRExpression>?,
        source: IRHandle<IRExpression>, in state: inout SPIRVFunctionState
    ) throws {
        let imageID = try value(image, in: &state)
        let position = try texelCoordinates(coordinate, layer: layer, in: &state)
        let data = try value(source, in: &state)
        let valid = try texelAccessCondition(image: imageID, position: position, level: nil, sample: nil)
        try effectWhen(valid) { writer in
            try writer.emit(99, [imageID, position.id, data]) // OpImageWrite
        }
    }

    mutating func writeImageQuery(
        image: IRHandle<IRExpression>, query: IRImageQuery, type: IRHandle<IRType>, in state: inout SPIRVFunctionState
    ) throws -> UInt32 {
        try requireCapability(50) // ImageQuery
        let kind = try valueKind(image, in: state)
        let imageID = try value(image, in: &state)
        if kind.isExternalTexture {
            guard let external = state.externalTextures[imageID], case .size(nil) = query else {
                throw unsupported("does not lower this external texture query")
            }
            return try externalDimensions(external)
        }
        switch query {
        case .numLevels:
            return try valueInstruction(106, type: type, operands: [imageID])
        case .numSamples:
            return try valueInstruction(107, type: type, operands: [imageID])
        case .size, .numLayers:
            break
        }
        let mipmapped = kind.isSampledTexture && !kind.isMultisampled2D
        let opcode: UInt32 = mipmapped ? 103 : 104 // OpImageQuerySizeLod / OpImageQuerySize
        var operands = [imageID]
        var validLevel: UInt32?
        if mipmapped {
            if case .size(let level?) = query {
                let mip = try safeQueryMip(image: imageID, level: unsignedTexelOperand(level, in: &state))
                operands.append(mip.level)
                validLevel = mip.valid
            } else { operands.append(try unsignedConstantID(0)) }
        }
        let value: UInt32
        if kind.isArrayedTexture {
            let dimensions = try imageInstruction(opcode, type: vectorTypeID(size: .three, scalar: .u32), operands: operands)
            if query == .numLayers { return try valueInstruction(81, type: type, operands: [dimensions, 2]) }
            value = try valueInstruction(79, type: type, operands: [dimensions, dimensions, 0, 1])
        } else {
            value = try valueInstruction(opcode, type: type, operands: operands)
        }
        if let validLevel {
            return try valueWhen(validLevel, type: typeID(type)) { _ in value }
        }
        return value
    }

    mutating func writeImageAtomic(
        image: IRHandle<IRExpression>, coordinate: IRHandle<IRExpression>, layer: IRHandle<IRExpression>?,
        operation: IRAtomicFunction, source: IRHandle<IRExpression>, in state: inout SPIRVFunctionState
    ) throws {
        guard let storage = try valueKind(image, in: state).storageTexture, storage.access.contains(.atomic),
              [.i32, .u32, .u64].contains(storage.scalar) else {
            throw unsupported("requires an integer storage image for atomics")
        }
        if storage.scalar == .u64 {
            guard operation == .min || operation == .max else {
                throw unsupported("R64Uint image atomics support min and max")
            }
            try requireCapability(12) // Int64Atomics
        }
        let imageID = try value(image, in: &state)
        guard let address = state.storageImageAddresses[imageID] else {
            throw unsupported("requires the descriptor address for an image atomic")
        }
        let position = try texelCoordinates(coordinate, layer: layer, in: &state)
        let data = try value(source, in: &state)
        let valid = try texelAccessCondition(image: imageID, position: position, level: nil, sample: nil)
        let type = try scalarTypeID(storage.scalar)
        let pointerType = try pointerTypeID(storageClass: 11, pointee: type) // Image
        markStorageDeviceScopeAtomic()
        let scope = try unsignedConstantID(1) // Device
        let zero = try unsignedConstantID(0) // Sample zero, relaxed memory semantics
        let opcode = try atomicOpcode(operation, signed: storage.scalar == .i32)
        let nonUniform = nonUniformResourceIDs.contains(imageID)
        try effectWhen(valid) { writer in
            let pointer = try writer.rawValueInstruction(60, type: pointerType,
                                                         operands: [address, position.id, zero]) // OpImageTexelPointer
            if nonUniform { try writer.markNonUniformResource(pointer) }
            _ = try writer.rawValueInstruction(opcode, type: type, operands: [pointer, scope, zero, data])
        }
    }

    private mutating func texelCoordinates(
        _ coordinate: IRHandle<IRExpression>, layer: IRHandle<IRExpression>?, in state: inout SPIRVFunctionState
    ) throws -> (id: UInt32, size: IRVectorSize?) {
        guard let shape = IRBitcastShape(try valueKind(coordinate, in: state)),
              [.i32, .u32].contains(shape.scalar) else { throw unsupported("requires integer texel coordinates") }
        let coordinateID = try packedImageCoordinate(coordinate, layer: layer, in: &state)
        let size: IRVectorSize?
        if layer != nil {
            guard let width = shape.size, let packed = IRVectorSize(rawValue: width.rawValue + 1) else {
                throw unsupported("requires space for a texture array layer")
            }
            size = packed
        } else {
            size = shape.size
        }
        if shape.scalar == .u32 { return (coordinateID, size) }
        let words = try scalarOrVectorTypeID(.u32, size: size)
        return (try rawValueInstruction(124, type: words, operands: [coordinateID]), size)
    }

    private mutating func unsignedTexelOperand(_ operand: IRHandle<IRExpression>, in state: inout SPIRVFunctionState) throws -> UInt32 {
        let id = try value(operand, in: &state)
        guard try valueKind(operand, in: state) == .scalar(.i32) else { return id }
        if let bits = constantWords[id] { return try unsignedConstantID(bits) }
        return try rawValueInstruction(124, type: scalarTypeID(.u32), operands: [id])
    }

    private mutating func safeQueryMip(image: UInt32, level: UInt32) throws -> (level: UInt32, valid: UInt32?) {
        // Mip zero exists for every bound texture. Never query a possibly invalid mip.
        if constantWords[level] == 0 { return (level, nil) }
        let u32 = try scalarTypeID(.u32)
        let count = try rawValueInstruction(106, type: u32, operands: [image]) // OpImageQueryLevels
        let valid = try rawValueInstruction(176, type: scalarTypeID(.bool), operands: [level, count])
        let safe = try rawValueInstruction(169, type: u32, operands: [valid, level, unsignedConstantID(0)])
        return (safe, valid)
    }

    private mutating func texelAccessCondition(
        image: UInt32, position: (id: UInt32, size: IRVectorSize?), level: UInt32?, sample: UInt32?
    ) throws -> UInt32 {
        try requireCapability(50) // ImageQuery
        let bool = try scalarTypeID(.bool)
        let words = try scalarOrVectorTypeID(.u32, size: position.size)
        let size: UInt32
        var validIndex: UInt32?
        if let level {
            let mip = try safeQueryMip(image: image, level: level)
            validIndex = mip.valid
            size = try rawValueInstruction(103, type: words, operands: [image, mip.level]) // OpImageQuerySizeLod
        } else {
            size = try rawValueInstruction(104, type: words, operands: [image]) // OpImageQuerySize
        }
        if let sample {
            let count = try rawValueInstruction(107, type: scalarTypeID(.u32), operands: [image]) // OpImageQuerySamples
            validIndex = try rawValueInstruction(176, type: bool, operands: [sample, count])
        }
        let predicates = try scalarOrVectorTypeID(.bool, size: position.size)
        let components = try rawValueInstruction(176, type: predicates, operands: [position.id, size])
        let validPosition = try position.size == nil ? components : rawValueInstruction(155, type: bool, operands: [components])
        if let validIndex {
            return try rawValueInstruction(167, type: bool, operands: [validIndex, validPosition])
        }
        return validPosition
    }

    private mutating func packedImageCoordinate(
        _ coordinate: IRHandle<IRExpression>, layer: IRHandle<IRExpression>?, in state: inout SPIRVFunctionState
    ) throws -> UInt32 {
        let position = try value(coordinate, in: &state)
        guard let layer else { return position }
        guard case .vector(let width, let scalar) = try valueKind(coordinate, in: state),
              let packedWidth = IRVectorSize(rawValue: width.rawValue + 1),
              case .scalar(let layerScalar) = try valueKind(layer, in: state),
              layerScalar == .u32 || layerScalar == .i32 else {
            throw unsupported("requires a vector coordinate and an integer array layer")
        }
        let layerID = try value(layer, in: &state)
        let converted: UInt32
        if scalar == .f32 {
            converted = try integerImageValueAsFloat(layer, id: layerID, in: state)
        } else if scalar != layerScalar {
            converted = try imageInstruction(124, type: scalarTypeID(scalar), operands: [layerID]) // OpBitcast
        } else {
            converted = layerID
        }
        return try imageInstruction(80, type: vectorTypeID(size: packedWidth, scalar: scalar), operands: [position, converted])
    }

    private mutating func integerImageValueAsFloat(
        _ operand: IRHandle<IRExpression>, id: UInt32, in state: SPIRVFunctionState
    ) throws -> UInt32 {
        switch try state.function.expressions.value(for: operand).kind {
        case .literal(.i32(let value)): return try literalID(.f32(Float(value)))
        case .literal(.u32(let value)): return try literalID(.f32(Float(value)))
        case .zeroValue: return try literalID(.f32(0))
        default: break
        }
        let opcode: UInt32
        switch try valueKind(operand, in: state) {
        case .scalar(.i32): opcode = 111 // OpConvertSToF
        case .scalar(.u32): opcode = 112 // OpConvertUToF
        default: throw unsupported("requires an integer texture index")
        }
        return try imageInstruction(opcode, type: scalarTypeID(.f32), operands: [id])
    }

    private mutating func imageOffsetConstant(
        _ handle: IRHandle<IRExpression>, in state: SPIRVFunctionState
    ) throws -> UInt32 {
        try budget.enterDepth()
        defer { try? budget.leaveDepth() }
        let expression = try state.function.expressions.value(for: handle)
        let elements: [UInt32]
        switch expression.kind {
        case .literal(let literal): return try literalID(literal)
        case .constant(let constant): return try constantID(constant)
        case .zeroValue(let type): return try zeroID(typeID: typeID(type))
        case .compose(_, let operands):
            var components: [UInt32] = []
            for operand in operands {
                guard case .scalar(.i32) = try valueKind(operand, in: state) else {
                    throw unsupported("does not lower vector constituents in a texture offset constant")
                }
                components.append(try imageOffsetConstant(operand, in: state))
            }
            elements = components
        case .splat(let size, let operand):
            elements = try Array(repeating: imageOffsetConstant(operand, in: state), count: Int(size.rawValue))
        default: throw unsupported("requires a constant texture offset")
        }
        let type = try typeID(expression.type)
        let id = try allocateID()
        try declare(44, [type, id] + elements) // OpConstantComposite
        return id
    }

    private mutating func imageInstruction(_ opcode: UInt32, type: UInt32, operands: [UInt32]) throws -> UInt32 {
        let id = try allocateID()
        try emit(opcode, [type, id] + operands)
        return id
    }

    private mutating func imageTexelInstruction(
        _ opcode: UInt32, type: IRHandle<IRType>, operands: [UInt32], expandDepth: Bool
    ) throws -> UInt32 {
        if expandDepth {
            let texel = try imageInstruction(opcode, type: vectorTypeID(size: .four, scalar: .f32), operands: operands)
            return try valueInstruction(81, type: type, operands: [texel, 0]) // OpCompositeExtract
        }
        return try valueInstruction(opcode, type: type, operands: operands)
    }
}
