import SAGAShaderIR

enum SPVImageSampleOperation: UInt32 {
    case implicit = 87, explicit = 88
    case compareImplicit = 89, compareExplicit = 90
    case projectImplicit = 91, projectExplicit = 92
    case projectCompareImplicit = 93, projectCompareExplicit = 94
    case gather = 96, gatherCompare = 97

    var comparison: Bool {
        self == .compareImplicit || self == .compareExplicit || self == .projectCompareImplicit ||
            self == .projectCompareExplicit || self == .gatherCompare
    }

    var projected: Bool {
        self == .projectImplicit || self == .projectExplicit || self == .projectCompareImplicit || self == .projectCompareExplicit
    }

    var explicitLevel: Bool {
        self == .explicit || self == .compareExplicit || self == .projectExplicit || self == .projectCompareExplicit
    }

    var gathering: Bool { self == .gather || self == .gatherCompare }
}

struct SPVSampledImagePair {
    let image: IRHandle<IRExpression>
    let imageAnchor: IRHandle<IRStatement>?
    let sampler: IRHandle<IRExpression>
    let imageType: UInt32
    let block: UInt32
}

extension SPVFunctionLowerer {
    mutating func lowerSampledImagePair(_ instruction: SPVInstruction, _ statements: inout [IRHandle<IRStatement>]) throws {
        let operands = instruction.a
        if instruction.op == 100 {
            try instruction.count(3)
            let pair = try sampledImagePair(operands[2], at: instruction)
            guard operands[0] == pair.imageType else {
                throw instruction.fail("OpImage result must match the paired image type", code: .typeMismatch)
            }
            if isDynamicallyIndexedDescriptorArrayElement(pair.image) {
                try checkDescriptorNonUniform([operands[1], operands[2]], at: instruction)
            }
            defineValue(operands[1], .init(expression: pair.image, type: operands[0], anchor: pair.imageAnchor))
            return
        }
        try instruction.count(4)
        guard let imageType = module.sampledImageTypes[operands[0]] else {
            throw instruction.fail("OpSampledImage requires a sampled-image result type", code: .typeMismatch)
        }
        let image = try imageObject(operands[2], at: instruction, &statements)
        let declared = try module.kind(imageType, instruction)
        guard case .image(let dimension, let arrayed, _) = image.kind,
              case .image(let declaredDimension, let declaredArrayed, _) = declared,
              dimension == declaredDimension, arrayed == declaredArrayed,
              let scalar = image.kind.sampledTextureScalar, scalar == declared.sampledTextureScalar,
              image.kind.isMultisampled2D == declared.isMultisampled2D else {
            throw instruction.fail("OpSampledImage image must match the paired type", code: .typeMismatch)
        }
        let sampler = try value(operands[3], instruction, &statements)
        guard module.pointers[sampler.type] == nil, try module.kind(sampler.type, instruction).isSampler else {
            throw instruction.fail("OpSampledImage requires a sampler object", code: .typeMismatch)
        }
        if isDynamicallyIndexedDescriptorArrayElement(image.expression) || isDynamicallyIndexedDescriptorArrayElement(sampler.expression) {
            try checkDescriptorNonUniform([operands[1], operands[2], operands[3]], at: instruction)
        }
        sampledImages[operands[1]] = .init(image: image.expression, imageAnchor: image.anchor,
                                           sampler: sampler.expression, imageType: imageType, block: currentBlock)
    }

    mutating func lowerImageSample(_ instruction: SPVInstruction, _ statements: inout [IRHandle<IRStatement>]) throws {
        guard let operation = SPVImageSampleOperation(rawValue: instruction.op) else {
            throw instruction.fail("Unknown image sample instruction")
        }
        let comparison = operation.comparison
        let operands = instruction.a
        let required = comparison || operation.gathering ? 5 : 4
        try instruction.minimum(required)
        let pair = try sampledImagePair(operands[2], at: instruction)
        let image = module.types[expressions[pair.image.index].type.index].kind
        let sampler = module.types[expressions[pair.sampler.index].type.index].kind
        guard let scalar = image.sampledTextureScalar, operation.gathering || scalar == .f32,
              !image.isMultisampled2D, !comparison || image.isDepthTexture,
              sampler == .sampler(comparison: comparison) else {
            throw instruction.fail("Invalid image, sampler or result type for sampling", code: .typeMismatch)
        }
        let resultType: IRTypeKind = comparison && !operation.gathering ? .scalar(.f32) : .vector(size: .four, scalar: scalar)
        guard try module.kind(operands[0], instruction) == resultType else {
            throw instruction.fail("Image sample result does not match its operation and texture type", code: .typeMismatch)
        }
        let coordinate = try value(operands[3], instruction, &statements)
        let (unprojected, layer) = try imageCoordinates(coordinate, image: image, normalized: true, at: instruction, &statements)
        let reference = comparison ? try floatImageOperand(operands[4], at: instruction, &statements) : nil
        let (position, depthReference) = if operation.projected {
            try projectedImageOperands(coordinate, position: unprojected, depthReference: reference,
                                       image: image, at: instruction, &statements)
        } else { (unprojected, reference) }
        let gather: IRSwizzleComponent?
        if operation == .gather {
            guard let component = module.constantMap[operands[4]]?.nonnegativeInteger, component <= 3 else {
                throw instruction.fail("Image gather component must be a constant 32-bit integer from 0 to 3", code: .typeMismatch)
            }
            gather = IRSwizzleComponent.allCases[Int(component)]
        } else {
            gather = operation.gathering ? .x : nil
        }
        let levelMask: UInt32 = if operation.gathering { 0 }
            else if operation.explicitLevel { 2 | 4 }
            else { comparison ? 0 : 1 }
        let extra = try imageOperands(instruction, at: required, allowed: levelMask | 8)
        let level: IRSampleLevel
        if operation.gathering {
            level = .zero
        } else if operation.explicitLevel {
            switch (extra[2]?.first, extra[4]) {
            case (let lod?, nil):
                if image.isDepthTexture {
                    level = try depthImageLevel(lod, comparison: comparison, at: instruction, &statements)
                } else {
                    level = .exact(try floatImageOperand(lod, at: instruction, &statements))
                }
            case (nil, let gradients?):
                let x = try value(gradients[0], instruction, &statements).expression
                let y = try value(gradients[1], instruction, &statements).expression
                level = .gradient(x: x, y: y)
            default:
                throw instruction.fail("Explicit image sampling requires one of Lod or Grad")
            }
        } else if let bias = extra[1]?.first {
            level = .bias(try floatImageOperand(bias, at: instruction, &statements))
        } else {
            level = .auto
        }
        let offset: IRHandle<IRExpression>?
        if let id = extra[8]?.first {
            guard module.constantMap[id] != nil else { throw instruction.fail("ConstOffset must be a constant") }
            offset = try value(id, instruction, &statements).expression
        } else {
            offset = nil
        }
        let sample = IRExpressionKind.imageSample(.init(image: pair.image, sampler: pair.sampler, gather: gather, coordinate: position,
                                                       arrayIndex: layer, offset: offset, level: level, depthReference: depthReference))
        try imageTexelResult(sample, depth: image.isDepthTexture && !comparison && !operation.gathering, at: instruction, &statements)
    }

    mutating func lowerImageFetch(_ instruction: SPVInstruction, _ statements: inout [IRHandle<IRStatement>]) throws {
        let operands = instruction.a
        try instruction.minimum(4)
        let image = try imageObject(operands[2], at: instruction, &statements)
        guard let scalar = image.kind.sampledTextureScalar, !image.kind.isCubeTexture,
              try module.kind(operands[0], instruction) == .vector(size: .four, scalar: scalar) else {
            throw instruction.fail("Image fetch requires a non-cube sampled image and a matching vec4 result", code: .typeMismatch)
        }
        let coordinate = try value(operands[3], instruction, &statements)
        let (position, layer) = try imageCoordinates(coordinate, image: image.kind, at: instruction, &statements)
        let extra = try imageOperands(instruction, at: 4, allowed: image.kind.isMultisampled2D ? 64 : 2)
        let index: IRHandle<IRExpression>
        if let id = extra[image.kind.isMultisampled2D ? 64 : 2]?.first {
            index = try imageIndexOperand(id, at: instruction, &statements)
        } else {
            guard !image.kind.isMultisampled2D else { throw instruction.fail("Multisampled image fetch requires Sample") }
            index = try expr(internLoweredType(.scalar(.i32)), .literal(.i32(0)), instruction)
        }
        let load = IRExpressionKind.imageLoad(.init(image: image.expression, coordinate: position, arrayIndex: layer,
                                                    sample: image.kind.isMultisampled2D ? index : nil,
                                                    level: image.kind.isMultisampled2D ? nil : index))
        try imageTexelResult(load, depth: image.kind.isDepthTexture, at: instruction, &statements)
    }

    private mutating func imageTexelResult(
        _ kind: IRExpressionKind, depth: Bool, at instruction: SPVInstruction, _ statements: inout [IRHandle<IRStatement>]
    ) throws {
        if depth {
            let scalar = try internLoweredType(.scalar(.f32))
            let value = try emitValue(scalar, kind, &statements, instruction)
            let zero = try expr(scalar, .literal(.f32(0)), instruction)
            let one = try expr(scalar, .literal(.f32(1)), instruction)
            let type = try module.type(instruction.a[0], instruction)
            try result(instruction, .compose(type: type, components: [value, zero, zero, one]), &statements)
        } else {
            try result(instruction, kind, &statements)
        }
    }

    private mutating func floatImageOperand(
        _ id: UInt32, at instruction: SPVInstruction, _ statements: inout [IRHandle<IRStatement>]
    ) throws -> IRHandle<IRExpression> {
        let operand = try value(id, instruction, &statements)
        guard try module.kind(operand.type, instruction) == .scalar(.f32) else {
            throw instruction.fail("Image LOD, bias or depth reference must be f32", code: .typeMismatch)
        }
        return operand.expression
    }

    private mutating func depthImageLevel(
        _ id: UInt32, comparison: Bool, at instruction: SPVInstruction, _ statements: inout [IRHandle<IRStatement>]
    ) throws -> IRSampleLevel {
        let level = try floatImageOperand(id, at: instruction, &statements)
        let constant = floatImageConstant(id)
        if constant == 0 { return .zero }
        if comparison { return .exact(level) }
        if let constant {
            if let signed = Int32(exactly: constant) {
                return try .exact(expr(internLoweredType(.scalar(.i32)), .literal(.i32(signed)), instruction))
            }
            if let unsigned = UInt32(exactly: constant) {
                return try .exact(expr(internLoweredType(.scalar(.u32)), .literal(.u32(unsigned)), instruction))
            }
        }
        // Recover the integer operand of OpConvertSToF/OpConvertUToF. A rounded
        // integer outside the exact f32 range is already outside the mip range.
        if case .cast(let source, .f32) = expressions[level.index].kind {
            let sourceKind = module.types[expressions[source.index].type.index].kind
            if sourceKind == .scalar(.i32) || sourceKind == .scalar(.u32) { return .exact(source) }
        }
        throw instruction.fail("Explicit depth sampling requires an integer LOD", code: .unsupportedConstruct)
    }

    private func floatImageConstant(_ id: UInt32) -> Float? {
        guard let constant = module.constantMap[id] else { return nil }
        switch module.globalExpressions[module.constants[constant.handle.index].initializer.index].kind {
        case .literal(.f32(let value)): return value
        case .zeroValue: return 0
        default: return nil
        }
    }

    private func imageOperands(_ instruction: SPVInstruction, at start: Int, allowed: UInt32) throws -> [UInt32: [UInt32]] {
        let operands = instruction.a
        guard operands.count > start else { return [:] }
        let mask = try imageOperandsWithoutMemoryHints(instruction, at: start)
        guard mask & ~allowed == 0 else {
            throw instruction.fail("Unsupported image operands for this instruction", code: .unsupportedConstruct)
        }
        var fields: [UInt32: [UInt32]] = [:]
        var index = start + 1
        for (bit, arity) in [(UInt32(1), 1), (2, 1), (4, 2), (8, 1), (64, 1)] where mask & bit != 0 {
            guard operands.count - index >= arity else { throw instruction.fail("Missing image operand") }
            fields[bit] = Array(operands[index..<(index + arity)])
            index += arity
        }
        guard index == operands.count else { throw instruction.fail("Unexpected image operands") }
        return fields
    }

    private func sampledImagePair(_ id: UInt32, at instruction: SPVInstruction) throws -> SPVSampledImagePair {
        guard let pair = sampledImages[id], pair.block == currentBlock else {
            throw instruction.fail("A sampled-image pair must be produced in the current block")
        }
        return pair
    }
}
