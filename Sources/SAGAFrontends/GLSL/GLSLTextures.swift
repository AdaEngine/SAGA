import SAGAShaderIR

private enum GLSLTextureOperation: String {
    case texture
    case textureLod
    case textureGrad
    case textureSize
    case textureQueryLevels
    case textureSamples
    case texelFetch

    static func variant(_ name: String) -> (operation: Self, offset: Bool, projected: Bool)? {
        switch name {
        case "textureOffset": (.texture, true, false)
        case "textureLodOffset": (.textureLod, true, false)
        case "textureGradOffset": (.textureGrad, true, false)
        case "textureProj": (.texture, false, true)
        case "textureProjOffset": (.texture, true, true)
        case "textureProjLod": (.textureLod, false, true)
        case "textureProjLodOffset": (.textureLod, true, true)
        case "textureProjGrad": (.textureGrad, false, true)
        case "textureProjGradOffset": (.textureGrad, true, true)
        case "texelFetchOffset": (.texelFetch, true, false)
        default: nil
        }
    }
}

extension GLSLParser {
    func glslOpaqueResourceKind(_ name: String) -> IRTypeKind? {
        switch name {
        case "sampler": .sampler(comparison: false)
        case "samplerShadow": .sampler(comparison: true)
        default: glslSampledImageKind(name, prefix: "texture")
        }
    }

    func isGLSLCombinedTextureType(_ name: String) -> Bool {
        glslSampledImageKind(name, prefix: "sampler") != nil || name.hasPrefix("sampler") && name.hasSuffix("Shadow")
    }

    private func glslSampledImageKind(_ name: String, prefix: String) -> IRTypeKind? {
        let scalar: IRScalarKind
        let suffix: Substring
        if name.hasPrefix(prefix) {
            scalar = .float
            suffix = name.dropFirst(prefix.count)
        } else if name.hasPrefix("i" + prefix) {
            scalar = .sint
            suffix = name.dropFirst(prefix.count + 1)
        } else if name.hasPrefix("u" + prefix) {
            scalar = .uint
            suffix = name.dropFirst(prefix.count + 1)
        } else {
            return nil
        }
        let (dimension, arrayed, multisampled): (IRImageDimension, Bool, Bool)
        switch suffix {
        case "1D": (dimension, arrayed, multisampled) = (.d1, false, false)
        case "2D": (dimension, arrayed, multisampled) = (.d2, false, false)
        case "2DArray": (dimension, arrayed, multisampled) = (.d2, true, false)
        case "2DMS": (dimension, arrayed, multisampled) = (.d2, false, true)
        case "3D": (dimension, arrayed, multisampled) = (.d3, false, false)
        case "Cube": (dimension, arrayed, multisampled) = (.cube, false, false)
        case "CubeArray": (dimension, arrayed, multisampled) = (.cube, true, false)
        default: return nil
        }
        return .image(dimension: dimension, arrayed: arrayed, class: .sampled(kind: scalar, multisampled: multisampled))
    }

    @discardableResult func declareGLSLOpaqueResource(
        _ kind: IRTypeKind, layout: [String: UInt32], flags: Set<String>, memory: Set<String>, at token: GLSLToken
    ) throws -> IRHandle<IRGlobalVariable> {
        guard flags.isEmpty, memory.isEmpty, layout.keys.allSatisfy({ $0 == "set" || $0 == "binding" }),
              let binding = layout["binding"] else {
            throw glslError(token, .invalidBinding, "Separate GLSL texture or sampler requires an explicit binding")
        }
        let name = try identifier()
        try expect(";")
        guard !hasGlobalName(name.text), !name.text.hasPrefix("gl_") else {
            throw glslError(name, .duplicateName, "Duplicate or reserved GLSL resource")
        }
        let resource = IRResourceBinding(group: layout["set"] ?? 0, binding: binding)
        try budget.consume(.work, amount: globals.count)
        guard !globals.contains(where: { $0.binding == resource }) else {
            throw glslError(name, .invalidBinding, "Duplicate resource binding")
        }
        let type = try addType(kind)
        try budget.consume(.nodes)
        let global = try IRHandle<IRGlobalVariable>(index: globals.count)
        globals.append(.init(name: name.text, space: .handle, binding: resource, type: type, location: try name.location))
        globalNames[name.text] = global
        return global
    }

    func acceptsGLSLCombinedSamplerArgument(_ name: String) -> Bool {
        GLSLTextureOperation(rawValue: name) != nil || GLSLTextureOperation.variant(name) != nil
    }

    func glslTextureCall(_ token: GLSLToken, arguments original: [GLSLValue]) throws -> GLSLValue? {
        var arguments = original
        let shadow = glslShadowImageKind(token.text)
        if let expected = shadow ?? glslSampledImageKind(token.text, prefix: "sampler") {
            guard arguments.count == 2, kind(arguments[0].type) == expected,
                  kind(arguments[1].type) == .sampler(comparison: shadow != nil) else {
                throw glslError(token, .typeMismatch, "Combined sampler constructor requires a matching texture and sampler kind")
            }
            if shadow != nil { try requireGLSLTextureRole(arguments[0], comparison: true, at: token) }
            return .init(expression: arguments[0].expression, type: arguments[0].type, pointer: false, writable: false,
                         combinedSampler: arguments[1].expression)
        }
        if isGLSLCombinedTextureType(token.text) {
            throw glslError(token, .unsupportedConstruct, "Unsupported GLSL comparison texture dimension")
        }
        let variant = GLSLTextureOperation.variant(token.text)
        guard let operation = variant?.operation ?? GLSLTextureOperation(rawValue: token.text) else { return nil }
        guard let pair = arguments.first, let sampler = pair.combinedSampler,
              let scalar = kind(pair.type).sampledTextureScalar else {
            throw glslError(token, .typeMismatch, "Texture operation requires a combined texture and sampler expression")
        }
        let image = kind(pair.type)
        let context = try requireContext()
        let comparison = kind(context.expressions[sampler.index].type) == .sampler(comparison: true)
        let projected = variant?.projected == true
        if projected, image.isArrayedTexture || image.isCubeTexture {
            throw glslError(token, .unsupportedConstruct, "Projective sampling requires a non-array 1D, 2D or 3D texture")
        }
        let offset: IRHandle<IRExpression>?
        if variant?.offset == true {
            guard !image.isCubeTexture, !image.isMultisampled2D else {
                throw glslError(token, .unsupportedConstruct, "Texture offsets require a non-cube, non-multisampled texture")
            }
            let index = switch operation {
            case .texture: 2
            case .textureLod, .texelFetch: 3
            case .textureGrad: 4
            default: throw ShaderFailure.invalidHandle
            }
            guard arguments.indices.contains(index) else {
                throw glslError(token, .typeMismatch, "Missing texture offset")
            }
            offset = try glslTextureOffset(arguments.remove(at: index), image: image, at: token)
        } else {
            offset = nil
        }
        switch operation {
        case .textureSize, .textureQueryLevels, .textureSamples:
            return try glslTextureQuery(operation, pair: pair, image: image, arguments: arguments, at: token)
        case .texelFetch:
            guard arguments.count == 3, !comparison, !image.isCubeTexture, kind(arguments[2].type) == .scalar(.i32) else {
                throw glslError(token, .typeMismatch, "Invalid texelFetch operands")
            }
            try requireGLSLTextureRole(pair, comparison: false, at: token)
            let (position, layer) = try glslTextureCoordinates(arguments[1], image: image, scalar: .i32, at: token)
            // Apply the offset before the load's bounds checks, keeping the array layer separate.
            let coordinate = if let offset {
                try expressionNode(context.expressions[offset.index].type,
                                   .binary(op: .add, left: position, right: offset), at: token).expression
            } else { position }
            let index = arguments[2].expression
            let load = IRImageLoad(image: pair.expression, coordinate: coordinate, arrayIndex: layer,
                                   sample: image.isMultisampled2D ? index : nil,
                                   level: image.isMultisampled2D ? nil : index)
            return try expressionNode(addType(.vector(size: .four, scalar: scalar)), .imageLoad(load), at: token)
        case .texture, .textureLod, .textureGrad:
            if comparison {
                return try glslShadowSample(operation, pair: pair, sampler: sampler, image: image, arguments: arguments,
                                            offset: offset, projected: projected, at: token)
            }
            guard scalar == .f32, !image.isMultisampled2D else {
                throw glslError(token, .unsupportedConstruct, "Filtered sampling requires a non-multisampled float texture")
            }
            try requireGLSLTextureRole(pair, comparison: false, at: token)
            let level: IRSampleLevel
            switch operation {
            case .texture:
                guard arguments.count == 2 || (arguments.count == 3 && stage == .fragment && !image.isLineTexture &&
                                               kind(arguments[2].type) == .scalar(.f32)) else {
                    throw glslError(token, .typeMismatch, "Invalid implicit sampling operands")
                }
                level = arguments.count == 3 ? .bias(arguments[2].expression) : stage == .fragment ? .auto : .zero
            case .textureLod:
                guard arguments.count == 3, kind(arguments[2].type) == .scalar(.f32) else {
                    throw glslError(token, .typeMismatch, "textureLod requires a float level")
                }
                level = .exact(arguments[2].expression)
            case .textureGrad:
                let gradient = glslSpatialCoordinate(image, scalar: .f32)
                guard arguments.count == 4,
                      kind(arguments[2].type) == gradient, kind(arguments[3].type) == gradient else {
                    throw glslError(token, .typeMismatch, "Invalid textureGrad dimension or gradient types")
                }
                level = .gradient(x: arguments[2].expression, y: arguments[3].expression)
            default: throw ShaderFailure.invalidHandle
            }
            let coordinates = projected ? try glslProjectedCoordinates(arguments[1],
                spatial: glslSpatialCoordinate(image, scalar: .f32), at: token) : arguments[1]
            let (position, layer) = try glslTextureCoordinates(coordinates, image: image, scalar: .f32, at: token)
            let sample = IRImageSample(image: pair.expression, sampler: sampler, coordinate: position,
                                       arrayIndex: layer, offset: offset, level: level)
            return try expressionNode(addType(.vector(size: .four, scalar: .f32)), .imageSample(sample), at: token)
        }
    }

    private func glslProjectedCoordinates(_ value: GLSLValue, spatial: IRTypeKind, at token: GLSLToken) throws -> GLSLValue {
        let width: IRVectorSize? = if case .vector(let size, _) = spatial { size } else { nil }
        let requiredWidth = (width?.rawValue ?? 1) + 1
        guard case .vector(let packedWidth, .f32) = kind(value.type),
              packedWidth.rawValue == requiredWidth || packedWidth == .four else {
            throw glslError(token, .typeMismatch, "Invalid projective texture coordinate type")
        }
        let scalar = try addType(.scalar(.f32))
        let divisor = try expressionNode(scalar, .accessIndex(base: value.expression, index: UInt32(packedWidth.rawValue - 1)),
                                         at: token)
        let type = try addType(spatial)
        let numerator: GLSLValue
        let denominator: GLSLValue
        if let width {
            numerator = try expressionNode(type, .swizzle(size: width, vector: value.expression,
                                          pattern: .init(x: .x, y: .y, z: .z, w: .w)), at: token)
            denominator = try expressionNode(type, .splat(size: width, value: divisor.expression), at: token)
        } else {
            numerator = try expressionNode(type, .accessIndex(base: value.expression, index: 0), at: token)
            denominator = divisor
        }
        return try expressionNode(type, .binary(op: .divide, left: numerator.expression, right: denominator.expression), at: token)
    }

    private func glslTextureOffset(_ value: GLSLValue, image: IRTypeKind, at token: GLSLToken) throws -> IRHandle<IRExpression> {
        guard kind(value.type) == glslSpatialCoordinate(image, scalar: .i32) else {
            throw glslError(token, .typeMismatch, "Texture offset must match the texture dimension and use int components")
        }
        let constant = try evaluateGlobalConstant(value, at: token)
        let component = try addType(.scalar(.i32))
        let lanes = try constant.lanes.map { lane -> IRHandle<IRExpression> in
            guard case .i32(let integer) = lane, (-8...7).contains(integer) else {
                throw glslError(token, .invalidType, "Texture offset components must be between -8 and 7")
            }
            return try expressionNode(component, .literal(lane), at: token, emit: false).expression
        }
        switch kind(value.type) {
        case .scalar where lanes.count == 1: return lanes[0]
        case .vector(let width, _) where lanes.count == Int(width.rawValue):
            return try expressionNode(value.type, .compose(type: value.type, components: lanes), at: token).expression
        default: throw ShaderFailure.invalidHandle
        }
    }

    private func glslShadowImageKind(_ name: String) -> IRTypeKind? {
        guard name.hasSuffix("Shadow"),
              let image = glslSampledImageKind(String(name.dropLast("Shadow".count)), prefix: "sampler"),
              case .image(let dimension, _, .sampled(.float, false)) = image,
              dimension == .d2 || dimension == .cube else { return nil }
        return image
    }

    private func glslShadowSample(
        _ operation: GLSLTextureOperation, pair: GLSLValue, sampler: IRHandle<IRExpression>,
        image: IRTypeKind, arguments: [GLSLValue], offset: IRHandle<IRExpression>?, projected: Bool, at token: GLSLToken
    ) throws -> GLSLValue {
        let separateReference = image.isCubeTexture && image.isArrayedTexture
        let level: IRSampleLevel
        switch operation {
        case .texture:
            guard arguments.count == (separateReference ? 3 : 2) else {
                throw glslError(token, .unsupportedConstruct, "Comparison sampling does not support a LOD bias")
            }
            level = stage == .fragment ? .auto : .zero
        case .textureLod:
            let context = try requireContext()
            guard arguments.count == 3, !image.isCubeTexture, !image.isArrayedTexture,
                  kind(arguments[2].type) == .scalar(.f32) else {
                throw glslError(token, .typeMismatch, "Comparison textureLod requires a 2D texture and a float level")
            }
            if case .literal(.f32(0)) = context.expressions[arguments[2].expression.index].kind {
                level = .zero
            } else {
                level = .exact(arguments[2].expression)
            }
        case .textureGrad:
            let gradient = glslSpatialCoordinate(image, scalar: .f32)
            guard !separateReference, arguments.count == 4,
                  kind(arguments[2].type) == gradient, kind(arguments[3].type) == gradient else {
                throw glslError(token, .typeMismatch, "Invalid comparison textureGrad dimension or gradient types")
            }
            level = .gradient(x: arguments[2].expression, y: arguments[3].expression)
        default:
            throw ShaderFailure.invalidHandle
        }
        let coordinate: GLSLValue
        let reference: GLSLValue
        if separateReference {
            coordinate = arguments[1]
            reference = arguments[2]
            guard kind(reference.type) == .scalar(.f32) else {
                throw glslError(token, .typeMismatch, "Comparison reference must be a float")
            }
        } else {
            // Project the comparison reference together with the spatial coordinates.
            let packed = projected ? try glslProjectedCoordinates(arguments[1],
                spatial: .vector(size: .three, scalar: .f32), at: token) : arguments[1]
            let width: IRVectorSize = image.isCubeTexture || image.isArrayedTexture ? .three : .two
            guard let full = IRVectorSize(rawValue: width.rawValue + 1),
                  kind(packed.type) == .vector(size: full, scalar: .f32) else {
                throw glslError(token, .typeMismatch, "Comparison coordinates require a final reference component")
            }
            coordinate = try expressionNode(addType(.vector(size: width, scalar: .f32)),
                .swizzle(size: width, vector: packed.expression, pattern: .init(x: .x, y: .y, z: .z, w: .w)), at: token)
            reference = try expressionNode(addType(.scalar(.f32)),
                .accessIndex(base: packed.expression, index: UInt32(width.rawValue)), at: token)
        }
        let (position, layer) = try glslTextureCoordinates(coordinate, image: image, scalar: .f32, at: token)
        let sample = IRImageSample(image: pair.expression, sampler: sampler, coordinate: position,
                                   arrayIndex: layer, offset: offset, level: level, depthReference: reference.expression)
        return try expressionNode(addType(.scalar(.f32)), .imageSample(sample), at: token)
    }

    private func glslSpatialCoordinate(_ image: IRTypeKind, scalar: IRScalar) -> IRTypeKind {
        image.isCubeTexture ? .vector(size: .three, scalar: scalar) : image.texelCoordinateType(scalar)
    }

    func glslTextureCoordinates(
        _ value: GLSLValue, image: IRTypeKind, scalar: IRScalar, at token: GLSLToken
    ) throws -> (IRHandle<IRExpression>, IRHandle<IRExpression>?) {
        let spatial = glslSpatialCoordinate(image, scalar: scalar)
        if !image.isArrayedTexture {
            guard kind(value.type) == spatial else { throw glslError(token, .typeMismatch, "Texture coordinate type mismatch") }
            return (value.expression, nil)
        }
        guard case .vector(let size, _) = spatial,
              let full = IRVectorSize(rawValue: size.rawValue + 1),
              kind(value.type) == .vector(size: full, scalar: scalar) else {
            throw glslError(token, .typeMismatch, "Array texture coordinate requires a final layer component")
        }
        let position = try expressionNode(addType(spatial), .swizzle(size: size, vector: value.expression,
                                           pattern: .init(x: .x, y: .y, z: .z, w: .w)), at: token)
        let lane = try addType(.scalar(scalar))
        var layer = try expressionNode(lane, .accessIndex(base: value.expression, index: UInt32(size.rawValue)), at: token)
        if scalar == .f32 {
            // Vulkan samples round a floating array coordinate to nearest-even before clamping.
            layer = try expressionNode(lane, .math(function: .round, argument: layer.expression,
                                                   argument1: nil, argument2: nil, argument3: nil), at: token)
            layer = try expressionNode(addType(.scalar(.i32)), .cast(expression: layer.expression, target: .i32), at: token)
        }
        return (position.expression, layer.expression)
    }

    private func glslTextureQuery(
        _ operation: GLSLTextureOperation, pair: GLSLValue, image: IRTypeKind,
        arguments: [GLSLValue], at token: GLSLToken
    ) throws -> GLSLValue {
        let query: IRImageQuery
        var shape: IRTypeKind = .scalar(.u32)
        switch operation {
        case .textureSize:
            let multisampled = image.isMultisampled2D
            guard arguments.count == (multisampled ? 1 : 2),
                  multisampled || kind(arguments[1].type) == .scalar(.i32) else {
                throw glslError(token, .typeMismatch, "textureSize needs a mip level only for non-multisampled textures")
            }
            query = .size(level: multisampled ? nil : arguments[1].expression)
            shape = image.texelCoordinateType(.u32)
        case .textureQueryLevels where arguments.count == 1 && !image.isMultisampled2D:
            query = .numLevels
        case .textureSamples where arguments.count == 1 && image.isMultisampled2D:
            query = .numSamples
        default:
            throw glslError(token, .typeMismatch, "Invalid texture query operands")
        }
        var value = try expressionNode(addType(shape), .imageQuery(image: pair.expression, query: query), at: token)
        if operation == .textureSize, image.isArrayedTexture, case .vector(let width, _) = shape,
           let full = IRVectorSize(rawValue: width.rawValue + 1) {
            let layers = try expressionNode(addType(.scalar(.u32)), .imageQuery(image: pair.expression, query: .numLayers), at: token)
            shape = .vector(size: full, scalar: .u32)
            let type = try addType(shape)
            value = try expressionNode(type, .compose(type: type, components: [value.expression, layers.expression]), at: token)
        }
        let signed: IRTypeKind = if case .vector(let width, _) = shape { .vector(size: width, scalar: .i32) }
            else { .scalar(.i32) }
        return try expressionNode(addType(signed), .cast(expression: value.expression, target: .i32), at: token)
    }
}
