import SAGAShaderIR

private enum GLSLTextureOperation: String {
    case texture
    case textureLod
    case textureGrad
    case textureSize
    case textureQueryLevels
    case textureSamples
    case texelFetch
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

    func declareGLSLOpaqueResource(
        _ kind: IRTypeKind, layout: [String: UInt32], flags: Set<String>, memory: Set<String>, at token: GLSLToken
    ) throws {
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
    }

    func acceptsGLSLCombinedSamplerArgument(_ name: String) -> Bool {
        GLSLTextureOperation(rawValue: name) != nil
    }

    func glslTextureCall(_ token: GLSLToken, arguments: [GLSLValue]) throws -> GLSLValue? {
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
        guard let operation = GLSLTextureOperation(rawValue: token.text) else { return nil }
        guard let pair = arguments.first, let sampler = pair.combinedSampler,
              let scalar = kind(pair.type).sampledTextureScalar else {
            throw glslError(token, .typeMismatch, "Texture operation requires a combined texture and sampler expression")
        }
        let image = kind(pair.type)
        let context = try requireContext()
        let comparison = kind(context.expressions[sampler.index].type) == .sampler(comparison: true)
        switch operation {
        case .textureSize, .textureQueryLevels, .textureSamples:
            return try glslTextureQuery(operation, pair: pair, image: image, arguments: arguments, at: token)
        case .texelFetch:
            guard arguments.count == 3, !comparison, !image.isCubeTexture, kind(arguments[2].type) == .scalar(.i32) else {
                throw glslError(token, .typeMismatch, "Invalid texelFetch operands")
            }
            try requireGLSLTextureRole(pair, comparison: false, at: token)
            let (position, layer) = try glslTextureCoordinates(arguments[1], image: image, scalar: .i32, at: token)
            let index = arguments[2].expression
            let load = IRImageLoad(image: pair.expression, coordinate: position, arrayIndex: layer,
                                   sample: image.isMultisampled2D ? index : nil,
                                   level: image.isMultisampled2D ? nil : index)
            return try expressionNode(addType(.vector(size: .four, scalar: scalar)), .imageLoad(load), at: token)
        case .texture, .textureLod, .textureGrad:
            if comparison {
                return try glslShadowSample(operation, pair: pair, sampler: sampler, image: image, arguments: arguments, at: token)
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
                guard !image.isLineTexture, arguments.count == 4,
                      kind(arguments[2].type) == gradient, kind(arguments[3].type) == gradient else {
                    throw glslError(token, .typeMismatch, "Invalid textureGrad dimension or gradient types")
                }
                level = .gradient(x: arguments[2].expression, y: arguments[3].expression)
            default: throw ShaderFailure.invalidHandle
            }
            let (position, layer) = try glslTextureCoordinates(arguments[1], image: image, scalar: .f32, at: token)
            let sample = IRImageSample(image: pair.expression, sampler: sampler, coordinate: position,
                                       arrayIndex: layer, level: level)
            return try expressionNode(addType(.vector(size: .four, scalar: .f32)), .imageSample(sample), at: token)
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
        image: IRTypeKind, arguments: [GLSLValue], at token: GLSLToken
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
                  case .literal(.f32(let lod)) = context.expressions[arguments[2].expression.index].kind, lod == 0 else {
                throw glslError(token, .unsupportedConstruct, "Comparison textureLod requires a 2D texture and literal zero LOD")
            }
            level = .zero
        default:
            throw glslError(token, .unsupportedConstruct, "Comparison sampling does not support explicit gradients")
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
            let packed = arguments[1]
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
                                   arrayIndex: layer, level: level, depthReference: reference.expression)
        return try expressionNode(addType(.scalar(.f32)), .imageSample(sample), at: token)
    }

    private func glslSpatialCoordinate(_ image: IRTypeKind, scalar: IRScalar) -> IRTypeKind {
        image.isCubeTexture ? .vector(size: .three, scalar: scalar) : image.texelCoordinateType(scalar)
    }

    private func glslTextureCoordinates(
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
