import SAGAShaderIR

extension GLSLParser {
    func glslStorageImageShape(_ name: String) -> (IRScalar, IRImageDimension, Bool)? {
        let scalar: IRScalar
        let suffix: Substring
        if name.hasPrefix("image") { (scalar, suffix) = (.f32, name.dropFirst(5)) }
        else if name.hasPrefix("iimage") { (scalar, suffix) = (.i32, name.dropFirst(6)) }
        else if name.hasPrefix("uimage") { (scalar, suffix) = (.u32, name.dropFirst(6)) }
        else { return nil }
        switch suffix {
        case "1D": return (scalar, .d1, false)
        case "2D": return (scalar, .d2, false)
        case "2DArray": return (scalar, .d2, true)
        case "3D": return (scalar, .d3, false)
        default: return nil
        }
    }

    func declareGLSLStorageImage(
        _ shape: (IRScalar, IRImageDimension, Bool), layout: [String: UInt32], flags: Set<String>,
        memory: Set<String>, at token: GLSLToken
    ) throws {
        guard flags.count == 1, let qualifier = flags.first,
              let format = IRStorageFormat.allCases.first(where: { $0.glslImageLayout == qualifier }),
              format.storageTexel.scalar == shape.0, memory.isSubset(of: ["readonly", "writeonly"]) else {
            throw glslError(token, .invalidBinding, "Storage image requires a matching format and supported access qualifiers")
        }
        let queryOnly = memory.contains("readonly") && memory.contains("writeonly")
        let access: IRStorageAccess = memory.contains("readonly") ? .load : memory.contains("writeonly") ? .store : [.load, .store]
        let kind = IRTypeKind.image(dimension: shape.1, arrayed: shape.2, class: .storage(format: format, access: access))
        let global = try declareGLSLOpaqueResource(kind, layout: layout, flags: [], memory: [], at: token)
        // IR has no query-only binding. A read-only binding can implement queries without granting source texel reads.
        if queryOnly { queryOnlyStorageImages.insert(global) }
    }

    func glslStorageImageCall(_ token: GLSLToken, arguments: [GLSLValue]) throws -> GLSLValue? {
        guard token.text == "imageLoad" || token.text == "imageSize" else { return nil }
        let load = token.text == "imageLoad"
        guard arguments.count == (load ? 2 : 1), let image = arguments.first,
              let texture = kind(image.type).storageTexture else {
            throw glslError(token, .typeMismatch, "Invalid storage image builtin operands")
        }
        if load {
            guard texture.access.contains(.load), try !isQueryOnlyImage(image) else {
                throw glslError(token, .invalidAddressSpace, "Storage image does not allow texel reads")
            }
            let (coordinate, layer) = try glslTextureCoordinates(arguments[1], image: kind(image.type), scalar: .i32, at: token)
            let type = try addType(.vector(size: .four, scalar: texture.scalar))
            return try expressionNode(type, .imageLoad(.init(image: image.expression, coordinate: coordinate,
                                                              arrayIndex: layer)), at: token)
        }
        let unsigned = kind(image.type).texelCoordinateType(.u32)
        let signed = kind(image.type).texelCoordinateType(.i32)
        let size = try expressionNode(addType(unsigned), .imageQuery(image: image.expression, query: .size(level: nil)), at: token)
        let dimensions = try expressionNode(addType(signed), .cast(expression: size.expression, target: .i32), at: token)
        guard kind(image.type).isArrayedTexture else { return dimensions }
        let layers = try expressionNode(addType(.scalar(.u32)), .imageQuery(image: image.expression, query: .numLayers), at: token)
        let layerCount = try expressionNode(addType(.scalar(.i32)), .cast(expression: layers.expression, target: .i32), at: token)
        let result = try addType(.vector(size: .three, scalar: .i32))
        return try expressionNode(result, .compose(type: result, components: [dimensions.expression, layerCount.expression]), at: token)
    }

    func glslImageStoreStatement(at token: GLSLToken) throws {
        _ = try advance()
        try expect("(")
        let image = try rvalue(expression(), at: token)
        try expect(",")
        let position = try rvalue(expression(), at: token)
        try expect(",")
        let data = try rvalue(expression(), at: token)
        try expect(")")
        guard let texture = kind(image.type).storageTexture, texture.access.contains(.store) else {
            throw glslError(token, .invalidAddressSpace, "Storage image does not allow texel writes")
        }
        let type = try addType(.vector(size: .four, scalar: texture.scalar))
        guard let value = try implicitlyConvert(data, to: type, at: token) else {
            throw glslError(token, .typeMismatch, "imageStore requires a matching four-component value")
        }
        let (coordinate, layer) = try glslTextureCoordinates(position, image: kind(image.type), scalar: .i32, at: token)
        try append(.imageStore(image: image.expression, coordinate: coordinate, arrayIndex: layer,
                               value: value.expression), at: token)
    }

    private func isQueryOnlyImage(_ image: GLSLValue) throws -> Bool {
        let context = try requireContext()
        guard case .globalVariable(let global) = context.expressions[image.expression.index].kind else { return false }
        return queryOnlyStorageImages.contains(global)
    }
}
