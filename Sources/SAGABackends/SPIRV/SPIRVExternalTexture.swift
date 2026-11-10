import SAGAShaderIR

// External texture lowering for SPIR-V. A `texture_external` binding becomes three sampled 2D float images (one per
// plane) and a uniform block of 13 uvec4 words (the same parameter words the text writers read), at fixed binding
// numbers derived from the WGSL binding so that the runtime can lay out descriptors without extra reflection. The
// sample, load and dimension operations expand inline to the same math as the HLSL helper functions.

struct SPIRVExternalGlobal {
    let planes: [UInt32]
    let imageType: UInt32
    let parameters: UInt32
}

/// The loaded plane images of an external texture value and the variable that holds its parameter words.
struct SPIRVExternalValue {
    let planes: [UInt32]
    let imageType: UInt32
    let parameters: UInt32
}

extension ComputeShaderArtifact.ExternalTextureLayout {
    /// Descriptor bindings (inside the descriptor set of the WGSL group) of an external texture's planes and parameter block.
    package static func vulkanBindings(for binding: IRResourceBinding) -> (planes: [UInt32], parameters: UInt32) {
        let base = 1024 + binding.binding * 4
        return ([base, base + 1, base + 2], base + 3)
    }
}

extension SPIRVBinaryWriter {
    mutating func externalGlobal(_ handle: IRHandle<IRGlobalVariable>) throws -> SPIRVExternalGlobal {
        try charge()
        if let cached = declarationState.externalGlobals[handle] { return cached }
        let global = try module.globalVariables.value(for: handle)
        guard global.space == .handle, let binding = global.binding, global.initializer == nil else {
            throw unsupported("requires an external texture binding")
        }
        let bindings = ComputeShaderArtifact.ExternalTextureLayout.vulkanBindings(for: binding)
        let imageType = try plainSampledImageTypeID()
        let imagePointer = try pointerTypeID(storageClass: 0, pointee: imageType)
        var planes: [UInt32] = []
        for index in 0..<3 {
            let id = try allocateID()
            try declare(59, [imagePointer, id, 0]) // OpVariable UniformConstant
            try decorate(71, [id, 34, binding.group])
            try decorate(71, [id, 33, bindings.planes[index]])
            try debugName(id, "\(global.name ?? "external")_plane_\(index)")
            planes.append(id)
        }
        // struct { uvec4 words[13]; } as a Block in the Uniform class.
        let words = try vectorTypeID(size: .four, scalar: .u32)
        let array = try allocateID()
        try declare(28, [array, words, unsignedConstantID(13)]) // OpTypeArray
        try decorate(71, [array, 6, 16]) // ArrayStride
        let block = try allocateID()
        try declare(30, [block, array]) // OpTypeStruct
        try decorate(71, [block, 2]) // Block
        try decorate(72, [block, 0, 35, 0]) // Offset
        let pointer = try pointerTypeID(storageClass: 2, pointee: block)
        let parameters = try allocateID()
        try declare(59, [pointer, parameters, 2]) // OpVariable Uniform
        try decorate(71, [parameters, 34, binding.group])
        try decorate(71, [parameters, 33, bindings.parameters])
        try debugName(parameters, "\(global.name ?? "external")_parameters")
        let result = SPIRVExternalGlobal(planes: planes, imageType: imageType, parameters: parameters)
        declarationState.externalGlobals[handle] = result
        return result
    }

    /// `OpTypeImage %float 2D depth=0 arrayed=0 ms=0 sampled=1`, the shape of every plane.
    private mutating func plainSampledImageTypeID() throws -> UInt32 {
        try imageTypeID(dimension: .d2, arrayed: false, imageClass: .sampled(kind: .float, multisampled: false))
    }

    /// Loads the plane images and registers the value so image operations can find the parameters.
    mutating func loadExternalTexture(
        _ handle: IRHandle<IRGlobalVariable>, in state: inout SPIRVFunctionState
    ) throws -> UInt32 {
        let global = try externalGlobal(handle)
        var planes: [UInt32] = []
        for variable in global.planes {
            let id = try allocateID()
            try emit(61, [global.imageType, id, variable]) // OpLoad
            planes.append(id)
        }
        state.externalTextures[planes[0]] = SPIRVExternalValue(planes: planes, imageType: global.imageType,
                                                              parameters: global.parameters)
        return planes[0]
    }

    // MARK: Math

    private struct Types {
        let f: UInt32, v2: UInt32, v4: UInt32, u: UInt32, u2: UInt32, u4: UInt32
        let bool: UInt32, b2: UInt32, b4: UInt32
    }

    private mutating func externalTypes() throws -> Types {
        try Types(f: scalarTypeID(.f32), v2: vectorTypeID(size: .two, scalar: .f32), v4: vectorTypeID(size: .four, scalar: .f32),
                  u: scalarTypeID(.u32), u2: vectorTypeID(size: .two, scalar: .u32), u4: vectorTypeID(size: .four, scalar: .u32),
                  bool: scalarTypeID(.bool), b2: vectorTypeID(size: .two, scalar: .bool), b4: vectorTypeID(size: .four, scalar: .bool))
    }

    /// Parameter words `index` as raw uvec4 and as vec4 float.
    private mutating func externalWords(_ value: SPIRVExternalValue, _ types: Types, index: UInt32) throws -> (raw: UInt32, float: UInt32) {
        let pointer = try pointerTypeID(storageClass: 2, pointee: types.u4)
        let address = try rawValueInstruction(65, type: pointer, operands: [value.parameters, unsignedConstantID(0), unsignedConstantID(index)])
        let raw = try rawValueInstruction(61, type: types.u4, operands: [address])
        return (raw, try rawValueInstruction(124, type: types.v4, operands: [raw])) // OpBitcast
    }

    private mutating func externalConvert(
        y: UInt32, chroma: UInt32, _ value: SPIRVExternalValue, _ t: Types
    ) throws -> UInt32 {
        func f(_ code: UInt32, _ a: UInt32, _ b: UInt32) throws -> UInt32 { try rawValueInstruction(code, type: t.f, operands: [a, b]) }
        func v4(_ code: UInt32, _ a: UInt32, _ b: UInt32) throws -> UInt32 { try rawValueInstruction(code, type: t.v4, operands: [a, b]) }
        func part(_ vector: UInt32, _ index: UInt32) throws -> UInt32 { try rawValueInstruction(81, type: t.f, operands: [vector, index]) }
        let one = try literalID(.f32(1)), zero = try literalID(.f32(0))
        let w0 = try externalWords(value, t, index: 0).float, w1 = try externalWords(value, t, index: 1).float
        let w2 = try externalWords(value, t, index: 2).float, w3 = try externalWords(value, t, index: 3).float
        let u = try part(chroma, 0), v = try part(chroma, 1)
        func scaled(_ vector: UInt32, _ scalar: UInt32) throws -> UInt32 {
            try rawValueInstruction(142, type: t.v4, operands: [vector, scalar]) // OpVectorTimesScalar
        }
        var encoded = try v4(129, scaled(w0, y), scaled(w1, u))
        encoded = try v4(129, encoded, scaled(w2, v))
        encoded = try v4(129, encoded, w3)
        let source = try externalWords(value, t, index: 7).float, target = try externalWords(value, t, index: 8).float
        let sx = try part(source, 0), sy = try part(source, 1), sz = try part(source, 2), sw = try part(source, 3)
        let tx = try part(target, 0), ty = try part(target, 1), tz = try part(target, 2), tw = try part(target, 3)
        // Source transfer function: linearize each lane.
        var linear: [UInt32] = []
        let knee = try f(133, sw, sy)
        let sxMinusOne = try f(131, sx, one)
        for lane in 0..<3 {
            let s = try part(encoded, UInt32(lane))
            let below = try rawValueInstruction(184, type: t.bool, operands: [s, knee]) // OpFOrdLessThan
            let linearPart = try f(136, s, sw)
            let shifted = try f(136, try f(129, s, sxMinusOne), sx)
            let powered = try glslMathInstruction(26, type: t.f, operands: [glslMathInstruction(40, type: t.f, operands: [shifted, zero]), sz])
            linear.append(try rawValueInstruction(169, type: t.f, operands: [below, linearPart, powered]))
        }
        let c0 = try externalWords(value, t, index: 4).float, c1 = try externalWords(value, t, index: 5).float
        let c2 = try externalWords(value, t, index: 6).float
        var converted = try v4(129, scaled(c0, linear[0]), scaled(c1, linear[1]))
        converted = try v4(129, converted, scaled(c2, linear[2]))
        // Destination transfer function: encode each lane.
        let inverseGamma = try f(136, one, tz)
        let offset = try f(131, tx, one)
        var lanes: [UInt32] = []
        for lane in 0..<3 {
            let d = try part(converted, UInt32(lane))
            let below = try rawValueInstruction(184, type: t.bool, operands: [d, ty])
            let linearPart = try f(133, tw, d)
            let powered = try glslMathInstruction(26, type: t.f, operands: [glslMathInstruction(40, type: t.f, operands: [d, zero]), inverseGamma])
            let curved = try f(131, try f(133, tx, powered), offset)
            lanes.append(try rawValueInstruction(169, type: t.f, operands: [below, linearPart, curved]))
        }
        return try rawValueInstruction(80, type: t.v4, operands: lanes + [one])
    }

    private mutating func splat(_ condition: UInt32, _ t: Types, width: Int) throws -> UInt32 {
        try rawValueInstruction(80, type: width == 2 ? t.b2 : t.b4, operands: Array(repeating: condition, count: width))
    }

    private mutating func planeSize(_ image: UInt32, _ t: Types) throws -> UInt32 {
        try requireCapability(50) // ImageQuery
        return try rawValueInstruction(103, type: t.u2, operands: [image, unsignedConstantID(0)]) // OpImageQuerySizeLod
    }

    /// The logical size when the parameters carry one, else the extent of the first plane.
    mutating func externalDimensions(_ value: SPIRVExternalValue) throws -> UInt32 {
        let t = try externalTypes()
        let raw = try externalWords(value, t, index: 12).raw
        let logical = try rawValueInstruction(79, type: t.u2, operands: [raw, raw, 0, 1]) // OpVectorShuffle
        let zero = try unsignedConstantID(0)
        let zeros = try rawValueInstruction(80, type: t.u2, operands: [zero, zero])
        let nonZero = try rawValueInstruction(171, type: t.b2, operands: [logical, zeros]) // OpINotEqual
        let hasLogical = try rawValueInstruction(155, type: t.bool, operands: [nonZero]) // OpAll
        return try rawValueInstruction(169, type: t.u2, operands: [splat(hasLogical, t, width: 2), logical, planeSize(value.planes[0], t)])
    }

    mutating func externalSample(_ value: SPIRVExternalValue, sampler: UInt32, coordinate uv: UInt32) throws -> UInt32 {
        let t = try externalTypes()
        func f(_ code: UInt32, _ a: UInt32, _ b: UInt32) throws -> UInt32 { try rawValueInstruction(code, type: t.f, operands: [a, b]) }
        func v2(_ code: UInt32, _ a: UInt32, _ b: UInt32) throws -> UInt32 { try rawValueInstruction(code, type: t.v2, operands: [a, b]) }
        func scaled(_ vector: UInt32, _ scalar: UInt32) throws -> UInt32 {
            try rawValueInstruction(142, type: t.v2, operands: [vector, scalar])
        }
        let transform = try externalWords(value, t, index: 9).float, originWords = try externalWords(value, t, index: 10).float
        let across = try rawValueInstruction(79, type: t.v2, operands: [transform, transform, 0, 1]) // xy
        let down = try rawValueInstruction(79, type: t.v2, operands: [transform, transform, 2, 3]) // zw
        let origin = try rawValueInstruction(79, type: t.v2, operands: [originWords, originWords, 0, 1])
        let opposite = try v2(129, v2(129, origin, across), down)
        let low = try glslMathInstruction(37, type: t.v2, operands: [origin, opposite])  // FMin
        let high = try glslMathInstruction(40, type: t.v2, operands: [origin, opposite]) // FMax
        let uvX = try rawValueInstruction(81, type: t.f, operands: [uv, 0]), uvY = try rawValueInstruction(81, type: t.f, operands: [uv, 1])
        let point = try v2(129, v2(129, origin, scaled(across, uvX)), scaled(down, uvY))
        let half = try literalID(.f32(0.5))
        let center = try scaled(v2(129, low, high), half)
        let halves = try rawValueInstruction(80, type: t.v2, operands: [half, half])
        let sampledType = try sampledImageTypeID(image: value.imageType)
        func sample(_ plane: UInt32) throws -> UInt32 {
            let size = try planeSize(plane, t)
            let sizeFloat = try rawValueInstruction(112, type: t.v2, operands: [size]) // OpConvertUToF
            let inset = try v2(136, halves, sizeFloat)
            let lower = try glslMathInstruction(37, type: t.v2, operands: [v2(129, low, inset), center])
            let upper = try glslMathInstruction(40, type: t.v2, operands: [v2(131, high, inset), center])
            let clamped = try glslMathInstruction(43, type: t.v2, operands: [point, lower, upper]) // FClamp
            let pair = try rawValueInstruction(86, type: sampledType, operands: [plane, sampler])
            return try rawValueInstruction(88, type: t.v4, operands: [pair, clamped, 2, literalID(.f32(0))]) // Lod 0
        }
        let first = try sample(value.planes[0])
        let second = try sample(value.planes[1]), third = try sample(value.planes[2])
        return try externalFinish(first: first, second: second, third: third, value, t)
    }

    /// Picks the RGBA texel for one plane or converts the YCbCr planes, as the helpers' early return does.
    private mutating func externalFinish(first: UInt32, second: UInt32, third: UInt32, _ value: SPIRVExternalValue, _ t: Types) throws -> UInt32 {
        let raw = try externalWords(value, t, index: 12).raw
        let kind = try rawValueInstruction(81, type: t.u, operands: [raw, 2])
        let isSingle = try rawValueInstruction(170, type: t.bool, operands: [kind, unsignedConstantID(1)])
        let isThree = try rawValueInstruction(170, type: t.bool, operands: [kind, unsignedConstantID(3)])
        let y = try rawValueInstruction(81, type: t.f, operands: [first, 0])
        let cb = try rawValueInstruction(81, type: t.f, operands: [second, 0])
        let crTwo = try rawValueInstruction(81, type: t.f, operands: [second, 1])
        let crThree = try rawValueInstruction(81, type: t.f, operands: [third, 0])
        let cr = try rawValueInstruction(169, type: t.f, operands: [isThree, crThree, crTwo])
        let chroma = try rawValueInstruction(80, type: t.v2, operands: [cb, cr])
        let converted = try externalConvert(y: y, chroma: chroma, value, t)
        return try rawValueInstruction(169, type: t.v4, operands: [splat(isSingle, t, width: 4), first, converted])
    }

    mutating func externalLoad(_ value: SPIRVExternalValue, pixel: UInt32) throws -> UInt32 {
        let t = try externalTypes()
        func u2(_ code: UInt32, _ a: UInt32, _ b: UInt32) throws -> UInt32 { try rawValueInstruction(code, type: t.u2, operands: [a, b]) }
        func v2(_ code: UInt32, _ a: UInt32, _ b: UInt32) throws -> UInt32 { try rawValueInstruction(code, type: t.v2, operands: [a, b]) }
        func scaled(_ vector: UInt32, _ scalar: UInt32) throws -> UInt32 {
            try rawValueInstruction(142, type: t.v2, operands: [vector, scalar])
        }
        let size = try externalDimensions(value)
        let outside = try rawValueInstruction(154, type: t.bool, operands: [rawValueInstruction(174, type: t.b2, operands: [pixel, size])]) // OpAny(UGreaterThanEqual)
        let transform = try externalWords(value, t, index: 11).float, originWords = try externalWords(value, t, index: 10).float
        let across = try rawValueInstruction(79, type: t.v2, operands: [transform, transform, 0, 1])
        let offset = try rawValueInstruction(79, type: t.v2, operands: [transform, transform, 2, 3])
        let scale = try rawValueInstruction(79, type: t.v2, operands: [originWords, originWords, 2, 3])
        let pixelFloat = try rawValueInstruction(112, type: t.v2, operands: [pixel])
        let px = try rawValueInstruction(81, type: t.f, operands: [pixelFloat, 0]), py = try rawValueInstruction(81, type: t.f, operands: [pixelFloat, 1])
        let point = try v2(129, v2(129, scaled(scale, px), scaled(across, py)), offset)
        let extent = try planeSize(value.planes[0], t)
        let one = try unsignedConstantID(1)
        let ones = try rawValueInstruction(80, type: t.u2, operands: [one, one])
        let lastFloat = try rawValueInstruction(112, type: t.v2, operands: [u2(130, extent, ones)])
        let zero = try literalID(.f32(0))
        let zeros = try rawValueInstruction(80, type: t.v2, operands: [zero, zero])
        let rounded = try glslMathInstruction(2, type: t.v2, operands: [point]) // RoundEven
        let clamped = try glslMathInstruction(43, type: t.v2, operands: [rounded, zeros, lastFloat])
        let c0 = try rawValueInstruction(109, type: t.u2, operands: [clamped]) // OpConvertFToU
        func fetch(_ plane: UInt32, _ coordinate: UInt32) throws -> UInt32 {
            try rawValueInstruction(95, type: t.v4, operands: [plane, coordinate, 2, unsignedConstantID(0)])
        }
        func subsampled(_ plane: UInt32) throws -> UInt32 {
            let planeExtent = try planeSize(plane, t)
            let scaledPosition = try v2(136, v2(133, rawValueInstruction(112, type: t.v2, operands: [c0]),
                                                 rawValueInstruction(112, type: t.v2, operands: [planeExtent])),
                                        rawValueInstruction(112, type: t.v2, operands: [extent]))
            let floored = try rawValueInstruction(109, type: t.u2, operands: [glslMathInstruction(8, type: t.v2, operands: [scaledPosition])])
            return try glslMathInstruction(38, type: t.u2, operands: [floored, u2(130, planeExtent, ones)]) // UMin
        }
        let first = try fetch(value.planes[0], c0)
        let second = try fetch(value.planes[1], subsampled(value.planes[1]))
        let third = try fetch(value.planes[2], subsampled(value.planes[2]))
        let result = try externalFinish(first: first, second: second, third: third, value, t)
        let transparent = try rawValueInstruction(80, type: t.v4, operands: [zero, zero, zero, zero])
        return try rawValueInstruction(169, type: t.v4, operands: [splat(outside, t, width: 4), transparent, result])
    }
}
