import SAGAShaderIR

enum SPVSampledImage {
    static let imageCubeArrayCapability: UInt32 = 34
    static let cubeArrayCapability: UInt32 = 45
}

enum SPVStorageImage {
    static let uniformConstant: UInt32 = 0
    static let multisampleCapability: UInt32 = 27
    static let sampled1DCapability: UInt32 = 43
    static let image1DCapability: UInt32 = 44
    static let extendedFormatsCapability: UInt32 = 49
}

extension SPVDecoder {
    mutating func declareImage(_ instruction: SPVInstruction) throws {
        let operands = instruction.a
        if operands.count == 9 {
            throw instruction.fail("OpTypeImage Access Qualifier requires Kernel", code: .unsupportedCapability)
        }
        try instruction.count(8)
        try define(operands[0], instruction)
        guard operands[3] <= 2, operands[4] <= 1, operands[5] <= 1, operands[6] <= 2 else {
            throw instruction.fail("Invalid OpTypeImage operands")
        }
        switch operands[6] {
        case 1: try declareSampledTexture(instruction)
        case 2: try declareStorageImage(instruction)
        default:
            throw instruction.fail("OpTypeImage requires an explicit Sampled operand of 1 or 2", code: .unsupportedConstruct)
        }
    }

    mutating func declareSampler(_ instruction: SPVInstruction) throws {
        try instruction.count(1)
        let id = instruction.a[0]
        try define(id, instruction)
        try addType(id, .sampler(comparison: false), instruction)
    }

    mutating func declareSampledImage(_ instruction: SPVInstruction) throws {
        try instruction.count(2)
        let id = instruction.a[0]
        let image = instruction.a[1]
        try define(id, instruction)
        guard pointers[image] == nil, try kind(image, instruction).isSampledTexture else {
            throw instruction.fail("OpTypeSampledImage requires an OpTypeImage for sampling", code: .typeMismatch)
        }
        // Combined texture/sampler values remain local to the SPIR-V decoder.
        sampledImageTypes[id] = image
    }

    private mutating func declareSampledTexture(_ instruction: SPVInstruction) throws {
        let operands = instruction.a
        guard operands[7] == 0 else {
            throw instruction.fail("Explicit formats for sampled textures are not supported", code: .unsupportedConstruct)
        }
        let scalar = try scalar(operands[1], instruction)
        let arrayed = operands[4] == 1
        let multisampled = operands[5] == 1
        let dimension: IRImageDimension
        switch operands[2] {
        case 0:
            guard capabilities.contains(SPVStorageImage.sampled1DCapability) ||
                  capabilities.contains(SPVStorageImage.image1DCapability) else {
                throw instruction.fail("1D sampled textures require Sampled1D", code: .unsupportedCapability)
            }
            dimension = .d1
        case 1: dimension = .d2
        case 2: dimension = .d3
        case 3:
            if arrayed && !capabilities.contains(SPVSampledImage.cubeArrayCapability) &&
                !capabilities.contains(SPVSampledImage.imageCubeArrayCapability) {
                throw instruction.fail("Sampled cube arrays require SampledCubeArray", code: .unsupportedCapability)
            }
            dimension = .cube
        default:
            throw instruction.fail("Unsupported sampled texture dimensionality", code: .unsupportedConstruct)
        }
        let imageClass: IRImageClass
        if operands[3] == 1 {
            guard scalar == .f32 else { throw instruction.fail("Depth textures require f32", code: .typeMismatch) }
            imageClass = .depth(multisampled: multisampled)
        } else {
            guard scalar == .f32 || scalar == .i32 || scalar == .u32 else {
                throw instruction.fail("Sampled texture scalar must be f32, i32 or u32", code: .typeMismatch)
            }
            imageClass = .sampled(kind: scalar.kind, multisampled: multisampled)
        }
        let kind = IRTypeKind.image(dimension: dimension, arrayed: arrayed, class: imageClass)
        guard kind.isSampledTexture else {
            throw instruction.fail("Unsupported sampled texture shape", code: .unsupportedConstruct)
        }
        try addType(operands[0], kind, instruction)
        imageDepths[operands[0]] = operands[3]
    }

    private mutating func declareStorageImage(_ instruction: SPVInstruction) throws {
        let operands = instruction.a
        let dim = operands[2]
        let depth = operands[3]
        let arrayed = operands[4]
        let multisampled = operands[5]
        let formatID = operands[7]
        guard depth != 1, multisampled == 0 else {
            throw instruction.fail("Only non-depth, single-sampled storage images are supported", code: .unsupportedConstruct)
        }
        let dimension: IRImageDimension
        switch (dim, arrayed) {
        case (0, 0):
            guard capabilities.contains(SPVStorageImage.image1DCapability) else {
                throw instruction.fail("1D storage images require Image1D", code: .unsupportedCapability)
            }
            dimension = .d1
        case (1, _): dimension = .d2
        case (2, 0): dimension = .d3
        default:
            throw instruction.fail("Unsupported storage image dimensionality", code: .unsupportedConstruct)
        }
        guard let format = SPIRVStorageImageFormat.decode(formatID) else {
            throw instruction.fail("Unsupported storage image format \(formatID)", code: .unsupportedConstruct)
        }
        if format == .r64Uint, !capabilities.contains(SPIRVStorageImageFormat.int64ImageCapability) {
            throw instruction.fail("R64ui storage image requires Int64ImageEXT", code: .unsupportedCapability)
        }
        if SPIRVStorageImageFormat.requiresExtendedFormats(formatID) && !capabilities.contains(SPVStorageImage.extendedFormatsCapability) {
            throw instruction.fail("Storage image format requires StorageImageExtendedFormats", code: .unsupportedCapability)
        }
        guard try scalar(operands[1], instruction) == format.storageTexel.scalar else {
            throw instruction.fail("OpTypeImage Sampled Type does not match the storage format", code: .typeMismatch)
        }
        // The access qualifier belongs to the resource variable in shader SPIR-V.
        try addType(operands[0], .image(dimension: dimension, arrayed: arrayed == 1,
                                       class: .storage(format: format, access: .load)), instruction)
        storageImageSampledTypes[operands[0]] = operands[1]
    }

    mutating func qualifyStorageImage(
        _ base: IRHandle<IRType>, variable: UInt32, at instruction: SPVInstruction
    ) throws -> IRHandle<IRType> {
        guard case .image(let dimension, let arrayed, .storage(let format, let baseAccess)) = types[base.index].kind else {
            throw instruction.fail("UniformConstant resource must be a storage image", code: .unsupportedConstruct)
        }
        let prohibited = SPVStorageAccessDecorations.prohibited(decorations[variable] ?? [:])
        let readWrite: IRStorageAccess = [.load, .store]
        let access = readWrite.subtracting(prohibited)
        guard !access.isEmpty else {
            throw instruction.fail("Storage image has both NonReadable and NonWritable decorations")
        }
        if access == baseAccess { return base }
        return try internType(.image(dimension: dimension, arrayed: arrayed, class: .storage(format: format, access: access)))
    }
}
