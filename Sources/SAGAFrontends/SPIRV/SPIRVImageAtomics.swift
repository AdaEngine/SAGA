import SAGAShaderIR

struct SPVImageTexelPointer {
    static let storageClass: UInt32 = 11
    static let int64AtomicsCapability: UInt32 = 12

    let instruction: SPVInstruction
    let block: UInt32
}

private extension IRStorageFormat {
    var supportsSPIRVImageAtomics: Bool {
        self == .r32Uint || self == .r32Sint || self == .r64Uint
    }
}

extension SPVDecoder {
    mutating func prepareImageAtomicResources() throws {
        guard hasImageTexelPointers else { return }
        var formals: [UInt32: [(slot: Int, instruction: SPVInstruction)]] = [:]
        var parameters: Set<UInt32> = []
        for function in functions {
            for (slot, parameter) in function.body.enumerated() {
                try charge()
                guard parameter.op == 55 else { break }
                guard parameter.a.count == 2, let type = typeMap[parameter.a[0]],
                      types[type.index].kind.storageTexture != nil else { continue }
                formals[function.header.a[1], default: []].append((slot, parameter))
                parameters.insert(parameter.a[1])
            }
        }

        var incoming: [UInt32: [(origin: UInt32, call: SPVInstruction)]] = [:]
        var atomicUses: [(id: UInt32, site: SPVInstruction)] = []
        for function in functions {
            var sources: [UInt32: UInt32] = [:]
            var texels: [UInt32: SPVInstruction] = [:]
            var consumers: [SPVInstruction] = []
            for instruction in function.body {
                try charge()
                let a = instruction.a
                switch (instruction.op, a.count) {
                case (61, 3), (83, 3), (65, 3), (66, 3): sources[a[1]] = a[2]
                case (65, 4), (66, 4):
                    if let raw = globalMap[a[2]]?.1 ?? functionValueTypes[a[2]], let base = pointers[raw],
                       let element = descriptorArrayElements[base.base], pointers[a[0]]?.base == element {
                        sources[a[1]] = a[2]
                    }
                case (60, 5): texels[a[1]] = instruction
                case (57, 3...), (234, 6), (236...242, 6): consumers.append(instruction)
                default: break
                }
            }
            var cache: [UInt32: UInt32] = [:]
            for consumer in consumers {
                try charge()
                let a = consumer.a
                if consumer.op == 57 {
                    for formal in formals[a[2]] ?? [] {
                        try charge()
                        guard formal.slot < a.count - 3 else { throw consumer.fail("Missing storage image function argument") }
                        let actual = a[formal.slot + 3]
                        guard (globalMap[actual]?.1 ?? functionValueTypes[actual]) == formal.instruction.a[0] else {
                            throw consumer.fail("Storage image function argument type mismatch", code: .typeMismatch)
                        }
                        guard let origin = try opaqueResourceOrigin(actual, sources: sources, parameters: parameters, cache: &cache) else {
                            throw consumer.fail("Cannot resolve a storage image function argument", code: .unsupportedConstruct)
                        }
                        incoming[formal.instruction.a[1], default: []].append((origin, consumer))
                    }
                } else if let texel = texels[a[2]] {
                    guard let origin = try opaqueResourceOrigin(texel.a[2], sources: sources, parameters: parameters, cache: &cache) else {
                        throw texel.fail("Image atomic requires a global image or descriptor array", code: .unsupportedConstruct)
                    }
                    atomicUses.append((origin, texel))
                }
            }
        }

        // Only atomic uses impose the atomic access qualifier on actual resources.
        var atomic: Set<UInt32> = []
        while let (id, site) = atomicUses.popLast() {
            try charge()
            guard atomic.insert(id).inserted else { continue }
            if let (global, _) = globalMap[id] {
                try upgradeAtomicStorageImage(global, at: site)
            } else {
                let arguments = incoming[id] ?? []
                try charge(arguments.count)
                atomicUses.append(contentsOf: arguments.map { ($0.origin, site) })
            }
        }

        // Functions are ordered callee-first. Infer formals from callers after all global qualifiers are known.
        for function in functions.reversed() {
            for formal in formals[function.header.a[1]] ?? [] {
                try charge()
                let parameter = formal.instruction
                let id = parameter.a[1]
                guard case .image(let dimension, let arrayed, .storage(let format, let declared)) = try kind(parameter.a[0], parameter) else {
                    throw parameter.fail("Expected a storage image parameter")
                }
                var common: IRStorageAccess?
                for argument in incoming[id] ?? [] {
                    try charge()
                    let actual: IRHandle<IRType>?
                    if let (global, _) = globalMap[argument.origin] {
                        actual = storageImageElement(globals[global.index].type).type
                    } else {
                        actual = storageImageParameterTypes[argument.origin]
                    }
                    guard let actual, let storage = types[actual.index].kind.storageTexture else {
                        throw argument.call.fail("Expected a storage image argument")
                    }
                    common = common.map { $0.intersection(storage.access) } ?? storage.access
                }
                let requiresAtomic = atomic.contains(id)
                let access = common ?? (requiresAtomic ? [.load, .store, .atomic] : declared)
                guard !access.isEmpty else {
                    throw parameter.fail("Storage image arguments have no common access", code: .unsupportedConstruct)
                }
                if requiresAtomic {
                    guard access.contains([.load, .store, .atomic]), format.supportsSPIRVImageAtomics else {
                        throw parameter.fail("Image atomic requires a read-write integer storage image", code: .unsupportedConstruct)
                    }
                }
                storageImageParameterTypes[id] = try internType(.image(dimension: dimension, arrayed: arrayed,
                                                                       class: .storage(format: format, access: access)))
            }
        }
    }

    private func storageImageElement(_ type: IRHandle<IRType>) -> (type: IRHandle<IRType>, array: IRArraySize?) {
        if case .bindingArray(let base, let size) = types[type.index].kind { return (base, size) }
        return (type, nil)
    }

    private mutating func upgradeAtomicStorageImage(_ handle: IRHandle<IRGlobalVariable>, at site: SPVInstruction) throws {
        let global = globals[handle.index]
        let element = storageImageElement(global.type)
        guard global.space == .handle,
              case .image(let dimension, let arrayed, .storage(let format, let access)) = types[element.type.index].kind,
              format.supportsSPIRVImageAtomics, access.contains([.load, .store]) else {
            throw site.fail("Image atomic requires a read-write r32uint, r32sint, or r64uint storage image", code: .unsupportedConstruct)
        }
        let atomic = try internType(.image(dimension: dimension, arrayed: arrayed,
                                           class: .storage(format: format, access: access.union(.atomic))))
        let type = try element.array.map { try internType(.bindingArray(base: atomic, size: $0)) } ?? atomic
        globals[handle.index] = .init(name: global.name, space: global.space, binding: global.binding, type: type,
                                      initializer: global.initializer, memoryDecorations: global.memoryDecorations,
                                      location: global.location)
    }
}

extension SPVFunctionLowerer {
    mutating func lowerImageTexelPointer(_ instruction: SPVInstruction, _ statements: inout [IRHandle<IRStatement>]) throws {
        try instruction.count(5)
        let a = instruction.a
        guard let result = module.pointers[a[0]], result.storage == SPVImageTexelPointer.storageClass else {
            throw instruction.fail("OpImageTexelPointer requires an Image pointer result")
        }
        let image = try value(a[2], instruction, &statements)
        let kind = module.types[expressions[image.expression.index].type.index].kind
        guard let imagePointer = module.pointers[image.type], imagePointer.storage == SPVStorageImage.uniformConstant,
              module.storageImageSampledTypes[imagePointer.base] == result.base,
              let storage = kind.storageTexture, storage.format.supportsSPIRVImageAtomics else {
            throw instruction.fail("OpImageTexelPointer requires a matching integer storage image pointer",
                                   code: .unsupportedConstruct)
        }
        guard let sample = module.constantMap[a[4]], sample.nonnegativeInteger == 0,
              try [.scalar(.i32), .scalar(.u32)].contains(module.kind(sample.type, instruction)) else {
            throw instruction.fail("Non-multisampled OpImageTexelPointer requires a constant zero sample")
        }
        let coordinate = try value(a[3], instruction, &statements)
        let spatial = kind.isLineTexture ? 1 : kind.isVolumeTexture ? 3 : 2
        guard let shape = IRBitcastShape(try module.kind(coordinate.type, instruction)),
              shape.scalar == .i32 || shape.scalar == .u32,
              Int(shape.size?.rawValue ?? 1) == spatial + (kind.isArrayedTexture ? 1 : 0) else {
            throw instruction.fail("OpImageTexelPointer coordinate does not match the image dimensions")
        }
        if isDynamicallyIndexedDescriptorArrayElement(image.expression) {
            try checkDescriptorNonUniform([a[1], a[2]], storage: true, at: instruction)
        }
        try node()
        imageTexelPointers[a[1]] = .init(instruction: instruction, block: currentBlock)
    }

    mutating func lowerImageAtomic(_ instruction: SPVInstruction, _ statements: inout [IRHandle<IRStatement>]) throws {
        guard [234, 236, 237, 238, 239, 240, 241, 242].contains(instruction.op) else {
            throw instruction.fail("This image atomic operation is not implemented", code: .unsupportedConstruct)
        }
        try instruction.count(6)
        let a = instruction.a
        guard let pointer = imageTexelPointers[a[2]], let type = module.pointers[pointer.instruction.a[0]], a[0] == type.base else {
            throw instruction.fail("Image atomic requires a direct OpImageTexelPointer and matching result type")
        }
        try proveDominance(of: pointer.block, at: currentBlock, instruction)
        let scope = module.constantMap[a[3]]?.nonnegativeInteger
        guard module.constantMap[a[4]]?.nonnegativeInteger == 0,
              try module.supportsRelaxedAtomicScope(scope, inStorage: true, at: instruction) else {
            throw instruction.fail("Only relaxed image atomics at supported storage scope are implemented", code: .unsupportedConstruct)
        }
        let image = try value(pointer.instruction.a[2], instruction, &statements)
        let kind = module.types[expressions[image.expression.index].type.index].kind
        guard let storage = kind.storageTexture, storage.access.contains([.load, .store, .atomic]) else {
            throw instruction.fail("Image atomic requires read-write atomic image access", code: .unsupportedConstruct)
        }
        if storage.scalar == .u64 {
            guard module.capabilities.contains(SPVImageTexelPointer.int64AtomicsCapability) else {
                throw instruction.fail("R64ui image atomics require Int64Atomics", code: .unsupportedCapability)
            }
            guard instruction.op == 237 || instruction.op == 239 else {
                throw instruction.fail("R64ui image atomics only support unsigned min and max", code: .unsupportedConstruct)
            }
        }
        let operation: IRAtomicFunction
        switch instruction.op {
        case 234: operation = .add
        case 236 where storage.scalar == .i32, 237 where storage.scalar.kind == .uint: operation = .min
        case 238 where storage.scalar == .i32, 239 where storage.scalar.kind == .uint: operation = .max
        case 240: operation = .and
        case 241: operation = .inclusiveOr
        case 242: operation = .exclusiveOr
        default: throw instruction.fail("Image atomic min/max opcode must match the image signedness", code: .unsupportedConstruct)
        }
        let coordinate = try value(pointer.instruction.a[3], instruction, &statements)
        let (position, layer) = try imageCoordinates(coordinate, image: kind, at: instruction, &statements)
        let operand = try typed(a[5], type.base, instruction, &statements)
        statements.append(try stmt(.imageAtomic(image: image.expression, coordinate: position, arrayIndex: layer,
                                                function: operation, value: operand), instruction))
        try node()
        discardedImageAtomicResults.insert(a[1])
    }
}
