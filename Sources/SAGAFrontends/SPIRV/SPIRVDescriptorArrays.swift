import SAGAShaderIR

enum SPVDescriptorIndexing {
    static let extensionName = "SPV_EXT_descriptor_indexing"
    static let dynamicUniformBufferArray: UInt32 = 28
    static let dynamicSampledArray: UInt32 = 29
    static let dynamicStorageBufferArray: UInt32 = 30
    static let dynamicStorageArray: UInt32 = 31
    static let nonUniform: UInt32 = 5300
    static let shaderNonUniform: UInt32 = 5301
    static let runtimeDescriptorArray: UInt32 = 5302
    static let nonUniformUniformBufferArray: UInt32 = 5306
    static let nonUniformSampledArray: UInt32 = 5307
    static let nonUniformStorageBufferArray: UInt32 = 5308
    static let nonUniformStorageArray: UInt32 = 5309
    static let nonUniformCapabilities: Set<UInt32> = [
        shaderNonUniform, nonUniformUniformBufferArray, nonUniformSampledArray,
        nonUniformStorageBufferArray, nonUniformStorageArray,
    ]
    static let coreVersion: UInt32 = 0x00010500
}

extension SPVDecoder {
    private func usesSimpleOpaqueDescriptorRules(_ kind: IRTypeKind) -> Bool {
        usesSimpleMemoryModel && kind.isTextureOrSampler && kind.storageTexture == nil
    }

    func validateArrayStrideDecoration(_ id: UInt32, at instruction: SPVInstruction) throws {
        guard let type = typeMap[id] else { throw instruction.fail("ArrayStride target must be array") }
        switch types[type.index].kind {
        case .array:
            return
        case .bindingArray(let element, _) where usesSimpleOpaqueDescriptorRules(types[element.index].kind):
            // The stride of a generic opaque descriptor array has no byte layout in IR.
            return
        default:
            throw instruction.fail("ArrayStride target must be array")
        }
    }

    func validateDescriptorIndexingCapabilities(at instruction: SPVInstruction) throws {
        if version < SPVDescriptorIndexing.coreVersion,
           capabilities.contains(SPVDescriptorIndexing.runtimeDescriptorArray) ||
             !capabilities.isDisjoint(with: SPVDescriptorIndexing.nonUniformCapabilities),
           !extensions.contains(SPVDescriptorIndexing.extensionName) {
            throw instruction.fail("Descriptor indexing requires SPV_EXT_descriptor_indexing before SPIR-V 1.5",
                                   code: .unsupportedCapability)
        }
    }

    func validateNonUniformDecoration(_ id: UInt32, at instruction: SPVInstruction) throws {
        guard !capabilities.isDisjoint(with: SPVDescriptorIndexing.nonUniformCapabilities) else {
            throw instruction.fail("NonUniform requires ShaderNonUniform", code: .unsupportedCapability)
        }
        guard let raw = functionValueTypes[id] ?? constantMap[id]?.type ?? overrideMap[id]?.type ?? globalMap[id]?.1 else {
            throw instruction.fail("NonUniform must target a typed value")
        }
        if sampledImageTypes[raw] != nil { return }
        if pointers[raw]?.storage == SPVImageTexelPointer.storageClass { return }
        switch try kind(raw, instruction) {
        case .scalar, .vector, .pointer, .image, .sampler, .bindingArray: return
        default: throw instruction.fail("NonUniform on this value type is not supported", code: .unsupportedConstruct)
        }
    }

    mutating func declareDescriptorArray(
        _ instruction: SPVInstruction, base: IRHandle<IRType>, size: IRArraySize
    ) throws -> Bool {
        let kind = types[base.index].kind
        if case .bindingArray = kind {
            throw instruction.fail("Nested descriptor arrays are not implemented", code: .unsupportedConstruct)
        }
        let element = instruction.a[1]
        let isBufferBlock: Bool
        if case .structure = kind {
            isBufferBlock = decorations[element]?[SPVLegacyStorage.block] != nil ||
                decorations[element]?[SPVLegacyStorage.bufferBlock] != nil
        } else {
            isBufferBlock = false
        }
        guard kind.isTextureOrSampler || isBufferBlock else { return false }
        let simpleOpaque = usesSimpleOpaqueDescriptorRules(kind)
        if size == .dynamic, !simpleOpaque, !capabilities.contains(SPVDescriptorIndexing.runtimeDescriptorArray) {
            throw instruction.fail("Runtime descriptor arrays require RuntimeDescriptorArray", code: .unsupportedCapability)
        }
        let id = instruction.a[0]
        if let stride = decorations[id]?[6]?.first {
            guard simpleOpaque else { throw instruction.fail("Descriptor arrays cannot have ArrayStride") }
            guard stride > 0 else { throw instruction.fail("Invalid ArrayStride") }
        }
        try addType(id, .bindingArray(base: base, size: size), instruction)
        descriptorArrayElements[id] = element
        return true
    }

    func validateDescriptorArrayPointer(_ pointer: SPVPointer, at instruction: SPVInstruction) throws {
        guard let element = descriptorArrayElements[pointer.base] else { return }
        if try kind(element, instruction).isTextureOrSampler {
            guard pointer.storage == SPVStorageImage.uniformConstant else {
                throw instruction.fail("Image and sampler descriptor arrays require UniformConstant pointers")
            }
        } else {
            let block = isLegacyStoragePointer(pointer) ? SPVLegacyStorage.bufferBlock : SPVLegacyStorage.block
            guard pointer.storage == SPVLegacyStorage.uniform || pointer.storage == SPVLegacyStorage.storageBuffer,
                  decorations[element]?[block] != nil else {
                throw instruction.fail("Buffer descriptor arrays require a Uniform or StorageBuffer pointer to a matching block")
            }
        }
    }
}

extension SPVFunctionLowerer {
    mutating func lowerDescriptorAccess(
        _ instruction: SPVInstruction, base: SPVValue, resultPointer: SPVPointer, rawElement: UInt32,
        statements: inout [IRHandle<IRStatement>]
    ) throws {
        try instruction.count(4)
        let a = instruction.a
        let baseExpression = expressions[base.expression.index]
        guard resultPointer.base == rawElement,
              case .globalVariable = baseExpression.kind,
              case .bindingArray(let element, let size) = module.types[baseExpression.type.index].kind else {
            throw instruction.fail("Descriptor access requires a matching element pointer and a bound global array")
        }
        let index = try value(a[3], instruction, &statements)
        let kind = try module.kind(index.type, instruction)
        guard kind == .scalar(.i32) || kind == .scalar(.u32) else {
            throw instruction.fail("Descriptor array index must be i32 or u32")
        }
        let selection: SPVAccessIndex
        if let constant = module.constantMap[a[3]] {
            guard let offset = constant.nonnegativeInteger else {
                throw instruction.fail("Constant descriptor array index is out of range")
            }
            if case .constant(let count) = size, offset >= count {
                throw instruction.fail("Constant descriptor array index is out of range")
            }
            selection = .constant(offset)
        } else {
            let storage = module.types[element.index].kind.storageTexture != nil
            try checkDescriptorNonUniform([a[1], a[2], a[3]], storage: storage, at: instruction)
            if size == .dynamic, module.usesSimpleMemoryModel,
               !module.capabilities.contains(SPVDescriptorIndexing.runtimeDescriptorArray),
               !module.capabilities.contains(SPVDescriptorIndexing.dynamicSampledArray),
               !module.capabilities.contains(SPVDescriptorIndexing.nonUniformSampledArray) {
                throw instruction.fail("Dynamic selection of a Simple runtime descriptor requires sampled array indexing capability",
                                       code: .unsupportedCapability)
            }
            selection = .dynamic(index)
        }
        let step = SPVAccessStep(type: element, index: selection)
        let expression = try project(base.expression, through: step, statements: &statements, at: instruction)
        accessPaths[expression] = .init(base: base, steps: [step])
        defineValue(a[1], .init(expression: expression, type: a[0], anchor: statements.last))
    }

    func isDynamicallyIndexedDescriptorArrayElement(_ expression: IRHandle<IRExpression>) -> Bool {
        guard let path = accessPaths[expression], case .dynamic? = path.steps.first?.index else { return false }
        let base = expressions[path.base.expression.index]
        if case .bindingArray = module.types[base.type.index].kind { return true }
        return false
    }

    func checkDescriptorNonUniform(_ ids: [UInt32], storage: Bool = false, at instruction: SPVInstruction) throws {
        let capability = storage ? SPVDescriptorIndexing.nonUniformStorageArray : SPVDescriptorIndexing.nonUniformSampledArray
        if ids.contains(where: { module.decorations[$0]?[SPVDescriptorIndexing.nonUniform] != nil }),
           !module.capabilities.contains(capability) {
            let name = storage ? "StorageImageArrayNonUniformIndexing" : "SampledImageArrayNonUniformIndexing"
            throw instruction.fail("Non-uniform descriptor access requires \(name)", code: .unsupportedCapability)
        }
    }

    mutating func checkBufferDescriptorNonUniform(
        _ ids: [UInt32], through expression: IRHandle<IRExpression>, at instruction: SPVInstruction
    ) throws {
        guard ids.contains(where: { module.decorations[$0]?[SPVDescriptorIndexing.nonUniform] != nil }) else { return }
        var cursor = expression
        while let path = accessPaths[cursor] {
            try work()
            cursor = path.base.expression
            guard case .pointer(let type, let space) = module.types[expressions[cursor.index].type.index].kind,
                  case .bindingArray = module.types[type.index].kind else { continue }
            // A divergent offset inside a statically chosen buffer is not divergent descriptor selection.
            guard let first = path.steps.first, case .dynamic = first.index else { return }
            let capability: UInt32
            let name: String
            switch space {
            case .uniform:
                capability = SPVDescriptorIndexing.nonUniformUniformBufferArray
                name = "UniformBufferArrayNonUniformIndexing"
            case .storage:
                capability = SPVDescriptorIndexing.nonUniformStorageBufferArray
                name = "StorageBufferArrayNonUniformIndexing"
            default: return
            }
            guard module.capabilities.contains(capability) else {
                throw instruction.fail("Non-uniform descriptor access requires \(name)", code: .unsupportedCapability)
            }
            return
        }
    }
}
