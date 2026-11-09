import SAGAShaderIR

struct SPIRVBufferGlobal {
    let id: UInt32
    /// Member index to prepend when addressing the original WGSL value.
    let wrapperMember: UInt32?
}

struct SPIRVResourceGlobal {
    let id: UInt32
    let valueType: UInt32
    /// Count supplied by the binding layout for an originally unbounded descriptor array.
    let layoutResolvedCount: UInt32?
}

struct SPIRVBufferArrayGlobal {
    let id: UInt32
    let storageClass: UInt32
    let blockType: UInt32
    let element: IRHandle<IRType>
    let layoutResolvedCount: UInt32?
}

struct SPIRVDeclarationState {
    fileprivate struct VectorKey: Hashable {
        let width: UInt8
        let scalar: IRScalar
    }
    fileprivate struct PointerKey: Hashable {
        let storageClass: UInt32
        let pointee: UInt32
    }
    fileprivate struct DescriptorArrayKey: Hashable {
        let element: UInt32
        let count: UInt32?
    }
    fileprivate struct MatrixKey: Hashable {
        let column: UInt32
        let count: UInt32
    }
    fileprivate struct ImageKey: Hashable {
        let scalar: IRScalar
        let dimension: UInt32
        let depth: UInt32
        let arrayed: UInt32
        let multisampled: UInt32
        let sampled: UInt32
        let format: UInt32
    }
    fileprivate struct LiteralKey: Hashable {
        let scalar: IRScalar
        let bits: UInt64
    }
    fileprivate struct EncodedConstant {
        let id: UInt32
        var lanes: [UInt32]? = nil
    }

    fileprivate var types: [IRHandle<IRType>: UInt32] = [:]
    fileprivate var cooperativeScalars: [UInt32: IRScalar] = [:]
    fileprivate var scalars: [IRScalar: UInt32] = [:]
    fileprivate var vectors: [VectorKey: UInt32] = [:]
    fileprivate var matrices: [MatrixKey: UInt32] = [:]
    fileprivate var pointers: [PointerKey: UInt32] = [:]
    fileprivate var descriptorArrays: [DescriptorArrayKey: UInt32] = [:]
    fileprivate var images: [ImageKey: UInt32] = [:]
    fileprivate var sampledImages: [UInt32: UInt32] = [:]
    fileprivate var sampler: UInt32?
    fileprivate var literals: [LiteralKey: UInt32] = [:]
    fileprivate var zeroes: [UInt32: UInt32] = [:]
    fileprivate var constantExpressions: [IRHandle<IRExpression>: EncodedConstant] = [:]
    fileprivate var globals: [IRHandle<IRGlobalVariable>: SPIRVBufferGlobal] = [:]
    fileprivate var resources: [IRHandle<IRGlobalVariable>: SPIRVResourceGlobal] = [:]
    fileprivate var bufferArrays: [IRHandle<IRGlobalVariable>: SPIRVBufferArrayGlobal] = [:]
    fileprivate var bufferBlocks: [IRHandle<IRType>: UInt32] = [:]
    fileprivate var workgroupGlobals: [IRHandle<IRGlobalVariable>: UInt32] = [:]
    fileprivate var privateGlobals: [IRHandle<IRGlobalVariable>: UInt32] = [:]
    fileprivate var workgroupTypes: [IRHandle<IRType>: UInt32] = [:]
}

extension SPIRVBinaryWriter {
    mutating func scalarTypeID(_ scalar: IRScalar) throws -> UInt32 {
        if let id = declarationState.scalars[scalar] { return id }
        let id = try allocateID()
        switch scalar {
        case .bool: try declare(20, [id])
        case .i16, .u16:
            try requireCapability(22) // Int16
            try declare(21, [id, 16, scalar == .i16 ? 1 : 0])
        case .i32: try declare(21, [id, 32, 1])
        case .u32: try declare(21, [id, 32, 0])
        case .i64, .u64:
            try requireCapability(11) // Int64
            try declare(21, [id, 64, scalar == .i64 ? 1 : 0])
        case .f16:
            try requireCapability(9) // Float16
            try declare(22, [id, 16])
        case .f32: try declare(22, [id, 32])
        case .f64:
            try requireCapability(10) // Float64
            try declare(22, [id, 64])
        default: throw ShaderFailure.unsupported("SPIR-V scalar type \(scalar.kind.rawValue) with width \(Int(scalar.width) * 8)")
        }
        declarationState.scalars[scalar] = id
        return id
    }

    mutating func vectorTypeID(size: IRVectorSize, scalar: IRScalar) throws -> UInt32 {
        let key = SPIRVDeclarationState.VectorKey(width: size.rawValue, scalar: scalar)
        if let id = declarationState.vectors[key] { return id }
        let element = try scalarTypeID(scalar)
        let id = try allocateID()
        try declare(23, [id, element, UInt32(size.rawValue)])
        declarationState.vectors[key] = id
        return id
    }

    mutating func scalarOrVectorTypeID(_ scalar: IRScalar, size: IRVectorSize?) throws -> UInt32 {
        if let size { return try vectorTypeID(size: size, scalar: scalar) }
        return try scalarTypeID(scalar)
    }

    mutating func pointerTypeID(storageClass: UInt32, pointee: UInt32) throws -> UInt32 {
        let key = SPIRVDeclarationState.PointerKey(storageClass: storageClass, pointee: pointee)
        if let id = declarationState.pointers[key] { return id }
        let id = try allocateID()
        try declare(32, [id, storageClass, pointee])
        declarationState.pointers[key] = id
        return id
    }

    private mutating func matrixTypeID(columns: IRVectorSize, rows: IRVectorSize, scalar: IRScalar) throws -> UInt32 {
        let key = SPIRVDeclarationState.MatrixKey(column: try vectorTypeID(size: rows, scalar: scalar),
                                                  count: UInt32(columns.rawValue))
        if let id = declarationState.matrices[key] { return id }
        let id = try allocateID()
        try declare(24, [id, key.column, key.count]) // OpTypeMatrix
        declarationState.matrices[key] = id
        return id
    }

    mutating func typeID(_ handle: IRHandle<IRType>) throws -> UInt32 {
        try charge()
        if let id = declarationState.types[handle] { return id }
        try budget.enterDepth()
        defer { try? budget.leaveDepth() }
        let type = try module.types.value(for: handle)
        let id: UInt32
        switch type.kind {
        case .scalar(let scalar), .atomic(let scalar):
            id = try scalarTypeID(scalar)
        case .vector(let size, let scalar):
            id = try vectorTypeID(size: size, scalar: scalar)
        case .matrix(let columns, let rows, let scalar):
            id = try matrixTypeID(columns: columns, rows: rows, scalar: scalar)
        case .cooperativeMatrix(let columns, let rows, let scalar, let role):
            try requireCooperativeMatrix()
            let component = try scalarTypeID(scalar)
            let scope = try unsignedConstantID(3) // Subgroup
            let rowCount = try unsignedConstantID(UInt32(rows.rawValue))
            let columnCount = try unsignedConstantID(UInt32(columns.rawValue))
            let use: UInt32 = switch role { case .a: 0; case .b: 1; case .c: 2 }
            let useID = try unsignedConstantID(use)
            id = try allocateID()
            try declare(4456, [id, component, scope, rowCount, columnCount, useID]) // OpTypeCooperativeMatrixKHR
            declarationState.cooperativeScalars[id] = scalar
        case .array(let element, let size, let stride):
            let elementID = try typeID(element)
            switch size {
            case .constant(let count):
                let length = try unsignedConstantID(count)
                id = try allocateID()
                try declare(28, [id, elementID, length])
            case .dynamic:
                id = try allocateID()
                try declare(29, [id, elementID])
            case .override:
                throw ShaderFailure.unsupported("SPIR-V array size must be specialized")
            }
            try decorate(71, [id, 6, stride]) // ArrayStride
            try debugName(id, type.name)
        case .structure(let members, let span):
            try charge(members.count)
            let memberTypes = try members.map { try typeID($0.type) }
            id = try allocateID()
            try declare(30, [id] + memberTypes)
            if try hasRuntimeArrayTail(type.kind) {
                try decorate(71, [id, 2]) // Vulkan runtime-array tails must be direct members of a Block.
            }
            for (index, member) in members.enumerated() {
                try decorateBufferMember(of: id, index: UInt32(index), type: member.type, offset: member.offset)
            }
            let name = members.isEmpty ? type.name : SPIRVStructureSpanName.encode(span: span, name: type.name)
            try debugName(id, name)
        case .pointer(let base, let space):
            let pointee = try space == .workgroup ? workgroupTypeID(base) : typeID(base)
            id = try pointerTypeID(storageClass: declarationStorageClass(space), pointee: pointee)
        case .valuePointer(let size, let scalar, let space):
            let pointee: UInt32
            if let size { pointee = try vectorTypeID(size: size, scalar: scalar) }
            else { pointee = try scalarTypeID(scalar) }
            id = try pointerTypeID(storageClass: declarationStorageClass(space), pointee: pointee)
        case .image(let dimension, let arrayed, let imageClass):
            id = try imageTypeID(dimension: dimension, arrayed: arrayed, imageClass: imageClass)
        case .bindingArray(let element, .constant(let count)):
            id = try descriptorArrayTypeID(element: element, count: count)
        case .bindingArray(let element, .dynamic):
            id = try descriptorArrayTypeID(element: element, count: nil)
        case .sampler:
            if let sampler = declarationState.sampler {
                id = sampler
            } else {
                id = try allocateID()
                try declare(26, [id]) // OpTypeSampler
                declarationState.sampler = id
            }
        default:
            throw ShaderFailure.unsupported("SPIR-V declaration for this shader type")
        }
        declarationState.types[handle] = id
        return id
    }

    private mutating func descriptorArrayTypeID(element: IRHandle<IRType>, count: UInt32?) throws -> UInt32 {
        let resource = try module.types.value(for: element).kind
        guard count != 0, resource.isTextureOrSampler, !resource.isExternalTexture else {
            throw unsupported("requires an image or sampler binding array")
        }
        return try descriptorArrayTypeID(elementType: typeID(element), count: count)
    }

    private mutating func descriptorArrayTypeID(elementType: UInt32, count: UInt32?) throws -> UInt32 {
        let key = SPIRVDeclarationState.DescriptorArrayKey(element: elementType, count: count)
        if let id = declarationState.descriptorArrays[key] { return id }
        let id = try allocateID()
        if let count {
            let length = try unsignedConstantID(count)
            try declare(28, [id, elementType, length]) // OpTypeArray
        } else {
            try requireRuntimeDescriptors()
            try declare(29, [id, elementType]) // OpTypeRuntimeArray
        }
        declarationState.descriptorArrays[key] = id
        return id
    }

    mutating func sampleMaskArrayTypeID() throws -> UInt32 {
        // A stage-interface array has no buffer stride decoration.
        try descriptorArrayTypeID(elementType: scalarTypeID(.i32), count: 1)
    }

    private mutating func imageTypeID(
        dimension: IRImageDimension, arrayed: Bool, imageClass: IRImageClass
    ) throws -> UInt32 {
        let kind = IRTypeKind.image(dimension: dimension, arrayed: arrayed, class: imageClass)
        let dim: UInt32 = switch dimension {
        case .d1: 0
        case .d2: 1
        case .d3: 2
        case .cube: 3
        }
        let key: SPIRVDeclarationState.ImageKey
        switch imageClass {
        case .sampled, .depth:
            guard let scalar = kind.sampledTextureScalar else { throw unsupported("does not lower this sampled image shape") }
            key = .init(scalar: scalar, dimension: dim, depth: kind.isDepthTexture ? 1 : 0, arrayed: arrayed ? 1 : 0,
                        multisampled: kind.isMultisampled2D ? 1 : 0, sampled: 1, format: 0)
        case .storage(let format, _):
            guard let storage = kind.storageTexture,
                  let number = SPIRVStorageImageFormat.encode(format) else {
                throw unsupported("does not lower this storage image type")
            }
            if format == .r64Uint { try requireInt64StorageImage() }
            key = .init(scalar: storage.scalar, dimension: dim, depth: 0, arrayed: arrayed ? 1 : 0,
                        multisampled: 0, sampled: 2, format: number)
        case .external:
            throw unsupported("does not lower external textures")
        }
        if let id = declarationState.images[key] { return id }
        if dimension == .d1 { try requireCapability(key.sampled == 1 ? 43 : 44) } // Sampled1D or Image1D
        if dimension == .cube && arrayed { try requireCapability(45) } // SampledCubeArray
        if SPIRVStorageImageFormat.requiresExtendedFormats(key.format) { try requireCapability(49) }
        let scalar = try scalarTypeID(key.scalar)
        let id = try allocateID()
        try declare(25, [id, scalar, dim, key.depth, key.arrayed, key.multisampled, key.sampled, key.format])
        declarationState.images[key] = id
        return id
    }

    mutating func sampledImageTypeID(image: UInt32) throws -> UInt32 {
        if let id = declarationState.sampledImages[image] { return id }
        let id = try allocateID()
        try declare(27, [id, image]) // OpTypeSampledImage
        declarationState.sampledImages[image] = id
        return id
    }

    mutating func literalID(_ literal: IRLiteral) throws -> UInt32 {
        let key: SPIRVDeclarationState.LiteralKey
        switch literal {
        case .bool(let value): key = .init(scalar: .bool, bits: value ? 1 : 0)
        case .i16(let value): key = .init(scalar: .i16, bits: UInt64(UInt32(bitPattern: Int32(value))))
        case .u16(let value): key = .init(scalar: .u16, bits: UInt64(value))
        case .i32(let value): key = .init(scalar: .i32, bits: UInt64(UInt32(bitPattern: value)))
        case .u32(let value): key = .init(scalar: .u32, bits: UInt64(value))
        case .i64(let value): key = .init(scalar: .i64, bits: UInt64(bitPattern: value))
        case .u64(let value): key = .init(scalar: .u64, bits: value)
        case .f16Bits(let bits): key = .init(scalar: .f16, bits: UInt64(bits))
        case .f32(let value): key = .init(scalar: .f32, bits: UInt64(value.bitPattern))
        case .f64(let value): key = .init(scalar: .f64, bits: value.bitPattern)
        default: throw ShaderFailure.unsupported("SPIR-V literal type")
        }
        if let id = declarationState.literals[key] { return id }
        let type = try scalarTypeID(key.scalar)
        let id = try allocateID()
        if key.scalar == .bool {
            try declare(key.bits == 0 ? 42 : 41, [type, id])
        } else {
            var words = [type, id, UInt32(truncatingIfNeeded: key.bits)]
            if key.scalar.width == 8 { words.append(UInt32(key.bits >> 32)) }
            try declare(43, words)
        }
        declarationState.literals[key] = id
        return id
    }

    mutating func unsignedConstantID(_ value: UInt32) throws -> UInt32 {
        try literalID(.u32(value))
    }

    mutating func splatLiteralID(_ literal: IRLiteral, type: UInt32, size: IRVectorSize?) throws -> UInt32 {
        let scalar = try literalID(literal)
        guard let size else { return scalar }
        let vector = try allocateID()
        try declare(44, [type, vector] + Array(repeating: scalar, count: Int(size.rawValue))) // OpConstantComposite
        return vector
    }

    mutating func zeroID(typeID: UInt32) throws -> UInt32 {
        if let id = declarationState.zeroes[typeID] { return id }
        if let scalar = declarationState.cooperativeScalars[typeID] {
            let component = try literalID(scalar == .f16 ? .f16Bits(0) : .f32(0))
            let id = try allocateID()
            try declare(44, [typeID, id, component]) // Cooperative matrix constant splat.
            declarationState.zeroes[typeID] = id
            return id
        }
        let id = try allocateID()
        try declare(46, [typeID, id])
        declarationState.zeroes[typeID] = id
        return id
    }

    mutating func constantID(_ handle: IRHandle<IRConstant>) throws -> UInt32 {
        let constant = try module.constants.value(for: handle)
        return try encodedConstant(constant.initializer).id
    }

    private mutating func encodedConstant(
        _ handle: IRHandle<IRExpression>
    ) throws -> SPIRVDeclarationState.EncodedConstant {
        try charge()
        if let cached = declarationState.constantExpressions[handle] { return cached }
        try budget.enterDepth()
        defer { try? budget.leaveDepth() }

        let expression = try module.globalExpressions.value(for: handle)
        let encoded: SPIRVDeclarationState.EncodedConstant
        switch expression.kind {
        case .literal(let literal):
            encoded = .init(id: try literalID(literal))
        case .constant(let reference):
            encoded = try encodedConstant(module.constants.value(for: reference).initializer)
        case .zeroValue(let type):
            let zero = try zeroID(typeID: typeID(type))
            if case .vector(let size, let scalar) = try module.types.value(for: type).kind {
                let lane = try zeroID(typeID: scalarTypeID(scalar))
                encoded = .init(id: zero, lanes: Array(repeating: lane, count: Int(size.rawValue)))
            } else {
                encoded = .init(id: zero)
            }
        case .compose(let type, let components):
            try charge(components.count)
            let typeID = try typeID(type)
            let vector = try module.types.value(for: type).kind
            var constituents: [UInt32] = []
            constituents.reserveCapacity(components.count)
            for component in components {
                let value = try encodedConstant(component)
                if case .vector = vector, let lanes = value.lanes {
                    constituents.append(contentsOf: lanes)
                } else {
                    constituents.append(value.id)
                }
            }
            let id = try allocateID()
            try declare(44, [typeID, id] + constituents)
            if case .vector = vector { encoded = .init(id: id, lanes: constituents) }
            else { encoded = .init(id: id) }
        case .splat(let size, let value):
            let type = try typeID(expression.type)
            let lane = try encodedConstant(value).id
            let lanes = Array(repeating: lane, count: Int(size.rawValue))
            let id = try allocateID()
            try declare(44, [type, id] + lanes)
            encoded = .init(id: id, lanes: lanes)
        default:
            throw ShaderFailure.unsupported("SPIR-V constant expression must be a literal, composite, or zero value")
        }
        declarationState.constantExpressions[handle] = encoded
        return encoded
    }

    mutating func privateGlobal(_ handle: IRHandle<IRGlobalVariable>) throws -> UInt32 {
        try charge()
        if let id = declarationState.privateGlobals[handle] { return id }
        let global = try module.globalVariables.value(for: handle)
        guard global.space == .private, global.binding == nil, global.memoryDecorations.isEmpty else {
            throw unsupported("requires an undecorated private variable")
        }
        let type = try typeID(global.type)
        let pointer = try pointerTypeID(storageClass: 6, pointee: type)
        let initializer = try global.initializer.map { try encodedConstant($0).id } ?? zeroID(typeID: type)
        let id = try allocateID()
        try declare(59, [pointer, id, 6, initializer]) // OpVariable Private
        try debugName(id, global.name)
        declarationState.privateGlobals[handle] = id
        return id
    }

    mutating func workgroupGlobal(_ handle: IRHandle<IRGlobalVariable>) throws -> UInt32 {
        try charge()
        if let id = declarationState.workgroupGlobals[handle] { return id }
        let global = try module.globalVariables.value(for: handle)
        guard global.space == .workgroup, global.binding == nil, global.initializer == nil,
              global.memoryDecorations.isEmpty else {
            throw unsupported("requires an undecorated workgroup variable without an initializer")
        }
        let pointer = try pointerTypeID(storageClass: 4, pointee: workgroupTypeID(global.type))
        let id = try allocateID()
        try declare(59, [pointer, id, 4]) // OpVariable Workgroup
        try debugName(id, global.name)
        declarationState.workgroupGlobals[handle] = id
        return id
    }

    mutating func workgroupTypeID(_ type: IRHandle<IRType>) throws -> UInt32 {
        try charge()
        if let id = declarationState.workgroupTypes[type] { return id }
        try budget.enterDepth()
        defer { try? budget.leaveDepth() }
        let id: UInt32
        // Workgroup variables use implicit layout in Vulkan 1.1.
        switch try module.types.value(for: type).kind {
        case .array(let element, .constant(let count), _):
            let element = try workgroupTypeID(element)
            id = try allocateID()
            try declare(28, [id, element, unsignedConstantID(count)])
        case .structure(let members, _):
            let types = try members.map { try workgroupTypeID($0.type) }
            id = try allocateID()
            try declare(30, [id] + types)
        default:
            return try typeID(type)
        }
        declarationState.workgroupTypes[type] = id
        return id
    }

    mutating func bufferGlobal(_ handle: IRHandle<IRGlobalVariable>) throws -> SPIRVBufferGlobal {
        try charge()
        if let cached = declarationState.globals[handle] { return cached }
        let global = try module.globalVariables.value(for: handle)
        guard (global.space == .immediate) == (global.binding == nil), global.initializer == nil else {
            throw unsupported("invalid uniform, storage or push constant buffer binding")
        }
        let storageClass: UInt32
        let readOnlyStorage: Bool
        switch global.space {
        case .immediate:
            (storageClass, readOnlyStorage) = (9, false)
            try require16BitBufferAccess(global.type, capability: 4435) // StoragePushConstant16
        case .uniform:
            (storageClass, readOnlyStorage) = (2, false)
            // Preserve WGSL offsets and matrix strides. Packed uniforms need Vulkan's
            // uniformBufferStandardLayout feature when they do not fit extended alignment.
            try require16BitBufferAccess(global.type, capability: 4434) // UniformAndStorageBuffer16BitAccess
        case .storage(let access) where access == .load || access == [.load, .store]:
            (storageClass, readOnlyStorage) = (12, access == .load)
            try require16BitBufferAccess(global.type, capability: 4433) // StorageBuffer16BitAccess
        default: throw unsupported("requires a uniform, read-only storage, or read-write storage buffer")
        }
        let type = try module.types.value(for: global.type).kind
        let valueType = try typeID(global.type)
        let blockType: UInt32
        let wrapperMember: UInt32?
        if try hasRuntimeArrayTail(type) {
            blockType = valueType
            wrapperMember = nil
        } else {
            blockType = try allocateID()
            try declare(30, [blockType, valueType])
            try decorate(71, [blockType, 2]) // Block
            try decorateBufferMember(of: blockType, index: 0, type: global.type, offset: 0)
            wrapperMember = 0
        }

        let pointer = try pointerTypeID(storageClass: storageClass, pointee: blockType)
        let id = try allocateID()
        try declare(59, [pointer, id, storageClass]) // OpVariable
        if let binding = global.binding {
            try decorate(71, [id, 34, binding.group])
            try decorate(71, [id, 33, binding.binding])
        }
        if readOnlyStorage { try decorate(71, [id, 24]) } // NonWritable
        try decorateStorageMemory(id, flags: global.memoryDecorations)
        try debugName(id, global.name)
        let result = SPIRVBufferGlobal(id: id, wrapperMember: wrapperMember)
        declarationState.globals[handle] = result
        return result
    }

    mutating func resourceGlobal(_ handle: IRHandle<IRGlobalVariable>) throws -> SPIRVResourceGlobal {
        try charge()
        if let cached = declarationState.resources[handle] { return cached }
        let global = try module.globalVariables.value(for: handle)
        guard global.space == .handle, let binding = global.binding, global.initializer == nil,
              global.memoryDecorations.isEmpty else {
            throw unsupported("requires an undecorated texture or sampler binding")
        }
        let kind = try module.types.value(for: global.type).kind
        let element: IRTypeKind
        let valueType: UInt32
        let layoutResolvedCount: UInt32?
        if case .bindingArray(let type, let size) = kind {
            element = try module.types.value(for: type).kind
            if size == .dynamic, bindingArrayLengths.isEmpty {
                valueType = try typeID(global.type)
                layoutResolvedCount = nil
            } else {
                let count = try resourceBindingArrayCount(size, binding: binding, location: global.location,
                                                           bindingArrayLengths: bindingArrayLengths)
                valueType = try descriptorArrayTypeID(element: type, count: count)
                layoutResolvedCount = size == .dynamic ? count : nil
            }
        } else {
            element = kind
            guard element.isTextureOrSampler else { throw unsupported("does not lower this resource type") }
            valueType = try typeID(global.type)
            layoutResolvedCount = nil
        }
        let pointer = try pointerTypeID(storageClass: 0, pointee: valueType)
        let id = try allocateID()
        try declare(59, [pointer, id, 0]) // UniformConstant
        try decorate(71, [id, 34, binding.group])
        try decorate(71, [id, 33, binding.binding])
        if let storage = element.storageTexture {
            if !storage.access.contains(.load) { try decorate(71, [id, 25]) } // NonReadable
            if !storage.access.contains(.store) { try decorate(71, [id, 24]) } // NonWritable
        }
        try debugName(id, global.name)
        let resource = SPIRVResourceGlobal(id: id, valueType: valueType, layoutResolvedCount: layoutResolvedCount)
        declarationState.resources[handle] = resource
        return resource
    }

    mutating func bufferArrayGlobal(_ handle: IRHandle<IRGlobalVariable>) throws -> SPIRVBufferArrayGlobal {
        try charge()
        if let existing = declarationState.bufferArrays[handle] { return existing }
        let global = try module.globalVariables.value(for: handle)
        guard let binding = global.binding, global.initializer == nil,
              case .bindingArray(let element, let size) = try module.types.value(for: global.type).kind else {
            throw unsupported("requires a bound uniform or storage buffer descriptor array")
        }
        let storageClass: UInt32
        let readOnlyStorage: Bool
        switch global.space {
        case .uniform:
            if !module.uniformBufferStandardLayout { try requireExtendedUniformMatrixLayout(element) }
            try require16BitBufferAccess(element, capability: 4434) // UniformAndStorageBuffer16BitAccess
            (storageClass, readOnlyStorage) = (2, false)
        case .storage(let access) where access == .load || access == [.load, .store]:
            try require16BitBufferAccess(element, capability: 4433)
            (storageClass, readOnlyStorage) = (12, access == .load)
        default: throw unsupported("does not lower this buffer descriptor address space")
        }
        let count: UInt32?
        if size == .dynamic, bindingArrayLengths.isEmpty {
            count = nil
        } else {
            count = try resourceBindingArrayCount(size, binding: binding, location: global.location,
                                                   bindingArrayLengths: bindingArrayLengths)
        }
        let layoutResolvedCount = size == .dynamic ? count : nil
        // OpAccessChain interprets even unsigned 32-bit indices as signed.
        if let layoutResolvedCount, layoutResolvedCount > 0x8000_0000 {
            throw ShaderFailure.diagnostics([.init(severity: .error, code: .invalidBinding,
                message: "SPIR-V buffer descriptor layout count exceeds the 32-bit access range", location: global.location)])
        }
        let block = try bufferDescriptorBlock(element)
        let array = try descriptorArrayTypeID(elementType: block, count: count)
        let pointer = try pointerTypeID(storageClass: storageClass, pointee: array)
        let id = try allocateID()
        try declare(59, [pointer, id, storageClass]) // OpVariable
        try decorate(71, [id, 34, binding.group])
        try decorate(71, [id, 33, binding.binding])
        if readOnlyStorage { try decorate(71, [id, 24]) } // NonWritable
        try decorateStorageMemory(id, flags: global.memoryDecorations)
        try debugName(id, global.name)
        let result = SPIRVBufferArrayGlobal(id: id, storageClass: storageClass, blockType: block,
                                           element: element, layoutResolvedCount: layoutResolvedCount)
        declarationState.bufferArrays[handle] = result
        return result
    }

    private mutating func decorateStorageMemory(_ id: UInt32, flags: IRMemoryDecorations) throws {
        if flags.contains(.coherent) || flags.contains(.volatile) { markCoherentOrVolatileStorage() }
        if flags.contains(.volatile) { try decorate(71, [id, 21]) }
        if flags.contains(.coherent) { try decorate(71, [id, 23]) }
    }

    private mutating func bufferDescriptorBlock(_ type: IRHandle<IRType>) throws -> UInt32 {
        let kind = try module.types.value(for: type).kind
        guard case .structure(let members, let span) = kind else {
            throw unsupported("requires a structure for each buffer descriptor")
        }
        if try hasRuntimeArrayTail(kind) { return try typeID(type) }
        if let existing = declarationState.bufferBlocks[type] { return existing }
        // A fixed-size structure may also appear as a value or inside another buffer.
        // Give its interface Block a separate type so those uses can retain the original type.
        try charge(members.count)
        let memberTypes = try members.map { try typeID($0.type) }
        let block = try allocateID()
        try declare(30, [block] + memberTypes)
        try decorate(71, [block, 2]) // Block
        for (index, member) in members.enumerated() {
            try decorateBufferMember(of: block, index: UInt32(index), type: member.type, offset: member.offset)
        }
        if !members.isEmpty { try debugName(block, SPIRVStructureSpanName.encode(span: span, name: nil)) }
        declarationState.bufferBlocks[type] = block
        return block
    }

    private mutating func decorateBufferMember(
        of structure: UInt32, index: UInt32, type: IRHandle<IRType>, offset: UInt32
    ) throws {
        try decorate(72, [structure, index, 35, offset]) // Offset
        var element = type
        while true {
            try charge()
            switch try module.types.value(for: element).kind {
            case .array(let child, _, _): element = child
            case .matrix(_, let rows, let scalar):
                let stride = matrixColumnStride(rows: rows, scalar: scalar)
                try decorate(72, [structure, index, 5]) // ColMajor
                try decorate(72, [structure, index, 7, stride]) // MatrixStride
                return
            default: return
            }
        }
    }

    private mutating func requireExtendedUniformMatrixLayout(_ type: IRHandle<IRType>) throws {
        var pending = [type]
        var seen: Set<IRHandle<IRType>> = []
        while let next = pending.popLast() {
            try charge()
            guard seen.insert(next).inserted else { continue }
            switch try module.types.value(for: next).kind {
            case .array(let element, _, _): pending.append(element)
            case .structure(let members, _):
                try charge(members.count)
                pending.append(contentsOf: members.map(\.type))
            case .matrix(_, let rows, let scalar) where matrixColumnStride(rows: rows, scalar: scalar) < 16:
                throw unsupported("requires uniform_buffer_standard_layout for packed uniform matrix columns")
            default: break
            }
        }
    }

    private func matrixColumnStride(rows: IRVectorSize, scalar: IRScalar) -> UInt32 {
        UInt32(rows == .three ? 4 : rows.rawValue) * UInt32(scalar.width)
    }

    private mutating func require16BitBufferAccess(_ type: IRHandle<IRType>, capability: UInt32) throws {
        var pending = [type]
        var seen: Set<IRHandle<IRType>> = []
        while let next = pending.popLast() {
            try charge()
            guard seen.insert(next).inserted else { continue }
            let kind = try module.types.value(for: next).kind
            if kind.usesFloat16 || IRBitcastShape(kind)?.scalar.width == 2 {
                try requireCapability(capability)
                return
            }
            switch kind {
            case .array(let element, _, _): pending.append(element)
            case .structure(let members, _):
                try charge(members.count)
                pending.append(contentsOf: members.map(\.type))
            default: break
            }
        }
    }

    private func hasRuntimeArrayTail(_ type: IRTypeKind) throws -> Bool {
        guard case .structure(let members, _) = type, let last = members.last,
              case .array(_, .dynamic, _) = try module.types.value(for: last.type).kind else { return false }
        return true
    }

    private func declarationStorageClass(_ space: IRAddressSpace) throws -> UInt32 {
        switch space {
        case .uniform: 2
        case .private: 6
        case .function: 7
        case .immediate: 9
        case .workgroup: 4
        case .storage: 12
        default: throw ShaderFailure.unsupported("SPIR-V pointer address space")
        }
    }
}
