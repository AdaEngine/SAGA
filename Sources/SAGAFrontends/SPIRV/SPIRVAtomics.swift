import SAGAShaderIR

enum SPVFloatAtomicAdd {
    static let extensionName = "SPV_EXT_shader_atomic_float_add"
    static let capability: UInt32 = 6033
    static let opcode: UInt32 = 6035
}

enum SPVAtomicStep: Hashable {
    case member(UInt32)
    case element
}

struct SPVDiscardableAtomicResult {
    let atomic: IRHandle<IRStatement>
    let aliasCast: IRHandle<IRStatement>?
}

private extension IRScalar {
    var isSPVAtomicInteger: Bool { [.i32, .u32, .i64, .u64].contains(self) }
}

extension SPVFunctionLowerer {
    mutating func lowerAtomic(_ instruction: SPVInstruction, _ statements: inout [IRHandle<IRStatement>]) throws {
        let opcode = instruction.op
        let a = instruction.a
        let store = opcode == 228
        let increment = opcode == 232 || opcode == 233
        try instruction.count(opcode == 230 ? 8 : store ? 4 : opcode == 227 || increment ? 5 : 6)
        let pointerIndex = store ? 0 : 2
        if let rawType = module.functionValueTypes[a[pointerIndex]],
           module.pointers[rawType]?.storage == SPVImageTexelPointer.storageClass {
            try lowerImageAtomic(instruction, &statements)
            return
        }
        let pointer = try value(a[pointerIndex], instruction, &statements)
        guard let pointee = module.pointers[pointer.type],
              case .pointer(let actual, let space) = module.types[expressions[pointer.expression.index].type.index].kind,
              case .scalar(let scalar) = try module.kind(pointee.base, instruction), scalar.isSPVAtomicInteger || scalar == .f32 else {
            throw instruction.fail("Atomics require a supported scalar in storage or workgroup memory", code: .unsupportedConstruct)
        }
        try module.validateInt64AtomicCapability(scalar, at: instruction)
        guard case .scalar(let memoryScalar) = module.types[actual.index].kind,
              memoryScalar == scalar || (memoryScalar == .u32 && scalar == .f32) ||
                  (memoryScalar.isSPVAtomicInteger && scalar.isSPVAtomicInteger && memoryScalar.width == scalar.width) else {
            throw instruction.fail("Unsupported physical scalar for an atomic view",
                                   code: .unsupportedConstruct)
        }
        let declared = try module.type(pointee.base, instruction)
        let inStorage = if case .storage = space { true } else { false }
        guard inStorage || space == .workgroup else {
            throw instruction.fail("Atomics require a supported scalar in storage or workgroup memory", code: .unsupportedConstruct)
        }
        if opcode == SPVFloatAtomicAdd.opcode {
            guard module.extensions.contains(SPVFloatAtomicAdd.extensionName), module.capabilities.contains(SPVFloatAtomicAdd.capability) else {
                throw instruction.fail("OpAtomicFAddEXT requires SPV_EXT_shader_atomic_float_add and AtomicFloat32AddEXT",
                                       code: .unsupportedCapability)
            }
            guard scalar == .f32, inStorage else {
                throw instruction.fail("OpAtomicFAddEXT requires a float32 storage pointer", code: .unsupportedConstruct)
            }
        } else if scalar == .f32 {
            guard opcode == 227 || opcode == 228 || (opcode == 229 && inStorage) else {
                throw instruction.fail("Float32 atomics support load, store and storage exchange", code: .unsupportedConstruct)
            }
        }
        let scope = module.constantMap[a[pointerIndex + 1]]?.nonnegativeInteger
        let semantics = module.constantMap[a[pointerIndex + 2]]?.nonnegativeInteger
        guard semantics == 0, try module.supportsRelaxedAtomicScope(scope, inStorage: inStorage, at: instruction) else {
            throw instruction.fail("Only relaxed SPIR-V atomics at workgroup or supported storage scope are implemented",
                                   code: .unsupportedConstruct)
        }
        try checkBufferDescriptorNonUniform([a[pointerIndex]], through: pointer.expression, at: instruction)
        let requiredAccess: IRStorageAccess = store ? .store : opcode == 227 ? .load : [.load, .store]
        try checkStorageAccess(pointer, requiring: requiredAccess, at: instruction)
        try recordAtomicPath(pointer.expression, at: instruction)
        if store {
            let operand = try atomicStorageOperand(a[3], declaredID: pointee.base, actual: actual, instruction, &statements)
            statements.append(try stmt(.store(pointer: pointer.expression, value: operand), instruction))
            return
        }
        guard try module.type(a[0], instruction) == declared else {
            throw instruction.fail("Atomic result type must match the pointee")
        }
        if opcode == 227 {
            try atomicValueResult(instruction, .load(pointer: pointer.expression), actual: actual, declared: declared, &statements)
            return
        }
        if opcode == 230 {
            guard module.constantMap[a[5]]?.nonnegativeInteger == 0 else {
                throw instruction.fail("Only relaxed compare-exchange failure semantics are implemented",
                                       code: .unsupportedConstruct)
            }
            let operand = try atomicStorageOperand(a[6], declaredID: pointee.base, actual: actual, instruction, &statements)
            let compare = try atomicStorageOperand(a[7], declaredID: pointee.base, actual: actual, instruction, &statements)
            // The raw SPIR-V declaration only exposes the declared result signedness during decode.
            module.budget = budget
            let pair = try module.internAtomicCompareType(memoryScalar, type: actual)
            budget = module.budget
            let pairResult = try expr(pair, .atomicResult(type: pair, comparison: true), instruction)
            statements.append(try stmt(.atomic(pointer: pointer.expression, function: .exchange(compare: compare, strong: true),
                                               value: operand, result: pairResult), instruction))
            try atomicValueResult(instruction, .accessIndex(base: pairResult, index: 0), actual: actual, declared: declared, &statements)
            return
        }
        let operation: IRAtomicFunction
        switch opcode {
        case 229: operation = .exchange(compare: nil)
        case 232, 234: operation = .add
        case SPVFloatAtomicAdd.opcode: operation = memoryScalar == .u32 ? .addFloat32Bits : .add
        case 233, 235: operation = .subtract
        case 236: operation = memoryScalar.kind == .sint ? .min : .minWithOppositeSignedness
        case 237: operation = memoryScalar.kind == .uint ? .min : .minWithOppositeSignedness
        case 238: operation = memoryScalar.kind == .sint ? .max : .maxWithOppositeSignedness
        case 239: operation = memoryScalar.kind == .uint ? .max : .maxWithOppositeSignedness
        case 240: operation = .and
        case 241: operation = .inclusiveOr
        case 242: operation = .exclusiveOr
        default: throw instruction.fail("Unsupported atomic opcode", code: .unsupportedConstruct)
        }
        let operand: IRHandle<IRExpression>
        if increment {
            let one: IRLiteral = switch memoryScalar {
            case .i32: .i32(1)
            case .u32: .u32(1)
            case .i64: .i64(1)
            case .u64: .u64(1)
            default: throw instruction.fail("Atomic increment/decrement requires an integer")
            }
            operand = try expr(actual, .literal(one), instruction)
        } else {
            operand = try atomicStorageOperand(a[5], declaredID: pointee.base, actual: actual, instruction, &statements)
        }
        let original = try expr(actual, .atomicResult(type: actual, comparison: false), instruction)
        let anchor = try stmt(.atomic(pointer: pointer.expression, function: operation, value: operand, result: original), instruction)
        statements.append(anchor)
        let aliasCast: IRHandle<IRStatement>?
        if memoryScalar == scalar {
            defineValue(a[1], .init(expression: original, type: a[0], anchor: anchor))
            aliasCast = nil
        } else {
            try result(instruction, .bitcast(expression: original, type: declared), &statements)
            aliasCast = statements.last
        }
        if memoryScalar.width == 8, inStorage, operation == .min || operation == .max {
            discardableAtomicResults[a[1]] = .init(atomic: anchor, aliasCast: aliasCast)
        }
    }

    mutating func discardUnusedAtomicResults() throws {
        // Unused wide storage extrema do not require the full atomic operation tier.
        // Keep statement handles stable for block and spill bookkeeping.
        for result in discardableAtomicResults.values {
            try work()
            let atomic = statements[result.atomic.index]
            guard case .atomic(let pointer, let function, let value, let oldValue?) = atomic.kind else {
                throw ShaderFailure.invalidHandle
            }
            statements[result.atomic.index] = .init(
                kind: .atomic(pointer: pointer, function: function, value: value, result: nil), location: atomic.location)
            let original = expressions[oldValue.index]
            expressions[oldValue.index] = .init(type: original.type, kind: .zeroValue(original.type), location: original.location)
            if let cast = result.aliasCast {
                statements[cast.index] = .init(kind: .emit([]), location: statements[cast.index].location)
            }
        }
    }

    private mutating func atomicStorageOperand(
        _ id: UInt32, declaredID: UInt32, actual: IRHandle<IRType>,
        _ instruction: SPVInstruction, _ statements: inout [IRHandle<IRStatement>]
    ) throws -> IRHandle<IRExpression> {
        let value = try typed(id, declaredID, instruction, &statements)
        let declared = try module.type(declaredID, instruction)
        return try reinterpretStorageAlias(value, from: declared, to: actual, instruction, &statements)
    }

    private mutating func atomicValueResult(
        _ instruction: SPVInstruction, _ kind: IRExpressionKind, actual: IRHandle<IRType>,
        declared: IRHandle<IRType>, _ statements: inout [IRHandle<IRStatement>]
    ) throws {
        if module.types[actual.index].kind == module.types[declared.index].kind {
            try result(instruction, kind, &statements)
        } else {
            let value = try emitValue(actual, kind, &statements, instruction)
            try result(instruction, .bitcast(expression: value, type: declared), &statements)
        }
    }

    private mutating func recordAtomicPath(_ pointer: IRHandle<IRExpression>, at site: SPVInstruction) throws {
        var cursor = pointer
        var reversed: [SPVAtomicStep] = []
        while true {
            try work()
            let base: IRHandle<IRExpression>
            let index: UInt32?
            switch expressions[cursor.index].kind {
            case .globalVariable(let global):
                atomicPaths[global, default: []].insert(Array(reversed.reversed()))
                return
            case .accessIndex(let root, let member):
                base = root
                index = member
            case .access(let root, _):
                base = root
                index = nil
            default:
                throw site.fail("Atomic pointer must originate in a global variable", code: .unsupportedConstruct)
            }
            guard case .pointer(let pointee, _) = module.types[expressions[base.index].type.index].kind else {
                throw site.fail("Atomic access base must be a pointer")
            }
            switch module.types[pointee.index].kind {
            case .array, .bindingArray: reversed.append(.element)
            case .structure where index != nil:
                if let index { reversed.append(.member(index)) }
            default:
                throw site.fail("Only atomic array elements and structure fields are implemented", code: .unsupportedConstruct)
            }
            cursor = base
        }
    }
}

extension SPVDecoder {
    func supportsRelaxedAtomicScope(_ scope: UInt32?, inStorage: Bool, at site: SPVInstruction) throws -> Bool {
        switch scope {
        case 2: // Workgroup
            return true
        case 5: // QueueFamily is the scope of WGSL storage atomics with the Vulkan memory model.
            return inStorage && usesVulkanMemoryModel
        case 1 where inStorage: // Device
            guard !usesVulkanMemoryModel || capabilities.contains(SPVVulkanMemoryModel.deviceScopeCapability) else {
                throw site.fail("Vulkan Device atomic scope requires VulkanMemoryModelDeviceScope",
                                code: .unsupportedCapability)
            }
            return true
        default:
            return false
        }
    }

    fileprivate func validateInt64AtomicCapability(_ scalar: IRScalar, at site: SPVInstruction) throws {
        if scalar.width == 8, !capabilities.contains(SPVImageTexelPointer.int64AtomicsCapability) {
            throw site.fail("64-bit atomics require Int64Atomics", code: .unsupportedCapability)
        }
    }

    mutating func prepareAtomicCompareType(at site: SPVInstruction) throws {
        try site.count(8)
        let type = try type(site.a[0], site)
        guard case .scalar(let scalar) = types[type.index].kind, scalar.isSPVAtomicInteger else {
            throw site.fail("Atomic compare-exchange requires a supported integer result", code: .unsupportedConstruct)
        }
        try validateInt64AtomicCapability(scalar, at: site)
        _ = try internAtomicCompareType(scalar, type: type)
    }

    mutating func internAtomicCompareType(_ scalar: IRScalar, type: IRHandle<IRType>) throws -> IRHandle<IRType> {
        let kind = IRPredeclaredType.atomicCompareExchangeWeakResult(scalar)
        if let existing = atomicCompareTypes.first(where: { $0.kind == kind }) { return existing.type }
        let boolean = try controlBoolType ?? internAtomicType(.scalar(.bool))
        controlBoolType = boolean
        let width = UInt32(scalar.width)
        let members: [IRStructMember] = [
            .init(name: "old_value", type: type, offset: 0),
            .init(name: "exchanged", type: boolean, offset: width),
        ]
        let result = try internAtomicType(.structure(members: members, span: width * 2))
        atomicCompareTypes.append(.init(kind: kind, type: result))
        return result
    }

    mutating func upgradeAtomicGlobals(in functions: inout [IRFunction], at site: SPVInstruction) throws {
        guard !atomicPaths.isEmpty else { return }
        for index in globals.indices {
            try charge()
            let handle = try IRHandle<IRGlobalVariable>(index: index)
            guard let paths = atomicPaths[handle] else { continue }
            let global = globals[index]
            guard global.initializer == nil else {
                throw site.fail("Atomic globals with explicit initializers are not implemented", code: .unsupportedConstruct)
            }
            let type = try atomicStorageType(global.type, paths: paths.map { $0[...] }, at: site)
            globals[index] = .init(name: global.name, space: global.space, binding: global.binding, type: type,
                                   memoryDecorations: global.memoryDecorations, location: global.location)
        }
        for index in functions.indices {
            let function = functions[index]
            var expressions = function.expressions.values
            var rewritten: [IRHandle<IRExpression>: (base: IRHandle<IRType>, space: IRAddressSpace)] = [:]
            for (index, expression) in expressions.enumerated() {
                try charge()
                let projection: (base: IRHandle<IRType>, space: IRAddressSpace)?
                switch expression.kind {
                case .globalVariable(let global) where atomicPaths[global] != nil:
                    let value = globals[global.index]
                    projection = (value.type, value.space)
                case .access(let base, _):
                    projection = try atomicProjection(rewritten[base], index: nil, at: site)
                case .accessIndex(let base, let member):
                    projection = try atomicProjection(rewritten[base], index: member, at: site)
                default:
                    projection = nil
                }
                guard let projection else { continue }
                let type = try internAtomicType(.pointer(base: projection.base, space: projection.space))
                expressions[index] = .init(type: type, kind: expression.kind, location: expression.location)
                rewritten[try IRHandle(index: index)] = projection
            }
            functions[index] = .init(name: function.name, arguments: function.arguments, result: function.result,
                localVariables: function.localVariables, expressions: .init(expressions), namedExpressions: function.namedExpressions,
                statements: function.statements, blocks: function.blocks, body: function.body,
                diagnosticFilter: function.diagnosticFilter, location: function.location)
        }
    }

    private mutating func atomicStorageType(
        _ type: IRHandle<IRType>, paths: [ArraySlice<SPVAtomicStep>], at site: SPVInstruction
    ) throws -> IRHandle<IRType> {
        try charge(paths.count)
        try budget.enterDepth()
        defer { try? budget.leaveDepth() }
        let kind = types[type.index].kind
        if paths.allSatisfy(\.isEmpty), case .scalar(let scalar) = kind {
            return try internAtomicType(.atomic(scalar))
        }
        switch kind {
        case .array(let element, let size, let stride):
            let upgraded = try atomicStorageType(element, paths: paths.map { $0.dropFirst() }, at: site)
            return try internAtomicType(.array(base: upgraded, size: size, stride: stride))
        case .bindingArray(let element, let size):
            guard paths.allSatisfy({ $0.first == .element }) else { throw site.fail("Invalid atomic descriptor array path") }
            let upgraded = try atomicStorageType(element, paths: paths.map { $0.dropFirst() }, at: site)
            return try internAtomicType(.bindingArray(base: upgraded, size: size))
        case .structure(let fields, let span):
            var byMember: [UInt32: [ArraySlice<SPVAtomicStep>]] = [:]
            for path in paths {
                guard case .member(let member)? = path.first else { throw site.fail("Invalid atomic structure path") }
                byMember[member, default: []].append(path.dropFirst())
            }
            var upgraded = fields
            for (index, field) in fields.enumerated() {
                try charge()
                guard let memberPaths = byMember[UInt32(index)] else { continue }
                let fieldType = try atomicStorageType(field.type, paths: memberPaths, at: site)
                upgraded[index] = .init(name: field.name, type: fieldType, binding: field.binding,
                                        offset: field.offset, location: field.location, explicitAlignment: field.explicitAlignment)
            }
            return try internAtomicType(.structure(members: upgraded, span: span))
        default: throw site.fail("Invalid atomic storage type")
        }
    }

    private func atomicProjection(
        _ pointer: (base: IRHandle<IRType>, space: IRAddressSpace)?, index: UInt32?, at site: SPVInstruction
    ) throws -> (base: IRHandle<IRType>, space: IRAddressSpace)? {
        guard let pointer else { return nil }
        switch types[pointer.base.index].kind {
        case .array(let element, _, _), .bindingArray(let element, _):
            return (element, pointer.space)
        case .structure(let fields, _):
            guard let index, UInt64(index) < UInt64(fields.count) else { throw site.fail("Invalid atomic field projection") }
            return (fields[Int(index)].type, pointer.space)
        case .vector, .matrix: return nil
        default: throw site.fail("Invalid access to atomic storage")
        }
    }

    private mutating func internAtomicType(_ kind: IRTypeKind) throws -> IRHandle<IRType> {
        try charge(types.count)
        if let index = types.firstIndex(where: { $0.kind == kind }) { return try IRHandle(index: index) }
        try node()
        let handle = try IRHandle<IRType>(index: types.count)
        types.append(.init(kind: kind))
        return handle
    }
}
