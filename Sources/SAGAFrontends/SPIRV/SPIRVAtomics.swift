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

extension SPVFunctionLowerer {
    mutating func lowerAtomic(_ instruction: SPVInstruction, _ statements: inout [IRHandle<IRStatement>]) throws {
        let opcode = instruction.op
        let a = instruction.a
        let store = opcode == 228
        let increment = opcode == 232 || opcode == 233
        try instruction.count(opcode == 230 ? 8 : store ? 4 : opcode == 227 || increment ? 5 : 6)
        let pointerIndex = store ? 0 : 2
        let pointer = try value(a[pointerIndex], instruction, &statements)
        guard let pointee = module.pointers[pointer.type], pointee.storage == 4 || pointee.storage == 12,
              case .scalar(let scalar) = try module.kind(pointee.base, instruction), [.i32, .u32, .f32].contains(scalar) else {
            throw instruction.fail("Atomics require a supported 32-bit scalar in storage or workgroup memory", code: .unsupportedConstruct)
        }
        if opcode == SPVFloatAtomicAdd.opcode {
            guard module.extensions.contains(SPVFloatAtomicAdd.extensionName), module.capabilities.contains(SPVFloatAtomicAdd.capability) else {
                throw instruction.fail("OpAtomicFAddEXT requires SPV_EXT_shader_atomic_float_add and AtomicFloat32AddEXT",
                                       code: .unsupportedCapability)
            }
            guard scalar == .f32, pointee.storage == 12 else {
                throw instruction.fail("OpAtomicFAddEXT requires a float32 storage pointer", code: .unsupportedConstruct)
            }
        } else if scalar == .f32 {
            guard opcode == 227 || opcode == 228 || (opcode == 229 && pointee.storage == 12) else {
                throw instruction.fail("Float32 atomics support load, store and storage exchange", code: .unsupportedConstruct)
            }
        }
        let scope = module.constantMap[a[pointerIndex + 1]]?.nonnegativeInteger
        let semantics = module.constantMap[a[pointerIndex + 2]]?.nonnegativeInteger
        guard semantics == 0, scope == 2 || (pointee.storage == 12 && scope == 1) else {
            throw instruction.fail("Only relaxed SPIR-V atomics at workgroup or storage device scope are implemented",
                                   code: .unsupportedConstruct)
        }
        try recordAtomicPath(pointer.expression, at: instruction)
        if store {
            let operand = try typed(a[3], pointee.base, instruction, &statements)
            statements.append(try stmt(.store(pointer: pointer.expression, value: operand), instruction))
            return
        }
        guard try module.type(a[0], instruction) == module.type(pointee.base, instruction) else {
            throw instruction.fail("Atomic result type must match the pointee")
        }
        if opcode == 227 {
            try result(instruction, .load(pointer: pointer.expression), &statements)
            return
        }
        if opcode == 230 {
            guard module.constantMap[a[5]]?.nonnegativeInteger == 0 else {
                throw instruction.fail("Only relaxed compare-exchange failure semantics are implemented",
                                       code: .unsupportedConstruct)
            }
            let operand = try typed(a[6], pointee.base, instruction, &statements)
            let compare = try typed(a[7], pointee.base, instruction, &statements)
            let kind = IRPredeclaredType.atomicCompareExchangeWeakResult(scalar)
            guard let pair = module.atomicCompareTypes.first(where: { $0.kind == kind })?.type else {
                throw instruction.fail("Missing atomic comparison result type")
            }
            let pairResult = try expr(pair, .atomicResult(type: pair, comparison: true), instruction)
            statements.append(try stmt(.atomic(pointer: pointer.expression, function: .exchange(compare: compare, strong: true),
                                               value: operand, result: pairResult), instruction))
            try result(instruction, .accessIndex(base: pairResult, index: 0), &statements)
            return
        }
        let operation: IRAtomicFunction
        switch opcode {
        case 229: operation = .exchange(compare: nil)
        case 232, 234, SPVFloatAtomicAdd.opcode: operation = .add
        case 233, 235: operation = .subtract
        case 236 where scalar == .i32, 237 where scalar == .u32: operation = .min
        case 238 where scalar == .i32, 239 where scalar == .u32: operation = .max
        case 240: operation = .and
        case 241: operation = .inclusiveOr
        case 242: operation = .exclusiveOr
        default: throw instruction.fail("Atomic opcode and integer signedness are incompatible", code: .unsupportedConstruct)
        }
        let type = try module.type(a[0], instruction)
        let operand: IRHandle<IRExpression>
        if increment {
            operand = try expr(type, .literal(scalar == .i32 ? .i32(1) : .u32(1)), instruction)
        } else {
            operand = try typed(a[5], pointee.base, instruction, &statements)
        }
        let result = try expr(type, .atomicResult(type: type, comparison: false), instruction)
        let anchor = try stmt(.atomic(pointer: pointer.expression, function: operation, value: operand, result: result), instruction)
        statements.append(anchor)
        defineValue(a[1], .init(expression: result, type: a[0], anchor: anchor))
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
            case .array: reversed.append(.element)
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
    mutating func prepareAtomicCompareType(at site: SPVInstruction) throws {
        try site.count(8)
        let type = try type(site.a[0], site)
        guard case .scalar(let scalar) = types[type.index].kind, scalar == .i32 || scalar == .u32 else {
            throw site.fail("Atomic compare-exchange requires a 32-bit integer result", code: .unsupportedConstruct)
        }
        let kind = IRPredeclaredType.atomicCompareExchangeWeakResult(scalar)
        guard !atomicCompareTypes.contains(where: { $0.kind == kind }) else { return }
        let boolean = try controlBoolType ?? internAtomicType(.scalar(.bool))
        controlBoolType = boolean
        let members: [IRStructMember] = [
            .init(name: "old_value", type: type, offset: 0),
            .init(name: "exchanged", type: boolean, offset: 4),
        ]
        let result = try internAtomicType(.structure(members: members, span: 8))
        atomicCompareTypes.append(.init(kind: kind, type: result))
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
                                        offset: field.offset, location: field.location)
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
        case .array(let element, _, _): return (element, pointer.space)
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
