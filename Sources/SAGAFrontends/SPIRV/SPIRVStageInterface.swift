import SAGAShaderIR

enum SPVBarycentric {
    static let extensionName = "SPV_KHR_fragment_shader_barycentric"
    static let capability: UInt32 = 5284
    static let perVertex: UInt32 = 5285
    static let perspective: UInt32 = 5286
    static let affine: UInt32 = 5287
}

enum SPVMultiview {
    static let extensionName = "SPV_KHR_multiview"
    static let capability: UInt32 = 4439
    static let viewIndex: UInt32 = 4440
}

enum SPVDrawParameters {
    static let extensionName = "SPV_KHR_shader_draw_parameters"
    static let capability: UInt32 = 4427
    static let drawIndex: UInt32 = 4426
}

enum SPVPrimitiveIndex {
    static let builtin: UInt32 = 7
    static let geometryCapability: UInt32 = 2
    static let tessellationCapability: UInt32 = 3
}

struct SPVInterfaceField {
    let global: IRHandle<IRGlobalVariable>
    let pointer: IRHandle<IRType>
    let type: IRHandle<IRType>
    let member: UInt32?
    let attributes: [UInt32: [UInt32]]
    let input: Bool
}

private struct SPVOutputSlot: Hashable {
    let location: UInt32
    let index: UInt32
}

private struct SPVBoundInterface {
    let field: SPVInterfaceField
    var binding: IRBinding
    let type: IRHandle<IRType>
    let memberPointer: IRHandle<IRType>?
    let sampleMaskElement: IRHandle<IRType>?

    var locationSlot: SPVOutputSlot? {
        guard let location = field.attributes[30]?.first else { return nil }
        return .init(location: location, index: field.attributes[32]?.first ?? 0)
    }
}

extension SPVDecoder {
    mutating func registerStageInterface(
        _ id: UInt32, global: IRHandle<IRGlobalVariable>, pointerType: UInt32,
        valueType: UInt32, input: Bool, at instruction: SPVInstruction
    ) throws {
        let attributes = decorations[id, default: [:]]
        let type = try type(valueType, instruction)
        let pointer = try self.type(pointerType, instruction)
        var fields: [SPVInterfaceField] = []
        if case .structure(let fieldsOfType, _) = types[type.index].kind {
            let baseLocation = attributes[30]?.first
            let hasMemberBindings = decorations[valueType]?[2] != nil && attributes.isEmpty
            guard hasMemberBindings || baseLocation != nil else {
                throw instruction.fail("Interface structures require member decorations", code: .unsupportedConstruct)
            }
            interfaceTypes.insert(valueType)
            for (index, member) in fieldsOfType.enumerated() {
                try charge()
                let index = UInt32(index)
                var memberAttributes = members[valueType]?[index] ?? [:]
                if let baseLocation {
                    switch types[member.type.index].kind {
                    case .scalar, .vector: break
                    default:
                        throw instruction.fail("Location-decorated interface structures require scalar or vector members",
                                               code: .unsupportedConstruct)
                    }
                    let (location, overflow) = baseLocation.addingReportingOverflow(index)
                    guard !overflow, memberAttributes[11] == nil, memberAttributes[30] == nil else {
                        throw instruction.fail("Invalid location on interface structure member")
                    }
                    for (key, value) in attributes where key != 30 && memberAttributes[key] == nil {
                        memberAttributes[key] = value
                    }
                    memberAttributes[30] = [location]
                }
                if !input, let builtin = memberAttributes[11]?.first,
                   builtin == 1 || builtin == 4 || (builtin == 3 && !capabilities.contains(32)),
                   memberAttributes[30] == nil {
                    omittedInterfaceMembers[global, default: []].insert(index)
                    continue
                }
                fields.append(.init(global: global, pointer: pointer, type: member.type, member: index,
                                    attributes: memberAttributes, input: input))
            }
        } else {
            guard attributes[11] != nil || attributes[30] != nil else {
                throw instruction.fail("Interface variable requires BuiltIn or Location", code: .unsupportedConstruct)
            }
            fields.append(.init(global: global, pointer: pointer, type: type, member: nil,
                                attributes: attributes, input: input))
            if input, let builtin = attributes[11]?.first, spirvForwardedBuiltin(builtin) != nil {
                forwardedInputGlobals.insert(global)
            }
        }
        stageInterfaces[id] = fields
        interfaceGlobals.insert(global)
    }

    private mutating func bindInterface(
        _ field: SPVInterfaceField, stage: IRStage, at instruction: SPVInstruction
    ) throws -> SPVBoundInterface {
        let attributes = field.attributes
        let kind = types[field.type.index].kind
        let binding: IRBinding
        var type = field.type
        var sampleMaskElement: IRHandle<IRType>?
        if let builtin = attributes[11]?.first {
            let fragmentSubgroupInput = stage == .fragment && field.input && (builtin == 36 || builtin == 41)
            var allowed: Set<UInt32> = [0, 11, 35]
            if builtin == 0 { allowed.insert(18) }
            let flatFragmentInput = stage == .fragment && field.input &&
                [18, 20, SPVMultiview.viewIndex].contains(builtin)
            if fragmentSubgroupInput || flatFragmentInput || builtin == SPVPrimitiveIndex.builtin {
                allowed.insert(14)
            }
            guard Set(attributes.keys).isSubset(of: allowed) else {
                throw instruction.fail("Invalid decorations on BuiltIn")
            }
            switch builtin {
            case 0 where stage == .vertex && !field.input && kind == .vector(size: .four, scalar: .f32):
                binding = .builtIn(.position(invariant: attributes[18] != nil))
            case 3:
                guard capabilities.contains(32) else {
                    throw instruction.fail("ClipDistance requires the SPIR-V ClipDistance capability", code: .unsupportedCapability)
                }
                guard stage == .vertex, !field.input else {
                    throw instruction.fail("Only vertex ClipDistance outputs are implemented", code: .unsupportedConstruct)
                }
                guard case .array(let element, .constant(let count), _) = kind,
                      types[element.index].kind == .scalar(.f32) else {
                    throw instruction.fail("ClipDistance requires a fixed array of f32")
                }
                guard (1...8).contains(count) else {
                    throw instruction.fail("ClipDistance supports up to eight planes", code: .unsupportedConstruct)
                }
                binding = .builtIn(.clipDistances)
            case 42, 43:
                guard stage == .vertex, field.input, kind == .scalar(.i32) || kind == .scalar(.u32) else {
                    throw instruction.fail("Vertex or instance index requires a vertex integer input")
                }
                binding = .builtIn(builtin == 42 ? .vertexIndex : .instanceIndex)
                type = try internType(.scalar(.u32))
            case SPVDrawParameters.drawIndex:
                guard capabilities.contains(SPVDrawParameters.capability) else {
                    throw instruction.fail("DrawIndex requires DrawParameters", code: .unsupportedCapability)
                }
                guard stage == .vertex, field.input, kind == .scalar(.i32) || kind == .scalar(.u32) else {
                    throw instruction.fail("DrawIndex requires a 32-bit integer vertex input")
                }
                binding = .builtIn(.drawIndex)
                type = try internType(.scalar(.u32))
            case SPVPrimitiveIndex.builtin:
                guard stage == .fragment, field.input, kind == .scalar(.i32) || kind == .scalar(.u32) else {
                    throw instruction.fail("PrimitiveId requires a 32-bit integer fragment input")
                }
                guard attributes[14] != nil else {
                    throw instruction.fail("PrimitiveId requires Flat interpolation")
                }
                guard capabilities.contains(SPVPrimitiveIndex.geometryCapability) ||
                      capabilities.contains(SPVPrimitiveIndex.tessellationCapability) else {
                    throw instruction.fail("PrimitiveId requires Geometry or Tessellation", code: .unsupportedCapability)
                }
                binding = .builtIn(.primitiveIndex)
                type = try internType(.scalar(.u32))
            case SPVMultiview.viewIndex:
                guard capabilities.contains(SPVMultiview.capability) else {
                    throw instruction.fail("ViewIndex requires the SPIR-V MultiView capability", code: .unsupportedCapability)
                }
                guard field.input, stage == .vertex || stage == .fragment,
                      kind == .scalar(.i32) || kind == .scalar(.u32) else {
                    throw instruction.fail("ViewIndex requires a 32-bit integer vertex or fragment input")
                }
                binding = .builtIn(.viewIndex)
                type = try internType(.scalar(.u32))
            case 15 where stage == .fragment && field.input && kind == .vector(size: .four, scalar: .f32):
                binding = .builtIn(.position(invariant: false))
            case 17 where stage == .fragment && field.input && kind == .scalar(.bool):
                binding = .builtIn(.frontFacing)
            case 18:
                guard capabilities.contains(35) else {
                    throw instruction.fail("SampleId requires SampleRateShading", code: .unsupportedCapability)
                }
                guard stage == .fragment, field.input, kind == .scalar(.i32) || kind == .scalar(.u32) else {
                    throw instruction.fail("SampleId requires a 32-bit integer fragment input")
                }
                binding = .builtIn(.sampleIndex)
                type = try internType(.scalar(.u32))
            case 20:
                guard stage == .fragment else {
                    throw instruction.fail("SampleMask requires a fragment interface")
                }
                guard case .array(let element, .constant(let count), _) = kind,
                      types[element.index].kind == .scalar(.i32) || types[element.index].kind == .scalar(.u32) else {
                    throw instruction.fail("SampleMask requires a fixed array of 32-bit integers")
                }
                guard count == 1 else {
                    throw instruction.fail("SampleMask supports one 32-bit coverage word", code: .unsupportedConstruct)
                }
                binding = .builtIn(.sampleMask)
                type = try internType(.scalar(.u32))
                sampleMaskElement = element
            case 22:
                guard stage == .fragment, !field.input, kind == .scalar(.f32) else {
                    throw instruction.fail("FragDepth requires an f32 fragment output")
                }
                binding = .builtIn(.fragDepth)
            case SPVBarycentric.perspective, SPVBarycentric.affine:
                guard capabilities.contains(SPVBarycentric.capability) else {
                    throw instruction.fail("Barycentric input requires FragmentBarycentricKHR", code: .unsupportedCapability)
                }
                guard stage == .fragment, field.input, kind == .vector(size: .three, scalar: .f32) else {
                    throw instruction.fail("Barycentric coordinates require a vec3<f32> fragment input")
                }
                binding = .builtIn(.barycentric(perspective: builtin == SPVBarycentric.perspective))
            case 24, 26...29, 36, 38, 40, 41:
                let subgroupInput = builtin >= 36
                if subgroupInput, capabilities.isDisjoint(with: SPVSubgroup.capabilities) {
                    throw instruction.fail("Subgroup BuiltIn requires GroupNonUniform", code: .unsupportedCapability)
                }
                guard stage == .compute || fragmentSubgroupInput, field.input, field.member == nil,
                      let builtin = spirvForwardedBuiltin(builtin) else {
                    throw instruction.fail("BuiltIn requires a direct input in a supported stage")
                }
                let scalar = subgroupInput || builtin == .localInvocationIndex
                let signed: IRTypeKind = scalar ? .scalar(.i32) : .vector(size: .three, scalar: .i32)
                let unsigned: IRTypeKind = scalar ? .scalar(.u32) : .vector(size: .three, scalar: .u32)
                guard kind == signed || kind == unsigned else { throw instruction.fail("Invalid BuiltIn input type") }
                binding = .builtIn(builtin)
                type = try internType(unsigned)
            case 0, 15, 17:
                throw instruction.fail("BuiltIn type, direction or stage mismatch")
            default:
                throw instruction.fail("BuiltIn \(builtin) is not implemented", code: .unsupportedConstruct)
            }
        } else if attributes[SPVBarycentric.perVertex] != nil {
            guard capabilities.contains(SPVBarycentric.capability) else {
                throw instruction.fail("PerVertexKHR requires FragmentBarycentricKHR", code: .unsupportedCapability)
            }
            guard stage == .fragment, field.input, field.member == nil, let location = attributes[30]?.first else {
                throw instruction.fail("PerVertexKHR requires a fragment input variable with Location")
            }
            guard Set(attributes.keys).isSubset(of: [0, 30, SPVBarycentric.perVertex]) else {
                throw instruction.fail("PerVertexKHR cannot be combined with interpolation or sampling decorations")
            }
            guard case .array(let element, .constant(3), _) = kind,
                  let shape = IRBitcastShape(types[element.index].kind),
                  [.f16, .f32, .i32, .u32].contains(shape.scalar) else {
                throw instruction.fail("PerVertexKHR requires an array of three supported numeric values")
            }
            binding = .location(index: location, interpolation: .perVertex, sampling: nil, blendSource: nil, perPrimitive: false)
        } else {
            guard stage != .compute, let location = attributes[30]?.first,
                  Set(attributes.keys).isSubset(of: [0,13,14,16,17,30,32,35]),
                  let shape = IRBitcastShape(kind), [.f16, .f32, .i32, .u32].contains(shape.scalar) else {
                throw instruction.fail("Location requires a supported numeric scalar or vector", code: .unsupportedConstruct)
            }
            let colorIndex = attributes[32]?.first
            if colorIndex != nil, stage != .fragment || field.input {
                throw instruction.fail("Index decoration requires a fragment color output")
            }
            if colorIndex == 1, location != 0 {
                throw instruction.fail("Dual-source fragment colors are supported only at location 0", code: .unsupportedConstruct)
            }
            let varying = (stage == .vertex && !field.input) || (stage == .fragment && field.input)
            let flat = attributes[14] != nil
            let linear = attributes[13] != nil
            let centroid = attributes[16] != nil
            let sample = attributes[17] != nil
            guard !(flat && linear), !(centroid && sample), !(flat && (centroid || sample)),
                  varying || !(flat || linear || centroid || sample),
                  !varying || shape.scalar.kind == .float || flat else {
                throw instruction.fail("Invalid interpolation decorations")
            }
            if sample && !capabilities.contains(35) {
                throw instruction.fail("Sample interpolation needs SampleRateShading", code: .unsupportedCapability)
            }
            let interpolation: IRInterpolation? = varying ? (flat ? .flat : linear ? .linear : .perspective) : nil
            let sampling: IRSampling? = varying && !flat ? (centroid ? .centroid : sample ? .sample : .center) : nil
            binding = .location(index: location, interpolation: interpolation, sampling: sampling,
                                blendSource: colorIndex == 1 ? 1 : nil, perPrimitive: false)
        }
        let memberPointer = try field.member.map { _ in try internType(.pointer(base: field.type, space: .private)) }
        return .init(field: field, binding: binding, type: type, memberPointer: memberPointer,
                     sampleMaskElement: sampleMaskElement)
    }

    mutating func lowerStageInterface(
        _ entry: SPVEntry, stage: IRStage, function: IRHandle<IRFunction>, functions: inout [IRFunction]
    ) throws -> IRHandle<IRFunction> {
        let listsAllGlobals = version >= 0x00010400
        let replacesDepth = modes[entry.function]?[12] != nil
        if !listsAllGlobals && !replacesDepth && entry.interfaces.isEmpty && interfaceGlobals.isEmpty { return function }
        var inputs: [SPVBoundInterface] = []
        var outputs: [SPVBoundInterface] = []
        var listed: Set<IRHandle<IRGlobalVariable>> = []
        var inputLocations: Set<UInt32> = []
        var outputLocations: Set<SPVOutputSlot> = []
        var inputBuiltins: Set<UInt32> = []
        var outputBuiltins: Set<UInt32> = []
        var inactiveDepthOutput: IRHandle<IRGlobalVariable>?
        for id in entry.interfaces {
            try charge()
            guard let global = globalMap[id]?.0 else {
                throw entry.inst.fail("Entry-point interface must list global variables")
            }
            guard listed.insert(global).inserted else { throw entry.inst.fail("Duplicate interface variable") }
            guard let fields = stageInterfaces[id] else {
                guard listsAllGlobals else {
                    throw entry.inst.fail("SPIR-V before 1.4 only lists stage inputs and outputs", code: .unsupportedConstruct)
                }
                continue
            }
            for field in fields {
                try charge()
                let bound = try bindInterface(field, stage: stage, at: entry.inst)
                if let slot = bound.locationSlot {
                    let inserted = field.input ? inputLocations.insert(slot.location).inserted : outputLocations.insert(slot).inserted
                    guard inserted else { throw entry.inst.fail("Duplicate interface Location") }
                } else if let builtin = field.attributes[11]?.first {
                    let inserted = field.input ? inputBuiltins.insert(builtin).inserted : outputBuiltins.insert(builtin).inserted
                    guard inserted else { throw entry.inst.fail("Duplicate interface BuiltIn") }
                    if builtin == 22, !field.input, !replacesDepth, field.member == nil {
                        // glslang can retain an unused gl_FragDepth declaration
                        // for a conservative-depth qualifier without DepthReplacing.
                        inactiveDepthOutput = field.global
                        continue
                    }
                }
                if field.input { inputs.append(bound) } else { outputs.append(bound) }
            }
        }
        let secondarySlot = SPVOutputSlot(location: 0, index: 1)
        if outputLocations.contains(secondarySlot) {
            // Vulkan treats a missing Index as zero. Expose source zero only
            // when a matching secondary output enables dual-source blending.
            try charge(outputs.count * 2)
            let primarySlot = SPVOutputSlot(location: 0, index: 0)
            guard outputLocations.count == 2,
                  let primary = outputs.firstIndex(where: { $0.locationSlot == primarySlot }),
                  let secondary = outputs.first(where: { $0.locationSlot == secondarySlot }),
                  types[outputs[primary].type.index].kind == types[secondary.type.index].kind else {
                throw entry.inst.fail("Dual-source outputs require two matching color types at location 0",
                                      code: .unsupportedConstruct)
            }
            outputs[primary].binding = .location(index: 0, interpolation: nil, sampling: nil,
                                                 blendSource: 0, perPrimitive: false)
        }
        guard replacesDepth == (outputBuiltins.contains(22) && inactiveDepthOutput == nil) else {
            throw entry.inst.fail("FragDepth output and DepthReplacing execution mode must be declared together")
        }
        let builtinInputs = builtinFunctionInputs[function.index]
        try charge(builtinInputs.count)
        guard builtinInputs.allSatisfy({ listed.contains($0) }) else {
            throw entry.inst.fail("Entry point omits a used builtin input")
        }
        try checkInterfaceUses(function, functions: functions, listed: listed, inactiveDepthOutput: inactiveDepthOutput, at: entry.inst)
        if inputs.isEmpty && outputs.isEmpty { return function }

        let result: IRFunctionResult?
        if outputs.count == 1 {
            result = .init(type: outputs[0].type, binding: outputs[0].binding)
        } else if outputs.isEmpty {
            result = nil
        } else {
            result = .init(type: try stageOutputType(outputs, at: entry.inst))
        }
        let positionLane = options.adjustCoordinateSpace && outputBuiltins.contains(0) ? try internType(.scalar(.f32)) : nil
        let site = entry.inst
        var wrapper = SPVFunctionLowerer(module: self, record: .init(header: site, body: []), budget: budget)
        var statements: [IRHandle<IRStatement>] = []
        var forwarded: [IRHandle<IRGlobalVariable>: IRHandle<IRExpression>] = [:]
        for input in inputs {
            let slot = UInt32(wrapper.parameters.count)
            try wrapper.node()
            wrapper.parameters.append(.init(type: input.type, binding: input.binding, location: site.location))
            var argument = try wrapper.expr(input.type, .functionArgument(slot), site)
            if let element = input.sampleMaskElement {
                if element != input.type {
                    argument = try wrapper.emitValue(element, .bitcast(expression: argument, type: element), &statements, site)
                }
                argument = try wrapper.emitValue(input.field.type, .compose(type: input.field.type, components: [argument]),
                                                  &statements, site)
            } else if input.type != input.field.type {
                argument = try wrapper.emitValue(input.field.type, .bitcast(expression: argument, type: input.field.type),
                                                           &statements, site)
            }
            if forwardedInputGlobals.contains(input.field.global) {
                forwarded[input.field.global] = argument
            } else {
                let pointer = try wrapper.interfacePointer(input, &statements, site)
                statements.append(try wrapper.stmt(.store(pointer: pointer, value: argument), site))
            }
        }
        let arguments = try builtinInputs.map { global in
            guard let argument = forwarded[global] else { throw site.fail("Missing builtin input argument") }
            return argument
        }
        statements.append(try wrapper.stmt(.call(function: function, arguments: arguments, result: nil), site))
        var values: [IRHandle<IRExpression>] = []
        for output in outputs {
            let pointer = try wrapper.interfacePointer(output, &statements, site)
            var value = try wrapper.emitValue(output.field.type, .load(pointer: pointer), &statements, site)
            if let element = output.sampleMaskElement {
                value = try wrapper.emitValue(element, .accessIndex(base: value, index: 0), &statements, site)
                if element != output.type {
                    value = try wrapper.emitValue(output.type, .bitcast(expression: value, type: output.type), &statements, site)
                }
            }
            if let lane = positionLane, case .builtIn(.position) = output.binding {
                var components: [IRHandle<IRExpression>] = []
                for index: UInt32 in 0..<4 {
                    let component = try wrapper.emitValue(lane, .accessIndex(base: value, index: index), &statements, site)
                    components.append(index == 1
                        ? try wrapper.emitValue(lane, .unary(op: .negate, expression: component), &statements, site)
                        : component)
                }
                value = try wrapper.emitValue(output.type, .compose(type: output.type, components: components),
                                                       &statements, site)
            }
            values.append(value)
        }
        let value: IRHandle<IRExpression>?
        if values.count > 1, let result {
            value = try wrapper.emitValue(result.type, .compose(type: result.type, components: values),
                                                   &statements, site)
        } else {
            value = values.first
        }
        statements.append(try wrapper.stmt(.returnValue(value), site))
        let body = try wrapper.block(statements, site)
        try wrapper.node()
        let handle = try IRHandle<IRFunction>(index: functions.count)
        functions.append(.init(arguments: wrapper.parameters, result: result, expressions: .init(wrapper.expressions),
                               statements: .init(wrapper.statements), blocks: .init(wrapper.irBlocks),
                               body: body, location: site.location))
        budget = wrapper.budget
        return handle
    }

    private mutating func checkInterfaceUses(
        _ entry: IRHandle<IRFunction>, functions: [IRFunction],
        listed: Set<IRHandle<IRGlobalVariable>>, inactiveDepthOutput: IRHandle<IRGlobalVariable>?, at site: SPVInstruction
    ) throws {
        let checksAliases = version >= 0x00010400 && !storageAliases.isEmpty
        if checksAliases { try charge(listed.count) }
        let effectiveListed = checksAliases ? listed.union(listed.map(canonicalStorageGlobal)) : listed
        var pending = [entry]
        var visited: Set<IRHandle<IRFunction>> = []
        while let function = pending.popLast() {
            try charge()
            if !visited.insert(function).inserted { continue }
            let body = functions[function.index]
            if checksAliases {
                let rawStorage = storageAliasInterfaceUses[function.index]
                try charge(rawStorage.count)
                guard rawStorage.allSatisfy(listed.contains) else {
                    throw site.fail("Entry point omits a used storage interface variable")
                }
            }
            for expression in body.expressions.values {
                try charge()
                if case .globalVariable(let global) = expression.kind {
                    if global == inactiveDepthOutput {
                        throw site.fail("Use of FragDepth output requires DepthReplacing execution mode")
                    }
                    if (version >= 0x00010400 || interfaceGlobals.contains(global)), !effectiveListed.contains(global) {
                        throw site.fail("Entry point omits a used interface variable")
                    }
                }
            }
            for statement in body.statements.values {
                try charge()
                if case .call(let target, _, _) = statement.kind { pending.append(target) }
            }
        }
    }

    private mutating func stageOutputType(
        _ outputs: [SPVBoundInterface], at site: SPVInstruction
    ) throws -> IRHandle<IRType> {
        var members: [IRStructMember] = []
        var size: UInt64 = 0
        var alignment: UInt64 = 1
        for output in outputs {
            try node()
            let layout = try valueLayout(output.type, site)
            let offset = layout.aligned(size)
            size = offset + layout.size
            guard size <= UInt64(Int32.max) else { throw site.fail("Stage output layout exceeds the size limit") }
            alignment = max(alignment, layout.alignment)
            members.append(.init(type: output.type, binding: output.binding, offset: UInt32(offset), location: site.location))
        }
        let span = SPVValueLayout(alignment: alignment, size: 0).aligned(size)
        guard span <= UInt64(Int32.max) else { throw site.fail("Stage output span exceeds the size limit") }
        try node()
        let handle = try IRHandle<IRType>(index: types.count)
        types.append(.init(kind: .structure(members: members, span: UInt32(span))))
        return handle
    }
}

extension SPVFunctionLowerer {
    fileprivate mutating func interfacePointer(
        _ interface: SPVBoundInterface, _ statements: inout [IRHandle<IRStatement>], _ site: SPVInstruction
    ) throws -> IRHandle<IRExpression> {
        let pointer = try expr(interface.field.pointer, .globalVariable(interface.field.global), site)
        if let index = interface.field.member, let type = interface.memberPointer {
            return try emitValue(type, .accessIndex(base: pointer, index: index), &statements, site)
        }
        return pointer
    }

    func checkOmittedInterfaceAccess(
        _ pointer: SPVValue, member: UInt32? = nil, at site: SPVInstruction
    ) throws {
        if case .globalVariable(let global) = expressions[pointer.expression.index].kind,
           let omitted = module.omittedInterfaceMembers[global],
           member.map({ omitted.contains($0) }) ?? true {
            throw site.fail("Access to an unsupported interface block member", code: .unsupportedConstruct)
        }
    }
}
