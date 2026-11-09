import SAGAShaderIR

struct ShaderStorage {
    enum WordAddressing: Equatable {
        case structured
        case byteAddressed
        case uniformRegisters
        case packedHalfWords(writable: Bool)
    }

    struct WordLayout {
        // Offsets and strides count scalar lanes, which can occupy two, four or eight bytes.
        let offset: UInt32
        let stride: UInt32
        let width: UInt8
        let scalar: IRScalar
        var addressing: WordAddressing = .structured
        var columns: UInt8? = nil
    }

    let reference: String
    let bound: String?
    let writable: Bool
    let words: WordLayout?
    let members: [ShaderStorage]
    let nested: NestedBufferLayout?
    let bindingArrayCount: UInt32?
    let splitBindingArray: Bool
    let externalTexture: ExternalTextureReference?
    let metalAtomicWriteGuard: String?

    init(reference: String, bound: String? = nil, writable: Bool,
         words: WordLayout? = nil, members: [ShaderStorage] = [], nested: NestedBufferLayout? = nil,
         bindingArrayCount: UInt32? = nil, splitBindingArray: Bool = false,
         externalTexture: ExternalTextureReference? = nil, metalAtomicWriteGuard: String? = nil) {
        self.reference = reference
        self.bound = bound
        self.writable = writable
        self.words = words
        self.members = members
        self.nested = nested
        self.bindingArrayCount = bindingArrayCount
        self.splitBindingArray = splitBindingArray
        self.externalTexture = externalTexture
        self.metalAtomicWriteGuard = metalAtomicWriteGuard
    }

    fileprivate func memory(at index: String? = nil) -> ShaderMemory {
        if let words {
            var firstWord = index.map { words.stride == 1 ? $0 : "(\($0)) * \(words.stride)u" }
            if words.offset > 0 {
                firstWord = firstWord.map { "\($0) + \(words.offset)u" } ?? "\(words.offset)u"
            }
            return .words(buffer: reference, first: firstWord ?? "0u", width: words.width,
                          scalar: words.scalar, addressing: words.addressing, columns: words.columns)
        }
        return .native(index.map { "\(reference)[\($0)]" } ?? reference)
    }
}

private enum ShaderMemory {
    case native(String)
    case aggregate(NestedBufferAccess)
    case glslAtomicWord(String)
    case words(buffer: String, first: String, width: UInt8, scalar: IRScalar,
               addressing: ShaderStorage.WordAddressing, columns: UInt8?)

    func component(_ index: String) throws -> Self {
        switch self {
        case .aggregate, .glslAtomicWord: throw ShaderFailure.invalidHandle
        case .native(let reference):
            return .native("\(reference)[\(index)]")
        case .words(let buffer, let first, let width, let scalar, let addressing, let columns):
            let stride = columns == nil ? 1 : width == 3 ? 4 : width
            let offset = stride == 1 ? "(\(index))" : "(\(index)) * \(stride)u"
            return .words(buffer: buffer, first: "\(first) + \(offset)", width: columns == nil ? 1 : width,
                          scalar: scalar, addressing: addressing, columns: nil)
        }
    }

    func load(in format: ShaderOutputFormat) throws -> String {
        switch self {
        case .native(let reference): return reference
        case .aggregate, .glslAtomicWord: throw ShaderFailure.invalidHandle
        case .words(let buffer, let first, let width, let scalar, let addressing, let columns):
            if let columns {
                let vectors = try (0..<columns).map { try component("\($0)u").load(in: format) }
                let type = shaderMatrixType(columns, width, scalar: scalar, in: format)
                return "\(type)(\(vectors.joined(separator: ", ")))"
            }
            let lanes = try (0..<width).map {
                let index = wordIndex(first: first, lane: $0)
                if scalar == .i64 || scalar == .u64 {
                    let bits: String
                    switch addressing {
                    case .structured: bits = "\(buffer)[\(index)]"
                    case .byteAddressed:
                        let offset = byteOffset(index, laneBytes: 8)
                        bits = joinedWideBits(low: "\(buffer).Load(\(offset))", high: "\(buffer).Load(\(offset) + 4u)")
                    case .uniformRegisters:
                        let register = "\(buffer)[(\(index)) / 2u]"
                        let low = "((\(index)) % 2u) * 2u"
                        bits = joinedWideBits(low: "\(register)[\(low)]", high: "\(register)[\(low) + 1u]")
                    case .packedHalfWords: throw ShaderFailure.invalidHandle
                    }
                    return scalar == .i64 ? try wideIntegerBits(bits, signed: true, in: format) : bits
                }
                let word: String
                switch addressing {
                case .structured: word = "\(buffer)[\(index)]"
                case .byteAddressed:
                    if scalar.width == 2 {
                        let type = try shaderVectorType(1, scalar: scalar, in: format)
                        return "\(buffer).Load<\(type)>(\(byteOffset(index, laneBytes: 2)))"
                    }
                    word = "\(buffer).Load(\(byteOffset(index)))"
                case .packedHalfWords(let writable):
                    let packed = "\(buffer)[(\(index)) / 2u]"
                    let word = writable ? "atomicOr(\(packed), 0u)" : packed
                    let lane = "((\(index)) % 2u)"
                    if scalar == .f16 { return "float16_t(unpackHalf2x16(\(word))[\(lane)])" }
                    return try integer16Bits("\(word) >> (\(lane) * 16u)", signed: scalar == .i16, in: format)
                case .uniformRegisters:
                    if scalar.width == 2 {
                        let register = "\(buffer)[(\(index)) / 8u][((\(index)) / 2u) % 4u]"
                        let lane = "((\(index)) % 2u)"
                        if scalar != .f16 {
                            return try integer16Bits("\(register) >> (\(lane) * 16u)", signed: scalar == .i16, in: format)
                        }
                        if format == .glsl {
                            return "unpackFloat2x16(\(register))[\(lane)]"
                        }
                        return "asfloat16(uint16_t(\(register) >> (\(lane) * 16u)))"
                    }
                    word = "\(buffer)[(\(index)) / 4u][(\(index)) % 4u]"
                }
                switch scalar {
                case .i32: return integerBits(word, signed: true, in: format)
                case .f32:
                    switch format {
                    case .msl: return "as_type<float>(\(word))"
                    case .glsl: return "uintBitsToFloat(\(word))"
                    default: return "asfloat(\(word))"
                    }
                default: return word
                }
            }
            if width == 1 { return lanes[0] }
            let type = try shaderVectorType(width, scalar: scalar, in: format)
            return "\(type)(\(lanes.joined(separator: ", ")))"
        }
    }

    func stores(_ value: String, temporary: String, in format: ShaderOutputFormat) throws -> [String] {
        switch self {
        case .native(let reference): return ["\(reference) = \(value);"]
        case .aggregate, .glslAtomicWord: throw ShaderFailure.invalidHandle
        case .words(let buffer, let first, let width, let scalar, let addressing, let columns):
            guard addressing != .uniformRegisters else { throw ShaderFailure.unsupported("Store to a uniform buffer") }
            if let columns {
                let type = shaderMatrixType(columns, width, scalar: scalar, in: format)
                var statements = ["const \(type) \(temporary) = \(value);"]
                for column in 0..<columns {
                    let target = try component("\(column)u")
                    statements += try target.stores("\(temporary)[\(column)u]",
                                                    temporary: "\(temporary)_column_\(column)", in: format)
                }
                return statements
            }
            var statements: [String] = []
            if width > 1 {
                let type = try shaderVectorType(width, scalar: scalar, in: format)
                statements.append("const \(type) \(temporary) = \(value);")
            }
            for lane in 0..<width {
                let component = width == 1 ? value : "\(temporary)[\(lane)]"
                let index = wordIndex(first: first, lane: lane)
                if case .packedHalfWords(let writable) = addressing {
                    guard writable else { throw ShaderFailure.invalidHandle }
                    statements += packedHalfStore(buffer: buffer, index: index, value: component, scalar: scalar,
                                                  temporary: temporary + "_half_\(lane)")
                    continue
                }
                let bits: String
                switch scalar {
                case .i64: bits = try wideIntegerBits(component, signed: false, in: format)
                case .i32: bits = integerBits(component, signed: false, in: format)
                case .f32:
                    switch format {
                    case .msl: bits = "as_type<uint>(\(component))"
                    case .glsl: bits = "floatBitsToUint(\(component))"
                    default: bits = "asuint(\(component))"
                    }
                default: bits = component
                }
                if addressing == .byteAddressed && scalar.width == 2 {
                    let type = try shaderVectorType(1, scalar: scalar, in: format)
                    statements.append("\(buffer).Store<\(type)>(\(byteOffset(index, laneBytes: 2)), \(component));")
                } else if addressing == .byteAddressed && scalar.width == 8 {
                    let packed = "\(temporary)_bits_\(lane)"
                    statements.append("const uint64_t \(packed) = \(bits);")
                    statements.append("\(buffer).Store2(\(byteOffset(index, laneBytes: 8)), uint2(uint(\(packed)), uint(\(packed) >> 32u)));")
                } else if addressing == .byteAddressed {
                    statements.append("\(buffer).Store(\(byteOffset(index)), \(bits));")
                } else {
                    statements.append("\(buffer)[\(index)] = \(bits);")
                }
            }
            return statements
        }
    }

    private func packedHalfStore(buffer: String, index: String, value: String, scalar: IRScalar, temporary: String) -> [String] {
        let slot = temporary + "_slot"
        let shift = temporary + "_shift"
        let bits = temporary + "_bits"
        let observed = temporary + "_observed"
        let expected = temporary + "_expected"
        let word = "\(buffer)[\(slot)]"
        let packed = scalar == .f16 ? "packHalf2x16(vec2(float(\(value)), 0.0))" : "uint(uint16_t(\(value)))"
        // Each halfword is an independent memory location. Preserve writes to the neighboring halfword.
        return [
            "uint \(slot) = (\(index)) / 2u;",
            "uint \(shift) = ((\(index)) % 2u) * 16u;",
            "uint \(bits) = (\(packed) & 65535u) << \(shift);",
            "uint \(observed) = atomicOr(\(word), 0u);",
            "uint \(expected);",
            "do {",
            "    \(expected) = \(observed);",
            "    \(observed) = atomicCompSwap(\(word), \(expected), (\(expected) & ~(65535u << \(shift))) | \(bits));",
            "} while (\(observed) != \(expected));",
        ]
    }

    var hlslAtomicByteAddress: (buffer: String, offset: String)? {
        guard case .words(let buffer, let first, 1, let scalar, .byteAddressed, nil) = self else { return nil }
        return (buffer, byteOffset(first, laneBytes: scalar.width))
    }

    private func wordIndex(first: String, lane: UInt8) -> String {
        lane == 0 ? first : "\(first) + \(lane)u"
    }

    private func byteOffset(_ index: String, laneBytes: UInt8 = 4) -> String {
        "(\(index)) * \(laneBytes)u"
    }

    private func joinedWideBits(low: String, high: String) -> String {
        "(uint64_t(\(low)) | (uint64_t(\(high)) << 32u))"
    }
}

private struct ShaderAddress {
    let memory: ShaderMemory
    let boundsCheck: String?
    let writable: Bool
}

private enum ShaderAtomicOperation {
    case load
    case store(String)
    case readModifyWrite(IRAtomicFunction, String)

    var bitwiseOperator: String? {
        switch self {
        case .readModifyWrite(.and, _): "&"
        case .readModifyWrite(.inclusiveOr, _): "|"
        case .readModifyWrite(.exclusiveOr, _): "^"
        default: nil
        }
    }
}

enum ShaderResultTarget {
    case direct
    case metalStructure(String)
    case metalVoidFragment(String)
    case glslPosition
    case glslVertexStructure(String, destinations: [String])
    case vertexClipStructure(String, destinations: [String], clipDistances: VertexClipDistances)
    case fragmentValue(FragmentOutputField, metalType: String?)
    case fragmentStructure(String, fields: [FragmentOutputField])
    case fragmentWithoutDepth(valueType: String, fields: [(offset: Int, element: FragmentOutputField)])
    case discardedFragmentDepth
}

private struct LoopTemporary {
    let expression: IRHandle<IRExpression>
    let afterLine: Int

    var name: String { "saga_loop_value_\(expression.index)" }
}

struct ShaderBodyWriter {
    let module: IRModule
    let function: IRFunction
    let arguments: [String]
    let storage: [ShaderStorage]
    let hasNestedBuffers: Bool
    let hasSplitBindingArrays: Bool
    let outputFormat: ShaderOutputFormat
    let resultTarget: ShaderResultTarget?
    let structures: [IRHandle<IRType>: ShaderStructure]
    let argumentFields: [Int: [String]]
    let callCaptures: [Int: [String]]
    let pointerWrites: [Int: Set<Int>]
    private(set) var budget: ShaderBudget
    private(set) var integerHelpers: Set<IntegerShaderHelper> = []
    private(set) var integerVectorHelpers: Set<IntegerVectorHelper> = []
    private var temporaries: Set<IRHandle<IRExpression>> = []
    private var temporaryOrder: [IRHandle<IRExpression>] = []
    private var continuingCaptures: [IRHandle<IRExpression>: Bool] = [:]
    private var selectedImages: [IRHandle<IRExpression>: ImageElement] = [:]
    private var webBufferElement: (access: IRHandle<IRExpression>, index: UInt32)?
    private var safeBufferRead = false
    private var indentation = "    "

    init(module: IRModule, function: IRFunction, arguments: [String], storage: [ShaderStorage],
         outputFormat: ShaderOutputFormat, resultTarget: ShaderResultTarget? = nil,
         structures: [IRHandle<IRType>: ShaderStructure] = [:], argumentFields: [Int: [String]] = [:],
         callCaptures: [Int: [String]] = [:], pointerWrites: [Int: Set<Int>] = [:],
         budget: ShaderBudget = ShaderBudget()) {
        self.module = module
        self.function = function
        self.arguments = arguments
        self.storage = storage
        self.hasNestedBuffers = storage.contains { $0.nested != nil }
        self.hasSplitBindingArrays = storage.contains(where: \.splitBindingArray)
        self.outputFormat = outputFormat
        self.resultTarget = resultTarget
        self.structures = structures
        self.argumentFields = argumentFields
        self.callCaptures = callCaptures
        self.pointerWrites = pointerWrites
        self.budget = budget
    }

    mutating func write() throws -> [String] {
        var lines: [String] = []
        for (index, local) in function.localVariables.values.enumerated() {
            if case .rayQuery(let vertexReturn) = try module.types.value(for: local.type).kind {
                guard (outputFormat == .msl || outputFormat == .hlsl || outputFormat == .glsl), !vertexReturn, local.initializer == nil else {
                    throw unsupported("ray query local target")
                }
                let name = localName(index)
                let queryType = outputFormat == .msl ? "metal::raytracing::intersection_query<metal::raytracing::instancing, metal::raytracing::triangle_data>" : outputFormat == .glsl ? "rayQueryEXT" : "RayQuery<RAY_FLAG_NONE>"
                try append("\(queryType) \(name);", to: &lines)
                try append("uint \(name)_state = 0u;", to: &lines)
                try append("float \(name)_tmax = 0.0f;", to: &lines)
                continue
            }
            let type = try typeName(local.type)
            let initial = try local.initializer.map { try value($0) } ?? zeroValue(local.type)
            try append("\(declaration(type, named: localName(index))) = \(initial);", to: &lines)
        }
        try writeBlock(function.body, nested: false, to: &lines)
        return lines
    }

    mutating func privateBindings(_ reachable: Set<Int>) throws -> PrivateShaderBindings {
        var bindings = PrivateShaderBindings()
        for index in reachable.sorted() {
            try budget.consume(.work)
            let global = module.globalVariables.values[index]
            guard global.space == .private else { continue }
            let name = "saga_private_\(index)"
            let type = try typeName(global.type)
            bindings.storage[index] = ShaderStorage(reference: name, writable: true)
            let explicit = try global.initializer.map { try constantValue($0) }
            if outputFormat == .wgsl {
                let initializer = explicit.map { " = \($0)" } ?? ""
                let declaration = "var<private> \(name): \(type)\(initializer);"
                try budget.consume(.sourceBytes, amount: declaration.utf8.count + 1)
                bindings.declarations.append(declaration)
                continue
            }
            let initial = try explicit ?? zeroValue(global.type)
            if outputFormat == .msl {
                bindings.captures[index] = [.init(type: "thread \(type)&", name: name)]
                try append("\(type) \(name) = \(initial);", to: &bindings.initialization)
            } else {
                // Native private globals are initialized from the entry point so aggregate
                // constructors run once per invocation in the same order as helper calls.
                let qualifier = outputFormat == .hlsl ? "static " : ""
                let declaration = "\(qualifier)\(type) \(name);"
                try budget.consume(.sourceBytes, amount: declaration.utf8.count + 1)
                bindings.declarations.append(declaration)
                try append("\(name) = \(initial);", to: &bindings.initialization)
            }
        }
        return bindings
    }

    @discardableResult
    private mutating func writeBlock(
        _ block: IRHandle<IRBlock>, nested: Bool = true, breakIf: IRHandle<IRExpression>? = nil,
        retainLoopBodyScope: Bool = false,
        to lines: inout [String]
    ) throws -> [LoopTemporary] {
        try budget.enterDepth()
        let mark = temporaryOrder.count
        var loopTemporaries: [LoopTemporary] = []
        if nested { indentation += "    " }
        defer {
            if !retainLoopBodyScope { discardTemporaries(from: mark) }
            if nested { indentation.removeLast(4) }
            try? budget.leaveDepth()
        }
        for handle in try function.blocks.value(for: block).statements {
            try budget.consume(.work)
            let beforeStatement = temporaryOrder.count
            let statement = try function.statements.value(for: handle).kind
            switch statement {
            case .rayQuery(let query, let operation):
                if outputFormat == .glsl { try writeGLSLRayQuery(query, operation: operation, to: &lines) }
                else if outputFormat == .hlsl { try writeHLSLRayQuery(query, operation: operation, to: &lines) }
                else { try writeMetalRayQuery(query, operation: operation, to: &lines) }
            case .emit(let handles):
                for expression in handles { try emit(expression, to: &lines) }
            case .call(let callee, let args, let result):
                try writeCall(callee, arguments: args, result: result, statementIndex: handle.index, to: &lines)
            case .returnValue(nil):
                try writeVoidReturn(to: &lines)
            case .returnValue(let result?):
                let value = try value(result)
                try writeResult(value, to: &lines)
            case .kill:
                try append(outputFormat == .msl ? "discard_fragment();" : "discard;", to: &lines)
            case .demoteToHelperInvocation:
                switch outputFormat {
                case .msl: try append("discard_fragment();", to: &lines)
                case .glsl: try append("demote;", to: &lines)
                default: try append("discard;", to: &lines)
                }
            case .subgroupGather(let mode, let argument, let result):
                try writeSubgroupGather(mode, argument: argument, result: result, to: &lines)
            case .subgroupBallot(let result, let predicate):
                try writeSubgroupBallot(predicate, result: result, to: &lines)
            case .subgroupCollectiveOperation(let operation, let collective, let argument, let result):
                try writeSubgroupCollective(operation, collective: collective, argument: argument, result: result, to: &lines)
            case .break:
                try append("break;", to: &lines)
            case .continue:
                try append("continue;", to: &lines)
            case .block(let child):
                try append("{", to: &lines)
                try writeBlock(child, to: &lines)
                try append("}", to: &lines)
            case .ifCondition(let condition, let accept, let reject):
                let test = try value(condition, parenthesized: false)
                try append("if (\(test)) {", to: &lines)
                try writeBlock(accept, to: &lines)
                if !(try function.blocks.value(for: reject).statements.isEmpty) {
                    try append("} else {", to: &lines)
                    try writeBlock(reject, to: &lines)
                }
                try append("}", to: &lines)
            case .switchValue(let selector, let cases):
                try writeSwitch(selector, cases: cases, index: handle.index, to: &lines)
            case .loop(let body, let continuing, let condition):
                try writeLoop(body, continuing: continuing, breakIf: condition,
                              index: handle.index, to: &lines)
            case .controlBarrier(let flags):
                for line in try shaderControlBarrier(flags, in: outputFormat) { try append(line, to: &lines) }
            case .debugPrintf(let format, let arguments):
                let callable: String
                switch outputFormat {
                case .wgsl: callable = "debugPrintf"
                case .msl: callable = "metal::os_log_default.log_info"
                default: throw unsupported("debugPrintf requires native WGSL, Metal or SPIR-V output")
                }
                var operands = [DebugPrintfFormat.sourceLiteral(format)]
                for argument in arguments {
                    let expression = try value(argument)
                    let type = try function.expressions.value(for: argument).type
                    if outputFormat == .msl, try module.types.value(for: type).kind == .scalar(.f16) {
                        operands.append("float(\(expression))")
                    } else {
                        operands.append(expression)
                    }
                }
                try append("\(callable)(\(operands.joined(separator: ", ")));", to: &lines)
            case .workgroupUniformLoad(let pointer, let result):
                try writeUniformLoad(pointer: pointer, result: result, to: &lines)
            case .imageStore(let image, let coordinate, let arrayIndex, let value):
                try writeImageVariants([image], named: "saga_texel_store_\(handle.index)", to: &lines) { writer, name, lines in
                    try writer.writeImageStore(image: image, coordinate: coordinate, arrayIndex: arrayIndex, value: value,
                                               named: name, to: &lines)
                }
            case .imageAtomic(let image, let coordinate, let arrayIndex, let function, let value):
                try writeImageVariants([image], named: "saga_texel_atomic_\(handle.index)", to: &lines) { writer, name, lines in
                    try writer.writeImageAtomic(image: image, coordinate: coordinate, arrayIndex: arrayIndex,
                                                function: function, value: value, named: name, to: &lines)
                }
            case .store(let pointer, let expression):
                try writeBufferVariants(splitBufferAccess(pointer), named: "saga_store_\(handle.index)", to: &lines) { writer, name, lines in
                    let target = try writer.address(pointer)
                    guard target.writable else { throw unsupported("store to read-only storage") }
                    let rhs = try writer.value(expression)
                    if let scalar = try writer.atomicScalar(at: pointer) {
                        let bits = try scalar == .f32 ? writer.wgslFloatAtomicConstantBits(expression) : nil
                        try writer.writeAtomic(.store(rhs), target: target, scalar: scalar, result: nil,
                                               temporary: name, constantFloatBits: bits, to: &lines)
                    } else {
                        let statements: [String]
                        if case .aggregate(let access) = target.memory {
                            let type = try writer.typeName(access.type)
                            statements = ["\(writer.declaration(type, named: name)) = \(rhs);"] +
                                (try writer.storeNested(access, value: name, temporary: name + "_field"))
                        } else {
                            statements = try target.memory.stores(rhs, temporary: name, in: writer.outputFormat)
                        }
                        try writer.writeStatements(statements, when: target.boundsCheck, to: &lines)
                    }
                }
            case .cooperativeStore(let target, let data):
                let tile = try value(target)
                let address = try cooperativeAddress(data, tileType: try function.expressions.value(for: target).type, named: "saga_coop_store_\(handle.index)", to: &lines)
                guard address.writable else { throw unsupported("cooperative store requires writable storage") }
                if outputFormat == .hlsl {
                    guard case .cooperativeMatrix(let columns, let rows, let scalar, _) = try module.types.value(for: function.expressions.value(for: target).type).kind,
                          scalar == .f16 || scalar == .f32 else {
                        throw unsupported("HLSL cooperative matrix store shape or scalar")
                    }
                    let count = Int(columns.rawValue) * Int(rows.rawValue) / 32
                    let index: (String) -> String = { element in
                        let row = "((\(element)) / \(columns.rawValue)u)"
                        let column = "((\(element)) % \(columns.rawValue)u)"
                        return data.rowMajor ? "\(address.index) + \(row) * \(address.stride) + \(column)" : "\(address.index) + \(column) * \(address.stride) + \(row)"
                    }
                    let store: (String, String) -> String = { element, component in
                        let offset = index(element)
                        return scalar == .f16
                            ? "\(address.buffer).Store<float16_t>((\(offset)) * 2u, float16_t(\(tile).\(component)));"
                            : "\(address.buffer)[\(offset)] = asuint(\(tile).\(component));"
                    }
                    let stores = (0..<count).map { component in
                        let field = component < 4 ? "lo[\(component)]" : "hi[\(component - 4)]"
                        return store("(WaveGetLaneIndex() * \(count)u + \(component)u)", field)
                    }
                    let minimumStride = data.rowMajor ? columns.rawValue : rows.rawValue
                    try writeStatements(stores, when: "(\(address.bounds)) && (\(address.stride) >= \(minimumStride)u)", to: &lines)
                    break
                }
                if outputFormat == .msl {
                    let tileType = try function.expressions.value(for: target).type
                    guard case .cooperativeMatrix(let columns, let rows, _, _) = try module.types.value(for: tileType).kind else {
                        throw unsupported("cooperative matrix store type")
                    }
                    let statements = (0..<Int(rows.rawValue / 8)).flatMap { row in
                        (0..<Int(columns.rawValue / 8)).map { column in
                            let part = rows == .eight && columns == .eight ? tile : "\(tile).t\(row)_\(column)"
                            let offset = data.rowMajor ? "\(row * 8)u * \(address.stride) + \(column * 8)u" : "\(column * 8)u * \(address.stride) + \(row * 8)u"
                            return "simdgroup_store(\(part), \(address.pointer) + (\(offset)), \(address.stride), ulong2(0), \(!data.rowMajor));"
                        }
                    }
                    let minor = data.rowMajor ? columns.rawValue : rows.rawValue
                    try writeStatements(statements, when: "(\(address.bounds)) && (\(address.stride) >= \(minor)u)", to: &lines)
                    break
                }
                let statement = outputFormat == .wgsl
                    ? "\(data.rowMajor ? "coopStoreT" : "coopStore")(\(tile), \(address.pointer), \(address.stride));"
                    : outputFormat == .glsl
                        ? "coopMatStore(\(tile), \(address.buffer), \(address.index), \(address.stride), \(data.rowMajor ? "gl_CooperativeMatrixLayoutRowMajor" : "gl_CooperativeMatrixLayoutColumnMajor"));"
                        : "simdgroup_store(\(tile), \(address.pointer), \(address.stride), ulong2(0), \(!data.rowMajor));"
                guard case .cooperativeMatrix(let columns, let rows, _, _) = try module.types.value(for: function.expressions.value(for: target).type).kind else { throw unsupported("cooperative matrix store type") }
                let minor = data.rowMajor ? columns.rawValue : rows.rawValue
                let storeBounds = "(\(address.bounds)) && (\(address.stride) >= \(minor)u)"
                try writeStatements([statement], when: storeBounds, to: &lines)
            case .atomic(let pointer, let operation, let operand, let result):
                let resultType = try result.map { try function.expressions.value(for: $0).type }
                let resultName = result.map { "saga_atomic_\($0.index)" } ?? "saga_atomic_statement_\(handle.index)"
                try writeBufferVariants(splitBufferAccess(pointer), named: resultName, resultType: resultType, to: &lines) { writer, name, lines in
                    let target = try writer.address(pointer)
                    guard target.writable, let scalar = try writer.atomicScalar(at: pointer) else {
                        throw unsupported("atomic read-modify-write address")
                    }
                    let value = try writer.value(operand)
                    let temporary = name == resultName ? "saga_atomic_tmp_\(handle.index)" : name + "_tmp"
                    if case .exchange(let compare?, let strong) = operation {
                        let expected = try writer.value(compare)
                        try writer.writeCompareExchange(target: target, scalar: scalar, expected: expected, replacement: value,
                            strong: strong, result: resultType.map { (name, $0) }, temporary: temporary, to: &lines)
                    } else {
                        let bits = try scalar == .f32 && operation == .exchange(compare: nil)
                            ? writer.wgslFloatAtomicConstantBits(operand) : nil
                        try writer.writeAtomic(.readModifyWrite(operation, value), target: target, scalar: scalar,
                                               result: result == nil ? nil : name, temporary: temporary,
                                               constantFloatBits: bits, to: &lines)
                    }
                }
                if let result { remember(result) }
            default: throw unsupported("statement")
            }
            if retainLoopBodyScope {
                for expression in temporaryOrder[beforeStatement...] {
                    try budget.consume(.work)
                    loopTemporaries.append(.init(expression: expression, afterLine: lines.count))
                }
            } else if statementEndsBlock(statement) {
                break
            }
        }
        if let breakIf {
            let test = try value(breakIf, parenthesized: false)
            try append("if (\(test)) { break; }", to: &lines)
        }
        return loopTemporaries
    }

    private func statementEndsBlock(_ statement: IRStatementKind) -> Bool {
        switch statement {
        case .returnValue, .break, .continue, .kill: return true
        default: return false
        }
    }

    private mutating func writeSwitch(
        _ selector: IRHandle<IRExpression>, cases: [IRSwitchCase], index: Int, to lines: inout [String]
    ) throws {
        var groups: [(labels: [String], body: IRHandle<IRBlock>, fallsThrough: Bool)] = []
        for entry in cases {
            try budget.consume(.work)
            let label: String
            switch entry.value {
            case .default: label = "default"
            case .u32(let number): label = "\(number)u"
            case .i32(let number):
                let suffix = outputFormat == .wgsl ? "i" : ""
                label = number == .min ? "(-2147483647\(suffix) - 1\(suffix))" : "\(number)\(suffix)"
            }
            if let last = groups.indices.last, groups[last].body == entry.body {
                groups[last].labels.append(label)
            } else {
                groups.append(([label], entry.body, entry.fallThrough))
            }
        }
        if outputFormat == .wgsl, groups.contains(where: \.fallsThrough) {
            try writeWGSLFallthroughSwitch(selector, groups: groups, index: index, to: &lines)
            return
        }
        try append("switch (\(try value(selector))) {", to: &lines)
        do {
            indentation += "    "
            defer { indentation.removeLast(4) }
            for group in groups {
                if outputFormat == .wgsl {
                    try append("\(wgslSwitchClause(group.labels)): {", to: &lines)
                    try writeBlock(group.body, to: &lines)
                    try append("}", to: &lines)
                } else {
                    for label in group.labels {
                        try append(label == "default" ? "default:" : "case \(label):", to: &lines)
                    }
                    try append("{", to: &lines)
                    try writeBlock(group.body, to: &lines)
                    try append("}", to: &lines)
                    if try !group.fallsThrough && !switchCaseFlow(group.body).terminates {
                        try append("break;", to: &lines)
                    }
                }
            }
        }
        try append("}", to: &lines)
    }

    private func wgslSwitchClause(_ labels: [String]) -> String {
        labels == ["default"] ? "default" : "case \(labels.joined(separator: ", "))"
    }

    private mutating func writeWGSLFallthroughSwitch(
        _ selector: IRHandle<IRExpression>,
        groups: [(labels: [String], body: IRHandle<IRBlock>, fallsThrough: Bool)],
        index: Int, to lines: inout [String]
    ) throws {
        let entry = "saga_switch_entry_\(index)"
        try append("var \(entry): u32;", to: &lines)
        try append("switch (\(try value(selector))) {", to: &lines)
        for (offset, group) in groups.enumerated() {
            try append("    \(wgslSwitchClause(group.labels)): { \(entry) = \(offset)u; }", to: &lines)
        }
        try append("}", to: &lines)

        // A synthetic switch catches case breaks while leaving continue aimed at the enclosing loop.
        try append("switch (0u) { default: {", to: &lines)
        indentation += "    "
        do {
            defer { indentation.removeLast(4) }
            for (offset, group) in groups.enumerated() {
                let last = offset == groups.count - 1
                // All remaining paths have entered by the final case.
                try append(last ? "{" : "if (\(entry) <= \(offset)u) {", to: &lines)
                try writeBlock(group.body, to: &lines)
                if try !last && !group.fallsThrough && !switchCaseFlow(group.body).terminates {
                    try append("    break;", to: &lines)
                }
                try append("}", to: &lines)
            }
        }
        try append("} }", to: &lines)
    }

    private mutating func switchCaseFlow(
        _ block: IRHandle<IRBlock>
    ) throws -> (terminates: Bool, breaks: Bool) {
        try budget.enterDepth()
        defer { try? budget.leaveDepth() }
        var terminates = false
        var breaks = false
        for statement in try function.blocks.value(for: block).statements {
            try budget.consume(.work)
            switch try function.statements.value(for: statement).kind {
            case .break:
                terminates = true
                breaks = true
            case .returnValue, .continue, .kill:
                terminates = true
            case .block(let child):
                let flow = try switchCaseFlow(child)
                terminates = flow.terminates
                breaks = breaks || flow.breaks
            case .ifCondition(_, let yes, let no):
                let accept = try switchCaseFlow(yes)
                let reject = try switchCaseFlow(no)
                terminates = accept.terminates && reject.terminates
                breaks = breaks || accept.breaks || reject.breaks
            case .switchValue(_, let cases):
                terminates = try switchTerminates(cases)
            default:
                terminates = false
            }
            if terminates { break }
        }
        return (terminates, breaks)
    }

    private mutating func switchTerminates(_ cases: [IRSwitchCase]) throws -> Bool {
        var previous: IRHandle<IRBlock>?
        for entry in cases {
            try budget.consume(.work)
            if previous != entry.body {
                let flow = try switchCaseFlow(entry.body)
                if flow.breaks || (!flow.terminates && !entry.fallThrough) { return false }
                previous = entry.body
            }
        }
        return true
    }

    private mutating func writeCall(
        _ callee: IRHandle<IRFunction>, arguments: [IRHandle<IRExpression>],
        result: IRHandle<IRExpression>?, statementIndex: Int, to lines: inout [String]
    ) throws {
        guard let captures = callCaptures[callee.index] else { throw unsupported("unplanned helper call") }
        let signature = try module.functions.value(for: callee).arguments
        var lowered: [String] = []
        var copyBack: [(assignment: String, valid: String?)] = []
        for (slot, pair) in zip(arguments, signature).enumerated() {
            try budget.consume(.work)
            let (handle, parameter) = pair
            guard case .pointer(let pointee, let space) = try module.types.value(for: parameter.type).kind else {
                lowered.append(try value(handle))
                continue
            }
            let target = try address(handle)
            guard case .native(let reference) = target.memory else {
                throw unsupported("pointer argument must address a native variable")
            }
            if !module.allowsPointerSubviews {
                guard target.boundsCheck == nil else {
                    throw unsupported("pointer argument must address an entire native variable")
                }
                lowered.append(outputFormat == .wgsl ? "&(\(reference))" : reference)
                continue
            }
            let temporary = "saga_pointer_\(statementIndex)_\(slot)"
            let valid: String?
            if let bounds = target.boundsCheck {
                let name = temporary + "_valid"
                let binding = outputFormat == .wgsl ? "let" : "const bool"
                try append("\(binding) \(name) = \(bounds);", to: &lines)
                valid = name
            } else {
                valid = nil
            }
            // These are per-invocation pointers, and alias analysis forbids competing writes.
            // WGSL passes the temporary as a whole function variable, even for private subviews.
            if valid != nil || (outputFormat == .wgsl && space == .private) {
                let type = try typeName(pointee)
                let binding = declaration(type, named: temporary)
                if let valid {
                    try append("\(binding) = \(try zeroValue(pointee));", to: &lines)
                    try writeStatements(["\(temporary) = \(reference);"], when: valid, to: &lines)
                } else {
                    try append("\(binding) = \(reference);", to: &lines)
                }
                lowered.append(outputFormat == .wgsl ? "&\(temporary)" : temporary)
                if pointerWrites[callee.index]?.contains(slot) == true {
                    copyBack.append(("\(reference) = \(temporary);", valid))
                }
            } else {
                lowered.append(outputFormat == .wgsl ? "&(\(reference))" : reference)
            }
            lowered.append(valid ?? "true")
        }
        lowered += captures
        let call = "saga_fn_\(callee.index)(\(lowered.joined(separator: ", ")))"
        if let result {
            let expression = try function.expressions.value(for: result)
            let type = try typeName(expression.type)
            let name = "saga_call_\(result.index)"
            let declaration = outputFormat == .wgsl ? "let \(name): \(type)" : "\(type) \(name)"
            try append("\(declaration) = \(call);", to: &lines)
            remember(result)
        } else {
            try append("\(call);", to: &lines)
        }
        for (assignment, valid) in copyBack {
            try writeStatements([assignment], when: valid, to: &lines)
        }
    }

    private mutating func writeSubgroupGather(
        _ mode: IRGatherMode, argument: IRHandle<IRExpression>, result: IRHandle<IRExpression>, to lines: inout [String]
    ) throws {
        let operand = try value(argument)
        let index = try mode.index.map { try value($0) }
        if case .quadBroadcastDynamic = mode, let index {
            let expression = try dynamicQuadBroadcast(operand: operand, index: index, result: result, to: &lines)
            try writeSubgroupResult(result, expression: expression, to: &lines)
            return
        }
        if case .quadBroadcast = mode, let index {
            let call: String
            switch outputFormat {
            case .msl: call = "quad_broadcast(\(operand), ushort(\(index)))"
            case .hlsl: call = "QuadReadLaneAt(\(operand), uint(\(index)))"
            case .glsl: call = "subgroupQuadBroadcast(\(operand), uint(\(index)))"
            default: call = "quadBroadcast(\(operand), \(index))"
            }
            try writeSubgroupResult(result, expression: call, to: &lines)
            return
        }
        if case .quadSwap(let direction) = mode {
            let names: (wgsl: String, hlsl: String, glsl: String, mask: UInt32)
            switch direction {
            case .x: names = ("quadSwapX", "QuadReadAcrossX", "subgroupQuadSwapHorizontal", 1)
            case .y: names = ("quadSwapY", "QuadReadAcrossY", "subgroupQuadSwapVertical", 2)
            case .diagonal: names = ("quadSwapDiagonal", "QuadReadAcrossDiagonal", "subgroupQuadSwapDiagonal", 3)
            }
            let call: String
            switch outputFormat {
            case .msl: call = "quad_shuffle_xor(\(operand), ushort(\(names.mask)))"
            case .hlsl: call = "\(names.hlsl)(\(operand))"
            case .glsl: call = "\(names.glsl)(\(operand))"
            default: call = "\(names.wgsl)(\(operand))"
            }
            try writeSubgroupResult(result, expression: call, to: &lines)
            return
        }
        let names: (wgsl: String, msl: String)
        switch mode {
        case .broadcastFirst: names = ("subgroupBroadcastFirst", "simd_broadcast_first")
        case .broadcast: names = ("subgroupBroadcast", "simd_broadcast")
        case .shuffle: names = ("subgroupShuffle", "simd_shuffle")
        case .shuffleDown: names = ("subgroupShuffleDown", "simd_shuffle_down")
        case .shuffleUp: names = ("subgroupShuffleUp", "simd_shuffle_up")
        case .shuffleXor: names = ("subgroupShuffleXor", "simd_shuffle_xor")
        default: throw unsupported("Unsupported subgroup gather")
        }
        let call: String
        switch outputFormat {
        case .msl:
            call = "\(names.msl)(\(operand)\(index.map { ", ushort(\($0))" } ?? ""))"
        case .hlsl:
            if let index {
                let lane: String
                switch mode {
                case .shuffleDown: lane = "WaveGetLaneIndex() + uint(\(index))"
                case .shuffleUp: lane = "WaveGetLaneIndex() - uint(\(index))"
                case .shuffleXor: lane = "WaveGetLaneIndex() ^ uint(\(index))"
                default: lane = "uint(\(index))"
                }
                call = "WaveReadLaneAt(\(operand), \(lane))"
            } else {
                call = "WaveReadLaneFirst(\(operand))"
            }
        case .glsl:
            call = "\(names.wgsl)(\(operand)\(index.map { ", uint(\($0))" } ?? ""))"
        default:
            call = "\(names.wgsl)(\(operand)\(index.map { ", \($0)" } ?? ""))"
        }
        try writeSubgroupResult(result, expression: call, to: &lines)
    }

    private mutating func dynamicQuadBroadcast(
        operand: String, index: String, result: IRHandle<IRExpression>, to lines: inout [String]
    ) throws -> String {
        switch outputFormat {
        case .msl: return "quad_broadcast(\(operand), ushort(\(index)))"
        case .hlsl: return "QuadReadLaneAt(\(operand), uint(\(index)))"
        default: break
        }

        // Constant quad indices work in WGSL and GLSL compiled to SPIR-V 1.3.
        // Read all four lanes before selecting so the quad operations execute
        // in the same control flow, including in fragment shaders.
        let wgsl = outputFormat == .wgsl
        let builtin = wgsl ? "quadBroadcast" : "subgroupQuadBroadcast"
        let type = try typeName(function.expressions.value(for: result).type)
        let lanes = (0..<4).map { "saga_quad_\(result.index)_\($0)" }
        for (lane, name) in lanes.enumerated() {
            let declaration = wgsl ? "let \(name): \(type)" : "\(type) \(name)"
            try append("\(declaration) = \(builtin)(\(operand), \(lane)u);", to: &lines)
        }
        var selected = lanes[0]
        for lane in 1..<4 {
            let test = "(\(index) == \(lane)u)"
            selected = wgsl ? "select(\(selected), \(lanes[lane]), \(test))"
                            : "(\(test) ? \(lanes[lane]) : \(selected))"
        }
        return selected
    }

    private mutating func writeSubgroupBallot(
        _ predicate: IRHandle<IRExpression>?, result: IRHandle<IRExpression>, to lines: inout [String]
    ) throws {
        let operand = try predicate.map { try value($0) } ?? "true"
        let expression: String
        switch outputFormat {
        case .msl:
            let mask = "saga_vote_\(result.index)"
            try append("ulong \(mask) = (ulong)simd_ballot(\(operand));", to: &lines)
            expression = "uint4(uint(\(mask)), uint(\(mask) >> 32), 0u, 0u)"
        case .hlsl:
            expression = "WaveActiveBallot(\(operand))"
        default:
            expression = "subgroupBallot(\(operand))"
        }
        try writeSubgroupResult(result, expression: expression, to: &lines)
    }

    private mutating func writeSubgroupCollective(
        _ operation: IRSubgroupOperation, collective: IRCollectiveOperation,
        argument: IRHandle<IRExpression>, result: IRHandle<IRExpression>, to lines: inout [String]
    ) throws {
        guard let name = operation.wgslName(collective) else { throw unsupported("Invalid subgroup collective") }
        let operand = try value(argument)
        let expression: String
        if outputFormat == .wgsl || outputFormat == .glsl {
            expression = "\(name)(\(operand))"
        } else if collective == .reduce {
            let names: (msl: String, hlsl: String)
            switch operation {
            case .all: names = ("simd_all", "WaveActiveAllTrue")
            case .any: names = ("simd_any", "WaveActiveAnyTrue")
            case .add: names = ("simd_sum", "WaveActiveSum")
            case .mul: names = ("simd_product", "WaveActiveProduct")
            case .min: names = ("simd_min", "WaveActiveMin")
            case .max: names = ("simd_max", "WaveActiveMax")
            case .and: names = ("simd_and", "WaveActiveBitAnd")
            case .or: names = ("simd_or", "WaveActiveBitOr")
            case .xor: names = ("simd_xor", "WaveActiveBitXor")
            }
            if outputFormat == .hlsl && operation.isBitwise {
                let type = try function.expressions.value(for: argument).type
                switch try module.types.value(for: type).kind {
                case .scalar(.i32), .vector(_, .i32):
                    expression = "asint(\(names.hlsl)(asuint(\(operand))))"
                default:
                    expression = "\(names.hlsl)(\(operand))"
                }
            } else {
                expression = "\(outputFormat == .msl ? names.msl : names.hlsl)(\(operand))"
            }
        } else if outputFormat == .msl {
            let mode = collective == .inclusiveScan ? "inclusive" : "exclusive"
            let operation = operation == .add ? "sum" : "product"
            expression = "simd_prefix_\(mode)_\(operation)(\(operand))"
        } else {
            let prefix = operation == .add ? "WavePrefixSum" : "WavePrefixProduct"
            let call = "\(prefix)(\(operand))"
            let combine = operation == .add ? "+" : "*"
            expression = collective == .inclusiveScan ? "(\(call) \(combine) \(operand))" : call
        }
        try writeSubgroupResult(result, expression: expression, to: &lines)
    }

    private mutating func writeSubgroupResult(
        _ result: IRHandle<IRExpression>, expression: String, to lines: inout [String]
    ) throws {
        let name = "saga_subgroup_\(result.index)"
        let type = try typeName(function.expressions.value(for: result).type)
        let declaration = outputFormat == .wgsl ? "let \(name): \(type)" : "\(type) \(name)"
        try append("\(declaration) = \(expression);", to: &lines)
        remember(result)
    }

    private mutating func remember(_ handle: IRHandle<IRExpression>) {
        temporaries.insert(handle)
        temporaryOrder.append(handle)
    }

    private mutating func discardTemporaries(from mark: Int) {
        for temporary in temporaryOrder[mark...] { temporaries.remove(temporary) }
        temporaryOrder.removeSubrange(mark...)
    }

    private mutating func writeVoidReturn(to lines: inout [String]) throws {
        if case .metalVoidFragment(let type) = resultTarget {
            try append("#ifdef \(RenderStageArtifact.metalPipelineSampleMaskMacro)", to: &lines)
            try append("return \(type){};", to: &lines)
            try append("#else", to: &lines)
            try append("return;", to: &lines)
            try append("#endif", to: &lines)
        } else {
            try append("return;", to: &lines)
        }
    }

    private mutating func writeResult(_ value: String, to lines: inout [String]) throws {
        switch resultTarget {
        case .direct:
            try append("return \(value);", to: &lines)
        case .metalStructure(let type):
            try append("return \(type){\(value)};", to: &lines)
        case .glslPosition:
            try append("gl_Position = \(value);", to: &lines)
            try finishGLSLVertex(to: &lines)
        case .glslVertexStructure(let type, let destinations):
            try append("\(type) saga_result = \(value);", to: &lines)
            for (index, destination) in destinations.enumerated() {
                try append("\(destination) = saga_result.saga_member_\(index);", to: &lines)
            }
            try finishGLSLVertex(to: &lines)
        case .vertexClipStructure(let type, let destinations, let clipDistances):
            try writeClipDistanceResult(value, type: type, destinations: destinations, clip: clipDistances, to: &lines)
        case .fragmentValue(let field, let metalType):
            let expression = fragmentOutputValue(value, field: field)
            if outputFormat == .glsl {
                try append("\(field.binding(in: .glsl)) = \(expression);", to: &lines)
                try append("return;", to: &lines)
            } else if let metalType {
                try append("return \(metalType){\(expression)};", to: &lines)
            } else {
                try append("return \(expression);", to: &lines)
            }
        case .fragmentStructure(let type, let fields):
            let converted = fields.enumerated().filter {
                ($0.element == .depth && outputFormat != .wgsl) ||
                    ($0.element == .sampleMask && outputFormat == .msl)
            }
            if outputFormat == .glsl {
                try append("\(type) saga_result = \(value);", to: &lines)
                for (index, field) in fields.enumerated() {
                    let expression = fragmentOutputValue("saga_result.saga_member_\(index)", field: field)
                    try append("\(field.binding(in: .glsl)) = \(expression);", to: &lines)
                }
                try append("return;", to: &lines)
            } else if !converted.isEmpty {
                try append("\(type) saga_result = \(value);", to: &lines)
                for (index, field) in converted {
                    let member = "saga_result.saga_member_\(index)"
                    try append("\(member) = \(fragmentOutputValue(member, field: field));", to: &lines)
                }
                try append("return saga_result;", to: &lines)
            } else {
                try append("return \(value);", to: &lines)
            }
        case .fragmentWithoutDepth(let valueType, let fields):
            try append("\(valueType) saga_result = \(value);", to: &lines)
            let initializer = outputFormat == .msl ? " = {}" : ""
            try append("SagaFragmentOutput saga_output\(initializer);", to: &lines)
            for (index, field) in fields {
                let member = "saga_member_\(index)"
                let expression = fragmentOutputValue("saga_result.\(member)", field: field)
                try append("saga_output.\(member) = \(expression);", to: &lines)
            }
            try append("return saga_output;", to: &lines)
        case .metalVoidFragment:
            try append("(void)(\(value));", to: &lines)
            try writeVoidReturn(to: &lines)
        case .discardedFragmentDepth:
            guard let result = function.result else { throw ShaderFailure.invalidHandle }
            let type = try typeName(result.type)
            try append("\(type) saga_ignored_depth = \(value);", to: &lines)
            try writeVoidReturn(to: &lines)
        case nil:
            throw unsupported("unexpected return value")
        }
    }

    private mutating func writeClipDistanceResult(
        _ value: String, type: String, destinations: [String], clip: VertexClipDistances, to lines: inout [String]
    ) throws {
        try append("\(type) saga_result = \(value);", to: &lines)
        let glsl = outputFormat == .glsl
        if !glsl {
            let initializer = outputFormat == .msl ? " = {}" : ""
            try append("SagaVertexOutput saga_output\(initializer);", to: &lines)
        }
        for (index, destination) in destinations.enumerated() where index != clip.member {
            let field = glsl ? destination : "saga_output.\(destination)"
            try append("\(field) = saga_result.saga_member_\(index);", to: &lines)
        }
        let source = "saga_result.saga_member_\(clip.member)"
        if outputFormat == .hlsl {
            for (group, range) in clip.hlslGroups.enumerated() {
                let values = range.map { source + outputFormat.valueArrayIndex(String($0)) }
                let vector = floatType(UInt8(range.count), in: .hlsl)
                try append("saga_output.saga_clip_\(group) = \(vector)(\(values.joined(separator: ", ")));", to: &lines)
            }
        } else {
            let destination = glsl ? "gl_ClipDistance" : "saga_output.\(destinations[clip.member])"
            for index in 0..<clip.count {
                let value = source + outputFormat.valueArrayIndex(String(index))
                try append("\(destination)[\(index)] = \(value);", to: &lines)
            }
        }
        if glsl {
            try finishGLSLVertex(to: &lines)
        } else {
            try append("return saga_output;", to: &lines)
        }
    }

    private func fragmentOutputValue(_ value: String, field: FragmentOutputField) -> String {
        switch field {
        case .depth where outputFormat == .msl:
            return "clamp(\(value), saga_viewport_depth.x, saga_viewport_depth.y)"
        case .depth: return "clamp(\(value), \(floatLiteral(0)), \(floatLiteral(1)))"
        case .sampleMask where outputFormat == .glsl: return "int(\(value))"
        case .sampleMask where outputFormat == .msl: return "saga_apply_pipeline_mask(\(value))"
        default: return value
        }
    }

    private mutating func finishGLSLVertex(to lines: inout [String]) throws {
        try append("gl_Position.y = -gl_Position.y;", to: &lines)
        try append("gl_Position.z = gl_Position.z * 2.0 - gl_Position.w;", to: &lines)
        try append("return;", to: &lines)
    }

    private mutating func writeLoop(
        _ body: IRHandle<IRBlock>, continuing: IRHandle<IRBlock>, breakIf: IRHandle<IRExpression>?,
        index: Int, to lines: inout [String]
    ) throws {
        let continuingBlock = try function.blocks.value(for: continuing)
        if continuingBlock.statements.isEmpty && breakIf == nil {
            try append(outputFormat == .wgsl ? "loop {" : "while (true) {", to: &lines)
            try writeBlock(body, to: &lines)
            try append("}", to: &lines)
            return
        }

        let mark = temporaryOrder.count
        defer { discardTemporaries(from: mark) }
        var bodyLines: [String] = []
        let definitions = try writeBlock(body, retainLoopBodyScope: true, to: &bodyLines)
        // Only references made while writing continuing need storage across iterations.
        for definition in definitions { continuingCaptures[definition.expression] = false }

        let first = "saga_first_\(index)"
        var continuingLines: [String] = []
        do {
            indentation += "    "
            defer { indentation.removeLast(4) }
            // Run continuing at the next iteration's head so native continue takes this path.
            try append("if (!\(first)) {", to: &continuingLines)
            try writeBlock(continuing, breakIf: breakIf, to: &continuingLines)
            try append("}", to: &continuingLines)
            try append("\(first) = false;", to: &continuingLines)
        }

        var captured: [LoopTemporary] = []
        for definition in definitions {
            try budget.consume(.work)
            if continuingCaptures.removeValue(forKey: definition.expression) == true {
                let type = try function.expressions.value(for: definition.expression).type
                // The first-iteration guard prevents reads before assignment. Initialize the
                // storage explicitly for native compilers that cannot prove this relationship.
                try append("\(declaration(try typeName(type), named: definition.name)) = \(try zeroValue(type));", to: &lines)
                captured.append(definition)
            }
        }
        try append("\(declaration("bool", named: first)) = true;", to: &lines)
        try append("while (true) {", to: &lines)
        try budget.consume(.work, amount: continuingLines.count + bodyLines.count)
        lines.append(contentsOf: continuingLines)
        indentation += "    "
        do {
            defer { indentation.removeLast(4) }
            var cursor = 0
            // Save each snapshot at its definition so a later continue cannot bypass the copy.
            for definition in captured {
                lines.append(contentsOf: bodyLines[cursor..<definition.afterLine])
                try append("\(definition.name) = \(try value(definition.expression));", to: &lines)
                cursor = definition.afterLine
            }
            lines.append(contentsOf: bodyLines[cursor...])
        }
        try append("}", to: &lines)
    }

    private mutating func emit(_ handle: IRHandle<IRExpression>, to lines: inout [String]) throws {
        let expression = try function.expressions.value(for: handle)
        switch expression.kind {
        case .cooperativeLoad(_, _, _, let data):
            guard outputFormat == .msl || outputFormat == .glsl || outputFormat == .hlsl || outputFormat == .wgsl else { throw unsupported("cooperative matrix target") }
            let name = "saga_coop_\(handle.index)"
            let location = try cooperativeAddress(data, tileType: expression.type, named: name, to: &lines)
            if outputFormat == .wgsl {
                let load = data.rowMajor ? "coopLoadT" : "coopLoad"
                let type = try typeName(expression.type)
                try append("var \(name): \(type) = \(type)();", to: &lines)
                try writeStatements(["\(name) = \(load)<\(type)>(\(location.pointer), \(location.stride));"],
                                    when: location.bounds, to: &lines)
            } else if outputFormat == .hlsl {
                guard case .cooperativeMatrix(let columns, let rows, let scalar, _) = try module.types.value(for: expression.type).kind else {
                    throw unsupported("HLSL cooperative matrix shape")
                }
                let count = Int(columns.rawValue) * Int(rows.rawValue) / 32
                let load: (String) -> String = { element in
                    let row = "((\(element)) / \(columns.rawValue)u)"
                    let column = "((\(element)) % \(columns.rawValue)u)"
                    let offset = data.rowMajor ? "\(location.index) + \(row) * \(location.stride) + \(column)" : "\(location.index) + \(column) * \(location.stride) + \(row)"
                    return scalar == .f16 ? "float(\(location.buffer).Load<float16_t>((\(offset)) * 2u))" : "asfloat(\(location.buffer)[\(offset)])"
                }
                try append("saga_coop_tile \(name) = (saga_coop_tile)0;", to: &lines)
                let loads = (0..<count).map { component in
                    let field = component < 4 ? "lo[\(component)]" : "hi[\(component - 4)]"
                    return "\(name).\(field) = \(load("(WaveGetLaneIndex() * \(count)u + \(component)u)"));"
                }
                try writeStatements(loads, when: location.bounds, to: &lines)
            } else if outputFormat == .glsl {
                guard case .cooperativeMatrix(_, _, let scalar, _) = try module.types.value(for: expression.type).kind else {
                    throw unsupported("GLSL cooperative matrix load result type")
                }
                let zero = scalar == .f16 ? "float16_t(0.0)" : "0.0"
                try append("\(try typeName(expression.type)) \(name) = \(try typeName(expression.type))(\(zero));", to: &lines)
                try writeStatements(["coopMatLoad(\(name), \(location.buffer), \(location.index), \(location.stride), \(data.rowMajor ? "gl_CooperativeMatrixLayoutRowMajor" : "gl_CooperativeMatrixLayoutColumnMajor"));"],
                                    when: location.bounds, to: &lines)
            } else {
                try append("\(try typeName(expression.type)) \(name) = \(try zeroValue(expression.type));", to: &lines)
                guard case .cooperativeMatrix(let columns, let rows, _, _) = try module.types.value(for: expression.type).kind else {
                    throw unsupported("cooperative matrix load type")
                }
                let statements = (0..<Int(rows.rawValue / 8)).flatMap { row in
                    (0..<Int(columns.rawValue / 8)).map { column in
                        let part = rows == .eight && columns == .eight ? name : "\(name).t\(row)_\(column)"
                        let offset = data.rowMajor ? "\(row * 8)u * \(location.stride) + \(column * 8)u" : "\(column * 8)u * \(location.stride) + \(row * 8)u"
                        return "simdgroup_load(\(part), \(location.pointer) + (\(offset)), \(location.stride), ulong2(0), \(!data.rowMajor));"
                    }
                }
                try writeStatements(statements, when: location.bounds, to: &lines)
            }
            remember(handle)
        case .cooperativeMultiplyAdd(let a, let b, let c):
            guard outputFormat == .msl || outputFormat == .glsl || outputFormat == .hlsl || outputFormat == .wgsl else { throw unsupported("cooperative matrix target") }
            let name = "saga_coop_\(handle.index)"
            if outputFormat == .wgsl {
                try append("let \(name) = coopMultiplyAdd(\(try value(a)), \(try value(b)), \(try value(c)));", to: &lines)
            } else if outputFormat == .hlsl {
                guard case .cooperativeMatrix(let columns, let rows, .f32, _) = try module.types.value(for: expression.type).kind else {
                    throw unsupported("HLSL cooperative multiply-add requires an f32 accumulator")
                }
                guard case .cooperativeMatrix(let inner, _, _, _) = try module.types.value(for: function.expressions.value(for: a).type).kind else { throw unsupported("HLSL cooperative multiply-add operand") }
                try append("\(try typeName(expression.type)) \(name) = saga_coop_muladd(\(try value(a)), \(try value(b)), \(try value(c)), \(rows.rawValue)u, \(columns.rawValue)u, \(inner.rawValue)u);", to: &lines)
            } else if outputFormat == .glsl {
                try append("\(try typeName(expression.type)) \(name) = coopMatMulAdd(\(try value(a)), \(try value(b)), \(try value(c)));", to: &lines)
            } else {
                try append("\(try typeName(expression.type)) \(name) = \(try zeroValue(expression.type));", to: &lines)
                guard case .cooperativeMatrix(let ac, let ar, _, _) = try module.types.value(for: function.expressions.value(for: a).type).kind,
                      case .cooperativeMatrix(let bc, let br, _, _) = try module.types.value(for: function.expressions.value(for: b).type).kind else {
                    throw unsupported("cooperative matrix multiply types")
                }
                let av = try value(a), bv = try value(b), cv = try value(c)
                func part(_ value: String, _ columns: IRCooperativeSize, _ rows: IRCooperativeSize, _ row: Int, _ column: Int) -> String {
                    columns == .eight && rows == .eight ? value : "\(value).t\(row)_\(column)"
                }
                for row in 0..<Int(ar.rawValue / 8) {
                    for column in 0..<Int(bc.rawValue / 8) {
                        let result = part(name, bc, ar, row, column)
                        for k in 0..<Int(ac.rawValue / 8) {
                            let accumulator = k == 0 ? part(cv, bc, ar, row, column) : result
                            try append("simdgroup_multiply_accumulate(\(result), \(part(av, ac, ar, row, k)), \(part(bv, bc, br, k, column)), \(accumulator));", to: &lines)
                        }
                    }
                }
            }
            remember(handle)
        case .rayQueryGetIntersection(let query, let committed):
            if outputFormat == .glsl {
                guard case .localVariable(let local) = try function.expressions.value(for: query).kind,
                      let layout = structures[expression.type] else { throw unsupported("GLSL ray query intersection") }
                let name = "saga_ray_intersection_\(handle.index)"
                for line in glslRayIntersectionLines(query: localName(local.index), name: name,
                                                      type: try typeName(expression.type),
                                                      fields: (0..<11).map { layout.memberName($0) },
                                                      committed: committed) {
                    try append(line, to: &lines)
                }
                remember(handle)
                break
            }
            if outputFormat == .hlsl {
                guard case .localVariable(let local) = try function.expressions.value(for: query).kind,
                      let layout = structures[expression.type] else { throw unsupported("HLSL ray query intersection") }
                let name = "saga_ray_intersection_\(handle.index)"
                for line in hlslRayIntersectionLines(query: localName(local.index), name: name,
                                                      type: try typeName(expression.type), layout: layout,
                                                      committed: committed) {
                    try append(line, to: &lines)
                }
                remember(handle)
                break
            }
            guard outputFormat == .msl,
                  case .localVariable(let local) = try function.expressions.value(for: query).kind,
                  let layout = structures[expression.type]
            else { throw unsupported("Metal ray query intersection") }
            let queryName = localName(local.index), state = "\(queryName)_state"
            let name = "saga_ray_intersection_\(handle.index)", type = try typeName(expression.type)
            let flavor = committed ? "committed" : "candidate"
            let condition = committed ? "(\(state) & 4u) != 0u" : "(\(state) & 2u) != 0u && (\(state) & 4u) == 0u"
            func field(_ index: Int) -> String { "\(name).\(layout.memberName(index))" }
            try append("\(type) \(name){};", to: &lines)
            try append("if (\(condition)) {", to: &lines)
            try append("    auto saga_type = \(queryName).get_\(flavor)_intersection_type();", to: &lines)
            try append("    if (saga_type == metal::raytracing::intersection_type::triangle) {", to: &lines)
            try append("        \(field(0)) = 1u;", to: &lines)
            if !committed { try append("        \(field(1)) = \(queryName).get_candidate_triangle_distance();", to: &lines) }
            try append("        \(field(7)) = \(queryName).get_\(flavor)_triangle_barycentric_coord();", to: &lines)
            try append("        \(field(8)) = \(queryName).is_\(flavor)_triangle_front_facing();", to: &lines)
            try append("    } else if (saga_type == metal::raytracing::intersection_type::bounding_box) { \(field(0)) = \(committed ? 2 : 3)u; }", to: &lines)
            try append("    if (saga_type != metal::raytracing::intersection_type::none) {", to: &lines)
            if committed { try append("        \(field(1)) = \(queryName).get_committed_distance();", to: &lines) }
            for (index, method) in [(2, "user_instance_id"), (3, "instance_id"), (5, "geometry_id"), (6, "primitive_id"), (9, "object_to_world_transform"), (10, "world_to_object_transform")] {
                try append("        \(field(index)) = \(queryName).get_\(flavor)_\(method)();", to: &lines)
            }
            try append("    }", to: &lines)
            try append("}", to: &lines)
            remember(handle)
        case .runtimeValue(let operand):
            let name = "saga_value_\(handle.index)"
            let binding = outputFormat == .wgsl ? "let \(name)" : "\(try typeName(expression.type)) \(name)"
            try append("\(binding) = \(try value(operand));", to: &lines)
            remember(handle)
        case .load(let pointer):
            // Loads are evaluated at their IR emit, before any later writes.
            let name = "saga_load_\(handle.index)"
            let selection = try splitBufferAccess(pointer)
            try writeBufferVariants(selection, named: name, resultType: expression.type,
                                    eager: selection?.writable == false, to: &lines) { writer, name, lines in
                let source = try writer.address(pointer)
                try writer.writeLoad(pointer: pointer, source: source, type: expression.type, name: name, to: &lines)
            }
            remember(handle)
        case .arrayLength(let pointer):
            if let selection = try splitBufferAccess(pointer) {
                try writeBufferVariants(selection, named: "saga_array_length_\(handle.index)", resultType: expression.type,
                                        eager: true, to: &lines) { writer, name, lines in
                    let binding = writer.outputFormat == .wgsl ? "let" : "uint"
                    try writer.append("\(binding) \(name) = \(try writer.value(handle));", to: &lines)
                }
                remember(handle)
            } else {
                _ = try value(handle)
            }
        case .imageSample(let sample):
            let name = "saga_image_\(handle.index)"
            try writeImageVariants([sample.image, sample.sampler], named: name, resultType: expression.type,
                                   eager: true, to: &lines) { writer, name, lines in
                let sampled: String
                if let component = sample.gather {
                    sampled = try writer.gatherExpression(sample, component: component, named: name, to: &lines)
                } else {
                    sampled = try writer.sampleExpression(sample, named: name, to: &lines)
                }
                let binding = writer.outputFormat == .wgsl ? "let \(name)" :
                    writer.declaration(try writer.typeName(expression.type), named: name)
                try writer.append("\(binding) = \(sampled);", to: &lines)
            }
            remember(handle)
        case .imageLoad(let load):
            let image = try function.expressions.value(for: load.image)
            let writable = try module.types.value(for: image.type).kind.storageTexture?.access.contains(.store) == true
            try writeImageVariants([load.image], named: "saga_image_\(handle.index)", resultType: expression.type,
                                   eager: !writable, to: &lines) { writer, name, lines in
                try writer.writeImageLoad(load, named: name, type: expression.type, to: &lines)
            }
            remember(handle)
        case .imageQuery(let image, let query):
            try writeImageVariants([image], named: "saga_image_\(handle.index)", resultType: expression.type,
                                   eager: true, to: &lines) { writer, name, lines in
                try writer.writeImageQuery(query, image: image, named: name, type: expression.type, to: &lines)
            }
            remember(handle)
        case .derivative(let axis, let control, let operand):
            let name = "saga_derivative_\(handle.index)"
            let value = try derivativeExpression(axis, control: control, operand: value(operand))
            let binding = outputFormat == .wgsl ? "let \(name)" : declaration(try typeName(expression.type), named: name)
            try append("\(binding) = \(value);", to: &lines)
            remember(handle)
        case .select(let condition, let accept, let reject):
            let name = "saga_select_\(handle.index)"
            let selected = try selectExpression(condition: condition, accept: accept, reject: reject, type: expression.type)
            let declaration = outputFormat == .wgsl ? "let \(name)" : "\(try typeName(expression.type)) \(name)"
            try append("\(declaration) = \(selected);", to: &lines)
            remember(handle)
        case .math(let function, let first, let second, let third, let fourth) where function.integerBitArity != nil:
            let scalar: IRScalar
            let width: UInt8?
            switch try module.types.value(for: expression.type).kind {
            case .scalar(let lane): scalar = lane; width = nil
            case .vector(let size, let lane): scalar = lane; width = size.rawValue
            default: throw unsupported("integer bit builtin result")
            }
            let arguments = try ([first] + [second, third, fourth].compactMap { $0 }).map { try value($0) }
            let statements = try integerBitStatements(function, arguments: arguments, type: typeName(expression.type),
                                                       scalar: scalar, width: width,
                                                       named: "saga_math_\(handle.index)", format: outputFormat)
            for statement in statements { try append(statement, to: &lines) }
            remember(handle)
        case .math(let function, let operand, _, _, _) where function.packingBuiltin != nil:
            guard let builtin = function.packingBuiltin else { throw ShaderFailure.invalidHandle }
            let statements = try packingStatements(function, builtin: builtin, operand: value(operand),
                                                   resultType: typeName(expression.type), named: "saga_math_\(handle.index)",
                                                   format: outputFormat)
            for statement in statements { try append(statement, to: &lines) }
            remember(handle)
        case .math(.ldexp, let mantissa, let exponent?, nil, nil):
            try writeLdexp(mantissa, exponent: exponent, result: handle, type: expression.type, to: &lines)
            remember(handle)
        case .math(let operation, let operand, nil, nil, nil) where operation.decomposesFloat:
            try writeDecomposition(operation, operand: operand, type: expression.type, result: handle, to: &lines)
            remember(handle)
        case .math(let operation, let first, let second, let third, nil) where operation.geometricArity != nil:
            let input = try function.expressions.value(for: first)
            let inputType = try module.types.value(for: input.type).kind
            let args = try ([first] + [second, third].compactMap { $0 }).map { try value($0) }
            let statements = try geometricStatements(operation, arguments: args, operand: inputType,
                                                      resultType: typeName(expression.type), named: "saga_math_\(handle.index)",
                                                      format: outputFormat)
            for statement in statements { try append(statement, to: &lines) }
            remember(handle)
        case .math(let operation, let first, let second, let third, nil) where operation.numericBuiltin != nil:
            let result = try module.types.value(for: expression.type).kind
            let operands = [first] + [second, third].compactMap { $0 }
            let args = try operands.map { try value($0) }
            let argumentTypes = try operands.map { try module.types.value(for: function.expressions.value(for: $0).type).kind }
            let statements = try numericBuiltinStatements(operation, arguments: args, argumentTypes: argumentTypes,
                                                            result: result, resultType: typeName(expression.type),
                                                            named: "saga_math_\(handle.index)", format: outputFormat)
            for statement in statements { try append(statement, to: &lines) }
            remember(handle)
        case .binary(.modulo, let left, let right) where outputFormat == .glsl:
            guard try module.types.value(for: expression.type).kind.concreteFloatShape != nil else {
                _ = try value(handle)
                break
            }
            let type = try typeName(expression.type)
            let name = "saga_remainder_\(handle.index)"
            let lhs = "\(name)_lhs", rhs = "\(name)_rhs"
            try append("\(type) \(lhs) = \(try value(left));", to: &lines)
            try append("\(type) \(rhs) = \(try value(right));", to: &lines)
            try append("\(type) \(name) = \(floatingRemainder(lhs, rhs));", to: &lines)
            remember(handle)
        case .cast(let operand, let target) where target.kind == .sint || target.kind == .uint:
            let input = try function.expressions.value(for: operand).type
            guard let shape = IRBitcastShape(try module.types.value(for: input).kind), shape.scalar == .f64 else {
                _ = try value(handle)
                break
            }
            try writeFloat64ToIntegerCast(operand, input: input, target: target, size: shape.size,
                                          result: handle, resultType: expression.type, to: &lines)
            remember(handle)
        case .bitcast(let operand, let target):
            let inputType = try function.expressions.value(for: operand).type
            let input = try bitcastShape(inputType)
            let output = try bitcastShape(target)
            guard let words = input.packedHalfWords(to: output), outputFormat == .hlsl || outputFormat == .glsl else {
                _ = try value(handle)
                break
            }
            let name = "saga_bitcast_\(handle.index)"
            let source = "\(name)_operand"
            try append("\(try typeName(inputType)) \(source) = \(try value(operand));", to: &lines)
            let type = try typeName(target)
            let cast = packedHalfBitcast(source, input: input, output: output, words: words,
                                         resultType: type, format: outputFormat)
            try append("\(type) \(name) = \(cast);", to: &lines)
            remember(handle)
        case .access(let base, let index):
            let type = try module.types.value(for: expression.type).kind
            if type.isTextureOrSampler { break }
            if case .pointer = type {
                if try splitBufferAccess(handle) == nil { _ = try address(handle) }
            } else {
                let source = try function.expressions.value(for: base)
                let sourceType = try module.types.value(for: source.type).kind
                let (index, nonnegative) = try indexedValue(index)
                let width: UInt32
                let subscriptPath: String
                if case .array(_, .constant(let count), _) = sourceType {
                    width = count
                    subscriptPath = outputFormat.valueArrayIndex(index)
                } else if let count = sourceType.componentCount {
                    width = UInt32(count)
                    subscriptPath = "[\(index)]"
                } else {
                    throw unsupported("dynamic value indexing")
                }
                let bounds = [nonnegative, "\(index) < \(width)u"].compactMap { $0 }.joined(separator: " && ")
                let name = "saga_index_\(handle.index)"
                let initial = try zeroValue(expression.type)
                try append("\(declaration(try typeName(expression.type), named: name)) = \(initial);", to: &lines)
                let operand = try value(base)
                try writeStatements(["\(name) = \(operand)\(subscriptPath);"], when: bounds, to: &lines)
                remember(handle)
            }
        case .accessIndex:
            let type = try module.types.value(for: expression.type).kind
            if type.isTextureOrSampler { break }
            if case .pointer = type {
                if try storageLocation(expression.kind) == nil, try splitBufferAccess(handle) == nil { _ = try address(handle) }
            } else {
                _ = try value(handle)
            }
        default:
            _ = try value(handle)
        }
    }

    private struct ImageElement {
        let reference: String
        let index: UInt32
    }

    private struct ImageVariant {
        var references: [IRHandle<IRExpression>: ImageElement] = [:]
        var conditions: [String] = []
    }

    private mutating func writeImageVariants(
        _ images: [IRHandle<IRExpression>], named name: String, resultType: IRHandle<IRType>? = nil,
        eager: Bool = false, to lines: inout [String],
        emit: (inout Self, String, inout [String]) throws -> Void
    ) throws {
        guard hasSplitBindingArrays else {
            try emit(&self, name, &lines)
            return
        }
        var seen: Set<IRHandle<IRExpression>> = []
        var variants = [ImageVariant()]
        for image in images where seen.insert(image).inserted {
            guard let access = try SplitImageAccess(image, in: function) else { continue }
            let array = try function.expressions.value(for: access.base)
            guard let location = try storageLocation(array.kind), location.splitBindingArray,
                  let count = location.bindingArrayCount, count > 0 else { continue }
            if let fixed = access.fixedElement(count: count) {
                for variant in variants.indices {
                    variants[variant].references[image] = .init(reference: "\(location.reference)_\(fixed)", index: fixed)
                }
                continue
            }
            guard let dynamic = access.dynamic else { throw ShaderFailure.invalidHandle }
            let selector = "\(name)_binding_\(image.index)"
            let index = try boundedImageIndex(dynamic, count: count)
            let declaration = outputFormat == .wgsl ? "let" : "uint"
            try append("\(declaration) \(selector) = \(index);", to: &lines)
            try budget.consume(.nodes, amount: variants.count * Int(count))
            variants = variants.flatMap { variant in
                (0..<count).map { element in
                    var choice = variant
                    choice.references[image] = .init(reference: "\(location.reference)_\(element)", index: element)
                    choice.conditions.append("\(selector) == \(element)u")
                    return choice
                }
            }
        }
        let choices = variants.map { ($0.references, $0.conditions.isEmpty ? nil : imageBounds($0.conditions)) }
        defer { selectedImages.removeAll(keepingCapacity: true) }
        try writeBindingVariants(choices, named: name, resultType: resultType, eager: eager, to: &lines,
                                 select: { $0.selectedImages = $1 }, emit: emit)
    }

    private mutating func writeBindingVariants<Selection>(
        _ choices: [(Selection, String?)], named name: String, resultType: IRHandle<IRType>?, eager: Bool,
        to lines: inout [String], select: (inout Self, Selection) -> Void,
        emit: (inout Self, String, inout [String]) throws -> Void
    ) throws {
        if choices.count == 1, let choice = choices.first, choice.1 == nil {
            select(&self, choice.0)
            try emit(&self, name, &lines)
            return
        }
        var selected = try resultType.map { try zeroValue($0) } ?? ""
        if choices.isEmpty {
            if let resultType {
                let binding = outputFormat == .wgsl ? "let \(name)" : declaration(try typeName(resultType), named: name)
                try append("\(binding) = \(selected);", to: &lines)
            }
            return
        }
        if let resultType, !eager || outputFormat == .glsl {
            try append("\(declaration(try typeName(resultType), named: name)) = \(selected);", to: &lines)
        }
        for (index, choice) in choices.enumerated() {
            let candidate = "\(name)_candidate_\(index)"
            guard let condition = choice.1 else { throw ShaderFailure.invalidHandle }
            select(&self, choice.0)
            if eager {
                guard let resultType else { throw ShaderFailure.invalidHandle }
                try emit(&self, candidate, &lines)
                if outputFormat == .glsl {
                    try writeStatements(["\(name) = \(candidate);"], when: condition, to: &lines)
                } else {
                    let result = index + 1 == choices.count ? name : "\(name)_selected_\(index)"
                    let value = try selectCompositeValue(resultType, reference: candidate, valid: condition, reject: selected)
                    try append("let \(result) = \(value);", to: &lines)
                    selected = result
                }
            } else {
                try append("if (\(condition)) {", to: &lines)
                indentation += "    "
                try emit(&self, candidate, &lines)
                if resultType != nil { try append("\(name) = \(candidate);", to: &lines) }
                indentation.removeLast(4)
                try append("}", to: &lines)
            }
        }
    }

    private struct SplitBufferAccess {
        let handle: IRHandle<IRExpression>
        let fixed: UInt32?
        let dynamic: IRHandle<IRExpression>?
        let count: UInt32
        let writable: Bool
    }

    private mutating func splitBufferAccess(_ pointer: IRHandle<IRExpression>) throws -> SplitBufferAccess? {
        guard hasSplitBindingArrays, hasNestedBuffers else { return nil }
        var handle = pointer
        while true {
            try budget.consume(.work)
            let base: IRHandle<IRExpression>
            let dynamic: IRHandle<IRExpression>?
            var fixed: UInt32?
            switch try function.expressions.value(for: handle).kind {
            case .accessIndex(let array, let index): (base, dynamic, fixed) = (array, nil, index)
            case .access(let array, let index): (base, dynamic, fixed) = (array, index, nil)
            default: return nil
            }
            if case .globalVariable(let global) = try function.expressions.value(for: base).kind {
                let location = storage[global.index]
                guard location.splitBindingArray, let array = location.nested?.bindingArray else { return nil }
                if let dynamic {
                    switch try function.expressions.value(for: dynamic).kind {
                    case .literal(.u32(let index)): fixed = index
                    case .literal(.i32(let index)): fixed = UInt32(bitPattern: index)
                    default: break
                    }
                }
                return .init(handle: handle, fixed: fixed, dynamic: dynamic, count: array.count, writable: location.writable)
            }
            handle = base
        }
    }

    private mutating func writeBufferVariants(
        _ access: SplitBufferAccess?, named name: String, resultType: IRHandle<IRType>? = nil,
        eager: Bool = false, to lines: inout [String],
        emit: (inout Self, String, inout [String]) throws -> Void
    ) throws {
        guard let access else {
            try emit(&self, name, &lines)
            return
        }
        let choices: [(UInt32, String?)]
        if let fixed = access.fixed {
            choices = fixed < access.count ? [(fixed, nil)] : []
        } else {
            guard let dynamic = access.dynamic else { throw ShaderFailure.invalidHandle }
            let selector = "\(name)_binding"
            let (index, _) = try indexedValue(dynamic)
            let binding = outputFormat == .wgsl ? "let" : "uint"
            try append("\(binding) \(selector) = \(index);", to: &lines)
            try budget.consume(.nodes, amount: Int(access.count))
            choices = (0..<access.count).map { ($0, "\(selector) == \($0)u") }
        }
        safeBufferRead = eager && outputFormat == .wgsl
        defer { webBufferElement = nil; safeBufferRead = false }
        try writeBindingVariants(choices, named: name, resultType: resultType, eager: eager, to: &lines,
                                 select: { $0.webBufferElement = (access.handle, $1) }, emit: emit)
    }

    private mutating func imageReference(_ image: IRHandle<IRExpression>) throws -> String {
        if let selected = selectedImages[image] { return selected.reference }
        let expression = try function.expressions.value(for: image)
        switch expression.kind {
        case .accessIndex(let base, let index):
            let (name, count) = try bindingArrayReference(base)
            return "\(name)[\(min(index, count - 1))u]"
        case .access(let base, let index):
            let (name, count) = try bindingArrayReference(base)
            let bounded = try boundedImageIndex(index, count: count)
            let subscriptValue = outputFormat == .hlsl ? "NonUniformResourceIndex(\(bounded))" : bounded
            return "\(name)[\(subscriptValue)]"
        default:
            guard let location = try storageLocation(expression.kind) else {
                throw unsupported("sampled texture must be a global binding")
            }
            return location.reference
        }
    }

    private func glslPair(_ sample: IRImageSample, externalPlane: UInt32? = nil) throws -> ShaderTextureSamplerPair {
        try ShaderTextureSamplerPair(sample, in: function, textureElement: selectedImages[sample.image]?.index,
                                     samplerElement: selectedImages[sample.sampler]?.index, externalPlane: externalPlane)
    }

    private func externalTextureReference(_ image: IRHandle<IRExpression>) throws -> ExternalTextureReference {
        let expression = try function.expressions.value(for: image)
        guard let reference = try storageLocation(expression.kind)?.externalTexture else {
            throw unsupported("Expected native external texture bindings")
        }
        return reference
    }

    private mutating func boundedImageIndex(_ index: IRHandle<IRExpression>, count: UInt32) throws -> String {
        let (offset, nonnegative) = try indexedValue(index)
        let upperBound = "min(\(offset), \(count - 1)u)"
        return nonnegative.map {
            outputFormat == .wgsl ? "select(0u, \(upperBound), \($0))" : "(\($0) ? \(upperBound) : 0u)"
        } ?? upperBound
    }

    private func bindingArrayReference(_ base: IRHandle<IRExpression>) throws -> (String, UInt32) {
        let expression = try function.expressions.value(for: base)
        guard let location = try storageLocation(expression.kind), !location.splitBindingArray,
              let count = location.bindingArrayCount, count > 0 else {
            throw unsupported("Expected a bounded texture or sampler binding array")
        }
        return (location.reference, count)
    }

    private mutating func clampedSampleLayer(
        _ sample: IRImageSample, image: String, named name: String, to lines: inout [String]
    ) throws -> String? {
        guard let layer = sample.arrayIndex else { return nil }
        if outputFormat == .wgsl { return try value(layer) }
        let (index, nonnegative) = try indexedValue(layer)
        let layers = "\(name)_layers"
        try writeStatements(imageLayers(image, named: layers), when: nil, to: &lines)
        let limit = "min(\(index), \(layers) - 1u)"
        return nonnegative.map { "(\($0) ? \(limit) : 0u)" } ?? limit
    }

    private mutating func sampleExpression(
        _ sample: IRImageSample, named name: String, to lines: inout [String]
    ) throws -> String {
        if let reference = sample.depthReference {
            return try comparisonExpression(sample, reference: reference, named: name, to: &lines)
        }
        let image = try function.expressions.value(for: sample.image)
        let kind = try module.types.value(for: image.type).kind
        if kind.isLineTexture, sample.offset != nil, outputFormat == .msl || outputFormat == .wgsl {
            throw unsupported("1D texture sample offset requires the HLSL or GLSL target")
        }
        if kind.isLineTexture, case .gradient = sample.level, outputFormat == .msl || outputFormat == .wgsl {
            throw unsupported("1D texture gradients require the HLSL or GLSL target")
        }
        if kind.isExternalTexture && outputFormat != .wgsl {
            let reference = try externalTextureReference(sample.image)
            let arguments: String
            if outputFormat == .glsl {
                let planes = try (0..<3).map { try glslPair(sample, externalPlane: $0).name }
                arguments = (planes + [reference.parameters]).joined(separator: ", ")
            } else {
                arguments = reference.arguments + ", " + (try imageReference(sample.sampler))
            }
            return "saga_external_sample(\(arguments), \(try value(sample.coordinate)))"
        }
        let depth = kind.isDepthTexture
        let cube = kind.isCubeTexture
        let texture = outputFormat == .glsl ? try glslPair(sample).name :
            try imageReference(sample.image)
        var uv = try value(sample.coordinate)
        if sample.clampToEdge {
            if outputFormat == .wgsl {
                let sampler = try imageReference(sample.sampler)
                return "textureSampleBaseClampToEdge(\(texture), \(sampler), \(uv))"
            }
            let extent = "\(name)_extent"
            for line in try imageDimensions(texture, level: "0u", named: extent) {
                try append(line, to: &lines)
            }
            let vector = floatType(2, in: outputFormat)
            let halfTexel = "\(name)_half_texel"
            try append("\(vector) \(halfTexel) = \(vector)(0.5) / \(vector)(\(extent));", to: &lines)
            uv = "clamp(\(uv), \(halfTexel), \(vector)(1.0) - \(halfTexel))"
        }
        let layer = try clampedSampleLayer(sample, image: texture, named: name, to: &lines)
        let names: (wgsl: String, hlsl: String, glsl: String)
        let lodArguments: [String]
        let metalModifier: String?
        switch sample.level {
        case .auto:
            names = ("textureSample", "Sample", "texture")
            lodArguments = []
            metalModifier = nil
        case .zero, .exact:
            let lod: String
            if case .exact(let operand) = sample.level {
                let value = try value(operand)
                lod = depth && outputFormat != .wgsl ? "float(\(value))" : value
            } else {
                lod = depth && outputFormat == .wgsl ? "0i" : floatLiteral(0)
            }
            names = ("textureSampleLevel", "SampleLevel", "textureLod")
            lodArguments = [lod]
            metalModifier = "level(\(lod))"
        case .bias(let operand):
            let bias = try value(operand)
            let clamped = "clamp(\(bias), \(floatLiteral(-16)), \(floatLiteral(15.99)))"
            names = ("textureSampleBias", "SampleBias", "texture")
            lodArguments = [outputFormat == .wgsl ? bias : clamped]
            metalModifier = "bias(\(clamped))"
        case .gradient(let x, let y):
            let dx = try value(x)
            let dy = try value(y)
            names = ("textureSampleGrad", "SampleGrad", "textureGrad")
            lodArguments = [dx, dy]
            let gradient = cube ? "gradientcube" : kind.isVolumeTexture ? "gradient3d" : "gradient2d"
            metalModifier = "\(gradient)(\(dx), \(dy))"
        }
        let offset = try sample.offset.map { try value($0) }
        let offsetArguments = offset.map { [$0] } ?? []
        if outputFormat == .glsl {
            let arguments: [String]
            if case .bias = sample.level {
                arguments = offsetArguments + lodArguments
            } else {
                arguments = lodArguments + offsetArguments
            }
            let coord = layer.map { "\(cube ? "vec4" : "vec3")(\(uv), float(\($0)))" } ?? uv
            let args = ([texture, coord] + arguments).joined(separator: ", ")
            let name = offset == nil ? names.glsl : "\(names.glsl)Offset"
            return "\(name)(\(args))" + (depth ? ".x" : "")
        }
        let sampler = try imageReference(sample.sampler)
        switch outputFormat {
        case .msl:
            let modifier = kind.isLineTexture ? nil : metalModifier
            let args = ([sampler, uv] + [layer, modifier].compactMap { $0 } + offsetArguments).joined(separator: ", ")
            return "\(texture).sample(\(args))"
        case .hlsl:
            let coord = layer.map { "\(cube ? "float4" : "float3")(\(uv), float(\($0)))" } ?? uv
            let args = ([sampler, coord] + lodArguments + offsetArguments).joined(separator: ", ")
            return "\(texture).\(names.hlsl)(\(args))"
        case .wgsl:
            let args = ([texture, sampler, uv] + [layer].compactMap { $0 } + lodArguments + offsetArguments).joined(separator: ", ")
            return "\(names.wgsl)(\(args))"
        default: throw unsupported("texture sample target")
        }
    }

    private mutating func comparisonExpression(
        _ sample: IRImageSample, reference: IRHandle<IRExpression>, named name: String, to lines: inout [String]
    ) throws -> String {
        let uv = try value(sample.coordinate)
        let reference = try value(reference)
        let offset = try sample.offset.map { try value($0) }
        let trailing = offset.map { [$0] } ?? []
        let levelZero = sample.level == .zero
        let imageType = try function.expressions.value(for: sample.image).type
        let cube = try module.types.value(for: imageType).kind.isCubeTexture
        let levelArguments: [String]
        switch sample.level {
        case .auto, .zero: levelArguments = []
        case .exact(let lod): levelArguments = [try value(lod)]
        case .gradient(let x, let y): levelArguments = try [value(x), value(y)]
        case .bias: throw unsupported("depth comparison with a LOD bias")
        }
        if outputFormat == .glsl {
            let pair = try glslPair(sample)
            let layer = try clampedSampleLayer(sample, image: pair.name, named: name, to: &lines)
            if cube, let layer {
                guard levelArguments.isEmpty else {
                    throw unsupported("GLSL cube-array shadow sampling with explicit gradients or LOD")
                }
                let coord = "vec4(\(uv), float(\(layer)))"
                return levelZero ? "textureLod(\(pair.name), \(coord), \(reference), 0.0)" :
                    "texture(\(pair.name), \(coord), \(reference))"
            }
            let coord = layer.map { "vec4(\(uv), float(\($0)), \(reference))" } ?? "vec3(\(uv), \(reference))"
            let shadowCoord = cube ? "vec4(\(uv), \(reference))" : coord
            // Zero gradients select the base mip for GLSL shadow samplers without a LOD overload.
            let operation: String
            let lod: [String]
            if case .gradient = sample.level {
                operation = "textureGrad"
                lod = levelArguments
            } else if case .exact = sample.level {
                guard !cube, layer == nil else {
                    throw unsupported("GLSL shadow sampling with explicit LOD requires a non-array 2D texture")
                }
                operation = "textureLod"
                lod = levelArguments
            } else if levelZero && (cube || layer != nil) {
                operation = "textureGrad"
                let gradient = cube ? "vec3(0.0)" : "vec2(0.0)"
                lod = [gradient, gradient]
            } else {
                operation = levelZero ? "textureLod" : "texture"
                lod = levelZero ? ["0.0"] : []
            }
            let args = ([pair.name, shadowCoord] + lod + trailing).joined(separator: ", ")
            return "\(operation)\(offset == nil ? "" : "Offset")(\(args))"
        }
        let texture = try imageReference(sample.image)
        let sampler = try imageReference(sample.sampler)
        let layer = try clampedSampleLayer(sample, image: texture, named: name, to: &lines)
        let base = [sampler, uv] + [layer].compactMap { $0 } + [reference]
        switch outputFormat {
        case .msl:
            let level: [String]
            switch sample.level {
            case .gradient:
                let modifier = cube ? "gradientcube" : "gradient2d"
                level = ["\(modifier)(\(levelArguments.joined(separator: ", ")))"]
            case .exact:
                level = ["level(\(levelArguments.joined()))"]
            default:
                level = levelZero ? ["level(0.0f)"] : []
            }
            let args = (base + level + trailing).joined(separator: ", ")
            return "\(texture).sample_compare(\(args))"
        case .hlsl:
            if cube, layer != nil, case .gradient = sample.level {
                throw unsupported("HLSL cube-array shadow sampling with explicit gradients")
            }
            let operation = switch sample.level {
            case .auto: "SampleCmp"
            case .zero: "SampleCmpLevelZero"
            case .exact: "SampleCmpLevel"
            case .gradient: "SampleCmpGrad"
            case .bias: throw unsupported("depth comparison with a LOD bias")
            }
            let coord = layer.map { "\(cube ? "float4" : "float3")(\(uv), float(\($0)))" } ?? uv
            let args = ([sampler, coord, reference] + levelArguments + trailing).joined(separator: ", ")
            return "\(texture).\(operation)(\(args))"
        case .wgsl:
            guard levelArguments.isEmpty else {
                throw unsupported("WGSL depth comparison with explicit gradients or nonzero LOD")
            }
            let operation = levelZero ? "textureSampleCompareLevel" : "textureSampleCompare"
            let args = ([texture] + base + trailing).joined(separator: ", ")
            return "\(operation)(\(args))"
        default: throw unsupported("depth comparison target")
        }
    }

    private mutating func gatherExpression(
        _ sample: IRImageSample, component: IRSwizzleComponent, named name: String, to lines: inout [String]
    ) throws -> String {
        guard let index = IRSwizzleComponent.allCases.firstIndex(of: component) else { throw ShaderFailure.invalidHandle }
        var uv = try value(sample.coordinate)
        let offset = try sample.offset.map { try value($0) }
        let offsetArguments = offset.map { [$0] } ?? []
        let reference = try sample.depthReference.map { try value($0) }
        let image = try function.expressions.value(for: sample.image)
        let kind = try module.types.value(for: image.type).kind
        let depth = kind.isDepthTexture
        let cube = kind.isCubeTexture
        if outputFormat == .glsl {
            let pair = try glslPair(sample)
            let trailing = reference.map { [$0] + offsetArguments } ?? (offsetArguments + [String(index)])
            let layer = try clampedSampleLayer(sample, image: pair.name, named: name, to: &lines)
            let coord = layer.map { "\(cube ? "vec4" : "vec3")(\(uv), float(\($0)))" } ?? uv
            let args = ([pair.name, coord] + trailing).joined(separator: ", ")
            let operation = offset == nil ? "textureGather" : "textureGatherOffset"
            return "\(operation)(\(args))"
        }
        let texture = try imageReference(sample.image)
        let sampler = try imageReference(sample.sampler)
        let layer = try clampedSampleLayer(sample, image: texture, named: name, to: &lines)
        if let offset, outputFormat == .msl || outputFormat == .wgsl {
            // Apple GPU gather ignored integer offsets in both native MSL and Chromium's WGSL path.
            // Center the gather between the four texels after moving the footprint in texel space.
            let coords = "\(name)_uv"
            let size = "\(name)_extent"
            let shift = "\(name)_offset"
            let vector = floatType(2, in: outputFormat)
            let declaration = outputFormat == .wgsl ? "let" : vector
            let dimensions = outputFormat == .wgsl ? "textureDimensions(\(texture))" :
                "\(texture).get_width(), \(texture).get_height()"
            try append("\(declaration) \(coords) = \(uv);", to: &lines)
            try append("\(declaration) \(size) = \(vector)(\(dimensions));", to: &lines)
            try append("\(declaration) \(shift) = \(vector)(\(offset));", to: &lines)
            let axes = ["x", "y"].map { axis in
                "(floor(\(coords).\(axis) * \(size).\(axis) - 0.5f) + \(shift).\(axis) + 1.0f) / \(size).\(axis)"
            }
            uv = "\(vector)(\(axes.joined(separator: ", ")))"
        }
        let base = [sampler, uv] + [layer].compactMap { $0 }
        switch outputFormat {
        case .msl:
            if let reference {
                let args = (base + [reference]).joined(separator: ", ")
                return "\(texture).gather_compare(\(args))"
            }
            let trailing = component == .x ? [] : (cube ? [] : ["int2(0)"]) + ["component::\(component.rawValue)"]
            let args = (base + trailing).joined(separator: ", ")
            return "\(texture).gather(\(args))"
        case .hlsl:
            let operation = depth ? (reference == nil ? "Gather" : "GatherCmp") :
                "Gather" + ["Red", "Green", "Blue", "Alpha"][index]
            // Select the explicit depth gather overload so glslang preserves a zero offset as ConstOffset.
            let offsets = depth && !cube && offsetArguments.isEmpty ? ["int2(0, 0)"] : offsetArguments
            let coord = layer.map { "\(cube ? "float4" : "float3")(\(uv), float(\($0)))" } ?? uv
            let args = ([sampler, coord] + [reference].compactMap { $0 } + offsets).joined(separator: ", ")
            return "\(texture).\(operation)(\(args))"
        case .wgsl:
            let args = ([texture] + base + [reference].compactMap { $0 }).joined(separator: ", ")
            if reference != nil { return "textureGatherCompare(\(args))" }
            let channel = depth ? "" : "\(index)u, "
            return "textureGather(\(channel)\(args))"
        default: throw unsupported("texture gather target")
        }
    }

    // Storage and multisampled views expose one level. Ordinary sampled textures take an explicit level.
    private func imageDimensions(
        _ image: String, level: String?, named name: String, kind: IRTypeKind = .sampledFloat2D
    ) throws -> [String] {
        let multisampled = kind.isMultisampled2D
        let arrayed = kind.isArrayedTexture
        let volume = kind.isVolumeTexture
        let line = kind.isLineTexture
        switch outputFormat {
        case .wgsl:
            let args = level.map { "\(image), \($0)" } ?? image
            return ["let \(name) = textureDimensions(\(args));"]
        case .msl:
            let mip = level ?? ""
            if line { return ["uint \(name) = \(image).get_width();"] }
            if kind.isCubeTexture { return ["uint2 \(name) = uint2(\(image).get_width(\(mip)));"] }
            if volume {
                return ["uint3 \(name) = uint3(\(image).get_width(\(mip)), \(image).get_height(\(mip)), \(image).get_depth(\(mip)));"]
            }
            return ["uint2 \(name) = uint2(\(image).get_width(\(mip)), \(image).get_height(\(mip)));"]
        case .glsl:
            let size = level.map { "textureSize(\(image), int(\($0)))" } ??
                (multisampled ? "textureSize(\(image))" : "imageSize(\(image))")
            let type = volume ? "uvec3" : line ? "uint" : "uvec2"
            return ["\(type) \(name) = \(type)(\(size)\(arrayed ? ".xy" : ""));"]
        case .hlsl:
            if line {
                if let level {
                    return ["uint \(name), \(name)_unused;", "\(image).GetDimensions(\(level), \(name), \(name)_unused);"]
                }
                return ["uint \(name);", "\(image).GetDimensions(\(name));"]
            }
            if volume, let level {
                return ["uint3 \(name);", "uint \(name)_unused;",
                        "\(image).GetDimensions(\(level), \(name).x, \(name).y, \(name).z, \(name)_unused);"]
            }
            if volume {
                return ["uint3 \(name);", "\(image).GetDimensions(\(name).x, \(name).y, \(name).z);"]
            }
            if arrayed, let level {
                return ["uint2 \(name);", "uint \(name)_layers, \(name)_unused;",
                        "\(image).GetDimensions(\(level), \(name).x, \(name).y, \(name)_layers, \(name)_unused);"]
            }
            if let level {
                return ["uint2 \(name);", "uint \(name)_unused;",
                        "\(image).GetDimensions(\(level), \(name).x, \(name).y, \(name)_unused);"]
            }
            if multisampled {
                return ["uint2 \(name);", "uint \(name)_unused;",
                        "\(image).GetDimensions(\(name).x, \(name).y, \(name)_unused);"]
            }
            if arrayed {
                return ["uint2 \(name);", "uint \(name)_layers;",
                        "\(image).GetDimensions(\(name).x, \(name).y, \(name)_layers);"]
            }
            return ["uint2 \(name);", "\(image).GetDimensions(\(name).x, \(name).y);"]
        default: throw unsupported("texture dimensions target")
        }
    }

    private func imageLevels(_ image: String, named name: String, kind: IRTypeKind) throws -> [String] {
        switch outputFormat {
        case .wgsl: return ["let \(name) = textureNumLevels(\(image));"]
        case .msl:
            let count = kind.isLineTexture ? "1u" : "\(image).get_num_mip_levels()"
            return ["uint \(name) = \(count);"]
        case .glsl: return ["uint \(name) = uint(textureQueryLevels(\(image)));"]
        case .hlsl:
            if kind.isLineTexture {
                return ["uint \(name), \(name)_width;", "\(image).GetDimensions(0u, \(name)_width, \(name));"]
            }
            if kind.isArrayedTexture || kind.isVolumeTexture {
                return ["uint \(name), \(name)_width, \(name)_height, \(name)_third;",
                        "\(image).GetDimensions(0u, \(name)_width, \(name)_height, \(name)_third, \(name));"]
            }
            return ["uint \(name), \(name)_width, \(name)_height;",
                    "\(image).GetDimensions(0u, \(name)_width, \(name)_height, \(name));"]
        default: throw unsupported("texture level count target")
        }
    }

    private func imageLayers(_ image: String, named name: String, storage: Bool = false) throws -> [String] {
        switch outputFormat {
        case .wgsl: return ["let \(name) = textureNumLayers(\(image));"]
        case .msl: return ["uint \(name) = \(image).get_array_size();"]
        case .glsl:
            let size = storage ? "imageSize(\(image))" : "textureSize(\(image), 0)"
            return ["uint \(name) = uint(\(size).z);"]
        case .hlsl:
            if storage {
                return ["uint \(name), \(name)_width, \(name)_height;",
                        "\(image).GetDimensions(\(name)_width, \(name)_height, \(name));"]
            }
            return ["uint \(name), \(name)_width, \(name)_height, \(name)_levels;",
                    "\(image).GetDimensions(0u, \(name)_width, \(name)_height, \(name), \(name)_levels);"]
        default: throw unsupported("texture array layer count target")
        }
    }

    private func imageSamples(_ image: String, named name: String) throws -> [String] {
        switch outputFormat {
        case .wgsl: return ["let \(name) = textureNumSamples(\(image));"]
        case .msl: return ["uint \(name) = \(image).get_num_samples();"]
        case .glsl: return ["uint \(name) = uint(textureSamples(\(image)));"]
        case .hlsl:
            return ["uint \(name), \(name)_width, \(name)_height;",
                    "\(image).GetDimensions(\(name)_width, \(name)_height, \(name));"]
        default: throw unsupported("texture sample count target")
        }
    }

    private func imageBounds(_ clauses: [String]) -> String {
        // WGSL's short-circuit operators lower to mutable locals in the frontend.
        clauses.map { "(\($0))" }.joined(separator: outputFormat == .wgsl ? " & " : " && ")
    }

    private mutating func writeImageQuery(
        _ query: IRImageQuery, image: IRHandle<IRExpression>, named name: String,
        type: IRHandle<IRType>, to lines: inout [String]
    ) throws {
        let imageType = try function.expressions.value(for: image).type
        let kind = try module.types.value(for: imageType).kind
        if kind.isExternalTexture && outputFormat != .wgsl {
            let reference = try externalTextureReference(image)
            let binding = declaration(try typeName(type), named: name)
            try append("\(binding) = saga_external_dimensions(\(reference.planes[0]), \(reference.parameters));", to: &lines)
            return
        }
        let image = try imageReference(image)
        switch query {
        case .numLevels:
            try writeStatements(imageLevels(image, named: name, kind: kind), when: nil, to: &lines)
        case .numSamples:
            try writeStatements(imageSamples(image, named: name), when: nil, to: &lines)
        case .numLayers:
            try writeStatements(imageLayers(image, named: name, storage: kind.storageTexture != nil),
                                when: nil, to: &lines)
        case .size(nil):
            let level = kind.storageTexture != nil || kind.isMultisampled2D || kind.isExternalTexture ? nil : "0u"
            try writeStatements(imageDimensions(image, level: level, named: name, kind: kind), when: nil, to: &lines)
        case .size(let level?):
            let (index, nonnegative) = try indexedValue(level)
            let levels = "\(name)_levels"
            try writeStatements(imageLevels(image, named: levels, kind: kind), when: nil, to: &lines)
            let initial = try zeroValue(type)
            let size = "\(name)_size"
            let bounds = imageBounds([nonnegative, "\(index) < \(levels)"].compactMap { $0 })
            if outputFormat == .wgsl {
                let valid = "\(name)_valid_level"
                try append("let \(valid) = \(bounds);", to: &lines)
                try writeStatements(imageDimensions(image, level: "select(0u, \(index), \(valid))", named: size, kind: kind),
                                    when: nil, to: &lines)
                try append("let \(name) = select(\(initial), \(size), \(valid));", to: &lines)
            } else {
                try append("\(declaration(try typeName(type), named: name)) = \(initial);", to: &lines)
                try writeStatements(imageDimensions(image, level: index, named: size, kind: kind) + ["\(name) = \(size);"],
                                    when: bounds, to: &lines)
            }
        }
    }

    private mutating func writeImageLoad(
        _ load: IRImageLoad, named name: String, type: IRHandle<IRType>, to lines: inout [String]
    ) throws {
        let imageType = try function.expressions.value(for: load.image).type
        let imageKind = try module.types.value(for: imageType).kind
        if imageKind.isExternalTexture {
            if outputFormat == .wgsl {
                try writeStorageImageLoad(load, named: name, type: type, scalar: .f32, components: 4,
                                          writable: false, kind: imageKind, to: &lines)
            } else {
                let reference = try externalTextureReference(load.image)
                let binding = declaration(try typeName(type), named: name)
                let coordinate = outputFormat == .glsl ? "uvec2" : "uint2"
                try append("\(binding) = saga_external_load(\(reference.arguments), \(coordinate)(\(try value(load.coordinate))));", to: &lines)
            }
            return
        }
        if let storage = imageKind.storageTexture {
            try writeStorageImageLoad(load, named: name, type: type, scalar: storage.scalar, components: storage.components,
                                      writable: storage.access.contains(.store), kind: imageKind, to: &lines)
            return
        }
        let multisampled = imageKind.isMultisampled2D
        let volume = imageKind.isVolumeTexture
        let line = imageKind.isLineTexture
        guard let operand = multisampled ? load.sample : load.level else {
            throw unsupported("sampled texture load without sample index or level")
        }
        let image = try imageReference(load.image)
        let (coordinates, unsigned, signed) = try imageCoordinates(load.coordinate, named: name, kind: imageKind, to: &lines)
        let (index, nonnegative) = try indexedValue(operand)
        let count = "\(name)_count"
        let query = multisampled ? try imageSamples(image, named: count) :
            try imageLevels(image, named: count, kind: imageKind)
        try writeStatements(query, when: nil, to: &lines)
        let initial = try zeroValue(type)
        let size = "\(name)_size"
        var indexChecks = [nonnegative, "\(index) < \(count)"].compactMap { $0 }
        let layer: String?
        if let arrayIndex = load.arrayIndex {
            let (value, nonnegativeLayer) = try indexedValue(arrayIndex)
            let layers = "\(name)_layers"
            try writeStatements(imageLayers(image, named: layers), when: nil, to: &lines)
            indexChecks += [nonnegativeLayer, "\(value) < \(layers)"].compactMap { $0 }
            layer = value
        } else {
            layer = nil
        }
        let indexBounds = imageBounds(indexChecks)
        let bounds = imageCoordinateBounds(coordinates, unsigned: unsigned, size: size, signed: signed, kind: imageKind)
        if outputFormat == .wgsl {
            // Use a valid address and select the zero result so uniform inputs stay uniform after re-parsing.
            let validIndex = "\(name)_valid_index"
            let safeIndex = "\(name)_index"
            let valid = "\(name)_valid"
            let safeCoordinates = "\(name)_safe_coordinates"
            try append("let \(validIndex) = \(indexBounds);", to: &lines)
            try append("let \(safeIndex) = select(0u, \(index), \(validIndex));", to: &lines)
            try writeStatements(imageDimensions(image, level: multisampled ? nil : safeIndex, named: size, kind: imageKind),
                                when: nil, to: &lines)
            try append("let \(valid) = \(validIndex) & \(bounds);", to: &lines)
            let zero = volume ? "vec3u(0u)" : line ? "0u" : "vec2u(0u)"
            try append("let \(safeCoordinates) = select(\(zero), \(unsigned), \(valid));", to: &lines)
            var args = [image, safeCoordinates]
            if let layer { args.append("select(0u, \(layer), \(validIndex))") }
            args.append(safeIndex)
            let texel = "textureLoad(\(args.joined(separator: ", ")))"
            try append("let \(name) = select(\(initial), \(texel), \(valid));", to: &lines)
            return
        }
        try append("\(declaration(try typeName(type), named: name)) = \(initial);", to: &lines)
        try append("if (\(indexBounds)) {", to: &lines)
        indentation += "    "
        try writeStatements(imageDimensions(image, level: multisampled ? nil : index, named: size, kind: imageKind),
                            when: nil, to: &lines)
        let texel: String
        switch outputFormat {
        case .msl:
            let args = [unsigned, layer, line ? nil : index].compactMap { $0 }.joined(separator: ", ")
            texel = "\(image).read(\(args))"
        case .hlsl:
            let location = "\(volume ? "int3" : "int2")(\(coordinates))"
            if line {
                texel = "\(image).Load(int2(int(\(coordinates)), int(\(index))))"
            } else if let layer {
                texel = "\(image).Load(int4(\(location), int(\(layer)), int(\(index))))"
            } else {
                texel = multisampled ? "\(image).Load(\(location), int(\(index)))" :
                    "\(image).Load(\(volume ? "int4" : "int3")(\(location), int(\(index))))"
            }
        case .glsl:
            let component = imageKind.isDepthTexture ? ".r" : ""
            let coordinate = "\(volume ? "ivec3" : line ? "int" : "ivec2")(\(coordinates))"
            let location = layer.map { "ivec3(\(coordinate), int(\($0)))" } ?? coordinate
            texel = "texelFetch(\(image), \(location), int(\(index)))\(component)"
        default: throw unsupported("sampled texture load target")
        }
        try writeStatements(["\(name) = \(texel);"], when: bounds, to: &lines)
        indentation.removeLast(4)
        try append("}", to: &lines)
    }

    private mutating func imageCoordinates(
        _ coordinate: IRHandle<IRExpression>, named name: String, kind: IRTypeKind, to lines: inout [String]
    ) throws -> (raw: String, unsigned: String, signed: Bool) {
        let type = try function.expressions.value(for: coordinate).type
        let signed = try module.types.value(for: type).kind == kind.texelCoordinateType(.i32)
        let coordinates = "\(name)_coordinates"
        let binding = outputFormat == .wgsl ? "let \(coordinates)" : "\(try typeName(type)) \(coordinates)"
        try append("\(binding) = \(try value(coordinate));", to: &lines)
        let width: UInt8? = kind.isLineTexture ? nil : kind.isVolumeTexture ? 3 : 2
        let unsigned = signed ? integerBits(coordinates, signed: false, width: width, in: outputFormat) : coordinates
        return (coordinates, unsigned, signed)
    }

    private mutating func writeStorageImageLoad(
        _ load: IRImageLoad, named name: String, type: IRHandle<IRType>,
        scalar: IRScalar, components: Int, writable: Bool, kind: IRTypeKind, to lines: inout [String]
    ) throws {
        let image = try imageReference(load.image)
        let (coordinates, unsigned, layer, bounds) = try storageImageAddress(
            image, coordinate: load.coordinate, arrayIndex: load.arrayIndex, named: name, kind: kind, to: &lines)
        let initial = try zeroValue(type)
        if outputFormat == .wgsl && !writable {
            let valid = "\(name)_valid"
            let safe = "\(name)_safe_coordinates"
            try append("let \(valid) = \(bounds);", to: &lines)
            let zero = kind.isVolumeTexture ? "vec3u(0u)" : kind.isLineTexture ? "0u" : "vec2u(0u)"
            try append("let \(safe) = select(\(zero), \(unsigned), \(valid));", to: &lines)
            let slice = layer.map { ", select(0u, \($0), \(valid))" } ?? ""
            try append("let \(name) = select(\(initial), textureLoad(\(image), \(safe)\(slice)), \(valid));", to: &lines)
            return
        }
        var texel: String
        switch outputFormat {
        case .wgsl: texel = "textureLoad(\(image), \(coordinates)\(layer.map { ", \($0)" } ?? ""))"
        case .msl: texel = "\(image).read(\(unsigned)\(layer.map { ", \($0)" } ?? ""))"
        case .hlsl:
            let location = layer.map { "uint3(\(unsigned), \($0))" } ?? unsigned
            texel = "\(image)[\(location)]"
        case .glsl:
            let coord = "\(kind.isVolumeTexture ? "ivec3" : kind.isLineTexture ? "int" : "ivec2")(\(coordinates))"
            let location = layer.map { "ivec3(\(coord), int(\($0)))" } ?? coord
            texel = "imageLoad(\(image), \(location))"
        default: throw unsupported("storage texture load target")
        }
        if components < 4 && (outputFormat == .msl || outputFormat == .hlsl) {
            let zero = scalarZero(scalar)
            let one = scalar == .u64 ? unsigned64Literal(1, in: outputFormat) :
                scalar == .u32 ? "1u" : scalar == .f32 ? "1.0f" : "1"
            if outputFormat == .msl || components == 3 { texel += "." + String("xyz".prefix(components)) }
            let lanes = [texel] + Array(repeating: zero, count: 3 - components) + [one]
            texel = "\(try typeName(type))(\(lanes.joined(separator: ", ")))"
        }
        try append("\(declaration(try typeName(type), named: name)) = \(initial);", to: &lines)
        try writeStatements(["\(name) = \(texel);"], when: bounds, to: &lines)
    }

    private func imageCoordinateBounds(_ coordinates: String, unsigned: String, size: String, signed: Bool,
                                       kind: IRTypeKind) -> String {
        let zero = outputFormat == .wgsl ? "0i" : "0"
        let lanes = kind.isLineTexture ? [""] : kind.isVolumeTexture ? [".x", ".y", ".z"] : [".x", ".y"]
        let positive = signed ? lanes.map { "\(coordinates)\($0) >= \(zero)" } : []
        return imageBounds(positive + lanes.map { "(\(unsigned))\($0) < \(size)\($0)" })
    }

    private mutating func writeImageStore(
        image: IRHandle<IRExpression>, coordinate: IRHandle<IRExpression>, arrayIndex: IRHandle<IRExpression>?,
        value rhs: IRHandle<IRExpression>,
        named name: String, to lines: inout [String]
    ) throws {
        let type = try function.expressions.value(for: image).type
        let kind = try module.types.value(for: type).kind
        guard let storage = kind.storageTexture else {
            throw unsupported("storage texture store type")
        }
        let image = try imageReference(image)
        let (coordinates, unsigned, layer, bounds) = try storageImageAddress(
            image, coordinate: coordinate, arrayIndex: arrayIndex, named: name, kind: kind, to: &lines)
        let color = try value(rhs)
        let store: String
        switch outputFormat {
        case .wgsl: store = "textureStore(\(image), \(coordinates)\(layer.map { ", \($0)" } ?? ""), \(color));"
        case .msl: store = "\(image).write(\(color), \(unsigned)\(layer.map { ", \($0)" } ?? ""));"
        case .hlsl:
            let value = switch storage.components {
            case 1: "(\(color)).x"
            case 2: "(\(color)).xy"
            default: color
            }
            let location = layer.map { "uint3(\(unsigned), \($0))" } ?? unsigned
            store = "\(image)[\(location)] = \(value);"
        case .glsl:
            let coord = "\(kind.isVolumeTexture ? "ivec3" : kind.isLineTexture ? "int" : "ivec2")(\(coordinates))"
            let location = layer.map { "ivec3(\(coord), int(\($0)))" } ?? coord
            store = "imageStore(\(image), \(location), \(color));"
        default: throw unsupported("storage texture target")
        }
        try writeStatements([store], when: bounds, to: &lines)
    }

    private mutating func storageImageAddress(
        _ image: String, coordinate: IRHandle<IRExpression>, arrayIndex: IRHandle<IRExpression>?,
        named name: String, kind: IRTypeKind, to lines: inout [String]
    ) throws -> (coordinates: String, unsigned: String, layer: String?, bounds: String) {
        let (coordinates, unsigned, signed) = try imageCoordinates(coordinate, named: name, kind: kind, to: &lines)
        let size = "\(name)_size"
        try writeStatements(imageDimensions(image, level: nil, named: size, kind: kind),
                            when: nil, to: &lines)
        let bounds = imageCoordinateBounds(coordinates, unsigned: unsigned, size: size, signed: signed, kind: kind)
        guard let arrayIndex else { return (coordinates, unsigned, nil, bounds) }
        let (layer, nonnegative) = try indexedValue(arrayIndex)
        let layers = "\(name)_layers"
        try writeStatements(imageLayers(image, named: layers, storage: true), when: nil, to: &lines)
        let layerBounds = [nonnegative, "\(layer) < \(layers)"].compactMap { $0 }
        return (coordinates, unsigned, layer, imageBounds([bounds] + layerBounds))
    }

    private mutating func writeImageAtomic(
        image: IRHandle<IRExpression>, coordinate: IRHandle<IRExpression>, arrayIndex: IRHandle<IRExpression>?,
        function operationKind: IRAtomicFunction, value operand: IRHandle<IRExpression>, named name: String, to lines: inout [String]
    ) throws {
        let operation: String = switch operationKind {
        case .add: "Add"
        case .min: "Min"
        case .max: "Max"
        case .and: "And"
        case .inclusiveOr: "Or"
        case .exclusiveOr: "Xor"
        default: throw unsupported("texture atomic operation")
        }
        let type = try function.expressions.value(for: image).type
        let kind = try module.types.value(for: type).kind
        let root = try SplitImageAccess(image, in: function)?.base ?? image
        let rootExpression = try function.expressions.value(for: root)
        let writeGuard = try storageLocation(rootExpression.kind)?.metalAtomicWriteGuard
        let image = try imageReference(image)
        let (coordinates, unsigned, layer, bounds) = try storageImageAddress(
            image, coordinate: coordinate, arrayIndex: arrayIndex, named: name, kind: kind, to: &lines)
        let rhs = try value(operand)
        let call: String
        switch outputFormat {
        case .wgsl:
            call = "textureAtomic\(operation)(\(image), \(coordinates)\(layer.map { ", \($0)" } ?? ""), \(rhs));"
        case .msl:
            let prefix = kind.storageTexture?.scalar == .u64 ? "atomic_" : "atomic_fetch_"
            call = "\(image).\(prefix)\(operation.lowercased())(\(unsigned)\(layer.map { ", \($0)" } ?? ""), \(rhs));"
        case .hlsl:
            let location = layer.map { "uint3(\(unsigned), \($0))" } ?? unsigned
            call = "Interlocked\(operation)(\(image)[\(location)], \(rhs));"
        case .glsl:
            let coord = "\(kind.isVolumeTexture ? "ivec3" : kind.isLineTexture ? "int" : "ivec2")(\(coordinates))"
            let location = layer.map { "ivec3(\(coord), int(\($0)))" } ?? coord
            call = "imageAtomic\(operation)(\(image), \(location), \(rhs));"
        default: throw unsupported("texture atomic target")
        }
        var statements = [call]
        if let writeGuard, let scalar = kind.storageTexture?.scalar {
            // Keep a texture write visible to the Metal fragment driver. The backend binds zero.
            let zero = scalarZero(scalar)
            let vector = try shaderVectorType(4, scalar: scalar, in: outputFormat)
            let location = "\(unsigned)\(layer.map { ", \($0)" } ?? "")"
            statements.append("if (\(writeGuard) != 0u) { \(image).write(\(vector)(\(zero)), \(location)); }")
        }
        try writeStatements(statements, when: bounds, to: &lines)
    }

    private mutating func selectExpression(
        condition: IRHandle<IRExpression>, accept: IRHandle<IRExpression>, reject: IRHandle<IRExpression>,
        type: IRHandle<IRType>
    ) throws -> String {
        let test = try value(condition)
        let accepted = try value(accept)
        let rejected = try value(reject)
        if outputFormat == .wgsl { return "select(\(rejected), \(accepted), \(test))" }
        let conditionType = try function.expressions.value(for: condition).type
        if case .vector(let width, .bool) = try module.types.value(for: conditionType).kind {
            let lanes = "xyzw".prefix(Int(width.rawValue))
            return try construct(type, components: lanes.map { lane in
                "((\(test)).\(lane) ? (\(accepted)).\(lane) : (\(rejected)).\(lane))"
            })
        }
        return "(\(test) ? \(accepted) : \(rejected))"
    }

    private mutating func writeLoad(
        pointer: IRHandle<IRExpression>, source: ShaderAddress, type: IRHandle<IRType>,
        name: String, to lines: inout [String]
    ) throws {
        if let scalar = try atomicScalar(at: pointer) {
            try writeAtomic(.load, target: source, scalar: scalar, result: name, temporary: name, to: &lines)
            return
        }
        let declaration = declaration(try typeName(type), named: name)
        let value: String
        if case .aggregate(let access) = source.memory {
            value = try loadNested(access)
        } else {
            value = try source.memory.load(in: outputFormat)
        }
        if let condition = source.boundsCheck, safeBufferRead {
            let captured = "\(name)_in_bounds"
            try append("let \(captured) = \(value);", to: &lines)
            try append("let \(name) = \(try selectCompositeValue(type, reference: captured, valid: condition));", to: &lines)
        } else if let condition = source.boundsCheck {
            let fallback = try zeroValue(type)
            try append("\(declaration) = \(fallback);", to: &lines)
            try writeStatements(["\(name) = \(value);"], when: condition, to: &lines)
        } else if outputFormat == .wgsl {
            try append("let \(name) = \(value);", to: &lines)
        } else {
            try append("\(declaration) = \(value);", to: &lines)
        }
    }

    private mutating func writeUniformLoad(
        pointer: IRHandle<IRExpression>, result: IRHandle<IRExpression>, to lines: inout [String]
    ) throws {
        let source = try address(pointer, guardIndexForUniformLoad: outputFormat == .wgsl)
        let type = try function.expressions.value(for: result).type
        let name = "saga_uniform_\(result.index)"
        if outputFormat == .wgsl {
            guard case .native(let reference) = source.memory else { throw unsupported("uniform load address") }
            var call = "workgroupUniformLoad(&\(reference))"
            if try atomicScalar(at: pointer) == .f32 { call = try floatAtomicBitcast(call, to: .f32) }
            if let bounds = source.boundsCheck {
                switch try module.types.value(for: type).kind {
                case .scalar, .vector:
                    let initial = try zeroValue(type)
                    try append("let \(name) = select(\(initial), \(call), \(bounds));", to: &lines)
                default:
                    let captured = name + "_captured"
                    let valid = name + "_valid"
                    try append("let \(captured) = \(call);", to: &lines)
                    try append("let \(valid) = \(bounds);", to: &lines)
                    let value = try selectCompositeValue(type, reference: captured, valid: valid)
                    try append("let \(name) = \(value);", to: &lines)
                }
            } else {
                try append("let \(name) = \(call);", to: &lines)
            }
        } else {
            let barrier = try shaderControlBarrier(.workgroup, in: outputFormat)
            for line in barrier { try append(line, to: &lines) }
            try writeLoad(pointer: pointer, source: source, type: type, name: name, to: &lines)
            for line in barrier { try append(line, to: &lines) }
        }
        remember(result)
    }

    // WGSL select accepts scalar and vector values. Rebuild larger values from
    // selected components so a uniform value stays uniform after bounds handling.
    private mutating func selectCompositeValue(
        _ type: IRHandle<IRType>, reference: String, valid: String, reject: String? = nil
    ) throws -> String {
        try budget.consume(.work)
        try budget.enterDepth()
        defer { try? budget.leaveDepth() }
        switch try module.types.value(for: type).kind {
        case .scalar, .vector:
            let fallback = try reject ?? zeroValue(type)
            return "select(\(fallback), \(reference), \(valid))"
        case .matrix(let columns, let rows, let scalar):
            let column = try shaderVectorType(rows.rawValue, scalar: scalar, in: outputFormat)
            let zero = "\(column)(\(scalarZero(scalar)))"
            let values = (0..<columns.rawValue).map { index in
                let fallback = reject.map { "\($0)[\(index)u]" } ?? zero
                return "select(\(fallback), \(reference)[\(index)u], \(valid))"
            }
            return try construct(type, components: values)
        case .array(let element, .constant(let count), _):
            try budget.consume(.nodes, amount: Int(count))
            let values = try (0..<count).map { index in
                try selectCompositeValue(element, reference: "\(reference)[\(index)u]", valid: valid,
                                      reject: reject.map { "\($0)[\(index)u]" })
            }
            return try construct(type, components: values)
        case .structure(let members, _):
            let values = try members.enumerated().map { index, field in
                let path = try member(UInt32(index), of: type)
                return try selectCompositeValue(field.type, reference: reference + path, valid: valid,
                                             reject: reject.map { $0 + path })
            }
            return try construct(type, components: values)
        default: throw unsupported("WGSL aggregate selection result type")
        }
    }

    private func atomicScalar(at handle: IRHandle<IRExpression>) throws -> IRScalar? {
        let expression = try function.expressions.value(for: handle)
        guard case .pointer(let type, _) = try module.types.value(for: expression.type).kind,
              case .atomic(let scalar) = try module.types.value(for: type).kind else { return nil }
        return scalar
    }

    private mutating func writeCompareExchange(
        target: ShaderAddress, scalar: IRScalar, expected: String, replacement: String, strong: Bool,
        result: (name: String, type: IRHandle<IRType>)?, temporary: String, to lines: inout [String]
    ) throws {
        if outputFormat == .msl, scalar.width == 8 {
            throw unsupported("Metal supports only unsigned 64-bit storage min/max atomics")
        }
        let lane = try shaderVectorType(1, scalar: scalar, in: outputFormat)
        let name = result?.name
        if let result {
            let initial = try zeroValue(result.type)
            try append("\(declaration(try typeName(result.type), named: result.name)) = \(initial);", to: &lines)
        }
        if outputFormat == .hlsl, let raw = target.memory.hlslAtomicByteAddress {
            let compared = "\(temporary)_expected"
            let signed32 = scalar == .i32
            let compareBits = signed32 ? "asuint(\(expected))" : expected
            let replacementBits = signed32 ? "asuint(\(replacement))" : replacement
            let bitsType = signed32 ? "uint" : lane
            let suffix = scalar.width == 8 ? "64" : ""
            var operations = [
                "\(bitsType) \(compared) = \(compareBits);",
                "\(bitsType) \(temporary);",
                "\(raw.buffer).InterlockedCompareExchange\(suffix)(\(raw.offset), \(compared), \(replacementBits), \(temporary));",
            ]
            if let name {
                let old = signed32 ? "asint(\(temporary))" : temporary
                operations += ["\(name).saga_member_0 = \(old);", "\(name).saga_member_1 = \(temporary) == \(compared);"]
            }
            try writeStatements(operations, when: target.boundsCheck, to: &lines)
            return
        }
        if case .glslAtomicWord(let reference) = target.memory {
            let compared = temporary + "_expected"
            let signed = scalar == .i32
            let expectedBits = signed ? "uint(\(expected))" : expected
            let replacementBits = signed ? "uint(\(replacement))" : replacement
            var operations = [
                "uint \(compared) = \(expectedBits);",
                "uint \(temporary) = atomicCompSwap(\(reference), \(compared), \(replacementBits));",
            ]
            if let name {
                let previous = signed ? "int(\(temporary))" : temporary
                operations += ["\(name).saga_member_0 = \(previous);", "\(name).saga_member_1 = \(temporary) == \(compared);"]
            }
            try writeStatements(operations, when: target.boundsCheck, to: &lines)
            return
        }
        guard case .native(let reference) = target.memory else { throw unsupported("atomic compare-exchange address") }
        let oldValue = name.map { "\($0).saga_member_0" } ?? temporary
        let statements: [String]
        switch outputFormat {
        case .msl:
            let newValue = strong ? temporary + "_replacement" : replacement
            let call = "atomic_compare_exchange_weak_explicit(&(\(reference)), &\(oldValue), \(newValue), " +
                "memory_order_relaxed, memory_order_relaxed)"
            if strong {
                let compared = temporary + "_expected"
                let exchanged = name.map { "\($0).saga_member_1" } ?? temporary + "_exchanged"
                var operations = ["const \(lane) \(compared) = \(expected);", "const \(lane) \(newValue) = \(replacement);"]
                if name == nil { operations += ["\(lane) \(oldValue);", "bool \(exchanged);"] }
                statements = operations + [
                    "do {",
                    "    \(oldValue) = \(compared);",
                    "    \(exchanged) = \(call);",
                    "} while (!\(exchanged) && \(oldValue) == \(compared));",
                ]
            } else {
                let start = name == nil ? "\(lane) \(oldValue)" : oldValue
                let assign = name.map { "\($0).saga_member_1 = " } ?? ""
                statements = ["\(start) = \(expected);", "\(assign)\(call);"]
            }
        case .hlsl, .glsl:
            let compared = "\(temporary)_expected"
            var operations = ["\(lane) \(compared) = \(expected);"]
            if outputFormat == .hlsl {
                if name == nil { operations.append("\(lane) \(oldValue);") }
                operations.append("InterlockedCompareExchange(\(reference), \(compared), \(replacement), \(oldValue));")
            } else {
                let call = "atomicCompSwap(\(reference), \(compared), \(replacement))"
                operations.append(name == nil ? "\(call);" : "\(oldValue) = \(call);")
            }
            if let name { operations.append("\(name).saga_member_1 = \(oldValue) == \(compared);") }
            statements = operations
        case .wgsl:
            if strong {
                let compared = temporary + "_expected"
                let newValue = temporary + "_replacement"
                var operations = [
                    "let \(compared) = \(expected);",
                    "let \(newValue) = \(replacement);",
                    "loop {",
                    "    let \(temporary) = atomicCompareExchangeWeak(&\(reference), \(compared), \(newValue));",
                    "    if (\(temporary).exchanged || \(temporary).old_value != \(compared)) {",
                ]
                if let name {
                    operations += ["        \(oldValue) = \(temporary).old_value;",
                                   "        \(name).saga_member_1 = \(temporary).exchanged;"]
                }
                statements = operations + ["        break;", "    }", "}"]
            } else if let name {
                let call = "atomicCompareExchangeWeak(&\(reference), \(expected), \(replacement))"
                statements = ["let \(temporary) = \(call);", "\(oldValue) = \(temporary).old_value;",
                              "\(name).saga_member_1 = \(temporary).exchanged;"]
            } else {
                statements = ["_ = atomicCompareExchangeWeak(&\(reference), \(expected), \(replacement));"]
            }
        default: throw unsupported("atomic compare-exchange target")
        }
        try writeStatements(statements, when: target.boundsCheck, to: &lines)
    }

    private mutating func writeAtomic(
        _ operation: ShaderAtomicOperation, target: ShaderAddress, scalar: IRScalar,
        result: String?, temporary: String, constantFloatBits: String? = nil, to lines: inout [String]
    ) throws {
        if case .readModifyWrite(.addFloat32Bits, let bits) = operation {
            guard scalar == .u32 else { throw unsupported("Float32 bit-pattern atomic add requires u32 storage") }
            let delta = try floatAtomicBitcast(bits, to: .f32)
            try writeFloatAtomic(.readModifyWrite(.add, delta), target: target, result: result,
                                 temporary: temporary, returnsBits: true, to: &lines)
            return
        }
        if scalar == .f32 {
            try writeFloatAtomic(operation, target: target, result: result, temporary: temporary,
                                 constantFloatBits: constantFloatBits, to: &lines)
            return
        }
        if outputFormat == .msl, scalar.width == 8 {
            guard scalar == .u64, result == nil, case .native(let reference) = target.memory,
                  case .readModifyWrite(let function, let value) = operation, function == .min || function == .max else {
                throw unsupported("Metal supports only unsigned 64-bit storage min/max atomics")
            }
            let name = function == .min ? "min" : "max"
            let call = "atomic_\(name)_explicit(&(\(reference)), \(value), memory_order_relaxed);"
            try writeStatements([call], when: target.boundsCheck, to: &lines)
            return
        }
        let zero = scalarZero(scalar)
        let type = try shaderVectorType(1, scalar: scalar, in: outputFormat)
        if let result { try append("\(declaration(type, named: result)) = \(zero);", to: &lines) }
        let assignment = result.map { "\($0) = " } ?? ""
        let loweredOperation: ShaderAtomicOperation
        let comparisonSigned: Bool
        if case .readModifyWrite(let function, let value) = operation,
           function == .minWithOppositeSignedness || function == .maxWithOppositeSignedness {
            guard scalar.kind == .sint || scalar.kind == .uint else {
                throw unsupported("Opposite-signed atomic min/max requires an integer")
            }
            let extremum: IRAtomicFunction = function == .minWithOppositeSignedness ? .min : .max
            if case .native(let reference) = target.memory {
                try writeOppositeSignedNativeAtomic(extremum, reference: reference, operand: value, scalar: scalar,
                                                   target: target, result: result, temporary: temporary, to: &lines)
                return
            }
            loweredOperation = .readModifyWrite(extremum, value)
            comparisonSigned = scalar.kind == .uint
        } else {
            loweredOperation = operation
            comparisonSigned = scalar.kind == .sint
        }
        let (intrinsic, operand) = try atomicIntrinsic(loweredOperation, scalar: scalar, zero: zero)
        if outputFormat == .hlsl, let raw = target.memory.hlslAtomicByteAddress, let operand {
            let statements: [String]
            if scalar.width == 8 {
                let unsignedBitwise = scalar == .i64 && operation.bitwiseOperator != nil
                let retag = unsignedBitwise || comparisonSigned != (scalar.kind == .sint)
                let operandType = comparisonSigned && !unsignedBitwise ? "int64_t" : "uint64_t"
                let input = retag ? "\(operandType)(\(operand))" : operand
                let old = retag ? temporary + "_bits" : result ?? temporary
                var operations = result == nil || retag ? ["\(operandType) \(old);"] : []
                operations.append("\(raw.buffer).\(intrinsic)64(\(raw.offset), \(input), \(old));")
                if retag, let result { operations.append("\(result) = \(type)(\(old));") }
                statements = operations
            } else {
                statements = hlslRawAtomic(loweredOperation, intrinsic: intrinsic, operand: operand, signed: scalar == .i32,
                                           comparisonSigned: comparisonSigned, raw: raw, result: result, temporary: "\(temporary)_bits")
            }
            try writeStatements(statements, when: target.boundsCheck, to: &lines)
            return
        }
        if case .glslAtomicWord(let reference) = target.memory, let operand {
            let statements = glslRawAtomic(loweredOperation, intrinsic: intrinsic, operand: operand, signed: scalar == .i32,
                                           comparisonSigned: comparisonSigned, reference: reference, result: result,
                                           temporary: temporary + "_bits")
            try writeStatements(statements, when: target.boundsCheck, to: &lines)
            return
        }
        guard case .native(let reference) = target.memory else { throw unsupported("atomic address representation") }
        let call: String
        switch outputFormat {
        case .msl:
            let arguments = operand.map { ", \($0)" } ?? ""
            call = "\(intrinsic)(&(\(reference))\(arguments), memory_order_relaxed)"
        case .wgsl:
            let arguments = operand.map { ", \($0)" } ?? ""
            call = "\(intrinsic)(&\(reference)\(arguments))"
        case .glsl:
            guard let operand else { throw unsupported("GLSL atomic operand") }
            call = "\(intrinsic)(\(reference), \(operand))"
        case .hlsl:
            guard let operand else { throw unsupported("HLSL atomic operand") }
            if scalar == .i64, let symbol = operation.bitwiseOperator {
                let previous = temporary + "_previous"
                let expected = temporary + "_expected"
                // HLSL exposes 64-bit bitwise intrinsics only on unsigned destinations.
                var statements = [
                    "int64_t \(previous);",
                    "InterlockedAdd(\(reference), \(zero), \(previous));",
                    "int64_t \(expected);",
                    "do {",
                    "    \(expected) = \(previous);",
                    "    InterlockedCompareExchange(\(reference), \(expected), \(expected) \(symbol) (\(operand)), \(previous));",
                    "} while (\(previous) != \(expected));",
                ]
                if let result { statements.append("\(result) = \(previous);") }
                try writeStatements(statements, when: target.boundsCheck, to: &lines)
                return
            }
            var statements: [String] = []
            if result == nil { statements.append("\(type) \(temporary);") }
            statements.append("\(intrinsic)(\(reference), \(operand), \(result ?? temporary));")
            try writeStatements(statements, when: target.boundsCheck, to: &lines)
            return
        default: throw unsupported("atomic operation target")
        }
        try writeStatements(["\(assignment)\(call);"], when: target.boundsCheck, to: &lines)
    }

    private mutating func writeOppositeSignedNativeAtomic(
        _ extremum: IRAtomicFunction, reference: String, operand: String, scalar: IRScalar,
        target: ShaderAddress, result: String?, temporary: String, to lines: inout [String]
    ) throws {
        let physicalType = try shaderVectorType(1, scalar: scalar, in: outputFormat)
        let comparisonScalar = IRScalar(kind: scalar.kind == .sint ? .uint : .sint, width: scalar.width)
        let comparisonType = try shaderVectorType(1, scalar: comparisonScalar, in: outputFormat)
        let input = temporary + "_operand"
        let current = temporary + "_current"
        let previous = temporary + "_previous"
        let name = extremum == .min ? "min" : "max"
        let comparedOperand = try bitcastValue(operand, from: scalar, to: comparisonScalar, type: comparisonType)
        let comparedPrevious = try bitcastValue(previous, from: scalar, to: comparisonScalar, type: comparisonType)
        let replacement = try bitcastValue("\(name)(\(comparedPrevious), \(input))", from: comparisonScalar,
                                           to: scalar, type: physicalType)
        var onSuccess: [String] = []
        if let result { onSuccess.append("        \(result) = \(previous);") }
        onSuccess.append("        break;")

        let statements: [String]
        if outputFormat == .wgsl {
            let attempt = temporary + "_attempt"
            statements = [
                "let \(input) = \(comparedOperand);",
                "var \(current) = atomicLoad(&\(reference));",
                "loop {",
                "    let \(previous) = \(current);",
                "    let \(attempt) = atomicCompareExchangeWeak(&\(reference), \(previous), \(replacement));",
                "    if (\(attempt).exchanged) {",
            ] + onSuccess + ["    }", "    \(current) = \(attempt).old_value;", "}"]
        } else {
            let zero = scalarZero(scalar)
            let load: [String]
            let attempt: [String]
            let succeeded: String
            switch outputFormat {
            case .msl:
                load = ["\(physicalType) \(current) = atomic_load_explicit(&(\(reference)), memory_order_relaxed);"]
                attempt = []
                succeeded = "atomic_compare_exchange_weak_explicit(&(\(reference)), &\(current), \(replacement), " +
                    "memory_order_relaxed, memory_order_relaxed)"
            case .hlsl:
                load = ["\(physicalType) \(current);", "InterlockedAdd(\(reference), \(zero), \(current));"]
                attempt = ["    InterlockedCompareExchange(\(reference), \(previous), \(replacement), \(current));"]
                succeeded = "\(current) == \(previous)"
            case .glsl:
                load = ["\(physicalType) \(current) = atomicAdd(\(reference), \(zero));"]
                attempt = ["    \(current) = atomicCompSwap(\(reference), \(previous), \(replacement));"]
                succeeded = "\(current) == \(previous)"
            default: throw unsupported("opposite-signed atomic min/max target")
            }
            let hint = outputFormat == .hlsl ? "[allow_uav_condition] " : ""
            statements = ["\(comparisonType) \(input) = \(comparedOperand);"] + load + [
                "\(hint)while (true) {", "    \(physicalType) \(previous) = \(current);",
            ] + attempt + ["    if (\(succeeded)) {"] + onSuccess + ["    }", "}"]
        }
        // The initial load and every retry must stay inside the same robust-address guard.
        try writeStatements(statements, when: target.boundsCheck, to: &lines)
    }

    private mutating func writeFloatAtomic(
        _ operation: ShaderAtomicOperation, target: ShaderAddress, result: String?,
        temporary: String, returnsBits: Bool = false, constantFloatBits: String? = nil, to lines: inout [String]
    ) throws {
        // Float atomics use the same four bytes as uint atomics. Compare bit patterns so NaNs and signed zero cannot
        // turn an otherwise successful compare-exchange into a failed floating-point comparison.
        let bits = temporary + "_bits"
        switch operation {
        case .store(let value):
            let replacement = try constantFloatBits ?? floatAtomicBitcast(value, to: .u32)
            try writeAtomic(.store(replacement), target: target, scalar: .u32,
                            result: nil, temporary: bits, to: &lines)
            return
        case .load:
            try writeAtomic(.load, target: target, scalar: .u32, result: bits, temporary: bits, to: &lines)
        case .readModifyWrite(.exchange(compare: nil, strong: false), let value):
            let replacement = try constantFloatBits ?? floatAtomicBitcast(value, to: .u32)
            try writeAtomic(.readModifyWrite(.exchange(compare: nil), replacement), target: target, scalar: .u32,
                            result: result == nil ? nil : bits, temporary: bits, to: &lines)
        case .readModifyWrite(let function, let value) where function == .add || function == .subtract:
            try writeAtomic(.load, target: target, scalar: .u32, result: bits, temporary: bits, to: &lines)
            let delta = temporary + "_delta"
            let previous = temporary + "_previous"
            let symbol = function == .add ? "+" : "-"
            let oldFloat = try floatAtomicBitcast(previous, to: .f32)
            let update = try floatAtomicBitcast("(\(oldFloat) \(symbol) \(delta))", to: .u32)
            let statements: [String]
            switch (outputFormat, target.memory) {
            case (.wgsl, .native(let reference)):
                let attempt = temporary + "_attempt"
                statements = [
                    "let \(delta) = \(value);",
                    "loop {",
                    "    let \(previous) = \(bits);",
                    "    let \(attempt) = atomicCompareExchangeWeak(&\(reference), \(previous), \(update));",
                    "    if (\(attempt).exchanged) { break; }",
                    "    \(bits) = \(attempt).old_value;",
                    "}",
                ]
            case (.msl, .native(let reference)):
                statements = [
                    "float \(delta) = \(value);",
                    "while (true) {",
                    "    uint \(previous) = \(bits);",
                    "    if (atomic_compare_exchange_weak_explicit(&(\(reference)), &\(bits), \(update), " +
                        "memory_order_relaxed, memory_order_relaxed)) { break; }",
                    "}",
                ]
            default:
                let exchange: String
                if outputFormat == .hlsl, let raw = target.memory.hlslAtomicByteAddress {
                    exchange = "\(raw.buffer).InterlockedCompareExchange(\(raw.offset), \(previous), \(update), \(bits));"
                } else {
                    switch (outputFormat, target.memory) {
                    case (.glsl, .native(let reference)), (.glsl, .glslAtomicWord(let reference)):
                        exchange = "\(bits) = atomicCompSwap(\(reference), \(previous), \(update));"
                    case (.hlsl, .native(let reference)):
                        exchange = "InterlockedCompareExchange(\(reference), \(previous), \(update), \(bits));"
                    default: throw unsupported("float atomic address")
                    }
                }
                let hint = outputFormat == .hlsl ? "[allow_uav_condition] " : ""
                statements = [
                    "float \(delta) = \(value);",
                    "\(hint)while (true) {",
                    "    uint \(previous) = \(bits);",
                    "    \(exchange)",
                    "    if (\(bits) == \(previous)) { break; }",
                    "}",
                ]
            }
            try writeStatements(statements, when: target.boundsCheck, to: &lines)
        default: throw unsupported("float atomic operation")
        }
        if let result {
            let type = try returnsBits ? shaderVectorType(1, scalar: .u32, in: outputFormat) : floatType(1, in: outputFormat)
            let value = try returnsBits ? bits : floatAtomicBitcast(bits, to: .f32)
            try append("\(declaration(type, named: result)) = \(value);", to: &lines)
        }
    }

    private func floatAtomicBitcast(_ value: String, to output: IRScalar) throws -> String {
        try bitcastValue(value, from: output == .f32 ? .u32 : .f32, to: output,
                         type: shaderVectorType(1, scalar: output, in: outputFormat))
    }

    private func wgslFloatAtomicConstantBits(_ operand: IRHandle<IRExpression>) throws -> String? {
        guard outputFormat == .wgsl else { return nil }
        let kind = try function.expressions.value(for: operand).kind
        let literal: IRExpressionKind
        if case .constant(let constant) = kind {
            let initializer = try module.constants.value(for: constant).initializer
            literal = try module.globalExpressions.value(for: initializer).kind
        } else {
            literal = kind
        }
        guard case .literal(.f32(let value)) = literal else { return nil }
        // Keep literal bits out of floating-point constant folding on the physical u32 path.
        return "0x\(String(value.bitPattern, radix: 16))u"
    }

    private func glslRawAtomic(
        _ operation: ShaderAtomicOperation, intrinsic: String, operand: String, signed: Bool,
        comparisonSigned: Bool, reference: String, result: String?, temporary: String
    ) -> [String] {
        if comparisonSigned, case .readModifyWrite(let function, _) = operation, function == .min || function == .max {
            // The raw SSBO word is unsigned. Compare in signed space, then CAS the bit pattern.
            let comparison = function == .min ? "min" : "max"
            let input = temporary + "_operand"
            let previous = temporary + "_previous"
            let next = temporary + "_next"
            var lines = [
                "int \(input) = \(signed ? operand : "int(\(operand))");",
                "uint \(temporary) = atomicOr(\(reference), 0u);",
                "while (true) {",
                "    uint \(previous) = \(temporary);",
                "    uint \(next) = uint(\(comparison)(int(\(previous)), \(input)));",
                "    \(temporary) = atomicCompSwap(\(reference), \(previous), \(next));",
                "    if (\(temporary) == \(previous)) { break; }",
                "}",
            ]
            if let result { lines.append("\(result) = \(signed ? "int(\(temporary))" : temporary);") }
            return lines
        }
        let bits = signed ? "uint(\(operand))" : operand
        let call = "\(intrinsic)(\(reference), \(bits))"
        guard let result else { return ["\(call);"] }
        let previous = signed ? "int(\(call))" : call
        return ["\(result) = \(previous);"]
    }

    private func hlslRawAtomic(
        _ operation: ShaderAtomicOperation, intrinsic: String, operand: String, signed: Bool,
        comparisonSigned: Bool, raw: (buffer: String, offset: String), result: String?, temporary: String
    ) -> [String] {
        var lines = ["uint \(temporary);"]
        if comparisonSigned, case .readModifyWrite(let function, _) = operation, function == .min || function == .max {
            // Some HLSL compilers select unsigned min/max for raw-buffer methods.
            let byteOffset = "\(temporary)_offset"
            let previous = "\(temporary)_previous"
            let next = "\(temporary)_next"
            let extremum = function == .min ? "min" : "max"
            lines += [
                "uint \(byteOffset) = \(raw.offset);",
                "\(raw.buffer).InterlockedAdd(\(byteOffset), 0u, \(temporary));",
                "[allow_uav_condition] while (true) {",
                "    uint \(previous) = \(temporary);",
                "    uint \(next) = asuint(\(extremum)(asint(\(previous)), \(signed ? operand : "asint(\(operand))")));",
                "    \(raw.buffer).InterlockedCompareExchange(\(byteOffset), \(previous), \(next), \(temporary));",
                "    if (\(temporary) == \(previous)) { break; }",
                "}",
            ]
        } else {
            let bits = signed ? "asuint(\(operand))" : operand
            lines.append("\(raw.buffer).\(intrinsic)(\(raw.offset), \(bits), \(temporary));")
        }
        if let result {
            let value = signed ? "asint(\(temporary))" : temporary
            lines.append("\(result) = \(value);")
        }
        return lines
    }

    private func atomicIntrinsic(
        _ operation: ShaderAtomicOperation, scalar: IRScalar, zero: String
    ) throws -> (String, String?) {
        let names: (msl: String, hlsl: String, glsl: String, wgsl: String)
        var operand: String?
        switch operation {
        case .load:
            names = ("atomic_load_explicit", "InterlockedAdd", "atomicAdd", "atomicLoad")
            operand = outputFormat == .glsl || outputFormat == .hlsl ? zero : nil
        case .store(let value):
            names = ("atomic_store_explicit", "InterlockedExchange", "atomicExchange", "atomicStore")
            operand = value
        case .readModifyWrite(let function, let value):
            guard let builtin = function.integerRMWBuiltin else { throw unsupported("atomic read-modify-write operation") }
            let metal: String
            let hlsl: String
            switch function {
            case .add: (metal, hlsl) = ("atomic_fetch_add_explicit", "InterlockedAdd")
            case .subtract: (metal, hlsl) = ("atomic_fetch_sub_explicit", "InterlockedAdd")
            case .min: (metal, hlsl) = ("atomic_fetch_min_explicit", "InterlockedMin")
            case .max: (metal, hlsl) = ("atomic_fetch_max_explicit", "InterlockedMax")
            case .minWithOppositeSignedness, .maxWithOppositeSignedness, .addFloat32Bits:
                throw unsupported("Atomic operation requires explicit lowering")
            case .and: (metal, hlsl) = ("atomic_fetch_and_explicit", "InterlockedAnd")
            case .inclusiveOr: (metal, hlsl) = ("atomic_fetch_or_explicit", "InterlockedOr")
            case .exclusiveOr: (metal, hlsl) = ("atomic_fetch_xor_explicit", "InterlockedXor")
            case .exchange: (metal, hlsl) = ("atomic_exchange_explicit", "InterlockedExchange")
            }
            names = (metal, hlsl, function == .subtract ? "atomicAdd" : builtin, builtin)
            if function == .subtract && (outputFormat == .glsl || outputFormat == .hlsl) {
                switch scalar {
                case .i32: operand = signedNegation(value, in: outputFormat)
                case .i64:
                    let bits = "\(unsigned64Literal(0, in: outputFormat)) - uint64_t(\(value))"
                    operand = "int64_t(\(bits))"
                default: operand = "\(zero) - (\(value))"
                }
            } else {
                operand = value
            }
        }
        switch outputFormat {
        case .msl: return (names.msl, operand)
        case .hlsl: return (names.hlsl, operand)
        case .glsl: return (names.glsl, operand)
        case .wgsl: return (names.wgsl, operand)
        default: throw unsupported("atomic operation target")
        }
    }

    private mutating func writeGLSLRayQuery(
        _ query: IRHandle<IRExpression>, operation: IRRayQueryFunction, to lines: inout [String]
    ) throws {
        guard case .localVariable(let local) = try function.expressions.value(for: query).kind,
              case .rayQuery(false) = try module.types.value(for: function.localVariables.value(for: local).type).kind
        else { throw unsupported("GLSL ray query must be a local without vertex return") }
        let name = localName(local.index), state = "\(name)_state"
        switch operation {
        case .begin: try append("\(state) = 0u;", to: &lines)
        case .initialize(let acceleration, let descriptor):
            guard let descriptorType = module.specialTypes.rayDescriptor, let layout = structures[descriptorType] else {
                throw unsupported("GLSL ray query descriptor layout")
            }
            let desc = "saga_ray_desc_\(query.index)"
            let fields = (0..<6).map { "\(desc).\(layout.memberName($0))" }
            let source = try value(descriptor), type = try typeName(descriptorType), accel = try value(acceleration)
            try append("{", to: &lines)
            try append("    \(state) = 0u;", to: &lines)
            try append("    \(type) \(desc) = \(source);", to: &lines)
            try append("    if (!any(isnan(\(fields[4]))) && !any(isinf(\(fields[4]))) && !any(isnan(\(fields[5]))) && !any(isinf(\(fields[5]))) && !isnan(\(fields[2])) && !isinf(\(fields[2])) && !isnan(\(fields[3])) && \(fields[2]) >= 0.0 && \(fields[2]) <= \(fields[3]) && \(glslRayFlagsValid(fields[0]))) {", to: &lines)
            try append("        rayQueryInitializeEXT(\(name), \(accel), \(fields[0]), \(fields[1]), \(fields[4]), \(fields[2]), \(fields[5]), \(fields[3]));", to: &lines)
            try append("        \(name)_tmax = \(fields[3]);", to: &lines)
            try append("        \(state) = 1u;", to: &lines)
            try append("    }", to: &lines)
            try append("}", to: &lines)
        case .proceed(let result):
            for line in glslRayQueryProceed(query: name, state: state, result: "saga_ray_proceed_\(result.index)") { try append(line, to: &lines) }
            remember(result)
        case .terminate: try append(glslRayQueryTerminate(query: name, state: state), to: &lines)
        case .confirmIntersection: try append(glslRayQueryConfirm(query: name, state: state), to: &lines)
        case .generateIntersection(let hit):
            let distance = try value(hit)
            try append("if ((\(state) & 2u) != 0u && (\(state) & 4u) == 0u && rayQueryGetIntersectionTypeEXT(\(name), false) == gl_RayQueryCandidateIntersectionAABBEXT) {", to: &lines)
            try append("    float saga_hit_t = \(distance);", to: &lines)
            try append("    float saga_hit_limit = rayQueryGetIntersectionTypeEXT(\(name), true) == gl_RayQueryCommittedIntersectionNoneEXT ? \(name)_tmax : rayQueryGetIntersectionTEXT(\(name), true);", to: &lines)
            try append("    if (rayQueryGetRayTMinEXT(\(name)) <= saga_hit_t && saga_hit_t <= saga_hit_limit) { rayQueryGenerateIntersectionEXT(\(name), saga_hit_t); }", to: &lines)
            try append("}", to: &lines)
        }
    }

    private mutating func writeHLSLRayQuery(
        _ query: IRHandle<IRExpression>, operation: IRRayQueryFunction, to lines: inout [String]
    ) throws {
        guard case .localVariable(let local) = try function.expressions.value(for: query).kind,
              case .rayQuery(false) = try module.types.value(for: function.localVariables.value(for: local).type).kind
        else { throw unsupported("HLSL ray query must be a local without vertex return") }
        let name = localName(local.index), state = "\(name)_state"
        switch operation {
        case .begin:
            try append("\(state) = 0u;", to: &lines)
        case .initialize(let acceleration, let descriptor):
            guard let descriptorType = module.specialTypes.rayDescriptor,
                  let layout = structures[descriptorType]
            else { throw unsupported("HLSL ray query requires a ray descriptor") }
            let desc = "saga_ray_desc_\(query.index)"
            let fields = (0..<6).map { "\(desc).\(layout.memberName($0))" }
            let source = try value(descriptor), structure = try typeName(descriptorType)
            let accel = try value(acceleration)
            try append("{", to: &lines)
            try append("    \(state) = 0u;", to: &lines)
            try append("    \(structure) \(desc) = \(source);", to: &lines)
            try append("    uint saga_ray_flags = \(fields[0]);", to: &lines)
            try append("    bool saga_ray_flags_valid = \(hlslRayFlagsValidation("saga_ray_flags"));", to: &lines)
            try append("    if (saga_ray_flags_valid && \(hlslRayDescriptorValidation(fields: fields))) {", to: &lines)
            try append("        RayDesc saga_ray;", to: &lines)
            try append("        saga_ray.Origin = \(fields[4]); saga_ray.Direction = \(fields[5]); saga_ray.TMin = \(fields[2]); saga_ray.TMax = \(fields[3]);", to: &lines)
            try append("        \(name).TraceRayInline(\(accel), saga_ray_flags, \(fields[1]), saga_ray);", to: &lines)
            try append("        \(state) = 1u;", to: &lines)
            try append("    }", to: &lines)
            try append("}", to: &lines)
        case .proceed(let result):
            let temporary = "saga_ray_proceed_\(result.index)"
            try append("bool \(temporary) = false;", to: &lines)
            try append("if ((\(state) & 1u) != 0u && (\(state) & 4u) == 0u) {", to: &lines)
            try append("    \(temporary) = \(name).Proceed();", to: &lines)
            try append("    \(state) |= \(temporary) ? 2u : 4u;", to: &lines)
            try append("}", to: &lines)
            remember(result)
        case .terminate:
            try append(hlslRayTerminate(query: name, state: state), to: &lines)
        case .confirmIntersection:
            try append("if ((\(state) & 2u) != 0u && (\(state) & 4u) == 0u && \(name).CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE) { \(name).CommitNonOpaqueTriangleHit(); }", to: &lines)
        case .generateIntersection(let hit):
            let distance = try value(hit)
            try append("if ((\(state) & 2u) != 0u && (\(state) & 4u) == 0u) {", to: &lines)
            try append("    float saga_hit_t = \(distance);", to: &lines)
            try append("    if (\(name).CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE && \(name).RayTMin() <= saga_hit_t && saga_hit_t <= \(name).CommittedRayT()) { \(name).CommitProceduralPrimitiveHit(saga_hit_t); }", to: &lines)
            try append("}", to: &lines)
        }
    }

    private mutating func writeMetalRayQuery(
        _ query: IRHandle<IRExpression>, operation: IRRayQueryFunction, to lines: inout [String]
    ) throws {
        guard outputFormat == .msl,
              case .localVariable(let local) = try function.expressions.value(for: query).kind,
              case .rayQuery(false) = try module.types.value(for: function.localVariables.value(for: local).type).kind
        else { throw unsupported("Metal ray query must be a local without vertex return") }
        let name = localName(local.index), state = "\(name)_state"
        switch operation {
        case .begin:
            try append("\(state) = 0u;", to: &lines)
        case .proceed(let result):
            let temporary = "saga_ray_proceed_\(result.index)"
            try append("bool \(temporary) = false;", to: &lines)
            try append("if ((\(state) & 1u) != 0u && (\(state) & 4u) == 0u) {", to: &lines)
            try append("    \(temporary) = \(name).next();", to: &lines)
            try append("    \(state) |= \(temporary) ? 2u : 6u;", to: &lines)
            try append("}", to: &lines)
            remember(result)
        case .initialize(let acceleration, let descriptor):
            guard let descriptorType = module.specialTypes.rayDescriptor,
                  let layout = structures[descriptorType]
            else { throw unsupported("Metal ray query descriptor layout") }
            let desc = "saga_ray_desc_\(query.index)"
            let fields = (0..<6).map { "\(desc).\(layout.memberName($0))" }
            let source = try value(descriptor)
            let structure = try typeName(descriptorType)
            let accel = try value(acceleration)
            try append("{", to: &lines)
            try append("    \(state) = 0u;", to: &lines)
            try append("    \(structure) \(desc) = \(source);", to: &lines)
            try append("    if (all(isfinite(\(fields[4]))) && all(isfinite(\(fields[5]))) && isfinite(\(fields[2])) && !isnan(\(fields[3])) && \(fields[2]) >= 0.0f && \(fields[2]) <= \(fields[3]) && any(\(fields[5]) != 0.0f) && (\(fields[0]) & ~1023u) == 0u && popcount(\(fields[0]) & 195u) < 2 && popcount(\(fields[0]) & 304u) < 2 && popcount(\(fields[0]) & 768u) < 2) {", to: &lines)
            try append("        metal::raytracing::intersection_params saga_params;", to: &lines)
            try append("        saga_params.set_triangle_front_facing_winding(metal::winding::clockwise);", to: &lines)
            try append("        saga_params.set_triangle_cull_mode((\(fields[0]) & 16u) != 0u ? metal::raytracing::triangle_cull_mode::back : ((\(fields[0]) & 32u) != 0u ? metal::raytracing::triangle_cull_mode::front : metal::raytracing::triangle_cull_mode::none));", to: &lines)
            try append("        saga_params.set_geometry_cull_mode((\(fields[0]) & 256u) != 0u ? metal::raytracing::geometry_cull_mode::triangle : ((\(fields[0]) & 512u) != 0u ? metal::raytracing::geometry_cull_mode::bounding_box : metal::raytracing::geometry_cull_mode::none));", to: &lines)
            try append("        metal::raytracing::opacity_cull_mode saga_cull = (\(fields[0]) & 64u) != 0u ? metal::raytracing::opacity_cull_mode::opaque : ((\(fields[0]) & 128u) != 0u ? metal::raytracing::opacity_cull_mode::non_opaque : metal::raytracing::opacity_cull_mode::none);", to: &lines)
            try append("        saga_params.set_opacity_cull_mode(saga_cull);", to: &lines)
            try append("        if (saga_cull == metal::raytracing::opacity_cull_mode::none) { saga_params.force_opacity((\(fields[0]) & 1u) != 0u ? metal::raytracing::forced_opacity::opaque : ((\(fields[0]) & 2u) != 0u ? metal::raytracing::forced_opacity::non_opaque : metal::raytracing::forced_opacity::none)); }", to: &lines)
            try append("        saga_params.accept_any_intersection((\(fields[0]) & 4u) != 0u);", to: &lines)
            try append("        metal::raytracing::ray saga_ray(\(fields[4]), \(fields[5]), \(fields[2]), \(fields[3]));", to: &lines)
            try append("        \(name).reset(saga_ray, \(accel), \(fields[1]), saga_params);", to: &lines)
            try append("        \(state) = 1u;", to: &lines)
            try append("        \(name)_tmax = \(fields[3]);", to: &lines)
            try append("    }", to: &lines)
            try append("}", to: &lines)
        case .confirmIntersection:
            try append("if ((\(state) & 2u) != 0u && (\(state) & 4u) == 0u && \(name).get_candidate_intersection_type() == metal::raytracing::intersection_type::triangle) { \(name).commit_triangle_intersection(); }", to: &lines)
        case .generateIntersection(let hitT):
            let hit = try value(hitT)
            try append("if ((\(state) & 2u) != 0u && (\(state) & 4u) == 0u) {", to: &lines)
            try append("    float saga_hit_t = \(hit);", to: &lines)
            try append("    float saga_max_t = \(name).get_committed_intersection_type() == metal::raytracing::intersection_type::none ? \(name)_tmax : \(name).get_committed_distance();", to: &lines)
            try append("    if (\(name).get_candidate_intersection_type() == metal::raytracing::intersection_type::bounding_box && \(name).get_ray_min_distance() <= saga_hit_t && saga_hit_t <= saga_max_t) { \(name).commit_bounding_box_intersection(saga_hit_t); }", to: &lines)
            try append("}", to: &lines)
        case .terminate:
            try append("if ((\(state) & 2u) != 0u && (\(state) & 4u) == 0u) { \(name).abort(); }", to: &lines)
        }
    }

    private mutating func writeStatements(
        _ statements: [String], when condition: String?, to lines: inout [String]
    ) throws {
        if let condition, statements.count == 1 {
            try append("if (\(condition)) { \(statements[0]) }", to: &lines)
        } else {
            if let condition {
                try append("if (\(condition)) {", to: &lines)
                indentation += "    "
            }
            for statement in statements { try append(statement, to: &lines) }
            if condition != nil {
                indentation.removeLast(4)
                try append("}", to: &lines)
            }
        }
    }

    private mutating func address(
        _ handle: IRHandle<IRExpression>, guardIndexForUniformLoad: Bool = false
    ) throws -> ShaderAddress {
        try budget.consume(.work)
        try budget.enterDepth()
        defer { try? budget.leaveDepth() }
        let pointer = try function.expressions.value(for: handle)
        if let selected = try nestedAccess(handle) {
            return try .init(memory: nestedMemory(selected.access), boundsCheck: selected.bounds,
                             writable: selected.access.buffer.writable)
        }
        if case .functionArgument(let argument) = pointer.kind,
           case .pointer = try module.types.value(for: pointer.type).kind {
            guard let slot = Int(exactly: argument), arguments.indices.contains(slot) else {
                throw ShaderFailure.invalidHandle
            }
            let name = arguments[slot]
            let reference = outputFormat == .wgsl ? "(*\(name))" : name
            let valid = module.allowsPointerSubviews ? name + "_valid" : nil
            return .init(memory: .native(reference), boundsCheck: valid, writable: true)
        }
        if case .localVariable(let local) = pointer.kind {
            _ = try function.localVariables.value(for: local)
            return .init(memory: .native(localName(local.index)), boundsCheck: nil, writable: true)
        }
        if case .accessIndex(let base, let index) = pointer.kind {
            let source = try function.expressions.value(for: base)
            if case .pointer(let pointee, let space) = try module.types.value(for: source.type).kind,
               space == .function || space == .workgroup || space == .private,
               case .structure = try module.types.value(for: pointee).kind {
                let parent = try address(base, guardIndexForUniformLoad: guardIndexForUniformLoad)
                guard case .native(let reference) = parent.memory else { throw unsupported("structure storage address") }
                let field = try member(index, of: pointee)
                return .init(memory: .native(reference + field),
                             boundsCheck: parent.boundsCheck, writable: parent.writable)
            }
        }
        if let field = try storageLocation(pointer.kind), field.bound == nil, field.members.isEmpty {
            return .init(memory: field.memory(), boundsCheck: nil, writable: field.writable)
        }
        let base: IRHandle<IRExpression>
        let index: String
        var nonnegative: String?
        switch pointer.kind {
        case .access(let source, let element):
            base = source
            (index, nonnegative) = try indexedValue(element)
        case .accessIndex(let source, let element):
            base = source
            index = "\(element)u"
        default:
            throw unsupported("storage access must name a scalar or an indexed array element")
        }
        let source = try function.expressions.value(for: base)
        if case .pointer(let pointee, let space) = try module.types.value(for: source.type).kind,
           space == .function || space == .workgroup || space == .private,
           case .array(_, .constant(let count), _) = try module.types.value(for: pointee).kind,
           try storageLocation(source.kind)?.bound == nil {
            let parent = try address(base, guardIndexForUniformLoad: guardIndexForUniformLoad)
            guard case .native(let reference) = parent.memory else { throw ShaderFailure.invalidHandle }
            let bounds = addressBounds([parent.boundsCheck, nonnegative, "\(index) < \(count)u"],
                                       uniform: guardIndexForUniformLoad)
            let safeIndex = guardIndexForUniformLoad ? "select(0u, \(index), \(bounds))" : index
            return .init(memory: .native(reference + outputFormat.valueArrayIndex(safeIndex)),
                         boundsCheck: bounds, writable: parent.writable)
        }
        if case .pointer(let pointee, _) = try module.types.value(for: source.type).kind,
           let width = try module.types.value(for: pointee).kind.componentCount {
            let parent = try address(base, guardIndexForUniformLoad: guardIndexForUniformLoad)
            let safeRead = guardIndexForUniformLoad || safeBufferRead
            let bounds = addressBounds([parent.boundsCheck, nonnegative, "\(index) < \(width)u"],
                                       uniform: safeRead)
            let safeIndex = safeRead ? "select(0u, \(index), \(bounds))" : index
            return try .init(memory: parent.memory.component(safeIndex), boundsCheck: bounds, writable: parent.writable)
        }
        guard let binding = try storageLocation(source.kind), let bound = binding.bound else {
            throw unsupported("storage access must name an indexed array element")
        }
        let upperBound = "\(index) < \(bound)"
        let bounds = addressBounds([nonnegative, upperBound], uniform: guardIndexForUniformLoad)
        let safeIndex = guardIndexForUniformLoad ? "select(0u, \(index), \(bounds))" : index
        return .init(memory: binding.memory(at: safeIndex), boundsCheck: bounds, writable: binding.writable)
    }

    private mutating func cooperativeAddress(
        _ data: IRCooperativeData, tileType: IRHandle<IRType>, named name: String, to lines: inout [String]
    ) throws -> (pointer: String, stride: String, bounds: String, writable: Bool, buffer: String, index: String) {
        guard outputFormat == .msl || outputFormat == .glsl || outputFormat == .hlsl || outputFormat == .wgsl else { throw unsupported("cooperative matrix target") }
        guard case .cooperativeMatrix(let columns, let rows, _, _) = try module.types.value(for: tileType).kind else { throw unsupported("cooperative matrix type") }
        let minor = data.rowMajor ? columns.rawValue : rows.rawValue
        let major = data.rowMajor ? rows.rawValue : columns.rawValue
        let pointer = try function.expressions.value(for: data.pointer)
        let base: IRHandle<IRExpression>
        switch pointer.kind {
        case .access(let source, _), .accessIndex(let source, _): base = source
        default: throw unsupported("cooperative matrix requires an indexed storage array")
        }
        let source = try function.expressions.value(for: base)
        guard case .globalVariable(let global) = source.kind,
              try module.globalVariables.value(for: global).memoryDecorations.isEmpty else {
            throw unsupported("cooperative matrix requires an undecorated storage binding")
        }
        guard let binding = try storageLocation(source.kind), let bound = binding.bound else {
            throw unsupported("cooperative matrix requires a bounded storage array")
        }
        let target = try address(data.pointer)
        let (stride, nonnegative) = try indexedValue(data.stride)
        let indexName = name + "_index"
        let strideName = name + "_stride"
        if outputFormat == .wgsl {
            guard binding.members.isEmpty,
                  case .pointer(let pointee, _) = try module.types.value(for: pointer.type).kind,
                  case .scalar(let scalar) = try module.types.value(for: pointee).kind,
                  scalar == .f32 || scalar == .f16 else {
                throw unsupported("WGSL cooperative matrix requires scalar f16/f32 storage")
            }
            let elementIndex: String
            let nonnegativeIndex: String?
            switch pointer.kind {
            case .access(_, let index): (elementIndex, nonnegativeIndex) = try indexedValue(index)
            case .accessIndex(_, let index): (elementIndex, nonnegativeIndex) = ("\(index)u", nil)
            default: throw unsupported("WGSL cooperative matrix requires indexed scalar storage")
            }
            try append("let \(indexName): u32 = \(elementIndex);", to: &lines)
            try append("let \(strideName): u32 = \(stride);", to: &lines)
            let pointerName = "&\(binding.reference)[\(indexName)]"
            let remaining = "(\(bound) - \(indexName))"
            let footprint = "(\(indexName) <= \(bound) && \(remaining) >= \(minor)u && \(strideName) <= (\(remaining) - \(minor)u) / \(major - 1)u)"
            let bounds = [target.boundsCheck, nonnegativeIndex, nonnegative, footprint].compactMap { $0 }
                .map { "(\($0))" }.joined(separator: " && ")
            return (pointerName, strideName, bounds, target.writable, "", indexName)
        }
        if outputFormat == .glsl {
            guard binding.members.isEmpty,
                  case .pointer(let pointee, _) = try module.types.value(for: pointer.type).kind,
                  case .scalar(let scalar) = try module.types.value(for: pointee).kind,
                  scalar == .f32 || scalar == .f16 else {
                throw unsupported("GLSL cooperative matrix requires contiguous f16 or f32 scalar storage")
            }
            let arrayIndex: String
            let nonnegativeIndex: String?
            switch pointer.kind {
            case .access(_, let index): (arrayIndex, nonnegativeIndex) = try indexedValue(index)
            case .accessIndex(_, let index): (arrayIndex, nonnegativeIndex) = ("\(index)u", nil)
            default: throw unsupported("cooperative matrix requires an indexed storage array")
            }
            try append("uint \(indexName) = \(arrayIndex);", to: &lines)
            try append("uint \(strideName) = \(stride);", to: &lines)
            let footprint = "(\(indexName) <= \(bound) && \(bound) - \(indexName) >= \(minor)u && \(strideName) <= (\(bound) - \(indexName) - \(minor)u) / \(major - 1)u)"
            let bounds = [target.boundsCheck, nonnegativeIndex, nonnegative, footprint].compactMap { $0 }
                .map { "(\($0))" }.joined(separator: " && ")
            return ("", strideName, bounds, target.writable, binding.reference, indexName)
        }
        guard let words = binding.words, words.offset == 0, words.stride == 1, words.width == 1,
              words.columns == nil,
              (words.addressing == .structured || (outputFormat == .hlsl && words.addressing == .byteAddressed && words.scalar == .f16)),
              words.scalar == .f32 || words.scalar == .f16 else {
            throw unsupported("cooperative matrix storage requires contiguous scalar layout")
        }
        guard case .words(let buffer, let first, _, let scalar, _, _) = target.memory else {
            throw unsupported("cooperative matrix storage address")
        }
        try append("uint \(indexName) = \(first);", to: &lines)
        try append("uint \(strideName) = \(stride);", to: &lines)
        if outputFormat == .glsl || outputFormat == .hlsl {
            guard scalar == .f32 || scalar == .f16 else { throw unsupported("GLSL cooperative matrix scalar storage") }
            let footprint = "(\(indexName) <= \(bound) && \(bound) - \(indexName) >= \(minor)u && \(strideName) <= (\(bound) - \(indexName) - \(minor)u) / \(major - 1)u)"
            let bounds = [target.boundsCheck, nonnegative, footprint].compactMap { $0 }
                .map { "(\($0))" }.joined(separator: " && ")
            return ("", strideName, bounds, target.writable, buffer, indexName)
        }
        let qualifier = target.writable ? "device" : "device const"
        let lane = scalar == .f32 ? "float" : "half"
        let address = "(reinterpret_cast<\(qualifier) \(lane)*>(\(buffer)) + \(indexName))"
        let footprint = "ulong(\(indexName)) + \(major - 1)ul * ulong(\(strideName)) + \(minor)ul <= ulong(\(bound))"
        return (address, strideName, [target.boundsCheck, nonnegative, footprint]
            .compactMap { $0 }.map { "(\($0))" }.joined(separator: " && "), target.writable, buffer, indexName)
    }

    private mutating func nestedAccess(_ handle: IRHandle<IRExpression>) throws
        -> (access: NestedBufferAccess, bounds: String?)? {
        guard hasNestedBuffers else { return nil }
        try budget.consume(.work)
        try budget.enterDepth()
        defer { try? budget.leaveDepth() }
        let expression = try function.expressions.value(for: handle)
        guard case .pointer(_, let space) = try module.types.value(for: expression.type).kind,
              space.isBufferAddressSpace else { return nil }
        if case .globalVariable(let global) = expression.kind, let buffer = storage[global.index].nested {
            return (NestedBufferAccess(buffer), nil)
        }
        let base: IRHandle<IRExpression>
        let staticIndex: UInt32?
        let dynamicIndex: IRHandle<IRExpression>?
        switch expression.kind {
        case .accessIndex(let parent, let index):
            (base, staticIndex, dynamicIndex) = (parent, index, nil)
        case .access(let parent, let index):
            (base, staticIndex, dynamicIndex) = (parent, nil, index)
        default: return nil
        }
        let parent = try function.expressions.value(for: base)
        guard case .pointer(let type, let parentSpace) = try module.types.value(for: parent.type).kind,
              parentSpace.isBufferAddressSpace else { return nil }
        func offset() throws -> (String, String?) {
            if let dynamicIndex { return try indexedValue(dynamicIndex) }
            if let staticIndex { return ("\(staticIndex)u", nil) }
            throw ShaderFailure.invalidHandle
        }
        switch try module.types.value(for: type).kind {
        case .bindingArray:
            guard let selected = try nestedAccess(base), let array = selected.access.buffer.bindingArray else {
                return nil
            }
            if array.splitsArray {
                guard let element = webBufferElement, element.access == handle else { throw ShaderFailure.invalidHandle }
                return (NestedBufferAccess(array.select(String(element.index), in: selected.access.buffer)), nil)
            }
            let (index, nonnegative) = try offset()
            let bounds = addressBounds([nonnegative, "\(index) < \(array.count)u"], uniform: false)
            // Even a zero-result arrayLength query needs a valid resource index.
            let safe = boundedBufferValue(index, when: bounds)
            let buffer = array.select(safe, in: selected.access.buffer)
            return (NestedBufferAccess(buffer), bounds)
        case .structure(let members, _):
            guard let staticIndex, let selected = try nestedAccess(base) else { return nil }
            return (selected.access.member(members[Int(staticIndex)], at: staticIndex), selected.bounds)
        case .array(let element, let size, let stride):
            guard let selected = try nestedAccess(base) else { return nil }
            let (index, nonnegative) = try offset()
            let count: String
            switch size {
            case .constant(let length): count = "\(length)u"
            case .dynamic:
                guard let length = selected.access.buffer.runtimeLength else { throw ShaderFailure.invalidHandle }
                count = length
            default: throw ShaderFailure.invalidHandle
            }
            let bounds = addressBounds([selected.bounds, nonnegative, "\(index) < \(count)"], uniform: safeBufferRead)
            let safeIndex = safeBufferRead ? boundedBufferValue(index, when: bounds) : index
            return (selected.access.element(element, stride: stride, index: safeIndex), bounds)
        default:
            // Vector lanes and matrix columns are handled by the ordinary component address path.
            return nil
        }
    }

    private func boundedBufferValue(_ value: String, when bounds: String) -> String {
        if outputFormat == .wgsl { return "select(0u, \(value), \(bounds))" }
        return "((\(bounds)) ? \(value) : 0u)"
    }

    private func nestedMemory(_ access: NestedBufferAccess) throws -> ShaderMemory {
        let width: UInt8
        let columns: UInt8?
        let scalar: IRScalar
        switch try module.types.value(for: access.type).kind {
        case .structure, .array: return .aggregate(access)
        case .atomic(let lane):
            switch outputFormat {
            case .wgsl: return .native(access.reference)
            case .msl:
                let atomic = lane == .u64 ? "atomic_ulong" : lane == .i32 ? "atomic_int" : "atomic_uint"
                let qualifier = access.buffer.metalMemoryQualifier
                let reference = "reinterpret_cast<\(qualifier)device \(atomic)*>(\(access.buffer.name))[(\(access.byteOffset)) / \(lane.width)u]"
                return .native(reference)
            case .glsl:
                if access.buffer.glslReference != nil { return .native(access.reference) }
                return .glslAtomicWord("\(access.buffer.name).words[(\(access.byteOffset)) / 4u]")
            case .hlsl:
                return .words(buffer: access.buffer.name, first: "(\(access.byteOffset)) / \(lane.width)u",
                              width: 1, scalar: lane, addressing: .byteAddressed, columns: nil)
            default: throw unsupported("nested atomic target")
            }
        case .scalar(let lane): (width, columns, scalar) = (1, nil, lane)
        case .vector(let size, let lane): (width, columns, scalar) = (size.rawValue, nil, lane)
        case .matrix(let count, let rows, let lane): (width, columns, scalar) = (rows.rawValue, count.rawValue, lane)
        default: throw ShaderFailure.invalidHandle
        }
        if outputFormat == .wgsl || access.buffer.glslReference != nil { return .native(access.reference) }
        if access.buffer.space == .uniform || access.buffer.space == .immediate {
            let buffer: String
            switch outputFormat {
            case .msl where scalar.width == 2 || scalar.width == 8:
                let pointer = scalar.width == 8 ? "ulong" : try shaderVectorType(1, scalar: scalar, in: .msl)
                buffer = "reinterpret_cast<constant \(pointer)*>(\(access.buffer.name))"
            case .glsl:
                buffer = access.buffer.name + ".words"
            default:
                buffer = access.buffer.name
            }
            return .words(buffer: buffer, first: "(\(access.byteOffset)) / \(scalar.width)u", width: width, scalar: scalar,
                          addressing: outputFormat == .msl ? .structured : .uniformRegisters, columns: columns)
        }
        let buffer: String
        let addressing: ShaderStorage.WordAddressing
        switch (outputFormat, scalar) {
        case (.msl, _) where scalar.width == 2 || scalar.width == 8:
            let qualifier = access.buffer.metalMemoryQualifier + (access.buffer.writable ? "device" : "device const")
            let pointer = scalar.width == 8 ? "ulong" : try shaderVectorType(1, scalar: scalar, in: .msl)
            buffer = "reinterpret_cast<\(qualifier) \(pointer)*>(\(access.buffer.name))"
            addressing = .structured
        case (.hlsl, _):
            buffer = access.buffer.name
            addressing = .byteAddressed
        case (.glsl, _) where scalar.width == 2:
            buffer = access.buffer.name + ".words"
            addressing = .packedHalfWords(writable: access.buffer.writable)
        case (.glsl, _):
            buffer = access.buffer.name + ".words"
            addressing = .structured
        default:
            buffer = access.buffer.name
            addressing = .structured
        }
        return .words(buffer: buffer, first: "(\(access.byteOffset)) / \(scalar.width)u", width: width, scalar: scalar,
                      addressing: addressing, columns: columns)
    }

    private mutating func loadNested(_ access: NestedBufferAccess) throws -> String {
        try budget.consume(.work)
        try budget.enterDepth()
        defer { try? budget.leaveDepth() }
        switch try module.types.value(for: access.type).kind {
        case .structure(let members, _):
            let fields = try members.enumerated().map { index, member in
                try loadNested(access.member(member, at: UInt32(index)))
            }
            return try construct(access.type, components: fields)
        case .array(let element, .constant(let count), let stride):
            try budget.consume(.nodes, amount: Int(count))
            let values = try (0..<count).map { index in
                try loadNested(access.element(element, stride: stride, index: "\(index)u"))
            }
            return try construct(access.type, components: values)
        default:
            let value = try nestedMemory(access).load(in: outputFormat)
            try budget.consume(.sourceBytes, amount: value.utf8.count)
            return value
        }
    }

    private mutating func storeNested(_ access: NestedBufferAccess, value: String, temporary: String) throws -> [String] {
        try budget.consume(.work)
        try budget.enterDepth()
        defer { try? budget.leaveDepth() }
        var lines: [String] = []
        switch try module.types.value(for: access.type).kind {
        case .structure(let members, _):
            for (index, field) in members.enumerated() {
                let suffix = try member(UInt32(index), of: access.type)
                lines += try storeNested(access.member(field, at: UInt32(index)), value: value + suffix,
                                         temporary: temporary + "_\(index)")
            }
        case .array(let element, .constant(let count), let stride):
            try budget.consume(.nodes, amount: Int(count))
            for index in 0..<count {
                let subscriptValue = "\(index)u"
                let suffix = outputFormat.valueArrayIndex(subscriptValue)
                lines += try storeNested(access.element(element, stride: stride, index: subscriptValue), value: value + suffix,
                                         temporary: temporary + "_\(index)")
            }
        default:
            lines = try nestedMemory(access).stores(value, temporary: temporary, in: outputFormat)
            for line in lines { try budget.consume(.sourceBytes, amount: line.utf8.count + 1) }
        }
        return lines
    }

    private func addressBounds(_ checks: [String?], uniform: Bool) -> String {
        let tests = checks.compactMap { $0 }
        if uniform && tests.count > 1 {
            // These checks are pure and uniform. Avoid short-circuit lowering through a mutable temporary.
            return tests.map { "(\($0))" }.joined(separator: " & ")
        }
        return tests.joined(separator: " && ")
    }

    private mutating func indexedValue(_ handle: IRHandle<IRExpression>) throws -> (String, String?) {
        let expression = try function.expressions.value(for: handle)
        if outputFormat == .hlsl, case .literal(.i32(let index)) = expression.kind {
            // Emit the known bits directly instead of relying on HLSL literal overload folding.
            return ("\(UInt32(bitPattern: index))u", index < 0 ? "false" : nil)
        }
        let raw = try value(handle)
        guard try module.types.value(for: expression.type).kind == .scalar(.i32) else { return (raw, nil) }
        let zero = outputFormat == .wgsl ? "0i" : "0"
        return (integerBits(raw, signed: false, in: outputFormat), "\(raw) >= \(zero)")
    }

    private func storageLocation(_ expression: IRExpressionKind) throws -> ShaderStorage? {
        switch expression {
        case .globalVariable(let global) where storage.indices.contains(global.index):
            return storage[global.index]
        case .accessIndex(let base, let member):
            guard case .globalVariable(let global) = try function.expressions.value(for: base).kind,
                  storage.indices.contains(global.index),
                  let index = Int(exactly: member), storage[global.index].members.indices.contains(index)
            else { return nil }
            return storage[global.index].members[index]
        default:
            return nil
        }
    }

    private mutating func append(_ line: String, to lines: inout [String]) throws {
        try budget.consume(.sourceBytes, amount: indentation.utf8.count + line.utf8.count + 1)
        lines.append(indentation + line)
    }

    private func scalar(_ handle: IRHandle<IRType>) throws -> IRScalar {
        guard case .scalar(let type) = try module.types.value(for: handle).kind,
              [.u16, .i16, .u32, .i32, .u64, .i64, .bool, .f64, .f32, .f16].contains(type)
        else { throw unsupported("expected a numeric or boolean scalar") }
        return type
    }

    private func localName(_ index: Int) -> String { "saga_local_\(index)" }

    private func derivativeExpression(_ axis: IRDerivativeAxis, control: IRDerivativeControl, operand: String) throws -> String {
        let operations: (metal: String, hlsl: String, glsl: String) = switch axis {
        case .x: ("dfdx", "ddx", "dFdx")
        case .y: ("dfdy", "ddy", "dFdy")
        case .width: ("fwidth", "fwidth", "fwidth")
        }
        let function: String
        switch outputFormat {
        case .wgsl: function = axis.wgslName(control: control)
        case .msl: function = operations.metal
        case .glsl: function = operations.glsl + control.nameSuffix
        case .hlsl:
            let suffix = control == .none ? "" : "_\(control.rawValue)"
            if axis == .width && control != .none {
                return "(abs(ddx\(suffix)(\(operand))) + abs(ddy\(suffix)(\(operand))))"
            }
            function = operations.hlsl + suffix
        default: throw unsupported("derivative target")
        }
        return "\(function)(\(operand))"
    }

    private func bitcastShape(_ type: IRHandle<IRType>) throws -> IRBitcastShape {
        guard let shape = try IRBitcastShape(module.types.value(for: type).kind),
              [.i16, .u16, .i32, .u32, .i64, .u64, .f32, .f16].contains(shape.scalar) else {
            throw unsupported("bitcast value type")
        }
        return shape
    }

    private mutating func bitcast(_ operand: IRHandle<IRExpression>, to target: IRHandle<IRType>) throws -> String {
        let inputShape = try bitcastShape(function.expressions.value(for: operand).type)
        let outputShape = try bitcastShape(target)
        guard inputShape.permits(outputShape) else { throw unsupported("bitcast lane types or shapes") }
        let input = inputShape.scalar
        let output = outputShape.scalar
        let source = try value(operand)
        guard input != output else { return source }
        if input.width == 8 {
            return try wideIntegerBits(source, signed: output == .i64, width: outputShape.size?.rawValue ?? 1,
                                       in: outputFormat)
        }
        if let words = inputShape.packedHalfWords(to: outputShape), outputFormat == .hlsl || outputFormat == .glsl {
            return try packedHalfBitcast("(\(source))", input: inputShape, output: outputShape, words: words,
                                         resultType: typeName(target), format: outputFormat)
        }
        return try bitcastValue(source, from: input, to: output, type: typeName(target))
    }

    private func bitcastValue(_ source: String, from input: IRScalar, to output: IRScalar, type: String) throws -> String {
        let operation: String
        switch outputFormat {
        case .wgsl: operation = "bitcast<\(type)>"
        case .msl: operation = "as_type<\(type)>"
        case .hlsl:
            switch (input, output) {
            case (.f16, .i16): operation = "asint16"
            case (.f16, .u16): operation = "asuint16"
            case (.i16, .f16), (.u16, .f16): operation = "asfloat16"
            case (.i16, .u16), (.u16, .i16), (.i64, .u64), (.u64, .i64): return numericCast(type, source)
            default: operation = output == .i32 ? "asint" : output == .u32 ? "asuint" : "asfloat"
            }
        case .glsl:
            switch (input, output) {
            case (.f16, .i16): operation = "float16BitsToInt16"
            case (.f16, .u16): operation = "float16BitsToUint16"
            case (.i16, .f16): operation = "int16BitsToFloat16"
            case (.u16, .f16): operation = "uint16BitsToFloat16"
            case (.f32, .i32): operation = "floatBitsToInt"
            case (.f32, .u32): operation = "floatBitsToUint"
            case (.i32, .f32): operation = "intBitsToFloat"
            case (.u32, .f32): operation = "uintBitsToFloat"
            default: operation = type
            }
        default: throw unsupported("bitcast target")
        }
        return "\(operation)(\(source))"
    }

    private func floatingRemainder(_ lhs: String, _ rhs: String) -> String {
        switch outputFormat {
        case .wgsl: return "(\(lhs) % \(rhs))"
        case .glsl: return "((\(lhs)) - (\(rhs)) * trunc((\(lhs)) / (\(rhs))))"
        default: return "fmod(\(lhs), \(rhs))"
        }
    }

    private func scalarZero(_ type: IRScalar) -> String {
        if type == .bool { return "false" }
        if type == .f64 { return doubleLiteral(0) }
        if type == .f32 { return floatLiteral(0) }
        if type == .f16 { return halfLiteral(0) }
        if type == .i16 || type == .u16 {
            return numericCast(integer16Type(signed: type == .i16, in: outputFormat), "0")
        }
        if type == .i32 { return outputFormat == .wgsl ? "0i" : "0" }
        if type == .i64 { return signed64Literal(0, in: outputFormat) }
        if type == .u64 { return unsigned64Literal(0, in: outputFormat) }
        return "0u"
    }

    private mutating func zeroValue(_ type: IRHandle<IRType>) throws -> String {
        try budget.consume(.work)
        try budget.enterDepth()
        defer { try? budget.leaveDepth() }
        switch try module.types.value(for: type).kind {
        case .scalar: return try scalarZero(scalar(type))
        case .cooperativeMatrix(_, _, let scalar, _) where outputFormat == .hlsl && (scalar == .f32 || scalar == .f16):
            return "(saga_coop_tile)0"
        case .cooperativeMatrix(_, _, _, _) where outputFormat == .wgsl:
            return "\(try typeName(type))()"
        case .cooperativeMatrix(let columns, let rows, let scalar, _) where outputFormat == .msl:
            let zero = "make_filled_simdgroup_matrix<\(scalar == .f16 ? "half" : "float"), 8, 8>(0)"
            if columns == .eight && rows == .eight { return zero }
            let count = Int(columns.rawValue / 8) * Int(rows.rawValue / 8)
            return "\(try typeName(type)){\(Array(repeating: zero, count: count).joined(separator: ", "))}"
        case .cooperativeMatrix(_, _, let scalar, _) where outputFormat == .glsl:
            let zero = scalar == .f16 ? "float16_t(0.0)" : "0.0"
            return "\(try typeName(type))(\(zero))"
        case .vector(let size, let scalar) where scalar.isWGSLConversionScalar:
            return try construct(type, components: Array(repeating: scalarZero(scalar), count: Int(size.rawValue)))
        case .matrix(let columns, let rows, let scalar) where [.f16, .f32, .f64].contains(scalar):
            let zero = scalarZero(scalar)
            let columnType = try shaderVectorType(rows.rawValue, scalar: scalar, in: outputFormat)
            let column = "\(columnType)(\(Array(repeating: zero, count: Int(rows.rawValue)).joined(separator: ", ")))"
            return try construct(type, components: Array(repeating: column, count: Int(columns.rawValue)))
        case .array(let element, .constant, _) where structures[type] != nil:
            switch outputFormat {
            case .wgsl, .msl: return try construct(type, components: [])
            case .hlsl: return try numericCast(typeName(type), "0")
            case .glsl: return try "saga_zero_array_\(type.index)(\(zeroValue(element)))"
            default: throw unsupported("array zero value target")
            }
        case .structure(let members, _) where structures[type] != nil:
            let components = try members.map { try zeroValue($0.type) }
            return try construct(type, components: components)
        default: throw unsupported("zero value for unsupported type")
        }
    }

    private func floatLiteral(_ value: Float) -> String {
        if outputFormat == .wgsl {
            if value.bitPattern == 0x80000000 { return "bitcast<f32>(0x80000000u)" }
            if value.magnitude == Float.greatestFiniteMagnitude {
                let sign = value.sign == .minus ? "-" : ""
                return "\(sign)0x1.fffffep+127f"
            }
        }
        return outputFormat == .glsl ? String(value) : "\(value)f"
    }

    private func halfLiteral(_ value: Float) -> String {
        if outputFormat == .wgsl { return "\(value)h" }
        return numericCast(halfType(1, in: outputFormat), floatLiteral(value))
    }

    private func doubleLiteral(_ value: Double) -> String {
        "\(value)\(outputFormat == .glsl ? "lf" : "L")"
    }

    private func numericCast(_ type: String, _ value: String) -> String {
        // Parenthesize C-style casts so a following swizzle applies to the converted vector.
        outputFormat == .hlsl ? "((\(type))(\(value)))" : "\(type)(\(value))"
    }

    private mutating func writeFloat64ToIntegerCast(
        _ operand: IRHandle<IRExpression>, input: IRHandle<IRType>, target: IRScalar, size: IRVectorSize?,
        result: IRHandle<IRExpression>, resultType: IRHandle<IRType>, to lines: inout [String]
    ) throws {
        let name = "saga_float64_cast_\(result.index)"
        let source = "\(name)_operand"
        try append("\(try typeName(input)) \(source) = \(try value(operand));", to: &lines)
        let bounds = try Float64IntegerBounds(target: target)
        let lower = doubleLiteral(bounds.lower)
        let upper = doubleLiteral(bounds.upper)
        let zero = doubleLiteral(0)
        let scalarType = try shaderVectorType(1, scalar: target, in: outputFormat)
        let sourceLanes = size.map { "xyzw".prefix(Int($0.rawValue)).map { "\(source).\($0)" } } ?? [source]
        var lanes: [String] = []
        for (index, lane) in sourceLanes.enumerated() {
            let words = "\(name)_words_\(index)"
            let magnitude = "\(name)_magnitude_\(index)"
            let finite = "\(name)_finite_\(index)"
            if outputFormat == .hlsl {
                try append("uint2 \(words);", to: &lines)
                try append("asuint(\(lane), \(words).x, \(words).y);", to: &lines)
            } else {
                try append("uvec2 \(words) = unpackDouble2x32(\(lane));", to: &lines)
            }
            // Keep NaN and infinities away from comparisons on targets with finite-math assumptions.
            try append("uint \(magnitude) = \(words).y & 0x7fffffffu;", to: &lines)
            try append("double \(finite) = \(magnitude) < 0x7ff00000u ? \(lane) : \(zero);", to: &lines)
            let infinity = "\(magnitude) == 0x7ff00000u && \(words).x == 0u"
            let saturation = "\(words).y > 0x7fffffffu ? \(lower) : \(upper)"
            let bounded = "\(finite) < \(lower) ? \(lower) : (\(finite) > \(upper) ? \(upper) : \(finite))"
            lanes.append(numericCast(scalarType, "\(infinity) ? (\(saturation)) : (\(bounded))"))
        }
        let type = try typeName(resultType)
        let converted = size == nil ? lanes[0] : "\(type)(\(lanes.joined(separator: ", ")))"
        try append("\(type) \(name) = \(converted);", to: &lines)
    }

    private func floatToIntegerCast(
        _ value: String, input: IRScalar, target: IRScalar, width: UInt8?, resultType: String
    ) -> String {
        // Clamp to integers exactly representable by the source format. The ordinary
        // native cast is only defined once the value lies within the destination range.
        let unsigned = target.kind == .uint
        let lower: String
        let upper: String
        if target.width == 2 {
            lower = unsigned ? "0.0" : "-32768.0"
            upper = unsigned ? "65535.0" : "32767.0"
        } else if input == .f16 {
            lower = unsigned ? "0.0" : "-65504.0"
            upper = "65504.0"
        } else if target.width == 8 {
            lower = unsigned ? "0.0" : "-9223372036854775808.0"
            upper = unsigned ? "18446742974197923840.0" : "9223371487098961920.0"
        } else {
            lower = unsigned ? "0.0" : "-2147483648.0"
            upper = unsigned ? "4294967040.0" : "2147483520.0"
        }
        let suffix = outputFormat == .glsl ? "" : "f"
        let floatingType = floatType(width ?? 1, in: outputFormat)
        let source = input == .f16 ? numericCast(floatingType, value) : value
        let bounds = [lower, upper].map { bound in
            let literal = bound + suffix
            return width == nil ? literal : numericCast(floatingType, literal)
        }
        return numericCast(resultType, "clamp(\(source), \(bounds[0]), \(bounds[1]))")
    }

    private func literalValue(_ literal: IRLiteral) throws -> String {
        switch literal {
        case .u16(let number):
            return numericCast(integer16Type(signed: false, in: outputFormat), "\(number)")
        case .i16(let number):
            return numericCast(integer16Type(signed: true, in: outputFormat), "\(number)")
        case .u32(let number): return "\(number)u"
        case .i32(let number):
            if number == .min { return integerBits("2147483648u", signed: true, in: outputFormat) }
            let value = outputFormat == .wgsl ? "\(number)i" : String(number)
            return number < 0 ? "(\(value))" : value
        case .f32(let number) where number.isFinite: return floatLiteral(number)
        case .f64(let number) where number.isFinite && (outputFormat == .glsl || outputFormat == .hlsl):
            return doubleLiteral(number)
        case .f16Bits(let bits) where bits & 0x7c00 != 0x7c00: return halfLiteral(Binary16.decode(bits))
        case .bool(let flag): return flag ? "true" : "false"
        case .u64(let value): return unsigned64Literal(value, in: outputFormat)
        case .i64(let value): return signed64Literal(value, in: outputFormat)
        default: throw unsupported("literal type")
        }
    }

    private mutating func constantValue(_ handle: IRHandle<IRExpression>) throws -> String {
        try budget.consume(.work)
        try budget.enterDepth()
        defer { try? budget.leaveDepth() }
        let expression = try module.globalExpressions.value(for: handle)
        switch expression.kind {
        case .literal(let literal):
            return try literalValue(literal)
        case .zeroValue(let type):
            return try zeroValue(type)
        case .compose(let type, let components):
            try budget.consume(.sourceBytes, amount: components.count + 1)
            let values = try components.map { try constantValue($0) }
            return try construct(type, components: values)
        default:
            throw unsupported("constant expression")
        }
    }

    private mutating func writeLdexp(
        _ mantissa: IRHandle<IRExpression>, exponent: IRHandle<IRExpression>,
        result: IRHandle<IRExpression>, type: IRHandle<IRType>, to lines: inout [String]
    ) throws {
        let name = "saga_math_\(result.index)"
        let x = try value(mantissa)
        let y = try value(exponent)
        let resultType = try typeName(type)
        let binding = outputFormat == .wgsl ? "let \(name)" : "\(resultType) \(name)"
        guard let shape = try IRBitcastShape(module.types.value(for: type).kind), [.f16, .f32, .f64].contains(shape.scalar) else {
            throw ShaderFailure.invalidHandle
        }
        let widen = shape.scalar == .f16 && outputFormat != .wgsl
        let nativeType = widen ? floatType(shape.size?.rawValue ?? 1, in: outputFormat) : resultType
        let input = widen ? numericCast(nativeType, x) : x
        let call: String
        if outputFormat == .hlsl {
            guard shape.scalar != .f64 else { throw unsupported("double-precision ldexp in HLSL") }
            let exponentType = try typeName(function.expressions.value(for: exponent).type)
            let power = "\(name)_exponent"
            try append("\(exponentType) \(power) = \(y);", to: &lines)
            // HLSL uses a float exponent. Keep its intermediate power of two finite at 128.
            let adjusted = "\(input) * (1.0f + \(nativeType)(\(power) > 127))"
            call = "ldexp(\(adjusted), \(nativeType)(min(\(power), 127)))"
        } else {
            let builtin = outputFormat == .msl ? "metal::ldexp" : "ldexp"
            call = "\(builtin)(\(input), \(y))"
        }
        let resultValue = widen ? numericCast(resultType, call) : call
        try append("\(binding) = \(resultValue);", to: &lines)
    }

    private mutating func writeDecomposition(
        _ operation: IRMathFunction, operand: IRHandle<IRExpression>, type: IRHandle<IRType>,
        result: IRHandle<IRExpression>, to lines: inout [String]
    ) throws {
        guard case .structure(let members, _) = try module.types.value(for: type).kind else {
            throw ShaderFailure.invalidHandle
        }
        let name = "saga_math_\(result.index)"
        let input = try value(operand)
        let call = operation.rawValue
        let other = "\(name)_other"
        if outputFormat == .wgsl {
            let native = "\(name)_native"
            try append("let \(native) = \(call)(\(input));", to: &lines)
            let constructed = try construct(type, components: ["\(native).fract", "\(native).\(operation.decompositionSecondMember)"])
            try append("let \(name) = \(constructed);", to: &lines)
            return
        }
        let fractionType = try typeName(members[0].type)
        let otherType = try typeName(members[1].type)
        guard let shape = try module.types.value(for: members[0].type).kind.concreteFloatShape else {
            throw ShaderFailure.invalidHandle
        }
        let widen = shape.scalar == .f16
        let nativeType = widen ? floatType(shape.size?.rawValue ?? 1, in: outputFormat) : fractionType
        let argument = widen ? numericCast(nativeType, input) : input
        let function = outputFormat == .msl ? "metal::\(call)" : call
        let nativeOther = operation == .modf ? nativeType : otherType
        let fraction = "\(name)_fract"
        if outputFormat == .hlsl && operation == .frexp {
            try writeHLSLFrexp(argument, size: shape.size, name: name, fraction: fraction,
                               fractionType: nativeType, exponent: other, exponentType: otherType, to: &lines)
        } else {
            try append("\(nativeOther) \(other);", to: &lines)
            try append("\(nativeType) \(fraction) = \(function)(\(argument), \(other));", to: &lines)
        }
        let narrowed = widen ? numericCast(fractionType, fraction) : fraction
        let second = nativeOther == otherType ? other : numericCast(otherType, other)
        let constructed = try construct(type, components: [narrowed, second])
        try append("\(try typeName(type)) \(name) = \(constructed);", to: &lines)
    }

    private mutating func writeHLSLFrexp(
        _ argument: String, size: IRVectorSize?, name: String, fraction: String, fractionType: String,
        exponent: String, exponentType: String, to lines: inout [String]
    ) throws {
        let bits = "\(name)_bits"
        let bitsType = try shaderVectorType(size?.rawValue ?? 1, scalar: .u32, in: .hlsl)
        try append("\(bitsType) \(bits) = asuint(\(argument));", to: &lines)
        try append("\(fractionType) \(fraction);", to: &lines)
        try append("\(exponentType) \(exponent);", to: &lines)
        // Normalize subnormal significands with integer shifts. Keep the sign bit,
        // including for signed zero, and avoid the compiler-specific frexp overloads.
        for lane in 0..<Int(size?.rawValue ?? 1) {
            let component = size == nil ? "" : ".\(["x", "y", "z", "w"][lane])"
            let word = bits + component
            let prefix = "\(name)_\(lane)"
            let magnitude = "\(prefix)_magnitude"
            let mantissa = "\(prefix)_mantissa"
            let biased = "\(prefix)_biased"
            let leading = "\(prefix)_leading"
            let normalized = "\(prefix)_normalized"
            let special = "\(prefix)_special"
            try append("uint \(magnitude) = \(word) & 0x7fffffffu;", to: &lines)
            try append("uint \(mantissa) = \(magnitude) & 0x007fffffu;", to: &lines)
            try append("uint \(biased) = \(magnitude) >> 23u;", to: &lines)
            try append("uint \(leading) = uint(firstbithigh(max(\(mantissa), 1u)));", to: &lines)
            try append("uint \(normalized) = \(biased) == 0u ? (\(mantissa) << (23u - \(leading))) & 0x007fffffu : \(mantissa);", to: &lines)
            try append("bool \(special) = \(magnitude) == 0u || \(biased) == 255u;", to: &lines)
            try append("\(fraction)\(component) = asfloat(\(special) ? \(word) : ((\(word) & 0x80000000u) | 0x3f000000u | \(normalized)));", to: &lines)
            try append("\(exponent)\(component) = \(special) ? 0 : (\(biased) == 0u ? int(\(leading)) - 148 : int(\(biased)) - 126);", to: &lines)
        }
    }

    private func declaration(_ type: String, named name: String) -> String {
        outputFormat == .wgsl ? "var \(name): \(type)" : "\(type) \(name)"
    }

    private func typeName(_ type: IRHandle<IRType>) throws -> String {
        try shaderValueType(type, module: module, format: outputFormat, structures: structures)
    }

    private mutating func construct(_ type: IRHandle<IRType>, components: [String]) throws -> String {
        let constructor = try structures[type]?.constructor ?? typeName(type)
        let aggregate = structures[type] != nil && outputFormat == .msl
        try budget.consume(.sourceBytes, amount: constructor.utf8.count + components.count * 2 + 2)
        let values = components.joined(separator: ", ")
        return aggregate ? "\(constructor){\(values)}" : "\(constructor)(\(values))"
    }

    private func member(_ index: UInt32, of type: IRHandle<IRType>) throws -> String {
        if case .matrix(let columns, _, let scalar) = try module.types.value(for: type).kind,
           [.f16, .f32, .f64].contains(scalar), index < UInt32(columns.rawValue) {
            return "[\(index)u]"
        }
        if case .vector(let width, let scalar) = try module.types.value(for: type).kind,
           scalar.isWGSLConversionScalar, index < UInt32(width.rawValue) {
            return [".x", ".y", ".z", ".w"][Int(index)]
        }
        if case .array(_, .constant(let count), _) = try module.types.value(for: type).kind, index < count {
            return outputFormat.valueArrayIndex("\(index)u")
        }
        if let structure = structures[type] { return ".\(structure.memberName(Int(index)))" }
        throw unsupported("access outside a supported vector or render structure")
    }

    private mutating func value(
        _ handle: IRHandle<IRExpression>, parenthesized: Bool = true
    ) throws -> String {
        try budget.consume(.work)
        try budget.enterDepth()
        defer { try? budget.leaveDepth() }
        if continuingCaptures[handle] != nil {
            continuingCaptures[handle] = true
            return "saga_loop_value_\(handle.index)"
        }
        let expression = try function.expressions.value(for: handle)
        switch expression.kind {
        case .cooperativeLoad where temporaries.contains(handle),
             .cooperativeMultiplyAdd where temporaries.contains(handle):
            return "saga_coop_\(handle.index)"
        case .literal(let literal): return try literalValue(literal)
        case .constant(let handle):
            let initializer = try module.constants.value(for: handle).initializer
            return try constantValue(initializer)
        case .zeroValue(let type): return try zeroValue(type)
        case .arrayLength where temporaries.contains(handle): return "saga_array_length_\(handle.index)"
        case .arrayLength(let pointer):
            if let nested = try nestedAccess(pointer), let length = nested.access.buffer.runtimeLength {
                return nested.bounds.map { boundedBufferValue(length, when: $0) } ?? length
            }
            guard let bound = try storageLocation(function.expressions.value(for: pointer).kind)?.bound
            else { throw unsupported("length of an array outside a supported storage binding") }
            return bound
        case .imageSample where temporaries.contains(handle), .imageLoad where temporaries.contains(handle),
             .imageQuery where temporaries.contains(handle):
            return "saga_image_\(handle.index)"
        case .runtimeValue where temporaries.contains(handle): return "saga_value_\(handle.index)"
        case .load where temporaries.contains(handle): return "saga_load_\(handle.index)"
        case .access where temporaries.contains(handle): return "saga_index_\(handle.index)"
        case .select where temporaries.contains(handle): return "saga_select_\(handle.index)"
        case .math where temporaries.contains(handle): return "saga_math_\(handle.index)"
        case .cast where temporaries.contains(handle): return "saga_float64_cast_\(handle.index)"
        case .bitcast where temporaries.contains(handle): return "saga_bitcast_\(handle.index)"
        case .binary where temporaries.contains(handle): return "saga_remainder_\(handle.index)"
        case .derivative where temporaries.contains(handle): return "saga_derivative_\(handle.index)"
        case .workgroupUniformLoadResult where temporaries.contains(handle): return "saga_uniform_\(handle.index)"
        case .rayQueryProceedResult where temporaries.contains(handle): return "saga_ray_proceed_\(handle.index)"
        case .rayQueryGetIntersection where temporaries.contains(handle): return "saga_ray_intersection_\(handle.index)"
        case .subgroupOperationResult, .subgroupBallotResult:
            guard temporaries.contains(handle) else { throw unsupported("Unavailable subgroup result") }
            return "saga_subgroup_\(handle.index)"
        case .atomicResult where temporaries.contains(handle): return "saga_atomic_\(handle.index)"
        case .callResult where temporaries.contains(handle): return "saga_call_\(handle.index)"
        case .globalVariable(let global) where outputFormat == .msl || outputFormat == .hlsl || outputFormat == .glsl:
            guard storage.indices.contains(global.index),
                  case .accelerationStructure(false) = try module.types.value(for: expression.type).kind
            else { throw unsupported("global value") }
            return storage[global.index].reference
        case .functionArgument(let argument):
            guard let index = Int(exactly: argument), arguments.indices.contains(index) else {
                throw ShaderFailure.invalidHandle
            }
            if !arguments[index].isEmpty { return arguments[index] }
            guard let fields = argumentFields[index] else { throw ShaderFailure.invalidHandle }
            return try construct(expression.type, components: fields)
        case .compose(let type, let components):
            // Charge before expansion so an expression shared by several constructors stays bounded.
            try budget.consume(.sourceBytes, amount: components.count + 1)
            let values = try components.map { try value($0) }
            return try construct(type, components: values)
        case .splat(let size, let operand):
            let lane = try value(operand)
            if outputFormat == .hlsl {
                return "(\(lane)).\(String(repeating: "x", count: Int(size.rawValue)))"
            }
            return try construct(expression.type, components: [lane])
        case .swizzle(let size, let vector, let pattern):
            let suffix = [pattern.x, pattern.y, pattern.z, pattern.w]
                .prefix(Int(size.rawValue)).map(\.rawValue).joined()
            return try value(vector) + "." + suffix
        case .accessIndex(let base, let index):
            let source = try function.expressions.value(for: base)
            if case .functionArgument(let argument) = source.kind, let slot = Int(exactly: argument),
               let fields = argumentFields[slot], let field = Int(exactly: index), fields.indices.contains(field) {
                return fields[field]
            }
            let field = try member(index, of: source.type)
            return try value(base) + field
        case .relational(let reduction, let operand) where reduction.isBooleanReduction:
            let operandType = try function.expressions.value(for: operand).type
            let argument = try value(operand)
            if case .scalar(.bool) = try module.types.value(for: operandType).kind { return argument }
            return "\(reduction.rawValue)(\(argument))"
        case .bitcast(let operand, let target): return try bitcast(operand, to: target)
        case .cast(let operand, let target):
            return try castValue(operand, target: target, resultType: expression.type)
        case .unary(let op, let argument):
            return try unaryValue(op, argument: argument, resultType: expression.type, parenthesized: parenthesized)
        case .binary(let op, let left, let right):
            return try binaryValue(op, left: left, right: right, resultType: expression.type, parenthesized: parenthesized)
        default: throw unsupported("expression")
        }
    }

    private mutating func castValue(
        _ operand: IRHandle<IRExpression>, target: IRScalar, resultType: IRHandle<IRType>
    ) throws -> String {
        let operandType = try function.expressions.value(for: operand).type
        let input: IRScalar
        let width: UInt8?
        switch try module.types.value(for: operandType).kind {
        case .vector(let size, let scalar):
            input = scalar
            width = size.rawValue
        case .scalar(let scalar):
            input = scalar
            width = nil
        case .matrix(_, _, let scalar) where [.f16, .f32, .f64].contains(scalar) && [.f16, .f32, .f64].contains(target):
            return try numericCast(typeName(resultType), value(operand))
        default: throw unsupported("cast operand type")
        }
        if input == .bool || target == .bool {
            let source = try value(operand)
            let type = try typeName(resultType)
            return booleanCast(type, source)
        }
        if input == .f64 || target == .f64 {
            guard target.kind == .float else { throw unsupported("f64 to integer conversion") }
            return try numericCast(typeName(resultType), value(operand))
        }
        if input == .i16 || input == .u16 || target == .i16 || target == .u16 {
            let source = try value(operand)
            let type = try typeName(resultType)
            if outputFormat == .wgsl { return numericCast(type, source) }
            if target == .i16 || target == .u16 {
                if input.kind == .float {
                    return floatToIntegerCast(source, input: input, target: target, width: width, resultType: type)
                }
                return try integer16Bits(source, signed: target == .i16, width: width ?? 1, in: outputFormat)
            }
            return numericCast(type, source)
        }
        if input.width == 8 || target.width == 8 {
            let source = try value(operand)
            let type = try typeName(resultType)
            if outputFormat == .wgsl { return numericCast(type, source) }
            if input.kind == .float {
                return floatToIntegerCast(source, input: input, target: target, width: width, resultType: type)
            }
            if input.width == 8, target.width == 8 {
                return try wideIntegerBits(source, signed: target == .i64, width: width ?? 1, in: outputFormat)
            }
            return numericCast(type, source)
        }
        if [.f32, .f16].contains(input), [.i32, .u32].contains(target) {
            // WGSL conversions already clamp to the destination range and round toward zero.
            if outputFormat == .wgsl { return try numericCast(typeName(resultType), value(operand)) }
            return try floatToIntegerCast(value(operand), input: input, target: target,
                                          width: width, resultType: typeName(resultType))
        }
        if target == .f16, [.f32, .i32, .u32].contains(input) {
            return try numericCast(typeName(resultType), value(operand))
        }
        if input == .f16, target == .f32 {
            return try numericCast(typeName(resultType), value(operand))
        }
        guard [.i32, .u32].contains(input), [.i32, .u32, .f32].contains(target) else {
            throw unsupported("cast outside the supported numeric profile")
        }
        let source = try value(operand)
        if target == .f32 { return "\(try typeName(resultType))(\(source))" }
        return input == target ? source : integerBits(source, signed: target == .i32, width: width, in: outputFormat)
    }

    private func booleanCast(_ type: String, _ value: String) -> String {
        outputFormat == .msl ? "static_cast<\(type)>(\(value))" : numericCast(type, value)
    }

    private func eagerBooleanBinary(_ symbol: String, left: String, right: String, width: UInt8) throws -> String {
        if outputFormat == .wgsl { return "(\(left) \(symbol) \(right))" }
        // Use integer bitwise operators so native scalar and vector targets
        // preserve eager Boolean evaluation.
        let word = try shaderVectorType(width, scalar: .u32, in: outputFormat)
        let result = try shaderVectorType(width, scalar: .bool, in: outputFormat)
        return booleanCast(result, "\(booleanCast(word, left)) \(symbol) \(booleanCast(word, right))")
    }

    private mutating func unaryValue(
        _ op: IRUnaryOperator, argument: IRHandle<IRExpression>, resultType: IRHandle<IRType>, parenthesized: Bool
    ) throws -> String {
        if op == .bitwiseNot, outputFormat != .wgsl,
           let shape = try IRBitcastShape(module.types.value(for: resultType).kind),
           shape.scalar == .i16 || shape.scalar == .u16 {
            let width = shape.size?.rawValue ?? 1
            let wide = try shaderVectorType(width, scalar: .u32, in: outputFormat)
            return try integer16Bits("~\(wide)(\(value(argument)))", signed: shape.scalar == .i16,
                                     width: width, in: outputFormat)
        }
        if case .vector(let size, let lane) = try module.types.value(for: resultType).kind,
           [.i16, .u16, .i32, .u32, .i64, .u64].contains(lane) {
            let operand = try value(argument)
            if op == .bitwiseNot { return "(~\(operand))" }
            guard op == .negate && lane.kind == .sint else { throw unsupported("integer vector unary operation") }
            let helper = IntegerVectorHelper(operation: .negate, scalar: lane, width: size.rawValue)
            integerVectorHelpers.insert(helper)
            return "\(helper.name)(\(operand))"
        }
        if op == .logicalNot, case .vector(_, .bool) = try module.types.value(for: resultType).kind {
            try budget.consume(.sourceBytes, amount: 5)
            let operand = try value(argument)
            return outputFormat == .glsl ? "not(\(operand))" : "(!\(operand))"
        }
        if op == .negate, let shape = try module.types.value(for: resultType).kind.concreteFloatShape,
           shape.size != nil {
            return "(-\(try value(argument)))"
        }
        let type = try scalar(resultType)
        if op == .negate && type.kind == .sint {
            return try integerNegation(value(argument), scalar: type, in: outputFormat)
        }
        let symbol: String
        switch op {
        case .logicalNot where type == .bool: symbol = "!"
        case .bitwiseNot where type.kind == .uint || type.kind == .sint: symbol = "~"
        case .negate where type.kind == .float: symbol = "-"
        default: throw unsupported("unary operation \(op.rawValue)")
        }
        try budget.consume(.sourceBytes, amount: 3)
        let unary = "\(symbol)\(try value(argument))"
        return parenthesized ? "(\(unary))" : unary
    }

    private mutating func binaryValue(
        _ op: IRBinaryOperator, left: IRHandle<IRExpression>, right: IRHandle<IRExpression>,
        resultType: IRHandle<IRType>, parenthesized: Bool
    ) throws -> String {
        // Validate and charge before expanding either child, as in the original traversal.
        try prepareBinaryValue(op, left: left, right: right)
        let lhs = try value(left)
        let rhs = try value(right)
        return try formatBinaryValue(op, left: left, right: right, lhs: lhs, rhs: rhs,
                                     resultType: resultType, parenthesized: parenthesized)
    }

    private mutating func prepareBinaryValue(
        _ op: IRBinaryOperator, left: IRHandle<IRExpression>, right: IRHandle<IRExpression>
    ) throws {
        let operandType = try function.expressions.value(for: left).type
        let lhsKind = try module.types.value(for: operandType).kind
        let rhsType = try function.expressions.value(for: right).type
        let rhsKind = try module.types.value(for: rhsType).kind
        if lhsKind.isMatrix || rhsKind.isMatrix {
            try budget.consume(.sourceBytes, amount: 8)
        } else if case .vector(let size, let lane) = lhsKind,
                  [.i16, .u16, .i32, .u32].contains(lane) || IntegerShaderHelper(operation: op, scalar: lane) != nil {
            let helper = IntegerVectorHelper(operation: .binary(op), scalar: lane, width: size.rawValue)
            try budget.consume(.sourceBytes, amount: helper.name.utf8.count + 4)
        } else if case .vector(_, let lane) = lhsKind, lane == .i64 || lane == .u64 {
            try budget.consume(.sourceBytes, amount: 8)
        } else if let shape = IRBitcastShape(lhsKind), [.f16, .f32, .f64].contains(shape.scalar), lhsKind == rhsKind {
            try budget.consume(.sourceBytes, amount: 8)
        } else if case .vector(_, .bool) = lhsKind {
            try budget.consume(.sourceBytes, amount: 64)
        } else {
            let type = try scalar(operandType)
            let shift = op == .shiftLeft || op == .shiftRight
            let rightType = try scalar(rhsType)
            guard rightType == (shift ? .u32 : type) else {
                throw unsupported("binary operand types")
            }
            try budget.consume(.sourceBytes, amount: 8)
        }
    }

    private mutating func formatBinaryValue(
        _ op: IRBinaryOperator, left: IRHandle<IRExpression>, right: IRHandle<IRExpression>,
        lhs: String, rhs: String, resultType: IRHandle<IRType>, parenthesized: Bool
    ) throws -> String {
        let operandType = try function.expressions.value(for: left).type
        let lhsKind = try module.types.value(for: operandType).kind
        let rhsType = try function.expressions.value(for: right).type
        let rhsKind = try module.types.value(for: rhsType).kind
        if lhsKind.isMatrix || rhsKind.isMatrix {
            // HLSL rows represent WGSL columns. The transpose of a product reverses the operands.
            if outputFormat == .hlsl, op == .multiply, !lhsKind.isScalar, !rhsKind.isScalar {
                return "mul(\(rhs), \(lhs))"
            }
            let symbol = try shaderBinarySymbol(op, type: .f32)
            let binary = "\(lhs) \(symbol) \(rhs)"
            return parenthesized ? "(\(binary))" : binary
        }
        if case .vector(let size, let lane) = lhsKind,
           [.i16, .u16, .i32, .u32].contains(lane) || IntegerShaderHelper(operation: op, scalar: lane) != nil {
            let helper = IntegerVectorHelper(operation: .binary(op), scalar: lane, width: size.rawValue)
            integerVectorHelpers.insert(helper)
            return "\(helper.name)(\(lhs), \(rhs))"
        }
        if case .vector(let size, let lane) = lhsKind, lane == .i64 || lane == .u64 {
            return try wideIntegerBinary(op, scalar: lane, width: size.rawValue, left: lhs, right: rhs, in: outputFormat)
        }
        if let shape = IRBitcastShape(lhsKind), [.f16, .f32, .f64].contains(shape.scalar), lhsKind == rhsKind {
            if op == .modulo {
                guard shape.scalar != .f64 else { throw unsupported("f64 remainder in a text backend") }
                let remainder = floatingRemainder(lhs, rhs)
                if outputFormat == .hlsl && shape.scalar == .f16 {
                    return try numericCast(typeName(resultType), remainder)
                }
                return remainder
            }
            if outputFormat == .glsl, shape.size != nil {
                let builtin: String? = switch op {
                case .equal: "equal"
                case .notEqual: "notEqual"
                case .less: "lessThan"
                case .lessEqual: "lessThanEqual"
                case .greater: "greaterThan"
                case .greaterEqual: "greaterThanEqual"
                default: nil
                }
                if let builtin { return "\(builtin)(\(lhs), \(rhs))" }
            }
            let symbol = try shaderBinarySymbol(op, type: shape.scalar)
            let binary = "\(lhs) \(symbol) \(rhs)"
            return parenthesized ? "(\(binary))" : binary
        }
        if case .vector(let width, .bool) = lhsKind {
            let symbol = try shaderBinarySymbol(op, type: .bool)
            if op == .and || op == .inclusiveOr {
                return try eagerBooleanBinary(symbol, left: lhs, right: rhs, width: width.rawValue)
            }
            if outputFormat == .glsl {
                let builtin = op == .equal ? "equal" : "notEqual"
                return "\(builtin)(\(lhs), \(rhs))"
            }
            return "(\(lhs) \(symbol) \(rhs))"
        }
        let type = try scalar(operandType)
        let shift = op == .shiftLeft || op == .shiftRight
        let rightType = try scalar(rhsType)
        guard rightType == (shift ? .u32 : type) else {
            throw unsupported("binary operand types")
        }
        if type == .bool, op == .and || op == .inclusiveOr {
            try budget.consume(.sourceBytes, amount: 64)
            return try eagerBooleanBinary(shaderBinarySymbol(op, type: type), left: lhs, right: rhs, width: 1)
        }
        if let lowered = try integerBinary(op, scalar: type, left: lhs, right: rhs,
                                           in: outputFormat, helpers: &integerHelpers) {
            return lowered
        }
        if type == .i64 || type == .u64 {
            return try wideIntegerBinary(op, scalar: type, left: lhs, right: rhs, in: outputFormat)
        }
        let symbol = try shaderBinarySymbol(op, type: type)
        let binary = "\(lhs) \(symbol) \(rhs)"
        return parenthesized ? "(\(binary))" : binary
    }
}

private extension IRTypeKind {
    var componentCount: UInt8? {
        switch self {
        case .vector(let size, _): size.rawValue
        case .matrix(let size, _, let scalar) where [.f16, .f32, .f64].contains(scalar): size.rawValue
        default: nil
        }
    }

    var isMatrix: Bool {
        if case .matrix = self { return true }
        return false
    }

    var isScalar: Bool {
        if case .scalar = self { return true }
        return false
    }
}

func shaderBinarySymbol(_ op: IRBinaryOperator, type: IRScalar) throws -> String {
    let lane: IRScalar = switch type {
    case .f16, .f64: .f32
    case .i16, .i64: .i32
    case .u16, .u64: .u32
    default: type
    }
    return switch (op, lane) {
    case (.equal, _): "=="
    case (.notEqual, _): "!="
    case (.logicalAnd, .bool): "&&"
    case (.logicalOr, .bool): "||"
    case (.add, .u32), (.add, .i32), (.add, .f32): "+"
    case (.subtract, .u32), (.subtract, .i32), (.subtract, .f32): "-"
    case (.multiply, .u32), (.multiply, .i32), (.multiply, .f32): "*"
    case (.divide, .f32): "/"
    case (.less, .u32), (.less, .i32), (.less, .f32): "<"
    case (.lessEqual, .u32), (.lessEqual, .i32), (.lessEqual, .f32): "<="
    case (.greater, .u32), (.greater, .i32), (.greater, .f32): ">"
    case (.greaterEqual, .u32), (.greaterEqual, .i32), (.greaterEqual, .f32): ">="
    case (.and, .u32), (.and, .i32), (.and, .bool): "&"
    case (.inclusiveOr, .u32), (.inclusiveOr, .i32), (.inclusiveOr, .bool): "|"
    case (.exclusiveOr, .u32), (.exclusiveOr, .i32): "^"
    default: throw unsupported("binary operation \(op.rawValue)")
    }
}

private func unsupported(_ reason: String) -> ShaderFailure {
    .unsupported("Shader body code generation: \(reason)")
}
