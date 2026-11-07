import SAGAShaderIR

struct ShaderStorage {
    enum WordAddressing: Equatable {
        case structured
        case byteAddressed
        case uniformRegisters
    }

    struct WordLayout {
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

    init(reference: String, bound: String? = nil, writable: Bool,
         words: WordLayout? = nil, members: [ShaderStorage] = []) {
        self.reference = reference
        self.bound = bound
        self.writable = writable
        self.words = words
        self.members = members
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
    case words(buffer: String, first: String, width: UInt8, scalar: IRScalar,
               addressing: ShaderStorage.WordAddressing, columns: UInt8?)

    func component(_ index: String) -> Self {
        switch self {
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
        case .words(let buffer, let first, let width, let scalar, let addressing, let columns):
            if let columns {
                let vectors = try (0..<columns).map { try component("\($0)u").load(in: format) }
                let type = shaderMatrixType(columns, width, in: format)
                return "\(type)(\(vectors.joined(separator: ", ")))"
            }
            let lanes = (0..<width).map {
                let index = wordIndex(first: first, lane: $0)
                let word: String
                switch addressing {
                case .structured: word = "\(buffer)[\(index)]"
                case .byteAddressed: word = "\(buffer).Load(\(byteOffset(index)))"
                case .uniformRegisters: word = "\(buffer)[(\(index)) / 4u][(\(index)) % 4u]"
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
        case .words(let buffer, let first, let width, let scalar, let addressing, let columns):
            guard addressing != .uniformRegisters else { throw ShaderFailure.unsupported("Store to a uniform buffer") }
            if let columns {
                let type = shaderMatrixType(columns, width, in: format)
                var statements = ["const \(type) \(temporary) = \(value);"]
                for column in 0..<columns {
                    let target = component("\(column)u")
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
                let bits: String
                switch scalar {
                case .i32: bits = integerBits(component, signed: false, in: format)
                case .f32: bits = format == .msl ? "as_type<uint>(\(component))" : "asuint(\(component))"
                default: bits = component
                }
                let index = wordIndex(first: first, lane: lane)
                if addressing == .byteAddressed {
                    statements.append("\(buffer).Store(\(byteOffset(index)), \(bits));")
                } else {
                    statements.append("\(buffer)[\(index)] = \(bits);")
                }
            }
            return statements
        }
    }

    var hlslAtomicByteAddress: (buffer: String, offset: String)? {
        guard case .words(let buffer, let first, 1, _, .byteAddressed, nil) = self else { return nil }
        return (buffer, byteOffset(first))
    }

    private func wordIndex(first: String, lane: UInt8) -> String {
        lane == 0 ? first : "\(first) + \(lane)u"
    }

    private func byteOffset(_ word: String) -> String {
        "(\(word)) * 4u"
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
}

enum ShaderResultTarget {
    case direct
    case metalStructure(String)
    case glslPosition
    case glslVertexStructure(String, destinations: [String])
    case fragmentValue(FragmentOutputField, metalType: String?)
    case fragmentStructure(String, fields: [FragmentOutputField])
}

struct ShaderBodyWriter {
    let module: IRModule
    let function: IRFunction
    let arguments: [String]
    let storage: [ShaderStorage]
    let outputFormat: ShaderOutputFormat
    let resultTarget: ShaderResultTarget?
    let structures: [IRHandle<IRType>: ShaderStructure]
    let argumentFields: [Int: [String]]
    let callCaptures: [Int: [String]]
    private(set) var budget: ShaderBudget
    private(set) var integerHelpers: Set<IntegerShaderHelper> = []
    private(set) var integerVectorHelpers: Set<IntegerVectorHelper> = []
    private var temporaries: Set<IRHandle<IRExpression>> = []
    private var temporaryOrder: [IRHandle<IRExpression>] = []
    private var indentation = "    "

    init(module: IRModule, function: IRFunction, arguments: [String], storage: [ShaderStorage],
         outputFormat: ShaderOutputFormat, resultTarget: ShaderResultTarget? = nil,
         structures: [IRHandle<IRType>: ShaderStructure] = [:], argumentFields: [Int: [String]] = [:],
         callCaptures: [Int: [String]] = [:], budget: ShaderBudget = ShaderBudget()) {
        self.module = module
        self.function = function
        self.arguments = arguments
        self.storage = storage
        self.outputFormat = outputFormat
        self.resultTarget = resultTarget
        self.structures = structures
        self.argumentFields = argumentFields
        self.callCaptures = callCaptures
        self.budget = budget
    }

    mutating func write() throws -> [String] {
        var lines: [String] = []
        for (index, local) in function.localVariables.values.enumerated() {
            let type = try typeName(local.type)
            let initial = try local.initializer.map { try value($0) } ?? zeroValue(local.type)
            try append("\(declaration(type, named: localName(index))) = \(initial);", to: &lines)
        }
        try writeBlock(function.body, nested: false, to: &lines)
        return lines
    }

    private mutating func writeBlock(
        _ block: IRHandle<IRBlock>, nested: Bool = true, breakIf: IRHandle<IRExpression>? = nil,
        to lines: inout [String]
    ) throws {
        try budget.enterDepth()
        let mark = temporaryOrder.count
        if nested { indentation += "    " }
        defer {
            for temporary in temporaryOrder[mark...] { temporaries.remove(temporary) }
            temporaryOrder.removeSubrange(mark...)
            if nested { indentation.removeLast(4) }
            try? budget.leaveDepth()
        }
        for handle in try function.blocks.value(for: block).statements {
            try budget.consume(.work)
            switch try function.statements.value(for: handle).kind {
            case .emit(let handles):
                for expression in handles { try emit(expression, to: &lines) }
            case .call(let callee, let args, let result):
                try writeCall(callee, arguments: args, result: result, to: &lines)
            case .returnValue(nil):
                try append("return;", to: &lines)
            case .returnValue(let result?):
                let value = try value(result)
                try writeResult(value, to: &lines)
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
                try writeSwitch(selector, cases: cases, to: &lines)
            case .loop(let body, let continuing, let condition):
                try writeLoop(body, continuing: continuing, breakIf: condition,
                              index: handle.index, to: &lines)
            case .controlBarrier(let flags):
                for line in try shaderControlBarrier(flags, in: outputFormat) { try append(line, to: &lines) }
            case .workgroupUniformLoad(let pointer, let result):
                try writeUniformLoad(pointer: pointer, result: result, to: &lines)
            case .imageStore(let image, let coordinate, let arrayIndex, let value):
                try writeImageStore(image: image, coordinate: coordinate, arrayIndex: arrayIndex, value: value,
                                    named: "saga_texel_store_\(handle.index)", to: &lines)
            case .store(let pointer, let expression):
                let target = try address(pointer)
                guard target.writable else { throw unsupported("store to read-only storage") }
                let rhs = try value(expression)
                let temporary = "saga_store_\(handle.index)"
                if let scalar = try atomicScalar(at: pointer) {
                    try writeAtomic(.store(rhs), target: target, scalar: scalar, result: nil, temporary: temporary, to: &lines)
                } else {
                    let statements = try target.memory.stores(rhs, temporary: temporary, in: outputFormat)
                    try writeStatements(statements, when: target.boundsCheck, to: &lines)
                }
            case .atomic(let pointer, let operation, let operand, let result):
                let target = try address(pointer)
                guard target.writable, let scalar = try atomicScalar(at: pointer) else {
                    throw unsupported("atomic read-modify-write address")
                }
                let operand = try value(operand)
                let temporary = "saga_atomic_tmp_\(handle.index)"
                if case .exchange(let compare?) = operation {
                    let expected = try value(compare)
                    try writeCompareExchange(target: target, scalar: scalar, expected: expected, replacement: operand,
                                             result: result, temporary: temporary, to: &lines)
                } else {
                    try writeAtomic(.readModifyWrite(operation, operand), target: target, scalar: scalar,
                                    result: result.map { "saga_atomic_\($0.index)" }, temporary: temporary, to: &lines)
                }
                if let result { remember(result) }
            default: throw unsupported("statement")
            }
        }
        if let breakIf {
            let test = try value(breakIf, parenthesized: false)
            try append("if (\(test)) { break; }", to: &lines)
        }
    }

    private mutating func writeSwitch(
        _ selector: IRHandle<IRExpression>, cases: [IRSwitchCase], to lines: inout [String]
    ) throws {
        try append("switch (\(try value(selector))) {", to: &lines)
        do {
            indentation += "    "
            defer { indentation.removeLast(4) }
            var index = 0
            while index < cases.count {
                let block = cases[index].body
                var labels: [String] = []
                repeat {
                    try budget.consume(.work)
                    let label: String
                    switch cases[index].value {
                    case .default: label = "default"
                    case .u32(let number): label = "\(number)u"
                    case .i32(let number):
                        let suffix = outputFormat == .wgsl ? "i" : ""
                        label = number == .min ? "(-2147483647\(suffix) - 1\(suffix))" : "\(number)\(suffix)"
                    }
                    labels.append(label)
                    index += 1
                } while index < cases.count && cases[index].body == block
                if outputFormat == .wgsl {
                    let clause = labels == ["default"] ? "default" : "case \(labels.joined(separator: ", "))"
                    try append("\(clause): {", to: &lines)
                    try writeBlock(block, to: &lines)
                    try append("}", to: &lines)
                } else {
                    for label in labels {
                        try append(label == "default" ? "default:" : "case \(label):", to: &lines)
                    }
                    try append("{", to: &lines)
                    try writeBlock(block, to: &lines)
                    try append("}", to: &lines)
                    if try !endsSwitchCase(block) { try append("break;", to: &lines) }
                }
            }
        }
        try append("}", to: &lines)
    }

    private mutating func endsSwitchCase(_ block: IRHandle<IRBlock>) throws -> Bool {
        try budget.consume(.work)
        try budget.enterDepth()
        defer { try? budget.leaveDepth() }
        guard let last = try function.blocks.value(for: block).statements.last else { return false }
        switch try function.statements.value(for: last).kind {
        case .returnValue, .break, .continue, .kill: return true
        case .block(let child): return try endsSwitchCase(child)
        case .ifCondition(_, let yes, let no): return try endsSwitchCase(yes) && endsSwitchCase(no)
        default: return false
        }
    }

    private mutating func writeCall(
        _ callee: IRHandle<IRFunction>, arguments: [IRHandle<IRExpression>],
        result: IRHandle<IRExpression>?, to lines: inout [String]
    ) throws {
        guard let captures = callCaptures[callee.index] else { throw unsupported("unplanned helper call") }
        let arguments = try arguments.map { try value($0) } + captures
        let call = "saga_fn_\(callee.index)(\(arguments.joined(separator: ", ")))"
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
    }

    private mutating func remember(_ handle: IRHandle<IRExpression>) {
        temporaries.insert(handle)
        temporaryOrder.append(handle)
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
        case nil:
            throw unsupported("unexpected return value")
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
        let hasContinuing = !continuingBlock.statements.isEmpty || breakIf != nil
        let first = "saga_first_\(index)"
        if hasContinuing {
            try append("\(declaration("bool", named: first)) = true;", to: &lines)
        }
        try append("while (true) {", to: &lines)
        do {
            indentation += "    "
            defer { indentation.removeLast(4) }
            if hasContinuing {
                // A native continue starts the next iteration, so run WGSL continuing at its head.
                try append("if (!\(first)) {", to: &lines)
                try writeBlock(continuing, breakIf: breakIf, to: &lines)
                try append("}", to: &lines)
                try append("\(first) = false;", to: &lines)
            }
            try writeBlock(body, nested: false, to: &lines)
        }
        try append("}", to: &lines)
    }

    private mutating func emit(_ handle: IRHandle<IRExpression>, to lines: inout [String]) throws {
        let expression = try function.expressions.value(for: handle)
        switch expression.kind {
        case .load(let pointer):
            // Loads are evaluated at their IR emit, before any later writes.
            let source = try address(pointer)
            let name = "saga_load_\(handle.index)"
            try writeLoad(pointer: pointer, source: source, type: expression.type, name: name, to: &lines)
            remember(handle)
        case .imageSample(let sample):
            let name = "saga_image_\(handle.index)"
            let sampled: String
            if let component = sample.gather {
                sampled = try gatherExpression(sample, component: component, named: name, to: &lines)
            } else {
                sampled = try sampleExpression(sample, named: name, to: &lines)
            }
            let binding = outputFormat == .wgsl ? "let \(name)" : declaration(try typeName(expression.type), named: name)
            try append("\(binding) = \(sampled);", to: &lines)
            remember(handle)
        case .imageLoad(let load):
            try writeImageLoad(load, result: handle, type: expression.type, to: &lines)
            remember(handle)
        case .imageQuery(let image, let query):
            try writeImageQuery(query, image: image, result: handle, type: expression.type, to: &lines)
            remember(handle)
        case .select(let condition, let accept, let reject):
            let name = "saga_select_\(handle.index)"
            let selected = try selectExpression(condition: condition, accept: accept, reject: reject, type: expression.type)
            let declaration = outputFormat == .wgsl ? "let \(name)" : "\(try typeName(expression.type)) \(name)"
            try append("\(declaration) = \(selected);", to: &lines)
            remember(handle)
        case .access(let base, let index):
            if case .pointer = try module.types.value(for: expression.type).kind {
                _ = try address(handle)
            } else {
                let source = try function.expressions.value(for: base)
                guard let width = try module.types.value(for: source.type).kind.componentCount else {
                    throw unsupported("dynamic value indexing")
                }
                let (index, nonnegative) = try indexedValue(index)
                let bounds = [nonnegative, "\(index) < \(width)u"].compactMap { $0 }.joined(separator: " && ")
                let name = "saga_index_\(handle.index)"
                let initial = try zeroValue(expression.type)
                try append("\(declaration(try typeName(expression.type), named: name)) = \(initial);", to: &lines)
                let operand = try value(base)
                try writeStatements(["\(name) = \(operand)[\(index)];"], when: bounds, to: &lines)
                remember(handle)
            }
        case .accessIndex:
            if case .pointer = try module.types.value(for: expression.type).kind {
                if try storageLocation(expression.kind) == nil { _ = try address(handle) }
            } else {
                _ = try value(handle)
            }
        default:
            _ = try value(handle)
        }
    }

    private func imageReference(_ image: IRHandle<IRExpression>) throws -> String {
        let expression = try function.expressions.value(for: image)
        guard let location = try storageLocation(expression.kind) else {
            throw unsupported("sampled texture must be a global binding")
        }
        return location.reference
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
        let depth = kind.isDepthTexture
        let cube = kind.isCubeTexture
        let texture = outputFormat == .glsl ? try ShaderTextureSamplerPair(sample, in: function).name :
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
            let args = ([sampler, uv] + [layer, metalModifier].compactMap { $0 } + offsetArguments).joined(separator: ", ")
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
        if outputFormat == .glsl {
            let pair = try ShaderTextureSamplerPair(sample, in: function)
            let layer = try clampedSampleLayer(sample, image: pair.name, named: name, to: &lines)
            if cube, let layer {
                let coord = "vec4(\(uv), float(\(layer)))"
                return levelZero ? "textureLod(\(pair.name), \(coord), \(reference), 0.0)" :
                    "texture(\(pair.name), \(coord), \(reference))"
            }
            let coord = layer.map { "vec4(\(uv), float(\($0)), \(reference))" } ?? "vec3(\(uv), \(reference))"
            let shadowCoord = cube ? "vec4(\(uv), \(reference))" : coord
            // Zero gradients select the base mip for GLSL shadow samplers without a LOD overload.
            let operation: String
            let lod: [String]
            if levelZero && (cube || layer != nil) {
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
            let level = levelZero ? ["level(0.0f)"] : []
            let args = (base + level + trailing).joined(separator: ", ")
            return "\(texture).sample_compare(\(args))"
        case .hlsl:
            let operation = levelZero ? "SampleCmpLevelZero" : "SampleCmp"
            let coord = layer.map { "\(cube ? "float4" : "float3")(\(uv), float(\($0)))" } ?? uv
            let args = ([sampler, coord, reference] + trailing).joined(separator: ", ")
            return "\(texture).\(operation)(\(args))"
        case .wgsl:
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
            let pair = try ShaderTextureSamplerPair(sample, in: function)
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
        _ image: String, level: String?, named name: String, multisampled: Bool = false, arrayed: Bool = false,
        cube: Bool = false, volume: Bool = false
    ) throws -> [String] {
        switch outputFormat {
        case .wgsl:
            let args = level.map { "\(image), \($0)" } ?? image
            return ["let \(name) = textureDimensions(\(args));"]
        case .msl:
            let mip = level ?? ""
            if cube { return ["uint2 \(name) = uint2(\(image).get_width(\(mip)));"] }
            if volume {
                return ["uint3 \(name) = uint3(\(image).get_width(\(mip)), \(image).get_height(\(mip)), \(image).get_depth(\(mip)));"]
            }
            return ["uint2 \(name) = uint2(\(image).get_width(\(mip)), \(image).get_height(\(mip)));"]
        case .glsl:
            let size = level.map { "textureSize(\(image), int(\($0)))" } ??
                (multisampled ? "textureSize(\(image))" : "imageSize(\(image))")
            let type = volume ? "uvec3" : "uvec2"
            return ["\(type) \(name) = \(type)(\(size)\(arrayed ? ".xy" : ""));"]
        case .hlsl:
            if volume, let level {
                return ["uint3 \(name);", "uint \(name)_unused;",
                        "\(image).GetDimensions(\(level), \(name).x, \(name).y, \(name).z, \(name)_unused);"]
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

    private func imageLevels(_ image: String, named name: String, arrayed: Bool = false, volume: Bool = false) throws -> [String] {
        switch outputFormat {
        case .wgsl: return ["let \(name) = textureNumLevels(\(image));"]
        case .msl: return ["uint \(name) = \(image).get_num_mip_levels();"]
        case .glsl: return ["uint \(name) = uint(textureQueryLevels(\(image)));"]
        case .hlsl:
            if arrayed || volume {
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
        _ query: IRImageQuery, image: IRHandle<IRExpression>, result: IRHandle<IRExpression>,
        type: IRHandle<IRType>, to lines: inout [String]
    ) throws {
        let imageType = try function.expressions.value(for: image).type
        let kind = try module.types.value(for: imageType).kind
        let image = try imageReference(image)
        let name = "saga_image_\(result.index)"
        switch query {
        case .numLevels:
            try writeStatements(imageLevels(image, named: name, arrayed: kind.isArrayedTexture, volume: kind.isVolumeTexture),
                                when: nil, to: &lines)
        case .numSamples:
            try writeStatements(imageSamples(image, named: name), when: nil, to: &lines)
        case .numLayers:
            try writeStatements(imageLayers(image, named: name, storage: kind.storageTexture2D != nil),
                                when: nil, to: &lines)
        case .size(nil):
            let level = kind.storageTexture2D != nil || kind.isMultisampled2D ? nil : "0u"
            try writeStatements(imageDimensions(image, level: level, named: name, multisampled: kind.isMultisampled2D,
                                                arrayed: kind.isArrayedTexture, cube: kind.isCubeTexture, volume: kind.isVolumeTexture),
                                when: nil, to: &lines)
        case .size(let level?):
            let (index, nonnegative) = try indexedValue(level)
            let levels = "\(name)_levels"
            try writeStatements(imageLevels(image, named: levels, arrayed: kind.isArrayedTexture, volume: kind.isVolumeTexture),
                                when: nil, to: &lines)
            let initial = try zeroValue(type)
            let size = "\(name)_size"
            let bounds = imageBounds([nonnegative, "\(index) < \(levels)"].compactMap { $0 })
            if outputFormat == .wgsl {
                let valid = "\(name)_valid_level"
                try append("let \(valid) = \(bounds);", to: &lines)
                try writeStatements(imageDimensions(image, level: "select(0u, \(index), \(valid))", named: size,
                                                    arrayed: kind.isArrayedTexture, cube: kind.isCubeTexture, volume: kind.isVolumeTexture),
                                    when: nil, to: &lines)
                try append("let \(name) = select(\(initial), \(size), \(valid));", to: &lines)
            } else {
                try append("\(declaration(try typeName(type), named: name)) = \(initial);", to: &lines)
                try writeStatements(imageDimensions(image, level: index, named: size, arrayed: kind.isArrayedTexture,
                                                    cube: kind.isCubeTexture, volume: kind.isVolumeTexture) + ["\(name) = \(size);"],
                                    when: bounds, to: &lines)
            }
        }
    }

    private mutating func writeImageLoad(
        _ load: IRImageLoad, result: IRHandle<IRExpression>, type: IRHandle<IRType>, to lines: inout [String]
    ) throws {
        let imageType = try function.expressions.value(for: load.image).type
        let imageKind = try module.types.value(for: imageType).kind
        if let storage = imageKind.storageTexture2D {
            try writeStorageImageLoad(load, result: result, type: type, scalar: storage.scalar, components: storage.components,
                                      writable: storage.access.contains(.store), to: &lines)
            return
        }
        let multisampled = imageKind.isMultisampled2D
        let arrayed = imageKind.isArrayedTexture
        let volume = imageKind.isVolumeTexture
        guard let operand = multisampled ? load.sample : load.level else {
            throw unsupported("sampled texture load without sample index or level")
        }
        let image = try imageReference(load.image)
        let name = "saga_image_\(result.index)"
        let (coordinates, unsigned, signed) = try imageCoordinates(load.coordinate, named: name, volume: volume, to: &lines)
        let (index, nonnegative) = try indexedValue(operand)
        let count = "\(name)_count"
        let query = multisampled ? try imageSamples(image, named: count) :
            try imageLevels(image, named: count, arrayed: arrayed, volume: volume)
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
        let bounds = imageCoordinateBounds(coordinates, unsigned: unsigned, size: size, signed: signed, volume: volume)
        if outputFormat == .wgsl {
            // Use a valid address and select the zero result so uniform inputs stay uniform after re-parsing.
            let validIndex = "\(name)_valid_index"
            let safeIndex = "\(name)_index"
            let valid = "\(name)_valid"
            let safeCoordinates = "\(name)_safe_coordinates"
            try append("let \(validIndex) = \(indexBounds);", to: &lines)
            try append("let \(safeIndex) = select(0u, \(index), \(validIndex));", to: &lines)
            try writeStatements(imageDimensions(image, level: multisampled ? nil : safeIndex, named: size,
                                                multisampled: multisampled, arrayed: arrayed, volume: volume), when: nil, to: &lines)
            try append("let \(valid) = \(validIndex) & \(bounds);", to: &lines)
            let zero = volume ? "vec3u(0u)" : "vec2u(0u)"
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
        try writeStatements(imageDimensions(image, level: multisampled ? nil : index, named: size,
                                            multisampled: multisampled, arrayed: arrayed, volume: volume), when: nil, to: &lines)
        let texel: String
        switch outputFormat {
        case .msl:
            let args = [unsigned, layer, index].compactMap { $0 }.joined(separator: ", ")
            texel = "\(image).read(\(args))"
        case .hlsl:
            let location = "\(volume ? "int3" : "int2")(\(coordinates))"
            if let layer {
                texel = "\(image).Load(int4(\(location), int(\(layer)), int(\(index))))"
            } else {
                texel = multisampled ? "\(image).Load(\(location), int(\(index)))" :
                    "\(image).Load(\(volume ? "int4" : "int3")(\(location), int(\(index))))"
            }
        case .glsl:
            let component = imageKind.isDepthTexture ? ".r" : ""
            let coordinate = "\(volume ? "ivec3" : "ivec2")(\(coordinates))"
            let location = layer.map { "ivec3(\(coordinate), int(\($0)))" } ?? coordinate
            texel = "texelFetch(\(image), \(location), int(\(index)))\(component)"
        default: throw unsupported("sampled texture load target")
        }
        try writeStatements(["\(name) = \(texel);"], when: bounds, to: &lines)
        indentation.removeLast(4)
        try append("}", to: &lines)
    }

    private mutating func imageCoordinates(
        _ coordinate: IRHandle<IRExpression>, named name: String, volume: Bool = false, to lines: inout [String]
    ) throws -> (raw: String, unsigned: String, signed: Bool) {
        let type = try function.expressions.value(for: coordinate).type
        let signed = try module.types.value(for: type).kind == .vector(size: volume ? .three : .two, scalar: .i32)
        let coordinates = "\(name)_coordinates"
        let binding = outputFormat == .wgsl ? "let \(coordinates)" : "\(try typeName(type)) \(coordinates)"
        try append("\(binding) = \(try value(coordinate));", to: &lines)
        let unsigned = signed ? integerBits(coordinates, signed: false, width: volume ? 3 : 2, in: outputFormat) : coordinates
        return (coordinates, unsigned, signed)
    }

    private mutating func writeStorageImageLoad(
        _ load: IRImageLoad, result: IRHandle<IRExpression>, type: IRHandle<IRType>,
        scalar: IRScalar, components: Int, writable: Bool, to lines: inout [String]
    ) throws {
        let image = try imageReference(load.image)
        let name = "saga_image_\(result.index)"
        let (coordinates, unsigned, layer, bounds) = try storageImageAddress(
            image, coordinate: load.coordinate, arrayIndex: load.arrayIndex, named: name, to: &lines)
        let initial = try zeroValue(type)
        if outputFormat == .wgsl && !writable {
            let valid = "\(name)_valid"
            let safe = "\(name)_safe_coordinates"
            try append("let \(valid) = \(bounds);", to: &lines)
            try append("let \(safe) = select(vec2u(0u), \(unsigned), \(valid));", to: &lines)
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
            let xy = "ivec2(\(coordinates))"
            let location = layer.map { "ivec3(\(xy), int(\($0)))" } ?? xy
            texel = "imageLoad(\(image), \(location))"
        default: throw unsupported("storage texture load target")
        }
        if components < 4 && (outputFormat == .msl || outputFormat == .hlsl) {
            let zero = scalar == .u32 ? "0u" : scalar == .f32 ? "0.0f" : "0"
            let one = scalar == .u32 ? "1u" : scalar == .f32 ? "1.0f" : "1"
            if outputFormat == .msl { texel += components == 1 ? ".x" : ".xy" }
            let lanes = [texel] + Array(repeating: zero, count: 3 - components) + [one]
            texel = "\(try typeName(type))(\(lanes.joined(separator: ", ")))"
        }
        try append("\(declaration(try typeName(type), named: name)) = \(initial);", to: &lines)
        try writeStatements(["\(name) = \(texel);"], when: bounds, to: &lines)
    }

    private func imageCoordinateBounds(_ coordinates: String, unsigned: String, size: String, signed: Bool,
                                       volume: Bool = false) -> String {
        let zero = outputFormat == .wgsl ? "0i" : "0"
        let axes = volume ? ["x", "y", "z"] : ["x", "y"]
        let positive = signed ? axes.map { "\(coordinates).\($0) >= \(zero)" } : []
        return imageBounds(positive + axes.map { "(\(unsigned)).\($0) < \(size).\($0)" })
    }

    private mutating func writeImageStore(
        image: IRHandle<IRExpression>, coordinate: IRHandle<IRExpression>, arrayIndex: IRHandle<IRExpression>?,
        value rhs: IRHandle<IRExpression>,
        named name: String, to lines: inout [String]
    ) throws {
        let type = try function.expressions.value(for: image).type
        guard let storage = try module.types.value(for: type).kind.storageTexture2D else {
            throw unsupported("storage texture store type")
        }
        let image = try imageReference(image)
        let (coordinates, unsigned, layer, bounds) = try storageImageAddress(
            image, coordinate: coordinate, arrayIndex: arrayIndex, named: name, to: &lines)
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
            let xy = "ivec2(\(coordinates))"
            let location = layer.map { "ivec3(\(xy), int(\($0)))" } ?? xy
            store = "imageStore(\(image), \(location), \(color));"
        default: throw unsupported("storage texture target")
        }
        try writeStatements([store], when: bounds, to: &lines)
    }

    private mutating func storageImageAddress(
        _ image: String, coordinate: IRHandle<IRExpression>, arrayIndex: IRHandle<IRExpression>?,
        named name: String, to lines: inout [String]
    ) throws -> (coordinates: String, unsigned: String, layer: String?, bounds: String) {
        let (coordinates, unsigned, signed) = try imageCoordinates(coordinate, named: name, to: &lines)
        let size = "\(name)_size"
        try writeStatements(imageDimensions(image, level: nil, named: size, arrayed: arrayIndex != nil),
                            when: nil, to: &lines)
        let bounds = imageCoordinateBounds(coordinates, unsigned: unsigned, size: size, signed: signed)
        guard let arrayIndex else { return (coordinates, unsigned, nil, bounds) }
        let (layer, nonnegative) = try indexedValue(arrayIndex)
        let layers = "\(name)_layers"
        try writeStatements(imageLayers(image, named: layers, storage: true), when: nil, to: &lines)
        let layerBounds = [nonnegative, "\(layer) < \(layers)"].compactMap { $0 }
        return (coordinates, unsigned, layer, imageBounds([bounds] + layerBounds))
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
        let value = try source.memory.load(in: outputFormat)
        if let condition = source.boundsCheck {
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
            let call = "workgroupUniformLoad(&\(reference))"
            if let bounds = source.boundsCheck {
                let initial = try zeroValue(type)
                try append("let \(name) = select(\(initial), \(call), \(bounds));", to: &lines)
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

    private func atomicScalar(at handle: IRHandle<IRExpression>) throws -> IRScalar? {
        let expression = try function.expressions.value(for: handle)
        guard case .pointer(let type, _) = try module.types.value(for: expression.type).kind,
              case .atomic(let scalar) = try module.types.value(for: type).kind else { return nil }
        return scalar
    }

    private mutating func writeCompareExchange(
        target: ShaderAddress, scalar: IRScalar, expected: String, replacement: String,
        result: IRHandle<IRExpression>?, temporary: String, to lines: inout [String]
    ) throws {
        let name = result.map { "saga_atomic_\($0.index)" }
        if let result, let name {
            let type = try function.expressions.value(for: result).type
            let initial = try zeroValue(type)
            try append("\(declaration(try typeName(type), named: name)) = \(initial);", to: &lines)
        }
        if outputFormat == .hlsl, let raw = target.memory.hlslAtomicByteAddress {
            let compared = "\(temporary)_expected"
            let compareBits = scalar == .i32 ? "asuint(\(expected))" : expected
            let replacementBits = scalar == .i32 ? "asuint(\(replacement))" : replacement
            var operations = [
                "uint \(compared) = \(compareBits);",
                "uint \(temporary);",
                "\(raw.buffer).InterlockedCompareExchange(\(raw.offset), \(compared), \(replacementBits), \(temporary));",
            ]
            if let name {
                let old = scalar == .i32 ? "asint(\(temporary))" : temporary
                operations += ["\(name).saga_member_0 = \(old);", "\(name).saga_member_1 = \(temporary) == \(compared);"]
            }
            try writeStatements(operations, when: target.boundsCheck, to: &lines)
            return
        }
        guard case .native(let reference) = target.memory else { throw unsupported("atomic compare-exchange address") }
        let lane = scalar == .i32 ? "int" : "uint"
        let oldValue = name.map { "\($0).saga_member_0" } ?? temporary
        let statements: [String]
        switch outputFormat {
        case .msl:
            let start = name == nil ? "\(lane) \(oldValue)" : oldValue
            let call = "atomic_compare_exchange_weak_explicit(&(\(reference)), &\(oldValue), \(replacement), " +
                "memory_order_relaxed, memory_order_relaxed)"
            let assign = name.map { "\($0).saga_member_1 = " } ?? ""
            statements = ["\(start) = \(expected);", "\(assign)\(call);"]
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
            let call = "atomicCompareExchangeWeak(&\(reference), \(expected), \(replacement))"
            if let name {
                statements = ["let \(temporary) = \(call);", "\(oldValue) = \(temporary).old_value;",
                              "\(name).saga_member_1 = \(temporary).exchanged;"]
            } else {
                statements = ["_ = \(call);"]
            }
        default: throw unsupported("atomic compare-exchange target")
        }
        try writeStatements(statements, when: target.boundsCheck, to: &lines)
    }

    private mutating func writeAtomic(
        _ operation: ShaderAtomicOperation, target: ShaderAddress, scalar: IRScalar,
        result: String?, temporary: String, to lines: inout [String]
    ) throws {
        let signed = scalar == .i32
        let zero = signed ? (outputFormat == .wgsl ? "0i" : "0") : "0u"
        let type = outputFormat == .wgsl ? (signed ? "i32" : "u32") : (signed ? "int" : "uint")
        if let result { try append("\(declaration(type, named: result)) = \(zero);", to: &lines) }
        let assignment = result.map { "\($0) = " } ?? ""
        let (intrinsic, operand) = try atomicIntrinsic(operation, signed: signed, zero: zero)
        if outputFormat == .hlsl, let raw = target.memory.hlslAtomicByteAddress, let operand {
            let statements = hlslRawAtomic(operation, intrinsic: intrinsic, operand: operand, signed: signed,
                                           raw: raw, result: result, temporary: "\(temporary)_bits")
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
            var statements: [String] = []
            if result == nil { statements.append("\(type) \(temporary);") }
            statements.append("\(intrinsic)(\(reference), \(operand), \(result ?? temporary));")
            try writeStatements(statements, when: target.boundsCheck, to: &lines)
            return
        default: throw unsupported("atomic operation target")
        }
        try writeStatements(["\(assignment)\(call);"], when: target.boundsCheck, to: &lines)
    }

    private func hlslRawAtomic(
        _ operation: ShaderAtomicOperation, intrinsic: String, operand: String, signed: Bool,
        raw: (buffer: String, offset: String), result: String?, temporary: String
    ) -> [String] {
        var lines = ["uint \(temporary);"]
        if signed, case .readModifyWrite(let function, _) = operation, function == .min || function == .max {
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
                "    uint \(next) = asuint(\(extremum)(asint(\(previous)), \(operand)));",
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
        _ operation: ShaderAtomicOperation, signed: Bool, zero: String
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
            case .and: (metal, hlsl) = ("atomic_fetch_and_explicit", "InterlockedAnd")
            case .inclusiveOr: (metal, hlsl) = ("atomic_fetch_or_explicit", "InterlockedOr")
            case .exclusiveOr: (metal, hlsl) = ("atomic_fetch_xor_explicit", "InterlockedXor")
            case .exchange: (metal, hlsl) = ("atomic_exchange_explicit", "InterlockedExchange")
            }
            names = (metal, hlsl, function == .subtract ? "atomicAdd" : builtin, builtin)
            if function == .subtract && (outputFormat == .glsl || outputFormat == .hlsl) {
                operand = signed ? signedNegation(value, in: outputFormat) : "0u - (\(value))"
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
        if case .localVariable(let local) = pointer.kind {
            _ = try function.localVariables.value(for: local)
            return .init(memory: .native(localName(local.index)), boundsCheck: nil, writable: true)
        }
        if case .accessIndex(let base, let index) = pointer.kind {
            let source = try function.expressions.value(for: base)
            if case .pointer(let pointee, .function) = try module.types.value(for: source.type).kind,
               case .structure = try module.types.value(for: pointee).kind {
                let parent = try address(base, guardIndexForUniformLoad: guardIndexForUniformLoad)
                guard case .native(let reference) = parent.memory else { throw unsupported("local structure storage") }
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
        if case .pointer(let pointee, _) = try module.types.value(for: source.type).kind,
           let width = try module.types.value(for: pointee).kind.componentCount {
            let parent = try address(base, guardIndexForUniformLoad: guardIndexForUniformLoad)
            let bounds = [parent.boundsCheck, nonnegative, "\(index) < \(width)u"]
                .compactMap { $0 }.joined(separator: " && ")
            let safeIndex = guardIndexForUniformLoad ? "select(0u, \(index), \(bounds))" : index
            return .init(memory: parent.memory.component(safeIndex), boundsCheck: bounds, writable: parent.writable)
        }
        guard let binding = try storageLocation(source.kind), let bound = binding.bound else {
            throw unsupported("storage access must name an indexed array element")
        }
        let upperBound = "\(index) < \(bound)"
        let bounds = nonnegative.map { "\($0) && \(upperBound)" } ?? upperBound
        let safeIndex = guardIndexForUniformLoad ? "select(0u, \(index), \(bounds))" : index
        return .init(memory: binding.memory(at: safeIndex), boundsCheck: bounds, writable: binding.writable)
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
              type == .u32 || type == .i32 || type == .bool || type == .f32
        else { throw unsupported("expected u32, i32, f32 or bool scalar") }
        return type
    }

    private func localName(_ index: Int) -> String { "saga_local_\(index)" }
    private func scalarZero(_ type: IRScalar) -> String {
        if type == .bool { return "false" }
        if type == .f32 { return floatLiteral(0) }
        if type == .i32 { return outputFormat == .wgsl ? "0i" : "0" }
        return "0u"
    }

    private mutating func zeroValue(_ type: IRHandle<IRType>) throws -> String {
        try budget.consume(.work)
        try budget.enterDepth()
        defer { try? budget.leaveDepth() }
        switch try module.types.value(for: type).kind {
        case .scalar: return try scalarZero(scalar(type))
        case .vector(let size, let scalar) where [.f32, .i32, .u32, .bool].contains(scalar):
            return try construct(type, components: Array(repeating: scalarZero(scalar), count: Int(size.rawValue)))
        case .matrix(let columns, let rows, .f32):
            let zero = scalarZero(.f32)
            let column = "\(floatType(rows.rawValue, in: outputFormat))(\(Array(repeating: zero, count: Int(rows.rawValue)).joined(separator: ", ")))"
            return try construct(type, components: Array(repeating: column, count: Int(columns.rawValue)))
        case .structure(let members, _) where structures[type] != nil:
            let components = try members.map { try zeroValue($0.type) }
            return try construct(type, components: components)
        default: throw unsupported("zero value for unsupported type")
        }
    }

    private func floatLiteral(_ value: Float) -> String {
        outputFormat == .glsl ? String(value) : "\(value)f"
    }

    private func literalValue(_ literal: IRLiteral) throws -> String {
        switch literal {
        case .u32(let number): return "\(number)u"
        case .i32(let number):
            if number == .min { return integerBits("2147483648u", signed: true, in: outputFormat) }
            let value = outputFormat == .wgsl ? "\(number)i" : String(number)
            return number < 0 ? "(\(value))" : value
        case .f32(let number) where number.isFinite: return floatLiteral(number)
        case .bool(let flag): return flag ? "true" : "false"
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
        if case .matrix(let columns, _, .f32) = try module.types.value(for: type).kind,
           index < UInt32(columns.rawValue) {
            return "[\(index)u]"
        }
        if case .vector(let width, let scalar) = try module.types.value(for: type).kind,
           [.f32, .u32, .i32, .bool].contains(scalar), index < UInt32(width.rawValue) {
            return [".x", ".y", ".z", ".w"][Int(index)]
        }
        if structures[type] != nil { return ".saga_member_\(index)" }
        throw unsupported("access outside a supported vector or render structure")
    }

    private mutating func value(
        _ handle: IRHandle<IRExpression>, parenthesized: Bool = true
    ) throws -> String {
        try budget.consume(.work)
        try budget.enterDepth()
        defer { try? budget.leaveDepth() }
        let expression = try function.expressions.value(for: handle)
        switch expression.kind {
        case .literal(let literal): return try literalValue(literal)
        case .constant(let handle):
            let initializer = try module.constants.value(for: handle).initializer
            return try constantValue(initializer)
        case .zeroValue(let type): return try zeroValue(type)
        case .arrayLength(let pointer):
            guard let bound = try storageLocation(function.expressions.value(for: pointer).kind)?.bound
            else { throw unsupported("length of an array outside a supported storage binding") }
            return bound
        case .imageSample where temporaries.contains(handle), .imageLoad where temporaries.contains(handle),
             .imageQuery where temporaries.contains(handle):
            return "saga_image_\(handle.index)"
        case .load where temporaries.contains(handle): return "saga_load_\(handle.index)"
        case .access where temporaries.contains(handle): return "saga_index_\(handle.index)"
        case .select where temporaries.contains(handle): return "saga_select_\(handle.index)"
        case .workgroupUniformLoadResult where temporaries.contains(handle): return "saga_uniform_\(handle.index)"
        case .atomicResult where temporaries.contains(handle): return "saga_atomic_\(handle.index)"
        case .callResult where temporaries.contains(handle): return "saga_call_\(handle.index)"
        case .functionArgument(let argument):
            guard let index = Int(exactly: argument), arguments.indices.contains(index) else {
                throw ShaderFailure.invalidHandle
            }
            guard !arguments[index].isEmpty else { throw unsupported("whole structure input value") }
            return arguments[index]
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
        case .cast(let operand, let target):
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
            default: throw unsupported("cast requires a scalar or vector")
            }
            guard [.i32, .u32].contains(input), [.i32, .u32, .f32].contains(target) else {
                throw unsupported("cast outside the supported numeric profile")
            }
            let source = try value(operand)
            if target == .f32 { return "\(try typeName(expression.type))(\(source))" }
            return input == target ? source : integerBits(source, signed: target == .i32, width: width, in: outputFormat)
        case .unary(let op, let argument):
            if case .vector(let size, let lane) = try module.types.value(for: expression.type).kind,
               lane == .i32 || lane == .u32 {
                let operand = try value(argument)
                if op == .bitwiseNot { return "(~\(operand))" }
                guard op == .negate && lane == .i32 else { throw unsupported("integer vector unary operation") }
                let helper = IntegerVectorHelper(operation: .negate, signed: true, width: size.rawValue)
                integerVectorHelpers.insert(helper)
                return "\(helper.name)(\(operand))"
            }
            let type = try scalar(expression.type)
            if op == .negate && type == .i32 {
                return signedNegation(try value(argument), in: outputFormat)
            }
            let symbol: String
            switch op {
            case .logicalNot where type == .bool: symbol = "!"
            case .bitwiseNot where type == .u32 || type == .i32: symbol = "~"
            case .negate where type == .f32 || type == .i32: symbol = "-"
            default: throw unsupported("unary operation \(op.rawValue)")
            }
            try budget.consume(.sourceBytes, amount: 3)
            let unary = "\(symbol)\(try value(argument))"
            return parenthesized ? "(\(unary))" : unary
        case .binary(let op, let left, let right):
            let operandType = try function.expressions.value(for: left).type
            let lhsKind = try module.types.value(for: operandType).kind
            let rhsType = try function.expressions.value(for: right).type
            let rhsKind = try module.types.value(for: rhsType).kind
            if lhsKind.isMatrix || rhsKind.isMatrix {
                try budget.consume(.sourceBytes, amount: 8)
                let lhs = try value(left)
                let rhs = try value(right)
                // HLSL rows represent WGSL columns. The transpose of a product reverses the operands.
                if outputFormat == .hlsl, op == .multiply, !lhsKind.isScalar, !rhsKind.isScalar {
                    return "mul(\(rhs), \(lhs))"
                }
                let symbol = try shaderBinarySymbol(op, type: .f32)
                let binary = "\(lhs) \(symbol) \(rhs)"
                return parenthesized ? "(\(binary))" : binary
            }
            if case .vector(let size, let lane) = lhsKind,
               lane == .i32 || lane == .u32 {
                let helper = IntegerVectorHelper(operation: .binary(op), signed: lane == .i32, width: size.rawValue)
                try budget.consume(.sourceBytes, amount: helper.name.utf8.count + 4)
                let a = try value(left)
                let b = try value(right)
                integerVectorHelpers.insert(helper)
                return "\(helper.name)(\(a), \(b))"
            }
            let type = try scalar(operandType)
            let shift = op == .shiftLeft || op == .shiftRight
            let rightType = try scalar(rhsType)
            guard rightType == (shift ? .u32 : type) else {
                throw unsupported("binary operand types")
            }
            // Charge each expansion before traversing children so shared IR cannot inflate output unchecked.
            try budget.consume(.sourceBytes, amount: 8)
            let lhs = try value(left)
            let rhs = try value(right)
            if let lowered = try integerBinary(op, scalar: type, left: lhs, right: rhs,
                                               in: outputFormat, helpers: &integerHelpers) {
                return lowered
            }
            let symbol = try shaderBinarySymbol(op, type: type)
            let binary = "\(lhs) \(symbol) \(rhs)"
            return parenthesized ? "(\(binary))" : binary
        default: throw unsupported("expression")
        }
    }

}

private extension IRTypeKind {
    var componentCount: UInt8? {
        switch self {
        case .vector(let size, _), .matrix(let size, _, .f32): size.rawValue
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
    switch (op, type) {
    case (.equal, _): "=="
    case (.notEqual, _): "!="
    case (.logicalAnd, .bool): "&&"
    case (.logicalOr, .bool): "||"
    case (.add, .u32), (.add, .i32), (.add, .f32): "+"
    case (.subtract, .u32), (.subtract, .i32), (.subtract, .f32): "-"
    case (.multiply, .u32), (.multiply, .i32), (.multiply, .f32): "*"
    case (.less, .u32), (.less, .i32), (.less, .f32): "<"
    case (.lessEqual, .u32), (.lessEqual, .i32), (.lessEqual, .f32): "<="
    case (.greater, .u32), (.greater, .i32), (.greater, .f32): ">"
    case (.greaterEqual, .u32), (.greaterEqual, .i32), (.greaterEqual, .f32): ">="
    case (.and, .u32), (.and, .i32): "&"
    case (.inclusiveOr, .u32), (.inclusiveOr, .i32): "|"
    case (.exclusiveOr, .u32), (.exclusiveOr, .i32): "^"
    default: throw unsupported("binary operation \(op.rawValue)")
    }
}

private func unsupported(_ reason: String) -> ShaderFailure {
    .unsupported("Shader body code generation: \(reason)")
}
