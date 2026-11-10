import SAGAShaderIR

func storageMemoryQualifier(
    _ flags: IRMemoryDecorations, format: ShaderOutputFormat, webGPU: Bool, writable: Bool, inferCoherence: Bool
) throws -> String {
    if flags.contains(.volatile) {
        guard format == .glsl || (format == .wgsl && !webGPU) else {
            throw unsupported("volatile storage buffers require a target with a volatile buffer qualifier")
        }
    }
    let coherent = flags.contains(.coherent)
    switch format {
    case .msl: return coherent ? "coherent " : ""
    case .hlsl:
        guard !coherent || writable else { throw unsupported("HLSL coherent storage requires a UAV binding") }
        return coherent ? "globallycoherent " : ""
    case .glsl: return (coherent || inferCoherence ? "coherent " : "") + (flags.contains(.volatile) ? "volatile " : "")
    case .wgsl where !webGPU:
        return (coherent ? "@coherent " : "") + (flags.contains(.volatile) ? "@volatile " : "")
    default: return ""
    }
}

extension IRAddressSpace {
    var isBufferAddressSpace: Bool {
        switch self {
        case .uniform, .storage, .immediate: return true
        default: return false
        }
    }
}

func hasNestedBufferLayout(_ type: IRHandle<IRType>, module: IRModule) throws -> Bool {
    func needsArrayValueAccess(_ element: IRHandle<IRType>, size: IRArraySize, stride: UInt32) throws -> Bool {
        let kind = try module.types.value(for: element).kind
        if kind.isShaderCompositeValue { return true }
        let natural: UInt32?
        switch kind {
        case .scalar(let scalar), .atomic(let scalar): natural = UInt32(scalar.width)
        case .vector(let size, let scalar):
            natural = UInt32(size == .three ? 4 : size.rawValue) * UInt32(scalar.width)
        case .matrix(let columns, let rows, let scalar):
            natural = UInt32(columns.rawValue) * UInt32(rows == .three ? 4 : rows.rawValue) * UInt32(scalar.width)
        default: natural = nil
        }
        if let natural, stride != natural { return true }
        guard case .constant = size else { return false }
        switch kind {
        case .scalar(.i64), .scalar(.u64), .vector(_, .i64), .vector(_, .u64): return true
        default: return false
        }
    }
    switch try module.types.value(for: type).kind {
    case .array(let element, let size, let stride):
        return try needsArrayValueAccess(element, size: size, stride: stride)
    case .structure(let members, _):
        return try members.contains { member in
            switch try module.types.value(for: member.type).kind {
            case .structure: return true
            case .array(let element, let size, let stride): return try needsArrayValueAccess(element, size: size, stride: stride)
            default: return false
            }
        }
    default: return false
    }
}

struct NestedBufferLayout {
    let name: String
    let type: IRHandle<IRType>
    let runtimeLength: String?
    let space: IRAddressSpace
    var glslReference: String? = nil
    var wrappedArrays: Set<IRHandle<IRType>> = []
    var bindingArray: BufferArrayAddress? = nil
    var metalMemoryQualifier = ""

    var writable: Bool {
        if case .storage(let access) = space { return access.contains(.store) }
        return false
    }
}

struct NestedBufferAccess {
    let buffer: NestedBufferLayout
    let type: IRHandle<IRType>
    let reference: String
    let byteOffset: String

    init(_ buffer: NestedBufferLayout) {
        self.init(buffer: buffer, type: buffer.type, reference: buffer.glslReference ?? buffer.name, byteOffset: "0u")
    }

    private init(buffer: NestedBufferLayout, type: IRHandle<IRType>, reference: String, byteOffset: String) {
        self.buffer = buffer
        self.type = type
        self.reference = reference
        self.byteOffset = byteOffset
    }

    func member(_ member: IRStructMember, at index: UInt32) -> Self {
        .init(buffer: buffer, type: member.type, reference: "\(reference).saga_member_\(index)",
              byteOffset: member.offset == 0 ? byteOffset : "\(byteOffset) + \(member.offset)u")
    }

    func element(_ type: IRHandle<IRType>, stride: UInt32, index: String) -> Self {
        let member = buffer.wrappedArrays.contains(self.type) ? ".saga_value" : ""
        return .init(buffer: buffer, type: type, reference: "\(reference)[\(index)]\(member)",
              byteOffset: "\(byteOffset) + (\(index)) * \(stride)u")
    }
}

struct NestedStorageBinding {
    let storage: ShaderStorage
    let parameters: [String]
    let captures: [ShaderParameter]
    let declarations: [String]
    let reflection: ComputeShaderArtifact.StorageBinding
    let usesGLSL16BitStorage: Bool

    init(
        global: IRGlobalVariable, index: Int, binding: IRResourceBinding, flat: UInt32, runtimeSlot: Int,
        metalLengths: ShaderParameter, writable: Bool, memoryQualifier: String, module: IRModule, format: ShaderOutputFormat,
        budget: inout ShaderBudget
    ) throws {
        let name = "saga_buffer_\(index)"
        var types = NestedBufferTypes(module: module, format: format, prefix: "SagaNestedStorage\(index)_")
        let root = try types.describe(global.type, budget: &budget)
        let tail = try Self.runtimeTail(global.type, module: module)
        let glslBlock = try types.glslStorageBlock(global.type, tailMember: tail?.member, budget: &budget)
        usesGLSL16BitStorage = glslBlock?.uses16BitStorage == true
        var declarations = types.declarations
        let length: String?
        let pointer: ShaderParameter?
        switch format {
        case .msl:
            pointer = .init(type: memoryQualifier + (writable ? "device uint*" : "device const uint*"), name: name)
            length = tail.map { _ in "\(metalLengths.name)[\(runtimeSlot)]" }
        case .hlsl:
            pointer = nil
            let qualifier = writable ? "RWByteAddressBuffer" : "ByteAddressBuffer"
            let register = writable ? "u" : "t"
            declarations.append("\(memoryQualifier)\(qualifier) \(name) : register(\(register)\(binding.binding), space\(binding.group));")
            length = tail.map { tail in
                let function = "saga_length_\(index)"
                declarations += ["uint \(function)() {", "    uint bytes;", "    \(name).GetDimensions(bytes);",
                                 "    if (bytes < \(tail.offset)u) { return 0u; }",
                                 "    return (bytes - \(tail.offset)u) / \(tail.stride)u;", "}"]
                return "\(function)()"
            }
        case .wgsl:
            pointer = nil
            let access = writable ? "read_write" : "read"
            declarations.append("@group(\(binding.group)) @binding(\(binding.binding)) \(memoryQualifier)var<storage, \(access)> \(name): \(root.name);")
            length = tail.map { tail in
                let reference = tail.member.map { "\(name).saga_member_\($0)" } ?? name
                return "arrayLength(&\(reference))"
            }
        case .glsl:
            pointer = nil
            let qualifier = memoryQualifier + (writable ? "buffer" : "readonly buffer")
            let members = glslBlock?.members ?? "uint words[];"
            declarations.append("layout(std430, binding = \(flat)) \(qualifier) SagaBuffer\(index) { \(members) } \(name);")
            if let glslBlock {
                length = tail.map { _ in "uint(\(name)\(glslBlock.tailSuffix).length())" }
            } else {
                length = tail.map { tail in
                    let function = "saga_length_\(index)"
                    declarations += ["uint \(function)() {", "    uint bytes = uint(\(name).words.length()) * 4u;",
                                     "    if (bytes < \(tail.offset)u) { return 0u; }",
                                     "    return (bytes - \(tail.offset)u) / \(tail.stride)u;", "}"]
                    return "\(function)()"
                }
            }
        default: throw ShaderFailure.unsupported("Nested storage buffer target")
        }
        storage = .init(reference: name, writable: writable,
                        nested: .init(name: name, type: global.type, runtimeLength: length, space: global.space,
                                      glslReference: glslBlock.map { name + $0.rootSuffix },
                                      wrappedArrays: format == .wgsl || glslBlock != nil ? types.paddedArrays : [],
                                      metalMemoryQualifier: format == .msl ? memoryQualifier : ""))
        if let pointer {
            parameters = ["\(pointer.type) \(name) [[buffer(\(flat))]]"]
            captures = tail == nil ? [pointer] : [pointer, metalLengths]
        } else {
            parameters = []
            captures = []
        }
        self.declarations = declarations
        reflection = .init(binding: binding, flatBindingIndex: format == .msl || format == .glsl ? flat : nil,
                           minimumByteSize: root.size, isReadOnly: !writable, runtimeArrayStride: tail?.stride,
                           runtimeArrayOffset: tail?.offset ?? 0, hlslUsesByteAddressBuffer: format == .hlsl,
                           memoryDecorations: global.memoryDecorations)
    }

    static func runtimeTail(
        _ type: IRHandle<IRType>, module: IRModule
    ) throws -> (offset: UInt32, stride: UInt32, member: Int?)? {
        switch try module.types.value(for: type).kind {
        case .array(_, .dynamic, let stride): return (0, stride, nil)
        case .structure(let members, _):
            if let last = members.last,
               case .array(_, .dynamic, let stride) = try module.types.value(for: last.type).kind {
                return (last.offset, stride, members.count - 1)
            }
            return nil
        default: return nil
        }
    }
}

struct GLSLNestedStorageBlock {
    let members: String
    let rootSuffix: String
    let tailSuffix: String
    let uses16BitStorage: Bool
}

struct NestedBufferTypes {
    struct Shape {
        let name: String
        let size: UInt64
        let alignment: UInt64
        var runtime = false
        var containsWideInteger = false
        var contains16Bit = false
    }

    let module: IRModule
    let format: ShaderOutputFormat
    let prefix: String
    var declarations: [String] = []
    private(set) var paddedArrays: Set<IRHandle<IRType>> = []
    private var known: [IRHandle<IRType>: Shape] = [:]
    private var glslStructures: Set<IRHandle<IRType>> = []
    private let glslHalfwordPadding: String

    init(module: IRModule, format: ShaderOutputFormat, prefix: String) {
        self.module = module
        self.format = format
        self.prefix = prefix
        glslHalfwordPadding = format == .glsl && moduleUsesInt16(module) ? "uint16_t" : "float16_t"
    }

    mutating func describe(_ handle: IRHandle<IRType>, budget: inout ShaderBudget) throws -> Shape {
        try budget.consume(.work)
        if let shape = known[handle] { return shape }
        try budget.enterDepth()
        defer { try? budget.leaveDepth() }
        let shape: Shape
        switch try module.types.value(for: handle).kind {
        case .array(let base, let size, let stride):
            let element = try describe(base, budget: &budget)
            let natural = aligned(element.size, to: element.alignment)
            guard UInt64(stride) >= natural, UInt64(stride) % element.alignment == 0 else {
                throw unsupported("invalid nested array stride")
            }
            var elementName = element.name
            if UInt64(stride) != natural {
                paddedArrays.insert(handle)
                if format == .wgsl {
                    elementName = strideWrapperName(handle)
                    let wrapper = ["struct \(elementName) {", "    @size(\(stride)) saga_value: \(element.name),", "}"]
                    for line in wrapper { try budget.consume(.sourceBytes, amount: line.utf8.count + 1) }
                    declarations += wrapper
                }
            }
            switch size {
            case .constant(let count):
                shape = .init(name: "array<\(elementName), \(count)>", size: UInt64(count) * UInt64(stride),
                              alignment: element.alignment, containsWideInteger: element.containsWideInteger,
                              contains16Bit: element.contains16Bit)
            case .dynamic:
                shape = .init(name: "array<\(elementName)>", size: UInt64(stride),
                              alignment: element.alignment, runtime: true, containsWideInteger: element.containsWideInteger,
                              contains16Bit: element.contains16Bit)
            default: throw unsupported("unspecialized nested array size")
            }
        case .structure(let members, let span):
            let fields = try members.map { try describe($0.type, budget: &budget) }
            let name = prefix + String(handle.index)
            let naturalAlignment = fields.map(\.alignment).max() ?? 1
            var alignment = format == .wgsl ? zip(members, fields).map { member, field in
                max(member.explicitAlignment.map(UInt64.init) ?? 1, field.alignment)
            }.max() ?? 1 : naturalAlignment
            if format != .wgsl, let last = fields.last, last.runtime, let offset = members.last?.offset {
                let end = UInt64(offset) + last.size
                while aligned(end, to: alignment) < UInt64(span) { alignment *= 2 }
                guard aligned(end, to: alignment) == UInt64(span) else {
                    throw unsupported("nested runtime structure alignment")
                }
            }
            if format == .wgsl {
                var declaration = ["struct \(name) {"]
                var cursor: UInt64 = 0
                for (index, field) in fields.enumerated() {
                    let start = UInt64(members[index].offset)
                    let end = index + 1 < members.count ? UInt64(members[index + 1].offset) : UInt64(span)
                    let memberAlignment = max(field.alignment, members[index].explicitAlignment.map(UInt64.init) ?? 1)
                    guard start == aligned(cursor, to: memberAlignment), end >= start,
                          end - start >= field.size else {
                        throw unsupported("nested WGSL structure layout cannot be represented")
                    }
                    let size = end - start
                    let align = members[index].explicitAlignment.map { "@align(\($0)) " } ?? ""
                    let padding = !field.runtime && size > field.size ? "@size(\(size)) " : ""
                    declaration.append("    \(align)\(padding)saga_member_\(index): \(field.name),")
                    cursor = start + (field.runtime ? field.size : size)
                }
                guard aligned(cursor, to: alignment) == UInt64(span) else {
                    throw unsupported("nested WGSL structure span cannot be represented")
                }
                declaration.append("}")
                for line in declaration { try budget.consume(.sourceBytes, amount: line.utf8.count + 1) }
                declarations += declaration
            }
            shape = .init(name: name, size: UInt64(span), alignment: alignment,
                          containsWideInteger: fields.contains(where: \.containsWideInteger),
                          contains16Bit: fields.contains(where: \.contains16Bit))
        case .atomic(let scalar) where [.i32, .u32, .i64, .u64, .f32].contains(scalar):
            if scalar == .i64, format == .msl {
                throw unsupported("Metal storage min/max atomics require unsigned 64-bit values")
            }
            let lane = "\(scalar.kind == .sint ? "i" : "u")\(Int(scalar.width) * 8)"
            let name: String
            if format == .glsl {
                switch scalar {
                case .i64: name = signed64Type(in: .glsl)
                case .u64: name = unsigned64Type(in: .glsl)
                case .i32: name = "int"
                default: name = "uint"
                }
            } else {
                name = "atomic<\(lane)>"
            }
            shape = .init(name: name, size: UInt64(scalar.width), alignment: UInt64(scalar.width),
                          containsWideInteger: scalar.width == 8)
        case .scalar(let scalar), .vector(_, let scalar), .matrix(_, _, let scalar):
            guard [.i16, .u16, .i32, .u32, .i64, .u64, .f32, .f16].contains(scalar) else {
                throw unsupported("unsupported nested storage scalar")
            }
            let laneBytes = UInt64(scalar.width)
            let size: UInt64
            let alignment: UInt64
            switch try module.types.value(for: handle).kind {
            case .scalar: (size, alignment) = (laneBytes, laneBytes)
            case .vector(let lanes, _):
                (size, alignment) = (UInt64(lanes.rawValue) * laneBytes, (lanes == .two ? 2 : 4) * laneBytes)
            case .matrix(let columns, let rows, _):
                alignment = (rows == .two ? 2 : 4) * laneBytes
                size = UInt64(columns.rawValue) * alignment
            default: throw ShaderFailure.invalidHandle
            }
            let name = try shaderValueType(handle, module: module, format: format, structures: [:])
            shape = .init(name: name, size: size, alignment: alignment, containsWideInteger: scalar.width == 8,
                          contains16Bit: scalar.width == 2)
        default: throw unsupported("unsupported nested storage element")
        }
        known[handle] = shape
        return shape
    }

    mutating func glslStorageBlock(
        _ handle: IRHandle<IRType>, tailMember: Int?, budget: inout ShaderBudget
    ) throws -> GLSLNestedStorageBlock? {
        guard format == .glsl, let root = known[handle], root.containsWideInteger || !paddedArrays.isEmpty else { return nil }
        if case .structure(let members, _) = try module.types.value(for: handle).kind {
            let fields = try glslMembers(members, span: nil, budget: &budget)
            return .init(members: fields.joined(separator: " "), rootSuffix: "",
                         tailSuffix: tailMember.map { ".saga_member_\($0)" } ?? "", uses16BitStorage: root.contains16Bit)
        }
        let field = try glslField(handle, named: "saga_data", budget: &budget)
        return .init(members: field, rootSuffix: ".saga_data", tailSuffix: ".saga_data",
                     uses16BitStorage: root.contains16Bit)
    }

    private mutating func glslField(
        _ handle: IRHandle<IRType>, named name: String, budget: inout ShaderBudget
    ) throws -> String {
        try budget.consume(.work)
        try budget.enterDepth()
        defer { try? budget.leaveDepth() }
        var base = handle
        var suffix = ""
        while case .array(let element, let size, let stride) = try module.types.value(for: base).kind {
            try budget.consume(.work)
            switch size {
            case .constant(let count): suffix += "[\(count)]"
            case .dynamic: suffix += "[]"
            default: throw ShaderFailure.invalidHandle
            }
            if paddedArrays.contains(base) {
                let wrapper = strideWrapperName(base)
                if glslStructures.insert(base).inserted {
                    guard let shape = known[element] else { throw ShaderFailure.invalidHandle }
                    let value = try glslField(element, named: "saga_value", budget: &budget)
                    let padding = glslPadding(from: shape.size, to: UInt64(stride), alignment: shape.alignment, at: 0)
                    let declaration = ["struct \(wrapper) {", "    \(value)"] + padding.map { "    \($0)" } + ["};"]
                    for line in declaration { try budget.consume(.sourceBytes, amount: line.utf8.count + 1) }
                    declarations += declaration
                }
                return "\(wrapper) \(name)\(suffix);"
            }
            base = element
        }
        guard let shape = known[base] else { throw ShaderFailure.invalidHandle }
        if case .structure(let members, let span) = try module.types.value(for: base).kind,
           glslStructures.insert(base).inserted {
            let fields = try glslMembers(members, span: UInt64(span), budget: &budget)
            declarations += ["struct \(shape.name) {"] + fields.map { "    \($0)" } + ["};"]
        }
        return "\(shape.name) \(name)\(suffix);"
    }

    private mutating func glslMembers(
        _ members: [IRStructMember], span: UInt64?, budget: inout ShaderBudget
    ) throws -> [String] {
        let shapes = try members.map { member in
            guard let shape = known[member.type] else { throw ShaderFailure.invalidHandle }
            return shape
        }
        let alignment = shapes.map(\.alignment).max() ?? 1
        var fields: [String] = []
        var cursor: UInt64 = 0
        for (index, member) in members.enumerated() {
            let shape = shapes[index]
            let offset = UInt64(member.offset)
            let natural = aligned(cursor, to: shape.alignment)
            guard natural <= offset else { throw unsupported("nested std430 member offset") }
            if natural < offset {
                fields += glslPadding(from: cursor, to: offset, alignment: alignment, at: index)
            }
            fields.append(try glslField(member.type, named: "saga_member_\(index)", budget: &budget))
            cursor = offset + shape.size
        }
        if let span {
            let natural = aligned(cursor, to: alignment)
            guard natural <= span else { throw unsupported("nested std430 structure size") }
            if natural < span {
                fields += glslPadding(from: cursor, to: span, alignment: alignment, at: members.count)
            }
        }
        return fields
    }

    private func glslPadding(from start: UInt64, to end: UInt64, alignment: UInt64, at index: Int) -> [String] {
        // Preserve the struct's base alignment so its array stride stays identical to WGSL.
        let name = "saga_pad_\(index)"
        if alignment < 4 {
            return ["\(glslHalfwordPadding) \(name)[\((end - start) / 2)];"]
        }
        var fields: [String] = []
        let wordsStart = aligned(start, to: 4)
        if wordsStart > start {
            fields.append("\(glslHalfwordPadding) \(name)_head;")
        }
        let bytes = end - wordsStart
        if bytes >= 4 {
            fields.append("uint \(name)[\(bytes / 4)];")
        }
        if bytes % 4 != 0 {
            fields.append("\(glslHalfwordPadding) \(name)_tail;")
        }
        return fields
    }

    private func aligned(_ value: UInt64, to alignment: UInt64) -> UInt64 {
        (value + alignment - 1) / alignment * alignment
    }

    private func strideWrapperName(_ type: IRHandle<IRType>) -> String { "\(prefix)Stride\(type.index)" }
}

private func unsupported(_ message: String) -> ShaderFailure {
    .unsupported("Compute code generation: \(message)")
}
