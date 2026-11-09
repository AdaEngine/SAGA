import SAGAShaderIR

struct BufferArrayAddress {
    enum Target {
        case metal(pointerType: String, elements: [String], lengths: (name: String, slot: Int)?)
        case hlsl(lengthFunction: String?)
        case wgsl(tailMember: Int?, split: Bool)
        case glsl(tail: (offset: UInt32, stride: UInt32)?)
        case glslTyped(rootSuffix: String, tailSuffix: String?)
    }

    let count: UInt32
    let target: Target

    var splitsArray: Bool {
        switch target {
        case .wgsl(_, true), .glsl, .glslTyped: true
        default: false
        }
    }

    func select(_ index: String, in buffer: NestedBufferLayout) -> NestedBufferLayout {
        let reference: String
        let length: String?
        var glslReference: String?
        switch target {
        case .metal(let pointerType, let elements, let lengths):
            reference = "array<\(pointerType), \(count)>{\(elements.joined(separator: ", "))}[\(index)]"
            length = lengths.map { "\($0.name)[\($0.slot)u + \(index)]" }
        case .hlsl(let lengthFunction):
            let member = buffer.space == .uniform ? ".words" : ""
            reference = "\(buffer.name)[NonUniformResourceIndex(\(index))]\(member)"
            length = lengthFunction.map { "\($0)(\(index))" }
        case .wgsl(let tailMember, let split):
            reference = split ? "\(buffer.name)_\(index)" : "\(buffer.name)[\(index)]"
            length = tailMember.map { "arrayLength(&\(reference).saga_member_\($0))" }
        case .glsl(let tail):
            reference = "\(buffer.name)_\(index)"
            length = tail.map { "(max(uint(\(reference).words.length()) * 4u, \($0.offset)u) - \($0.offset)u) / \($0.stride)u" }
        case .glslTyped(let rootSuffix, let tailSuffix):
            reference = "\(buffer.name)_\(index)"
            length = tailSuffix.map { "uint(\(reference)\($0).length())" }
            glslReference = reference + rootSuffix
        }
        return .init(name: reference, type: buffer.type, runtimeLength: length, space: buffer.space,
                     glslReference: glslReference, wrappedArrays: buffer.wrappedArrays,
                     metalMemoryQualifier: buffer.metalMemoryQualifier)
    }
}

struct BufferBindingArray {
    let declarations: [String]
    let parameters: [String]
    let captures: [ShaderParameter]
    let storage: ShaderStorage
    let layout: ComputeShaderArtifact.BindingArrayLayout
    let minimumByteSize: UInt64
    let runtimeStride: UInt32?
    let runtimeOffset: UInt32
    let usesGLSL16BitStorage: Bool

    init(
        global: IRGlobalVariable, index: Int, flat: UInt32, module: IRModule, format: ShaderOutputFormat, count: UInt32,
        runtimeSlot: Int? = nil, metalLengths: ShaderParameter? = nil, webBindingBase: UInt32?, memoryQualifier: String = "",
        budget: inout ShaderBudget
    ) throws {
        guard let binding = global.binding,
              case .bindingArray(let element, _) = try module.types.value(for: global.type).kind
        else { throw ShaderFailure.invalidHandle }

        let name = "saga_buffer_\(index)"
        let writable: Bool
        if case .storage(let access) = global.space { writable = access.contains(.store) } else { writable = false }
        let uniform = global.space == .uniform
        var types = NestedBufferTypes(module: module, format: format, prefix: "SagaBufferArray\(index)_")
        let root = try types.describe(element, budget: &budget)
        guard !uniform || root.size <= 65_536 else { throw ShaderFailure.unsupported("Uniform buffer exceeds 64 KiB") }
        let tail = try NestedStorageBinding.runtimeTail(element, module: module)
        let glslBlock = try uniform ? nil : types.glslStorageBlock(element, tailMember: tail?.member, budget: &budget)
        usesGLSL16BitStorage = glslBlock?.uses16BitStorage == true
        let register = hlslBindingArrayRegisterBase(binding)
        let address: BufferArrayAddress.Target
        var declarations = types.declarations
        var parameters: [String] = []
        var captures: [ShaderParameter] = []

        switch format {
        case .msl:
            let pointer = memoryQualifier + (uniform ? "constant uint*" : writable ? "device uint*" : "device const uint*")
            for offset in 0..<count {
                let elementName = "\(name)_\(offset)"
                parameters.append("\(pointer) \(elementName) [[buffer(\(flat + offset))]]")
                captures.append(.init(type: pointer, name: elementName))
            }
            let elements = captures.map(\.name)
            let lengths: (name: String, slot: Int)?
            if tail != nil {
                guard let runtimeSlot, let metalLengths else { throw ShaderFailure.invalidHandle }
                captures.append(metalLengths)
                lengths = (metalLengths.name, runtimeSlot)
            } else {
                lengths = nil
            }
            address = .metal(pointerType: pointer, elements: elements, lengths: lengths)
        case .hlsl:
            if uniform {
                let type = "SagaUniformArray\(index)"
                declarations.append("struct \(type) { uint4 words[\((root.size + 15) / 16)]; };")
                declarations.append("ConstantBuffer<\(type)> \(name)[\(count)] : register(b\(register), space\(binding.group));")
            } else {
                let type = writable ? "RWByteAddressBuffer" : "ByteAddressBuffer"
                let space = writable ? "u" : "t"
                declarations.append("\(memoryQualifier)\(type) \(name)[\(count)] : register(\(space)\(register), space\(binding.group));")
            }
            let lengthFunction = tail.map { tail in
                let helper = "saga_length_\(index)"
                declarations += ["uint \(helper)(uint index) {", "    uint bytes;",
                    "    \(name)[NonUniformResourceIndex(index)].GetDimensions(bytes);",
                    "    return bytes < \(tail.offset)u ? 0u : (bytes - \(tail.offset)u) / \(tail.stride)u;", "}"]
                return helper
            }
            address = .hlsl(lengthFunction: lengthFunction)
        case .wgsl:
            let space = uniform ? "uniform" : writable ? "storage, read_write" : "storage, read"
            if let webBindingBase {
                declarations += (0..<count).map {
                    "@group(\(binding.group)) @binding(\(webBindingBase + $0)) \(memoryQualifier)var<\(space)> \(name)_\($0): \(root.name);"
                }
            } else {
                declarations.append("@group(\(binding.group)) @binding(\(binding.binding)) \(memoryQualifier)var<\(space)> \(name): binding_array<\(root.name), \(count)>;")
            }
            address = .wgsl(tailMember: tail?.member, split: webBindingBase != nil)
        case .glsl:
            if uniform {
                let registers = (root.size + 15) / 16
                declarations += (0..<count).map {
                    "layout(std140, binding = \(flat + $0)) uniform SagaUniformArray\(index)_\($0) { uvec4 words[\(registers)]; } \(name)_\($0);"
                }
            } else {
                let qualifier = memoryQualifier + (writable ? "buffer" : "readonly buffer")
                let members = glslBlock?.members ?? "uint words[];"
                declarations += (0..<count).map {
                    "layout(std430, binding = \(flat + $0)) \(qualifier) SagaStorageArray\(index)_\($0) { \(members) } \(name)_\($0);"
                }
            }
            if let glslBlock {
                address = .glslTyped(rootSuffix: glslBlock.rootSuffix, tailSuffix: tail.map { _ in glslBlock.tailSuffix })
            } else {
                address = .glsl(tail: tail.map { ($0.offset, $0.stride) })
            }
        default:
            throw ShaderFailure.unsupported("Buffer binding array target")
        }
        self.declarations = declarations
        self.parameters = parameters
        self.captures = captures
        self.minimumByteSize = root.size
        self.runtimeStride = tail?.stride
        self.runtimeOffset = tail?.offset ?? 0
        layout = .init(count: count, hlslRegisterBase: format == .hlsl ? register : nil, webBindingBase: webBindingBase)
        let bufferArray = BufferArrayAddress(count: count, target: address)
        storage = .init(reference: name, writable: writable,
                        nested: .init(name: name, type: element, runtimeLength: nil, space: global.space,
                                      wrappedArrays: format == .wgsl || glslBlock != nil ? types.paddedArrays : [],
                                      bindingArray: bufferArray, metalMemoryQualifier: format == .msl ? memoryQualifier : ""),
                        splitBindingArray: bufferArray.splitsArray)
    }
}
