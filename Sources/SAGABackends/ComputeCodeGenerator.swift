import SAGAShaderIR
import SAGAValidation

/// Initial compute writer. All unsupported IR fails closed before an artifact is returned.
package struct ComputeCodeGenerator: ShaderCodeGenerator {
    package let outputFormat: ShaderOutputFormat

    package init?(_ output: ShaderOutputFormat) {
        switch output {
        case .msl, .hlsl, .glsl, .wgsl: outputFormat = output
        default: return nil
        }
    }

    package func generate(_ validated: ValidatedModule<IRModule>) throws -> ShaderArtifact {
        try generateCompute(validated).artifact
    }

    package func generateCompute(_ validated: ValidatedModule<IRModule>) throws -> ComputeShaderArtifact {
        let module = validated.module
        guard module.entryPoints.count == 1, let entry = module.entryPoints.first,
              entry.stage == .compute, entry.workgroupOverrides == nil,
              module.overrides.count == 0
        else { throw unsupported("expected one compute entry point with no overrides") }
        let function = try module.functions.value(for: entry.function)
        guard function.result == nil else {
            throw unsupported("compute return values")
        }

        let size = entry.workgroupSize
        guard size.x > 0, size.y > 0, size.z > 0,
              size.x <= 1024, size.y <= 1024, size.z <= 64,
              size.x * size.y * size.z <= 1024
        else { throw unsupported("workgroup size exceeds the compute profile") }

        var inputs = try ComputeInputs(arguments: function.arguments, module: module, format: outputFormat)
        var helpers = try ShaderFunctions(module: module, entry: entry.function, captureGlobals: outputFormat == .msl)
        var declarations = inputs.declarations
        var structures = inputs.structures
        for entry in module.specialTypes.predeclared {
            guard case .atomicCompareExchangeWeakResult(let scalar) = entry.kind else {
                throw unsupported("predeclared type")
            }
            let signed = scalar == .i32
            let suffix = signed ? "I32" : "U32"
            let name = "SagaAtomicResult\(suffix)"
            let constructor = outputFormat == .hlsl ? "saga_make_atomic_\(suffix)" : name
            let structure = ShaderStructure(typeName: name, constructor: constructor)
            let lane = outputFormat == .wgsl ? (signed ? "i32" : "u32") : (signed ? "int" : "uint")
            declarations += structure.declarations(memberTypes: [lane, "bool"], in: outputFormat)
            structures[entry.type] = structure
        }
        var resourceParameters: [String] = []

        var storage: [ShaderStorage] = []
        var metalStorage: [[ShaderParameter]] = []
        var layout: [ComputeShaderArtifact.StorageBinding] = []
        var uniformLayout: [ComputeShaderArtifact.UniformBinding] = []
        var sampledTextures: [ComputeShaderArtifact.SampledTextureBinding] = []
        var storageTextures: [ComputeShaderArtifact.StorageTextureBinding] = []
        var samplers: [ComputeShaderArtifact.SamplerBinding] = []
        let flatBindings = try flatResourceBindingIndices(
            module.globalVariables.values.filter { $0.space != .workgroup && $0.space != .uniform && $0.space != .handle },
            capacity: ComputeShaderArtifact.storageBindingCapacity)
        let flatUniforms = try flatResourceBindingIndices(
            module.globalVariables.values.filter { $0.space == .uniform },
            capacity: ComputeShaderArtifact.uniformBindingCapacity)
        let globals = module.globalVariables.values
        let handles = try handleResourceGlobals(globals, module: module)
        let flatTextures = try flatResourceBindingIndices(handles.textures,
            capacity: ComputeShaderArtifact.sampledTextureBindingCapacity)
        let flatStorageTextures = try flatResourceBindingIndices(handles.storageTextures,
            capacity: ComputeShaderArtifact.storageTextureBindingCapacity)
        let flatSamplers = try flatResourceBindingIndices(handles.samplers,
            capacity: ComputeShaderArtifact.samplerBindingCapacity)
        let nonFilteringSamplers = try helpers.nonFilteringSamplers(
            module: module, entry: entry.function, sampledTextures: handles.textures)
        let glslBindings = outputFormat == .glsl
            ? try helpers.glslTextureSamplerBindings(module: module, entry: entry.function, textureSlots: Set(flatTextures.values))
            : []
        declarations += glslBindings.map { $0.declaration }
        let usesFlatBindings = outputFormat == .msl || outputFormat == .glsl
        let metalLengths = ShaderParameter(type: "constant uint*", name: "saga_lengths")
        var runtimeIndex = 0
        var workgroupVariables: [WorkgroupVariable] = []
        var workgroupBytes: UInt64 = 0
        for (index, global) in module.globalVariables.values.enumerated() {
            if global.space == .workgroup {
                let field = try ComputeStorageField(type: global.type, offset: 0, module: module)
                guard !field.runtime, global.memoryDecorations.isEmpty else {
                    throw unsupported("expected a fixed workgroup variable")
                }
                workgroupBytes += (field.minimumByteSize + 15) / 16 * 16
                guard workgroupBytes <= 16 * 1024 else {
                    throw unsupported("workgroup storage exceeds 16 KiB")
                }
                let variable = WorkgroupVariable(name: "saga_shared_\(index)", field: field, format: outputFormat,
                                                 type: try field.valueType(in: outputFormat, module: module))
                workgroupVariables.append(variable)
                if outputFormat != .msl { declarations.append(variable.declaration) }
                storage.append(.init(reference: variable.reference, bound: field.count.map { "\($0)u" }, writable: true))
                if outputFormat == .msl {
                    metalStorage.append([ShaderParameter(type: "threadgroup \(variable.type)*", name: variable.name)])
                }
                continue
            }
            if global.space == .uniform {
                guard let binding = global.binding, let flat = flatUniforms[binding] else {
                    throw unsupported("invalid compute uniform binding")
                }
                let uniform = try UniformShaderBinding(global: global, index: index, flat: flat, module: module, format: outputFormat)
                declarations += uniform.declarations
                resourceParameters += uniform.parameters
                storage.append(uniform.storage)
                metalStorage.append(uniform.captures)
                uniformLayout.append(uniform.reflection)
                continue
            }
            if global.space == .handle, let binding = global.binding, let flat = flatSamplers[binding] {
                let sampler = try SamplerShaderBinding(global: global, index: index, flat: flat, module: module,
                                                      format: outputFormat, requiresNonFiltering: nonFilteringSamplers.contains(index))
                declarations += sampler.declarations
                resourceParameters += sampler.parameters
                storage.append(sampler.storage)
                metalStorage.append(sampler.captures)
                samplers.append(sampler.reflection)
                continue
            }
            if global.space == .handle, let binding = global.binding, let flat = flatStorageTextures[binding] {
                let texture = try StorageTextureShaderBinding(
                    global: global, index: index, flat: flat, module: module, format: outputFormat,
                    coherent: helpers.controlBarriers.contains(.texture))
                declarations += texture.declarations
                resourceParameters += texture.parameters
                storage.append(texture.storage)
                metalStorage.append(texture.captures)
                storageTextures.append(texture.reflection)
                continue
            }
            if global.space == .handle {
                guard let binding = global.binding, let flat = flatTextures[binding] else {
                    throw unsupported("invalid compute sampled texture binding")
                }
                let texture = try SampledTextureShaderBinding(global: global, index: index, flat: flat,
                                                              module: module, format: outputFormat)
                declarations += texture.declarations
                resourceParameters += texture.parameters
                storage.append(texture.storage)
                metalStorage.append(texture.captures)
                sampledTextures.append(texture.reflection)
                continue
            }
            let writable: Bool
            switch global.space {
            case .storage(.load): writable = false
            case .storage([.load, .store]): writable = true
            default: throw unsupported("expected a uniform or storage buffer binding")
            }
            guard let binding = global.binding,
                  let flatBinding = flatBindings[binding],
                  global.initializer == nil, global.memoryDecorations.isEmpty
            else { throw unsupported("invalid compute buffer binding") }
            let structureSpan: UInt32?
            let fields: [ComputeStorageField]
            if case .structure(let members, let span) = try module.types.value(for: global.type).kind {
                structureSpan = span
                fields = try members.map { try ComputeStorageField(type: $0.type, offset: $0.offset, module: module) }
            } else {
                structureSpan = nil
                fields = [try ComputeStorageField(type: global.type, offset: 0, module: module)]
            }
            guard let tail = fields.last else { throw unsupported("empty storage structure") }
            let structured = structureSpan != nil
            let rawHLSL = outputFormat == .hlsl && structured && fields.contains(where: \.atomic)
            let name = "saga_buffer_\(index)"
            let typeName = "SagaBuffer\(index)"
            let minimum = structureSpan.map(UInt64.init) ?? tail.minimumByteSize
            let fieldNames = structured ? fields.indices.map { "saga_member_\($0)" } : ["value"]
            let runtimeLength: String?
            switch outputFormat {
            case .msl:
                let qualifier = writable ? "device" : "device const"
                let element = tail.atomic && !structured ? tail.metalAtomicType : "uint"
                let pointer = ShaderParameter(type: "\(qualifier) \(element)*", name: name)
                resourceParameters.append("\(pointer.type) \(name) [[buffer(\(flatBinding))]]")
                metalStorage.append(tail.runtime ? [pointer, metalLengths] : [pointer])
                runtimeLength = tail.runtime ? "\(metalLengths.name)[\(runtimeIndex)]" : nil
            case .hlsl:
                let register = writable ? "u" : "t"
                let type: String
                if rawHLSL {
                    type = "RWByteAddressBuffer"
                } else {
                    let buffer = writable ? "RWStructuredBuffer" : "StructuredBuffer"
                    let element = tail.atomic ? try tail.valueType(in: .hlsl, module: module) : "uint"
                    type = "\(buffer)<\(element)>"
                }
                declarations.append("\(type) \(name) : register(\(register)\(binding.binding), space\(binding.group));")
                if tail.runtime {
                    let helper = "saga_length_\(index)"
                    let unit: UInt32 = rawHLSL ? 1 : 4
                    let offset = tail.offset / unit
                    let stride = tail.stride / unit
                    let elements = offset == 0 ? "count" : "(count - \(offset)u)"
                    let count = stride == 1 ? elements : "\(elements) / \(stride)u"
                    declarations += [
                        "uint \(helper)() {",
                        rawHLSL ? "    uint count;" : "    uint count, stride;",
                        "    \(name).GetDimensions(\(rawHLSL ? "count" : "count, stride"));",
                    ]
                    if offset > 0 { declarations.append("    if (count < \(offset)u) { return 0u; }") }
                    declarations += ["    return \(count);", "}"]
                    runtimeLength = "\(helper)()"
                } else {
                    runtimeLength = nil
                }
            case .wgsl:
                let access = writable ? "read_write" : "read"
                let type = try structured ? typeName : tail.valueType(in: .wgsl, module: module)
                if let structureSpan {
                    let firstAlignment: String
                    if tail.runtime {
                        let natural = fields.map(\.alignment).max() ?? 4
                        let end = UInt64(tail.offset) + tail.minimumByteSize
                        var align = UInt64(natural)
                        func rounded() -> UInt64 { (end + align - 1) / align * align }
                        while rounded() < UInt64(structureSpan) { align *= 2 }
                        guard rounded() == UInt64(structureSpan) else { throw unsupported("runtime structure alignment") }
                        firstAlignment = align > UInt64(natural) ? "@align(\(align)) " : ""
                    } else { firstAlignment = "" }
                    declarations.append("struct \(typeName) {")
                    for (position, field) in fields.enumerated() {
                        var padding = ""
                        let end = position + 1 < fields.count ? fields[position + 1].offset :
                            (field.runtime ? nil : structureSpan)
                        if let end {
                            let memberSize = end - field.offset
                            if UInt64(memberSize) > field.minimumByteSize { padding = "@size(\(memberSize)) " }
                        }
                        let type = try field.valueType(in: .wgsl, module: module)
                        let align = position == 0 ? firstAlignment : ""
                        declarations.append("    \(align)\(padding)\(fieldNames[position]): \(type),")
                    }
                    declarations.append("}")
                }
                declarations.append("@group(\(binding.group)) @binding(\(binding.binding)) var<storage, \(access)> \(name): \(type);")
                let array = structured ? "\(name).\(fieldNames.last!)" : name
                runtimeLength = tail.runtime ? "arrayLength(&\(array))" : nil
            default:
                let coherence = helpers.controlBarriers.contains(.storage) && writable ? "coherent " : ""
                let qualifier = coherence + (writable ? "buffer" : "readonly buffer")
                let members = try fields.enumerated().map { position, field in
                    let offset = structured ? "layout(offset = \(field.offset)) " : ""
                    let type = try field.valueType(in: .glsl, module: module)
                    return "\(offset)\(type) \(fieldNames[position])\(field.glslSuffix);"
                }.joined(separator: " ")
                declarations.append("layout(std430, binding = \(flatBinding)) \(qualifier) \(typeName) { \(members) } \(name);")
                runtimeLength = tail.runtime ? "uint(\(name).\(fieldNames.last!).length())" : nil
            }
            let locations = fields.enumerated().map { position, field in
                let reference: String
                var words: ShaderStorage.WordLayout?
                switch outputFormat {
                case .hlsl where rawHLSL:
                    reference = name
                    words = field.wordLayout(addressing: .byteAddressed)
                case .msl, .hlsl:
                    if field.atomic {
                        let pointer = structured ? "reinterpret_cast<device \(field.metalAtomicType)*>(\(name) + \(field.offset / 4)u)" : name
                        reference = field.isArray ? pointer : "\(pointer)[0]"
                    } else if field.isArray || field.width > 1 || field.scalar != .u32 {
                        reference = name
                        words = field.wordLayout()
                    } else if outputFormat == .msl && !structured {
                        reference = "*\(name)"
                    } else {
                        reference = "\(name)[\(field.offset / 4)]"
                    }
                case .wgsl where !structured:
                    reference = name
                default:
                    reference = "\(name).\(fieldNames[position])"
                }
                let bound = field.runtime ? runtimeLength : field.count.map { "\($0)u" }
                return ShaderStorage(reference: reference, bound: bound, writable: writable, words: words)
            }
            storage.append(structured ? ShaderStorage(reference: name, writable: writable, members: locations) : locations[0])
            let targetIndex = usesFlatBindings ? flatBinding : nil
            layout.append(.init(binding: binding, flatBindingIndex: targetIndex, minimumByteSize: minimum,
                                isReadOnly: !writable, runtimeArrayStride: tail.runtime ? tail.stride : nil,
                                runtimeArrayOffset: tail.runtime ? tail.offset : 0,
                                hlslUsesByteAddressBuffer: rawHLSL))
            if tail.runtime { runtimeIndex += 1 }
        }

        var workgroupSetup: [String] = []
        if outputFormat != .wgsl && !workgroupVariables.isEmpty {
            let lane = inputs.workgroupInitializationIndex()
            if outputFormat == .msl { workgroupSetup = workgroupVariables.map(\.declaration) }
            let threads = size.x * size.y * size.z
            workgroupSetup += workgroupVariables.flatMap { $0.initialize(lane: lane, threads: threads) }
            workgroupSetup += try shaderControlBarrier(.workgroup, in: outputFormat).map { "    \($0)" }
        }
        var parameters = inputs.parameters + resourceParameters
        let lengthBuffer = outputFormat == .msl && runtimeIndex > 0 ? ComputeShaderArtifact.storageBindingCapacity : nil
        if let lengthBuffer {
            parameters.append("\(metalLengths.type) \(metalLengths.name) [[buffer(\(lengthBuffer))]]")
        }
        let parameterList = parameters.joined(separator: ", ")
        let entryPoint = outputFormat == .glsl ? "main" : "saga_main"
        if outputFormat == .msl { try helpers.bindMetalStorage(metalStorage) }
        let helperDeclarations = try helpers.declarations(module: module, format: outputFormat,
                                                          storage: storage, structures: structures)
        let signature: String
        let attributes: [String]
        switch outputFormat {
        case .msl:
            declarations.insert(contentsOf: ["#include <metal_stdlib>", "using namespace metal;"], at: 0)
            attributes = []
            signature = "kernel void \(entryPoint)(\(parameterList))"
        case .hlsl:
            attributes = ["[numthreads(\(size.x), \(size.y), \(size.z))]"]
            signature = "void \(entryPoint)(\(parameterList))"
        case .wgsl:
            attributes = ["@compute @workgroup_size(\(size.x), \(size.y), \(size.z))"]
            signature = "fn \(entryPoint)(\(parameterList))"
        default:
            declarations.insert(contentsOf: [
                "#version 450 core",
            ] + helpers.glslExtensions + [
                "layout(local_size_x = \(size.x), local_size_y = \(size.y), local_size_z = \(size.z)) in;",
            ], at: 0)
            attributes = []
            signature = "void \(entryPoint)()"
        }

        var body = ShaderBodyWriter(module: module, function: function, arguments: inputs.references,
                                    storage: storage, outputFormat: outputFormat,
                                    structures: structures,
                                    callCaptures: helpers.callCaptures, budget: helpers.budget)
        let lines = try body.write()
        let numeric = try integerPrelude(helpers.integerHelpers.union(body.integerHelpers),
                                         vectors: helpers.integerVectorHelpers.union(body.integerVectorHelpers),
                                         in: outputFormat)
        let source = (declarations + numeric + helperDeclarations + attributes + [signature + " {"] +
                      workgroupSetup + inputs.initializers + lines + ["}", ""])
            .joined(separator: "\n")
        return .init(
            artifact: .text(format: outputFormat, source: source), sourceEntryPoint: entry.name,
            entryPoint: entryPoint, workgroupSize: size, storageBindings: layout, uniformBindings: uniformLayout,
            sampledTextures: sampledTextures, storageTextures: storageTextures, samplerBindings: samplers,
            glslTextureSamplerBindings: glslBindings.map { $0.reflection },
            workgroupStorageByteSize: workgroupBytes, metalRuntimeLengthBuffer: lengthBuffer,
            hlslWorkgroupCountBinding: inputs.hlslWorkgroupCountBinding,
            usesTextureBarrier: helpers.controlBarriers.contains(.texture))
    }
}

private struct WorkgroupVariable {
    let name: String
    let field: ComputeStorageField
    let format: ShaderOutputFormat
    let type: String

    var declaration: String {
        switch format {
        case .msl: return "    threadgroup \(type) \(name)[\(field.count ?? 1)];"
        case .hlsl: return "groupshared \(type) \(name)\(field.glslSuffix);"
        case .glsl: return "shared \(type) \(name)\(field.glslSuffix);"
        default: return "var<workgroup> \(name): \(type);"
        }
    }

    var reference: String {
        format == .msl && !field.isArray ? "\(name)[0]" : name
    }

    func initialize(lane: String, threads: UInt32) -> [String] {
        let zero = field.scalar == .u32 ? "0u" : field.scalar == .i32 ? "0" : format == .glsl ? "0.0" : "0.0f"
        let components = Array(repeating: zero, count: Int(field.width)).joined(separator: ", ")
        let value: String
        if let columns = field.columns {
            let column = "\(floatType(field.width, in: format))(\(components))"
            value = "\(type)(\(Array(repeating: column, count: Int(columns)).joined(separator: ", ")))"
        } else {
            value = field.width == 1 ? zero : "\(type)(\(components))"
        }
        func store(_ location: String) -> String {
            if field.atomic && format == .msl {
                return "atomic_store_explicit(&\(location), \(value), memory_order_relaxed);"
            }
            return "\(location) = \(value);"
        }
        guard let count = field.count else {
            return ["    if (\(lane) == 0u) { \(store(reference)) }"]
        }
        return [
            "    for (uint saga_init = \(lane); saga_init < \(count)u; saga_init += \(threads)u) {",
            "        \(store("\(name)[saga_init]"))",
            "    }",
        ]
    }
}

func shaderControlBarrier(_ flags: IRBarrier, in format: ShaderOutputFormat) throws -> [String] {
    guard !flags.isEmpty, flags.subtracting([.workgroup, .storage, .texture]).isEmpty else {
        throw unsupported("unsupported control barrier memory scope")
    }
    let shared = flags.contains(.workgroup)
    let storage = flags.contains(.storage)
    let texture = flags.contains(.texture)
    switch format {
    case .msl:
        let memory = [shared ? "mem_flags::mem_threadgroup" : nil,
                      storage ? "mem_flags::mem_device" : nil,
                      texture ? "mem_flags::mem_texture" : nil]
            .compactMap { $0 }.joined(separator: " | ")
        return ["threadgroup_barrier(\(memory));"]
    case .hlsl:
        if shared && (storage || texture) { return ["AllMemoryBarrierWithGroupSync();"] }
        return [shared ? "GroupMemoryBarrierWithGroupSync();" : "DeviceMemoryBarrierWithGroupSync();"]
    case .glsl:
        return [shared ? "memoryBarrierShared();" : nil, storage ? "memoryBarrierBuffer();" : nil,
                texture ? "memoryBarrierImage();" : nil, "barrier();"].compactMap { $0 }
    default:
        return [shared ? "workgroupBarrier();" : nil, storage ? "storageBarrier();" : nil,
                texture ? "textureBarrier();" : nil].compactMap { $0 }
    }
}

func flatResourceBindingIndices(
    _ globals: [IRGlobalVariable], capacity: UInt32
) throws -> [IRResourceBinding: UInt32] {
    guard globals.count <= Int(capacity) else { throw unsupported("resource binding count exceeds \(capacity)") }
    let bindings = globals.compactMap(\.binding)
    guard bindings.count == globals.count, Set(bindings).count == bindings.count,
          bindings.allSatisfy({ $0.group < ComputeShaderArtifact.portableBindGroupCount && $0.binding < 1_000 })
    else { throw unsupported("expected unique bindings in groups 0..<4 and binding indices 0..<1000") }

    // Keep existing low group-zero indices, then fill free slots in logical binding order.
    let legacy = bindings.filter { $0.group == 0 && $0.binding < capacity }
    var indices = Dictionary(uniqueKeysWithValues: legacy.map { ($0, $0.binding) })
    let occupied = Set(indices.values)
    let remaining = bindings.filter { indices[$0] == nil }.sorted {
        ($0.group, $0.binding) < ($1.group, $1.binding)
    }
    for (binding, slot) in zip(remaining, (0..<capacity).filter { !occupied.contains($0) }) {
        indices[binding] = slot
    }
    return indices
}

func handleResourceGlobals(_ globals: [IRGlobalVariable], module: IRModule) throws
    -> (textures: [IRGlobalVariable], samplers: [IRGlobalVariable], storageTextures: [IRGlobalVariable])
{
    var textures: [IRGlobalVariable] = []
    var samplers: [IRGlobalVariable] = []
    var storageTextures: [IRGlobalVariable] = []
    for global in globals where global.space == .handle {
        switch try module.types.value(for: global.type).kind {
        case let kind where kind.isSampledTexture: textures.append(global)
        case let kind where kind.storageTexture2D != nil: storageTextures.append(global)
        case .sampler: samplers.append(global)
        default: throw ShaderFailure.unsupported("Unsupported texture or sampler binding")
        }
    }
    return (textures, samplers, storageTextures)
}

struct SamplerShaderBinding {
    var declarations: [String] = []
    var parameters: [String] = []
    var captures: [ShaderParameter] = []
    let storage: ShaderStorage
    let reflection: ComputeShaderArtifact.SamplerBinding

    init(global: IRGlobalVariable, index: Int, flat: UInt32, module: IRModule, format: ShaderOutputFormat,
         requiresNonFiltering: Bool) throws {
        guard let binding = global.binding,
              case .sampler(let comparison) = try module.types.value(for: global.type).kind else {
            throw unsupported("invalid sampler binding")
        }
        let name = "saga_sampler_\(index)"
        switch format {
        case .msl:
            parameters.append("sampler \(name) [[sampler(\(flat))]]")
            captures.append(.init(type: "sampler", name: name))
        case .wgsl:
            let type = comparison ? "sampler_comparison" : "sampler"
            declarations.append("@group(\(binding.group)) @binding(\(binding.binding)) var \(name): \(type);")
        case .hlsl:
            let type = comparison ? "SamplerComparisonState" : "SamplerState"
            declarations.append("\(type) \(name) : register(s\(binding.binding), space\(binding.group));")
        default: break
        }
        storage = .init(reference: name, writable: false)
        reflection = .init(binding: binding, flatBindingIndex: format == .msl ? flat : nil, isComparison: comparison,
                           requiresNonFiltering: requiresNonFiltering)
    }
}

struct SampledTextureShaderBinding {
    var declarations: [String] = []
    var parameters: [String] = []
    var captures: [ShaderParameter] = []
    let storage: ShaderStorage
    let reflection: ComputeShaderArtifact.SampledTextureBinding

    init(global: IRGlobalVariable, index: Int, flat: UInt32, module: IRModule, format: ShaderOutputFormat) throws {
        let kind = try module.types.value(for: global.type).kind
        guard let scalar = kind.sampledTextureScalar, let binding = global.binding else {
            throw unsupported("invalid sampled texture binding")
        }
        let name = "saga_texture_\(index)"
        let depth = kind.isDepthTexture
        let multisampled = kind.isMultisampled2D
        let arrayed = kind.isArrayedTexture
        let cube = kind.isCubeTexture
        let volume = kind.isVolumeTexture
        let nativeScalar = scalar == .i32 ? "int" : scalar == .u32 ? "uint" : "float"
        switch format {
        case .msl:
            let shape = (depth ? "depth" : "texture") + (cube ? "cube" : volume ? "3d" : "2d") +
                (multisampled ? "_ms" : arrayed ? "_array" : "")
            let access = multisampled ? "read" : "sample"
            let type = "\(shape)<\(nativeScalar), access::\(access)>"
            let parameter = ShaderParameter(type: type, name: name)
            parameters.append("\(parameter.type) \(name) [[texture(\(flat))]]")
            captures.append(parameter)
        case .hlsl:
            let type = depth ? nativeScalar : "\(nativeScalar)4"
            let shape = (cube ? "TextureCube" : volume ? "Texture3D" : "Texture2D") +
                (multisampled ? "MS" : arrayed ? "Array" : "")
            declarations.append("\(shape)<\(type)> \(name) : register(t\(binding.binding), space\(binding.group));")
        case .glsl:
            let prefix = scalar == .i32 ? "i" : scalar == .u32 ? "u" : ""
            let type = prefix + (cube ? "samplerCube" : volume ? "sampler3D" : "sampler2D") +
                (multisampled ? "MS" : arrayed ? "Array" : "")
            declarations.append("layout(binding=\(flat)) uniform \(type) \(name);")
        case .wgsl:
            let shape = (multisampled ? "_multisampled_2d" : cube ? "_cube" : volume ? "_3d" : "_2d") + (arrayed ? "_array" : "")
            let element = scalar == .i32 ? "i32" : scalar == .u32 ? "u32" : "f32"
            let type = depth ? "texture_depth\(shape)" : "texture\(shape)<\(element)>"
            declarations.append("@group(\(binding.group)) @binding(\(binding.binding)) var \(name): \(type);")
        default: throw unsupported("sampled texture target")
        }
        storage = .init(reference: name, writable: false)
        reflection = .init(binding: binding, flatBindingIndex: format == .msl || format == .glsl ? flat : nil,
                           isDepth: depth, isMultisampled: multisampled, isArrayed: arrayed, sampleKind: scalar.kind,
                           dimension: cube ? .cube : volume ? .d3 : .d2)
    }
}

struct StorageTextureShaderBinding {
    var declarations: [String] = []
    var parameters: [String] = []
    var captures: [ShaderParameter] = []
    let storage: ShaderStorage
    let reflection: ComputeShaderArtifact.StorageTextureBinding

    init(global: IRGlobalVariable, index: Int, flat: UInt32, module: IRModule, format: ShaderOutputFormat,
         coherent: Bool = false) throws {
        let kind = try module.types.value(for: global.type).kind
        guard let binding = global.binding, let texture = kind.storageTexture2D else {
            throw unsupported("storage texture type")
        }
        let arrayed = kind.isArrayedTexture
        let name = "saga_texture_\(index)"
        let scalar = storageTextureScalarName(texture.scalar)
        let access: String
        let qualifier: String
        switch texture.access {
        case .load: (access, qualifier) = ("read", "readonly ")
        case .store: (access, qualifier) = ("write", "writeonly ")
        default: (access, qualifier) = ("read_write", "")
        }
        let emittedIndex: UInt32?
        switch format {
        case .msl:
            let slot = ComputeShaderArtifact.sampledTextureBindingCapacity + flat
            let shape = arrayed ? "texture2d_array" : "texture2d"
            let parameter = ShaderParameter(type: "\(shape)<\(scalar), access::\(access)>", name: name)
            parameters.append("\(parameter.type) \(name) [[texture(\(slot))]]")
            captures.append(parameter)
            emittedIndex = slot
        case .hlsl:
            let normalization = switch texture.format {
            case .rgba8Unorm: "unorm "
            case .rgba8Snorm: "snorm "
            default: ""
            }
            let element = texture.components == 1 ? scalar : "\(scalar)\(texture.components)"
            let shape = arrayed ? "RWTexture2DArray" : "RWTexture2D"
            declarations.append("\(shape)<\(normalization)\(element)> \(name) : register(u\(binding.binding), space\(binding.group));")
            emittedIndex = nil
        case .glsl:
            let layout = try glslStorageTextureLayout(texture.format)
            let prefix = texture.scalar == .u32 ? "u" : texture.scalar == .i32 ? "i" : ""
            let type = prefix + "image2D" + (arrayed ? "Array" : "")
            let memory = coherent && texture.access.contains(.store) ? "coherent " : ""
            declarations.append("layout(\(layout), binding=\(flat)) \(memory)\(qualifier)uniform \(type) \(name);")
            emittedIndex = flat
        case .wgsl:
            let layout = texture.format.rawValue.lowercased()
            let shape = arrayed ? "texture_storage_2d_array" : "texture_storage_2d"
            declarations.append("@group(\(binding.group)) @binding(\(binding.binding)) var \(name): \(shape)<\(layout), \(access)>;")
            emittedIndex = nil
        default: throw unsupported("storage texture target")
        }
        storage = .init(reference: name, writable: texture.access.contains(.store))
        reflection = .init(binding: binding, format: texture.format, access: texture.access,
                           flatBindingIndex: emittedIndex, isArrayed: arrayed)
    }
}

private func storageTextureScalarName(_ scalar: IRScalar) -> String {
    switch scalar {
    case .u32: "uint"
    case .i32: "int"
    default: "float"
    }
}

private func glslStorageTextureLayout(_ format: IRStorageFormat) throws -> String {
    switch format {
    case .rgba8Unorm: "rgba8"
    case .rgba8Snorm: "rgba8_snorm"
    case .rgba8Uint: "rgba8ui"
    case .rgba8Sint: "rgba8i"
    case .rgba16Uint: "rgba16ui"
    case .rgba16Sint: "rgba16i"
    case .rgba16Float: "rgba16f"
    case .r32Uint: "r32ui"
    case .r32Sint: "r32i"
    case .r32Float: "r32f"
    case .rg32Uint: "rg32ui"
    case .rg32Sint: "rg32i"
    case .rg32Float: "rg32f"
    case .rgba32Uint: "rgba32ui"
    case .rgba32Sint: "rgba32i"
    case .rgba32Float: "rgba32f"
    default: throw unsupported("GLSL storage texture layout")
    }
}

struct UniformShaderBinding {
    var declarations: [String] = []
    var parameters: [String] = []
    var captures: [ShaderParameter] = []
    let storage: ShaderStorage
    let reflection: ComputeShaderArtifact.UniformBinding

    init(global: IRGlobalVariable, index: Int, flat: UInt32, module: IRModule, format: ShaderOutputFormat) throws {
        guard global.space == .uniform, let binding = global.binding else { throw unsupported("invalid uniform buffer") }
        let span: UInt32?
        let fields: [ComputeStorageField]
        if case .structure(let members, let size) = try module.types.value(for: global.type).kind {
            span = size
            fields = try members.map { try ComputeStorageField(type: $0.type, offset: $0.offset, module: module) }
        } else {
            span = nil
            fields = [try ComputeStorageField(type: global.type, offset: 0, module: module)]
        }
        guard let first = fields.first, fields.allSatisfy({ !$0.atomic && !$0.runtime }) else {
            throw unsupported("expected a fixed uniform buffer")
        }
        let minimum = span.map(UInt64.init) ?? first.minimumByteSize
        guard minimum <= 65_536 else { throw unsupported("uniform buffer exceeds 64 KiB") }
        let flatBinding = ComputeShaderArtifact.uniformBindingBase + flat
        let name = "saga_buffer_\(index)"
        let type = "SagaUniform\(index)"
        let names = span == nil ? ["value"] : fields.indices.map { "saga_member_\($0)" }
        // Two-row WGSL matrices pack columns at eight bytes, while std140 requires sixteen.
        let packedGLSL = format == .glsl && fields.contains { $0.columns != nil && $0.width == 2 }
        switch format {
        case .msl:
            captures = [.init(type: "constant uint*", name: name)]
            parameters = ["constant uint* \(name) [[buffer(\(flatBinding))]]"]
        case .hlsl:
            declarations = [
                "cbuffer \(type) : register(b\(binding.binding), space\(binding.group)) {",
                "    uint4 \(name)[\((minimum + 15) / 16)];", "};",
            ]
        case .wgsl:
            if let span {
                declarations.append("struct \(type) {")
                for (position, field) in fields.enumerated() {
                    let end = position + 1 < fields.count ? fields[position + 1].offset : span
                    let size = end - field.offset
                    let padding = UInt64(size) > field.minimumByteSize ? "@size(\(size)) " : ""
                    declarations.append("    \(padding)\(names[position]): \(try field.valueType(in: format, module: module)),")
                }
                declarations.append("}")
            }
            let valueType = try span == nil ? first.valueType(in: format, module: module) : type
            declarations.append("@group(\(binding.group)) @binding(\(binding.binding)) var<uniform> \(name): \(valueType);")
        default:
            let members: String
            if packedGLSL {
                members = "uvec4 words[\((minimum + 15) / 16)];"
            } else {
                members = try fields.enumerated().map { position, field in
                    "layout(offset = \(field.offset)) \(try field.valueType(in: format, module: module)) \(names[position])\(field.glslSuffix);"
                }.joined(separator: " ")
            }
            declarations = ["layout(std140, binding = \(flatBinding)) uniform \(type) { \(members) } \(name);"]
        }
        let locations = fields.enumerated().map { position, field in
            let bound = field.count.map { "\($0)u" }
            if format == .msl || format == .hlsl || packedGLSL {
                let reference = packedGLSL ? "\(name).words" : name
                return ShaderStorage(reference: reference, bound: bound, writable: false,
                                     words: field.wordLayout(addressing: format == .msl ? .structured : .uniformRegisters))
            }
            let reference = format == .wgsl && span == nil ? name : "\(name).\(names[position])"
            return ShaderStorage(reference: reference, bound: bound, writable: false)
        }
        storage = span == nil ? locations[0] : .init(reference: name, writable: false, members: locations)
        reflection = .init(binding: binding, flatBindingIndex: format == .msl || format == .glsl ? flatBinding : nil,
                           minimumByteSize: minimum)
    }
}

private struct ComputeStorageField {
    let offset: UInt32
    let element: IRHandle<IRType>
    let scalar: IRScalar
    let width: UInt8
    let columns: UInt8?
    let stride: UInt32
    let count: UInt32?
    let runtime: Bool
    let atomic: Bool

    init(type: IRHandle<IRType>, offset: UInt32, module: IRModule) throws {
        self.offset = offset
        let arrayStride: UInt32?
        if case .array(let base, let size, let bytes) = try module.types.value(for: type).kind {
            element = base
            arrayStride = bytes
            switch size {
            case .constant(let length) where length > 0 && length <= 32 * 1024 * 1024:
                count = length
                runtime = false
            case .dynamic:
                count = nil
                runtime = true
            default:
                throw unsupported("storage array size")
            }
        } else {
            element = type
            arrayStride = nil
            count = nil
            runtime = false
        }
        switch try module.types.value(for: element).kind {
        case .scalar(let lane):
            scalar = lane
            width = 1
            columns = nil
            atomic = false
        case .atomic(let lane):
            scalar = lane
            width = 1
            columns = nil
            atomic = true
        case .vector(let size, let lane):
            scalar = lane
            width = size.rawValue
            columns = nil
            atomic = false
        case .matrix(let count, let rows, .f32):
            scalar = .f32
            width = rows.rawValue
            columns = count.rawValue
            atomic = false
        default:
            throw unsupported("expected a supported scalar, vector or matrix buffer element")
        }
        guard scalar == .u32 || scalar == .i32 || scalar == .f32 else {
            throw unsupported("storage scalar type")
        }
        let columnSize = UInt32(width == 3 ? 4 : width) * 4
        let byteSize = columns.map { UInt32($0) * columnSize } ?? UInt32(width) * 4
        let naturalStride = max(byteSize, columnSize)
        guard arrayStride == nil || arrayStride == naturalStride else {
            throw unsupported("nonstandard storage array stride")
        }
        stride = arrayStride ?? byteSize
    }

    var isArray: Bool { runtime || count != nil }
    var alignment: UInt32 { width == 3 ? 16 : UInt32(width) * 4 }
    var metalAtomicType: String { scalar == .i32 ? "atomic_int" : "atomic_uint" }
    var minimumByteSize: UInt64 { UInt64(count ?? 1) * UInt64(stride) }
    var glslSuffix: String { runtime ? "[]" : (count.map { "[\($0)]" } ?? "") }

    func wordLayout(addressing: ShaderStorage.WordAddressing = .structured) -> ShaderStorage.WordLayout {
        .init(offset: offset / 4, stride: stride / 4, width: width, scalar: scalar, addressing: addressing, columns: columns)
    }

    func valueType(in format: ShaderOutputFormat, module: IRModule) throws -> String {
        let base: String
        if atomic {
            let signed = scalar == .i32
            switch format {
            case .msl: base = metalAtomicType
            case .wgsl: base = signed ? "atomic<i32>" : "atomic<u32>"
            default: base = signed ? "int" : "uint"
            }
        } else {
            base = try shaderValueType(element, module: module, format: format, structures: [:])
        }
        guard format == .wgsl else { return base }
        if runtime { return "array<\(base)>" }
        return count.map { "array<\(base), \($0)>" } ?? base
    }
}

private func unsupported(_ reason: String) -> ShaderFailure {
    .unsupported("Compute code generation: \(reason)")
}
