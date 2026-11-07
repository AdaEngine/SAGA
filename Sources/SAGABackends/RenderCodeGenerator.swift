import SAGAShaderIR
import SAGAValidation

private struct RenderField {
    let index: Int
    let type: IRHandle<IRType>
    let varying: RenderStageArtifact.Varying?
    var name: String { "saga_member_\(index)" }
    var global: String { varying.map { "saga_varying_\($0.location)" } ?? "gl_Position" }
}

enum FragmentOutputField: Equatable {
    case color(UInt32, IRScalar)
    case depth
    case sampleMask

    var colorScalar: IRScalar? {
        if case .color(_, let scalar) = self { return scalar }
        return nil
    }

    func type(in format: ShaderOutputFormat) throws -> String {
        switch self {
        case .color(_, let scalar): return try shaderVectorType(4, scalar: scalar, in: format)
        case .depth: return floatType(1, in: format)
        case .sampleMask: return format == .wgsl ? "u32" : "uint"
        }
    }

    func binding(in format: ShaderOutputFormat) -> String {
        switch (format, self) {
        case (.msl, .color(let location, _)): "color(\(location))"
        case (.msl, .depth): "depth(any)"
        case (.msl, .sampleMask): "sample_mask"
        case (.hlsl, .color(let location, _)): "SV_Target\(location)"
        case (.hlsl, .depth): "SV_Depth"
        case (.hlsl, .sampleMask): "SV_Coverage"
        case (.wgsl, .color(let location, _)): "@location(\(location))"
        case (.wgsl, .depth): "@builtin(frag_depth)"
        case (.wgsl, .sampleMask): "@builtin(sample_mask)"
        case (.glsl, .color(let location, _)): location == 0 ? "saga_color" : "saga_color_\(location)"
        case (.glsl, .depth): "gl_FragDepth"
        case (.glsl, .sampleMask): "gl_SampleMask[0]"
        default: ""
        }
    }
}

package struct RenderCodeGenerator {
    private static let metalPointOutput = [
        "#ifdef \(RenderStageArtifact.metalPointTopologyMacro)",
        "    float saga_point_size [[point_size]] = 1.0f;",
        "#endif",
    ]

    private static let metalPipelineMaskOutput = [
        "#ifdef \(RenderStageArtifact.metalPipelineSampleMaskMacro)",
        "    uint saga_pipeline_sample_mask [[sample_mask]] = uint(\(RenderStageArtifact.metalPipelineSampleMaskMacro));",
        "#endif",
    ]

    let format: ShaderOutputFormat

    package init(_ format: ShaderOutputFormat) { self.format = format }

    package func generate(_ validated: ValidatedModule<IRModule>, entryPoint: String) throws -> RenderStageArtifact {
        let module = validated.module
        guard let entry = module.entryPoints.first(where: { $0.name == entryPoint }) else {
            throw unsupported("missing entry point '\(entryPoint)'")
        }
        let vertex = entry.stage == .vertex
        guard vertex || entry.stage == .fragment else { throw unsupported("expected a vertex or fragment stage") }
        guard module.overrides.count == 0, entry.earlyDepthTest == nil
        else { throw unsupported("overrides and early depth") }

        var helpers = try ShaderFunctions(module: module, entry: entry.function, captureGlobals: true)
        let globals = module.globalVariables.values
        let reachable = helpers.reachableGlobals
        let active = globals.enumerated().filter { reachable.contains($0.offset) }.map(\.element)
        guard active.allSatisfy({ $0.space == .uniform || $0.space == .handle }) else {
            throw unsupported("render resource must be a uniform buffer, texture or sampler")
        }
        let activeUniforms = active.filter { $0.space == .uniform }
        let activeHandles = try handleResourceGlobals(active, module: module)
        guard activeUniforms.count <= Int(ComputeShaderArtifact.uniformBindingCapacity),
              activeHandles.textures.count <= Int(ComputeShaderArtifact.sampledTextureBindingCapacity),
              activeHandles.storageTextures.count <= Int(ComputeShaderArtifact.storageTextureBindingCapacity),
              activeHandles.samplers.count <= Int(ComputeShaderArtifact.samplerBindingCapacity) else {
            throw unsupported("render resource count exceeds the per-stage limit")
        }
        // OpenGL uses shared binding namespaces across the linked stages.
        let mapped = format == .glsl ? globals : active
        let namespaceCount: UInt32 = format == .glsl ? 2 : 1
        let flatUniforms = try flatResourceBindingIndices(mapped.filter { $0.space == .uniform },
            capacity: ComputeShaderArtifact.uniformBindingCapacity * namespaceCount)
        let sampled = format == .glsl ? try handleResourceGlobals(globals, module: module) : activeHandles
        let flatTextures = try flatResourceBindingIndices(sampled.textures,
            capacity: ComputeShaderArtifact.sampledTextureBindingCapacity * namespaceCount)
        let renderEntries = module.entryPoints.filter { $0.stage == .vertex || $0.stage == .fragment }.map(\.function)
        let mappedStorage: [IRGlobalVariable]
        if format == .glsl, !activeHandles.storageTextures.isEmpty {
            mappedStorage = try helpers.glslSharedStorageTextures(module: module, entry: entry.function, sharedEntries: renderEntries)
        } else {
            mappedStorage = activeHandles.storageTextures
        }
        let flatStorageTextures = try flatResourceBindingIndices(mappedStorage,
            capacity: ComputeShaderArtifact.storageTextureBindingCapacity * namespaceCount)
        let flatSamplers = try flatResourceBindingIndices(sampled.samplers,
            capacity: ComputeShaderArtifact.samplerBindingCapacity * namespaceCount)
        let glslBindings = format == .glsl ? try helpers.glslTextureSamplerBindings(
            module: module, entry: entry.function, textureSlots: Set(flatTextures.values),
            sharedEntries: renderEntries) : []
        let uniforms = try globals.enumerated().map { index, global -> UniformShaderBinding? in
            guard reachable.contains(index), let binding = global.binding, let slot = flatUniforms[binding] else { return nil }
            return try UniformShaderBinding(global: global, index: index, flat: slot, module: module, format: format)
        }
        let textures = try globals.enumerated().map { index, global -> SampledTextureShaderBinding? in
            guard reachable.contains(index), let binding = global.binding, let slot = flatTextures[binding] else { return nil }
            return try SampledTextureShaderBinding(global: global, index: index, flat: slot, module: module, format: format)
        }
        let storageTextures = try globals.enumerated().map { index, global -> StorageTextureShaderBinding? in
            guard reachable.contains(index), let binding = global.binding, let slot = flatStorageTextures[binding] else {
                return nil
            }
            return try StorageTextureShaderBinding(global: global, index: index, flat: slot, module: module, format: format)
        }
        let nonFilteringSamplers = try helpers.nonFilteringSamplers(
            module: module, entry: entry.function, sampledTextures: activeHandles.textures)
        let samplers = try globals.enumerated().map { index, global -> SamplerShaderBinding? in
            guard reachable.contains(index), let binding = global.binding, let slot = flatSamplers[binding] else { return nil }
            return try SamplerShaderBinding(global: global, index: index, flat: slot, module: module,
                                            format: format, requiresNonFiltering: nonFilteringSamplers.contains(index))
        }
        let usedUniforms = uniforms.compactMap { $0 }
        let usedTextures = textures.compactMap { $0 }
        let usedStorageTextures = storageTextures.compactMap { $0 }
        let usedSamplers = samplers.compactMap { $0 }
        let storage = globals.indices.map {
            uniforms[$0]?.storage ?? textures[$0]?.storage ?? storageTextures[$0]?.storage ??
                samplers[$0]?.storage ?? ShaderStorage(reference: "", writable: false)
        }
        if format == .msl {
            try helpers.bindMetalStorage(globals.indices.map {
                uniforms[$0]?.captures ?? textures[$0]?.captures ?? storageTextures[$0]?.captures ??
                    samplers[$0]?.captures ?? []
            })
        }

        let function = try module.functions.value(for: entry.function)
        guard let result = function.result else { throw unsupported("missing render stage result") }
        let outputFields = try vertex ? vertexOutputs(result, module: module) : []
        let fragmentOutput = try vertex ? nil : fragmentResult(result, module: module)
        let fragmentFields = fragmentOutput?.fields ?? []
        let directField = fragmentFields.first ?? .color(0, .f32)
        let directType = try directField.type(in: format)
        let directBinding = directField.binding(in: format)
        let colors = fragmentFields.compactMap { field -> (UInt32, IRScalarKind)? in
            guard case .color(let location, let scalar) = field else { return nil }
            return (location, scalar.kind)
        }

        var parameter = ""
        var arguments: [String] = []
        var argumentFields: [Int: [String]] = [:]
        var inputFields: [RenderField] = []
        var inputDeclarations: [String] = []
        var vertexInputs: [RenderStageArtifact.VertexInput] = []
        var vertexBuiltins: [IRBuiltIn] = []
        if vertex {
            let inputs = try vertexArguments(function.arguments, module: module)
            parameter = inputs.parameter
            arguments = inputs.references
            argumentFields = inputs.fields
            inputDeclarations = inputs.declarations
            vertexInputs = inputs.inputs
            vertexBuiltins = inputs.builtins
        } else {
            let inputs = try fragmentArguments(function.arguments, module: module)
            parameter = inputs.parameter
            arguments = inputs.references
            argumentFields = inputs.fields
            inputDeclarations = inputs.declarations
            inputFields = inputs.varyings
        }

        let stage: ShaderStage = vertex ? .vertex : .fragment
        var parameters = parameter.isEmpty ? [] : [parameter]
        parameters += usedUniforms.flatMap(\.parameters)
        parameters += usedTextures.flatMap(\.parameters)
        parameters += usedStorageTextures.flatMap(\.parameters)
        parameters += usedSamplers.flatMap(\.parameters)
        if format == .msl && fragmentFields.contains(.depth) {
            parameters.append("constant float2& saga_viewport_depth [[buffer(\(RenderStageArtifact.metalViewportDepthBufferIndex))]]")
        }
        parameter = parameters.joined(separator: ", ")
        let name = format == .glsl ? "main" : "saga_\(stage.rawValue)"
        var signature: String
        var declarations: [String] = []
        var resultTarget: ShaderResultTarget
        var structures: [IRHandle<IRType>: ShaderStructure] = [:]
        var hlslProfile: String?
        switch format {
        case .msl:
            let output = vertex ? "SagaVertexOutput" : "SagaFragmentOutput"
            let member = directField.colorScalar == nil ? "value" : "color"
            let field = vertex ? "position [[position]]" : "\(member) [[\(directBinding)]]"
            declarations = ["#include <metal_stdlib>", "using namespace metal;"]
            if vertex {
                declarations += ["struct \(output) {", "    \(directType) \(field);"] + Self.metalPointOutput + ["};"]
            } else if fragmentOutput?.isStructure != true {
                let maskOutput = directField == .sampleMask ? [] : Self.metalPipelineMaskOutput
                declarations += ["struct \(output) {", "    \(directType) \(field);"] + maskOutput + ["};"]
            }
            signature = "\(stage.rawValue) \(output) \(name)(\(parameter))"
            resultTarget = vertex ? .metalStructure(output) : .fragmentValue(directField, metalType: output)
        case .hlsl:
            hlslProfile = vertex ? (vertexBuiltins.isEmpty ? "vs_6_0" : "vs_6_8") : "ps_6_0"
            let semantic = vertex ? "SV_Position" : directBinding
            signature = "\(directType) \(name)(\(parameter)) : \(semantic)"
            resultTarget = vertex ? .direct : .fragmentValue(directField, metalType: nil)
        case .glsl:
            declarations = [vertexBuiltins.contains(.instanceIndex) ? "#version 460 core" : "#version 450 core"]
            declarations += helpers.glslExtensions
            for field in fragmentFields {
                guard case .color(let location, _) = field else { continue }
                let type = try field.type(in: format)
                declarations.append("layout(location = \(location)) out \(type) \(field.binding(in: format));")
            }
            signature = "void \(name)()"
            resultTarget = vertex ? .glslPosition : .fragmentValue(directField, metalType: nil)
        case .wgsl:
            let binding = vertex ? "@builtin(position)" : directBinding
            signature = "@\(stage.rawValue) fn \(name)(\(parameter)) -> \(binding) \(directType)"
            resultTarget = .direct
        default: throw unsupported("output format \(format.rawValue)")
        }
        if !outputFields.isEmpty {
            let typeName = "SagaVertexOutput"
            let make = format == .hlsl ? "saga_make_vertex_output" : typeName
            let output = ShaderStructure(typeName: typeName, constructor: make)
            structures[result.type] = output
            switch format {
            case .msl:
                declarations = ["#include <metal_stdlib>", "using namespace metal;"]
                signature = "vertex \(typeName) \(name)(\(parameter))"
                resultTarget = .direct
            case .hlsl: signature = "\(typeName) \(name)(\(parameter))"
            case .wgsl: signature = "@vertex fn \(name)(\(parameter)) -> \(typeName)"
            case .glsl:
                declarations += try glslVaryings(outputFields, output: true, module: module)
                resultTarget = .glslVertexStructure(typeName, destinations: outputFields.map(\.global))
            default: break
            }
            declarations += try structure(typeName, fields: outputFields, vertexOutput: true, module: module)
            if format == .hlsl {
                let types = try outputFields.map {
                    try shaderValueType($0.type, module: module, format: format, structures: [:])
                }
                declarations += output.hlslConstructor(memberTypes: types)
            }
        }
        if fragmentOutput?.isStructure == true {
            let typeName = "SagaFragmentOutput"
            let structure = ShaderStructure(typeName: typeName,
                constructor: format == .hlsl ? "saga_make_fragment_output" : typeName)
            structures[result.type] = structure
            let types = try fragmentFields.map { try $0.type(in: format) }
            let members = fragmentFields.enumerated().map { index, field -> String in
                let member = "saga_member_\(index)"
                let type = types[index]
                let binding = field.binding(in: format)
                switch format {
                case .msl: return "    \(type) \(member) [[\(binding)]];"
                case .hlsl: return "    \(type) \(member) : \(binding);"
                case .wgsl: return "    \(binding) \(member): \(type),"
                default: return "    \(type) \(member);"
                }
            }
            let maskOutput = format == .msl && !fragmentFields.contains(.sampleMask) ? Self.metalPipelineMaskOutput : []
            declarations += ["struct \(typeName) {"] + members + maskOutput + [format == .wgsl ? "}" : "};"]
            switch format {
            case .msl: signature = "fragment \(typeName) \(name)(\(parameter))"
            case .hlsl:
                signature = "\(typeName) \(name)(\(parameter))"
                declarations += structure.hlslConstructor(memberTypes: types)
            case .wgsl: signature = "@fragment fn \(name)(\(parameter)) -> \(typeName)"
            default: break
            }
            resultTarget = .fragmentStructure(typeName, fields: fragmentFields)
        }
        if format == .msl && fragmentFields.contains(.sampleMask) {
            let mask = RenderStageArtifact.metalPipelineSampleMaskMacro
            declarations += ["inline uint saga_apply_pipeline_mask(uint mask) {",
                             "#ifdef \(mask)", "    return mask & uint(\(mask));",
                             "#else", "    return mask;", "#endif", "}"]
        }
        declarations += inputDeclarations + usedUniforms.flatMap(\.declarations) + usedTextures.flatMap(\.declarations)
        declarations += usedStorageTextures.flatMap(\.declarations) + usedSamplers.flatMap(\.declarations)
        declarations += glslBindings.map { $0.declaration }

        let helperDeclarations = try helpers.declarations(module: module, format: format, storage: storage, structures: structures)
        var writer = ShaderBodyWriter(module: module, function: function, arguments: arguments,
                                      storage: storage, outputFormat: format, resultTarget: resultTarget,
                                      structures: structures, argumentFields: argumentFields,
                                      callCaptures: helpers.callCaptures, budget: helpers.budget)
        let body = try writer.write()
        let numeric = try integerPrelude(helpers.integerHelpers.union(writer.integerHelpers),
                                         vectors: helpers.integerVectorHelpers.union(writer.integerVectorHelpers),
                                         in: format)
        let source = (declarations + numeric + helperDeclarations + [signature + " {"] + body + ["}", ""])
            .joined(separator: "\n")
        return .init(artifact: .text(format: format, source: source), stage: stage,
                     sourceEntryPoint: entry.name, entryPoint: name,
                     varyings: (vertex ? outputFields : inputFields).compactMap(\.varying).sorted { $0.location < $1.location },
                     vertexInputs: vertexInputs, hlslProfile: hlslProfile,
                     uniformBindings: usedUniforms.map(\.reflection), sampledTextures: usedTextures.map(\.reflection),
                     storageTextures: usedStorageTextures.map(\.reflection), samplerBindings: usedSamplers.map(\.reflection),
                     glslTextureSamplerBindings: glslBindings.map { $0.reflection },
                     colorOutputKinds: Dictionary(uniqueKeysWithValues: colors), writesDepth: fragmentFields.contains(.depth),
                     writesSampleMask: fragmentFields.contains(.sampleMask))
    }

    private func vertexArguments(
        _ arguments: [IRFunctionArgument], module: IRModule
    ) throws -> (parameter: String, references: [String], fields: [Int: [String]],
                 declarations: [String], inputs: [RenderStageArtifact.VertexInput], builtins: [IRBuiltIn]) {
        var attributes: [(input: RenderStageArtifact.VertexInput, type: IRHandle<IRType>)] = []
        var references: [String] = []
        var fields: [Int: [String]] = [:]
        var builtinParameters: [String] = []
        var builtins: [IRBuiltIn] = []

        func reference(type: IRHandle<IRType>, binding: IRBinding?) throws -> String {
            let kind = try module.types.value(for: type).kind
            if binding == .builtIn(.vertexIndex), kind == .scalar(.u32) {
                builtins.append(.vertexIndex)
                switch format {
                case .msl: builtinParameters.append("uint saga_index [[vertex_id]]")
                case .hlsl:
                    builtinParameters += ["uint saga_index : SV_VertexID", "int saga_first_vertex : SV_StartVertexLocation"]
                    return "(saga_index + asuint(saga_first_vertex))"
                case .wgsl: builtinParameters.append("@builtin(vertex_index) saga_index: u32")
                default: return "uint(gl_VertexID)"
                }
                return "saga_index"
            }
            if binding == .builtIn(.instanceIndex), kind == .scalar(.u32) {
                builtins.append(.instanceIndex)
                switch format {
                case .msl: builtinParameters.append("uint saga_instance [[instance_id]]")
                case .hlsl:
                    builtinParameters += ["uint saga_instance : SV_InstanceID", "uint saga_first_instance : SV_StartInstanceLocation"]
                    return "(saga_instance + saga_first_instance)"
                case .wgsl: builtinParameters.append("@builtin(instance_index) saga_instance: u32")
                default: return "(uint(gl_InstanceID) + uint(gl_BaseInstance))"
                }
                return "saga_instance"
            }
            guard case .location(let location, nil, nil, nil, false) = binding, location < 16 else {
                throw unsupported("expected vertex_index, instance_index or a vertex attribute at location 0..<16")
            }
            let width: UInt8
            let scalar: IRScalar
            switch kind {
            case .scalar(let lane): (width, scalar) = (1, lane)
            case .vector(let size, let lane): (width, scalar) = (size.rawValue, lane)
            default: throw unsupported("vertex attribute must be a scalar or vector")
            }
            guard [.f32, .i32, .u32].contains(scalar) else {
                throw unsupported("vertex attribute must use f32, i32 or u32")
            }
            attributes.append((.init(location: location, componentCount: width, scalar: scalar), type))
            let name = "saga_attribute_\(location)"
            return format == .glsl ? name : "saga_vertex_input.\(name)"
        }

        for (index, argument) in arguments.enumerated() {
            if case .structure(let members, _) = try module.types.value(for: argument.type).kind {
                fields[index] = try members.map { try reference(type: $0.type, binding: $0.binding) }
                references.append("")
            } else {
                references.append(try reference(type: argument.type, binding: argument.binding))
            }
        }

        attributes.sort { $0.input.location < $1.input.location }
        var parameters: [String] = []
        var declarations: [String] = []
        if !attributes.isEmpty {
            let lines = try attributes.map { attribute -> String in
                let input = attribute.input
                let type = try shaderValueType(attribute.type, module: module, format: format, structures: [:])
                let name = "saga_attribute_\(input.location)"
                switch format {
                case .msl: return "    \(type) \(name) [[attribute(\(input.location))]];"
                case .hlsl: return "    \(type) \(name) : TEXCOORD\(input.location);"
                case .wgsl: return "    @location(\(input.location)) \(name): \(type),"
                default: return "layout(location = \(input.location)) in \(type) \(name);"
                }
            }
            if format == .glsl {
                declarations = lines
            } else {
                declarations = ["struct SagaVertexInput {"] + lines + [format == .wgsl ? "}" : "};"]
                switch format {
                case .msl: parameters.append("SagaVertexInput saga_vertex_input [[stage_in]]")
                case .hlsl: parameters.append("SagaVertexInput saga_vertex_input")
                case .wgsl: parameters.append("saga_vertex_input: SagaVertexInput")
                default: break
                }
            }
        }
        parameters += builtinParameters
        return (parameters.joined(separator: ", "), references, fields, declarations,
                attributes.map { $0.input }, builtins)
    }

    private func fragmentArguments(
        _ arguments: [IRFunctionArgument], module: IRModule
    ) throws -> (parameter: String, references: [String], fields: [Int: [String]],
                 declarations: [String], varyings: [RenderField]) {
        var references: [String] = []
        var fields: [Int: [String]] = [:]
        var varyings: [RenderField] = []
        var builtins: [String] = []
        var hlslBuiltins: [String] = []
        func reference(type: IRHandle<IRType>, binding: IRBinding?) throws -> String {
            let builtin: (name: String, msl: String, hlsl: String, wgsl: String, glsl: String)?
            switch binding {
            case .builtIn(.sampleIndex):
                builtin = ("saga_sample_index", "sample_id", "SV_SampleIndex", "sample_index", "uint(gl_SampleID)")
            case .builtIn(.sampleMask):
                builtin = ("saga_sample_mask", "sample_mask", "SV_Coverage", "sample_mask", "uint(gl_SampleMaskIn[0])")
            case .builtIn(.position):
                builtin = ("saga_position", "position", "SV_Position", "position", "gl_FragCoord")
            case .builtIn(.frontFacing):
                builtin = ("saga_front_facing", "front_facing", "SV_IsFrontFace", "front_facing", "gl_FrontFacing")
            default:
                builtin = nil
            }
            if let builtin {
                let type = try shaderValueType(type, module: module, format: format, structures: [:])
                switch format {
                case .msl: builtins.append("\(type) \(builtin.name) [[\(builtin.msl)]]")
                case .hlsl:
                    hlslBuiltins.append("    \(type) \(builtin.name) : \(builtin.hlsl);")
                    let reference = "saga_input.\(builtin.name)"
                    if case .builtIn(.position) = binding {
                        // SV_Position.w contains clip-space w, while WGSL exposes its reciprocal.
                        return "float4(\(reference).xyz, 1.0f / \(reference).w)"
                    }
                    return reference
                case .wgsl: builtins.append("@builtin(\(builtin.wgsl)) \(builtin.name): \(type)")
                default: return builtin.glsl
                }
                return builtin.name
            }
            let varying = try field(varyings.count, type: type, binding: binding, module: module)
            varyings.append(varying)
            return format == .glsl ? varying.global : "saga_input.\(varying.name)"
        }
        for (index, argument) in arguments.enumerated() {
            if case .structure(let members, _) = try module.types.value(for: argument.type).kind {
                fields[index] = try members.map { try reference(type: $0.type, binding: $0.binding) }
                references.append("")
            } else {
                references.append(try reference(type: argument.type, binding: argument.binding))
            }
        }
        if format == .glsl {
            return ("", references, fields, try glslVaryings(varyings, output: false, module: module), varyings)
        }
        var parameters: [String] = []
        var declarations: [String] = []
        if !varyings.isEmpty || !hlslBuiltins.isEmpty {
            declarations = try structure("SagaFragmentInput", fields: varyings, vertexOutput: false, module: module)
            // glslang preserves an input SV_Coverage when it is a structure member.
            declarations.insert(contentsOf: hlslBuiltins, at: declarations.count - 1)
            switch format {
            case .msl: parameters.append("SagaFragmentInput saga_input [[stage_in]]")
            case .hlsl: parameters.append("SagaFragmentInput saga_input")
            case .wgsl: parameters.append("saga_input: SagaFragmentInput")
            default: break
            }
        }
        return ((parameters + builtins).joined(separator: ", "), references, fields, declarations, varyings)
    }

    private func fragmentResult(
        _ result: IRFunctionResult, module: IRModule
    ) throws -> (fields: [FragmentOutputField], isStructure: Bool) {
        func field(type: IRHandle<IRType>, binding: IRBinding?) throws -> FragmentOutputField {
            switch binding {
            case .builtIn(.fragDepth): return .depth
            case .builtIn(.sampleMask): return .sampleMask
            case .location(let location, nil, nil, nil, false) where location < 8:
                guard case .vector(.four, let scalar) = try module.types.value(for: type).kind,
                      [.f32, .i32, .u32].contains(scalar) else {
                    throw unsupported("expected a vec4<f32>, vec4<i32> or vec4<u32> color")
                }
                return .color(location, scalar)
            default: throw unsupported("expected a color at location 0..<8, frag_depth or sample_mask")
            }
        }
        if result.binding == nil, case .structure(let members, _) = try module.types.value(for: result.type).kind {
            return (try members.map { try field(type: $0.type, binding: $0.binding) }, true)
        }
        return ([try field(type: result.type, binding: result.binding)], false)
    }

    private func vertexOutputs(_ result: IRFunctionResult, module: IRModule) throws -> [RenderField] {
        let type = try module.types.value(for: result.type).kind
        let position = IRBinding.builtIn(.position(invariant: false))
        if case .structure(let members, _) = type, result.binding == nil {
            return try members.enumerated().map { index, member in
                if member.binding == position,
                   try module.types.value(for: member.type).kind == .vector(size: .four, scalar: .f32) {
                    return RenderField(index: index, type: member.type, varying: nil)
                }
                return try field(index, type: member.type, binding: member.binding, module: module)
            }
        }
        guard result.binding == position, type == .vector(size: .four, scalar: .f32) else {
            throw unsupported("expected a vec4<f32> position")
        }
        return []
    }

    private func field(
        _ index: Int, type: IRHandle<IRType>, binding: IRBinding?, module: IRModule
    ) throws -> RenderField {
        guard case .location(let location, let declaredInterpolation, let declaredSampling, nil, false) = binding,
              location < 16 else {
            throw unsupported("expected a varying at location 0..<16")
        }
        let interpolation = declaredInterpolation ?? .perspective
        let sampling = declaredSampling ?? (interpolation == .flat ? .first : .center)
        switch interpolation {
        case .flat:
            guard sampling == .first || sampling == .either else { throw unsupported("unsupported flat sampling") }
        case .perspective, .linear:
            guard sampling == .center || sampling == .centroid || sampling == .sample else {
                throw unsupported("unsupported interpolation sampling")
            }
        case .perVertex:
            throw unsupported("per-vertex varying interpolation")
        }
        let width: UInt8
        let scalar: IRScalar
        switch try module.types.value(for: type).kind {
        case .scalar(let lane): (width, scalar) = (1, lane)
        case .vector(let size, let lane): (width, scalar) = (size.rawValue, lane)
        default: throw unsupported("expected a scalar or vector varying")
        }
        guard scalar == .f32 || (interpolation == .flat && (scalar == .i32 || scalar == .u32)) else {
            throw unsupported("unsupported varying scalar type")
        }
        return RenderField(index: index, type: type, varying: .init(
            location: location, componentCount: width, scalar: scalar, interpolation: interpolation, sampling: sampling))
    }

    private func structure(
        _ name: String, fields: [RenderField], vertexOutput: Bool, module: IRModule
    ) throws -> [String] {
        let lines = try fields.map { field -> String in
            let type = try shaderValueType(field.type, module: module, format: format, structures: [:])
            guard let varying = field.varying else {
                switch format {
                case .msl: return "    float4 \(field.name) [[position]];"
                case .hlsl: return "    float4 \(field.name) : SV_Position;"
                case .wgsl: return "    @builtin(position) \(field.name): \(type),"
                default: return "    \(type) \(field.name);"
                }
            }
            switch format {
            case .msl:
                let modifier = varying.interpolation == .flat ? "flat" :
                    "\(varying.sampling.rawValue)_\(varying.interpolation == .linear ? "no_perspective" : "perspective")"
                let attribute = vertexOutput ? "" : ", \(modifier)"
                return "    \(type) \(field.name) [[user(locn\(varying.location))\(attribute)]];"
            case .hlsl:
                let qualifier = varyingQualifiers(varying, flat: "nointerpolation")
                return "    \(qualifier)\(type) \(field.name) : TEXCOORD\(varying.location);"
            case .wgsl:
                let attribute: String
                switch (varying.interpolation, varying.sampling) {
                case (.perspective, .center): attribute = ""
                case (.linear, .center), (.flat, .first):
                    attribute = " @interpolate(\(varying.interpolation.rawValue))"
                default:
                    attribute = " @interpolate(\(varying.interpolation.rawValue), \(varying.sampling.rawValue))"
                }
                return "    @location(\(varying.location))\(attribute) \(field.name): \(type),"
            default:
                return "    \(type) \(field.name);"
            }
        }
        let pointOutput = format == .msl && vertexOutput ? Self.metalPointOutput : []
        return ["struct \(name) {"] + lines + pointOutput + [format == .wgsl ? "}" : "};"]
    }

    private func varyingQualifiers(_ varying: RenderStageArtifact.Varying, flat: String) -> String {
        if varying.interpolation == .flat { return flat + " " }
        let interpolation = varying.interpolation == .linear ? "noperspective " : ""
        let sampling = varying.sampling == .center ? "" : "\(varying.sampling.rawValue) "
        return interpolation + sampling
    }

    private func glslVaryings(_ fields: [RenderField], output: Bool, module: IRModule) throws -> [String] {
        try fields.compactMap { field in
            guard let varying = field.varying else { return nil }
            let type = try shaderValueType(field.type, module: module, format: .glsl, structures: [:])
            let qualifier = varyingQualifiers(varying, flat: "flat")
            return "layout(location = \(varying.location)) \(qualifier)\(output ? "out" : "in") \(type) \(field.global);"
        }
    }

}

func floatType(_ width: UInt8, in format: ShaderOutputFormat) -> String {
    if width == 1 { return format == .wgsl ? "f32" : "float" }
    switch format {
    case .wgsl: return "vec\(width)<f32>"
    case .glsl: return "vec\(width)"
    default: return "float\(width)"
    }
}

private func unsupported(_ reason: String) -> ShaderFailure {
    .unsupported("Render code generation: \(reason)")
}
