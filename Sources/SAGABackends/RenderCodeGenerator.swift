import SAGAShaderIR
import SAGAValidation

private struct RenderField {
    let index: Int
    let varying: RenderStageArtifact.Varying?
    var name: String { "saga_member_\(index)" }
    var width: UInt8 { varying?.componentCount ?? 4 }
    var global: String { varying.map { "saga_varying_\($0.location)" } ?? "gl_Position" }
}

package struct RenderCodeGenerator {
    let format: ShaderOutputFormat

    package init(_ format: ShaderOutputFormat) { self.format = format }

    package func generate(_ validated: ValidatedModule<IRModule>, entryPoint: String) throws -> RenderStageArtifact {
        let module = validated.module
        guard let entry = module.entryPoints.first(where: { $0.name == entryPoint }) else {
            throw unsupported("missing entry point '\(entryPoint)'")
        }
        let vertex = entry.stage == .vertex
        guard vertex || entry.stage == .fragment else { throw unsupported("expected a vertex or fragment stage") }
        guard module.globalVariables.count == 0, module.constants.count == 0,
              module.overrides.count == 0,
              entry.earlyDepthTest == nil
        else { throw unsupported("resources, constants, overrides and early depth") }

        let function = try module.functions.value(for: entry.function)
        guard let result = function.result else { throw unsupported("missing render stage result") }
        let outputFields = try vertex ? vertexOutputs(result, module: module) : []
        if !vertex {
            let color = IRBinding.location(index: 0, interpolation: nil, sampling: nil, blendSource: nil, perPrimitive: false)
            guard result.binding == color,
                  try module.types.value(for: result.type).kind == .vector(size: .four, scalar: .f32)
            else { throw unsupported("expected a vec4<f32> color at location 0") }
        }

        var parameter = ""
        var arguments: [String] = []
        var argumentFields: [Int: [String]] = [:]
        var inputFields: [RenderField] = []
        var inputDeclarations: [String] = []
        var vertexInputs: [RenderStageArtifact.VertexInput] = []
        if vertex {
            let inputs = try vertexArguments(function.arguments, module: module)
            parameter = inputs.parameter
            arguments = inputs.references
            argumentFields = inputs.fields
            inputDeclarations = inputs.declarations
            vertexInputs = inputs.inputs
        } else if !function.arguments.isEmpty {
            let structured: Bool
            if function.arguments.count == 1,
               case .structure(let members, _) = try module.types.value(for: function.arguments[0].type).kind {
                structured = true
                inputFields = try members.enumerated().map { index, member in
                    try field(index, type: member.type, binding: member.binding, module: module)
                }
            } else {
                structured = false
                inputFields = try function.arguments.enumerated().map { index, argument in
                    try field(index, type: argument.type, binding: argument.binding, module: module)
                }
            }
            if format == .glsl {
                inputDeclarations = glslVaryings(inputFields, output: false)
            } else {
                inputDeclarations = structure("SagaFragmentInput", fields: inputFields, vertexOutput: false)
                switch format {
                case .msl: parameter = "SagaFragmentInput saga_input [[stage_in]]"
                case .hlsl: parameter = "SagaFragmentInput saga_input"
                case .wgsl: parameter = "saga_input: SagaFragmentInput"
                default: break
                }
            }
            let references = inputFields.map { format == .glsl ? $0.global : "saga_input.\($0.name)" }
            if structured {
                arguments = [format == .glsl ? "" : "saga_input"]
                argumentFields[0] = references
            } else {
                arguments = references
            }
        }

        let stage: ShaderStage = vertex ? .vertex : .fragment
        let name = format == .glsl ? "main" : "saga_\(stage.rawValue)"
        var signature: String
        var declarations: [String] = []
        var resultTarget: ShaderResultTarget
        var structures: [IRHandle<IRType>: ShaderStructure] = [:]
        switch format {
        case .msl:
            let output = vertex ? "SagaVertexOutput" : "SagaFragmentOutput"
            let field = vertex ? "position [[position]]" : "color [[color(0)]]"
            declarations = ["#include <metal_stdlib>", "using namespace metal;", "struct \(output) { float4 \(field); };"]
            signature = "\(stage.rawValue) \(output) \(name)(\(parameter))"
            resultTarget = .metalStructure(output)
        case .hlsl:
            let semantic = vertex ? "SV_Position" : "SV_Target0"
            signature = "float4 \(name)(\(parameter)) : \(semantic)"
            resultTarget = .direct
        case .glsl:
            declarations = ["#version 450 core"]
            if !vertex { declarations.append("layout(location = 0) out vec4 saga_color;") }
            signature = "void \(name)()"
            resultTarget = vertex ? .glslPosition : .glslColor
        case .wgsl:
            let binding = vertex ? "@builtin(position)" : "@location(0)"
            signature = "@\(stage.rawValue) fn \(name)(\(parameter)) -> \(binding) vec4<f32>"
            resultTarget = .direct
        default: throw unsupported("output format \(format.rawValue)")
        }
        if !outputFields.isEmpty {
            let typeName = "SagaVertexOutput"
            let make = format == .hlsl ? "saga_make_vertex_output" : typeName
            structures[result.type] = ShaderStructure(typeName: typeName, constructor: make)
            switch format {
            case .msl:
                declarations = ["#include <metal_stdlib>", "using namespace metal;"]
                signature = "vertex \(typeName) \(name)(\(parameter))"
                resultTarget = .direct
            case .hlsl: signature = "\(typeName) \(name)(\(parameter))"
            case .wgsl: signature = "@vertex fn \(name)(\(parameter)) -> \(typeName)"
            case .glsl:
                declarations += glslVaryings(outputFields, output: true)
                resultTarget = .glslVertexStructure(typeName, destinations: outputFields.map(\.global))
            default: break
            }
            declarations += structure(typeName, fields: outputFields, vertexOutput: true)
            if format == .hlsl {
                let parameters = outputFields.map { "\(floatType($0.width, in: format)) arg\($0.index)" }
                declarations.append("\(typeName) \(make)(\(parameters.joined(separator: ", "))) {")
                declarations.append("    \(typeName) result;")
                declarations += outputFields.map { "    result.\($0.name) = arg\($0.index);" }
                declarations += ["    return result;", "}"]
            }
        }
        declarations += inputDeclarations

        var helpers = try ShaderFunctions(module: module, entry: entry.function)
        let helperDeclarations = try helpers.declarations(module: module, format: format, storage: [], structures: structures)
        var writer = ShaderBodyWriter(module: module, function: function, arguments: arguments,
                                      storage: [], outputFormat: format, resultTarget: resultTarget,
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
                     vertexInputs: vertexInputs)
    }

    private func vertexArguments(
        _ arguments: [IRFunctionArgument], module: IRModule
    ) throws -> (parameter: String, references: [String], fields: [Int: [String]],
                 declarations: [String], inputs: [RenderStageArtifact.VertexInput]) {
        var attributes: [(input: RenderStageArtifact.VertexInput, type: IRHandle<IRType>)] = []
        var references: [String] = []
        var fields: [Int: [String]] = [:]
        var indexParameter: String?

        func reference(type: IRHandle<IRType>, binding: IRBinding?) throws -> String {
            let kind = try module.types.value(for: type).kind
            if binding == .builtIn(.vertexIndex), kind == .scalar(.u32) {
                switch format {
                case .msl: indexParameter = "uint saga_index [[vertex_id]]"
                case .hlsl: indexParameter = "uint saga_index : SV_VertexID"
                case .wgsl: indexParameter = "@builtin(vertex_index) saga_index: u32"
                default: return "uint(gl_VertexID)"
                }
                return "saga_index"
            }
            guard case .location(let location, nil, nil, nil, false) = binding, location < 16 else {
                throw unsupported("expected vertex_index or a vertex attribute at location 0..<16")
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
        if let indexParameter { parameters.append(indexParameter) }
        return (parameters.joined(separator: ", "), references, fields, declarations, attributes.map { $0.input })
    }

    private func vertexOutputs(_ result: IRFunctionResult, module: IRModule) throws -> [RenderField] {
        let type = try module.types.value(for: result.type).kind
        let position = IRBinding.builtIn(.position(invariant: false))
        if case .structure(let members, _) = type, result.binding == nil {
            return try members.enumerated().map { index, member in
                if member.binding == position,
                   try module.types.value(for: member.type).kind == .vector(size: .four, scalar: .f32) {
                    return RenderField(index: index, varying: nil)
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
        guard case .location(let location, let interpolation, let declaredSampling, nil, false) = binding,
              location < 16, interpolation == nil || interpolation == .perspective else {
            throw unsupported("expected a perspective varying at location 0..<16")
        }
        let sampling = declaredSampling ?? .center
        guard sampling == .center || sampling == .centroid || sampling == .sample else {
            throw unsupported("unsupported perspective sampling")
        }
        let width: UInt8
        switch try module.types.value(for: type).kind {
        case .scalar(.f32): width = 1
        case .vector(let size, .f32): width = size.rawValue
        default: throw unsupported("expected an f32 scalar or vector varying")
        }
        return RenderField(index: index, varying: .init(location: location, componentCount: width,
                                                          scalar: .f32, interpolation: .perspective, sampling: sampling))
    }

    private func structure(_ name: String, fields: [RenderField], vertexOutput: Bool) -> [String] {
        let lines = fields.map { field -> String in
            let type = floatType(field.width, in: format)
            guard let varying = field.varying else {
                switch format {
                case .msl: return "    float4 \(field.name) [[position]];"
                case .hlsl: return "    float4 \(field.name) : SV_Position;"
                case .wgsl: return "    @builtin(position) \(field.name): \(type),"
                default: return "    \(type) \(field.name);"
                }
            }
            let qualifier = varying.sampling == .center ? "" : "\(varying.sampling.rawValue) "
            switch format {
            case .msl:
                let sampling = vertexOutput ? "" : ", \(varying.sampling.rawValue)_perspective"
                return "    \(type) \(field.name) [[user(locn\(varying.location))\(sampling)]];"
            case .hlsl:
                return "    \(qualifier)\(type) \(field.name) : TEXCOORD\(varying.location);"
            case .wgsl:
                let sampling = varying.sampling == .center ? "" : " @interpolate(perspective, \(varying.sampling.rawValue))"
                return "    @location(\(varying.location))\(sampling) \(field.name): \(type),"
            default:
                return "    \(type) \(field.name);"
            }
        }
        return ["struct \(name) {"] + lines + [format == .wgsl ? "}" : "};"]
    }

    private func glslVaryings(_ fields: [RenderField], output: Bool) -> [String] {
        fields.compactMap { field in
            guard let varying = field.varying else { return nil }
            let qualifier = varying.sampling == .center ? "" : "\(varying.sampling.rawValue) "
            return "layout(location = \(varying.location)) \(qualifier)\(output ? "out" : "in") \(floatType(field.width, in: .glsl)) \(field.global);"
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
