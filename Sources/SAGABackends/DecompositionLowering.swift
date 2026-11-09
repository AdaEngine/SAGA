import SAGAShaderIR

func decompositionDeclarations(
    module: IRModule, format: ShaderOutputFormat, structures: inout [IRHandle<IRType>: ShaderStructure]
) throws -> [String] {
    var declarations: [String] = []
    for entry in module.specialTypes.predeclared {
        guard let result = entry.kind.decomposition else { continue }
        guard case .structure(let members, _) = try module.types.value(for: entry.type).kind else {
            throw ShaderFailure.invalidHandle
        }
        let width = result.size?.rawValue ?? 1
        let precision = result.scalar == .f16 ? "_f16" : ""
        let suffix = "\(result.function.rawValue)_\(width)\(precision)"
        let name = "SagaResult_\(suffix)"
        let constructor = format == .hlsl ? "saga_make_\(suffix)" : name
        let structure = ShaderStructure(typeName: name, constructor: constructor,
                                        memberNames: ["fract", result.function.decompositionSecondMember])
        let types = try members.map { try shaderValueType($0.type, module: module, format: format, structures: structures) }
        declarations += structure.declarations(memberTypes: types, in: format)
        structures[entry.type] = structure
    }
    return declarations
}

func halfType(_ width: UInt8, in format: ShaderOutputFormat) -> String {
    switch (format, width) {
    case (.wgsl, 1): "f16"
    case (.wgsl, _): "vec\(width)<f16>"
    case (.glsl, 1): "float16_t"
    case (.glsl, _): "f16vec\(width)"
    case (_, 1): "half"
    default: "half\(width)"
    }
}

let glslFloat16Extension = "#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require"
let glslStorage16BitExtension = "#extension GL_EXT_shader_16bit_storage : require"
