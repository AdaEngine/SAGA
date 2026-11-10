import SAGACompiler
import SAGAShaderIR
import SAGAValidation
import Testing

@Test func perPrimitiveFragmentInputRequiresMeshCapability() throws {
    let source = """
    enable wgpu_mesh_shader;
    @fragment fn fs(@location(0) @per_primitive value: u32) -> @location(0) u32 { return value; }
    """
    let compiler = IndependentShaderCompiler()
    let module = try compiler.parseAndValidate(.wgsl(source), capabilities: [.meshShader]).module
    #expect(throws: ShaderFailure.self) { try ShaderSemanticValidator().validate(module) }
}

@Test(arguments: [
    "@vertex fn vs(@location(0) @per_primitive v: f32) -> @builtin(position) vec4f { return vec4f(v); }",
    "@fragment fn fs() -> @location(0) @per_primitive f32 { return 1.0; }",
]) func perPrimitiveRejectsOtherInterfaces(entry: String) {
    #expect(throws: ShaderFailure.self) {
        try IndependentShaderCompiler().parseAndValidate(.wgsl("enable wgpu_mesh_shader;\n" + entry), capabilities: [.meshShader])
    }
}
