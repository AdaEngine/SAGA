import SAGACompiler
import SAGAShaderIR
import Testing

@Test(arguments: [false, true])
func wgslInt16BuiltinProfileRejectsOnlyTheNarrowEntry(_ usesBuiltins: Bool) throws {
    let narrowBody = usesBuiltins ? """
        let lanes = vec2<u16>(u16(narrowOutput), u16(2));
        let reversed = reverseBits(lanes);
        narrowOutput = u32(dot(reversed, vec2<u16>(u16(1), u16(2))));
        """ : "narrowOutput = u32(u16(narrowOutput) + u16(1));"
    let source = ShaderInput.wgsl("""
        enable wgpu_int16;
        @group(0) @binding(0) var<storage, read_write> narrowOutput: u32;
        @group(0) @binding(1) var<storage, read_write> portableOutput: u32;
        @compute @workgroup_size(1) fn narrow() {
            \(narrowBody)
        }
        @compute @workgroup_size(1) fn portable() {
            portableOutput += 7u;
        }
        """)
    let compiler = IndependentShaderCompiler()
    func expectEntryProfiles(_ input: ShaderInput) throws {
        let native = try compiler.compileCompute(input, entryPoint: "narrow", to: .wgsl)
        #expect(native.storageBindings.map(\.binding) == [.init(group: 0, binding: 0)])
        #expect(native.requiresShaderInt16)
        #expect(throws: ShaderFailure.unsupported("16-bit integer types require native shader output")) {
            try compiler.compileWebGPUCompute(input, entryPoint: "narrow")
        }

        let portable = try compiler.compileWebGPUCompute(input, entryPoint: "portable")
        #expect(portable.storageBindings.map(\.binding) == [.init(group: 0, binding: 1)])
        #expect(portable.storageBindings.map(\.minimumByteSize) == [4])
        #expect(!portable.requiresShaderInt16)
        guard case .text(.wgsl, let text) = portable.artifact else { throw ShaderFailure.incompatibleArtifact }
        #expect(!text.contains("wgpu_int16"))
    }

    try expectEntryProfiles(source)
    guard case .spirv(let words) = try compiler.compile(source, to: .spirv) else {
        throw ShaderFailure.incompatibleArtifact
    }
    try expectEntryProfiles(.spirv(words))
}
