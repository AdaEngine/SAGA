import SAGACompiler
import SAGAShaderIR
import Testing

@Test func wgslInt16AcceptsNativeTypesAndPreservesConstantWidths() throws {
    let source = """
        enable wgpu_int16, f16;
        alias Signed = i16;
        alias Pair = vec2<u16>;
        const MINIMUM: Signed = -32768;
        const MAXIMUM: u16 = 65535;
        const WRAPPED = i16(65535) + i16(2);
        const CONCRETE = i16(65535u);
        const CLAMPED = i16(f16(33000.0));
        const ZERO = u16(-4.75f);
        const NOT = ~u16(0);
        const NEGATED = -i16(-32768);
        const PAIR = Pair(vec2u(65536, 65537));
        const BITS = bitcast<u16>(i16(-1));
        const FLOAT_BITS = bitcast<f16>(u16(0x3c00));
        const_assert MINIMUM == i16(-32768);
        const_assert MAXIMUM == u16(65535);
        const_assert WRAPPED == i16(1);
        const_assert CONCRETE == i16(-1);
        const_assert CLAMPED == i16(32767);
        const_assert ZERO == u16(0);
        const_assert NOT == MAXIMUM;
        const_assert NEGATED == MINIMUM;
        const_assert PAIR.x == u16(0) && PAIR.y == u16(1);
        const_assert BITS == MAXIMUM;
        const_assert FLOAT_BITS == 1h;
        @compute @workgroup_size(1) fn main() {}
        """
    let validated = try IndependentShaderCompiler().parseAndValidate(.wgsl(source), capabilities: [.shaderInt16, .shaderFloat16])
    #expect(validated.module.types.values.contains { $0.kind == .scalar(.init(kind: .sint, width: 2)) })
    #expect(validated.module.types.values.contains { $0.kind == .vector(size: .two, scalar: .init(kind: .uint, width: 2)) })
}

@Test func wgslInt16ValidatesDynamicConversionsArithmeticAndIndexing() throws {
    let source = """
        enable wgpu_int16;
        @group(0) @binding(0) var<storage, read_write> output: i32;
        fn convert(value: i32, lanes: vec2i, count: u32) -> i32 {
            var signed = i16(value);
            let pair = vec2<i16>(lanes);
            signed += pair.x;
            signed = ~(signed << count) + i16(3);
            let values = array<i16, 2>(signed, pair.y);
            let selected = values[u16(count)];
            return i32(selected) + i32(bitcast<i16>(bitcast<u16>(signed)));
        }
        @compute @workgroup_size(1) fn main(@builtin(local_invocation_index) index: u32) {
            output = convert(output, vec2i(output), index);
        }
        """
    _ = try IndependentShaderCompiler().parseAndValidate(.wgsl(source), capabilities: [.shaderInt16])
}

@Test func wgslInt16AcceptsNonnegativeArrayLengthsAndNumericAttributes() throws {
    let source = """
        enable wgpu_int16;
        struct Packed {
            @align(u16(4)) @size(i16(6)) first: u16,
            values: array<i16, u16(3)>,
        }
        @group(u16(0)) @binding(i16(0)) var<storage, read_write> buffer: Packed;
        @compute @workgroup_size(u16(2), 1) fn main() {
            buffer.values[0] = i16(7);
        }
        """
    let result = try IndependentShaderCompiler().parseAndValidate(.wgsl(source), capabilities: [.shaderInt16])
    #expect(result.module.entryPoints.first?.workgroupSize.x == 2)
}

@Test func wgslInt16AcceptsPrivateInitializersAndConversionsFromOverrides() throws {
    let source = """
        enable wgpu_int16;
        override base: i32 = 5;
        var<private> fixed: i16 = i16(-2);
        var<private> dependent: u16 = u16(base) + u16(1);
        @compute @workgroup_size(1) fn main() {
            _ = fixed;
            _ = dependent;
        }
        """
    let result = try IndependentShaderCompiler().parseAndValidate(.wgsl(source), capabilities: [.shaderInt16])
    #expect(result.module.globalVariables.values.allSatisfy { $0.initializer != nil })
}

@Test func wgslInt16SpecializesDirectOverridesForValuesAndWorkgroupSizes() throws {
    let source = ShaderInput.wgsl("""
        enable wgpu_int16;
        override count: u16 = u16(2);
        @id(7) override bias: i16 = i16(-3);
        var<workgroup> values: array<u32, count + u16(1)>;
        @group(0) @binding(0) var<storage, read_write> output: u32;
        @compute @workgroup_size(count) fn main(@builtin(local_invocation_index) lane: u32) {
            values[lane] = bitcast<u32>(i32(bias));
            if lane == 0u { output = values[lane]; }
        }
        """)
    let compiler = IndependentShaderCompiler()
    let defaults = try compiler.compileCompute(source, to: .wgsl)
    let overridden = try compiler.compileCompute(source, to: .wgsl, constants: ["count": 5.75, "7": -7.25])
    #expect(defaults.workgroupSize == .init(x: 2))
    #expect(overridden.workgroupSize == .init(x: 5))
    #expect(overridden.workgroupStorageByteSize > defaults.workgroupStorageByteSize)
    for constants: [String: Double] in [["count": -1], ["count": 65536], ["7": -32769], ["7": 32768]] {
        #expect(throws: ShaderFailure.self) {
            _ = try compiler.compileCompute(source, to: .wgsl, constants: constants)
        }
    }
}

@Test func wgslInt16RejectsUnsupportedProfilesAndShapes() throws {
    let compiler = IndependentShaderCompiler()
    #expect(throws: ShaderFailure.self) {
        _ = try compiler.parseAndValidate(.wgsl("enable wgpu_int16; const x = i16(1);"), capabilities: [])
    }
    for source in [
        "const x = i16(1);",
        "alias X = vec2<u16>;",
        "enable wgpu_int16; const x = 1s;",
        "enable wgpu_int16; const x = 1us;",
        "enable wgpu_int16; const x: i16 = 1i;",
        "enable wgpu_int16; const x = i16(1) + 1u;",
        "enable wgpu_int16; const x = i16(-32768) / i16(-1);",
        "enable wgpu_int16; const x = u16(1) << 16u;",
        "enable wgpu_int16; alias X = mat2x2<i16>;",
        "enable wgpu_int16; @group(0) @binding(0) var<storage, read_write> x: atomic<u16>;",
        "enable wgpu_int16; const x = bitcast<i16>(1u);",
        "enable wgpu_int16; const x = vec3<i16>(vec2i(1));",
        "enable wgpu_int16; alias X = array<i16, i16(-1)>;",
        "enable wgpu_int16; @group(0) @binding(i16(-1)) var<storage, read_write> x: i16;",
        "enable wgpu_int16; @compute @workgroup_size(i16(-1)) fn main() {}",
    ] {
        #expect(throws: ShaderFailure.self) {
            _ = try compiler.parseAndValidate(.wgsl(source), capabilities: [.shaderInt16])
        }
    }
}
