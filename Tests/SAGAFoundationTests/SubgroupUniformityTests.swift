import Testing
import SAGACompiler
import SAGACore
import SAGAMetal
import SAGAShaderIR
import SAGATypes

private let subgroupUniformityShader = ShaderInput.wgsl("""
    requires subgroup_id, subgroup_uniformity;
    enable subgroups;
    @group(0) @binding(0) var<storage, read_write> output: array<vec2u>;
    fn selected(group: u32) -> bool { return group == 0u; }
    fn count_lanes(group: u32) -> u32 {
        if selected(group) { return subgroupAdd(1u); }
        return 0u;
    }
    fn forward(group: u32) -> u32 { return count_lanes(group); }
    @compute @workgroup_size(128)
    fn run(@builtin(subgroup_id) group: u32, @builtin(local_invocation_index) index: u32) {
        let shared_group = subgroupBroadcastFirst(group);
        output[index] = vec2u(group, forward(shared_group));
    }
    """)

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func subgroupUniformityAllowsSubgroupScopedBranches(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(subgroupUniformityShader, to: format)
    #expect(shader.requiresSubgroupUniformity)
    if case .text(.wgsl, let text) = shader.artifact {
        #expect(text.contains("requires subgroup_uniformity;"))
        _ = try compiler.parseAndValidate(.wgsl(text), capabilities: .referenceDefault.union(.subgroup))
    }
}

@Test(arguments: [
    "if group == 0u { workgroupBarrier(); }",
    "if group == 0u { synchronize(); }",
    "if lane == 0u { _ = subgroupAdd(1u); }",
    "_ = subgroupShuffleXor(1u, lane);",
    "if group == 0u { return; } workgroupBarrier();",
])
func subgroupUniformityRejectsOperationsOutsideTheirScope(_ body: String) {
    let source = """
        requires subgroup_id, subgroup_uniformity;
        enable subgroups;
        fn synchronize() { workgroupBarrier(); }
        @compute @workgroup_size(64)
        fn run(@builtin(subgroup_id) group: u32, @builtin(subgroup_invocation_id) lane: u32) {
            \(body)
        }
        """
    #expect(throws: ShaderFailure.self) {
        try IndependentShaderCompiler().parseAndValidate(.wgsl(source), capabilities: .referenceDefault.union(.subgroup))
    }
}

@Test func subgroupUniformitySurvivesAnEntireSubgroupReturning() throws {
    let source = """
        requires subgroup_id, subgroup_uniformity;
        enable subgroups;
        @compute @workgroup_size(64) fn run(@builtin(subgroup_id) group: u32) {
            if group == 0u { return; }
            _ = subgroupAdd(1u);
        }
        """
    _ = try IndependentShaderCompiler().parseAndValidate(.wgsl(source), capabilities: .referenceDefault.union(.subgroup))
}

@Test(arguments: [false, true])
func uniformHelperArgumentsAreCheckedAtTheirCallSites(varying: Bool) throws {
    let source = ShaderInput.wgsl("""
        requires subgroup_id, subgroup_uniformity;
        enable subgroups;
        fn gate(condition: u32, unused: u32) -> bool { return condition == 0u; }
        fn exchange(selector: u32, value: u32) -> u32 { return subgroupShuffleXor(value, selector); }
        fn run_collective(condition: u32, value: u32) {
            if gate(condition, value) { _ = exchange(condition, value); }
        }
        @compute @workgroup_size(64)
        fn run(@builtin(subgroup_id) group: u32, @builtin(subgroup_invocation_id) lane: u32) {
            run_collective(\(varying ? "lane" : "group"), lane);
        }
        """)
    let compiler = IndependentShaderCompiler()
    if varying {
        do {
            _ = try compiler.parseAndValidate(source, capabilities: .referenceDefault.union(.subgroup))
            Issue.record("Expected a non-uniform argument diagnostic")
        } catch ShaderFailure.diagnostics(let diagnostics) {
            #expect(diagnostics.first?.code == .nonUniformControlFlow)
            #expect(diagnostics.first?.message.contains("uniform argument") == true)
        }
    } else {
        _ = try compiler.compileCompute(source, to: .msl)
    }
}

@Test(arguments: [false, true])
func workgroupHelperRequiresAWorkgroupUniformArgument(useSubgroup: Bool) throws {
    let source = ShaderInput.wgsl("""
        requires subgroup_id, subgroup_uniformity;
        enable subgroups;
        fn synchronize(group: u32) { if group == 0u { workgroupBarrier(); } }
        @compute @workgroup_size(64)
        fn run(@builtin(workgroup_id) workgroup: vec3u, @builtin(subgroup_id) subgroup: u32) {
            synchronize(\(useSubgroup ? "subgroup" : "workgroup.x"));
        }
        """)
    let compiler = IndependentShaderCompiler()
    if useSubgroup {
        #expect(throws: ShaderFailure.self) { try compiler.compileCompute(source, to: .wgsl) }
    } else {
        _ = try compiler.compileCompute(source, to: .wgsl)
    }
}

@Test(arguments: [false, true])
func helperReturnDependsOnTheControlFlowThatSelectsIt(varying: Bool) throws {
    let source = ShaderInput.wgsl("""
        requires subgroup_id, subgroup_uniformity;
        enable subgroups;
        fn choose(value: u32) -> bool {
            if value == 0u { return true; }
            return false;
        }
        @compute @workgroup_size(64)
        fn run(@builtin(subgroup_id) group: u32, @builtin(subgroup_invocation_id) lane: u32) {
            if choose(\(varying ? "lane" : "group")) { _ = subgroupAdd(1u); }
        }
        """)
    let compiler = IndependentShaderCompiler()
    if varying {
        #expect(throws: ShaderFailure.self) { try compiler.compileCompute(source, to: .wgsl) }
    } else {
        _ = try compiler.compileCompute(source, to: .wgsl)
    }
}

@Test func subgroupUniformityKeepsTheWholeScopeWithoutTheDirective() {
    let source = """
        requires subgroup_id;
        enable subgroups;
        @compute @workgroup_size(64) fn run(@builtin(subgroup_id) group: u32) {
            if group == 0u { _ = subgroupAdd(1u); }
        }
        """
    #expect(throws: ShaderFailure.self) {
        try IndependentShaderCompiler().parseAndValidate(.wgsl(source), capabilities: .referenceDefault.union(.subgroup))
    }
}

@Test func subgroupUniformityIsReflectedInFragmentShaders() throws {
    let source = ShaderInput.wgsl("""
        requires subgroup_uniformity;
        enable subgroups;
        @fragment fn shade() -> @location(0) vec4f {
            let shared_value = subgroupBroadcastFirst(1u);
            if shared_value != 0u { return vec4f(f32(subgroupAdd(1u))); }
            return vec4f(0);
        }
        """)
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileWebGPURenderStage(source, entryPoint: "shade")
    #expect(shader.requiresSubgroupUniformity)
    guard case .text(.wgsl, let text) = shader.artifact else { Issue.record("Expected WGSL"); return }
    #expect(text.contains("requires subgroup_uniformity;"))
    _ = try compiler.parseAndValidate(.wgsl(text), capabilities: .referenceDefault.union(.subgroup))
}

@Test func subgroupUniformityReportsFilteredWarningsAndKeepsTheWGSLRule() throws {
    let source = ShaderInput.wgsl("""
        requires subgroup_uniformity;
        enable subgroups;
        diagnostic(warning, subgroup_uniformity);
        @diagnostic(off, subgroup_uniformity)
        fn quiet(index: u32) { if index == 0u { _ = subgroupAdd(1u); } }
        @compute @workgroup_size(64) fn run(@builtin(local_invocation_index) index: u32) {
            if index == 0u { _ = subgroupAdd(1u); }
            _ = subgroupShuffleXor(1u, index);
            quiet(index);
        }
        """)
    let compiler = IndependentShaderCompiler()
    let module = try compiler.parseAndValidate(source, capabilities: .referenceDefault.union(.subgroup))
    #expect(module.diagnostics.count == 2)
    #expect(module.diagnostics.allSatisfy { $0.code == .nonUniformControlFlow && $0.severity == .warning })
    let shader = try compiler.compileCompute(source, to: .wgsl)
    guard case .text(_, let text) = shader.artifact else { Issue.record("Expected WGSL"); return }
    #expect(text.contains("diagnostic(warning, subgroup_uniformity);"))
    #expect(text.contains("@diagnostic(off, subgroup_uniformity)"))
}

#if canImport(Metal)
@Test func subgroupUniformityExecutesOnlyTheSelectedMetalSubgroup() async throws {
    let shader = try IndependentShaderCompiler().compileCompute(subgroupUniformityShader, to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    let pipeline = try await device.createComputePipeline(shader)
    let output = try await device.createBuffer(.init(size: 1024, usage: [.storage, .copySource]))
    let readback = try await device.createBuffer(.init(size: 1024, usage: [.mapRead, .copyDestination]))
    let results = try await device.submitCommands([
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
        .copy(source: output, destination: readback, byteCount: 1024),
        .read(buffer: readback, byteCount: 1024),
    ]).wait()
    let words = try #require(results.first)
    try #require(words.count == 256)
    let lanes = stride(from: 0, to: words.count, by: 2)
    let firstGroupSize = UInt32(lanes.filter { words[$0] == 0 }.count)
    #expect(firstGroupSize > 0)
    #expect(lanes.allSatisfy { words[$0 + 1] == (words[$0] == 0 ? firstGroupSize : 0) })
    await device.close()
}
#endif
