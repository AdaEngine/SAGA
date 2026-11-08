import Testing
import SAGACompiler
import SAGACore
import SAGAMetal
import SAGAShaderIR
import SAGATypes

private let sharedLoopShader = """
    var<workgroup> scratch: array<u32, 4>;
    @group(0) @binding(0) var<storage, read_write> output: array<u32>;
    fn sync() { workgroupBarrier(); }
    @compute @workgroup_size(4)
    fn run(@builtin(local_invocation_index) lane: u32) {
        var total = 0u;
        for (var step = 0u; step < 3u; step++) {
            if step == 1u { continue; }
            scratch[lane] = step * 10u + lane;
            sync();
            total += scratch[(lane + 1u) % 4u];
            workgroupBarrier();
        }
        output[lane] = total;
    }
    """

private func workgroupLoopSource(_ body: String, helpers: String = "") -> ShaderInput {
    .wgsl("""
        \(helpers)
        @compute @workgroup_size(4)
        fn run(@builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) group: vec3u) {
            \(body)
        }
        """)
}

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func loopUniformityEmitsCountedLoopsWithBarriers(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(sharedLoopShader), to: format)
    if case .text(.wgsl, let text) = shader.artifact {
        _ = try compiler.parseAndValidate(.wgsl(text))
    }
}

@Test(arguments: [
    "loop { workgroupBarrier(); break; }",
    "var i = 0u; while i < group.x { workgroupBarrier(); i++; }",
    "var i = 0u; loop { workgroupBarrier(); continuing { i++; break if i >= 3u; } }",
    "var slot: u32; loop { slot = group.x; break; } if slot == 0u { workgroupBarrier(); }",
    "loop { switch lane { case 0u: { break; } default: {} } workgroupBarrier(); break; }",
    "loop { loop { workgroupBarrier(); break; } break; }",
    "var i = 0u; loop { if group.x == 0u { continue; } i += 1u; continuing { workgroupBarrier(); break if i > 2u; } }",
    "loop { if lane == 0u { break; } else { break; } } workgroupBarrier();",
])
func loopUniformityAcceptsWorkgroupControl(_ body: String) throws {
    _ = try IndependentShaderCompiler().parseAndValidate(workgroupLoopSource(body))
}

@Test(arguments: [
    "loop { workgroupBarrier(); if lane == 0u { break; } }",
    "loop { if lane == 0u { continue; } workgroupBarrier(); break; }",
    "loop { workgroupBarrier(); continuing { break if lane == 0u; } }",
    "var value = 0u; loop { if value == 0u { workgroupBarrier(); } value = lane; continuing { break if group.x == 0u; } }",
    "var a = 0u; var b = 0u; loop { if a == 0u { workgroupBarrier(); } a = b; b = lane; continuing { break if group.x == 0u; } }",
    "var slot: u32; loop { if lane == 0u { slot = 1u; break; } break; } if slot == 0u { workgroupBarrier(); }",
    "var i = 0u; loop { switch lane { case 0u: { continue; } default: {} } workgroupBarrier(); continuing { i++; break if i > 2u; } }",
])
func loopUniformityRejectsControlThatCanDiverge(_ body: String) {
    do {
        _ = try IndependentShaderCompiler().parseAndValidate(workgroupLoopSource(body))
        Issue.record("Accepted a divergent barrier")
    } catch ShaderFailure.diagnostics(let issues) {
        #expect(issues.contains { $0.code == .nonUniformControlFlow })
    } catch {
        Issue.record("Unexpected validation failure: \(error)")
    }
}

@Test func loopUniformityPropagatesAHelperArgumentRequirement() throws {
    let helper = """
        fn repeatUniform(limit: u32) {
            for (var i = 0u; i < limit; i++) { workgroupBarrier(); }
        }
        """
    let compiler = IndependentShaderCompiler()
    _ = try compiler.parseAndValidate(workgroupLoopSource("repeatUniform(group.x);", helpers: helper))
    do {
        _ = try compiler.parseAndValidate(workgroupLoopSource("repeatUniform(lane);", helpers: helper))
        Issue.record("Accepted a varying loop bound")
    } catch ShaderFailure.diagnostics(let issues) {
        #expect(issues.contains { $0.code == .nonUniformControlFlow })
    } catch {
        Issue.record("Unexpected validation failure: \(error)")
    }
}

@Test func loopUniformityChecksBindingIndicesWithTheBackedgeConnected() throws {
    let declaration = """
        enable wgpu_binding_array;
        @group(0) @binding(0) var images: binding_array<texture_2d<f32>, 2>;
        """
    let compiler = IndependentShaderCompiler()
    let capability = ShaderCapabilities.textureAndSamplerBindingArray
    let uniform = workgroupLoopSource("""
        for (var i = 0u; i < 2u; i++) { _ = textureDimensions(images[i]); }
        """, helpers: declaration)
    _ = try compiler.parseAndValidate(uniform, capabilities: capability)
    let varying = workgroupLoopSource("""
        var i = 0u;
        var index = 0u;
        loop {
            _ = textureDimensions(images[index]);
            index = lane % 2u;
            continuing { i++; break if i >= 2u; }
        }
        """, helpers: declaration)
    _ = try compiler.parseAndValidate(varying, capabilities: capability.union(.textureAndSamplerBindingArrayNonUniformIndexing))
    do {
        _ = try compiler.parseAndValidate(varying, capabilities: capability)
        Issue.record("Accepted varying resource indexing without the capability")
    } catch ShaderFailure.diagnostics(let issues) {
        #expect(issues.contains { $0.code == .unsupportedCapability })
    } catch {
        Issue.record("Unexpected validation failure: \(error)")
    }
}

@Test func loopUniformityKeepsSubgroupAndWorkgroupControlSeparate() throws {
    let source = """
        requires subgroup_id, subgroup_uniformity;
        enable subgroups;
        @compute @workgroup_size(128) fn run(@builtin(subgroup_id) group: u32) {
            for (var i = 0u; i < group; i++) { _ = subgroupAdd(1u); }
        }
        """
    let compiler = IndependentShaderCompiler()
    let capabilities = ShaderCapabilities.referenceDefault.union(.subgroup)
    _ = try compiler.parseAndValidate(.wgsl(source), capabilities: capabilities)
    #expect(throws: ShaderFailure.self) {
        try compiler.parseAndValidate(.wgsl("""
            requires subgroup_id, subgroup_uniformity;
            enable subgroups;
            @compute @workgroup_size(128) fn run(@builtin(subgroup_id) group: u32) {
                for (var i = 0u; i < group; i++) { workgroupBarrier(); }
            }
            """), capabilities: capabilities)
    }
}

#if canImport(Metal)
@Test func loopUniformitySharesValuesAcrossIterationsOnMetal() async throws {
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(sharedLoopShader), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let output = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 16, usage: [.mapRead, .copyDestination]))
        let result = try await device.submitCommands([
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
            .copy(source: output, destination: readback, byteCount: 16),
            .read(buffer: readback, byteCount: 16),
        ]).wait()
        #expect(result == [[22, 24, 26, 20]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
