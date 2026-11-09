import SAGACompiler
import SAGAShaderIR
import Testing
#if canImport(Metal)
import SAGACore
import SAGAMetal
import SAGATypes
#endif

private let memoryCapabilities: ShaderCapabilities = [.memoryDecorationCoherent, .memoryDecorationVolatile]
private let storageMemorySource = ShaderInput.wgsl("""
    @group(0) @binding(0) @coherent var<storage, read_write> first: array<u32, 2>;
    @group(0) @binding(1) @volatile var<storage, read_write> second: array<u32, 2>;
    @group(0) @binding(2) @coherent @volatile var<storage, read_write> third: array<u32, 2>;
    @group(0) @binding(3) var<storage, read_write> plain: array<u32, 2>;
    @compute @workgroup_size(1) fn run() {
        first[0] = second[0];
        third[0] = plain[0];
    }
    """)

@Test func wgslStorageMemoryPreservesIndependentQualifiersAcrossTargets() throws {
    let compiler = IndependentShaderCompiler()
    let expected: [IRMemoryDecorations] = [.coherent, .volatile, [.coherent, .volatile], []]
    let module = try compiler.parseAndValidate(storageMemorySource, capabilities: memoryCapabilities).module
    #expect(module.globalVariables.values.map(\.memoryDecorations) == expected)
    guard case .spirv(let words) = try compiler.compile(storageMemorySource, to: .spirv) else {
        throw ShaderFailure.incompatibleArtifact
    }
    let imported = ShaderInput.spirv(words)
    let importedModule = try compiler.parseAndValidate(imported).module
    #expect(importedModule.globalVariables.values.map(\.memoryDecorations) == expected)
    for source in [storageMemorySource, imported] {
        let wgsl = try compiler.compileCompute(source, to: .wgsl)
        guard case .text(.wgsl, let text) = wgsl.artifact else { throw ShaderFailure.incompatibleArtifact }
        #expect(try compiler.parseAndValidate(.wgsl(text), capabilities: memoryCapabilities)
            .module.globalVariables.values.map(\.memoryDecorations) == expected)
        let glsl = try compiler.compileCompute(source, to: .glsl)
        guard case .text(.glsl, let text) = glsl.artifact else { throw ShaderFailure.incompatibleArtifact }
        #expect(text.contains(") coherent buffer") && text.contains(") volatile buffer") &&
                text.contains(") coherent volatile buffer"))
        #expect(throws: ShaderFailure.self) { try compiler.compileWebGPUCompute(source) }
        #expect(throws: ShaderFailure.self) { try compiler.compileCompute(source, to: .msl) }
        #expect(throws: ShaderFailure.self) { try compiler.compileCompute(source, to: .hlsl) }
    }
}

@Test func wgslStorageMemoryRequiresCapabilitiesAndStorageGlobals() throws {
    let compiler = IndependentShaderCompiler()
    for capabilities: ShaderCapabilities in [[], [.memoryDecorationCoherent], [.memoryDecorationVolatile]] {
        #expect(throws: ShaderFailure.self) { try compiler.parseAndValidate(storageMemorySource, capabilities: capabilities) }
    }
    for text in [
        "@coherent var<private> invalid: u32;",
        "@volatile var<workgroup> invalid: u32;",
        "@group(0) @binding(0) @coherent var<uniform> invalid: vec4u;",
        "@group(0) @binding(0) @volatile var invalid: texture_storage_2d<r32uint, write>;",
        "@group(0) @binding(0) @coherent() var<storage> invalid: u32;",
        "@group(0) @binding(0) @volatile @volatile var<storage> invalid: u32;",
    ] {
        #expect(throws: ShaderFailure.self) { try compiler.parseAndValidate(.wgsl(text), capabilities: memoryCapabilities) }
    }
}

private let coherentStorageSource = ShaderInput.wgsl("""
    @group(0) @binding(0) @coherent var<storage, read_write> mailbox: array<u32, 2>;
    @group(0) @binding(1) var<storage, read_write> output: array<u32, 2>;
    @compute @workgroup_size(2) fn run(@builtin(local_invocation_index) lane: u32) {
        mailbox[lane] = 17u + lane;
        storageBarrier();
        output[lane] = mailbox[1u - lane];
    }
    """)

@Test func wgslCoherentStorageUsesTargetSpecificQualifiers() throws {
    let compiler = IndependentShaderCompiler()
    for (format, spelling): (ShaderOutputFormat, String) in [(.msl, "coherent device"), (.hlsl, "globallycoherent")] {
        let shader = try compiler.compileCompute(coherentStorageSource, to: format)
        guard case .text(_, let text) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
        #expect(text.contains(spelling))
        #expect(shader.storageBindings.map(\.memoryDecorations) == [.coherent, []])
    }
    #expect(throws: ShaderFailure.unsupported("Browser WGSL does not support native storage memory attributes")) {
        try compiler.compileWebGPUCompute(coherentStorageSource)
    }
    guard case .spirv(let words) = try compiler.compile(coherentStorageSource, to: .spirv) else {
        throw ShaderFailure.incompatibleArtifact
    }
    let browser = try compiler.compileWebGPUCompute(.spirv(words))
    guard case .text(.wgsl, let text) = browser.artifact else { throw ShaderFailure.incompatibleArtifact }
    #expect(!text.contains("@coherent") && text.contains("storageBarrier()"))
    _ = try compiler.parseAndValidate(.wgsl(text))
}

#if canImport(Metal)
@Test func wgslCoherentStorageReadbackUsesTheMetalFeature() async throws {
    let shader = try IndependentShaderCompiler().compileCompute(coherentStorageSource, to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        if device.capabilities.features.contains(.memoryDecorationCoherent) {
            let pipeline = try await device.createComputePipeline(shader)
            let shared = try await device.createBuffer(.init(size: 8, usage: .storage))
            let output = try await device.createBuffer(.init(size: 8, usage: [.storage, .copySource]))
            let read = try await device.createBuffer(.init(size: 8, usage: [.copyDestination, .mapRead]))
            let values = try await device.submitCommands([
                .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): shared, .init(binding: 1): output],
                                 workgroups: .init(x: 1)),
                .copy(source: output, destination: read, byteCount: 8), .read(buffer: read, byteCount: 8),
            ]).wait()
            #expect(values == [[18, 17]])
        } else {
            await #expect(throws: GPUError.validation("Feature not advertised")) {
                try await device.createComputePipeline(shader)
            }
        }
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
