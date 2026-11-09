import SAGACompiler
import SAGAShaderIR
import Testing
#if canImport(Metal)
import SAGACore
import SAGAMetal
import SAGATypes
#endif

private let debugSource = ShaderInput.wgsl(#"""
    enable wgpu_debug_printf;
    @group(0) @binding(0) var<storage, read_write> output: u32;
    fn report(value: u32) {
        debugPrintf("saga \"debug\" 100%% id=%03u", value);
    }
    @compute @workgroup_size(1) fn run(@builtin(global_invocation_id) id: vec3u) {
        report(id.x + 7u);
        output = id.x + 41u;
    }
    @compute @workgroup_size(1) fn portable() { output = 19u; }
    """#)

@Test func shaderDebugPrintfPreservesLiteralAndOperandsAcrossWGSLAndSPIRV() throws {
    let compiler = IndependentShaderCompiler()
    guard case .spirv(let words) = try compiler.compile(debugSource, to: .spirv) else {
        throw ShaderFailure.incompatibleArtifact
    }
    for input in [debugSource, .spirv(words)] {
        let module = try compiler.parseAndValidate(input, capabilities: [.debugPrintf]).module
        let formats = module.functions.values.flatMap { $0.statements.values }.compactMap { statement -> String? in
            if case .debugPrintf(let format, let arguments) = statement.kind, arguments.count == 1 { return format }
            return nil
        }
        #expect(formats == [#"saga "debug" 100%% id=%03u"#])
        let native = try compiler.compileCompute(input, entryPoint: "run", to: .wgsl)
        guard case .text(.wgsl, let text) = native.artifact else { throw ShaderFailure.incompatibleArtifact }
        #expect(text.contains("enable wgpu_debug_printf;") && text.contains("debugPrintf("))
        _ = try compiler.parseAndValidate(.wgsl(text), capabilities: [.debugPrintf])
        let metal = try compiler.compileCompute(input, entryPoint: "run", to: .msl)
        #expect(metal.requiresDebugPrintf)
        guard case .text(.msl, let source) = metal.artifact else { throw ShaderFailure.incompatibleArtifact }
        #expect(source.contains("metal::os_log_default.log_info("))
        #expect(throws: ShaderFailure.self) { try compiler.compileWebGPUCompute(input, entryPoint: "run") }
        let web = try compiler.compileWebGPUCompute(input, entryPoint: "portable")
        #expect(!web.requiresDebugPrintf)
        for format in [ShaderOutputFormat.hlsl, .glsl] {
            #expect(throws: ShaderFailure.self) { try compiler.compileCompute(input, entryPoint: "run", to: format) }
        }
    }
}

@Test(arguments: [#"debugPrintf("%n", 1i);"#, #"debugPrintf("%u", 1i);"#,
                  #"debugPrintf("%u %f", 1u);"#, #"debugPrintf("plain", 1u);"#,
                  #"debugPrintf("%u", vec2u(1));"#, #"debugPrintf("bad\q");"#,
                  #"let x = debugPrintf("no value");"#, #"debugPrintf("dangling %");"#])
func shaderDebugPrintfRejectsInvalidFormatsTypesAndValueUse(_ body: String) {
    let input = ShaderInput.wgsl("enable wgpu_debug_printf; @compute @workgroup_size(1) fn run() { \(body) }")
    #expect(throws: ShaderFailure.self) {
        try IndependentShaderCompiler().parseAndValidate(input, capabilities: [.debugPrintf])
    }
}

@Test func shaderDebugPrintfRequiresDirectiveAndCapability() {
    let compiler = IndependentShaderCompiler()
    let missingEnable = ShaderInput.wgsl(#"@compute @workgroup_size(1) fn run() { debugPrintf("test"); }"#)
    #expect(throws: ShaderFailure.self) { try compiler.parseAndValidate(debugSource) }
    #expect(throws: ShaderFailure.self) { try compiler.parseAndValidate(missingEnable, capabilities: [.debugPrintf]) }
}

@Test func shaderDebugPrintfSPIRVRequiresVoidResultAndAStringID() throws {
    let compiler = IndependentShaderCompiler()
    guard case .spirv(let words) = try compiler.compile(debugSource, to: .spirv) else {
        throw ShaderFailure.incompatibleArtifact
    }
    var offset = 5
    while offset < words.count, words[offset] & 0xffff != 12 {
        let count = Int(words[offset] >> 16)
        try #require(count > 0 && count <= words.count - offset)
        offset += count
    }
    try #require(offset + 5 < words.count)
    var invalidFormat = words
    invalidFormat[offset + 5] = words[offset + 1] // The result type is not an OpString.
    var invalidResult = words
    invalidResult[offset + 1] = words[offset + 5] // The format ID is not an OpTypeVoid.
    for input in [invalidFormat, invalidResult] {
        #expect(throws: ShaderFailure.self) { try compiler.parseAndValidate(.spirv(input), capabilities: [.debugPrintf]) }
    }
}

#if canImport(Metal)
@Test func shaderDebugPrintfLogsAndRetainsStorageSideEffectsOnMetal() async throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(debugSource, entryPoint: "run", to: .msl)
    let base = try MetalBackend.makeDevice()
    guard base.capabilities.features.contains(.shaderDebugPrintf) else { return }
    let (messages, continuation) = AsyncStream<String>.makeStream(bufferingPolicy: .bufferingNewest(16))
    defer { continuation.finish() }
    let backend = try MetalBackend.makeDevice(shaderLogHandler: { _ = continuation.yield($0) })
    let device = try await Device.make(backend: backend)
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let output = try await device.createBuffer(.init(size: 4, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 4, usage: [.copyDestination, .mapRead]))
        let words = try await device.submitCommands([
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
            .copy(source: output, destination: readback, byteCount: 4), .read(buffer: readback, byteCount: 4),
        ]).wait()
        #expect(words == [[41]])
        let logged = await withTaskGroup(of: Bool.self) { group in
            group.addTask {
                for await message in messages where message.contains(#"saga "debug" 100% id=007"#) { return true }
                return false
            }
            group.addTask { try? await Task.sleep(for: .seconds(3)); return false }
            let result = await group.next() ?? false
            group.cancelAll()
            return result
        }
        #expect(logged)
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
