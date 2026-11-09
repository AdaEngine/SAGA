import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

private func storageMemoryGLSLInput(_ qualifiers: String) -> ShaderInput {
    .glsl("""
    #version 450
    layout(local_size_x=2) in;
    layout(std430, set=0, binding=0) \(qualifiers) buffer Mailbox { uint values[4]; } mailbox;
    layout(std430, set=0, binding=1) buffer Result { uint values[4]; } outputData;

    void main() {
        uint lane = gl_LocalInvocationIndex;
        uint group = gl_WorkGroupID.x;
        uint base = group * 2u;
        mailbox.values[base + lane] = 100u * group + lane + 10u;
        outputData.values[base + lane] = mailbox.values[base + lane] + 3u;
    }
    """, stage: .compute, defines: [:])
}

private func exportCoherentStorage(_ compiler: IndependentShaderCompiler) throws -> [UInt32] {
    guard case .spirv(let words) = try compiler.compile(storageMemoryGLSLInput("coherent"), to: .spirv) else {
        throw ShaderFailure.incompatibleArtifact
    }
    return words
}

@Test(arguments: [ShaderOutputFormat.msl, .wgsl, .hlsl, .glsl])
func spirvCoherentStorageCodeGeneratorPreservesMemoryDecoration(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let words = try exportCoherentStorage(compiler)
    try #require(words.count > 5)
    var coherent: Set<UInt32> = []
    var bindings: [UInt32: UInt32] = [:]
    var offset = 5
    while offset < words.count {
        let count = Int(words[offset] >> 16)
        try #require(count > 0 && count <= words.count - offset)
        if words[offset] & 0xffff == 71 { // OpDecorate
            if count == 3 && words[offset + 2] == 23 { coherent.insert(words[offset + 1]) }
            if count == 4 && words[offset + 2] == 33 { bindings[words[offset + 3]] = words[offset + 1] }
        }
        offset += count
    }
    let mailbox = try #require(bindings[0])
    #expect(coherent == [mailbox])
    let input = ShaderInput.spirv(words)
    let module = try compiler.parseAndValidate(input).module
    let data = try #require(module.globalVariables.values.first { $0.binding == .init(group: 0, binding: 0) })
    let result = try #require(module.globalVariables.values.first { $0.binding == .init(group: 0, binding: 1) })
    #expect(data.memoryDecorations == .coherent && result.memoryDecorations.isEmpty)
    let shader = try compiler.compileCompute(input, to: format)
    guard case .text(let actual, let text) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
    #expect(actual == format)
    if format == .hlsl { #expect(text.contains("globallycoherent ")) }
    if format == .glsl { #expect(text.contains("coherent buffer")) }
    if format == .wgsl {
        _ = try compiler.parseAndValidate(.wgsl(text), capabilities: [.memoryDecorationCoherent])
    }
}

extension SPIRVFrontendTests {
    // glslang decorates block members individually for a GLSL coherent buffer block.
    static func storageMemberMemoryModule(decoration: UInt32, both: Bool) -> [UInt32] {
        var words = pre
        for instruction in [
            i(15, 5, 30, 0x6e69616d, 0), i(16, 30, 17, 1, 1, 1), i(71, 3, 2),
            i(72, 3, 0, 35, 0), i(72, 3, 1, 35, 4), i(72, 3, 0, decoration),
        ] { words += instruction }
        if both { words += i(72, 3, 1, decoration) }
        for instruction in [
            i(71, 20, 34, 0), i(71, 20, 33, 0), i(19, 1), i(21, 2, 32, 0), i(30, 3, 2, 2),
            i(32, 4, 12, 3), i(32, 5, 12, 2), i(33, 6, 1), i(43, 2, 10, 0), i(43, 2, 11, 1),
            i(43, 2, 12, 42), i(59, 4, 20, 12), i(54, 1, 30, 0, 6), i(248, 31),
            i(65, 5, 32, 20, 10), i(62, 32, 12), i(65, 5, 33, 20, 11), i(62, 33, 12), i(253), i(56),
        ] { words += instruction }
        return words
    }
}

@Test func spirvCoherentStorageCodeGeneratorImportsFullyCoherentBlockMembers() throws {
    let compiler = IndependentShaderCompiler()
    let input = ShaderInput.spirv(SPIRVFrontendTests.storageMemberMemoryModule(decoration: 23, both: true))
    let module = try compiler.parseAndValidate(input).module
    let buffer = try #require(module.globalVariables.values.first { $0.binding == .init(group: 0, binding: 0) })
    #expect(buffer.memoryDecorations == .coherent)
}

@Test(arguments: [UInt32(21), 23])
func spirvCoherentStorageCodeGeneratorRejectsPartialMemberQualifiers(_ decoration: UInt32) throws {
    let compiler = IndependentShaderCompiler()
    let input = ShaderInput.spirv(SPIRVFrontendTests.storageMemberMemoryModule(decoration: decoration, both: false))
    do {
        _ = try compiler.parseAndValidate(input)
        Issue.record("Promoted a partial storage member decoration to the whole buffer")
    } catch ShaderFailure.diagnostics(let diagnostics) {
        #expect(diagnostics.contains { $0.code == .unsupportedConstruct })
    }
}

@Test func spirvCoherentStorageCodeGeneratorPreservesVolatileOnlyOnSupportingTargets() throws {
    let compiler = IndependentShaderCompiler()
    guard case .spirv(let words) = try compiler.compile(storageMemoryGLSLInput("coherent volatile"), to: .spirv) else {
        throw ShaderFailure.incompatibleArtifact
    }
    let input = ShaderInput.spirv(words)
    let module = try compiler.parseAndValidate(input).module
    let mailbox = try #require(module.globalVariables.values.first { $0.binding == .init(group: 0, binding: 0) })
    #expect(mailbox.memoryDecorations == [.coherent, .volatile])
    let glsl = try compiler.compileCompute(input, to: .glsl)
    guard case .text(.glsl, let text) = glsl.artifact else { throw ShaderFailure.incompatibleArtifact }
    #expect(text.contains("coherent ") && text.contains("volatile "))
    for format: ShaderOutputFormat in [.msl, .hlsl] {
        do {
            _ = try compiler.compileCompute(input, to: format)
            Issue.record("Compiled volatile storage for a target without a volatile storage qualifier")
        } catch ShaderFailure.unsupported(let message) {
            #expect(message.lowercased().contains("volatile"))
        }
    }
    #expect(throws: ShaderFailure.self) { try compiler.compileWebGPUCompute(input) }
}

@Test func spirvCoherentStorageCodeGeneratorReexportsCoherentBinaryWithBuiltinInputs() throws {
    let compiler = IndependentShaderCompiler()
    let input = ShaderInput.spirv(try exportCoherentStorage(compiler))
    guard case .spirv(let words) = try compiler.compile(input, to: .spirv) else {
        throw ShaderFailure.incompatibleArtifact
    }
    let module = try compiler.parseAndValidate(.spirv(words)).module
    let mailbox = try #require(module.globalVariables.values.first { $0.binding == .init(group: 0, binding: 0) })
    #expect(mailbox.memoryDecorations == .coherent)
}

#if canImport(Metal)
@Test func spirvCoherentStorageCodeGeneratorRetainsCoherentWritesOnMetal() async throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.spirv(try exportCoherentStorage(compiler)), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let mailbox = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
        let output = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 16, usage: [.copyDestination, .mapRead]))
        let snapshots = try await device.submitCommands([
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): mailbox, .init(binding: 1): output],
                             workgroups: .init(x: 2)),
            .copy(source: output, destination: readback, byteCount: 16), .read(buffer: readback, byteCount: 16),
            .copy(source: mailbox, destination: readback, byteCount: 16), .read(buffer: readback, byteCount: 16),
        ]).wait()
        #expect(snapshots == [[13, 14, 113, 114], [10, 11, 110, 111]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
