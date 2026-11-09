import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

private let spirvBufferArrayInput = ShaderInput.wgsl("""
    enable wgpu_binding_array;
    struct Settings { data: vec4u }
    struct Record { head: u32, tail: array<u32> }
    @group(0) @binding(0) var<uniform> settings: binding_array<Settings>;
    @group(0) @binding(1) var<storage, read_write> records: binding_array<Record>;
    @group(0) @binding(2) var<storage, read_write> output: array<vec4u, 2>;

    @compute @workgroup_size(2) fn run(@builtin(local_invocation_index) lane: u32) {
        let config = settings[lane].data;
        let count = arrayLength(&records[lane].tail);
        let previous = records[lane].tail[0];
        output[lane] = vec4u(config.x, count, records[lane].head, previous);
        records[lane].tail[0] = previous + config.y;
    }
    """)

private let spirvBufferArrayLayout: [IRResourceBinding: UInt32] = [
    .init(group: 0, binding: 0): 2,
    .init(group: 0, binding: 1): 2,
]

private func exportBufferArrays(_ compiler: IndependentShaderCompiler) throws -> [UInt32] {
    guard case .spirv(let words) = try compiler.compile(spirvBufferArrayInput, to: .spirv) else {
        throw ShaderFailure.incompatibleArtifact
    }
    return words
}

private func expectRuntimeBufferArrayMetadata(_ words: [UInt32]) throws {
    try #require(words.count > 5)
    var capabilities: Set<UInt32> = []
    var runtimeArrays: Set<UInt32> = []
    var strides: [UInt32: UInt32] = [:]
    var lengthMembers: [UInt32] = []
    var offset = 5
    while offset < words.count {
        let count = Int(words[offset] >> 16)
        try #require(count > 0 && count <= words.count - offset)
        switch words[offset] & 0xffff {
        case 17 where count == 2: capabilities.insert(words[offset + 1])
        case 29 where count == 3: runtimeArrays.insert(words[offset + 1]) // OpTypeRuntimeArray
        case 68 where count == 5: lengthMembers.append(words[offset + 4]) // OpArrayLength
        case 71 where count == 4 && words[offset + 2] == 6: strides[words[offset + 1]] = words[offset + 3] // ArrayStride
        default: break
        }
        offset += count
    }
    #expect(capabilities.isSuperset(of: [5301, 5302, 5306, 5308]))
    #expect(runtimeArrays.count == 3)
    #expect(runtimeArrays.compactMap { strides[$0] } == [4])
    #expect(Set(lengthMembers) == [1]) // Explicit queries and implicit bounds use the same tail member.
}

@Test(arguments: [ShaderOutputFormat.msl, .wgsl, .hlsl, .glsl])
func spirvBufferDescriptorArrayCodeGeneratorKeepsDescriptorCountsAndDataStridesSeparate(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let words = try exportBufferArrays(compiler)
    try expectRuntimeBufferArrayMetadata(words)
    let input = ShaderInput.spirv(words)
    let module = try compiler.parseAndValidate(input, capabilities: .allKnown).module
    for binding in spirvBufferArrayLayout.keys {
        let global = try #require(module.globalVariables.values.first { $0.binding == binding })
        guard case .bindingArray(_, .dynamic) = try module.types.value(for: global.type).kind else {
            Issue.record("Expected an unbounded buffer descriptor array at binding \(binding.binding)")
            continue
        }
    }
    do {
        _ = try compiler.compileCompute(input, to: format)
        Issue.record("Compiled a runtime buffer descriptor array without its layout")
    } catch ShaderFailure.diagnostics(let diagnostics) {
        #expect(diagnostics.contains { $0.code == .invalidBinding })
    }
    let shader = try compiler.compileCompute(input, to: format, bindingArrayLengths: spirvBufferArrayLayout)
    let records = try #require(shader.storageBindings.first { $0.binding == .init(group: 0, binding: 1) })
    #expect(shader.uniformBindings.map { $0.bindingArray?.count } == [2])
    #expect(records.bindingArray?.count == 2 && !records.isReadOnly)
    #expect(records.runtimeArrayOffset == 4 && records.runtimeArrayStride == 4)
    if format == .wgsl {
        guard case .text(.wgsl, let text) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
        _ = try compiler.parseAndValidate(.wgsl(text), capabilities: .allKnown)
    }
}

#if canImport(Metal)
@Test func spirvBufferDescriptorArrayCodeGeneratorSpecializesBufferLayoutAndGuardsBoundsOnMetal() async throws {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl("""
        enable wgpu_binding_array;
        struct Settings { data: vec4u }
        struct Record { head: u32, tail: array<u32> }
        @group(0) @binding(0) var<uniform> settings: binding_array<Settings>;
        @group(0) @binding(1) var<storage, read_write> records: binding_array<Record>;
        @group(0) @binding(2) var<storage, read_write> output: array<vec4u, 5>;

        @compute @workgroup_size(5) fn run(@builtin(local_invocation_index) lane: u32) {
            let index = i32(lane) - 1i;
            let record = bitcast<u32>(index);
            let config = settings[index];
            let count = arrayLength(&records[record].tail);
            let previous = records[record].tail[0];
            output[lane] = vec4u(config.data.x, count, records[record].head, previous);
            records[record].tail[0] = previous + config.data.y + 7u;
        }
        """)
    let settingsBinding = IRResourceBinding(group: 0, binding: 0)
    let recordsBinding = IRResourceBinding(group: 0, binding: 1)
    let layout: [IRResourceBinding: UInt32] = [settingsBinding: 2, recordsBinding: 3]
    guard case .spirv(let words) = try compiler.compile(source, to: .spirv, bindingArrayLengths: layout) else {
        throw ShaderFailure.incompatibleArtifact
    }
    do {
        _ = try compiler.compile(source, to: .spirv, bindingArrayLengths: [settingsBinding: 2])
        Issue.record("Accepted a partial buffer descriptor layout")
    } catch ShaderFailure.diagnostics(let diagnostics) {
        #expect(diagnostics.contains { $0.code == .invalidBinding })
    }
    let shader = try compiler.compileCompute(.spirv(words), to: .msl)
    let recordsLayout = try #require(shader.storageBindings.first { $0.binding == recordsBinding })
    #expect(shader.uniformBindings.map { $0.bindingArray?.count } == [2])
    #expect(recordsLayout.bindingArray?.count == 3)
    #expect(recordsLayout.runtimeArrayOffset == 4 && recordsLayout.runtimeArrayStride == 4)

    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let settings = try await device.createBuffer(.init(size: 512, usage: [.uniform, .copyDestination]))
        let records = try await device.createBuffer(.init(size: 768, usage: [.storage, .copyDestination, .copySource]))
        let output = try await device.createBuffer(.init(size: 80, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 80, usage: [.copyDestination, .mapRead]))
        let snapshots = try await device.submitCommands([
            .write(buffer: settings, offset: 0, words: [11, 55, 0, 0]),
            .write(buffer: settings, offset: 256, words: [22, 77, 0, 0]),
            .write(buffer: records, offset: 0, words: [101, 5]),
            .write(buffer: records, offset: 256, words: [202, 7, 8, 9]),
            .write(buffer: records, offset: 512, words: [303, 13, 14]),
            .dispatchComputeRanges(pipeline: pipeline, bindings: [
                .init(binding: 0, arrayElement: 0): .init(buffer: settings, size: 16),
                .init(binding: 0, arrayElement: 1): .init(buffer: settings, offset: 256, size: 16),
                .init(binding: 1, arrayElement: 0): .init(buffer: records, size: 8),
                .init(binding: 1, arrayElement: 1): .init(buffer: records, offset: 256, size: 16),
                .init(binding: 1, arrayElement: 2): .init(buffer: records, offset: 512, size: 12),
                .init(binding: 2): .init(buffer: output),
            ], workgroups: .init(x: 1)),
            .copy(source: output, destination: readback, byteCount: 80), .read(buffer: readback, byteCount: 80),
            .copy(source: records, destination: readback, byteCount: 8), .read(buffer: readback, byteCount: 8),
            .copy(source: records, sourceOffset: 256, destination: readback, byteCount: 16), .read(buffer: readback, byteCount: 16),
            .copy(source: records, sourceOffset: 512, destination: readback, byteCount: 12), .read(buffer: readback, byteCount: 12),
        ]).wait()
        let expectedOutput: [UInt32] = [0, 0, 0, 0, 11, 1, 101, 5, 22, 3, 202, 7, 0, 2, 303, 13, 0, 0, 0, 0]
        #expect(snapshots == [expectedOutput, [101, 67], [202, 91, 8, 9], [303, 20, 14]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}

@Test func spirvBufferDescriptorArrayCodeGeneratorRoundTripsFixedWholeRecordsOnMetal() async throws {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl("""
        enable wgpu_binding_array;
        struct R { fields: vec4u }
        @group(0) @binding(0) var<uniform> settings: binding_array<R, 2>;
        @group(0) @binding(1) var<storage, read_write> records: binding_array<R, 2>;

        @compute @workgroup_size(2) fn run(@builtin(local_invocation_index) lane: u32) {
            let config = settings[1u - lane];
            let previous = records[lane];
            records[lane] = R(previous.fields + config.fields);
        }
        """)
    guard case .spirv(let words) = try compiler.compile(source, to: .spirv) else { throw ShaderFailure.incompatibleArtifact }
    let shader = try compiler.compileCompute(.spirv(words), to: .msl)
    #expect(shader.uniformBindings.map { $0.bindingArray?.count } == [2])
    #expect(shader.storageBindings.map { $0.bindingArray?.count } == [2])
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let settings = try await device.createBuffer(.init(size: 512, usage: [.uniform, .copyDestination]))
        let records = try await device.createBuffer(.init(size: 512, usage: [.storage, .copyDestination, .copySource]))
        let readback = try await device.createBuffer(.init(size: 16, usage: [.copyDestination, .mapRead]))
        let snapshots = try await device.submitCommands([
            .write(buffer: settings, offset: 0, words: [100, 200, 300, 400]),
            .write(buffer: settings, offset: 256, words: [10, 20, 30, 40]),
            .write(buffer: records, offset: 0, words: [5, 6, 7, 8]),
            .write(buffer: records, offset: 256, words: [1, 2, 3, 4]),
            .dispatchComputeRanges(pipeline: pipeline, bindings: [
                .init(binding: 0, arrayElement: 0): .init(buffer: settings, size: 16),
                .init(binding: 0, arrayElement: 1): .init(buffer: settings, offset: 256, size: 16),
                .init(binding: 1, arrayElement: 0): .init(buffer: records, size: 16),
                .init(binding: 1, arrayElement: 1): .init(buffer: records, offset: 256, size: 16),
            ], workgroups: .init(x: 1)),
            .copy(source: records, destination: readback, byteCount: 16), .read(buffer: readback, byteCount: 16),
            .copy(source: records, sourceOffset: 256, destination: readback, byteCount: 16), .read(buffer: readback, byteCount: 16),
        ]).wait()
        #expect(snapshots == [[15, 26, 37, 48], [101, 202, 303, 404]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}

@Test func spirvBufferDescriptorArrayCodeGeneratorUsesEachBuffersTailLengthOnMetal() async throws {
    let compiler = IndependentShaderCompiler()
    let input = ShaderInput.spirv(try exportBufferArrays(compiler))
    let shader = try compiler.compileCompute(input, to: .msl, bindingArrayLengths: spirvBufferArrayLayout)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let settings = try await device.createBuffer(.init(size: 512, usage: [.uniform, .copyDestination]))
        let records = try await device.createBuffer(.init(size: 512, usage: [.storage, .copyDestination, .copySource]))
        let output = try await device.createBuffer(.init(size: 32, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 32, usage: [.copyDestination, .mapRead]))
        let snapshots = try await device.submitCommands([
            .write(buffer: settings, offset: 0, words: [11, 55, 0, 0]),
            .write(buffer: settings, offset: 256, words: [22, 77, 0, 0]),
            .write(buffer: records, offset: 0, words: [101, 5]),
            .write(buffer: records, offset: 256, words: [202, 7, 8, 9]),
            .dispatchComputeRanges(pipeline: pipeline, bindings: [
                .init(binding: 0, arrayElement: 0): .init(buffer: settings, size: 16),
                .init(binding: 0, arrayElement: 1): .init(buffer: settings, offset: 256, size: 16),
                .init(binding: 1, arrayElement: 0): .init(buffer: records, size: 8),
                .init(binding: 1, arrayElement: 1): .init(buffer: records, offset: 256, size: 16),
                .init(binding: 2): .init(buffer: output),
            ], workgroups: .init(x: 1)),
            .copy(source: output, destination: readback, byteCount: 32), .read(buffer: readback, byteCount: 32),
            .copy(source: records, destination: readback, byteCount: 8), .read(buffer: readback, byteCount: 8),
            .copy(source: records, sourceOffset: 256, destination: readback, byteCount: 16), .read(buffer: readback, byteCount: 16),
        ]).wait()
        #expect(snapshots == [[11, 1, 101, 5, 22, 3, 202, 7], [101, 60], [202, 84, 8, 9]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
