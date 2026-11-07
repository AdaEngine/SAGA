@main struct SAGAWebDiagnostic {
    static func main() {
        #if !os(WASI)
        print("Build SAGAWebDiagnostic with the Swift Wasm SDK and run it in a WebGPU browser.")
        #endif
    }
}

#if os(WASI) && compiler(>=6.3)
import SAGATypes
import SAGABackend
import SAGACore
import SAGAShaderIR
import SAGACompiler
import SAGAWebGPU

func runWebDiagnostic(capabilities: DeviceCapabilities, supportsPresentation: Bool) async throws -> String {
    let source = """
        @group(0) @binding(1) var<storage, read> input: array<u32, 4>;
        @group(0) @binding(3) var<storage, read_write> output: array<u32, 4>;
        fn baseValue(index: u32) -> u32 { return input[index] * 3u + 7u; }
        fn writeValue(index: u32, value: u32) { output[index] = value; }
        fn writeAndRead(index: u32, value: u32) -> u32 {
            writeValue(index, value);
            return output[index];
        }
        @compute @workgroup_size(2, 2)
        fn run(@builtin(global_invocation_id) id: vec3<u32>) {
            let i = id.x + id.y * 2u;
            var bonus: u32;
            let base = bonus;
            if id.x == 1u && id.y == 0u {
                bonus = 1u;
            } else if id.y > 0u {
                bonus = 2u;
            }
            var progress = 0u;
            loop {
                if progress == bonus { break; }
                continue;
                continuing {
                    progress += 1u;
                    break if progress >= bonus;
                }
            }
            let original = writeAndRead(i, baseValue(i));
            writeValue(i, 0u);
            output[i] = original + base + progress + 1u;
        }
        """
    let host = WebGPUHost(capabilities: capabilities, supportsPresentation: supportsPresentation)
    let device = try await Device.make(backend: BrowserComputeBackend(host: host))
    do {
        let shader = try IndependentShaderCompiler().compileCompute(.wgsl(source), to: .wgsl)
        let pipeline = try await device.createComputePipeline(shader)
        let input = try await device.createBuffer(.init(size: 16, usage: [.storage, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
        let small = try await device.createBuffer(.init(size: 12, usage: .storage))
        let readback = try await device.createBuffer(.init(size: 16, usage: [.copyDestination, .mapRead]))
        let inputSlot = ComputeBindingSlot(binding: 1)
        let outputSlot = ComputeBindingSlot(binding: 3)
        let scope = try await device.pushErrorScope(.validation)
        do {
            _ = try await device.submitCommands([
                .dispatchCompute(pipeline: pipeline, bindings: [inputSlot: input, outputSlot: small],
                                 workgroups: .init(x: 1))
            ])
            throw GPUError.backendFailure("Swift accepted the undersized buffer")
        } catch GPUError.validation {}
        guard case .validation = try await device.popErrorScope(scope) else {
            throw GPUError.backendFailure("Swift validation scope did not capture the error")
        }
        let bindings = [inputSlot: input, outputSlot: output]
        let ticket = try await device.submitCommands([
            .write(buffer: input, offset: 0, words: [0, 1, 2, .max]),
            .dispatchCompute(pipeline: pipeline, bindings: bindings, workgroups: .init(x: 0)),
            .copy(source: output, destination: readback, byteCount: 16),
            .read(buffer: readback, byteCount: 16),
            .dispatchCompute(pipeline: pipeline, bindings: bindings, workgroups: .init(x: 1, y: 2)),
            .copy(source: output, destination: readback, byteCount: 16),
            .read(buffer: readback, byteCount: 16),
        ])
        try await device.destroyBuffer(input)
        let result = try await ticket.wait()
        guard result == [[0, 0, 0, 0], [8, 12, 16, 7]] else {
            throw GPUError.backendFailure("Unexpected WebGPU readback: \(result)")
        }
        let integerWords = try await integerDiagnostic(device)
        let runtimeArrayWords = try await runtimeArrayDiagnostic(device)
        let storageTailWords = try await storageTailDiagnostic(device)
        let storageVectors = try await storageVectorDiagnostic(device)
        let groupedStorage = try await groupedStorageDiagnostic(device)
        let builtinInvocations = try await computeBuiltinDiagnostic(device)
        let workgroupWords = try await workgroupMemoryDiagnostic(device)
        let storageBarrierWords = try await storageBarrierDiagnostic(device)
        let atomicTickets = try await atomicDiagnostic(device)
        let atomicRMWWords = try await atomicRMWDiagnostic(device)
        let atomicCompareWords = try await atomicCompareDiagnostic(device)
        let mixedAtomicWords = try await mixedAtomicDiagnostic(device)
        let uniformLoadWords = try await uniformLoadDiagnostic(device)
        let selectWords = try await selectDiagnostic(device)
        let constantWords = try await constantSelectDiagnostic(device)
        let uniformWords = try await computeUniformDiagnostic(device)
        let renderUniformPixels = try await renderUniformDiagnostic(device)
        let matrixValues = try await matrixDiagnostic(device)
        let matrixBufferValues = try await matrixBufferDiagnostic(device)
        let switchWords = try await switchDiagnostic(device)
        let textureUploadWords = try await textureUploadDiagnostic(device)
        let textureLoadWords = try await textureLoadDiagnostic(device)
        let renderTexturePixels = try await renderTextureDiagnostic(device)
        let samplerValues = try await samplerDiagnostic(device)
        let renderSamplerPixels = try await renderSamplerDiagnostic(device)
        let mipTextureWords = try await mipTextureDiagnostic(device)
        let textureViewWords = try await textureViewDiagnostic(device)
        let textureGatherLanes = try await textureGatherDiagnostic(device)
        let storageTextureWords = try await storageTextureDiagnostic(device)
        let fragmentStorageTextureWords = try await fragmentStorageTextureDiagnostic(device)
        let supportsReadableStorage = device.capabilities.features.contains(.storageTextureReadWrite)
        let readWriteStorageWords = supportsReadableStorage ? try await readWriteStorageDiagnostic(device) : []
        let textureBarrierWords = supportsReadableStorage ? try await textureBarrierDiagnostic(device) : []
        let renderReadableStorageWords = supportsReadableStorage ? try await renderReadableStorageDiagnostic(device) : []
        let depthAttachmentWords = try await depthAttachmentDiagnostic(device)
        let depthTextureWords = try await depthTextureDiagnostic(device)
        let depthComparisonWords = try await depthComparisonDiagnostic(device)
        let depthGatherLanes = try await depthGatherDiagnostic(device, comparison: true).map {
            UInt32(Float(bitPattern: $0))
        }
        let rawDepthGather = try await depthGatherDiagnostic(device, comparison: false).map {
            String(Float(bitPattern: $0))
        }.joined(separator: ",")
        let multisampleWords = try await multisampleDiagnostic(device)
        let integerTextureWords = try await integerTextureDiagnostic(device)
        let integerGatherWords = try await integerGatherDiagnostic(device)
        let sampleCoverageWords = try await sampleCoverageDiagnostic(device)
        let alphaCoverageWords = try await alphaCoverageDiagnostic(device)
        let multipleColorWords = try await multipleColorDiagnostic(device)
        let viewportWords = try await viewportDiagnostic(device)
        let depthBiasWords = try await depthBiasDiagnostic(device)
        let stencilWords = try await stencilDiagnostic(device, sampleCount: 1)
        let msaaStencilWords = try await stencilDiagnostic(device, sampleCount: 4)
        let readOnlyDepthWords = try await readOnlyDepthDiagnostic(device)
        let aspectWords = try await depthStencilAspectDiagnostic(device, sampleCount: 1)
        let msaaAspectWords = try await depthStencilAspectDiagnostic(device, sampleCount: 4)
        let stencilCopyWords = try await stencilCopyDiagnostic(device)
        let textureCopyWords = try await textureCopyDiagnostic(device)
        let arrayWords = try await arrayTextureDiagnostic(device)
        let layerViewWords = try await arrayLayerViewDiagnostic(device)
        let arraySamplingWords = try await arraySamplingDiagnostic(device)
        let storageArrayWords = try await storageArrayDiagnostic(device)
        let storageFormats = try await storageFormatDiagnostic(device)
        let cubeWords = try await cubeTextureDiagnostic(device)
        let volumeWords = try await volumeTextureDiagnostic(device)
        let fragmentDepthWords = try await fragmentDepthDiagnostic(device)
        let fragmentRasterWords = try await fragmentRasterDiagnostic(device)
        let interpolationWord = try await interpolationDiagnostic(device)
        let topologyWords = try await topologyDiagnostic(device)
        let blendWords = try await blendDiagnostic(device)
        let render = try await renderDiagnostic(device)
        await device.close()
        func csv(_ words: [UInt32]) -> String { words.map(String.init).joined(separator: ",") }
        let snapshots = result.map(csv)
        let metrics: [String] = [String(render.checked), csv(render.probe), String(integerWords), String(render.vertexStreams),
                       String(render.indexedVertices), String(render.instancesPerDraw), String(runtimeArrayWords),
                       String(storageTailWords), String(storageVectors), csv(groupedStorage),
                       String(builtinInvocations), csv(workgroupWords), csv(storageBarrierWords), String(atomicTickets),
                       String(atomicRMWWords), String(atomicCompareWords), String(mixedAtomicWords), csv(uniformLoadWords),
                       csv(selectWords), csv(constantWords), csv(uniformWords), csv(renderUniformPixels),
                       matrixValues.map { String($0) }.joined(separator: ","),
                       matrixBufferValues.map { String($0) }.joined(separator: ","), csv(switchWords),
                       csv(textureUploadWords), csv(textureLoadWords), csv(renderTexturePixels), csv(samplerValues),
                       csv(renderSamplerPixels), csv(mipTextureWords), csv(textureViewWords), csv(textureGatherLanes),
                       csv(storageTextureWords), csv(fragmentStorageTextureWords), csv(readWriteStorageWords),
                       csv(textureBarrierWords), csv(renderReadableStorageWords), csv(depthAttachmentWords), csv(depthTextureWords),
                       csv(depthComparisonWords), csv(depthGatherLanes), rawDepthGather, csv(multisampleWords), csv(integerTextureWords), csv(integerGatherWords), csv(sampleCoverageWords), csv(fragmentDepthWords), csv(fragmentRasterWords), String(interpolationWord), csv(topologyWords), csv(blendWords), csv(alphaCoverageWords), csv(multipleColorWords), csv(viewportWords), csv(depthBiasWords), csv(stencilWords + msaaStencilWords), csv(readOnlyDepthWords), csv(aspectWords + msaaAspectWords), csv(stencilCopyWords), csv(textureCopyWords), csv(arrayWords), csv(layerViewWords), csv(arraySamplingWords), csv(storageArrayWords), String(storageFormats), csv(cubeWords), csv(volumeWords)]
        return (["swift"] + snapshots + metrics).joined(separator: "|")
    } catch {
        await device.close()
        throw error
    }
}

private func integerDiagnostic(_ device: Device) async throws -> Int {
    let source = """
        @group(0) @binding(0) var<storage, read> input: array<u32, 6>;
        @group(0) @binding(1) var<storage, read_write> output: array<u32, 20>;
        @compute @workgroup_size(1)
        fn run() {
            let a = vec2u(input[0u], input[3u]);
            let b = vec2u(input[1u], input[4u]);
            let shift = vec2u(input[2u], input[5u]);
            let lhs = vec2i(a);
            let rhs = vec2i(b);
            let sum = vec2u(lhs + rhs);
            let neg = vec2u(-lhs);
            let div = vec2u(lhs / rhs);
            let rem = vec2u(lhs % rhs);
            let shr = vec2u(lhs >> shift);
            let shl = vec2u(lhs << shift);
            let udiv = a / b;
            let urem = a % b;
            let bias = vec2u(lhs + 1i);
            let cmp = lhs < rhs;
            output[0u] = sum.x; output[1u] = neg.x; output[2u] = div.x; output[3u] = rem.x;
            output[4u] = shr.x; output[5u] = shl.x; output[6u] = udiv.x; output[7u] = urem.x;
            output[8u] = sum.y; output[9u] = neg.y; output[10u] = div.y; output[11u] = rem.y;
            output[12u] = shr.y; output[13u] = shl.y; output[14u] = udiv.y; output[15u] = urem.y;
            output[16u] = bias.x; output[17u] = bias.y;
            if cmp.x { output[18u] = 1u; }
            if cmp.y { output[19u] = 1u; }
        }
        """
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(source), to: .wgsl)
    let pipeline = try await device.createComputePipeline(shader)
    let input = try await device.createBuffer(.init(size: 24, usage: [.storage, .copyDestination]))
    let output = try await device.createBuffer(.init(size: 80, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 80, usage: [.mapRead, .copyDestination]))
    let ticket = try await device.submitCommands([
        .write(buffer: input, offset: 0, words: [0x80000000, .max, 32, 0xfffffff9, 0, 63]),
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): input, .init(binding: 1): output],
                         workgroups: .init(x: 1)),
        .copy(source: output, destination: read, byteCount: 80),
        .read(buffer: read, byteCount: 80),
    ])
    let result = try await ticket.wait()
    let expected: [UInt32] = [
        0x7fffffff, 0x80000000, 0x80000000, 0, 0x80000000, 0x80000000, 0, 0x80000000,
        0xfffffff9, 7, 0xfffffff9, 0, .max, 0x80000000, 0xfffffff9, 0,
        0x80000001, 0xfffffffa, 1, 1,
    ]
    guard result == [expected] else { throw GPUError.backendFailure("Unexpected integer WebGPU readback: \(result)") }
    return expected.count
}

private func runtimeArrayDiagnostic(_ device: Device) async throws -> Int {
    let source = """
        @group(0) @binding(1) var<storage, read> src: array<u32>;
        @group(0) @binding(3) var<storage, read_write> dst: array<u32>;
        fn read(index: i32) -> u32 { return src[index]; }
        @compute @workgroup_size(1)
        fn run(@builtin(global_invocation_id) id: vec3u) {
            dst[id.x] = arrayLength(&src) * 100u + arrayLength(&dst) * 10u + read(i32(id.x) - 1i);
        }
        """
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(source), to: .wgsl)
    let pipeline = try await device.createComputePipeline(shader)
    let input = try await device.createBuffer(.init(size: 272, usage: [.storage, .copyDestination]))
    let output = try await device.createBuffer(.init(size: 280, usage: [.storage, .copySource, .copyDestination]))
    let read = try await device.createBuffer(.init(size: 280, usage: [.mapRead, .copyDestination]))
    let expected: [UInt32] = [350, 351, 352, 353, 350]
    let ticket = try await device.submitCommands([
        .write(buffer: input, offset: 256, words: [1, 2, 3, 777]),
        .write(buffer: output, offset: 256, words: [0, 0, 0, 0, 0, 999]),
        .dispatchComputeRanges(pipeline: pipeline, bindings: [
            .init(binding: 1): .init(buffer: input, offset: 256, size: 12),
            .init(binding: 3): .init(buffer: output, offset: 256, size: 20),
        ], workgroups: .init(x: 6)),
        .copy(source: output, destination: read, byteCount: 280),
        .read(buffer: read, byteCount: 280),
    ])
    let result = try await ticket.wait()
    guard result.count == 1, let words = result.first, words.count == 70,
          words.prefix(64).allSatisfy({ $0 == 0 }),
          words.suffix(6).elementsEqual(expected + [999])
    else { throw GPUError.backendFailure("Unexpected runtime array WebGPU readback: \(result)") }
    return expected.count
}

private func storageTailDiagnostic(_ device: Device) async throws -> Int {
    let source = """
        struct Input {
            @size(12) bias: u32,
            factors: array<u32, 2>,
            @align(32) values: array<u32>,
        }
        struct Output {
            count: u32,
            @align(16) values: array<u32>,
        }
        @group(0) @binding(1) var<storage, read> src: Input;
        @group(0) @binding(3) var<storage, read_write> dst: Output;
        fn measure() -> u32 { return arrayLength(&src.values) * 100u + arrayLength(&dst.values) * 10u + src.bias; }
        @compute @workgroup_size(1)
        fn run(@builtin(global_invocation_id) id: vec3u) {
            if id.x == 0u { dst.count = measure(); }
            dst.values[id.x] = measure() + src.factors[id.x] + src.values[i32(id.x) - 1i];
        }
        """
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(source), to: .wgsl)
    let pipeline = try await device.createComputePipeline(shader)
    let input = try await device.createBuffer(.init(size: 324, usage: [.storage, .copyDestination]))
    let output = try await device.createBuffer(.init(size: 296, usage: [.storage, .copySource, .copyDestination]))
    let read = try await device.createBuffer(.init(size: 296, usage: [.mapRead, .copyDestination]))
    let inputBinding = StorageBufferBinding(buffer: input, offset: 256, size: 64)
    let ticket = try await device.submitCommands([
        .write(buffer: input, offset: 256, words: [7, 888, 888, 10, 20, 888, 888, 888, 1, 2, 3, 0, 0, 0, 0, 0, 777]),
        .write(buffer: output, offset: 256, words: [0, 888, 888, 888, 0, 0, 0, 0, 0, 999]),
        .dispatchComputeRanges(pipeline: pipeline, bindings: [
            .init(binding: 1): inputBinding,
            .init(binding: 3): .init(buffer: output, offset: 256, size: 36),
        ], workgroups: .init(x: 6)),
        .copy(source: output, destination: read, byteCount: 296),
        .read(buffer: read, byteCount: 296),
        .dispatchComputeRanges(pipeline: pipeline, bindings: [
            .init(binding: 1): inputBinding,
            .init(binding: 3): .init(buffer: output, offset: 256, size: 32),
        ], workgroups: .init(x: 1)),
        .copy(source: output, destination: read, byteCount: 296),
        .read(buffer: read, byteCount: 296),
    ])
    let expected: [[UInt32]] = [
        [857, 888, 888, 888, 867, 878, 859, 860, 857, 999],
        [847, 888, 888, 888, 857, 878, 859, 860, 857, 999],
    ]
    let result = try await ticket.wait()
    let prefix = Array(repeating: UInt32(0), count: 64)
    guard result == expected.map({ prefix + $0 }) else {
        throw GPUError.backendFailure("Unexpected storage tail WebGPU readback: \(result)")
    }
    return 5
}

private func storageVectorDiagnostic(_ device: Device) async throws -> Int {
    let source = """
        @group(0) @binding(1) var<storage, read> src: array<vec3u>;
        @group(0) @binding(3) var<storage, read_write> dst: array<vec3i>;
        @group(0) @binding(5) var<storage, read_write> pair: array<vec2f, 1>;
        @compute @workgroup_size(1)
        fn run(@builtin(global_invocation_id) id: vec3u) {
            dst[id.x] = vec3i(src[i32(id.x) - 1i]) +
                vec3i(i32(arrayLength(&src)), i32(arrayLength(&dst)), -1i);
            if id.x == 0u {
                dst[id.x].z = -7i;
                let value = pair[0u];
                pair[0u] = vec2f(value.y, value.x);
            }
            if id.x == 1u { dst[id.x][i32(id.x) - 2i] = -9i; }
            if id.x == 2u { dst[id.x][id.x - 1u] = dst[id.x].x + 100i; }
        }
        """
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(source), to: .wgsl)
    let pipeline = try await device.createComputePipeline(shader)
    let input = try await device.createBuffer(.init(size: 304, usage: [.storage, .copyDestination]))
    let output = try await device.createBuffer(.init(size: 336, usage: [.storage, .copyDestination, .copySource]))
    let pair = try await device.createBuffer(.init(size: 8, usage: [.storage, .copyDestination, .copySource]))
    let read = try await device.createBuffer(.init(size: 336, usage: [.copyDestination, .mapRead]))
    let ticket = try await device.submitCommands([
        .write(buffer: input, offset: 256, words: [10, 20, 30, 801, 40, 50, 60, 802, 777, 778, 779, 803]),
        .write(buffer: output, offset: 256, words: [
            0, 0, 0, 701, 0, 0, 0, 702, 0, 0, 0, 703, 0, 0, 0, 704, 901, 902, 903, 904,
        ]),
        .write(buffer: pair, offset: 0, words: [Float(-1.25).bitPattern, Float(3.5).bitPattern]),
        .dispatchComputeRanges(pipeline: pipeline, bindings: [
            .init(binding: 1): .init(buffer: input, offset: 256, size: 36),
            .init(binding: 3): .init(buffer: output, offset: 256, size: 68),
            .init(binding: 5): .init(buffer: pair),
        ], workgroups: .init(x: 5)),
        .copy(source: output, destination: read, byteCount: 336),
        .read(buffer: read, byteCount: 336),
        .copy(source: pair, destination: read, byteCount: 8),
        .read(buffer: read, byteCount: 8),
    ])
    let expected: [UInt32] = [
        2, 4, UInt32(bitPattern: -7), 701, 12, 24, 29, 702, 42, 142, 59, 703,
        2, 4, .max, 704, 901, 902, 903, 904,
    ]
    let result = try await ticket.wait()
    guard result == [Array(repeating: 0, count: 64) + expected, [Float(3.5).bitPattern, Float(-1.25).bitPattern]] else {
        throw GPUError.backendFailure("Unexpected vector storage WebGPU readback: \(result)")
    }
    return 4
}

private func groupedStorageDiagnostic(_ device: Device) async throws -> [UInt32] {
    let source = """
        @group(3) @binding(999) var<storage, read> src: array<u32>;
        @group(0) @binding(5) var<storage, read> gain: u32;
        @group(1) @binding(999) var<storage, read_write> dst: array<u32>;
        fn scaled(i: u32) -> u32 { return src[i] * gain + arrayLength(&src) * 10u + arrayLength(&dst); }
        @compute @workgroup_size(1)
        fn run(@builtin(global_invocation_id) id: vec3u) { dst[id.x] = scaled(id.x); }
        """
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(source), to: .wgsl)
    let pipeline = try await device.createComputePipeline(shader)
    let input = try await device.createBuffer(.init(size: 268, usage: [.storage, .copyDestination]))
    let gain = try await device.createBuffer(.init(size: 4, usage: [.storage, .copyDestination]))
    let output = try await device.createBuffer(.init(size: 272, usage: [.storage, .copyDestination, .copySource]))
    let read = try await device.createBuffer(.init(size: 272, usage: [.copyDestination, .mapRead]))
    let ticket = try await device.submitCommands([
        .write(buffer: input, offset: 256, words: [3, 5, 777]),
        .write(buffer: gain, offset: 0, words: [2]),
        .write(buffer: output, offset: 256, words: [0, 0, 0, 999]),
        .dispatchComputeRanges(pipeline: pipeline, bindings: [
            .init(group: 3, binding: 999): .init(buffer: input, offset: 256, size: 8),
            .init(binding: 5): .init(buffer: gain),
            .init(group: 1, binding: 999): .init(buffer: output, offset: 256, size: 12),
        ], workgroups: .init(x: 4)),
        .copy(source: output, destination: read, byteCount: 272),
        .read(buffer: read, byteCount: 272),
    ])
    let result = try await ticket.wait()
    guard result.count == 1, let words = result.first, words.count == 68,
          words.prefix(64).allSatisfy({ $0 == 0 }),
          words.suffix(4).elementsEqual([29, 33, 23, 999])
    else { throw GPUError.backendFailure("Unexpected grouped storage WebGPU readback: \(result)") }
    return Array(words.suffix(4))
}

private func computeBuiltinDiagnostic(_ device: Device) async throws -> Int {
    let source = """
        struct Invocation {
            @builtin(local_invocation_id) local: vec3u,
            @builtin(local_invocation_index) lane: u32,
            @builtin(workgroup_id) group: vec3u,
        }
        @group(0) @binding(2) var<storage, read_write> output: array<u32>;
        fn pack(v: vec3u) -> u32 { return v.x + v.y * 16u + v.z * 256u; }
        fn packLocal(v: Invocation) -> u32 { return pack(v.local); }
        @compute @workgroup_size(2, 3, 2)
        fn run(@builtin(global_invocation_id) global: vec3u, v: Invocation,
               @builtin(num_workgroups) count: vec3u) {
            let width = count.x * 2u;
            let height = count.y * 3u;
            let i = 5u * (global.x + width * (global.y + height * global.z));
            output[i] = pack(global);
            output[i + 1u] = packLocal(v);
            output[i + 2u] = v.lane;
            output[i + 3u] = pack(v.group);
            output[i + 4u] = pack(count);
        }
        """
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(source), to: .wgsl)
    let pipeline = try await device.createComputePipeline(shader)
    var checked = 0
    func pack(_ x: UInt32, _ y: UInt32, _ z: UInt32) -> UInt32 { x + y * 16 + z * 256 }
    for groups in [DispatchWorkgroups(x: 3, y: 2, z: 2), .init(x: 1, y: 2, z: 1)] {
        var expected: [UInt32] = []
        for z in 0..<(groups.z * 2) {
            for y in 0..<(groups.y * 3) {
                for x in 0..<(groups.x * 2) {
                    let (lx, ly, lz) = (x % 2, y % 3, z % 2)
                    expected += [
                        pack(x, y, z), pack(lx, ly, lz), lx + 2 * ly + 6 * lz,
                        pack(x / 2, y / 3, z / 2), pack(groups.x, groups.y, groups.z),
                    ]
                }
            }
        }
        let size = UInt64(expected.count * 4)
        let buffer = try await device.createBuffer(.init(size: size, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: size, usage: [.copyDestination, .mapRead]))
        let ticket = try await device.submitCommands([
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 2): buffer], workgroups: groups),
            .copy(source: buffer, destination: readback, byteCount: size),
            .read(buffer: readback, byteCount: size),
        ])
        guard try await ticket.wait() == [expected] else {
            throw GPUError.backendFailure("Unexpected WebGPU compute builtin values for \(groups)")
        }
        checked += expected.count / 5
    }
    return checked
}

private func workgroupMemoryDiagnostic(_ device: Device) async throws -> [UInt32] {
    let source = """
        var<workgroup> scratch: array<u32, 7>;
        @group(0) @binding(2) var<storage, read_write> output: array<u32>;
        var<workgroup> flag: u32;
        fn fill(lane: u32, value: u32) { scratch[lane] = value; }
        fn sync() { workgroupBarrier(); }
        @compute @workgroup_size(4)
        fn run(@builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) group: vec3u) {
            let initial = scratch[lane + 3u] + flag;
            workgroupBarrier();
            fill(lane, group.x * 100u + lane + 10u);
            if lane == 0u { flag = 7u; }
            sync();
            output[group.x * 4u + lane] = initial * 1000u + scratch[(lane + 1u) % 4u] + flag;
        }
        """
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(source), to: .wgsl)
    let pipeline = try await device.createComputePipeline(shader)
    let buffer = try await device.createBuffer(.init(size: 48, usage: [.storage, .copySource]))
    let readback = try await device.createBuffer(.init(size: 48, usage: [.copyDestination, .mapRead]))
    let ticket = try await device.submitCommands([
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 2): buffer], workgroups: .init(x: 3)),
        .copy(source: buffer, destination: readback, byteCount: 48),
        .read(buffer: readback, byteCount: 48),
    ])
    let result = try await ticket.wait()
    let expected: [UInt32] = [18, 19, 20, 17, 118, 119, 120, 117, 218, 219, 220, 217]
    guard result == [expected] else {
        throw GPUError.backendFailure("Unexpected workgroup memory WebGPU readback: \(result)")
    }
    return expected
}

private func storageBarrierDiagnostic(_ device: Device) async throws -> [UInt32] {
    let source = """
        @group(0) @binding(1) var<storage, read_write> scratch: array<u32>;
        @group(2) @binding(1) var<storage, read_write> output: array<u32>;
        fn syncDevice() { storageBarrier(); }
        fn rendezvous() { syncDevice(); }
        @compute @workgroup_size(128)
        fn run(@builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) group: vec3u) {
            let start = group.x * 128u;
            scratch[start + lane] = group.x * 1000u + lane + 10u;
            rendezvous();
            output[start + lane] = scratch[start + (lane + 64u) % 128u] + 7u;
        }
        """
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(source), to: .wgsl)
    let pipeline = try await device.createComputePipeline(shader)
    let outputBytes: UInt64 = 3 * 128 * 4
    let sentinelOffset = 256 + outputBytes
    let scratchBytes = sentinelOffset + 4
    let scratch = try await device.createBuffer(.init(size: scratchBytes, usage: [.storage, .copyDestination, .copySource]))
    let output = try await device.createBuffer(.init(size: outputBytes, usage: [.storage, .copySource]))
    let readback = try await device.createBuffer(.init(size: scratchBytes, usage: [.copyDestination, .mapRead]))
    let ticket = try await device.submitCommands([
        .write(buffer: scratch, offset: sentinelOffset, words: [999]),
        .dispatchComputeRanges(pipeline: pipeline, bindings: [
            .init(binding: 1): .init(buffer: scratch, offset: 256, size: outputBytes),
            .init(group: 2, binding: 1): .init(buffer: output),
        ], workgroups: .init(x: 3)),
        .copy(source: output, destination: readback, byteCount: outputBytes),
        .read(buffer: readback, byteCount: outputBytes),
        .copy(source: scratch, destination: readback, byteCount: scratchBytes),
        .read(buffer: readback, byteCount: scratchBytes),
    ])
    let writes = (0..<384).map { (index: Int) -> UInt32 in
        let group = UInt32(index / 128)
        let lane = UInt32(index % 128)
        return group * 1000 + lane + 10
    }
    let rotated = (0..<384).map { (index: Int) -> UInt32 in
        let start = (index / 128) * 128
        let lane = (index + 64) % 128
        return writes[start + lane] + 7
    }
    let result = try await ticket.wait()
    let scratchExpected = Array(repeating: UInt32(0), count: 64) + writes + [999]
    guard result == [rotated, scratchExpected] else {
        throw GPUError.backendFailure("Unexpected storage barrier WebGPU readback")
    }
    return [0, 127, 128, 255, 256, 383].map { result[0][$0] }
}

private func atomicDiagnostic(_ device: Device) async throws -> Int {
    let source = """
        @group(0) @binding(0) var<storage, read_write> counter: array<atomic<u32>>;
        @group(0) @binding(1) var<storage, read_write> output: array<u32>;
        var<workgroup> local: atomic<i32>;
        fn ticket() -> u32 { return atomicAdd(&counter[0u], 1u); }
        @compute @workgroup_size(64)
        fn run(@builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) group: vec3u) {
            if lane == 0u { atomicStore(&local, -10i); }
            workgroupBarrier();
            let old = atomicAdd(&local, 1i);
            let global = ticket();
            let outside = atomicAdd(&counter[group.x + 1u], 1u);
            workgroupBarrier();
            let offset = 4u * (group.x * 64u + lane);
            output[offset] = global;
            output[offset + 1u] = u32(old);
            output[offset + 2u] = u32(atomicLoad(&local));
            output[offset + 3u] = outside;
        }
        """
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(source), to: .wgsl)
    let pipeline = try await device.createComputePipeline(shader)
    let counter = try await device.createBuffer(.init(size: 264, usage: [.storage, .copyDestination, .copySource]))
    let output = try await device.createBuffer(.init(size: 2048, usage: [.storage, .copySource]))
    let readback = try await device.createBuffer(.init(size: 2048, usage: [.copyDestination, .mapRead]))
    let ticket = try await device.submitCommands([
        .write(buffer: counter, offset: 256, words: [0, 999]),
        .dispatchComputeRanges(pipeline: pipeline, bindings: [
            .init(binding: 0): .init(buffer: counter, offset: 256, size: 4),
            .init(binding: 1): .init(buffer: output),
        ], workgroups: .init(x: 2)),
        .copy(source: output, destination: readback, byteCount: 2048),
        .read(buffer: readback, byteCount: 2048),
        .copy(source: counter, destination: readback, byteCount: 264),
        .read(buffer: readback, byteCount: 264),
    ])
    let result = try await ticket.wait()
    let expectedCounter = Array(repeating: UInt32(0), count: 64) + [128, 999]
    guard result.count == 2, result[0].count == 512, result[1] == expectedCounter else {
        throw GPUError.backendFailure("Unexpected WebGPU atomic counter")
    }
    let words = result[0]
    let slots = stride(from: 0, to: words.count, by: 4)
    let tickets = slots.map { words[$0] }.sorted()
    guard tickets == Array(0..<128), slots.allSatisfy({ words[$0 + 2] == 54 && words[$0 + 3] == 0 }) else {
        throw GPUError.backendFailure("Unexpected WebGPU atomic results")
    }
    for start in [0, 256] {
        let lanes = stride(from: start, to: start + 256, by: 4)
        guard lanes.map({ Int32(bitPattern: words[$0 + 1]) }).sorted() == Array(-10..<54) else {
            throw GPUError.backendFailure("Unexpected WebGPU workgroup atomic sequence")
        }
    }
    return tickets.count
}

private func atomicRMWDiagnostic(_ device: Device) async throws -> Int {
    let source = """
        @group(0) @binding(0) var<storage, read_write> signedValue: atomic<i32>;
        @group(0) @binding(1) var<storage, read_write> unsignedValues: array<atomic<u32>>;
        @group(0) @binding(2) var<storage, read_write> output: array<u32, 17>;
        @compute @workgroup_size(1)
        fn run(@builtin(global_invocation_id) id: vec3u) {
            output[0] = u32(atomicSub(&signedValue, i32(2147483648u)));
            output[1] = u32(atomicMin(&signedValue, -7i));
            output[2] = u32(atomicMax(&signedValue, -7i));
            output[3] = u32(atomicAnd(&signedValue, 15i));
            output[4] = u32(atomicOr(&signedValue, 32i));
            output[5] = u32(atomicXor(&signedValue, 3i));
            output[6] = u32(atomicExchange(&signedValue, -1i));
            atomicOr(&signedValue, 0i);
            output[7] = u32(atomicLoad(&signedValue));
            output[8] = atomicSub(&unsignedValues[0], 1u);
            output[9] = atomicMax(&unsignedValues[0], 7u);
            output[10] = atomicMin(&unsignedValues[0], 7u);
            output[11] = atomicAnd(&unsignedValues[0], 6u);
            output[12] = atomicOr(&unsignedValues[0], 8u);
            output[13] = atomicXor(&unsignedValues[0], 3u);
            output[14] = atomicExchange(&unsignedValues[0], 23u);
            output[15] = atomicLoad(&unsignedValues[0]);
            output[16] = atomicExchange(&unsignedValues[id.x + 1u], 777u);
        }
        """
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(source), to: .wgsl)
    let pipeline = try await device.createComputePipeline(shader)
    let signed = try await device.createBuffer(.init(size: 4, usage: .storage))
    let unsigned = try await device.createBuffer(.init(size: 8, usage: [.storage, .copyDestination, .copySource]))
    let output = try await device.createBuffer(.init(size: 68, usage: [.storage, .copySource]))
    let readback = try await device.createBuffer(.init(size: 68, usage: [.copyDestination, .mapRead]))
    let ticket = try await device.submitCommands([
        .write(buffer: unsigned, offset: 4, words: [999]),
        .dispatchComputeRanges(pipeline: pipeline, bindings: [
            .init(binding: 0): .init(buffer: signed),
            .init(binding: 1): .init(buffer: unsigned, size: 4),
            .init(binding: 2): .init(buffer: output),
        ], workgroups: .init(x: 1)),
        .copy(source: output, destination: readback, byteCount: 68),
        .read(buffer: readback, byteCount: 68),
        .copy(source: unsigned, destination: readback, byteCount: 8),
        .read(buffer: readback, byteCount: 8),
    ])
    let expected: [UInt32] = [
        0, 0x80000000, 0x80000000, 0xfffffff9, 9, 41, 42, .max,
        0, .max, .max, 7, 6, 14, 13, 23, 0,
    ]
    guard try await ticket.wait() == [expected, [23, 999]] else {
        throw GPUError.backendFailure("Unexpected WebGPU atomic read-modify-write results")
    }
    return expected.count
}

private func atomicCompareDiagnostic(_ device: Device) async throws -> Int {
    let source = """
        @group(0) @binding(0) var<storage, read_write> counter: array<atomic<u32>>;
        @group(0) @binding(1) var<storage, read_write> output: array<u32, 10>;
        var<workgroup> signedValue: atomic<i32>;
        @compute @workgroup_size(1)
        fn run(@builtin(global_invocation_id) id: vec3u) {
            atomicStore(&signedValue, -9i);
            var signedResult = atomicCompareExchangeWeak(&signedValue, -9i, 11i);
            output[0] = u32(signedResult.old_value);
            if signedResult.exchanged { output[1] = 1u; }
            output[2] = u32(atomicLoad(&signedValue));
            signedResult = atomicCompareExchangeWeak(&signedValue, 999i, 12i);
            output[3] = u32(signedResult.old_value);
            if signedResult.exchanged { output[4] = 1u; }
            let unsignedResult = atomicCompareExchangeWeak(&counter[0u], 0u, 7u);
            output[5] = unsignedResult.old_value;
            if unsignedResult.exchanged { output[6] = 1u; }
            output[7] = atomicLoad(&counter[0u]);
            let outside = atomicCompareExchangeWeak(&counter[id.x + 1u], 0u, 888u);
            output[8] = outside.old_value;
            if outside.exchanged { output[9] = 1u; }
        }
        """
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(source), to: .wgsl)
    let pipeline = try await device.createComputePipeline(shader)
    let counter = try await device.createBuffer(.init(size: 8, usage: [.storage, .copyDestination, .copySource]))
    let output = try await device.createBuffer(.init(size: 40, usage: [.storage, .copySource]))
    let readback = try await device.createBuffer(.init(size: 40, usage: [.copyDestination, .mapRead]))
    let ticket = try await device.submitCommands([
        .write(buffer: counter, offset: 4, words: [999]),
        .dispatchComputeRanges(pipeline: pipeline, bindings: [
            .init(binding: 0): .init(buffer: counter, size: 4),
            .init(binding: 1): .init(buffer: output),
        ], workgroups: .init(x: 1)),
        .copy(source: output, destination: readback, byteCount: 40),
        .read(buffer: readback, byteCount: 40),
        .copy(source: counter, destination: readback, byteCount: 8),
        .read(buffer: readback, byteCount: 8),
    ])
    let snapshots = try await ticket.wait()
    guard snapshots.count == 2, snapshots[0].count == 10 else {
        throw GPUError.backendFailure("Unexpected WebGPU atomic comparison readback size")
    }
    let result = snapshots[0]
    let signedStart = UInt32(bitPattern: -9)
    let signedAfter: UInt32 = result[1] == 1 ? 11 : signedStart
    let unsignedAfter: UInt32 = result[6] == 1 ? 7 : 0
    let expected: [UInt32] = [signedStart, result[1], signedAfter, signedAfter, 0, 0, result[6], unsignedAfter, 0, 0]
    guard result[1] <= 1, result[6] <= 1, result == expected, snapshots[1] == [unsignedAfter, 999] else {
        throw GPUError.backendFailure("Unexpected WebGPU atomic comparison results")
    }
    return expected.count
}

private func mixedAtomicDiagnostic(_ device: Device) async throws -> Int {
    let source = """
        struct Mixed {
            @size(12) bias: u32,
            count: atomic<i32>,
            payload: vec2f,
            @align(32) values: array<atomic<u32>>,
        }
        @group(0) @binding(0) var<storage, read_write> data: Mixed;
        @group(0) @binding(1) var<storage, read_write> output: array<u32, 6>;
        @compute @workgroup_size(1)
        fn run(@builtin(global_invocation_id) id: vec3u) {
            output[0] = u32(atomicMin(&data.count, -20i));
            output[1] = u32(atomicMax(&data.count, -6i));
            atomicOr(&data.count, 0i);
            output[2] = u32(atomicLoad(&data.count));
            data.bias += atomicAdd(&data.values[0u], 2u);
            let value = data.payload;
            data.payload = vec2f(value.y, value.x);
            output[3] = arrayLength(&data.values);
            let exchanged = atomicCompareExchangeWeak(&data.values[1u], 4u, 11u);
            if exchanged.exchanged { output[4] = 1u; }
            let outside = atomicCompareExchangeWeak(&data.values[id.x + arrayLength(&data.values)], 0u, 888u);
            output[5] = outside.old_value;
            if outside.exchanged { output[5] = 1u; }
        }
        """
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(source), to: .wgsl)
    let pipeline = try await device.createComputePipeline(shader)
    let data = try await device.createBuffer(.init(size: 324, usage: [.storage, .copyDestination, .copySource]))
    let output = try await device.createBuffer(.init(size: 24, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 324, usage: [.copyDestination, .mapRead]))
    let initial: [UInt32] = [9, 701, 702, UInt32(bitPattern: -10), Float(1.5).bitPattern, Float(-2).bitPattern,
                            703, 704, 3, 4, 0, 0, 0, 0, 0, 0, 999]
    let ticket = try await device.submitCommands([
        .write(buffer: data, offset: 256, words: initial),
        .dispatchComputeRanges(pipeline: pipeline, bindings: [
            .init(binding: 0): .init(buffer: data, offset: 256, size: 64),
            .init(binding: 1): .init(buffer: output),
        ], workgroups: .init(x: 1)),
        .copy(source: output, destination: read, byteCount: 24),
        .read(buffer: read, byteCount: 24),
        .copy(source: data, destination: read, byteCount: 324),
        .read(buffer: read, byteCount: 324),
    ])
    let result = try await ticket.wait()
    guard result.count == 2, result[0].count == 6 else {
        throw GPUError.backendFailure("Unexpected mixed atomic readback size")
    }
    let exchanged = result[0][4]
    let expected: [UInt32] = [
        UInt32(bitPattern: -10), UInt32(bitPattern: -20), UInt32(bitPattern: -6), 8, exchanged, 0,
    ]
    var expectedData = initial
    expectedData[0] = 12
    expectedData[3] = UInt32(bitPattern: -6)
    expectedData.swapAt(4, 5)
    expectedData[8] = 5
    expectedData[9] = exchanged == 1 ? 11 : 4
    guard exchanged <= 1, result == [expected, Array(repeating: UInt32(0), count: 64) + expectedData] else {
        throw GPUError.backendFailure("Unexpected mixed atomic readback")
    }
    return expectedData.count
}

private func uniformLoadDiagnostic(_ device: Device) async throws -> [UInt32] {
    let source = """
        var<workgroup> flag: u32;
        var<workgroup> values: array<vec2u, 2>;
        var<workgroup> counter: atomic<i32>;
        @group(0) @binding(0) var<storage, read_write> output: array<u32, 12>;
        @compute @workgroup_size(4)
        fn run(@builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) group: vec3u) {
            if lane == 0u {
                flag = 11u + group.x;
                values[group.x] = vec2u(7u + group.x, 2u);
                atomicStore(&counter, -5i + i32(group.x));
            }
            let snapshot = workgroupUniformLoad(&flag);
            if lane == 0u { flag = 666u; }
            let next = workgroupUniformLoad(&flag);
            if snapshot != 0u { workgroupBarrier(); }
            let pair = workgroupUniformLoad(&values[group.x]);
            if pair.x != 0u { workgroupBarrier(); }
            let signedValue = workgroupUniformLoad(&counter);
            if signedValue < 0i { workgroupBarrier(); }
            output[group.x * 4u + lane] = snapshot + pair.x + pair.y + u32(signedValue + 10i) + next;
        }
        """
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(source), to: .wgsl)
    let pipeline = try await device.createComputePipeline(shader)
    let output = try await device.createBuffer(.init(size: 48, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 48, usage: [.copyDestination, .mapRead]))
    let ticket = try await device.submitCommands([
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 3)),
        .copy(source: output, destination: read, byteCount: 48),
        .read(buffer: read, byteCount: 48),
    ])
    let words: [UInt32] = [691, 691, 691, 691, 694, 694, 694, 694, 686, 686, 686, 686]
    guard try await ticket.wait() == [words] else { throw GPUError.backendFailure("Unexpected WebGPU uniform load") }
    return words
}

private func selectDiagnostic(_ device: Device) async throws -> [UInt32] {
    let source = """
        @group(0) @binding(0) var<storage, read_write> trace: u32;
        @group(0) @binding(1) var<storage, read_write> output: array<u32, 8>;
        fn mark(n: u32) -> u32 { trace = trace * 10u + n; return n; }
        @compute @workgroup_size(1)
        fn run(@builtin(global_invocation_id) id: vec3u) {
            let scalar = select(mark(2u), mark(3u), true);
            let mask = vec3u(id.x, id.x + 1u, id.x + 2u) < vec3u(1u, 1u, 3u);
            let lanes = select(vec3u(10u, 20u, 30u), vec3u(40u, 50u, 60u), mask);
            let pair = select(vec2i(-1i, -2i), vec2i(3i, 4i), false);
            let floats = select(vec2f(1.5f, 2.5f), vec2f(3.5f, 4.5f), vec2<bool>(false, true));
            let valid = select(false, true, floats.x == 1.5f && floats.y == 4.5f);
            output[0] = scalar;
            output[1] = trace;
            output[2] = lanes.x;
            output[3] = lanes.y;
            output[4] = lanes.z;
            output[5] = u32(pair.x);
            output[6] = u32(pair.y);
            output[7] = select(0u, 1, valid);
        }
        """
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(source), to: .wgsl)
    let pipeline = try await device.createComputePipeline(shader)
    let trace = try await device.createBuffer(.init(size: 4, usage: .storage))
    let output = try await device.createBuffer(.init(size: 32, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 32, usage: [.copyDestination, .mapRead]))
    let ticket = try await device.submitCommands([
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): trace, .init(binding: 1): output],
                         workgroups: .init(x: 1)),
        .copy(source: output, destination: read, byteCount: 32),
        .read(buffer: read, byteCount: 32),
    ])
    let words: [UInt32] = [3, 23, 40, 20, 60, .max, .max - 1, 1]
    guard try await ticket.wait() == [words] else { throw GPUError.backendFailure("Unexpected WebGPU select results") }
    return words
}

private func constantSelectDiagnostic(_ device: Device) async throws -> [UInt32] {
    let source = """
        const enabled = select(false, true, true);
        const scale = select(4u, 8u, enabled);
        const bias = select(-2i, -9i, false);
        const factor = select(0.25f, 0.5f, enabled);
        const lanes = select(vec4u(1u, 2u, 3u, 4u), vec4u(8u, 9u, 10u, 11u), vec4<bool>(true, false, true, false));
        const_assert select(0u, scale, enabled) == 8u;
        @group(0) @binding(0) var<storage, read_write> output: array<u32, select(2u, 4u, enabled)>;
        fn adjustment() -> u32 { return scale + u32(bias + 10i); }
        @compute @workgroup_size(select(2, 4, enabled))
        fn run(@builtin(global_invocation_id) id: vec3u) {
            if enabled && factor == 0.5f {
                output[id.x] = adjustment() + lanes.x + lanes.y + lanes.z + lanes.w + id.x;
            }
        }
        """
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(source), to: .wgsl)
    let pipeline = try await device.createComputePipeline(shader)
    let output = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 16, usage: [.copyDestination, .mapRead]))
    let ticket = try await device.submitCommands([
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
        .copy(source: output, destination: read, byteCount: 16),
        .read(buffer: read, byteCount: 16),
    ])
    let words: [UInt32] = [40, 41, 42, 43]
    guard try await ticket.wait() == [words] else { throw GPUError.backendFailure("Unexpected WebGPU constant select results") }
    return words
}

private func computeUniformDiagnostic(_ device: Device) async throws -> [UInt32] {
    let source = """
        struct Config {
            normal: vec3u,
            bias: u32,
            @align(32) table: array<vec4u, 2>,
        }
        @group(0) @binding(0) var<uniform> cfg: Config;
        @group(3) @binding(9) var<uniform> tuning: u32;
        @group(0) @binding(1) var<storage, read_write> output: array<u32, 4>;
        fn score(i: u32) -> u32 {
            let row = cfg.table[i];
            return cfg.normal.z + cfg.bias + tuning + row.x + row.w;
        }
        @compute @workgroup_size(4)
        fn run(@builtin(global_invocation_id) id: vec3u) { output[id.x] = score(id.x); }
        """
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(source), to: .wgsl)
    let pipeline = try await device.createComputePipeline(shader)
    let cfg = try await device.createBuffer(.init(size: 320, usage: [.uniform, .copyDestination]))
    let tuning = try await device.createBuffer(.init(size: 4, usage: [.uniform, .copyDestination]))
    let output = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 16, usage: [.copyDestination, .mapRead]))
    let ticket = try await device.submitCommands([
        .write(buffer: cfg, offset: 256, words: [1, 2, 3, 4, 701, 702, 703, 704, 5, 6, 7, 8, 9, 10, 11, 12]),
        .write(buffer: tuning, offset: 0, words: [10]),
        .dispatchComputeRanges(pipeline: pipeline, bindings: [
            .init(binding: 0): .init(buffer: cfg, offset: 256, size: 64),
            .init(group: 3, binding: 9): .init(buffer: tuning),
            .init(binding: 1): .init(buffer: output),
        ], workgroups: .init(x: 1)),
        .copy(source: output, destination: read, byteCount: 16),
        .read(buffer: read, byteCount: 16),
    ])
    let words: [UInt32] = [30, 38, 17, 17]
    guard try await ticket.wait() == [words] else { throw GPUError.backendFailure("Unexpected WebGPU uniform results") }
    return words
}

private func renderUniformDiagnostic(_ device: Device) async throws -> [UInt32] {
    let source = ShaderInput.wgsl("""
        struct DrawConfig { offset: vec2f, color: vec4f }
        @group(3) @binding(11) var<uniform> cfg: DrawConfig;
        @group(0) @binding(0) var<uniform> jitter: vec2f;
        @group(1) @binding(1) var<uniform> unused: vec4f;
        fn tint() -> vec4f { return cfg.color; }
        @vertex fn vs(@location(0) point: vec2f) -> @builtin(position) vec4f {
            return vec4f(point.x + cfg.offset.x + jitter.x, point.y + cfg.offset.y + jitter.y, 0.0f, 1.0f);
        }
        @fragment fn fs() -> @location(0) vec4f { return tint(); }
        """)
    let compiler = IndependentShaderCompiler()
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .wgsl)
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: .wgsl)
    let pipeline = try await device.createRenderPipeline(
        vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm,
        vertexBuffers: [.init(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float32x2)])])
    let uniform = try await device.createBuffer(.init(size: 288, usage: [.uniform, .copyDestination]))
    let jitter = try await device.createBuffer(.init(size: 8, usage: [.uniform, .copyDestination]))
    let vertices = try await device.createBuffer(.init(size: 24, usage: [.vertex, .copyDestination]))
    let indices = try await device.createBuffer(.init(size: 8, usage: [.index, .copyDestination]))
    let texture = try await device.createTexture(.init(width: 8, height: 8, format: .rgba8Unorm,
                                                       usage: [.renderAttachment, .copySource]))
    let read = try await device.createBuffer(.init(size: 1824, usage: [.copyDestination, .mapRead]))
    let clear = RenderClearColor(red: 1, green: 0, blue: 0, alpha: 1)
    let bindings: [ComputeBindingSlot: ComputeBufferBinding] = [
        .init(group: 3, binding: 11): .init(buffer: uniform, offset: 256, size: 32),
        .init(binding: 0): .init(buffer: jitter),
    ]
    let streams = [VertexBufferBinding(buffer: vertices)]
    let ticket = try await device.submitCommands([
        .write(buffer: uniform, offset: 256, words: [Float(0.75).bitPattern, 0, 0, 0, 0, Float(1).bitPattern, 0, Float(1).bitPattern]),
        .write(buffer: jitter, offset: 0, words: [Float(0.25).bitPattern, 0]),
        .write(buffer: vertices, offset: 0, words: [Float(-1), -1, 3, -1, -1, 3].map(\.bitPattern)),
        .write(buffer: indices, offset: 0, words: [0x00010000, 2]),
        .render(pipeline: pipeline, target: texture, clear: clear, vertexCount: 3,
                vertexBuffers: streams, uniforms: bindings),
        .copyTextureToBuffer(source: texture, destination: read, bytesPerRow: 256),
        .read(buffer: read, byteCount: 1824),
        .renderIndexed(pipeline: pipeline, target: texture, clear: clear,
                       indexBuffer: .init(buffer: indices, format: .uint16), draw: .init(indexCount: 3),
                       vertexBuffers: streams, uniforms: bindings),
        .copyTextureToBuffer(source: texture, destination: read, bytesPerRow: 256),
        .read(buffer: read, byteCount: 1824),
    ])
    let snapshots = try await ticket.wait()
    guard snapshots.count == 2, snapshots.allSatisfy({ $0.count == 456 }) else {
        throw GPUError.backendFailure("Unexpected uniform render readback size")
    }
    let pixels = snapshots.flatMap { [$0[4 * 64 + 1], $0[4 * 64 + 6]] }
    guard pixels == [0xff0000ff, 0xff00ff00, 0xff0000ff, 0xff00ff00] else {
        throw GPUError.backendFailure("Unexpected uniform render pixels: \(pixels)")
    }
    return pixels
}

private func matrixDiagnostic(_ device: Device) async throws -> [Float] {
    let source = """
        @group(0) @binding(0) var<storage, read_write> output: array<f32, 18>;
        fn product(a: mat2x3f, b: mat4x2f) -> mat4x3f { return a * b; }
        fn column(a: mat2x3f, v: vec2f) -> vec3f { return a * v; }
        fn row(v: vec3f, a: mat2x3f) -> vec2f { return v * a; }
        fn seed(i: u32) -> f32 { if i == 0u { return 1.0f; } return 2.0f; }
        @compute @workgroup_size(1)
        fn run(@builtin(global_invocation_id) id: vec3u) {
            let a = mat2x3f(vec3f(seed(id.x), 2.0f, 3.0f), vec3f(4.0f, 5.0f, 6.0f));
            let b = mat4x2f(vec2f(7.0f, 8.0f), vec2f(9.0f, 10.0f), vec2f(11.0f, 12.0f), vec2f(13.0f, 14.0f));
            let p = product(a, b);
            let first = p[0u];
            let last = p[id.x + 3u];
            let cv = column(a, vec2f(2.0f, 3.0f));
            let rv = row(vec3f(2.0f, 3.0f, 4.0f), a);
            let scaled = (a * 2.0f) + (3.0f * a) - a;
            var scratch: mat2x3f;
            scratch[id.x + 1u] = vec3f(7.0f, 8.0f, 9.0f);
            scratch[i32(id.x) - 1i] = vec3f(999.0f);
            let selected = scratch[id.x + 1u];
            let outside = scratch[id.x + 2u];
            output[0] = first.x; output[1] = first.y; output[2] = first.z;
            output[3] = last.x; output[4] = last.y; output[5] = last.z;
            output[6] = cv.x; output[7] = cv.y; output[8] = cv.z;
            output[9] = rv.x; output[10] = rv.y; output[11] = scaled[1u].z;
            output[12] = selected.x; output[13] = selected.y; output[14] = selected.z;
            output[15] = outside.x; output[16] = outside.y; output[17] = outside.z;
        }
        """
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(source), to: .wgsl)
    let pipeline = try await device.createComputePipeline(shader)
    let output = try await device.createBuffer(.init(size: 72, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 72, usage: [.copyDestination, .mapRead]))
    let ticket = try await device.submitCommands([
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
        .copy(source: output, destination: read, byteCount: 72),
        .read(buffer: read, byteCount: 72),
    ])
    let expected: [Float] = [39, 54, 69, 69, 96, 123, 14, 19, 24, 20, 47, 24, 7, 8, 9, 0, 0, 0]
    guard try await ticket.wait() == [expected.map(\.bitPattern)] else {
        throw GPUError.backendFailure("Unexpected WebGPU matrix results")
    }
    return expected
}

private func matrixBufferDiagnostic(_ device: Device) async throws -> [Float] {
    let source = """
        struct Data { basis: mat2x3f, transforms: array<mat2x3f> }
        @group(0) @binding(0) var<storage, read_write> data: Data;
        @group(0) @binding(1) var<uniform> scale: mat2x2f;
        @group(0) @binding(2) var<storage, read_write> output: array<f32, 4>;
        @compute @workgroup_size(1)
        fn run(@builtin(global_invocation_id) id: vec3u) {
            let saved = data.basis;
            data.basis = saved * scale;
            let i = id.x + 1u;
            data.transforms[i] = data.transforms[0u] * scale;
            data.transforms[i][id.x] = vec3f(21.0f, 22.0f, 23.0f);
            data.transforms[i][1u][1u] = 55.0f;
            data.transforms[id.x + 2u] = saved;
            let missing = data.transforms[id.x + 2u];
            output[0] = missing[0u].x;
            if arrayLength(&data.transforms) == 2u { output[1] = 1.0f; }
            output[2] = data.transforms[i][1u].y;
            output[3] = data.basis[1u].z;
        }
        """
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(source), to: .wgsl)
    let pipeline = try await device.createComputePipeline(shader)
    let data = try await device.createBuffer(.init(size: 356, usage: [.storage, .copyDestination, .copySource]))
    let scale = try await device.createBuffer(.init(size: 16, usage: [.uniform, .copyDestination]))
    let output = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 356, usage: [.copyDestination, .mapRead]))
    func column(_ x: Float, _ y: Float, _ z: Float, _ padding: UInt32) -> [UInt32] {
        [x.bitPattern, y.bitPattern, z.bitPattern, padding]
    }
    let initial = [
        column(1, 2, 3, 701), column(4, 5, 6, 702),
        column(7, 8, 9, 703), column(10, 11, 12, 704),
        column(13, 14, 15, 705), column(16, 17, 18, 706), [999],
    ].flatMap { $0 }
    let expected = [
        column(2, 4, 6, 701), column(12, 15, 18, 702),
        column(7, 8, 9, 703), column(10, 11, 12, 704),
        column(21, 22, 23, 705), column(30, 55, 36, 706), [999],
    ].flatMap { $0 }
    let ticket = try await device.submitCommands([
        .write(buffer: data, offset: 256, words: initial),
        .write(buffer: scale, offset: 0, words: [Float(2).bitPattern, 0, 0, Float(3).bitPattern]),
        .dispatchComputeRanges(pipeline: pipeline, bindings: [
            .init(binding: 0): .init(buffer: data, offset: 256, size: 96),
            .init(binding: 1): .init(buffer: scale),
            .init(binding: 2): .init(buffer: output),
        ], workgroups: .init(x: 1)),
        .copy(source: output, destination: read, byteCount: 16),
        .read(buffer: read, byteCount: 16),
        .copy(source: data, destination: read, byteCount: 356),
        .read(buffer: read, byteCount: 356),
    ])
    let values: [Float] = [0, 1, 55, 18]
    let result = try await ticket.wait()
    guard result == [values.map(\.bitPattern), Array(repeating: UInt32(0), count: 64) + expected] else {
        throw GPUError.backendFailure("Unexpected WebGPU matrix buffer readback: \(result.map { Array($0.suffix(25)) })")
    }
    return values
}

private func switchDiagnostic(_ device: Device) async throws -> [UInt32] {
    let source = """
        @group(0) @binding(0) var<storage, read_write> output: array<u32, 8>;
        const alternative = select(-2i, -1i, true);
        fn mapped(i: i32) -> u32 {
            switch i {
                case -2i, alternative: { return 7u; }
                case 0i: { return 11u; }
                case 9i, (-2147483647i - 1i), default: { return 19u; }
            }
        }
        fn choose() -> u32 { output[7] += 1u; return 2u; }
        @compute @workgroup_size(1)
        fn run() {
            var total = 0u;
            var i = 0u;
            loop {
                switch i {
                    case 0u, 2u: { total += 2u; continue; }
                    case 1u: { total += 10u; break; }
                    default: { total += 1u; }
                }
                total += 100u;
                continuing { i += 1u; break if i >= 4u; }
            }
            output[0] = mapped(-2i); output[1] = mapped(-1i); output[2] = mapped(0i);
            output[3] = mapped(9i); output[4] = mapped(5i); output[5] = total;
            switch choose() {
                case 2u: { output[6] = 23u; }
                default: { output[6] = 99u; }
            }
        }
        """
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(source), to: .wgsl)
    let pipeline = try await device.createComputePipeline(shader)
    let output = try await device.createBuffer(.init(size: 32, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 32, usage: [.copyDestination, .mapRead]))
    let ticket = try await device.submitCommands([
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
        .copy(source: output, destination: read, byteCount: 32),
        .read(buffer: read, byteCount: 32),
    ])
    let words: [UInt32] = [7, 7, 11, 19, 19, 215, 23, 1]
    guard try await ticket.wait() == [words] else { throw GPUError.backendFailure("Unexpected WebGPU switch results") }
    return words
}

private func textureUploadDiagnostic(_ device: Device) async throws -> [UInt32] {
    let source = try await device.createBuffer(.init(size: 268, usage: [.copySource, .copyDestination]))
    let target = try await device.createTexture(.init(width: 4, height: 3, format: .rgba8Unorm,
                                                      usage: [.copyDestination, .copySource]))
    let full = try await device.createTexture(.init(width: 2, height: 1, format: .bgra8Unorm,
                                                    usage: [.copyDestination, .copySource]))
    let read = try await device.createBuffer(.init(size: 528, usage: [.copyDestination, .mapRead]))
    let texels: [UInt32] = [0xff112233, 0xff445566, 0xff778899, 0xffaabbcc]
    let ticket = try await device.submitCommands([
        .write(buffer: source, offset: 4, words: Array(texels.prefix(2))),
        .write(buffer: source, offset: 260, words: Array(texels.suffix(2))),
        .copyBufferToTexture(source: source, layout: .init(offset: 4, bytesPerRow: 256),
                             destination: target, region: .init(x: 1, y: 1, width: 2, height: 2)),
        .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256),
        .read(buffer: read, byteCount: 528),
        .copyBufferToTexture(source: source, layout: .init(offset: 8),
                             destination: target, region: .init(x: 3, width: 1, height: 1)),
        .copyBufferToTexture(source: source, layout: .init(offset: 268),
                             destination: target, region: .init(x: 4, y: 3, width: 0, height: 0)),
        .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256),
        .read(buffer: read, byteCount: 528),
        .copyBufferToTexture(source: source, layout: .init(offset: 4),
                             destination: full, region: .init(width: 2, height: 1)),
        .copyTextureToBuffer(source: full, destination: read, bytesPerRow: 256),
        .read(buffer: read, byteCount: 8),
    ])
    try await device.destroyBuffer(source)
    let result = try await ticket.wait()
    var first = Array(repeating: UInt32(0), count: 132)
    first[65] = texels[0]; first[66] = texels[1]
    first[129] = texels[2]; first[130] = texels[3]
    var second = first
    second[3] = texels[1]
    guard result == [first, second, Array(texels.prefix(2))] else {
        throw GPUError.backendFailure("Unexpected WebGPU texture upload readback")
    }
    return [3, 65, 66, 129, 130].map { result[1][$0] }
}

private func textureLoadDiagnostic(_ device: Device) async throws -> [UInt32] {
    let source = ShaderInput.wgsl("""
        @group(2) @binding(5) var image: texture_2d<f32>;
        @group(2) @binding(0) var<storage, read_write> out: array<vec4f, 7>;

        fn fetch(p: vec2i, level: i32) -> vec4f { return textureLoad(image, p, level); }

        @compute @workgroup_size(7)
        fn run(@builtin(global_invocation_id) id: vec3u) {
            if id.x == 4u {
                out[id.x] = fetch(vec2i(0), 1i);
            } else if id.x == 5u {
                out[id.x] = textureLoad(image, vec2u(0xffffffffu, 0u), 0u);
            } else if id.x == 6u {
                out[id.x] = textureLoad(image, vec2u(0u), 0xffffffffu);
            } else {
                out[id.x] = fetch(vec2i(i32(id.x) - 1i, 0i), 0i);
            }
        }
        """)
    let shader = try IndependentShaderCompiler().compileCompute(source, to: .wgsl)
    let pipeline = try await device.createComputePipeline(shader)
    let input = try await device.createBuffer(.init(size: 8, usage: [.copyDestination, .copySource]))
    let target = try await device.createTexture(.init(width: 2, height: 1, format: .bgra8Unorm,
                                                      usage: [.copyDestination, .textureBinding]))
    let fresh = try await device.createTexture(.init(width: 2, height: 1, format: .rgba8Unorm, usage: .textureBinding))
    let size: UInt64 = 7 * 16
    let out = try await device.createBuffer(.init(size: size, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: size, usage: [.copyDestination, .mapRead]))
    func dispatch(_ texture: ResourceID) -> BufferCommand {
        .dispatchComputeRanges(pipeline: pipeline, bindings: [.init(group: 2, binding: 0): .init(buffer: out)],
                               workgroups: .init(x: 1), textures: [.init(group: 2, binding: 5): texture])
    }
    let ticket = try await device.submitCommands([
        .write(buffer: input, offset: 0, words: [0xffff0000, 0xff00ff00]),
        .copyBufferToTexture(source: input, layout: .init(), destination: target, region: .init(width: 2, height: 1)),
        dispatch(target),
        .copy(source: out, destination: read, byteCount: size),
        .read(buffer: read, byteCount: size),
        dispatch(fresh),
        .copy(source: out, destination: read, byteCount: size),
        .read(buffer: read, byteCount: size),
    ])
    try await device.destroyTexture(target)
    let result = try await ticket.wait()
    let expected: [Float] = [
        0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]
    guard result == [expected.map(\.bitPattern), Array(repeating: 0, count: 28)] else {
        throw GPUError.backendFailure("Unexpected WebGPU sampled texture readback: \(result)")
    }
    return (0..<7).map { i in
        let pixel = result[0][(i * 4)..<(i * 4 + 4)]
        return zip(pixel, [16, 8, 0, 24]).reduce(0) { $0 | (UInt32(Float(bitPattern: $1.0) * 255) << $1.1) }
    }
}

private func renderTextureDiagnostic(_ device: Device) async throws -> [UInt32] {
    let source = ShaderInput.wgsl("""
        @group(2) @binding(5) var image: texture_2d<f32>;
        @group(1) @binding(4) var auxiliary: texture_2d<f32>;
        @group(3) @binding(6) var unused: texture_2d<f32>;
        @group(2) @binding(8) var<uniform> gain: f32;

        fn readImage(x: i32) -> vec4f { return textureLoad(image, vec2i(x, 0i), 0i); }

        @vertex fn vs(@location(0) point: vec2f) -> @builtin(position) vec4f {
            return vec4f(point.x + readImage(0i).r, point.y, 0.0f, 1.0f);
        }

        @fragment fn fs() -> @location(0) vec4f {
            let blue = textureLoad(auxiliary, vec2u(0u), 0u).b;
            return vec4f(readImage(1i).g * gain, blue, 0.0f, 1.0f);
        }
        """)
    let compiler = IndependentShaderCompiler()
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .wgsl)
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: .wgsl)
    let pipeline = try await device.createRenderPipeline(
        vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm,
        vertexBuffers: [.init(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float32x2)])])
    let upload = try await device.createBuffer(.init(size: 12, usage: [.copySource, .copyDestination]))
    let vertices = try await device.createBuffer(.init(size: 24, usage: [.vertex, .copyDestination]))
    let indices = try await device.createBuffer(.init(size: 8, usage: [.index, .copyDestination]))
    let gain = try await device.createBuffer(.init(size: 4, usage: [.uniform, .copyDestination]))
    let image = try await device.createTexture(.init(width: 2, height: 1, format: .rgba8Unorm,
                                                    usage: [.copyDestination, .textureBinding]))
    let aux = try await device.createTexture(.init(width: 1, height: 1, format: .bgra8Unorm,
                                                  usage: [.copyDestination, .textureBinding]))
    let fresh = try await device.createTexture(.init(width: 1, height: 1, format: .bgra8Unorm, usage: .textureBinding))
    let target = try await device.createTexture(.init(width: 8, height: 8, format: .rgba8Unorm,
                                                      usage: [.renderAttachment, .copySource, .textureBinding]))
    let read = try await device.createBuffer(.init(size: 1824, usage: [.copyDestination, .mapRead]))
    let clear = RenderClearColor(red: 0, green: 0, blue: 1, alpha: 1)
    let shared = ComputeBindingSlot(group: 2, binding: 5)
    let auxiliary = ComputeBindingSlot(group: 1, binding: 4)
    let uniforms: [ComputeBindingSlot: ComputeBufferBinding] = [.init(group: 2, binding: 8): .init(buffer: gain)]
    let streams = [VertexBufferBinding(buffer: vertices)]
    func draw(_ textures: [ComputeBindingSlot: ResourceID]) -> BufferCommand {
        .render(pipeline: pipeline, target: target, clear: clear, vertexCount: 3, vertexBuffers: streams,
                uniforms: uniforms, textures: textures)
    }
    do {
        _ = try await device.submitCommands([draw([shared: target, auxiliary: aux])])
        throw GPUError.backendFailure("Accepted a sampled render attachment")
    } catch GPUError.validation("Render attachment cannot be bound as a sampled texture") {}
    let ticket = try await device.submitCommands([
        .write(buffer: upload, offset: 0, words: [0xff0000ff, 0xff00ff00, 0xff0000ff]),
        .copyBufferToTexture(source: upload, layout: .init(), destination: image, region: .init(width: 2, height: 1)),
        .copyBufferToTexture(source: upload, layout: .init(offset: 8), destination: aux, region: .init(width: 1, height: 1)),
        .write(buffer: gain, offset: 0, words: [Float(1).bitPattern]),
        .write(buffer: vertices, offset: 0, words: [Float(-1), -1, 3, -1, -1, 3].map(\.bitPattern)),
        .write(buffer: indices, offset: 0, words: [0x00010000, 2]),
        draw([shared: image, auxiliary: aux]),
        .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256),
        .read(buffer: read, byteCount: 1824),
        .renderIndexed(pipeline: pipeline, target: target, clear: clear,
                       indexBuffer: .init(buffer: indices, format: .uint16), draw: .init(indexCount: 3),
                       vertexBuffers: streams, uniforms: uniforms, textures: [shared: image, auxiliary: fresh]),
        .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256),
        .read(buffer: read, byteCount: 1824),
    ])
    try await device.destroyTexture(image)
    let snapshots = try await ticket.wait()
    guard snapshots.count == 2, snapshots.allSatisfy({ $0.count == 456 }) else {
        throw GPUError.backendFailure("Unexpected textured render readback size")
    }
    let pixels = snapshots.flatMap { [$0[4 * 64 + 1], $0[4 * 64 + 6]] }
    guard pixels == [0xffff0000, 0xff00ffff, 0xffff0000, 0xff0000ff] else {
        throw GPUError.backendFailure("Unexpected textured render pixels: \(pixels)")
    }
    return pixels
}

private func samplerDiagnostic(_ device: Device) async throws -> [UInt32] {
    let source = ShaderInput.wgsl("""
        @group(2) @binding(0) var<storage, read_write> out: array<vec4f, 4>;
        @group(2) @binding(5) var image: texture_2d<f32>;
        @group(2) @binding(6) var pointSampler: sampler;
        @group(2) @binding(7) var linearSampler: sampler;
        const east = vec2i(2, -1) + vec2i(-1, 1);
        fn blend(uv: vec2f, mip: f32) -> vec4f { return textureSampleLevel(image, linearSampler, uv, mip); }
        @compute @workgroup_size(1) fn run() {
            let uv = vec2f(0.4375f, 0.5f);
            out[0] = textureSampleLevel(image, pointSampler, uv, 0, east,);
            out[1] = blend(uv, 0.0);
            out[2] = textureSampleGrad(image, pointSampler, vec2f(-0.25f, 0.5f), vec2f(0.25f, 0.0f), vec2f(0.0f, 0.25f), vec2i(-1, 0));
            out[3] = textureLoad(image, vec2u(1u, 0u), 0u);
        }
        """)
    let shader = try IndependentShaderCompiler().compileCompute(source, to: .wgsl)
    let pipeline = try await device.createComputePipeline(shader)
    let point = try await device.createSampler(.init(addressModeU: .repeat))
    let linear = try await device.createSampler(.init(minFilter: .linear, magFilter: .linear))
    let input = try await device.createBuffer(.init(size: 8, usage: [.copyDestination, .copySource]))
    let image = try await device.createTexture(.init(width: 2, height: 1, format: .bgra8Unorm,
                                                    usage: [.copyDestination, .textureBinding]))
    let out = try await device.createBuffer(.init(size: 64, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 64, usage: [.copyDestination, .mapRead]))
    func dispatch(_ point: ResourceID) -> BufferCommand {
        .dispatchComputeRanges(pipeline: pipeline, bindings: [.init(group: 2, binding: 0): .init(buffer: out)],
                               workgroups: .init(x: 1), textures: [.init(group: 2, binding: 5): image],
                               samplers: [.init(group: 2, binding: 6): point, .init(group: 2, binding: 7): linear])
    }
    do {
        _ = try await device.submitCommands([dispatch(image)])
        throw GPUError.backendFailure("Accepted a texture as a sampler binding")
    } catch GPUError.validation("Invalid sampler binding") {}
    let ticket = try await device.submitCommands([
        .write(buffer: input, offset: 0, words: [0xffff0000, 0xff00ff00]),
        .copyBufferToTexture(source: input, layout: .init(), destination: image, region: .init(width: 2, height: 1)),
        dispatch(point),
        .copy(source: out, destination: read, byteCount: 64),
        .read(buffer: read, byteCount: 64),
    ])
    try await device.destroySampler(point)
    let snapshots = try await ticket.wait()
    let actual = snapshots.first?.map(Float.init(bitPattern:)) ?? []
    let expected: [Float] = [0, 1, 0, 1, 0.625, 0.375, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1]
    guard actual.count == expected.count, zip(actual, expected).allSatisfy({ abs($0.0 - $0.1) < 0.004 }) else {
        throw GPUError.backendFailure("Unexpected WebGPU sampler values: \(actual)")
    }
    return actual.map { UInt32(($0 * 16).rounded()) }
}

private func renderSamplerDiagnostic(_ device: Device) async throws -> [UInt32] {
    let source = ShaderInput.wgsl("""
        @group(2) @binding(5) var image: texture_2d<f32>;
        @group(1) @binding(4) var linearSampler: sampler;
        @group(2) @binding(7) var pointSampler: sampler;

        fn point(uv: vec2f) -> vec4f { return textureSampleLevel(image, pointSampler, uv, 0); }

        @vertex fn vs(@location(0) position: vec2f) -> @builtin(position) vec4f {
            let offset = point(vec2f(0.25, 0.5)).r;
            return vec4f(position.x + offset, position.y, 0.0, 1.0);
        }

        @fragment fn fs() -> @location(0) vec4f {
            let uv = vec2f(0.4375, 0.5);
            const east = vec2(1, 0);
            let filtered = textureSample(image, linearSampler, vec2f(-0.0625, 0.5), east);
            let biased = textureSampleBias(image, linearSampler, uv, 2.0, east);
            let grad = textureSampleGrad(image, pointSampler, vec2f(0.75, 0.5), vec2f(0.25, 0.0), vec2f(0.0, 0.25));
            return vec4f(point(uv).r * grad.g, filtered.g, biased.g, 1.0);
        }
        """)
    let compiler = IndependentShaderCompiler()
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .wgsl)
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: .wgsl)
    let pipeline = try await device.createRenderPipeline(
        vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm,
        vertexBuffers: [.init(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float32x2)])])
    let upload = try await device.createBuffer(.init(size: 8, usage: [.copyDestination, .copySource]))
    let vertices = try await device.createBuffer(.init(size: 24, usage: [.vertex, .copyDestination]))
    let indices = try await device.createBuffer(.init(size: 8, usage: [.index, .copyDestination]))
    let image = try await device.createTexture(.init(width: 2, height: 1, format: .rgba8Unorm,
                                                    usage: [.copyDestination, .textureBinding]))
    let target = try await device.createTexture(.init(width: 8, height: 8, format: .rgba8Unorm,
                                                      usage: [.renderAttachment, .copySource]))
    let read = try await device.createBuffer(.init(size: 1824, usage: [.copyDestination, .mapRead]))
    let point = try await device.createSampler()
    let linear = try await device.createSampler(.init(minFilter: .linear, magFilter: .linear))
    let textures: [ComputeBindingSlot: ResourceID] = [.init(group: 2, binding: 5): image]
    let sharedSlot = ComputeBindingSlot(group: 2, binding: 7)
    let linearSlot = ComputeBindingSlot(group: 1, binding: 4)
    let clear = RenderClearColor(red: 0, green: 0, blue: 1, alpha: 1)
    let streams = [VertexBufferBinding(buffer: vertices)]
    func draw(_ filtered: ResourceID) -> BufferCommand {
        .render(pipeline: pipeline, target: target, clear: clear, vertexCount: 3, vertexBuffers: streams,
                textures: textures, samplers: [sharedSlot: point, linearSlot: filtered])
    }
    do {
        _ = try await device.submitCommands([draw(image)])
        throw GPUError.backendFailure("Accepted a texture as a render sampler binding")
    } catch GPUError.validation("Invalid sampler binding") {}
    let ticket = try await device.submitCommands([
        .write(buffer: upload, offset: 0, words: [0xff0000ff, 0xff00ff00]),
        .copyBufferToTexture(source: upload, layout: .init(), destination: image, region: .init(width: 2, height: 1)),
        .write(buffer: vertices, offset: 0, words: [Float(-1), -1, 3, -1, -1, 3].map(\.bitPattern)),
        .write(buffer: indices, offset: 0, words: [0x00010000, 2]),
        draw(linear),
        .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256),
        .read(buffer: read, byteCount: 1824),
        .renderIndexed(pipeline: pipeline, target: target, clear: clear,
                       indexBuffer: .init(buffer: indices, format: .uint16), draw: .init(indexCount: 3),
                       vertexBuffers: streams, textures: textures, samplers: [sharedSlot: point, linearSlot: point]),
        .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256),
        .read(buffer: read, byteCount: 1824),
    ])
    try await device.destroySampler(linear)
    let snapshots = try await ticket.wait()
    guard snapshots.count == 2, snapshots.allSatisfy({ $0.count == 456 }) else {
        throw GPUError.backendFailure("Unexpected render sampler readback size")
    }
    let pixels = snapshots.flatMap { [$0[4 * 64 + 1], $0[4 * 64 + 6]] }
    guard pixels == [0xffff0000, 0xffff60ff, 0xffff0000, 0xffff00ff] else {
        throw GPUError.backendFailure("Unexpected render sampler pixels: \(pixels)")
    }
    return pixels
}

private func mipTextureDiagnostic(_ device: Device) async throws -> [UInt32] {
    let source = ShaderInput.wgsl("""
        @group(0) @binding(0) var<storage, read_write> out: array<vec4f, 7>;
        @group(0) @binding(1) var image: texture_2d<f32>;
        @group(0) @binding(2) var filter: sampler;
        @compute @workgroup_size(1) fn run() {
            out[0] = textureLoad(image, vec2u(4, 2), 0u);
            out[1] = textureLoad(image, vec2u(1, 0), 1u);
            out[2] = textureLoad(image, vec2u(0, 0), 2u);
            out[3] = textureSampleLevel(image, filter, vec2f(0.75, 0.5), 0.5);
            out[4] = textureLoad(image, vec2u(0, 0), 1u);
            let size = vec2f(textureDimensions(image, 1u));
            out[5] = vec4f(f32(textureNumLevels(image)), size.x, size.y, f32(i32(textureDimensions(image, 2u).x)));
            out[6] = textureLoad(image, vec2u(2, 0), 1u);
        }
        """)
    let compiler = IndependentShaderCompiler()
    let compute = try await device.createComputePipeline(compiler.compileCompute(source, to: .wgsl))
    let clearShader = ShaderInput.wgsl("""
        @vertex fn vs() -> @builtin(position) vec4f { return vec4f(0, 0, 0, 1); }
        @fragment fn fs() -> @location(0) vec4f { return vec4f(0, 0, 1, 1); }
        """)
    let render = try await device.createRenderPipeline(
        vertex: compiler.compileRenderStage(clearShader, entryPoint: "vs", to: .wgsl),
        fragment: compiler.compileRenderStage(clearShader, entryPoint: "fs", to: .wgsl), colorFormat: .rgba8Unorm)
    let image = try await device.createTexture(.init(width: 5, height: 3, format: .rgba8Unorm,
        usage: [.copySource, .copyDestination, .textureBinding, .renderAttachment], mipLevelCount: 3))
    let upload = try await device.createBuffer(.init(size: 536, usage: [.copySource, .copyDestination]))
    let out = try await device.createBuffer(.init(size: 112, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 112, usage: [.copyDestination, .mapRead]))
    let filter = try await device.createSampler(.init(mipmapFilter: .linear))
    let dispatch = BufferCommand.dispatchCompute(
        pipeline: compute, bindings: [.init(binding: 0): out], workgroups: .init(x: 1),
        textures: [.init(binding: 1): image], samplers: [.init(binding: 2): filter])
    let red = Array(repeating: UInt32(0xff0000ff), count: 5)
    let green: UInt32 = 0xff00ff00
    let blue = RenderClearColor(red: 0, green: 0, blue: 1, alpha: 1)
    do {
        _ = try await device.submitCommands([
            .render(pipeline: render, target: image, clear: blue, vertexCount: 0, targetMipLevel: 3),
        ])
        throw GPUError.backendFailure("Accepted an invalid render mip level")
    } catch GPUError.validation {}
    let first = try await device.submitCommands([
        .write(buffer: upload, offset: 0, words: red),
        .write(buffer: upload, offset: 256, words: red),
        .write(buffer: upload, offset: 512, words: red + [green]),
        .copyBufferToTexture(source: upload, layout: .init(bytesPerRow: 256), destination: image,
                             region: .init(width: 5, height: 3)),
        .copyBufferToTexture(source: upload, layout: .init(offset: 532), destination: image,
                             region: .init(x: 1, width: 1, height: 1, mipLevel: 1)),
        dispatch,
        .copy(source: out, destination: read, byteCount: 112),
        .read(buffer: read, byteCount: 112),
        .copyTextureToBuffer(source: image, destination: read, bytesPerRow: 256, mipLevel: 1),
        .read(buffer: read, byteCount: 8),
    ])
    let initial: [Float] = [
        1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0.5, 0.5, 0, 1,
        0, 0, 0, 0, 3, 2, 1, 1, 0, 0, 0, 0,
    ]
    let beforeRender = try await first.wait()
    guard beforeRender == [initial.map(\.bitPattern), [0, green]] else {
        throw GPUError.backendFailure("Unexpected mip upload and sampling results: \(beforeRender)")
    }
    let second = try await device.submitCommands([
        .render(pipeline: render, target: image, clear: blue, vertexCount: 0, targetMipLevel: 2),
        dispatch,
        .copy(source: out, destination: read, byteCount: 112),
        .read(buffer: read, byteCount: 112),
        .copyTextureToBuffer(source: image, destination: read, bytesPerRow: 256, mipLevel: 2),
        .read(buffer: read, byteCount: 4),
    ])
    var rendered = initial
    rendered.replaceSubrange(8..<12, with: [0, 0, 1, 1])
    let afterRender = try await second.wait()
    guard afterRender == [rendered.map(\.bitPattern), [0xffff0000]] else {
        throw GPUError.backendFailure("Unexpected mip render results: \(afterRender)")
    }
    return beforeRender[1] + afterRender[1] + Array(afterRender[0][12..<14])
}

private func textureViewDiagnostic(_ device: Device) async throws -> [UInt32] {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl("""
        @group(0) @binding(0) var image: texture_2d<f32>;
        @vertex fn vs(@location(0) p: vec2f) -> @builtin(position) vec4f {
            return vec4f(p, 0.0, 1.0);
        }
        @fragment fn fs() -> @location(0) vec4f {
            return textureLoad(image, vec2u(0, 0), 0u);
        }
        """)
    let render = try await device.createRenderPipeline(
        vertex: compiler.compileRenderStage(source, entryPoint: "vs", to: .wgsl),
        fragment: compiler.compileRenderStage(source, entryPoint: "fs", to: .wgsl),
        colorFormat: .rgba8Unorm, vertexBuffers: [.init(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float32x2)])])
    let compute = try await device.createComputePipeline(compiler.compileCompute(.wgsl("""
        @group(0) @binding(0) var image: texture_2d<f32>;
        @group(0) @binding(1) var<storage, read_write> out: array<vec4f, 2>;
        @group(0) @binding(2) var state: sampler;
        @compute @workgroup_size(1) fn cs() {
            let p = textureLoad(image, vec2u(0), 0u);
            let q = textureLoad(image, vec2u(0), 1u);
            let invalid = textureLoad(image, vec2u(0), 2u);
            out[0] = vec4f(p.g, q.a + invalid.a, f32(textureDimensions(image).x), f32(textureNumLevels(image)));
            let left = textureSampleBaseClampToEdge(image, state, vec2f(-0.25, 0.25));
            let bottom = textureSampleBaseClampToEdge(image, state, vec2f(0.75, 1.25));
            let center = textureSampleBaseClampToEdge(image, state, vec2f(0.5));
            out[1] = vec4f(left.g, left.r, bottom.b, center.r);
        }
        """), to: .wgsl))
    let image = try await device.createTexture(.init(width: 4, height: 4, format: .rgba8Unorm,
        usage: [.copySource, .copyDestination, .textureBinding, .renderAttachment], mipLevelCount: 3))
    let sampled = try await device.createTextureView(texture: image, .init(baseMipLevel: 1))
    let attachment = try await device.createTextureView(texture: image, .init(mipLevelCount: 1))
    let overlap = try await device.createTextureView(texture: image, .init(baseMipLevel: 1, mipLevelCount: 1))
    let state = try await device.createSampler(.init(addressModeU: .repeat, addressModeV: .repeat,
                                                     minFilter: .linear, magFilter: .linear, mipmapFilter: .linear))
    let upload = try await device.createBuffer(.init(size: 264, usage: [.copySource, .copyDestination]))
    let vertices = try await device.createBuffer(.init(size: 24, usage: [.vertex, .copyDestination]))
    let out = try await device.createBuffer(.init(size: 32, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 784, usage: [.copyDestination, .mapRead]))
    let clear = RenderClearColor(red: 0, green: 0, blue: 0, alpha: 1)
    let textureSlot = ComputeBindingSlot(binding: 0)
    func draw(to target: ResourceID) -> BufferCommand {
        .render(pipeline: render, target: target, clear: clear, vertexCount: 3,
                vertexBuffers: [.init(buffer: vertices)], textures: [textureSlot: sampled])
    }
    do {
        _ = try await device.submitCommands([draw(to: overlap)])
        throw GPUError.backendFailure("Accepted an overlapping render texture view")
    } catch GPUError.validation("Render attachment cannot be bound as a sampled texture") {}
    let dispatch = BufferCommand.dispatchCompute(
        pipeline: compute, bindings: [.init(binding: 1): out], workgroups: .init(x: 1), textures: [textureSlot: sampled],
        samplers: [.init(binding: 2): state])
    let green: UInt32 = 0xff00ff00
    let ticket = try await device.submitCommands([
        .write(buffer: upload, offset: 0, words: [green, 0xff0000ff]),
        .write(buffer: upload, offset: 256, words: [0xffff0000, 0xffffffff]),
        .copyBufferToTexture(source: upload, layout: .init(bytesPerRow: 256), destination: image,
                             region: .init(width: 2, height: 2, mipLevel: 1)),
        .write(buffer: vertices, offset: 0, words: [Float(-1), -1, 3, -1, -1, 3].map(\.bitPattern)),
        draw(to: attachment),
        dispatch,
        .copy(source: out, destination: read, byteCount: 32),
        .read(buffer: read, byteCount: 32),
        .copyTextureToBuffer(source: image, destination: read, bytesPerRow: 256),
        .read(buffer: read, byteCount: 16),
    ])
    let expected = [Float(1), 0, 2, 2, 1, 0, 1, 0.5].map(\.bitPattern)
    let snapshots = try await ticket.wait()
    guard snapshots == [expected, Array(repeating: green, count: 4)] else {
        throw GPUError.backendFailure("Unexpected render and compute texture view results: \(snapshots)")
    }
    let viewOnly = try await device.submitCommands([
        dispatch,
        .copy(source: out, destination: read, byteCount: 32),
        .read(buffer: read, byteCount: 32),
    ])
    try await device.destroyTexture(image)
    do {
        _ = try await device.submitCommands([dispatch])
        throw GPUError.backendFailure("Accepted a view of a destroyed texture")
    } catch GPUError.validation("Texture view has a destroyed parent texture") {}
    try await device.destroyTextureView(sampled)
    try await device.destroyTextureView(attachment)
    let viewSnapshot = try await viewOnly.wait()
    guard viewSnapshot == [expected] else {
        throw GPUError.backendFailure("Unexpected retained texture view results: \(viewSnapshot)")
    }
    return viewSnapshot[0] + [snapshots[1][0]]
}

private func textureGatherDiagnostic(_ device: Device) async throws -> [UInt32] {
    let source = ShaderInput.wgsl("""
        @group(0) @binding(0) var<storage, read_write> out: array<vec4f, 4>;
        @group(0) @binding(1) var image: texture_2d<f32>;
        @group(0) @binding(2) var state: sampler;
        const green = 2u - 1u;
        fn gatherRed(uv: vec2f) -> vec4f { return textureGather(0, image, state, uv); }
        @compute @workgroup_size(1) fn run(@builtin(global_invocation_id) id: vec3u) {
            if id.x == 0u {
                let uv = vec2f(0.33333334, 0.5);
                out[0] = gatherRed(uv);
                out[1] = textureGather(green, image, state, uv, vec2(1, 0));
                out[2] = textureGather(2i, image, state, uv, vec2i(-1, 0),);
                out[3] = textureGather(3u, image, state, uv);
            }
        }
        """)
    let shader = try IndependentShaderCompiler().compileCompute(source, to: .wgsl)
    let pipeline = try await device.createComputePipeline(shader)
    let sampler = try await device.createSampler(.init(addressModeU: .repeat))
    let image = try await device.createTexture(.init(width: 6, height: 4, format: .rgba8Unorm,
        usage: [.copyDestination, .textureBinding], mipLevelCount: 2))
    let view = try await device.createTextureView(texture: image, .init(baseMipLevel: 1))
    let input = try await device.createBuffer(.init(size: 268, usage: [.copyDestination, .copySource]))
    let out = try await device.createBuffer(.init(size: 64, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 64, usage: [.copyDestination, .mapRead]))
    func dispatch(_ texture: ResourceID) -> BufferCommand {
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): out], workgroups: .init(x: 1),
                         textures: [.init(binding: 1): texture], samplers: [.init(binding: 2): sampler])
    }
    let ticket = try await device.submitCommands([
        .write(buffer: input, offset: 0, words: [0x8c190a14, 0x962d1e28, 0xa041323c]),
        .write(buffer: input, offset: 256, words: [0xaa554650, 0xb4695a64, 0xbe7d6e78]),
        .copyBufferToTexture(source: input, layout: .init(bytesPerRow: 256), destination: image,
                             region: .init(width: 3, height: 2, mipLevel: 1)),
        dispatch(view),
        .copy(source: out, destination: read, byteCount: 64),
        .read(buffer: read, byteCount: 64),
        dispatch(image),
        .copy(source: out, destination: read, byteCount: 64),
        .read(buffer: read, byteCount: 64),
    ])
    let snapshots = try await ticket.wait()
    guard snapshots.count == 2, snapshots[1] == Array(repeating: 0, count: 16) else {
        throw GPUError.backendFailure("Unexpected texture gather base mip result: \(snapshots)")
    }
    let lanes = snapshots[0].map { (Float(bitPattern: $0) * 255).rounded() }
    guard lanes == [80, 100, 40, 20, 90, 110, 50, 30, 125, 85, 25, 65, 170, 180, 150, 140] else {
        throw GPUError.backendFailure("Unexpected texture gather channel or order: \(lanes)")
    }
    return lanes.map(UInt32.init)
}

private func storageTextureDiagnostic(_ device: Device) async throws -> [UInt32] {
    let source = ShaderInput.wgsl("""
        @group(2) @binding(3) var source: texture_2d<f32>;
        @group(2) @binding(4) var destination: texture_storage_2d<rgba8unorm, write>;
        fn writeColor(x: i32, color: vec4f) { textureStore(destination, vec2i(x, 0), color); }
        @compute @workgroup_size(5) fn run(@builtin(global_invocation_id) id: vec3u) {
            let x = i32(id.x) - 1;
            if id.x == 4u {
                textureStore(destination, vec2u(0xffffffffu, 0u), vec4f(1, 1, 0, 1));
            } else if x != 0 {
                let dimensions = textureDimensions(destination);
                let red = textureLoad(source, vec2u(0), 0).r;
                writeColor(x, vec4f(red, 0.0, f32(dimensions.x - 1u), 1.0));
            }
        }
        """)
    let shader = try IndependentShaderCompiler().compileCompute(source, to: .wgsl)
    let pipeline = try await device.createComputePipeline(shader)
    let image = try await device.createTexture(.init(width: 4, height: 2, format: .rgba8Unorm,
        usage: [.copySource, .copyDestination, .textureBinding, .storageBinding], mipLevelCount: 2))
    let input = try await device.createTextureView(texture: image, .init(mipLevelCount: 1))
    let output = try await device.createTextureView(texture: image, .init(baseMipLevel: 1))
    let upload = try await device.createBuffer(.init(size: 272, usage: [.copySource, .copyDestination]))
    let read = try await device.createBuffer(.init(size: 272, usage: [.copyDestination, .mapRead]))
    let inputSlot = ComputeBindingSlot(group: 2, binding: 3)
    let outputSlot = ComputeBindingSlot(group: 2, binding: 4)
    func dispatch(_ input: ResourceID, _ output: ResourceID) -> BufferCommand {
        .dispatchCompute(pipeline: pipeline, bindings: [:], workgroups: .init(x: 1),
                         textures: [inputSlot: input, outputSlot: output])
    }
    do {
        _ = try await device.submitCommands([dispatch(output, output)])
        throw GPUError.backendFailure("Accepted overlapping sampled and writable storage texture bindings")
    } catch GPUError.validation("Aliased writable compute textures") {}
    let red = Array(repeating: UInt32(0xff0000ff), count: 4)
    let ticket = try await device.submitCommands([
        .write(buffer: upload, offset: 0, words: red),
        .write(buffer: upload, offset: 256, words: red),
        .copyBufferToTexture(source: upload, layout: .init(bytesPerRow: 256), destination: image,
                             region: .init(width: 4, height: 2)),
        dispatch(input, output),
        .copyTextureToBuffer(source: image, destination: read, bytesPerRow: 256, mipLevel: 1),
        .read(buffer: read, byteCount: 8),
        .copyTextureToBuffer(source: image, destination: read, bytesPerRow: 256),
        .read(buffer: read, byteCount: 272),
    ])
    let snapshots = try await ticket.wait()
    guard snapshots == [[0, 0xffff00ff], red + Array(repeating: 0, count: 60) + red] else {
        throw GPUError.backendFailure("Unexpected storage texture pixels: \(snapshots)")
    }
    return snapshots[0] + [snapshots[1][0]]
}

private func fragmentStorageTextureDiagnostic(_ device: Device) async throws -> [UInt32] {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl("""
        @group(2) @binding(4) var destination: texture_storage_2d<rgba8unorm, write>;
        struct VertexOut {
            @builtin(position) position: vec4f,
            @location(0) x: f32,
        }
        @vertex fn vs(@location(0) point: vec2f) -> VertexOut {
            return VertexOut(vec4f(point, 0, 1), point.x);
        }
        fn writeColor(x: i32, color: vec4f) {
            textureStore(destination, vec2i(x, 0), color);
        }
        @fragment fn fs(@location(0) x: f32) -> @location(0) vec4f {
            if x < 0.0 { writeColor(1, vec4f(1, 0, 0, 1)); }
            else { writeColor(i32(textureDimensions(destination).x) - 2i, vec4f(0, 0, 1, 1)); }
            return vec4f(0, 1, 0, 1);
        }
        """)
    let pipeline = try await device.createRenderPipeline(
        vertex: compiler.compileRenderStage(source, entryPoint: "vs", to: .wgsl),
        fragment: compiler.compileRenderStage(source, entryPoint: "fs", to: .wgsl),
        colorFormat: .rgba8Unorm, vertexBuffers: [.init(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float32x2)])])
    let image = try await device.createTexture(.init(width: 4, height: 2, format: .rgba8Unorm,
        usage: [.renderAttachment, .copySource, .storageBinding], mipLevelCount: 2))
    let storage = try await device.createTextureView(texture: image, .init(mipLevelCount: 1))
    let attachment = try await device.createTextureView(texture: image, .init(baseMipLevel: 1))
    let overlapping = try await device.createTextureView(texture: image, .init(baseMipLevel: 1))
    let storageIndexed = try await device.createTexture(.init(width: 4, height: 2, format: .rgba8Unorm,
        usage: [.copySource, .storageBinding]))
    let vertices = try await device.createBuffer(.init(size: 24, usage: [.vertex, .copyDestination]))
    let indices = try await device.createBuffer(.init(size: 8, usage: [.index, .copyDestination]))
    let read = try await device.createBuffer(.init(size: 272, usage: [.copyDestination, .mapRead]))
    let clear = RenderClearColor(red: 0, green: 0, blue: 0, alpha: 1)
    let slot = ComputeBindingSlot(group: 2, binding: 4)
    let streams = [VertexBufferBinding(buffer: vertices)]
    func draw(_ output: ResourceID) -> BufferCommand {
        .render(pipeline: pipeline, target: attachment, clear: clear, vertexCount: 3,
                vertexBuffers: streams, textures: [slot: output])
    }
    do {
        _ = try await device.submitCommands([draw(overlapping)])
        throw GPUError.backendFailure("Accepted overlapping color and storage texture views")
    } catch GPUError.validation("Aliased writable render textures") {}
    let ticket = try await device.submitCommands([
        .write(buffer: vertices, offset: 0, words: [Float(-1), -1, 3, -1, -1, 3].map(\.bitPattern)),
        .write(buffer: indices, offset: 0, words: [0x00010000, 2]),
        draw(storage),
        .copyTextureToBuffer(source: image, destination: read, bytesPerRow: 256),
        .read(buffer: read, byteCount: 272),
        .copyTextureToBuffer(source: image, destination: read, bytesPerRow: 256, mipLevel: 1),
        .read(buffer: read, byteCount: 8),
        .renderIndexed(pipeline: pipeline, target: attachment, clear: clear,
                       indexBuffer: .init(buffer: indices, format: .uint16), draw: .init(indexCount: 3),
                       vertexBuffers: streams, textures: [slot: storageIndexed]),
        .copyTextureToBuffer(source: storageIndexed, destination: read, bytesPerRow: 256),
        .read(buffer: read, byteCount: 272),
    ])
    let expectedStorage: [UInt32] = [0, 0xff0000ff, 0xffff0000, 0] + Array(repeating: 0, count: 64)
    let expected: [[UInt32]] = [expectedStorage, [0xff00ff00, 0xff00ff00], expectedStorage]
    let snapshots = try await ticket.wait()
    guard snapshots == expected else { throw GPUError.backendFailure("Unexpected fragment storage pixels: \(snapshots)") }
    return Array(snapshots[0][1...2]) + snapshots[1] + Array(snapshots[2][1...2])
}

private func readWriteStorageDiagnostic(_ device: Device) async throws -> [UInt32] {
    let source = ShaderInput.wgsl("""
        @group(0) @binding(0) var mask: texture_storage_2d<rgba8unorm, read>;
        @group(0) @binding(1) var image: texture_storage_2d<r32uint, read_write>;
        @group(0) @binding(2) var<storage, read_write> before: array<vec4u, 4>;
        @compute @workgroup_size(4) fn run(@builtin(global_invocation_id) id: vec3u) {
            if id.x == 3u {
                before[3u] = textureLoad(image, vec2u(0xffffffffu, 0u));
            } else {
                let coords = vec2i(i32(id.x) - 1i, 0i);
                let old = textureLoad(image, coords);
                before[id.x] = old;
                if textureLoad(mask, coords).g > 0.5 {
                    textureStore(image, coords, old + vec4u(5u, 99u, 99u, 99u));
                }
            }
        }
        """)
    let shader = try IndependentShaderCompiler().compileCompute(source, to: .wgsl)
    let pipeline = try await device.createComputePipeline(shader)
    let mask = try await device.createTexture(.init(width: 2, height: 1, format: .rgba8Unorm,
        usage: [.copyDestination, .storageBinding]))
    let image = try await device.createTexture(.init(width: 2, height: 1, format: .r32Uint,
        usage: [.copySource, .copyDestination, .storageBinding]))
    let upload = try await device.createBuffer(.init(size: 16, usage: [.copySource, .copyDestination]))
    let out = try await device.createBuffer(.init(size: 64, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 64, usage: [.copyDestination, .mapRead]))
    let dispatch = BufferCommand.dispatchCompute(
        pipeline: pipeline, bindings: [.init(binding: 2): out], workgroups: .init(x: 1),
        textures: [.init(binding: 0): mask, .init(binding: 1): image])
    let ticket = try await device.submitCommands([
        .write(buffer: upload, offset: 0, words: [0xff00ff00, 0xff0000ff, 10, 20]),
        .copyBufferToTexture(source: upload, layout: .init(), destination: mask,
                             region: .init(width: 2, height: 1)),
        dispatch,
        .copy(source: out, destination: read, byteCount: 64),
        .read(buffer: read, byteCount: 64),
        .copyTextureToBuffer(source: image, destination: read, bytesPerRow: 256),
        .read(buffer: read, byteCount: 8),
        .copyBufferToTexture(source: upload, layout: .init(offset: 8), destination: image,
                             region: .init(width: 2, height: 1)),
        dispatch,
        .copy(source: out, destination: read, byteCount: 64),
        .read(buffer: read, byteCount: 64),
        .copyTextureToBuffer(source: image, destination: read, bytesPerRow: 256),
        .read(buffer: read, byteCount: 8),
    ])
    let expected: [[UInt32]] = [
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0], [5, 0],
        [0, 0, 0, 0, 10, 0, 0, 1, 20, 0, 0, 1, 0, 0, 0, 0], [15, 20],
    ]
    let snapshots = try await ticket.wait()
    guard snapshots == expected else { throw GPUError.backendFailure("Unexpected readable storage results: \(snapshots)") }
    return snapshots[1] + [0, 4, 7, 8, 11, 12].map { snapshots[2][$0] } + snapshots[3]
}

private func textureBarrierDiagnostic(_ device: Device) async throws -> [UInt32] {
    let source = ShaderInput.wgsl("""
        @group(2) @binding(0) var image: texture_storage_2d<r32uint, read_write>;
        @group(0) @binding(0) var<storage, read_write> output: array<u32, 384>;
        fn syncImage() { textureBarrier(); }
        fn rendezvous() { syncImage(); }
        @compute @workgroup_size(128)
        fn run(@builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) group: vec3u) {
            let start = group.x * 128u;
            let index = start + lane;
            textureStore(image, vec2u(index, 0u), vec4u(group.x * 1000u + lane + 10u));
            rendezvous();
            let neighbor = start + (lane + 64u) % 128u;
            output[index] = textureLoad(image, vec2u(neighbor, 0u)).r + 7u;
        }
        """)
    let shader = try IndependentShaderCompiler().compileCompute(source, to: .wgsl)
    let pipeline = try await device.createComputePipeline(shader)
    let image = try await device.createTexture(.init(width: 384, height: 1, format: .r32Uint,
        usage: [.copySource, .storageBinding]))
    let output = try await device.createBuffer(.init(size: 1536, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 1536, usage: [.copyDestination, .mapRead]))
    let ticket = try await device.submitCommands([
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 3),
                         textures: [.init(group: 2, binding: 0): image]),
        .copy(source: output, destination: read, byteCount: 1536),
        .read(buffer: read, byteCount: 1536),
        .copyTextureToBuffer(source: image, destination: read, bytesPerRow: 1536),
        .read(buffer: read, byteCount: 1536),
    ])
    let writes = (0..<384).map { (index: Int) -> UInt32 in
        let group = UInt32(index / 128)
        let lane = UInt32(index % 128)
        return group * 1000 + lane + 10
    }
    let neighbors = (0..<384).map { (index: Int) -> UInt32 in
        let start = index / 128 * 128
        let rotated = (index + 64) % 128
        return writes[start + rotated] + 7
    }
    let snapshots = try await ticket.wait()
    guard snapshots == [neighbors, writes] else { throw GPUError.backendFailure("Unexpected texture barrier readback") }
    return [0, 127, 128, 255, 256, 383].map { snapshots[0][$0] }
}

private func renderReadableStorageDiagnostic(_ device: Device) async throws -> [UInt32] {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl("""
        @group(2) @binding(5) var mask: texture_storage_2d<rgba8unorm, read>;
        @group(1) @binding(4) var image: texture_storage_2d<r32uint, read_write>;

        struct VertexOut {
            @builtin(position) position: vec4f,
            @location(0) x: f32,
        }
        @vertex fn vs(@location(0) point: vec2f) -> VertexOut {
            let x = point.x + textureLoad(mask, vec2u(0u)).r;
            return VertexOut(vec4f(x, point.y, 0.0, 1.0), x);
        }
        @fragment fn fs(@location(0) x: f32) -> @location(0) vec4f {
            let coords = vec2u(0u);
            let old = textureLoad(image, coords);
            if x > 0.0 { textureStore(image, coords, old + vec4u(5u)); }
            let red = select(0.0, 1.0, old.r == 10u);
            return vec4f(red, textureLoad(mask, coords).g, 0.0, 1.0);
        }
        """)
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .wgsl)
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: .wgsl)
    let pipeline = try await device.createRenderPipeline(
        vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm,
        vertexBuffers: [.init(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float32x2)])])
    let mask = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm,
        usage: [.copyDestination, .storageBinding]))
    let image = try await device.createTexture(.init(width: 1, height: 1, format: .r32Uint,
        usage: [.copySource, .copyDestination, .storageBinding]))
    let target = try await device.createTexture(.init(width: 2, height: 1, format: .rgba8Unorm,
        usage: [.copySource, .renderAttachment]))
    let upload = try await device.createBuffer(.init(size: 8, usage: [.copySource, .copyDestination]))
    let vertices = try await device.createBuffer(.init(size: 24, usage: [.vertex, .copyDestination]))
    let read = try await device.createBuffer(.init(size: 8, usage: [.copyDestination, .mapRead]))
    let clear = RenderClearColor(red: 0, green: 0, blue: 1, alpha: 1)
    let ticket = try await device.submitCommands([
        .write(buffer: upload, offset: 0, words: [0xff00ffff, 10]),
        .copyBufferToTexture(source: upload, layout: .init(), destination: mask, region: .init(width: 1, height: 1)),
        .copyBufferToTexture(source: upload, layout: .init(offset: 4), destination: image, region: .init(width: 1, height: 1)),
        .write(buffer: vertices, offset: 0, words: [Float(-1), -1, 3, -1, -1, 3].map(\.bitPattern)),
        .render(pipeline: pipeline, target: target, clear: clear, vertexCount: 3,
                vertexBuffers: [.init(buffer: vertices)],
                textures: [.init(group: 2, binding: 5): mask, .init(group: 1, binding: 4): image]),
        .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256),
        .read(buffer: read, byteCount: 8),
        .copyTextureToBuffer(source: image, destination: read, bytesPerRow: 256),
        .read(buffer: read, byteCount: 4),
    ])
    let snapshots = try await ticket.wait()
    guard snapshots == [[0xffff0000, 0xff00ffff], [15]] else {
        throw GPUError.backendFailure("Unexpected readable render storage results: \(snapshots)")
    }
    return snapshots[0] + snapshots[1]
}

private func depthAttachmentDiagnostic(_ device: Device) async throws -> [UInt32] {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl("""
        struct VertexOut {
            @builtin(position) position: vec4f,
            @location(0) tint: vec4f,
        }
        @vertex fn vs(@location(0) point: vec3f, @location(1) tint: vec4f) -> VertexOut {
            return VertexOut(vec4f(point, 1.0), tint);
        }
        @fragment fn fs(@location(0) tint: vec4f) -> @location(0) vec4f { return tint; }
        """)
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .wgsl)
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: .wgsl)
    let layouts = [VertexBufferLayout(stride: 28, attributes: [
        .init(location: 0, offset: 0, format: .float32x3),
        .init(location: 1, offset: 12, format: .float32x4),
    ])]
    let pipeline = try await device.createRenderPipeline(
        vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm, vertexBuffers: layouts,
        depthState: .init(compare: .less, writeEnabled: true))
    let readOnly = try await device.createRenderPipeline(
        vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm, vertexBuffers: layouts,
        depthState: .init(compare: .always, writeEnabled: false))
    let target = try await device.createTexture(.init(width: 2, height: 1, format: .rgba8Unorm,
        usage: [.copySource, .renderAttachment]))
    let depth = try await device.createTexture(.init(width: 4, height: 2, format: .depth32Float,
        usage: [.copySource, .renderAttachment], mipLevelCount: 2))
    let depthView = try await device.createTextureView(texture: depth, .init(baseMipLevel: 1))
    let vertices = try await device.createBuffer(.init(size: 168, usage: [.vertex, .copyDestination]))
    let indices = try await device.createBuffer(.init(size: 8, usage: [.index, .copyDestination]))
    let read = try await device.createBuffer(.init(size: 8, usage: [.copyDestination, .mapRead]))
    let blue = RenderClearColor(red: 0, green: 0, blue: 1, alpha: 1)
    let streams = [VertexBufferBinding(buffer: vertices)]
    func draw(_ count: UInt32, depth attachment: RenderDepthAttachment?) -> BufferCommand {
        .render(pipeline: pipeline, target: target, clear: blue, vertexCount: count,
                vertexBuffers: streams, depth: attachment)
    }
    let load = RenderDepthAttachment(texture: depthView, loadAction: .load)
    let clear = RenderDepthAttachment(texture: depthView, loadAction: .clear(1))
    do {
        _ = try await device.submitCommands([draw(3, depth: nil)])
        throw GPUError.backendFailure("Accepted a render command without its required depth attachment")
    } catch GPUError.validation("Depth attachment does not match the render pipeline") {}
    let triangles: [Float] = [
        -1, -1, 0.25, 0, 1, 0, 1,
         3, -1, 0.25, 0, 1, 0, 1,
        -1,  3, 0.25, 0, 1, 0, 1,
        -1, -1, 0.75, 1, 0, 0, 1,
         3, -1, 0.75, 1, 0, 0, 1,
        -1,  3, 0.75, 1, 0, 0, 1,
    ]
    let ticket = try await device.submitCommands([
        .write(buffer: vertices, offset: 0, words: triangles.map(\.bitPattern)),
        .write(buffer: indices, offset: 0, words: [0x00040003, 5]),
        draw(3, depth: load),
        .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256),
        .read(buffer: read, byteCount: 8),
        draw(6, depth: clear),
        .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256),
        .read(buffer: read, byteCount: 8),
        .renderIndexed(pipeline: pipeline, target: target, clear: blue,
                       indexBuffer: .init(buffer: indices, format: .uint16), draw: .init(indexCount: 3),
                       vertexBuffers: streams, depth: load),
        .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256),
        .read(buffer: read, byteCount: 8),
        .renderIndexed(pipeline: readOnly, target: target, clear: blue,
                       indexBuffer: .init(buffer: indices, format: .uint16), draw: .init(indexCount: 3),
                       vertexBuffers: streams, depth: load),
        .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256),
        .read(buffer: read, byteCount: 8),
        .copyTextureToBuffer(source: depth, destination: read, bytesPerRow: 256, mipLevel: 1),
        .read(buffer: read, byteCount: 8),
    ])
    try await device.destroyTextureView(depthView)
    let expected: [[UInt32]] = [
        [0xffff0000, 0xffff0000], [0xff00ff00, 0xff00ff00],
        [0xffff0000, 0xffff0000], [0xff0000ff, 0xff0000ff],
        Array(repeating: Float(0.25).bitPattern, count: 2),
    ]
    let snapshots = try await ticket.wait()
    guard snapshots == expected else { throw GPUError.backendFailure("Unexpected depth render results: \(snapshots)") }
    return snapshots.map { $0[0] }
}

private func depthTextureDiagnostic(_ device: Device) async throws -> [UInt32] {
    let compiler = IndependentShaderCompiler()
    let compute = try await device.createComputePipeline(compiler.compileCompute(.wgsl("""
        @group(2) @binding(1) var image: texture_depth_2d;
        @group(0) @binding(0) var<storage, read_write> output: array<f32, 8>;
        fn fetch(p: vec2i) -> f32 { return textureLoad(image, p, 0i); }
        @compute @workgroup_size(1) fn run() {
            let size = textureDimensions(image, 0u);
            output[0u] = fetch(vec2i(0));
            output[1u] = fetch(vec2i(-1, 0));
            output[2u] = textureLoad(image, vec2u(0xffffffffu, 0u), 0u);
            output[3u] = textureLoad(image, vec2u(0), 1u);
            output[4u] = f32(size.x);
            output[5u] = f32(size.y);
            output[6u] = f32(textureNumLevels(image));
            output[7u] = textureLoad(image, vec2u(1, 0), 0u);
        }
        """), to: .wgsl))
    let source = ShaderInput.wgsl("""
        @group(2) @binding(1) var image: texture_depth_2d;
        @vertex fn vs(@location(0) point: vec2f) -> @builtin(position) vec4f {
            return vec4f(point, 0.25, 1.0);
        }
        @fragment fn writeDepth() -> @location(0) vec4f { return vec4f(0, 1, 0, 1); }
        @fragment fn readDepth() -> @location(0) vec4f {
            return vec4f(textureLoad(image, vec2u(0u), 0u) * 4.0, 0, 0, 1);
        }
        """)
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .wgsl)
    let writer = try compiler.compileRenderStage(source, entryPoint: "writeDepth", to: .wgsl)
    let reader = try compiler.compileRenderStage(source, entryPoint: "readDepth", to: .wgsl)
    let layouts = [VertexBufferLayout(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float32x2)])]
    let render = try await device.createRenderPipeline(
        vertex: vertex, fragment: writer, colorFormat: .rgba8Unorm, vertexBuffers: layouts,
        depthState: .init(compare: .less))
    let preview = try await device.createRenderPipeline(
        vertex: vertex, fragment: reader, colorFormat: .rgba8Unorm, vertexBuffers: layouts)
    let conflicting = try await device.createRenderPipeline(
        vertex: vertex, fragment: reader, colorFormat: .rgba8Unorm, vertexBuffers: layouts,
        depthState: .init(compare: .always, writeEnabled: false))
    let depth = try await device.createTexture(.init(width: 4, height: 2, format: .depth32Float,
        usage: [.renderAttachment, .textureBinding], mipLevelCount: 2))
    let view = try await device.createTextureView(texture: depth, .init(baseMipLevel: 1))
    let color = try await device.createTexture(.init(width: 2, height: 1, format: .rgba8Unorm,
        usage: [.renderAttachment, .textureBinding, .copySource]))
    let vertices = try await device.createBuffer(.init(size: 24, usage: [.vertex, .copyDestination]))
    let output = try await device.createBuffer(.init(size: 32, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 32, usage: [.copyDestination, .mapRead]))
    let clear = RenderClearColor(red: 0, green: 0, blue: 1, alpha: 1)
    let depthSlot = ComputeBindingSlot(group: 2, binding: 1)
    let streams = [VertexBufferBinding(buffer: vertices)]
    let depthAttachment = RenderDepthAttachment(texture: view, loadAction: .clear(1))
    func dispatch(_ texture: ResourceID) -> BufferCommand {
        .dispatchCompute(pipeline: compute, bindings: [.init(binding: 0): output],
                         workgroups: .init(x: 1), textures: [depthSlot: texture])
    }
    do {
        _ = try await device.submitCommands([
            .render(pipeline: conflicting, target: color, clear: clear, vertexCount: 3,
                    vertexBuffers: streams, textures: [depthSlot: view], depth: depthAttachment),
        ])
        throw GPUError.backendFailure("Accepted overlapping sampled and attached depth views")
    } catch GPUError.validation("Render attachment cannot be bound as a sampled texture") {}
    let ticket = try await device.submitCommands([
        .write(buffer: vertices, offset: 0, words: [Float(-1), -1, 3, -1, -1, 3].map(\.bitPattern)),
        .render(pipeline: render, target: color, clear: clear, vertexCount: 3,
                vertexBuffers: streams, depth: depthAttachment),
        .render(pipeline: preview, target: color, clear: clear, vertexCount: 3,
                vertexBuffers: streams, textures: [depthSlot: view]),
        .copyTextureToBuffer(source: color, destination: read, bytesPerRow: 256),
        .read(buffer: read, byteCount: 8),
        dispatch(view),
        .copy(source: output, destination: read, byteCount: 32),
        .read(buffer: read, byteCount: 32),
        dispatch(depth),
        .copy(source: output, destination: read, byteCount: 32),
        .read(buffer: read, byteCount: 32),
    ])
    let viewed: [Float] = [0.25, 0, 0, 0, 2, 1, 1, 0.25]
    let whole: [Float] = [0, 0, 0, 0.25, 4, 2, 2, 0]
    let expected: [[UInt32]] = [[0xff0000ff, 0xff0000ff], viewed.map(\.bitPattern), whole.map(\.bitPattern)]
    let snapshots = try await ticket.wait()
    guard snapshots == expected else { throw GPUError.backendFailure("Unexpected depth texture results: \(snapshots)") }
    return [snapshots[0][0], snapshots[1][0], snapshots[1][3], snapshots[1][4], snapshots[2][3]]
}

private func depthComparisonDiagnostic(_ device: Device) async throws -> [UInt32] {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl("""
        @group(2) @binding(1) var image: texture_depth_2d;
        @group(2) @binding(2) var state: sampler_comparison;
        @group(2) @binding(3) var nearest: sampler;
        @vertex fn vs(@location(0) point: vec2f) -> @builtin(position) vec4f {
            return vec4f(point, point.x * 0.5 + 0.5, 1.0);
        }
        @fragment fn writeDepth() -> @location(0) vec4f { return vec4f(0, 0, 0, 1); }
        @fragment fn fs() -> @location(0) vec4f {
            let uv = vec2f(0.25, 0.5);
            let a = textureSampleCompare(image, state, uv, 0.5);
            let b = textureSampleCompareLevel(image, state, uv, 0.5, vec2i(1, 0));
            let c = textureSample(image, nearest, uv);
            let d = textureSample(image, nearest, uv, vec2i(1, 0));
            return vec4f(a, b, c, d);
        }
        """)
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .wgsl)
    let writer = try compiler.compileRenderStage(source, entryPoint: "writeDepth", to: .wgsl)
    let reader = try compiler.compileRenderStage(source, entryPoint: "fs", to: .wgsl)
    let layouts = [VertexBufferLayout(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float32x2)])]
    let render = try await device.createRenderPipeline(
        vertex: vertex, fragment: writer, colorFormat: .rgba8Unorm, vertexBuffers: layouts,
        depthState: .init(compare: .less))
    let preview = try await device.createRenderPipeline(
        vertex: vertex, fragment: reader, colorFormat: .rgba8Unorm, vertexBuffers: layouts)
    let compute = try await device.createComputePipeline(
        compiler.compileCompute(.wgsl("""
        @group(2) @binding(1) var image: texture_depth_2d;
        @group(2) @binding(2) var state: sampler_comparison;
        @group(2) @binding(3) var nearest: sampler;
        @group(0) @binding(0) var<storage, read_write> out: array<vec4f, 2>;
        @compute @workgroup_size(1) fn run() {
            let a = textureSampleCompareLevel(image, state, vec2f(0.25, 0.5), 0.5);
            let b = textureSampleCompareLevel(image, state, vec2f(0.75, 0.5), 0.5);
            out[0] = vec4f(a, b, textureLoad(image, vec2u(0), 0u), textureLoad(image, vec2u(1, 0), 0u));
            let uv = vec2f(0.25, 0.5);
            let lod = 1i;
            out[1] = vec4f(textureSampleLevel(image, nearest, uv, 0),
                           textureSampleLevel(image, nearest, uv, 0u, vec2i(1, 0)),
                           textureSampleLevel(image, nearest, uv, lod),
                           textureSampleLevel(image, nearest, uv, u32(lod)));
        }
        """), to: .wgsl))
    let depth = try await device.createTexture(.init(width: 4, height: 2, format: .depth32Float,
        usage: [.renderAttachment, .textureBinding], mipLevelCount: 3))
    let view = try await device.createTextureView(texture: depth, .init(baseMipLevel: 1))
    let compare = try await device.createSampler(.init(compare: .less))
    let plain = try await device.createSampler()
    let color = try await device.createTexture(.init(width: 2, height: 1, format: .rgba8Unorm,
        usage: [.renderAttachment, .copySource], mipLevelCount: 2))
    let vertices = try await device.createBuffer(.init(size: 24, usage: [.vertex, .copyDestination]))
    let output = try await device.createBuffer(.init(size: 32, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 32, usage: [.copyDestination, .mapRead]))
    let clear = RenderClearColor(red: 0, green: 0, blue: 0, alpha: 1)
    let textures: [ComputeBindingSlot: ResourceID] = [.init(group: 2, binding: 1): view]
    let samplerSlot = ComputeBindingSlot(group: 2, binding: 2)
    let rawSamplerSlot = ComputeBindingSlot(group: 2, binding: 3)
    let streams = [VertexBufferBinding(buffer: vertices)]
    func visualize(_ sampler: ResourceID) -> BufferCommand {
        .render(pipeline: preview, target: color, clear: clear, vertexCount: 3,
                vertexBuffers: streams, textures: textures, samplers: [samplerSlot: sampler, rawSamplerSlot: plain])
    }
    do {
        _ = try await device.submitCommands([visualize(plain)])
        throw GPUError.backendFailure("Accepted a plain sampler for comparison")
    } catch GPUError.validation("Invalid sampler binding") {}
    let ticket = try await device.submitCommands([
        .write(buffer: vertices, offset: 0, words: [Float(-1), -1, 3, -1, -1, 3].map(\.bitPattern)),
        .render(pipeline: render, target: color, clear: clear, vertexCount: 3, vertexBuffers: streams,
                depth: .init(texture: depth, mipLevel: 1)),
        .render(pipeline: render, target: color, clear: clear, vertexCount: 3, vertexBuffers: streams,
                targetMipLevel: 1, depth: .init(texture: depth, mipLevel: 2)),
        visualize(compare),
        .copyTextureToBuffer(source: color, destination: read, bytesPerRow: 256),
        .read(buffer: read, byteCount: 8),
        .dispatchCompute(pipeline: compute, bindings: [.init(binding: 0): output], workgroups: .init(x: 1),
                         textures: textures, samplers: [samplerSlot: compare, rawSamplerSlot: plain]),
        .copy(source: output, destination: read, byteCount: 32),
        .read(buffer: read, byteCount: 32),
    ])
    let compared: [Float] = [0, 1, 0.25, 0.75, 0.25, 0.75, 0.5, 0.5]
    let expected: [[UInt32]] = [[0xbf40ff00, 0xbf40ff00], compared.map(\.bitPattern)]
    let snapshots = try await ticket.wait()
    guard snapshots == expected else { throw GPUError.backendFailure("Unexpected depth sampling results: \(snapshots)") }
    return snapshots.flatMap { $0 }
}

private func depthGatherDiagnostic(_ device: Device, comparison: Bool) async throws -> [UInt32] {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl("""
        @vertex fn vs(@location(0) point: vec3f) -> @builtin(position) vec4f { return vec4f(point, 1.0); }
        @fragment fn fs() -> @location(0) vec4f { return vec4f(1, 0, 0, 1); }
        """)
    let render = try await device.createRenderPipeline(
        vertex: compiler.compileRenderStage(source, entryPoint: "vs", to: .wgsl),
        fragment: compiler.compileRenderStage(source, entryPoint: "fs", to: .wgsl),
        colorFormat: .rgba8Unorm,
        vertexBuffers: [.init(stride: 12, attributes: [.init(location: 0, offset: 0, format: .float32x3)])],
        depthState: .init())
    let gatherSource = comparison ? """
        @group(0) @binding(0) var<storage, read_write> out: array<vec4f, 4>;
        @group(0) @binding(1) var image: texture_depth_2d;
        @group(0) @binding(2) var state: sampler_comparison;
        fn gather(uv: vec2f) -> vec4f { return textureGatherCompare(image, state, uv, 0.375); }
        @compute @workgroup_size(1) fn run(@builtin(global_invocation_id) id: vec3u) {
            if id.x == 0u {
                let uv = vec2f(0.33333334, 0.5);
                out[0u] = gather(uv);
                out[1u] = textureGatherCompare(image, state, uv, 0.625, vec2i(1, 0));
                out[2u] = textureGatherCompare(image, state, uv, 0.375, vec2i(-1, 0));
                out[3u] = textureGatherCompare(image, state, uv, 0.25);
            }
        }
        """ : """
        @group(0) @binding(0) var<storage, read_write> out: array<vec4f, 4>;
        @group(0) @binding(1) var image: texture_depth_2d;
        @group(0) @binding(2) var state: sampler;
        fn gather(uv: vec2f) -> vec4f { return textureGather(image, state, uv); }
        @compute @workgroup_size(1) fn run() {
            let uv = vec2f(0.33333334, 0.5);
            out[0u] = gather(uv);
            out[1u] = textureGather(image, state, uv, vec2i(1, 0));
            out[2u] = textureGather(image, state, uv, vec2i(-1, 0));
            out[3u] = textureGather(image, state, uv);
        }
        """
    let compute = try await device.createComputePipeline(compiler.compileCompute(.wgsl(gatherSource), to: .wgsl))
    let depth = try await device.createTexture(.init(width: 6, height: 4, format: .depth32Float,
        usage: [.renderAttachment, .textureBinding], mipLevelCount: 2))
    let view = try await device.createTextureView(texture: depth, .init(baseMipLevel: 1))
    let color = try await device.createTexture(.init(width: 3, height: 2, format: .rgba8Unorm, usage: .renderAttachment))
    let sampler = try await device.createSampler(.init(addressModeU: .repeat, compare: comparison ? .less : nil))
    let vertices = try await device.createBuffer(.init(size: 432, usage: [.vertex, .copyDestination]))
    let output = try await device.createBuffer(.init(size: 64, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 64, usage: [.copyDestination, .mapRead]))
    let x: [Float] = [-1, -1 / 3, 1 / 3, 1]
    let y: [Float] = [1, 0, -1]
    let values: [[Float]] = [[0.125, 0.375, 0.625], [0.25, 0.5, 0.875]]
    var grid: [Float] = []
    for row in 0..<2 {
        for column in 0..<3 {
            let (l, r, t, b, z) = (x[column], x[column + 1], y[row], y[row + 1], values[row][column])
            grid += [l, t, z, l, b, z, r, t, z, r, t, z, l, b, z, r, b, z]
        }
    }
    let ticket = try await device.submitCommands([
        .write(buffer: vertices, offset: 0, words: grid.map(\.bitPattern)),
        .render(pipeline: render, target: color, clear: .init(red: 0, green: 0, blue: 0, alpha: 1),
                vertexCount: 36, vertexBuffers: [.init(buffer: vertices)], depth: .init(texture: view)),
        .dispatchCompute(pipeline: compute, bindings: [.init(binding: 0): output], workgroups: .init(x: 1),
                         textures: [.init(binding: 1): view], samplers: [.init(binding: 2): sampler]),
        .copy(source: output, destination: read, byteCount: 64),
        .read(buffer: read, byteCount: 64),
    ])
    let expected: [Float] = comparison
        ? [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0]
        : [0.25, 0.5, 0.375, 0.125, 0.5, 0.875, 0.625, 0.375,
           0.875, 0.25, 0.125, 0.625, 0.25, 0.5, 0.375, 0.125]
    let snapshots = try await ticket.wait()
    guard snapshots == [expected.map(\.bitPattern)] else {
        throw GPUError.backendFailure("Unexpected depth gather readback: \(snapshots)")
    }
    return snapshots[0]
}

private func multisampleDiagnostic(_ device: Device) async throws -> [UInt32] {
    let compiler = IndependentShaderCompiler()
    let stages = ShaderInput.wgsl("""
        @vertex fn vs(@location(0) xy: vec2f) -> @builtin(position) vec4f { return vec4f(xy, 0.25, 1.0); }
        @fragment fn fs() -> @location(0) vec4f { return vec4f(1, 0, 0, 1); }
        """)
    let render = try await device.createRenderPipeline(
        vertex: compiler.compileRenderStage(stages, entryPoint: "vs", to: .wgsl),
        fragment: compiler.compileRenderStage(stages, entryPoint: "fs", to: .wgsl),
        colorFormat: .rgba8Unorm,
        vertexBuffers: [.init(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float32x2)])],
        depthState: .init(compare: .less), sampleCount: 4)
    let compute = try await device.createComputePipeline(compiler.compileCompute(.wgsl("""
        @group(0) @binding(0) var color: texture_multisampled_2d<f32>;
        @group(0) @binding(1) var depth: texture_depth_multisampled_2d;
        @group(0) @binding(2) var<storage, read_write> out: array<u32, 12>;
        @compute @workgroup_size(1) fn run() {
            for (var lane = 0u; lane < 4u; lane++) {
                let r = textureLoad(color, vec2u(0), lane).r;
                let d = textureLoad(depth, vec2i(0), i32(lane));
                out[lane] = select(select(2u, 0u, r == 0.0), 1u, r == 1.0);
                out[4u + lane] = select(select(0u, 4u, d == 1.0), 1u, d == 0.25);
            }
            let dimensions = textureDimensions(color);
            let negative = textureLoad(color, vec2i(0), -1i);
            let outside = textureLoad(depth, vec2u(0xffffffffu), 0u);
            let invalid = textureLoad(depth, vec2u(0), 0xffffffffu);
            out[8] = textureNumSamples(color);
            out[9] = textureNumSamples(depth);
            out[10] = dimensions.x + dimensions.y;
            out[11] = select(1u, 0u, negative.a == 0.0 && outside == 0.0 && invalid == 0.0);
        }
        """), to: .wgsl))
    let color = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm,
        usage: [.renderAttachment, .textureBinding, .copySource], sampleCount: 4))
    let depth = try await device.createTexture(.init(width: 1, height: 1, format: .depth32Float,
        usage: [.renderAttachment, .textureBinding, .copySource], sampleCount: 4))
    let resolveDescriptor = TextureDescriptor(width: 2, height: 2, format: .rgba8Unorm,
        usage: [.renderAttachment, .copySource], mipLevelCount: 2)
    let resolved = try await device.createTexture(resolveDescriptor)
    let resolvedIndexed = try await device.createTexture(resolveDescriptor)
    let resolvedView = try await device.createTextureView(texture: resolvedIndexed, .init(baseMipLevel: 1))
    let colorCopy = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm,
        usage: [.renderAttachment, .textureBinding, .copyDestination], sampleCount: 4))
    let depthCopy = try await device.createTexture(.init(width: 1, height: 1, format: .depth32Float,
        usage: [.renderAttachment, .textureBinding, .copyDestination], sampleCount: 4))
    let colorView = try await device.createTextureView(texture: colorCopy)
    let vertices = try await device.createBuffer(.init(size: 24, usage: [.vertex, .copyDestination]))
    let indices = try await device.createBuffer(.init(size: 12, usage: [.index, .copyDestination]))
    let out = try await device.createBuffer(.init(size: 48, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 264, usage: [.copyDestination, .mapRead]))
    let clear = RenderClearColor(red: 0, green: 0, blue: 0, alpha: 1)
    let ticket = try await device.submitCommands([
        .write(buffer: vertices, offset: 0, words: [Float(-3), -1, 0, -1, 0, 5].map(\.bitPattern)),
        .write(buffer: indices, offset: 0, words: [0, 1, 2]),
        .render(pipeline: render, target: color, clear: clear,
                vertexCount: 3, vertexBuffers: [.init(buffer: vertices)], depth: .init(texture: depth),
                resolve: .init(texture: resolved, mipLevel: 1)),
        .renderIndexed(pipeline: render, target: color, clear: clear,
                       indexBuffer: .init(buffer: indices, format: .uint32), draw: .init(indexCount: 3),
                       vertexBuffers: [.init(buffer: vertices)], depth: .init(texture: depth),
                       resolve: .init(texture: resolvedView)),
        .copyTextureToTexture(source: color, destination: colorCopy, sourceRegion: .init(width: 1, height: 1)),
        .copyTextureToTexture(source: depth, destination: depthCopy, sourceRegion: .init(width: 1, height: 1)),
        .dispatchCompute(pipeline: compute, bindings: [.init(binding: 2): out], workgroups: .init(x: 1),
                         textures: [.init(binding: 0): colorView, .init(binding: 1): depthCopy]),
        .copy(source: out, destination: read, byteCount: 48), .read(buffer: read, byteCount: 48),
        .copyTextureToBuffer(source: resolved, destination: read, bytesPerRow: 256, mipLevel: 1),
        .read(buffer: read, byteCount: 4),
        .copyTextureToBuffer(source: resolvedIndexed, destination: read, bytesPerRow: 256, mipLevel: 1),
        .read(buffer: read, byteCount: 4),
        .copyTextureToBuffer(source: resolvedIndexed, destination: read, bytesPerRow: 256),
        .read(buffer: read, byteCount: 8),
    ])
    let snapshots = try await ticket.wait()
    guard snapshots.count == 4, let words = snapshots.first, words.count == 12,
          words[8..<12].elementsEqual([4, 4, 2, 0]),
          words[0..<4].sorted() == [0, 0, 1, 1],
          snapshots[1] == [0xff000080], snapshots[2] == [0xff000080], snapshots[3] == [0, 0],
          zip(words[0..<4], words[4..<8]).allSatisfy({ color, depth in
              color <= 1 && depth == (color == 1 ? 1 : 4)
          }) else {
        throw GPUError.backendFailure("Unexpected multisampled texture or resolve readback: \(snapshots)")
    }
    return snapshots.flatMap { $0 }
}

private func integerTextureDiagnostic(_ device: Device) async throws -> [UInt32] {
    let compiler = IndependentShaderCompiler()
    let draw = ShaderInput.wgsl("""
        @vertex fn vs(@location(0) xy: vec2f) -> @builtin(position) vec4f { return vec4f(xy, 0, 1); }
        @fragment fn signedColor() -> @location(0) vec4i { return vec4i(-17, 23, -3, 1); }
        @fragment fn unsignedColor() -> @location(0) vec4u { return vec4u(249, 1, 200, 7); }
        """)
    let vertex = try compiler.compileRenderStage(draw, entryPoint: "vs", to: .wgsl)
    let signed = try compiler.compileRenderStage(draw, entryPoint: "signedColor", to: .wgsl)
    let unsigned = try compiler.compileRenderStage(draw, entryPoint: "unsignedColor", to: .wgsl)
    let layouts = [VertexBufferLayout(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float32x2)])]
    let signedPipeline = try await device.createRenderPipeline(
        vertex: vertex, fragment: signed, colorFormat: .rgba8Sint, vertexBuffers: layouts, sampleCount: 4)
    let unsignedPipeline = try await device.createRenderPipeline(
        vertex: vertex, fragment: unsigned, colorFormat: .rgba8Uint, vertexBuffers: layouts, sampleCount: 4)
    let singlePipeline = try await device.createRenderPipeline(
        vertex: vertex, fragment: unsigned, colorFormat: .rgba8Uint, vertexBuffers: layouts)
    let reader = try await device.createComputePipeline(compiler.compileCompute(.wgsl("""
        @group(0) @binding(0) var signedImage: texture_multisampled_2d<i32>;
        @group(0) @binding(1) var unsignedImage: texture_multisampled_2d<u32>;
        @group(0) @binding(2) var single: texture_2d<u32>;
        @group(0) @binding(3) var<storage, read_write> signedOut: array<vec4i, 5>;
        @group(0) @binding(4) var<storage, read_write> unsignedOut: array<vec4u, 7>;
        @compute @workgroup_size(4) fn run(@builtin(local_invocation_index) lane: u32) {
            signedOut[lane] = textureLoad(signedImage, vec2i(0), i32(lane));
            unsignedOut[lane] = textureLoad(unsignedImage, vec2u(0), lane);
            if lane == 0u {
                signedOut[4] = textureLoad(signedImage, vec2i(0), -1i);
                unsignedOut[4] = textureLoad(unsignedImage, vec2u(0), 0xffffffffu);
                unsignedOut[5] = textureLoad(single, vec2u(0), 0u);
                unsignedOut[6] = vec4u(textureNumSamples(signedImage), textureNumSamples(unsignedImage),
                                      textureDimensions(unsignedImage));
            }
        }
        """), to: .wgsl))
    let usage: TextureUsage = [.renderAttachment, .textureBinding]
    let signedTexture = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Sint, usage: usage, sampleCount: 4))
    let unsignedTexture = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Uint, usage: usage, sampleCount: 4))
    let singleTexture = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Uint, usage: usage))
    let vertices = try await device.createBuffer(.init(size: 24, usage: [.vertex, .copyDestination]))
    let signedOut = try await device.createBuffer(.init(size: 80, usage: [.storage, .copySource]))
    let unsignedOut = try await device.createBuffer(.init(size: 112, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 112, usage: [.copyDestination, .mapRead]))
    let clear = RenderClearColor(red: 0, green: 0, blue: 0, alpha: 0)
    func render(_ pipeline: RenderPipeline, to target: ResourceID) -> BufferCommand {
        .render(pipeline: pipeline, target: target, clear: clear, vertexCount: 3, vertexBuffers: [.init(buffer: vertices)])
    }
    let ticket = try await device.submitCommands([
        .write(buffer: vertices, offset: 0, words: [Float(-1), -1, 3, -1, -1, 3].map(\.bitPattern)),
        render(signedPipeline, to: signedTexture), render(unsignedPipeline, to: unsignedTexture),
        render(singlePipeline, to: singleTexture),
        .dispatchCompute(pipeline: reader, bindings: [.init(binding: 3): signedOut, .init(binding: 4): unsignedOut],
                         workgroups: .init(x: 1),
                         textures: [.init(binding: 0): signedTexture, .init(binding: 1): unsignedTexture,
                                    .init(binding: 2): singleTexture]),
        .copy(source: signedOut, destination: read, byteCount: 80), .read(buffer: read, byteCount: 80),
        .copy(source: unsignedOut, destination: read, byteCount: 112), .read(buffer: read, byteCount: 112),
    ])
    let signedWords: [UInt32] = Array(repeating: [Int32(-17), 23, -3, 1].map(UInt32.init(bitPattern:)), count: 4)
        .flatMap { $0 } + [0, 0, 0, 0]
    let unsignedSample: [UInt32] = [249, 1, 200, 7]
    let unsignedWords: [UInt32] = Array(repeating: unsignedSample, count: 4).flatMap { $0 } +
        [0, 0, 0, 0] + unsignedSample + [4, 4, 1, 1]
    let snapshots = try await ticket.wait()
    guard snapshots == [signedWords, unsignedWords] else {
        throw GPUError.backendFailure("Unexpected integer texture readback: \(snapshots)")
    }
    return snapshots.flatMap { $0 }
}

private func integerGatherDiagnostic(_ device: Device) async throws -> [UInt32] {
    let shader = ShaderInput.wgsl("""
        @group(0) @binding(0) var signedImage: texture_2d<i32>;
        @group(0) @binding(1) var unsignedImage: texture_2d<u32>;
        @group(0) @binding(2) var state: sampler;
        @group(0) @binding(3) var<storage, read_write> signedOut: array<vec4i, 2>;
        @group(0) @binding(4) var<storage, read_write> unsignedOut: array<vec4u, 2>;
        const alpha = 1u + 2u;
        fn gatherSigned(uv: vec2f) -> vec4i { return textureGather(alpha, signedImage, state, uv); }
        @compute @workgroup_size(1) fn run() {
            let uv = vec2f(0.33333334, 0.5);
            signedOut[0] = gatherSigned(uv);
            signedOut[1] = textureGather(alpha, signedImage, state, uv, vec2i(1, 0));
            unsignedOut[0] = textureGather(2i, unsignedImage, state, uv, vec2i(-1, 0));
            unsignedOut[1] = textureGather(alpha, unsignedImage, state, uv);
        }
        """)
    let compute = try await device.createComputePipeline(IndependentShaderCompiler().compileCompute(shader, to: .wgsl))
    let signed = try await device.createTexture(.init(width: 6, height: 4, format: .rgba8Sint,
        usage: [.copyDestination, .textureBinding], mipLevelCount: 2))
    let unsigned = try await device.createTexture(.init(width: 6, height: 4, format: .rgba8Uint,
        usage: [.copyDestination, .textureBinding], mipLevelCount: 2))
    let signedView = try await device.createTextureView(texture: signed, .init(baseMipLevel: 1))
    let unsignedView = try await device.createTextureView(texture: unsigned, .init(baseMipLevel: 1))
    let nearest = try await device.createSampler(.init(addressModeU: .repeat))
    let input = try await device.createBuffer(.init(size: 268, usage: [.copyDestination, .copySource]))
    let signedOut = try await device.createBuffer(.init(size: 32, usage: [.storage, .copySource]))
    let unsignedOut = try await device.createBuffer(.init(size: 32, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 32, usage: [.copyDestination, .mapRead]))
    let ticket = try await device.submitCommands([
        .write(buffer: input, offset: 0, words: [0x8c190a14, 0x962d1e28, 0xa041323c]),
        .write(buffer: input, offset: 256, words: [0xaa554650, 0xb4695a64, 0xbe7d6e78]),
        .copyBufferToTexture(source: input, layout: .init(bytesPerRow: 256), destination: signed,
                             region: .init(width: 3, height: 2, mipLevel: 1)),
        .copyBufferToTexture(source: input, layout: .init(bytesPerRow: 256), destination: unsigned,
                             region: .init(width: 3, height: 2, mipLevel: 1)),
        .dispatchCompute(pipeline: compute, bindings: [.init(binding: 3): signedOut, .init(binding: 4): unsignedOut],
                         workgroups: .init(x: 1), textures: [.init(binding: 0): signedView, .init(binding: 1): unsignedView],
                         samplers: [.init(binding: 2): nearest]),
        .copy(source: signedOut, destination: read, byteCount: 32), .read(buffer: read, byteCount: 32),
        .copy(source: unsignedOut, destination: read, byteCount: 32), .read(buffer: read, byteCount: 32),
    ])
    let signedLanes: [Int32] = [-86, -76, -106, -116, -76, -66, -96, -106]
    let snapshots = try await ticket.wait()
    guard snapshots == [signedLanes.map(UInt32.init(bitPattern:)), [125, 85, 25, 65, 170, 180, 150, 140]] else {
        throw GPUError.backendFailure("Unexpected integer texture gather: \(snapshots)")
    }
    return snapshots.flatMap { $0 }
}

private func sampleCoverageDiagnostic(_ device: Device) async throws -> [UInt32] {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl("""
        struct VertexOutput {
            @builtin(position) position: vec4f,
            @location(0) marker: f32,
        }
        struct SampleInput {
            @builtin(sample_mask) coverage: u32,
            @location(0) marker: f32,
            @builtin(sample_index) index: u32,
        }
        struct SampleOutput {
            @builtin(sample_mask) coverage: u32,
            @location(0) color: vec4u,
        }
        @vertex fn vs(@location(0) xy: vec2f) -> VertexOutput {
            return VertexOutput(vec4f(xy, 0, 1), 1.0);
        }
        @fragment fn allSamples(@builtin(sample_index) index: u32) -> @location(0) vec4u {
            return vec4u(index + 1u, 0u, 0u, 1u);
        }
        @fragment fn maskedSamples(input: SampleInput) -> SampleOutput {
            let covered = (input.coverage & (1u << input.index)) != 0u && input.marker == 1.0;
            return SampleOutput(5u, vec4u(input.index + 1u, select(0u, 9u, covered), 0u, 1u));
        }
        """)
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .wgsl)
    let layouts = [VertexBufferLayout(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float32x2)])]
    let unmasked = try await device.createRenderPipeline(
        vertex: vertex, fragment: compiler.compileRenderStage(source, entryPoint: "allSamples", to: .wgsl),
        colorFormat: .rgba8Uint, vertexBuffers: layouts, sampleCount: 4, sampleMask: 0b1010)
    let masked = try await device.createRenderPipeline(
        vertex: vertex, fragment: compiler.compileRenderStage(source, entryPoint: "maskedSamples", to: .wgsl),
        colorFormat: .rgba8Uint, vertexBuffers: layouts, sampleCount: 4, sampleMask: 0b0110)
    let reader = try await device.createComputePipeline(compiler.compileCompute(.wgsl("""
        @group(0) @binding(0) var unmasked: texture_multisampled_2d<u32>;
        @group(0) @binding(1) var masked: texture_multisampled_2d<u32>;
        @group(0) @binding(2) var<storage, read_write> out: array<vec4u, 8>;
        @compute @workgroup_size(4) fn run(@builtin(local_invocation_index) lane: u32) {
            out[lane] = textureLoad(unmasked, vec2u(0), lane);
            out[lane + 4u] = textureLoad(masked, vec2u(0), lane);
        }
        """), to: .wgsl))
    let descriptor = TextureDescriptor(width: 1, height: 1, format: .rgba8Uint,
                                       usage: [.renderAttachment, .textureBinding], sampleCount: 4)
    let allSamples = try await device.createTexture(descriptor)
    let maskedSamples = try await device.createTexture(descriptor)
    let vertices = try await device.createBuffer(.init(size: 24, usage: [.vertex, .copyDestination]))
    let out = try await device.createBuffer(.init(size: 128, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 128, usage: [.copyDestination, .mapRead]))
    let clear = RenderClearColor(red: 0, green: 0, blue: 0, alpha: 0)
    let ticket = try await device.submitCommands([
        .write(buffer: vertices, offset: 0, words: [Float(-1), -1, 3, -1, -1, 3].map(\.bitPattern)),
        .render(pipeline: unmasked, target: allSamples, clear: clear, vertexCount: 3, vertexBuffers: [.init(buffer: vertices)]),
        .render(pipeline: masked, target: maskedSamples, clear: clear, vertexCount: 3, vertexBuffers: [.init(buffer: vertices)]),
        .dispatchCompute(pipeline: reader, bindings: [.init(binding: 2): out], workgroups: .init(x: 1),
                         textures: [.init(binding: 0): allSamples, .init(binding: 1): maskedSamples]),
        .copy(source: out, destination: read, byteCount: 128), .read(buffer: read, byteCount: 128),
    ])
    let expected: [UInt32] = [
        0, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 0, 4, 0, 0, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 3, 9, 0, 1, 0, 0, 0, 0,
    ]
    let snapshots = try await ticket.wait()
    guard snapshots == [expected] else {
        throw GPUError.backendFailure("Unexpected fragment sample coverage readback: \(snapshots)")
    }
    return snapshots[0]
}

private func alphaCoverageDiagnostic(_ device: Device) async throws -> [UInt32] {
    let source = ShaderInput.wgsl("""
        struct FragmentOutput {
            @location(0) color: vec4f,
            @builtin(sample_mask) mask: u32,
        }
        @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
            if i == 0u { return vec4f(-1, -1, 0, 1); }
            if i == 1u { return vec4f(3, -1, 0, 1); }
            return vec4f(-1, 3, 0, 1);
        }
        @fragment fn zero() -> @location(0) vec4f { return vec4f(1, 0, 0, 0); }
        @fragment fn half() -> @location(0) vec4f { return vec4f(1, 0, 0, 0.5); }
        @fragment fn one() -> @location(0) vec4f { return vec4f(1, 0, 0, 1); }
        @fragment fn masked() -> FragmentOutput { return FragmentOutput(vec4f(1), 5u); }
        @fragment fn integer() -> @location(0) vec4u { return vec4u(1); }
        @fragment fn depthOnly() -> @builtin(frag_depth) f32 { return 0.25; }
        """)
    let compiler = IndependentShaderCompiler()
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .wgsl)
    var pipelines: [RenderPipeline] = []
    for (entry, mask) in [("zero", UInt32.max), ("half", UInt32.max), ("one", 0b1010), ("zero", 0b1010)] {
        let fragment = try compiler.compileRenderStage(source, entryPoint: entry, to: .wgsl)
        pipelines.append(try await device.createRenderPipeline(
            vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm,
            sampleCount: 4, sampleMask: mask, alphaToCoverageEnabled: true))
    }
    let reader = try await device.createComputePipeline(compiler.compileCompute(.wgsl("""
        @group(0) @binding(0) var image: texture_multisampled_2d<f32>;
        @group(0) @binding(1) var<storage, read_write> out: array<u32, 4>;
        @compute @workgroup_size(4) fn run(@builtin(local_invocation_index) lane: u32) {
            out[lane] = select(0u, 1u, textureLoad(image, vec2u(0), lane).r == 1.0);
        }
        """), to: .wgsl))
    let image = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm,
        usage: [.renderAttachment, .textureBinding], sampleCount: 4))
    let out = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 16, usage: [.copyDestination, .mapRead]))
    var commands: [BufferCommand] = []
    for pipeline in pipelines {
        commands += [
            .render(pipeline: pipeline, target: image, clear: .transparentBlack, vertexCount: 3),
            .dispatchCompute(pipeline: reader, bindings: [.init(binding: 1): out], workgroups: .init(x: 1),
                             textures: [.init(binding: 0): image]),
            .copy(source: out, destination: read, byteCount: 16), .read(buffer: read, byteCount: 16),
        ]
    }
    let onlyDepth = try compiler.compileRenderStage(source, entryPoint: "depthOnly", to: .wgsl)
    let noSamples = try await device.createRenderPipeline(
        vertex: vertex, fragment: onlyDepth, colorFormat: .rgba8Unorm,
        depthState: .init(compare: .always), sampleMask: 0)
    let singleColor = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm,
        usage: .renderAttachment))
    let singleDepth = try await device.createTexture(.init(width: 1, height: 1, format: .depth32Float,
        usage: [.renderAttachment, .copySource]))
    commands += [
        .render(pipeline: noSamples, target: singleColor, clear: .transparentBlack, vertexCount: 3,
                depth: .init(texture: singleDepth, loadAction: .clear(0.75))),
        .copyTextureToBuffer(source: singleDepth, destination: read, bytesPerRow: 256),
        .read(buffer: read, byteCount: 4),
    ]
    let ticket = try await device.submitCommands(commands)
    let snapshots = try await ticket.wait()
    guard snapshots.count == 5,
          snapshots[0] == [0, 0, 0, 0],
          snapshots[1].contains(0), snapshots[1].contains(1),
          snapshots[2] == [0, 1, 0, 1],
          snapshots[3] == [0, 0, 0, 0],
          snapshots[4] == [Float(0.75).bitPattern] else {
        throw GPUError.backendFailure("Unexpected alpha-to-coverage readback: \(snapshots)")
    }
    return snapshots.flatMap { $0 }
}

private func fragmentDepthDiagnostic(_ device: Device) async throws -> [UInt32] {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl("""
        struct FragmentOut {
            @builtin(frag_depth) depth: f32,
            @builtin(sample_mask) mask: u32,
            @location(0) color: vec4f,
        }
        @vertex fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
            if index == 0u { return vec4f(-1, -1, 0.75, 1); }
            if index == 1u { return vec4f(3, -1, 0.75, 1); }
            return vec4f(-1, 3, 0.75, 1);
        }
        @fragment fn colorAndDepth() -> FragmentOut {
            return FragmentOut(0.25, 1u, vec4f(1, 0, 0, 1));
        }
        @fragment fn depthOnly() -> @builtin(frag_depth) f32 { return 2.0; }
        """)
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .wgsl)
    let combined = try compiler.compileRenderStage(source, entryPoint: "colorAndDepth", to: .wgsl)
    let onlyDepth = try compiler.compileRenderStage(source, entryPoint: "depthOnly", to: .wgsl)
    let writesColor = try await device.createRenderPipeline(
        vertex: vertex, fragment: combined, colorFormat: .rgba8Unorm, depthState: .init(compare: .less))
    let writesDepth = try await device.createRenderPipeline(
        vertex: vertex, fragment: onlyDepth, colorFormat: .rgba8Unorm, depthState: .init(compare: .always))
    let color = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm,
        usage: [.renderAttachment, .copySource]))
    let depth = try await device.createTexture(.init(width: 1, height: 1, format: .depth32Float,
        usage: [.renderAttachment, .copySource]))
    let read = try await device.createBuffer(.init(size: 4, usage: [.copyDestination, .mapRead]))
    let blue = RenderClearColor(red: 0, green: 0, blue: 1, alpha: 1)
    let ticket = try await device.submitCommands([
        .render(pipeline: writesColor, target: color, clear: blue, vertexCount: 3,
                depth: .init(texture: depth, loadAction: .clear(0.5))),
        .copyTextureToBuffer(source: color, destination: read, bytesPerRow: 256), .read(buffer: read, byteCount: 4),
        .copyTextureToBuffer(source: depth, destination: read, bytesPerRow: 256), .read(buffer: read, byteCount: 4),
        .render(pipeline: writesDepth, target: color, clear: blue, vertexCount: 3,
                depth: .init(texture: depth, loadAction: .load)),
        .copyTextureToBuffer(source: color, destination: read, bytesPerRow: 256), .read(buffer: read, byteCount: 4),
        .copyTextureToBuffer(source: depth, destination: read, bytesPerRow: 256), .read(buffer: read, byteCount: 4),
    ])
    let expected: [[UInt32]] = [[0xff0000ff], [Float(0.25).bitPattern], [0xffff0000], [Float(1).bitPattern]]
    let snapshots = try await ticket.wait()
    guard snapshots == expected else {
        throw GPUError.backendFailure("Unexpected fragment depth readback: \(snapshots)")
    }
    return snapshots.flatMap { $0 }
}

private func fragmentRasterDiagnostic(_ device: Device) async throws -> [UInt32] {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl("""
        struct RasterInput {
            @builtin(position) pixel: vec4f,
        }
        @vertex fn vs(@location(0) clip: vec2f) -> @builtin(position) vec4f {
            return vec4f(clip, 0.5, 2.0);
        }
        @fragment fn fs(input: RasterInput, @builtin(front_facing) front: bool) -> @location(0) vec4u {
            let p = input.pixel;
            let x = select(0u, 1u, p.x == 0.5) + select(0u, 2u, p.x == 1.5);
            let y = select(0u, 1u, p.y == 0.5) + select(0u, 2u, p.y == 1.5);
            let zw = select(0u, 37u, p.z == 0.25) + select(0u, 64u, p.w == 0.5);
            return vec4u(x, y, zw, select(0u, 1u, front));
        }
        """)
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .wgsl)
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: .wgsl)
    let layouts = [VertexBufferLayout(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float32x2)])]
    let defaults = try await device.createRenderPipeline(
        vertex: vertex, fragment: fragment, colorFormat: .rgba8Uint, vertexBuffers: layouts)
    let clockwise = try await device.createRenderPipeline(
        vertex: vertex, fragment: fragment, colorFormat: .rgba8Uint, vertexBuffers: layouts,
        primitive: .init(frontFace: .clockwise, cullMode: .back))
    let cullFront = try await device.createRenderPipeline(
        vertex: vertex, fragment: fragment, colorFormat: .rgba8Uint, vertexBuffers: layouts,
        primitive: .init(cullMode: .front))
    let texture = try await device.createTexture(.init(width: 2, height: 2, format: .rgba8Uint,
        usage: [.renderAttachment, .copySource]))
    let vertices = try await device.createBuffer(.init(size: 48, usage: [.vertex, .copyDestination]))
    let read = try await device.createBuffer(.init(size: 264, usage: [.copyDestination, .mapRead]))
    let clear = RenderClearColor(red: 0, green: 0, blue: 0, alpha: 0)
    let triangles: [Float] = [-2, -2, 6, -2, -2, 6, -2, -2, -2, 6, 6, -2]
    var commands: [BufferCommand] = [.write(buffer: vertices, offset: 0, words: triangles.map(\.bitPattern))]
    for (pipeline, firstVertex) in [(defaults, 0), (defaults, 3), (clockwise, 0), (clockwise, 3), (cullFront, 0)] {
        commands += [
            .render(pipeline: pipeline, target: texture, clear: clear, vertexCount: 3,
                    vertexBuffers: [.init(buffer: vertices)], firstVertex: UInt32(firstVertex)),
            .copyTextureToBuffer(source: texture, destination: read, bytesPerRow: 256),
            .read(buffer: read, byteCount: 264),
        ]
    }
    let ticket = try await device.submitCommands(commands)
    let snapshots = try await ticket.wait()
    guard snapshots.count == 5, snapshots.allSatisfy({ $0.count == 66 }) else {
        throw GPUError.backendFailure("Unexpected fragment raster readback size")
    }
    let pixels = snapshots.map { [$0[0], $0[1], $0[64], $0[65]] }
    let front: [UInt32] = [0x01650101, 0x01650102, 0x01650201, 0x01650202]
    let back = front.map { $0 & 0x00ff_ffff }
    let culled: [UInt32] = [0, 0, 0, 0]
    guard pixels == [front, back, culled, front, culled] else {
        throw GPUError.backendFailure("Unexpected fragment raster inputs: \(pixels)")
    }
    return pixels.flatMap { $0 }
}

private func interpolationDiagnostic(_ device: Device) async throws -> UInt32 {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl("""
        struct VertexOutput {
            @builtin(position) position: vec4f,
            @location(0) @interpolate(linear) affine: f32,
            @location(1) smooth: f32,
            @location(2) @interpolate(flat) signed: vec2i,
            @location(3) @interpolate(flat, either) unsigned: u32,
        }
        fn vertex(position: vec4f, value: f32, tag: vec2i, code: u32) -> VertexOutput {
            return VertexOutput(position, value, value, tag, code);
        }
        @vertex fn vs(@builtin(vertex_index) index: u32) -> VertexOutput {
            if index == 0u { return vertex(vec4f(-1, 1, 0.25, 1), 0.0, vec2i(-17, 5), 249u); }
            if index == 1u { return vertex(vec4f(6, 2, 0.5, 2), 1.0, vec2i(1, 7), 11u); }
            return vertex(vec4f(-4, -12, 1, 4), 0.0, vec2i(3, 9), 200u);
        }
        @fragment fn fs(
            @location(0) @interpolate(linear) affine: f32,
            @location(1) smooth: f32,
            @location(2) @interpolate(flat, first) signed: vec2i,
            @location(3) @interpolate(flat, either) unsigned: u32,
        ) -> @location(0) vec4u {
            return vec4u(select(0u, 21u, affine > 0.24 && affine < 0.26),
                         select(0u, 42u, smooth > 0.18 && smooth < 0.19),
                         select(0u, 63u, signed.x == -17 && signed.y == 5),
                         select(0u, 84u, unsigned == 249u || unsigned == 200u));
        }
        """)
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .wgsl)
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: .wgsl)
    let pipeline = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Uint)
    let target = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Uint,
        usage: [.renderAttachment, .copySource]))
    let read = try await device.createBuffer(.init(size: 4, usage: [.copyDestination, .mapRead]))
    let ticket = try await device.submitCommands([
        .render(pipeline: pipeline, target: target, clear: .init(red: 0, green: 0, blue: 0, alpha: 0), vertexCount: 3),
        .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256), .read(buffer: read, byteCount: 4),
    ])
    let snapshots = try await ticket.wait()
    guard snapshots == [[0x543f2a15]] else {
        throw GPUError.backendFailure("Unexpected interstage interpolation readback: \(snapshots)")
    }
    return snapshots[0][0]
}

private func topologyDiagnostic(_ device: Device) async throws -> [UInt32] {
    let source = ShaderInput.wgsl("""
        @vertex fn vs(@location(0) p: vec2f) -> @builtin(position) vec4f { return vec4f(p, 0, 1); }
        @fragment fn fs() -> @location(0) vec4f { return vec4f(0, 1, 0, 1); }
        """)
    let compiler = IndependentShaderCompiler()
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .wgsl)
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: .wgsl)
    let layouts = [VertexBufferLayout(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float32x2)])]
    let states: [RenderPrimitiveState] = [
        .init(topology: .pointList), .init(topology: .lineList), .init(topology: .lineStrip),
        .init(topology: .triangleStrip), .init(topology: .triangleStrip, stripIndexFormat: .uint16),
        .init(topology: .lineStrip, stripIndexFormat: .uint32),
    ]
    var pipelines: [RenderPipeline] = []
    for primitive in states {
        pipelines.append(try await device.createRenderPipeline(
            vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm, vertexBuffers: layouts, primitive: primitive))
    }
    let target = try await device.createTexture(.init(width: 8, height: 4, format: .rgba8Unorm,
        usage: [.renderAttachment, .copySource]))
    let vertices = try await device.createBuffer(.init(size: 176, usage: [.vertex, .copyDestination]))
    let short = try await device.createBuffer(.init(size: 20, usage: [.index, .copyDestination]))
    let long = try await device.createBuffer(.init(size: 20, usage: [.index, .copyDestination]))
    let read = try await device.createBuffer(.init(size: 800, usage: [.copyDestination, .mapRead]))
    let positions: [(Float, Float)] = [
        (0.5, 0.5), (7.5, 3.5),
        (1.5, 0.5), (6.5, 0.5), (6.5, 3.5), (1.5, 3.5),
        (1, 0), (1, 4), (7, 0), (7, 4),
        (0, 0), (0, 4), (2, 0), (2, 4), (6, 0), (6, 4), (8, 0), (8, 4),
        (0.5, 1.5), (2.5, 1.5), (5.5, 1.5), (7.5, 1.5),
    ]
    let clear = RenderClearColor(red: 0, green: 0, blue: 0, alpha: 1)
    var commands: [BufferCommand] = [
        .write(buffer: vertices, offset: 0, words: positions.flatMap { x, y in
            [(x / 4 - 1).bitPattern, (1 - y / 2).bitPattern]
        }),
        .write(buffer: short, offset: 0, words: [0x00010009, 0x00030002, 0xffff0004, 0x00060005, 0x00080007]),
        .write(buffer: long, offset: 0, words: [0, 1, .max, 2, 3]),
    ]
    let draws: [BufferCommand] = [
        .render(pipeline: pipelines[0], target: target, clear: clear, vertexCount: 2, vertexBuffers: [.init(buffer: vertices)]),
        .render(pipeline: pipelines[1], target: target, clear: clear, vertexCount: 4, vertexBuffers: [.init(buffer: vertices)], firstVertex: 2),
        .render(pipeline: pipelines[2], target: target, clear: clear, vertexCount: 4, vertexBuffers: [.init(buffer: vertices)], firstVertex: 2),
        .render(pipeline: pipelines[3], target: target, clear: clear, vertexCount: 4, vertexBuffers: [.init(buffer: vertices)], firstVertex: 6),
        .renderIndexed(pipeline: pipelines[4], target: target, clear: clear, indexBuffer: .init(buffer: short, format: .uint16),
                       draw: .init(indexCount: 9, firstIndex: 1, baseVertex: -1), vertexBuffers: [.init(buffer: vertices, offset: 80)]),
        .renderIndexed(pipeline: pipelines[5], target: target, clear: clear, indexBuffer: .init(buffer: long, format: .uint32),
                       draw: .init(indexCount: 5), vertexBuffers: [.init(buffer: vertices, offset: 144)]),
    ]
    for draw in draws {
        commands += [draw, .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256),
                     .read(buffer: read, byteCount: 800)]
    }
    let ticket = try await device.submitCommands(commands)
    let snapshots = try await ticket.wait()
    guard snapshots.count == 6, snapshots.allSatisfy({ $0.count == 200 }) else {
        throw GPUError.backendFailure("Unexpected primitive topology readback size")
    }
    let pixels = [
        snapshots[0][0], snapshots[0][199], snapshots[0][68],
        snapshots[1][3], snapshots[1][70], snapshots[1][195],
        snapshots[2][3], snapshots[2][70], snapshots[2][195],
        snapshots[3][3], snapshots[3][195], snapshots[3][64],
        snapshots[4][64], snapshots[4][68], snapshots[4][71],
        snapshots[5][65], snapshots[5][68], snapshots[5][70],
    ]
    let green: UInt32 = 0xff00ff00
    let black: UInt32 = 0xff000000
    let pointPixels = (0..<4).flatMap { row in snapshots[0][(row * 64)..<(row * 64 + 8)] }
    guard pointPixels.filter({ $0 == green }).count == 2,
          pixels == [green, green, black, green, black, green, green, green, green,
                     green, green, black, green, black, green, green, black, green] else {
        throw GPUError.backendFailure("Unexpected primitive topology readback: \(pixels)")
    }
    return pixels
}

private func blendDiagnostic(_ device: Device) async throws -> [UInt32] {
    let source = ShaderInput.wgsl("""
        @vertex fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
            if index == 0u { return vec4f(-1, -1, 0, 1); }
            if index == 1u { return vec4f(3, -1, 0, 1); }
            return vec4f(-1, 3, 0, 1);
        }
        @fragment fn base() -> @location(0) vec4f { return vec4f(0, 0.5, 1, 1); }
        @fragment fn overlay() -> @location(0) vec4f { return vec4f(1, 0, 0, 0.5); }
        @fragment fn integerColor() -> @location(0) vec4u { return vec4u(1); }
        """)
    let compiler = IndependentShaderCompiler()
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .wgsl)
    let base = try compiler.compileRenderStage(source, entryPoint: "base", to: .wgsl)
    let overlay = try compiler.compileRenderStage(source, entryPoint: "overlay", to: .wgsl)
    let paint = try await device.createRenderPipeline(vertex: vertex, fragment: base, colorFormat: .rgba8Unorm)
    let blend = try await device.createRenderPipeline(
        vertex: vertex, fragment: overlay, colorFormat: .rgba8Unorm,
        colorState: .init(blend: .init(color: .init(sourceFactor: .sourceAlpha, destinationFactor: .oneMinusSourceAlpha)),
                          writeMask: [.red, .green, .alpha]))
    let constantBlend = try await device.createRenderPipeline(
        vertex: vertex, fragment: overlay, colorFormat: .rgba8Unorm,
        colorState: .init(blend: .init(color: .init(operation: .reverseSubtract, sourceFactor: .constant, destinationFactor: .one),
                                      alpha: .init(operation: .min, sourceFactor: .one, destinationFactor: .one))))
    let texture = try await device.createTexture(.init(width: 2, height: 2, format: .rgba8Unorm,
        usage: [.renderAttachment, .copySource], mipLevelCount: 2))
    let view = try await device.createTextureView(texture: texture, .init(baseMipLevel: 1))
    let indices = try await device.createBuffer(.init(size: 12, usage: [.index, .copyDestination]))
    let read = try await device.createBuffer(.init(size: 4, usage: [.copyDestination, .mapRead]))
    let ticket = try await device.submitCommands([
        .render(pipeline: paint, target: view, colorLoad: .load, vertexCount: 0),
        .copyTextureToBuffer(source: texture, destination: read, bytesPerRow: 256, mipLevel: 1),
        .read(buffer: read, byteCount: 4),
        .render(pipeline: paint, target: view, clear: .init(red: 0, green: 0, blue: 0, alpha: 0), vertexCount: 3),
        .write(buffer: indices, offset: 0, words: [0, 1, 2]),
        .renderIndexed(pipeline: blend, target: view, colorLoad: .load,
                       indexBuffer: .init(buffer: indices, format: .uint32), draw: .init(indexCount: 3)),
        .copyTextureToBuffer(source: texture, destination: read, bytesPerRow: 256, mipLevel: 1),
        .read(buffer: read, byteCount: 4),
        .render(pipeline: constantBlend, target: view, clear: .init(red: 1, green: 0.5, blue: 0.25, alpha: 1),
                vertexCount: 3, blendConstant: .init(red: 0.25, green: 0, blue: 0, alpha: 0)),
        .copyTextureToBuffer(source: texture, destination: read, bytesPerRow: 256, mipLevel: 1),
        .read(buffer: read, byteCount: 4),
    ])
    let snapshots = try await ticket.wait()
    guard snapshots == [[0], [0x80ff4080], [0x804080bf]] else {
        throw GPUError.backendFailure("Unexpected color blend readback: \(snapshots)")
    }
    return snapshots.flatMap { $0 }
}

private func multipleColorDiagnostic(_ device: Device) async throws -> [UInt32] {
    let source = ShaderInput.wgsl("""
        struct FragmentOutput {
            @location(2) tag: vec4u,
            @location(0) color: vec4f,
        }
        struct FloatOutput {
            @location(0) first: vec4f,
            @location(2) second: vec4f,
        }
        @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
            if i == 0u { return vec4f(-1, -1, 0, 1); }
            if i == 1u { return vec4f(3, -1, 0, 1); }
            return vec4f(-1, 3, 0, 1);
        }
        @fragment fn fs() -> FragmentOutput { return FragmentOutput(vec4u(17, 29, 33, 49), vec4f(1, 0, 0, 1)); }
        @fragment fn bothFloat() -> FloatOutput { return FloatOutput(vec4f(0), vec4f(1)); }
        """)
    let compiler = IndependentShaderCompiler()
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .wgsl)
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: .wgsl)
    let pipeline = try await device.createRenderPipeline(
        vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm,
        additionalColorTargets: [2: .init(format: .rgba8Uint, colorState: .init(writeMask: [.red, .blue]))])
    let primary = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm,
        usage: [.renderAttachment, .copySource]))
    let tag = try await device.createTexture(.init(width: 2, height: 2, format: .rgba8Uint,
        usage: [.renderAttachment, .copySource], mipLevelCount: 2))
    let view = try await device.createTextureView(texture: tag, .init(baseMipLevel: 1))
    let indices = try await device.createBuffer(.init(size: 12, usage: [.index, .copyDestination]))
    let read = try await device.createBuffer(.init(size: 4, usage: [.copyDestination, .mapRead]))
    let colors: [UInt32: RenderColorAttachment] = [
        2: .init(texture: view, loadAction: .clear(.init(red: 0, green: 5, blue: 0, alpha: 7))),
    ]
    let loaded: [UInt32: RenderColorAttachment] = [2: .init(texture: view)]
    let draw = BufferCommand.renderIndexed(
        pipeline: pipeline, target: primary, clear: .transparentBlack,
        indexBuffer: .init(buffer: indices, format: .uint32), draw: .init(indexCount: 3), additionalColors: loaded)
    let floatFragment = try compiler.compileRenderStage(source, entryPoint: "bothFloat", to: .wgsl)
    let constant = RenderBlendComponent(sourceFactor: .constant)
    let multisampled = try await device.createRenderPipeline(
        vertex: vertex, fragment: floatFragment, colorFormat: .rgba8Unorm, sampleCount: 4,
        additionalColorTargets: [2: .init(format: .rgba8Unorm,
            colorState: .init(blend: .init(color: constant, alpha: constant)))])
    let multisample = TextureDescriptor(width: 1, height: 1, format: .rgba8Unorm, usage: .renderAttachment, sampleCount: 4)
    let msPrimary = try await device.createTexture(multisample)
    let msSecondary = try await device.createTexture(multisample)
    let ticket = try await device.submitCommands([
        .write(buffer: indices, offset: 0, words: [0, 1, 2]),
        .render(pipeline: pipeline, target: primary, clear: .transparentBlack,
                vertexCount: 0, additionalColors: colors),
        draw,
        .copyTextureToBuffer(source: primary, destination: read, bytesPerRow: 256), .read(buffer: read, byteCount: 4),
        .copyTextureToBuffer(source: tag, destination: read, bytesPerRow: 256, mipLevel: 1),
        .read(buffer: read, byteCount: 4),
        .render(pipeline: multisampled, target: msPrimary, clear: .transparentBlack, vertexCount: 3,
                blendConstant: .init(red: 0.25, green: 0.5, blue: 1, alpha: 1),
                additionalColors: [2: .init(texture: msSecondary, resolve: .init(texture: primary))]),
        .copyTextureToBuffer(source: primary, destination: read, bytesPerRow: 256), .read(buffer: read, byteCount: 4),
    ])
    let snapshots = try await ticket.wait()
    guard snapshots == [[0xff0000ff], [0x07210511], [0xffff8040]] else {
        throw GPUError.backendFailure("Unexpected multiple color attachment readback: \(snapshots)")
    }
    return snapshots.flatMap { $0 }
}

private func viewportDiagnostic(_ device: Device) async throws -> [UInt32] {
    let source = ShaderInput.wgsl("""
        @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
            if i == 0u { return vec4f(-1, -1, 0.25, 1); }
            if i == 1u { return vec4f(3, -1, 0.25, 1); }
            return vec4f(-1, 3, 0.25, 1);
        }
        @fragment fn color(@builtin(position) pixel: vec4f) -> @location(0) vec4f {
            return select(vec4f(1, 0, 0, 1), vec4f(0, 1, 0, 1), pixel.z == 0.625);
        }
        @fragment fn depth() -> @builtin(frag_depth) f32 { return 0.1; }
        """)
    let compiler = IndependentShaderCompiler()
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .wgsl)
    let color = try await device.createRenderPipeline(vertex: vertex,
        fragment: compiler.compileRenderStage(source, entryPoint: "color", to: .wgsl), colorFormat: .rgba8Unorm)
    let depth = try await device.createRenderPipeline(vertex: vertex,
        fragment: compiler.compileRenderStage(source, entryPoint: "depth", to: .wgsl), colorFormat: .rgba8Unorm,
        depthState: .init(compare: .less))
    let target = try await device.createTexture(.init(width: 8, height: 8, format: .rgba8Unorm,
        usage: [.renderAttachment, .copySource], mipLevelCount: 2))
    let view = try await device.createTextureView(texture: target, .init(baseMipLevel: 1))
    let depthTarget = try await device.createTexture(.init(width: 4, height: 4, format: .depth32Float,
        usage: [.renderAttachment, .copySource]))
    let indices = try await device.createBuffer(.init(size: 12, usage: [.index, .copyDestination]))
    let read = try await device.createBuffer(.init(size: 784, usage: [.copyDestination, .mapRead]))
    let raster = RenderRasterState(viewport: .init(x: 1, y: 1, width: 3, height: 2, minDepth: 0.5),
                                  scissor: .init(x: 0, y: 2, width: 3, height: 2))
    let ticket = try await device.submitCommands([
        .write(buffer: indices, offset: 0, words: [0, 1, 2]),
        .render(pipeline: color, target: view, clear: .transparentBlack, vertexCount: 3, raster: raster),
        .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256, mipLevel: 1),
        .read(buffer: read, byteCount: 784),
        .renderIndexed(pipeline: depth, target: view, colorLoad: .load, indexBuffer: .init(buffer: indices, format: .uint32),
                       draw: .init(indexCount: 3), depth: .init(texture: depthTarget, loadAction: .clear(0.75)), raster: raster),
        .copyTextureToBuffer(source: depthTarget, destination: read, bytesPerRow: 256), .read(buffer: read, byteCount: 784),
        .render(pipeline: color, target: view, clear: .init(red: 0, green: 0, blue: 1, alpha: 1), vertexCount: 3,
                raster: .init(scissor: .init(x: 4, y: 4, width: 0, height: 0))),
        .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256, mipLevel: 1),
        .read(buffer: read, byteCount: 784),
    ])
    let snapshots = try await ticket.wait()
    guard snapshots.count == 3, snapshots.allSatisfy({ $0.count == 196 }) else {
        throw GPUError.backendFailure("Unexpected viewport readback size")
    }
    var words: [UInt32] = []
    for y in 0..<4 {
        for x in 0..<4 {
            let inside = y == 2 && (x == 1 || x == 2)
            let color = snapshots[0][y * 64 + x]
            let depth = snapshots[1][y * 64 + x]
            guard color == (inside ? 0xff00ff00 : 0), depth == (inside ? Float(0.5) : 0.75).bitPattern else {
                throw GPUError.backendFailure("Unexpected viewport sample at \(x), \(y): \(color), \(depth)")
            }
            guard snapshots[2][y * 64 + x] == 0xffff0000 else {
                throw GPUError.backendFailure("Empty scissor must preserve the pass clear")
            }
            words += [color, depth]
        }
    }
    return words
}

private func depthBiasDiagnostic(_ device: Device) async throws -> [UInt32] {
    let source = ShaderInput.wgsl("""
        @vertex fn flatDepth(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
            if i == 0u { return vec4f(-1, -1, 0.5, 1); }
            if i == 1u { return vec4f(3, -1, 0.5, 1); }
            return vec4f(-1, 3, 0.5, 1);
        }
        @vertex fn slope(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
            if i == 0u { return vec4f(-1, -1, 0.25, 1); }
            if i == 1u { return vec4f(3, -1, 0.75, 1); }
            return vec4f(-1, 3, 0.25, 1);
        }
        @fragment fn fs() -> @location(0) vec4f { return vec4f(1, 0, 0, 1); }
        """)
    let compiler = IndependentShaderCompiler()
    let flat = try compiler.compileRenderStage(source, entryPoint: "flatDepth", to: .wgsl)
    let slope = try compiler.compileRenderStage(source, entryPoint: "slope", to: .wgsl)
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: .wgsl)
    let color = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm,
        usage: [.renderAttachment, .copySource]))
    let depth = try await device.createTexture(.init(width: 1, height: 1, format: .depth32Float,
        usage: [.renderAttachment, .copySource]))
    let read = try await device.createBuffer(.init(size: 4, usage: [.copyDestination, .mapRead]))
    let cases: [(RenderStageArtifact, RenderDepthBiasState, Double)] = [
        (flat, .init(), 0.53125),
        (flat, .init(constant: 1 << 20), 0.53125),
        (flat, .init(constant: 1 << 20), 1),
        (flat, .init(constant: -(1 << 20)), 1),
        (slope, .init(slopeScale: 1, clamp: 0.03125), 1),
        (slope, .init(slopeScale: -1, clamp: -0.03125), 1),
    ]
    var commands: [BufferCommand] = []
    for (vertex, bias, clear) in cases {
        let pipeline = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm,
                                                             depthState: .init(bias: bias))
        commands += [
            .render(pipeline: pipeline, target: color, clear: .transparentBlack, vertexCount: 3,
                    depth: .init(texture: depth, loadAction: .clear(clear))),
            .copyTextureToBuffer(source: color, destination: read, bytesPerRow: 256), .read(buffer: read, byteCount: 4),
            .copyTextureToBuffer(source: depth, destination: read, bytesPerRow: 256), .read(buffer: read, byteCount: 4),
        ]
    }
    let ticket = try await device.submitCommands(commands)
    let red: UInt32 = 0xff0000ff
    let expected: [[UInt32]] = [
        [red], [Float(0.5).bitPattern],
        [0], [Float(0.53125).bitPattern],
        [red], [Float(0.5625).bitPattern],
        [red], [Float(0.4375).bitPattern],
        [red], [Float(0.40625).bitPattern],
        [red], [Float(0.34375).bitPattern],
    ]
    let snapshots = try await ticket.wait()
    guard snapshots == expected else {
        throw GPUError.backendFailure("Unexpected depth bias readback: \(snapshots)")
    }
    return snapshots.flatMap { $0 }
}

private func stencilDiagnostic(_ device: Device, sampleCount: UInt32) async throws -> [UInt32] {
    let source = ShaderInput.wgsl("""
        @vertex fn vs(@location(0) p: vec2f) -> @builtin(position) vec4f { return vec4f(p, 0.5, 1); }
        @fragment fn fs() -> @location(0) vec4f { return vec4f(0, 1, 0, 1); }
        """)
    let compiler = IndependentShaderCompiler()
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .wgsl)
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: .wgsl)
    let layouts = [VertexBufferLayout(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float32x2)])]
    func pipeline(_ compare: CompareFunction, _ stencil: RenderStencilState) async throws -> RenderPipeline {
        try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm,
            vertexBuffers: layouts, depthState: .init(format: .depth24PlusStencil8, compare: compare,
                writeEnabled: false, stencil: stencil), sampleCount: sampleCount)
    }
    let writer = try await pipeline(.less, .init(front: .init(pass: .replace), back: .init(pass: .invert), writeMask: 0x0f))
    let equal = RenderStencilFaceState(compare: .equal)
    let reader = try await pipeline(.always, .init(front: equal, back: equal, readMask: 0x0f, writeMask: 0))
    let failure = try await pipeline(.never, .init(
        front: .init(compare: .equal, fail: .incrementWrap, depthFail: .decrementWrap), readMask: 0x0f, writeMask: 0x0f))
    let color = try await device.createTexture(.init(width: 2, height: 1, format: .rgba8Unorm,
        usage: [.renderAttachment, .copySource]))
    let target: ResourceID
    let resolve: RenderResolveAttachment?
    if sampleCount > 1 {
        target = try await device.createTexture(.init(width: 2, height: 1, format: .rgba8Unorm,
            usage: .renderAttachment, sampleCount: sampleCount))
        resolve = .init(texture: color)
    } else {
        target = color
        resolve = nil
    }
    let baseMip: UInt32 = sampleCount == 1 ? 1 : 0
    let depth = try await device.createTexture(.init(width: 2 << baseMip, height: 1 << baseMip, format: .depth24PlusStencil8,
        usage: .renderAttachment, mipLevelCount: baseMip + 1, sampleCount: sampleCount))
    let view = try await device.createTextureView(texture: depth, .init(baseMipLevel: baseMip))
    let vertices = try await device.createBuffer(.init(size: 48, usage: [.vertex, .copyDestination]))
    let read = try await device.createBuffer(.init(size: 8, usage: [.copyDestination, .mapRead]))
    func draw(_ pipeline: RenderPipeline, reference: UInt32, first: UInt32 = 0,
              stencil: RenderStencilAttachmentState.LoadAction = .load,
              depthLoad: RenderDepthAttachment.LoadAction = .load, x: UInt32? = nil) -> BufferCommand {
        .render(pipeline: pipeline, target: target, clear: .transparentBlack, vertexCount: 3,
                vertexBuffers: [.init(buffer: vertices)], firstVertex: first,
                depth: .init(texture: view, loadAction: depthLoad,
                             stencil: .init(loadAction: stencil, reference: reference)), resolve: resolve,
                raster: .init(scissor: x.map { .init(x: $0, y: 0, width: 1, height: 1) }))
    }
    let copy = BufferCommand.copyTextureToBuffer(source: color, destination: read, bytesPerRow: 256)
    let snapshot = BufferCommand.read(buffer: read, byteCount: 8)
    let positions: [Float] = [-1, -1, 3, -1, -1, 3, -1, -1, -1, 3, 3, -1]
    let ticket = try await device.submitCommands([
        .write(buffer: vertices, offset: 0, words: positions.map(\.bitPattern)),
        draw(reader, reference: 0, stencil: .readOnly, depthLoad: .clear(0.75)), copy, snapshot,
        draw(writer, reference: 5, stencil: .clear(0x1a0), depthLoad: .readOnly, x: 0),
        draw(writer, reference: 0, first: 3, depthLoad: .readOnly, x: 1),
        draw(reader, reference: 5, stencil: .readOnly), copy, snapshot,
        draw(reader, reference: 15, stencil: .readOnly), copy, snapshot,
        draw(failure, reference: 5),
        draw(reader, reference: 4, stencil: .readOnly), copy, snapshot,
        draw(reader, reference: 0, stencil: .readOnly), copy, snapshot,
    ])
    let green: UInt32 = 0xff00ff00
    let snapshots = try await ticket.wait()
    guard snapshots == [[green, green], [green, 0], [0, green], [green, 0], [0, green]] else {
        throw GPUError.backendFailure("Unexpected stencil readback: \(snapshots)")
    }
    return snapshots.flatMap { $0 }
}

private func readOnlyDepthDiagnostic(_ device: Device) async throws -> [UInt32] {
    let source = ShaderInput.wgsl("""
        @group(0) @binding(0) var image: texture_depth_2d;
        fn position(i: u32, z: f32) -> vec4f {
            if i == 0u { return vec4f(-1, -1, z, 1); }
            if i == 1u { return vec4f(3, -1, z, 1); }
            return vec4f(-1, 3, z, 1);
        }
        @vertex fn seed(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f { return position(i, 0.25); }
        @vertex fn near(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f { return position(i, 0.125); }
        @vertex fn far(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f { return position(i, 0.5); }
        @fragment fn clearColor() -> @location(0) vec4f { return vec4f(0); }
        @fragment fn sampleDepth() -> @location(0) vec4f {
            return select(vec4f(1, 0, 0, 1), vec4f(0, 1, 0, 1), textureLoad(image, vec2u(0), 0u) == 0.25);
        }
        """)
    let compiler = IndependentShaderCompiler()
    let writer = try await device.createRenderPipeline(
        vertex: compiler.compileRenderStage(source, entryPoint: "seed", to: .wgsl),
        fragment: compiler.compileRenderStage(source, entryPoint: "clearColor", to: .wgsl),
        colorFormat: .rgba8Unorm, depthState: .init())
    let reader = try compiler.compileRenderStage(source, entryPoint: "sampleDepth", to: .wgsl)
    let near = try await device.createRenderPipeline(
        vertex: compiler.compileRenderStage(source, entryPoint: "near", to: .wgsl), fragment: reader,
        colorFormat: .rgba8Unorm, depthState: .init(writeEnabled: false))
    let far = try await device.createRenderPipeline(
        vertex: compiler.compileRenderStage(source, entryPoint: "far", to: .wgsl), fragment: reader,
        colorFormat: .rgba8Unorm, depthState: .init(writeEnabled: false))
    let color = try await device.createTexture(.init(width: 2, height: 1, format: .rgba8Unorm,
        usage: [.renderAttachment, .copySource]))
    let depth = try await device.createTexture(.init(width: 4, height: 2, format: .depth32Float,
        usage: [.renderAttachment, .textureBinding, .copySource], mipLevelCount: 2))
    let attachment = try await device.createTextureView(texture: depth, .init(baseMipLevel: 1))
    let sampled = try await device.createTextureView(texture: depth, .init(baseMipLevel: 1))
    let read = try await device.createBuffer(.init(size: 8, usage: [.copyDestination, .mapRead]))
    func draw(_ pipeline: RenderPipeline, load: RenderDepthAttachment.LoadAction) -> BufferCommand {
        .render(pipeline: pipeline, target: color, clear: .transparentBlack, vertexCount: 3,
                textures: [.init(binding: 0): sampled], depth: .init(texture: attachment, loadAction: load))
    }
    let ticket = try await device.submitCommands([
        .render(pipeline: writer, target: color, clear: .transparentBlack, vertexCount: 3,
                depth: .init(texture: attachment)),
        draw(near, load: .readOnly),
        .copyTextureToBuffer(source: color, destination: read, bytesPerRow: 256), .read(buffer: read, byteCount: 8),
        draw(far, load: .readOnly),
        .copyTextureToBuffer(source: color, destination: read, bytesPerRow: 256), .read(buffer: read, byteCount: 8),
        .copyTextureToBuffer(source: depth, destination: read, bytesPerRow: 256, mipLevel: 1),
        .read(buffer: read, byteCount: 8),
    ])
    let snapshots = try await ticket.wait()
    guard snapshots == [[0xff00ff00, 0xff00ff00], [0, 0], Array(repeating: Float(0.25).bitPattern, count: 2)] else {
        throw GPUError.backendFailure("Unexpected read-only depth readback: \(snapshots)")
    }
    return snapshots.flatMap { $0 }
}

private func depthStencilAspectDiagnostic(_ device: Device, sampleCount: UInt32) async throws -> [UInt32] {
    let depthType = sampleCount > 1 ? "texture_depth_multisampled_2d" : "texture_depth_2d"
    let stencilType = sampleCount > 1 ? "texture_multisampled_2d<u32>" : "texture_2d<u32>"
    let source = ShaderInput.wgsl("""
        @group(0) @binding(0) var depthImage: \(depthType);
        @group(0) @binding(1) var stencilImage: \(stencilType);
        @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
            if i == 0u { return vec4f(-1, -1, 0.25, 1); }
            if i == 1u { return vec4f(3, -1, 0.25, 1); }
            return vec4f(-1, 3, 0.25, 1);
        }
        @fragment fn seed() -> @location(0) vec4f { return vec4f(0); }
        @fragment fn useDepth() -> @location(0) vec4f {
            let d = textureLoad(depthImage, vec2u(0), 0u);
            return select(vec4f(1, 0, 0, 1), vec4f(0, 1, 0, 1), d > 0.24 && d < 0.26);
        }
        @fragment fn useStencil() -> @location(0) vec4f {
            let s = textureLoad(stencilImage, vec2u(0), 0u).r;
            return select(vec4f(1, 0, 0, 1), vec4f(0, 1, 0, 1), s == 7u);
        }
        """)
    let compiler = IndependentShaderCompiler()
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .wgsl)
    let replace = RenderStencilFaceState(pass: .replace)
    let stencilWriter = RenderStencilState(front: replace, back: replace)
    func pipeline(_ entry: String, depthWrite: Bool, stencil: RenderStencilState? = nil) async throws -> RenderPipeline {
        try await device.createRenderPipeline(vertex: vertex,
            fragment: compiler.compileRenderStage(source, entryPoint: entry, to: .wgsl), colorFormat: .rgba8Unorm,
            depthState: .init(format: .depth24PlusStencil8, compare: .always, writeEnabled: depthWrite, stencil: stencil),
            sampleCount: sampleCount)
    }
    let seed = try await pipeline("seed", depthWrite: true, stencil: stencilWriter)
    let readDepth = try await pipeline("useDepth", depthWrite: false, stencil: stencilWriter)
    let readStencil = try await pipeline("useStencil", depthWrite: true)
    let compute = try await device.createComputePipeline(compiler.compileCompute(.wgsl("""
        @group(0) @binding(0) var depthImage: \(depthType);
        @group(0) @binding(1) var stencilImage: \(stencilType);
        @group(0) @binding(2) var<storage, read_write> out: vec4u;
        @compute @workgroup_size(1) fn readAspects() {
            let d = textureLoad(depthImage, vec2u(0), 0u);
            out = vec4u(select(0u, 1u, d > 0.24 && d < 0.26),
                        textureLoad(stencilImage, vec2u(0), 0u).r, textureDimensions(stencilImage));
        }
        """), to: .wgsl))
    let baseMip: UInt32 = sampleCount == 1 ? 1 : 0
    let combined = try await device.createTexture(.init(width: 2 << baseMip, height: 1 << baseMip, format: .depth24PlusStencil8,
        usage: [.renderAttachment, .textureBinding], mipLevelCount: baseMip + 1, sampleCount: sampleCount))
    let attachment = try await device.createTextureView(texture: combined, .init(baseMipLevel: baseMip))
    let depth = try await device.createTextureView(texture: combined, .init(baseMipLevel: baseMip, aspect: .depthOnly))
    let stencil = try await device.createTextureView(texture: combined, .init(baseMipLevel: baseMip, aspect: .stencilOnly))
    let color = try await device.createTexture(.init(width: 2, height: 1, format: .rgba8Unorm,
        usage: [.renderAttachment, .copySource]))
    let target: ResourceID
    let resolve: RenderResolveAttachment?
    if sampleCount > 1 {
        target = try await device.createTexture(.init(width: 2, height: 1, format: .rgba8Unorm,
            usage: .renderAttachment, sampleCount: sampleCount))
        resolve = .init(texture: color)
    } else {
        target = color
        resolve = nil
    }
    let out = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 16, usage: [.copyDestination, .mapRead]))
    let copyColor = BufferCommand.copyTextureToBuffer(source: color, destination: read, bytesPerRow: 256)
    let readColor = BufferCommand.read(buffer: read, byteCount: 8)
    let ticket = try await device.submitCommands([
        .render(pipeline: seed, target: target, clear: .transparentBlack, vertexCount: 3,
                depth: .init(texture: attachment, stencil: .init(reference: 173)), resolve: resolve),
        .dispatchCompute(pipeline: compute, bindings: [.init(binding: 2): out], workgroups: .init(x: 1),
                         textures: [.init(binding: 0): depth, .init(binding: 1): stencil]),
        .copy(source: out, destination: read, byteCount: 16), .read(buffer: read, byteCount: 16),
        .render(pipeline: readDepth, target: target, clear: .transparentBlack, vertexCount: 3,
                textures: [.init(binding: 0): depth],
                depth: .init(texture: attachment, loadAction: .readOnly,
                             stencil: .init(loadAction: .load, reference: 7)), resolve: resolve),
        copyColor, readColor,
        .render(pipeline: readStencil, target: target, clear: .transparentBlack, vertexCount: 3,
                textures: [.init(binding: 1): stencil],
                depth: .init(texture: attachment, stencil: .init(loadAction: .readOnly)), resolve: resolve),
        copyColor, readColor,
    ])
    let snapshots = try await ticket.wait()
    guard snapshots == [[1, 173, 2, 1], [0xff00ff00, 0xff00ff00], [0xff00ff00, 0xff00ff00]] else {
        throw GPUError.backendFailure("Unexpected depth and stencil aspect readback: \(snapshots)")
    }
    return snapshots.flatMap { $0 }
}

private func stencilCopyDiagnostic(_ device: Device) async throws -> [UInt32] {
    let source = ShaderInput.wgsl("""
        @group(0) @binding(0) var depthImage: texture_depth_2d;
        @group(0) @binding(1) var stencilImage: texture_2d<u32>;
        @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
            if i == 0u { return vec4f(-1, -1, 0.25, 1); }
            if i == 1u { return vec4f(3, -1, 0.25, 1); }
            return vec4f(-1, 3, 0.25, 1);
        }
        @fragment fn seed() -> @location(0) vec4f { return vec4f(0); }
        @fragment fn useDepth() -> @location(0) vec4f {
            let d = textureLoad(depthImage, vec2u(0), 0u);
            return select(vec4f(1, 0, 0, 1), vec4f(0, 1, 0, 1), d > 0.24 && d < 0.26);
        }
        @fragment fn useStencil() -> @location(0) vec4f {
            let s = textureLoad(stencilImage, vec2u(0), 0u).r;
            return select(vec4f(1, 0, 0, 1), vec4f(0, 1, 0, 1), s == 7u);
        }
        """)
    let compiler = IndependentShaderCompiler()
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .wgsl)
    let seed = try await device.createRenderPipeline(vertex: vertex,
        fragment: compiler.compileRenderStage(source, entryPoint: "seed", to: .wgsl), colorFormat: .rgba8Unorm,
        depthState: .init(format: .depth24PlusStencil8))
    let compute = try await device.createComputePipeline(
        compiler.compileCompute(.wgsl("""
        @group(0) @binding(0) var depthImage: texture_depth_2d;
        @group(0) @binding(1) var stencilImage: texture_2d<u32>;
        @group(0) @binding(2) var<storage, read_write> out: vec4u;
        @compute @workgroup_size(1) fn readAspects() {
            let d = textureLoad(depthImage, vec2u(0), 0u);
            out = vec4u(select(0u, 1u, d > 0.24 && d < 0.26),
                        textureLoad(stencilImage, vec2u(0), 0u).r, textureDimensions(stencilImage));
        }
        """), to: .wgsl))
    let combined = try await device.createTexture(.init(width: 6, height: 4, format: .depth24PlusStencil8,
        usage: [.renderAttachment, .textureBinding, .copySource, .copyDestination], mipLevelCount: 2))
    let attachment = try await device.createTextureView(texture: combined, .init(baseMipLevel: 1))
    let clone = try await device.createTexture(.init(width: 3, height: 2, format: .depth24PlusStencil8,
        usage: [.textureBinding, .copySource, .copyDestination]))
    let depth = try await device.createTextureView(texture: clone, .init(aspect: .depthOnly))
    let stencil = try await device.createTextureView(texture: clone, .init(aspect: .stencilOnly))
    let color = try await device.createTexture(.init(width: 3, height: 2, format: .rgba8Unorm, usage: .renderAttachment))
    let upload = try await device.createBuffer(.init(size: 264, usage: [.copySource, .copyDestination]))
    let read = try await device.createBuffer(.init(size: 264, usage: [.copyDestination, .mapRead]))
    let out = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
    let ticket = try await device.submitCommands([
        .render(pipeline: seed, target: color, clear: .transparentBlack, vertexCount: 3,
                depth: .init(texture: attachment, stencil: .init())),
        .write(buffer: upload, offset: 4, words: [0xaaf08001]),
        .write(buffer: upload, offset: 260, words: [0xbba50711]),
        .copyBufferToTexture(source: upload, layout: .init(offset: 4, bytesPerRow: 256), destination: combined,
                             region: .init(width: 3, height: 2, mipLevel: 1), aspect: .stencilOnly),
        .copyTextureToTexture(source: combined, destination: clone, sourceRegion: .init(width: 3, height: 2, mipLevel: 1)),
        .copyTextureToBuffer(source: clone, destination: read, bytesPerRow: 256,
                             aspect: .stencilOnly, destinationOffset: 4),
        .read(buffer: read, byteCount: 264),
        .dispatchCompute(pipeline: compute, bindings: [.init(binding: 2): out], workgroups: .init(x: 1),
                         textures: [.init(binding: 0): depth, .init(binding: 1): stencil]),
        .copy(source: out, destination: read, byteCount: 16), .read(buffer: read, byteCount: 16),
    ])
    var expected: [UInt32] = Array(repeating: 0, count: 66)
    expected[1] = 0x00f08001
    expected[65] = 0x00a50711
    let snapshots = try await ticket.wait()
    guard snapshots == [expected, [1, 1, 3, 2]] else {
        throw GPUError.backendFailure("Unexpected stencil copy readback: \(snapshots)")
    }
    return [snapshots[0][1], snapshots[0][65]] + snapshots[1]
}

private func textureCopyDiagnostic(_ device: Device) async throws -> [UInt32] {
    let image = try await device.createTexture(.init(width: 4, height: 2, format: .rgba8Unorm,
        usage: [.copySource, .copyDestination], mipLevelCount: 2))
    let destination = try await device.createTexture(.init(width: 2, height: 1, format: .rgba8Unorm,
        usage: [.copySource, .copyDestination]))
    let upload = try await device.createBuffer(.init(size: 16, usage: [.copySource, .copyDestination]))
    let read = try await device.createBuffer(.init(size: 8, usage: [.copyDestination, .mapRead]))
    let ticket = try await device.submitCommands([
        .write(buffer: upload, offset: 0, words: [0xff00ff00, 0xff0000ff, 0xffff0000, 0xffffffff]),
        .copyBufferToTexture(source: upload, layout: .init(), destination: image, region: .init(y: 1, width: 4, height: 1)),
        .copyTextureToTexture(source: image, destination: image,
                              sourceRegion: .init(x: 1, y: 1, width: 1, height: 1),
                              destinationOrigin: .init(x: 1, mipLevel: 1)),
        .copyTextureToTexture(source: image, destination: destination, sourceRegion: .init(width: 2, height: 1, mipLevel: 1)),
        .copyTextureToBuffer(source: destination, destination: read, bytesPerRow: 256),
        .read(buffer: read, byteCount: 8),
    ])
    let snapshots = try await ticket.wait()
    guard snapshots == [[0, 0xff0000ff]] else {
        throw GPUError.backendFailure("Unexpected color texture copy readback: \(snapshots)")
    }
    return snapshots[0]
}

private func arrayTextureDiagnostic(_ device: Device) async throws -> [UInt32] {
    let source = ShaderInput.wgsl("""
        @group(0) @binding(0) var image: texture_2d_array<u32>;
        @group(0) @binding(1) var<storage, read_write> out: array<vec4u, 5>;
        @compute @workgroup_size(1) fn run() {
            out[0] = textureLoad(image, vec2u(0), 0u, 0u);
            out[1] = textureLoad(image, vec2i(1, 0), 1i, 0i);
            out[2] = textureLoad(image, vec2u(0), 2u, 0u);
            out[3] = textureLoad(image, vec2i(0), -1i, 0i) +
                     textureLoad(image, vec2u(0), 0xffffffffu, 0u) +
                     textureLoad(image, vec2u(0), 1u, 99u);
            out[4] = vec4u(textureDimensions(image, 0), textureNumLayers(image), textureNumLevels(image));
        }
        """)
    let shader = try IndependentShaderCompiler().compileCompute(source, to: .wgsl)
    let pipeline = try await device.createComputePipeline(shader)
    let image = try await device.createTexture(.init(width: 4, height: 2, format: .rgba8Uint,
        usage: [.copySource, .copyDestination, .textureBinding], mipLevelCount: 2, arrayLayerCount: 3))
    let view = try await device.createTextureView(texture: image, .init(baseMipLevel: 1))
    let upload = try await device.createBuffer(.init(size: 16, usage: [.copySource, .copyDestination]))
    let out = try await device.createBuffer(.init(size: 80, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 80, usage: [.copyDestination, .mapRead]))
    func load(_ texture: ResourceID) -> BufferCommand {
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 1): out], workgroups: .init(x: 1),
                         textures: [.init(binding: 0): texture])
    }
    let ticket = try await device.submitCommands([
        .write(buffer: upload, offset: 0, words: [0xf0800201, 0x04030201, 0x08070605, 0xffeeddcc]),
        .copyBufferToTexture(source: upload, layout: .init(), destination: image,
                             region: .init(width: 2, height: 1, mipLevel: 1, arrayLayer: 0)),
        .copyBufferToTexture(source: upload, layout: .init(offset: 8), destination: image,
                             region: .init(width: 2, height: 1, mipLevel: 1, arrayLayer: 1)),
        load(view),
        .copy(source: out, destination: read, byteCount: 80), .read(buffer: read, byteCount: 80),
        .copyTextureToTexture(source: image, destination: image,
                              sourceRegion: .init(width: 2, height: 1, mipLevel: 1, arrayLayer: 0),
                              destinationOrigin: .init(mipLevel: 1, arrayLayer: 2)),
        .copyTextureToBuffer(source: image, destination: read, bytesPerRow: 256, mipLevel: 1, arrayLayer: 2),
        .read(buffer: read, byteCount: 8),
    ])
    let expected: [UInt32] = [1, 2, 128, 240, 204, 221, 238, 255, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 3, 1]
    let snapshots = try await ticket.wait()
    guard snapshots == [expected, [0xf0800201, 0x04030201]] else {
        throw GPUError.backendFailure("Unexpected array texture readback: \(snapshots)")
    }
    return snapshots.flatMap { $0 }
}

private func arrayLayerViewDiagnostic(_ device: Device) async throws -> [UInt32] {
    let source = ShaderInput.wgsl("""
        @group(0) @binding(0) var source: texture_2d_array<u32>;
        @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
            if i == 0u { return vec4f(-1, -1, 0, 1); }
            if i == 1u { return vec4f(3, -1, 0, 1); }
            return vec4f(-1, 3, 0, 1);
        }
        @fragment fn fs(@builtin(position) p: vec4f) -> @location(0) vec4u {
            let x = select(0u, 1u, p.x > 1.0);
            let texel = textureLoad(source, vec2u(x, 0), 0u, 0u);
            return vec4u(texel.rgb, textureNumLayers(source));
        }
        """)
    let compiler = IndependentShaderCompiler()
    let pipeline = try await device.createRenderPipeline(
        vertex: compiler.compileRenderStage(source, entryPoint: "vs", to: .wgsl),
        fragment: compiler.compileRenderStage(source, entryPoint: "fs", to: .wgsl), colorFormat: .rgba8Uint)
    let image = try await device.createTexture(.init(width: 4, height: 2, format: .rgba8Uint,
        usage: [.renderAttachment, .textureBinding, .copySource, .copyDestination], mipLevelCount: 2, arrayLayerCount: 3))
    let target = try await device.createTextureView(texture: image,
        .init(baseMipLevel: 1, dimension: .d2, baseArrayLayer: 1, arrayLayerCount: 1))
    let sourceView = try await device.createTextureView(texture: image,
        .init(baseMipLevel: 1, dimension: .d2Array, baseArrayLayer: 2, arrayLayerCount: 1))
    let upload = try await device.createBuffer(.init(size: 8, usage: [.copySource, .copyDestination]))
    let read = try await device.createBuffer(.init(size: 8, usage: [.copyDestination, .mapRead]))
    func draw(_ sampled: ResourceID) -> BufferCommand {
        .render(pipeline: pipeline, target: target, clear: .transparentBlack, vertexCount: 3,
                textures: [.init(binding: 0): sampled])
    }
    let compute = try await device.createComputePipeline(compiler.compileCompute(.wgsl("""
        @group(0) @binding(0) var image: texture_storage_2d<rgba8unorm, write>;
        @compute @workgroup_size(1) fn run() {
            textureStore(image, vec2u(0), vec4f(1, 0, 0, 1));
        }
        """), to: .wgsl))
    let storage = try await device.createTexture(.init(width: 2, height: 1, format: .rgba8Unorm,
        usage: [.storageBinding, .copySource], arrayLayerCount: 2))
    let slice = try await device.createTextureView(texture: storage, .init(dimension: .d2, baseArrayLayer: 1))
    let ticket = try await device.submitCommands([
        .write(buffer: upload, offset: 0, words: [0x281e140a, 0x50463c32]),
        .copyBufferToTexture(source: upload, layout: .init(), destination: image,
                             region: .init(width: 2, height: 1, mipLevel: 1, arrayLayer: 2)),
        draw(sourceView),
        .copyTextureToBuffer(source: image, destination: read, bytesPerRow: 256, mipLevel: 1, arrayLayer: 1),
        .read(buffer: read, byteCount: 8),
        .copyTextureToBuffer(source: image, destination: read, bytesPerRow: 256, mipLevel: 1, arrayLayer: 2),
        .read(buffer: read, byteCount: 8),
        .dispatchCompute(pipeline: compute, bindings: [:], workgroups: .init(x: 1), textures: [.init(binding: 0): slice]),
        .copyTextureToBuffer(source: storage, destination: read, bytesPerRow: 256, arrayLayer: 1),
        .read(buffer: read, byteCount: 8),
        .copyTextureToBuffer(source: storage, destination: read, bytesPerRow: 256, arrayLayer: 0),
        .read(buffer: read, byteCount: 8),
    ])
    let snapshots = try await ticket.wait()
    guard snapshots == [[0x011e140a, 0x01463c32], [0x281e140a, 0x50463c32], [0xff0000ff, 0], [0, 0]] else {
        throw GPUError.backendFailure("Unexpected array layer view readback: \(snapshots)")
    }
    return snapshots.flatMap { $0 }
}

private func arraySamplingDiagnostic(_ device: Device) async throws -> [UInt32] {
    let source = ShaderInput.wgsl("""
        @vertex fn vs() -> @builtin(position) vec4f { return vec4f(0, 0, 0, 1); }
        @fragment fn fs() -> @location(0) vec4f { return vec4f(0); }
        """)
    let compiler = IndependentShaderCompiler()
    let render = try await device.createRenderPipeline(
        vertex: compiler.compileRenderStage(source, entryPoint: "vs", to: .wgsl),
        fragment: compiler.compileRenderStage(source, entryPoint: "fs", to: .wgsl),
        colorFormat: .rgba8Unorm, depthState: .init())
    let pipeline = try await device.createComputePipeline(compiler.compileCompute(.wgsl("""
        @group(0) @binding(0) var color: texture_2d_array<f32>;
        @group(0) @binding(1) var depth: texture_depth_2d_array;
        @group(0) @binding(2) var nearest: sampler;
        @group(0) @binding(3) var comparison: sampler_comparison;
        @group(0) @binding(4) var<storage, read_write> out: array<vec4f, 6>;
        @compute @workgroup_size(1) fn run() {
            let uv = vec2f(0.5);
            out[0] = textureSampleLevel(color, nearest, uv, -5i, 0.0);
            out[1] = textureSampleLevel(color, nearest, uv, 0xffffffffu, 0.0);
            out[2] = textureGather(0u, color, nearest, uv, -1i, vec2i(1, 0));
            let less = textureSampleCompareLevel(depth, comparison, uv, -1i, 0.5, vec2i(0));
            let greater = textureSampleCompareLevel(depth, comparison, uv, 99u, 0.5);
            let raw = textureSampleLevel(depth, nearest, uv, 0u, 0u);
            out[3] = vec4f(less, greater, raw, textureGather(depth, nearest, uv, 99u).x);
            out[4] = textureGatherCompare(depth, comparison, uv, 99u, 0.5);
            out[5] = textureSampleGrad(color, nearest, uv, 1u, vec2f(0), vec2f(0));
        }
        """), to: .wgsl))
    let color = try await device.createTexture(.init(width: 2, height: 2, format: .rgba8Unorm,
        usage: [.copyDestination, .textureBinding], arrayLayerCount: 3))
    let depth = try await device.createTexture(.init(width: 2, height: 2, format: .depth32Float,
        usage: [.renderAttachment, .textureBinding], arrayLayerCount: 3))
    let colorView = try await device.createTextureView(texture: color, .init(baseArrayLayer: 1, arrayLayerCount: 2))
    let depthView = try await device.createTextureView(texture: depth, .init(baseArrayLayer: 1, arrayLayerCount: 2))
    let firstDepth = try await device.createTextureView(texture: depth, .init(dimension: .d2, baseArrayLayer: 1))
    let lastDepth = try await device.createTextureView(texture: depth, .init(dimension: .d2, baseArrayLayer: 2))
    let target = try await device.createTexture(.init(width: 2, height: 2, format: .rgba8Unorm, usage: .renderAttachment))
    let nearest = try await device.createSampler(.init(addressModeU: .repeat))
    let compare = try await device.createSampler(.init(compare: .less))
    let upload = try await device.createBuffer(.init(size: 776, usage: [.copySource, .copyDestination]))
    let out = try await device.createBuffer(.init(size: 96, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 96, usage: [.copyDestination, .mapRead]))
    let ticket = try await device.submitCommands([
        .write(buffer: upload, offset: 0, words: [0xff0000ff, 0xff0000ff]),
        .write(buffer: upload, offset: 256, words: [0xff0000ff, 0xff0000ff]),
        .write(buffer: upload, offset: 512, words: [0xff00ff00, 0xff00ff00]),
        .write(buffer: upload, offset: 768, words: [0xff00ff00, 0xff00ff00]),
        .copyBufferToTexture(source: upload, layout: .init(bytesPerRow: 256), destination: color,
                             region: .init(width: 2, height: 2, arrayLayer: 1)),
        .copyBufferToTexture(source: upload, layout: .init(offset: 512, bytesPerRow: 256), destination: color,
                             region: .init(width: 2, height: 2, arrayLayer: 2)),
        .render(pipeline: render, target: target, clear: .transparentBlack, vertexCount: 0,
                depth: .init(texture: firstDepth, loadAction: .clear(0.25))),
        .render(pipeline: render, target: target, clear: .transparentBlack, vertexCount: 0,
                depth: .init(texture: lastDepth, loadAction: .clear(0.75))),
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 4): out], workgroups: .init(x: 1),
                         textures: [.init(binding: 0): colorView, .init(binding: 1): depthView],
                         samplers: [.init(binding: 2): nearest, .init(binding: 3): compare]),
        .copy(source: out, destination: read, byteCount: 96), .read(buffer: read, byteCount: 96),
    ])
    let expected: [Float] = [1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0.25, 0.75, 1, 1, 1, 1, 0, 1, 0, 1]
    let snapshots = try await ticket.wait()
    guard snapshots == [expected.map(\.bitPattern)] else {
        throw GPUError.backendFailure("Unexpected array sampling readback: \(snapshots)")
    }
    return snapshots[0]
}

private func storageArrayDiagnostic(_ device: Device) async throws -> [UInt32] {
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl("""
        @group(0) @binding(0) var source: texture_storage_2d_array<r32uint, read>;
        @group(0) @binding(1) var image: texture_storage_2d_array<r32uint, read_write>;
        @group(0) @binding(2) var<storage, read_write> result: array<u32, 10>;
        @compute @workgroup_size(1) fn run() {
            let xy = vec2i(0);
            let size = textureDimensions(image);
            let first = textureLoad(image, xy, 0i);
            let second = textureLoad(image, xy, 1u);
            let input = textureLoad(source, xy, 0u);
            result[0] = size.x;
            result[1] = size.y;
            result[2] = textureNumLayers(source);
            result[3] = textureNumLayers(image);
            result[4] = input.x;
            result[5] = first.x;
            result[6] = second.x;
            result[7] = textureLoad(image, xy, -1i).w;
            result[8] = textureLoad(image, xy, 0xffffffffu).w;
            result[9] = textureLoad(source, xy, 1u).w;
            textureStore(image, xy, 0i, first + input);
            textureStore(image, xy, 1u, second + vec4u(7u));
            textureStore(image, xy, -1i, vec4u(99u));
            textureStore(image, xy, 2u, vec4u(99u));
            textureStore(image, vec2u(0xffffffffu, 0u), 0u, vec4u(99u));
        }
        """), to: .wgsl)
    let pipeline = try await device.createComputePipeline(shader)
    let texture = try await device.createTexture(.init(width: 4, height: 2, format: .r32Uint,
        usage: [.copySource, .copyDestination, .storageBinding], mipLevelCount: 2, arrayLayerCount: 4))
    let source = try await device.createTextureView(texture: texture,
        .init(baseMipLevel: 1, dimension: .d2Array, arrayLayerCount: 1))
    let image = try await device.createTextureView(texture: texture,
        .init(baseMipLevel: 1, baseArrayLayer: 1, arrayLayerCount: 2))
    let slice = try await device.createTextureView(texture: texture,
        .init(baseMipLevel: 1, dimension: .d2, baseArrayLayer: 1))
    let upload = try await device.createBuffer(.init(size: 16, usage: [.copySource, .copyDestination]))
    let output = try await device.createBuffer(.init(size: 40, usage: [.storage, .copySource]))
    let readback = try await device.createBuffer(.init(size: 40, usage: [.copyDestination, .mapRead]))
    func dispatch(source: ResourceID, image: ResourceID) -> BufferCommand {
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 2): output], workgroups: .init(x: 1),
                         textures: [.init(binding: 0): source, .init(binding: 1): image])
    }
    do {
        _ = try await device.submitCommands([dispatch(source: source, image: slice)])
        throw GPUError.backendFailure("Accepted a 2D view in an array storage binding")
    } catch GPUError.validation {}
    do {
        _ = try await device.submitCommands([dispatch(source: image, image: image)])
        throw GPUError.backendFailure("Accepted overlapping writable storage array views")
    } catch GPUError.validation {}
    var commands: [BufferCommand] = [.write(buffer: upload, offset: 0, words: [6, 10, 20, 40])]
    for layer in UInt32(0)..<4 {
        commands.append(.copyBufferToTexture(source: upload, layout: .init(offset: UInt64(layer) * 4),
            destination: texture, region: .init(width: 1, height: 1, mipLevel: 1, arrayLayer: layer)))
    }
    commands += [dispatch(source: source, image: image),
                 .copy(source: output, destination: readback, byteCount: 40), .read(buffer: readback, byteCount: 40)]
    for layer in UInt32(0)..<4 {
        commands += [.copyTextureToBuffer(source: texture, destination: readback, bytesPerRow: 256,
                                          mipLevel: 1, arrayLayer: layer), .read(buffer: readback, byteCount: 8)]
    }
    let compiler = IndependentShaderCompiler()
    let drawSource = ShaderInput.wgsl("""
        @group(1) @binding(3) var image: texture_storage_2d_array<rgba8unorm, write>;
        @vertex fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
            if index == 0u { return vec4f(-1, -1, 0, 1); }
            if index == 1u { return vec4f(3, -1, 0, 1); }
            return vec4f(-1, 3, 0, 1);
        }
        fn paint(value: vec4f) { textureStore(image, vec2i(0), 1u, value); }
        @fragment fn fs() -> @location(0) vec4f {
            paint(vec4f(1, 0, 1, 1));
            return vec4f(0, 1, 0, 1);
        }
        """)
    let draw = try await device.createRenderPipeline(
        vertex: compiler.compileRenderStage(drawSource, entryPoint: "vs", to: .wgsl),
        fragment: compiler.compileRenderStage(drawSource, entryPoint: "fs", to: .wgsl), colorFormat: .rgba8Unorm)
    let target = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm, usage: .renderAttachment))
    let color = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm,
        usage: [.storageBinding, .copySource], arrayLayerCount: 2))
    commands.append(.render(pipeline: draw, target: target, clear: .transparentBlack, vertexCount: 3,
                            textures: [.init(group: 1, binding: 3): color]))
    for layer in UInt32(0)..<2 {
        commands += [.copyTextureToBuffer(source: color, destination: readback, bytesPerRow: 256, arrayLayer: layer),
                     .read(buffer: readback, byteCount: 4)]
    }
    let ticket = try await device.submitCommands(commands)
    let snapshots = try await ticket.wait()
    guard snapshots == [[2, 1, 1, 2, 6, 10, 20, 0, 0, 0], [6, 0], [16, 0], [27, 0], [40, 0], [0], [0xffff00ff]] else {
        throw GPUError.backendFailure("Unexpected storage array readback: \(snapshots)")
    }
    return snapshots.flatMap { $0 }
}

private func storageFormatDiagnostic(_ device: Device) async throws -> Int {
    let formats: [(name: String, vector: String, packed: [UInt32], lanes: [UInt32])] = [
        ("rgba8unorm", "vec4f", [0x00ff00ff], [1, 0, 1, 0].map(Float.init).map(\.bitPattern)),
        ("rgba8snorm", "vec4f", [0x007f0081], [-1, 0, 1, 0].map(Float.init).map(\.bitPattern)),
        ("rgba8uint", "vec4u", [0xff020180], [128, 1, 2, 255]),
        ("rgba8sint", "vec4i", [0xff020180], [0xffffff80, 1, 2, .max]),
        ("rgba16uint", "vec4u", [0x0001ffff, 0x80000002], [65535, 1, 2, 32768]),
        ("rgba16sint", "vec4i", [0x0001ffff, 0x80000002], [.max, 1, 2, 0xffff8000]),
        ("rgba16float", "vec4f", [0xc0003c00, 0x40003800], [1, -2, 0.5, 2].map(Float.init).map(\.bitPattern)),
        ("r32uint", "vec4u", [0xdeadbeef], [0xdeadbeef, 0, 0, 1]),
        ("r32sint", "vec4i", [0x80000005], [0x80000005, 0, 0, 1]),
        ("r32float", "vec4f", [0xc0000000], [-2, 0, 0, 1].map(Float.init).map(\.bitPattern)),
        ("rg32uint", "vec4u", [0xdeadbeef, 99], [0xdeadbeef, 99, 0, 1]),
        ("rg32sint", "vec4i", [0x80000005, 99], [0x80000005, 99, 0, 1]),
        ("rg32float", "vec4f", [0xc0000000, 0x3f000000], [-2, 0.5, 0, 1].map(Float.init).map(\.bitPattern)),
        ("rgba32uint", "vec4u", [0xdeadbeef, 1, 2, 3], [0xdeadbeef, 1, 2, 3]),
        ("rgba32sint", "vec4i", [0x80000005, 1, 2, 3], [0x80000005, 1, 2, 3]),
        ("rgba32float", "vec4f", [0xc0000000, 0x3f000000, 0x40000000, 0x3f800000], [-2, 0.5, 2, 1].map(Float.init).map(\.bitPattern)),
    ]
    let compiler = IndependentShaderCompiler()
    let upload = try await device.createBuffer(.init(size: 32, usage: [.copySource, .copyDestination]))
    let output = try await device.createBuffer(.init(size: 48, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 64, usage: [.copyDestination, .mapRead]))
    let region = TextureCopyRegion(width: 1, height: 1)
    var commands: [BufferCommand] = []
    var expected: [[UInt32]] = []
    for fixture in formats {
        let readWrite = fixture.name.hasPrefix("r32")
        let shader = try compiler.compileCompute(.wgsl("""
            @group(0) @binding(0) var input: texture_storage_2d<\(fixture.name), read>;
            @group(0) @binding(1) var output: texture_storage_2d<\(fixture.name), \(readWrite ? "read_write" : "write")>;
            @group(0) @binding(2) var<storage, read_write> result: array<\(fixture.vector), 3>;
            @compute @workgroup_size(1) fn run() {
                let xy = vec2i(0);
                let value = textureLoad(input, xy);
                result[0] = value;
                result[1] = textureLoad(input, vec2i(-1));
                \(readWrite ? "result[2] = textureLoad(output, xy);" : "result[2] = \(fixture.vector)(0);")
                textureStore(output, xy, value);
            }
            """), to: .wgsl)
        guard let binding = shader.storageTextures.first else {
            throw GPUError.backendFailure("Missing \(fixture.name) storage reflection")
        }
        let format = try binding.textureFormat()
        let pipeline = try await device.createComputePipeline(shader)
        let source = try await device.createTexture(.init(width: 1, height: 1, format: format,
            usage: [.storageBinding, .copyDestination]))
        let destination = try await device.createTexture(.init(width: 1, height: 1, format: format,
            usage: [.storageBinding, .copySource]))
        commands += [
            .write(buffer: upload, offset: 16, words: fixture.packed),
            .copyBufferToTexture(source: upload, layout: .init(offset: 16), destination: source, region: region),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 2): output], workgroups: .init(x: 1),
                             textures: [.init(binding: 0): source, .init(binding: 1): destination]),
            .copy(source: output, destination: read, byteCount: 48), .read(buffer: read, byteCount: 48),
            .copyTextureToBuffer(source: destination, destination: read, bytesPerRow: 256),
            .read(buffer: read, byteCount: UInt64(fixture.packed.count) * 4),
        ]
        let alpha: UInt32 = fixture.vector == "vec4f" ? Float(1).bitPattern : 1
        expected += [fixture.lanes + [0, 0, 0, 0, 0, 0, 0, readWrite ? alpha : 0], fixture.packed]
    }
    let snapshots = try await device.submitCommands(commands).wait()
    guard snapshots.count == expected.count else {
        throw GPUError.backendFailure("Unexpected storage format snapshot count: \(snapshots.count)")
    }
    for (index, snapshot) in snapshots.enumerated() where snapshot != expected[index] {
        throw GPUError.backendFailure("Unexpected \(formats[index / 2].name) storage readback: \(snapshot)")
    }
    return formats.count
}

private func cubeTextureDiagnostic(_ device: Device) async throws -> [UInt32] {
    let compiler = IndependentShaderCompiler()
    let renderSource = ShaderInput.wgsl("""
        @vertex fn vs() -> @builtin(position) vec4f { return vec4f(0, 0, 0, 1); }
        @fragment fn fs() -> @location(0) vec4f { return vec4f(0); }
        """)
    let render = try await device.createRenderPipeline(
        vertex: compiler.compileRenderStage(renderSource, entryPoint: "vs", to: .wgsl),
        fragment: compiler.compileRenderStage(renderSource, entryPoint: "fs", to: .wgsl),
        colorFormat: .rgba8Unorm, depthState: .init())
    let pipeline = try await device.createComputePipeline(compiler.compileCompute(.wgsl("""
        @group(0) @binding(0) var cube: texture_cube<f32>;
        @group(0) @binding(1) var cubes: texture_cube_array<f32>;
        @group(0) @binding(2) var depth: texture_depth_cube;
        @group(0) @binding(3) var depths: texture_depth_cube_array;
        @group(0) @binding(4) var nearest: sampler;
        @group(0) @binding(5) var comparison: sampler_comparison;
        @group(0) @binding(6) var<storage, read_write> out: array<vec4f, 11>;
        @compute @workgroup_size(1) fn run() {
            let direction = vec3f(1, 0, 0);
            out[0] = textureSampleLevel(cube, nearest, direction, 0.0);
            out[1] = textureSampleLevel(cube, nearest, vec3f(-1, 0, 0), 0.0);
            out[2] = textureSampleLevel(cube, nearest, vec3f(0, 1, 0), 0.0);
            out[3] = textureSampleLevel(cube, nearest, vec3f(0, -1, 0), 0.0);
            out[4] = textureSampleLevel(cube, nearest, vec3f(0, 0, 1), 0.0);
            out[5] = textureSampleLevel(cube, nearest, vec3f(0, 0, -1), 0.0);
            out[6] = textureSampleGrad(cubes, nearest, direction, -5i, vec3f(0), vec3f(0));
            out[7] = textureSampleLevel(cubes, nearest, direction, 0xffffffffu, 0.0);
            out[8] = textureGather(2, cubes, nearest, direction, 1u);
            let size = textureDimensions(cubes, 0u);
            out[9] = vec4f(f32(size.x), f32(size.y), f32(textureNumLayers(cubes)), f32(textureNumLevels(cube)));
            out[10] = vec4f(textureSampleLevel(depth, nearest, direction, 0u),
                textureSampleCompareLevel(depth, comparison, direction, 0.5),
                textureSampleCompareLevel(depths, comparison, direction, -1i, 0.5),
                textureGatherCompare(depths, comparison, direction, 0xffffffffu, 0.5).x);
        }
        """), to: .wgsl))
    let drawSource = ShaderInput.wgsl("""
        @group(0) @binding(0) var cube: texture_cube<f32>;
        @group(0) @binding(1) var cubes: texture_cube_array<f32>;
        @group(0) @binding(2) var depths: texture_depth_cube_array;
        @group(0) @binding(3) var nearest: sampler;
        @group(0) @binding(4) var comparison: sampler_comparison;
        @vertex fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
            if index == 0u { return vec4f(-1, -1, 0, 1); }
            if index == 1u { return vec4f(3, -1, 0, 1); }
            return vec4f(-1, 3, 0, 1);
        }
        @fragment fn fs() -> @location(0) vec4f {
            let direction = vec3f(1, 0, 0);
            let a = textureSample(cube, nearest, direction);
            let b = textureSampleBias(cubes, nearest, direction, 1i, 0.25);
            let cmp = textureSampleCompare(depths, comparison, direction, 1i, 0.5);
            let raw = textureSample(depths, nearest, direction, 1i);
            return vec4f(a.b, b.b, cmp, raw);
        }
        """)
    let draw = try await device.createRenderPipeline(
        vertex: compiler.compileRenderStage(drawSource, entryPoint: "vs", to: .wgsl),
        fragment: compiler.compileRenderStage(drawSource, entryPoint: "fs", to: .wgsl), colorFormat: .rgba8Unorm)
    let color = try await device.createTexture(.init(width: 4, height: 4, format: .rgba8Unorm,
        usage: [.copyDestination, .textureBinding], mipLevelCount: 2, arrayLayerCount: 13))
    let depth = try await device.createTexture(.init(width: 4, height: 4, format: .depth32Float,
        usage: [.renderAttachment, .textureBinding], mipLevelCount: 2, arrayLayerCount: 13))
    let cube = TextureViewDescriptor(baseMipLevel: 1, dimension: .cube, baseArrayLayer: 7)
    let array = TextureViewDescriptor(baseMipLevel: 1, dimension: .cubeArray, baseArrayLayer: 1)
    let singleColor = try await device.createTextureView(texture: color, cube)
    let arrayColor = try await device.createTextureView(texture: color, array)
    let singleDepth = try await device.createTextureView(texture: depth, cube)
    let arrayDepth = try await device.createTextureView(texture: depth, array)
    let firstFace = try await device.createTextureView(texture: depth, .init(baseMipLevel: 1, dimension: .d2, baseArrayLayer: 1))
    let lastFace = try await device.createTextureView(texture: depth, .init(baseMipLevel: 1, dimension: .d2, baseArrayLayer: 7))
    do {
        _ = try await device.createTextureView(texture: color, .init(dimension: .cubeArray))
        throw GPUError.backendFailure("Accepted an invalid cube view")
    } catch GPUError.validation {}
    do {
        _ = try await device.createTextureView(texture: color, .init(dimension: .cube, arrayLayerCount: 5))
        throw GPUError.backendFailure("Accepted an invalid cube view")
    } catch GPUError.validation {}
    let rectangular = try await device.createTexture(.init(width: 4, height: 2, format: .rgba8Unorm,
        usage: .textureBinding, arrayLayerCount: 6))
    do {
        _ = try await device.createTextureView(texture: rectangular, .init(dimension: .cube))
        throw GPUError.backendFailure("Accepted an invalid cube view")
    } catch GPUError.validation {}
    let target = try await device.createTexture(.init(width: 2, height: 2, format: .rgba8Unorm,
        usage: [.renderAttachment, .copySource]))
    let nearest = try await device.createSampler(.init())
    let compare = try await device.createSampler(.init(compare: .less))
    let output = try await device.createBuffer(.init(size: 176, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 264, usage: [.copyDestination, .mapRead]))
    let upload = try await device.createBuffer(.init(size: 7 * 512, usage: [.copySource, .copyDestination]))
    let faces: [(UInt32, UInt32)] = [(1, 0xff0000ff), (7, 0xffff0000), (8, 0xff0000ff), (9, 0xff00ff00),
                                  (10, 0xff00ffff), (11, 0xffffff00), (12, 0xffff00ff)]
    var commands: [BufferCommand] = []
    for (index, (layer, texel)) in faces.enumerated() {
        let offset = UInt64(index) * 512
        commands += [.write(buffer: upload, offset: offset, words: [texel, texel]),
                     .write(buffer: upload, offset: offset + 256, words: [texel, texel]),
                     .copyBufferToTexture(source: upload, layout: .init(offset: offset, bytesPerRow: 256),
                                          destination: color, region: .init(width: 2, height: 2, mipLevel: 1, arrayLayer: layer))]
    }
    func dispatch(_ cube: ResourceID) -> BufferCommand {
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 6): output], workgroups: .init(x: 1),
                         textures: [.init(binding: 0): cube, .init(binding: 1): arrayColor,
                                    .init(binding: 2): singleDepth, .init(binding: 3): arrayDepth],
                         samplers: [.init(binding: 4): nearest, .init(binding: 5): compare])
    }
    do {
        _ = try await device.submitCommands([dispatch(arrayColor)])
        throw GPUError.backendFailure("Accepted a cube array view for a cube binding")
    } catch GPUError.validation("Invalid sampled texture binding") {}
    commands += [
        .render(pipeline: render, target: target, clear: .transparentBlack, vertexCount: 0,
                depth: .init(texture: firstFace, loadAction: .clear(0.25))),
        .render(pipeline: render, target: target, clear: .transparentBlack, vertexCount: 0,
                depth: .init(texture: lastFace, loadAction: .clear(0.75))),
        dispatch(singleColor), .copy(source: output, destination: read, byteCount: 176), .read(buffer: read, byteCount: 176),
        .render(pipeline: draw, target: target, clear: .transparentBlack, vertexCount: 3,
                textures: [.init(binding: 0): singleColor, .init(binding: 1): arrayColor, .init(binding: 2): arrayDepth],
                samplers: [.init(binding: 3): nearest, .init(binding: 4): compare]),
        .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256), .read(buffer: read, byteCount: 8),
    ]
    let expected: [Float] = [0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1,
                             1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 0.75, 1, 0, 1]
    let ticket = try await device.submitCommands(commands)
    let snapshots = try await ticket.wait()
    guard snapshots == [expected.map(\.bitPattern), [0xbfffffff, 0xbfffffff]] else {
        throw GPUError.backendFailure("Unexpected cube texture readback: \(snapshots)")
    }
    return snapshots.flatMap { $0 }
}

private func volumeTextureDiagnostic(_ device: Device) async throws -> [UInt32] {
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl("""
        @group(0) @binding(0) var volume: texture_3d<f32>;
        @group(0) @binding(1) var nearest: sampler;
        @group(0) @binding(2) var<storage, read_write> result: array<vec4f, 9>;
        @compute @workgroup_size(1) fn run() {
            result[0] = textureLoad(volume, vec3i(0, 0, 0), 0i);
            result[1] = textureLoad(volume, vec3u(0, 0, 1), 0u);
            result[2] = textureLoad(volume, vec3i(0, 0, 2), 0i);
            result[3] = textureLoad(volume, vec3i(0, 0, -1), 0i);
            result[4] = textureLoad(volume, vec3u(0, 0, 0xffffffffu), 0i);
            result[5] = textureSampleLevel(volume, nearest, vec3f(0.25, 0.5, 0.5), 0.0);
            result[6] = textureSampleGrad(volume, nearest, vec3f(0.25, 0.5, 0.16666667), vec3f(0), vec3f(0), vec3i(0, 0, 1));
            let size = textureDimensions(volume, 0u);
            result[7] = vec4f(f32(size.x), f32(size.y), f32(size.z), f32(textureNumLevels(volume)));
            let invalid = textureDimensions(volume, 1u);
            result[8] = vec4f(f32(invalid.x), f32(invalid.y), f32(invalid.z), textureLoad(volume, vec3u(0), -1i).a);
        }
        """), to: .wgsl)
    let pipeline = try await device.createComputePipeline(shader)
    let texture = TextureDescriptor(width: 4, height: 2, format: .rgba8Unorm,
        usage: [.copySource, .copyDestination, .textureBinding], mipLevelCount: 2, dimension: .d3, depth: 6)
    let source = try await device.createTexture(texture)
    let volume = try await device.createTexture(texture)
    let view = try await device.createTextureView(texture: volume, .init(baseMipLevel: 1))
    do {
        _ = try await device.createTextureView(texture: volume, .init(dimension: .d2))
        throw GPUError.backendFailure("Accepted a 2D view of a volume texture")
    } catch GPUError.validation("Texture view dimension does not match the texture") {}
    let nearest = try await device.createSampler(.init())
    let upload = try await device.createBuffer(.init(size: 264, usage: [.copySource, .copyDestination]))
    let out = try await device.createBuffer(.init(size: 144, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 520, usage: [.copyDestination, .mapRead]))
    let region = TextureCopyRegion(width: 2, height: 1, mipLevel: 1, depth: 2)
    do {
        _ = try await device.submitCommands([.copyTextureToBuffer(source: volume, destination: read,
            bytesPerRow: 256, mipLevel: 1, z: 2, depth: 2)])
        throw GPUError.backendFailure("Accepted volume copy outside the mip depth")
    } catch GPUError.validation("Invalid texture copy source or aspect") {}
    let ticket = try await device.submitCommands([
        .write(buffer: upload, offset: 0, words: [0xff0000ff, 0xff0000ff]),
        .write(buffer: upload, offset: 256, words: [0xff00ff00, 0xff00ff00]),
        .copyBufferToTexture(source: upload, layout: .init(bytesPerRow: 256), destination: source, region: region),
        .copyTextureToTexture(source: source, destination: volume, sourceRegion: region, destinationOrigin: .init(mipLevel: 1)),
        .copyTextureToTexture(source: source, destination: volume,
            sourceRegion: .init(width: 1, height: 1, mipLevel: 1, z: 1), destinationOrigin: .init(x: 1, mipLevel: 1, z: 2)),
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 2): out], workgroups: .init(x: 1),
                         textures: [.init(binding: 0): view], samplers: [.init(binding: 1): nearest]),
        .copy(source: out, destination: read, byteCount: 144), .read(buffer: read, byteCount: 144),
        .copyTextureToBuffer(source: volume, destination: read, bytesPerRow: 256, mipLevel: 1, depth: 3),
        .read(buffer: read, byteCount: 520),
    ])
    let snapshots = try await ticket.wait()
    let expected: [Float] = [1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 1, 0, 1, 0, 1, 0, 1, 2, 1, 3, 1, 0, 0, 0, 0]
    let planes = [0, 1, 64, 65, 128, 129].map { snapshots[1][$0] }
    guard snapshots[0] == expected.map(\.bitPattern),
          planes == [0xff0000ff, 0xff0000ff, 0xff00ff00, 0xff00ff00, 0, 0xff00ff00] else {
        throw GPUError.backendFailure("Unexpected volume texture readback: \(snapshots[0]), \(planes)")
    }
    return snapshots[0] + planes
}

private func renderDiagnostic(
    _ device: Device
) async throws -> (checked: Int, probe: [UInt32], vertexStreams: Int, indexedVertices: Int, instancesPerDraw: Int) {
    let source = ShaderInput.wgsl("""
        struct VertexInput {
            @location(4) clip: vec4f,
            @location(1) tint: vec4f,
            @location(6) gain: i32,
            @builtin(instance_index) instance: u32,
        }
        struct VertexOutput {
            @location(2) tint: vec4f,
            @builtin(position) clip: vec4f,
            @location(5) @interpolate(perspective, centroid) gain: f32,
        }
        struct FragmentInput {
            @location(5) @interpolate(perspective, centroid) gain: f32,
            @location(2) tint: vec4f,
        }
        fn makeVertex(tint: vec4f, clip: vec4f, gain: f32) -> VertexOutput {
            var output: VertexOutput;
            output.tint = tint;
            output.clip = clip;
            output.gain = gain;
            return output;
        }
        @vertex fn vertexMain(input: VertexInput) -> VertexOutput {
            if input.gain == -1i && input.instance == 2u {
                return makeVertex(input.tint, input.clip, 1.0);
            }
            return makeVertex(input.tint, vec4f(0.0, 0.0, 0.0, 1.0), 0.0);
        }
        @fragment fn fragmentMain(input: FragmentInput) -> @location(0) vec4f {
            return shade(input.tint, input.gain);
        }
        fn shade(tint: vec4f, gain: f32) -> vec4f {
            var color = tint;
            let before = color;
            color.r = 0.0;
            return vec4f(before.r * gain, color.g, color.b, before.a);
        }
        """)
    let compiler = IndependentShaderCompiler()
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vertexMain", to: .wgsl)
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fragmentMain", to: .wgsl)
    let layouts: [VertexBufferLayout] = [
        .init(stride: 32, attributes: [.init(location: 4, offset: 0, format: .float32x4),
                                       .init(location: 1, offset: 16, format: .float32x3)]),
        .init(stride: 8, stepMode: .instance, attributes: [.init(location: 6, offset: 4, format: .sint32)]),
    ]
    let pipeline = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm,
                                                         vertexBuffers: layouts)
    let geometry = try await device.createBuffer(.init(size: 128, usage: [.vertex, .copyDestination]))
    let gain = try await device.createBuffer(.init(size: 24, usage: [.vertex, .copyDestination]))
    let shortIndices = try await device.createBuffer(.init(size: 12, usage: [.index, .copyDestination]))
    let wordIndices = try await device.createBuffer(.init(size: 24, usage: [.index, .copyDestination]))
    let vertices: [Float] = [
        99, 4, 4, 0, 1, 0, 0, 0,
        99, -1, 1, 0.25, 1, 1, 0, 0,
        99, 2, 2, 0.5, 2, 0, 1, 0,
        99, -4, -4, 1, 4, 0, 0, 1,
    ]
    let clear = RenderClearColor(red: 0, green: 0, blue: 1, alpha: 1)
    let texture = try await device.createTexture(.init(width: 8, height: 8, format: .rgba8Unorm,
                                                       usage: [.renderAttachment, .copySource]))
    let byteCount: UInt64 = 1824
    let buffer = try await device.createBuffer(.init(size: byteCount, usage: [.copyDestination, .mapRead]))
    let vertexBindings: [VertexBufferBinding] = [.init(buffer: geometry, offset: 4), .init(buffer: gain)]
    let instances = DrawInstances(count: 2, first: 1)
    let draw = IndexedDraw(indexCount: 3, firstIndex: 1, baseVertex: -1, instances: instances)
    if device.supportsPresentation {
        let initial = try await device.createTexture(.init(width: 4, height: 4, format: .rgba8Unorm, usage: .copySource))
        let frame = try await device.present(initial)
        try await device.destroyTexture(initial)
        try await frame.wait()
    }
    do {
        _ = try await device.submitCommands([.copyTextureToBuffer(source: texture, destination: buffer, bytesPerRow: 32)])
        throw GPUError.backendFailure("Swift accepted an invalid texture copy stride")
    } catch GPUError.validation {}
    do {
        _ = try await device.submitCommands([
            .render(pipeline: pipeline, target: texture, clear: clear, vertexCount: 4,
                    vertexBuffers: [.init(buffer: geometry, offset: 8), .init(buffer: gain)]),
        ])
        throw GPUError.backendFailure("Swift accepted an out-of-range vertex fetch")
    } catch GPUError.validation {}
    do {
        _ = try await device.submitCommands([
            .renderIndexed(pipeline: pipeline, target: texture, clear: clear,
                           indexBuffer: .init(buffer: shortIndices, format: .uint16, offset: 2, size: 6),
                           draw: draw, vertexBuffers: vertexBindings),
        ])
        throw GPUError.backendFailure("Swift accepted an out-of-range index slice")
    } catch GPUError.validation {}
    do {
        _ = try await device.submitCommands([
            .renderIndexed(pipeline: pipeline, target: texture, clear: clear,
                           indexBuffer: .init(buffer: wordIndices, format: .uint32, offset: 4),
                           draw: .init(indexCount: 3, firstIndex: 1, baseVertex: -1,
                                       instances: .init(count: 3, first: 1)),
                           vertexBuffers: vertexBindings),
        ])
        throw GPUError.backendFailure("Swift accepted an out-of-range instance fetch")
    } catch GPUError.validation {}
    let ticket = try await device.submitCommands([
        .write(buffer: geometry, offset: 0, words: vertices.map(\.bitPattern)),
        .write(buffer: gain, offset: 0, words: [99, 0, 99, 0, 99, .max]),
        .write(buffer: shortIndices, offset: 0, words: [0x0000ffff, 0x00030002, 0xffff0004]),
        .write(buffer: wordIndices, offset: 0, words: [.max, 0, 2, 3, 4, .max]),
        .copyTextureToBuffer(source: texture, destination: buffer, bytesPerRow: 256),
        .read(buffer: buffer, byteCount: byteCount),
        .renderIndexed(pipeline: pipeline, target: texture, clear: clear,
                       indexBuffer: .init(buffer: shortIndices, format: .uint16, offset: 2, size: 8),
                       draw: draw, vertexBuffers: vertexBindings),
        .copyTextureToBuffer(source: texture, destination: buffer, bytesPerRow: 256),
        .read(buffer: buffer, byteCount: byteCount),
        .renderIndexed(pipeline: pipeline, target: texture, clear: clear,
                       indexBuffer: .init(buffer: wordIndices, format: .uint32, offset: 4, size: 16),
                       draw: draw, vertexBuffers: vertexBindings),
        .copyTextureToBuffer(source: texture, destination: buffer, bytesPerRow: 256),
        .read(buffer: buffer, byteCount: byteCount),
        .render(pipeline: pipeline, target: texture, clear: clear, vertexCount: 3, vertexBuffers: vertexBindings,
                firstVertex: 1, instances: instances),
        .copyTextureToBuffer(source: texture, destination: buffer, bytesPerRow: 256),
        .read(buffer: buffer, byteCount: byteCount),
    ])
    let frame = device.supportsPresentation ? try await device.present(texture) : nil
    try await device.destroyBuffer(geometry)
    try await device.destroyBuffer(shortIndices)
    try await device.destroyTexture(texture)
    let snapshots = try await ticket.wait()
    try await frame?.wait()
    guard snapshots.count == 4, snapshots.allSatisfy({ $0.count == Int(byteCount / 4) }),
          snapshots[0].allSatisfy({ $0 == 0 }), snapshots[1] == snapshots[2],
          snapshots[1] == snapshots[3] else {
        throw GPUError.backendFailure("Unexpected WebGPU texture snapshots for instanced rendering")
    }
    var checked = 0
    let pixels = snapshots[1]
    for y in 0..<8 {
        for x in 0..<8 where x + y != 7 {
            let pixel = pixels[y * 64 + x]
            if x + y < 7 {
                let u = (Double(x) + 0.5) / 8, v = (Double(y) + 0.5) / 8
                let weights = [1 - u - v, u / 2, v / 4]
                let sum = weights.reduce(0, +)
                guard pixel >> 24 == 255, weights.enumerated().allSatisfy({ channel, weight in
                    let expected = Int((255 * weight / sum).rounded())
                    return abs(Int((pixel >> (8 * channel)) & 255) - expected) <= 1
                }) else { throw GPUError.backendFailure("Unexpected WebGPU interpolation at \(x), \(y)") }
            } else if pixel != 0xffff0000 {
                throw GPUError.backendFailure("Unexpected WebGPU clear pixel at \(x), \(y)")
            }
            checked += 1
        }
    }
    return (checked, (0..<3).map { (pixels[0] >> (8 * $0)) & 255 }, pipeline.vertexBuffers.count,
            2 * Int(draw.indexCount), Int(instances.count))
}
#endif
