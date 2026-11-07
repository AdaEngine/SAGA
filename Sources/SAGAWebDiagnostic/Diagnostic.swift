@main struct SAGAWebDiagnostic {
    static func main() {
        #if !os(WASI)
        print("Build SAGAWebDiagnostic with the Swift Wasm SDK and run it in a WebGPU browser.")
        #endif
    }
}

#if os(WASI) && compiler(>=6.3)
import SAGATypes
import SAGACore
import SAGAShaderIR
import SAGACompiler
import SAGAWebGPU

func runWebDiagnostic(capabilities: DeviceCapabilities) async throws -> String {
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
    let host = WebGPUHost(capabilities: capabilities)
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
        let render = try await renderDiagnostic(device)
        await device.close()
        let snapshots = result.map { $0.map(String.init).joined(separator: ",") }
        let probe = render.probe.map(String.init).joined(separator: ",")
        let metrics = [String(render.checked), probe, String(integerWords), String(render.vertexStreams)]
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

private func renderDiagnostic(_ device: Device) async throws -> (checked: Int, probe: [UInt32], vertexStreams: Int) {
    let source = ShaderInput.wgsl("""
        struct VertexInput {
            @location(4) clip: vec4f,
            @location(1) tint: vec4f,
            @location(6) gain: i32,
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
            if input.gain == -1i { return makeVertex(input.tint, input.clip, 1.0); }
            return makeVertex(input.tint, input.clip, 0.0);
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
        .init(stride: 0, attributes: [.init(location: 6, offset: 4, format: .sint32)]),
    ]
    let pipeline = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm,
                                                         vertexBuffers: layouts)
    let geometry = try await device.createBuffer(.init(size: 96, usage: [.vertex, .copyDestination]))
    let gain = try await device.createBuffer(.init(size: 8, usage: [.vertex, .copyDestination]))
    let vertices: [Float] = [
        99, -1, 1, 0.25, 1, 1, 0, 0,
        99, 2, 2, 0.5, 2, 0, 1, 0,
        99, -4, -4, 1, 4, 0, 0, 1,
    ]
    let clear = RenderClearColor(red: 0, green: 0, blue: 1, alpha: 1)
    let texture = try await device.createTexture(.init(width: 8, height: 8, format: .rgba8Unorm,
                                                       usage: [.renderAttachment, .copySource]))
    let byteCount: UInt64 = 1824
    let buffer = try await device.createBuffer(.init(size: byteCount, usage: [.copyDestination, .mapRead]))
    do {
        _ = try await device.submitCommands([.copyTextureToBuffer(source: texture, destination: buffer, bytesPerRow: 32)])
        throw GPUError.backendFailure("Swift accepted an invalid texture copy stride")
    } catch GPUError.validation {}
    do {
        _ = try await device.submitCommands([
            .render(pipeline: pipeline, target: texture, clear: clear, vertexCount: 3,
                    vertexBuffers: [.init(buffer: geometry, offset: 8), .init(buffer: gain)]),
        ])
        throw GPUError.backendFailure("Swift accepted an out-of-range vertex fetch")
    } catch GPUError.validation {}
    let ticket = try await device.submitCommands([
        .write(buffer: geometry, offset: 0, words: vertices.map(\.bitPattern)),
        .write(buffer: gain, offset: 0, words: [99, .max]),
        .copyTextureToBuffer(source: texture, destination: buffer, bytesPerRow: 256),
        .read(buffer: buffer, byteCount: byteCount),
        .render(pipeline: pipeline, target: texture, clear: clear, vertexCount: 3,
                vertexBuffers: [.init(buffer: geometry, offset: 4), .init(buffer: gain)]),
        .copyTextureToBuffer(source: texture, destination: buffer, bytesPerRow: 256),
        .read(buffer: buffer, byteCount: byteCount),
    ])
    try await device.destroyBuffer(geometry)
    try await device.destroyTexture(texture)
    let snapshots = try await ticket.wait()
    guard snapshots.count == 2, snapshots.allSatisfy({ $0.count == Int(byteCount / 4) }),
          snapshots[0].allSatisfy({ $0 == 0 }) else { throw GPUError.backendFailure("Unexpected WebGPU texture snapshots") }
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
    return (checked, (0..<3).map { (pixels[0] >> (8 * $0)) & 255 }, pipeline.vertexBuffers.count)
}
#endif
