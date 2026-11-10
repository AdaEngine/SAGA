#if os(WASI) && compiler(>=6.3)
import SAGATypes
import SAGACore
import SAGACompiler

// Check host-visible byte offsets, not just values read back through the same shader layout.
func explicitLayoutDiagnostic(_ device: Device) async throws {
    let source = """
        struct Entry {
            @align(16) first: u32,
            @size(16) second: u32,
        }
        struct Payload {
            head: u32,
            @align(16) entries: array<Entry, 2>,
            tail: u32,
            nested: array<array<u32, 2>, 2>,
        }
        @group(0) @binding(0) var<storage, read_write> output: Payload;
        @compute @workgroup_size(1) fn run() {
            output.head = 11u;
            output.entries[0].first = 21u;
            output.entries[0].second = 22u;
            output.entries[1].first = 31u;
            output.entries[1].second = 32u;
            output.tail = 41u;
            output.nested[0][0] = 51u;
            output.nested[0][1] = 52u;
            output.nested[1][0] = 61u;
            output.nested[1][1] = 62u;
        }
        """
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(source), to: .wgsl)
    let pipeline = try await device.createComputePipeline(shader)
    let output = try await device.createBuffer(.init(size: 112, usage: [.storage, .copySource]))
    let readback = try await device.createBuffer(.init(size: 112, usage: [.copyDestination, .mapRead]))
    let ticket = try await device.submitCommands([
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
        .copy(source: output, destination: readback, byteCount: 112),
        .read(buffer: readback, byteCount: 112),
    ])
    let actual = try await ticket.wait()
    let expected: [UInt32] = [
        11, 0, 0, 0, 21, 22, 0, 0, 0, 0, 0, 0, 31, 32, 0, 0,
        0, 0, 0, 0, 41, 51, 52, 61, 62, 0, 0, 0,
    ]
    guard actual == [expected] else {
        throw GPUError.backendFailure("Unexpected explicit layout WebGPU readback: \(actual)")
    }
}
#endif
