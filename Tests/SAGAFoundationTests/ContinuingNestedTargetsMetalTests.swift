import Testing
import SAGACompiler
import SAGACore
import SAGAMetal
import SAGATypes

#if canImport(Metal)
@Test func continuingNestedTargetsExecuteOnMetal() async throws {
    let source = """
        @group(0) @binding(0) var<storage, read_write> output: u32;
        @compute @workgroup_size(1) fn run() {
            var n = 0u;
            var total = 0u;
            loop {
                continuing {
                    switch n {
                        case 0u: { total += 2u; break; }
                        default: { total += 5u; }
                    }
                    var j = 0u;
                    loop {
                        j++;
                        if j < 2u { continue; }
                        total += j;
                        break;
                    }
                    n++;
                    break if n == 3u;
                }
            }
            output = total;
        }
        """
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(source), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let output = try await device.createBuffer(.init(size: 4, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 4, usage: [.mapRead, .copyDestination]))
        let result = try await device.submitCommands([
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
            .copy(source: output, destination: readback, byteCount: 4),
            .read(buffer: readback, byteCount: 4),
        ]).wait()
        #expect(result == [[18]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
