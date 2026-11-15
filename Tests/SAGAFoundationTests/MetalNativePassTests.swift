#if canImport(Metal)
import Testing
import SAGACompiler
import SAGACore
import SAGAMetal
import SAGATypes

@Test(.gpuExclusive) func metalNativePassRejectsTicketAndQueueRemainsUsable() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let buffer = try await device.createBuffer(.init(size: 16, usage: [.copyDestination, .copySource]))
        let readback = try await device.createBuffer(.init(size: 16, usage: [.copyDestination, .mapRead]))
        let retired = try await device.createBuffer(.init(size: 16, usage: .copyDestination))
        _ = try await device.submitCommands([.write(buffer: buffer, offset: 0, words: [7])]).wait()
        let pass = NativePass(label: "unsupported", uses: [.init(retired, .write)]) { _ in
            Issue.record("Metal must not execute a native pass")
        }
        let ticket = try await device.submitCommands([
            .write(buffer: buffer, offset: 0, words: [99]), .native(pass),
        ])
        try await device.destroyBuffer(retired)
        await #expect(throws: GPUError.unsupported("Metal does not support native passes")) {
            _ = try await ticket.wait()
        }
        await #expect(throws: GPUError.self) {
            _ = try await device.submitCommands([.write(buffer: retired, offset: 0, words: [1])])
        }
        let next = try await device.submitCommands([
            .copy(source: buffer, destination: readback, byteCount: 16), .read(buffer: readback, byteCount: 16),
        ])
        #expect(try await next.wait() == [[7, 0, 0, 0]])
        try await device.destroyBuffer(buffer)
        try await device.destroyBuffer(readback)
        await device.close()
    } catch {
        await device.close()
        throw error
    }
}

@Test(.enabled(if: try MetalBackend.makeDevice().capabilities.features.contains(.timestampQuery), "Metal adapter does not advertise timestampQuery"), .gpuExclusive)
func metalNativePassRejectsAfterQuerySplit() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice(), requiredFeatures: [.timestampQuery])
    do {
        let buffer = try await device.createBuffer(.init(size: 16, usage: [.copyDestination, .copySource]))
        let readback = try await device.createBuffer(.init(size: 16, usage: [.copyDestination, .mapRead]))
        let result = try await device.createBuffer(.init(size: 256, usage: [.queryResolve, .copySource]))
        let query = try await device.createQuerySet(.init(type: .timestamp, count: 1))
        let shader = try IndependentShaderCompiler().compileCompute(.wgsl("@compute @workgroup_size(1) fn run() {}"), to: .msl)
        let pipeline = try await device.createComputePipeline(shader)
        let pass = NativePass(label: "unsupported", uses: [.init(buffer, .write)]) { _ in
            Issue.record("Metal must not execute a native pass")
        }
        _ = try await device.submitCommands([.write(buffer: buffer, offset: 0, words: [7])]).wait()
        let timed = BufferCommand.timestamped(
            pass: .dispatchCompute(pipeline: pipeline, bindings: [:], workgroups: .init(x: 1)),
            writes: .init(querySet: query, beginningOfPassWriteIndex: 0)
        )
        let ticket = try await device.submitCommands([
            .write(buffer: buffer, offset: 0, words: [99]), timed,
            .resolveQuerySet(querySet: query, firstQuery: 0, queryCount: 1, destination: result), .native(pass),
        ])
        await #expect(throws: GPUError.unsupported("Metal does not support native passes")) {
            _ = try await ticket.wait()
        }
        let queryReadback = try await device.createBuffer(.init(size: 8, usage: [.copyDestination, .mapRead]))
        let next = try await device.submitCommands([
            .copy(source: buffer, destination: readback, byteCount: 16), .read(buffer: readback, byteCount: 16),
            .resolveQuerySet(querySet: query, firstQuery: 0, queryCount: 1, destination: result),
            .copy(source: result, destination: queryReadback, byteCount: 8), .read(buffer: queryReadback, byteCount: 8),
        ])
        #expect(try await next.wait() == [[7, 0, 0, 0], [0, 0]])
        try await device.destroyBuffer(queryReadback)
        try await device.destroyBuffer(buffer)
        try await device.destroyBuffer(readback)
        try await device.destroyBuffer(result)
        try await device.destroyQuerySet(query)
        await device.close()
    } catch {
        await device.close()
        throw error
    }
}
#endif
