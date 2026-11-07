import Testing
import SAGATypes
import SAGABackend
import SAGACore
@testable import SAGAMetal

#if canImport(Metal)
@Test func metalDiagnosticComputesEveryElementOnGPU() async throws {
    let backend = try MetalBackend.makeDevice()
    let device = try await Device.make(backend: backend)
    do {
        let pipeline = try await backend.makeDiagnosticPipeline(owner: device.identity)
        for count in [1, 63, 64, 65, 4099] {
            let input = (0..<count).map { UInt32(truncatingIfNeeded: $0 &* 104729) ^ UInt32.max }
            let size = UInt64(count * 4)
            let a = try await device.createBuffer(.init(size: size, usage: [.storage, .copyDestination]))
            let b = try await device.createBuffer(.init(size: size, usage: [.storage, .copySource]))
            let r = try await device.createBuffer(.init(size: size, usage: [.copyDestination, .mapRead]))
            let ticket = try await device.submitCommands([
                .write(buffer: a, offset: 0, words: input),
                .dispatch(pipeline: pipeline, input: a, output: b, count: UInt32(count)),
                .copy(source: b, destination: r, byteCount: size),
                .read(buffer: r, byteCount: size),
            ])
            let result = try await ticket.wait()
            #expect(result == [input.map { $0 &* 3 &+ 7 }])
            for id in [a, b, r] { try await device.destroyBuffer(id) }
        }
        await device.close()
    } catch { await device.close(); throw error }
}

#endif

#if canImport(Metal)
@Test func metalReadbacksAreZeroInitializedAndSnapshotOrdered() async throws {
    let backend = try MetalBackend.makeDevice()
    let device = try await Device.make(backend: backend)
    do {
        let a = try await device.createBuffer(.init(size: 16, usage: [.copyDestination, .copySource]))
        let r = try await device.createBuffer(.init(size: 16, usage: [.copyDestination, .mapRead]))
        let ticket = try await device.submitCommands([
            .read(buffer: r, byteCount: 16),
            .write(buffer: a, offset: 4, words: [42, 99]),
            .copy(source: a, destination: r, byteCount: 16),
            .read(buffer: r, byteCount: 16),
        ])
        #expect(try await ticket.wait() == [[0, 0, 0, 0], [0, 42, 99, 0]])
        await device.close()
    } catch { await device.close(); throw error }
}

@Test func metalRejectsUnknownPipelineAndQueueRemainsUsable() async throws {
    let backend = try MetalBackend.makeDevice()
    let device = try await Device.make(backend: backend)
    do {
        let a = try await device.createBuffer(.init(size: 4, usage: [.storage, .copySource]))
        let b = try await device.createBuffer(.init(size: 4, usage: [.storage, .copyDestination]))
        let invalid = try await device.submitCommands([
            .dispatch(pipeline: .init(device: device.identity, serial: 999), input: a, output: b, count: 1)
        ])
        await #expect(throws: GPUError.validation("Unknown Metal diagnostic pipeline")) { try await invalid.wait() }
        let valid = try await device.submitCommands([.copy(source: a, destination: b, byteCount: 4)])
        #expect(try await valid.wait().isEmpty)
        await device.close()
    } catch { await device.close(); throw error }
}
#endif

@Test func completionDeadlineAndLateCallbackResolveOnlyOnce() async throws {
    let completion = BackendCompletion()
    await #expect(throws: GPUError.timeout) { try await completion.wait(timeout: .zero) }
    await completion.finish(.success(()))
    await #expect(throws: GPUError.validation("Completion already has a consumer")) {
        try await completion.wait(timeout: .seconds(1))
    }
    let early = BackendCompletion()
    await early.finish(.success(()))
    try await early.wait(timeout: .zero)
    let failed = BackendCompletion()
    await failed.finish(.failure(.deviceLost("injected completion error")))
    await #expect(throws: GPUError.deviceLost("injected completion error")) {
        try await failed.wait(timeout: .seconds(1))
    }
}

@Test func completionTimerActuallyExpiresWithoutCallback() async {
    let completion = BackendCompletion()
    await #expect(throws: GPUError.timeout) { try await completion.wait(timeout: .milliseconds(1)) }
}

#if canImport(Metal)
@Test func metalTimeoutQuarantinesNativeResourcesThroughCoreCloseAndLateRetirement() async throws {
    let backend = try MetalBackend.makeDevice(timeout: .milliseconds(10))
    let callbackArrived = Gate()
    let releaseCallback = Gate()
    await backend.setRetirementHookForTesting {
        await callbackArrived.release()
        await releaseCallback.wait()
    }
    let device = try await Device.make(backend: backend)
    _ = try await backend.makeDiagnosticPipeline(owner: device.identity)
    let r = try await device.createBuffer(.init(size: 4, usage: [.mapRead, .copyDestination]))
    let ticket = try await device.submitCommands([.read(buffer: r, byteCount: 4)])
    await callbackArrived.wait()
    await #expect(throws: GPUError.timeout) { try await ticket.wait() }
    await device.close()
    let before = await backend.resourceCountsForTesting()
    #expect(before.buffers == 0 && before.pipelines == 0)
    #expect(before.batches == 1 && before.retainedBuffers == 2)
    await releaseCallback.release()
    await backend.waitForRetirementForTesting()
    let after = await backend.resourceCountsForTesting()
    #expect(after.batches == 0 && after.retainedBuffers == 0)
    await #expect(throws: GPUError.self) {
        try await backend.makeDiagnosticPipeline(owner: device.identity)
    }
}

@Test func repeatedCoreCloseReleasesPipelineHandlesOnRetainedBackend() async throws {
    let backend = try MetalBackend.makeDevice()
    for _ in 0..<10 {
        let device = try await Device.make(backend: backend)
        let first = try await backend.makeDiagnosticPipeline(owner: device.identity)
        #expect(try await backend.makeDiagnosticPipeline(owner: device.identity) == first)
        #expect(await backend.resourceCountsForTesting().pipelines == 1)
        await device.close()
        #expect(await backend.resourceCountsForTesting().pipelines == 0)
    }
}
#endif
