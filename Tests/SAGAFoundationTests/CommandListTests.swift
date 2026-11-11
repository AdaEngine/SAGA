import Testing
import SAGATypes
import SAGABackend
import SAGACore
import SAGACompiler
@testable import SAGAVulkan

private let device: UInt64 = 1
private func id(_ serial: UInt64) -> ResourceID { ResourceID(device: device, serial: serial) }
private func slot(_ binding: UInt32) -> ComputeBindingSlot { ComputeBindingSlot(binding: binding) }

private func pipeline(
    reads: [UInt32] = [0], writes: [UInt32] = [1], serial: UInt64 = 100, immediateBytes: UInt64 = 0, deviceID: UInt64 = device
) -> ComputePipeline {
    ComputePipeline(
        id: ResourceID(device: deviceID, serial: serial),
        storageBindings: reads.map { .init(slot: slot($0), minimumByteSize: 4, isReadOnly: true) }
            + writes.map { .init(slot: slot($0), minimumByteSize: 4, isReadOnly: false) },
        immediateByteSize: immediateBytes)
}

private let descriptors: [ResourceID: BufferDescriptor] = [
    id(1): .init(size: 1024, usage: [.storage, .copySource, .copyDestination]),
    id(2): .init(size: 1024, usage: [.storage, .copySource, .copyDestination]),
    id(3): .init(size: 1024, usage: [.copyDestination, .mapRead]),
    id(4): .init(size: 1024, usage: [.mapRead]),
    id(5): .init(size: 2048, usage: [.storage, .uniform]),
]

/// Runs the compact validator and the original command validator and requires the same outcome.
private func requireSameOutcome(_ command: BufferCommand, limit: UInt32 = 65_535, sourceLocation: SourceLocation = #_sourceLocation) {
    let list = CommandList([command])
    #expect(list.count == 1, sourceLocation: sourceLocation)
    let record = list.records[0]
    var original: GPUError?
    do { try command.validate(device: device, descriptors: descriptors, maxComputeWorkgroupsPerDimension: limit) }
    catch let error as GPUError { original = error } catch { Issue.record("unexpected error \(error)", sourceLocation: sourceLocation) }
    var compact: GPUError?
    if record.kind == .legacy {
        // Not compacted: the list defers to the command, so the outcome is the original one by construction.
        compact = original
    } else {
        do { try list.validate(record, device: device, descriptors: descriptors, maxComputeWorkgroupsPerDimension: limit) }
        catch let error as GPUError { compact = error } catch { Issue.record("unexpected error \(error)", sourceLocation: sourceLocation) }
    }
    #expect(compact == original, "list: \(String(describing: compact)), command: \(String(describing: original))", sourceLocation: sourceLocation)
}

@Test func commandListCompactsHotCommandsAndRoundTripsThem() {
    let p = pipeline()
    let commands: [BufferCommand] = [
        .write(buffer: id(1), offset: 4, words: [1, 2, 3]),
        .dispatchCompute(pipeline: p, bindings: [slot(0): id(1), slot(1): id(2)], workgroups: .init(x: 2, y: 3, z: 4)),
        .dispatchComputeRanges(pipeline: p, bindings: [slot(0): .init(buffer: id(5), offset: 256, size: 64), slot(1): .init(buffer: id(2))],
                               workgroups: .init(x: 1)),
        .copy(source: id(1), sourceOffset: 8, destination: id(2), destinationOffset: 4, byteCount: 16),
        .copy(source: id(1), destination: id(2)),
        .clearBuffer(buffer: id(1), offset: 4, byteCount: 8),
        .clearBuffer(buffer: id(1)),
        .read(buffer: id(3), byteCount: 64),
    ]
    let list = CommandList(commands)
    #expect(list.legacyCommands.isEmpty)
    #expect(list.records.allSatisfy { $0.kind != .legacy })
    #expect(list.expandedCommands() == commands)
    #expect(Set(list.resources) == Set(commands.flatMap(\.resources)))
}

@Test func commandListKeepsUncompactableCommandsAsGeneralCommands() {
    let textured = ComputePipeline(
        id: id(101), storageBindings: [],
        textureBindings: [.init(slot: slot(2), sampleType: .float)])
    let commands: [BufferCommand] = [
        .copy(source: id(1), destination: id(2), byteCount: .max),
        .writeBytes(buffer: id(1), offset: 0, bytes: [1, 2, 3, 4]),
        .dispatchCompute(pipeline: textured, bindings: [:], workgroups: .init(x: 1), textures: [slot(2): id(9)]),
    ]
    let list = CommandList(commands)
    #expect(list.records.map(\.kind) == [.legacy, .legacy, .legacy])
    #expect(list.expandedCommands() == commands)
}

@Test func commandListRecorderMatchesAppendedCommands() {
    let p = pipeline()
    var recorded = CommandList()
    recorded.write(buffer: id(1), offset: 0, words: [9, 9])
    recorded.dispatch(p, bindings: [.init(slot(0), id(1)), .init(slot(1), id(2), offset: 256, size: 32)],
                      workgroups: .init(x: 5), immediates: [])
    recorded.copy(from: id(1), to: id(2), byteCount: 8)
    recorded.read(buffer: id(3), byteCount: 8)
    recorded.clear(buffer: id(2))
    let appended = CommandList(recorded.expandedCommands())
    #expect(appended.expandedCommands() == recorded.expandedCommands())
    #expect(recorded.count == 5)
    recorded.removeAll()
    #expect(recorded.isEmpty && recorded.resources.isEmpty)
}

@Test func compactValidationMatchesCommandValidationForDispatches() {
    let p = pipeline()
    requireSameOutcome(.dispatchCompute(pipeline: p, bindings: [slot(0): id(1), slot(1): id(2)], workgroups: .init(x: 1)))
    // Missing binding, extra binding, wrong binding slot.
    requireSameOutcome(.dispatchCompute(pipeline: p, bindings: [slot(0): id(1)], workgroups: .init(x: 1)))
    requireSameOutcome(.dispatchCompute(pipeline: p, bindings: [slot(0): id(1), slot(1): id(2), slot(2): id(2)], workgroups: .init(x: 1)))
    requireSameOutcome(.dispatchCompute(pipeline: p, bindings: [slot(0): id(1), slot(7): id(2)], workgroups: .init(x: 1)))
    // Usage, unknown buffer, foreign buffer.
    requireSameOutcome(.dispatchCompute(pipeline: p, bindings: [slot(0): id(1), slot(1): id(3)], workgroups: .init(x: 1)))
    requireSameOutcome(.dispatchCompute(pipeline: p, bindings: [slot(0): id(1), slot(1): id(99)], workgroups: .init(x: 1)))
    requireSameOutcome(.dispatchCompute(pipeline: p, bindings: [slot(0): id(1), slot(1): ResourceID(device: 2, serial: 2)], workgroups: .init(x: 1)))
    // Aliasing: a writable binding overlapping a read, and the same buffer with different access modes.
    requireSameOutcome(.dispatchCompute(pipeline: p, bindings: [slot(0): id(1), slot(1): id(1)], workgroups: .init(x: 1)))
    requireSameOutcome(.dispatchComputeRanges(
        pipeline: p, bindings: [slot(0): .init(buffer: id(1), offset: 0, size: 256), slot(1): .init(buffer: id(1), offset: 256, size: 256)],
        workgroups: .init(x: 1)))
    requireSameOutcome(.dispatchComputeRanges(
        pipeline: p, bindings: [slot(0): .init(buffer: id(5), offset: 256), slot(1): .init(buffer: id(2), offset: 128)],
        workgroups: .init(x: 1)))
    requireSameOutcome(.dispatchComputeRanges(
        pipeline: p, bindings: [slot(0): .init(buffer: id(5), offset: 0, size: 4096), slot(1): .init(buffer: id(2))],
        workgroups: .init(x: 1)))
    // Workgroup limits and immediates and foreign pipelines.
    requireSameOutcome(.dispatchCompute(pipeline: p, bindings: [slot(0): id(1), slot(1): id(2)], workgroups: .init(x: 70_000)))
    requireSameOutcome(.dispatchCompute(pipeline: p, bindings: [slot(0): id(1), slot(1): id(2)], workgroups: .init(x: 5, y: 5)), limit: 4)
    requireSameOutcome(.dispatchCompute(pipeline: p, bindings: [slot(0): id(1), slot(1): id(2)], workgroups: .init(x: 1), immediates: [1]))
    let needsImmediate = pipeline(immediateBytes: 4)
    requireSameOutcome(.dispatchCompute(pipeline: needsImmediate, bindings: [slot(0): id(1), slot(1): id(2)], workgroups: .init(x: 1)))
    requireSameOutcome(.dispatchCompute(pipeline: needsImmediate, bindings: [slot(0): id(1), slot(1): id(2)], workgroups: .init(x: 1), immediates: [7]))
    let foreign = pipeline(deviceID: 2)
    requireSameOutcome(.dispatchCompute(pipeline: foreign, bindings: [slot(0): id(1), slot(1): id(2)], workgroups: .init(x: 1)))
    // Pipelines without bindings.
    requireSameOutcome(.dispatchCompute(pipeline: pipeline(reads: [], writes: []), bindings: [:], workgroups: .init(x: 1)))
    requireSameOutcome(.dispatchCompute(pipeline: pipeline(reads: [], writes: []), bindings: [slot(0): id(1)], workgroups: .init(x: 1)))
}

@Test func compactValidationMatchesCommandValidationForTransfers() {
    requireSameOutcome(.copy(source: id(1), destination: id(2), byteCount: 16))
    requireSameOutcome(.copy(source: id(1), destination: id(1), byteCount: 16))
    requireSameOutcome(.copy(source: id(1), sourceOffset: 2, destination: id(2), byteCount: 16))
    requireSameOutcome(.copy(source: id(1), sourceOffset: 1020, destination: id(2), byteCount: 16))
    requireSameOutcome(.copy(source: id(4), destination: id(2), byteCount: 16))
    requireSameOutcome(.copy(source: id(1), destination: id(4)))
    requireSameOutcome(.copy(source: id(1), destination: id(2), byteCount: 0))
    requireSameOutcome(.write(buffer: id(1), offset: 0, words: [1, 2]))
    requireSameOutcome(.write(buffer: id(1), offset: 1020, words: [1, 2]))
    requireSameOutcome(.write(buffer: id(1), offset: 2, words: [1]))
    requireSameOutcome(.write(buffer: id(4), offset: 0, words: [1]))
    requireSameOutcome(.write(buffer: id(1), offset: 0, words: []))
    requireSameOutcome(.write(buffer: id(99), offset: 0, words: [1]))
    requireSameOutcome(.clearBuffer(buffer: id(1), offset: 0, byteCount: 8))
    requireSameOutcome(.clearBuffer(buffer: id(1), offset: 4, byteCount: 6))
    requireSameOutcome(.clearBuffer(buffer: id(4)))
    requireSameOutcome(.read(buffer: id(3), byteCount: 16))
    requireSameOutcome(.read(buffer: id(3), byteCount: 0))
    requireSameOutcome(.read(buffer: id(1), byteCount: 16))
    requireSameOutcome(.read(buffer: id(3), byteCount: 2048))
}

private let vulkanAvailable = VulkanBackend.requestAdapter() != nil

@Test(.enabled(if: vulkanAvailable), .gpuExclusive) func commandListRunsOnVulkanLikeTheEquivalentCommands() async throws {
    let backend = try VulkanBackend.makeDevice()
    let device = try await Device.make(backend: backend)
    do {
        let shader = try IndependentShaderCompiler().compileCompute(.wgsl("""
            @group(0) @binding(0) var<storage, read> source: array<u32>;
            @group(0) @binding(1) var<storage, read_write> destination: array<u32>;
            @compute @workgroup_size(64)
            fn run(@builtin(global_invocation_id) id: vec3<u32>) {
                if (id.x < arrayLength(&destination)) { destination[id.x] = source[id.x] * 3u + 7u; }
            }
            """), to: .spirv)
        let pipeline = try await device.createComputePipeline(shader)
        let count = 1000
        let size = UInt64(count * 4)
        let input = try await device.createBuffer(.init(size: size, usage: [.storage, .copyDestination]))
        let output = try await device.createBuffer(.init(size: size, usage: [.storage, .copySource]))
        let copy = try await device.createBuffer(.init(size: size, usage: [.copyDestination, .copySource]))
        let readback = try await device.createBuffer(.init(size: size, usage: [.copyDestination, .mapRead]))
        let data = (0..<count).map { UInt32($0) }
        let groups = DispatchWorkgroups(x: UInt32((count + 63) / 64))
        let bindings: [CommandList.Binding] = [.init(ComputeBindingSlot(binding: 0), input), .init(ComputeBindingSlot(binding: 1), output)]

        var list = CommandList()
        list.write(buffer: input, words: data)
        list.dispatch(pipeline, bindings: bindings, workgroups: groups)
        list.dispatch(pipeline, bindings: bindings, workgroups: groups)  // repeated dispatch reuses the pipeline entry
        list.clear(buffer: copy)
        list.copy(from: output, to: copy, byteCount: size)
        list.copy(from: copy, to: readback)
        list.read(buffer: readback, byteCount: size)
        let fromList = try await device.submit(list).wait()

        let fromCommands = try await device.submitCommands([
            .write(buffer: input, offset: 0, words: data),
            .dispatchCompute(pipeline: pipeline, bindings: [ComputeBindingSlot(binding: 0): input, ComputeBindingSlot(binding: 1): output], workgroups: groups),
            .clearBuffer(buffer: copy),
            .copy(source: output, destination: copy, byteCount: size),
            .copy(source: copy, destination: readback),
            .read(buffer: readback, byteCount: size),
        ]).wait()
        #expect(fromList == [data.map { $0 &* 3 &+ 7 }])
        #expect(fromList == fromCommands)

        // The same list can be submitted again, and errors surface as GPUError without poisoning the queue.
        #expect(try await device.submit(list).wait() == fromList)
        var invalid = CommandList()
        invalid.dispatch(pipeline, bindings: [.init(ComputeBindingSlot(binding: 0), input)], workgroups: groups)
        await #expect(throws: GPUError.self) { try await device.submit(invalid).wait() }
        #expect(try await device.submit(list).wait() == fromList)
        await device.close()
    } catch { await device.close(); throw error }
}
