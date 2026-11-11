/// A recorded sequence of commands stored as compact, trivially copyable records.
///
/// `BufferCommand` is convenient but heavy: every case carries dictionaries and arrays, so walking it costs
/// allocations and reference counting. The hot commands (buffer-only compute dispatches, copies, writes, reads
/// and clears) are stored here as plain records that point into shared arenas. Every other command is kept
/// as-is and replayed through the existing paths, so a list can hold any `BufferCommand`.
///
/// A list is a value: record it once, submit it any number of times.
public struct CommandList: Sendable {
    /// One buffer binding of a compact dispatch.
    public struct Binding: Sendable, Equatable {
        public let slot: ComputeBindingSlot
        public let buffer: ResourceID
        /// Byte offset, aligned to 256 bytes.
        public let offset: UInt64
        /// Byte count, or the rest of the buffer when omitted.
        public let size: UInt64?

        public init(_ slot: ComputeBindingSlot, _ buffer: ResourceID, offset: UInt64 = 0, size: UInt64? = nil) {
            self.slot = slot
            self.buffer = buffer
            self.offset = offset
            self.size = size
        }
    }

    package enum Kind: UInt8, Sendable { case dispatch, copy, write, read, clear, legacy }

    /// Field use by kind:
    /// - dispatch: `index` pipeline table index, `start`/`count` binding range, `x`/`y`/`z` workgroups,
    ///   `offsetA`/`size` immediate-word range.
    /// - copy: `a` source, `b` destination, `offsetA`/`offsetB` offsets, `size` byte count (`.max` means the rest).
    /// - write: `a` buffer, `offsetA` byte offset, `index` into `writes`, `count` words.
    /// - read: `a` buffer, `size` byte count.
    /// - clear: `a` buffer, `offsetA` byte offset, `size` byte count (`.max` means the rest).
    /// - legacy: `index` into `legacy`.
    package struct Record: Sendable {
        package var kind: Kind
        package var index: UInt32 = 0
        package var start: UInt32 = 0
        package var count: UInt32 = 0
        package var x: UInt32 = 0
        package var y: UInt32 = 0
        package var z: UInt32 = 0
        package var a = ResourceID(device: 0, serial: 0)
        package var b = ResourceID(device: 0, serial: 0)
        package var offsetA: UInt64 = 0
        package var offsetB: UInt64 = 0
        package var size: UInt64 = 0
    }

    package private(set) var records: [Record] = []
    package private(set) var pipelines: [ComputePipeline] = []
    package private(set) var bindings: [Binding] = []
    package private(set) var immediates: [UInt32] = []
    /// Word arrays of write commands, kept by reference so recording a large upload copies nothing.
    package private(set) var writes: [[UInt32]] = []
    package private(set) var legacy: [BufferCommand] = []
    /// Every resource the commands touch, in order. Duplicates are expected.
    package private(set) var resources: [ResourceID] = []

    public init() {}

    /// Records `commands`, compacting the ones that have a compact form.
    public init(_ commands: [BufferCommand]) {
        records.reserveCapacity(commands.count)
        for command in commands { append(command) }
    }

    public var count: Int { records.count }
    public var isEmpty: Bool { records.isEmpty }

    public mutating func removeAll(keepingCapacity keep: Bool = true) {
        records.removeAll(keepingCapacity: keep)
        pipelines.removeAll(keepingCapacity: keep)
        bindings.removeAll(keepingCapacity: keep)
        immediates.removeAll(keepingCapacity: keep)
        writes.removeAll(keepingCapacity: keep)
        legacy.removeAll(keepingCapacity: keep)
        resources.removeAll(keepingCapacity: keep)
    }

    // MARK: Recording

    /// Records a compute dispatch whose pipeline binds only storage or uniform buffers. Other pipelines are
    /// stored as a general command and validated the same way.
    public mutating func dispatch(
        _ pipeline: ComputePipeline, bindings suppliedBindings: [Binding] = [], workgroups: DispatchWorkgroups,
        immediates suppliedImmediates: [UInt32] = []
    ) {
        guard pipeline.hasOnlyBufferBindings else {
            append(Self.dispatchCommand(pipeline, suppliedBindings, workgroups, suppliedImmediates))
            return
        }
        appendDispatch(pipeline, suppliedBindings, workgroups, suppliedImmediates)
    }

    public mutating func copy(
        from source: ResourceID, sourceOffset: UInt64 = 0, to destination: ResourceID, destinationOffset: UInt64 = 0,
        byteCount: UInt64? = nil
    ) {
        if byteCount == .max { appendGeneral(.copy(source: source, sourceOffset: sourceOffset, destination: destination, destinationOffset: destinationOffset, byteCount: byteCount)); return }
        var record = Record(kind: .copy)
        record.a = source
        record.b = destination
        record.offsetA = sourceOffset
        record.offsetB = destinationOffset
        record.size = byteCount ?? .max
        records.append(record)
        resources.append(source)
        resources.append(destination)
    }

    public mutating func write(buffer: ResourceID, offset: UInt64 = 0, words newWords: [UInt32]) {
        var record = Record(kind: .write)
        record.a = buffer
        record.offsetA = offset
        record.index = UInt32(writes.count)
        record.count = UInt32(newWords.count)
        writes.append(newWords)
        records.append(record)
        resources.append(buffer)
    }

    public mutating func read(buffer: ResourceID, byteCount: UInt64) {
        var record = Record(kind: .read)
        record.a = buffer
        record.size = byteCount
        records.append(record)
        resources.append(buffer)
    }

    public mutating func clear(buffer: ResourceID, offset: UInt64 = 0, byteCount: UInt64? = nil) {
        if byteCount == .max { appendGeneral(.clearBuffer(buffer: buffer, offset: offset, byteCount: byteCount)); return }
        var record = Record(kind: .clear)
        record.a = buffer
        record.offsetA = offset
        record.size = byteCount ?? .max
        records.append(record)
        resources.append(buffer)
    }

    /// Records any command, in compact form when there is one.
    public mutating func append(_ command: BufferCommand) {
        switch command {
        case .dispatchCompute(let pipeline, let buffers, let groups, let textures, let samplers, let immediates, let structures)
            where textures.isEmpty && samplers.isEmpty && structures.isEmpty && pipeline.hasOnlyBufferBindings:
            appendDispatch(pipeline, buffers.map { Binding($0.key, $0.value) }, groups, immediates)
        case .dispatchComputeRanges(let pipeline, let ranged, let groups, let textures, let samplers, let immediates, let structures)
            where textures.isEmpty && samplers.isEmpty && structures.isEmpty && pipeline.hasOnlyBufferBindings:
            appendDispatch(pipeline, ranged.map { Binding($0.key, $0.value.buffer, offset: $0.value.offset, size: $0.value.size) },
                           groups, immediates)
        case .copy(let source, let sourceOffset, let destination, let destinationOffset, let byteCount):
            copy(from: source, sourceOffset: sourceOffset, to: destination, destinationOffset: destinationOffset, byteCount: byteCount)
        case .write(let buffer, let offset, let words):
            write(buffer: buffer, offset: offset, words: words)
        case .read(let buffer, let byteCount):
            read(buffer: buffer, byteCount: byteCount)
        case .clearBuffer(let buffer, let offset, let byteCount):
            clear(buffer: buffer, offset: offset, byteCount: byteCount)
        default:
            appendGeneral(command)
        }
    }

    private mutating func appendGeneral(_ command: BufferCommand) {
        var record = Record(kind: .legacy)
        record.index = UInt32(legacy.count)
        legacy.append(command)
        records.append(record)
        resources.append(contentsOf: command.resources)
    }

    private mutating func appendDispatch(
        _ pipeline: ComputePipeline, _ suppliedBindings: [Binding], _ groups: DispatchWorkgroups, _ suppliedImmediates: [UInt32]
    ) {
        var record = Record(kind: .dispatch)
        if let last = pipelines.last, last == pipeline {
            record.index = UInt32(pipelines.count - 1)
        } else {
            record.index = UInt32(pipelines.count)
            pipelines.append(pipeline)
        }
        record.start = UInt32(bindings.count)
        record.count = UInt32(suppliedBindings.count)
        record.x = groups.x
        record.y = groups.y
        record.z = groups.z
        record.offsetA = UInt64(immediates.count)
        record.size = UInt64(suppliedImmediates.count)
        if !suppliedBindings.isEmpty {
            bindings.append(contentsOf: suppliedBindings)
            for binding in suppliedBindings { resources.append(binding.buffer) }
        }
        if !suppliedImmediates.isEmpty { immediates.append(contentsOf: suppliedImmediates) }
        records.append(record)
    }

    private static func dispatchCommand(
        _ pipeline: ComputePipeline, _ bindings: [Binding], _ groups: DispatchWorkgroups, _ immediates: [UInt32]
    ) -> BufferCommand {
        if bindings.allSatisfy({ $0.offset == 0 && $0.size == nil }) {
            var dictionary: [ComputeBindingSlot: ResourceID] = [:]
            for binding in bindings { dictionary[binding.slot] = binding.buffer }
            return .dispatchCompute(pipeline: pipeline, bindings: dictionary, workgroups: groups, immediates: immediates)
        }
        var dictionary: [ComputeBindingSlot: StorageBufferBinding] = [:]
        for binding in bindings {
            dictionary[binding.slot] = StorageBufferBinding(buffer: binding.buffer, offset: binding.offset, size: binding.size)
        }
        return .dispatchComputeRanges(pipeline: pipeline, bindings: dictionary, workgroups: groups, immediates: immediates)
    }

    // MARK: Reading

    /// The commands this list stands for, for backends that have no native list path.
    package func expandedCommands() -> [BufferCommand] {
        var commands: [BufferCommand] = []
        commands.reserveCapacity(records.count)
        for record in records { commands.append(command(for: record)) }
        return commands
    }

    package func command(for record: Record) -> BufferCommand {
        switch record.kind {
        case .legacy:
            return legacy[Int(record.index)]
        case .dispatch:
            let slice = Array(bindings[Int(record.start)..<Int(record.start + record.count)])
            let words = Array(immediates[Int(record.offsetA)..<Int(record.offsetA + record.size)])
            return Self.dispatchCommand(pipelines[Int(record.index)], slice,
                                        DispatchWorkgroups(x: record.x, y: record.y, z: record.z), words)
        case .copy:
            return .copy(source: record.a, sourceOffset: record.offsetA, destination: record.b,
                         destinationOffset: record.offsetB, byteCount: record.size == .max ? nil : record.size)
        case .write:
            return .write(buffer: record.a, offset: record.offsetA,
                          words: writes[Int(record.index)])
        case .read:
            return .read(buffer: record.a, byteCount: record.size)
        case .clear:
            return .clearBuffer(buffer: record.a, offset: record.offsetA, byteCount: record.size == .max ? nil : record.size)
        }
    }

    /// Commands that were not compacted, in recording order.
    package var legacyCommands: [BufferCommand] { legacy }

    /// Device of the first pipeline a compact dispatch uses, for lists that touch no resource.
    package var firstPipelineDevice: UInt64? { pipelines.first?.id.device }

    // MARK: Validation

    /// Validates a compact record exactly as `BufferCommand.validate` validates the command it stands for.
    /// Legacy records are validated through their command instead.
    package func validate(
        _ record: Record, device: UInt64, descriptors: [ResourceID: BufferDescriptor], maxComputeWorkgroupsPerDimension: UInt32
    ) throws {
        @discardableResult
        func require(
            _ id: ResourceID, _ usage: BufferUsage, _ size: UInt64?, offset: UInt64 = 0, allowEmpty: Bool = false
        ) throws -> UInt64 {
            guard id.device == device, let descriptor = descriptors[id], descriptor.usage.contains(usage),
                  offset % 4 == 0, offset <= descriptor.size else {
                throw GPUError.validation("Command has invalid resource, usage, size or alignment")
            }
            let count = size ?? (descriptor.size - offset)
            guard (allowEmpty || count > 0), count % 4 == 0, count <= descriptor.size - offset else {
                throw GPUError.validation("Command has invalid resource, usage, size or alignment")
            }
            return count
        }
        switch record.kind {
        case .legacy:
            break
        case .write:
            try require(record.a, .copyDestination, UInt64(record.count) * 4, offset: record.offsetA, allowEmpty: true)
        case .clear:
            try require(record.a, .copyDestination, record.size == .max ? nil : record.size, offset: record.offsetA, allowEmpty: true)
        case .read:
            try require(record.a, .mapRead, record.size)
        case .copy:
            guard record.a != record.b else { throw GPUError.validation("Aliased copy") }
            let count = try require(record.a, .copySource, record.size == .max ? nil : record.size,
                                    offset: record.offsetA, allowEmpty: true)
            try require(record.b, .copyDestination, count, offset: record.offsetB, allowEmpty: true)
        case .dispatch:
            let pipeline = pipelines[Int(record.index)]
            guard pipeline.id.device == device else { throw GPUError.validation("Foreign compute pipeline") }
            guard record.size == pipeline.immediateByteSize / 4 else {
                throw GPUError.validation("Compute immediate data size does not match the pipeline layout")
            }
            let groups = DispatchWorkgroups(x: record.x, y: record.y, z: record.z)
            guard groups.x <= maxComputeWorkgroupsPerDimension, groups.y <= maxComputeWorkgroupsPerDimension,
                  groups.z <= maxComputeWorkgroupsPerDimension else {
                throw GPUError.validation("Workgroup count exceeds the supported limit")
            }
            if let limit = pipeline.maximumLinearWorkgroupCount, !groups.isEmpty, groups.z > limit / groups.x / groups.y {
                throw GPUError.validation("Linear indexing dispatch exceeds u32")
            }
            guard Int(record.count) == pipeline.bufferBindings.count else {
                throw GPUError.validation("Compute bindings do not match the pipeline layout")
            }
            guard !pipeline.bufferBindings.isEmpty else { return }
            let supplied = bindings[Int(record.start)..<Int(record.start + record.count)]
            var used: [(buffer: ResourceID, range: Range<UInt64>, readOnly: Bool)] = []
            used.reserveCapacity(supplied.count)
            for requirement in pipeline.bufferBindings {
                guard let binding = supplied.first(where: { $0.slot == requirement.slot }) else {
                    throw GPUError.validation("Compute bindings do not match the pipeline layout")
                }
                guard binding.buffer.device == device, let descriptor = descriptors[binding.buffer] else {
                    throw GPUError.validation("Command has invalid resource, usage, size or alignment")
                }
                let range = try StorageBufferBinding(buffer: binding.buffer, offset: binding.offset, size: binding.size)
                    .range(in: descriptor, usage: requirement.usage)
                guard range.upperBound - range.lowerBound >= requirement.minimumByteSize else {
                    throw GPUError.validation("Command has invalid resource, usage, size or alignment")
                }
                for previous in used where previous.buffer == binding.buffer {
                    if (!previous.readOnly || !requirement.isReadOnly) && previous.range.overlaps(range) {
                        throw GPUError.validation("Aliased writable compute buffers")
                    }
                    guard previous.readOnly == requirement.isReadOnly else {
                        throw GPUError.validation("Incompatible storage access modes for the same buffer")
                    }
                }
                used.append((binding.buffer, range, requirement.isReadOnly))
            }
        }
    }
}

extension ComputePipeline {
    /// True when the pipeline binds nothing but storage and uniform buffers.
    package var hasOnlyBufferBindings: Bool {
        textureBindings.isEmpty && storageTextureBindings.isEmpty && samplerBindings.isEmpty
            && externalTextureBindings.isEmpty && accelerationStructureBindings.isEmpty
    }
}

extension CommandList {
    /// Calls `body` with every resource a compact record touches, in the order `BufferCommand.resources` lists them.
    package func forEachResource(of record: Record, _ body: (ResourceID) throws -> Void) rethrows {
        switch record.kind {
        case .dispatch:
            for binding in bindings[Int(record.start)..<Int(record.start + record.count)] { try body(binding.buffer) }
        case .copy:
            try body(record.a)
            try body(record.b)
        case .write, .read, .clear:
            try body(record.a)
        case .legacy:
            for id in legacy[Int(record.index)].resources { try body(id) }
        }
    }
}
