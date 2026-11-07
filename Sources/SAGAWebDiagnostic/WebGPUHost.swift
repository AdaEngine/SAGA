#if os(WASI) && compiler(>=6.3)
import SAGATypes
import SAGABackend
import SAGAShaderIR
import SAGAWebGPU

private struct Packet {
    var bytes: [UInt8] = []
    init(_ opcode: UInt32) { add(opcode) }
    mutating func add<T: FixedWidthInteger>(_ value: T) {
        withUnsafeBytes(of: value.littleEndian) { bytes.append(contentsOf: $0) }
    }
    mutating func add(_ id: ResourceID) { add(id.device); add(id.serial) }
    mutating func add(_ text: String) {
        add(UInt32(text.utf8.count))
        bytes.append(contentsOf: text.utf8)
    }
    func send() async throws -> [UInt8] { try await BrowserRPC.shared.send(bytes) }
}

struct WebGPUHost: BrowserGPURenderHost {
    let capabilities: DeviceCapabilities

    func allocateBuffer(id: ResourceID, descriptor: BufferDescriptor) async throws {
        var packet = Packet(1)
        packet.add(id); packet.add(descriptor.size); packet.add(descriptor.usage.rawValue)
        _ = try await packet.send()
    }
    func destroyBuffer(id: ResourceID) async {
        var packet = Packet(2)
        packet.add(id)
        _ = try? await packet.send()
    }
    func allocateComputePipeline(_ pipeline: ComputePipeline, shader: ComputeShaderArtifact) async throws {
        guard case .text(.wgsl, let source) = shader.artifact else { throw GPUError.validation("Expected WGSL") }
        var packet = Packet(3)
        packet.add(pipeline.id); packet.add(shader.entryPoint); packet.add(source)
        packet.add(UInt32(shader.storageBindings.count))
        for binding in shader.storageBindings {
            packet.add(binding.binding.group); packet.add(binding.binding.binding)
            packet.add(binding.minimumByteSize); packet.add(UInt32(binding.isReadOnly ? 1 : 0))
        }
        _ = try await packet.send()
    }
    func destroyComputePipeline(id: ResourceID) async {
        var packet = Packet(4)
        packet.add(id)
        _ = try? await packet.send()
    }
    func allocateTexture(id: ResourceID, descriptor: TextureDescriptor) async throws {
        var packet = Packet(8)
        packet.add(id); packet.add(descriptor.width); packet.add(descriptor.height)
        packet.add(descriptor.format.rawValue); packet.add(descriptor.usage.rawValue)
        _ = try await packet.send()
    }
    func destroyTexture(id: ResourceID) async {
        var packet = Packet(9)
        packet.add(id)
        _ = try? await packet.send()
    }
    func allocateRenderPipeline(
        _ pipeline: RenderPipeline, vertex: RenderStageArtifact, fragment: RenderStageArtifact
    ) async throws {
        guard case .text(.wgsl, let vertexSource) = vertex.artifact,
              case .text(.wgsl, let fragmentSource) = fragment.artifact else {
            throw GPUError.validation("Expected WGSL render stages")
        }
        var packet = Packet(10)
        packet.add(pipeline.id); packet.add(pipeline.colorFormat.rawValue)
        packet.add(vertex.entryPoint); packet.add(vertexSource)
        packet.add(fragment.entryPoint); packet.add(fragmentSource)
        packet.add(UInt32(pipeline.vertexBuffers.count))
        for layout in pipeline.vertexBuffers {
            packet.add(layout.stride)
            packet.add(UInt32(layout.attributes.count))
            for attribute in layout.attributes {
                packet.add(attribute.location); packet.add(attribute.offset); packet.add(attribute.format.rawValue)
            }
        }
        _ = try await packet.send()
    }
    func destroyRenderPipeline(id: ResourceID) async {
        var packet = Packet(11)
        packet.add(id)
        _ = try? await packet.send()
    }
    func close(owner: UInt64) async {
        var packet = Packet(5)
        packet.add(owner)
        _ = try? await packet.send()
    }
    func submit(_ submission: Submission) async throws {
        _ = try await Packet(7).send()
    }
    func submitCommands(_ submission: Submission, commands: [BufferCommand]) async throws -> [[UInt32]] {
        var packet = Packet(6)
        packet.add(UInt32(commands.count))
        for command in commands {
            switch command {
            case .write(let id, let offset, let words):
                let headerSize = 32 // opcode, resource identity, offset and word count
                guard words.count <= (BrowserRPC.packetLimit - packet.bytes.count - headerSize) / 4 else {
                    throw GPUError.validation("Browser upload exceeds its packet limit")
                }
                packet.add(UInt32(1)); packet.add(id); packet.add(offset)
                packet.add(UInt32(words.count))
                for word in words { packet.add(word) }
            case .dispatchCompute(let pipeline, let bindings, let groups):
                packet.add(UInt32(2)); packet.add(pipeline.id)
                packet.add(groups.x); packet.add(groups.y); packet.add(groups.z)
                packet.add(UInt32(bindings.count))
                for (slot, id) in bindings {
                    packet.add(slot.group); packet.add(slot.binding); packet.add(id)
                }
            case .copy(let source, let destination, let size):
                packet.add(UInt32(3)); packet.add(source); packet.add(destination); packet.add(size)
            case .read(let id, let size):
                packet.add(UInt32(4)); packet.add(id); packet.add(size)
            case .dispatch:
                throw GPUError.unsupported("Metal diagnostic dispatch is unavailable on WebGPU")
            case .render(let pipeline, let target, let clear, let vertexCount, let vertexBuffers):
                packet.add(UInt32(5)); packet.add(pipeline.id); packet.add(target); packet.add(vertexCount)
                for component in [clear.red, clear.green, clear.blue, clear.alpha] {
                    packet.add(component.bitPattern)
                }
                packet.add(UInt32(vertexBuffers.count))
                for binding in vertexBuffers {
                    packet.add(binding.buffer); packet.add(binding.offset)
                }
            case .copyTextureToBuffer(let source, let destination, let bytesPerRow):
                packet.add(UInt32(6)); packet.add(source); packet.add(destination); packet.add(bytesPerRow)
            }
            guard packet.bytes.count <= BrowserRPC.packetLimit else {
                throw GPUError.validation("Browser submission packet exceeds its limit")
            }
        }
        let result = try await packet.send()
        var offset = 0
        func word() throws -> UInt32 {
            guard offset + 4 <= result.count else { throw GPUError.backendFailure("Truncated browser readback") }
            defer { offset += 4 }
            return result.withUnsafeBytes {
                UInt32(littleEndian: $0.loadUnaligned(fromByteOffset: offset, as: UInt32.self))
            }
        }
        let count = try word()
        var reads: [[UInt32]] = []
        for _ in 0..<count {
            let length = try word()
            guard length <= (result.count - offset) / 4 else {
                throw GPUError.backendFailure("Invalid browser readback length")
            }
            reads.append(try (0..<length).map { _ in try word() })
        }
        guard offset == result.count else { throw GPUError.backendFailure("Trailing browser readback bytes") }
        return reads
    }
}
#endif
