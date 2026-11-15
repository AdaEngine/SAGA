#if canImport(Metal)
import Metal
import SAGACompiler
import SAGACore
import SAGAMetal
import SAGATypes
import Testing

@Suite
struct MetalNativePassTests {
    @MainActor
    private final class SubmissionEvents {
        var values: [String] = []
    }

    @MainActor
    @Test(.gpuExclusive) func recordsInAnOrderedMetalSubmission() async throws {
        let device = try await Device.make(backend: MetalBackend.makeDevice())
        do {
            let buffer = try await device.createBuffer(.init(size: 16, usage: [.copySource, .copyDestination]))
            let result = try await device.createBuffer(.init(size: 16, usage: [.copyDestination, .mapRead]))
            let events = SubmissionEvents()
            let pass = NativePass.onMainActor(label: "Native marker", uses: [.init(buffer, .read)]) { context in
                let metal = try #require(context as? MetalNativePassContext)
                events.values.append("recorded")
                metal.addCommittedHandler { events.values.append("committed") }
            }
            let ticket = try await device.submitCommands([
                .write(buffer: buffer, offset: 0, words: [7]),
                .native(pass),
                .copy(source: buffer, destination: result, byteCount: 16),
                .read(buffer: result, byteCount: 16),
            ])
            #expect(try await ticket.wait() == [[7, 0, 0, 0]])
            #expect(events.values == ["recorded", "committed"])
        } catch {
            await device.close()
            throw error
        }
        await device.close()
    }

    @Test(.gpuExclusive) func preservesUntouchedPixelsInANativePartialWrite() async throws {
        let device = try await Device.make(backend: MetalBackend.makeDevice())
        do {
            let texture = try await device.createTexture(.init(width: 2, height: 1, format: .rgba8Unorm,
                usage: [.copySource, .copyDestination]))
            let upload = try await device.createBuffer(.init(size: 256, usage: [.copySource, .copyDestination]))
            let readback = try await device.createBuffer(.init(size: 256, usage: [.copyDestination, .mapRead]))
            let pass = NativePass(label: "Copy second pixel", uses: [.init(texture, .write), .init(upload, .read)]) { context in
                let metal = try #require(context as? MetalNativePassContext)
                let source = try metal.buffer(upload)
                let destination = try metal.texture(texture)
                let blit = try #require(metal.commandBuffer.makeBlitCommandEncoder())
                blit.copy(from: source, sourceOffset: 0, sourceBytesPerRow: 256, sourceBytesPerImage: 256,
                          sourceSize: MTLSize(width: 1, height: 1, depth: 1), to: destination,
                          destinationSlice: 0, destinationLevel: 0, destinationOrigin: MTLOrigin(x: 1, y: 0, z: 0))
                blit.endEncoding()
            }
            let ticket = try await device.submitCommands([
                .write(buffer: upload, offset: 0, words: [0x44332211]),
                .native(pass),
                .copyTextureToBuffer(source: texture, destination: readback, bytesPerRow: 256),
                .read(buffer: readback, byteCount: 8),
            ])
            #expect(try await ticket.wait() == [[0, 0x44332211]])
        } catch {
            await device.close()
            throw error
        }
        await device.close()
    }

    @Test(.gpuExclusive) func reportsUndeclaredHandlesAndLeavesTheQueueUsable() async throws {
        let device = try await Device.make(backend: MetalBackend.makeDevice())
        do {
            let target = try await device.createBuffer(.init(size: 16, usage: [.copySource, .copyDestination]))
            let other = try await device.createBuffer(.init(size: 16, usage: [.copySource]))
            let readback = try await device.createBuffer(.init(size: 16, usage: [.copyDestination, .mapRead]))
            _ = try await device.submitCommands([.write(buffer: target, offset: 0, words: [7])]).wait()
            let pass = NativePass(label: "Invalid buffer lookup", uses: [.init(target, .write)]) { context in
                let metal = try #require(context as? MetalNativePassContext)
                _ = try metal.buffer(other)
            }
            let ticket = try await device.submitCommands([
                .write(buffer: target, offset: 0, words: [99]), .native(pass),
            ])
            await #expect(throws: GPUError.validation("The native pass did not name this buffer")) {
                _ = try await ticket.wait()
            }
            let next = try await device.submitCommands([
                .copy(source: target, destination: readback, byteCount: 16), .read(buffer: readback, byteCount: 16),
            ])
            #expect(try await next.wait() == [[7, 0, 0, 0]])
        } catch {
            await device.close()
            throw error
        }
        await device.close()
    }

    @Test(.enabled(if: try MetalBackend.makeDevice().capabilities.features.contains(.timestampQuery)), .gpuExclusive)
    func recordingFailureDiscardsCommandsBeforeAQuerySplit() async throws {
        let device = try await Device.make(backend: MetalBackend.makeDevice(), requiredFeatures: [.timestampQuery])
        do {
            let buffer = try await device.createBuffer(.init(size: 16, usage: [.copySource, .copyDestination]))
            let readback = try await device.createBuffer(.init(size: 16, usage: [.copyDestination, .mapRead]))
            let queryResult = try await device.createBuffer(.init(size: 256, usage: [.queryResolve, .copySource]))
            let queryReadback = try await device.createBuffer(.init(size: 8, usage: [.copyDestination, .mapRead]))
            let query = try await device.createQuerySet(.init(type: .timestamp, count: 1))
            let shader = try IndependentShaderCompiler().compileCompute(.wgsl("@compute @workgroup_size(1) fn run() {}"), to: .msl)
            let pipeline = try await device.createComputePipeline(shader)
            let timed = BufferCommand.timestamped(
                pass: .dispatchCompute(pipeline: pipeline, bindings: [:], workgroups: .init(x: 1)),
                writes: .init(querySet: query, beginningOfPassWriteIndex: 0)
            )
            let failure = GPUError.validation("Native pass could not encode")
            let pass = NativePass(label: "Fail recording", uses: []) { _ in throw failure }
            let marker = NativePass(label: "Record before split", uses: []) { context in
                let metal = try #require(context as? MetalNativePassContext)
                metal.addCommittedHandler { Issue.record("A failed submission committed an earlier native pass") }
            }
            _ = try await device.submitCommands([.write(buffer: buffer, offset: 0, words: [7])]).wait()
            let ticket = try await device.submitCommands([
                .write(buffer: buffer, offset: 0, words: [99]),
                .native(marker),
                timed,
                .resolveQuerySet(querySet: query, firstQuery: 0, queryCount: 1, destination: queryResult),
                .native(pass),
            ])
            await #expect(throws: failure) { _ = try await ticket.wait() }
            let next = try await device.submitCommands([
                .copy(source: buffer, destination: readback, byteCount: 16),
                .read(buffer: readback, byteCount: 16),
                .resolveQuerySet(querySet: query, firstQuery: 0, queryCount: 1, destination: queryResult),
                .copy(source: queryResult, destination: queryReadback, byteCount: 8),
                .read(buffer: queryReadback, byteCount: 8),
            ])
            #expect(try await next.wait() == [[7, 0, 0, 0], [0, 0]])
        } catch {
            await device.close()
            throw error
        }
        await device.close()
    }
}
#endif
