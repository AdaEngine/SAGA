import Testing
import SAGABackend
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private struct StorageFormatCase: Sendable {
    let format: IRStorageFormat
    let wgsl: String
    let vector: String
    let hlsl: String
    let glsl: String
    let packed: [UInt32]
    let lanes: [UInt32]
    var readWrite: Bool { [.r32Uint, .r32Sint, .r32Float].contains(format) }

    init(_ format: IRStorageFormat, _ wgsl: String, _ vector: String, _ hlsl: String, _ glsl: String,
         _ packed: [UInt32], _ lanes: [UInt32]) {
        self.format = format
        self.wgsl = wgsl
        self.vector = vector
        self.hlsl = hlsl
        self.glsl = glsl
        self.packed = packed
        self.lanes = lanes
    }

    var shader: String {
        """
        @group(0) @binding(0) var source: texture_storage_2d<\(wgsl), read>;
        @group(0) @binding(1) var destination: texture_storage_2d<\(wgsl), \(readWrite ? "read_write" : "write")>;
        @group(0) @binding(2) var<storage, read_write> result: array<\(vector), 3>;
        @compute @workgroup_size(1) fn run() {
            let xy = vec2i(0);
            let value: \(vector) = textureLoad(source, xy);
            result[0] = value;
            result[1] = textureLoad(source, vec2i(-1));
            \(readWrite ? "result[2] = textureLoad(destination, xy);" : "result[2] = \(vector)(0);")
            textureStore(destination, xy, value);
        }
        """
    }
}

private let storageFormatCases: [StorageFormatCase] = [
    .init(.rgba8Unorm, "rgba8unorm", "vec4f", "unorm float4", "rgba8", [0x00ff00ff], [1, 0, 1, 0].map(Float.init).map(\.bitPattern)),
    .init(.rgba8Snorm, "rgba8snorm", "vec4f", "snorm float4", "rgba8_snorm", [0x007f0081], [-1, 0, 1, 0].map(Float.init).map(\.bitPattern)),
    .init(.rgba8Uint, "rgba8uint", "vec4u", "uint4", "rgba8ui", [0xff020180], [128, 1, 2, 255]),
    .init(.rgba8Sint, "rgba8sint", "vec4i", "int4", "rgba8i", [0xff020180], [0xffffff80, 1, 2, .max]),
    .init(.rgba16Uint, "rgba16uint", "vec4u", "uint4", "rgba16ui", [0x0001ffff, 0x80000002], [65535, 1, 2, 32768]),
    .init(.rgba16Sint, "rgba16sint", "vec4i", "int4", "rgba16i", [0x0001ffff, 0x80000002], [.max, 1, 2, 0xffff8000]),
    .init(.rgba16Float, "rgba16float", "vec4f", "float4", "rgba16f", [0xc0003c00, 0x40003800], [1, -2, 0.5, 2].map(Float.init).map(\.bitPattern)),
    .init(.r32Uint, "r32uint", "vec4u", "uint", "r32ui", [0xdeadbeef], [0xdeadbeef, 0, 0, 1]),
    .init(.r32Sint, "r32sint", "vec4i", "int", "r32i", [0x80000005], [0x80000005, 0, 0, 1]),
    .init(.r32Float, "r32float", "vec4f", "float", "r32f", [0xc0000000], [-2, 0, 0, 1].map(Float.init).map(\.bitPattern)),
    .init(.rg32Uint, "rg32uint", "vec4u", "uint2", "rg32ui", [0xdeadbeef, 99], [0xdeadbeef, 99, 0, 1]),
    .init(.rg32Sint, "rg32sint", "vec4i", "int2", "rg32i", [0x80000005, 99], [0x80000005, 99, 0, 1]),
    .init(.rg32Float, "rg32float", "vec4f", "float2", "rg32f", [0xc0000000, 0x3f000000], [-2, 0.5, 0, 1].map(Float.init).map(\.bitPattern)),
    .init(.rgba32Uint, "rgba32uint", "vec4u", "uint4", "rgba32ui", [0xdeadbeef, 1, 2, 3], [0xdeadbeef, 1, 2, 3]),
    .init(.rgba32Sint, "rgba32sint", "vec4i", "int4", "rgba32i", [0x80000005, 1, 2, 3], [0x80000005, 1, 2, 3]),
    .init(.rgba32Float, "rgba32float", "vec4f", "float4", "rgba32f", [0xc0000000, 0x3f000000, 0x40000000, 0x3f800000], [-2, 0.5, 2, 1].map(Float.init).map(\.bitPattern)),
]

@Test(arguments: storageFormatCases, [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
private func emitsCoreStorageTextureFormats(fixture: StorageFormatCase, target: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(fixture.shader), to: target)
    #expect(shader.storageTextures.map(\.format) == [fixture.format, fixture.format])
    #expect(shader.storageTextures.map(\.access) == [.load, fixture.readWrite ? [.load, .store] : .store])
    guard case .text(_, let source) = shader.artifact else { Issue.record("Expected source"); return }
    switch target {
    case .wgsl:
        #expect(source.contains("texture_storage_2d<\(fixture.wgsl), read>"))
        _ = try compiler.parseAndValidate(.wgsl(source))
    case .msl:
        let scalar = fixture.vector == "vec4i" ? "int" : fixture.vector == "vec4u" ? "uint" : "float"
        #expect(source.contains("texture2d<\(scalar), access::read>"))
    case .hlsl: #expect(source.contains("RWTexture2D<\(fixture.hlsl)>"))
    case .glsl: #expect(source.contains("layout(\(fixture.glsl), binding=0) readonly"))
    default: Issue.record("Unexpected format")
    }
}

#if canImport(Metal)
@Test(arguments: storageFormatCases)
private func roundTripsCoreStorageTextureFormatsOnMetal(fixture: StorageFormatCase) async throws {
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(fixture.shader), to: .msl)
    let format = try #require(shader.storageTextures.first).textureFormat()
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let source = try await device.createTexture(.init(width: 1, height: 1, format: format,
            usage: [.storageBinding, .copyDestination]))
        let destination = try await device.createTexture(.init(width: 1, height: 1, format: format,
            usage: [.storageBinding, .copySource]))
        let upload = try await device.createBuffer(.init(size: 32, usage: [.copySource, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 48, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: 64, usage: [.copyDestination, .mapRead]))
        let region = TextureCopyRegion(width: 1, height: 1)
        if fixture.packed.count > 1 {
            await #expect(throws: GPUError.validation("Invalid texture upload buffer, offset or size")) {
                try await device.submitCommands([.copyBufferToTexture(source: upload, layout: .init(offset: 4),
                    destination: source, region: region)])
            }
            await #expect(throws: GPUError.validation("Invalid texture copy buffer, offset or size")) {
                try await device.submitCommands([.copyTextureToBuffer(source: destination, destination: read,
                    bytesPerRow: 256, destinationOffset: 4)])
            }
        }
        let ticket = try await device.submitCommands([
            .write(buffer: upload, offset: 16, words: fixture.packed),
            .copyBufferToTexture(source: upload, layout: .init(offset: 16), destination: source, region: region),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 2): output], workgroups: .init(x: 1),
                             textures: [.init(binding: 0): source, .init(binding: 1): destination]),
            .copy(source: output, destination: read, byteCount: 48), .read(buffer: read, byteCount: 48),
            .copyTextureToBuffer(source: destination, destination: read, bytesPerRow: 256),
            .read(buffer: read, byteCount: UInt64(fixture.packed.count) * 4),
        ])
        let alpha: UInt32 = fixture.vector == "vec4f" ? Float(1).bitPattern : 1
        let expected = fixture.lanes + [0, 0, 0, 0, 0, 0, 0, fixture.readWrite ? alpha : 0]
        #expect(try await ticket.wait() == [expected, fixture.packed])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
