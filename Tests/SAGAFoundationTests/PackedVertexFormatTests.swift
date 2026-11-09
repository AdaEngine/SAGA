import SAGACore
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

private struct PackedVertexFixture {
    let id: UInt32
    let components: Int
    let bytes: [UInt8]
    let dataType: VertexDataType
    let expected: String
}

private func packedVertexFixtures() -> [PackedVertexFixture] {
    let families: [(UInt32, [UInt8], VertexDataType, String)] = [
        (16, [201], .uint, "201u"), (19, [0x80], .sint, "-128i"),
        (22, [128], .float, "128.0 / 255.0"), (25, [0x80], .float, "-1.0"),
        (28, [0xef, 0xbe], .uint, "48879u"), (31, [0, 0x80], .sint, "-32768i"),
        (34, [0, 0x80], .float, "32768.0 / 65535.0"), (37, [0, 0x80], .float, "-1.0"),
    ]
    let linear = families.flatMap { (first, bytes, dataType, value) in
        [1, 2, 4].enumerated().map { index, count in
            let scalar = switch dataType {
            case .uint: "u32"
            case .sint: "i32"
            case .float: "f32"
            }
            let expected = count == 1 ? value : "vec\(count)<\(scalar)>(\(value))"
            return PackedVertexFixture(id: first + UInt32(index), components: count,
                bytes: (0..<count).flatMap { _ in bytes }, dataType: dataType, expected: expected)
        }
    }
    return linear + [
        .init(id: 40, components: 4, bytes: [0xff, 0x03, 0, 0xe0], dataType: .float,
              expected: "vec4f(1, 0, 512.0 / 1023.0, 1)"),
        .init(id: 41, components: 4, bytes: [32, 64, 128, 255], dataType: .float,
              expected: "vec4f(128.0 / 255.0, 64.0 / 255.0, 32.0 / 255.0, 1)"),
        .init(id: 42, components: 4, bytes: [0, 0x02, 0xf0, 0x9f], dataType: .float,
              expected: "vec4f(-1, 0, 1, -1)"),
    ]
}

@Test func packedVertexFormatsRetainByteWidthAndShaderScalarKind() throws {
    for fixture in packedVertexFixtures() {
        let format = try #require(VertexFormat(rawValue: fixture.id))
        #expect(format.componentCount == UInt32(fixture.components))
        #expect(format.byteSize == UInt32(fixture.bytes.count))
        #expect(format.dataType == fixture.dataType)
    }
    #expect(VertexFormat.float32.rawValue == 1 && VertexFormat.float16x4.rawValue == 15)
}

@Test func packedVertexFormatsValidateAlignmentAndOptionalSupport() throws {
    let caps = try DeviceCapabilities(maxBufferSize: 64)
    _ = try VertexBufferLayout.validate([.init(stride: 4, attributes: [
        .init(location: 0, offset: 1, format: .uint8),
        .init(location: 1, offset: 2, format: .sint16),
    ])], against: caps)
    #expect(throws: GPUError.validation("Invalid, duplicate or unaligned vertex attribute")) {
        try VertexBufferLayout.validate([.init(stride: 4, attributes: [
            .init(location: 0, offset: 1, format: .uint8x2),
        ])], against: caps)
    }
    let packed: [VertexBufferLayout] = [
        .init(stride: 4, attributes: [.init(location: 0, offset: 0, format: .snorm1010102)]),
    ]
    #expect(throws: GPUError.validation("Feature not advertised")) {
        try VertexBufferLayout.validate(packed, against: caps)
    }
    let enabled = try DeviceCapabilities(maxBufferSize: 64, features: [.vertexFormatSnorm1010102])
    _ = try VertexBufferLayout.validate(packed, against: enabled)
}

#if canImport(Metal)
@Test(arguments: [false, true])
func packedVertexFormatsFetchAllAddedFormatsOnMetal(_ pulling: Bool) async throws {
    let fixtures = packedVertexFixtures()
    var bytes: [UInt8] = []
    var attributes: [VertexAttribute] = []
    var fields: [String] = []
    var checks: [String] = []
    for (index, fixture) in fixtures.enumerated() {
        let format = try #require(VertexFormat(rawValue: fixture.id))
        let alignment = min(4, fixture.bytes.count)
        while bytes.count % alignment != 0 { bytes.append(0) }
        attributes.append(.init(location: UInt32(index), offset: UInt32(bytes.count), format: format))
        bytes += fixture.bytes
        let scalar = switch fixture.dataType {
        case .uint: "u32"
        case .sint: "i32"
        case .float: "f32"
        }
        let type = fixture.components == 1 ? scalar : "vec\(fixture.components)<\(scalar)>"
        fields.append("@location(\(index)) a\(index): \(type),")
        let difference = fixture.dataType == .float
            ? "abs(input.a\(index) - (\(fixture.expected))) < \(type)(0.001)"
            : "input.a\(index) == \(fixture.expected)"
        let matches = fixture.components == 1 ? difference : "all(\(difference))"
        checks.append("if \(matches) { mask |= \(1 << index)u; }")
    }
    while bytes.count % 4 != 0 { bytes.append(0) }
    let shader = ShaderInput.wgsl("""
        struct Inputs { \(fields.joined(separator: "\n")) }
        struct Raster {
            @builtin(position) position: vec4f,
            @location(0) @interpolate(flat) mask: u32,
        }
        @vertex fn vs(input: Inputs, @builtin(vertex_index) i: u32) -> Raster {
            let points = array(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
            var mask = 0u;
            \(checks.joined(separator: "\n"))
            return Raster(vec4f(points[i], 0, 1), mask);
        }
        @fragment fn fs(in: Raster) -> @location(0) vec4u {
            return (vec4u(in.mask) >> vec4u(0, 8, 16, 24)) & vec4u(255u);
        }
        """)
    let compiler = IndependentShaderCompiler()
    let layouts: [VertexBufferLayout] = [.init(stride: 0, stepMode: .instance, attributes: attributes)]
    let vertex = pulling
        ? try compiler.compileMetalVertexStage(shader, entryPoint: "vs", vertexBuffers: layouts)
        : try compiler.compileRenderStage(shader, entryPoint: "vs", to: .msl)
    let fragment = try compiler.compileRenderStage(shader, entryPoint: "fs", to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Uint,
            vertexBuffers: layouts)
        let stream = try await device.createBuffer(.init(size: UInt64(bytes.count), usage: [.vertex, .copyDestination]))
        let texture = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Uint,
                                                            usage: [.renderAttachment, .copySource]))
        let readback = try await device.createBuffer(.init(size: 4, usage: [.copyDestination, .mapRead]))
        let snapshots = try await device.submitCommands([
            .writeBytes(buffer: stream, offset: 0, bytes: bytes),
            .render(pipeline: pipeline, target: texture, clear: .transparentBlack, vertexCount: 3,
                    vertexBuffers: [.init(buffer: stream)]),
            .copyTextureToBuffer(source: texture, destination: readback, bytesPerRow: 256),
            .read(buffer: readback, byteCount: 4),
        ]).wait()
        #expect(snapshots == [[(1 << fixtures.count) - 1]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
