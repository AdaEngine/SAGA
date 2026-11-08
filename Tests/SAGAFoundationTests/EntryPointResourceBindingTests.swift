import Testing
import SAGACompiler
import SAGACore
import SAGAMetal
import SAGAShaderIR
import SAGATypes

private let bindingVariants = """
    @group(0) @binding(0) var<storage, read_write> wide: array<u32, 4>;
    @group(0) @binding(0) var<storage, read_write> narrow: vec2u;
    fn writeWide() { wide = array<u32, 4>(3u, 5u, 7u, 11u); }
    fn writeNarrow() { narrow = vec2u(13u, 17u); }
    @compute @workgroup_size(1) fn first() { writeWide(); }
    @compute @workgroup_size(1) fn second() { writeNarrow(); }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func entryPointBindingsCompileWithReusedSlots(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    for (entry, bytes) in [("first", UInt64(16)), ("second", UInt64(8))] {
        let shader = try compiler.compileCompute(.wgsl(bindingVariants), entryPoint: entry, to: format)
        #expect(shader.storageBindings.map(\.binding) == [.init(group: 0, binding: 0)])
        #expect(shader.storageBindings.map(\.minimumByteSize) == [bytes])
        if case .text(.wgsl, let text) = shader.artifact { _ = try compiler.parseAndValidate(.wgsl(text)) }
    }
    let implicit = try compiler.compileCompute(.wgsl("""
        @group(0) @binding(0) var unused: texture_2d<f32>;
        @group(0) @binding(0) var<storage, read_write> live: u32;
        @compute @workgroup_size(1) fn main() { live = 7u; }
        """), to: format)
    #expect(implicit.sampledTextures.isEmpty)
    #expect(implicit.storageBindings.map(\.binding) == [.init(group: 0, binding: 0)])
}

@Test func entryPointBindingsRejectATransitiveCollision() throws {
    let source = ShaderInput.wgsl("""
        @group(0) @binding(0) var<storage, read> a: u32;
        @group(0) @binding(0) var<uniform> b: u32;
        fn readA() -> u32 { return a; }
        fn readB() -> u32 { return b; }
        fn pair() -> u32 { return readA() + readB(); }
        @compute @workgroup_size(1) fn run() { _ = pair(); }
        """)
    do {
        _ = try IndependentShaderCompiler().parseAndValidate(source)
        Issue.record("Accepted two globals sharing a binding in one entry point")
    } catch ShaderFailure.diagnostics(let diagnostics) {
        #expect(diagnostics.first?.code == .invalidBinding)
        #expect(diagnostics.first?.message.contains("entry point 'run'") == true)
    }
}

@Test func entryPointBindingsKeepTheLargestGLSLRangeForLinkedVariants() throws {
    let input = ShaderInput.wgsl("""
        enable wgpu_binding_array;
        struct Params { value: vec4f, }
        @group(0) @binding(0) var<uniform> small: binding_array<Params, 2>;
        @group(0) @binding(0) var<uniform> large: binding_array<Params, 3>;
        @group(0) @binding(1) var<uniform> common: vec4f;
        @vertex fn vs() -> @builtin(position) vec4f { return small[1].value + common; }
        @fragment fn fs() -> @location(0) vec4f { return large[2].value + common; }
        """)
    let compiler = IndependentShaderCompiler()
    for (entry, count) in [("vs", UInt32(2)), ("fs", UInt32(3))] {
        let shader = try compiler.compileRenderStage(input, entryPoint: entry, to: .glsl)
        #expect(shader.uniformBindings.first?.bindingArray?.count == count)
        #expect(shader.uniformBindings.map(\.flatBindingIndex) == [9, 12])
    }
}

@Test func entryPointBindingsKeepDifferentResourceKindsInTheirGLSLNamespaces() throws {
    let input = ShaderInput.wgsl("""
        @group(0) @binding(0) var<uniform> point: vec4f;
        @group(0) @binding(0) var image: texture_2d<f32>;
        @vertex fn vs() -> @builtin(position) vec4f { return point; }
        @fragment fn fs() -> @location(0) vec4f { return textureLoad(image, vec2i(0), 0); }
        """)
    let compiler = IndependentShaderCompiler()
    let vertex = try compiler.compileRenderStage(input, entryPoint: "vs", to: .glsl)
    let fragment = try compiler.compileRenderStage(input, entryPoint: "fs", to: .glsl)
    #expect(vertex.uniformBindings.map(\.binding) == [.init(group: 0, binding: 0)])
    #expect(vertex.sampledTextures.isEmpty)
    #expect(fragment.uniformBindings.isEmpty)
    #expect(fragment.sampledTextures.map(\.binding) == [.init(group: 0, binding: 0)])
}

#if canImport(Metal)
@Test func entryPointBindingsReuseOneSlotWithDifferentBufferSizesOnMetal() async throws {
    let compiler = IndependentShaderCompiler()
    let input = ShaderInput.wgsl(bindingVariants)
    let first = try compiler.compileCompute(input, entryPoint: "first", to: .msl)
    let second = try compiler.compileCompute(input, entryPoint: "second", to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let wide = try await device.createComputePipeline(first)
        let narrow = try await device.createComputePipeline(second)
        let a = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
        let b = try await device.createBuffer(.init(size: 8, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 16, usage: [.copyDestination, .mapRead]))
        let snapshots = try await device.submitCommands([
            .dispatchCompute(pipeline: wide, bindings: [.init(binding: 0): a], workgroups: .init(x: 1)),
            .dispatchCompute(pipeline: narrow, bindings: [.init(binding: 0): b], workgroups: .init(x: 1)),
            .copy(source: a, destination: readback, byteCount: 16), .read(buffer: readback, byteCount: 16),
            .copy(source: b, destination: readback, byteCount: 8), .read(buffer: readback, byteCount: 8),
        ]).wait()
        #expect(snapshots == [[3, 5, 7, 11], [13, 17]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
