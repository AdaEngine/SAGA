import Testing
import SAGACompiler
import SAGACore
import SAGAMetal
import SAGAShaderIR
import SAGATypes

private let computeEntries = """
    override width = 2u;
    override start: u32;
    @group(0) @binding(0) var<storage, read_write> products: array<u32, 4>;
    @group(1) @binding(3) var<storage, read_write> sequence: array<u32, 3>;
    @group(2) @binding(2) var<uniform> color: vec4f;

    fn product(index: u32) -> u32 { return (index + 1u) * width; }

    @compute @workgroup_size(width)
    fn scale(@builtin(local_invocation_index) lane: u32) {
        products[lane] = product(lane);
    }

    @compute @workgroup_size(3)
    fn fill(@builtin(local_invocation_index) lane: u32) {
        sequence[lane] = start + lane;
    }

    @fragment fn shade() -> @location(0) vec4f { return color; }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func selectsComputeEntryPointAndItsBindings(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let input = ShaderInput.wgsl(computeEntries)
    let scale = try compiler.compileCompute(input, entryPoint: "scale", to: format, constants: ["width": 4])
    let fill = try compiler.compileCompute(input, entryPoint: "fill", to: format, constants: ["start": 70])
    #expect(scale.sourceEntryPoint == "scale")
    #expect(fill.sourceEntryPoint == "fill")
    #expect(scale.workgroupSize == .init(x: 4))
    #expect(fill.workgroupSize == .init(x: 3))
    #expect(scale.storageBindings.map(\.binding) == [.init(group: 0, binding: 0)])
    #expect(fill.storageBindings.map(\.binding) == [.init(group: 1, binding: 3)])
    #expect(scale.uniformBindings.isEmpty && fill.uniformBindings.isEmpty)
    if format == .wgsl {
        for shader in [scale, fill] {
            guard case .text(.wgsl, let source) = shader.artifact else { throw ShaderFailure.invalidHandle }
            let module = try compiler.parseAndValidate(.wgsl(source)).module
            #expect(module.entryPoints.map(\.name) == [shader.entryPoint])
            #expect(module.globalVariables.values.compactMap(\.binding) == shader.storageBindings.map(\.binding))
        }
    }
}

@Test func infersOnlyComputeEntryPointInMixedModule() throws {
    let source = ShaderInput.wgsl("""
        @fragment fn draw() -> @location(0) vec4f { return vec4f(1.0); }
        @compute @workgroup_size(4) fn dispatch() {}
        """)
    let shader = try IndependentShaderCompiler().compileCompute(source, to: .msl)
    #expect(shader.sourceEntryPoint == "dispatch")
    #expect(shader.workgroupSize == .init(x: 4))
}

@Test func diagnosesUnknownAndAmbiguousComputeEntryPoints() throws {
    let compiler = IndependentShaderCompiler()
    let input = ShaderInput.wgsl(computeEntries)
    for (name, code) in [("missing", ShaderDiagnosticCode.unresolvedName), ("shade", .invalidBinding)] {
        do {
            _ = try compiler.compileCompute(input, entryPoint: name, to: .msl)
            Issue.record("Accepted an invalid compute entry point")
        } catch ShaderFailure.diagnostics(let diagnostics) {
            #expect(diagnostics.first?.code == code)
        }
    }
    do {
        _ = try compiler.compileCompute(input, to: .msl)
        Issue.record("Accepted ambiguous compute entry points")
    } catch ShaderFailure.diagnostics(let diagnostics) {
        #expect(diagnostics.first?.code == .invalidBinding)
    }
    #expect(throws: ShaderFailure.self) {
        try compiler.compileCompute(input, entryPoint: "fill", to: .msl)
    }
}

#if canImport(Metal)
@Test func dispatchesDifferentComputeEntryPointsFromOneModuleOnMetal() async throws {
    let compiler = IndependentShaderCompiler()
    let input = ShaderInput.wgsl(computeEntries)
    let scale = try compiler.compileCompute(input, entryPoint: "scale", to: .msl, constants: ["width": 4])
    let fill = try compiler.compileCompute(input, entryPoint: "fill", to: .msl, constants: ["start": 70])
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let scalePipeline = try await device.createComputePipeline(scale)
        let fillPipeline = try await device.createComputePipeline(fill)
        let first = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
        let second = try await device.createBuffer(.init(size: 12, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 16, usage: [.copyDestination, .mapRead]))
        let snapshots = try await device.submitCommands([
            .dispatchCompute(pipeline: scalePipeline, bindings: [.init(binding: 0): first], workgroups: .init(x: 1)),
            .dispatchCompute(pipeline: fillPipeline, bindings: [.init(group: 1, binding: 3): second], workgroups: .init(x: 1)),
            .copy(source: first, destination: readback, byteCount: 16), .read(buffer: readback, byteCount: 16),
            .copy(source: second, destination: readback, byteCount: 12), .read(buffer: readback, byteCount: 12),
        ]).wait()
        #expect(snapshots == [[4, 8, 12, 16], [70, 71, 72]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
