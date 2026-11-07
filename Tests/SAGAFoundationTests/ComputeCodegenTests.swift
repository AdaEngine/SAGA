import Testing
import SAGACompiler
import SAGAShaderIR

private func storageShader(
    _ body: String = "output = id.x * 3u + 7u;", group: UInt32 = 0
) -> String {
    """
    @group(\(group)) @binding(3) var<storage, read_write> output: u32;

    @compute @workgroup_size(1)
    fn main(@builtin(global_invocation_id) id: vec3<u32>) {
        \(body)
    }
    """
}

private let arrayStorageShader = """
    @group(0) @binding(2) var<storage, read> input: array<u32, 4>;
    @group(0) @binding(3) var<storage, read_write> output: array<u32, 6>;

    @compute @workgroup_size(2)
    fn run(@builtin(global_invocation_id) id: vec3<u32>) {
        let old = output[id.x];
        output[id.x] = input[id.x] * 3u + 7u;
        output[id.x] = old + output[id.x];
    }
    """

private let branchingStorageShader = """
    @group(0) @binding(3) var<storage, read_write> output: array<u32, 4>;
    @compute @workgroup_size(2)
    fn run(@builtin(global_invocation_id) id: vec3<u32>) {
        if id.x >= 4u { return; }
        var value: u32;
        let before = value;
        value = id.x + 5u;
        var firstPair = id.x < 2u;
        if firstPair && id.x != 0u {
            let saved = value;
            {
                var value = 2u;
                value += 1u;
                output[id.x] = value + saved;
            }
            value += saved;
        } else if !firstPair {
            value *= 2u;
        } else {
            value += 4u;
        }
        output[id.x] = output[id.x] + before + value;
    }
    """

private let loopingStorageShader = """
    @group(0) @binding(3) var<storage, read_write> output: array<u32, 4>;
    @compute @workgroup_size(2)
    fn run(@builtin(global_invocation_id) id: vec3<u32>) {
        var result = 0u;
        for (var i = 0u; i < 4u; i += 1u) {
            var perIteration: u32;
            perIteration += 1u;
            if i == 1u { continue; }
            result += perIteration + i;
        }
        var counter = 0u;
        loop {
            if counter == 3u { break; }
            if counter != 0u { result += 5u; }
            continue;
            continuing {
                counter += 1u;
                if counter == 1u { result += id.x; }
                break if counter >= 2u && id.x == 0u;
            }
        }
        var j = 0u;
        while j < 2u {
            result += 2u;
            j += 1u;
        }
        output[id.x] = result + counter;
    }
    """

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .glsl])
func translatesComputeStorageWrite(to format: ShaderOutputFormat) throws {
    let artifact = try IndependentShaderCompiler().compile(.wgsl(storageShader()), to: format)
    let expected: String
    switch format {
    case .msl:
        expected = """
            #include <metal_stdlib>
            using namespace metal;
            kernel void saga_main(uint3 saga_arg_0 [[thread_position_in_grid]], device uint* saga_buffer_0 [[buffer(3)]]) {
                *saga_buffer_0 = ((saga_arg_0.x * 3u) + 7u);
                return;
            }
            """
    case .hlsl:
        expected = """
            RWStructuredBuffer<uint> saga_buffer_0 : register(u3, space0);
            [numthreads(1, 1, 1)]
            void saga_main(uint3 saga_arg_0 : SV_DispatchThreadID) {
                saga_buffer_0[0] = ((saga_arg_0.x * 3u) + 7u);
                return;
            }
            """
    case .glsl:
        expected = """
            #version 450 core
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
            layout(std430, binding = 3) buffer SagaBuffer0 { uint value; } saga_buffer_0;
            void main() {
                saga_buffer_0.value = ((gl_GlobalInvocationID.x * 3u) + 7u);
                return;
            }
            """
    default:
        Issue.record("Unexpected target")
        return
    }
    #expect(artifact == .text(format: format, source: expected + "\n"))
}

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .glsl])
func computeCodegenEmitsReturnsAndTargetWorkgroupLayout(to format: ShaderOutputFormat) throws {
    let artifact = try IndependentShaderCompiler().compile(
        .wgsl("@compute @workgroup_size(2, 3, 4) fn launch() { return; }"), to: format)
    guard case .text(_, let source) = artifact else {
        Issue.record("Expected a text shader")
        return
    }
    switch format {
    case .msl: #expect(source.contains("kernel void saga_main()"))
    case .hlsl: #expect(source.contains("[numthreads(2, 3, 4)]"))
    case .glsl: #expect(source.contains("local_size_x = 2, local_size_y = 3, local_size_z = 4"))
    default: Issue.record("Unexpected target")
    }
    #expect(source.contains("return;"))
}

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .glsl])
func computeCodegenRejectsInvalidWGSLAndUnsupportedIR(to format: ShaderOutputFormat) {
    let compiler = IndependentShaderCompiler()
    let invalid = storageShader("output = undefinedValue;")
    #expect(throws: ShaderFailure.self) {
        do {
            _ = try compiler.compile(.wgsl(invalid), to: format)
        } catch let failure as ShaderFailure {
            guard case .diagnostics(let diagnostics) = failure else {
                Issue.record("Expected frontend diagnostics, got \(failure)")
                throw failure
            }
            #expect(!diagnostics.isEmpty)
            throw failure
        }
    }
    for source in [
        "@compute @workgroup_size(1) fn a() {} @compute @workgroup_size(1) fn b() {}",
        "@compute @workgroup_size(32, 33) fn oversized() {}",
        "@compute @workgroup_size(1, 1, 65) fn oversizedZ() {}",
        storageShader(group: 1),
        storageShader("output = min(id.x, 3u);"),
        storageShader("var x: array<u32, 2>; _ = x; output = 1u;"),
        """
        @group(0) @binding(0) var<storage, read_write> data: array<u32>;
        @compute @workgroup_size(1) fn run() { data[0] = 1u; }
        """,
    ] {
        #expect(throws: ShaderFailure.self) {
            do {
                _ = try compiler.compile(.wgsl(source), to: format)
            } catch let failure as ShaderFailure {
                if case .unsupported = failure {} else {
                    Issue.record("Expected unsupported lowering, got \(failure)")
                }
                throw failure
            }
        }
    }
}

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .glsl])
func translatesRobustFixedStorageArrays(to format: ShaderOutputFormat) throws {
    let artifact = try IndependentShaderCompiler().compile(.wgsl(arrayStorageShader), to: format)
    guard case .text(_, let source) = artifact else {
        Issue.record("Expected a text shader")
        return
    }
    switch format {
    case .msl: #expect(source.contains("device const uint* saga_buffer_0 [[buffer(2)]]"))
    case .hlsl: #expect(source.contains("StructuredBuffer<uint> saga_buffer_0 : register(t2, space0);"))
    case .glsl: #expect(source.contains("readonly buffer SagaBuffer0 { uint value[4]; }"))
    default: Issue.record("Unexpected target")
    }
    let lines = source.split(separator: "\n")
    #expect(lines.filter { $0.contains(" < 4u)") }.count == 1)
    #expect(lines.filter { $0.contains(" < 6u)") }.count == 4)
    #expect(lines.filter { $0.contains(" = 0u;") }.count == 3)
}

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .glsl])
func computeReflectionMatchesEmittedBindingsAndEntryPoint(to format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let input = ShaderInput.wgsl(arrayStorageShader)
    let shader = try compiler.compileCompute(input, to: format)
    #expect(shader.artifact == (try compiler.compile(input, to: format)))
    #expect(shader.sourceEntryPoint == "run")
    #expect(shader.entryPoint == (format == .glsl ? "main" : "saga_main"))
    #expect(shader.workgroupSize == .init(x: 2, y: 1, z: 1))
    #expect(shader.storageBindings.map(\.binding) == [.init(group: 0, binding: 2), .init(group: 0, binding: 3)])
    #expect(shader.storageBindings.map(\.minimumByteSize) == [16, 24])
    #expect(shader.storageBindings.map(\.isReadOnly) == [true, false])
    let scalar = try compiler.compileCompute(.wgsl(storageShader()), to: format)
    #expect(scalar.storageBindings.map(\.minimumByteSize) == [4])
}

@Test func emittedGLSLRoundTripsThroughSwiftValidation() throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(arrayStorageShader), to: .glsl)
    guard case .text(.glsl, let source) = shader.artifact else {
        Issue.record("Expected GLSL")
        return
    }
    let validated = try compiler.parseAndValidate(.glsl(source, stage: .compute, defines: [:]))
    let entry = try #require(validated.module.entryPoints.first)
    #expect(entry.name == shader.entryPoint)
    #expect(entry.stage == .compute)
    #expect(entry.workgroupSize == shader.workgroupSize)
    #expect(validated.module.globalVariables.values.compactMap(\.binding) == shader.storageBindings.map(\.binding))
}

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .glsl, .wgsl])
func translatesBranchesAndScalarLocals(to format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(branchingStorageShader), to: format)
    guard case .text(_, let source) = shader.artifact else {
        Issue.record("Expected a text shader")
        return
    }
    #expect(source.contains("} else {"))
    #expect(source.contains(" >= 4u"))
    #expect(source.contains(" != 0u"))
    #expect(source.contains(" = false;"))
    #expect(source.contains("return;"))
    if format == .wgsl || format == .glsl {
        let input: ShaderInput = format == .wgsl ? .wgsl(source) : .glsl(source, stage: .compute, defines: [:])
        let validated = try compiler.parseAndValidate(input)
        #expect(validated.module.entryPoints.first?.workgroupSize == shader.workgroupSize)
    }
}

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .glsl, .wgsl])
func translatesLoopsWithContinuationAndRuntimeBreakCondition(to format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(loopingStorageShader), to: format)
    guard case .text(_, let source) = shader.artifact else {
        Issue.record("Expected a text shader")
        return
    }
    #expect(source.contains("continue;"))
    #expect(source.contains("break;"))
    #expect(source.contains(" >= 2u"))
    if format == .wgsl || format == .glsl {
        let input: ShaderInput = format == .wgsl ? .wgsl(source) : .glsl(source, stage: .compute, defines: [:])
        let validated = try compiler.parseAndValidate(input)
        #expect(validated.module.entryPoints.first?.workgroupSize == shader.workgroupSize)
    }
}

@Test func continuingRejectsReferencesToDeclarationsBypassedByContinue() throws {
    for declaration in ["var step = 2u;", "let step = 2u;", "const step = 2u;"] {
        let source = """
            @compute @workgroup_size(1) fn run() {
                var i = 0u;
                loop {
                    if i == 0u { continue; }
                    \(declaration)
                    continuing {
                        i += step;
                        break if true;
                    }
                }
            }
            """
        do {
            _ = try IndependentShaderCompiler().parseAndValidate(.wgsl(source))
            Issue.record("Accepted a declaration bypassed by continue")
        } catch ShaderFailure.diagnostics(let diagnostics) {
            #expect(diagnostics.first?.code == .invalidControlFlow)
        }
    }
    let valid = """
        @compute @workgroup_size(1) fn run() {
            var i = 0u;
            loop {
                var step = 2u;
                if i == 0u { continue; }
                continuing {
                    i += step;
                    break if i >= 2u;
                }
            }
        }
        """
    _ = try IndependentShaderCompiler().compile(.wgsl(valid), to: .wgsl)
}

#if canImport(Metal)
import Metal

@Test func emittedMSLStoresValueOnMetal() throws {
    let result = try dispatchMSL(
        storageShader(), initial: [3: [.max]], readSlot: 3, groups: 1)
    #expect(result == [7])
}

@Test func emittedMSLBoundsChecksArraysAndPreservesLoadSnapshotsOnMetal() throws {
    let result = try dispatchMSL(
        arrayStorageShader,
        initial: [2: [0, 1, 2, .max, 100, 200], 3: [20, 21, 22, 23, 24, 25, 999, 999]],
        readSlot: 3, groups: 4)
    #expect(result == [27, 31, 35, 27, 31, 32, 999, 999])
}

@Test func emittedMSLBranchesPreserveLocalSnapshotsAndShadowingOnMetal() throws {
    let result = try dispatchMSL(branchingStorageShader, initial: [3: [10, 10, 10, 10, 99, 99]], readSlot: 3, groups: 3)
    #expect(result == [19, 21, 24, 26, 99, 99])
}

@Test func emittedMSLLoopsRunContinuingAndReinitializeLocalsOnMetal() throws {
    let result = try dispatchMSL(loopingStorageShader, initial: [3: [0, 0, 0, 0]], readSlot: 3, groups: 2)
    #expect(result == [19, 26, 27, 28])
}

private func dispatchMSL(
    _ wgsl: String, initial: [Int: [UInt32]], readSlot: Int, groups: Int
) throws -> [UInt32] {
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(wgsl), to: .msl)
    guard case .text(.msl, let source) = shader.artifact else {
        Issue.record("Expected MSL")
        return []
    }
    let device = try #require(MTLCreateSystemDefaultDevice())
    let library = try device.makeLibrary(source: source, options: nil)
    let function = try #require(library.makeFunction(name: shader.entryPoint))
    let pipeline = try device.makeComputePipelineState(function: function)
    let queue = try #require(device.makeCommandQueue())
    let commands = try #require(queue.makeCommandBuffer())
    let encoder = try #require(commands.makeComputeCommandEncoder())
    encoder.setComputePipelineState(pipeline)
    var buffers: [Int: any MTLBuffer] = [:]
    for binding in shader.storageBindings {
        let slot = Int(binding.binding.binding)
        let words = try #require(initial[slot])
        try #require(UInt64(words.count) * 4 >= binding.minimumByteSize)
        let buffer = try #require(words.withUnsafeBytes {
            device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
        })
        buffers[slot] = buffer
        encoder.setBuffer(buffer, offset: 0, index: slot)
    }
    let size = shader.workgroupSize
    encoder.dispatchThreadgroups(
        MTLSize(width: groups, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: Int(size.x), height: Int(size.y), depth: Int(size.z)))
    encoder.endEncoding()
    commands.commit()
    commands.waitUntilCompleted()
    try #require(commands.status == .completed)
    let buffer = try #require(buffers[readSlot])
    let count = try #require(initial[readSlot]?.count)
    let words = buffer.contents().bindMemory(to: UInt32.self, capacity: count)
    return Array(UnsafeBufferPointer(start: words, count: count))
}
#endif
