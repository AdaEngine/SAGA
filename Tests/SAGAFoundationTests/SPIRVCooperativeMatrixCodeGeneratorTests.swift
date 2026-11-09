import SAGACompiler
import SAGABackends
import SAGAFrontends
import SAGAShaderIR
import SAGAValidation
import Testing

@Test(arguments: [UInt32(0), 1])
func spirvCooperativeLoadSupportsStaticArrayElementPointer(_ index: UInt32) throws {
    let source = """
        enable wgpu_cooperative_matrix;
        @group(0) @binding(0) var<storage, read> input: array<f32, 64>;
        @compute @workgroup_size(32) fn main() {
            let tile = coopLoad<coop_mat8x8<f32, A>>(&input[0]);
        }
        """
    let parsed = try WGSLFrontend(options: .init(capabilities: [.cooperativeMatrix])).parse(.wgsl(source))
    let original = try #require(parsed.functions.values.first)
    var changed = false
    let expressions = original.expressions.values.map { expression in
        if case .access(let base, _) = expression.kind, !changed {
            changed = true
            return IRExpression(type: expression.type, kind: .accessIndex(base: base, index: index), location: expression.location)
        }
        return expression
    }
    #expect(changed)
    let function = IRFunction(name: original.name, arguments: original.arguments, result: original.result,
        localVariables: original.localVariables, expressions: .init(expressions), namedExpressions: original.namedExpressions,
        statements: original.statements, blocks: original.blocks, body: original.body,
        diagnosticFilter: original.diagnosticFilter, location: original.location)
    let module = IRModule(types: parsed.types, specialTypes: parsed.specialTypes, constants: parsed.constants,
        overrides: parsed.overrides, globalVariables: parsed.globalVariables, globalExpressions: parsed.globalExpressions,
        functions: .init([function]), entryPoints: parsed.entryPoints, diagnosticFilters: parsed.diagnosticFilters,
        diagnosticFilter: parsed.diagnosticFilter, documentation: parsed.documentation,
        allowsPointerSubviews: parsed.allowsPointerSubviews, uniformBufferStandardLayout: parsed.uniformBufferStandardLayout,
        requiresSubgroupUniformity: parsed.requiresSubgroupUniformity)
    let validated = try ValidatedModule.checking(module, with: ShaderSemanticValidator(capabilities: [.cooperativeMatrix]))
    guard case .spirv(let words) = try SPIRVCodeGenerator().generate(validated) else { throw ShaderFailure.incompatibleArtifact }
    var offset = 5
    var opcodes: [UInt32] = []
    while offset < words.count {
        let count = Int(words[offset] >> 16)
        try #require(count > 0 && count <= words.count - offset)
        opcodes.append(words[offset] & 0xffff)
        offset += count
    }
    #expect(opcodes.contains(4457))
    #expect(opcodes.contains(250))
}

@Test func spirvCooperativeMatrixPreservesShapeLayoutAndMixedAccumulator() throws {
    let source = """
        enable f16;
        enable wgpu_cooperative_matrix;
        @group(0) @binding(0) var<storage, read> factors: array<f16>;
        @group(0) @binding(1) var<storage, read_write> accumulators: array<f32, 256>;

        @compute @workgroup_size(8, 8)
        fn main() {
            let a = coopLoadT<coop_mat16x8<f16, A>>(&factors[0]);
            let b = coopLoad<coop_mat8x16<f16, B>>(&factors[128]);
            let c = coopLoadT<coop_mat8x8<f32, C>>(&accumulators[0], 12u);
            coopStore(coopMultiplyAdd(a, b, c), &accumulators[128]);
        }
        """
    let artifact = try IndependentShaderCompiler().compile(.wgsl(source), to: .spirv)
    guard case .spirv(let words) = artifact else { throw ShaderFailure.incompatibleArtifact }
    var instructions: [UInt32: [[UInt32]]] = [:]
    var offset = 5
    while offset < words.count {
        let count = Int(words[offset] >> 16)
        try #require(count > 0 && count <= words.count - offset)
        instructions[words[offset] & 0xffff, default: []].append(Array(words[(offset + 1)..<(offset + count)]))
        offset += count
    }

    let capabilities = Set((instructions[17] ?? []).compactMap(\.first))
    #expect(capabilities.isSuperset(of: [9, 4433, 5345, 6022])) // Float16, StorageBuffer16BitAccess, VulkanMemoryModel, CooperativeMatrixKHR.
    #expect(!capabilities.contains(11)) // Full-footprint guarding does not need Int64.
    #expect(!capabilities.contains(5346)) // Device scope capability is not needed without storage atomics.
    #expect(instructions[14] == [[0, 3]]) // Logical VulkanKHR.
    let extensions = (instructions[10] ?? []).map {
        String(decoding: $0.flatMap { word in (0..<4).map { UInt8(truncatingIfNeeded: word >> ($0 * 8)) } }
            .prefix(while: { $0 != 0 }), as: UTF8.self)
    }
    #expect(extensions.contains("SPV_KHR_cooperative_matrix"))
    #expect(extensions.contains("SPV_KHR_vulkan_memory_model"))

    let constants = Dictionary(uniqueKeysWithValues: (instructions[43] ?? []).filter { $0.count == 3 }.map { ($0[1], $0[2]) })
    let scalarWidths = Dictionary(uniqueKeysWithValues: (instructions[22] ?? []).map { ($0[0], $0[1]) })
    let types = try #require(instructions[4456])
    try #require(types.count == 3 && types.allSatisfy { $0.count == 6 })
    let typesByRole = Dictionary(uniqueKeysWithValues: types.map { (constants[$0[5]], $0) })
    for (role, width, rows, columns) in [(0, 16, 8, 16), (1, 16, 16, 8), (2, 32, 8, 8)] {
        let type = try #require(typesByRole[UInt32(role)])
        #expect(scalarWidths[type[1]] == UInt32(width))
        #expect(constants[type[2]] == 3) // Subgroup.
        #expect(constants[type[3]] == UInt32(rows))
        #expect(constants[type[4]] == UInt32(columns))
    }

    let loads = try #require(instructions[4457])
    try #require(loads.count == 3 && loads.allSatisfy { $0.count == 6 })
    #expect(loads.map { constants[$0[3]] } == [0, 1, 0])
    #expect(loads.map { constants[$0[4]] } == [16, 16, 12])
    let multiply = try #require(instructions[4459]?.onlyElement)
    try #require(multiply.count == 5)
    #expect(multiply[0] == typesByRole[2]?[0])
    let store = try #require(instructions[4458]?.onlyElement)
    try #require(store.count == 4)
    #expect(store[1] == multiply[1])
    #expect(constants[store[2]] == 1)
    #expect(constants[store[3]] == 8)
    #expect(instructions[134]?.count == 4) // OpUDiv bounds the full footprint before each transfer.
}

private extension Array {
    var onlyElement: Element? { count == 1 ? first : nil }
}

@Test func spirvCooperativeLoadAllowsZeroStrideButStoreGuardsIt() throws {
    let source = """
        enable wgpu_cooperative_matrix;
        @group(0) @binding(0) var<storage, read> input: array<f32>;
        @group(0) @binding(1) var<storage, read_write> output: array<f32, 64>;
        @compute @workgroup_size(8, 8)
        fn main() {
            let tile = coopLoad<coop_mat8x8<f32, A>>(&input[0], 0u);
            coopStore(tile, &output[0], 0u);
        }
        """
    let artifact = try IndependentShaderCompiler().compile(.wgsl(source), to: .spirv)
    guard case .spirv(let words) = artifact else { throw ShaderFailure.incompatibleArtifact }
    var instructions: [UInt32: [[UInt32]]] = [:]
    var offset = 5
    while offset < words.count {
        let count = Int(words[offset] >> 16)
        try #require(count > 0 && count <= words.count - offset)
        instructions[words[offset] & 0xffff, default: []].append(Array(words[(offset + 1)..<(offset + count)]))
        offset += count
    }
    let constants = Dictionary(uniqueKeysWithValues: (instructions[43] ?? []).filter { $0.count == 3 }.map { ($0[1], $0[2]) })
    let load = try #require(instructions[4457]?.onlyElement)
    #expect(load.count == 6)
    #expect(constants[load[4]] == 0) // KHR load permits a zero element stride.
    let store = try #require(instructions[4458]?.onlyElement)
    #expect(store.count == 4)
    #expect(constants[store[3]] == 1) // The guarded false path passes a spec-valid positive operand.
    #expect((instructions[250] ?? []).contains { $0.count == 3 }) // OpBranchConditional guards the store.
}

@Test func spirvCooperativeLoadRejectsUntrackedStructMemberFootprint() throws {
    let source = """
        enable wgpu_cooperative_matrix;
        struct ScalarBuffer { value: f32, }
        @group(0) @binding(0) var<storage, read> input: ScalarBuffer;
        @compute @workgroup_size(8, 8)
        fn main() {
            let tile = coopLoad<coop_mat8x8<f32, A>>(&input.value);
        }
        """
    #expect(throws: ShaderFailure.unsupported("SPIR-V writer cooperative matrix footprint requires an indexed array")) {
        try IndependentShaderCompiler().compile(.wgsl(source), to: .spirv)
    }
}

@Test func spirvStorageDeviceScopeCapabilityRequiresVulkanCooperativeMemoryModel() throws {
    let atomicOnly = """
        @group(0) @binding(0) var<storage, read_write> counter: atomic<u32>;
        @group(0) @binding(1) var<storage, read_write> output: array<u32, 1>;
        @compute @workgroup_size(1)
        fn main() { output[0] = atomicAdd(&counter, 1u); }
        """
    let combined = """
        enable wgpu_cooperative_matrix;
        @group(0) @binding(0) var<storage, read_write> counter: atomic<u32>;
        @group(0) @binding(1) var<storage, read> input: array<f32>;
        @group(0) @binding(2) var<storage, read_write> output: array<f32, 64>;
        @compute @workgroup_size(8, 8)
        fn main() {
            let previous = atomicAdd(&counter, 1u);
            output[0] = f32(previous);
            let tile = coopLoad<coop_mat8x8<f32, A>>(&input[0]);
            coopStore(tile, &output[0]);
        }
        """
    func capabilities(in words: [UInt32]) throws -> Set<UInt32> {
        var result = Set<UInt32>()
        var offset = 5
        while offset < words.count {
            let count = Int(words[offset] >> 16)
            try #require(count > 0 && count <= words.count - offset)
            if words[offset] & 0xffff == 17, count == 2 { result.insert(words[offset + 1]) }
            offset += count
        }
        return result
    }
    guard case .spirv(let atomicWords) = try IndependentShaderCompiler().compile(.wgsl(atomicOnly), to: .spirv),
          case .spirv(let combinedWords) = try IndependentShaderCompiler().compile(.wgsl(combined), to: .spirv) else {
        throw ShaderFailure.incompatibleArtifact
    }
    let atomicCapabilities = try capabilities(in: atomicWords)
    let combinedCapabilities = try capabilities(in: combinedWords)
    #expect(!atomicCapabilities.contains(5346))
    #expect(combinedCapabilities.contains(5346))
}

@Test func spirvImageDeviceScopeCapabilityRequiresVulkanCooperativeMemoryModel() throws {
    let imageAtomicOnly = """
        @group(0) @binding(0) var image: texture_storage_2d<r32uint, atomic>;
        @compute @workgroup_size(1)
        fn main() { textureAtomicAdd(image, vec2<i32>(0, 0), 1u); }
        """
    let combined = """
        enable wgpu_cooperative_matrix;
        @group(0) @binding(0) var image: texture_storage_2d<r32uint, atomic>;
        @group(0) @binding(1) var<storage, read> input: array<f32>;
        @group(0) @binding(2) var<storage, read_write> output: array<f32, 64>;
        @compute @workgroup_size(8, 8)
        fn main() {
            textureAtomicAdd(image, vec2<i32>(0, 0), 1u);
            let tile = coopLoad<coop_mat8x8<f32, A>>(&input[0]);
            coopStore(tile, &output[0]);
        }
        """
    func capabilities(in words: [UInt32]) throws -> Set<UInt32> {
        var result = Set<UInt32>()
        var offset = 5
        while offset < words.count {
            let count = Int(words[offset] >> 16)
            try #require(count > 0 && count <= words.count - offset)
            if words[offset] & 0xffff == 17, count == 2 { result.insert(words[offset + 1]) }
            offset += count
        }
        return result
    }
    guard case .spirv(let imageOnlyWords) = try IndependentShaderCompiler().compile(.wgsl(imageAtomicOnly), to: .spirv),
          case .spirv(let combinedWords) = try IndependentShaderCompiler().compile(.wgsl(combined), to: .spirv) else {
        throw ShaderFailure.incompatibleArtifact
    }
    let imageOnlyCapabilities = try capabilities(in: imageOnlyWords)
    let combinedCapabilities = try capabilities(in: combinedWords)
    #expect(!imageOnlyCapabilities.contains(5346))
    #expect(combinedCapabilities.contains(5346))
}

@Test func spirvCooperativeMatricesPreserveWorkgroupAndStorageBarriers() throws {
    let source = """
        enable wgpu_cooperative_matrix;
        @group(0) @binding(0) var<storage, read> input: array<f32>;
        @group(0) @binding(1) var<storage, read_write> output: array<f32, 64>;
        @compute @workgroup_size(8, 8)
        fn main() {
            workgroupBarrier();
            storageBarrier();
            let tile = coopLoad<coop_mat8x8<f32, A>>(&input[0]);
            coopStore(tile, &output[0]);
        }
        """
    let artifact = try IndependentShaderCompiler().compile(.wgsl(source), to: .spirv)
    guard case .spirv(let words) = artifact else { throw ShaderFailure.incompatibleArtifact }
    var instructions: [UInt32: [[UInt32]]] = [:]
    var offset = 5
    while offset < words.count {
        let count = Int(words[offset] >> 16)
        try #require(count > 0 && count <= words.count - offset)
        instructions[words[offset] & 0xffff, default: []].append(Array(words[(offset + 1)..<(offset + count)]))
        offset += count
    }
    let constants = Dictionary(uniqueKeysWithValues: (instructions[43] ?? []).filter { $0.count == 3 }.map { ($0[1], $0[2]) })
    #expect(instructions[14] == [[0, 3]]) // Logical VulkanKHR.
    let barriers = try #require(instructions[224])
    #expect(barriers.count == 2)
    #expect(barriers.compactMap { $0.count == 3 ? constants[$0[2]] : nil } == [0x108, 0x48])
}

@Test func spirvCooperativeMatricesRejectCoherentOrVolatileStorageDecorations() throws {
    let unsupported = ShaderFailure.unsupported(
        "SPIR-V writer does not support coherent or volatile storage decorations with the Vulkan memory model")
    for decoration in ["@coherent", "@volatile", "@coherent @volatile"] {
        let source = """
            enable wgpu_cooperative_matrix;
            @group(0) @binding(0) var<storage, read> input: array<f32>;
            @group(0) @binding(1) \(decoration) var<storage, read_write> output: array<f32, 64>;
            @compute @workgroup_size(8, 8)
            fn main() {
                storageBarrier();
                let tile = coopLoad<coop_mat8x8<f32, A>>(&input[0]);
                coopStore(tile, &output[0]);
            }
            """
        #expect(throws: unsupported) {
            try IndependentShaderCompiler().compile(.wgsl(source), to: .spirv)
        }
    }
}

@Test func spirvCooperativeMatrixNestedStorageArrayPreservesBounds() throws {
    let source = """
        enable wgpu_cooperative_matrix;
        struct Tile { values: array<f32, 128>, }
        struct Buffer { tiles: array<Tile, 4>, }
        @group(0) @binding(0) var<storage, read_write> buffer: Buffer;
        @compute @workgroup_size(8, 8)
        fn main(@builtin(workgroup_id) id: vec3<u32>) {
            let tile = coopLoad<coop_mat8x8<f32, A>>(&buffer.tiles[id.x].values[id.y], 8u);
            coopStore(tile, &buffer.tiles[id.x].values[id.y], 8u);
        }
        """
    guard case .spirv(let words) = try IndependentShaderCompiler().compile(.wgsl(source), to: .spirv) else {
        throw ShaderFailure.incompatibleArtifact
    }
    var counts: [UInt32: Int] = [:]
    var offset = 5
    while offset < words.count {
        let count = Int(words[offset] >> 16)
        try #require(count > 0 && count <= words.count - offset)
        counts[words[offset] & 0xffff, default: 0] += 1
        offset += count
    }
    #expect(counts[4457] == 1)
    #expect(counts[4458] == 1)
    #expect(counts[134] == 2)
    #expect((counts[250] ?? 0) >= 2)
}
