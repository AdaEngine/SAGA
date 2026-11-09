import SAGAShaderIR
import SAGAFrontends
import Testing

@Suite struct WGSLCooperativeMatrixFrontendTests {
    private let frontend = WGSLFrontend(options: .init(capabilities: [.cooperativeMatrix, .shaderFloat16]))

    @Test func preservesPrivateMatricesAndDefaultElementStrides() throws {
        let source = """
            enable wgpu_cooperative_matrix;
            var<private> left: coop_mat8x8<f32, A>;
            var<private> right: coop_mat8x8<f32, B>;
            @group(0) @binding(0) var<storage, read_write> buffer: array<f32>;
            @compute @workgroup_size(8, 8, 1) fn main() {
                var accumulator = coopLoad<coop_mat8x8<f32, C>>(&buffer[4]);
                var result = coopMultiplyAdd(left, right, accumulator);
                coopStore(result, &buffer[0]);
                accumulator = result;
            }
            """
        let module = try frontend.parse(.wgsl(source))
        #expect(module.globalVariables.values.prefix(2).map {
            module.types.values[$0.type.index].kind
        } == [
            .cooperativeMatrix(columns: .eight, rows: .eight, scalar: .f32, role: .a),
            .cooperativeMatrix(columns: .eight, rows: .eight, scalar: .f32, role: .b),
        ])
        let function = try #require(module.functions.values.first)
        let load = try #require(function.expressions.values.first {
            if case .cooperativeLoad = $0.kind { true } else { false }
        })
        guard case .cooperativeLoad(.eight, .eight, .c, let input) = load.kind else {
            Issue.record("Expected an 8×8 accumulator load")
            return
        }
        let accumulator = IRTypeKind.cooperativeMatrix(columns: .eight, rows: .eight, scalar: .f32, role: .c)
        #expect(module.types.values[load.type.index].kind == accumulator)
        let multiply = try #require(function.expressions.values.first {
            if case .cooperativeMultiplyAdd = $0.kind { true } else { false }
        })
        #expect(module.types.values[multiply.type.index].kind == accumulator)
        let output = try #require(function.statements.values.compactMap { statement -> IRCooperativeData? in
            if case .cooperativeStore(_, let data) = statement.kind { data } else { nil }
        }.first)
        for data in [input, output] {
            #expect(!data.rowMajor)
            #expect(function.expressions.values[data.stride.index].kind == .literal(.u32(8)))
            let pointerType = function.expressions.values[data.pointer.index].type
            guard case .pointer(let pointee, .storage(let access)) = module.types.values[pointerType.index].kind else {
                Issue.record("Expected a storage element pointer")
                continue
            }
            #expect(access.contains([.load, .store]))
            #expect(module.types.values[pointee.index].kind == .scalar(.f32))
        }
    }

    @Test func preservesTransposedLayoutsExplicitStridesAndMixedAccumulator() throws {
        let module = try frontend.parse(.wgsl("""
            enable f16, wgpu_cooperative_matrix;
            alias HalfLeft = coop_mat16x16<f16, A>;
            @group(0) @binding(0) var<storage, read> inputs: array<f16>;
            @group(0) @binding(1) var<storage, read_write> sums: array<f32>;
            @compute @workgroup_size(32) fn main() {
                let left = coopLoadT<HalfLeft>(&inputs[0], 32);
                let right = coopLoad<coop_mat16x16<f16, B>>(&inputs[16]);
                let accumulator = coopLoad<coop_mat16x16<f32, C>>(&sums[0]);
                coopStoreT(coopMultiplyAdd(left, right, accumulator), &sums[8], 48u);
            }
            """))
        let function = try #require(module.functions.values.first)
        let loads = function.expressions.values.filter {
            if case .cooperativeLoad = $0.kind { true } else { false }
        }
        #expect(loads.map { module.types.values[$0.type.index].kind } == [
            .cooperativeMatrix(columns: .sixteen, rows: .sixteen, scalar: .f16, role: .a),
            .cooperativeMatrix(columns: .sixteen, rows: .sixteen, scalar: .f16, role: .b),
            .cooperativeMatrix(columns: .sixteen, rows: .sixteen, scalar: .f32, role: .c),
        ])
        let transposedLoad = try #require(loads.first)
        guard case .cooperativeLoad(_, _, _, let input) = transposedLoad.kind else { return }
        #expect(input.rowMajor)
        #expect(function.expressions.values[input.stride.index].kind == .literal(.u32(32)))
        let output = try #require(function.statements.values.compactMap { statement -> IRCooperativeData? in
            if case .cooperativeStore(_, let data) = statement.kind { data } else { nil }
        }.first)
        #expect(output.rowMajor)
        #expect(function.expressions.values[output.stride.index].kind == .literal(.u32(48)))
    }

    @Test func rejectsDisabledExtensionAndMalformedOperands() throws {
        #expect(throws: ShaderFailure.self) {
            try WGSLFrontend(options: .init(capabilities: [])).parse(.wgsl("enable wgpu_cooperative_matrix;"))
        }
        for source in [
            "alias M = coop_mat8x8<f32, A>;",
            "fn f() { _ = coopMultiplyAdd(1, 2, 3); }",
        ] {
            #expect(throws: ShaderFailure.self) { try frontend.parse(.wgsl(source)) }
        }
        for source in [
            "alias M = coop_mat8x8<f32, D>;",
            "alias M = coop_mat8x8<u32, A>;",
            "fn f() { var data: array<f32, 64>; _ = coopLoad<coop_mat8x8<f32, C>>(&data); }",
            "fn f() { var data: array<f32, 64>; _ = coopLoad<coop_mat8x8<f32, C>>(&data[0], 1.0); }",
            """
            @group(0) @binding(0) var<storage, read> data: array<f32>;
            fn f(m: coop_mat8x8<f32, C>) { coopStore(m, &data[0]); }
            """,
        ] {
            #expect(throws: ShaderFailure.self) {
                try frontend.parse(.wgsl("enable wgpu_cooperative_matrix; " + source))
            }
        }
    }
}
