import Testing
import SAGACompiler
import SAGAShaderIR

private indirect enum BufferLayout: Equatable {
    case leaf
    case array(count: UInt32?, stride: UInt32, element: BufferLayout)
    case structure(offsets: [UInt32], span: UInt32, members: [BufferLayout])
}

private func bufferLayout(_ module: IRModule, _ type: IRHandle<IRType>) throws -> BufferLayout {
    switch try module.types.value(for: type).kind {
    case .structure(let members, let span):
        return .structure(offsets: members.map(\.offset), span: span,
                          members: try members.map { try bufferLayout(module, $0.type) })
    case .array(let element, let size, let stride):
        let count: UInt32?
        switch size {
        case .constant(let value): count = value
        case .dynamic: count = nil
        default: throw LayoutTestError.unspecializedArray
        }
        return try .array(count: count, stride: stride, element: bufferLayout(module, element))
    default:
        return .leaf
    }
}

private enum LayoutTestError: Error { case unspecializedArray }

private func storageLayout(_ module: IRModule) throws -> BufferLayout {
    let global = try #require(module.globalVariables.values.first { $0.binding != nil })
    return try bufferLayout(module, global.type)
}

@Test func wgslNestedExplicitAlignmentRoundTripsAcrossStages() throws {
    let declarations = """
        struct Cell { @align(32) value: vec4u, @size(16) tail: u32, }
        struct Data { cells: array<Cell, 2>, }
        @group(0) @binding(0) var<storage, read> data: Data;
        """
    let compute = declarations + """
        @compute @workgroup_size(1) fn main() { let x = data.cells[1].tail; }
        """
    let render = declarations + """
        @vertex fn main() -> @builtin(position) vec4f {
            return vec4f(f32(data.cells[1].tail));
        }
        """
    let compiler = IndependentShaderCompiler()
    let artifacts = [try compiler.compileCompute(.wgsl(compute), to: .wgsl).artifact,
                     try compiler.compileRenderStage(.wgsl(render), entryPoint: "main", to: .wgsl).artifact]
    for artifact in artifacts {
        guard case .text(_, let source) = artifact else { Issue.record("Expected WGSL"); continue }
        #expect(source.contains("@align(32)"))
        let original = try compiler.parseAndValidate(.wgsl(compute)).module
        let roundTrip = try compiler.parseAndValidate(.wgsl(source)).module
        #expect(try storageLayout(original) == storageLayout(roundTrip))
    }
}

@Test func wgslExplicitLayoutRuntimeArrayRoundTrips() throws {
    let source = """
        struct Cell { @align(32) value: vec4u, @size(16) tail: u32, }
        struct Data { @size(16) header: u32, cells: array<Cell>, }
        @group(0) @binding(0) var<storage, read> data: Data;
        @compute @workgroup_size(1) fn main() { let x = data.cells[0].tail; }
        """
    let compiler = IndependentShaderCompiler()
    let original = try compiler.parseAndValidate(.wgsl(source)).module
    let artifact = try compiler.compileCompute(.wgsl(source), to: .wgsl).artifact
    guard case .text(_, let output) = artifact else { Issue.record("Expected WGSL"); return }
    let roundTrip = try compiler.parseAndValidate(.wgsl(output)).module
    #expect(try storageLayout(original) == storageLayout(roundTrip))
}
