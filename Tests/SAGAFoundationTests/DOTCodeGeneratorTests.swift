import SAGACompiler
import SAGAShaderIR
import Testing

private let dotInput = ShaderInput.wgsl("""
    @id(11) override unresolvedWidth: u32;
    @id(12) override gain = 7u;
    @group(0) @binding(0) var<storage, read_write> output: u32;

    fn dormant(deadArgument: u32) -> u32 { return deadArgument ^ gain; }
    fn stepValue(operand: u32) -> u32 { return operand + gain; }

    @vertex fn draw(@builtin(vertex_index) vertex: u32) -> @builtin(position) vec4f {
        return vec4f(f32(stepValue(vertex)), 0, 0, 1);
    }
    @fragment fn shade() -> @location(0) vec4f { return vec4f(1); }
    @compute @workgroup_size(unresolvedWidth) fn simulate(@builtin(local_invocation_index) lane: u32) {
        var index = lane;
        loop {
            if index == 2u { break; }
            switch index {
                case 0u: { index = stepValue(index); continue; }
                case 1u: { break; }
                default: { index += 1u; }
            }
            index += 2u;
            continuing { index += 1u; break if index > 4u; }
        }
        output = index;
    }
    """)

private struct DOTGraph {
    struct Edge: Hashable {
        let from: String
        let to: String
        let label: String?
    }

    var nodes: [String: String] = [:]
    var edges: [Edge] = []

    init(_ source: String) {
        for line in source.split(separator: "\n") {
            let tokens = line.prefix { $0 != "[" }.split { $0.isWhitespace || $0 == ";" }
            let label = line.firstIndex(of: "[").flatMap { start -> String? in
                let attribute = line[line.index(after: start)...].drop(while: \.isWhitespace)
                let prefix = #"label=""#
                guard attribute.hasPrefix(prefix) else { return nil }
                return String(attribute.dropFirst(prefix.count).prefix { $0 != "\"" })
            }
            if tokens.count == 3, tokens[1] == "->" {
                edges.append(.init(from: String(tokens[0]), to: String(tokens[2]), label: label))
            } else if tokens.count == 1, let label {
                nodes[String(tokens[0])] = label
            }
        }
    }

    func node(withLabel prefix: String) -> String? {
        nodes.first { $0.value.hasPrefix(prefix) }?.key
    }
}

private func dotText(_ artifact: ShaderArtifact) throws -> String {
    guard case .text(.dot, let source) = artifact else { throw ShaderFailure.incompatibleArtifact }
    return source
}

@Test func dotCodeGeneratorPreservesAllEntriesAndUnspecializedData() throws {
    let compiler = IndependentShaderCompiler()
    let artifact = try compiler.compile(dotInput, to: .dot)
    let source = try dotText(artifact)
    let graph = DOTGraph(source)
    #expect(source.hasPrefix("digraph "))
    #expect(try compiler.compile(dotInput, to: .dot) == artifact)
    for (stage, name) in [("vertex", "draw"), ("fragment", "shade"), ("compute", "simulate")] {
        let entry = try #require(graph.node(withLabel: "@\(stage) \(name)"))
        let function = try #require(graph.node(withLabel: "Function \(name)"))
        #expect(graph.edges.contains { $0.from == entry && $0.to == function })
    }
    #expect(graph.nodes.values.contains { $0.contains("unresolvedWidth") })
    let gain = try #require(graph.nodes.first { $0.key.hasPrefix("o") && $0.value.contains("gain") }?.key)
    let dormant = try #require(graph.node(withLabel: "Function dormant"))
    #expect(graph.nodes["\(dormant)_arg0"]?.contains("deadArgument") == true)
    let step = try #require(graph.node(withLabel: "Function stepValue"))
    #expect(graph.nodes["\(step)_arg0"]?.contains("operand") == true)
    let addition = try #require(graph.nodes.first { $0.key.hasPrefix("\(step)_e") && $0.value.contains("binary add") }?.key)
    #expect(Set(graph.edges.filter { $0.from == addition }.compactMap(\.label)).isSuperset(of: ["operand 0", "operand 1"]))
    #expect(graph.edges.contains { $0.to == gain && $0.label == "reference" })
    #expect(graph.edges.contains { $0.to == "\(step)_arg0" && $0.label == "reference" })
    #expect(graph.edges.filter { $0.to == step && $0.label == "call" }.count == 2)
    let compute = try #require(graph.node(withLabel: "@compute simulate"))
    let workgroup = try #require(graph.edges.first { $0.from == compute && $0.label == "workgroup x" })
    let width = try #require(graph.nodes.first { $0.key.hasPrefix("o") && $0.value.contains("unresolvedWidth") }?.key)
    #expect(graph.edges.contains { $0.from == workgroup.to && $0.to == width && $0.label == "reference" })
}

@Test func dotCodeGeneratorLinksStructuredControlFlowTargets() throws {
    let source = try dotText(IndependentShaderCompiler().compile(dotInput, to: .dot))
    let graph = DOTGraph(source)
    let function = try #require(graph.node(withLabel: "Function simulate"))
    let edges = graph.edges.filter { $0.from.hasPrefix("\(function)_") }
    let loop = try #require(edges.first { $0.label == "body" })
    let breakIf = "\(loop.from)_break_if"
    #expect(graph.nodes[breakIf]?.contains("break if") == true)
    let leaveLoop = try #require(edges.first { $0.from == breakIf && $0.label == "true" })
    #expect(edges.contains { $0.from == breakIf && $0.to == loop.to && $0.label == "false" })
    #expect(edges.contains { $0.to == leaveLoop.to && $0.label == "break" })
    let continuing = try #require(edges.first { $0.label == "continuing" })
    #expect(edges.contains { $0.to == continuing.to && $0.label == "continue" })
    #expect(edges.contains { $0.to != leaveLoop.to && $0.to != continuing.to && $0.label == "break" })
    let conditional = try #require(edges.first { $0.label == "true" && $0.from != breakIf })
    #expect(edges.contains { $0.from == conditional.from && $0.to != conditional.to && $0.label == "false" })
    #expect(Set(edges.compactMap(\.label)).isSuperset(of: ["case 0u", "case 1u", "default"]))
}

@Test func dotCodeGeneratorCFGOnlyPreservesControlFlowWithoutDataEdges() throws {
    let compiler = IndependentShaderCompiler()
    let fullArtifact = try compiler.compile(dotInput, to: .dot)
    #expect(try compiler.compileDOT(dotInput) == fullArtifact)
    let full = DOTGraph(try dotText(fullArtifact))
    let cfg = DOTGraph(try dotText(compiler.compileDOT(dotInput, cfgOnly: true)))
    #expect(cfg.nodes.count < full.nodes.count)
    #expect(cfg.node(withLabel: "@compute simulate") != nil)
    #expect(cfg.node(withLabel: "Function dormant") != nil)
    #expect(cfg.nodes.keys.allSatisfy { $0.hasPrefix("f") || $0.hasPrefix("ep") })
    #expect(!cfg.nodes.keys.contains { $0.contains("_e") && !$0.hasSuffix("_exit") || $0.contains("_arg") || $0.contains("_local") })
    let controlEdges = full.edges.filter { cfg.nodes[$0.from] != nil && cfg.nodes[$0.to] != nil && $0.label != "call" }
    #expect(Set(cfg.edges) == Set(controlEdges))
}

@Test func dotCodeGeneratorReportsInvalidShaderSource() throws {
    let invalid = ShaderInput.wgsl("@compute @workgroup_size(1) fn invalid() { let number: u32 = true; }")
    do {
        _ = try IndependentShaderCompiler().compile(invalid, to: .dot)
        Issue.record("DOT compilation accepted invalid WGSL")
    } catch ShaderFailure.diagnostics(let diagnostics) {
        #expect(diagnostics.contains { $0.code == .typeMismatch && $0.span != nil })
    }
}

@Test func dotCodeGeneratorEscapesSPIRVNames() throws {
    var name = Array("quoted\"slash\\line\nend".utf8) + [0]
    while !name.count.isMultiple(of: 4) { name.append(0) }
    let packed = packLittleEndianWords(name)
    let i = SPIRVFrontendTests.i
    var words: [UInt32] = SPIRVFrontendTests.pre
    words.append(UInt32(packed.count + 3) << 16 | 15)
    words += [5, 10]
    words += packed
    for instruction in [i(16, 10, 17, 1, 1, 1), i(19, 1), i(33, 2, 1), i(54, 1, 10, 0, 2), i(248, 11), i(253), i(56)] {
        words += instruction
    }
    let source = try dotText(IndependentShaderCompiler().compile(.spirv(words), to: .dot))
    #expect(source.contains(#"quoted\"slash\\line\nend"#))
    #expect(!source.contains("line\nend"))
}
