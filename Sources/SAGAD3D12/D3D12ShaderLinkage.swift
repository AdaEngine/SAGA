#if os(Windows)
enum D3D12ShaderLinkage {
    static func fragmentSource(_ fragment: String, matchingVertexSource vertex: String) -> String {
        let vertexLines = vertex.split(separator: "\n", omittingEmptySubsequences: false)
        var fragmentLines = fragment.split(separator: "\n", omittingEmptySubsequences: false)
        guard let outputRange = fields(of: "SagaVertexOutput", in: vertexLines),
              let inputRange = fields(of: "SagaFragmentInput", in: fragmentLines) else { return fragment }

        let outputs = vertexLines[outputRange].compactMap(Field.init)
        let inputs = fragmentLines[inputRange].compactMap(Field.init)
        guard outputs.count == outputRange.count, inputs.count == inputRange.count else { return fragment }

        let inputsBySemantic = Dictionary(uniqueKeysWithValues: inputs.map { ($0.semantic, $0.declaration) })
        let vertexSemantics = Set(outputs.map(\.semantic))
        guard let lastSharedIndex = outputs.lastIndex(where: { inputsBySemantic[$0.semantic] != nil }) else { return fragment }

        // DXC packs each stage separately. Preserve the vertex signature's order and retain unused
        // fields that precede a shared input so both stages assign that input the same register.
        var linked = outputs[...lastSharedIndex].enumerated().map { index, output in
            inputsBySemantic[output.semantic] ?? output.unusedDeclaration(index: index)
        }
        // The rasterizer provides system values such as SV_IsFrontFace.
        linked += inputs.filter { !vertexSemantics.contains($0.semantic) }.map(\.declaration)
        fragmentLines.replaceSubrange(inputRange, with: linked)
        return fragmentLines.joined(separator: "\n")
    }

    // The HLSL writer emits one declaration per line in these generated interface structures.
    private static func fields(of structure: String, in lines: [Substring]) -> Range<Int>? {
        guard let header = lines.firstIndex(of: "struct \(structure) {"[...]),
              let end = lines[(header + 1)...].firstIndex(of: "};") else { return nil }
        return (header + 1)..<end
    }

    private struct Field {
        let declaration: Substring
        let name: Substring
        let semantic: Substring

        init?(_ declaration: Substring) {
            let parts = declaration.split(separator: ":", maxSplits: 1)
            guard parts.count == 2, let name = parts[0].split(separator: " ").last,
                  let semantic = parts[1].split(whereSeparator: { $0 == " " || $0 == ";" || $0 == ":" }).first else { return nil }
            self.declaration = declaration
            self.name = name
            self.semantic = semantic
        }

        func unusedDeclaration(index: Int) -> Substring {
            String(declaration).replacingAll(" \(name) :", with: " saga_unused_input_\(index) :")[...]
        }
    }
}
#endif
