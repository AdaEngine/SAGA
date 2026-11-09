import SAGAShaderIR
import SAGATypes

struct MetalVertexPulling {
    let layouts: [VertexBufferLayout]

    func generate(
        _ inputs: [RenderStageArtifact.VertexInput], builtins: [IRBuiltIn]
    ) throws -> (parameters: [String], declarations: [String], initializers: [String]) {
        let formats: [UInt32: VertexFormat]
        do {
            formats = try VertexBufferLayout.validate(layouts,
                maxBuffers: RenderStageArtifact.metalVertexPullingSizeBufferIndex, maxAttributes: .max, maxStride: .max)
        } catch GPUError.validation(let message) {
            throw ShaderFailure.unsupported("MSL vertex pulling: \(message)")
        }
        guard !inputs.isEmpty else { return ([], [], []) }
        var locations: [UInt32: (slot: Int, offset: UInt32)] = [:]
        for (slot, layout) in layouts.enumerated() {
            for attribute in layout.attributes { locations[attribute.location] = (slot, attribute.offset) }
        }

        var parameters = layouts.indices.map { "device const uchar* saga_vertex_bytes_\($0) [[buffer(\($0))]]" }
        parameters.append("constant ulong* saga_vertex_sizes [[buffer(\(RenderStageArtifact.metalVertexPullingSizeBufferIndex))]]")
        if !builtins.contains(.vertexIndex), layouts.contains(where: { $0.stride != 0 && $0.stepMode == .vertex }) {
            parameters.append("uint saga_index [[vertex_id]]")
        }
        if !builtins.contains(.instanceIndex), layouts.contains(where: { $0.stride != 0 && $0.stepMode == .instance }) {
            parameters.append("uint saga_instance [[instance_id]]")
        }
        let declarations = ["""
            inline uint saga_vertex_read(device const uchar* bytes, uint count) {
                uint bits = 0u;
                for (uint i = 0u; i < count; ++i) {
                    bits |= uint(bytes[i]) << (i * 8u);
                }
                return bits;
            }
            """]
        var initializers = ["    SagaVertexInput saga_vertex_input = {};"]
        for input in inputs {
            let location = input.location
            guard let format = formats[location], let (slot, offset) = locations[location],
                  compatible(format.dataType, with: input.scalar) else {
                throw ShaderFailure.unsupported("MSL vertex pulling: missing or incompatible vertex attribute at location \(location)")
            }
            let layout = layouts[slot]
            let index = layout.stepMode == .instance ? "saga_instance" : "saga_index"
            let address = layout.stride == 0 ? "\(offset)ul" : "ulong(\(index)) * \(layout.stride)ul + \(offset)ul"
            let offsetName = "saga_vertex_offset_\(location)"
            let size = "saga_vertex_sizes[\(slot)]"
            let bytes = "saga_vertex_data_\(location)"
            initializers += [
                "    const ulong \(offsetName) = \(address);",
                "    if (\(offsetName) <= \(size) && \(format.byteSize)ul <= \(size) - \(offsetName)) {",
                "        device const uchar* \(bytes) = saga_vertex_bytes_\(slot) + \(offsetName);",
            ]
            let packed = format == .unorm1010102 || format == .snorm1010102
            let packedWord = "saga_vertex_packed_\(location)"
            if packed { initializers.append("        const uint \(packedWord) = saga_vertex_read(\(bytes), 4u);") }
            let lanes = (0..<UInt32(input.componentCount)).map { lane -> String in
                guard lane < format.componentCount else { return lane == 3 ? "1" : "0" }
                let bits: UInt32
                let word: String
                if packed {
                    bits = lane == 3 ? 2 : 10
                    let mask = (1 << bits) - 1
                    word = "((\(packedWord) >> \(lane * 10)u) & \(mask)u)"
                } else {
                    let laneBytes = format.byteSize / format.componentCount
                    bits = laneBytes * 8
                    let offset = format == .unorm8x4BGRA && lane < 3 ? 2 - lane : lane
                    word = "saga_vertex_read(\(bytes) + \(offset * laneBytes)u, \(laneBytes)u)"
                }
                return component(format, word: word, bits: bits)
            }
            let type = try shaderVectorType(input.componentCount, scalar: input.scalar, in: .msl)
            initializers.append("        saga_vertex_input.saga_attribute_\(location) = \(type)(\(lanes.joined(separator: ", ")));")
            initializers.append("    }")
        }
        return (parameters, declarations, initializers)
    }

    private func compatible(_ format: VertexDataType, with scalar: IRScalar) -> Bool {
        switch (format, scalar) {
        case (.float, .f32), (.float, .f16), (.sint, .i32), (.uint, .u32): true
        default: false
        }
    }

    private func component(_ format: VertexFormat, word: String, bits: UInt32) -> String {
        let shift = 32 - bits
        let signed = shift == 0 ? "as_type<int>(\(word))" : "(as_type<int>(\(word) << \(shift)u) >> \(shift)u)"
        switch format {
        case .float16, .float16x2, .float16x4: return "as_type<half>(ushort(\(word)))"
        case .float32, .float32x2, .float32x3, .float32x4: return "as_type<float>(\(word))"
        case .snorm8, .snorm8x2, .snorm8x4, .snorm16, .snorm16x2, .snorm16x4, .snorm1010102:
            return "max(-1.0f, float(\(signed)) / \((1 << (bits - 1)) - 1).0f)"
        case .unorm8, .unorm8x2, .unorm8x4, .unorm16, .unorm16x2, .unorm16x4, .unorm1010102, .unorm8x4BGRA:
            return "(float(\(word)) / \((1 << bits) - 1).0f)"
        default:
            return format.dataType == .sint ? signed : word
        }
    }
}
