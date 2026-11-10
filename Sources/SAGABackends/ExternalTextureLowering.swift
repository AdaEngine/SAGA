import SAGAShaderIR

struct ExternalTextureReference {
    let planes: [String]
    let parameters: String

    var arguments: String { (planes + [parameters]).joined(separator: ", ") }
}

struct ExternalTextureShaderBinding {
    let binding: IRResourceBinding
    let declarations: [String]
    let parameters: [String]
    let captures: [ShaderParameter]
    let storage: ShaderStorage
    let layout: ComputeShaderArtifact.ExternalTextureLayout?

    init(
        global: IRGlobalVariable, index: Int, format: ShaderOutputFormat, planeBase: UInt32?, parametersIndex: UInt32?
    ) throws {
        guard let binding = global.binding, binding.group < ComputeShaderArtifact.portableBindGroupCount,
              binding.binding < ComputeShaderArtifact.portableBindingIndexLimit else {
            throw ShaderFailure.unsupported("Invalid external texture binding")
        }
        self.binding = binding
        let name = "saga_external_texture_\(index)"
        if format == .wgsl {
            declarations = ["@group(\(binding.group)) @binding(\(binding.binding)) var \(name): texture_external;"]
            parameters = []
            captures = []
            storage = .init(reference: name, writable: false)
            layout = nil
            return
        }
        guard let language = ExternalTextureLanguage(format) else {
            throw ShaderFailure.unsupported("External texture output format")
        }
        let planes = (0..<3).map { "\(name)_plane_\($0)" }
        let base: UInt32
        let uniform: UInt32
        if format == .hlsl {
            base = hlslBindingArrayRegisterBase(binding)
            uniform = base
        } else {
            guard let planeBase, let parametersIndex else { throw ShaderFailure.invalidHandle }
            base = planeBase
            uniform = parametersIndex
        }
        let units = (0..<3).map { base + $0 }
        let words = "\(name)_words"
        let reference: String
        switch format {
        case .msl:
            captures = planes.map { .init(type: language.texture, name: $0) } +
                [.init(type: "constant uint4*", name: words)]
            parameters = zip(planes, units).map { "\(language.texture) \($0.0) [[texture(\($0.1))]]" } +
                ["constant uint4* \(words) [[buffer(\(uniform))]]"]
            declarations = []
            reference = words
        case .hlsl:
            parameters = []
            captures = []
            declarations = zip(planes, units).map {
                "Texture2D<float4> \($0.0) : register(t\($0.1), space\(binding.group));"
            } + ["cbuffer \(name)_parameters : register(b\(uniform), space\(binding.group)) { uint4 \(words)[13]; };"]
            reference = words
        case .glsl:
            parameters = []
            captures = []
            declarations = zip(planes, units).map { "layout(binding = \($0.1)) uniform sampler2D \($0.0);" } +
                ["layout(std140, binding = \(uniform)) uniform SagaExternalParameters\(index) { uvec4 words[13]; } \(words);"]
            reference = words + ".words"
        default: throw ShaderFailure.invalidHandle
        }
        storage = .init(reference: name, writable: false, externalTexture: .init(planes: planes, parameters: reference))
        layout = .init(binding: binding, planeBindingIndices: units, parametersBindingIndex: uniform)
    }
}

private struct ExternalTextureLanguage {
    let format: ShaderOutputFormat

    init?(_ format: ShaderOutputFormat) {
        guard format == .msl || format == .hlsl || format == .glsl else { return nil }
        self.format = format
    }

    var texture: String {
        switch format {
        case .msl: "texture2d<float, access::sample>"
        case .hlsl: "Texture2D<float4>"
        default: "sampler2D"
        }
    }

    var sampler: String {
        switch format {
        case .msl: ", sampler state"
        case .hlsl: ", SamplerState state"
        default: ""
        }
    }

    var words: String {
        switch format {
        case .msl: "constant uint4* words"
        case .hlsl: "uint4 words[13]"
        default: "uvec4 words[13]"
        }
    }

    var f2: String { format == .glsl ? "vec2" : "float2" }
    var f3: String { format == .glsl ? "vec3" : "float3" }
    var f4: String { format == .glsl ? "vec4" : "float4" }
    var u2: String { format == .glsl ? "uvec2" : "uint2" }
    func floatWords(_ index: Int) -> String {
        switch format {
        case .msl: "as_type<float4>(words[\(index)])"
        case .hlsl: "asfloat(words[\(index)])"
        default: "uintBitsToFloat(words[\(index)])"
        }
    }

    func sample(_ plane: String, coordinate: String) -> String {
        switch format {
        case .msl: "\(plane).sample(state, \(coordinate), level(0.0f))"
        case .hlsl: "\(plane).SampleLevel(state, \(coordinate), 0.0)"
        default: "textureLod(\(plane), \(coordinate), 0.0)"
        }
    }

    func load(_ plane: String, coordinate: String) -> String {
        switch format {
        case .msl: "\(plane).read(\(coordinate), 0u)"
        case .hlsl: "\(plane).Load(int3(\(coordinate), 0))"
        default: "texelFetch(\(plane), ivec2(\(coordinate)), 0)"
        }
    }

    var sizeBody: String {
        switch format {
        case .msl: "return uint2(plane.get_width(), plane.get_height());"
        case .hlsl: "uint width, height; plane.GetDimensions(width, height); return uint2(width, height);"
        default: "return uvec2(textureSize(plane, 0));"
        }
    }

    var compareExtent: String { format == .glsl ? "any(greaterThanEqual(pixel, size))" : "any(pixel >= size)" }
    var hasLogicalSize: String { format == .glsl ? "all(notEqual(words[12].xy, uvec2(0u, 0u)))" : "all(words[12].xy != uint2(0u, 0u))" }
}

func externalTexturePrelude(_ format: ShaderOutputFormat) -> [String] {
    guard let lang = ExternalTextureLanguage(format) else { return [] }
    let f2 = lang.f2, f3 = lang.f3, f4 = lang.f4, u2 = lang.u2
    let words = lang.words
    let planes = " \(lang.texture) p0, \(lang.texture) p1, \(lang.texture) p2"
    let bit = lang.floatWords
    // The block is 13 vectors of raw words, using the offsets of Naga's external texture parameters.
    return ["""
        \(u2) saga_external_plane_size(\(lang.texture) plane) { \(lang.sizeBody) }
        \(u2) saga_external_dimensions(\(lang.texture) p0, \(words)) {
            return \(lang.hasLogicalSize) ? words[12].xy : saga_external_plane_size(p0);
        }
        \(f4) saga_external_convert(float y, \(f2) uv, \(words)) {
            \(f3) encoded = (\(bit(0)) * y + \(bit(1)) * uv.x + \(bit(2)) * uv.y + \(bit(3))).xyz;
            \(f4) source = \(bit(7));
            \(f4) target = \(bit(8));
            \(f3) lin;
            for (int inputLane = 0; inputLane < 3; ++inputLane) {
                float sourceValue = encoded[inputLane];
                lin[inputLane] = sourceValue < source.w * source.y ? sourceValue / source.w :
                    pow(max((sourceValue + source.x - 1.0) / source.x, 0.0), source.z);
            }
            \(f3) converted = \(bit(4)).xyz * lin.x + \(bit(5)).xyz * lin.y + \(bit(6)).xyz * lin.z;
            for (int outputLane = 0; outputLane < 3; ++outputLane) {
                float destinationValue = converted[outputLane];
                encoded[outputLane] = destinationValue < target.y ? target.w * destinationValue :
                    target.x * pow(max(destinationValue, 0.0), 1.0 / target.z) - (target.x - 1.0);
            }
            return \(f4)(encoded, 1.0);
        }
        \(f2) saga_external_sample_coords(\(f2) pt, \(f2) low, \(f2) high, \(u2) size) {
            \(f2) center = (low + high) * 0.5;
            \(f2) inset = \(f2)(0.5, 0.5) / \(f2)(size);
            return clamp(pt, min(low + inset, center), max(high - inset, center));
        }
        \(f4) saga_external_sample(\(planes), \(words)\(lang.sampler), \(f2) uv) {
            \(f4) transform = \(bit(9));
            \(f2) origin = \(bit(10)).xy;
            \(f2) opposite = origin + transform.xy + transform.zw;
            \(f2) low = min(origin, opposite);
            \(f2) high = max(origin, opposite);
            \(f2) pt = origin + transform.xy * uv.x + transform.zw * uv.y;
            \(f2) c0 = saga_external_sample_coords(pt, low, high, saga_external_plane_size(p0));
            \(f4) first = \(lang.sample("p0", coordinate: "c0"));
            if (words[12].z == 1u) { return first; }
            \(f2) c1 = saga_external_sample_coords(pt, low, high, saga_external_plane_size(p1));
            \(f2) chroma = \(lang.sample("p1", coordinate: "c1")).xy;
            if (words[12].z == 3u) {
                \(f2) c2 = saga_external_sample_coords(pt, low, high, saga_external_plane_size(p2));
                chroma.y = \(lang.sample("p2", coordinate: "c2")).x;
            }
            return saga_external_convert(first.x, chroma, words);
        }
        \(u2) saga_external_subsampled_pixel(\(u2) pixel, \(u2) fromSize, \(u2) toSize) {
            return min(\(u2)(floor(\(f2)(pixel) * \(f2)(toSize) / \(f2)(fromSize))), toSize - \(u2)(1u, 1u));
        }
        \(f4) saga_external_load(\(planes), \(words), \(u2) pixel) {
            \(u2) size = saga_external_dimensions(p0, words);
            if (\(lang.compareExtent)) { return \(f4)(0.0, 0.0, 0.0, 0.0); }
            \(f4) transform = \(bit(11));
            \(f2) pt = \(bit(10)).zw * float(pixel.x) + transform.xy * float(pixel.y) + transform.zw;
            \(u2) planeSize = saga_external_plane_size(p0);
            \(u2) c0 = \(u2)(clamp(round(pt), \(f2)(0.0, 0.0), \(f2)(planeSize - \(u2)(1u, 1u))));
            \(f4) first = \(lang.load("p0", coordinate: "c0"));
            if (words[12].z == 1u) { return first; }
            \(u2) c1 = saga_external_subsampled_pixel(c0, planeSize, saga_external_plane_size(p1));
            \(f2) chroma = \(lang.load("p1", coordinate: "c1")).xy;
            if (words[12].z == 3u) {
                \(u2) c2 = saga_external_subsampled_pixel(c0, planeSize, saga_external_plane_size(p2));
                chroma.y = \(lang.load("p2", coordinate: "c2")).x;
            }
            return saga_external_convert(first.x, chroma, words);
        }
        """]
}
