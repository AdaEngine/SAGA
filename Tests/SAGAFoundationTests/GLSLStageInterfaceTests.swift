import SAGACompiler
import SAGAShaderIR
import Testing

extension GLSLFrontendTests {
    @Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl, .spirv, .dot])
    func glslFlatIntegerVaryingReachesCodeGenerators(format: ShaderOutputFormat) throws {
        let source = """
            #version 450
            layout(location = 2) flat in ivec2 cell;
            layout(location = 3) in vec2 uv;
            layout(location = 0) out vec4 color;
            void main() { color = vec4(uv, vec2(cell)); }
            """
        let module = try parse(source, stage: .fragment)
        let entry = try module.functions.value(for: module.entryPoints[0].function)
        #expect(entry.arguments.map(\.binding) == [
            .location(index: 2, interpolation: .flat, sampling: nil, blendSource: nil, perPrimitive: false),
            .location(index: 3, interpolation: .perspective, sampling: .center, blendSource: nil, perPrimitive: false),
        ])
        let input = ShaderInput.glsl(source, stage: .fragment, defines: [:])
        let compiler = IndependentShaderCompiler()
        _ = try compiler.compile(input, to: format)
        if format == .wgsl {
            let web = try compiler.compileWebGPURenderStage(input, entryPoint: "main")
            #expect(web.varyings.map(\.interpolation) == [.flat, .perspective])
        }
    }

    @Test func glslInterpolationQualifiersCanAppearInEitherOrder() throws {
        let source = """
            #version 450
            centroid layout(location = 0) noperspective in vec2 edge;
            layout(location = 1) in smooth sample float coverage;
            void main() {}
            """
        let module = try parse(source, stage: .fragment)
        let entry = try module.functions.value(for: module.entryPoints[0].function)
        #expect(entry.arguments.map(\.binding) == [
            .location(index: 0, interpolation: .linear, sampling: .centroid, blendSource: nil, perPrimitive: false),
            .location(index: 1, interpolation: .perspective, sampling: .sample, blendSource: nil, perPrimitive: false),
        ])
    }

    @Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl, .spirv, .dot])
    func glslDualSourceOutputsReachCodeGenerators(format: ShaderOutputFormat) throws {
        let source = """
            #version 450
            layout(location = 0, index = 1) out vec4 coverage;
            layout(location = 0, index = 0) out vec4 color;
            void main() {
                coverage = vec4(0.5);
                color = vec4(0.25, 0.75, 0.0, 1.0);
            }
            """
        let module = try parse(source, stage: .fragment)
        let entry = try module.functions.value(for: module.entryPoints[0].function)
        let result = try #require(entry.result)
        guard case .structure(let outputs, _) = try module.types.value(for: result.type).kind else {
            throw ShaderFailure.invalidHandle
        }
        #expect(outputs.map(\.binding) == [
            .location(index: 0, interpolation: nil, sampling: nil, blendSource: 1, perPrimitive: false),
            .location(index: 0, interpolation: nil, sampling: nil, blendSource: 0, perPrimitive: false),
        ])
        let input = ShaderInput.glsl(source, stage: .fragment, defines: [:])
        let compiler = IndependentShaderCompiler()
        _ = try compiler.compile(input, to: format)
        if format == .wgsl {
            let web = try compiler.compileWebGPURenderStage(input, entryPoint: "main")
            #expect(web.usesDualSourceBlending)
        }
    }

    @Test func glslUnpairedZeroColorIndexIsAnOrdinaryOutput() throws {
        let module = try parse("layout(location = 1, index = 0) out vec4 color; void main() {}", stage: .fragment)
        let entry = try module.functions.value(for: module.entryPoints[0].function)
        #expect(entry.result?.binding == .location(index: 1, interpolation: nil, sampling: nil,
                                                   blendSource: nil, perPrimitive: false))
    }

    @Test func glslDualSourceCanUseAnImplicitPrimaryColorIndex() throws {
        let source = """
            layout(location = 0) out vec4 color;
            layout(location = 0, index = 1) out vec4 coverage;
            void main() { color = vec4(1.0); coverage = vec4(0.5); }
            """
        let input = ShaderInput.glsl(source, stage: .fragment, defines: [:])
        let web = try IndependentShaderCompiler().compileWebGPURenderStage(input, entryPoint: "main")
        #expect(web.usesDualSourceBlending)
    }

    @Test(arguments: [
        "layout(location = 0) flat out vec4 color;",
        "layout(location = 0) in ivec2 cell;",
        "layout(location = 0) centroid flat in int cell;",
        "layout(location = 0) smooth flat in float value;",
        "layout(location = 0) sample centroid in float value;",
        "layout(location = 0, index = 0) in vec4 color;",
        "layout(location = 0, index = 2) out vec4 color;",
        "layout(location = 1, index = 1) out vec4 coverage;",
        "layout(location = 0, index = 1) out vec4 coverage;",
        "layout(location = 0, index = 1) out vec4 coverage; layout(location = 0) out vec3 color;",
        "layout(location = 0, index = 1) out vec4 coverage; layout(location = 0, index = 1) out vec4 color;",
        "layout(location = 0, index = 0) out vec4 coverage; layout(location = 0) out vec4 color;",
        "layout(location = 0, index = 1) out vec4 coverage; layout(location = 0) out vec4 color; layout(location = 1) out vec4 other;",
    ])
    func glslRejectsInvalidInterfaceQualifiers(declaration: String) throws {
        #expect(throws: ShaderFailure.self) { try parse(declaration + " void main() {}", stage: .fragment) }
    }

    @Test func glslColorIndexRequiresAFragmentOutput() {
        #expect(throws: ShaderFailure.self) {
            try parse("layout(location = 0, index = 0) out vec4 color; void main() {}", stage: .vertex)
        }
    }
}
