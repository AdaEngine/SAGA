import SAGACompiler
import SAGAShaderIR
import Testing

@Test func glslStructConstructorsImplicitlyConvertFieldTypes() throws {
    let input = ShaderInput.glsl("""
        #version 450
        struct Surface { float roughness; vec2 slope; };
        layout(location = 0) out vec4 color;
        void main() {
            Surface surface = Surface(2u, ivec2(3, 4));
            color = vec4(surface.roughness, surface.slope, 1.0);
        }
        """, stage: .fragment, defines: [:])
    let compiler = IndependentShaderCompiler()
    _ = try compiler.compileWebGPURenderStage(input, entryPoint: "main")
}

@Test(arguments: ["true", "vec2(1.0)", ""])
func glslStructConstructorsRejectIncompatibleFields(_ argument: String) {
    let input = ShaderInput.glsl("""
        #version 450
        struct Surface { float roughness; };
        void main() { Surface surface = Surface(\(argument)); }
        """, stage: .fragment, defines: [:])
    #expect(throws: ShaderFailure.self) { try IndependentShaderCompiler().parseAndValidate(input) }
}

@Test func glslStructMembersAllowDeclaratorLists() throws {
    let source = """
        #version 450
        struct Params { float first, weights[3], last; vec2 offset, scale; };
        layout(location = 0) out vec4 color;
        void main() {
            Params values;
            values.weights[1] = 0.5;
            color = vec4(values.first, values.offset, values.weights[1] + values.last);
        }
        """
    let input = ShaderInput.glsl(source, stage: .fragment, defines: [:])
    let compiler = IndependentShaderCompiler()
    let validated = try compiler.parseAndValidate(input)
    let layouts = validated.module.types.values.compactMap { type -> [IRStructMember]? in
        guard case .structure(let members, _) = type.kind, members.first?.name == "first" else { return nil }
        return members
    }
    let members = try #require(layouts.first)
    #expect(members.map(\.name) == ["first", "weights", "last", "offset", "scale"])
    #expect(members.map(\.offset) == [0, 4, 16, 24, 32])
    _ = try compiler.compileWebGPURenderStage(input, entryPoint: "main")
}

@Test(arguments: [
    "struct Bad { float duplicate, duplicate; };",
    "layout(std430, binding = 0) buffer Bad { float tail[], other; } bad;",
])
func glslStructDeclaratorListsPreserveMemberValidation(_ declaration: String) {
    let input = ShaderInput.glsl("#version 450\n\(declaration)\nvoid main() {}", stage: .fragment, defines: [:])
    #expect(throws: ShaderFailure.self) { try IndependentShaderCompiler().parseAndValidate(input) }
}

@Test func glslUninitializedLocalConstIsAnImmutableLocal() throws {
    let source = """
        #version 450
        struct Surface { float roughness; };
        out vec4 color;
        void main() {
            const Surface surface;
            color = vec4(surface.roughness);
        }
        """
    let input = ShaderInput.glsl(source, stage: .fragment, defines: [:])
    let compiler = IndependentShaderCompiler()
    let module = try compiler.parseAndValidate(input).module
    let local = try #require(module.functions.values.flatMap { $0.localVariables.values }.first { $0.name == "surface" })
    #expect(local.initializer == nil)
    _ = try compiler.compileWebGPURenderStage(input, entryPoint: "main")
}

@Test func glslUninitializedLocalConstRejectsWritesToMembers() {
    let input = ShaderInput.glsl("""
        #version 450
        struct Surface { float roughness; };
        void main() {
            const Surface surface;
            surface.roughness = 1.0;
        }
        """, stage: .fragment, defines: [:])
    #expect(throws: ShaderFailure.self) { try IndependentShaderCompiler().parseAndValidate(input) }
}

@Test(arguments: [ShaderOutputFormat.wgsl, .msl])
func glslLocalStructTypesAreScoped(_ format: ShaderOutputFormat) throws {
    let input = ShaderInput.glsl("""
        #version 440 core
        layout(local_size_x = 1) in;
        layout(std430, binding = 0) buffer Result { int value; } outputValue;
        struct Item { int data; };

        void main() {
            Item outer = Item(2);
            {
                struct Item { float data; };
                struct Container { Item item; } wrapped = Container(Item(3.0));
                outputValue.value = int(wrapped.item.data);
            }
            {
                Item Item = Item(1);
                outputValue.value += Item.data;
            }
            Item restored = Item(4);
            outputValue.value += outer.data + restored.data;
        }
        """, stage: .compute, defines: [:])
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(input, to: format)
    #expect(shader.storageBindings.map(\.minimumByteSize) == [4])
    if case .text(.wgsl, let source) = shader.artifact {
        _ = try compiler.parseAndValidate(.wgsl(source))
    }
}

@Test(arguments: [
    "void main() { { struct Item { int data; }; } Item escaped = Item(1); }",
    "void helper() { struct Item { int data; }; } void main() { Item escaped = Item(1); }",
    "void main() { struct Item { int data; }; struct Item { int other; }; }",
    "void main() { int Item = 0; struct Item { int data; }; }",
    "void main() { struct Item { int data; }; int Item = 0; }",
    "void main() { int Item = 0; { struct Item { int data; }; int wrong = Item; } }",
    "struct Item { int data; }; void main() { int Item = 0; Item wrong = Item(1); }",
])
func glslLocalStructTypesRejectOutOfScopeUsesAndConflicts(_ declarations: String) {
    let input = ShaderInput.glsl("#version 440 core\n\(declarations)", stage: .fragment, defines: [:])
    #expect(throws: ShaderFailure.self) { try IndependentShaderCompiler().parseAndValidate(input) }
}
