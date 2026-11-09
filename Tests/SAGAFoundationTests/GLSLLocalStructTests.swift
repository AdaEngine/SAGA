import SAGACompiler
import SAGAShaderIR
import Testing

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
])
func glslLocalStructTypesRejectOutOfScopeUsesAndConflicts(_ declarations: String) {
    let input = ShaderInput.glsl("#version 440 core\n\(declarations)", stage: .fragment, defines: [:])
    #expect(throws: ShaderFailure.self) { try IndependentShaderCompiler().parseAndValidate(input) }
}
