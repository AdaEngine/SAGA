@testable import SAGACompiler
import SAGAShaderIR
import SAGAValidation
import Testing

@Test func specializationPreservesExplicitStructAlignment() throws {
    let source = ShaderInput.wgsl("""
        struct Aligned { @align(32) value: u32, other: u32, }
        override count: u32 = 2u;
        @compute @workgroup_size(count) fn main() {}
        """)
    let validated = try IndependentShaderCompiler().parseAndValidate(source)
    let entry = try #require(validated.module.entryPoints.first)
    let specialized = try specializeOverrides(
        validated, entry: entry, constants: ["count": 4], retainLinkedStages: false,
        validator: ShaderSemanticValidator())
    let structures = specialized.module.types.values.compactMap { type -> [IRStructMember]? in
        guard case .structure(let members, _) = type.kind else { return nil }
        return members
    }
    let members = try #require(structures.first { $0.first?.name == "value" })
    #expect(members.first?.explicitAlignment == 32)
    try ShaderSemanticValidator().validate(specialized.module)
}
