import Testing
import SAGAShaderIR
import SAGACompiler

@Suite struct LoopOwnershipTests {
  @Test func whileLoweringOwnsDistinctBranchesAndContinuing() throws {
    for condition in ["true", "x < 4u"] {
      _ = try IndependentShaderCompiler().parseAndValidate(.wgsl("fn f() { var x = 0u; while " + condition + " { x += 1u; } }"))
    }
  }
  @Test func glslLoopsRetainOwnedChildBlocks() throws {
    for loop in ["for (int i = 0; i < 1; i += 1) {}", "while (true) { break; }"] {
      _ = try IndependentShaderCompiler().parseAndValidate(.glsl("#version 450\nvoid main() { " + loop + " }",stage:.fragment,defines:[:]))
    }
  }
  @Test func omittedForConditionDoesNotLeaveOrphanBlock() throws {
    for header in [";;", "var x = 0u;;x += 1u"] {
      _ = try IndependentShaderCompiler().parseAndValidate(.wgsl("fn f() { for (" + header + ") { break; } }"))
    }
  }
}
