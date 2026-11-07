import SAGAFrontends
import SAGAShaderIR
import SAGAValidation
@main struct Repro {
  static func main() throws {
    let source = "struct S { items: array<u32>, } @group(0) @binding(0) var<storage> s: S;"
    let raw = try WGSLFrontend().parse(.wgsl(source))
    for t in raw.types.values {
      if case .structure(_, let span) = t.kind { print("WGSL frontend runtime-array struct span=\(span)") }
    }
    do { try ShaderSemanticValidator().validate(raw); print("Unexpected acceptance") }
    catch { print("Concrete validator: \(error)") }
  }
}
