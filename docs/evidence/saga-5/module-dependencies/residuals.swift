import SAGAShaderIR
import SAGAFrontends
import SAGAValidation
@main struct DependencyResiduals {
  static func main() {
    for (name,source) in [
      ("forward-align-resolved", "struct S { @align(A) x:f32, } const A=16u;"),
      ("abstract-align-resolved", "const A=16; struct S { @align(A) x:f32, }"),
      ("abstract-u32-use-resolved", "fn f()->u32 { return A; } const A=4294967295;"),
      ("abstract-float-declaration", "const A=1.5; fn f()->f32 { return A; }"),
      ("boolean-const-comparison", "const A=1 < 2;"),
      ("module-const-assert", "const A=16; const_assert A == 16;")
    ] {
      do {
        let m=try WGSLFrontend().parse(.wgsl(source))
        do {try ShaderSemanticValidator().validate(m);print("\(name): parsed and validated")}
        catch {print("\(name): validator: \(error)")}
      } catch {print("\(name): frontend: \(error)")}
    }
  }
}
