import SAGAShaderIR
import SAGAFrontends
import SAGAValidation
@main struct BooleanResiduals {
  static func main() {
    for (name,source) in [
      ("original-comparison", "const A=1<2; const_assert A; fn f()->bool {return A;}"),
      ("function-assert-local", "fn f(){const x=1; const_assert x==1;}"),
      ("normative-short-circuit", "const_assert true || (i32(5000000000)==0);"),
      ("skipped-mandatory-division", "const_assert true || (1/0==0);"),
      ("float-assertion", "const_assert 1.0 < 2.0;"),
      ("function-float-local-assertion", "fn f(){const x=1.0; const_assert x<2.0;}"),
      ("vector-all-assertion", "const_assert all(vec2<bool>(true,true));"),
      ("type-alias-constructor", "alias B=bool; const_assert B(1);")
    ] {
      do {
        let m=try WGSLFrontend().parse(.wgsl(source))
        try ShaderSemanticValidator().validate(m)
        print("\(name): parsed and validated")
      } catch {print("\(name): \(error)")}
    }
  }
}
