import SAGAShaderIR
import SAGAFrontends
import SAGAValidation
@main struct FloatResiduals {
  static func main() {
    for (name,source) in [
      ("float-assert-resolved","const_assert 1.0<2.0;"),
      ("local-float-resolved","fn f(){const x=1.0; const_assert x<2.0;}"),
      ("alias-resolved","alias B=bool; const_assert B(1);"),
      ("f32-step-rounding","const_assert (16777216.0f+1.0f)-16777216.0f==0.0f;"),
      ("abstract-step-precision","const_assert (16777216.0+1.0)-16777216.0==1.0;"),
      ("vector-all-next","const_assert all(vec2<bool>(true,true));"),
      ("vector-alias-next","alias V=vec2<bool>; const_assert all(V(true,true));"),
      ("float-builtin-next","const_assert sqrt(4.0)==2.0;"),
      ("hex-float-next","const_assert 0x1p0 < 2.0;")
    ] {
      do {let m=try WGSLFrontend().parse(.wgsl(source));try ShaderSemanticValidator().validate(m);print("\(name): parsed and validated")}
      catch {print("\(name): \(error)")}
    }
  }
}
