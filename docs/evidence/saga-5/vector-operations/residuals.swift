import SAGAShaderIR
import SAGAFrontends
import SAGAValidation
@main struct Residuals {
 static func main() {
  for (name,source) in [
   ("vector-arithmetic-resolved","const_assert all(vec2(1,2)+vec2(3,4)==vec2(4,6));"),
   ("vector-select-resolved","const_assert all(select(vec2<bool>(false),vec2<bool>(true),vec2<bool>(true)));"),
   ("vector-swizzle-index-resolved","const V=vec2(1,2); const_assert V.yx[0]==2;"),
   ("runtime-invalid-index-rejected","const V=vec2<bool>(true); fn f()->bool{return V[2];}"),
   ("matrix-const-index-resolved","const M=mat2x2<f32>(1.0,0.0,0.0,1.0); const_assert M[1].y==1.0f;"),
   ("matrix-runtime-next","const M=mat2x2<f32>(); fn f()->mat2x2<f32>{return M;}"),
   ("matrix-local-next","fn f(){const M=mat2x2<f32>();const_assert M[0].x==0.0f;}"),
   ("matrix-arithmetic-next","const M=mat2x2<f32>();const N=M+M;"),
   ("array-aggregate-next","const A=array(1,2);const_assert A[0]==1;"),
   ("f16-next","const V=vec2<f16>(1.0h);")
  ] {
   do {let m=try WGSLFrontend().parse(.wgsl(source));try ShaderSemanticValidator().validate(m);print("\(name): parsed and validated")}
   catch {print("\(name): \(error)")}
  }
 }
}
