import SAGAShaderIR
import SAGAFrontends
import SAGAValidation
@main struct VectorResiduals {
 static func main() {
  for (name,source) in [
   ("generic-vector-resolved","const_assert all(vec2<bool>(true,true));"),
   ("vector-alias-resolved","alias V=vec2<bool>; const_assert all(V(true,true));"),
   ("sqrt-hex-resolved","const_assert sqrt(0x1p2f)==2.0f;"),
   ("runtime-context-resolved","const V=vec2(4294967295,2); fn f()->vec2<u32>{return V;}"),
   ("vector-arithmetic-next","const_assert all((vec2(1,2)+vec2(3,4))==vec2(4,6));"),
   ("vector-select-next","const_assert all(select(vec2<bool>(false),vec2<bool>(true),vec2<bool>(true)));"),
   ("matrix-constructor-next","const M=mat2x2<f32>(1.0,0.0,0.0,1.0);"),
   ("f16-next","const V=vec2<f16>(1.0h);"),
   ("vector-swizzle-next","const V=vec2(1,2); const_assert V.x==1;")
  ] {
   do {let m=try WGSLFrontend().parse(.wgsl(source));try ShaderSemanticValidator().validate(m);print("\(name): parsed and validated")}
   catch {print("\(name): \(error)")}
  }
 }
}
