import SAGAShaderIR
import SAGAValidation
@main struct BoundaryAudit {
 static func h<T:Sendable>(_ n:Int) throws -> IRHandle<T> {try .init(index:n)}
 static func main() throws {
  let types=IRTable([IRType(kind:.scalar(.f32)),IRType(kind:.vector(size:.two,scalar:.f32)),IRType(kind:.matrix(columns:.two,rows:.two,scalar:.f32))])
  let expressions=IRTable([
   IRExpression(type:try h(0),kind:.literal(.f32(0))),
   IRExpression(type:try h(1),kind:.compose(type:try h(1),components:[try h(0),try h(0)])),
   IRExpression(type:try h(2),kind:.compose(type:try h(2),components:[try h(1),try h(1)]))])
  let function=IRFunction(expressions:expressions,statements:IRTable([IRStatement(kind:.emit([try h(1),try h(2)])),IRStatement(kind:.returnValue(nil))]),blocks:IRTable([IRBlock(statements:[try h(0),try h(1)])]),body:try h(0))
  do {try ShaderSemanticValidator().validate(IRModule(types:types,functions:IRTable([function])));print("matrix compose: accepted")}
  catch {print("matrix compose: \(error)")}
  // Separate raw validator residual: Access (dynamic IR node) with a literal index is not checked
  // like AccessIndex. Frontend now rejects the original WGSL case; no raw validator edit authorized.
  let rawTypes=IRTable([IRType(kind:.scalar(.f32)),IRType(kind:.vector(size:.two,scalar:.f32)),IRType(kind:.scalar(.u32))])
  let xs=IRTable([IRExpression(type:try h(0),kind:.literal(.f32(0))),IRExpression(type:try h(1),kind:.compose(type:try h(1),components:[try h(0),try h(0)])),IRExpression(type:try h(2),kind:.literal(.u32(2))),IRExpression(type:try h(0),kind:.access(base:try h(1),index:try h(2)))])
  let f=IRFunction(expressions:xs,statements:IRTable([IRStatement(kind:.emit([try h(1),try h(3)])),IRStatement(kind:.returnValue(nil))]),blocks:IRTable([IRBlock(statements:[try h(0),try h(1)])]),body:try h(0))
  do {_ = try ValidatedModule.checking(IRModule(types:rawTypes,functions:IRTable([f])),with:ShaderSemanticValidator());print("raw literal Access vector index2 width2: accepted (OPEN semantic rule gap, not conformance)")}
  catch {print("raw literal Access: \(error)")}
 }
}
