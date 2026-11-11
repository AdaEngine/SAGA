import Foundation
import Metal
let device = MTLCreateSystemDefaultDevice()!
let source = try String(contentsOfFile: "/tmp/saga-loop.metal", encoding: .utf8)
for mode in [0,1,2] {
 let options = MTLCompileOptions()
 if mode == 1 { options.optimizationLevel = .size }
 let src = mode == 2 ? source.replacingOccurrences(of:"if (saga_load_5 < 6u) {\n        } else {", with:"if (!(saga_load_5 < 6u)) {") : source
 let lib = try device.makeLibrary(source: src, options: options)
 let pipe = try device.makeComputePipelineState(function:lib.makeFunction(name:"saga_main")!)
 let buf=device.makeBuffer(length:4)!
 buf.contents().storeBytes(of:UInt32(12345),as:UInt32.self)
 let c=device.makeCommandQueue()!.makeCommandBuffer()!;let e=c.makeComputeCommandEncoder()!
 e.setComputePipelineState(pipe);e.setBuffer(buf,offset:0,index:0);e.dispatchThreadgroups(MTLSize(width:1,height:1,depth:1),threadsPerThreadgroup:MTLSize(width:1,height:1,depth:1));e.endEncoding();c.commit();c.waitUntilCompleted();print(mode,buf.contents().load(as:UInt32.self), c.error as Any)
}
