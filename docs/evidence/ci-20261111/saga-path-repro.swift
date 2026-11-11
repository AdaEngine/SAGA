import Metal
import Foundation
let device = MTLCreateSystemDefaultDevice()!
let source = try String(contentsOfFile: "/tmp/saga-path-repro.metal", encoding: .utf8)
let lib = try device.makeLibrary(source: source, options: nil)
let pso = try device.makeComputePipelineState(function: lib.makeFunction(name: "saga_main")!)
let buffer = device.makeBuffer(length: 20, options: .storageModeShared)!
let ptr=buffer.contents().bindMemory(to: UInt32.self, capacity: 5)
for (i,v) in [UInt32(0),10,20,30,40].enumerated() { ptr[i]=v }
let cb=device.makeCommandQueue()!.makeCommandBuffer()!
let enc=cb.makeComputeCommandEncoder()!
enc.setComputePipelineState(pso);enc.setBuffer(buffer, offset:0,index:0)
enc.dispatchThreadgroups(MTLSize(width:1,height:1,depth:1),threadsPerThreadgroup:MTLSize(width:1,height:1,depth:1))
enc.endEncoding();cb.commit();cb.waitUntilCompleted()
print(Array(UnsafeBufferPointer(start:ptr,count:5)))
