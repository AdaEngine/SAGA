# Running a compute shader on Metal

Compile WGSL and submit a dispatch to the device queue on a Metal-capable Mac.

## Compile and dispatch

Add the `SAGA`, `SAGACompiler`, and `SAGAMetal` library products to your SwiftPM
target. Compile WGSL to MSL, pass the resulting shader artifact to the device,
then wait for the submission ticket to complete.

```swift
import SAGA
import SAGACompiler
import SAGAMetal

func runComputeShader() async throws {
    let shader = try IndependentShaderCompiler().compileCompute(
        .wgsl("@compute @workgroup_size(1) fn main() {}"),
        to: .msl
    )
    let backend = try MetalBackend.makeDevice()
    let device = try await GPUDevice.make(backend: backend)

    do {
        let pipeline = try await device.createComputePipeline(shader)
        let ticket = try await device.submitCommands([
            .dispatchCompute(pipeline: pipeline, bindings: [:], workgroups: .init(x: 1))
        ])
        _ = try await ticket.wait()
    } catch {
        await device.close()
        throw error
    }

    await device.close()
}
```

The shader above declares no resources. For a shader with storage buffers, create
the buffers on the same device and supply them using the group and binding
indices reflected in the shader artifact. Request any required device features
when creating the device.
