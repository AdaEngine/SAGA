@main struct SAGAWebDiagnostic {
    static func main() {
        #if !os(WASI)
        print("Build SAGAWebDiagnostic with the Swift Wasm SDK and run it in a WebGPU browser.")
        #endif
    }
}

#if os(WASI) && compiler(>=6.3)
import SAGATypes
import SAGABackend
import SAGACore
import SAGAShaderIR
import SAGACompiler
import SAGAWebGPU
import SAGAWebHost

func runWebDiagnostic(
    capabilities: DeviceCapabilities, supportsPresentation: Bool, videoFrames: (UInt64, UInt64, UInt64)
) async throws -> String {
    let source = """
        @group(0) @binding(1) var<storage, read> input: array<u32, 4>;
        @group(0) @binding(3) var<storage, read_write> output: array<u32, 4>;
        fn baseValue(index: u32) -> u32 { return input[index] * 3u + 7u; }
        fn writeValue(index: u32, value: u32) { output[index] = value; }
        fn writeAndRead(index: u32, value: u32) -> u32 {
            writeValue(index, value);
            return output[index];
        }
        @compute @workgroup_size(2, 2)
        fn run(@builtin(global_invocation_id) id: vec3<u32>) {
            let i = id.x + id.y * 2u;
            var bonus: u32;
            let base = bonus;
            if id.x == 1u && id.y == 0u {
                bonus = 1u;
            } else if id.y > 0u {
                bonus = 2u;
            }
            var progress = 0u;
            loop {
                if progress == bonus { break; }
                let next = progress + 1u;
                continue;
                continuing {
                    progress = next;
                    break if next >= bonus;
                }
            }
            let original = writeAndRead(i, baseValue(i));
            writeValue(i, 0u);
            output[i] = original + base + progress + 1u;
        }
        """
    let host = makeWebGPUHost(
        capabilities: capabilities, supportsPresentation: supportsPresentation, transport: BrowserRPC.shared
    )
    let device = try await Device.make(backend: BrowserComputeBackend(host: host))
    do {
        try await validateNativeEarlyDepthFeature(device)
        let shader = try IndependentShaderCompiler().compileCompute(.wgsl(source), to: .wgsl)
        let pipeline = try await device.createComputePipeline(shader)
        let input = try await device.createBuffer(.init(size: 16, usage: [.storage, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
        let small = try await device.createBuffer(.init(size: 12, usage: .storage))
        let readback = try await device.createBuffer(.init(size: 16, usage: [.copyDestination, .mapRead]))
        let inputSlot = ComputeBindingSlot(binding: 1)
        let outputSlot = ComputeBindingSlot(binding: 3)
        let scope = try await device.pushErrorScope(.validation)
        do {
            _ = try await device.submitCommands([
                .dispatchCompute(pipeline: pipeline, bindings: [inputSlot: input, outputSlot: small],
                                 workgroups: .init(x: 1))
            ])
            throw GPUError.backendFailure("Swift accepted the undersized buffer")
        } catch GPUError.validation {}
        guard case .validation = try await device.popErrorScope(scope) else {
            throw GPUError.backendFailure("Swift validation scope did not capture the error")
        }
        let bindings = [inputSlot: input, outputSlot: output]
        let ticket = try await device.submitCommands([
            .write(buffer: input, offset: 0, words: [0, 1, 2, .max]),
            .write(buffer: input, offset: 16, words: []),
            .dispatchCompute(pipeline: pipeline, bindings: bindings, workgroups: .init(x: 0)),
            .copy(source: output, destination: readback, byteCount: 16),
            .read(buffer: readback, byteCount: 16),
            .dispatchCompute(pipeline: pipeline, bindings: bindings, workgroups: .init(x: 1, y: 2)),
            .copy(source: output, destination: readback, byteCount: 16),
            .read(buffer: readback, byteCount: 16),
        ])
        try await device.destroyBuffer(input)
        let result = try await ticket.wait()
        guard result == [[0, 0, 0, 0], [8, 12, 16, 7]] else {
            throw GPUError.backendFailure("Unexpected WebGPU readback: \(result)")
        }
        try await explicitLayoutDiagnostic(device)
        let integerWords = try await integerDiagnostic(device)
        let runtimeArrayWords = try await runtimeArrayDiagnostic(device)
        let storageTailWords = try await storageTailDiagnostic(device)
        let storageVectors = try await storageVectorDiagnostic(device)
        let groupedStorage = try await groupedStorageDiagnostic(device)
        let builtinInvocations = try await computeBuiltinDiagnostic(device)
        let linearIndexingInvocations = capabilities.features.contains(.linearIndexing)
            ? try await linearIndexingDiagnostic(device) : 0
        let workgroupWords = try await workgroupMemoryDiagnostic(device)
        let storageBarrierWords = try await storageBarrierDiagnostic(device)
        let atomicTickets = try await atomicDiagnostic(device)
        let atomicRMWWords = try await atomicRMWDiagnostic(device)
        let atomicCompareWords = try await atomicCompareDiagnostic(device)
        let mixedAtomicWords = try await mixedAtomicDiagnostic(device)
        let uniformLoadWords = try await uniformLoadDiagnostic(device)
        let selectWords = try await selectDiagnostic(device)
        let constantWords = try await constantSelectDiagnostic(device)
        let uniformWords = try await computeUniformDiagnostic(device)
        let renderUniformPixels = try await renderUniformDiagnostic(device)
        let matrixValues = try await matrixDiagnostic(device)
        let matrixBufferValues = try await matrixBufferDiagnostic(device)
        let switchWords = try await switchDiagnostic(device)
        let continuingTargetsWords = try await continuingTargetsDiagnostic(device)
        let textureUploadWords = try await textureUploadDiagnostic(device)
        let textureLoadWords = try await textureLoadDiagnostic(device)
        let renderTexturePixels = try await renderTextureDiagnostic(device)
        let samplerValues = try await samplerDiagnostic(device)
        try await samplerLODClampDiagnostic(device)
        let renderSamplerPixels = try await renderSamplerDiagnostic(device)
        let mipTextureWords = try await mipTextureDiagnostic(device)
        let textureViewWords = try await textureViewDiagnostic(device)
        let textureGatherLanes = try await textureGatherDiagnostic(device)
        let storageTextureWords = try await storageTextureDiagnostic(device)
        let fragmentStorageTextureWords = try await fragmentStorageTextureDiagnostic(device)
        let supportsReadableStorage = device.capabilities.features.contains(.storageTextureReadWrite)
        let readWriteStorageWords = supportsReadableStorage ? try await readWriteStorageDiagnostic(device) : []
        let textureBarrierWords = supportsReadableStorage ? try await textureBarrierDiagnostic(device) : []
        let renderReadableStorageWords = supportsReadableStorage ? try await renderReadableStorageDiagnostic(device) : []
        let depthAttachmentWords = try await depthAttachmentDiagnostic(device)
        let depthTextureWords = try await depthTextureDiagnostic(device)
        let depthComparisonWords = try await depthComparisonDiagnostic(device)
        let depthGatherLanes = try await depthGatherDiagnostic(device, comparison: true).map {
            UInt32(Float(bitPattern: $0))
        }
        let rawDepthGather = try await depthGatherDiagnostic(device, comparison: false).map {
            String(Float(bitPattern: $0))
        }.joined(separator: ",")
        let multisampleWords = try await multisampleDiagnostic(device)
        let integerTextureWords = try await integerTextureDiagnostic(device)
        let integerGatherWords = try await integerGatherDiagnostic(device)
        let sampleCoverageWords = try await sampleCoverageDiagnostic(device)
        let alphaCoverageWords = try await alphaCoverageDiagnostic(device)
        let multipleColorWords = try await multipleColorDiagnostic(device)
        let viewportWords = try await viewportDiagnostic(device)
        let depthBiasWords = try await depthBiasDiagnostic(device)
        let stencilWords = try await stencilDiagnostic(device, sampleCount: 1)
        let msaaStencilWords = try await stencilDiagnostic(device, sampleCount: 4)
        let readOnlyDepthWords = try await readOnlyDepthDiagnostic(device)
        let aspectWords = try await depthStencilAspectDiagnostic(device, sampleCount: 1)
        let msaaAspectWords = try await depthStencilAspectDiagnostic(device, sampleCount: 4)
        let stencilCopyWords = try await stencilCopyDiagnostic(device)
        let textureCopyWords = try await textureCopyDiagnostic(device)
        let arrayWords = try await arrayTextureDiagnostic(device)
        let layerViewWords = try await arrayLayerViewDiagnostic(device)
        let arraySamplingWords = try await arraySamplingDiagnostic(device)
        let storageArrayWords = try await storageArrayDiagnostic(device)
        let storageFormats = try await storageFormatDiagnostic(device)
        let cubeWords = try await cubeTextureDiagnostic(device)
        let volumeWords = try await volumeTextureDiagnostic(device)
        let volumeStorageWords = try await volumeStorageDiagnostic(device)
        let volumeRenderWords = try await volumeRenderDiagnostic(device)
        let lineTextureWords = try await lineTextureDiagnostic(device)
        let bitcastWords = try await bitcastDiagnostic(device)
        let integerBitWords = try await integerBitDiagnostic(device)
        let packingWords = try await packingDiagnostic(device)
        let geometricWords = try await geometricDiagnostic(device)
        let matrixBuiltinWords = try await matrixBuiltinDiagnostic(device)
        let derivativeWords = try await derivativeDiagnostic(device)
        let derivativeFilterWords = try await derivativeFilterDiagnostic(device)
        let numericBuiltinWords = try await numericBuiltinDiagnostic(device)
        let transcendentalWords = try await transcendentalDiagnostic(device)
        let decompositionWords = try await decompositionDiagnostic(device)
        let halfPrecisionWords = try await halfPrecisionDiagnostic(device)
        let halfBufferWords = try await halfBufferDiagnostic(device)
        let halfMatrixWords = try await halfMatrixDiagnostic(device)
        let halfGeometryWords = try await halfGeometryDiagnostic(device)
        let halfBitcastWords = try await halfBitcastDiagnostic(device)
        let halfNumericWords = try await halfNumericDiagnostic(device)
        let quantizeWords = try await quantizeDiagnostic(device)
        let floatingVectorWords = try await floatingVectorDiagnostic(device)
        let overrideWords = try await overrideDiagnostic(device)
        let computedOverrideWords = try await computedOverrideDiagnostic(device)
        let overrideArrayWords = try await overrideArrayDiagnostic(device)
        let pipelineBodyWords = try await pipelineBodyDiagnostic(device)
        let pipelineAggregateWords = try await pipelineAggregateDiagnostic(device)
        let pipelineCompositeWords = try await pipelineCompositeDiagnostic(device)
        let arrayConstantWords = try await arrayConstantDiagnostic(device)
        let pipelineBuiltinWords = try await pipelineBuiltinDiagnostic(device)
        let conversionWords = try await conversionDiagnostic(device)
        let halfConversionWords = try await halfConversionDiagnostic(device)
        let localArrayWords = try await localArrayDiagnostic(device)
        let workgroupAggregateWords = try await workgroupAggregateDiagnostic(device)
        let privateVariableWords = try await privateVariableDiagnostic(device)
        let privateRenderWords = try await privateRenderDiagnostic(device)
        let pointerWords = try await pointerDiagnostic(device)
        let pointerSubviewWords = try await pointerSubviewDiagnostic(device)
        let sharedPointerWords = try await sharedPointerDiagnostic(device)
        let swizzleAssignmentWords = try await swizzleAssignmentDiagnostic(device)
        let nestedStorageWords = try await nestedStorageDiagnostic(device)
        let nestedHalfStorageWords = try await nestedHalfStorageDiagnostic(device)
        let nestedAtomicWords = try await nestedAtomicDiagnostic(device)
        let nestedUniformWords = try await nestedUniformDiagnostic(device)
        let nestedHalfUniformWords = try await nestedHalfUniformDiagnostic(device)
        let standardUniformWords = try await standardUniformDiagnostic(device)
        let renderStorageBufferWords = try await renderStorageBufferDiagnostic(device)
        let indirectDispatchWords = try await indirectDispatchDiagnostic(device)
        let indirectDrawWords = try await indirectDrawDiagnostic(device)
        let clipDistancePixels = try await clipDistanceDiagnostic(device)
        try await verifyNativeBarycentricsAreUnavailable(device)
        let primitiveIndexPixels = try await primitiveIndexDiagnostic(device)
        let invariantPositionPixels = try await invariantPositionDiagnostic(device)
        let voidFragmentPixels = try await voidFragmentDiagnostic(device)
        let discardContinuationWords = try await discardContinuationDiagnostic(device)
        let depthClipWords = try await depthClipControlDiagnostic(device)
        let timestampQueryWords = try await timestampQueryDiagnostic(device)
        let occlusionQueryWords = try await occlusionQueryDiagnostic(device)
        let mappedBufferWords = try await mappedBufferDiagnostic(device)
        let bufferRangeWords = try await bufferRangeDiagnostic(device)
        let queueTextureWords = try await queueTextureDiagnostic(device)
        let subgroupGatherWords = try await subgroupGatherDiagnostic(device)
        let subgroupCollectiveWords = try await subgroupCollectiveDiagnostic(device)
        let subgroupIndexWords = try await subgroupIndexDiagnostic(device)
        let requestedSubgroupSize = try await subgroupSizeControlDiagnostic(device)
        let subgroupUniformityWords = try await subgroupUniformityDiagnostic(device)
        let dualSourcePixels = try await dualSourceDiagnostic(device)
        let bcTextureWords = try await bcTextureDiagnostic(device)
        let etc2TextureWords = try await etc2TextureDiagnostic(device)
        let astcTextureWords = try await astcTextureDiagnostic(device)
        let bcVolumeWords = try await compressedVolumeDiagnostic(device, format: .bc1RGBAUnorm, feature: .textureCompressionBCSliced3D)
        let astcVolumeWords = try await compressedVolumeDiagnostic(device, format: .astc5x4Unorm, feature: .textureCompressionASTCSliced3D)
        let callStatementWords = try await callStatementDiagnostic(device)
        let matrixCompoundWords = try await matrixCompoundDiagnostic(device)
        let float32TextureWords = try await float32TextureDiagnostic(device)
        let float32ColorWords = try await float32ColorDiagnostic(device)
        let halfTexelWords = try await halfTexelTransferDiagnostic(device).flatMap { $0 }
        let halfColorWords = try await halfColorResolveDiagnostic(device).flatMap { $0 }
        let halfStorageWords = try await halfStorageDiagnostic(device)
        let integer16ColorWords = try await integer16ColorDiagnostic(device)
        let integer16MSAAWords = try await integer16ColorDiagnostic(device, sampleCount: 4)
        let integer16StorageWords = try await integer16StorageDiagnostic(device)
        let normalized16StorageWords = try await normalized16StorageDiagnostic(device)
        let normalized16ColorWords = try await normalized16ColorDiagnostic(device)
        let packedColorWords = try await packedColorDiagnostic(device)
        let packedStorageWords = try await packedStorageDiagnostic(device)
        let srgbWords = try await srgbDiagnostic(device)
        let sharedExponentWords = try await sharedExponentDiagnostic(device)
        let byteTransferWords = try await byteTransferDiagnostic(device)
        let byteColorWords = try await byteColorDiagnostic(device)
        let byteStorageWords = try await byteStorageDiagnostic(device)
        let depthOnlyFormatWords = try await depthOnlyFormatDiagnostic(device)
        let depth32StencilWords = try await depth32StencilDiagnostic(device)
        let stencilOnlyWords = try await stencilOnlyDiagnostic(device)
        let bgraStorageWords = try await bgraStorageDiagnostic(device)
        let wideStorageWords = try await wideStorageDiagnostic(device)
        let forwardedResourceWords = try await forwardedResourcesDiagnostic(device)
        let webBufferArrayWords = try await webBufferArraysDiagnostic(device)
        let externalTextureWords = videoFrames.0 == 0 ? [] : try await webExternalTextureDiagnostic(device, frames: videoFrames)
        let immediateWords = capabilities.maxImmediateSize < 32 ? [] : try await computeImmediateDiagnostic(device)
        let renderImmediateWords = capabilities.maxImmediateSize < 32 ? [] : try await renderImmediateDiagnostic(device)
        let fragmentDepthWords = try await fragmentDepthDiagnostic(device)
        let fragmentRasterWords = try await fragmentRasterDiagnostic(device)
        let interpolationWord = try await interpolationDiagnostic(device)
        let topologyWords = try await topologyDiagnostic(device)
        let blendWords = try await blendDiagnostic(device)
        let render = try await renderDiagnostic(device)
        await device.close()
        func csv(_ words: [UInt32]) -> String { words.map(String.init).joined(separator: ",") }
        let snapshots = result.map(csv)
        let metrics: [String] = [String(render.checked), csv(render.probe), String(integerWords), String(render.vertexStreams),
                       String(render.indexedVertices), String(render.instancesPerDraw), String(runtimeArrayWords),
                       String(storageTailWords), String(storageVectors), csv(groupedStorage),
                       String(builtinInvocations), csv(workgroupWords), csv(storageBarrierWords), String(atomicTickets),
                       String(atomicRMWWords), String(atomicCompareWords), String(mixedAtomicWords), csv(uniformLoadWords),
                       csv(selectWords), csv(constantWords), csv(uniformWords), csv(renderUniformPixels),
                       matrixValues.map { String($0) }.joined(separator: ","),
                       matrixBufferValues.map { String($0) }.joined(separator: ","), csv(switchWords + continuingTargetsWords),
                       csv(textureUploadWords), csv(textureLoadWords), csv(renderTexturePixels), csv(samplerValues),
                       csv(renderSamplerPixels), csv(mipTextureWords), csv(textureViewWords), csv(textureGatherLanes),
                       csv(storageTextureWords), csv(fragmentStorageTextureWords), csv(readWriteStorageWords),
                       csv(textureBarrierWords), csv(renderReadableStorageWords), csv(depthAttachmentWords), csv(depthTextureWords),
                       csv(depthComparisonWords), csv(depthGatherLanes), rawDepthGather, csv(multisampleWords), csv(integerTextureWords), csv(integerGatherWords), csv(sampleCoverageWords), csv(fragmentDepthWords), csv(fragmentRasterWords), String(interpolationWord), csv(topologyWords), csv(blendWords), csv(alphaCoverageWords), csv(multipleColorWords), csv(viewportWords), csv(depthBiasWords), csv(stencilWords + msaaStencilWords), csv(readOnlyDepthWords), csv(aspectWords + msaaAspectWords), csv(stencilCopyWords), csv(textureCopyWords), csv(arrayWords), csv(layerViewWords), csv(arraySamplingWords), csv(storageArrayWords), String(storageFormats), csv(cubeWords), csv(volumeWords), csv(volumeStorageWords), csv(volumeRenderWords), csv(lineTextureWords), csv(bitcastWords), csv(integerBitWords), csv(packingWords), csv(geometricWords), csv(matrixBuiltinWords), csv(derivativeWords), csv(derivativeFilterWords), csv(numericBuiltinWords), csv(transcendentalWords), csv(decompositionWords), csv(halfPrecisionWords), csv(halfBufferWords), csv(halfMatrixWords), csv(halfGeometryWords), csv(halfBitcastWords), csv(halfNumericWords), csv(quantizeWords), csv(floatingVectorWords), csv(overrideWords), csv(computedOverrideWords), csv(overrideArrayWords), csv(pipelineBodyWords), csv(pipelineAggregateWords), csv(pipelineCompositeWords), csv(arrayConstantWords), csv(pipelineBuiltinWords), csv(conversionWords), csv(halfConversionWords), csv(localArrayWords), csv(workgroupAggregateWords), csv(privateVariableWords), csv(privateRenderWords), csv(pointerWords), csv(pointerSubviewWords), csv(sharedPointerWords), csv(swizzleAssignmentWords), csv(nestedStorageWords), csv(nestedHalfStorageWords), csv(nestedAtomicWords), csv(nestedUniformWords), csv(nestedHalfUniformWords), csv(standardUniformWords), csv(renderStorageBufferWords), csv(indirectDispatchWords), csv(indirectDrawWords), csv(clipDistancePixels), csv(primitiveIndexPixels), csv(invariantPositionPixels), csv(voidFragmentPixels), csv(discardContinuationWords), csv(depthClipWords), csv(timestampQueryWords), csv(occlusionQueryWords), csv(mappedBufferWords), csv(bufferRangeWords), csv(queueTextureWords), csv(subgroupGatherWords), csv(subgroupCollectiveWords), csv(subgroupIndexWords), csv(dualSourcePixels), csv(bcTextureWords), csv(etc2TextureWords), csv(astcTextureWords), csv(bcVolumeWords), csv(astcVolumeWords), csv(callStatementWords), csv(matrixCompoundWords), csv(float32TextureWords), csv(float32ColorWords), csv(halfTexelWords), csv(halfColorWords), csv(halfStorageWords), csv(integer16ColorWords), csv(integer16MSAAWords), csv(integer16StorageWords), csv(normalized16StorageWords), csv(normalized16ColorWords), csv(packedColorWords), csv(packedStorageWords), csv(srgbWords), csv(sharedExponentWords), csv(byteTransferWords), csv(byteColorWords), csv(byteStorageWords), csv(depthOnlyFormatWords), csv(depth32StencilWords), csv(stencilOnlyWords), csv(bgraStorageWords), csv(wideStorageWords), csv(forwardedResourceWords), csv(webBufferArrayWords), csv(externalTextureWords)]
        return (["swift"] + snapshots + metrics + [csv(immediateWords), csv(renderImmediateWords),
                                                   String(linearIndexingInvocations), String(requestedSubgroupSize),
                                                   csv(subgroupUniformityWords), String(device.capabilities.maxBindGroupsPlusVertexBuffers)]).joined(separator: "|")
    } catch {
        await device.close()
        throw error
    }
}

private func computeImmediateDiagnostic(_ device: Device) async throws -> [UInt32] {
    let source = ShaderInput.glsl("""
        #version 450
        layout(local_size_x=1) in;
        struct Pair { uint first; uint second; };
        layout(push_constant) uniform Settings { uvec3 axis; Pair pair; uint last; } settings;
        layout(binding=0) buffer Output { uint words[4]; } outputData;
        uint second() { return settings.pair.second; }
        void main() {
            outputData.words[0] = settings.axis.z;
            outputData.words[1] = settings.pair.first;
            outputData.words[2] = second();
            outputData.words[3] = settings.last;
        }
        """, stage: .compute, defines: [:])
    let compiler = IndependentShaderCompiler()
    guard case .spirv(let binary) = try compiler.compile(source, to: .spirv) else {
        throw GPUError.backendFailure("Expected binary SPIR-V for GLSL push constants")
    }
    let shader = try compiler.compileWebGPUCompute(.spirv(binary))
    let pipeline = try await device.createComputePipeline(shader)
    let output = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
    let readback = try await device.createBuffer(.init(size: 16, usage: [.copyDestination, .mapRead]))
    let arguments = try await device.createBuffer(.init(size: 12, usage: [.copyDestination, .indirect]))
    let slot = ComputeBindingSlot(binding: 0)
    let words = try await device.submitCommands([
        .write(buffer: arguments, offset: 0, words: [1, 1, 1]),
        .dispatchCompute(pipeline: pipeline, bindings: [slot: output], workgroups: .init(x: 1),
                         immediates: [1, 2, 0x1122_3344, 0xaabb_ccdd, 0x8123_4567, 9, .max, .max]),
        .copy(source: output, destination: readback, byteCount: 16),
        .read(buffer: readback, byteCount: 16),
        .dispatchComputeIndirect(pipeline: pipeline, bindings: [slot: .init(buffer: output)], indirectBuffer: arguments,
                                 immediates: [1, 2, 21, 22, 23, 0, .max, .max]),
        .copy(source: output, destination: readback, byteCount: 16),
        .read(buffer: readback, byteCount: 16),
    ]).wait()
    guard words == [[0x1122_3344, 0xaabb_ccdd, 0x8123_4567, 9], [21, 22, 23, 0]] else {
        throw GPUError.backendFailure("Unexpected compute immediate values: \(words)")
    }
    return words.flatMap { $0 }
}

private func renderImmediateDiagnostic(_ device: Device) async throws -> [UInt32] {
    let source = ShaderInput.wgsl("""
        requires immediate_address_space;
        struct FragmentData { reserved: vec2f, color: vec4f, }
        var<immediate> vertex_data: vec2f;
        var<immediate> fragment_data: FragmentData;
        @vertex fn vs(@location(0) position: vec2f) -> @builtin(position) vec4f {
            return vec4f(position + vertex_data, 0, 1);
        }
        @fragment fn fs() -> @location(0) vec4f { return fragment_data.color; }
        """)
    let compiler = IndependentShaderCompiler()
    let vertex = try compiler.compileWebGPURenderStage(source, entryPoint: "vs")
    let fragment = try compiler.compileWebGPURenderStage(source, entryPoint: "fs")
    let pipeline = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm,
        vertexBuffers: [.init(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float32x2)])])
    let positions = try await device.createBuffer(.init(size: 24, usage: [.vertex, .copyDestination]))
    let indices = try await device.createBuffer(.init(size: 12, usage: [.index, .copyDestination]))
    let arguments = try await device.createBuffer(.init(size: 36, usage: [.indirect, .copyDestination]))
    let target = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm,
                                                      usage: [.renderAttachment, .copySource]))
    let readback = try await device.createBuffer(.init(size: 256, usage: [.mapRead, .copyDestination]))
    let streams = [VertexBufferBinding(buffer: positions)]
    let index = IndexBufferBinding(buffer: indices, format: .uint32)
    let clear = RenderClearColor(red: 0, green: 0, blue: 1, alpha: 1)
    func words(_ x: Float, _ r: Float, _ g: Float, _ b: Float) -> [UInt32] {
        [x, 0, -7, -9, r, g, b, 1].map(\.bitPattern)
    }
    let setup: [BufferCommand] = [
        .write(buffer: positions, offset: 0, words: [Float(-1), -1, 3, -1, -1, 3].map(\.bitPattern)),
        .write(buffer: indices, offset: 0, words: [0, 1, 2]),
        .write(buffer: arguments, offset: 0, words: [3, 1, 0, 0, 3, 1, 0, 0, 0]),
    ]
    let draws: [BufferCommand] = [
        .render(pipeline: pipeline, target: target, clear: clear, vertexCount: 3,
                vertexBuffers: streams, immediates: words(0, 1, 0, 0)),
        .renderIndexed(pipeline: pipeline, target: target, clear: clear, indexBuffer: index,
                       draw: .init(indexCount: 3), vertexBuffers: streams, immediates: words(0, 0, 1, 0)),
        .renderIndirect(pipeline: pipeline, target: target, colorLoad: .clear(clear), indirectBuffer: arguments,
                        vertexBuffers: streams, immediates: words(4, 1, 0, 0)),
        .renderIndirect(pipeline: pipeline, target: target, colorLoad: .clear(clear), indirectBuffer: arguments,
                        indirectOffset: 16, indexBuffer: index, vertexBuffers: streams, immediates: words(0, 1, 1, 0)),
    ]
    let capture: [BufferCommand] = [
        .copyTextureToBuffer(source: target, destination: readback, bytesPerRow: 256),
        .read(buffer: readback, byteCount: 4),
    ]
    let snapshots = try await device.submitCommands(setup + draws.flatMap { [$0] + capture }).wait()
    guard snapshots == [[0xff00_00ff], [0xff00_ff00], [0xffff_0000], [0xff00_ffff]] else {
        throw GPUError.backendFailure("Unexpected render immediate colors: \(snapshots)")
    }
    return snapshots.flatMap { $0 }
}

private func integerDiagnostic(_ device: Device) async throws -> Int {
    let source = """
        @group(0) @binding(0) var<storage, read> input: array<u32, 6>;
        @group(0) @binding(1) var<storage, read_write> output: array<u32, 24>;
        @compute @workgroup_size(1)
        fn run() {
            let a = vec2u(input[0u], input[3u]);
            let b = vec2u(input[1u], input[4u]);
            let shift = vec2u(input[2u], input[5u]);
            let lhs = vec2i(a);
            let rhs = vec2i(b);
            let sum = vec2u(lhs + rhs);
            let neg = vec2u(-lhs);
            let div = vec2u(lhs / rhs);
            let rem = vec2u(lhs % rhs);
            let shr = vec2u(lhs >> shift);
            let shl = vec2u(lhs << shift);
            let udiv = a / b;
            let urem = a % b;
            let bias = vec2u(lhs + 1i);
            let cmp = lhs < rhs;
            let abstractNeg = vec2i(-(vec2(1, 2)));
            let abstractNot = vec2i(~(vec2(1, 2)));
            output[0u] = sum.x; output[1u] = neg.x; output[2u] = div.x; output[3u] = rem.x;
            output[4u] = shr.x; output[5u] = shl.x; output[6u] = udiv.x; output[7u] = urem.x;
            output[8u] = sum.y; output[9u] = neg.y; output[10u] = div.y; output[11u] = rem.y;
            output[12u] = shr.y; output[13u] = shl.y; output[14u] = udiv.y; output[15u] = urem.y;
            output[16u] = bias.x; output[17u] = bias.y;
            if cmp.x { output[18u] = 1u; }
            if cmp.y { output[19u] = 1u; }
            output[20u] = bitcast<u32>(abstractNeg.x); output[21u] = bitcast<u32>(abstractNeg.y);
            output[22u] = bitcast<u32>(abstractNot.x); output[23u] = bitcast<u32>(abstractNot.y);
        }
        """
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(source), to: .wgsl)
    let pipeline = try await device.createComputePipeline(shader)
    let input = try await device.createBuffer(.init(size: 24, usage: [.storage, .copyDestination]))
    let output = try await device.createBuffer(.init(size: 96, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 96, usage: [.mapRead, .copyDestination]))
    let ticket = try await device.submitCommands([
        .write(buffer: input, offset: 0, words: [0x80000000, .max, 32, 0xfffffff9, 0, 63]),
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): input, .init(binding: 1): output],
                         workgroups: .init(x: 1)),
        .copy(source: output, destination: read, byteCount: 96),
        .read(buffer: read, byteCount: 96),
    ])
    let result = try await ticket.wait()
    let expected: [UInt32] = [
        0x7fffffff, 0x80000000, 0x80000000, 0, 0x80000000, 0x80000000, 0, 0x80000000,
        0xfffffff9, 7, 0xfffffff9, 0, .max, 0x80000000, 0xfffffff9, 0,
        0x80000001, 0xfffffffa, 1, 1,
        0xffffffff, 0xfffffffe, 0xfffffffe, 0xfffffffd,
    ]
    guard result == [expected] else { throw GPUError.backendFailure("Unexpected integer WebGPU readback: \(result)") }
    return expected.count
}

private func runtimeArrayDiagnostic(_ device: Device) async throws -> Int {
    let source = """
        @group(0) @binding(1) var<storage, read> src: array<u32>;
        @group(0) @binding(3) var<storage, read_write> dst: array<u32>;
        fn read(index: i32) -> u32 { return src[index]; }
        @compute @workgroup_size(1)
        fn run(@builtin(global_invocation_id) id: vec3u) {
            dst[id.x] = arrayLength(&src) * 100u + arrayLength(&dst) * 10u + read(i32(id.x) - 1i);
        }
        """
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(source), to: .wgsl)
    let pipeline = try await device.createComputePipeline(shader)
    let roundTripSource = ShaderInput.wgsl("""
        struct Destination { result: array<u32, 3>, tail: array<u32> }
        @group(0) @binding(1) var<storage, read> indices: array<u32, 4>;
        @group(0) @binding(3) var<storage, read_write> dst: Destination;
        @compute @workgroup_size(1) fn round_trip() {
            let negative = bitcast<i32>(indices[0]);
            let huge = indices[1];
            let valid = indices[2];
            let beyondShort = indices[3];
            dst.result[0] = 100u * dst.tail[negative] + 10u * dst.tail[huge] + dst.tail[beyondShort];
            dst.result[1] = dst.tail[valid];
            dst.tail[negative] = 111u;
            dst.tail[huge] = 222u;
            dst.tail[beyondShort] = 333u;
            dst.tail[valid] = 444u;
            dst.result[2] = dst.tail[valid] + indices[huge];
        }
        """)
    guard case .spirv(let binary) = try compiler.compile(roundTripSource, to: .spirv) else {
        throw GPUError.backendFailure("Expected binary SPIR-V for guarded data indexing")
    }
    var opcodes = Set<UInt32>()
    var offset = 5 // SPIR-V header.
    while offset < binary.count {
        let header = binary[offset]
        let count = Int(header >> 16)
        guard count > 0, count <= binary.count - offset else {
            throw GPUError.backendFailure("Invalid SPIR-V instruction length")
        }
        opcodes.insert(header & 0xffff)
        offset += count
    }
    // The source has neither arrayLength nor control flow. Both must be present before reimport.
    guard opcodes.isSuperset(of: [68, 245, 247, 250]) else { // OpArrayLength, OpPhi, OpSelectionMerge, OpBranchConditional
        throw GPUError.backendFailure("Missing robust data indexing guards in raw SPIR-V")
    }
    let roundTripShader = try compiler.compileWebGPUCompute(.spirv(binary), entryPoint: "round_trip")
    let roundTrip = try await device.createComputePipeline(roundTripShader)
    let input = try await device.createBuffer(.init(size: 272, usage: [.storage, .copyDestination]))
    let output = try await device.createBuffer(.init(size: 280, usage: [.storage, .copySource, .copyDestination]))
    let read = try await device.createBuffer(.init(size: 280, usage: [.mapRead, .copyDestination]))
    let expected: [UInt32] = [350, 351, 352, 353, 350]
    func dispatchRoundTrip(size: UInt64) -> BufferCommand {
        .dispatchComputeRanges(pipeline: roundTrip, bindings: [
            .init(binding: 1): .init(buffer: input, offset: 256, size: 16),
            .init(binding: 3): .init(buffer: output, offset: 256, size: size),
        ], workgroups: .init(x: 1))
    }
    do {
        _ = try await device.submitCommands([dispatchRoundTrip(size: 12)])
        throw GPUError.backendFailure("Accepted a runtime array binding with only its header")
    } catch GPUError.validation("Command has invalid resource, usage, size or alignment") {}
    var commands: [BufferCommand] = [
        .write(buffer: input, offset: 256, words: [1, 2, 3, 777]),
        .write(buffer: output, offset: 256, words: [0, 0, 0, 0, 0, 999]),
        .dispatchComputeRanges(pipeline: pipeline, bindings: [
            .init(binding: 1): .init(buffer: input, offset: 256, size: 12),
            .init(binding: 3): .init(buffer: output, offset: 256, size: 20),
        ], workgroups: .init(x: 6)),
        .copy(source: output, destination: read, byteCount: 280),
        .read(buffer: read, byteCount: 280),
        .write(buffer: input, offset: 256, words: [.max, .max, 0, 1]),
    ]
    for size: UInt64 in [16, 20] {
        commands += [
            .write(buffer: output, offset: 256, words: [0, 0, 0, 27, 53, 999]),
            dispatchRoundTrip(size: size),
            .copy(source: output, sourceOffset: 256, destination: read, byteCount: 24),
            .read(buffer: read, byteCount: 24),
        ]
    }
    let result = try await device.submitCommands(commands).wait()
    let roundTripExpected: [[UInt32]] = [[0, 27, 444, 444, 53, 999], [53, 27, 444, 444, 333, 999]]
    guard result.count == 3, let words = result.first, words.count == 70,
          words.prefix(64).allSatisfy({ $0 == 0 }),
          words.suffix(6).elementsEqual(expected + [999]),
          result.dropFirst().elementsEqual(roundTripExpected)
    else { throw GPUError.backendFailure("Unexpected runtime array WebGPU readback: \(result)") }
    return expected.count + roundTripExpected.reduce(0) { $0 + $1.count }
}

private func storageTailDiagnostic(_ device: Device) async throws -> Int {
    let source = """
        struct Input {
            @size(12) bias: u32,
            factors: array<u32, 2>,
            @align(32) values: array<u32>,
        }
        struct Output {
            count: u32,
            @align(16) values: array<u32>,
        }
        @group(0) @binding(1) var<storage, read> src: Input;
        @group(0) @binding(3) var<storage, read_write> dst: Output;
        fn measure() -> u32 { return arrayLength(&src.values) * 100u + arrayLength(&dst.values) * 10u + src.bias; }
        @compute @workgroup_size(1)
        fn run(@builtin(global_invocation_id) id: vec3u) {
            if id.x == 0u { dst.count = measure(); }
            dst.values[id.x] = measure() + src.factors[id.x] + src.values[i32(id.x) - 1i];
        }
        """
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(source), to: .wgsl)
    let pipeline = try await device.createComputePipeline(shader)
    let spirvShader = try compiler.compileWebGPUCompute(.spirv(runtimeStorageTailSPIRV()))
    let spirvPipeline = try await device.createComputePipeline(spirvShader)
    let paddedShader = try compiler.compileWebGPUCompute(.spirv(runtimeStorageTailSPIRV(stride: 16)))
    let paddedPipeline = try await device.createComputePipeline(paddedShader)
    func makeAliasPipeline(_ words: [UInt32], size: UInt64 = 16) async throws -> ComputePipeline {
        let shader = try compiler.compileWebGPUCompute(.spirv(words))
        guard shader.storageBindings.count == 1,
              shader.storageBindings[0].binding == .init(group: 0, binding: 0),
              shader.storageBindings[0].minimumByteSize == size, !shader.storageBindings[0].isReadOnly else {
            throw GPUError.backendFailure("Unexpected SPIR-V storage alias reflection")
        }
        return try await device.createComputePipeline(shader)
    }
    let signedExtrema = try await makeAliasPipeline(atomicExtremaAliasSPIRV(signed: true), size: 20)
    let unsignedExtrema = try await makeAliasPipeline(atomicExtremaAliasSPIRV(signed: false), size: 20)
    let sentinel: UInt32 = 0xfeedbeef
    let floatAtomicInput: [UInt32] = [0x3fa00000, 0, 0, 0]
    let floatAtomicExpected: [UInt32] = [0xbfa00000, 0x3fa00000, 0x3fe00000, 0xbfe00000]
    let aliases: [(pipeline: ComputePipeline, initial: [UInt32], expected: [UInt32])] = [
        (try await makeAliasPipeline(storageAliasSPIRV(.unqualified)), [0xfffffff5, 0, 0, 0],
         [0x7ffffffd, 0xfffffffb, 0x80000003, 0x40000001]),
        (try await makeAliasPipeline(storageAliasSPIRV(.qualified)), [0xfffffff5, 0, 0, 0],
         [5, 0xfffffffb, 7, 0xfffffff9]),
        (try await makeAliasPipeline(storageAliasSPIRV(.atomic)), [0xfffffff7, 0, 0, 0],
         [1, 0xfffffffc, 0x7ffffffd, 0xfffffffa]),
        (signedExtrema, [3, 0, 0, 0, sentinel], [4, 3, 0xfffffffb, 4, sentinel]),
        (unsignedExtrema, [0xfffffffe, 0, 0, 0, sentinel], [0x80000005, 0xfffffffe, 3, 0x80000005, sentinel]),
        (signedExtrema, [3, 4, 0, 0, sentinel], [3, 0, 0, 3, sentinel]),
        (try await makeAliasPipeline(atomicFloatAliasSPIRV(floatFirst: true)), floatAtomicInput, floatAtomicExpected),
        (try await makeAliasPipeline(atomicFloatAliasSPIRV(floatFirst: false)), floatAtomicInput, floatAtomicExpected),
    ]
    let input = try await device.createBuffer(.init(size: 324, usage: [.storage, .copyDestination]))
    let output = try await device.createBuffer(.init(size: 296, usage: [.storage, .copySource, .copyDestination]))
    let read = try await device.createBuffer(.init(size: 296, usage: [.mapRead, .copyDestination]))
    let inputBinding = StorageBufferBinding(buffer: input, offset: 256, size: 64)
    var commands: [BufferCommand] = [
        .write(buffer: input, offset: 256, words: [7, 888, 888, 10, 20, 888, 888, 888, 1, 2, 3, 0, 0, 0, 0, 0, 777]),
        .write(buffer: output, offset: 256, words: [0, 888, 888, 888, 0, 0, 0, 0, 0, 999]),
        .dispatchComputeRanges(pipeline: pipeline, bindings: [
            .init(binding: 1): inputBinding,
            .init(binding: 3): .init(buffer: output, offset: 256, size: 36),
        ], workgroups: .init(x: 6)),
        .copy(source: output, destination: read, byteCount: 296),
        .read(buffer: read, byteCount: 296),
        .dispatchComputeRanges(pipeline: pipeline, bindings: [
            .init(binding: 1): inputBinding,
            .init(binding: 3): .init(buffer: output, offset: 256, size: 32),
        ], workgroups: .init(x: 1)),
        .copy(source: output, destination: read, byteCount: 296),
        .read(buffer: read, byteCount: 296),
        .write(buffer: input, offset: 256, words: [7, 2, 0, 0, 100, 101, 102, 103, 4, 10, 12, 33, 99, 0, 0, 0]),
        .dispatchComputeRanges(pipeline: spirvPipeline, bindings: [
            .init(binding: 0): inputBinding,
            .init(binding: 1): .init(buffer: output, offset: 256, size: 8),
        ], workgroups: .init(x: 1)),
        .copy(source: output, sourceOffset: 256, destination: read, byteCount: 8),
        .read(buffer: read, byteCount: 8),
        .write(buffer: input, offset: 260, words: [4]),
        .dispatchComputeRanges(pipeline: spirvPipeline, bindings: [
            .init(binding: 0): .init(buffer: input, offset: 256, size: 48),
            .init(binding: 1): .init(buffer: output, offset: 256, size: 8),
        ], workgroups: .init(x: 1)),
        .copy(source: output, sourceOffset: 256, destination: read, byteCount: 8),
        .read(buffer: read, byteCount: 8),
        .write(buffer: input, offset: 256, words: [7, 1, 0, 0, 100, 101, 102, 103, 4, 201, 202, 203, 12, 301, 302, 303]),
        .dispatchComputeRanges(pipeline: paddedPipeline, bindings: [
            .init(binding: 0): inputBinding,
            .init(binding: 1): .init(buffer: output, offset: 256, size: 8),
        ], workgroups: .init(x: 1)),
        .copy(source: output, sourceOffset: 256, destination: read, byteCount: 8),
        .read(buffer: read, byteCount: 8),
        .dispatchComputeRanges(pipeline: paddedPipeline, bindings: [
            .init(binding: 0): .init(buffer: input, offset: 256, size: 48),
            .init(binding: 1): .init(buffer: output, offset: 256, size: 8),
        ], workgroups: .init(x: 1)),
        .copy(source: output, sourceOffset: 256, destination: read, byteCount: 8),
        .read(buffer: read, byteCount: 8),
    ]
    for alias in aliases {
        let size = UInt64(alias.initial.count) * 4
        commands += [
            .write(buffer: output, offset: 256, words: alias.initial),
            .dispatchComputeRanges(pipeline: alias.pipeline, bindings: [
                .init(binding: 0): .init(buffer: output, offset: 256, size: size),
            ], workgroups: .init(x: 1)),
            .copy(source: output, sourceOffset: 256, destination: read, byteCount: size),
            .read(buffer: read, byteCount: size),
        ]
    }
    let ticket = try await device.submitCommands(commands)
    let expected: [[UInt32]] = [
        [857, 888, 888, 888, 867, 878, 859, 860, 857, 999],
        [847, 888, 888, 888, 857, 878, 859, 860, 857, 999],
    ]
    let result = try await ticket.wait()
    let prefix = Array(repeating: UInt32(0), count: 64)
    guard result == expected.map({ prefix + $0 }) + [[19, 8], [7, 4], [19, 2], [7, 1]] + aliases.map({ $0.expected }) else {
        throw GPUError.backendFailure("Unexpected storage tail WebGPU readback: \(result)")
    }
    return 5
}

private func runtimeStorageTailSPIRV(stride: UInt32 = 4) -> [UInt32] {
    func op(_ code: UInt32, _ operands: UInt32...) -> [UInt32] {
        [UInt32(operands.count + 1) << 16 | code] + operands
    }
    var words: [UInt32] = [0x07230203, 0x00010300, 0, 100, 0]
    for instruction in [
        op(17, 1), op(14, 0, 1), op(15, 5, 30, 0x6e69616d, 0), op(16, 30, 17, 1, 1, 1),
        op(71, 5, 6, stride), op(71, 6, 2), op(71, 7, 2),
        op(72, 6, 0, 35, 0), op(72, 6, 1, 35, 32), op(72, 7, 0, 35, 0), op(72, 7, 1, 35, 4),
        op(71, 20, 34, 0), op(71, 20, 33, 0), op(71, 21, 34, 0), op(71, 21, 33, 1),
        op(19, 1), op(21, 2, 32, 0), op(23, 4, 2, 4), op(29, 5, 2), op(30, 6, 4, 5),
        op(30, 7, 2, 2), op(32, 8, 12, 6), op(32, 10, 12, 7), op(32, 11, 12, 2), op(33, 12, 1),
        op(43, 2, 22, 0), op(43, 2, 23, 1), op(59, 8, 20, 12), op(59, 10, 21, 12),
        op(54, 1, 30, 0, 12), op(248, 31),
        op(65, 11, 33, 20, 22, 22), op(61, 2, 34, 33), op(65, 11, 35, 20, 22, 23), op(61, 2, 36, 35),
        op(65, 11, 37, 20, 23, 36), op(61, 2, 38, 37), op(128, 2, 39, 34, 38),
        op(65, 11, 40, 21, 22), op(62, 40, 39), op(68, 2, 41, 20, 1),
        op(65, 11, 42, 21, 23), op(62, 42, 41), op(253), op(56),
    ] { words += instruction }
    return words
}

private enum StorageAliasFixture { case unqualified, qualified, atomic }

private func storageAliasSPIRV(_ fixture: StorageAliasFixture) -> [UInt32] {
    func op(_ code: UInt32, _ operands: UInt32...) -> [UInt32] {
        [UInt32(operands.count + 1) << 16 | code] + operands
    }
    var words: [UInt32] = [0x07230203, 0x00010300, 0, 64, 0]
    words += op(17, 1) + op(14, 0, 1) + op(15, 5, 30, 0x6e69616d, 0) + op(16, 30, 17, 1, 1, 1)
    for block: UInt32 in [4, 5] {
        words += op(71, block, 2)
        for member: UInt32 in 0..<4 { words += op(72, block, member, 35, member * 4) }
    }
    if fixture == .qualified {
        words += op(71, 20, 24) + op(71, 21, 25)
    }
    for instruction in [
        op(71, 20, 34, 0), op(71, 20, 33, 0), op(71, 20, 20),
        op(71, 21, 34, 0), op(71, 21, 33, 0), op(71, 21, 20),
        op(19, 1), op(21, 2, 32, 0), op(21, 3, 32, 1), op(30, 4, 3, 3, 3, 3), op(30, 5, 2, 2, 2, 2),
        op(32, 6, 12, 4), op(32, 7, 12, 5), op(32, 8, 12, 3), op(32, 9, 12, 2), op(33, 10, 1),
        op(43, 2, 25, 0), op(43, 2, 26, 1), op(43, 2, 27, 2), op(43, 2, 28, 3), op(43, 3, 29, 2),
    ] { words += instruction }
    if fixture == .atomic {
        words += op(43, 3, 50, 3) + op(43, 2, 51, 7)
        // Put the unsigned representative first so the signed atomic needs a bit reinterpretation.
        words += op(59, 7, 21, 12) + op(59, 6, 20, 12)
    } else {
        words += op(59, 6, 20, 12) + op(59, 7, 21, 12)
    }
    words += op(54, 1, 30, 0, 10) + op(248, 31)
    switch fixture {
    case .unqualified:
        for instruction in [
            op(65, 8, 32, 20, 25), op(61, 3, 33, 32), op(135, 3, 34, 33, 29),
            op(65, 8, 35, 20, 26), op(62, 35, 34),
            op(65, 9, 36, 21, 26), op(61, 2, 37, 36), op(194, 2, 38, 37, 26),
            op(65, 9, 39, 21, 25), op(62, 39, 38),
            op(61, 3, 40, 32), op(126, 3, 41, 40), op(65, 8, 42, 20, 27), op(62, 42, 41),
            op(65, 9, 43, 21, 27), op(61, 2, 44, 43), op(194, 2, 45, 44, 26),
            op(65, 9, 46, 21, 28), op(62, 46, 45),
        ] { words += instruction }
    case .qualified:
        for instruction in [
            op(65, 8, 32, 20, 25), op(61, 3, 33, 32), op(135, 3, 34, 33, 29),
            op(124, 2, 35, 34), op(65, 9, 36, 21, 26), op(62, 36, 35),
            op(65, 8, 37, 20, 26), op(61, 3, 38, 37), op(126, 3, 39, 38),
            op(124, 2, 40, 39), op(65, 9, 41, 21, 25), op(62, 41, 40),
            op(61, 3, 42, 32), op(128, 3, 43, 42, 29), op(124, 2, 44, 43),
            op(65, 9, 45, 21, 27), op(62, 45, 44),
            op(65, 8, 46, 20, 27), op(61, 3, 47, 46), op(126, 3, 48, 47),
            op(124, 2, 49, 48), op(65, 9, 52, 21, 28), op(62, 52, 49),
        ] { words += instruction }
    case .atomic:
        for instruction in [
            op(65, 8, 32, 20, 25), op(234, 3, 33, 32, 26, 25, 50), op(135, 3, 34, 33, 29),
            op(65, 8, 35, 20, 26), op(62, 35, 34),
            op(65, 9, 36, 21, 25), op(227, 2, 37, 36, 26, 25), op(194, 2, 38, 37, 26),
            op(65, 9, 39, 21, 27), op(62, 39, 38),
            op(234, 2, 40, 36, 26, 25, 51), op(65, 9, 41, 21, 28), op(62, 41, 40),
        ] { words += instruction }
    }
    return words + op(253) + op(56)
}

private func atomicExtremaAliasSPIRV(signed: Bool) -> [UInt32] {
    func op(_ code: UInt32, _ operands: UInt32...) -> [UInt32] {
        [UInt32(operands.count + 1) << 16 | code] + operands
    }
    let aliasScalar: UInt32 = signed ? 3 : 2
    let aliasPointer: UInt32 = signed ? 10 : 15
    let aliasVariable: UInt32 = signed ? 20 : 21
    let physicalScalar: UInt32 = signed ? 2 : 3
    let physicalPointer: UInt32 = signed ? 15 : 10
    let physicalVariable: UInt32 = signed ? 21 : 20
    var words: [UInt32] = [0x07230203, 0x00010300, 0, 64, 0]
    words += op(17, 1) + op(14, 0, 1) + op(15, 5, 30, 0x6e69616d, 0) + op(16, 30, 17, 1, 1, 1)
    words += op(71, 4, 6, 4) + op(71, 5, 6, 4)
    for block: UInt32 in [6, 7] {
        words += op(71, block, 2)
        for member: UInt32 in 0..<5 { words += op(72, block, member, 35, member * 4) }
    }
    for instruction in [
        op(71, 20, 34, 0), op(71, 20, 33, 0), op(71, 20, 20),
        op(71, 21, 34, 0), op(71, 21, 33, 0), op(71, 21, 20),
        op(19, 1), op(21, 2, 32, 0), op(21, 3, 32, 1),
        op(43, 2, 11, 0), op(43, 2, 12, 1), op(43, 2, 13, 2), op(43, 2, 14, 3),
        op(28, 4, 3, 12), op(28, 5, 2, 12), op(30, 6, 4, 3, 3, 3, 3), op(30, 7, 5, 2, 2, 2, 2),
        op(32, 8, 12, 6), op(32, 9, 12, 7), op(32, 10, 12, 3), op(32, 15, 12, 2), op(33, 16, 1),
        op(43, aliasScalar, 17, signed ? 0xfffffffb : 3), op(43, aliasScalar, 18, signed ? 4 : 0x80000005),
    ] { words += instruction }
    // The first global is the physical representative, with the opposite signedness to the extrema.
    words += signed ? op(59, 9, 21, 12) + op(59, 8, 20, 12) : op(59, 8, 20, 12) + op(59, 9, 21, 12)
    for instruction in [
        op(54, 1, 30, 0, 16), op(248, 31), op(65, 15, 32, 21, 12), op(61, 2, 33, 32),
        op(65, aliasPointer, 34, aliasVariable, 11, 33),
        op(signed ? 236 : 237, aliasScalar, 35, 34, 12, 11, 17),
        op(signed ? 238 : 239, aliasScalar, 36, 34, 12, 11, 18),
        op(65, aliasPointer, 37, aliasVariable, 12), op(62, 37, 35),
        op(65, aliasPointer, 38, aliasVariable, 13), op(62, 38, 36),
        op(65, physicalPointer, 39, physicalVariable, 11, 11), op(227, physicalScalar, 40, 39, 12, 11),
        op(65, physicalPointer, 41, physicalVariable, 14), op(62, 41, 40), op(253), op(56),
    ] { words += instruction }
    return words
}

private func atomicFloatAliasSPIRV(floatFirst: Bool) -> [UInt32] {
    func op(_ code: UInt32, _ operands: UInt32...) -> [UInt32] {
        [UInt32(operands.count + 1) << 16 | code] + operands
    }
    var words: [UInt32] = [0x07230203, 0x00010300, 0, 64, 0]
    words += op(17, 1) + op(17, 6033)
    words += op(10, 0x5f565053, 0x5f545845, 0x64616873, 0x615f7265, 0x696d6f74, 0x6c665f63, 0x5f74616f, 0x646461)
    words += op(14, 0, 1) + op(15, 5, 30, 0x6e69616d, 0) + op(16, 30, 17, 1, 1, 1)
    for block: UInt32 in [4, 5] {
        words += op(71, block, 2)
        for member: UInt32 in 0..<4 { words += op(72, block, member, 35, member * 4) }
    }
    for instruction in [
        op(71, 20, 34, 0), op(71, 20, 33, 0), op(71, 20, 20),
        op(71, 21, 34, 0), op(71, 21, 33, 0), op(71, 21, 20),
        op(19, 1), op(21, 2, 32, 0), op(22, 3, 32), op(30, 4, 3, 3, 3, 3), op(30, 5, 2, 2, 2, 2),
        op(32, 6, 12, 4), op(32, 7, 12, 5), op(32, 8, 12, 3), op(32, 9, 12, 2), op(33, 10, 1),
        op(43, 2, 25, 0), op(43, 2, 26, 1), op(43, 2, 27, 2), op(43, 2, 28, 3),
        op(43, 3, 17, 0x3f000000), op(43, 2, 18, 0x80000000),
    ] { words += instruction }
    words += floatFirst ? op(59, 6, 20, 12) + op(59, 7, 21, 12) : op(59, 7, 21, 12) + op(59, 6, 20, 12)
    for instruction in [
        op(54, 1, 30, 0, 10), op(248, 31), op(65, 8, 32, 20, 25),
        op(6035, 3, 33, 32, 26, 25, 17), op(65, 8, 34, 20, 26), op(62, 34, 33),
        op(65, 9, 35, 21, 25), op(242, 2, 36, 35, 26, 25, 18), op(65, 9, 37, 21, 27), op(62, 37, 36),
        op(6035, 3, 38, 32, 26, 25, 17), op(65, 8, 39, 20, 28), op(62, 39, 38),
        op(253), op(56),
    ] { words += instruction }
    return words
}

private func storageVectorDiagnostic(_ device: Device) async throws -> Int {
    let source = """
        @group(0) @binding(1) var<storage, read> src: array<vec3u>;
        @group(0) @binding(3) var<storage, read_write> dst: array<vec3i>;
        @group(0) @binding(5) var<storage, read_write> pair: array<vec2f, 1>;
        @compute @workgroup_size(1)
        fn run(@builtin(global_invocation_id) id: vec3u) {
            dst[id.x] = vec3i(src[i32(id.x) - 1i]) +
                vec3i(i32(arrayLength(&src)), i32(arrayLength(&dst)), -1i);
            if id.x == 0u {
                dst[id.x].z = -7i;
                let value = pair[0u];
                pair[0u] = vec2f(value.y, value.x);
            }
            if id.x == 1u { dst[id.x][i32(id.x) - 2i] = -9i; }
            if id.x == 2u { dst[id.x][id.x - 1u] = dst[id.x].x + 100i; }
        }
        """
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(source), to: .wgsl)
    let pipeline = try await device.createComputePipeline(shader)
    let input = try await device.createBuffer(.init(size: 304, usage: [.storage, .copyDestination]))
    let output = try await device.createBuffer(.init(size: 336, usage: [.storage, .copyDestination, .copySource]))
    let pair = try await device.createBuffer(.init(size: 8, usage: [.storage, .copyDestination, .copySource]))
    let read = try await device.createBuffer(.init(size: 336, usage: [.copyDestination, .mapRead]))
    let ticket = try await device.submitCommands([
        .write(buffer: input, offset: 256, words: [10, 20, 30, 801, 40, 50, 60, 802, 777, 778, 779, 803]),
        .write(buffer: output, offset: 256, words: [
            0, 0, 0, 701, 0, 0, 0, 702, 0, 0, 0, 703, 0, 0, 0, 704, 901, 902, 903, 904,
        ]),
        .write(buffer: pair, offset: 0, words: [Float(-1.25).bitPattern, Float(3.5).bitPattern]),
        .dispatchComputeRanges(pipeline: pipeline, bindings: [
            .init(binding: 1): .init(buffer: input, offset: 256, size: 36),
            .init(binding: 3): .init(buffer: output, offset: 256, size: 68),
            .init(binding: 5): .init(buffer: pair),
        ], workgroups: .init(x: 5)),
        .copy(source: output, destination: read, byteCount: 336),
        .read(buffer: read, byteCount: 336),
        .copy(source: pair, destination: read, byteCount: 8),
        .read(buffer: read, byteCount: 8),
    ])
    let expected: [UInt32] = [
        2, 4, UInt32(bitPattern: -7), 701, 12, 24, 29, 702, 42, 142, 59, 703,
        2, 4, .max, 704, 901, 902, 903, 904,
    ]
    let result = try await ticket.wait()
    guard result == [Array(repeating: 0, count: 64) + expected, [Float(3.5).bitPattern, Float(-1.25).bitPattern]] else {
        throw GPUError.backendFailure("Unexpected vector storage WebGPU readback: \(result)")
    }
    return 4
}

private func groupedStorageDiagnostic(_ device: Device) async throws -> [UInt32] {
    let source = """
        @group(3) @binding(999) var<storage, read> src: array<u32>;
        @group(0) @binding(5) var<storage, read> gain: u32;
        @group(1) @binding(999) var<storage, read_write> dst: array<u32>;
        fn scaled(i: u32) -> u32 { return src[i] * gain + arrayLength(&src) * 10u + arrayLength(&dst); }
        @compute @workgroup_size(1)
        fn run(@builtin(global_invocation_id) id: vec3u) { dst[id.x] = scaled(id.x); }
        """
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(source), to: .wgsl)
    let pipeline = try await device.createComputePipeline(shader)
    let roundTripSource = ShaderInput.wgsl("""
        @group(3) @binding(999) var<storage, read> src: array<u32, 2>;
        @group(0) @binding(5) var<storage, read> gain: i32;
        @group(1) @binding(999) var<storage, read_write> dst: array<u32, 3>;
        @compute @workgroup_size(1)
        fn round_trip() {
            let a = src[0u];
            let b = src[1u];
            let signedValue = (gain - 9i) * 3i + 1i;
            var count = 0u;
            var sum = 0u;
            loop {
                if count >= 4u { break; }
                if count == 0u { sum += a; } else { sum += b; }
                count += 1u;
                continuing {
                    sum += 2u;
                    break if count >= 3u;
                }
            }
            switch a {
                case 3u: { dst[0u] = sum + 100u; }
                default: { dst[0u] = sum + 200u; }
            }
            dst[1u] = bitcast<u32>(signedValue);
            dst[2u] = count * 10u + sum;
        }
        """)
    guard case .spirv(let words) = try compiler.compile(roundTripSource, to: .spirv),
          words.first == 0x0723_0203 else {
        throw GPUError.backendFailure("Expected a binary SPIR-V shader")
    }
    let roundTripShader = try compiler.compileWebGPUCompute(.spirv(words), entryPoint: "round_trip")
    let roundTrip = try await device.createComputePipeline(roundTripShader)
    let input = try await device.createBuffer(.init(size: 268, usage: [.storage, .copyDestination]))
    let gain = try await device.createBuffer(.init(size: 4, usage: [.storage, .copyDestination]))
    let output = try await device.createBuffer(.init(size: 272, usage: [.storage, .copyDestination, .copySource]))
    let read = try await device.createBuffer(.init(size: 272, usage: [.copyDestination, .mapRead]))
    let bindings: [ComputeBindingSlot: StorageBufferBinding] = [
        .init(group: 3, binding: 999): .init(buffer: input, offset: 256, size: 8),
        .init(binding: 5): .init(buffer: gain),
        .init(group: 1, binding: 999): .init(buffer: output, offset: 256, size: 12),
    ]
    let ticket = try await device.submitCommands([
        .write(buffer: input, offset: 256, words: [3, 5, 777]),
        .write(buffer: gain, offset: 0, words: [2]),
        .write(buffer: output, offset: 256, words: [0, 0, 0, 999]),
        .dispatchComputeRanges(pipeline: pipeline, bindings: bindings, workgroups: .init(x: 4)),
        .copy(source: output, destination: read, byteCount: 272),
        .read(buffer: read, byteCount: 272),
        .dispatchComputeRanges(pipeline: roundTrip, bindings: bindings, workgroups: .init(x: 1)),
        .copy(source: output, sourceOffset: 256, destination: read, byteCount: 16),
        .read(buffer: read, byteCount: 16),
    ])
    let result = try await ticket.wait()
    guard result.count == 2, let original = result.first, original.count == 68,
          original.prefix(64).allSatisfy({ $0 == 0 }),
          original.suffix(4).elementsEqual([29, 33, 23, 999]),
          result[1] == [119, 0xffff_ffec, 49, 999]
    else { throw GPUError.backendFailure("Unexpected grouped storage WebGPU readback: \(result)") }
    return Array(original.suffix(4)) + result[1]
}

private func computeBuiltinDiagnostic(_ device: Device) async throws -> Int {
    let source = """
        struct Invocation {
            @builtin(local_invocation_id) local: vec3u,
            @builtin(local_invocation_index) lane: u32,
            @builtin(workgroup_id) group: vec3u,
        }
        @group(0) @binding(2) var<storage, read_write> output: array<u32>;
        fn pack(v: vec3u) -> u32 { return v.x + v.y * 16u + v.z * 256u; }
        fn packLocal(v: Invocation) -> u32 { return pack(v.local); }
        @compute @workgroup_size(2, 3, 2)
        fn run(@builtin(global_invocation_id) global: vec3u, v: Invocation,
               @builtin(num_workgroups) count: vec3u) {
            let width = count.x * 2u;
            let height = count.y * 3u;
            let i = 5u * (global.x + width * (global.y + height * global.z));
            output[i] = pack(global);
            output[i + 1u] = packLocal(v);
            output[i + 2u] = v.lane;
            output[i + 3u] = pack(v.group);
            output[i + 4u] = pack(count);
        }
        """
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(source), to: .wgsl)
    let pipeline = try await device.createComputePipeline(shader)
    let roundTripSource = ShaderInput.wgsl("""
        @group(0) @binding(2) var<storage, read_write> output: array<u32>;
        fn pack(v: vec3u) -> u32 { return 1024u + v.x + v.y * 16u + v.z * 256u; }
        @compute @workgroup_size(2, 3, 2)
        fn round_trip(@builtin(global_invocation_id) gid: vec3u,
                      @builtin(local_invocation_id) lid: vec3u,
                      @builtin(local_invocation_index) lane: u32,
                      @builtin(workgroup_id) wid: vec3u,
                      @builtin(num_workgroups) count: vec3u) {
            let i = 5u * (gid.x + (count.x * 2u) * (gid.y + (count.y * 3u) * gid.z));
            output[i] = pack(gid);
            output[i + 1u] = pack(lid);
            output[i + 2u] = 1024u + lane;
            output[i + 3u] = pack(wid);
            output[i + 4u] = pack(count);
        }
        """)
    guard case .spirv(let words) = try compiler.compile(roundTripSource, to: .spirv) else {
        throw GPUError.backendFailure("Expected binary SPIR-V for compute builtins")
    }
    let roundTripShader = try compiler.compileWebGPUCompute(.spirv(words), entryPoint: "round_trip")
    guard roundTripShader.workgroupSize == .init(x: 2, y: 3, z: 2) else {
        throw GPUError.backendFailure("Unexpected round-trip workgroup size")
    }
    let roundTrip = try await device.createComputePipeline(roundTripShader)
    var checked = 0
    func pack(_ x: UInt32, _ y: UInt32, _ z: UInt32) -> UInt32 { x + y * 16 + z * 256 }
    let spirv = try compiler.compileWebGPUCompute(.spirv(computeBuiltinsSPIRV()))
    let spirvPipeline = try await device.createComputePipeline(spirv)
    for (index, groups) in [DispatchWorkgroups(x: 3, y: 2, z: 2), .init(x: 1, y: 2, z: 1)].enumerated() {
        var expected: [UInt32] = []
        for z in 0..<(groups.z * 2) {
            for y in 0..<(groups.y * 3) {
                for x in 0..<(groups.x * 2) {
                    let (lx, ly, lz) = (x % 2, y % 3, z % 2)
                    expected += [
                        pack(x, y, z), pack(lx, ly, lz), lx + 2 * ly + 6 * lz,
                        pack(x / 2, y / 3, z / 2), pack(groups.x, groups.y, groups.z),
                    ]
                }
            }
        }
        let size = UInt64(expected.count * 4)
        let buffer = try await device.createBuffer(.init(size: size, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: size, usage: [.copyDestination, .mapRead]))
        var commands: [BufferCommand] = [
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 2): buffer], workgroups: groups),
            .copy(source: buffer, destination: readback, byteCount: size),
            .read(buffer: readback, byteCount: size),
            .dispatchCompute(pipeline: roundTrip, bindings: [.init(binding: 2): buffer], workgroups: groups),
            .copy(source: buffer, destination: readback, byteCount: size),
            .read(buffer: readback, byteCount: size),
        ]
        var results = [expected, expected.map { $0 + 1024 }]
        if index == 0 {
            commands += [
                .dispatchCompute(pipeline: spirvPipeline, bindings: [.init(binding: 0): buffer], workgroups: .init(x: 3)),
                .copy(source: buffer, destination: readback, byteCount: 24), .read(buffer: readback, byteCount: 24),
            ]
            results.append([192, 197, 208, 213, 224, 229])
            checked += 6
        }
        guard try await device.submitCommands(commands).wait() == results else {
            throw GPUError.backendFailure("Unexpected WebGPU compute builtin values for \(groups)")
        }
        checked += 2 * (expected.count / 5)
    }
    return checked
}

private func computeBuiltinsSPIRV() -> [UInt32] {
    func op(_ code: UInt32, _ operands: UInt32...) -> [UInt32] {
        [UInt32(operands.count + 1) << 16 | code] + operands
    }
    var words: [UInt32] = [0x07230203, 0x00010300, 0, 100, 0]
    for instruction in [
        op(17, 1), op(14, 0, 1), op(15, 5, 12, 0x6e69616d, 0, 15, 16, 17, 18, 19), op(16, 12, 17, 1, 1, 1),
        op(71, 15, 11, 28), op(71, 16, 11, 27), op(71, 17, 11, 26), op(71, 18, 11, 24), op(71, 19, 11, 29),
        op(71, 28, 11, 25), op(71, 5, 6, 4), op(71, 6, 2), op(72, 6, 0, 35, 0), op(71, 14, 34, 0), op(71, 14, 33, 0),
        op(19, 1), op(21, 2, 32, 0), op(20, 3), op(23, 4, 2, 3),
        op(43, 2, 20, 0), op(43, 2, 21, 1), op(43, 2, 22, 2), op(43, 2, 23, 4),
        op(43, 2, 24, 16), op(43, 2, 25, 64), op(43, 2, 26, 0x108), op(43, 2, 27, 32),
        op(44, 4, 28, 22, 21, 21), op(28, 5, 2, 27), op(30, 6, 5),
        op(32, 7, 1, 4), op(32, 8, 1, 2), op(32, 9, 12, 6), op(32, 10, 12, 2), op(33, 11, 1),
        op(59, 9, 14, 12), op(59, 7, 15, 1), op(59, 7, 16, 1), op(59, 7, 17, 1), op(59, 7, 18, 1), op(59, 8, 19, 1),
        op(54, 1, 12, 0, 11), op(248, 30), op(57, 1, 31, 13), op(253), op(56),
        op(54, 1, 13, 0, 11), op(248, 40),
        op(65, 8, 41, 15, 20), op(61, 2, 42, 41), op(65, 8, 43, 16, 20), op(61, 2, 44, 43),
        op(65, 8, 45, 17, 20), op(61, 2, 46, 45), op(65, 8, 47, 18, 20), op(61, 2, 48, 47), op(61, 2, 49, 19),
        op(170, 3, 50, 46, 20), op(247, 53, 0), op(250, 50, 52, 53),
        op(248, 52), op(224, 22, 22, 26), op(249, 53), op(248, 53),
        op(132, 2, 54, 49, 23), op(132, 2, 55, 46, 24), op(132, 2, 56, 48, 25),
        op(128, 2, 57, 44, 54), op(128, 2, 58, 57, 55), op(128, 2, 59, 58, 56),
        op(65, 10, 60, 14, 20, 42), op(62, 60, 59), op(253), op(56),
    ] { words += instruction }
    return words
}

private func linearIndexingDiagnostic(_ device: Device) async throws -> Int {
    let source = ShaderInput.wgsl("""
        requires linear_indexing;
        struct Inputs { @builtin(workgroup_index) group: u32, }
        @group(0) @binding(0) var<storage, read_write> output: array<vec2u>;
        @compute @workgroup_size(2, 3, 2)
        fn run(input: Inputs, @builtin(global_invocation_index) index: u32) {
            output[index] = vec2u(index, input.group);
        }
        """)
    let shader = try IndependentShaderCompiler().compileWebGPUCompute(source)
    let pipeline = try await device.createComputePipeline(shader)
    let direct = DispatchWorkgroups(x: 2, y: 3, z: 2)
    let indirect = DispatchWorkgroups(x: 3, y: 2, z: 2)
    func expected(_ counts: DispatchWorkgroups) -> [UInt32] {
        var words: [UInt32] = []
        for z in 0..<(2 * counts.z) {
            for y in 0..<(3 * counts.y) {
                for x in 0..<(2 * counts.x) {
                    words += [UInt32(words.count / 2), x / 2 + counts.x * (y / 3 + counts.y * (z / 2))]
                }
            }
        }
        return words
    }
    let first = expected(direct), second = expected(indirect)
    let byteCount = UInt64(first.count) * 4
    let output = try await device.createBuffer(.init(size: byteCount, usage: [.storage, .copySource]))
    let readback = try await device.createBuffer(.init(size: byteCount, usage: [.mapRead, .copyDestination]))
    let arguments = try await device.createBuffer(.init(size: 12, usage: [.indirect, .copyDestination]))
    let slot = ComputeBindingSlot(binding: 0)
    let results = try await device.submitCommands([
        .write(buffer: arguments, offset: 0, words: [indirect.x, indirect.y, indirect.z]),
        .dispatchCompute(pipeline: pipeline, bindings: [slot: output], workgroups: direct),
        .copy(source: output, destination: readback, byteCount: byteCount),
        .read(buffer: readback, byteCount: byteCount),
        .dispatchComputeIndirect(pipeline: pipeline, bindings: [slot: .init(buffer: output)], indirectBuffer: arguments),
        .copy(source: output, destination: readback, byteCount: byteCount),
        .read(buffer: readback, byteCount: byteCount),
    ]).wait()
    guard results == [first, second] else { throw GPUError.backendFailure("Unexpected linear index readback") }
    return (first.count + second.count) / 2
}

private func workgroupMemoryDiagnostic(_ device: Device) async throws -> [UInt32] {
    let source = """
        var<workgroup> scratch: array<u32, 7>;
        @group(0) @binding(2) var<storage, read_write> output: array<u32>;
        var<workgroup> flag: u32;
        fn fill(lane: u32, value: u32) { scratch[lane] = value; }
        fn sync() { workgroupBarrier(); }
        @compute @workgroup_size(4)
        fn run(@builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) group: vec3u) {
            let initial = scratch[lane + 3u] + flag;
            workgroupBarrier();
            var step = 0u;
            loop {
                fill(lane, group.x * 100u + lane + 9u + step);
                if lane == 0u { flag = 7u; }
                sync();
                continuing {
                    step++;
                    break if step >= 2u;
                }
            }
            output[group.x * 4u + lane] = initial * 1000u + scratch[(lane + 1u) % 4u] + flag;
        }
        """
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(source), to: .wgsl)
    let pipeline = try await device.createComputePipeline(shader)
    let roundTripSource = ShaderInput.wgsl("""
        var<workgroup> scratch: array<u32, 4>;
        var<workgroup> flag: u32;
        var<workgroup> signedCount: atomic<i32>;
        var<workgroup> unsignedCount: atomic<u32>;
        @group(0) @binding(2) var<storage, read_write> output: array<u32, 12>;

        @compute @workgroup_size(4)
        fn round_trip(@builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) group: vec3u) {
            let initial = scratch[lane] | flag | bitcast<u32>(atomicLoad(&signedCount)) | atomicLoad(&unsignedCount);
            workgroupBarrier();
            scratch[lane] = group.x * 100u + 30u + lane;
            if lane == 0u {
                flag = 10u;
                atomicStore(&signedCount, -9i);
                atomicStore(&unsignedCount, 20u);
            }
            workgroupBarrier();
            atomicAdd(&signedCount, 1i);
            atomicAdd(&unsignedCount, 2u);
            workgroupBarrier();
            let counts = bitcast<u32>(atomicLoad(&signedCount) + 9i) * 16u + atomicLoad(&unsignedCount) * 8u;
            let neighbor = scratch[(lane + 1u) % 4u];
            let divisor = i32(initial);
            let quotient = -19i / divisor;
            let remainder = -19i % divisor;
            let failed = initial != 0u || quotient != -19i || remainder != 0i;
            output[group.x * 4u + lane] = neighbor + flag + counts + select(0u, 1000u, failed);
        }
        """)
    guard case .spirv(let words) = try compiler.compile(roundTripSource, to: .spirv) else {
        throw GPUError.backendFailure("Expected binary SPIR-V for workgroup memory")
    }
    let roundTripShader = try compiler.compileWebGPUCompute(.spirv(words), entryPoint: "round_trip")
    let roundTrip = try await device.createComputePipeline(roundTripShader)
    let largeWorkgroup = ShaderInput.wgsl("""
        override scratchWords: u32 = 8188u;
        var<workgroup> scratch: array<u32, scratchWords>;
        var<workgroup> flag: u32;
        @group(0) @binding(0) var<storage, read_write> output: array<u32, 4>;

        @compute @workgroup_size(4)
        fn run(@builtin(local_invocation_index) lane: u32) {
            let index = scratchWords - 1u - lane;
            let initial = scratch[index] + flag;
            workgroupBarrier();
            scratch[index] = 10u + lane;
            if lane == 0u { flag = 70u; }
            workgroupBarrier();
            output[lane] = initial * 1000u + flag + scratch[scratchWords - 1u - (lane + 1u) % 4u];
        }
        """)
    let limitWords = device.capabilities.maxComputeWorkgroupStorageSize / 16 * 4
    let oversized = try compiler.compileWebGPUCompute(largeWorkgroup, constants: ["scratchWords": Double(limitWords)])
    do {
        _ = try await device.createComputePipeline(oversized)
        throw GPUError.backendFailure("Workgroup storage accepted above the negotiated limit")
    } catch GPUError.validation("Workgroup storage exceeds the device limit") {
        // The limit is checked before invoking the browser pipeline allocator.
    }
    let largeShader = try compiler.compileWebGPUCompute(largeWorkgroup,
        constants: ["scratchWords": Double(min(limitWords, 8192) - 4)])
    let largePipeline = try await device.createComputePipeline(largeShader)
    let limits = device.capabilities
    let width = min(512, limits.maxComputeWorkgroupSize.x, limits.maxComputeInvocationsPerWorkgroup)
    let wideShader = try compiler.compileWebGPUCompute(.wgsl("""
        @group(0) @binding(0) var<storage, read_write> output: u32;
        @compute @workgroup_size(\(width)u)
        fn run(@builtin(local_invocation_index) lane: u32) {
            if lane == \(width - 1)u { output = lane + 1u; }
        }
        """))
    let widePipeline = try await device.createComputePipeline(wideShader)
    if limits.maxComputeWorkgroupSize.x < .max {
        let width = limits.maxComputeWorkgroupSize.x + 1
        let oversized = try compiler.compileWebGPUCompute(.wgsl("@compute @workgroup_size(\(width)u) fn run() {}"))
        do {
            _ = try await device.createComputePipeline(oversized)
            throw GPUError.backendFailure("Workgroup size accepted above the negotiated limit")
        } catch GPUError.validation("Workgroup size exceeds the device limits") {}
    }
    let buffer = try await device.createBuffer(.init(size: 48, usage: [.storage, .copySource]))
    let readback = try await device.createBuffer(.init(size: 48, usage: [.copyDestination, .mapRead]))
    if limits.maxComputeWorkgroupsPerDimension < .max {
        do {
            _ = try await device.submitCommands([
                .dispatchCompute(pipeline: widePipeline, bindings: [.init(binding: 0): buffer],
                                 workgroups: .init(x: limits.maxComputeWorkgroupsPerDimension + 1))
            ])
            throw GPUError.backendFailure("Direct dispatch accepted above the negotiated limit")
        } catch GPUError.validation("Workgroup count exceeds the supported limit") {}
    }
    let ticket = try await device.submitCommands([
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 2): buffer], workgroups: .init(x: 3)),
        .copy(source: buffer, destination: readback, byteCount: 48),
        .read(buffer: readback, byteCount: 48),
        .dispatchCompute(pipeline: largePipeline, bindings: [.init(binding: 0): buffer], workgroups: .init(x: 1)),
        .copy(source: buffer, destination: readback, byteCount: 16),
        .read(buffer: readback, byteCount: 16),
        .dispatchCompute(pipeline: widePipeline, bindings: [.init(binding: 0): buffer], workgroups: .init(x: 1)),
        .copy(source: buffer, destination: readback, byteCount: 4),
        .read(buffer: readback, byteCount: 4),
        .dispatchCompute(pipeline: roundTrip, bindings: [.init(binding: 2): buffer], workgroups: .init(x: 3)),
        .copy(source: buffer, destination: readback, byteCount: 48),
        .read(buffer: readback, byteCount: 48),
    ])
    let result = try await ticket.wait()
    let expected: [UInt32] = [18, 19, 20, 17, 118, 119, 120, 117, 218, 219, 220, 217]
    let largeExpected: [UInt32] = [81, 82, 83, 80]
    let roundTripExpected: [UInt32] = [329, 330, 331, 328, 429, 430, 431, 428, 529, 530, 531, 528]
    guard result == [expected, largeExpected, [width], roundTripExpected] else {
        throw GPUError.backendFailure("Unexpected workgroup memory WebGPU readback: \(result)")
    }
    return expected + largeExpected + [width] + result[3]
}

private func storageBarrierDiagnostic(_ device: Device) async throws -> [UInt32] {
    let source = """
        enable wgpu_debug_printf;
        @group(0) @binding(1) var<storage, read_write> scratch: array<u32>;
        @group(2) @binding(1) var<storage, read_write> output: array<u32>;
        fn syncDevice() { storageBarrier(); }
        fn rendezvous() { syncDevice(); }
        @compute @workgroup_size(128)
        fn run(@builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) group: vec3u) {
            let start = group.x * 128u;
            scratch[start + lane] = group.x * 1000u + lane + 10u;
            rendezvous();
            output[start + lane] = scratch[start + (lane + 64u) % 128u] + 7u;
        }
        @compute @workgroup_size(1) fn native_debug() { debugPrintf("native profile %u", 1u); }
        """
    let compiler = IndependentShaderCompiler()
    let native = ShaderInput.wgsl("""
        @group(0) @binding(0) @coherent var<storage, read_write> data: u32;
        @compute @workgroup_size(1) fn run() { data = 1u; }
        """)
    do {
        _ = try compiler.compileWebGPUCompute(native)
        throw GPUError.backendFailure("Browser profile accepted a native WGSL memory attribute")
    } catch ShaderFailure.unsupported("Browser WGSL does not support native storage memory attributes") {}
    do {
        _ = try compiler.compileWebGPUCompute(.wgsl(source), entryPoint: "native_debug")
        throw GPUError.backendFailure("Browser profile accepted shader debugPrintf")
    } catch ShaderFailure.unsupported {}
    let shader = try compiler.compileWebGPUCompute(.wgsl(source), entryPoint: "run")
    let pipeline = try await device.createComputePipeline(shader)
    let outputBytes: UInt64 = 3 * 128 * 4
    let sentinelOffset = 256 + outputBytes
    let scratchBytes = sentinelOffset + 4
    let scratch = try await device.createBuffer(.init(size: scratchBytes, usage: [.storage, .copyDestination, .copySource]))
    let output = try await device.createBuffer(.init(size: outputBytes, usage: [.storage, .copySource]))
    let readback = try await device.createBuffer(.init(size: scratchBytes, usage: [.copyDestination, .mapRead]))
    let spirv = try compiler.compileWebGPUCompute(.spirv(storageBarrierSPIRV()))
    let spirvPipeline = try await device.createComputePipeline(spirv)
    let coherentInput = ShaderInput.spirv(storageBarrierSPIRV(coherent: true))
    guard case .spirv(let coherentWords) = try compiler.compile(coherentInput, to: .spirv) else {
        throw GPUError.backendFailure("Expected binary SPIR-V for coherent storage")
    }
    let coherentModule = try compiler.parseAndValidate(.spirv(coherentWords)).module
    guard coherentModule.globalVariables.values.contains(where: {
        $0.binding == .init(group: 0, binding: 0) && $0.memoryDecorations.contains(.coherent)
    }) else {
        throw GPUError.backendFailure("Coherent storage decoration did not round trip")
    }
    let coherentShader = try compiler.compileWebGPUCompute(.spirv(coherentWords), entryPoint: "main")
    let coherentPipeline = try await device.createComputePipeline(coherentShader)
    let spirvBindings: [ComputeBindingSlot: ResourceID] = [.init(binding: 0): scratch, .init(binding: 1): output]
    let ticket = try await device.submitCommands([
        .write(buffer: scratch, offset: sentinelOffset, words: [999]),
        .dispatchComputeRanges(pipeline: pipeline, bindings: [
            .init(binding: 1): .init(buffer: scratch, offset: 256, size: outputBytes),
            .init(group: 2, binding: 1): .init(buffer: output),
        ], workgroups: .init(x: 3)),
        .copy(source: output, destination: readback, byteCount: outputBytes),
        .read(buffer: readback, byteCount: outputBytes),
        .copy(source: scratch, destination: readback, byteCount: scratchBytes),
        .read(buffer: readback, byteCount: scratchBytes),
        .dispatchCompute(pipeline: spirvPipeline, bindings: spirvBindings, workgroups: .init(x: 3)),
        .copy(source: output, destination: readback, byteCount: outputBytes),
        .read(buffer: readback, byteCount: outputBytes),
        .dispatchCompute(pipeline: coherentPipeline, bindings: spirvBindings, workgroups: .init(x: 3)),
        .copy(source: output, destination: readback, byteCount: outputBytes),
        .read(buffer: readback, byteCount: outputBytes),
    ])
    let writes = (0..<384).map { (index: Int) -> UInt32 in
        let group = UInt32(index / 128)
        let lane = UInt32(index % 128)
        return group * 1000 + lane + 10
    }
    let rotated = (0..<384).map { (index: Int) -> UInt32 in
        let start = (index / 128) * 128
        let lane = (index + 64) % 128
        return writes[start + lane] + 7
    }
    let result = try await ticket.wait()
    let scratchExpected = Array(repeating: UInt32(0), count: 64) + writes + [999]
    let spirvExpected = (0..<384).map { (index: Int) -> UInt32 in UInt32((index ^ 64) + 10) }
    let coherentExpected = spirvExpected.map { $0 + 10 }
    guard result == [rotated, scratchExpected, spirvExpected, coherentExpected] else {
        throw GPUError.backendFailure("Unexpected storage barrier WebGPU readback")
    }
    let probes = [0, 127, 128, 255, 256, 383]
    return [0, 2, 3].flatMap { snapshot in probes.map { result[snapshot][$0] } }
}

private func storageBarrierSPIRV(coherent: Bool = false) -> [UInt32] {
    func op(_ code: UInt32, _ operands: UInt32...) -> [UInt32] {
        [UInt32(operands.count + 1) << 16 | code] + operands
    }
    var words: [UInt32] = [0x07230203, 0x00010300, 0, 100, 0]
    for instruction in [
        op(17, 1), op(14, 0, 1), op(15, 5, 11, 0x6e69616d, 0, 16), op(16, 11, 17, 128, 1, 1),
        op(71, 16, 11, 28), op(71, 5, 6, 4), op(71, 6, 2), op(72, 6, 0, 35, 0),
        coherent ? op(71, 14, 23) : [], // Coherent on the exchanged storage buffer.
        op(71, 14, 34, 0), op(71, 14, 33, 0), op(71, 15, 34, 0), op(71, 15, 33, 1),
        op(19, 1), op(21, 2, 32, 0), op(23, 3, 2, 3), op(43, 2, 4, 384),
        op(28, 5, 2, 4), op(30, 6, 5), op(32, 7, 12, 6), op(32, 8, 12, 2), op(32, 9, 1, 3),
        op(32, 10, 1, 2), op(33, 12, 1), op(43, 2, 20, 0), op(43, 2, 21, coherent ? 20 : 10), op(43, 2, 22, 64),
        op(43, 2, 23, 2), op(43, 2, 25, 0x48), op(43, 2, 26, 0x108),
        op(59, 7, 14, 12), op(59, 7, 15, 12), op(59, 9, 16, 1),
        op(54, 1, 11, 0, 12), op(248, 30), op(65, 10, 31, 16, 20), op(61, 2, 32, 31),
        op(65, 8, 33, 14, 20, 32), op(128, 2, 34, 32, 21), op(62, 33, 34),
        op(225, 23, 25), op(224, 23, 23, 26),
        op(198, 2, 35, 32, 22), op(65, 8, 36, 14, 20, 35), op(61, 2, 37, 36),
        op(65, 8, 38, 15, 20, 32), op(62, 38, 37), op(253), op(56),
    ] { words += instruction }
    return words
}

private func atomicDiagnostic(_ device: Device) async throws -> Int {
    let source = """
        @group(0) @binding(0) var<storage, read_write> counter: array<atomic<u32>>;
        @group(0) @binding(1) var<storage, read_write> output: array<u32>;
        var<workgroup> local: atomic<i32>;
        fn ticket() -> u32 { return atomicAdd(&counter[0u], 1u); }
        @compute @workgroup_size(64)
        fn run(@builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) group: vec3u) {
            if lane == 0u { atomicStore(&local, -10i); }
            workgroupBarrier();
            let old = atomicAdd(&local, 1i);
            let global = ticket();
            let outside = atomicAdd(&counter[group.x + 1u], 1u);
            workgroupBarrier();
            let offset = 4u * (group.x * 64u + lane);
            output[offset] = global;
            output[offset + 1u] = u32(old);
            output[offset + 2u] = u32(atomicLoad(&local));
            output[offset + 3u] = outside;
        }
        """
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(source), to: .wgsl)
    let pipeline = try await device.createComputePipeline(shader)
    let spirv = try compiler.compileWebGPUCompute(.spirv(atomicTicketsSPIRV()))
    let spirvPipeline = try await device.createComputePipeline(spirv)
    let legacy = try compiler.compileWebGPUCompute(.spirv(atomicTicketsSPIRV(legacy: true)))
    let legacyPipeline = try await device.createComputePipeline(legacy)
    let counter = try await device.createBuffer(.init(size: 516, usage: [.storage, .copyDestination, .copySource]))
    let output = try await device.createBuffer(.init(size: 2048, usage: [.storage, .copySource]))
    let readback = try await device.createBuffer(.init(size: 2048, usage: [.copyDestination, .mapRead]))
    let ticket = try await device.submitCommands([
        .write(buffer: counter, offset: 256, words: [0, 999]),
        .dispatchComputeRanges(pipeline: pipeline, bindings: [
            .init(binding: 0): .init(buffer: counter, offset: 256, size: 4),
            .init(binding: 1): .init(buffer: output),
        ], workgroups: .init(x: 2)),
        .copy(source: output, destination: readback, byteCount: 2048),
        .read(buffer: readback, byteCount: 2048),
        .copy(source: counter, destination: readback, byteCount: 264),
        .read(buffer: readback, byteCount: 264),
        .write(buffer: counter, offset: 0, words: [0, 0, 128]),
        .dispatchCompute(pipeline: spirvPipeline, bindings: [.init(binding: 0): counter, .init(binding: 1): output],
                         workgroups: .init(x: 2)),
        .copy(source: output, destination: readback, byteCount: 512), .read(buffer: readback, byteCount: 512),
        .copy(source: counter, destination: readback, byteCount: 12), .read(buffer: readback, byteCount: 12),
        .write(buffer: counter, offset: 0, words: [0, 0, 128]),
        .dispatchCompute(pipeline: legacyPipeline, bindings: [.init(binding: 0): counter, .init(binding: 1): output],
                         workgroups: .init(x: 2)),
        .copy(source: output, destination: readback, byteCount: 512), .read(buffer: readback, byteCount: 512),
        .copy(source: counter, destination: readback, byteCount: 12), .read(buffer: readback, byteCount: 12),
    ])
    let result = try await ticket.wait()
    let expectedCounter = Array(repeating: UInt32(0), count: 64) + [128, 999]
    let expectedTickets = Array(UInt32(0)..<128)
    guard result.count == 6, result[0].count == 512, result[1] == expectedCounter,
          result[2].sorted() == expectedTickets, result[3] == [128, 17, 0],
          result[4].sorted() == expectedTickets, result[5] == [128, 17, 0] else {
        throw GPUError.backendFailure("Unexpected WebGPU atomic counter")
    }
    let words = result[0]
    let slots = stride(from: 0, to: words.count, by: 4)
    let tickets = slots.map { words[$0] }.sorted()
    guard tickets == Array(0..<128), slots.allSatisfy({ words[$0 + 2] == 54 && words[$0 + 3] == 0 }) else {
        throw GPUError.backendFailure("Unexpected WebGPU atomic results")
    }
    for start in [0, 256] {
        let lanes = stride(from: start, to: start + 256, by: 4)
        guard lanes.map({ Int32(bitPattern: words[$0 + 1]) }).sorted() == Array(-10..<54) else {
            throw GPUError.backendFailure("Unexpected WebGPU workgroup atomic sequence")
        }
    }
    return tickets.count + result[2].count + result[4].count
}

private func atomicTicketsSPIRV(legacy: Bool = false) -> [UInt32] {
    func op(_ code: UInt32, _ operands: UInt32...) -> [UInt32] {
        [UInt32(operands.count + 1) << 16 | code] + operands
    }
    let storageClass: UInt32 = legacy ? 2 : 12
    let blockDecoration: UInt32 = legacy ? 3 : 2
    var words: [UInt32] = [0x07230203, legacy ? 0x00010000 : 0x00010300, 0, 100, 0]
    for instruction in [
        op(17, 1), op(14, 0, 1), op(15, 5, 11, 0x6e69616d, 0, 16), op(16, 11, 17, 64, 1, 1),
        op(71, 16, 11, 28), op(71, 5, 6, 4), op(71, 6, blockDecoration), op(72, 6, 0, 35, 0), op(72, 6, 1, 35, 512),
        op(71, 14, 34, 0), op(71, 14, 33, 0), op(71, 15, 34, 0), op(71, 15, 33, 1),
        op(19, 1), op(21, 2, 32, 0), op(23, 3, 2, 3), op(43, 2, 4, 128),
        op(28, 5, 2, 4), op(30, 6, 5, 2), op(32, 7, storageClass, 6), op(32, 8, storageClass, 2), op(32, 9, 1, 3),
        op(32, 10, 1, 2), op(33, 12, 1), op(43, 2, 20, 0), op(43, 2, 21, 1), op(43, 2, 22, 2),
        op(43, 2, 23, 0), op(43, 2, 24, 17), op(59, 7, 14, storageClass), op(59, 7, 15, storageClass), op(59, 9, 16, 1),
        op(54, 1, 11, 0, 12), op(248, 30), op(65, 10, 31, 16, 20), op(61, 2, 32, 31),
        op(65, 8, 33, 14, 20, 20), op(234, 2, 34, 33, 21, 23, 21),
        op(65, 8, 35, 14, 20, 21), op(228, 35, 21, 23, 24), op(227, 2, 36, 35, 21, 23),
        op(65, 8, 37, 14, 20, 22), op(233, 2, 38, 37, 21, 23),
        op(65, 8, 39, 15, 20, 32), op(62, 39, 34), op(253), op(56),
    ] { words += instruction }
    return words
}

private func atomicRMWDiagnostic(_ device: Device) async throws -> Int {
    let source = """
        @group(0) @binding(0) var<storage, read_write> signedValue: atomic<i32>;
        @group(0) @binding(1) var<storage, read_write> unsignedValues: array<atomic<u32>>;
        @group(0) @binding(2) var<storage, read_write> output: array<u32, 17>;
        @compute @workgroup_size(1)
        fn run(@builtin(global_invocation_id) id: vec3u) {
            output[0] = u32(atomicSub(&signedValue, i32(2147483648u)));
            output[1] = u32(atomicMin(&signedValue, -7i));
            output[2] = u32(atomicMax(&signedValue, -7i));
            output[3] = u32(atomicAnd(&signedValue, 15i));
            output[4] = u32(atomicOr(&signedValue, 32i));
            output[5] = u32(atomicXor(&signedValue, 3i));
            output[6] = u32(atomicExchange(&signedValue, -1i));
            atomicOr(&signedValue, 0i);
            output[7] = u32(atomicLoad(&signedValue));
            output[8] = atomicSub(&unsignedValues[0], 1u);
            output[9] = atomicMax(&unsignedValues[0], 7u);
            output[10] = atomicMin(&unsignedValues[0], 7u);
            output[11] = atomicAnd(&unsignedValues[0], 6u);
            output[12] = atomicOr(&unsignedValues[0], 8u);
            output[13] = atomicXor(&unsignedValues[0], 3u);
            output[14] = atomicExchange(&unsignedValues[0], 23u);
            output[15] = atomicLoad(&unsignedValues[0]);
            output[16] = atomicExchange(&unsignedValues[id.x + 1u], 777u);
        }
        """
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(source), to: .wgsl)
    let pipeline = try await device.createComputePipeline(shader)
    let floatShader = try compiler.compileWebGPUCompute(.wgsl("""
        struct Data { counters: array<array<atomic<f32>, 2>, 1>, @size(12) sentinel: u32 }
        @group(0) @binding(0) var<storage, read_write> scalar: atomic<f32>;
        @group(0) @binding(1) var<storage, read_write> data: Data;
        @group(0) @binding(2) var<storage, read_write> output: array<vec4u, 4>;
        var<workgroup> sharedValue: atomic<f32>;

        fn increment(p: ptr<storage, atomic<f32>, read_write>) -> f32 {
            return atomicAdd(p, 0.5);
        }

        @compute @workgroup_size(4)
        fn run(@builtin(local_invocation_index) lane: u32) {
            if lane == 0u {
                atomicStore(&scalar, -0.0);
                atomicStore(&data.counters[0][0], 1.25);
                atomicStore(&data.counters[0][1], 5.0);
                atomicStore(&sharedValue, -0.0);
            }
            storageBarrier();
            workgroupBarrier();
            let sharedBits = bitcast<u32>(workgroupUniformLoad(&sharedValue));
            let before = increment(&scalar);
            let nested = atomicSub(&data.counters[0][0], 0.25);
            let invalid = increment(&data.counters[0][lane + 2u]);
            output[lane] = vec4u(bitcast<u32>(before), bitcast<u32>(nested), bitcast<u32>(invalid), sharedBits);
            storageBarrier();
            if lane == 0u {
                output[0].z = bitcast<u32>(atomicExchange(&data.counters[0][1], -0.0));
                output[0].w = bitcast<u32>(atomicLoad(&scalar));
                output[1].w = bitcast<u32>(atomicLoad(&data.counters[0][0]));
                output[2].w = bitcast<u32>(atomicLoad(&data.counters[0][1]));
            }
        }
        """))
    let floatPipeline = try await device.createComputePipeline(floatShader)
    let spirvShader = try compiler.compileWebGPUCompute(.spirv(floatAtomicSPIRV()))
    let spirvPipeline = try await device.createComputePipeline(spirvShader)
    let roundTripSource = ShaderInput.wgsl("""
        @group(0) @binding(0) var<storage, read_write> counter: atomic<f32>;
        @group(0) @binding(1) var<storage, read_write> output: array<u32, 6>;
        var<workgroup> sharedValue: atomic<f32>;

        @compute @workgroup_size(1)
        fn round_trip() {
            let sharedInitial = atomicLoad(&sharedValue);
            atomicStore(&sharedValue, -0.0);
            let sharedStored = atomicLoad(&sharedValue);
            atomicStore(&counter, -0.0);
            let beforeAdd = atomicAdd(&counter, 1.5);
            let beforeSub = atomicSub(&counter, 0.25);
            let beforeExchange = atomicExchange(&counter, -2.5);
            let current = atomicLoad(&counter);
            output[0] = bitcast<u32>(beforeSub);
            output[1] = bitcast<u32>(beforeExchange);
            output[2] = bitcast<u32>(beforeAdd);
            output[3] = bitcast<u32>(current);
            output[4] = bitcast<u32>(sharedInitial);
            output[5] = bitcast<u32>(sharedStored);
        }
        """)
    guard case .spirv(let roundTripBinary) = try compiler.compile(roundTripSource, to: .spirv) else {
        throw GPUError.backendFailure("Expected binary SPIR-V for float atomics")
    }
    let roundTripShader = try compiler.compileWebGPUCompute(.spirv(roundTripBinary), entryPoint: "round_trip")
    let roundTrip = try await device.createComputePipeline(roundTripShader)
    let signed = try await device.createBuffer(.init(size: 4, usage: .storage))
    let unsigned = try await device.createBuffer(.init(size: 20, usage: [.storage, .copyDestination, .copySource]))
    let output = try await device.createBuffer(.init(size: 68, usage: [.storage, .copySource]))
    let readback = try await device.createBuffer(.init(size: 68, usage: [.copyDestination, .mapRead]))
    let ticket = try await device.submitCommands([
        .write(buffer: unsigned, offset: 4, words: [999]),
        .dispatchComputeRanges(pipeline: pipeline, bindings: [
            .init(binding: 0): .init(buffer: signed),
            .init(binding: 1): .init(buffer: unsigned, size: 4),
            .init(binding: 2): .init(buffer: output),
        ], workgroups: .init(x: 1)),
        .copy(source: output, destination: readback, byteCount: 68),
        .read(buffer: readback, byteCount: 68),
        .copy(source: unsigned, destination: readback, byteCount: 8),
        .read(buffer: readback, byteCount: 8),
        .write(buffer: unsigned, offset: 8, words: [99, 100, 101]),
        .dispatchCompute(pipeline: floatPipeline, bindings: [
            .init(binding: 0): signed, .init(binding: 1): unsigned, .init(binding: 2): output,
        ], workgroups: .init(x: 1)),
        .copy(source: output, destination: readback, byteCount: 64),
        .read(buffer: readback, byteCount: 64),
        .copy(source: unsigned, destination: readback, byteCount: 20),
        .read(buffer: readback, byteCount: 20),
        .dispatchCompute(pipeline: spirvPipeline, bindings: [.init(binding: 0): unsigned, .init(binding: 1): output],
                         workgroups: .init(x: 1)),
        .copy(source: output, destination: readback, byteCount: 20),
        .read(buffer: readback, byteCount: 20),
        .copy(source: unsigned, destination: readback, byteCount: 8),
        .read(buffer: readback, byteCount: 8),
        .dispatchCompute(pipeline: roundTrip, bindings: [.init(binding: 0): unsigned, .init(binding: 1): output],
                         workgroups: .init(x: 1)),
        .copy(source: output, destination: readback, byteCount: 24),
        .read(buffer: readback, byteCount: 24),
        .copy(source: unsigned, destination: readback, byteCount: 8),
        .read(buffer: readback, byteCount: 8),
    ])
    let expected: [UInt32] = [
        0, 0x80000000, 0x80000000, 0xfffffff9, 9, 41, 42, .max,
        0, .max, .max, 7, 6, 14, 13, 23, 0,
    ]
    let snapshots = try await ticket.wait()
    guard snapshots.count == 8, snapshots[0] == expected, snapshots[1] == [23, 999],
          snapshots[2].count == 16 else {
        throw GPUError.backendFailure("Unexpected WebGPU atomic read-modify-write results")
    }
    let floats = snapshots[2]
    guard [0, 4, 8, 12].map({ floats[$0] }).sorted() == [0x3f000000, 0x3f800000, 0x3fc00000, 0x80000000],
          [1, 5, 9, 13].map({ floats[$0] }).sorted() == [0x3f000000, 0x3f400000, 0x3f800000, 0x3fa00000],
          [2, 6, 10, 14].map({ floats[$0] }) == [0x40a00000, 0, 0, 0],
          [3, 7, 11, 15].map({ floats[$0] }) == [0x40000000, 0x3e800000, 0x80000000, 0x80000000],
          snapshots[3] == [0x3e800000, 0x80000000, 99, 100, 101] else {
        throw GPUError.backendFailure("Unexpected WebGPU float atomic results: \(snapshots)")
    }
    let spirvExpected: [UInt32] = [0x80000000, 0x3f000000, 0x40e00000, 0, 0x40100000]
    guard snapshots[4] == spirvExpected, snapshots[5] == [0x40e00000, 0x40100000] else {
        throw GPUError.backendFailure("Unexpected WebGPU SPIR-V float atomic results")
    }
    let roundTripExpected: [UInt32] = [0x3fc00000, 0x3fa00000, 0x80000000, 0xc0200000, 0, 0x80000000]
    guard snapshots[6] == roundTripExpected, snapshots[7] == [0xc0200000, 0x40100000] else {
        throw GPUError.backendFailure("Unexpected WebGPU float atomic round-trip results: \(snapshots[6...7])")
    }
    return expected.count + floats.count + spirvExpected.count + roundTripExpected.count
}

private func floatAtomicSPIRV() -> [UInt32] {
    func op(_ code: UInt32, _ operands: UInt32...) -> [UInt32] {
        [UInt32(operands.count + 1) << 16 | code] + operands
    }
    var words: [UInt32] = [0x07230203, 0x00010300, 0, 100, 0]
    for instruction in [
        op(17, 1), op(17, 6033),
        op(10, 0x5f565053, 0x5f545845, 0x64616873, 0x615f7265, 0x696d6f74, 0x6c665f63, 0x5f74616f, 0x646461),
        op(14, 0, 1), op(15, 5, 20, 0x6e69616d, 0, 17), op(16, 20, 17, 1, 1, 1),
        op(4, 0x455f4c47, 0x735f5458, 0x65646168, 0x74615f72, 0x63696d6f, 0x6f6c665f, 0x7461),
        op(71, 17, 11, 28), op(71, 5, 6, 4), op(71, 6, 2), op(72, 6, 0, 35, 0), op(72, 6, 1, 35, 4),
        op(71, 7, 2), op(72, 7, 0, 35, 0), op(72, 7, 1, 35, 4), op(72, 7, 2, 35, 8),
        op(72, 7, 3, 35, 12), op(72, 7, 4, 35, 16),
        op(71, 15, 34, 0), op(71, 15, 33, 0), op(71, 16, 34, 0), op(71, 16, 33, 1),
        op(19, 1), op(21, 2, 32, 0), op(22, 3, 32), op(23, 4, 2, 3),
        op(43, 2, 30, 0), op(43, 2, 31, 1), op(43, 2, 32, 2), op(43, 2, 33, 3), op(43, 2, 34, 4),
        op(43, 3, 35, 0x80000000), op(43, 3, 36, 0x3f000000), op(43, 3, 37, 0x40e00000),
        op(43, 3, 38, 0x40100000), op(43, 2, 39, 0),
        op(28, 5, 3, 31), op(30, 6, 5, 3), op(30, 7, 2, 2, 2, 2, 2),
        op(32, 8, 12, 6), op(32, 9, 12, 7), op(32, 10, 12, 3), op(32, 11, 12, 2),
        op(32, 12, 1, 4), op(32, 13, 1, 2), op(33, 14, 1),
        op(59, 8, 15, 12), op(59, 9, 16, 12), op(59, 12, 17, 1), op(54, 1, 20, 0, 14), op(248, 21),
        op(65, 10, 50, 15, 30, 30), op(228, 50, 31, 30, 35),
        op(6035, 3, 51, 50, 31, 39, 36), op(229, 3, 52, 50, 31, 30, 37),
        op(65, 13, 53, 17, 30), op(61, 2, 54, 53), op(128, 2, 55, 54, 31),
        op(65, 10, 56, 15, 30, 55), op(6035, 3, 57, 56, 31, 39, 36),
        op(227, 3, 58, 50, 31, 30), op(65, 10, 59, 15, 31), op(62, 59, 38), op(61, 3, 60, 59),
        op(124, 2, 61, 51), op(124, 2, 62, 52), op(124, 2, 63, 58), op(124, 2, 64, 57), op(124, 2, 65, 60),
        op(65, 11, 70, 16, 30), op(62, 70, 61), op(65, 11, 71, 16, 31), op(62, 71, 62),
        op(65, 11, 72, 16, 32), op(62, 72, 63), op(65, 11, 73, 16, 33), op(62, 73, 64),
        op(65, 11, 74, 16, 34), op(62, 74, 65), op(253), op(56),
    ] { words += instruction }
    return words
}

private func atomicCompareDiagnostic(_ device: Device) async throws -> Int {
    let source = """
        @group(0) @binding(0) var<storage, read_write> counter: array<atomic<u32>>;
        @group(0) @binding(1) var<storage, read_write> output: array<u32, 10>;
        var<workgroup> signedValue: atomic<i32>;
        @compute @workgroup_size(1)
        fn run(@builtin(global_invocation_id) id: vec3u) {
            atomicStore(&signedValue, -9i);
            var signedResult = atomicCompareExchangeWeak(&signedValue, -9i, 11i);
            output[0] = u32(signedResult.old_value);
            if signedResult.exchanged { output[1] = 1u; }
            output[2] = u32(atomicLoad(&signedValue));
            signedResult = atomicCompareExchangeWeak(&signedValue, 999i, 12i);
            output[3] = u32(signedResult.old_value);
            if signedResult.exchanged { output[4] = 1u; }
            let unsignedResult = atomicCompareExchangeWeak(&counter[0u], 0u, 7u);
            output[5] = unsignedResult.old_value;
            if unsignedResult.exchanged { output[6] = 1u; }
            output[7] = atomicLoad(&counter[0u]);
            let outside = atomicCompareExchangeWeak(&counter[id.x + 1u], 0u, 888u);
            output[8] = outside.old_value;
            if outside.exchanged { output[9] = 1u; }
        }
        """
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(source), to: .wgsl)
    let pipeline = try await device.createComputePipeline(shader)
    let spirv = try compiler.compileWebGPUCompute(.spirv(atomicCompareExchangeSPIRV()))
    let spirvPipeline = try await device.createComputePipeline(spirv)
    let roundTripSource = ShaderInput.wgsl("""
        struct Counters { signedValue: atomic<i32>, unsignedValue: atomic<u32> }
        @group(0) @binding(0) var<storage, read_write> counters: Counters;
        @group(0) @binding(1) var<storage, read_write> output: array<u32, 10>;
        @compute @workgroup_size(1) fn round_trip() {
            atomicStore(&counters.signedValue, -5i);
            atomicStore(&counters.unsignedValue, 0xfffffff0u);
            output[0] = bitcast<u32>(atomicAdd(&counters.signedValue, 2i));
            output[1] = bitcast<u32>(atomicMin(&counters.signedValue, -9i));
            output[2] = atomicMax(&counters.unsignedValue, 7u);
            output[3] = atomicMin(&counters.unsignedValue, 11u);
            let mismatch = atomicCompareExchangeWeak(&counters.signedValue, -3i, -7i);
            let match = atomicCompareExchangeWeak(&counters.signedValue, -9i, 17i);
            output[4] = bitcast<u32>(mismatch.old_value);
            output[5] = select(0u, 1u, mismatch.exchanged);
            output[6] = bitcast<u32>(match.old_value);
            output[7] = select(0u, 1u, match.exchanged);
            output[8] = bitcast<u32>(atomicLoad(&counters.signedValue));
            output[9] = atomicLoad(&counters.unsignedValue);
        }
        """)
    guard case .spirv(let words) = try compiler.compile(roundTripSource, to: .spirv) else {
        throw GPUError.backendFailure("Expected binary SPIR-V for storage atomics")
    }
    let roundTripShader = try compiler.compileWebGPUCompute(.spirv(words), entryPoint: "round_trip")
    let roundTrip = try await device.createComputePipeline(roundTripShader)
    let counter = try await device.createBuffer(.init(size: 12, usage: [.storage, .copyDestination, .copySource]))
    let output = try await device.createBuffer(.init(size: 40, usage: [.storage, .copySource]))
    let readback = try await device.createBuffer(.init(size: 40, usage: [.copyDestination, .mapRead]))
    let ticket = try await device.submitCommands([
        .write(buffer: counter, offset: 4, words: [999]),
        .dispatchComputeRanges(pipeline: pipeline, bindings: [
            .init(binding: 0): .init(buffer: counter, size: 4),
            .init(binding: 1): .init(buffer: output),
        ], workgroups: .init(x: 1)),
        .copy(source: output, destination: readback, byteCount: 40),
        .read(buffer: readback, byteCount: 40),
        .copy(source: counter, destination: readback, byteCount: 8),
        .read(buffer: readback, byteCount: 8),
        .write(buffer: counter, offset: 0, words: [0]),
        .dispatchComputeRanges(pipeline: spirvPipeline, bindings: [
            .init(binding: 0): .init(buffer: counter, size: 4),
            .init(binding: 1): .init(buffer: output),
        ], workgroups: .init(x: 1)),
        .copy(source: output, destination: readback, byteCount: 16), .read(buffer: readback, byteCount: 16),
        .copy(source: counter, destination: readback, byteCount: 8), .read(buffer: readback, byteCount: 8),
        .write(buffer: counter, offset: 8, words: [0xfeed_beef]),
        .dispatchComputeRanges(pipeline: roundTrip, bindings: [
            .init(binding: 0): .init(buffer: counter, size: 8),
            .init(binding: 1): .init(buffer: output),
        ], workgroups: .init(x: 1)),
        .copy(source: output, destination: readback, byteCount: 40), .read(buffer: readback, byteCount: 40),
        .copy(source: counter, destination: readback, byteCount: 12), .read(buffer: readback, byteCount: 12),
    ])
    let snapshots = try await ticket.wait()
    guard snapshots.count == 6, snapshots[0].count == 10 else {
        throw GPUError.backendFailure("Unexpected WebGPU atomic comparison readback size")
    }
    let result = snapshots[0]
    let signedStart = UInt32(bitPattern: -9)
    let signedAfter: UInt32 = result[1] == 1 ? 11 : signedStart
    let unsignedAfter: UInt32 = result[6] == 1 ? 7 : 0
    let expected: [UInt32] = [signedStart, result[1], signedAfter, signedAfter, 0, 0, result[6], unsignedAfter, 0, 0]
    guard result[1] <= 1, result[6] <= 1, result == expected, snapshots[1] == [unsignedAfter, 999] else {
        throw GPUError.backendFailure("Unexpected WebGPU atomic comparison results")
    }
    let expectedSPIRV: [UInt32] = [0, 7, 7, 0]
    guard snapshots[2] == expectedSPIRV, snapshots[3] == [7, 999] else {
        throw GPUError.backendFailure("Unexpected WebGPU SPIR-V compare-exchange results")
    }
    let expectedRoundTrip: [UInt32] = [
        0xffff_fffb, 0xffff_fffd, 0xffff_fff0, 0xffff_fff0, 0xffff_fff7, 0, 0xffff_fff7, 1, 17, 11,
    ]
    guard snapshots[4] == expectedRoundTrip, snapshots[5] == [17, 11, 0xfeed_beef] else {
        throw GPUError.backendFailure("Unexpected round-trip atomic results: \(snapshots[4...5])")
    }
    return expected.count + expectedSPIRV.count + expectedRoundTrip.count
}

private func atomicCompareExchangeSPIRV() -> [UInt32] {
    func op(_ code: UInt32, _ operands: UInt32...) -> [UInt32] {
        [UInt32(operands.count + 1) << 16 | code] + operands
    }
    var words: [UInt32] = [0x07230203, 0x00010300, 0, 100, 0]
    for instruction in [
        op(17, 1), op(14, 0, 1),
        op(15, 5, 10, 0x6e69616d, 0, 14), op(16, 10, 17, 1, 1, 1),
        op(71, 14, 11, 28), op(71, 3, 6, 4), op(71, 4, 2), op(72, 4, 0, 35, 0),
        op(71, 5, 2), op(72, 5, 0, 35, 0), op(72, 5, 1, 35, 4), op(72, 5, 2, 35, 8), op(72, 5, 3, 35, 12),
        op(71, 11, 34, 0), op(71, 11, 33, 0), op(71, 12, 34, 0), op(71, 12, 33, 1),
        op(19, 1), op(21, 2, 32, 0), op(43, 2, 20, 0), op(43, 2, 21, 1), op(43, 2, 22, 2), op(43, 2, 23, 3),
        op(43, 2, 24, 7), op(43, 2, 25, 99), op(43, 2, 26, 11), op(43, 2, 27, 0),
        op(28, 3, 2, 21), op(30, 4, 3), op(30, 5, 2, 2, 2, 2), op(32, 6, 12, 4), op(32, 7, 12, 5),
        op(32, 8, 12, 2), op(33, 9, 1), op(23, 15, 2, 3), op(32, 16, 1, 15), op(32, 17, 1, 2),
        op(59, 6, 11, 12), op(59, 7, 12, 12), op(59, 16, 14, 1),
        op(54, 1, 10, 0, 9), op(248, 13),
        op(65, 8, 30, 11, 20, 20),
        op(230, 2, 31, 30, 21, 20, 27, 24, 20), // Successful exchange returns 0.
        op(230, 2, 32, 30, 21, 20, 27, 26, 25), // Mismatch returns 7.
        op(65, 17, 33, 14, 20), op(61, 2, 34, 33), op(128, 2, 35, 34, 21),
        op(65, 8, 36, 11, 20, 35),
        op(230, 2, 37, 36, 21, 20, 27, 25, 20), // Robust out-of-bounds exchange returns 0.
        op(227, 2, 38, 30, 21, 20),
        op(65, 8, 40, 12, 20), op(62, 40, 31), op(65, 8, 41, 12, 21), op(62, 41, 32),
        op(65, 8, 42, 12, 22), op(62, 42, 38), op(65, 8, 43, 12, 23), op(62, 43, 37),
        op(253), op(56),
    ] { words += instruction }
    return words
}

private func mixedAtomicDiagnostic(_ device: Device) async throws -> Int {
    let source = """
        struct Mixed {
            @size(12) bias: u32,
            count: atomic<i32>,
            payload: vec2f,
            @align(32) values: array<atomic<u32>>,
        }
        @group(0) @binding(0) var<storage, read_write> data: Mixed;
        @group(0) @binding(1) var<storage, read_write> output: array<u32, 6>;
        @compute @workgroup_size(1)
        fn run(@builtin(global_invocation_id) id: vec3u) {
            output[0] = u32(atomicMin(&data.count, -20i));
            output[1] = u32(atomicMax(&data.count, -6i));
            atomicOr(&data.count, 0i);
            output[2] = u32(atomicLoad(&data.count));
            data.bias += atomicAdd(&data.values[0u], 2u);
            let value = data.payload;
            data.payload = vec2f(value.y, value.x);
            output[3] = arrayLength(&data.values);
            let exchanged = atomicCompareExchangeWeak(&data.values[1u], 4u, 11u);
            if exchanged.exchanged { output[4] = 1u; }
            let outside = atomicCompareExchangeWeak(&data.values[id.x + arrayLength(&data.values)], 0u, 888u);
            output[5] = outside.old_value;
            if outside.exchanged { output[5] = 1u; }
        }
        """
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(source), to: .wgsl)
    let pipeline = try await device.createComputePipeline(shader)
    let data = try await device.createBuffer(.init(size: 324, usage: [.storage, .copyDestination, .copySource]))
    let output = try await device.createBuffer(.init(size: 24, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 324, usage: [.copyDestination, .mapRead]))
    let initial: [UInt32] = [9, 701, 702, UInt32(bitPattern: -10), Float(1.5).bitPattern, Float(-2).bitPattern,
                            703, 704, 3, 4, 0, 0, 0, 0, 0, 0, 999]
    let ticket = try await device.submitCommands([
        .write(buffer: data, offset: 256, words: initial),
        .dispatchComputeRanges(pipeline: pipeline, bindings: [
            .init(binding: 0): .init(buffer: data, offset: 256, size: 64),
            .init(binding: 1): .init(buffer: output),
        ], workgroups: .init(x: 1)),
        .copy(source: output, destination: read, byteCount: 24),
        .read(buffer: read, byteCount: 24),
        .copy(source: data, destination: read, byteCount: 324),
        .read(buffer: read, byteCount: 324),
    ])
    let result = try await ticket.wait()
    guard result.count == 2, result[0].count == 6 else {
        throw GPUError.backendFailure("Unexpected mixed atomic readback size")
    }
    let exchanged = result[0][4]
    let expected: [UInt32] = [
        UInt32(bitPattern: -10), UInt32(bitPattern: -20), UInt32(bitPattern: -6), 8, exchanged, 0,
    ]
    var expectedData = initial
    expectedData[0] = 12
    expectedData[3] = UInt32(bitPattern: -6)
    expectedData.swapAt(4, 5)
    expectedData[8] = 5
    expectedData[9] = exchanged == 1 ? 11 : 4
    guard exchanged <= 1, result == [expected, Array(repeating: UInt32(0), count: 64) + expectedData] else {
        throw GPUError.backendFailure("Unexpected mixed atomic readback")
    }
    return expectedData.count
}

private func uniformLoadDiagnostic(_ device: Device) async throws -> [UInt32] {
    let source = """
        var<workgroup> flag: u32;
        var<workgroup> values: array<vec2u, 2>;
        var<workgroup> counter: atomic<i32>;
        @group(0) @binding(0) var<storage, read_write> output: array<u32, 12>;
        @compute @workgroup_size(4)
        fn run(@builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) group: vec3u) {
            if lane == 0u {
                flag = 11u + group.x;
                values[group.x] = vec2u(7u + group.x, 2u);
                atomicStore(&counter, -5i + i32(group.x));
            }
            let snapshot = workgroupUniformLoad(&flag);
            if lane == 0u { flag = 666u; }
            let next = workgroupUniformLoad(&flag);
            if snapshot != 0u { workgroupBarrier(); }
            let pair = workgroupUniformLoad(&values[group.x]);
            if pair.x != 0u { workgroupBarrier(); }
            let signedValue = workgroupUniformLoad(&counter);
            if signedValue < 0i { workgroupBarrier(); }
            output[group.x * 4u + lane] = snapshot + pair.x + pair.y + u32(signedValue + 10i) + next;
        }
        """
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(source), to: .wgsl)
    let pipeline = try await device.createComputePipeline(shader)
    let roundTripSource = ShaderInput.wgsl("""
        var<workgroup> gates: array<u32, 3>;
        var<workgroup> scratch: array<u32, 4>;
        var<workgroup> atomicGate: atomic<i32>;
        @group(0) @binding(0) var<storage, read_write> output: array<u32, 12>;
        @compute @workgroup_size(4)
        fn round_trip(@builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) group: vec3u) {
            if lane == 0u {
                gates[group.x] = 11u + group.x;
                atomicStore(&atomicGate, 21i + i32(group.x));
            }
            let gate = workgroupUniformLoad(&gates[group.x]);
            let atomicValue = workgroupUniformLoad(&atomicGate);
            scratch[lane] = group.x * 100u + lane + 1u;
            let nextLane = (lane + 1u) % 4u;
            var selected = select(scratch[lane], 0u, group.x == 2u);
            if gate == 12u {
                workgroupBarrier();
                selected = scratch[nextLane];
            }
            if atomicValue == 23i {
                workgroupBarrier();
                selected = scratch[nextLane] - nextLane + lane;
            }
            let mismatch = atomicValue != 21i + i32(group.x);
            output[group.x * 4u + lane] = gate * 100u + selected + select(0u, 1000u, mismatch);
        }
        """)
    guard case .spirv(let words) = try compiler.compile(roundTripSource, to: .spirv) else {
        throw GPUError.backendFailure("Expected binary SPIR-V for workgroup uniform load")
    }
    let roundTripShader = try compiler.compileWebGPUCompute(.spirv(words), entryPoint: "round_trip")
    let roundTrip = try await device.createComputePipeline(roundTripShader)
    let output = try await device.createBuffer(.init(size: 48, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 48, usage: [.copyDestination, .mapRead]))
    let ticket = try await device.submitCommands([
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 3)),
        .copy(source: output, destination: read, byteCount: 48),
        .read(buffer: read, byteCount: 48),
        .dispatchCompute(pipeline: roundTrip, bindings: [.init(binding: 0): output], workgroups: .init(x: 3)),
        .copy(source: output, destination: read, byteCount: 48),
        .read(buffer: read, byteCount: 48),
    ])
    let directWords: [UInt32] = [691, 691, 691, 691, 694, 694, 694, 694, 686, 686, 686, 686]
    let roundTripWords: [UInt32] = [1101, 1102, 1103, 1104, 1302, 1303, 1304, 1301, 1501, 1502, 1503, 1504]
    let snapshots = try await ticket.wait()
    guard snapshots == [directWords, roundTripWords] else {
        throw GPUError.backendFailure("Unexpected WebGPU uniform load readback: \(snapshots)")
    }
    return snapshots.flatMap { $0 }
}

private func selectDiagnostic(_ device: Device) async throws -> [UInt32] {
    let source = """
        @group(0) @binding(0) var<storage, read_write> trace: u32;
        @group(0) @binding(1) var<storage, read_write> output: array<u32, 8>;
        fn mark(n: u32) -> u32 { trace = trace * 10u + n; return n; }
        @compute @workgroup_size(1)
        fn run(@builtin(global_invocation_id) id: vec3u) {
            let scalar = select(mark(2u), mark(3u), true);
            let mask = vec3u(id.x, id.x + 1u, id.x + 2u) < vec3u(1u, 1u, 3u);
            let lanes = select(vec3u(10u, 20u, 30u), vec3u(40u, 50u, 60u), mask);
            let pair = select(vec2i(-1i, -2i), vec2i(3i, 4i), false);
            let floats = select(vec2f(1.5f, 2.5f), vec2f(3.5f, 4.5f), vec2<bool>(false, true));
            let valid = select(false, true, floats.x == 1.5f && floats.y == 4.5f);
            output[0] = scalar;
            output[1] = trace;
            output[2] = lanes.x;
            output[3] = lanes.y;
            output[4] = lanes.z;
            output[5] = u32(pair.x);
            output[6] = u32(pair.y);
            output[7] = select(0u, 1, valid);
        }
        """
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(source), to: .wgsl)
    let pipeline = try await device.createComputePipeline(shader)
    let trace = try await device.createBuffer(.init(size: 4, usage: .storage))
    let output = try await device.createBuffer(.init(size: 32, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 32, usage: [.copyDestination, .mapRead]))
    let ticket = try await device.submitCommands([
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): trace, .init(binding: 1): output],
                         workgroups: .init(x: 1)),
        .copy(source: output, destination: read, byteCount: 32),
        .read(buffer: read, byteCount: 32),
    ])
    let words: [UInt32] = [3, 23, 40, 20, 60, .max, .max - 1, 1]
    guard try await ticket.wait() == [words] else { throw GPUError.backendFailure("Unexpected WebGPU select results") }
    return words
}

private func constantSelectDiagnostic(_ device: Device) async throws -> [UInt32] {
    let source = """
        const enabled = select(false, true, true);
        const scale = select(4u, 8u, enabled);
        const bias = select(-2i, -9i, false);
        const factor = select(0.25f, 0.5f, enabled);
        const lanes = select(vec4u(1u, 2u, 3u, 4u), vec4u(8u, 9u, 10u, 11u), vec4<bool>(true, false, true, false));
        const_assert select(0u, scale, enabled) == 8u;
        @group(0) @binding(0) var<storage, read_write> output: array<u32, select(2u, 4u, enabled)>;
        fn adjustment() -> u32 { return scale + u32(bias + 10i); }
        @compute @workgroup_size(select(2, 4, enabled))
        fn run(@builtin(global_invocation_id) id: vec3u) {
            if enabled && factor == 0.5f {
                output[id.x] = adjustment() + lanes.x + lanes.y + lanes.z + lanes.w + id.x;
            }
        }
        """
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(source), to: .wgsl)
    let pipeline = try await device.createComputePipeline(shader)
    let output = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 16, usage: [.copyDestination, .mapRead]))
    let ticket = try await device.submitCommands([
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
        .copy(source: output, destination: read, byteCount: 16),
        .read(buffer: read, byteCount: 16),
    ])
    let words: [UInt32] = [40, 41, 42, 43]
    guard try await ticket.wait() == [words] else { throw GPUError.backendFailure("Unexpected WebGPU constant select results") }
    return words
}

private func computeUniformDiagnostic(_ device: Device) async throws -> [UInt32] {
    let source = """
        struct Config {
            normal: vec3u,
            bias: u32,
            @align(32) table: array<vec4u, 2>,
        }
        @group(0) @binding(0) var<uniform> cfg: Config;
        @group(3) @binding(9) var<uniform> tuning: u32;
        @group(0) @binding(1) var<storage, read_write> output: array<u32, 4>;
        fn score(i: u32) -> u32 {
            let row = cfg.table[i];
            return cfg.normal.z + cfg.bias + tuning + row.x + row.w;
        }
        @compute @workgroup_size(4)
        fn run(@builtin(global_invocation_id) id: vec3u) { output[id.x] = score(id.x); }
        """
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(source), to: .wgsl)
    let pipeline = try await device.createComputePipeline(shader)
    let cfg = try await device.createBuffer(.init(size: 320, usage: [.uniform, .copyDestination]))
    let tuning = try await device.createBuffer(.init(size: 4, usage: [.uniform, .copyDestination]))
    let output = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 16, usage: [.copyDestination, .mapRead]))
    let ticket = try await device.submitCommands([
        .write(buffer: cfg, offset: 256, words: [1, 2, 3, 4, 701, 702, 703, 704, 5, 6, 7, 8, 9, 10, 11, 12]),
        .write(buffer: tuning, offset: 0, words: [10]),
        .dispatchComputeRanges(pipeline: pipeline, bindings: [
            .init(binding: 0): .init(buffer: cfg, offset: 256, size: 64),
            .init(group: 3, binding: 9): .init(buffer: tuning),
            .init(binding: 1): .init(buffer: output),
        ], workgroups: .init(x: 1)),
        .copy(source: output, destination: read, byteCount: 16),
        .read(buffer: read, byteCount: 16),
    ])
    let words: [UInt32] = [30, 38, 17, 17]
    guard try await ticket.wait() == [words] else { throw GPUError.backendFailure("Unexpected WebGPU uniform results") }
    return words
}

private func packedVertexProbes(
    _ device: Device, compiler: IndependentShaderCompiler, readback: ResourceID
) async throws -> (commands: [BufferCommand], masks: [UInt32]) {
    let families: [([VertexFormat], [UInt8], String)] = [
        ([.uint8, .uint8x2, .uint8x4], [202], "202u"),
        ([.sint8, .sint8x2, .sint8x4], [0x81], "-127i"),
        ([.unorm8, .unorm8x2, .unorm8x4], [255], "1.0"),
        ([.snorm8, .snorm8x2, .snorm8x4], [0x80], "-1.0"),
        ([.uint16, .uint16x2, .uint16x4], [0x34, 0x12], "4660u"),
        ([.sint16, .sint16x2, .sint16x4], [0xff, 0xfe], "-257i"),
        ([.unorm16, .unorm16x2, .unorm16x4], [0, 0x40], "16384.0 / 65535.0"),
        ([.snorm16, .snorm16x2, .snorm16x4], [0, 0x80], "-1.0"),
    ]
    typealias Sample = (format: VertexFormat, bytes: [UInt8], expected: String)
    func scalar(_ format: VertexFormat) -> String {
        switch format.dataType {
        case .float: "f32"
        case .sint: "i32"
        case .uint: "u32"
        }
    }
    func type(_ format: VertexFormat) -> String {
        format.componentCount == 1 ? scalar(format) : "vec\(format.componentCount)<\(scalar(format))>"
    }
    var samples: [Sample] = families.flatMap { formats, bytes, value in
        formats.map { format in
            let data = (0..<format.componentCount).flatMap { _ in bytes }
            return (format, data, format.componentCount == 1 ? value : "\(type(format))(\(value))")
        }
    }
    samples += [
        (.unorm1010102, [0, 0xfc, 0x0f, 0x40], "vec4f(0, 1, 0, 1.0 / 3.0)"),
        (.unorm8x4BGRA, [16, 32, 96, 255], "vec4f(96.0 / 255.0, 32.0 / 255.0, 16.0 / 255.0, 1)"),
    ]
    if device.capabilities.features.contains(.vertexFormatSnorm1010102) {
        samples.append((.snorm1010102, [0, 0x02, 0xf0, 0x9f], "vec4f(-1, 0, 1, -1)"))
    } else {
        let source = ShaderInput.wgsl("""
            @vertex fn vs(@location(0) value: vec4f) -> @builtin(position) vec4f { return value; }
            @fragment fn fs() -> @location(0) vec4u { return vec4u(0); }
            """)
        let vertex = try compiler.compileWebGPURenderStage(source, entryPoint: "vs")
        let fragment = try compiler.compileWebGPURenderStage(source, entryPoint: "fs")
        do {
            _ = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Uint,
                vertexBuffers: [.init(stride: 4, attributes: [.init(location: 0, offset: 0, format: .snorm1010102)])])
            throw GPUError.backendFailure("Browser accepted an unsupported packed vertex format")
        } catch GPUError.validation("Feature not advertised") {}
    }

    let capacity = Int(min(15, device.capabilities.maxVertexAttributes - 1))
    let target = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Uint,
        usage: [.renderAttachment, .copySource]))
    var commands: [BufferCommand] = []
    var masks: [UInt32] = []
    for start in stride(from: 0, to: samples.count, by: capacity) {
        let chunk = samples[start..<min(samples.count, start + capacity)]
        var bytes: [UInt8] = []
        var attributes: [VertexAttribute] = []
        var fields: [String] = []
        var checks: [String] = []
        for (index, sample) in chunk.enumerated() {
            while bytes.count % min(4, sample.bytes.count) != 0 { bytes.append(0) }
            attributes.append(.init(location: UInt32(index), offset: UInt32(bytes.count), format: sample.format))
            bytes += sample.bytes
            let value = "input.a\(index)"
            let name = type(sample.format)
            fields.append("@location(\(index)) a\(index): \(name),")
            let test = sample.format.dataType == .float
                ? "abs(\(value) - (\(sample.expected))) < \(name)(0.001)"
                : "\(value) == \(sample.expected)"
            let condition = sample.format.componentCount == 1 ? test : "all(\(test))"
            checks.append("if \(condition) { mask |= \(1 << index)u; }")
        }
        while bytes.count % 4 != 0 { bytes.append(0) }
        let source = ShaderInput.wgsl("""
            struct Inputs { \(fields.joined(separator: "\n")) }
            struct Raster { @builtin(position) position: vec4f, @location(0) @interpolate(flat) mask: u32, }
            @vertex fn vs(input: Inputs, @builtin(vertex_index) i: u32) -> Raster {
                let points = array(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
                var mask = 0u;
                \(checks.joined(separator: "\n"))
                return Raster(vec4f(points[i], 0, 1), mask);
            }
            @fragment fn fs(in: Raster) -> @location(0) vec4u {
                return (vec4u(in.mask) >> vec4u(0, 8, 16, 24)) & vec4u(255u);
            }
            """)
        let vertex = try compiler.compileWebGPURenderStage(source, entryPoint: "vs")
        let fragment = try compiler.compileWebGPURenderStage(source, entryPoint: "fs")
        let pipeline = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Uint,
            vertexBuffers: [.init(stride: 0, attributes: []),
                            .init(stride: 0, stepMode: .instance, attributes: attributes)])
        let stream = try await device.createBuffer(.init(size: UInt64(bytes.count), usage: [.vertex, .copyDestination]))
        commands += [
            .writeBytes(buffer: stream, offset: 0, bytes: bytes),
            .render(pipeline: pipeline, target: target, clear: .transparentBlack, vertexCount: 3,
                    vertexBuffers: [.init(buffer: stream, slot: device.capabilities.maxVertexBuffers - 1),
                                    .init(buffer: stream, slot: 1)]),
            .copyTextureToBuffer(source: target, destination: readback, bytesPerRow: 256),
            .read(buffer: readback, byteCount: 4),
        ]
        masks.append((1 << chunk.count) - 1)
    }
    return (commands, masks)
}

private func renderUniformDiagnostic(_ device: Device) async throws -> [UInt32] {
    let source = ShaderInput.wgsl("""
        struct DrawConfig { offset: vec2f, color: vec4f }
        @group(3) @binding(11) var<uniform> cfg: DrawConfig;
        @group(0) @binding(0) var<uniform> jitter: vec2f;
        @group(1) @binding(1) var<uniform> unused: vec4f;
        fn tint() -> vec4f { return cfg.color; }
        @vertex fn vs(@location(0) point: vec2f) -> @builtin(position) vec4f {
            return vec4f(point.x + cfg.offset.x + jitter.x, point.y + cfg.offset.y + jitter.y, 0.0f, 1.0f);
        }
        @fragment fn fs() -> @location(0) vec4f { return tint(); }
        """)
    let compiler = IndependentShaderCompiler()
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .wgsl)
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: .wgsl)
    let pipeline = try await device.createRenderPipeline(
        vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm,
        vertexBuffers: [.init(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float32x2)])])
    let attributeLimit = device.capabilities.maxVertexAttributes
    let interstageLimit = device.capabilities.maxInterStageShaderVariables
    guard attributeLimit > 1, interstageLimit > 0 else {
        throw GPUError.backendFailure("No additional vertex attributes or inter-stage variables available")
    }
    func highInput(_ location: UInt32, varying: UInt32) -> ShaderInput {
        .wgsl("""
            struct Raster {
                @builtin(position) position: vec4f,
                @location(\(varying)u) @interpolate(flat) tag: u32,
            }
            @vertex fn vs(@location(0) point: vec2f, @location(\(location)u) tag: u32) -> Raster {
                return Raster(vec4f(point, 0, 1), tag);
            }
            @fragment fn fs(in: Raster) -> @location(0) vec4u {
                return vec4u(in.tag, 0, 0, 255);
            }
            """)
    }
    let positionLayout = VertexBufferLayout(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float32x2)])
    func tagLayout(_ location: UInt32) -> VertexBufferLayout {
        .init(stride: 0, stepMode: .instance, attributes: [.init(location: location, offset: 0, format: .uint32)])
    }
    let highLocation = attributeLimit - 1
    let highSource = highInput(highLocation, varying: interstageLimit - 1)
    let highFragment = try compiler.compileWebGPURenderStage(highSource, entryPoint: "fs")
    let highVertex = try compiler.compileWebGPURenderStage(highSource, entryPoint: "vs")
    let highPipeline = try await device.createRenderPipeline(vertex: highVertex, fragment: highFragment,
        colorFormat: .rgba8Uint, vertexBuffers: [positionLayout, tagLayout(highLocation)])
    let overVertex = try compiler.compileWebGPURenderStage(highInput(attributeLimit, varying: interstageLimit - 1),
                                                          entryPoint: "vs")
    do {
        _ = try await device.createRenderPipeline(vertex: overVertex, fragment: highFragment,
            colorFormat: .rgba8Uint, vertexBuffers: [positionLayout, tagLayout(attributeLimit)])
        throw GPUError.backendFailure("Pipeline accepted an attribute past the negotiated vertex input limit")
    } catch GPUError.validation("Invalid, duplicate or unaligned vertex attribute") {}
    let overInterstageSource = highInput(highLocation, varying: interstageLimit)
    let overInterstageVertex = try compiler.compileWebGPURenderStage(overInterstageSource, entryPoint: "vs")
    let overInterstageFragment = try compiler.compileWebGPURenderStage(overInterstageSource, entryPoint: "fs")
    do {
        _ = try await device.createRenderPipeline(vertex: overInterstageVertex, fragment: overInterstageFragment,
            colorFormat: .rgba8Uint, vertexBuffers: [positionLayout, tagLayout(highLocation)])
        throw GPUError.backendFailure("Pipeline accepted a varying past the negotiated inter-stage limit")
    } catch GPUError.validation("Vertex outputs exceed the inter-stage limit") {}
    let tag = try await device.createBuffer(.init(size: 4, usage: [.vertex, .copyDestination]))
    let highTarget = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Uint,
        usage: [.renderAttachment, .copySource]))
    let uniform = try await device.createBuffer(.init(size: 288, usage: [.uniform, .copyDestination]))
    let jitter = try await device.createBuffer(.init(size: 8, usage: [.uniform, .copyDestination]))
    let vertices = try await device.createBuffer(.init(size: 24, usage: [.vertex, .copyDestination]))
    let indices = try await device.createBuffer(.init(size: 8, usage: [.index, .copyDestination]))
    let texture = try await device.createTexture(.init(width: 8, height: 8, format: .rgba8Unorm,
                                                       usage: [.renderAttachment, .copySource]))
    let read = try await device.createBuffer(.init(size: 1824, usage: [.copyDestination, .mapRead]))
    let packedProbes = try await packedVertexProbes(device, compiler: compiler, readback: read)
    let clear = RenderClearColor(red: 1, green: 0, blue: 0, alpha: 1)
    let bindings: [ComputeBindingSlot: ComputeBufferBinding] = [
        .init(group: 3, binding: 11): .init(buffer: uniform, offset: 256, size: 32),
        .init(binding: 0): .init(buffer: jitter),
    ]
    let streams = [VertexBufferBinding(buffer: vertices)]
    let ticket = try await device.submitCommands([
        .write(buffer: uniform, offset: 256, words: [Float(0.75).bitPattern, 0, 0, 0, 0, Float(1).bitPattern, 0, Float(1).bitPattern]),
        .write(buffer: jitter, offset: 0, words: [Float(0.25).bitPattern, 0]),
        .write(buffer: vertices, offset: 0, words: [Float(-1), -1, 3, -1, -1, 3].map(\.bitPattern)),
        .write(buffer: indices, offset: 0, words: [0x00010000, 2]),
        .render(pipeline: pipeline, target: texture, clear: clear, vertexCount: 3,
                vertexBuffers: streams, uniforms: bindings),
        .copyTextureToBuffer(source: texture, destination: read, bytesPerRow: 256),
        .read(buffer: read, byteCount: 1824),
        .renderIndexed(pipeline: pipeline, target: texture, clear: clear,
                       indexBuffer: .init(buffer: indices, format: .uint16), draw: .init(indexCount: 3),
                       vertexBuffers: streams, uniforms: bindings),
        .copyTextureToBuffer(source: texture, destination: read, bytesPerRow: 256),
        .read(buffer: read, byteCount: 1824),
        .write(buffer: tag, offset: 0, words: [77]),
        .render(pipeline: highPipeline, target: highTarget, clear: .transparentBlack, vertexCount: 3,
                vertexBuffers: streams + [.init(buffer: tag)]),
        .copyTextureToBuffer(source: highTarget, destination: read, bytesPerRow: 256),
        .read(buffer: read, byteCount: 4),
    ] + packedProbes.commands)
    let snapshots = try await ticket.wait()
    guard snapshots.map(\.count) == [456, 456, 1] + packedProbes.masks.map({ _ in 1 }) else {
        throw GPUError.backendFailure("Unexpected uniform render readback size")
    }
    let pixels = snapshots.prefix(2).flatMap { [$0[4 * 64 + 1], $0[4 * 64 + 6]] } + snapshots.dropFirst(2).flatMap { $0 }
    let expected: [UInt32] = [0xff0000ff, 0xff00ff00, 0xff0000ff, 0xff00ff00, 0xff00004d] + packedProbes.masks
    guard pixels == expected else {
        throw GPUError.backendFailure("Unexpected uniform render pixels: \(pixels)")
    }
    return pixels
}

private func matrixDiagnostic(_ device: Device) async throws -> [Float] {
    let source = """
        @group(0) @binding(0) var<storage, read_write> output: array<f32, 18>;
        fn product(a: mat2x3f, b: mat4x2f) -> mat4x3f { return a * b; }
        fn column(a: mat2x3f, v: vec2f) -> vec3f { return a * v; }
        fn row(v: vec3f, a: mat2x3f) -> vec2f { return v * a; }
        fn seed(i: u32) -> f32 { if i == 0u { return 1.0f; } return 2.0f; }
        @compute @workgroup_size(1)
        fn run(@builtin(global_invocation_id) id: vec3u) {
            let a = mat2x3f(vec3f(seed(id.x), 2.0f, 3.0f), vec3f(4.0f, 5.0f, 6.0f));
            let b = mat4x2f(vec2f(7.0f, 8.0f), vec2f(9.0f, 10.0f), vec2f(11.0f, 12.0f), vec2f(13.0f, 14.0f));
            let p = product(a, b);
            let first = p[0u];
            let last = p[id.x + 3u];
            let cv = column(a, vec2f(2.0f, 3.0f));
            let rv = row(vec3f(2.0f, 3.0f, 4.0f), a);
            let scaled = (a * 2.0f) + (3.0f * a) - a;
            var scratch: mat2x3f;
            scratch[id.x + 1u] = vec3f(7.0f, 8.0f, 9.0f);
            scratch[i32(id.x) - 1i] = vec3f(999.0f);
            let selected = scratch[id.x + 1u];
            let outside = scratch[id.x + 2u];
            output[0] = first.x; output[1] = first.y; output[2] = first.z;
            output[3] = last.x; output[4] = last.y; output[5] = last.z;
            output[6] = cv.x; output[7] = cv.y; output[8] = cv.z;
            output[9] = rv.x; output[10] = rv.y; output[11] = scaled[1u].z;
            output[12] = selected.x; output[13] = selected.y; output[14] = selected.z;
            output[15] = outside.x; output[16] = outside.y; output[17] = outside.z;
        }
        """
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(source), to: .wgsl)
    let pipeline = try await device.createComputePipeline(shader)
    let roundTripSource = ShaderInput.wgsl("""
        @group(0) @binding(0) var<storage, read_write> output: array<f32, 18>;
        @compute @workgroup_size(1)
        fn round_trip(@builtin(global_invocation_id) id: vec3u) {
            let a = mat2x3f(vec3f(1.0 + f32(id.x), -2.0, 3.0), vec3f(4.0, 5.0, -6.0));
            let b = mat4x2f(vec2f(2.0, 3.0), vec2f(5.0, 7.0), vec2f(11.0, 13.0), vec2f(17.0, 19.0));
            let product = a * b;
            let transposed = transpose(a);
            output[0] = product[0].x; output[1] = product[0].y; output[2] = product[0].z;
            output[3] = product[1].x; output[4] = product[1].y; output[5] = product[1].z;
            output[6] = product[2].x; output[7] = product[2].y; output[8] = product[2].z;
            output[9] = product[3].x; output[10] = product[3].y; output[11] = product[3].z;
            output[12] = transposed[0].x; output[13] = transposed[0].y;
            output[14] = transposed[1].x; output[15] = transposed[1].y;
            output[16] = transposed[2].x; output[17] = transposed[2].y;
        }
        """)
    guard case .spirv(let binary) = try compiler.compile(roundTripSource, to: .spirv) else {
        throw GPUError.backendFailure("Expected binary SPIR-V for matrix arithmetic")
    }
    let roundTripShader = try compiler.compileWebGPUCompute(.spirv(binary), entryPoint: "round_trip")
    let roundTrip = try await device.createComputePipeline(roundTripShader)
    let output = try await device.createBuffer(.init(size: 72, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 72, usage: [.copyDestination, .mapRead]))
    let ticket = try await device.submitCommands([
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
        .copy(source: output, destination: read, byteCount: 72),
        .read(buffer: read, byteCount: 72),
        .dispatchCompute(pipeline: roundTrip, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
        .copy(source: output, destination: read, byteCount: 72),
        .read(buffer: read, byteCount: 72),
    ])
    let expected: [Float] = [39, 54, 69, 69, 96, 123, 14, 19, 24, 20, 47, 24, 7, 8, 9, 0, 0, 0]
    let roundTripExpected: [Float] = [14, 11, -12, 33, 25, -27, 63, 43, -45, 93, 61, -63, 1, 4, -2, 5, 3, -6]
    let snapshots = try await ticket.wait()
    guard snapshots == [expected.map(\.bitPattern), roundTripExpected.map(\.bitPattern)] else {
        throw GPUError.backendFailure("Unexpected WebGPU matrix results: \(snapshots)")
    }
    return snapshots.flatMap { $0.map(Float.init(bitPattern:)) }
}

private func matrixBufferDiagnostic(_ device: Device) async throws -> [Float] {
    let source = """
        struct Data { basis: mat2x3f, transforms: array<mat2x3f> }
        @group(0) @binding(0) var<storage, read_write> data: Data;
        @group(0) @binding(1) var<uniform> scale: mat2x2f;
        @group(0) @binding(2) var<storage, read_write> output: array<f32, 4>;
        @compute @workgroup_size(1)
        fn run(@builtin(global_invocation_id) id: vec3u) {
            let saved = data.basis;
            data.basis = saved * scale;
            let i = id.x + 1u;
            data.transforms[i] = data.transforms[0u] * scale;
            data.transforms[i][id.x] = vec3f(21.0f, 22.0f, 23.0f);
            data.transforms[i][1u][1u] = 55.0f;
            data.transforms[id.x + 2u] = saved;
            let missing = data.transforms[id.x + 2u];
            output[0] = missing[0u].x;
            if arrayLength(&data.transforms) == 2u { output[1] = 1.0f; }
            output[2] = data.transforms[i][1u].y;
            output[3] = data.basis[1u].z;
        }
        """
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(source), to: .wgsl)
    let pipeline = try await device.createComputePipeline(shader)
    let roundTripSource = ShaderInput.wgsl("""
        struct Data { basis: mat2x3f, transforms: array<mat2x3f> }
        @group(0) @binding(0) var<storage, read_write> data: Data;
        @group(0) @binding(2) var<storage, read_write> output: array<f32, 4>;
        @compute @workgroup_size(1)
        fn round_trip(@builtin(global_invocation_id) id: vec3u) {
            let factor = mat2x2f(vec2f(2.0, 1.0), vec2f(-1.0, 3.0));
            data.basis = data.basis * factor;
            let index = id.x + 1u;
            data.transforms[index] = data.transforms[0u] * factor;
            data.transforms[index][id.x] = vec3f(21.0, 22.0, 23.0);
            data.transforms[index][1u][1u] = 45.0;
            output[0] = f32(arrayLength(&data.transforms));
            output[1] = data.transforms[index][id.x].x;
            output[2] = data.transforms[index][1u].y;
            output[3] = data.basis[1u].z;
        }
        """)
    guard case .spirv(let binary) = try compiler.compile(roundTripSource, to: .spirv) else {
        throw GPUError.backendFailure("Expected binary SPIR-V for matrix buffers")
    }
    let roundTripShader = try compiler.compileWebGPUCompute(.spirv(binary), entryPoint: "round_trip")
    let roundTrip = try await device.createComputePipeline(roundTripShader)
    let data = try await device.createBuffer(.init(size: 356, usage: [.storage, .copyDestination, .copySource]))
    let scale = try await device.createBuffer(.init(size: 16, usage: [.uniform, .copyDestination]))
    let output = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 356, usage: [.copyDestination, .mapRead]))
    func column(_ x: Float, _ y: Float, _ z: Float, _ padding: UInt32) -> [UInt32] {
        [x.bitPattern, y.bitPattern, z.bitPattern, padding]
    }
    let initial = [
        column(1, 2, 3, 701), column(4, 5, 6, 702),
        column(7, 8, 9, 703), column(10, 11, 12, 704),
        column(13, 14, 15, 705), column(16, 17, 18, 706), [999],
    ].flatMap { $0 }
    let expected = [
        column(2, 4, 6, 701), column(12, 15, 18, 702),
        column(7, 8, 9, 703), column(10, 11, 12, 704),
        column(21, 22, 23, 705), column(30, 55, 36, 706), [999],
    ].flatMap { $0 }
    let roundTripExpected = [
        column(6, 9, 12, 701), column(11, 13, 15, 702),
        column(7, 8, 9, 703), column(10, 11, 12, 704),
        column(21, 22, 23, 705), column(23, 45, 27, 706), [999],
    ].flatMap { $0 }
    let range = ComputeBufferBinding(buffer: data, offset: 256, size: 96)
    let ticket = try await device.submitCommands([
        .write(buffer: data, offset: 256, words: initial),
        .write(buffer: scale, offset: 0, words: [Float(2).bitPattern, 0, 0, Float(3).bitPattern]),
        .dispatchComputeRanges(pipeline: pipeline, bindings: [
            .init(binding: 0): range,
            .init(binding: 1): .init(buffer: scale),
            .init(binding: 2): .init(buffer: output),
        ], workgroups: .init(x: 1)),
        .copy(source: output, destination: read, byteCount: 16),
        .read(buffer: read, byteCount: 16),
        .copy(source: data, destination: read, byteCount: 356),
        .read(buffer: read, byteCount: 356),
        .write(buffer: data, offset: 256, words: initial),
        .dispatchComputeRanges(pipeline: roundTrip, bindings: [
            .init(binding: 0): range,
            .init(binding: 2): .init(buffer: output),
        ], workgroups: .init(x: 1)),
        .copy(source: output, destination: read, byteCount: 16),
        .read(buffer: read, byteCount: 16),
        .copy(source: data, sourceOffset: 256, destination: read, byteCount: 100),
        .read(buffer: read, byteCount: 100),
    ])
    let values: [Float] = [0, 1, 55, 18]
    let roundTripValues: [Float] = [2, 21, 45, 15]
    let result = try await ticket.wait()
    guard result == [values.map(\.bitPattern), Array(repeating: UInt32(0), count: 64) + expected,
                     roundTripValues.map(\.bitPattern), roundTripExpected] else {
        throw GPUError.backendFailure("Unexpected WebGPU matrix buffer readback: \(result.map { Array($0.suffix(25)) })")
    }
    return values + roundTripValues
}

private func continuingTargetsDiagnostic(_ device: Device) async throws -> [UInt32] {
    let source = """
        @group(0) @binding(0) var<storage, read_write> output: u32;
        @compute @workgroup_size(1) fn run() {
            var n = 0u;
            var total = 0u;
            loop {
                continuing {
                    switch n {
                        case 0u: { total += 2u; break; }
                        default: { total += 5u; }
                    }
                    var j = 0u;
                    loop {
                        j++;
                        if j < 2u { continue; }
                        total += j;
                        break;
                    }
                    n++;
                    break if n == 3u;
                }
            }
            output = total;
        }
        """
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(source), to: .wgsl)
    let pipeline = try await device.createComputePipeline(shader)
    let output = try await device.createBuffer(.init(size: 4, usage: [.storage, .copySource]))
    let readback = try await device.createBuffer(.init(size: 4, usage: [.mapRead, .copyDestination]))
    let result = try await device.submitCommands([
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
        .copy(source: output, destination: readback, byteCount: 4),
        .read(buffer: readback, byteCount: 4),
    ]).wait()
    guard result == [[18]] else {
        throw GPUError.backendFailure("Unexpected continuing targets readback: \(result)")
    }
    return result[0]
}

private func switchDiagnostic(_ device: Device) async throws -> [UInt32] {
    let source = ShaderInput.glsl("""
        precision highp float;
        layout(constant_id=5) const uint adjustment = 4u;
        layout(local_size_x=1) in;
        layout(set=0,binding=0) buffer Data { uint words[8]; } data;
        uint mapped(int i) {
            const int alternative = -2 + 1;
            switch (i) {
                case -2: case alternative: return 7u;
                case 0: return 11u;
                case 9: case (-2147483647 - 1): default: return 19u;
            }
        }
        uint mapped(uint i) { return mapped(int(i)) + adjustment; }
        void advanceValue(inout uint value, out uint previous) {
            previous = value;
            value += 1u;
        }
        uint choose() { data.words[7] += 1u; return 2u; }
        void main() {
            uint total = 0u;
            uint i;
            uint iterations;
            for (i = 0u, iterations = 0u; i < 4u; ++i, iterations += 1u) {
                switch (i) {
                    case 0u: case 2u: total += 2u; continue;
                    case 1u: total += 9u;
                    default: total += 1u; break;
                }
                total += 100u;
            }
            bvec2 expected = equal(lessThan(uvec2(iterations, total), uvec2(5u, 200u)), bvec2(true, false));
            if (all(expected)) { total += iterations; } else { total = 0u; }
            uvec2 endpoints = uvec2(mapped(-2), mapped(0));
            endpoints.yx = endpoints.xy;
            mat2 reciprocal = inverse(mat2(4, 2, 2, 2));
            uint math = uint(4.0 * mod(-2.5, 2.0) + roundEven(1.5) + length(vec2(3, 4)) + reciprocal[1].y);
            uint selected = mapped(5u);
            uint previous;
            advanceValue(selected, previous);
            uint variants[2] = uint[2](previous, selected);
            data.words[0] = endpoints.y; data.words[1] = mapped(-1); data.words[2] = endpoints.x;
            data.words[3] = mapped(9); data.words[4] = variants[0] + variants[1] + math - 26u; data.words[5] = total;
            switch (choose()) {
                case 2u: data.words[6] = 23u; break;
                default: data.words[6] = 99u;
            }
        }
        """, stage: .compute, defines: [:])
    let shader = try IndependentShaderCompiler().compileWebGPUCompute(source, constants: ["5": 6])
    let pipeline = try await device.createComputePipeline(shader)
    let spirvShader = try IndependentShaderCompiler().compileWebGPUCompute(.spirv(switchPhiSPIRV()))
    let spirvPipeline = try await device.createComputePipeline(spirvShader)
    let output = try await device.createBuffer(.init(size: 32, usage: [.storage, .copySource, .copyDestination]))
    let read = try await device.createBuffer(.init(size: 32, usage: [.copyDestination, .mapRead]))
    var commands: [BufferCommand] = [
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
        .copy(source: output, destination: read, byteCount: 32),
        .read(buffer: read, byteCount: 32),
    ]
    for selector: UInt32 in [.max, 0, 2, 9, 17] {
        commands += [
            .write(buffer: output, offset: 0, words: [selector]),
            .dispatchCompute(pipeline: spirvPipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
            .copy(source: output, sourceOffset: 4, destination: read, byteCount: 4),
            .read(buffer: read, byteCount: 4),
        ]
    }
    for postTest in [false, true] {
        let loopShader = try IndependentShaderCompiler().compileWebGPUCompute(.spirv(structuredLoopSPIRV(postTest: postTest)))
        let loopPipeline = try await device.createComputePipeline(loopShader)
        commands += [
            .dispatchCompute(pipeline: loopPipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
            .copy(source: output, destination: read, byteCount: 4),
            .read(buffer: read, byteCount: 4),
        ]
    }
    let snapshotShader = try IndependentShaderCompiler().compileWebGPUCompute(.spirv(escapingLoopValueSPIRV()))
    let snapshotPipeline = try await device.createComputePipeline(snapshotShader)
    commands += [
        .write(buffer: output, offset: 0, words: [7, 0, 0]),
        .dispatchCompute(pipeline: snapshotPipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
        .copy(source: output, destination: read, byteCount: 12),
        .read(buffer: read, byteCount: 12),
    ]
    let pathShader = try IndependentShaderCompiler().compileWebGPUCompute(.spirv(escapingAccessPathSPIRV()))
    let pathPipeline = try await device.createComputePipeline(pathShader)
    commands += [
        .write(buffer: output, offset: 0, words: [0, 10, 20, 30, 40]),
        .dispatchCompute(pipeline: pathPipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
        .copy(source: output, destination: read, byteCount: 20),
        .read(buffer: read, byteCount: 20),
    ]
    let callShader = try IndependentShaderCompiler().compileWebGPUCompute(.spirv(forwardLoopSPIRV()))
    let callPipeline = try await device.createComputePipeline(callShader)
    commands += [
        .dispatchCompute(pipeline: callPipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
        .copy(source: output, destination: read, byteCount: 4),
        .read(buffer: read, byteCount: 4),
    ]
    let words: [UInt32] = [7, 7, 11, 19, 39, 219, 23, 1]
    let result = try await device.submitCommands(commands).wait()
    guard result == [words, [28], [28], [50], [99], [38], [22], [64], [99, 40, 43], [0, 10, 20, 99, 40], [3]] else {
        throw GPUError.backendFailure("Unexpected WebGPU switch results: \(result)")
    }
    return result.flatMap { $0 }
}

private func switchPhiSPIRV() -> [UInt32] {
    func op(_ code: UInt32, _ operands: UInt32...) -> [UInt32] {
        [UInt32(operands.count + 1) << 16 | code] + operands
    }
    // Grouped cases enter the default and then the next case, with separate phi inputs on each edge.
    var words: [UInt32] = [0x07230203, 0x00010300, 0, 100, 0]
    for instruction in [
        op(17, 1), op(14, 0, 1), op(15, 5, 10, 0x6e69616d, 0), op(16, 10, 17, 1, 1, 1),
        op(71, 4, 2), op(72, 4, 0, 35, 0), op(72, 4, 1, 35, 4), op(71, 8, 34, 0), op(71, 8, 33, 0),
        op(19, 1), op(21, 2, 32, 1), op(21, 3, 32, 0), op(30, 4, 2, 2), op(32, 5, 12, 4), op(32, 6, 12, 2), op(33, 7, 1),
        op(43, 3, 11, 0), op(43, 3, 12, 1), op(43, 2, 13, 10), op(43, 2, 14, 11),
        op(43, 2, 15, 31), op(43, 2, 16, 7), op(43, 2, 17, 50), op(43, 2, 18, 99), op(59, 5, 8, 12),
        op(54, 1, 10, 0, 7), op(248, 20), op(65, 6, 25, 8, 11), op(61, 2, 26, 25), op(247, 24, 0),
        op(251, 26, 22, .max, 21, 0, 21, 2, 23, 9, 24),
        op(248, 21), op(128, 2, 27, 13, 14), op(249, 22),
        op(248, 22), op(245, 2, 28, 15, 20, 27, 21), op(128, 2, 29, 28, 16), op(249, 23),
        op(248, 23), op(245, 2, 30, 17, 20, 29, 22), op(249, 24),
        op(248, 24), op(245, 2, 31, 18, 20, 30, 23), op(65, 6, 32, 8, 12), op(62, 32, 31), op(253), op(56),
    ] {
        words += instruction
    }
    return words
}

private func structuredLoopSPIRV(postTest: Bool) -> [UInt32] {
    func op(_ code: UInt32, _ operands: UInt32...) -> [UInt32] {
        [UInt32(operands.count + 1) << 16 | code] + operands
    }
    // A switch continues or exits the loop while the header and latch carry separate phi values.
    var words: [UInt32] = [0x07230203, 0x00010300, 0, 100, 0]
    for instruction in [
        op(17, 1), op(14, 0, 1), op(15, 5, 10, 0x6e69616d, 0), op(16, 10, 17, 1, 1, 1),
        op(71, 5, 2), op(72, 5, 0, 35, 0), op(71, 8, 34, 0), op(71, 8, 33, 0),
        op(19, 1), op(20, 2), op(21, 3, 32, 0), op(30, 5, 3), op(32, 6, 12, 5), op(32, 7, 12, 3), op(33, 9, 1),
        op(43, 3, 11, 0), op(43, 3, 12, 1), op(43, 3, 13, 6), op(43, 3, 14, 10), op(59, 6, 8, 12),
        op(54, 1, 10, 0, 9), op(248, 20), op(249, 21),
        op(248, 21), op(245, 3, 32, 11, 20, 41, 28), op(245, 3, 33, 11, 20, 40, 28),
        op(246, 29, 28, 0), op(249, 22), op(248, 22),
    ] {
        words += instruction
    }
    if postTest { words += op(249, 23) }
    else { words += op(176, 2, 34, 32, 13) + op(250, 34, 23, 29) }
    for instruction in [
        op(248, 23), op(247, 27, 0), op(251, 32, 26, 1, 24, postTest ? 9 : 3, 25),
        op(248, 24), op(249, 28), op(248, 25), op(249, 29), op(248, 26), op(249, 27),
        op(248, 27), op(128, 3, 37, 32, 14), op(128, 3, 38, 33, 37), op(249, 28),
        op(248, 28), op(245, 3, 40, 33, 24, 38, 27), op(128, 3, 41, 32, 12),
    ] {
        words += instruction
    }
    if postTest {
        words += op(176, 2, 42, 41, 13) + op(250, 42, 21, 29)
        words += op(248, 29) + op(245, 3, 43, 33, 25, 40, 28)
    } else {
        words += op(249, 21) + op(248, 29) + op(245, 3, 43, 33, 22, 33, 25)
    }
    return words + op(65, 7, 44, 8, 11) + op(62, 44, 43) + op(253) + op(56)
}

private func escapingLoopValueSPIRV() -> [UInt32] {
    func op(_ code: UInt32, _ operands: UInt32...) -> [UInt32] {
        [UInt32(operands.count + 1) << 16 | code] + operands
    }
    // The final header load and computation survive a write to the same buffer after the loop.
    var words: [UInt32] = [0x07230203, 0x00010300, 0, 100, 0]
    for instruction in [
        op(17, 1), op(14, 0, 1), op(15, 5, 10, 0x6e69616d, 0), op(16, 10, 17, 1, 1, 1),
        op(71, 4, 2), op(72, 4, 0, 35, 0), op(72, 4, 1, 35, 4), op(72, 4, 2, 35, 8), op(71, 8, 34, 0), op(71, 8, 33, 0),
        op(19, 1), op(20, 2), op(21, 3, 32, 0), op(30, 4, 3, 3, 3), op(32, 5, 12, 4), op(32, 6, 12, 3), op(33, 9, 1),
        op(43, 3, 11, 0), op(43, 3, 12, 1), op(43, 3, 13, 2), op(43, 3, 14, 3), op(43, 3, 15, 10), op(43, 3, 16, 99),
        op(59, 5, 8, 12), op(33, 40, 3, 3, 3),
        op(54, 3, 41, 0, 40), op(55, 3, 42), op(55, 3, 43), op(248, 44), op(128, 3, 45, 42, 43), op(254, 45), op(56),
        op(54, 1, 10, 0, 9), op(248, 20),
        op(65, 6, 21, 8, 11), op(65, 6, 22, 8, 12), op(65, 6, 23, 8, 13), op(249, 24),
        op(248, 24), op(245, 3, 30, 11, 20, 35, 25), op(61, 3, 31, 21), op(83, 3, 32, 31),
        op(57, 3, 33, 41, 32, 30), op(176, 2, 34, 30, 14), op(246, 26, 25, 0), op(250, 34, 25, 26), op(248, 25),
        op(128, 3, 36, 33, 15), op(62, 21, 36), op(128, 3, 35, 30, 12), op(249, 24),
        op(248, 26), op(62, 21, 16), op(62, 22, 32), op(62, 23, 33), op(253), op(56),
    ] {
        words += instruction
    }
    return words
}

private func escapingAccessPathSPIRV() -> [UInt32] {
    func op(_ code: UInt32, _ operands: UInt32...) -> [UInt32] {
        [UInt32(operands.count + 1) << 16 | code] + operands
    }
    // The row and column come from the last loop iteration, before the index is cleared.
    var words: [UInt32] = [0x07230203, 0x00010300, 0, 100, 0]
    for instruction in [
        op(17, 1), op(14, 0, 1), op(15, 5, 10, 0x6e69616d, 0), op(16, 10, 17, 1, 1, 1),
        op(71, 4, 6, 4), op(71, 42, 6, 8), op(71, 5, 2), op(72, 5, 0, 35, 0), op(72, 5, 1, 35, 4), op(71, 8, 34, 0), op(71, 8, 33, 0),
        op(19, 1), op(20, 2), op(21, 3, 32, 0), op(21, 40, 32, 1), op(43, 40, 41, 1), op(43, 3, 13, 2),
        op(28, 4, 3, 13), op(28, 42, 4, 13), op(30, 5, 3, 42),
        op(32, 6, 12, 5), op(32, 7, 12, 3), op(32, 9, 12, 4), op(32, 43, 12, 42), op(33, 15, 1),
        op(43, 3, 11, 0), op(43, 3, 12, 1), op(43, 3, 16, 99), op(59, 6, 8, 12),
        op(54, 1, 10, 0, 15), op(248, 20), op(65, 7, 21, 8, 11), op(249, 22),
        op(248, 22), op(245, 3, 30, 11, 20, 37, 23), op(61, 3, 31, 21),
        op(194, 3, 45, 31, 12), op(199, 3, 46, 31, 12),
        op(65, 9, 32, 8, 41, 45), op(66, 7, 33, 32, 46), op(65, 7, 34, 33),
        op(176, 2, 35, 30, 13), op(246, 24, 23, 0), op(250, 35, 23, 24), op(248, 23),
        op(128, 3, 36, 31, 12), op(62, 21, 36), op(128, 3, 37, 30, 12), op(249, 22),
        op(248, 24), op(62, 21, 11), op(62, 34, 16), op(253), op(56),
    ] {
        words += instruction
    }
    return words
}

private func forwardLoopSPIRV() -> [UInt32] {
    func op(_ code: UInt32, _ operands: UInt32...) -> [UInt32] {
        [UInt32(operands.count + 1) << 16 | code] + operands
    }
    // The entry calls a later function and passes the local counter updated by its loop.
    var words: [UInt32] = [0x07230203, 0x00010300, 0, 100, 0]
    for instruction in [
        op(17, 1), op(11, 80, 0x4c534c47, 0x6474732e, 0x3035342e, 0),
        op(14, 0, 1), op(15, 5, 10, 0x6e69616d, 0), op(16, 10, 17, 1, 1, 1),
        op(3, 2, 450), op(5, 20, 0x6574616c, 0x72), op(6, 4, 0, 0x64726f77, 0),
        op(71, 4, 2), op(72, 4, 0, 35, 0), op(71, 8, 34, 0), op(71, 8, 33, 0),
        op(19, 1), op(20, 2), op(21, 3, 32, 0), op(30, 4, 3), op(32, 5, 12, 4), op(32, 6, 12, 3), op(32, 7, 7, 3),
        op(33, 9, 1), op(33, 11, 1, 7), op(43, 3, 12, 0), op(43, 3, 13, 1), op(43, 3, 14, 3), op(59, 5, 8, 12),
        op(54, 1, 10, 0, 9), op(248, 21), op(59, 7, 22, 7), op(62, 22, 12), op(57, 1, 23, 20, 22),
        op(61, 3, 24, 22), op(65, 6, 25, 8, 12), op(62, 25, 24), op(253), op(56),
        op(54, 1, 20, 0, 11), op(55, 7, 26), op(248, 27), op(249, 28),
        op(248, 28), op(246, 31, 30, 0), op(249, 29),
        op(248, 29), op(61, 3, 32, 26), op(128, 3, 33, 32, 13), op(62, 26, 33), op(249, 30),
        op(248, 30), op(61, 3, 34, 26), op(176, 2, 35, 34, 14), op(250, 35, 28, 31), op(248, 31), op(253), op(56),
    ] {
        words += instruction
    }
    return words
}

private func textureUploadDiagnostic(_ device: Device) async throws -> [UInt32] {
    let source = try await device.createBuffer(.init(size: 268, usage: [.copySource, .copyDestination]))
    let target = try await device.createTexture(.init(width: 4, height: 3, format: .rgba8Unorm,
                                                      usage: [.copyDestination, .copySource]))
    let full = try await device.createTexture(.init(width: 2, height: 1, format: .bgra8Unorm,
                                                    usage: [.copyDestination, .copySource]))
    let read = try await device.createBuffer(.init(size: 528, usage: [.copyDestination, .mapRead]))
    let texels: [UInt32] = [0xff112233, 0xff445566, 0xff778899, 0xffaabbcc]
    // Distinct image strides exercise both directions and preserve the gaps.
    let layered = try await device.createTexture(.init(width: 2, height: 2, format: .rgba8Unorm,
        usage: [.copyDestination, .copySource], arrayLayerCount: 2))
    let layeredSource = try await device.createBuffer(.init(size: 1036, usage: [.copySource, .copyDestination]))
    let layeredRead = try await device.createBuffer(.init(size: 1296, usage: [.copyDestination, .mapRead]))
    let sentinel: UInt32 = 0xdeadbeef
    let ticket = try await device.submitCommands([
        .write(buffer: source, offset: 4, words: Array(texels.prefix(2))),
        .write(buffer: source, offset: 260, words: Array(texels.suffix(2))),
        .copyBufferToTexture(source: source, layout: .init(offset: 4, bytesPerRow: 256),
                             destination: target, region: .init(x: 1, y: 1, width: 2, height: 2)),
        .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256),
        .read(buffer: read, byteCount: 528),
        .copyBufferToTexture(source: source, layout: .init(offset: 8),
                             destination: target, region: .init(x: 3, width: 1, height: 1)),
        .copyBufferToTexture(source: source, layout: .init(offset: 268),
                             destination: target, region: .init(x: 4, y: 3, width: 0, height: 0)),
        .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256),
        .read(buffer: read, byteCount: 528),
        .copyBufferToTexture(source: source, layout: .init(offset: 4),
                             destination: full, region: .init(width: 2, height: 1)),
        .copyTextureToBuffer(source: full, destination: read, bytesPerRow: 256),
        .read(buffer: read, byteCount: 8),
        .write(buffer: layeredSource, offset: 4, words: [texels[0], texels[1]]),
        .write(buffer: layeredSource, offset: 260, words: [texels[2], texels[3]]),
        .write(buffer: layeredSource, offset: 772, words: [texels[1], texels[0]]),
        .write(buffer: layeredSource, offset: 1028, words: [texels[3], texels[2]]),
        .write(buffer: layeredRead, offset: 0, words: Array(repeating: sentinel, count: 324)),
        .copyBufferToTexture(source: layeredSource, layout: .init(offset: 4, bytesPerRow: 256, rowsPerImage: 3),
                             destination: layered, region: .init(width: 2, height: 2, depth: 2)),
        .copyTextureToBuffer(source: layered, destination: layeredRead, bytesPerRow: 256,
                             destinationOffset: 8, depth: 2, rowsPerImage: 4),
        .copyTextureToBuffer(source: layered, destination: layeredRead, bytesPerRow: 256,
                             destinationOffset: 1296, arrayLayer: 2, depth: 0),
        .read(buffer: layeredRead, byteCount: 1296),
        .write(buffer: layeredRead, offset: 0, words: Array(repeating: sentinel, count: 324)),
        .copyTextureRegionToBuffer(source: layered, destination: layeredRead,
            layout: .init(offset: 8, bytesPerRow: 256, rowsPerImage: 4),
            region: .init(x: 1, y: 1, width: 1, height: 1, depth: 2)),
        .read(buffer: layeredRead, byteCount: 1296),
        .write(buffer: layeredRead, offset: 0, words: Array(repeating: sentinel, count: 324)),
        .copyTextureRegionToBuffer(source: layered, destination: layeredRead,
            layout: .init(offset: 8, bytesPerRow: 256),
            region: .init(x: 1, width: 1, height: 2, depth: 2)),
        .read(buffer: layeredRead, byteCount: 1296),
    ])
    try await device.destroyBuffer(source)
    let result = try await ticket.wait()
    var first = Array(repeating: UInt32(0), count: 132)
    first[65] = texels[0]; first[66] = texels[1]
    first[129] = texels[2]; first[130] = texels[3]
    var second = first
    second[3] = texels[1]
    var layeredExpected = Array(repeating: sentinel, count: 324)
    layeredExpected.replaceSubrange(2..<4, with: [texels[0], texels[1]])
    layeredExpected.replaceSubrange(66..<68, with: [texels[2], texels[3]])
    layeredExpected.replaceSubrange(258..<260, with: [texels[1], texels[0]])
    layeredExpected.replaceSubrange(322..<324, with: [texels[3], texels[2]])
    var regionExpected = Array(repeating: sentinel, count: 324)
    regionExpected[2] = texels[3]
    regionExpected[258] = texels[2]
    var packedExpected = Array(repeating: sentinel, count: 324)
    packedExpected[2] = texels[1]; packedExpected[66] = texels[3]
    packedExpected[130] = texels[0]; packedExpected[194] = texels[2]
    guard result == [first, second, Array(texels.prefix(2)), layeredExpected, regionExpected, packedExpected] else {
        throw GPUError.backendFailure("Unexpected WebGPU texture upload readback")
    }
    return [3, 65, 66, 129, 130].map { result[1][$0] }
}

private func textureLoadDiagnostic(_ device: Device) async throws -> [UInt32] {
    let source = ShaderInput.wgsl("""
        @group(2) @binding(5) var image: texture_2d<f32>;
        @group(2) @binding(0) var<storage, read_write> out: array<vec4f, 7>;

        fn fetch(p: vec2i, level: i32) -> vec4f { return textureLoad(image, p, level); }

        @compute @workgroup_size(7)
        fn run(@builtin(global_invocation_id) id: vec3u) {
            if id.x == 4u {
                out[id.x] = fetch(vec2i(0), 1i);
            } else if id.x == 5u {
                out[id.x] = textureLoad(image, vec2u(0xffffffffu, 0u), 0u);
            } else if id.x == 6u {
                out[id.x] = textureLoad(image, vec2u(0u), 0xffffffffu);
            } else {
                out[id.x] = fetch(vec2i(i32(id.x) - 1i, 0i), 0i);
            }
        }
        """)
    let shader = try IndependentShaderCompiler().compileCompute(source, to: .wgsl)
    let pipeline = try await device.createComputePipeline(shader)
    let input = try await device.createBuffer(.init(size: 8, usage: [.copyDestination, .copySource]))
    let target = try await device.createTexture(.init(width: 2, height: 1, format: .bgra8Unorm,
                                                      usage: [.copyDestination, .textureBinding]))
    let fresh = try await device.createTexture(.init(width: 2, height: 1, format: .rgba8Unorm, usage: .textureBinding))
    let size: UInt64 = 7 * 16
    let out = try await device.createBuffer(.init(size: size, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: size, usage: [.copyDestination, .mapRead]))
    func dispatch(_ texture: ResourceID) -> BufferCommand {
        .dispatchComputeRanges(pipeline: pipeline, bindings: [.init(group: 2, binding: 0): .init(buffer: out)],
                               workgroups: .init(x: 1), textures: [.init(group: 2, binding: 5): texture])
    }
    let ticket = try await device.submitCommands([
        .write(buffer: input, offset: 0, words: [0xffff0000, 0xff00ff00]),
        .copyBufferToTexture(source: input, layout: .init(), destination: target, region: .init(width: 2, height: 1)),
        dispatch(target),
        .copy(source: out, destination: read, byteCount: size),
        .read(buffer: read, byteCount: size),
        dispatch(fresh),
        .copy(source: out, destination: read, byteCount: size),
        .read(buffer: read, byteCount: size),
    ])
    try await device.destroyTexture(target)
    let result = try await ticket.wait()
    let expected: [Float] = [
        0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]
    guard result == [expected.map(\.bitPattern), Array(repeating: 0, count: 28)] else {
        throw GPUError.backendFailure("Unexpected WebGPU sampled texture readback: \(result)")
    }
    return (0..<7).map { i in
        let pixel = result[0][(i * 4)..<(i * 4 + 4)]
        return zip(pixel, [16, 8, 0, 24]).reduce(0) { $0 | (UInt32(Float(bitPattern: $1.0) * 255) << $1.1) }
    }
}

private func renderTextureDiagnostic(_ device: Device) async throws -> [UInt32] {
    let source = ShaderInput.wgsl("""
        @group(2) @binding(5) var image: texture_2d<f32>;
        @group(1) @binding(4) var auxiliary: texture_2d<f32>;
        @group(3) @binding(6) var unused: texture_2d<f32>;
        @group(2) @binding(8) var<uniform> gain: f32;

        fn readImage(x: i32) -> vec4f { return textureLoad(image, vec2i(x, 0i), 0i); }

        @vertex fn vs(@location(0) point: vec2f) -> @builtin(position) vec4f {
            return vec4f(point.x + readImage(0i).r, point.y, 0.0f, 1.0f);
        }

        @fragment fn fs() -> @location(0) vec4f {
            let blue = textureLoad(auxiliary, vec2u(0u), 0u).b;
            return vec4f(readImage(1i).g * gain, blue, 0.0f, 1.0f);
        }
        """)
    let compiler = IndependentShaderCompiler()
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .wgsl)
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: .wgsl)
    let pipeline = try await device.createRenderPipeline(
        vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm,
        vertexBuffers: [.init(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float32x2)])])
    let upload = try await device.createBuffer(.init(size: 12, usage: [.copySource, .copyDestination]))
    let vertices = try await device.createBuffer(.init(size: 24, usage: [.vertex, .copyDestination]))
    let indices = try await device.createBuffer(.init(size: 8, usage: [.index, .copyDestination]))
    let gain = try await device.createBuffer(.init(size: 4, usage: [.uniform, .copyDestination]))
    let image = try await device.createTexture(.init(width: 2, height: 1, format: .rgba8Unorm,
                                                    usage: [.copyDestination, .textureBinding]))
    let aux = try await device.createTexture(.init(width: 1, height: 1, format: .bgra8Unorm,
                                                  usage: [.copyDestination, .textureBinding]))
    let fresh = try await device.createTexture(.init(width: 1, height: 1, format: .bgra8Unorm, usage: .textureBinding))
    let target = try await device.createTexture(.init(width: 8, height: 8, format: .rgba8Unorm,
                                                      usage: [.renderAttachment, .copySource, .textureBinding]))
    let read = try await device.createBuffer(.init(size: 1824, usage: [.copyDestination, .mapRead]))
    let clear = RenderClearColor(red: 0, green: 0, blue: 1, alpha: 1)
    let shared = ComputeBindingSlot(group: 2, binding: 5)
    let auxiliary = ComputeBindingSlot(group: 1, binding: 4)
    let uniforms: [ComputeBindingSlot: ComputeBufferBinding] = [.init(group: 2, binding: 8): .init(buffer: gain)]
    let streams = [VertexBufferBinding(buffer: vertices)]
    func draw(_ textures: [ComputeBindingSlot: ResourceID]) -> BufferCommand {
        .render(pipeline: pipeline, target: target, clear: clear, vertexCount: 3, vertexBuffers: streams,
                uniforms: uniforms, textures: textures)
    }
    do {
        _ = try await device.submitCommands([draw([shared: target, auxiliary: aux])])
        throw GPUError.backendFailure("Accepted a sampled render attachment")
    } catch GPUError.validation("Render attachment cannot be bound as a sampled texture") {}
    let ticket = try await device.submitCommands([
        .write(buffer: upload, offset: 0, words: [0xff0000ff, 0xff00ff00, 0xff0000ff]),
        .copyBufferToTexture(source: upload, layout: .init(), destination: image, region: .init(width: 2, height: 1)),
        .copyBufferToTexture(source: upload, layout: .init(offset: 8), destination: aux, region: .init(width: 1, height: 1)),
        .write(buffer: gain, offset: 0, words: [Float(1).bitPattern]),
        .write(buffer: vertices, offset: 0, words: [Float(-1), -1, 3, -1, -1, 3].map(\.bitPattern)),
        .write(buffer: indices, offset: 0, words: [0x00010000, 2]),
        draw([shared: image, auxiliary: aux]),
        .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256),
        .read(buffer: read, byteCount: 1824),
        .renderIndexed(pipeline: pipeline, target: target, clear: clear,
                       indexBuffer: .init(buffer: indices, format: .uint16), draw: .init(indexCount: 3),
                       vertexBuffers: streams, uniforms: uniforms, textures: [shared: image, auxiliary: fresh]),
        .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256),
        .read(buffer: read, byteCount: 1824),
    ])
    try await device.destroyTexture(image)
    let snapshots = try await ticket.wait()
    guard snapshots.count == 2, snapshots.allSatisfy({ $0.count == 456 }) else {
        throw GPUError.backendFailure("Unexpected textured render readback size")
    }
    let pixels = snapshots.flatMap { [$0[4 * 64 + 1], $0[4 * 64 + 6]] }
    guard pixels == [0xffff0000, 0xff00ffff, 0xffff0000, 0xff0000ff] else {
        throw GPUError.backendFailure("Unexpected textured render pixels: \(pixels)")
    }
    return pixels
}

private func samplerLODClampDiagnostic(_ device: Device) async throws {
    let source = ShaderInput.wgsl("""
        @group(0) @binding(0) var<storage, read_write> out: array<vec4f, 2>;
        @group(0) @binding(1) var image: texture_2d<f32>;
        @group(0) @binding(2) var state: sampler;
        @compute @workgroup_size(1) fn run() {
            out[0] = textureSampleLevel(image, state, vec2f(0.5), 0.0);
            out[1] = textureSampleLevel(image, state, vec2f(0.5), 8.0);
        }
        """)
    let shader = try IndependentShaderCompiler().compileCompute(source, to: .wgsl)
    let pipeline = try await device.createComputePipeline(shader)
    let sampler = try await device.createSampler(.init(minFilter: .linear, magFilter: .linear,
        mipmapFilter: .linear, lodMinClamp: 1, lodMaxClamp: 1, maxAnisotropy: 32))
    let image = try await device.createTexture(.init(width: 4, height: 1, format: .rgba8Unorm,
        usage: [.copyDestination, .textureBinding], mipLevelCount: 3))
    let upload = try await device.createBuffer(.init(size: 28, usage: [.copyDestination, .copySource]))
    let out = try await device.createBuffer(.init(size: 32, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 32, usage: [.copyDestination, .mapRead]))
    let ticket = try await device.submitCommands([
        .write(buffer: upload, offset: 0, words: [0xff0000ff, 0xff0000ff, 0xff0000ff, 0xff0000ff,
                                                0xff00ff00, 0xff00ff00, 0xffff0000]),
        .copyBufferToTexture(source: upload, layout: .init(), destination: image, region: .init(width: 4, height: 1)),
        .copyBufferToTexture(source: upload, layout: .init(offset: 16), destination: image,
                             region: .init(width: 2, height: 1, mipLevel: 1)),
        .copyBufferToTexture(source: upload, layout: .init(offset: 24), destination: image,
                             region: .init(width: 1, height: 1, mipLevel: 2)),
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): out], workgroups: .init(x: 1),
                         textures: [.init(binding: 1): image], samplers: [.init(binding: 2): sampler]),
        .copy(source: out, destination: read, byteCount: 32),
        .read(buffer: read, byteCount: 32),
    ])
    let snapshots = try await ticket.wait()
    let expected: [Float] = [0, 1, 0, 1, 0, 1, 0, 1]
    guard snapshots.first == expected.map(\.bitPattern) else {
        throw GPUError.backendFailure("Unexpected WebGPU sampler LOD clamp readback: \(snapshots)")
    }
}

private func samplerDiagnostic(_ device: Device) async throws -> [UInt32] {
    let source = ShaderInput.wgsl("""
        @group(2) @binding(0) var<storage, read_write> out: array<vec4f, 4>;
        @group(2) @binding(5) var image: texture_2d<f32>;
        @group(2) @binding(6) var pointSampler: sampler;
        @group(2) @binding(7) var linearSampler: sampler;
        const east = vec2i(2, -1) + vec2i(-1, 1);
        fn blend(uv: vec2f, mip: f32) -> vec4f { return textureSampleLevel(image, linearSampler, uv, mip); }
        @compute @workgroup_size(1) fn run() {
            let uv = vec2f(0.4375f, 0.5f);
            out[0] = textureSampleLevel(image, pointSampler, uv, 0, east,);
            out[1] = blend(uv, 0.0);
            out[2] = textureSampleGrad(image, pointSampler, vec2f(-0.25f, 0.5f), vec2f(0.25f, 0.0f), vec2f(0.0f, 0.25f), vec2i(-1, 0));
            out[3] = textureLoad(image, vec2u(1u, 0u), 0u);
        }
        """)
    let shader = try IndependentShaderCompiler().compileCompute(source, to: .wgsl)
    let pipeline = try await device.createComputePipeline(shader)
    let point = try await device.createSampler(.init(addressModeU: .repeat))
    let linear = try await device.createSampler(.init(minFilter: .linear, magFilter: .linear))
    let input = try await device.createBuffer(.init(size: 8, usage: [.copyDestination, .copySource]))
    let image = try await device.createTexture(.init(width: 2, height: 1, format: .bgra8Unorm,
                                                    usage: [.copyDestination, .textureBinding]))
    let out = try await device.createBuffer(.init(size: 64, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 64, usage: [.copyDestination, .mapRead]))
    func dispatch(_ point: ResourceID) -> BufferCommand {
        .dispatchComputeRanges(pipeline: pipeline, bindings: [.init(group: 2, binding: 0): .init(buffer: out)],
                               workgroups: .init(x: 1), textures: [.init(group: 2, binding: 5): image],
                               samplers: [.init(group: 2, binding: 6): point, .init(group: 2, binding: 7): linear])
    }
    do {
        _ = try await device.submitCommands([dispatch(image)])
        throw GPUError.backendFailure("Accepted a texture as a sampler binding")
    } catch GPUError.validation("Invalid sampler binding") {}
    let ticket = try await device.submitCommands([
        .write(buffer: input, offset: 0, words: [0xffff0000, 0xff00ff00]),
        .copyBufferToTexture(source: input, layout: .init(), destination: image, region: .init(width: 2, height: 1)),
        dispatch(point),
        .copy(source: out, destination: read, byteCount: 64),
        .read(buffer: read, byteCount: 64),
    ])
    try await device.destroySampler(point)
    let snapshots = try await ticket.wait()
    let actual = snapshots.first?.map(Float.init(bitPattern:)) ?? []
    let expected: [Float] = [0, 1, 0, 1, 0.625, 0.375, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1]
    guard actual.count == expected.count, zip(actual, expected).allSatisfy({ abs($0.0 - $0.1) < 0.004 }) else {
        throw GPUError.backendFailure("Unexpected WebGPU sampler values: \(actual)")
    }
    return actual.map { UInt32(($0 * 16).rounded()) }
}

private func renderSamplerDiagnostic(_ device: Device) async throws -> [UInt32] {
    let source = ShaderInput.wgsl("""
        @group(2) @binding(5) var image: texture_2d<f32>;
        @group(1) @binding(4) var linearSampler: sampler;
        @group(2) @binding(7) var pointSampler: sampler;

        fn point(uv: vec2f) -> vec4f { return textureSampleLevel(image, pointSampler, uv, 0); }

        @vertex fn vs(@location(0) position: vec2f) -> @builtin(position) vec4f {
            let offset = point(vec2f(0.25, 0.5)).r;
            return vec4f(position.x + offset, position.y, 0.0, 1.0);
        }

        @fragment fn fs() -> @location(0) vec4f {
            let uv = vec2f(0.4375, 0.5);
            const east = vec2(1, 0);
            let filtered = textureSample(image, linearSampler, vec2f(-0.0625, 0.5), east);
            let biased = textureSampleBias(image, linearSampler, uv, 2.0, east);
            let grad = textureSampleGrad(image, pointSampler, vec2f(0.75, 0.5), vec2f(0.25, 0.0), vec2f(0.0, 0.25));
            return vec4f(point(uv).r * grad.g, filtered.g, biased.g, 1.0);
        }
        """)
    let compiler = IndependentShaderCompiler()
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .wgsl)
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: .wgsl)
    let pipeline = try await device.createRenderPipeline(
        vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm,
        vertexBuffers: [.init(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float32x2)])])
    let upload = try await device.createBuffer(.init(size: 8, usage: [.copyDestination, .copySource]))
    let vertices = try await device.createBuffer(.init(size: 24, usage: [.vertex, .copyDestination]))
    let indices = try await device.createBuffer(.init(size: 8, usage: [.index, .copyDestination]))
    let image = try await device.createTexture(.init(width: 2, height: 1, format: .rgba8Unorm,
                                                    usage: [.copyDestination, .textureBinding]))
    let target = try await device.createTexture(.init(width: 8, height: 8, format: .rgba8Unorm,
                                                      usage: [.renderAttachment, .copySource]))
    let read = try await device.createBuffer(.init(size: 1824, usage: [.copyDestination, .mapRead]))
    let point = try await device.createSampler()
    let linear = try await device.createSampler(.init(minFilter: .linear, magFilter: .linear))
    let textures: [ComputeBindingSlot: ResourceID] = [.init(group: 2, binding: 5): image]
    let sharedSlot = ComputeBindingSlot(group: 2, binding: 7)
    let linearSlot = ComputeBindingSlot(group: 1, binding: 4)
    let clear = RenderClearColor(red: 0, green: 0, blue: 1, alpha: 1)
    let streams = [VertexBufferBinding(buffer: vertices)]
    func draw(_ filtered: ResourceID) -> BufferCommand {
        .render(pipeline: pipeline, target: target, clear: clear, vertexCount: 3, vertexBuffers: streams,
                textures: textures, samplers: [sharedSlot: point, linearSlot: filtered])
    }
    do {
        _ = try await device.submitCommands([draw(image)])
        throw GPUError.backendFailure("Accepted a texture as a render sampler binding")
    } catch GPUError.validation("Invalid sampler binding") {}
    let ticket = try await device.submitCommands([
        .write(buffer: upload, offset: 0, words: [0xff0000ff, 0xff00ff00]),
        .copyBufferToTexture(source: upload, layout: .init(), destination: image, region: .init(width: 2, height: 1)),
        .write(buffer: vertices, offset: 0, words: [Float(-1), -1, 3, -1, -1, 3].map(\.bitPattern)),
        .write(buffer: indices, offset: 0, words: [0x00010000, 2]),
        draw(linear),
        .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256),
        .read(buffer: read, byteCount: 1824),
        .renderIndexed(pipeline: pipeline, target: target, clear: clear,
                       indexBuffer: .init(buffer: indices, format: .uint16), draw: .init(indexCount: 3),
                       vertexBuffers: streams, textures: textures, samplers: [sharedSlot: point, linearSlot: point]),
        .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256),
        .read(buffer: read, byteCount: 1824),
    ])
    try await device.destroySampler(linear)
    let snapshots = try await ticket.wait()
    guard snapshots.count == 2, snapshots.allSatisfy({ $0.count == 456 }) else {
        throw GPUError.backendFailure("Unexpected render sampler readback size")
    }
    let pixels = snapshots.flatMap { [$0[4 * 64 + 1], $0[4 * 64 + 6]] }
    guard pixels == [0xffff0000, 0xffff60ff, 0xffff0000, 0xffff00ff] else {
        throw GPUError.backendFailure("Unexpected render sampler pixels: \(pixels)")
    }
    return pixels
}

private func mipTextureDiagnostic(_ device: Device) async throws -> [UInt32] {
    let source = ShaderInput.glsl("""
        #version 450
        layout(local_size_x=1) in;
        layout(binding=0) buffer Output { vec4 values[7]; } outputData;
        layout(binding=1) uniform texture2D sourceImage;
        layout(binding=2) uniform sampler filterState;
        void main() {
            outputData.values[0] = texelFetch(sampler2D(sourceImage, filterState), ivec2(4, 2), 0);
            outputData.values[1] = texelFetchOffset(sampler2D(sourceImage, filterState), ivec2(-1, 0), 1, ivec2(2, 0));
            outputData.values[2] = texelFetch(sampler2D(sourceImage, filterState), ivec2(0), 2);
            outputData.values[3] = textureProjLodOffset(sampler2D(sourceImage, filterState), vec3(-0.5, -1.0, -2.0), 0.5, ivec2(1, 0));
            outputData.values[4] = texelFetch(sampler2D(sourceImage, filterState), ivec2(0), 1);
            ivec2 size = textureSize(sampler2D(sourceImage, filterState), 1);
            int levels = textureQueryLevels(sampler2D(sourceImage, filterState));
            int lastWidth = textureSize(sampler2D(sourceImage, filterState), 2).x;
            outputData.values[5] = vec4(levels, size.x, size.y, lastWidth);
            outputData.values[6] = texelFetchOffset(sampler2D(sourceImage, filterState), ivec2(1, 0), 1, ivec2(1, 0));
        }
        """, stage: .compute, defines: [:])
    let compiler = IndependentShaderCompiler()
    let compute = try await device.createComputePipeline(compiler.compileWebGPUCompute(source))
    let spirvShader = try compiler.compileWebGPUCompute(.spirv(sampledTextureSPIRV()))
    guard spirvShader.sampledTextures.count == 1, let texture = spirvShader.sampledTextures.first,
          texture.binding == .init(group: 0, binding: 1), texture.sampleKind == .float,
          texture.dimension == .d2, !texture.isArrayed, !texture.isDepth, !texture.isMultisampled,
          texture.samplingSamplers == [.init(group: 0, binding: 2)],
          spirvShader.samplerBindings.count == 1, let sampler = spirvShader.samplerBindings.first,
          sampler.binding == .init(group: 0, binding: 2), !sampler.isComparison else {
        throw GPUError.backendFailure("Unexpected SPIR-V sampled texture reflection")
    }
    let spirvCompute = try await device.createComputePipeline(spirvShader)
    let roundTripSource = ShaderInput.wgsl("""
        @group(0) @binding(0) var<storage, read_write> out: array<vec4f, 3>;
        @group(0) @binding(1) var image: texture_2d<f32>;
        @group(0) @binding(2) var filter: sampler;
        @compute @workgroup_size(1) fn round_trip() {
            out[0u] = textureSampleLevel(image, filter, vec2f(0.75, 0.5), 1.5);
            out[1u] = textureLoad(image, vec2i(4, 2), 0i);
            let baseSize = textureDimensions(image, 0u);
            let mipSize = textureDimensions(image, 1u);
            out[2u] = vec4f(f32(baseSize.x), f32(baseSize.y), f32(mipSize.x), f32(textureNumLevels(image)));
        }
        """)
    guard case .spirv(let words) = try compiler.compile(roundTripSource, to: .spirv) else {
        throw GPUError.backendFailure("Expected binary SPIR-V for sampled textures")
    }
    let roundTripShader = try compiler.compileWebGPUCompute(.spirv(words), entryPoint: "round_trip")
    let roundTrip = try await device.createComputePipeline(roundTripShader)
    let clearShader = ShaderInput.wgsl("""
        @vertex fn vs() -> @builtin(position) vec4f { return vec4f(0, 0, 0, 1); }
        @fragment fn fs() -> @location(0) vec4f { return vec4f(0, 0, 1, 1); }
        """)
    let render = try await device.createRenderPipeline(
        vertex: compiler.compileRenderStage(clearShader, entryPoint: "vs", to: .wgsl),
        fragment: compiler.compileRenderStage(clearShader, entryPoint: "fs", to: .wgsl), colorFormat: .rgba8Unorm)
    let image = try await device.createTexture(.init(width: 5, height: 3, format: .rgba8Unorm,
        usage: [.copySource, .copyDestination, .textureBinding, .renderAttachment], mipLevelCount: 3))
    let upload = try await device.createBuffer(.init(size: 536, usage: [.copySource, .copyDestination]))
    let out = try await device.createBuffer(.init(size: 112, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 112, usage: [.copyDestination, .mapRead]))
    let filter = try await device.createSampler(.init(mipmapFilter: .linear))
    let dispatch = BufferCommand.dispatchCompute(
        pipeline: compute, bindings: [.init(binding: 0): out], workgroups: .init(x: 1),
        textures: [.init(binding: 1): image], samplers: [.init(binding: 2): filter])
    let red = Array(repeating: UInt32(0xff0000ff), count: 5)
    let green: UInt32 = 0xff00ff00
    let blue = RenderClearColor(red: 0, green: 0, blue: 1, alpha: 1)
    do {
        _ = try await device.submitCommands([
            .render(pipeline: render, target: image, clear: blue, vertexCount: 0, targetMipLevel: 3),
        ])
        throw GPUError.backendFailure("Accepted an invalid render mip level")
    } catch GPUError.validation {}
    let first = try await device.submitCommands([
        .write(buffer: upload, offset: 0, words: red),
        .write(buffer: upload, offset: 256, words: red),
        .write(buffer: upload, offset: 512, words: red + [green]),
        .copyBufferToTexture(source: upload, layout: .init(bytesPerRow: 256), destination: image,
                             region: .init(width: 5, height: 3)),
        .copyBufferToTexture(source: upload, layout: .init(offset: 532), destination: image,
                             region: .init(x: 1, width: 1, height: 1, mipLevel: 1)),
        dispatch,
        .copy(source: out, destination: read, byteCount: 112),
        .read(buffer: read, byteCount: 112),
        .copyTextureToBuffer(source: image, destination: read, bytesPerRow: 256, mipLevel: 1),
        .read(buffer: read, byteCount: 8),
    ])
    let initial: [Float] = [
        1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0.5, 0.5, 0, 1,
        0, 0, 0, 0, 3, 2, 1, 1, 0, 0, 0, 0,
    ]
    let beforeRender = try await first.wait()
    guard beforeRender == [initial.map(\.bitPattern), [0, green]] else {
        throw GPUError.backendFailure("Unexpected mip upload and sampling results: \(beforeRender)")
    }
    let second = try await device.submitCommands([
        .render(pipeline: render, target: image, clear: blue, vertexCount: 0, targetMipLevel: 2),
        dispatch,
        .copy(source: out, destination: read, byteCount: 112),
        .read(buffer: read, byteCount: 112),
        .copyTextureToBuffer(source: image, destination: read, bytesPerRow: 256, mipLevel: 2),
        .read(buffer: read, byteCount: 4),
        .dispatchCompute(pipeline: spirvCompute, bindings: [.init(binding: 0): out], workgroups: .init(x: 1),
                         textures: [.init(binding: 1): image], samplers: [.init(binding: 2): filter]),
        .copy(source: out, destination: read, byteCount: 32),
        .read(buffer: read, byteCount: 32),
        .dispatchCompute(pipeline: roundTrip, bindings: [.init(binding: 0): out], workgroups: .init(x: 1),
                         textures: [.init(binding: 1): image], samplers: [.init(binding: 2): filter]),
        .copy(source: out, destination: read, byteCount: 48),
        .read(buffer: read, byteCount: 48),
    ])
    var rendered = initial
    rendered.replaceSubrange(8..<12, with: [0, 0, 1, 1])
    let afterRender = try await second.wait()
    let sampledAndFetched: [Float] = [0, 1, 0, 1, 0, 0, 1, 1]
    let roundTripValues: [Float] = [0, 0.5, 0.5, 1, 1, 0, 0, 1, 5, 3, 2, 3]
    guard afterRender == [rendered.map(\.bitPattern), [0xffff0000], sampledAndFetched.map(\.bitPattern),
                          roundTripValues.map(\.bitPattern)] else {
        throw GPUError.backendFailure("Unexpected mip render results: \(afterRender)")
    }
    return beforeRender[1] + afterRender[1] + Array(afterRender[0][12..<14]) + afterRender[2] + afterRender[3]
}

private func sampledTextureSPIRV() -> [UInt32] {
    func op(_ code: UInt32, _ operands: UInt32...) -> [UInt32] {
        [UInt32(operands.count + 1) << 16 | code] + operands
    }
    var words: [UInt32] = [0x07230203, 0x00010300, 0, 56, 0]
    for instruction in [
        op(17, 1), op(14, 0, 1), op(15, 5, 40, 0x6e69616d, 0), op(16, 40, 17, 1, 1, 1),
        op(71, 13, 2), op(72, 13, 0, 35, 0), op(72, 13, 1, 35, 16),
        op(71, 20, 34, 0), op(71, 20, 33, 0), op(71, 21, 34, 0), op(71, 21, 33, 1),
        op(71, 22, 34, 0), op(71, 22, 33, 2),
        op(19, 1), op(22, 2, 32), op(21, 3, 32, 1), op(23, 4, 2, 2), op(23, 5, 2, 4), op(23, 6, 3, 2),
        // Separate 2D sampled texture and sampler, with a combined sampled-image SSA value.
        op(25, 7, 2, 1, 0, 0, 0, 1, 0), op(26, 8), op(27, 9, 7), op(32, 10, 0, 7), op(32, 11, 0, 8),
        op(33, 12, 1), op(30, 13, 5, 5), op(32, 14, 12, 13), op(32, 15, 12, 5),
        op(43, 3, 25, 0), op(43, 3, 26, 1), op(43, 3, 27, 2), op(44, 6, 28, 25, 25),
        op(43, 2, 29, 0x3f400000), op(43, 2, 30, 0x3f000000), op(44, 4, 31, 29, 30), op(43, 2, 32, 0x3f800000),
        op(59, 14, 20, 12), op(59, 10, 21, 0), op(59, 11, 22, 0),
        op(54, 1, 40, 0, 12), op(248, 41), op(61, 7, 42, 21), op(61, 8, 43, 22), op(86, 9, 44, 42, 43),
        // Sample (0.75, 0.5) at mip 1 and fetch texel (0, 0) at mip 2. Image Operands mask 2 is Lod.
        op(88, 5, 45, 44, 31, 2, 32), op(95, 5, 46, 42, 28, 2, 27),
        op(65, 15, 47, 20, 25), op(62, 47, 45), op(65, 15, 48, 20, 26), op(62, 48, 46),
        op(253), op(56),
    ] { words += instruction }
    return words
}

private func textureViewDiagnostic(_ device: Device) async throws -> [UInt32] {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl("""
        @group(0) @binding(0) var image: texture_2d<f32>;
        @vertex fn vs(@location(0) p: vec2f) -> @builtin(position) vec4f {
            return vec4f(p, 0.0, 1.0);
        }
        @fragment fn fs() -> @location(0) vec4f {
            return textureLoad(image, vec2u(0, 0), 0u);
        }
        """)
    let render = try await device.createRenderPipeline(
        vertex: compiler.compileRenderStage(source, entryPoint: "vs", to: .wgsl),
        fragment: compiler.compileRenderStage(source, entryPoint: "fs", to: .wgsl),
        colorFormat: .rgba8Unorm, vertexBuffers: [.init(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float32x2)])])
    let compute = try await device.createComputePipeline(compiler.compileCompute(.wgsl("""
        @group(0) @binding(0) var image: texture_2d<f32>;
        @group(0) @binding(1) var<storage, read_write> out: array<vec4f, 2>;
        @group(0) @binding(2) var state: sampler;
        @compute @workgroup_size(1) fn cs() {
            let p = textureLoad(image, vec2u(0), 0u);
            let q = textureLoad(image, vec2u(0), 1u);
            let invalid = textureLoad(image, vec2u(0), 2u);
            out[0] = vec4f(p.g, q.a + invalid.a, f32(textureDimensions(image).x), f32(textureNumLevels(image)));
            let left = textureSampleBaseClampToEdge(image, state, vec2f(-0.25, 0.25));
            let bottom = textureSampleBaseClampToEdge(image, state, vec2f(0.75, 1.25));
            let center = textureSampleBaseClampToEdge(image, state, vec2f(0.5));
            out[1] = vec4f(left.g, left.r, bottom.b, center.r);
        }
        """), to: .wgsl))
    let image = try await device.createTexture(.init(width: 4, height: 4, format: .rgba8Unorm,
        usage: [.copySource, .copyDestination, .textureBinding, .renderAttachment], mipLevelCount: 3))
    let sampled = try await device.createTextureView(texture: image, .init(baseMipLevel: 1))
    let attachment = try await device.createTextureView(texture: image, .init(mipLevelCount: 1))
    let overlap = try await device.createTextureView(texture: image, .init(baseMipLevel: 1, mipLevelCount: 1))
    let state = try await device.createSampler(.init(addressModeU: .repeat, addressModeV: .repeat,
                                                     minFilter: .linear, magFilter: .linear, mipmapFilter: .linear))
    let upload = try await device.createBuffer(.init(size: 264, usage: [.copySource, .copyDestination]))
    let vertices = try await device.createBuffer(.init(size: 24, usage: [.vertex, .copyDestination]))
    let out = try await device.createBuffer(.init(size: 32, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 784, usage: [.copyDestination, .mapRead]))
    let clear = RenderClearColor(red: 0, green: 0, blue: 0, alpha: 1)
    let textureSlot = ComputeBindingSlot(binding: 0)
    func draw(to target: ResourceID) -> BufferCommand {
        .render(pipeline: render, target: target, clear: clear, vertexCount: 3,
                vertexBuffers: [.init(buffer: vertices)], textures: [textureSlot: sampled])
    }
    do {
        _ = try await device.submitCommands([draw(to: overlap)])
        throw GPUError.backendFailure("Accepted an overlapping render texture view")
    } catch GPUError.validation("Render attachment cannot be bound as a sampled texture") {}
    let dispatch = BufferCommand.dispatchCompute(
        pipeline: compute, bindings: [.init(binding: 1): out], workgroups: .init(x: 1), textures: [textureSlot: sampled],
        samplers: [.init(binding: 2): state])
    let green: UInt32 = 0xff00ff00
    let ticket = try await device.submitCommands([
        .write(buffer: upload, offset: 0, words: [green, 0xff0000ff]),
        .write(buffer: upload, offset: 256, words: [0xffff0000, 0xffffffff]),
        .copyBufferToTexture(source: upload, layout: .init(bytesPerRow: 256), destination: image,
                             region: .init(width: 2, height: 2, mipLevel: 1)),
        .write(buffer: vertices, offset: 0, words: [Float(-1), -1, 3, -1, -1, 3].map(\.bitPattern)),
        draw(to: attachment),
        dispatch,
        .copy(source: out, destination: read, byteCount: 32),
        .read(buffer: read, byteCount: 32),
        .copyTextureToBuffer(source: image, destination: read, bytesPerRow: 256),
        .read(buffer: read, byteCount: 16),
    ])
    let expected = [Float(1), 0, 2, 2, 1, 0, 1, 0.5].map(\.bitPattern)
    let snapshots = try await ticket.wait()
    guard snapshots == [expected, Array(repeating: green, count: 4)] else {
        throw GPUError.backendFailure("Unexpected render and compute texture view results: \(snapshots)")
    }
    let viewOnly = try await device.submitCommands([
        dispatch,
        .copy(source: out, destination: read, byteCount: 32),
        .read(buffer: read, byteCount: 32),
    ])
    try await device.destroyTexture(image)
    do {
        _ = try await device.submitCommands([dispatch])
        throw GPUError.backendFailure("Accepted a view of a destroyed texture")
    } catch GPUError.validation("Texture view has a destroyed parent texture") {}
    try await device.destroyTextureView(sampled)
    try await device.destroyTextureView(attachment)
    let viewSnapshot = try await viewOnly.wait()
    guard viewSnapshot == [expected] else {
        throw GPUError.backendFailure("Unexpected retained texture view results: \(viewSnapshot)")
    }
    return viewSnapshot[0] + [snapshots[1][0]]
}

private func textureGatherDiagnostic(_ device: Device) async throws -> [UInt32] {
    let source = ShaderInput.wgsl("""
        @group(0) @binding(0) var<storage, read_write> out: array<vec4f, 4>;
        @group(0) @binding(1) var image: texture_2d<f32>;
        @group(0) @binding(2) var state: sampler;
        const green = 2u - 1u;
        fn gatherRed(uv: vec2f) -> vec4f { return textureGather(0, image, state, uv); }
        @compute @workgroup_size(1) fn run(@builtin(global_invocation_id) id: vec3u) {
            if id.x == 0u {
                let uv = vec2f(0.33333334, 0.5);
                out[0] = gatherRed(uv);
                out[1] = textureGather(green, image, state, uv, vec2(1, 0));
                out[2] = textureGather(2i, image, state, uv, vec2i(-1, 0),);
                out[3] = textureGather(3u, image, state, uv);
            }
        }
        """)
    let shader = try IndependentShaderCompiler().compileCompute(source, to: .wgsl)
    let pipeline = try await device.createComputePipeline(shader)
    let spirvPipeline = try await spirvGatherPipeline(device, comparison: false)
    let sampler = try await device.createSampler(.init(addressModeU: .repeat))
    let image = try await device.createTexture(.init(width: 6, height: 4, format: .rgba8Unorm,
        usage: [.copyDestination, .textureBinding], mipLevelCount: 2))
    let view = try await device.createTextureView(texture: image, .init(baseMipLevel: 1))
    let input = try await device.createBuffer(.init(size: 268, usage: [.copyDestination, .copySource]))
    let out = try await device.createBuffer(.init(size: 64, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 64, usage: [.copyDestination, .mapRead]))
    func dispatch(_ texture: ResourceID) -> BufferCommand {
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): out], workgroups: .init(x: 1),
                         textures: [.init(binding: 1): texture], samplers: [.init(binding: 2): sampler])
    }
    let ticket = try await device.submitCommands([
        .write(buffer: input, offset: 0, words: [0x8c190a14, 0x962d1e28, 0xa041323c]),
        .write(buffer: input, offset: 256, words: [0xaa554650, 0xb4695a64, 0xbe7d6e78]),
        .copyBufferToTexture(source: input, layout: .init(bytesPerRow: 256), destination: image,
                             region: .init(width: 3, height: 2, mipLevel: 1)),
        dispatch(view),
        .copy(source: out, destination: read, byteCount: 64),
        .read(buffer: read, byteCount: 64),
        dispatch(image),
        .copy(source: out, destination: read, byteCount: 64),
        .read(buffer: read, byteCount: 64),
        .dispatchCompute(pipeline: spirvPipeline, bindings: [.init(binding: 0): out], workgroups: .init(x: 1),
                         textures: [.init(binding: 1): view], samplers: [.init(binding: 2): sampler]),
        .copy(source: out, destination: read, byteCount: 16),
        .read(buffer: read, byteCount: 16),
    ])
    let snapshots = try await ticket.wait()
    guard snapshots.count == 3, snapshots[1] == Array(repeating: 0, count: 16) else {
        throw GPUError.backendFailure("Unexpected texture gather base mip result: \(snapshots)")
    }
    let lanes = snapshots[0].map { (Float(bitPattern: $0) * 255).rounded() }
    guard lanes == [80, 100, 40, 20, 90, 110, 50, 30, 125, 85, 25, 65, 170, 180, 150, 140] else {
        throw GPUError.backendFailure("Unexpected texture gather channel or order: \(lanes)")
    }
    let spirvLanes = snapshots[2].map { (Float(bitPattern: $0) * 255).rounded() }
    guard spirvLanes == [90, 110, 50, 30] else {
        throw GPUError.backendFailure("Unexpected SPIR-V texture gather readback: \(spirvLanes)")
    }
    return (lanes + spirvLanes).map(UInt32.init)
}

private func spirvGatherPipeline(_ device: Device, comparison: Bool) async throws -> ComputePipeline {
    let shader = try IndependentShaderCompiler().compileWebGPUCompute(.spirv(textureGatherSPIRV(comparison: comparison)))
    let samplerBinding = IRResourceBinding(group: 0, binding: 2)
    guard shader.sampledTextures.count == 1, let texture = shader.sampledTextures.first,
          texture.binding == .init(group: 0, binding: 1), texture.dimension == .d2,
          texture.isDepth == comparison, comparison || texture.samplingSamplers == [samplerBinding],
          shader.samplerBindings.count == 1, let sampler = shader.samplerBindings.first,
          sampler.binding == samplerBinding, sampler.isComparison == comparison else {
        throw GPUError.backendFailure("Unexpected SPIR-V gather resource reflection")
    }
    return try await device.createComputePipeline(shader)
}

private func textureGatherSPIRV(comparison: Bool) -> [UInt32] {
    func op(_ code: UInt32, _ operands: UInt32...) -> [UInt32] {
        [UInt32(operands.count + 1) << 16 | code] + operands
    }
    let depth: UInt32 = comparison ? 1 : 0
    let offsetX: UInt32 = comparison ? UInt32.max : 1
    let gather = comparison
        ? op(97, 5, 45, 44, 32, 33, 8, 35) // Dref 0.375, ConstOffset (-1, 0).
        : op(96, 5, 45, 44, 32, 26, 8, 35) // Green component, ConstOffset (1, 0).
    var words: [UInt32] = [0x07230203, 0x00010300, 0, 52, 0]
    for instruction in [
        op(17, 1), op(14, 0, 1), op(15, 5, 40, 0x6e69616d, 0), op(16, 40, 17, 1, 1, 1),
        op(71, 13, 2), op(72, 13, 0, 35, 0),
        op(71, 20, 34, 0), op(71, 20, 33, 0), op(71, 21, 34, 0), op(71, 21, 33, 1),
        op(71, 22, 34, 0), op(71, 22, 33, 2),
        op(19, 1), op(22, 2, 32), op(21, 3, 32, 1), op(23, 4, 2, 2), op(23, 5, 2, 4), op(23, 6, 3, 2),
        op(25, 7, 2, 1, depth, 0, 0, 1, 0), op(26, 8), op(27, 9, 7), op(32, 10, 0, 7), op(32, 11, 0, 8),
        op(33, 12, 1), op(30, 13, 5), op(32, 14, 12, 13), op(32, 15, 12, 5),
        op(43, 3, 25, 0), op(43, 3, 26, 1), op(43, 3, 27, offsetX), op(44, 6, 35, 27, 25),
        op(43, 2, 30, 0x3eaaaaab), op(43, 2, 31, 0x3f000000), op(44, 4, 32, 30, 31), op(43, 2, 33, 0x3ec00000),
        op(59, 14, 20, 12), op(59, 10, 21, 0), op(59, 11, 22, 0),
        op(54, 1, 40, 0, 12), op(248, 41), op(61, 7, 42, 21), op(61, 8, 43, 22), op(86, 9, 44, 42, 43),
        gather, op(65, 15, 46, 20, 25), op(62, 46, 45), op(253), op(56),
    ] { words += instruction }
    return words
}

private func storageTextureDiagnostic(_ device: Device) async throws -> [UInt32] {
    let source = ShaderInput.wgsl("""
        @group(2) @binding(3) var source: texture_2d<f32>;
        @group(2) @binding(4) var destination: texture_storage_2d<rgba8unorm, write>;
        fn writeColor(x: i32, color: vec4f) { textureStore(destination, vec2i(x, 0), color); }
        @compute @workgroup_size(5) fn run(@builtin(global_invocation_id) id: vec3u) {
            let x = i32(id.x) - 1;
            if id.x == 4u {
                textureStore(destination, vec2u(0xffffffffu, 0u), vec4f(1, 1, 0, 1));
            } else if x != 0 {
                let dimensions = textureDimensions(destination);
                let red = textureLoad(source, vec2u(0), 0).r;
                writeColor(x, vec4f(red, 0.0, f32(dimensions.x - 1u), 1.0));
            }
        }
        """)
    let shader = try IndependentShaderCompiler().compileCompute(source, to: .wgsl)
    let pipeline = try await device.createComputePipeline(shader)
    let image = try await device.createTexture(.init(width: 4, height: 2, format: .rgba8Unorm,
        usage: [.copySource, .copyDestination, .textureBinding, .storageBinding], mipLevelCount: 2))
    let input = try await device.createTextureView(texture: image, .init(mipLevelCount: 1))
    let output = try await device.createTextureView(texture: image, .init(baseMipLevel: 1))
    let upload = try await device.createBuffer(.init(size: 272, usage: [.copySource, .copyDestination]))
    let read = try await device.createBuffer(.init(size: 272, usage: [.copyDestination, .mapRead]))
    let inputSlot = ComputeBindingSlot(group: 2, binding: 3)
    let outputSlot = ComputeBindingSlot(group: 2, binding: 4)
    func dispatch(_ input: ResourceID, _ output: ResourceID) -> BufferCommand {
        .dispatchCompute(pipeline: pipeline, bindings: [:], workgroups: .init(x: 1),
                         textures: [inputSlot: input, outputSlot: output])
    }
    do {
        _ = try await device.submitCommands([dispatch(output, output)])
        throw GPUError.backendFailure("Accepted overlapping sampled and writable storage texture bindings")
    } catch GPUError.validation("Aliased writable compute textures") {}
    let red = Array(repeating: UInt32(0xff0000ff), count: 4)
    let ticket = try await device.submitCommands([
        .write(buffer: upload, offset: 0, words: red),
        .write(buffer: upload, offset: 256, words: red),
        .copyBufferToTexture(source: upload, layout: .init(bytesPerRow: 256), destination: image,
                             region: .init(width: 4, height: 2)),
        dispatch(input, output),
        .copyTextureToBuffer(source: image, destination: read, bytesPerRow: 256, mipLevel: 1),
        .read(buffer: read, byteCount: 8),
        .copyTextureToBuffer(source: image, destination: read, bytesPerRow: 256),
        .read(buffer: read, byteCount: 272),
    ])
    let snapshots = try await ticket.wait()
    guard snapshots == [[0, 0xffff00ff], red + Array(repeating: 0, count: 60) + red] else {
        throw GPUError.backendFailure("Unexpected storage texture pixels: \(snapshots)")
    }
    return snapshots[0] + [snapshots[1][0]]
}

private func fragmentStorageTextureDiagnostic(_ device: Device) async throws -> [UInt32] {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl("""
        @group(2) @binding(4) var destination: texture_storage_2d<rgba8unorm, write>;
        struct VertexOut {
            @builtin(position) position: vec4f,
            @location(0) x: f32,
        }
        @vertex fn vs(@location(0) point: vec2f) -> VertexOut {
            return VertexOut(vec4f(point, 0, 1), point.x);
        }
        fn writeColor(x: i32, color: vec4f) {
            textureStore(destination, vec2i(x, 0), color);
        }
        @fragment fn fs(@location(0) x: f32) -> @location(0) vec4f {
            if x < 0.0 { writeColor(1, vec4f(1, 0, 0, 1)); }
            else { writeColor(i32(textureDimensions(destination).x) - 2i, vec4f(0, 0, 1, 1)); }
            return vec4f(0, 1, 0, 1);
        }
        """)
    let pipeline = try await device.createRenderPipeline(
        vertex: compiler.compileRenderStage(source, entryPoint: "vs", to: .wgsl),
        fragment: compiler.compileRenderStage(source, entryPoint: "fs", to: .wgsl),
        colorFormat: .rgba8Unorm, vertexBuffers: [.init(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float32x2)])])
    let image = try await device.createTexture(.init(width: 4, height: 2, format: .rgba8Unorm,
        usage: [.renderAttachment, .copySource, .storageBinding], mipLevelCount: 2))
    let storage = try await device.createTextureView(texture: image, .init(mipLevelCount: 1))
    let attachment = try await device.createTextureView(texture: image, .init(baseMipLevel: 1))
    let overlapping = try await device.createTextureView(texture: image, .init(baseMipLevel: 1))
    let storageIndexed = try await device.createTexture(.init(width: 4, height: 2, format: .rgba8Unorm,
        usage: [.copySource, .storageBinding]))
    let vertices = try await device.createBuffer(.init(size: 24, usage: [.vertex, .copyDestination]))
    let indices = try await device.createBuffer(.init(size: 8, usage: [.index, .copyDestination]))
    let read = try await device.createBuffer(.init(size: 272, usage: [.copyDestination, .mapRead]))
    let clear = RenderClearColor(red: 0, green: 0, blue: 0, alpha: 1)
    let slot = ComputeBindingSlot(group: 2, binding: 4)
    let streams = [VertexBufferBinding(buffer: vertices)]
    func draw(_ output: ResourceID) -> BufferCommand {
        .render(pipeline: pipeline, target: attachment, clear: clear, vertexCount: 3,
                vertexBuffers: streams, textures: [slot: output])
    }
    do {
        _ = try await device.submitCommands([draw(overlapping)])
        throw GPUError.backendFailure("Accepted overlapping color and storage texture views")
    } catch GPUError.validation("Aliased writable render textures") {}
    let ticket = try await device.submitCommands([
        .write(buffer: vertices, offset: 0, words: [Float(-1), -1, 3, -1, -1, 3].map(\.bitPattern)),
        .write(buffer: indices, offset: 0, words: [0x00010000, 2]),
        draw(storage),
        .copyTextureToBuffer(source: image, destination: read, bytesPerRow: 256),
        .read(buffer: read, byteCount: 272),
        .copyTextureToBuffer(source: image, destination: read, bytesPerRow: 256, mipLevel: 1),
        .read(buffer: read, byteCount: 8),
        .renderIndexed(pipeline: pipeline, target: attachment, clear: clear,
                       indexBuffer: .init(buffer: indices, format: .uint16), draw: .init(indexCount: 3),
                       vertexBuffers: streams, textures: [slot: storageIndexed]),
        .copyTextureToBuffer(source: storageIndexed, destination: read, bytesPerRow: 256),
        .read(buffer: read, byteCount: 272),
    ])
    let expectedStorage: [UInt32] = [0, 0xff0000ff, 0xffff0000, 0] + Array(repeating: 0, count: 64)
    let expected: [[UInt32]] = [expectedStorage, [0xff00ff00, 0xff00ff00], expectedStorage]
    let snapshots = try await ticket.wait()
    guard snapshots == expected else { throw GPUError.backendFailure("Unexpected fragment storage pixels: \(snapshots)") }
    return Array(snapshots[0][1...2]) + snapshots[1] + Array(snapshots[2][1...2])
}

private func readWriteStorageDiagnostic(_ device: Device) async throws -> [UInt32] {
    let source = ShaderInput.glsl("""
        #version 450
        layout(local_size_x=4) in;
        layout(rgba8, binding=0) readonly uniform image2D mask;
        layout(r32ui, binding=1) uniform uimage2D image;
        layout(binding=2) buffer Output { uvec4 before[4]; } outputData;
        void main() {
            uint lane = gl_GlobalInvocationID.x;
            if (lane == 3u) {
                outputData.before[3] = imageLoad(image, ivec2(-1, 0));
            } else {
                ivec2 coords = ivec2(int(lane) - 1, 0);
                uvec4 old = imageLoad(image, coords);
                outputData.before[lane] = old;
                if (imageLoad(mask, coords).g > 0.5) {
                    imageStore(image, coords, old + uvec4(5u, 99u, 99u, 99u));
                }
            }
        }
        """, stage: .compute, defines: [:])
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileWebGPUCompute(source)
    let pipeline = try await device.createComputePipeline(shader)
    let spirvShader = try compiler.compileWebGPUCompute(.spirv(readWriteStorageSPIRV()))
    guard spirvShader.storageTextures.count == 1, let binding = spirvShader.storageTextures.first,
          binding.binding == .init(group: 0, binding: 1), binding.format == .r32Uint,
          binding.access == [.load, .store], binding.dimension == .d2, !binding.isArrayed else {
        throw GPUError.backendFailure("Unexpected SPIR-V storage texture reflection")
    }
    let spirvPipeline = try await device.createComputePipeline(spirvShader)
    let mask = try await device.createTexture(.init(width: 2, height: 1, format: .rgba8Unorm,
        usage: [.copyDestination, .storageBinding]))
    let image = try await device.createTexture(.init(width: 2, height: 1, format: .r32Uint,
        usage: [.copySource, .copyDestination, .storageBinding]))
    let upload = try await device.createBuffer(.init(size: 16, usage: [.copySource, .copyDestination]))
    let out = try await device.createBuffer(.init(size: 64, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 64, usage: [.copyDestination, .mapRead]))
    let dispatch = BufferCommand.dispatchCompute(
        pipeline: pipeline, bindings: [.init(binding: 2): out], workgroups: .init(x: 1),
        textures: [.init(binding: 0): mask, .init(binding: 1): image])
    let ticket = try await device.submitCommands([
        .write(buffer: upload, offset: 0, words: [0xff00ff00, 0xff0000ff, 10, 20]),
        .copyBufferToTexture(source: upload, layout: .init(), destination: mask,
                             region: .init(width: 2, height: 1)),
        dispatch,
        .copy(source: out, destination: read, byteCount: 64),
        .read(buffer: read, byteCount: 64),
        .copyTextureToBuffer(source: image, destination: read, bytesPerRow: 256),
        .read(buffer: read, byteCount: 8),
        .copyBufferToTexture(source: upload, layout: .init(offset: 8), destination: image,
                             region: .init(width: 2, height: 1)),
        dispatch,
        .copy(source: out, destination: read, byteCount: 64),
        .read(buffer: read, byteCount: 64),
        .copyTextureToBuffer(source: image, destination: read, bytesPerRow: 256),
        .read(buffer: read, byteCount: 8),
        .dispatchCompute(pipeline: spirvPipeline, bindings: [.init(binding: 2): out], workgroups: .init(x: 1),
                         textures: [.init(binding: 1): image]),
        .copy(source: out, destination: read, byteCount: 24),
        .read(buffer: read, byteCount: 24),
        .copyTextureToBuffer(source: image, destination: read, bytesPerRow: 256),
        .read(buffer: read, byteCount: 8),
    ])
    let expected: [[UInt32]] = [
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0], [5, 0],
        [0, 0, 0, 0, 10, 0, 0, 1, 20, 0, 0, 1, 0, 0, 0, 0], [15, 20],
        [20, 0, 0, 1, 2, 1], [0x01020318, 20],
    ]
    let snapshots = try await ticket.wait()
    guard snapshots == expected else { throw GPUError.backendFailure("Unexpected readable storage results: \(snapshots)") }
    let wgslWords = snapshots[1] + [0, 4, 7, 8, 11, 12].map { snapshots[2][$0] } + snapshots[3]
    return wgslWords + snapshots[4] + snapshots[5]
}

private func readWriteStorageSPIRV() -> [UInt32] {
    func op(_ code: UInt32, _ operands: UInt32...) -> [UInt32] {
        [UInt32(operands.count + 1) << 16 | code] + operands
    }
    var words: [UInt32] = [0x07230203, 0x00010300, 0, 80, 0]
    words += op(17, 1) + op(17, 50) + op(14, 0, 1)
    words += op(15, 5, 40, 0x6e69616d, 0) + op(16, 40, 17, 1, 1, 1)
    words += op(71, 9, 2)
    for member: UInt32 in 0..<6 { words += op(72, 9, member, 35, member * 4) }
    for instruction in [
        op(71, 20, 34, 0), op(71, 20, 33, 1), op(71, 21, 34, 0), op(71, 21, 33, 2),
        op(19, 1), op(21, 2, 32, 0), op(21, 3, 32, 1), op(23, 4, 3, 2), op(23, 5, 2, 2), op(23, 6, 2, 4),
        // Sampled=2, format=R32ui, UniformConstant pointer.
        op(25, 7, 2, 1, 0, 0, 0, 2, 33), op(32, 8, 0, 7),
        op(30, 9, 2, 2, 2, 2, 2, 2), op(32, 10, 12, 9), op(32, 11, 12, 2), op(33, 12, 1),
    ] { words += instruction }
    for index: UInt32 in 0..<6 { words += op(43, 2, 25 + index, index) }
    for instruction in [
        op(43, 3, 31, 0), op(43, 3, 32, 1), op(44, 4, 33, 31, 31), op(44, 4, 34, 32, 31),
        op(43, 2, 35, 0x01020304), op(44, 6, 36, 35, 25, 25, 25),
        op(59, 8, 20, 0), op(59, 10, 21, 12), op(54, 1, 40, 0, 12), op(248, 41),
        op(61, 7, 42, 20), op(98, 6, 43, 42, 34), op(104, 5, 44, 42),
        op(128, 6, 45, 43, 36), op(99, 42, 33, 45),
    ] { words += instruction }
    for index: UInt32 in 0..<6 {
        let vector: UInt32 = index < 4 ? 43 : 44
        let lane = index < 4 ? index : index - 4
        words += op(81, 2, 50 + index, vector, lane)
        words += op(65, 11, 60 + index, 21, 25 + index) + op(62, 60 + index, 50 + index)
    }
    return words + op(253) + op(56)
}

private func textureBarrierDiagnostic(_ device: Device) async throws -> [UInt32] {
    let source = ShaderInput.wgsl("""
        @group(2) @binding(0) var image: texture_storage_2d<r32uint, read_write>;
        @group(0) @binding(0) var<storage, read_write> output: array<u32, 384>;
        fn syncImage() { textureBarrier(); }
        fn rendezvous() { syncImage(); }
        @compute @workgroup_size(128)
        fn run(@builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) group: vec3u) {
            let start = group.x * 128u;
            let index = start + lane;
            textureStore(image, vec2u(index, 0u), vec4u(group.x * 1000u + lane + 10u));
            rendezvous();
            let neighbor = start + (lane + 64u) % 128u;
            output[index] = textureLoad(image, vec2u(neighbor, 0u)).r + 7u;
        }
        """)
    let shader = try IndependentShaderCompiler().compileCompute(source, to: .wgsl)
    let pipeline = try await device.createComputePipeline(shader)
    let image = try await device.createTexture(.init(width: 384, height: 1, format: .r32Uint,
        usage: [.copySource, .storageBinding]))
    let output = try await device.createBuffer(.init(size: 1536, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 1536, usage: [.copyDestination, .mapRead]))
    let ticket = try await device.submitCommands([
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 3),
                         textures: [.init(group: 2, binding: 0): image]),
        .copy(source: output, destination: read, byteCount: 1536),
        .read(buffer: read, byteCount: 1536),
        .copyTextureToBuffer(source: image, destination: read, bytesPerRow: 1536),
        .read(buffer: read, byteCount: 1536),
    ])
    let writes = (0..<384).map { (index: Int) -> UInt32 in
        let group = UInt32(index / 128)
        let lane = UInt32(index % 128)
        return group * 1000 + lane + 10
    }
    let neighbors = (0..<384).map { (index: Int) -> UInt32 in
        let start = index / 128 * 128
        let rotated = (index + 64) % 128
        return writes[start + rotated] + 7
    }
    let snapshots = try await ticket.wait()
    guard snapshots == [neighbors, writes] else { throw GPUError.backendFailure("Unexpected texture barrier readback") }
    return [0, 127, 128, 255, 256, 383].map { snapshots[0][$0] }
}

private func renderReadableStorageDiagnostic(_ device: Device) async throws -> [UInt32] {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl("""
        @group(2) @binding(5) var mask: texture_storage_2d<rgba8unorm, read>;
        @group(1) @binding(4) var image: texture_storage_2d<r32uint, read_write>;

        struct VertexOut {
            @builtin(position) position: vec4f,
            @location(0) x: f32,
        }
        @vertex fn vs(@location(0) point: vec2f) -> VertexOut {
            let x = point.x + textureLoad(mask, vec2u(0u)).r;
            return VertexOut(vec4f(x, point.y, 0.0, 1.0), x);
        }
        @fragment fn fs(@location(0) x: f32) -> @location(0) vec4f {
            let coords = vec2u(0u);
            let old = textureLoad(image, coords);
            if x > 0.0 { textureStore(image, coords, old + vec4u(5u)); }
            let red = select(0.0, 1.0, old.r == 10u);
            return vec4f(red, textureLoad(mask, coords).g, 0.0, 1.0);
        }
        """)
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .wgsl)
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: .wgsl)
    let pipeline = try await device.createRenderPipeline(
        vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm,
        vertexBuffers: [.init(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float32x2)])])
    let mask = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm,
        usage: [.copyDestination, .storageBinding]))
    let image = try await device.createTexture(.init(width: 1, height: 1, format: .r32Uint,
        usage: [.copySource, .copyDestination, .storageBinding]))
    let target = try await device.createTexture(.init(width: 2, height: 1, format: .rgba8Unorm,
        usage: [.copySource, .renderAttachment]))
    let upload = try await device.createBuffer(.init(size: 8, usage: [.copySource, .copyDestination]))
    let vertices = try await device.createBuffer(.init(size: 24, usage: [.vertex, .copyDestination]))
    let read = try await device.createBuffer(.init(size: 8, usage: [.copyDestination, .mapRead]))
    let clear = RenderClearColor(red: 0, green: 0, blue: 1, alpha: 1)
    let ticket = try await device.submitCommands([
        .write(buffer: upload, offset: 0, words: [0xff00ffff, 10]),
        .copyBufferToTexture(source: upload, layout: .init(), destination: mask, region: .init(width: 1, height: 1)),
        .copyBufferToTexture(source: upload, layout: .init(offset: 4), destination: image, region: .init(width: 1, height: 1)),
        .write(buffer: vertices, offset: 0, words: [Float(-1), -1, 3, -1, -1, 3].map(\.bitPattern)),
        .render(pipeline: pipeline, target: target, clear: clear, vertexCount: 3,
                vertexBuffers: [.init(buffer: vertices)],
                textures: [.init(group: 2, binding: 5): mask, .init(group: 1, binding: 4): image]),
        .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256),
        .read(buffer: read, byteCount: 8),
        .copyTextureToBuffer(source: image, destination: read, bytesPerRow: 256),
        .read(buffer: read, byteCount: 4),
    ])
    let snapshots = try await ticket.wait()
    guard snapshots == [[0xffff0000, 0xff00ffff], [15]] else {
        throw GPUError.backendFailure("Unexpected readable render storage results: \(snapshots)")
    }
    return snapshots[0] + snapshots[1]
}

private func depthAttachmentDiagnostic(_ device: Device) async throws -> [UInt32] {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl("""
        struct VertexOut {
            @builtin(position) position: vec4f,
            @location(0) tint: vec4f,
        }
        @vertex fn vs(@location(0) point: vec3f, @location(1) tint: vec4f) -> VertexOut {
            return VertexOut(vec4f(point, 1.0), tint);
        }
        @fragment fn fs(@location(0) tint: vec4f) -> @location(0) vec4f { return tint; }
        """)
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .wgsl)
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: .wgsl)
    let layouts = [VertexBufferLayout(stride: 28, attributes: [
        .init(location: 0, offset: 0, format: .float32x3),
        .init(location: 1, offset: 12, format: .float32x4),
    ])]
    let pipeline = try await device.createRenderPipeline(
        vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm, vertexBuffers: layouts,
        depthState: .init(compare: .less, writeEnabled: true))
    let readOnly = try await device.createRenderPipeline(
        vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm, vertexBuffers: layouts,
        depthState: .init(compare: .always, writeEnabled: false))
    let target = try await device.createTexture(.init(width: 2, height: 1, format: .rgba8Unorm,
        usage: [.copySource, .renderAttachment]))
    let depth = try await device.createTexture(.init(width: 4, height: 2, format: .depth32Float,
        usage: [.copySource, .renderAttachment], mipLevelCount: 2))
    let depthView = try await device.createTextureView(texture: depth, .init(baseMipLevel: 1))
    let vertices = try await device.createBuffer(.init(size: 168, usage: [.vertex, .copyDestination]))
    let indices = try await device.createBuffer(.init(size: 8, usage: [.index, .copyDestination]))
    let read = try await device.createBuffer(.init(size: 8, usage: [.copyDestination, .mapRead]))
    let blue = RenderClearColor(red: 0, green: 0, blue: 1, alpha: 1)
    let streams = [VertexBufferBinding(buffer: vertices)]
    func draw(_ count: UInt32, depth attachment: RenderDepthAttachment?) -> BufferCommand {
        .render(pipeline: pipeline, target: target, clear: blue, vertexCount: count,
                vertexBuffers: streams, depth: attachment)
    }
    let load = RenderDepthAttachment(texture: depthView, loadAction: .load)
    let clear = RenderDepthAttachment(texture: depthView, loadAction: .clear(1))
    do {
        _ = try await device.submitCommands([draw(3, depth: nil)])
        throw GPUError.backendFailure("Accepted a render command without its required depth attachment")
    } catch GPUError.validation("Depth attachment does not match the render pipeline") {}
    let triangles: [Float] = [
        -1, -1, 0.25, 0, 1, 0, 1,
         3, -1, 0.25, 0, 1, 0, 1,
        -1,  3, 0.25, 0, 1, 0, 1,
        -1, -1, 0.75, 1, 0, 0, 1,
         3, -1, 0.75, 1, 0, 0, 1,
        -1,  3, 0.75, 1, 0, 0, 1,
    ]
    let ticket = try await device.submitCommands([
        .write(buffer: vertices, offset: 0, words: triangles.map(\.bitPattern)),
        .write(buffer: indices, offset: 0, words: [0x00040003, 5]),
        draw(3, depth: load),
        .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256),
        .read(buffer: read, byteCount: 8),
        draw(6, depth: clear),
        .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256),
        .read(buffer: read, byteCount: 8),
        .renderIndexed(pipeline: pipeline, target: target, clear: blue,
                       indexBuffer: .init(buffer: indices, format: .uint16), draw: .init(indexCount: 3),
                       vertexBuffers: streams, depth: load),
        .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256),
        .read(buffer: read, byteCount: 8),
        .renderIndexed(pipeline: readOnly, target: target, clear: blue,
                       indexBuffer: .init(buffer: indices, format: .uint16), draw: .init(indexCount: 3),
                       vertexBuffers: streams, depth: load),
        .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256),
        .read(buffer: read, byteCount: 8),
        .copyTextureToBuffer(source: depth, destination: read, bytesPerRow: 256, mipLevel: 1),
        .read(buffer: read, byteCount: 8),
    ])
    try await device.destroyTextureView(depthView)
    let expected: [[UInt32]] = [
        [0xffff0000, 0xffff0000], [0xff00ff00, 0xff00ff00],
        [0xffff0000, 0xffff0000], [0xff0000ff, 0xff0000ff],
        Array(repeating: Float(0.25).bitPattern, count: 2),
    ]
    let snapshots = try await ticket.wait()
    guard snapshots == expected else { throw GPUError.backendFailure("Unexpected depth render results: \(snapshots)") }
    return snapshots.map { $0[0] }
}

private func depthTextureDiagnostic(_ device: Device) async throws -> [UInt32] {
    let compiler = IndependentShaderCompiler()
    let compute = try await device.createComputePipeline(compiler.compileCompute(.wgsl("""
        @group(2) @binding(1) var image: texture_depth_2d;
        @group(0) @binding(0) var<storage, read_write> output: array<f32, 8>;
        fn fetch(p: vec2i) -> f32 { return textureLoad(image, p, 0i); }
        @compute @workgroup_size(1) fn run() {
            let size = textureDimensions(image, 0u);
            output[0u] = fetch(vec2i(0));
            output[1u] = fetch(vec2i(-1, 0));
            output[2u] = textureLoad(image, vec2u(0xffffffffu, 0u), 0u);
            output[3u] = textureLoad(image, vec2u(0), 1u);
            output[4u] = f32(size.x);
            output[5u] = f32(size.y);
            output[6u] = f32(textureNumLevels(image));
            output[7u] = textureLoad(image, vec2u(1, 0), 0u);
        }
        """), to: .wgsl))
    let source = ShaderInput.wgsl("""
        @group(2) @binding(1) var image: texture_depth_2d;
        @vertex fn vs(@location(0) point: vec2f) -> @builtin(position) vec4f {
            return vec4f(point, 0.25, 1.0);
        }
        @fragment fn writeDepth() -> @location(0) vec4f { return vec4f(0, 1, 0, 1); }
        @fragment fn readDepth() -> @location(0) vec4f {
            return vec4f(textureLoad(image, vec2u(0u), 0u) * 4.0, 0, 0, 1);
        }
        """)
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .wgsl)
    let writer = try compiler.compileRenderStage(source, entryPoint: "writeDepth", to: .wgsl)
    let reader = try compiler.compileRenderStage(source, entryPoint: "readDepth", to: .wgsl)
    let layouts = [VertexBufferLayout(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float32x2)])]
    let render = try await device.createRenderPipeline(
        vertex: vertex, fragment: writer, colorFormat: .rgba8Unorm, vertexBuffers: layouts,
        depthState: .init(compare: .less))
    let preview = try await device.createRenderPipeline(
        vertex: vertex, fragment: reader, colorFormat: .rgba8Unorm, vertexBuffers: layouts)
    let conflicting = try await device.createRenderPipeline(
        vertex: vertex, fragment: reader, colorFormat: .rgba8Unorm, vertexBuffers: layouts,
        depthState: .init(compare: .always, writeEnabled: false))
    let depth = try await device.createTexture(.init(width: 4, height: 2, format: .depth32Float,
        usage: [.renderAttachment, .textureBinding], mipLevelCount: 2))
    let view = try await device.createTextureView(texture: depth, .init(baseMipLevel: 1))
    let color = try await device.createTexture(.init(width: 2, height: 1, format: .rgba8Unorm,
        usage: [.renderAttachment, .textureBinding, .copySource]))
    let vertices = try await device.createBuffer(.init(size: 24, usage: [.vertex, .copyDestination]))
    let output = try await device.createBuffer(.init(size: 32, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 32, usage: [.copyDestination, .mapRead]))
    let clear = RenderClearColor(red: 0, green: 0, blue: 1, alpha: 1)
    let depthSlot = ComputeBindingSlot(group: 2, binding: 1)
    let streams = [VertexBufferBinding(buffer: vertices)]
    let depthAttachment = RenderDepthAttachment(texture: view, loadAction: .clear(1))
    func dispatch(_ texture: ResourceID) -> BufferCommand {
        .dispatchCompute(pipeline: compute, bindings: [.init(binding: 0): output],
                         workgroups: .init(x: 1), textures: [depthSlot: texture])
    }
    do {
        _ = try await device.submitCommands([
            .render(pipeline: conflicting, target: color, clear: clear, vertexCount: 3,
                    vertexBuffers: streams, textures: [depthSlot: view], depth: depthAttachment),
        ])
        throw GPUError.backendFailure("Accepted overlapping sampled and attached depth views")
    } catch GPUError.validation("Render attachment cannot be bound as a sampled texture") {}
    let ticket = try await device.submitCommands([
        .write(buffer: vertices, offset: 0, words: [Float(-1), -1, 3, -1, -1, 3].map(\.bitPattern)),
        .render(pipeline: render, target: color, clear: clear, vertexCount: 3,
                vertexBuffers: streams, depth: depthAttachment),
        .render(pipeline: preview, target: color, clear: clear, vertexCount: 3,
                vertexBuffers: streams, textures: [depthSlot: view]),
        .copyTextureToBuffer(source: color, destination: read, bytesPerRow: 256),
        .read(buffer: read, byteCount: 8),
        dispatch(view),
        .copy(source: output, destination: read, byteCount: 32),
        .read(buffer: read, byteCount: 32),
        dispatch(depth),
        .copy(source: output, destination: read, byteCount: 32),
        .read(buffer: read, byteCount: 32),
    ])
    let viewed: [Float] = [0.25, 0, 0, 0, 2, 1, 1, 0.25]
    let whole: [Float] = [0, 0, 0, 0.25, 4, 2, 2, 0]
    let expected: [[UInt32]] = [[0xff0000ff, 0xff0000ff], viewed.map(\.bitPattern), whole.map(\.bitPattern)]
    let snapshots = try await ticket.wait()
    guard snapshots == expected else { throw GPUError.backendFailure("Unexpected depth texture results: \(snapshots)") }
    return [snapshots[0][0], snapshots[1][0], snapshots[1][3], snapshots[1][4], snapshots[2][3]]
}

private func depthComparisonDiagnostic(_ device: Device) async throws -> [UInt32] {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl("""
        @group(2) @binding(1) var image: texture_depth_2d;
        @group(2) @binding(2) var state: sampler_comparison;
        @group(2) @binding(3) var nearest: sampler;
        @vertex fn vs(@location(0) point: vec2f) -> @builtin(position) vec4f {
            return vec4f(point, point.x * 0.5 + 0.5, 1.0);
        }
        @fragment fn writeDepth() -> @location(0) vec4f { return vec4f(0, 0, 0, 1); }
        @fragment fn fs() -> @location(0) vec4f {
            let uv = vec2f(0.25, 0.5);
            let a = textureSampleCompare(image, state, uv, 0.5);
            let b = textureSampleCompareLevel(image, state, uv, 0.5, vec2i(1, 0));
            let c = textureSample(image, nearest, uv);
            let d = textureSample(image, nearest, uv, vec2i(1, 0));
            return vec4f(a, b, c, d);
        }
        """)
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .wgsl)
    let writer = try compiler.compileRenderStage(source, entryPoint: "writeDepth", to: .wgsl)
    let reader = try compiler.compileRenderStage(source, entryPoint: "fs", to: .wgsl)
    let layouts = [VertexBufferLayout(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float32x2)])]
    let render = try await device.createRenderPipeline(
        vertex: vertex, fragment: writer, colorFormat: .rgba8Unorm, vertexBuffers: layouts,
        depthState: .init(compare: .less))
    let preview = try await device.createRenderPipeline(
        vertex: vertex, fragment: reader, colorFormat: .rgba8Unorm, vertexBuffers: layouts)
    let compute = try await device.createComputePipeline(
        compiler.compileCompute(.wgsl("""
        @group(2) @binding(1) var image: texture_depth_2d;
        @group(2) @binding(2) var state: sampler_comparison;
        @group(2) @binding(3) var nearest: sampler;
        @group(0) @binding(0) var<storage, read_write> out: array<vec4f, 2>;
        @compute @workgroup_size(1) fn run() {
            let a = textureSampleCompareLevel(image, state, vec2f(0.25, 0.5), 0.5);
            let b = textureSampleCompareLevel(image, state, vec2f(0.75, 0.5), 0.5);
            out[0] = vec4f(a, b, textureLoad(image, vec2u(0), 0u), textureLoad(image, vec2u(1, 0), 0u));
            let uv = vec2f(0.25, 0.5);
            let lod = 1i;
            out[1] = vec4f(textureSampleLevel(image, nearest, uv, 0),
                           textureSampleLevel(image, nearest, uv, 0u, vec2i(1, 0)),
                           textureSampleLevel(image, nearest, uv, lod),
                           textureSampleLevel(image, nearest, uv, u32(lod)));
        }
        """), to: .wgsl))
    let spirvShader = try compiler.compileWebGPUCompute(.spirv(depthComparisonSPIRV()))
    guard spirvShader.sampledTextures.count == 1, let texture = spirvShader.sampledTextures.first,
          texture.binding == .init(group: 2, binding: 1), texture.isDepth, texture.dimension == .d2,
          spirvShader.samplerBindings.count == 1, let sampler = spirvShader.samplerBindings.first,
          sampler.binding == .init(group: 2, binding: 2), sampler.isComparison else {
        throw GPUError.backendFailure("Unexpected SPIR-V depth comparison reflection")
    }
    let spirvCompute = try await device.createComputePipeline(spirvShader)
    let glslComparison = ShaderInput.glsl("""
        #version 450
        layout(local_size_x = 1) in;
        layout(set=2, binding=1) uniform texture2D image;
        layout(set=2, binding=2) uniform samplerShadow compareState;
        layout(set=0, binding=0) buffer Result { float values[2]; } outputData;
        float compareDepth(texture2D source, samplerShadow state, vec2 uv) {
            return textureLod(sampler2DShadow(source, state), vec3(uv, 0.5), 0.0);
        }
        void main() {
            outputData.values[0] = compareDepth(image, compareState, vec2(0.25, 0.5));
            outputData.values[1] = compareDepth(image, compareState, vec2(0.75, 0.5));
        }
        """, stage: .compute, defines: [:])
    let glslCompute = try await device.createComputePipeline(compiler.compileWebGPUCompute(glslComparison))
    let depth = try await device.createTexture(.init(width: 4, height: 2, format: .depth32Float,
        usage: [.renderAttachment, .textureBinding], mipLevelCount: 3))
    let view = try await device.createTextureView(texture: depth, .init(baseMipLevel: 1))
    let compare = try await device.createSampler(.init(compare: .less))
    let plain = try await device.createSampler()
    let color = try await device.createTexture(.init(width: 2, height: 1, format: .rgba8Unorm,
        usage: [.renderAttachment, .copySource], mipLevelCount: 2))
    let vertices = try await device.createBuffer(.init(size: 24, usage: [.vertex, .copyDestination]))
    let output = try await device.createBuffer(.init(size: 32, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 32, usage: [.copyDestination, .mapRead]))
    let clear = RenderClearColor(red: 0, green: 0, blue: 0, alpha: 1)
    let textures: [ComputeBindingSlot: ResourceID] = [.init(group: 2, binding: 1): view]
    let samplerSlot = ComputeBindingSlot(group: 2, binding: 2)
    let rawSamplerSlot = ComputeBindingSlot(group: 2, binding: 3)
    let streams = [VertexBufferBinding(buffer: vertices)]
    func visualize(_ sampler: ResourceID) -> BufferCommand {
        .render(pipeline: preview, target: color, clear: clear, vertexCount: 3,
                vertexBuffers: streams, textures: textures, samplers: [samplerSlot: sampler, rawSamplerSlot: plain])
    }
    do {
        _ = try await device.submitCommands([visualize(plain)])
        throw GPUError.backendFailure("Accepted a plain sampler for comparison")
    } catch GPUError.validation("Invalid sampler binding") {}
    let ticket = try await device.submitCommands([
        .write(buffer: vertices, offset: 0, words: [Float(-1), -1, 3, -1, -1, 3].map(\.bitPattern)),
        .render(pipeline: render, target: color, clear: clear, vertexCount: 3, vertexBuffers: streams,
                depth: .init(texture: depth, mipLevel: 1)),
        .render(pipeline: render, target: color, clear: clear, vertexCount: 3, vertexBuffers: streams,
                targetMipLevel: 1, depth: .init(texture: depth, mipLevel: 2)),
        visualize(compare),
        .copyTextureToBuffer(source: color, destination: read, bytesPerRow: 256),
        .read(buffer: read, byteCount: 8),
        .dispatchCompute(pipeline: compute, bindings: [.init(binding: 0): output], workgroups: .init(x: 1),
                         textures: textures, samplers: [samplerSlot: compare, rawSamplerSlot: plain]),
        .copy(source: output, destination: read, byteCount: 32),
        .read(buffer: read, byteCount: 32),
        .dispatchCompute(pipeline: spirvCompute, bindings: [.init(binding: 0): output], workgroups: .init(x: 1),
                         textures: textures, samplers: [samplerSlot: compare]),
        .copy(source: output, destination: read, byteCount: 8),
        .read(buffer: read, byteCount: 8),
        .dispatchCompute(pipeline: glslCompute, bindings: [.init(binding: 0): output], workgroups: .init(x: 1),
                         textures: textures, samplers: [samplerSlot: compare]),
        .copy(source: output, destination: read, byteCount: 8),
        .read(buffer: read, byteCount: 8),
    ])
    let compared: [Float] = [0, 1, 0.25, 0.75, 0.25, 0.75, 0.5, 0.5]
    let expected: [[UInt32]] = [[0xbf40ff00, 0xbf40ff00], compared.map(\.bitPattern), [Float(1).bitPattern, 0]]
    let snapshots = try await ticket.wait()
    guard snapshots == expected + [[0, Float(1).bitPattern]] else {
        throw GPUError.backendFailure("Unexpected depth sampling results: \(snapshots)")
    }
    return snapshots.dropLast().flatMap { $0 }
}

private func depthComparisonSPIRV() -> [UInt32] {
    func op(_ code: UInt32, _ operands: UInt32...) -> [UInt32] {
        [UInt32(operands.count + 1) << 16 | code] + operands
    }
    var words: [UInt32] = [0x07230203, 0x00010300, 0, 70, 0]
    for instruction in [
        op(17, 1), op(14, 0, 1), op(15, 5, 40, 0x6e69616d, 0), op(16, 40, 17, 1, 1, 1),
        op(71, 11, 2), op(72, 11, 0, 35, 0), op(72, 11, 1, 35, 4),
        op(71, 20, 34, 0), op(71, 20, 33, 0), op(71, 21, 34, 2), op(71, 21, 33, 1),
        op(71, 22, 34, 2), op(71, 22, 33, 2),
        op(19, 1), op(22, 2, 32), op(21, 3, 32, 1), op(23, 4, 2, 2),
        // Depth=1, Sampled=1. The image and sampler are separate UniformConstant bindings.
        op(25, 5, 2, 1, 1, 0, 0, 1, 0), op(26, 6), op(27, 7, 5), op(32, 8, 0, 5), op(32, 9, 0, 6),
        op(33, 10, 1), op(30, 11, 2, 2), op(32, 12, 12, 11), op(32, 13, 12, 2), op(33, 14, 2, 5, 6, 4),
        op(43, 3, 25, 0), op(43, 3, 26, 1), op(43, 2, 27, 0), op(43, 2, 28, 0x3e800000),
        op(43, 2, 29, 0x3f000000), op(43, 2, 30, 0x3f400000), op(44, 4, 31, 28, 29), op(44, 4, 32, 30, 29),
        op(59, 12, 20, 12), op(59, 8, 21, 0), op(59, 9, 22, 0),
        // The inner helper pairs the forwarded handles and compares reference 0.5 at LOD 0.
        op(54, 2, 50, 0, 14), op(55, 5, 51), op(55, 6, 52), op(55, 4, 53), op(248, 54),
        op(86, 7, 55, 51, 52), op(90, 2, 56, 55, 53, 29, 2, 27), op(254, 56), op(56),
        // The outer helper forwards both resources and the coordinates unchanged.
        op(54, 2, 60, 0, 14), op(55, 5, 61), op(55, 6, 62), op(55, 4, 63), op(248, 64),
        op(57, 2, 65, 50, 61, 62, 63), op(254, 65), op(56),
        op(54, 1, 40, 0, 10), op(248, 41), op(61, 5, 42, 21), op(61, 6, 43, 22),
        // Sample right then left to distinguish this readback from the preceding WGSL dispatch.
        op(57, 2, 45, 60, 42, 43, 32), op(57, 2, 46, 60, 42, 43, 31),
        op(65, 13, 47, 20, 25), op(62, 47, 45), op(65, 13, 48, 20, 26), op(62, 48, 46),
        op(253), op(56),
    ] { words += instruction }
    return words
}

private func depthGatherDiagnostic(_ device: Device, comparison: Bool) async throws -> [UInt32] {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl("""
        @vertex fn vs(@location(0) point: vec3f) -> @builtin(position) vec4f { return vec4f(point, 1.0); }
        @fragment fn fs() -> @location(0) vec4f { return vec4f(1, 0, 0, 1); }
        """)
    let render = try await device.createRenderPipeline(
        vertex: compiler.compileRenderStage(source, entryPoint: "vs", to: .wgsl),
        fragment: compiler.compileRenderStage(source, entryPoint: "fs", to: .wgsl),
        colorFormat: .rgba8Unorm,
        vertexBuffers: [.init(stride: 12, attributes: [.init(location: 0, offset: 0, format: .float32x3)])],
        depthState: .init())
    let gatherSource = comparison ? """
        @group(0) @binding(0) var<storage, read_write> out: array<vec4f, 4>;
        @group(0) @binding(1) var image: texture_depth_2d;
        @group(0) @binding(2) var state: sampler_comparison;
        fn gather(uv: vec2f) -> vec4f { return textureGatherCompare(image, state, uv, 0.375); }
        @compute @workgroup_size(1) fn run(@builtin(global_invocation_id) id: vec3u) {
            if id.x == 0u {
                let uv = vec2f(0.33333334, 0.5);
                out[0u] = gather(uv);
                out[1u] = textureGatherCompare(image, state, uv, 0.625, vec2i(1, 0));
                out[2u] = textureGatherCompare(image, state, uv, 0.375, vec2i(-1, 0));
                out[3u] = textureGatherCompare(image, state, uv, 0.25);
            }
        }
        """ : """
        @group(0) @binding(0) var<storage, read_write> out: array<vec4f, 4>;
        @group(0) @binding(1) var image: texture_depth_2d;
        @group(0) @binding(2) var state: sampler;
        fn gather(uv: vec2f) -> vec4f { return textureGather(image, state, uv); }
        @compute @workgroup_size(1) fn run() {
            let uv = vec2f(0.33333334, 0.5);
            out[0u] = gather(uv);
            out[1u] = textureGather(image, state, uv, vec2i(1, 0));
            out[2u] = textureGather(image, state, uv, vec2i(-1, 0));
            out[3u] = textureGather(image, state, uv);
        }
        """
    let compute = try await device.createComputePipeline(compiler.compileCompute(.wgsl(gatherSource), to: .wgsl))
    let spirvPipeline = comparison ? try await spirvGatherPipeline(device, comparison: true) : nil
    let depth = try await device.createTexture(.init(width: 6, height: 4, format: .depth32Float,
        usage: [.renderAttachment, .textureBinding], mipLevelCount: 2))
    let view = try await device.createTextureView(texture: depth, .init(baseMipLevel: 1))
    let color = try await device.createTexture(.init(width: 3, height: 2, format: .rgba8Unorm, usage: .renderAttachment))
    let sampler = try await device.createSampler(.init(addressModeU: .repeat, compare: comparison ? .less : nil))
    let vertices = try await device.createBuffer(.init(size: 432, usage: [.vertex, .copyDestination]))
    let output = try await device.createBuffer(.init(size: 64, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 64, usage: [.copyDestination, .mapRead]))
    let x: [Float] = [-1, -1 / 3, 1 / 3, 1]
    let y: [Float] = [1, 0, -1]
    let values: [[Float]] = [[0.125, 0.375, 0.625], [0.25, 0.5, 0.875]]
    var grid: [Float] = []
    for row in 0..<2 {
        for column in 0..<3 {
            let (l, r, t, b, z) = (x[column], x[column + 1], y[row], y[row + 1], values[row][column])
            grid += [l, t, z, l, b, z, r, t, z, r, t, z, l, b, z, r, b, z]
        }
    }
    var commands: [BufferCommand] = [
        .write(buffer: vertices, offset: 0, words: grid.map(\.bitPattern)),
        .render(pipeline: render, target: color, clear: .init(red: 0, green: 0, blue: 0, alpha: 1),
                vertexCount: 36, vertexBuffers: [.init(buffer: vertices)], depth: .init(texture: view)),
        .dispatchCompute(pipeline: compute, bindings: [.init(binding: 0): output], workgroups: .init(x: 1),
                         textures: [.init(binding: 1): view], samplers: [.init(binding: 2): sampler]),
        .copy(source: output, destination: read, byteCount: 64),
        .read(buffer: read, byteCount: 64),
    ]
    if let spirvPipeline {
        commands += [
            .dispatchCompute(pipeline: spirvPipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1),
                             textures: [.init(binding: 1): view], samplers: [.init(binding: 2): sampler]),
            .copy(source: output, destination: read, byteCount: 16),
            .read(buffer: read, byteCount: 16),
        ]
    }
    let ticket = try await device.submitCommands(commands)
    let expected: [Float] = comparison
        ? [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0]
        : [0.25, 0.5, 0.375, 0.125, 0.5, 0.875, 0.625, 0.375,
           0.875, 0.25, 0.125, 0.625, 0.25, 0.5, 0.375, 0.125]
    let snapshots = try await ticket.wait()
    let spirvExpected: [Float] = [1, 0, 0, 1]
    let expectedSnapshots = [expected.map(\.bitPattern)] + (comparison ? [spirvExpected.map(\.bitPattern)] : [])
    guard snapshots == expectedSnapshots else {
        throw GPUError.backendFailure("Unexpected depth gather readback: \(snapshots)")
    }
    return snapshots.flatMap { $0 }
}

private func multisampleDiagnostic(_ device: Device) async throws -> [UInt32] {
    let compiler = IndependentShaderCompiler()
    let stages = ShaderInput.wgsl("""
        @group(0) @binding(3) var<storage, read_write> visits: atomic<u32>;
        @vertex fn vs(@location(0) xy: vec2f) -> @builtin(position) vec4f { return vec4f(xy, 0.25, 1.0); }
        @vertex fn native_vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
            _ = atomicOr(&visits, 1u << index);
            return vec4f(0, 0, 0, 1);
        }
        @fragment fn fs() -> @location(0) vec4f { return vec4f(1, 0, 0, 1); }
        """)
    guard case .spirv(let renderBinary) = try compiler.compile(stages, to: .spirv) else {
        throw GPUError.backendFailure("Expected SPIR-V for vertex storage profile validation")
    }
    for source in [stages, .spirv(renderBinary)] {
        do {
            _ = try compiler.compileWebGPURenderStage(source, entryPoint: "native_vs")
            throw GPUError.backendFailure("Browser profile accepted vertex writable storage")
        } catch ShaderFailure.unsupported("Browser WebGPU does not support vertex writable storage") {}
    }
    let vertex = try compiler.compileWebGPURenderStage(.spirv(renderBinary), entryPoint: "vs")
    let render = try await device.createRenderPipeline(
        vertex: vertex, fragment: compiler.compileWebGPURenderStage(stages, entryPoint: "fs"),
        colorFormat: .rgba8Unorm,
        vertexBuffers: [.init(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float32x2)])],
        depthState: .init(compare: .less), sampleCount: 4)
    let vertexOnly = try await device.createRenderPipeline(
        vertex: vertex, fragment: nil,
        colorFormat: nil, vertexBuffers: render.vertexBuffers, depthState: .init(compare: .always),
        sampleCount: 4, sampleMask: 0b1010)
    let input = ShaderInput.wgsl("""
        @group(0) @binding(0) var color: texture_multisampled_2d<f32>;
        @group(0) @binding(1) var depth: texture_depth_multisampled_2d;
        @group(0) @binding(2) var<storage, read_write> out: array<u32, 12>;
        @compute @workgroup_size(1) fn run() {
            for (var lane = 0u; lane < 4u; lane++) {
                let r = textureLoad(color, vec2u(0), lane).r;
                let d = textureLoad(depth, vec2i(0), i32(lane));
                out[lane] = select(select(2u, 0u, r == 0.0), 1u, r == 1.0);
                out[4u + lane] = select(select(0u, 4u, d == 1.0), 1u, d == 0.25);
            }
            let dimensions = textureDimensions(color);
            let negative = textureLoad(color, vec2i(0), -1i);
            let outside = textureLoad(depth, vec2u(0xffffffffu), 0u);
            let invalid = textureLoad(depth, vec2u(0), 0xffffffffu);
            out[8] = textureNumSamples(color);
            out[9] = textureNumSamples(depth);
            out[10] = dimensions.x + dimensions.y;
            out[11] = select(1u, 0u, negative.a == 0.0 && outside == 0.0 && invalid == 0.0);
        }
        """)
    let compute = try await device.createComputePipeline(compiler.compileWebGPUCompute(input))
    guard case .spirv(var binary) = try compiler.compile(input, to: .spirv) else {
        throw GPUError.backendFailure("Expected a SPIR-V multisample shader")
    }
    binary.insert(contentsOf: [0x00020011, 27], at: 5) // StorageImageMultisample, declared but unused.
    let imported = try await device.createComputePipeline(compiler.compileWebGPUCompute(.spirv(binary)))
    let color = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm,
        usage: [.renderAttachment, .textureBinding, .copySource], sampleCount: 4))
    let depth = try await device.createTexture(.init(width: 1, height: 1, format: .depth32Float,
        usage: [.renderAttachment, .textureBinding, .copySource], sampleCount: 4))
    let resolveDescriptor = TextureDescriptor(width: 2, height: 2, format: .rgba8Unorm,
        usage: [.renderAttachment, .copySource], mipLevelCount: 2)
    let resolved = try await device.createTexture(resolveDescriptor)
    let resolvedIndexed = try await device.createTexture(resolveDescriptor)
    let resolvedView = try await device.createTextureView(texture: resolvedIndexed, .init(baseMipLevel: 1))
    let colorCopy = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm,
        usage: [.renderAttachment, .textureBinding, .copyDestination], sampleCount: 4))
    let depthCopy = try await device.createTexture(.init(width: 1, height: 1, format: .depth32Float,
        usage: [.renderAttachment, .textureBinding, .copyDestination], sampleCount: 4))
    let colorView = try await device.createTextureView(texture: colorCopy)
    let vertices = try await device.createBuffer(.init(size: 24, usage: [.vertex, .copyDestination]))
    let indices = try await device.createBuffer(.init(size: 12, usage: [.index, .copyDestination]))
    let out = try await device.createBuffer(.init(size: 48, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 264, usage: [.copyDestination, .mapRead]))
    let clear = RenderClearColor(red: 0, green: 0, blue: 0, alpha: 1)
    let ticket = try await device.submitCommands([
        .write(buffer: vertices, offset: 0, words: [Float(-3), -1, 0, -1, 0, 5].map(\.bitPattern)),
        .write(buffer: indices, offset: 0, words: [0, 1, 2]),
        .render(pipeline: render, target: color, clear: clear,
                vertexCount: 3, vertexBuffers: [.init(buffer: vertices)], depth: .init(texture: depth),
                resolve: .init(texture: resolved, mipLevel: 1)),
        .renderIndexed(pipeline: render, target: color, clear: clear,
                       indexBuffer: .init(buffer: indices, format: .uint32), draw: .init(indexCount: 3),
                       vertexBuffers: [.init(buffer: vertices)], depth: .init(texture: depth),
                       resolve: .init(texture: resolvedView)),
        .copyTextureToTexture(source: color, destination: colorCopy, sourceRegion: .init(width: 1, height: 1)),
        .copyTextureToTexture(source: depth, destination: depthCopy, sourceRegion: .init(width: 1, height: 1)),
        .dispatchCompute(pipeline: imported, bindings: [.init(binding: 2): out], workgroups: .init(x: 1),
                         textures: [.init(binding: 0): colorView, .init(binding: 1): depthCopy]),
        .copy(source: out, destination: read, byteCount: 48), .read(buffer: read, byteCount: 48),
        .copyTextureToBuffer(source: resolved, destination: read, bytesPerRow: 256, mipLevel: 1),
        .read(buffer: read, byteCount: 4),
        .copyTextureToBuffer(source: resolvedIndexed, destination: read, bytesPerRow: 256, mipLevel: 1),
        .read(buffer: read, byteCount: 4),
        .copyTextureToBuffer(source: resolvedIndexed, destination: read, bytesPerRow: 256),
        .read(buffer: read, byteCount: 8),
        .write(buffer: vertices, offset: 0, words: [Float(-1), -1, 3, -1, -1, 3].map(\.bitPattern)),
        .render(pipeline: vertexOnly, target: nil, colorLoad: .load, vertexCount: 3,
                vertexBuffers: [.init(buffer: vertices)], depth: .init(texture: depth)),
        .dispatchCompute(pipeline: compute, bindings: [.init(binding: 2): out], workgroups: .init(x: 1),
                         textures: [.init(binding: 0): colorView, .init(binding: 1): depth]),
        .copy(source: out, sourceOffset: 16, destination: read, byteCount: 16),
        .read(buffer: read, byteCount: 16),
        .render(pipeline: render, target: color, clear: clear, vertexCount: 3,
                vertexBuffers: [.init(buffer: vertices)], depth: .init(texture: depth),
                resolve: .init(texture: resolved, mipLevel: 1), colorStore: .discard),
        .copyTextureToBuffer(source: resolved, destination: read, bytesPerRow: 256, mipLevel: 1),
        .read(buffer: read, byteCount: 4),
        .render(pipeline: render, target: color, colorLoad: .load, vertexCount: 0,
                vertexBuffers: [.init(buffer: vertices)], depth: .init(texture: depth),
                resolve: .init(texture: resolved, mipLevel: 1)),
        .copyTextureToBuffer(source: resolved, destination: read, bytesPerRow: 256, mipLevel: 1),
        .read(buffer: read, byteCount: 4),
    ])
    let snapshots = try await ticket.wait()
    guard snapshots.count == 7, let words = snapshots.first, words.count == 12,
          words[8..<12].elementsEqual([4, 4, 2, 0]),
          words[0..<4].sorted() == [0, 0, 1, 1],
          snapshots[1] == [0xff000080], snapshots[2] == [0xff000080], snapshots[3] == [0, 0],
          snapshots[4] == [4, 1, 4, 1],
          snapshots[5] == [0xff0000ff], snapshots[6] == [0],
          zip(words[0..<4], words[4..<8]).allSatisfy({ color, depth in
              color <= 1 && depth == (color == 1 ? 1 : 4)
          }) else {
        throw GPUError.backendFailure("Unexpected multisampled texture or resolve readback: \(snapshots)")
    }
    return snapshots.flatMap { $0 }
}

private func integerTextureDiagnostic(_ device: Device) async throws -> [UInt32] {
    let compiler = IndependentShaderCompiler()
    let draw = ShaderInput.wgsl("""
        @vertex fn vs(@location(0) xy: vec2f) -> @builtin(position) vec4f { return vec4f(xy, 0, 1); }
        @fragment fn signedColor() -> @location(0) vec4i { return vec4i(-17, 23, -3, 1); }
        @fragment fn unsignedColor() -> @location(0) vec4u { return vec4u(249, 1, 200, 7); }
        """)
    let vertex = try compiler.compileRenderStage(draw, entryPoint: "vs", to: .wgsl)
    let signed = try compiler.compileRenderStage(draw, entryPoint: "signedColor", to: .wgsl)
    let unsigned = try compiler.compileRenderStage(draw, entryPoint: "unsignedColor", to: .wgsl)
    let layouts = [VertexBufferLayout(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float32x2)])]
    let signedPipeline = try await device.createRenderPipeline(
        vertex: vertex, fragment: signed, colorFormat: .rgba8Sint, vertexBuffers: layouts, sampleCount: 4)
    let unsignedPipeline = try await device.createRenderPipeline(
        vertex: vertex, fragment: unsigned, colorFormat: .rgba8Uint, vertexBuffers: layouts, sampleCount: 4)
    let singlePipeline = try await device.createRenderPipeline(
        vertex: vertex, fragment: unsigned, colorFormat: .rgba8Uint, vertexBuffers: layouts)
    let reader = try await device.createComputePipeline(compiler.compileCompute(.wgsl("""
        @group(0) @binding(0) var signedImage: texture_multisampled_2d<i32>;
        @group(0) @binding(1) var unsignedImage: texture_multisampled_2d<u32>;
        @group(0) @binding(2) var single: texture_2d<u32>;
        @group(0) @binding(3) var<storage, read_write> signedOut: array<vec4i, 5>;
        @group(0) @binding(4) var<storage, read_write> unsignedOut: array<vec4u, 7>;
        @compute @workgroup_size(4) fn run(@builtin(local_invocation_index) lane: u32) {
            signedOut[lane] = textureLoad(signedImage, vec2i(0), i32(lane));
            unsignedOut[lane] = textureLoad(unsignedImage, vec2u(0), lane);
            if lane == 0u {
                signedOut[4] = textureLoad(signedImage, vec2i(0), -1i);
                unsignedOut[4] = textureLoad(unsignedImage, vec2u(0), 0xffffffffu);
                unsignedOut[5] = textureLoad(single, vec2u(0), 0u);
                unsignedOut[6] = vec4u(textureNumSamples(signedImage), textureNumSamples(unsignedImage),
                                      textureDimensions(unsignedImage));
            }
        }
        """), to: .wgsl))
    let usage: TextureUsage = [.renderAttachment, .textureBinding]
    let signedTexture = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Sint, usage: usage, sampleCount: 4))
    let unsignedTexture = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Uint, usage: usage, sampleCount: 4))
    let singleTexture = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Uint, usage: usage))
    let vertices = try await device.createBuffer(.init(size: 24, usage: [.vertex, .copyDestination]))
    let signedOut = try await device.createBuffer(.init(size: 80, usage: [.storage, .copySource]))
    let unsignedOut = try await device.createBuffer(.init(size: 112, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 112, usage: [.copyDestination, .mapRead]))
    let clear = RenderClearColor(red: 0, green: 0, blue: 0, alpha: 0)
    func render(_ pipeline: RenderPipeline, to target: ResourceID) -> BufferCommand {
        .render(pipeline: pipeline, target: target, clear: clear, vertexCount: 3, vertexBuffers: [.init(buffer: vertices)])
    }
    let ticket = try await device.submitCommands([
        .write(buffer: vertices, offset: 0, words: [Float(-1), -1, 3, -1, -1, 3].map(\.bitPattern)),
        render(signedPipeline, to: signedTexture), render(unsignedPipeline, to: unsignedTexture),
        render(singlePipeline, to: singleTexture),
        .dispatchCompute(pipeline: reader, bindings: [.init(binding: 3): signedOut, .init(binding: 4): unsignedOut],
                         workgroups: .init(x: 1),
                         textures: [.init(binding: 0): signedTexture, .init(binding: 1): unsignedTexture,
                                    .init(binding: 2): singleTexture]),
        .copy(source: signedOut, destination: read, byteCount: 80), .read(buffer: read, byteCount: 80),
        .copy(source: unsignedOut, destination: read, byteCount: 112), .read(buffer: read, byteCount: 112),
    ])
    let signedWords: [UInt32] = Array(repeating: [Int32(-17), 23, -3, 1].map(UInt32.init(bitPattern:)), count: 4)
        .flatMap { $0 } + [0, 0, 0, 0]
    let unsignedSample: [UInt32] = [249, 1, 200, 7]
    let unsignedWords: [UInt32] = Array(repeating: unsignedSample, count: 4).flatMap { $0 } +
        [0, 0, 0, 0] + unsignedSample + [4, 4, 1, 1]
    let snapshots = try await ticket.wait()
    guard snapshots == [signedWords, unsignedWords] else {
        throw GPUError.backendFailure("Unexpected integer texture readback: \(snapshots)")
    }
    return snapshots.flatMap { $0 }
}

private func integerGatherDiagnostic(_ device: Device) async throws -> [UInt32] {
    let shader = ShaderInput.wgsl("""
        @group(0) @binding(0) var signedImage: texture_2d<i32>;
        @group(0) @binding(1) var unsignedImage: texture_2d<u32>;
        @group(0) @binding(2) var state: sampler;
        @group(0) @binding(3) var<storage, read_write> signedOut: array<vec4i, 2>;
        @group(0) @binding(4) var<storage, read_write> unsignedOut: array<vec4u, 2>;
        const alpha = 1u + 2u;
        fn gatherSigned(uv: vec2f) -> vec4i { return textureGather(alpha, signedImage, state, uv); }
        @compute @workgroup_size(1) fn run() {
            let uv = vec2f(0.33333334, 0.5);
            signedOut[0] = gatherSigned(uv);
            signedOut[1] = textureGather(alpha, signedImage, state, uv, vec2i(1, 0));
            unsignedOut[0] = textureGather(2i, unsignedImage, state, uv, vec2i(-1, 0));
            unsignedOut[1] = textureGather(alpha, unsignedImage, state, uv);
        }
        """)
    let compute = try await device.createComputePipeline(IndependentShaderCompiler().compileCompute(shader, to: .wgsl))
    let signed = try await device.createTexture(.init(width: 6, height: 4, format: .rgba8Sint,
        usage: [.copyDestination, .textureBinding], mipLevelCount: 2))
    let unsigned = try await device.createTexture(.init(width: 6, height: 4, format: .rgba8Uint,
        usage: [.copyDestination, .textureBinding], mipLevelCount: 2))
    let signedView = try await device.createTextureView(texture: signed, .init(baseMipLevel: 1))
    let unsignedView = try await device.createTextureView(texture: unsigned, .init(baseMipLevel: 1))
    let nearest = try await device.createSampler(.init(addressModeU: .repeat))
    let input = try await device.createBuffer(.init(size: 268, usage: [.copyDestination, .copySource]))
    let signedOut = try await device.createBuffer(.init(size: 32, usage: [.storage, .copySource]))
    let unsignedOut = try await device.createBuffer(.init(size: 32, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 32, usage: [.copyDestination, .mapRead]))
    let ticket = try await device.submitCommands([
        .write(buffer: input, offset: 0, words: [0x8c190a14, 0x962d1e28, 0xa041323c]),
        .write(buffer: input, offset: 256, words: [0xaa554650, 0xb4695a64, 0xbe7d6e78]),
        .copyBufferToTexture(source: input, layout: .init(bytesPerRow: 256), destination: signed,
                             region: .init(width: 3, height: 2, mipLevel: 1)),
        .copyBufferToTexture(source: input, layout: .init(bytesPerRow: 256), destination: unsigned,
                             region: .init(width: 3, height: 2, mipLevel: 1)),
        .dispatchCompute(pipeline: compute, bindings: [.init(binding: 3): signedOut, .init(binding: 4): unsignedOut],
                         workgroups: .init(x: 1), textures: [.init(binding: 0): signedView, .init(binding: 1): unsignedView],
                         samplers: [.init(binding: 2): nearest]),
        .copy(source: signedOut, destination: read, byteCount: 32), .read(buffer: read, byteCount: 32),
        .copy(source: unsignedOut, destination: read, byteCount: 32), .read(buffer: read, byteCount: 32),
    ])
    let signedLanes: [Int32] = [-86, -76, -106, -116, -76, -66, -96, -106]
    let snapshots = try await ticket.wait()
    guard snapshots == [signedLanes.map(UInt32.init(bitPattern:)), [125, 85, 25, 65, 170, 180, 150, 140]] else {
        throw GPUError.backendFailure("Unexpected integer texture gather: \(snapshots)")
    }
    return snapshots.flatMap { $0 }
}

private func sampleCoverageDiagnostic(_ device: Device) async throws -> [UInt32] {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl("""
        struct VertexOutput {
            @builtin(position) position: vec4f,
            @location(0) @interpolate(linear, sample) marker: vec2f,
        }
        struct SampleInput {
            @builtin(sample_mask) coverage: u32,
            @location(0) @interpolate(linear, sample) marker: vec2f,
            @builtin(sample_index) index: u32,
        }
        struct SampleOutput {
            @builtin(sample_mask) coverage: u32,
            @location(0) color: vec4u,
        }
        @vertex fn vs(@location(0) xy: vec2f) -> VertexOutput {
            return VertexOutput(vec4f(xy, 0, 1), xy * vec2f(0.5, -0.5) + vec2f(0.5));
        }
        @fragment fn allSamples(@builtin(sample_index) index: u32,
                                @location(0) @interpolate(linear, sample) marker: vec2f) -> @location(0) vec4u {
            return vec4u(index + 1u, vec2u(marker * 128.0), 1u);
        }
        @fragment fn maskedSamples(input: SampleInput) -> SampleOutput {
            let covered = (input.coverage & (1u << input.index)) != 0u &&
                all(input.marker >= vec2f(0.0)) && all(input.marker <= vec2f(1.0));
            return SampleOutput(5u, vec4u(input.index + 1u, select(0u, 9u, covered), 0u, 1u));
        }
        """)
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .wgsl)
    let layouts = [VertexBufferLayout(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float32x2)])]
    let unmasked = try await device.createRenderPipeline(
        vertex: vertex, fragment: compiler.compileRenderStage(source, entryPoint: "allSamples", to: .wgsl),
        colorFormat: .rgba8Uint, vertexBuffers: layouts, sampleCount: 4, sampleMask: 0b1010)
    let masked = try await device.createRenderPipeline(
        vertex: vertex, fragment: compiler.compileRenderStage(source, entryPoint: "maskedSamples", to: .wgsl),
        colorFormat: .rgba8Uint, vertexBuffers: layouts, sampleCount: 4, sampleMask: 0b0110)
    guard case .spirv(let binary) = try compiler.compile(source, to: .spirv) else {
        throw GPUError.backendFailure("Expected binary SPIR-V for sample coverage")
    }
    let roundTrip = ShaderInput.spirv(binary)
    let importedVertex = try compiler.compileWebGPURenderStage(roundTrip, entryPoint: "vs")
    let importedAllSamples = try compiler.compileWebGPURenderStage(roundTrip, entryPoint: "allSamples")
    let importedMaskedSamples = try compiler.compileWebGPURenderStage(roundTrip, entryPoint: "maskedSamples")
    guard [importedVertex, importedAllSamples, importedMaskedSamples].allSatisfy({
        $0.varyings.count == 1 && $0.varyings[0].interpolation == .linear && $0.varyings[0].sampling == .sample
    }) else {
        throw GPUError.backendFailure("SPIR-V round trip lost per-sample interpolation")
    }
    let importedUnmasked = try await device.createRenderPipeline(
        vertex: importedVertex, fragment: importedAllSamples,
        colorFormat: .rgba8Uint, vertexBuffers: layouts, sampleCount: 4, sampleMask: 0b1010)
    let importedMasked = try await device.createRenderPipeline(
        vertex: importedVertex, fragment: importedMaskedSamples,
        colorFormat: .rgba8Uint, vertexBuffers: layouts, sampleCount: 4, sampleMask: 0b0110)
    let reader = try await device.createComputePipeline(compiler.compileCompute(.wgsl("""
        @group(0) @binding(0) var unmasked: texture_multisampled_2d<u32>;
        @group(0) @binding(1) var masked: texture_multisampled_2d<u32>;
        @group(0) @binding(2) var<storage, read_write> out: array<vec4u, 8>;
        @compute @workgroup_size(4) fn run(@builtin(local_invocation_index) lane: u32) {
            out[lane] = textureLoad(unmasked, vec2u(0), lane);
            out[lane + 4u] = textureLoad(masked, vec2u(0), lane);
        }
        """), to: .wgsl))
    let descriptor = TextureDescriptor(width: 1, height: 1, format: .rgba8Uint,
                                       usage: [.renderAttachment, .textureBinding], sampleCount: 4)
    let allSamples = try await device.createTexture(descriptor)
    let maskedSamples = try await device.createTexture(descriptor)
    let vertices = try await device.createBuffer(.init(size: 24, usage: [.vertex, .copyDestination]))
    let out = try await device.createBuffer(.init(size: 128, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 128, usage: [.copyDestination, .mapRead]))
    let clear = RenderClearColor(red: 0, green: 0, blue: 0, alpha: 0)
    var commands: [BufferCommand] = [
        .write(buffer: vertices, offset: 0, words: [Float(-1), -1, 3, -1, -1, 3].map(\.bitPattern)),
    ]
    for (allPipeline, maskedPipeline) in [(unmasked, masked), (importedUnmasked, importedMasked)] {
        commands += [
            .render(pipeline: allPipeline, target: allSamples, clear: clear, vertexCount: 3,
                    vertexBuffers: [.init(buffer: vertices)]),
            .render(pipeline: maskedPipeline, target: maskedSamples, clear: clear, vertexCount: 3,
                    vertexBuffers: [.init(buffer: vertices)]),
            .dispatchCompute(pipeline: reader, bindings: [.init(binding: 2): out], workgroups: .init(x: 1),
                             textures: [.init(binding: 0): allSamples, .init(binding: 1): maskedSamples]),
            .copy(source: out, destination: read, byteCount: 128), .read(buffer: read, byteCount: 128),
        ]
    }
    let snapshots = try await device.submitCommands(commands).wait()
    guard snapshots.map(\.count) == [32, 32], snapshots[0] == snapshots[1] else {
        throw GPUError.backendFailure("Unexpected fragment sample coverage readback: \(snapshots)")
    }
    return snapshots[0]
}

private func alphaCoverageDiagnostic(_ device: Device) async throws -> [UInt32] {
    let source = ShaderInput.wgsl("""
        struct FragmentOutput {
            @location(0) color: vec4f,
            @builtin(sample_mask) mask: u32,
        }
        @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
            if i == 0u { return vec4f(-1, -1, 0, 1); }
            if i == 1u { return vec4f(3, -1, 0, 1); }
            return vec4f(-1, 3, 0, 1);
        }
        @fragment fn zero() -> @location(0) vec4f { return vec4f(1, 0, 0, 0); }
        @fragment fn half() -> @location(0) vec4f { return vec4f(1, 0, 0, 0.5); }
        @fragment fn one() -> @location(0) vec4f { return vec4f(1, 0, 0, 1); }
        @fragment fn masked() -> FragmentOutput { return FragmentOutput(vec4f(1), 5u); }
        @fragment fn integer() -> @location(0) vec4u { return vec4u(1); }
        @fragment fn depthOnly() -> @builtin(frag_depth) f32 { return 0.25; }
        """)
    let compiler = IndependentShaderCompiler()
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .wgsl)
    var pipelines: [RenderPipeline] = []
    for (entry, mask) in [("zero", UInt32.max), ("half", UInt32.max), ("one", 0b1010), ("zero", 0b1010)] {
        let fragment = try compiler.compileRenderStage(source, entryPoint: entry, to: .wgsl)
        pipelines.append(try await device.createRenderPipeline(
            vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm,
            sampleCount: 4, sampleMask: mask, alphaToCoverageEnabled: true))
    }
    let reader = try await device.createComputePipeline(compiler.compileCompute(.wgsl("""
        @group(0) @binding(0) var image: texture_multisampled_2d<f32>;
        @group(0) @binding(1) var<storage, read_write> out: array<u32, 4>;
        @compute @workgroup_size(4) fn run(@builtin(local_invocation_index) lane: u32) {
            out[lane] = select(0u, 1u, textureLoad(image, vec2u(0), lane).r == 1.0);
        }
        """), to: .wgsl))
    let image = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm,
        usage: [.renderAttachment, .textureBinding], sampleCount: 4))
    let out = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 16, usage: [.copyDestination, .mapRead]))
    var commands: [BufferCommand] = []
    for pipeline in pipelines {
        commands += [
            .render(pipeline: pipeline, target: image, clear: .transparentBlack, vertexCount: 3),
            .dispatchCompute(pipeline: reader, bindings: [.init(binding: 1): out], workgroups: .init(x: 1),
                             textures: [.init(binding: 0): image]),
            .copy(source: out, destination: read, byteCount: 16), .read(buffer: read, byteCount: 16),
        ]
    }
    let onlyDepth = try compiler.compileRenderStage(source, entryPoint: "depthOnly", to: .wgsl)
    let noSamples = try await device.createRenderPipeline(
        vertex: vertex, fragment: onlyDepth, colorFormat: .rgba8Unorm,
        depthState: .init(compare: .always), sampleMask: 0)
    let singleColor = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm,
        usage: .renderAttachment))
    let singleDepth = try await device.createTexture(.init(width: 1, height: 1, format: .depth32Float,
        usage: [.renderAttachment, .copySource]))
    commands += [
        .render(pipeline: noSamples, target: singleColor, clear: .transparentBlack, vertexCount: 3,
                depth: .init(texture: singleDepth, loadAction: .clear(0.75))),
        .copyTextureToBuffer(source: singleDepth, destination: read, bytesPerRow: 256),
        .read(buffer: read, byteCount: 4),
    ]
    let ticket = try await device.submitCommands(commands)
    let snapshots = try await ticket.wait()
    guard snapshots.count == 5,
          snapshots[0] == [0, 0, 0, 0],
          snapshots[1].contains(0), snapshots[1].contains(1),
          snapshots[2] == [0, 1, 0, 1],
          snapshots[3] == [0, 0, 0, 0],
          snapshots[4] == [Float(0.75).bitPattern] else {
        throw GPUError.backendFailure("Unexpected alpha-to-coverage readback: \(snapshots)")
    }
    return snapshots.flatMap { $0 }
}

private func fragmentDepthDiagnostic(_ device: Device) async throws -> [UInt32] {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl("""
        enable wgpu_per_vertex;

        struct FragmentOut {
            @builtin(frag_depth) depth: f32,
            @builtin(sample_mask) mask: u32,
            @location(0) color: vec4f,
        }
        @vertex fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
            if index == 0u { return vec4f(-1, -1, 0.75, 1); }
            if index == 1u { return vec4f(3, -1, 0.75, 1); }
            return vec4f(-1, 3, 0.75, 1);
        }
        @fragment fn colorAndDepth() -> FragmentOut {
            return FragmentOut(0.25, 1u, vec4f(1, 0, 0, 1));
        }
        @fragment fn depthOnly() -> @builtin(frag_depth) f32 { return 2.0; }
        @fragment fn sparseColor() -> @location(2) vec4f { return vec4f(0, 1, 0, 1); }
        @fragment @early_depth_test(greater_equal)
        fn conservativeDepth(@builtin(position) point: vec4f) -> @builtin(frag_depth) f32 {
            return max(point.z, 0.875);
        }
        @fragment
        fn nativePerVertex(@location(0) @interpolate(per_vertex) values: array<f32, 3>) -> @location(0) vec4f {
            return vec4f(values[0], values[1], values[2], 1.0);
        }
        """)
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .wgsl)
    let combined = try compiler.compileRenderStage(source, entryPoint: "colorAndDepth", to: .wgsl)
    let onlyDepth = try compiler.compileRenderStage(source, entryPoint: "depthOnly", to: .wgsl)
    let writesColor = try await device.createRenderPipeline(
        vertex: vertex, fragment: combined, colorFormat: .rgba8Unorm, depthState: .init(compare: .less))
    let writesDepth = try await device.createRenderPipeline(
        vertex: vertex, fragment: onlyDepth, colorFormat: .rgba8Unorm, depthState: .init(compare: .always))
    let depthPass = try await device.createRenderPipeline(
        vertex: vertex, fragment: onlyDepth, colorFormat: nil, depthState: .init(compare: .always))
    let vertexOnly = try await device.createRenderPipeline(
        vertex: vertex, fragment: nil, colorFormat: nil, depthState: .init(compare: .always))
    let maskedVertexOnly = try await device.createRenderPipeline(
        vertex: vertex, fragment: nil, colorFormat: nil, depthState: .init(compare: .always), sampleMask: 0)
    let sparseFragment = try compiler.compileWebGPURenderStage(source, entryPoint: "sparseColor")
    let sparsePass = try await device.createRenderPipeline(vertex: vertex, fragment: sparseFragment, colorFormat: nil,
                                                          additionalColorTargets: [2: .init(format: .rgba8Unorm)])
    guard case .spirv(let binary) = try compiler.compile(source, to: .spirv) else {
        throw GPUError.backendFailure("Expected binary SPIR-V for fragment depth")
    }
    let roundTrip = ShaderInput.spirv(binary)
    for input in [source, roundTrip] {
        do {
            _ = try compiler.compileWebGPURenderStage(input, entryPoint: "nativePerVertex")
            throw GPUError.backendFailure("WebGPU accepted reachable per-vertex interpolation")
        } catch ShaderFailure.unsupported("Render code generation: WebGPU per-vertex interpolation") {}
    }
    let importedVertex = try compiler.compileWebGPURenderStage(roundTrip, entryPoint: "vs")
    let importedCombined = try compiler.compileWebGPURenderStage(roundTrip, entryPoint: "colorAndDepth")
    let importedDepthOnly = try compiler.compileWebGPURenderStage(roundTrip, entryPoint: "depthOnly")
    let conservativeDepth = try compiler.compileWebGPURenderStage(roundTrip, entryPoint: "conservativeDepth")
    guard importedCombined.earlyDepthTest == nil, importedDepthOnly.earlyDepthTest == nil,
          conservativeDepth.earlyDepthTest == .allow(.greaterEqual) else {
        throw GPUError.backendFailure("SPIR-V conservative depth mode was not isolated to its entry point")
    }
    let scope = try await device.pushErrorScope(.validation)
    do {
        _ = try await device.createRenderPipeline(
            vertex: importedVertex, fragment: conservativeDepth, colorFormat: nil, depthState: .init(compare: .always))
        throw GPUError.backendFailure("Browser accepted the native conservative depth mode")
    } catch GPUError.validation("Feature not advertised") {
        guard case .validation("Feature not advertised") = try await device.popErrorScope(scope) else {
            throw GPUError.backendFailure("Conservative depth feature failure did not reach the error scope")
        }
    }
    let importedColor = try await device.createRenderPipeline(
        vertex: importedVertex, fragment: importedCombined,
        colorFormat: .rgba8Unorm, depthState: .init(compare: .less))
    let importedDepth = try await device.createRenderPipeline(
        vertex: importedVertex, fragment: importedDepthOnly,
        colorFormat: nil, depthState: .init(compare: .always))
    let arguments = try await device.createBuffer(.init(size: 16, usage: [.indirect, .copyDestination]))
    let color = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm,
        usage: [.renderAttachment, .copySource]))
    let depth = try await device.createTexture(.init(width: 1, height: 1, format: .depth32Float,
        usage: [.renderAttachment, .copySource]))
    let read = try await device.createBuffer(.init(size: 4, usage: [.copyDestination, .mapRead]))
    let blue = RenderClearColor(red: 0, green: 0, blue: 1, alpha: 1)
    let ticket = try await device.submitCommands([
        .render(pipeline: writesColor, target: color, clear: blue, vertexCount: 3,
                depth: .init(texture: depth, loadAction: .clear(0.5))),
        .copyTextureToBuffer(source: color, destination: read, bytesPerRow: 256), .read(buffer: read, byteCount: 4),
        .copyTextureToBuffer(source: depth, destination: read, bytesPerRow: 256), .read(buffer: read, byteCount: 4),
        .render(pipeline: writesDepth, target: color, clear: blue, vertexCount: 3,
                depth: .init(texture: depth, loadAction: .load)),
        .copyTextureToBuffer(source: color, destination: read, bytesPerRow: 256), .read(buffer: read, byteCount: 4),
        .copyTextureToBuffer(source: depth, destination: read, bytesPerRow: 256), .read(buffer: read, byteCount: 4),
        .write(buffer: arguments, offset: 0, words: [3, 1, 0, 0]),
        .renderIndirect(pipeline: depthPass, target: nil, colorLoad: .load, indirectBuffer: arguments,
                        depth: .init(texture: depth, loadAction: .clear(0.125))),
        .copyTextureToBuffer(source: depth, destination: read, bytesPerRow: 256, aspect: .depthOnly),
        .read(buffer: read, byteCount: 4),
        .render(pipeline: sparsePass, target: nil, colorLoad: .load, vertexCount: 3,
                additionalColors: [2: .init(texture: color, loadAction: .clear(.transparentBlack))]),
        .copyTextureToBuffer(source: color, destination: read, bytesPerRow: 256), .read(buffer: read, byteCount: 4),
        .render(pipeline: vertexOnly, target: nil, colorLoad: .load, vertexCount: 3,
                depth: .init(texture: depth, loadAction: .clear(0.125))),
        .copyTextureToBuffer(source: depth, destination: read, bytesPerRow: 256, aspect: .depthOnly),
        .read(buffer: read, byteCount: 4),
        .render(pipeline: maskedVertexOnly, target: nil, colorLoad: .load, vertexCount: 3,
                depth: .init(texture: depth, loadAction: .clear(0.125))),
        .copyTextureToBuffer(source: depth, destination: read, bytesPerRow: 256, aspect: .depthOnly),
        .read(buffer: read, byteCount: 4),
        .render(pipeline: importedColor, target: color, clear: blue, vertexCount: 3,
                depth: .init(texture: depth, loadAction: .clear(0.5))),
        .copyTextureToBuffer(source: color, destination: read, bytesPerRow: 256), .read(buffer: read, byteCount: 4),
        .copyTextureToBuffer(source: depth, destination: read, bytesPerRow: 256), .read(buffer: read, byteCount: 4),
        .renderIndirect(pipeline: importedDepth, target: nil, colorLoad: .load, indirectBuffer: arguments,
                        depth: .init(texture: depth, loadAction: .clear(0.125))),
        .copyTextureToBuffer(source: depth, destination: read, bytesPerRow: 256, aspect: .depthOnly),
        .read(buffer: read, byteCount: 4),
    ])
    let expected: [[UInt32]] = [
        [0xff0000ff], [Float(0.25).bitPattern], [0xffff0000], [Float(1).bitPattern], [Float(1).bitPattern], [0xff00ff00],
        [Float(0.75).bitPattern], [Float(0.125).bitPattern],
    ]
    let snapshots = try await ticket.wait()
    guard snapshots == expected + [expected[0], expected[1], expected[4]] else {
        throw GPUError.backendFailure("Unexpected fragment depth readback: \(snapshots)")
    }
    return snapshots.prefix(expected.count).flatMap { $0 }
}

private func fragmentRasterDiagnostic(_ device: Device) async throws -> [UInt32] {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl("""
        struct RasterInput {
            @builtin(position) pixel: vec4f,
        }
        @vertex fn vs(@location(0) clip: vec2f) -> @builtin(position) vec4f {
            return vec4f(clip, 0.5, 2.0);
        }
        @fragment fn fs(input: RasterInput, @builtin(front_facing) front: bool) -> @location(0) vec4u {
            let p = input.pixel;
            let x = select(0u, 1u, p.x == 0.5) + select(0u, 2u, p.x == 1.5);
            let y = select(0u, 1u, p.y == 0.5) + select(0u, 2u, p.y == 1.5);
            let zw = select(0u, 37u, p.z == 0.25) + select(0u, 64u, p.w == 0.5);
            return vec4u(x, y, zw, select(0u, 1u, front));
        }
        """)
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .wgsl)
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: .wgsl)
    let layouts = [VertexBufferLayout(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float32x2)])]
    let defaults = try await device.createRenderPipeline(
        vertex: vertex, fragment: fragment, colorFormat: .rgba8Uint, vertexBuffers: layouts)
    let clockwise = try await device.createRenderPipeline(
        vertex: vertex, fragment: fragment, colorFormat: .rgba8Uint, vertexBuffers: layouts,
        primitive: .init(frontFace: .clockwise, cullMode: .back))
    let cullFront = try await device.createRenderPipeline(
        vertex: vertex, fragment: fragment, colorFormat: .rgba8Uint, vertexBuffers: layouts,
        primitive: .init(cullMode: .front))
    let texture = try await device.createTexture(.init(width: 2, height: 2, format: .rgba8Uint,
        usage: [.renderAttachment, .copySource]))
    let vertices = try await device.createBuffer(.init(size: 48, usage: [.vertex, .copyDestination]))
    let read = try await device.createBuffer(.init(size: 264, usage: [.copyDestination, .mapRead]))
    let clear = RenderClearColor(red: 0, green: 0, blue: 0, alpha: 0)
    let triangles: [Float] = [-2, -2, 6, -2, -2, 6, -2, -2, -2, 6, 6, -2]
    var commands: [BufferCommand] = [.write(buffer: vertices, offset: 0, words: triangles.map(\.bitPattern))]
    for (pipeline, firstVertex) in [(defaults, 0), (defaults, 3), (clockwise, 0), (clockwise, 3), (cullFront, 0)] {
        commands += [
            .render(pipeline: pipeline, target: texture, clear: clear, vertexCount: 3,
                    vertexBuffers: [.init(buffer: vertices)], firstVertex: UInt32(firstVertex)),
            .copyTextureToBuffer(source: texture, destination: read, bytesPerRow: 256),
            .read(buffer: read, byteCount: 264),
        ]
    }
    let locationVertex = try compiler.compileWebGPURenderStage(.spirv(vertexInterfaceSPIRV()), entryPoint: "main")
    let locationFragment = try compiler.compileWebGPURenderStage(.spirv(fragmentLocationsSPIRV()), entryPoint: "main")
    let locationPipeline = try await device.createRenderPipeline(vertex: locationVertex, fragment: locationFragment,
                                                                 colorFormat: .rgba8Unorm)
    let locationTexture = try await device.createTexture(.init(width: 2, height: 2, format: .rgba8Unorm,
                                                               usage: [.renderAttachment, .copySource]))
    let roundTripSource = ShaderInput.wgsl("""
        struct RasterOutput {
            @builtin(position) position: vec4f,
            @location(2) uv: vec2f,
        }
        @vertex fn round_trip_vs(@location(0) clip: vec2f, @builtin(vertex_index) index: u32) -> RasterOutput {
            let uv = vec2f(select(0.0, 2.0, index == 1u), select(0.0, 2.0, index == 2u));
            return RasterOutput(vec4f(clip.x, -clip.y, 0.0, 2.0), uv);
        }
        @fragment fn round_trip_fs(@location(2) uv: vec2f) -> @location(0) vec4f {
            return vec4f(select(0.0, 1.0, uv.x > 0.5), select(0.0, 1.0, uv.y < 0.5), 1.0, 1.0);
        }
        """)
    guard case .spirv(let words) = try compiler.compile(roundTripSource, to: .spirv) else {
        throw GPUError.backendFailure("Expected binary SPIR-V for render stages")
    }
    let roundTripVertex = try compiler.compileWebGPURenderStage(.spirv(words), entryPoint: "round_trip_vs")
    let roundTripFragment = try compiler.compileWebGPURenderStage(.spirv(words), entryPoint: "round_trip_fs")
    let roundTrip = try await device.createRenderPipeline(
        vertex: roundTripVertex, fragment: roundTripFragment, colorFormat: .rgba8Unorm, vertexBuffers: layouts)
    commands += [
        .render(pipeline: locationPipeline, target: locationTexture, clear: clear, vertexCount: 3),
        .copyTextureToBuffer(source: locationTexture, destination: read, bytesPerRow: 256),
        .read(buffer: read, byteCount: 264),
        .render(pipeline: roundTrip, target: locationTexture, clear: clear, vertexCount: 3,
                vertexBuffers: [.init(buffer: vertices)]),
        .copyTextureToBuffer(source: locationTexture, destination: read, bytesPerRow: 256),
        .read(buffer: read, byteCount: 264),
    ]
    let ticket = try await device.submitCommands(commands)
    let snapshots = try await ticket.wait()
    guard snapshots.count == 7, snapshots.allSatisfy({ $0.count == 66 }),
          snapshots[5][0] == 0xff004039, snapshots[5][65] == 0 else {
        throw GPUError.backendFailure("Unexpected fragment raster readback size")
    }
    let locationWords = snapshots[5]
    let pixels = snapshots.prefix(5).map { [$0[0], $0[1], $0[64], $0[65]] }
    let front: [UInt32] = [0x01650101, 0x01650102, 0x01650201, 0x01650202]
    let back = front.map { $0 & 0x00ff_ffff }
    let culled: [UInt32] = [0, 0, 0, 0]
    guard pixels == [front, back, culled, front, culled] else {
        throw GPUError.backendFailure("Unexpected fragment raster inputs: \(pixels)")
    }
    let roundTripPixels = [0, 1, 64, 65].map { snapshots[6][$0] }
    guard roundTripPixels == [0xffffff00, 0xffffffff, 0xffff0000, 0xffff00ff] else {
        throw GPUError.backendFailure("Unexpected render round-trip pixels: \(roundTripPixels)")
    }
    return pixels.flatMap { $0 } + [locationWords[0], locationWords[65]] + roundTripPixels
}

private func vertexInterfaceSPIRV() -> [UInt32] {
    func op(_ code: UInt32, _ operands: UInt32...) -> [UInt32] {
        [UInt32(operands.count + 1) << 16 | code] + operands
    }
    var words: [UInt32] = [0x07230203, 0x00010300, 0, 100, 0]
    for instruction in [
        op(17, 1), op(14, 0, 1), op(15, 0, 10, 0x6e69616d, 0, 11, 12, 13),
        op(71, 11, 11, 42), op(71, 13, 30, 1), op(71, 13, 14),
        op(71, 8, 2), op(72, 8, 0, 11, 0), op(72, 8, 1, 11, 1),
        op(72, 8, 2, 11, 3), op(72, 8, 3, 11, 4),
        op(19, 1), op(20, 2), op(21, 3, 32, 1), op(22, 4, 32), op(23, 5, 4, 4),
        op(32, 6, 1, 3), op(32, 7, 3, 5), op(32, 9, 3, 4), op(33, 14, 1),
        op(43, 3, 20, 0), op(43, 3, 21, 1), op(43, 3, 22, 2),
        op(43, 4, 23, Float(-1).bitPattern), op(43, 4, 24, Float(1).bitPattern),
        op(43, 4, 25, 0), op(43, 4, 26, Float(90).bitPattern),
        op(28, 15, 4, 21), op(30, 8, 5, 4, 15, 15), op(32, 16, 3, 8),
        op(59, 6, 11, 1), op(59, 16, 12, 3), op(59, 9, 13, 3),
        op(54, 1, 10, 0, 14), op(248, 30), op(61, 3, 31, 11),
        op(170, 2, 32, 31, 21), op(170, 2, 33, 31, 22),
        op(169, 4, 34, 32, 24, 23), op(169, 4, 35, 33, 24, 23),
        op(80, 5, 36, 34, 35, 25, 24), op(65, 7, 37, 12, 20),
        op(62, 37, 36), op(62, 13, 26), op(253), op(56),
    ] { words += instruction }
    return words
}

private func fragmentLocationsSPIRV() -> [UInt32] {
    func op(_ code: UInt32, _ operands: UInt32...) -> [UInt32] {
        [UInt32(operands.count + 1) << 16 | code] + operands
    }
    var words: [UInt32] = [0x07230203, 0x00010300, 0, 100, 0]
    for instruction in [
        op(17, 1), op(11, 80, 0x4c534c47, 0x6474732e, 0x3035342e, 0), op(14, 0, 1),
        op(15, 4, 10, 0x6e69616d, 0, 12, 11), op(16, 10, 7), op(71, 11, 30, 1), op(71, 11, 14), op(71, 12, 30, 0),
        op(19, 1), op(22, 2, 32), op(23, 3, 2, 4), op(32, 4, 1, 2), op(32, 5, 3, 3), op(33, 6, 1), op(33, 7, 2),
        op(43, 2, 20, Float(400).bitPattern), op(43, 2, 21, Float(0.25).bitPattern),
        op(43, 2, 22, 0), op(43, 2, 23, Float(1).bitPattern), op(59, 4, 11, 1), op(59, 5, 12, 3),
        op(54, 1, 10, 0, 6), op(248, 30), op(57, 2, 33, 14),
        op(136, 2, 34, 33, 20), op(80, 3, 35, 34, 21, 22, 23), op(62, 12, 35), op(253), op(56),
        op(54, 2, 14, 0, 7), op(248, 39), op(61, 2, 31, 11),
        op(12, 2, 32, 80, 11, 31), op(12, 2, 36, 80, 12, 32), op(254, 36), op(56),
    ] { words += instruction }
    return words
}

private func interpolationDiagnostic(_ device: Device) async throws -> UInt32 {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl("""
        struct VertexOutput {
            @builtin(position) position: vec4f,
            @location(0) @interpolate(linear) affine: f32,
            @location(1) smooth: f32,
            @location(2) @interpolate(flat) signed: vec2i,
            @location(3) @interpolate(flat, either) unsigned: u32,
        }
        fn vertex(position: vec4f, value: f32, tag: vec2i, code: u32) -> VertexOutput {
            return VertexOutput(position, value, value, tag, code);
        }
        @vertex fn vs(@builtin(vertex_index) index: u32) -> VertexOutput {
            if index == 0u { return vertex(vec4f(-1, 1, 0.25, 1), 0.0, vec2i(-17, 5), 249u); }
            if index == 1u { return vertex(vec4f(6, 2, 0.5, 2), 1.0, vec2i(1, 7), 11u); }
            return vertex(vec4f(-4, -12, 1, 4), 0.0, vec2i(3, 9), 200u);
        }
        @fragment fn fs(
            @location(0) @interpolate(linear) affine: f32,
            @location(1) smooth: f32,
            @location(2) @interpolate(flat, first) signed: vec2i,
            @location(3) @interpolate(flat, either) unsigned: u32,
        ) -> @location(0) vec4u {
            return vec4u(select(0u, 21u, affine > 0.24 && affine < 0.26),
                         select(0u, 42u, smooth > 0.18 && smooth < 0.19),
                         select(0u, 63u, signed.x == -17 && signed.y == 5),
                         select(0u, 84u, unsigned == 249u || unsigned == 200u));
        }
        """)
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .wgsl)
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: .wgsl)
    let pipeline = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Uint)
    let target = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Uint,
        usage: [.renderAttachment, .copySource]))
    let read = try await device.createBuffer(.init(size: 4, usage: [.copyDestination, .mapRead]))
    let ticket = try await device.submitCommands([
        .render(pipeline: pipeline, target: target, clear: .init(red: 0, green: 0, blue: 0, alpha: 0), vertexCount: 3),
        .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256), .read(buffer: read, byteCount: 4),
    ])
    let snapshots = try await ticket.wait()
    guard snapshots == [[0x543f2a15]] else {
        throw GPUError.backendFailure("Unexpected interstage interpolation readback: \(snapshots)")
    }
    return snapshots[0][0]
}

private func topologyDiagnostic(_ device: Device) async throws -> [UInt32] {
    let source = ShaderInput.wgsl("""
        @vertex fn vs(@location(0) p: vec2f) -> @builtin(position) vec4f { return vec4f(p, 0, 1); }
        @fragment fn fs() -> @location(0) vec4f { return vec4f(0, 1, 0, 1); }
        """)
    let compiler = IndependentShaderCompiler()
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .wgsl)
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: .wgsl)
    let layouts = [VertexBufferLayout(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float32x2)])]
    let states: [RenderPrimitiveState] = [
        .init(topology: .pointList), .init(topology: .lineList), .init(topology: .lineStrip),
        .init(topology: .triangleStrip), .init(topology: .triangleStrip, stripIndexFormat: .uint16),
        .init(topology: .lineStrip, stripIndexFormat: .uint32),
    ]
    var pipelines: [RenderPipeline] = []
    for primitive in states {
        pipelines.append(try await device.createRenderPipeline(
            vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm, vertexBuffers: layouts, primitive: primitive))
    }
    let target = try await device.createTexture(.init(width: 8, height: 4, format: .rgba8Unorm,
        usage: [.renderAttachment, .copySource]))
    let vertices = try await device.createBuffer(.init(size: 176, usage: [.vertex, .copyDestination]))
    let short = try await device.createBuffer(.init(size: 20, usage: [.index, .copyDestination]))
    let long = try await device.createBuffer(.init(size: 20, usage: [.index, .copyDestination]))
    let read = try await device.createBuffer(.init(size: 800, usage: [.copyDestination, .mapRead]))
    let positions: [(Float, Float)] = [
        (0.5, 0.5), (7.5, 3.5),
        (1.5, 0.5), (6.5, 0.5), (6.5, 3.5), (1.5, 3.5),
        (1, 0), (1, 4), (7, 0), (7, 4),
        (0, 0), (0, 4), (2, 0), (2, 4), (6, 0), (6, 4), (8, 0), (8, 4),
        (0.5, 1.5), (2.5, 1.5), (5.5, 1.5), (7.5, 1.5),
    ]
    let clear = RenderClearColor(red: 0, green: 0, blue: 0, alpha: 1)
    var commands: [BufferCommand] = [
        .write(buffer: vertices, offset: 0, words: positions.flatMap { x, y in
            [(x / 4 - 1).bitPattern, (1 - y / 2).bitPattern]
        }),
        .write(buffer: short, offset: 0, words: [0x00010009, 0x00030002, 0xffff0004, 0x00060005, 0x00080007]),
        .write(buffer: long, offset: 0, words: [0, 1, .max, 2, 3]),
    ]
    let draws: [BufferCommand] = [
        .render(pipeline: pipelines[0], target: target, clear: clear, vertexCount: 2, vertexBuffers: [.init(buffer: vertices)]),
        .render(pipeline: pipelines[1], target: target, clear: clear, vertexCount: 4, vertexBuffers: [.init(buffer: vertices)], firstVertex: 2),
        .render(pipeline: pipelines[2], target: target, clear: clear, vertexCount: 4, vertexBuffers: [.init(buffer: vertices)], firstVertex: 2),
        .render(pipeline: pipelines[3], target: target, clear: clear, vertexCount: 4, vertexBuffers: [.init(buffer: vertices)], firstVertex: 6),
        .renderIndexed(pipeline: pipelines[4], target: target, clear: clear, indexBuffer: .init(buffer: short, format: .uint16),
                       draw: .init(indexCount: 9, firstIndex: 1, baseVertex: -1), vertexBuffers: [.init(buffer: vertices, offset: 80)]),
        .renderIndexed(pipeline: pipelines[5], target: target, clear: clear, indexBuffer: .init(buffer: long, format: .uint32),
                       draw: .init(indexCount: 5), vertexBuffers: [.init(buffer: vertices, offset: 144)]),
    ]
    for draw in draws {
        commands += [draw, .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256),
                     .read(buffer: read, byteCount: 800)]
    }
    let ticket = try await device.submitCommands(commands)
    let snapshots = try await ticket.wait()
    guard snapshots.count == 6, snapshots.allSatisfy({ $0.count == 200 }) else {
        throw GPUError.backendFailure("Unexpected primitive topology readback size")
    }
    let pixels = [
        snapshots[0][0], snapshots[0][199], snapshots[0][68],
        snapshots[1][3], snapshots[1][70], snapshots[1][195],
        snapshots[2][3], snapshots[2][70], snapshots[2][195],
        snapshots[3][3], snapshots[3][195], snapshots[3][64],
        snapshots[4][64], snapshots[4][68], snapshots[4][71],
        snapshots[5][65], snapshots[5][68], snapshots[5][70],
    ]
    let green: UInt32 = 0xff00ff00
    let black: UInt32 = 0xff000000
    let pointPixels = (0..<4).flatMap { row in snapshots[0][(row * 64)..<(row * 64 + 8)] }
    guard pointPixels.filter({ $0 == green }).count == 2,
          pixels == [green, green, black, green, black, green, green, green, green,
                     green, green, black, green, black, green, green, black, green] else {
        throw GPUError.backendFailure("Unexpected primitive topology readback: \(pixels)")
    }
    return pixels
}

private func blendDiagnostic(_ device: Device) async throws -> [UInt32] {
    let source = ShaderInput.wgsl("""
        @vertex fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
            if index == 0u { return vec4f(-1, -1, 0, 1); }
            if index == 1u { return vec4f(3, -1, 0, 1); }
            return vec4f(-1, 3, 0, 1);
        }
        @fragment fn base() -> @location(0) vec4f { return vec4f(0, 0.5, 1, 1); }
        @fragment fn overlay() -> @location(0) vec4f { return vec4f(1, 0, 0, 0.5); }
        @fragment fn integerColor() -> @location(0) vec4u { return vec4u(1); }
        """)
    let compiler = IndependentShaderCompiler()
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .wgsl)
    let base = try compiler.compileRenderStage(source, entryPoint: "base", to: .wgsl)
    let overlay = try compiler.compileRenderStage(source, entryPoint: "overlay", to: .wgsl)
    let paint = try await device.createRenderPipeline(vertex: vertex, fragment: base, colorFormat: .rgba8Unorm)
    let blend = try await device.createRenderPipeline(
        vertex: vertex, fragment: overlay, colorFormat: .rgba8Unorm,
        colorState: .init(blend: .init(color: .init(sourceFactor: .sourceAlpha, destinationFactor: .oneMinusSourceAlpha)),
                          writeMask: [.red, .green, .alpha]))
    let constantBlend = try await device.createRenderPipeline(
        vertex: vertex, fragment: overlay, colorFormat: .rgba8Unorm,
        colorState: .init(blend: .init(color: .init(operation: .reverseSubtract, sourceFactor: .constant, destinationFactor: .one),
                                      alpha: .init(operation: .min, sourceFactor: .one, destinationFactor: .one))))
    let texture = try await device.createTexture(.init(width: 2, height: 2, format: .rgba8Unorm,
        usage: [.renderAttachment, .copySource], mipLevelCount: 2))
    let view = try await device.createTextureView(texture: texture, .init(baseMipLevel: 1))
    let indices = try await device.createBuffer(.init(size: 12, usage: [.index, .copyDestination]))
    let read = try await device.createBuffer(.init(size: 4, usage: [.copyDestination, .mapRead]))
    let ticket = try await device.submitCommands([
        .render(pipeline: paint, target: view, colorLoad: .load, vertexCount: 0),
        .copyTextureToBuffer(source: texture, destination: read, bytesPerRow: 256, mipLevel: 1),
        .read(buffer: read, byteCount: 4),
        .render(pipeline: paint, target: view, clear: .init(red: 0, green: 0, blue: 0, alpha: 0), vertexCount: 3),
        .write(buffer: indices, offset: 0, words: [0, 1, 2]),
        .renderIndexed(pipeline: blend, target: view, colorLoad: .load,
                       indexBuffer: .init(buffer: indices, format: .uint32), draw: .init(indexCount: 3)),
        .copyTextureToBuffer(source: texture, destination: read, bytesPerRow: 256, mipLevel: 1),
        .read(buffer: read, byteCount: 4),
        .render(pipeline: constantBlend, target: view, clear: .init(red: 1, green: 0.5, blue: 0.25, alpha: 1),
                vertexCount: 3, blendConstant: .init(red: 0.25, green: 0, blue: 0, alpha: 0)),
        .copyTextureToBuffer(source: texture, destination: read, bytesPerRow: 256, mipLevel: 1),
        .read(buffer: read, byteCount: 4),
    ])
    let snapshots = try await ticket.wait()
    guard snapshots == [[0], [0x80ff4080], [0x804080bf]] else {
        throw GPUError.backendFailure("Unexpected color blend readback: \(snapshots)")
    }
    return snapshots.flatMap { $0 }
}

private func multipleColorDiagnostic(_ device: Device) async throws -> [UInt32] {
    let source = ShaderInput.wgsl("""
        @group(0) @binding(0) var<storage, read_write> calls: atomic<u32>;
        struct FragmentOutput {
            @location(2) tag: vec4u,
            @location(0) color: vec4f,
        }
        struct FloatOutput {
            @location(0) first: vec4f,
            @location(2) second: vec4f,
        }
        @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
            if i == 0u { return vec4f(-1, -1, 0, 1); }
            if i == 1u { return vec4f(3, -1, 0, 1); }
            return vec4f(-1, 3, 0, 1);
        }
        @fragment fn fs() -> FragmentOutput { return FragmentOutput(vec4u(17, 29, 33, 49), vec4f(1, 0, 0, 1)); }
        @fragment fn bothFloat() -> FloatOutput { return FloatOutput(vec4f(0), vec4f(1)); }
        fn extraColor() -> vec4u { return vec4u(atomicAdd(&calls, 1u), 11u, 13u, 255u); }
        @fragment fn unboundColor() -> FragmentOutput { return FragmentOutput(extraColor(), vec4f(0, 1, 0, 1)); }
        @fragment fn onlyUnboundColor() -> @location(2) vec4u { return extraColor(); }
        """)
    let compiler = IndependentShaderCompiler()
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .wgsl)
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: .wgsl)
    let pipeline = try await device.createRenderPipeline(
        vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm,
        additionalColorTargets: [2: .init(format: .rgba8Uint, colorState: .init(writeMask: [.red, .blue]))])
    let primary = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm,
        usage: [.renderAttachment, .copySource]))
    let tag = try await device.createTexture(.init(width: 2, height: 2, format: .rgba8Uint,
        usage: [.renderAttachment, .copySource], mipLevelCount: 2))
    let view = try await device.createTextureView(texture: tag, .init(baseMipLevel: 1))
    let indices = try await device.createBuffer(.init(size: 12, usage: [.index, .copyDestination]))
    let read = try await device.createBuffer(.init(size: 16, usage: [.copyDestination, .mapRead]))
    let colors: [UInt32: RenderColorAttachment] = [
        2: .init(texture: view, loadAction: .clear(.init(red: 0, green: 5, blue: 0, alpha: 7))),
    ]
    let loaded: [UInt32: RenderColorAttachment] = [2: .init(texture: view)]
    let draw = BufferCommand.renderIndexed(
        pipeline: pipeline, target: primary, clear: .transparentBlack,
        indexBuffer: .init(buffer: indices, format: .uint32), draw: .init(indexCount: 3), additionalColors: loaded)
    let floatFragment = try compiler.compileRenderStage(source, entryPoint: "bothFloat", to: .wgsl)
    let constant = RenderBlendComponent(sourceFactor: .constant)
    let multisampled = try await device.createRenderPipeline(
        vertex: vertex, fragment: floatFragment, colorFormat: .rgba8Unorm, sampleCount: 4,
        additionalColorTargets: [2: .init(format: .rgba8Unorm,
            colorState: .init(blend: .init(color: constant, alpha: constant)))])
    let multisample = TextureDescriptor(width: 1, height: 1, format: .rgba8Unorm, usage: .renderAttachment, sampleCount: 4)
    let msPrimary = try await device.createTexture(multisample)
    let msSecondary = try await device.createTexture(multisample)
    let limits = device.capabilities
    let targetCount = Int(min(limits.maxColorAttachments, limits.maxColorAttachmentBytesPerSample / 16))
    guard targetCount > 0 else { throw GPUError.backendFailure("No Float32 color attachments fit the device limit") }
    let fields = (0..<targetCount).map { "@location(\($0)) color\($0): vec4f," }.joined(separator: "\n")
    let colorsAtLimit = (0..<targetCount).map { "vec4f(\($0).25, 0.5, 0.75, 1)" }.joined(separator: ", ")
    let wideFragment = try compiler.compileWebGPURenderStage(.wgsl("""
        struct Outputs { \(fields) }
        @fragment fn fs() -> Outputs { return Outputs(\(colorsAtLimit)); }
        """), entryPoint: "fs")
    let formats = Dictionary(uniqueKeysWithValues: (UInt32(1)..<UInt32(targetCount)).map {
        ($0, RenderColorTarget(format: .rgba32Float))
    })
    let widePipeline = try await device.createRenderPipeline(vertex: vertex, fragment: wideFragment, colorFormat: .rgba32Float,
                                                              additionalColorTargets: formats)
    var outOfRange = formats
    outOfRange[limits.maxColorAttachments] = .init(format: .rgba32Float)
    do {
        _ = try await device.createRenderPipeline(vertex: vertex, fragment: wideFragment, colorFormat: .rgba32Float,
                                                   additionalColorTargets: outOfRange)
        throw GPUError.backendFailure("Color target accepted beyond the negotiated slot limit")
    } catch GPUError.validation("Color target location exceeds the device attachment limit") {}
    var wideTargets: [ResourceID] = []
    for _ in 0..<targetCount {
        wideTargets.append(try await device.createTexture(.init(width: 1, height: 1, format: .rgba32Float,
            usage: [.renderAttachment, .copySource])))
    }
    let wideAttachments = Dictionary(uniqueKeysWithValues: wideTargets.dropFirst().enumerated().map { index, texture in
        (UInt32(index + 1), RenderColorAttachment(texture: texture, loadAction: .clear(.transparentBlack)))
    })
    let unboundFragment = try compiler.compileWebGPURenderStage(source, entryPoint: "unboundColor")
    let unbound = try await device.createRenderPipeline(vertex: vertex, fragment: unboundFragment, colorFormat: .rgba8Unorm)
    let onlyUnbound = try compiler.compileWebGPURenderStage(source, entryPoint: "onlyUnboundColor")
    let noWrites = try await device.createRenderPipeline(vertex: vertex, fragment: onlyUnbound, colorFormat: .rgba8Unorm)
    let outOfRangeFragment = try compiler.compileWebGPURenderStage(.wgsl("""
        @fragment fn fs() -> @location(\(limits.maxColorAttachments)u) vec4f { return vec4f(1); }
        """), entryPoint: "fs")
    do {
        _ = try await device.createRenderPipeline(vertex: vertex, fragment: outOfRangeFragment, colorFormat: .rgba8Unorm)
        throw GPUError.backendFailure("Ignored color output accepted beyond the device location limit")
    } catch GPUError.validation("Fragment output location exceeds the device attachment limit") {}
    let calls = try await device.createBuffer(.init(size: 4, usage: [.storage, .copySource]))
    let callBinding = [ComputeBindingSlot(binding: 0): ComputeBufferBinding(buffer: calls)]
    var commands: [BufferCommand] = [
        .write(buffer: indices, offset: 0, words: [0, 1, 2]),
        .render(pipeline: pipeline, target: primary, clear: .transparentBlack,
                vertexCount: 0, additionalColors: colors),
        draw,
        .copyTextureToBuffer(source: primary, destination: read, bytesPerRow: 256), .read(buffer: read, byteCount: 4),
        .copyTextureToBuffer(source: tag, destination: read, bytesPerRow: 256, mipLevel: 1),
        .read(buffer: read, byteCount: 4),
        .render(pipeline: multisampled, target: msPrimary, clear: .transparentBlack, vertexCount: 3,
                blendConstant: .init(red: 0.25, green: 0.5, blue: 1, alpha: 1),
                additionalColors: [2: .init(texture: msSecondary, resolve: .init(texture: primary))]),
        .copyTextureToBuffer(source: primary, destination: read, bytesPerRow: 256), .read(buffer: read, byteCount: 4),
    ]
    commands.append(.render(pipeline: widePipeline, target: wideTargets[0], clear: .transparentBlack,
                            vertexCount: 3, additionalColors: wideAttachments))
    for texture in wideTargets {
        commands.append(.copyTextureToBuffer(source: texture, destination: read, bytesPerRow: 256))
        commands.append(.read(buffer: read, byteCount: 16))
    }
    for sparsePipeline in [unbound, noWrites] {
        commands += [
            .render(pipeline: sparsePipeline, target: primary, clear: .transparentBlack, vertexCount: 3,
                    storageBuffers: callBinding),
            .copyTextureToBuffer(source: primary, destination: read, bytesPerRow: 256),
            .read(buffer: read, byteCount: 4),
            .copy(source: calls, destination: read, byteCount: 4), .read(buffer: read, byteCount: 4),
        ]
    }
    let snapshots = try await device.submitCommands(commands).wait()
    let expectedWide = (0..<targetCount).map { [Float($0) + 0.25, 0.5, 0.75, 1].map(\.bitPattern) }
    let sparseExpected: [[UInt32]] = [[0xff00ff00], [1], [0], [2]]
    guard snapshots == [[0xff0000ff], [0x07210511], [0xffff8040]] + expectedWide + sparseExpected else {
        throw GPUError.backendFailure("Unexpected multiple color attachment readback: \(snapshots)")
    }
    return snapshots.flatMap { $0 }
}

private func viewportDiagnostic(_ device: Device) async throws -> [UInt32] {
    let source = ShaderInput.wgsl("""
        @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
            if i == 0u { return vec4f(-1, -1, 0.25, 1); }
            if i == 1u { return vec4f(3, -1, 0.25, 1); }
            return vec4f(-1, 3, 0.25, 1);
        }
        @fragment fn color(@builtin(position) pixel: vec4f) -> @location(0) vec4f {
            return select(vec4f(1, 0, 0, 1), vec4f(0, 1, 0, 1), pixel.z == 0.625);
        }
        @fragment fn depth() -> @builtin(frag_depth) f32 { return 0.1; }
        """)
    let compiler = IndependentShaderCompiler()
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .wgsl)
    let color = try await device.createRenderPipeline(vertex: vertex,
        fragment: compiler.compileRenderStage(source, entryPoint: "color", to: .wgsl), colorFormat: .rgba8Unorm)
    let depth = try await device.createRenderPipeline(vertex: vertex,
        fragment: compiler.compileRenderStage(source, entryPoint: "depth", to: .wgsl), colorFormat: .rgba8Unorm,
        depthState: .init(compare: .less))
    let target = try await device.createTexture(.init(width: 8, height: 8, format: .rgba8Unorm,
        usage: [.renderAttachment, .copySource], mipLevelCount: 2))
    let view = try await device.createTextureView(texture: target, .init(baseMipLevel: 1))
    let depthTarget = try await device.createTexture(.init(width: 4, height: 4, format: .depth32Float,
        usage: [.renderAttachment, .copySource]))
    let indices = try await device.createBuffer(.init(size: 12, usage: [.index, .copyDestination]))
    let read = try await device.createBuffer(.init(size: 784, usage: [.copyDestination, .mapRead]))
    let raster = RenderRasterState(viewport: .init(x: 1, y: 1, width: 3, height: 2, minDepth: 0.5),
                                  scissor: .init(x: 0, y: 2, width: 3, height: 2))
    let ticket = try await device.submitCommands([
        .write(buffer: indices, offset: 0, words: [0, 1, 2]),
        .render(pipeline: color, target: view, clear: .transparentBlack, vertexCount: 3, raster: raster),
        .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256, mipLevel: 1),
        .read(buffer: read, byteCount: 784),
        .renderIndexed(pipeline: depth, target: view, colorLoad: .load, indexBuffer: .init(buffer: indices, format: .uint32),
                       draw: .init(indexCount: 3), depth: .init(texture: depthTarget, loadAction: .clear(0.75)), raster: raster),
        .copyTextureToBuffer(source: depthTarget, destination: read, bytesPerRow: 256), .read(buffer: read, byteCount: 784),
        .render(pipeline: color, target: view, clear: .init(red: 0, green: 0, blue: 1, alpha: 1), vertexCount: 3,
                raster: .init(scissor: .init(x: 4, y: 4, width: 0, height: 0))),
        .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256, mipLevel: 1),
        .read(buffer: read, byteCount: 784),
    ])
    let snapshots = try await ticket.wait()
    guard snapshots.count == 3, snapshots.allSatisfy({ $0.count == 196 }) else {
        throw GPUError.backendFailure("Unexpected viewport readback size")
    }
    var words: [UInt32] = []
    for y in 0..<4 {
        for x in 0..<4 {
            let inside = y == 2 && (x == 1 || x == 2)
            let color = snapshots[0][y * 64 + x]
            let depth = snapshots[1][y * 64 + x]
            guard color == (inside ? 0xff00ff00 : 0), depth == (inside ? Float(0.5) : 0.75).bitPattern else {
                throw GPUError.backendFailure("Unexpected viewport sample at \(x), \(y): \(color), \(depth)")
            }
            guard snapshots[2][y * 64 + x] == 0xffff0000 else {
                throw GPUError.backendFailure("Empty scissor must preserve the pass clear")
            }
            words += [color, depth]
        }
    }
    return words
}

private func depthBiasDiagnostic(_ device: Device) async throws -> [UInt32] {
    let source = ShaderInput.wgsl("""
        @vertex fn flatDepth(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
            if i == 0u { return vec4f(-1, -1, 0.5, 1); }
            if i == 1u { return vec4f(3, -1, 0.5, 1); }
            return vec4f(-1, 3, 0.5, 1);
        }
        @vertex fn slope(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
            if i == 0u { return vec4f(-1, -1, 0.25, 1); }
            if i == 1u { return vec4f(3, -1, 0.75, 1); }
            return vec4f(-1, 3, 0.25, 1);
        }
        @fragment fn fs() -> @location(0) vec4f { return vec4f(1, 0, 0, 1); }
        """)
    let compiler = IndependentShaderCompiler()
    let flat = try compiler.compileRenderStage(source, entryPoint: "flatDepth", to: .wgsl)
    let slope = try compiler.compileRenderStage(source, entryPoint: "slope", to: .wgsl)
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: .wgsl)
    let color = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm,
        usage: [.renderAttachment, .copySource]))
    let depth = try await device.createTexture(.init(width: 1, height: 1, format: .depth32Float,
        usage: [.renderAttachment, .copySource]))
    let read = try await device.createBuffer(.init(size: 4, usage: [.copyDestination, .mapRead]))
    let cases: [(RenderStageArtifact, RenderDepthBiasState, Double)] = [
        (flat, .init(), 0.53125),
        (flat, .init(constant: 1 << 20), 0.53125),
        (flat, .init(constant: 1 << 20), 1),
        (flat, .init(constant: -(1 << 20)), 1),
        (slope, .init(slopeScale: 1, clamp: 0.03125), 1),
        (slope, .init(slopeScale: -1, clamp: -0.03125), 1),
    ]
    var commands: [BufferCommand] = []
    for (vertex, bias, clear) in cases {
        let pipeline = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm,
                                                             depthState: .init(bias: bias))
        commands += [
            .render(pipeline: pipeline, target: color, clear: .transparentBlack, vertexCount: 3,
                    depth: .init(texture: depth, loadAction: .clear(clear))),
            .copyTextureToBuffer(source: color, destination: read, bytesPerRow: 256), .read(buffer: read, byteCount: 4),
            .copyTextureToBuffer(source: depth, destination: read, bytesPerRow: 256), .read(buffer: read, byteCount: 4),
        ]
    }
    let ticket = try await device.submitCommands(commands)
    let red: UInt32 = 0xff0000ff
    let expected: [[UInt32]] = [
        [red], [Float(0.5).bitPattern],
        [0], [Float(0.53125).bitPattern],
        [red], [Float(0.5625).bitPattern],
        [red], [Float(0.4375).bitPattern],
        [red], [Float(0.40625).bitPattern],
        [red], [Float(0.34375).bitPattern],
    ]
    let snapshots = try await ticket.wait()
    guard snapshots == expected else {
        throw GPUError.backendFailure("Unexpected depth bias readback: \(snapshots)")
    }
    return snapshots.flatMap { $0 }
}

private func stencilDiagnostic(_ device: Device, sampleCount: UInt32) async throws -> [UInt32] {
    let source = ShaderInput.wgsl("""
        @vertex fn vs(@location(0) p: vec2f) -> @builtin(position) vec4f { return vec4f(p, 0.5, 1); }
        @fragment fn fs() -> @location(0) vec4f { return vec4f(0, 1, 0, 1); }
        """)
    let compiler = IndependentShaderCompiler()
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .wgsl)
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: .wgsl)
    let layouts = [VertexBufferLayout(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float32x2)])]
    func pipeline(_ compare: CompareFunction, _ stencil: RenderStencilState) async throws -> RenderPipeline {
        try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm,
            vertexBuffers: layouts, depthState: .init(format: .depth24PlusStencil8, compare: compare,
                writeEnabled: false, stencil: stencil), sampleCount: sampleCount)
    }
    let writer = try await pipeline(.less, .init(front: .init(pass: .replace), back: .init(pass: .invert), writeMask: 0x0f))
    let equal = RenderStencilFaceState(compare: .equal)
    let reader = try await pipeline(.always, .init(front: equal, back: equal, readMask: 0x0f, writeMask: 0))
    let failure = try await pipeline(.never, .init(
        front: .init(compare: .equal, fail: .incrementWrap, depthFail: .decrementWrap), readMask: 0x0f, writeMask: 0x0f))
    let color = try await device.createTexture(.init(width: 2, height: 1, format: .rgba8Unorm,
        usage: [.renderAttachment, .copySource]))
    let target: ResourceID
    let resolve: RenderResolveAttachment?
    if sampleCount > 1 {
        target = try await device.createTexture(.init(width: 2, height: 1, format: .rgba8Unorm,
            usage: .renderAttachment, sampleCount: sampleCount))
        resolve = .init(texture: color)
    } else {
        target = color
        resolve = nil
    }
    let baseMip: UInt32 = sampleCount == 1 ? 1 : 0
    let depth = try await device.createTexture(.init(width: 2 << baseMip, height: 1 << baseMip, format: .depth24PlusStencil8,
        usage: .renderAttachment, mipLevelCount: baseMip + 1, sampleCount: sampleCount))
    let view = try await device.createTextureView(texture: depth, .init(baseMipLevel: baseMip))
    let vertices = try await device.createBuffer(.init(size: 48, usage: [.vertex, .copyDestination]))
    let read = try await device.createBuffer(.init(size: 8, usage: [.copyDestination, .mapRead]))
    func draw(_ pipeline: RenderPipeline, reference: UInt32, first: UInt32 = 0,
              stencil: RenderStencilAttachmentState.LoadAction = .load,
              depthLoad: RenderDepthAttachment.LoadAction = .load, x: UInt32? = nil) -> BufferCommand {
        .render(pipeline: pipeline, target: target, clear: .transparentBlack, vertexCount: 3,
                vertexBuffers: [.init(buffer: vertices)], firstVertex: first,
                depth: .init(texture: view, loadAction: depthLoad,
                             stencil: .init(loadAction: stencil, reference: reference)), resolve: resolve,
                raster: .init(scissor: x.map { .init(x: $0, y: 0, width: 1, height: 1) }))
    }
    let copy = BufferCommand.copyTextureToBuffer(source: color, destination: read, bytesPerRow: 256)
    let snapshot = BufferCommand.read(buffer: read, byteCount: 8)
    let positions: [Float] = [-1, -1, 3, -1, -1, 3, -1, -1, -1, 3, 3, -1]
    let ticket = try await device.submitCommands([
        .write(buffer: vertices, offset: 0, words: positions.map(\.bitPattern)),
        draw(reader, reference: 0, stencil: .readOnly, depthLoad: .clear(0.75)), copy, snapshot,
        draw(writer, reference: 5, stencil: .clear(0x1a0), depthLoad: .readOnly, x: 0),
        draw(writer, reference: 0, first: 3, depthLoad: .readOnly, x: 1),
        draw(reader, reference: 5, stencil: .readOnly), copy, snapshot,
        draw(reader, reference: 15, stencil: .readOnly), copy, snapshot,
        draw(failure, reference: 5),
        draw(reader, reference: 4, stencil: .readOnly), copy, snapshot,
        draw(reader, reference: 0, stencil: .readOnly), copy, snapshot,
    ])
    let green: UInt32 = 0xff00ff00
    let snapshots = try await ticket.wait()
    guard snapshots == [[green, green], [green, 0], [0, green], [green, 0], [0, green]] else {
        throw GPUError.backendFailure("Unexpected stencil readback: \(snapshots)")
    }
    return snapshots.flatMap { $0 }
}

private func readOnlyDepthDiagnostic(_ device: Device) async throws -> [UInt32] {
    let source = ShaderInput.wgsl("""
        @group(0) @binding(0) var image: texture_depth_2d;
        fn position(i: u32, z: f32) -> vec4f {
            if i == 0u { return vec4f(-1, -1, z, 1); }
            if i == 1u { return vec4f(3, -1, z, 1); }
            return vec4f(-1, 3, z, 1);
        }
        @vertex fn seed(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f { return position(i, 0.25); }
        @vertex fn near(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f { return position(i, 0.125); }
        @vertex fn far(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f { return position(i, 0.5); }
        @fragment fn clearColor() -> @location(0) vec4f { return vec4f(0); }
        @fragment fn sampleDepth() -> @location(0) vec4f {
            return select(vec4f(1, 0, 0, 1), vec4f(0, 1, 0, 1), textureLoad(image, vec2u(0), 0u) == 0.25);
        }
        """)
    let compiler = IndependentShaderCompiler()
    let writer = try await device.createRenderPipeline(
        vertex: compiler.compileRenderStage(source, entryPoint: "seed", to: .wgsl),
        fragment: compiler.compileRenderStage(source, entryPoint: "clearColor", to: .wgsl),
        colorFormat: .rgba8Unorm, depthState: .init())
    let reader = try compiler.compileRenderStage(source, entryPoint: "sampleDepth", to: .wgsl)
    let near = try await device.createRenderPipeline(
        vertex: compiler.compileRenderStage(source, entryPoint: "near", to: .wgsl), fragment: reader,
        colorFormat: .rgba8Unorm, depthState: .init(writeEnabled: false))
    let far = try await device.createRenderPipeline(
        vertex: compiler.compileRenderStage(source, entryPoint: "far", to: .wgsl), fragment: reader,
        colorFormat: .rgba8Unorm, depthState: .init(writeEnabled: false))
    let color = try await device.createTexture(.init(width: 2, height: 1, format: .rgba8Unorm,
        usage: [.renderAttachment, .copySource]))
    let depth = try await device.createTexture(.init(width: 4, height: 2, format: .depth32Float,
        usage: [.renderAttachment, .textureBinding, .copySource], mipLevelCount: 2))
    let attachment = try await device.createTextureView(texture: depth, .init(baseMipLevel: 1))
    let sampled = try await device.createTextureView(texture: depth, .init(baseMipLevel: 1))
    let read = try await device.createBuffer(.init(size: 8, usage: [.copyDestination, .mapRead]))
    func draw(_ pipeline: RenderPipeline, load: RenderDepthAttachment.LoadAction) -> BufferCommand {
        .render(pipeline: pipeline, target: color, clear: .transparentBlack, vertexCount: 3,
                textures: [.init(binding: 0): sampled], depth: .init(texture: attachment, loadAction: load))
    }
    let ticket = try await device.submitCommands([
        .render(pipeline: writer, target: color, clear: .transparentBlack, vertexCount: 3,
                depth: .init(texture: attachment)),
        draw(near, load: .readOnly),
        .copyTextureToBuffer(source: color, destination: read, bytesPerRow: 256), .read(buffer: read, byteCount: 8),
        draw(far, load: .readOnly),
        .copyTextureToBuffer(source: color, destination: read, bytesPerRow: 256), .read(buffer: read, byteCount: 8),
        .copyTextureToBuffer(source: depth, destination: read, bytesPerRow: 256, mipLevel: 1),
        .read(buffer: read, byteCount: 8),
    ])
    let snapshots = try await ticket.wait()
    guard snapshots == [[0xff00ff00, 0xff00ff00], [0, 0], Array(repeating: Float(0.25).bitPattern, count: 2)] else {
        throw GPUError.backendFailure("Unexpected read-only depth readback: \(snapshots)")
    }
    return snapshots.flatMap { $0 }
}

private func depthStencilAspectDiagnostic(_ device: Device, sampleCount: UInt32) async throws -> [UInt32] {
    let depthType = sampleCount > 1 ? "texture_depth_multisampled_2d" : "texture_depth_2d"
    let stencilType = sampleCount > 1 ? "texture_multisampled_2d<u32>" : "texture_2d<u32>"
    let source = ShaderInput.wgsl("""
        @group(0) @binding(0) var depthImage: \(depthType);
        @group(0) @binding(1) var stencilImage: \(stencilType);
        @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
            if i == 0u { return vec4f(-1, -1, 0.25, 1); }
            if i == 1u { return vec4f(3, -1, 0.25, 1); }
            return vec4f(-1, 3, 0.25, 1);
        }
        @fragment fn seed() -> @location(0) vec4f { return vec4f(0); }
        @fragment fn useDepth() -> @location(0) vec4f {
            let d = textureLoad(depthImage, vec2u(0), 0u);
            return select(vec4f(1, 0, 0, 1), vec4f(0, 1, 0, 1), d > 0.24 && d < 0.26);
        }
        @fragment fn useStencil() -> @location(0) vec4f {
            let s = textureLoad(stencilImage, vec2u(0), 0u).r;
            return select(vec4f(1, 0, 0, 1), vec4f(0, 1, 0, 1), s == 7u);
        }
        """)
    let compiler = IndependentShaderCompiler()
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .wgsl)
    let replace = RenderStencilFaceState(pass: .replace)
    let stencilWriter = RenderStencilState(front: replace, back: replace)
    func pipeline(_ entry: String, depthWrite: Bool, stencil: RenderStencilState? = nil) async throws -> RenderPipeline {
        try await device.createRenderPipeline(vertex: vertex,
            fragment: compiler.compileRenderStage(source, entryPoint: entry, to: .wgsl), colorFormat: .rgba8Unorm,
            depthState: .init(format: .depth24PlusStencil8, compare: .always, writeEnabled: depthWrite, stencil: stencil),
            sampleCount: sampleCount)
    }
    let seed = try await pipeline("seed", depthWrite: true, stencil: stencilWriter)
    let readDepth = try await pipeline("useDepth", depthWrite: false, stencil: stencilWriter)
    let readStencil = try await pipeline("useStencil", depthWrite: true)
    let compute = try await device.createComputePipeline(compiler.compileCompute(.wgsl("""
        @group(0) @binding(0) var depthImage: \(depthType);
        @group(0) @binding(1) var stencilImage: \(stencilType);
        @group(0) @binding(2) var<storage, read_write> out: vec4u;
        @compute @workgroup_size(1) fn readAspects() {
            let d = textureLoad(depthImage, vec2u(0), 0u);
            out = vec4u(select(0u, 1u, d > 0.24 && d < 0.26),
                        textureLoad(stencilImage, vec2u(0), 0u).r, textureDimensions(stencilImage));
        }
        """), to: .wgsl))
    let baseMip: UInt32 = sampleCount == 1 ? 1 : 0
    let combined = try await device.createTexture(.init(width: 2 << baseMip, height: 1 << baseMip, format: .depth24PlusStencil8,
        usage: [.renderAttachment, .textureBinding], mipLevelCount: baseMip + 1, sampleCount: sampleCount))
    let attachment = try await device.createTextureView(texture: combined, .init(baseMipLevel: baseMip))
    let depth = try await device.createTextureView(texture: combined, .init(baseMipLevel: baseMip, aspect: .depthOnly))
    let stencil = try await device.createTextureView(texture: combined, .init(baseMipLevel: baseMip, aspect: .stencilOnly))
    let color = try await device.createTexture(.init(width: 2, height: 1, format: .rgba8Unorm,
        usage: [.renderAttachment, .copySource]))
    let target: ResourceID
    let resolve: RenderResolveAttachment?
    if sampleCount > 1 {
        target = try await device.createTexture(.init(width: 2, height: 1, format: .rgba8Unorm,
            usage: .renderAttachment, sampleCount: sampleCount))
        resolve = .init(texture: color)
    } else {
        target = color
        resolve = nil
    }
    let out = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 16, usage: [.copyDestination, .mapRead]))
    let copyColor = BufferCommand.copyTextureToBuffer(source: color, destination: read, bytesPerRow: 256)
    let readColor = BufferCommand.read(buffer: read, byteCount: 8)
    let ticket = try await device.submitCommands([
        .render(pipeline: seed, target: target, clear: .transparentBlack, vertexCount: 3,
                depth: .init(texture: attachment, stencil: .init(reference: 173)), resolve: resolve),
        .dispatchCompute(pipeline: compute, bindings: [.init(binding: 2): out], workgroups: .init(x: 1),
                         textures: [.init(binding: 0): depth, .init(binding: 1): stencil]),
        .copy(source: out, destination: read, byteCount: 16), .read(buffer: read, byteCount: 16),
        .render(pipeline: readDepth, target: target, clear: .transparentBlack, vertexCount: 3,
                textures: [.init(binding: 0): depth],
                depth: .init(texture: attachment, loadAction: .readOnly,
                             stencil: .init(loadAction: .load, reference: 7)), resolve: resolve),
        copyColor, readColor,
        .render(pipeline: readStencil, target: target, clear: .transparentBlack, vertexCount: 3,
                textures: [.init(binding: 1): stencil],
                depth: .init(texture: attachment, stencil: .init(loadAction: .readOnly)), resolve: resolve),
        copyColor, readColor,
    ])
    let snapshots = try await ticket.wait()
    guard snapshots == [[1, 173, 2, 1], [0xff00ff00, 0xff00ff00], [0xff00ff00, 0xff00ff00]] else {
        throw GPUError.backendFailure("Unexpected depth and stencil aspect readback: \(snapshots)")
    }
    return snapshots.flatMap { $0 }
}

private func stencilCopyDiagnostic(_ device: Device) async throws -> [UInt32] {
    let source = ShaderInput.wgsl("""
        @group(0) @binding(0) var depthImage: texture_depth_2d;
        @group(0) @binding(1) var stencilImage: texture_2d<u32>;
        @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
            if i == 0u { return vec4f(-1, -1, 0.25, 1); }
            if i == 1u { return vec4f(3, -1, 0.25, 1); }
            return vec4f(-1, 3, 0.25, 1);
        }
        @fragment fn seed() -> @location(0) vec4f { return vec4f(0); }
        @fragment fn useDepth() -> @location(0) vec4f {
            let d = textureLoad(depthImage, vec2u(0), 0u);
            return select(vec4f(1, 0, 0, 1), vec4f(0, 1, 0, 1), d > 0.24 && d < 0.26);
        }
        @fragment fn useStencil() -> @location(0) vec4f {
            let s = textureLoad(stencilImage, vec2u(0), 0u).r;
            return select(vec4f(1, 0, 0, 1), vec4f(0, 1, 0, 1), s == 7u);
        }
        """)
    let compiler = IndependentShaderCompiler()
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .wgsl)
    let seed = try await device.createRenderPipeline(vertex: vertex,
        fragment: compiler.compileRenderStage(source, entryPoint: "seed", to: .wgsl), colorFormat: .rgba8Unorm,
        depthState: .init(format: .depth24PlusStencil8))
    let compute = try await device.createComputePipeline(
        compiler.compileCompute(.wgsl("""
        @group(0) @binding(0) var depthImage: texture_depth_2d;
        @group(0) @binding(1) var stencilImage: texture_2d<u32>;
        @group(0) @binding(2) var<storage, read_write> out: vec4u;
        @compute @workgroup_size(1) fn readAspects() {
            let d = textureLoad(depthImage, vec2u(0), 0u);
            out = vec4u(select(0u, 1u, d > 0.24 && d < 0.26),
                        textureLoad(stencilImage, vec2u(0), 0u).r, textureDimensions(stencilImage));
        }
        """), to: .wgsl))
    let combined = try await device.createTexture(.init(width: 6, height: 4, format: .depth24PlusStencil8,
        usage: [.renderAttachment, .textureBinding, .copySource, .copyDestination], mipLevelCount: 2))
    let attachment = try await device.createTextureView(texture: combined, .init(baseMipLevel: 1))
    let clone = try await device.createTexture(.init(width: 3, height: 2, format: .depth24PlusStencil8,
        usage: [.textureBinding, .copySource, .copyDestination]))
    let depth = try await device.createTextureView(texture: clone, .init(aspect: .depthOnly))
    let stencil = try await device.createTextureView(texture: clone, .init(aspect: .stencilOnly))
    let color = try await device.createTexture(.init(width: 3, height: 2, format: .rgba8Unorm, usage: .renderAttachment))
    let upload = try await device.createBuffer(.init(size: 264, usage: [.copySource, .copyDestination]))
    let read = try await device.createBuffer(.init(size: 264, usage: [.copyDestination, .mapRead]))
    let out = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
    let ticket = try await device.submitCommands([
        .render(pipeline: seed, target: color, clear: .transparentBlack, vertexCount: 3,
                depth: .init(texture: attachment, stencil: .init())),
        .write(buffer: upload, offset: 4, words: [0xaaf08001]),
        .write(buffer: upload, offset: 260, words: [0xbba50711]),
        .copyBufferToTexture(source: upload, layout: .init(offset: 4, bytesPerRow: 256), destination: combined,
                             region: .init(width: 3, height: 2, mipLevel: 1), aspect: .stencilOnly),
        .copyTextureToTexture(source: combined, destination: clone, sourceRegion: .init(width: 3, height: 2, mipLevel: 1)),
        .copyTextureToBuffer(source: clone, destination: read, bytesPerRow: 256,
                             aspect: .stencilOnly, destinationOffset: 4),
        .read(buffer: read, byteCount: 264),
        .dispatchCompute(pipeline: compute, bindings: [.init(binding: 2): out], workgroups: .init(x: 1),
                         textures: [.init(binding: 0): depth, .init(binding: 1): stencil]),
        .copy(source: out, destination: read, byteCount: 16), .read(buffer: read, byteCount: 16),
        .render(pipeline: seed, target: color, clear: .transparentBlack, vertexCount: 0,
                depth: .init(texture: attachment, loadAction: .load,
                             stencil: .init(loadAction: .load, storeAction: .discard))),
        .copyTextureToTexture(source: combined, destination: clone, sourceRegion: .init(width: 3, height: 2, mipLevel: 1)),
        .dispatchCompute(pipeline: compute, bindings: [.init(binding: 2): out], workgroups: .init(x: 1),
                         textures: [.init(binding: 0): depth, .init(binding: 1): stencil]),
        .copy(source: out, destination: read, byteCount: 16), .read(buffer: read, byteCount: 16),
        .render(pipeline: seed, target: color, clear: .transparentBlack, vertexCount: 0,
                depth: .init(texture: attachment, loadAction: .load,
                             stencil: .init(loadAction: .clear(17)), storeAction: .discard)),
        .copyTextureToTexture(source: combined, destination: clone, sourceRegion: .init(width: 3, height: 2, mipLevel: 1)),
        .dispatchCompute(pipeline: compute, bindings: [.init(binding: 2): out], workgroups: .init(x: 1),
                         textures: [.init(binding: 0): depth, .init(binding: 1): stencil]),
        .copy(source: out, destination: read, byteCount: 16), .read(buffer: read, byteCount: 16),
    ])
    var expected: [UInt32] = Array(repeating: 0, count: 66)
    expected[1] = 0x00f08001
    expected[65] = 0x00a50711
    let snapshots = try await ticket.wait()
    guard snapshots == [expected, [1, 1, 3, 2], [1, 0, 3, 2], [0, 17, 3, 2]] else {
        throw GPUError.backendFailure("Unexpected stencil copy readback: \(snapshots)")
    }
    return [snapshots[0][1], snapshots[0][65]] + snapshots.dropFirst().flatMap { $0 }
}

private func textureCopyDiagnostic(_ device: Device) async throws -> [UInt32] {
    let image = try await device.createTexture(.init(width: 4, height: 2, format: .rgba8Unorm,
        usage: [.copySource, .copyDestination], mipLevelCount: 2))
    let destination = try await device.createTexture(.init(width: 2, height: 1, format: .rgba8Unorm,
        usage: [.copySource, .copyDestination]))
    let upload = try await device.createBuffer(.init(size: 16, usage: [.copySource, .copyDestination]))
    let read = try await device.createBuffer(.init(size: 8, usage: [.copyDestination, .mapRead]))
    let ticket = try await device.submitCommands([
        .write(buffer: upload, offset: 0, words: [0xff00ff00, 0xff0000ff, 0xffff0000, 0xffffffff]),
        .copyBufferToTexture(source: upload, layout: .init(), destination: image, region: .init(y: 1, width: 4, height: 1)),
        .copyTextureToTexture(source: image, destination: image,
                              sourceRegion: .init(x: 1, y: 1, width: 1, height: 1),
                              destinationOrigin: .init(x: 1, mipLevel: 1)),
        .copyTextureToTexture(source: image, destination: destination, sourceRegion: .init(width: 2, height: 1, mipLevel: 1)),
        .copyTextureToBuffer(source: destination, destination: read, bytesPerRow: 256),
        .read(buffer: read, byteCount: 8),
    ])
    let snapshots = try await ticket.wait()
    guard snapshots == [[0, 0xff0000ff]] else {
        throw GPUError.backendFailure("Unexpected color texture copy readback: \(snapshots)")
    }
    return snapshots[0]
}

private func arrayTextureDiagnostic(_ device: Device) async throws -> [UInt32] {
    let source = ShaderInput.wgsl("""
        @group(0) @binding(0) var image: texture_2d_array<u32>;
        @group(0) @binding(1) var<storage, read_write> out: array<vec4u, 5>;
        @compute @workgroup_size(1) fn run() {
            out[0] = textureLoad(image, vec2u(0), 0u, 0u);
            out[1] = textureLoad(image, vec2i(1, 0), 1i, 0i);
            out[2] = textureLoad(image, vec2u(0), 2u, 0u);
            out[3] = textureLoad(image, vec2i(0), -1i, 0i) +
                     textureLoad(image, vec2u(0), 0xffffffffu, 0u) +
                     textureLoad(image, vec2u(0), 1u, 99u);
            out[4] = vec4u(textureDimensions(image, 0), textureNumLayers(image), textureNumLevels(image));
        }
        """)
    let shader = try IndependentShaderCompiler().compileCompute(source, to: .wgsl)
    let pipeline = try await device.createComputePipeline(shader)
    let image = try await device.createTexture(.init(width: 4, height: 2, format: .rgba8Uint,
        usage: [.copySource, .copyDestination, .textureBinding], mipLevelCount: 2, arrayLayerCount: 3))
    let copied = try await device.createTexture(.init(width: 2, height: 1, format: .rgba8Uint,
        usage: [.copySource, .copyDestination], arrayLayerCount: 3))
    let view = try await device.createTextureView(texture: image, .init(baseMipLevel: 1))
    let upload = try await device.createBuffer(.init(size: 16, usage: [.copySource, .copyDestination]))
    let out = try await device.createBuffer(.init(size: 80, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 80, usage: [.copyDestination, .mapRead]))
    let layeredRead = try await device.createBuffer(.init(size: 264, usage: [.copyDestination, .mapRead]))
    func load(_ texture: ResourceID) -> BufferCommand {
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 1): out], workgroups: .init(x: 1),
                         textures: [.init(binding: 0): texture])
    }
    let ticket = try await device.submitCommands([
        .write(buffer: upload, offset: 0, words: [0xf0800201, 0x04030201, 0x08070605, 0xffeeddcc]),
        .copyBufferToTexture(source: upload, layout: .init(), destination: image,
                             region: .init(width: 2, height: 1, mipLevel: 1, arrayLayer: 0)),
        .copyBufferToTexture(source: upload, layout: .init(offset: 8), destination: image,
                             region: .init(width: 2, height: 1, mipLevel: 1, arrayLayer: 1)),
        load(view),
        .copy(source: out, destination: read, byteCount: 80), .read(buffer: read, byteCount: 80),
        .copyTextureToTexture(source: image, destination: image,
                              sourceRegion: .init(width: 2, height: 1, mipLevel: 1, arrayLayer: 0),
                              destinationOrigin: .init(mipLevel: 1, arrayLayer: 2)),
        .copyTextureToBuffer(source: image, destination: read, bytesPerRow: 256, mipLevel: 1, arrayLayer: 2),
        .read(buffer: read, byteCount: 8),
        .copyTextureToTexture(source: image, destination: copied,
                              sourceRegion: .init(width: 2, height: 1, mipLevel: 1, depth: 2),
                              destinationOrigin: .init(arrayLayer: 1)),
        .copyTextureToBuffer(source: copied, destination: read, bytesPerRow: 256, arrayLayer: 1),
        .read(buffer: read, byteCount: 8),
        .copyTextureToBuffer(source: copied, destination: read, bytesPerRow: 256, arrayLayer: 2),
        .read(buffer: read, byteCount: 8),
        .copyTextureToBuffer(source: copied, destination: layeredRead, bytesPerRow: 256, arrayLayer: 1, depth: 2),
        .read(buffer: layeredRead, byteCount: 264),
    ])
    let expected: [UInt32] = [1, 2, 128, 240, 204, 221, 238, 255, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 3, 1]
    let snapshots = try await ticket.wait()
    guard snapshots.count == 5,
          Array(snapshots.prefix(4)) == [expected, [0xf0800201, 0x04030201], [0xf0800201, 0x04030201], [0x08070605, 0xffeeddcc]],
          snapshots[4].count == 66,
          Array(snapshots[4][0..<2]) == [0xf0800201, 0x04030201],
          Array(snapshots[4][64..<66]) == [0x08070605, 0xffeeddcc] else {
        throw GPUError.backendFailure("Unexpected array texture readback: \(snapshots)")
    }
    return snapshots.prefix(4).flatMap { $0 }
}

private func arrayLayerViewDiagnostic(_ device: Device) async throws -> [UInt32] {
    let source = ShaderInput.wgsl("""
        @group(0) @binding(0) var source: texture_2d_array<u32>;
        @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
            if i == 0u { return vec4f(-1, -1, 0, 1); }
            if i == 1u { return vec4f(3, -1, 0, 1); }
            return vec4f(-1, 3, 0, 1);
        }
        @fragment fn fs(@builtin(position) p: vec4f) -> @location(0) vec4u {
            let x = select(0u, 1u, p.x > 1.0);
            let texel = textureLoad(source, vec2u(x, 0), 0u, 0u);
            return vec4u(texel.rgb, textureNumLayers(source));
        }
        """)
    let compiler = IndependentShaderCompiler()
    let pipeline = try await device.createRenderPipeline(
        vertex: compiler.compileRenderStage(source, entryPoint: "vs", to: .wgsl),
        fragment: compiler.compileRenderStage(source, entryPoint: "fs", to: .wgsl), colorFormat: .rgba8Uint)
    let image = try await device.createTexture(.init(width: 4, height: 2, format: .rgba8Uint,
        usage: [.renderAttachment, .textureBinding, .copySource, .copyDestination], mipLevelCount: 2, arrayLayerCount: 3))
    let target = try await device.createTextureView(texture: image,
        .init(baseMipLevel: 1, dimension: .d2, baseArrayLayer: 1, arrayLayerCount: 1))
    let sourceView = try await device.createTextureView(texture: image,
        .init(baseMipLevel: 1, dimension: .d2Array, baseArrayLayer: 2, arrayLayerCount: 1))
    let upload = try await device.createBuffer(.init(size: 8, usage: [.copySource, .copyDestination]))
    let read = try await device.createBuffer(.init(size: 8, usage: [.copyDestination, .mapRead]))
    func draw(_ sampled: ResourceID) -> BufferCommand {
        .render(pipeline: pipeline, target: target, clear: .transparentBlack, vertexCount: 3,
                textures: [.init(binding: 0): sampled])
    }
    let compute = try await device.createComputePipeline(compiler.compileCompute(.wgsl("""
        @group(0) @binding(0) var image: texture_storage_2d<rgba8unorm, write>;
        @compute @workgroup_size(1) fn run() {
            textureStore(image, vec2u(0), vec4f(1, 0, 0, 1));
        }
        """), to: .wgsl))
    let storage = try await device.createTexture(.init(width: 2, height: 1, format: .rgba8Unorm,
        usage: [.storageBinding, .copySource], arrayLayerCount: 2))
    let slice = try await device.createTextureView(texture: storage, .init(dimension: .d2, baseArrayLayer: 1))
    let ticket = try await device.submitCommands([
        .write(buffer: upload, offset: 0, words: [0x281e140a, 0x50463c32]),
        .copyBufferToTexture(source: upload, layout: .init(), destination: image,
                             region: .init(width: 2, height: 1, mipLevel: 1, arrayLayer: 2)),
        draw(sourceView),
        .copyTextureToBuffer(source: image, destination: read, bytesPerRow: 256, mipLevel: 1, arrayLayer: 1),
        .read(buffer: read, byteCount: 8),
        .copyTextureToBuffer(source: image, destination: read, bytesPerRow: 256, mipLevel: 1, arrayLayer: 2),
        .read(buffer: read, byteCount: 8),
        .dispatchCompute(pipeline: compute, bindings: [:], workgroups: .init(x: 1), textures: [.init(binding: 0): slice]),
        .copyTextureToBuffer(source: storage, destination: read, bytesPerRow: 256, arrayLayer: 1),
        .read(buffer: read, byteCount: 8),
        .copyTextureToBuffer(source: storage, destination: read, bytesPerRow: 256, arrayLayer: 0),
        .read(buffer: read, byteCount: 8),
    ])
    let snapshots = try await ticket.wait()
    guard snapshots == [[0x011e140a, 0x01463c32], [0x281e140a, 0x50463c32], [0xff0000ff, 0], [0, 0]] else {
        throw GPUError.backendFailure("Unexpected array layer view readback: \(snapshots)")
    }
    return snapshots.flatMap { $0 }
}

private func guardedTexturePipeline(
    _ source: String, requiring operations: Set<UInt32>, compiler: IndependentShaderCompiler, device: Device
) async throws -> ComputePipeline {
    guard case .spirv(let words) = try compiler.compile(.wgsl(source), to: .spirv) else {
        throw GPUError.backendFailure("Expected binary SPIR-V for guarded texture access")
    }
    var opcodes = Set<UInt32>()
    var offset = 5
    while offset < words.count {
        let header = words[offset]
        let count = Int(header >> 16)
        guard count > 0, count <= words.count - offset else {
            throw GPUError.backendFailure("Invalid SPIR-V instruction length")
        }
        opcodes.insert(header & 0xffff)
        offset += count
    }
    // The source has no explicit queries or conditional control flow.
    let missing = operations.union([245, 247, 250]).subtracting(opcodes)
    guard missing.isEmpty else {
        throw GPUError.backendFailure("Missing robust texture guards in raw SPIR-V: \(missing.sorted())")
    }
    let shader = try compiler.compileWebGPUCompute(.spirv(words), entryPoint: "round_trip")
    return try await device.createComputePipeline(shader)
}

private func arraySamplingDiagnostic(_ device: Device) async throws -> [UInt32] {
    let source = ShaderInput.wgsl("""
        @vertex fn vs() -> @builtin(position) vec4f { return vec4f(0, 0, 0, 1); }
        @fragment fn fs() -> @location(0) vec4f { return vec4f(0); }
        """)
    let compiler = IndependentShaderCompiler()
    let render = try await device.createRenderPipeline(
        vertex: compiler.compileRenderStage(source, entryPoint: "vs", to: .wgsl),
        fragment: compiler.compileRenderStage(source, entryPoint: "fs", to: .wgsl),
        colorFormat: .rgba8Unorm, depthState: .init())
    let pipeline = try await device.createComputePipeline(compiler.compileCompute(.wgsl("""
        @group(0) @binding(0) var color: texture_2d_array<f32>;
        @group(0) @binding(1) var depth: texture_depth_2d_array;
        @group(0) @binding(2) var nearest: sampler;
        @group(0) @binding(3) var comparison: sampler_comparison;
        @group(0) @binding(4) var<storage, read_write> out: array<vec4f, 6>;
        @compute @workgroup_size(1) fn run() {
            let uv = vec2f(0.5);
            out[0] = textureSampleLevel(color, nearest, uv, -5i, 0.0);
            out[1] = textureSampleLevel(color, nearest, uv, 0xffffffffu, 0.0);
            out[2] = textureGather(0u, color, nearest, uv, -1i, vec2i(1, 0));
            let less = textureSampleCompareLevel(depth, comparison, uv, -1i, 0.5, vec2i(0));
            let greater = textureSampleCompareLevel(depth, comparison, uv, 99u, 0.5);
            let raw = textureSampleLevel(depth, nearest, uv, 0u, 0u);
            out[3] = vec4f(less, greater, raw, textureGather(depth, nearest, uv, 99u).x);
            out[4] = textureGatherCompare(depth, comparison, uv, 99u, 0.5);
            out[5] = textureSampleGrad(color, nearest, uv, 1u, vec2f(0), vec2f(0));
        }
        """), to: .wgsl))
    let roundTrip = try await guardedTexturePipeline("""
        @group(0) @binding(0) var image: texture_2d_array<f32>;
        @group(0) @binding(4) var<storage, read_write> result: array<u32, 10>;
        @compute @workgroup_size(1) fn round_trip(@builtin(global_invocation_id) id: vec3u) {
            let negative = i32(id.x) - 1i;
            let huge = id.x + 0xffffffffu;
            let layer = id.x;
            let level = id.x + 1u;
            let xy = vec2i(0);
            let edge = vec2u(id.x + 1u, 0u);
            result[0] = bitcast<u32>(textureLoad(image, xy, layer, level).x);
            result[1] = bitcast<u32>(textureLoad(image, vec2i(negative, 0i), layer, level).w);
            result[2] = bitcast<u32>(textureLoad(image, vec2u(huge, 0u), layer, level).w);
            result[3] = bitcast<u32>(textureLoad(image, xy, negative, level).w);
            result[4] = bitcast<u32>(textureLoad(image, xy, huge, level).w);
            result[5] = bitcast<u32>(textureLoad(image, xy, layer, negative).w);
            result[6] = bitcast<u32>(textureLoad(image, xy, layer, huge).w);
            result[7] = bitcast<u32>(textureLoad(image, edge, layer, level).w);
            result[8] = bitcast<u32>(textureLoad(image, edge, layer, id.x).w);
            result[9] = bitcast<u32>(textureLoad(image, xy, layer + 1u, level).y);
        }
        """, requiring: [95, 103, 106], compiler: compiler, device: device) // OpImageFetch, OpImageQuerySizeLod, OpImageQueryLevels.
    let color = try await device.createTexture(.init(width: 2, height: 2, format: .rgba8Unorm,
        usage: [.copyDestination, .textureBinding], mipLevelCount: 2, arrayLayerCount: 3))
    let depth = try await device.createTexture(.init(width: 2, height: 2, format: .depth32Float,
        usage: [.renderAttachment, .textureBinding], arrayLayerCount: 3))
    let colorView = try await device.createTextureView(texture: color, .init(baseArrayLayer: 1, arrayLayerCount: 2))
    let depthView = try await device.createTextureView(texture: depth, .init(baseArrayLayer: 1, arrayLayerCount: 2))
    let firstDepth = try await device.createTextureView(texture: depth, .init(dimension: .d2, baseArrayLayer: 1))
    let lastDepth = try await device.createTextureView(texture: depth, .init(dimension: .d2, baseArrayLayer: 2))
    let target = try await device.createTexture(.init(width: 2, height: 2, format: .rgba8Unorm, usage: .renderAttachment))
    let nearest = try await device.createSampler(.init(addressModeU: .repeat))
    let compare = try await device.createSampler(.init(compare: .less))
    let upload = try await device.createBuffer(.init(size: 776, usage: [.copySource, .copyDestination]))
    let out = try await device.createBuffer(.init(size: 96, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 96, usage: [.copyDestination, .mapRead]))
    let ticket = try await device.submitCommands([
        .write(buffer: upload, offset: 0, words: [0xff0000ff, 0xff0000ff]),
        .write(buffer: upload, offset: 256, words: [0xff0000ff, 0xff0000ff]),
        .write(buffer: upload, offset: 512, words: [0xff00ff00, 0xff00ff00]),
        .write(buffer: upload, offset: 768, words: [0xff00ff00, 0xff00ff00]),
        .copyBufferToTexture(source: upload, layout: .init(bytesPerRow: 256), destination: color,
                             region: .init(width: 2, height: 2, arrayLayer: 1)),
        .copyBufferToTexture(source: upload, layout: .init(offset: 512, bytesPerRow: 256), destination: color,
                             region: .init(width: 2, height: 2, arrayLayer: 2)),
        .render(pipeline: render, target: target, clear: .transparentBlack, vertexCount: 0,
                depth: .init(texture: firstDepth, loadAction: .clear(0.25))),
        .render(pipeline: render, target: target, clear: .transparentBlack, vertexCount: 0,
                depth: .init(texture: lastDepth, loadAction: .clear(0.75))),
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 4): out], workgroups: .init(x: 1),
                         textures: [.init(binding: 0): colorView, .init(binding: 1): depthView],
                         samplers: [.init(binding: 2): nearest, .init(binding: 3): compare]),
        .copy(source: out, destination: read, byteCount: 96), .read(buffer: read, byteCount: 96),
        .copyBufferToTexture(source: upload, layout: .init(), destination: color,
                             region: .init(width: 1, height: 1, mipLevel: 1, arrayLayer: 1)),
        .copyBufferToTexture(source: upload, layout: .init(offset: 512), destination: color,
                             region: .init(width: 1, height: 1, mipLevel: 1, arrayLayer: 2)),
        .dispatchCompute(pipeline: roundTrip, bindings: [.init(binding: 4): out], workgroups: .init(x: 1),
                         textures: [.init(binding: 0): colorView]),
        .copy(source: out, destination: read, byteCount: 40), .read(buffer: read, byteCount: 40),
    ])
    let expected: [Float] = [1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0.25, 0.75, 1, 1, 1, 1, 0, 1, 0, 1]
    let roundTripExpected: [Float] = [1, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    let snapshots = try await ticket.wait()
    guard snapshots == [expected.map(\.bitPattern), roundTripExpected.map(\.bitPattern)] else {
        throw GPUError.backendFailure("Unexpected array sampling readback: \(snapshots)")
    }
    return snapshots.flatMap { $0 }
}

private func storageArrayDiagnostic(_ device: Device) async throws -> [UInt32] {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl("""
        @group(0) @binding(0) var source: texture_storage_2d_array<r32uint, read>;
        @group(0) @binding(1) var image: texture_storage_2d_array<r32uint, read_write>;
        @group(0) @binding(2) var<storage, read_write> result: array<u32, 10>;
        @compute @workgroup_size(1) fn run() {
            let xy = vec2i(0);
            let size = textureDimensions(image);
            let first = textureLoad(image, xy, 0i);
            let second = textureLoad(image, xy, 1u);
            let input = textureLoad(source, xy, 0u);
            result[0] = size.x;
            result[1] = size.y;
            result[2] = textureNumLayers(source);
            result[3] = textureNumLayers(image);
            result[4] = input.x;
            result[5] = first.x;
            result[6] = second.x;
            result[7] = textureLoad(image, xy, -1i).w;
            result[8] = textureLoad(image, xy, 0xffffffffu).w;
            result[9] = textureLoad(source, xy, 1u).w;
            textureStore(image, xy, 0i, first + input);
            textureStore(image, xy, 1u, second + vec4u(7u));
            textureStore(image, xy, -1i, vec4u(99u));
            textureStore(image, xy, 2u, vec4u(99u));
            textureStore(image, vec2u(0xffffffffu, 0u), 0u, vec4u(99u));
        }
        """), to: .wgsl)
    let pipeline = try await device.createComputePipeline(shader)
    let storageRoundTrip = try await guardedTexturePipeline("""
        @group(0) @binding(1) var image: texture_storage_2d_array<r32uint, read_write>;
        @group(0) @binding(2) var<storage, read_write> result: array<u32, 10>;
        @group(0) @binding(3) var wideSource: texture_storage_2d<r64uint, read>;
        @group(0) @binding(4) var wideDestination: texture_storage_2d<r64uint, write>;

        @compute @workgroup_size(1) fn wide_copy(@builtin(global_invocation_id) id: vec3u) {
            let coordinate = vec2u(id.x, 0u);
            textureStore(wideDestination, coordinate, textureLoad(wideSource, coordinate));
        }
        @compute @workgroup_size(1) fn round_trip(@builtin(global_invocation_id) id: vec3u) {
            let negative = i32(id.x) - 1i;
            let huge = id.x + 0xffffffffu;
            let layer = id.x;
            let xy = vec2i(0);
            let edge = vec2u(id.x + 2u, 0u);
            result[0] = textureLoad(image, xy, layer).x;
            result[1] = textureLoad(image, vec2i(negative, 0i), layer).w;
            result[2] = textureLoad(image, vec2u(huge, 0u), layer).w;
            result[3] = textureLoad(image, xy, negative).w;
            result[4] = textureLoad(image, xy, huge).w;
            result[5] = textureLoad(image, edge, layer).w;
            result[6] = textureLoad(image, xy, layer + 2u).w;
            result[7] = textureLoad(image, xy, layer + 1u).x;
            result[8] = textureLoad(image, vec2i(1, 0), layer).w;
            result[9] = textureLoad(image, xy, layer).w;
            textureStore(image, xy, layer, vec4u(61u));
            textureStore(image, xy, layer + 1u, vec4u(72u));
            textureStore(image, vec2i(negative, 0i), layer, vec4u(91u));
            textureStore(image, vec2u(huge, 0u), layer, vec4u(92u));
            textureStore(image, xy, negative, vec4u(93u));
            textureStore(image, xy, huge, vec4u(94u));
            textureStore(image, edge, layer, vec4u(95u));
            textureStore(image, xy, layer + 2u, vec4u(96u));
        }
        """, requiring: [98, 99, 104], compiler: compiler, device: device) // OpImageRead, OpImageWrite, OpImageQuerySize.
    let texture = try await device.createTexture(.init(width: 4, height: 2, format: .r32Uint,
        usage: [.copySource, .copyDestination, .storageBinding], mipLevelCount: 2, arrayLayerCount: 4))
    let source = try await device.createTextureView(texture: texture,
        .init(baseMipLevel: 1, dimension: .d2Array, arrayLayerCount: 1))
    let image = try await device.createTextureView(texture: texture,
        .init(baseMipLevel: 1, baseArrayLayer: 1, arrayLayerCount: 2))
    let slice = try await device.createTextureView(texture: texture,
        .init(baseMipLevel: 1, dimension: .d2, baseArrayLayer: 1))
    let upload = try await device.createBuffer(.init(size: 16, usage: [.copySource, .copyDestination]))
    let output = try await device.createBuffer(.init(size: 40, usage: [.storage, .copySource]))
    let readback = try await device.createBuffer(.init(size: 40, usage: [.copyDestination, .mapRead]))
    func dispatch(source: ResourceID, image: ResourceID) -> BufferCommand {
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 2): output], workgroups: .init(x: 1),
                         textures: [.init(binding: 0): source, .init(binding: 1): image])
    }
    do {
        _ = try await device.submitCommands([dispatch(source: source, image: slice)])
        throw GPUError.backendFailure("Accepted a 2D view in an array storage binding")
    } catch GPUError.validation {}
    do {
        _ = try await device.submitCommands([dispatch(source: image, image: image)])
        throw GPUError.backendFailure("Accepted overlapping writable storage array views")
    } catch GPUError.validation {}
    var commands: [BufferCommand] = [.write(buffer: upload, offset: 0, words: [6, 10, 20, 40])]
    for layer in UInt32(0)..<4 {
        commands.append(.copyBufferToTexture(source: upload, layout: .init(offset: UInt64(layer) * 4),
            destination: texture, region: .init(width: 1, height: 1, mipLevel: 1, arrayLayer: layer)))
    }
    commands += [dispatch(source: source, image: image),
                 .copy(source: output, destination: readback, byteCount: 40), .read(buffer: readback, byteCount: 40)]
    for layer in UInt32(0)..<4 {
        commands += [.copyTextureToBuffer(source: texture, destination: readback, bytesPerRow: 256,
                                          mipLevel: 1, arrayLayer: layer), .read(buffer: readback, byteCount: 8)]
    }
    let drawSource = ShaderInput.wgsl("""
        @group(1) @binding(3) var image: texture_storage_2d_array<rgba8unorm, write>;
        @vertex fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
            if index == 0u { return vec4f(-1, -1, 0, 1); }
            if index == 1u { return vec4f(3, -1, 0, 1); }
            return vec4f(-1, 3, 0, 1);
        }
        fn paint(value: vec4f) { textureStore(image, vec2i(0), 1u, value); }
        @fragment fn fs() -> @location(0) vec4f {
            paint(vec4f(1, 0, 1, 1));
            return vec4f(0, 1, 0, 1);
        }
        """)
    let draw = try await device.createRenderPipeline(
        vertex: compiler.compileRenderStage(drawSource, entryPoint: "vs", to: .wgsl),
        fragment: compiler.compileRenderStage(drawSource, entryPoint: "fs", to: .wgsl), colorFormat: .rgba8Unorm)
    let target = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm, usage: .renderAttachment))
    let color = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm,
        usage: [.storageBinding, .copySource], arrayLayerCount: 2))
    commands.append(.render(pipeline: draw, target: target, clear: .transparentBlack, vertexCount: 3,
                            textures: [.init(group: 1, binding: 3): color]))
    for layer in UInt32(0)..<2 {
        commands += [.copyTextureToBuffer(source: color, destination: readback, bytesPerRow: 256, arrayLayer: layer),
                     .read(buffer: readback, byteCount: 4)]
    }
    commands += [
        .dispatchCompute(pipeline: storageRoundTrip, bindings: [.init(binding: 2): output], workgroups: .init(x: 1),
                         textures: [.init(binding: 1): image]),
        .copy(source: output, destination: readback, byteCount: 40), .read(buffer: readback, byteCount: 40),
    ]
    for layer in UInt32(0)..<4 {
        commands += [.copyTextureToBuffer(source: texture, destination: readback, bytesPerRow: 256,
                                          mipLevel: 1, arrayLayer: layer), .read(buffer: readback, byteCount: 8)]
    }
    let ticket = try await device.submitCommands(commands)
    let snapshots = try await ticket.wait()
    let expected: [[UInt32]] = [
        [2, 1, 1, 2, 6, 10, 20, 0, 0, 0], [6, 0], [16, 0], [27, 0], [40, 0], [0], [0xffff00ff],
        [16, 0, 0, 0, 0, 0, 0, 27, 1, 1],
        [6, 0], [61, 0], [72, 0], [40, 0],
    ]
    guard snapshots == expected else {
        throw GPUError.backendFailure("Unexpected storage array readback: \(snapshots)")
    }
    return snapshots.flatMap { $0 }
}

private func storageFormatDiagnostic(_ device: Device) async throws -> Int {
    let formats: [(name: String, vector: String, packed: [UInt32], lanes: [UInt32])] = [
        ("rgba8unorm", "vec4f", [0x00ff00ff], [1, 0, 1, 0].map(Float.init).map(\.bitPattern)),
        ("rgba8snorm", "vec4f", [0x007f0081], [-1, 0, 1, 0].map(Float.init).map(\.bitPattern)),
        ("rgba8uint", "vec4u", [0xff020180], [128, 1, 2, 255]),
        ("rgba8sint", "vec4i", [0xff020180], [0xffffff80, 1, 2, .max]),
        ("rgba16uint", "vec4u", [0x0001ffff, 0x80000002], [65535, 1, 2, 32768]),
        ("rgba16sint", "vec4i", [0x0001ffff, 0x80000002], [.max, 1, 2, 0xffff8000]),
        ("rgba16float", "vec4f", [0xc0003c00, 0x40003800], [1, -2, 0.5, 2].map(Float.init).map(\.bitPattern)),
        ("r32uint", "vec4u", [0xdeadbeef], [0xdeadbeef, 0, 0, 1]),
        ("r32sint", "vec4i", [0x80000005], [0x80000005, 0, 0, 1]),
        ("r32float", "vec4f", [0xc0000000], [-2, 0, 0, 1].map(Float.init).map(\.bitPattern)),
        ("rg32uint", "vec4u", [0xdeadbeef, 99], [0xdeadbeef, 99, 0, 1]),
        ("rg32sint", "vec4i", [0x80000005, 99], [0x80000005, 99, 0, 1]),
        ("rg32float", "vec4f", [0xc0000000, 0x3f000000], [-2, 0.5, 0, 1].map(Float.init).map(\.bitPattern)),
        ("rgba32uint", "vec4u", [0xdeadbeef, 1, 2, 3], [0xdeadbeef, 1, 2, 3]),
        ("rgba32sint", "vec4i", [0x80000005, 1, 2, 3], [0x80000005, 1, 2, 3]),
        ("rgba32float", "vec4f", [0xc0000000, 0x3f000000, 0x40000000, 0x3f800000], [-2, 0.5, 2, 1].map(Float.init).map(\.bitPattern)),
    ]
    let compiler = IndependentShaderCompiler()
    let upload = try await device.createBuffer(.init(size: 32, usage: [.copySource, .copyDestination]))
    let output = try await device.createBuffer(.init(size: 48, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 64, usage: [.copyDestination, .mapRead]))
    let region = TextureCopyRegion(width: 1, height: 1)
    var commands: [BufferCommand] = []
    var expected: [[UInt32]] = []
    for fixture in formats {
        let readWrite = fixture.name.hasPrefix("r32")
        let shader = try compiler.compileCompute(.wgsl("""
            @group(0) @binding(0) var input: texture_storage_2d<\(fixture.name), read>;
            @group(0) @binding(1) var output: texture_storage_2d<\(fixture.name), \(readWrite ? "read_write" : "write")>;
            @group(0) @binding(2) var<storage, read_write> result: array<\(fixture.vector), 3>;
            @compute @workgroup_size(1) fn run() {
                let xy = vec2i(0);
                let value = textureLoad(input, xy);
                result[0] = value;
                result[1] = textureLoad(input, vec2i(-1));
                \(readWrite ? "result[2] = textureLoad(output, xy);" : "result[2] = \(fixture.vector)(0);")
                textureStore(output, xy, value);
            }
            """), to: .wgsl)
        guard let binding = shader.storageTextures.first else {
            throw GPUError.backendFailure("Missing \(fixture.name) storage reflection")
        }
        let format = binding.textureFormat()
        let pipeline = try await device.createComputePipeline(shader)
        let source = try await device.createTexture(.init(width: 1, height: 1, format: format,
            usage: [.storageBinding, .copyDestination]))
        let destination = try await device.createTexture(.init(width: 1, height: 1, format: format,
            usage: [.storageBinding, .copySource]))
        commands += [
            .write(buffer: upload, offset: 16, words: fixture.packed),
            .copyBufferToTexture(source: upload, layout: .init(offset: 16), destination: source, region: region),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 2): output], workgroups: .init(x: 1),
                             textures: [.init(binding: 0): source, .init(binding: 1): destination]),
            .copy(source: output, destination: read, byteCount: 48), .read(buffer: read, byteCount: 48),
            .copyTextureToBuffer(source: destination, destination: read, bytesPerRow: 256),
            .read(buffer: read, byteCount: UInt64(fixture.packed.count) * 4),
        ]
        let alpha: UInt32 = fixture.vector == "vec4f" ? Float(1).bitPattern : 1
        expected += [fixture.lanes + [0, 0, 0, 0, 0, 0, 0, readWrite ? alpha : 0], fixture.packed]
    }
    let snapshots = try await device.submitCommands(commands).wait()
    guard snapshots.count == expected.count else {
        throw GPUError.backendFailure("Unexpected storage format snapshot count: \(snapshots.count)")
    }
    for (index, snapshot) in snapshots.enumerated() where snapshot != expected[index] {
        throw GPUError.backendFailure("Unexpected \(formats[index / 2].name) storage readback: \(snapshot)")
    }
    return formats.count
}

private func cubeTextureDiagnostic(_ device: Device) async throws -> [UInt32] {
    let compiler = IndependentShaderCompiler()
    let renderSource = ShaderInput.wgsl("""
        @vertex fn vs() -> @builtin(position) vec4f { return vec4f(0, 0, 0, 1); }
        @fragment fn fs() -> @location(0) vec4f { return vec4f(0); }
        """)
    let render = try await device.createRenderPipeline(
        vertex: compiler.compileRenderStage(renderSource, entryPoint: "vs", to: .wgsl),
        fragment: compiler.compileRenderStage(renderSource, entryPoint: "fs", to: .wgsl),
        colorFormat: .rgba8Unorm, depthState: .init())
    let pipeline = try await device.createComputePipeline(compiler.compileCompute(.wgsl("""
        @group(0) @binding(0) var cube: texture_cube<f32>;
        @group(0) @binding(1) var cubes: texture_cube_array<f32>;
        @group(0) @binding(2) var depth: texture_depth_cube;
        @group(0) @binding(3) var depths: texture_depth_cube_array;
        @group(0) @binding(4) var nearest: sampler;
        @group(0) @binding(5) var comparison: sampler_comparison;
        @group(0) @binding(6) var<storage, read_write> out: array<vec4f, 11>;
        @compute @workgroup_size(1) fn run() {
            let direction = vec3f(1, 0, 0);
            out[0] = textureSampleLevel(cube, nearest, direction, 0.0);
            out[1] = textureSampleLevel(cube, nearest, vec3f(-1, 0, 0), 0.0);
            out[2] = textureSampleLevel(cube, nearest, vec3f(0, 1, 0), 0.0);
            out[3] = textureSampleLevel(cube, nearest, vec3f(0, -1, 0), 0.0);
            out[4] = textureSampleLevel(cube, nearest, vec3f(0, 0, 1), 0.0);
            out[5] = textureSampleLevel(cube, nearest, vec3f(0, 0, -1), 0.0);
            out[6] = textureSampleGrad(cubes, nearest, direction, -5i, vec3f(0), vec3f(0));
            out[7] = textureSampleLevel(cubes, nearest, direction, 0xffffffffu, 0.0);
            out[8] = textureGather(2, cubes, nearest, direction, 1u);
            let size = textureDimensions(cubes, 0u);
            out[9] = vec4f(f32(size.x), f32(size.y), f32(textureNumLayers(cubes)), f32(textureNumLevels(cube)));
            out[10] = vec4f(textureSampleLevel(depth, nearest, direction, 0u),
                textureSampleCompareLevel(depth, comparison, direction, 0.5),
                textureSampleCompareLevel(depths, comparison, direction, -1i, 0.5),
                textureGatherCompare(depths, comparison, direction, 0xffffffffu, 0.5).x);
        }
        """), to: .wgsl))
    let drawSource = ShaderInput.wgsl("""
        @group(0) @binding(0) var cube: texture_cube<f32>;
        @group(0) @binding(1) var cubes: texture_cube_array<f32>;
        @group(0) @binding(2) var depths: texture_depth_cube_array;
        @group(0) @binding(3) var nearest: sampler;
        @group(0) @binding(4) var comparison: sampler_comparison;
        @vertex fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
            if index == 0u { return vec4f(-1, -1, 0, 1); }
            if index == 1u { return vec4f(3, -1, 0, 1); }
            return vec4f(-1, 3, 0, 1);
        }
        @fragment fn fs() -> @location(0) vec4f {
            let direction = vec3f(1, 0, 0);
            let a = textureSample(cube, nearest, direction);
            let b = textureSampleBias(cubes, nearest, direction, 1i, 0.25);
            let cmp = textureSampleCompare(depths, comparison, direction, 1i, 0.5);
            let raw = textureSample(depths, nearest, direction, 1i);
            return vec4f(a.b, b.b, cmp, raw);
        }
        """)
    let draw = try await device.createRenderPipeline(
        vertex: compiler.compileRenderStage(drawSource, entryPoint: "vs", to: .wgsl),
        fragment: compiler.compileRenderStage(drawSource, entryPoint: "fs", to: .wgsl), colorFormat: .rgba8Unorm)
    let color = try await device.createTexture(.init(width: 4, height: 4, format: .rgba8Unorm,
        usage: [.copyDestination, .textureBinding], mipLevelCount: 2, arrayLayerCount: 13))
    let depth = try await device.createTexture(.init(width: 4, height: 4, format: .depth32Float,
        usage: [.renderAttachment, .textureBinding], mipLevelCount: 2, arrayLayerCount: 13))
    let cube = TextureViewDescriptor(baseMipLevel: 1, dimension: .cube, baseArrayLayer: 7)
    let array = TextureViewDescriptor(baseMipLevel: 1, dimension: .cubeArray, baseArrayLayer: 1)
    let singleColor = try await device.createTextureView(texture: color, cube)
    let arrayColor = try await device.createTextureView(texture: color, array)
    let singleDepth = try await device.createTextureView(texture: depth, cube)
    let arrayDepth = try await device.createTextureView(texture: depth, array)
    let firstFace = try await device.createTextureView(texture: depth, .init(baseMipLevel: 1, dimension: .d2, baseArrayLayer: 1))
    let lastFace = try await device.createTextureView(texture: depth, .init(baseMipLevel: 1, dimension: .d2, baseArrayLayer: 7))
    do {
        _ = try await device.createTextureView(texture: color, .init(dimension: .cubeArray))
        throw GPUError.backendFailure("Accepted an invalid cube view")
    } catch GPUError.validation {}
    do {
        _ = try await device.createTextureView(texture: color, .init(dimension: .cube, arrayLayerCount: 5))
        throw GPUError.backendFailure("Accepted an invalid cube view")
    } catch GPUError.validation {}
    let rectangular = try await device.createTexture(.init(width: 4, height: 2, format: .rgba8Unorm,
        usage: .textureBinding, arrayLayerCount: 6))
    do {
        _ = try await device.createTextureView(texture: rectangular, .init(dimension: .cube))
        throw GPUError.backendFailure("Accepted an invalid cube view")
    } catch GPUError.validation {}
    let target = try await device.createTexture(.init(width: 2, height: 2, format: .rgba8Unorm,
        usage: [.renderAttachment, .copySource]))
    let nearest = try await device.createSampler(.init())
    let compare = try await device.createSampler(.init(compare: .less))
    let output = try await device.createBuffer(.init(size: 176, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 264, usage: [.copyDestination, .mapRead]))
    let upload = try await device.createBuffer(.init(size: 7 * 512, usage: [.copySource, .copyDestination]))
    let faces: [(UInt32, UInt32)] = [(1, 0xff0000ff), (7, 0xffff0000), (8, 0xff0000ff), (9, 0xff00ff00),
                                  (10, 0xff00ffff), (11, 0xffffff00), (12, 0xffff00ff)]
    var commands: [BufferCommand] = []
    for (index, (layer, texel)) in faces.enumerated() {
        let offset = UInt64(index) * 512
        commands += [.write(buffer: upload, offset: offset, words: [texel, texel]),
                     .write(buffer: upload, offset: offset + 256, words: [texel, texel]),
                     .copyBufferToTexture(source: upload, layout: .init(offset: offset, bytesPerRow: 256),
                                          destination: color, region: .init(width: 2, height: 2, mipLevel: 1, arrayLayer: layer))]
    }
    func dispatch(_ cube: ResourceID) -> BufferCommand {
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 6): output], workgroups: .init(x: 1),
                         textures: [.init(binding: 0): cube, .init(binding: 1): arrayColor,
                                    .init(binding: 2): singleDepth, .init(binding: 3): arrayDepth],
                         samplers: [.init(binding: 4): nearest, .init(binding: 5): compare])
    }
    do {
        _ = try await device.submitCommands([dispatch(arrayColor)])
        throw GPUError.backendFailure("Accepted a cube array view for a cube binding")
    } catch GPUError.validation("Invalid sampled texture binding") {}
    commands += [
        .render(pipeline: render, target: target, clear: .transparentBlack, vertexCount: 0,
                depth: .init(texture: firstFace, loadAction: .clear(0.25))),
        .render(pipeline: render, target: target, clear: .transparentBlack, vertexCount: 0,
                depth: .init(texture: lastFace, loadAction: .clear(0.75))),
        dispatch(singleColor), .copy(source: output, destination: read, byteCount: 176), .read(buffer: read, byteCount: 176),
        .render(pipeline: draw, target: target, clear: .transparentBlack, vertexCount: 3,
                textures: [.init(binding: 0): singleColor, .init(binding: 1): arrayColor, .init(binding: 2): arrayDepth],
                samplers: [.init(binding: 3): nearest, .init(binding: 4): compare]),
        .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256), .read(buffer: read, byteCount: 8),
    ]
    let expected: [Float] = [0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1,
                             1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 0.75, 1, 0, 1]
    let ticket = try await device.submitCommands(commands)
    let snapshots = try await ticket.wait()
    guard snapshots == [expected.map(\.bitPattern), [0xbfffffff, 0xbfffffff]] else {
        throw GPUError.backendFailure("Unexpected cube texture readback: \(snapshots)")
    }
    return snapshots.flatMap { $0 }
}

private func volumeTextureDiagnostic(_ device: Device) async throws -> [UInt32] {
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl("""
        @group(0) @binding(0) var volume: texture_3d<f32>;
        @group(0) @binding(1) var nearest: sampler;
        @group(0) @binding(2) var<storage, read_write> result: array<vec4f, 11>;
        @compute @workgroup_size(1) fn run() {
            result[0] = textureLoad(volume, vec3i(0, 0, 0), 0i);
            result[1] = textureLoad(volume, vec3u(0, 0, 1), 0u);
            result[2] = textureLoad(volume, vec3i(0, 0, 2), 0i);
            result[3] = textureLoad(volume, vec3i(0, 0, -1), 0i);
            result[4] = textureLoad(volume, vec3u(0, 0, 0xffffffffu), 0i);
            result[5] = textureSampleLevel(volume, nearest, vec3f(0.25, 0.5, 0.5), 0.0);
            result[6] = textureSampleGrad(volume, nearest, vec3f(0.25, 0.5, 0.16666667), vec3f(0), vec3f(0), vec3i(0, 0, 1));
            let size = textureDimensions(volume, 0u);
            result[7] = vec4f(f32(size.x), f32(size.y), f32(size.z), f32(textureNumLevels(volume)));
            let invalid = textureDimensions(volume, 1u);
            result[8] = vec4f(f32(invalid.x), f32(invalid.y), f32(invalid.z), textureLoad(volume, vec3u(0), -1i).a);
            result[10] = textureSampleLevel(volume, nearest, vec3f(0.25, 0.5, 1.5), 0.0);
            result[9] = textureSampleLevel(volume, nearest, vec3f(0.25, 0.5, 0.5), 0.0, vec3i(0, 0, -1));
        }
        """), to: .wgsl)
    let pipeline = try await device.createComputePipeline(shader)
    let texture = TextureDescriptor(width: 4, height: 2, format: .rgba8Unorm,
        usage: [.copySource, .copyDestination, .textureBinding], mipLevelCount: 2, dimension: .d3, depth: 6)
    let source = try await device.createTexture(texture)
    let volume = try await device.createTexture(texture)
    let arraySource = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm,
        usage: [.copySource, .copyDestination], arrayLayerCount: 4))
    let arrayDestination = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm,
        usage: [.copySource, .copyDestination], arrayLayerCount: 4))
    let mixedVolume = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm,
        usage: [.copySource, .copyDestination], dimension: .d3, depth: 4))
    let view = try await device.createTextureView(texture: volume, .init(baseMipLevel: 1))
    do {
        _ = try await device.createTextureView(texture: volume, .init(dimension: .d2))
        throw GPUError.backendFailure("Accepted a 2D view of a volume texture")
    } catch GPUError.validation("Texture view dimension does not match the texture") {}
    let nearest = try await device.createSampler(.init())
    let repeating = try await device.createSampler(.init(addressModeW: .repeat))
    let mirroring = try await device.createSampler(.init(addressModeW: .mirrorRepeat))
    let upload = try await device.createBuffer(.init(size: 264, usage: [.copySource, .copyDestination]))
    let out = try await device.createBuffer(.init(size: 176, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 772, usage: [.copyDestination, .mapRead]))
    let region = TextureCopyRegion(width: 2, height: 1, mipLevel: 1, depth: 2)
    do {
        _ = try await device.submitCommands([.copyTextureToBuffer(source: volume, destination: read,
            bytesPerRow: 256, mipLevel: 1, z: 2, depth: 2)])
        throw GPUError.backendFailure("Accepted volume copy outside the mip depth")
    } catch GPUError.validation("Invalid texture copy source or aspect") {}
    let ticket = try await device.submitCommands([
        .write(buffer: upload, offset: 0, words: [0xff0000ff, 0xff0000ff]),
        .write(buffer: upload, offset: 256, words: [0xff00ff00, 0xff00ff00]),
        .copyBufferToTexture(source: upload, layout: .init(bytesPerRow: 256), destination: source, region: region),
        .copyTextureToTexture(source: source, destination: volume, sourceRegion: region, destinationOrigin: .init(mipLevel: 1)),
        .copyTextureToTexture(source: source, destination: volume,
            sourceRegion: .init(width: 1, height: 1, mipLevel: 1, z: 1), destinationOrigin: .init(x: 1, mipLevel: 1, z: 2)),
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 2): out], workgroups: .init(x: 1),
                         textures: [.init(binding: 0): view], samplers: [.init(binding: 1): nearest]),
        .copy(source: out, destination: read, byteCount: 176), .read(buffer: read, byteCount: 176),
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 2): out], workgroups: .init(x: 1),
                         textures: [.init(binding: 0): view], samplers: [.init(binding: 1): repeating]),
        .copy(source: out, destination: read, byteCount: 176), .read(buffer: read, byteCount: 176),
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 2): out], workgroups: .init(x: 1),
                         textures: [.init(binding: 0): view], samplers: [.init(binding: 1): mirroring]),
        .copy(source: out, destination: read, byteCount: 176), .read(buffer: read, byteCount: 176),
        .copyTextureToBuffer(source: volume, destination: read, bytesPerRow: 256, mipLevel: 1, depth: 3),
        .read(buffer: read, byteCount: 520),
        .writeTexture(destination: arraySource, data: [11, 0, 0, 255, 21, 0, 0, 255, 31, 0, 0, 255, 41, 0, 0, 255],
                      layout: .init(bytesPerRow: 4, rowsPerImage: 1), region: .init(width: 1, height: 1, depth: 4)),
        .writeTexture(destination: mixedVolume, data: [51, 0, 0, 255, 52, 0, 0, 255, 53, 0, 0, 255, 54, 0, 0, 255],
                      layout: .init(bytesPerRow: 4, rowsPerImage: 1), region: .init(width: 1, height: 1, depth: 4)),
        .writeTexture(destination: arrayDestination, data: [61, 0, 0, 255, 62, 0, 0, 255, 63, 0, 0, 255, 64, 0, 0, 255],
                      layout: .init(bytesPerRow: 4, rowsPerImage: 1), region: .init(width: 1, height: 1, depth: 4)),
        .copyTextureToTexture(source: arraySource, destination: mixedVolume,
                              sourceRegion: .init(width: 1, height: 1, arrayLayer: 1, depth: 2),
                              destinationOrigin: .init(z: 1)),
        .copyTextureToBuffer(source: mixedVolume, destination: read, bytesPerRow: 256, depth: 4),
        .read(buffer: read, byteCount: 772),
        .copyTextureToTexture(source: mixedVolume, destination: arrayDestination,
                              sourceRegion: .init(width: 1, height: 1, z: 1, depth: 2),
                              destinationOrigin: .init(arrayLayer: 1)),
        .copyTextureToBuffer(source: arrayDestination, destination: read, bytesPerRow: 256, depth: 4),
        .read(buffer: read, byteCount: 772),
        .writeTexture(destination: mixedVolume, data: [51, 0, 0, 255, 52, 0, 0, 255, 53, 0, 0, 255, 54, 0, 0, 255],
                      layout: .init(bytesPerRow: 4, rowsPerImage: 1), region: .init(width: 1, height: 1, depth: 4)),
        .copyTextureToTexture(source: mixedVolume, destination: mixedVolume,
                              sourceRegion: .init(width: 1, height: 1), destinationOrigin: .init(z: 2)),
        .copyTextureToBuffer(source: mixedVolume, destination: read, bytesPerRow: 256, depth: 4),
        .read(buffer: read, byteCount: 772),
    ])
    let snapshots = try await ticket.wait()
    let expected: [Float] = [1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 1, 0, 1, 0, 1, 0, 1, 2, 1, 3, 1, 0, 0, 0, 0, 1, 0, 0, 1]
    let planes = [0, 1, 64, 65, 128, 129].map { snapshots[3][$0] }
    guard Array(snapshots[0].prefix(40)) == expected.map(\.bitPattern),
          Array(snapshots[0].suffix(4)) == [0, 0, 0, 0],
          Array(snapshots[1].suffix(4)) == [0, 1, 0, 1].map({ Float($0).bitPattern }),
          Array(snapshots[2].suffix(4)) == [0, 1, 0, 1].map({ Float($0).bitPattern }),
          planes == [0xff0000ff, 0xff0000ff, 0xff00ff00, 0xff00ff00, 0, 0xff00ff00],
          [0, 64, 128, 192].map({ snapshots[4][$0] }) == [0xff000033, 0xff000015, 0xff00001f, 0xff000036],
          [0, 64, 128, 192].map({ snapshots[5][$0] }) == [0xff00003d, 0xff000015, 0xff00001f, 0xff000040],
          [0, 64, 128, 192].map({ snapshots[6][$0] }) == [0xff000033, 0xff000034, 0xff000033, 0xff000036] else {
        throw GPUError.backendFailure("Unexpected volume texture readback: \(snapshots[0]), \(planes)")
    }
    return Array(snapshots[0].prefix(40)) + planes
}

private func volumeStorageDiagnostic(_ device: Device) async throws -> [UInt32] {
    let compiler = IndependentShaderCompiler()
    let compute = try await device.createComputePipeline(compiler.compileCompute(.wgsl("""
        @group(0) @binding(0) var source: texture_storage_3d<r32uint, read>;
        @group(0) @binding(1) var image: texture_storage_3d<r32uint, read_write>;
        @group(0) @binding(2) var<storage, read_write> result: array<vec4u, 5>;
        @compute @workgroup_size(1) fn run() {
            let input = textureLoad(source, vec3u(0, 0, 2));
            let previous = textureLoad(image, vec3i(0, 0, 1));
            result[0] = vec4u(textureDimensions(image), input.x);
            result[1] = previous;
            result[2] = textureLoad(source, vec3u(0, 0, 4));
            result[3] = textureLoad(image, vec3i(0, 0, -1));
            result[4] = textureLoad(image, vec3u(0, 0, 0xffffffffu));
            textureStore(image, vec3i(0, 0, 1), previous + input);
            textureStore(image, vec3u(1, 0, 0), input + vec4u(1u));
            textureStore(image, vec3i(0, 0, -1), vec4u(99u));
            textureStore(image, vec3u(0, 0, 2), vec4u(99u));
            textureStore(image, vec3u(0, 0, 0xffffffffu), vec4u(99u));
        }
        """), to: .wgsl))
    let shader = ShaderInput.wgsl("""
        @group(1) @binding(3) var volume: texture_storage_3d<rgba8unorm, write>;
        @vertex fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
            if index == 0u { return vec4f(-1, -1, 0, 1); }
            if index == 1u { return vec4f(3, -1, 0, 1); }
            return vec4f(-1, 3, 0, 1);
        }
        @fragment fn fs() -> @location(0) vec4f {
            textureStore(volume, vec3u(0, 0, 1), vec4f(1, 0, 1, 1));
            return vec4f(0, 1, 0, 1);
        }
        """)
    let draw = try await device.createRenderPipeline(
        vertex: compiler.compileRenderStage(shader, entryPoint: "vs", to: .wgsl),
        fragment: compiler.compileRenderStage(shader, entryPoint: "fs", to: .wgsl), colorFormat: .rgba8Unorm)
    let volume = try await device.createTexture(.init(width: 4, height: 2, format: .r32Uint,
        usage: [.storageBinding, .copySource, .copyDestination], mipLevelCount: 2, dimension: .d3, depth: 4))
    let source = try await device.createTextureView(texture: volume, .init(mipLevelCount: 1))
    let image = try await device.createTextureView(texture: volume, .init(baseMipLevel: 1))
    let plane = try await device.createTexture(.init(width: 2, height: 1, format: .r32Uint, usage: .storageBinding))
    let color = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm,
        usage: [.storageBinding, .copySource], dimension: .d3, depth: 2))
    let target = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm, usage: .renderAttachment))
    let upload = try await device.createBuffer(.init(size: 12, usage: [.copyDestination, .copySource]))
    let output = try await device.createBuffer(.init(size: 80, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 80, usage: [.mapRead, .copyDestination]))
    func dispatch(_ input: ResourceID, _ outputImage: ResourceID) -> BufferCommand {
        .dispatchCompute(pipeline: compute, bindings: [.init(binding: 2): output], workgroups: .init(x: 1),
                         textures: [.init(binding: 0): input, .init(binding: 1): outputImage])
    }
    do {
        _ = try await device.submitCommands([dispatch(source, plane)])
        throw GPUError.backendFailure("Accepted a 2D view for a 3D storage binding")
    } catch GPUError.validation("Storage texture view dimension does not match the shader binding") {}
    do {
        _ = try await device.submitCommands([dispatch(volume, image)])
        throw GPUError.backendFailure("Accepted a multi-mip storage binding")
    } catch GPUError.validation("Storage texture binding requires one mip level of the declared format") {}
    do {
        _ = try await device.submitCommands([dispatch(image, image)])
        throw GPUError.backendFailure("Accepted overlapping writable storage mips")
    } catch GPUError.validation("Aliased writable compute textures") {}
    let ticket = try await device.submitCommands([
        .write(buffer: upload, offset: 0, words: [7, 20, 21]),
        .copyBufferToTexture(source: upload, layout: .init(), destination: volume,
                             region: .init(width: 1, height: 1, z: 2)),
        .copyBufferToTexture(source: upload, layout: .init(offset: 4), destination: volume,
                             region: .init(width: 2, height: 1, mipLevel: 1, z: 1)),
        dispatch(source, image),
        .copy(source: output, destination: read, byteCount: 80), .read(buffer: read, byteCount: 80),
        .copyTextureToBuffer(source: volume, destination: read, bytesPerRow: 256, mipLevel: 1), .read(buffer: read, byteCount: 8),
        .copyTextureToBuffer(source: volume, destination: read, bytesPerRow: 256, mipLevel: 1, z: 1), .read(buffer: read, byteCount: 8),
        .render(pipeline: draw, target: target, clear: .transparentBlack, vertexCount: 3,
                textures: [.init(group: 1, binding: 3): color]),
        .copyTextureToBuffer(source: color, destination: read, bytesPerRow: 256), .read(buffer: read, byteCount: 4),
        .copyTextureToBuffer(source: color, destination: read, bytesPerRow: 256, z: 1), .read(buffer: read, byteCount: 4),
    ])
    let expected: [[UInt32]] = [[2, 1, 2, 7, 20, 0, 0, 1] + Array(repeating: 0, count: 12),
                                [0, 8], [27, 21], [0], [0xffff00ff]]
    let snapshots = try await ticket.wait()
    guard snapshots == expected else {
        throw GPUError.backendFailure("Unexpected volume storage readback: \(snapshots)")
    }
    return snapshots.flatMap { $0 }
}

private func volumeRenderDiagnostic(_ device: Device) async throws -> [UInt32] {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl("""
        @group(0) @binding(0) var input: texture_3d<f32>;
        struct FragmentOutput {
            @location(0) red: vec4f,
            @location(2) green: vec4f,
        }
        @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
            if i == 0u { return vec4f(-1, -1, 0, 1); }
            if i == 1u { return vec4f(3, -1, 0, 1); }
            return vec4f(-1, 3, 0, 1);
        }
        @fragment fn fs() -> FragmentOutput {
            let red = textureLoad(input, vec3i(0), 0i).r;
            return FragmentOutput(vec4f(red, 0, 0, 1), vec4f(0, 1, 0, 1));
        }
        """)
    let pipeline = try await device.createRenderPipeline(
        vertex: compiler.compileRenderStage(source, entryPoint: "vs", to: .wgsl),
        fragment: compiler.compileRenderStage(source, entryPoint: "fs", to: .wgsl),
        colorFormat: .rgba8Unorm, colorState: .init(writeMask: .red),
        additionalColorTargets: [2: .init(format: .rgba8Unorm, colorState: .init(writeMask: .green))])
    let texture = try await device.createTexture(.init(width: 4, height: 2, format: .rgba8Unorm,
        usage: [.renderAttachment, .textureBinding, .copySource, .copyDestination], mipLevelCount: 2,
        dimension: .d3, depth: 6))
    let input = try await device.createTextureView(texture: texture, .init(mipLevelCount: 1))
    let output = try await device.createTextureView(texture: texture, .init(baseMipLevel: 1))
    let multiMip = try await device.createTextureView(texture: texture, .init())
    let flat = try await device.createTexture(.init(width: 2, height: 1, format: .rgba8Unorm, usage: .renderAttachment))
    let upload = try await device.createBuffer(.init(size: 12, usage: [.copySource, .copyDestination]))
    let indices = try await device.createBuffer(.init(size: 12, usage: [.index, .copyDestination]))
    let read = try await device.createBuffer(.init(size: 520, usage: [.copyDestination, .mapRead]))
    let firstColors: [UInt32: RenderColorAttachment] = [
        2: .init(texture: output, loadAction: .clear(.init(red: 0.25, green: 0, blue: 0, alpha: 1)), depthSlice: 1),
    ]
    func draw(target: ResourceID, slice: UInt32?, sampled: ResourceID, colors: [UInt32: RenderColorAttachment]) -> BufferCommand {
        .render(pipeline: pipeline, target: target, colorLoad: .load, vertexCount: 3,
                textures: [.init(binding: 0): sampled], targetDepthSlice: slice, additionalColors: colors)
    }
    for invalid in [draw(target: output, slice: nil, sampled: input, colors: firstColors),
                    draw(target: output, slice: 3, sampled: input, colors: firstColors),
                    draw(target: flat, slice: 0, sampled: input, colors: firstColors),
                    draw(target: multiMip, slice: 0, sampled: input, colors: firstColors)] {
        do {
            _ = try await device.submitCommands([invalid])
            throw GPUError.backendFailure("Accepted an invalid volume color attachment")
        } catch GPUError.validation("Invalid render pipeline, color attachment or clear value") {}
    }
    do {
        _ = try await device.submitCommands([draw(target: output, slice: 1, sampled: input, colors: firstColors)])
        throw GPUError.backendFailure("Accepted the same volume slice twice")
    } catch GPUError.validation("Aliased render attachments") {}
    do {
        _ = try await device.submitCommands([draw(target: output, slice: 0, sampled: output, colors: firstColors)])
        throw GPUError.backendFailure("Accepted sampling an attached volume mip")
    } catch GPUError.validation("Render attachment cannot be bound as a sampled texture") {}
    let ticket = try await device.submitCommands([
        .write(buffer: upload, offset: 0, words: [0xff0000ff, 0xff00ffff, 0xff00ffff]),
        .write(buffer: indices, offset: 0, words: [0, 1, 2]),
        .copyBufferToTexture(source: upload, layout: .init(), destination: texture, region: .init(width: 1, height: 1)),
        .copyBufferToTexture(source: upload, layout: .init(offset: 4), destination: texture,
                             region: .init(width: 2, height: 1, mipLevel: 1, z: 2)),
        draw(target: output, slice: 0, sampled: input, colors: firstColors),
        .copyTextureToBuffer(source: texture, destination: read, bytesPerRow: 256, mipLevel: 1, depth: 3),
        .read(buffer: read, byteCount: 520),
        .renderIndexed(pipeline: pipeline, target: texture, colorLoad: .load,
                       indexBuffer: .init(buffer: indices, format: .uint32), draw: .init(indexCount: 3),
                       textures: [.init(binding: 0): input], targetMipLevel: 1, targetDepthSlice: 1,
                       additionalColors: [2: .init(texture: output, depthSlice: 0)],
                       raster: .init(scissor: .init(x: 0, y: 0, width: 1, height: 1))),
        .copyTextureToBuffer(source: texture, destination: read, bytesPerRow: 256, mipLevel: 1, depth: 3),
        .read(buffer: read, byteCount: 520),
    ])
    let snapshots = try await ticket.wait()
    let planes = snapshots.map { words in [0, 1, 64, 65, 128, 129].map { words[$0] } }
    guard planes == [[0xff, 0xff, 0xff00ff40, 0xff00ff40, 0xff00ffff, 0xff00ffff],
                       [0xffff, 0xff, 0xff00ffff, 0xff00ff40, 0xff00ffff, 0xff00ffff]] else {
        throw GPUError.backendFailure("Unexpected volume color attachment readback: \(planes)")
    }
    return planes.flatMap { $0 }
}

private func lineTextureDiagnostic(_ device: Device) async throws -> [UInt32] {
    let compiler = IndependentShaderCompiler()
    let compute = try await device.createComputePipeline(compiler.compileCompute(.wgsl("""
        @group(0) @binding(0) var line: texture_1d<f32>;
        @group(0) @binding(1) var nearest: sampler;
        @group(0) @binding(2) var readLine: texture_storage_1d<r32sint, read>;
        @group(0) @binding(3) var writeLine: texture_storage_1d<r32sint, read_write>;
        @group(0) @binding(4) var<storage, read_write> result: array<vec4f, 4>;
        @compute @workgroup_size(1) fn run() {
            result[0] = textureLoad(line, 0i, 0i);
            result[1] = textureSampleLevel(line, nearest, 0.75, 3.0);
            let previous = textureLoad(writeLine, 1u);
            let input = textureLoad(readLine, 0u);
            result[2] = vec4f(f32(textureDimensions(line)), f32(textureNumLevels(line)),
                             f32(textureDimensions(writeLine)), f32(previous.x + input.x));
            result[3] = vec4f(textureLoad(line, -1i, 0i).a, textureLoad(line, 0xffffffffu, 0u).a,
                             f32(textureDimensions(line, 1u)), f32(textureLoad(readLine, 2u).a));
            textureStore(writeLine, 1i, previous + input);
            textureStore(writeLine, -1i, vec4i(99));
            textureStore(writeLine, 0xffffffffu, vec4i(99));
        }
        """), to: .wgsl))
    let source = ShaderInput.wgsl("""
        @group(0) @binding(0) var line: texture_1d<f32>;
        @group(0) @binding(1) var nearest: sampler;
        @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
            if i == 0u { return vec4f(-1, -1, 0, 1); }
            if i == 1u { return vec4f(3, -1, 0, 1); }
            return vec4f(-1, 3, 0, 1);
        }
        @fragment fn fs() -> @location(0) vec4f { return textureSample(line, nearest, 0.75); }
        """)
    let draw = try await device.createRenderPipeline(
        vertex: compiler.compileRenderStage(source, entryPoint: "vs", to: .wgsl),
        fragment: compiler.compileRenderStage(source, entryPoint: "fs", to: .wgsl), colorFormat: .rgba8Unorm)
    let line = try await device.createTexture(.init(width: 2, height: 1, format: .rgba8Unorm,
        usage: [.textureBinding, .copyDestination], dimension: .d1))
    let view = try await device.createTextureView(texture: line, .init())
    do {
        _ = try await device.createTextureView(texture: line, .init(dimension: .d2))
        throw GPUError.backendFailure("Accepted a 2D view of a 1D texture")
    } catch GPUError.validation("Texture view dimension does not match the texture") {}
    do {
        _ = try await device.createTexture(.init(width: 2, height: 1, format: .rgba8Unorm,
            usage: .textureBinding, mipLevelCount: 2, dimension: .d1))
        throw GPUError.backendFailure("Accepted multiple mips for a 1D texture")
    } catch GPUError.validation("Invalid 1D texture extent, mip levels, format or usage") {}
    let desc = TextureDescriptor(width: 2, height: 1, format: .r32Sint,
        usage: [.storageBinding, .copyDestination, .copySource], dimension: .d1)
    let readLine = try await device.createTexture(desc)
    let writeLine = try await device.createTexture(desc)
    let target = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm,
        usage: [.renderAttachment, .copySource]))
    let sampler = try await device.createSampler(.init())
    let upload = try await device.createBuffer(.init(size: 16, usage: [.copySource, .copyDestination]))
    let output = try await device.createBuffer(.init(size: 64, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 64, usage: [.mapRead, .copyDestination]))
    let ticket = try await device.submitCommands([
        .write(buffer: upload, offset: 0, words: [0xff0000ff, 0xff00ff00, 5, UInt32(bitPattern: -3)]),
        .copyBufferToTexture(source: upload, layout: .init(), destination: line, region: .init(width: 2, height: 1)),
        .copyBufferToTexture(source: upload, layout: .init(offset: 8), destination: readLine, region: .init(width: 1, height: 1)),
        .copyBufferToTexture(source: upload, layout: .init(offset: 12), destination: writeLine,
                             region: .init(x: 1, width: 1, height: 1)),
        .dispatchCompute(pipeline: compute, bindings: [.init(binding: 4): output], workgroups: .init(x: 1),
                         textures: [.init(binding: 0): view, .init(binding: 2): readLine, .init(binding: 3): writeLine],
                         samplers: [.init(binding: 1): sampler]),
        .copy(source: output, destination: read, byteCount: 64), .read(buffer: read, byteCount: 64),
        .copyTextureToBuffer(source: writeLine, destination: read, bytesPerRow: 256), .read(buffer: read, byteCount: 8),
        .render(pipeline: draw, target: target, clear: .transparentBlack, vertexCount: 3,
                textures: [.init(binding: 0): view], samplers: [.init(binding: 1): sampler]),
        .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256), .read(buffer: read, byteCount: 4),
    ])
    let expected: [Float] = [1, 0, 0, 1, 0, 1, 0, 1, 2, 1, 2, 2, 0, 0, 0, 0]
    let snapshots = try await ticket.wait()
    guard snapshots == [expected.map(\.bitPattern), [0, 2], [0xff00ff00]] else {
        throw GPUError.backendFailure("Unexpected 1D texture readback: \(snapshots)")
    }
    return snapshots.flatMap { $0 }
}

private func bitcastDiagnostic(_ device: Device) async throws -> [UInt32] {

    let shader = try IndependentShaderCompiler().compileCompute(.wgsl("""
        alias Words = vec4<u32>;
        const FLOATS = bitcast<vec4f>(vec4u(0x3f800000u, 0xc0200000u, 0x80000000u, 0x3f000000u));
        const PACKED: Words = bitcast<Words>(FLOATS);
        const_assert bitcast<u32>(4294967295) == 0xffffffffu;
        const_assert bitcast<vec2u>(vec2(1, 4294967295)).y == 0xffffffffu;
        const_assert bitcast<u32>(bitcast<f32>(-2147483648)) == 0x80000000u;
        @group(0) @binding(0) var<storage, read> input: array<vec4u, 1>;
        @group(0) @binding(1) var<storage, read_write> output: array<vec4u, 5>;
        @compute @workgroup_size(1) fn run() {
            let words = input[0];
            let floats = bitcast<vec4<f32>>(words);
            output[0] = bitcast<vec4u>(floats);
            output[1] = bitcast<vec4u>(bitcast<vec4i>(floats));
            output[2] = bitcast<vec4u>(bitcast<vec4f>(bitcast<vec4i>(words)));
            let pairs = bitcast<vec2u>(bitcast<vec2f>(words.xy));
            let triple = bitcast<vec3u>(bitcast<vec3f>(words.yzw));
            output[3] = vec4u(pairs, triple.z, bitcast<u32>(floats.x + 1.0));
            const high = bitcast<u32>(-1i);
            output[4] = vec4u(PACKED.xy, bitcast<u32>(-0.0f), high);
        }
        """), to: .wgsl)
    let pipeline = try await device.createComputePipeline(shader)
    let input = try await device.createBuffer(.init(size: 16, usage: [.storage, .copyDestination]))
    let output = try await device.createBuffer(.init(size: 80, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 80, usage: [.mapRead, .copyDestination]))
    let words: [UInt32] = [0x3f800000, 0xc0200000, 0x3f000000, 0x00800000]
    let ticket = try await device.submitCommands([
        .write(buffer: input, offset: 0, words: words),
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): input, .init(binding: 1): output],
                         workgroups: .init(x: 1)),
        .copy(source: output, destination: read, byteCount: 80), .read(buffer: read, byteCount: 80),
    ])
    let expected = words + words + words + [words[0], words[1], words[3], 0x40000000] +
        [0x3f800000, 0xc0200000, 0x80000000, 0xffffffff]
    let snapshots = try await ticket.wait()
    guard snapshots == [expected] else {
        throw GPUError.backendFailure("Unexpected bitcast readback: \(snapshots)")
    }
    return snapshots[0]
}

private func integerBitDiagnostic(_ device: Device) async throws -> [UInt32] {

    let shader = try IndependentShaderCompiler().compileCompute(.wgsl("""
        const edges = vec4u(countLeadingZeros(0u), countTrailingZeros(0u), countOneBits(0xffffffffu), reverseBits(1u));
        const_assert firstLeadingBit(-1i) == -1i;
        const_assert firstLeadingBit(-2147483647i - 1i) == 30i;
        const_assert firstTrailingBit(0u) == 0xffffffffu;
        const_assert extractBits(vec2i(-1, 8), 0u, 4u).y == -8i;
        const_assert insertBits(7u, 9u, 32u, 0u) == 7u;
        const_assert insertBits(vec2(0, 1), vec2u(3, 2), 28u, 4u).y == 0x20000001u;
        @group(0) @binding(0) var<storage, read> input: array<vec4u, 2>;
        @group(0) @binding(1) var<storage, read_write> output: array<vec4u, 14>;
        @compute @workgroup_size(1) fn run() {
            let bits = input[0];
            let signed = bitcast<vec4i>(bits);
            let ranges = input[1];
            output[0] = countLeadingZeros(bits);
            output[1] = bitcast<vec4u>(countTrailingZeros(signed));
            output[2] = countOneBits(bits);
            output[3] = bitcast<vec4u>(reverseBits(signed));
            output[4] = firstLeadingBit(bits);
            output[5] = bitcast<vec4u>(firstLeadingBit(signed));
            output[6] = bitcast<vec4u>(firstTrailingBit(signed));
            output[7] = extractBits(bits, ranges.x, ranges.y);
            output[8] = bitcast<vec4u>(extractBits(signed, ranges.x, ranges.y));
            output[9] = insertBits(bits, vec4u(5u), ranges.x, ranges.y);
            output[10] = bitcast<vec4u>(extractBits(signed, 0u, ranges.w));
            output[11] = bitcast<vec4u>(insertBits(signed, vec4i(-1), ranges.z, ranges.y));
            output[12] = vec4u(extractBits(bits.w, ranges.z, ranges.y), firstTrailingBit(bits.x),
                               bitcast<u32>(countLeadingZeros(signed.z)), insertBits(bits.x, 7, 0u, ranges.w));
            output[13] = edges;
        }
        """), to: .wgsl)
    let pipeline = try await device.createComputePipeline(shader)
    let booleanSource = ShaderInput.wgsl("""
        const_assert all((vec2(true, false) & vec2(true, true)) == vec2(true, false));
        const_assert all(!(vec2(false, true) | vec2(false, false)) == vec2(true, false));

        @group(0) @binding(0) var<storage, read_write> output: array<vec4u, 6>;
        var<private> trace: u32;

        fn mark(tag: u32, value: bool) -> bool {
            trace = trace * 10u + tag;
            return value;
        }

        fn markVector(tag: u32, value: vec4<bool>) -> vec4<bool> {
            trace = trace * 10u + tag;
            return value;
        }

        @compute @workgroup_size(1) fn run() {
            let both = mark(1u, false) & mark(2u, true);
            let either = mark(3u, true) | mark(4u, false);
            let short = (false && mark(9u, true)) || (true || mark(9u, false));
            output[0] = vec4u(u32(both), u32(either), trace, u32(short));
            let lhs = vec4(true, true, false, false);
            let rhs = vec4(true, false, true, false);
            output[1] = select(vec4u(0), vec4u(1), markVector(5u, lhs) & markVector(6u, rhs));
            output[2] = select(vec4u(0), vec4u(1), !(markVector(7u, lhs) | markVector(8u, rhs)));
            output[3] = select(vec4u(0), vec4u(1), lhs == rhs);
            output[4] = select(vec4u(0), vec4u(1), lhs != rhs);
            var combined = lhs;
            combined &= rhs;
            combined |= vec4(false, false, false, true);
            output[5] = vec4u(trace, u32(combined.x), u32(combined.y), u32(combined.w));
        }
        """)
    let boolean = try IndependentShaderCompiler().compileCompute(booleanSource, to: .wgsl)
    let booleanPipeline = try await device.createComputePipeline(boolean)
    let composite = try IndependentShaderCompiler().compileWebGPUCompute(.spirv(vectorCompositeSPIRV()))
    let compositePipeline = try await device.createComputePipeline(composite)
    let bitInstructions = try IndependentShaderCompiler().compileWebGPUCompute(.spirv(bitInstructionsSPIRV()))
    let bitPipeline = try await device.createComputePipeline(bitInstructions)
    let shiftInstructions = try IndependentShaderCompiler().compileWebGPUCompute(.spirv(shiftInstructionsSPIRV()))
    let shiftPipeline = try await device.createComputePipeline(shiftInstructions)
    let roundTripSource = ShaderInput.wgsl("""
        enable wgpu_int16;
        @group(0) @binding(0) var<storage, read> input: array<vec4u, 2>;
        @group(0) @binding(1) var<storage, read_write> output: array<vec4u, 13>;
        @group(0) @binding(2) var<storage, read_write> narrowOutput: u32;
        @group(0) @binding(3) var<storage, read_write> wideOutput: u32;
        @group(0) @binding(4) var<storage, read_write> wideAtomic: atomic<u64>;
        @compute @workgroup_size(1) fn native_int16() {
            let packed = input[0];
            let unsigned = vec2<u16>(packed.xy);
            let signed = vec2<i16>(packed.xy);
            let unsignedDivisor = vec2<u16>(packed.zw);
            let signedDivisor = vec2<i16>(packed.zw);
            let unsignedResult = ((unsigned << packed.zw) >> packed.wz)
                + unsigned / unsignedDivisor + unsigned % unsignedDivisor;
            let signedResult = ((signed << packed.zw) >> packed.wz)
                + signed / signedDivisor + signed % signedDivisor;
            narrowOutput = u32(dot(reverseBits(unsignedResult), vec2<u16>(u16(1), u16(2))))
                + u32(bitcast<u16>(signedResult.x)) + u32(bitcast<u16>(signedResult.y));
        }
        @compute @workgroup_size(1) fn native_int64() {
            let packed = input[0];
            let ranges = input[1];
            let unsigned = (vec2<u64>(packed.xy) << vec2u(32u)) | vec2<u64>(packed.zw);
            let signed = bitcast<vec2<i64>>(unsigned);
            let unsignedDivisor = vec2<u64>(ranges.xy);
            let signedDivisor = vec2<i64>(bitcast<vec2i>(ranges.zw));
            let unsignedResult = ((unsigned << ranges.xy) >> ranges.zw)
                + unsigned / unsignedDivisor + unsigned % unsignedDivisor;
            let signedResult = ((signed << ranges.yx) >> ranges.wz)
                + signed / signedDivisor + signed % signedDivisor;
            let unsignedFlag = select(0u, 1u, unsignedResult.x > unsignedResult.y);
            let signedFlag = select(0u, 2u, signedResult.x < signedResult.y);
            let unsignedField = extractBits(reverseBits(unsignedResult), ranges.x, ranges.y);
            let signedField = extractBits(signedResult, ranges.x, ranges.y);
            let unsignedInserted = insertBits(unsignedResult, unsignedField, ranges.x, ranges.y);
            let signedInserted = insertBits(signedField, signedResult, ranges.x, ranges.y);
            let population = countOneBits(unsignedInserted);
            let leading = firstLeadingBit(signedInserted);
            wideOutput = u32(unsignedInserted.x) + u32(bitcast<u64>(signedInserted.y))
                + u32(population.y) + u32(bitcast<u64>(leading.x)) + unsignedFlag + signedFlag;
        }
        @compute @workgroup_size(1) fn native_atomic64() {
            let packed = input[0];
            let high = 0x8000000000000000lu | (u64(packed.x) << 32u) | u64(packed.y);
            let low = (u64(packed.z) << 32u) | u64(packed.w);
            let previousMin = atomicMin(&wideAtomic, high);
            let previousMax = atomicMax(&wideAtomic, low);
            wideOutput = u32(previousMin >> 32u) ^ u32(previousMax >> 32u);
        }
        @compute @workgroup_size(1) fn round_trip() {
            let bits = input[0];
            let signed = bitcast<vec4i>(bits);
            let ranges = input[1];
            output[0] = bits >> ranges;
            output[1] = bitcast<vec4u>(signed >> ranges);
            output[2] = bits << ranges;
            output[3] = bitcast<vec4u>(signed << ranges);
            output[4] = extractBits(bits, ranges.x, ranges.y);
            output[5] = bitcast<vec4u>(extractBits(signed, ranges.x, ranges.y));
            output[6] = insertBits(bits, vec4u(5u), ranges.x, ranges.y);
            output[7] = bitcast<vec4u>(insertBits(signed, vec4i(-2i), ranges.x, ranges.y));
            output[8] = extractBits(bits, ranges.z + 1u, ranges.y);
            output[9] = insertBits(bits, vec4u(5u), ranges.z + 1u, ranges.y);
            output[10] = insertBits(bits, vec4u(5u), ranges.x, ranges.w);
            output[11] = reverseBits(bits);
            output[12] = countOneBits(bits);
        }
        """)
    let compiler = IndependentShaderCompiler()
    guard case .spirv(let binary) = try compiler.compile(roundTripSource, to: .spirv) else {
        throw GPUError.backendFailure("Expected binary SPIR-V for integer bit operations")
    }
    do {
        _ = try compiler.compileWebGPUCompute(.spirv(binary), entryPoint: "native_int16")
        throw GPUError.backendFailure("WebGPU accepted a reachable 16-bit integer entry")
    } catch ShaderFailure.unsupported("16-bit integer types require native shader output") {}
    do {
        _ = try compiler.compileWebGPUCompute(.spirv(binary), entryPoint: "native_int64")
        throw GPUError.backendFailure("WebGPU accepted a reachable 64-bit integer entry")
    } catch ShaderFailure.unsupported("64-bit integer types require native shader output") {}
    do {
        _ = try compiler.compileWebGPUCompute(.spirv(binary), entryPoint: "native_atomic64")
        throw GPUError.backendFailure("WebGPU accepted reachable 64-bit atomics")
    } catch ShaderFailure.unsupported("64-bit integer types require native shader output") {}
    let roundTripShader = try compiler.compileWebGPUCompute(.spirv(binary), entryPoint: "round_trip")
    guard !roundTripShader.requiresShaderInt16, !roundTripShader.requiresShaderInt64,
          !roundTripShader.requiresShaderInt64AtomicMinMax, !roundTripShader.requiresShaderInt64AtomicAllOps else {
        throw GPUError.backendFailure("Unreachable native integer types changed the WebGPU entry profile")
    }
    let roundTrip = try await device.createComputePipeline(roundTripShader)
    guard case .spirv(let booleanBinary) = try compiler.compile(booleanSource, to: .spirv) else {
        throw GPUError.backendFailure("Expected binary SPIR-V for eager Boolean operations")
    }
    let booleanRoundTripShader = try compiler.compileWebGPUCompute(.spirv(booleanBinary))
    let booleanRoundTrip = try await device.createComputePipeline(booleanRoundTripShader)
    let input = try await device.createBuffer(.init(size: 32, usage: [.storage, .copyDestination]))
    let output = try await device.createBuffer(.init(size: 224, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 224, usage: [.mapRead, .copyDestination]))
    let bitInput: [UInt32] = [0x80000008, 0x0123456f, 0xffffffff, 0x12345677]
    let ticket = try await device.submitCommands([
        .write(buffer: input, offset: 0, words: [0, 1, 0x80000000, 0xffffffff, 28, 8, 0xffffffff, 32]),
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): input, .init(binding: 1): output],
                         workgroups: .init(x: 1)),
        .copy(source: output, destination: read, byteCount: 224), .read(buffer: read, byteCount: 224),
        .dispatchCompute(pipeline: booleanPipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
        .copy(source: output, destination: read, byteCount: 96), .read(buffer: read, byteCount: 96),
        .write(buffer: input, offset: 0, words: [11, 22, 33, 2]),
        .dispatchCompute(pipeline: compositePipeline, bindings: [.init(binding: 0): input, .init(binding: 1): output],
                         workgroups: .init(x: 1)),
        .copy(source: output, destination: read, byteCount: 32), .read(buffer: read, byteCount: 32),
        .write(buffer: input, offset: 0, words: bitInput + [0, 4, 0, 0]),
        .dispatchCompute(pipeline: bitPipeline, bindings: [.init(binding: 0): input, .init(binding: 1): output],
                         workgroups: .init(x: 1)),
        .copy(source: output, destination: read, byteCount: 80), .read(buffer: read, byteCount: 80),
        .write(buffer: input, offset: 16, words: [32, 0]),
        .dispatchCompute(pipeline: bitPipeline, bindings: [.init(binding: 0): input, .init(binding: 1): output],
                         workgroups: .init(x: 1)),
        .copy(source: output, destination: read, byteCount: 80), .read(buffer: read, byteCount: 80),
        .write(buffer: input, offset: 0, words: [0x80000008, 0xf123456f, 0xffffffff, 0x12345677, 1, 4, 31, 0]),
        .dispatchCompute(pipeline: shiftPipeline, bindings: [.init(binding: 0): input, .init(binding: 1): output],
                         workgroups: .init(x: 1)),
        .copy(source: output, destination: read, byteCount: 48), .read(buffer: read, byteCount: 48),
        .write(buffer: input, offset: 0, words: bitInput + [28, 0xffffffff, 32, 0]),
        .dispatchCompute(pipeline: roundTrip, bindings: [.init(binding: 0): input, .init(binding: 1): output],
                         workgroups: .init(x: 1)),
        .copy(source: output, destination: read, byteCount: 208), .read(buffer: read, byteCount: 208),
        .dispatchCompute(pipeline: booleanRoundTrip, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
        .copy(source: output, destination: read, byteCount: 96), .read(buffer: read, byteCount: 96),
    ])
    let expected: [UInt32] = [
        32, 31, 0, 0, 32, 0, 31, 0, 0, 1, 1, 32, 0, 0x80000000, 1, 0xffffffff,
        0xffffffff, 0, 31, 31, 0xffffffff, 0, 30, 0xffffffff, 0xffffffff, 0, 31, 0,
        0, 0, 8, 15, 0, 0, 0xfffffff8, 0xffffffff, 0x50000000, 0x50000001, 0x50000000, 0x5fffffff,
        0, 1, 0x80000000, 0xffffffff, 0, 1, 0x80000000, 0xffffffff, 0, 0xffffffff, 0, 7,
        32, 32, 32, 0x80000000,
    ]
    let snapshots = try await ticket.wait()
    let booleanExpected: [UInt32] = [0, 1, 1234, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1,
                                     0, 1, 1, 0, 12345678, 1, 0, 1]
    let reverseAndCount: [UInt32] = [0x10000001, 0xf6a2c480, 0xffffffff, 0xee6a2c48, 2, 13, 32, 15]
    let bitExpected: [UInt32] = [
        0x80000000, 0x01234561, 0xfffffff2, 0x12345673,
        0xfffffff8, 0xffffffff, 0xffffffff, 7, 8, 15, 15, 7,
    ] + reverseAndCount
    let emptyFieldExpected = bitInput + Array(repeating: 0, count: 8) + reverseAndCount
    let shiftExpected: [UInt32] = [
        0x40000004, 0x0f123456, 1, 0x12345677,
        0xc0000004, 0xff123456, 0xffffffff, 0x12345677,
        0x10, 0x123456f0, 0x80000000, 0x12345677,
    ]
    let roundTripExpected: [UInt32] = [
        8, 0, 0xffffffff, 0x12345677,
        0xfffffff8, 0, 0xffffffff, 0x12345677,
        0x80000000, 0x80000000, 0xffffffff, 0x12345677,
        0x80000000, 0x80000000, 0xffffffff, 0x12345677,
        8, 0, 15, 1,
        0xfffffff8, 0, 0xffffffff, 1,
        0x50000008, 0x5123456f, 0x5fffffff, 0x52345677,
        0xe0000008, 0xe123456f, 0xefffffff, 0xe2345677,
        0, 0, 0, 0,
    ] + bitInput + bitInput + reverseAndCount
    guard snapshots == [expected, booleanExpected, [202, 900, 11, 0, 101, 202, 33, 900],
                        bitExpected, emptyFieldExpected, shiftExpected, roundTripExpected, booleanExpected] else {
        throw GPUError.backendFailure("Unexpected integer bit builtin readback: \(snapshots)")
    }
    return snapshots.flatMap { $0 }
}

private func bitInstructionsSPIRV() -> [UInt32] {
    func op(_ code: UInt32, _ operands: UInt32...) -> [UInt32] {
        [UInt32(operands.count + 1) << 16 | code] + operands
    }
    var words: [UInt32] = [0x07230203, 0x00010300, 0, 80, 0]
    words += op(17, 1) + op(14, 0, 1) + op(15, 5, 40, 0x6e69616d, 0) + op(16, 40, 17, 1, 1, 1)
    for (block, members): (UInt32, UInt32) in [(6, 2), (7, 5)] {
        words += op(71, block, 2)
        for member in 0..<members { words += op(72, block, member, 35, member * 16) }
    }
    for instruction in [
        op(71, 20, 34, 0), op(71, 20, 33, 0), op(71, 21, 34, 0), op(71, 21, 33, 1),
        op(19, 1), op(21, 2, 32, 0), op(21, 3, 32, 1), op(23, 4, 2, 4), op(23, 5, 3, 4),
        op(30, 6, 4, 4), op(30, 7, 4, 4, 4, 4, 4),
        op(32, 8, 12, 6), op(32, 9, 12, 7), op(32, 10, 12, 4), op(33, 11, 1),
        op(43, 2, 25, 0), op(43, 2, 26, 1), op(43, 2, 27, 2), op(43, 2, 28, 3), op(43, 2, 29, 4),
        op(44, 4, 30, 25, 26, 27, 28), op(59, 8, 20, 12), op(59, 9, 21, 12),
        op(54, 1, 40, 0, 11), op(248, 41),
        op(65, 10, 42, 20, 25), op(61, 4, 43, 42), op(65, 10, 44, 20, 26), op(61, 4, 45, 44),
        op(81, 2, 46, 45, 0), op(81, 2, 47, 45, 1), op(124, 3, 48, 47), op(124, 5, 49, 43),
        op(201, 4, 50, 43, 30, 46, 48), op(202, 4, 51, 43, 46, 48), op(203, 5, 52, 49, 46, 48),
        op(124, 4, 53, 52), op(204, 5, 54, 49), op(124, 4, 55, 54), op(205, 5, 56, 43), op(124, 4, 57, 56),
    ] { words += instruction }
    for (index, result) in [UInt32(50), 51, 53, 55, 57].enumerated() {
        let address = 60 + UInt32(index)
        words += op(65, 10, address, 21, 25 + UInt32(index)) + op(62, address, result)
    }
    return words + op(253) + op(56)
}

private func shiftInstructionsSPIRV() -> [UInt32] {
    func op(_ code: UInt32, _ operands: UInt32...) -> [UInt32] {
        [UInt32(operands.count + 1) << 16 | code] + operands
    }
    var words: [UInt32] = [0x07230203, 0x00010300, 0, 80, 0]
    for instruction in [
        op(17, 1), op(14, 0, 1), op(15, 5, 40, 0x6e69616d, 0), op(16, 40, 17, 1, 1, 1),
        op(71, 6, 2), op(72, 6, 0, 35, 0), op(72, 6, 1, 35, 16),
        op(71, 7, 2), op(72, 7, 0, 35, 0), op(72, 7, 1, 35, 16), op(72, 7, 2, 35, 32),
        op(71, 20, 34, 0), op(71, 20, 33, 0), op(71, 21, 34, 0), op(71, 21, 33, 1),
        op(19, 1), op(21, 2, 32, 0), op(21, 3, 32, 1), op(23, 4, 2, 4), op(23, 5, 3, 4),
        op(30, 6, 4, 4), op(30, 7, 4, 4, 4), op(32, 8, 12, 6), op(32, 9, 12, 7), op(32, 10, 12, 4), op(33, 11, 1),
        op(43, 2, 25, 0), op(43, 2, 26, 1), op(43, 2, 27, 2), op(59, 8, 20, 12), op(59, 9, 21, 12),
        op(54, 1, 40, 0, 11), op(248, 41),
        op(65, 10, 42, 20, 25), op(61, 4, 43, 42), op(65, 10, 44, 20, 26), op(61, 4, 45, 44),
        op(124, 5, 46, 43), op(124, 5, 47, 45),
        op(194, 5, 50, 46, 45), op(124, 4, 51, 50), op(195, 4, 52, 43, 47), op(196, 4, 53, 43, 47),
        op(65, 10, 60, 21, 25), op(62, 60, 51), op(65, 10, 61, 21, 26), op(62, 61, 52),
        op(65, 10, 62, 21, 27), op(62, 62, 53), op(253), op(56),
    ] { words += instruction }
    return words
}

private func vectorCompositeSPIRV() -> [UInt32] {
    func instruction(_ opcode: UInt32, _ operands: UInt32...) -> [UInt32] {
        [UInt32(operands.count + 1) << 16 | opcode] + operands
    }
    var words: [UInt32] = [0x07230203, 0x00010300, 0, 70, 0]
    for item in [
        instruction(17, 1), instruction(14, 0, 1), instruction(15, 5, 40, 0x6e69616d, 0), instruction(16, 40, 17, 1, 1, 1),
        instruction(71, 6, 2), instruction(72, 6, 0, 35, 0), instruction(71, 7, 2),
        instruction(72, 7, 0, 35, 0), instruction(72, 7, 1, 35, 16),
        instruction(71, 20, 34, 0), instruction(71, 20, 33, 0), instruction(71, 21, 34, 0), instruction(71, 21, 33, 1),
        instruction(19, 1), instruction(21, 2, 32, 0), instruction(21, 3, 32, 1), instruction(23, 4, 2, 4),
        instruction(23, 5, 2, 2), instruction(30, 6, 4), instruction(30, 7, 4, 4),
        instruction(32, 8, 12, 6), instruction(32, 9, 12, 7), instruction(32, 10, 12, 4), instruction(33, 11, 1),
        instruction(43, 2, 25, 0), instruction(43, 2, 26, 1), instruction(43, 2, 27, 101), instruction(43, 2, 28, 202),
        instruction(43, 2, 29, 900), instruction(44, 5, 30, 27, 28), instruction(59, 8, 20, 12), instruction(59, 9, 21, 12),
        instruction(54, 1, 40, 0, 11), instruction(248, 41), instruction(65, 10, 42, 20, 25), instruction(61, 4, 43, 42),
        instruction(81, 2, 44, 43, 3), instruction(124, 3, 45, 44), instruction(77, 2, 46, 43, 45),
        instruction(78, 4, 47, 43, 29, 45), instruction(79, 4, 48, 47, 30, 5, 2, 0, .max),
        instruction(80, 4, 49, 30, 46, 29), instruction(65, 10, 50, 21, 25), instruction(62, 50, 48),
        instruction(65, 10, 51, 21, 26), instruction(62, 51, 49), instruction(253), instruction(56),
    ] { words += item }
    return words
}

private func packingDiagnostic(_ device: Device) async throws -> [UInt32] {

    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl("""
        const packed = vec4u(pack4x8snorm(vec4f(-2, -0.5, 0.5, 2)), pack4x8unorm(vec4f(-2, -0.5, 0.5, 2)),
                            pack2x16snorm(vec2f(-0.5, 0.5)), pack2x16float(vec2f(1, -2)));
        const_assert unpack4xI8(0x7f00ff80u).x == -128i;
        const_assert unpack4xU8(0x7f00ff80u).w == 127u;
        const_assert unpack2x16float(pack2x16float(vec2f(1.00048828125, 5.960464477539063e-8))).x == 1.0f;
        const_assert pack2x16float(vec2f(5.960464477539063e-8, -0.0)) == 0x80000001u;
        const_assert pack4xI8Clamp(vec4(-129, -1, 127, 128)) == 0x7f7fff80u;
        @group(0) @binding(0) var<storage, read> input: array<vec4u, 5>;
        @group(0) @binding(1) var<storage, read_write> output: array<vec4u, 9>;
        @compute @workgroup_size(1) fn run() {
            let normalized = bitcast<vec4f>(input[0]);
            let halfValues = bitcast<vec2f>(input[1].xy);
            let word = input[1].z;
            let signed = bitcast<vec4i>(input[2]);
            let unsigned = input[3];
            let channels = input[4];
            output[0] = vec4u(pack4x8snorm(normalized), pack4x8unorm(normalized),
                              pack2x16snorm(normalized.yz), pack2x16unorm(normalized.zw));
            output[1] = vec4u(pack4xI8(signed), pack4xI8Clamp(signed), pack4xU8(unsigned), pack4xU8Clamp(unsigned));
            output[2] = bitcast<vec4u>(unpack4xI8(word));
            output[3] = unpack4xU8(word);
            output[4] = bitcast<vec4u>(unpack4x8snorm(channels.x));
            output[5] = bitcast<vec4u>(unpack4x8unorm(channels.y));
            output[6] = vec4u(bitcast<vec2u>(unpack2x16snorm(channels.z)), bitcast<vec2u>(unpack2x16unorm(channels.w)));
            output[7] = vec4u(pack2x16float(halfValues), bitcast<vec2u>(unpack2x16float(input[1].w)), pack2x16float(unpack2x16float(input[1].w)));
            output[8] = packed;
        }
        """), to: .wgsl)
    let pipeline = try await device.createComputePipeline(shader)
    let roundTripSource = ShaderInput.wgsl("""
        @group(0) @binding(0) var<storage, read> input: array<vec4u, 5>;
        @group(0) @binding(1) var<storage, read_write> output: array<vec4u, 9>;
        @compute @workgroup_size(1) fn round_trip() {
            let normalized = bitcast<vec4f>(input[0]);
            let words = input[1];
            let signed = bitcast<vec4i>(input[2]);
            let unsigned = input[3];
            let channels = input[4];
            let halves = unpack2x16float(words.w);
            output[0] = vec4u(pack4x8snorm(normalized), pack4x8unorm(normalized),
                              pack2x16snorm(normalized.yz), pack2x16unorm(normalized.zw));
            output[1] = vec4u(pack4xI8(signed), pack4xI8Clamp(signed), pack4xU8(unsigned), pack4xU8Clamp(unsigned));
            output[2] = bitcast<vec4u>(unpack4xI8(words.z));
            output[3] = unpack4xU8(words.z);
            output[4] = bitcast<vec4u>(unpack4x8snorm(channels.x));
            output[5] = bitcast<vec4u>(unpack4x8unorm(channels.y));
            output[6] = vec4u(bitcast<vec2u>(unpack2x16snorm(channels.z)), bitcast<vec2u>(unpack2x16unorm(channels.w)));
            output[7] = vec4u(pack2x16float(bitcast<vec2f>(words.xy)), bitcast<vec2u>(halves), pack2x16float(halves));
            output[8] = vec4u(bitcast<u32>(dot4I8Packed(words.z, words.w)), dot4U8Packed(words.z, words.w),
                              bitcast<u32>(dot4I8Packed(channels.x, channels.y)), dot4U8Packed(channels.x, channels.y));
        }
        """)
    guard case .spirv(let binary) = try compiler.compile(roundTripSource, to: .spirv) else {
        throw GPUError.backendFailure("Expected binary SPIR-V for packing and packed dot")
    }
    let roundTripShader = try compiler.compileWebGPUCompute(.spirv(binary), entryPoint: "round_trip")
    let roundTrip = try await device.createComputePipeline(roundTripShader)
    let input = try await device.createBuffer(.init(size: 80, usage: [.storage, .copyDestination]))
    let output = try await device.createBuffer(.init(size: 144, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 144, usage: [.mapRead, .copyDestination]))
    let packingInputWords: [UInt32] = [
        0xc0000000, 0xbf000000, 0x3f000000, 0x40000000,
        0x3f800000, 0xc0000000, 0x7f00ff80, 0xc0003c00,
        0xffffff7f, 0xffffffff, 127, 128, 256, 255, 511, 0x12345678,
        0x7f008081, 0xff00ff00, 0x7fff8000, 0xffff0000,
    ]
    let ticket = try await device.submitCommands([
        .write(buffer: input, offset: 0, words: packingInputWords),
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): input, .init(binding: 1): output],
                         workgroups: .init(x: 1)),
        .copy(source: output, destination: read, byteCount: 144), .read(buffer: read, byteCount: 144),
        .dispatchCompute(pipeline: roundTrip, bindings: [.init(binding: 0): input, .init(binding: 1): output],
                         workgroups: .init(x: 1)),
        .copy(source: output, destination: read, byteCount: 144), .read(buffer: read, byteCount: 144),
    ])
    let expected: [UInt32] = [
        0x7f40c181, 0xff800000, 0x4000c001, 0xffff8000,
        0x807fff7f, 0x7f7fff80, 0x78ffff00, 0xffffffff,
        0xffffff80, 0xffffffff, 0, 127, 128, 255, 0, 127,
        0xbf800000, 0xbf800000, 0, 0x3f800000, 0, 0x3f800000, 0, 0x3f800000,
        0xbf800000, 0x3f800000, 0, 0x3f800000,
        0xc0003c00, 0x3f800000, 0xc0000000, 0xc0003c00,
        0x7f40c181, 0xff800000, 0x4000c001, 0xc0003c00,
    ]
    let roundTripExpected = Array(expected.prefix(32)) + [0xffffe004, 39684, 1, 65025]
    let snapshots = try await ticket.wait()
    guard snapshots == [expected, roundTripExpected] else {
        throw GPUError.backendFailure("Unexpected packing readback: \(snapshots)")
    }
    return snapshots[0]
}

private func geometricDiagnostic(_ device: Device) async throws -> [UInt32] {

    let shader = try IndependentShaderCompiler().compileCompute(.wgsl("""
        const_assert dot(vec3(1, 2, 3), vec3(3, 2, 1)) == 10;
        const_assert length(vec2(3, 4)) == 5.0;
        const_assert cross(vec3f(1, 0, 0), vec3f(0, 1, 0)).z == 1.0f;
        const_assert dot(vec2i(2147483647, 1), vec2i(2, 1)) == -1i;
        const_assert dot4I8Packed(0x7f00ff80u, 0x01020304u) == -388i;
        const_assert dot4U8Packed(0x7f00ff80u, 0x01020304u) == 1404u;
        const_assert refract(vec2f(1, 0), vec2f(0, 1), 2.0f).x == 0.0f;
        @group(0) @binding(0) var<storage, read> input: array<vec4u, 4>;
        @group(0) @binding(1) var<storage, read_write> output: array<vec4u, 7>;
        @compute @workgroup_size(1) fn run() {
            let a = bitcast<vec4f>(input[0]);
            let b = bitcast<vec4f>(input[1]);
            let axis = vec3f(0, 1, 0);
            let incidence = vec3f(0, -1, 0);
            let tangent = vec3f(1, 0, 0);
            output[0] = bitcast<vec4u>(vec4f(dot(a.xy, a.xy), length(a.xy), distance(a.xy, b.xy), distance(a.x, -2.0)));
            output[1] = bitcast<vec4u>(vec4f(cross(a.xyz, b.xyz), length(-a.y)));
            output[2] = bitcast<vec4u>(vec4f(normalize(vec3f(0, a.y, 0)), dot(a.xyz, b.xyz)));
            output[3] = bitcast<vec4u>(vec4f(faceForward(axis, incidence, axis), faceForward(axis, axis, axis).y));
            output[4] = bitcast<vec4u>(vec4f(reflect(incidence, axis), refract(incidence, axis, 0.5).y));
            output[5] = bitcast<vec4u>(vec4f(refract(tangent, axis, a.x), dot(a, b)));
            let words = input[2];
            let products = input[3];
            output[6] = vec4u(bitcast<u32>(dot(bitcast<vec4i>(words), vec4i(2, 2, 3, 4))), dot(words, vec4u(1)),
                              bitcast<u32>(dot4I8Packed(products.x, products.y)), dot4U8Packed(products.x, products.y));
        }
        """), to: .wgsl)
    let pipeline = try await device.createComputePipeline(shader)
    let input = try await device.createBuffer(.init(size: 64, usage: [.storage, .copyDestination]))
    let output = try await device.createBuffer(.init(size: 112, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 112, usage: [.mapRead, .copyDestination]))
    let geometricInput: [UInt32] = [
        0x40400000, 0x40800000, 0, 0, 0, 0, 0x3f800000, 0,
        0x7fffffff, 0x80000000, 0xffffffff, 2, 0x7f00ff80, 0x01020304, 0, 0,
    ]
    let ticket = try await device.submitCommands([
        .write(buffer: input, offset: 0, words: geometricInput),
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): input, .init(binding: 1): output],
                         workgroups: .init(x: 1)),
        .copy(source: output, destination: read, byteCount: 112), .read(buffer: read, byteCount: 112),
    ])
    let expectedGeometry: [Float] = [
        25, 5, 5, 5, 4, -3, 0, 4, 0, 1, 0, 0,
        0, 1, 0, -1, 0, 1, 0, -1, 0, 0, 0, 0,
    ]
    let expectedPackedDots: [UInt32] = [3, 0, 0xfffffe7c, 1404]
    guard let snapshot = try await ticket.wait().first,
          snapshot.prefix(24).map({ Float(bitPattern: $0) }) == expectedGeometry,
          Array(snapshot.suffix(4)) == expectedPackedDots else {
        throw GPUError.backendFailure("Unexpected geometric builtin readback")
    }
    return snapshot.prefix(24).map { $0 == 0x80000000 ? 0 : $0 } + snapshot.suffix(4)
}

private func matrixBuiltinDiagnostic(_ device: Device) async throws -> [UInt32] {

    let shader = try IndependentShaderCompiler().compileCompute(.wgsl("""
        const FLIP = transpose(mat2x3(1, 2, 3, 4, 5, 6));
        const_assert FLIP[2][1] == 6.0;
        const_assert transpose(FLIP)[1][0] == 4.0;
        const_assert determinant(mat3x3(1, 0, 5, 2, 1, 6, 3, 4, 0)) == 1.0;
        const_assert determinant(mat4x4f(0, 2, 0, 0, 3, 0, 0, 0, 1, 2, 4, 0, 5, 6, 7, -1)) == 24.0f;
        @group(0) @binding(0) var<storage, read> input: array<vec4f, 9>;
        @group(0) @binding(1) var<storage, read_write> output: array<vec4f, 3>;
        @group(0) @binding(2) var<storage, read_write> columns: array<mat3x2f, 1>;
        fn flip(m: mat2x3f) -> mat3x2f { return transpose(m); }
        @compute @workgroup_size(1) fn run() {
            let rect = mat2x3f(input[0].xyz, input[1].xyz);
            let square = mat2x2f(input[0].xy, input[1].xy);
            let m3 = mat3x3f(input[2].xyz, input[3].xyz, input[4].xyz);
            let m4 = mat4x4f(input[5], input[6], input[7], input[8]);
            let turned = flip(rect);
            let restored = transpose(turned);
            const local = determinant(mat2x2f(1, 2, 3, 4));
            columns[0] = turned;
            output[0] = vec4f(determinant(square), determinant(m3), determinant(m4), local);
            output[1] = vec4f(restored[0], turned[2].y);
            output[2] = vec4f(turned * vec3f(1, 2, 3), restored[1].xy);
        }
        """), to: .wgsl)
    let pipeline = try await device.createComputePipeline(shader)
    let input = try await device.createBuffer(.init(size: 144, usage: [.storage, .copyDestination]))
    let output = try await device.createBuffer(.init(size: 48, usage: [.storage, .copySource]))
    let columns = try await device.createBuffer(.init(size: 24, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 116, usage: [.mapRead, .copyDestination]))
    let inverseScale: Float = 1e-12
    let inverseShader = try IndependentShaderCompiler().compileWebGPUCompute(.spirv(inverseMatrixSPIRV(scale: inverseScale)))
    let inversePipeline = try await device.createComputePipeline(inverseShader)
    let inverses = try await device.createBuffer(.init(size: 116, usage: [.storage, .copySource]))
    let matrixInstructions = try IndependentShaderCompiler().compileWebGPUCompute(.spirv(matrixInstructionsSPIRV()))
    let instructionPipeline = try await device.createComputePipeline(matrixInstructions)
    let matrixBuiltinInput: [Float] = [
        1, 2, 3, 0, 4, 5, 6, 0,
        1, 0, 5, 0, 2, 1, 6, 0, 3, 4, 0, 0,
        0, 2, 0, 0, 3, 0, 0, 0, 1, 2, 4, 0, 5, 6, 7, -1,
    ]
    let ticket = try await device.submitCommands([
        .write(buffer: input, offset: 0, words: matrixBuiltinInput.map(\.bitPattern)),
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): input, .init(binding: 1): output,
                         .init(binding: 2): columns], workgroups: .init(x: 1)),
        .copy(source: output, destination: read, byteCount: 48), .read(buffer: read, byteCount: 48),
        .copy(source: columns, destination: read, byteCount: 24), .read(buffer: read, byteCount: 24),
        .dispatchCompute(pipeline: inversePipeline, bindings: [.init(binding: 0): inverses], workgroups: .init(x: 1)),
        .copy(source: inverses, destination: read, byteCount: 116), .read(buffer: read, byteCount: 116),
        .write(buffer: input, offset: 0, words: [Float(1), 2, 3, 4, 5, 6].map(\.bitPattern)),
        .dispatchCompute(pipeline: instructionPipeline, bindings: [.init(binding: 0): input, .init(binding: 1): output],
                         workgroups: .init(x: 1)),
        .copy(source: output, destination: read, byteCount: 48), .read(buffer: read, byteCount: 48),
    ])
    let expected: [[Float]] = [[-3, 1, 24, -2, 1, 2, 3, 6, 14, 32, 4, 5], [1, 4, 2, 5, 3, 6]]
    let inverseExpected: [Float] = [
        0.5, 0, -0.125, 0.25, -24, 20, -5, 18, -15, 4, 5, -4, 1,
        0.5, 0, 0, 0, -0.125, 0.25, 0, 0, 0.03125, -0.0625, 0.125, 0, -0.0078125, 0.015625, -0.03125, 0.0625,
    ]
    let instructionExpected: [Float] = [2, 5, 16, 12, 62, 152, 39, 54, 69, 36, 88, 39]
    let snapshots = try await ticket.wait()
    guard snapshots.count == 4, Array(snapshots.prefix(2)) == expected.map({ $0.map(\.bitPattern) }),
          snapshots[2].count == inverseExpected.count,
          zip(snapshots[2], inverseExpected).allSatisfy({
              abs(Float(bitPattern: $0) * inverseScale - $1) <= max(1, abs($1)) * 0.0008
          }), snapshots[3] == instructionExpected.map(\.bitPattern) else {
        throw GPUError.backendFailure("Unexpected matrix builtin readback: \(snapshots)")
    }
    return snapshots.flatMap { $0 }
}

private func matrixInstructionsSPIRV() -> [UInt32] {
    func inst(_ code: UInt32, _ operands: UInt32...) -> [UInt32] {
        [UInt32(operands.count + 1) << 16 | code] + operands
    }
    var words: [UInt32] = [0x07230203, 0x00010300, 0, 128, 0]
    words += inst(17, 1) + inst(14, 0, 1) + inst(15, 5, 18, 0x6e69616d, 0) + inst(16, 18, 17, 1, 1, 1)
    for (block, count): (UInt32, UInt32) in [(9, 6), (10, 12)] {
        words += inst(71, block, 2)
        for index in 0..<count { words += inst(72, block, index, 35, index * 4) }
    }
    words += inst(71, 16, 34, 0) + inst(71, 16, 33, 0) + inst(71, 17, 34, 0) + inst(71, 17, 33, 1)
    for instruction in [
        inst(19, 1), inst(21, 2, 32, 0), inst(22, 3, 32), inst(23, 4, 3, 2), inst(23, 5, 3, 3),
        inst(24, 6, 5, 2), inst(24, 7, 4, 3), inst(24, 8, 5, 3),
        [UInt32(8) << 16 | 30, 9] + Array(repeating: 3, count: 6),
        [UInt32(14) << 16 | 30, 10] + Array(repeating: 3, count: 12),
        inst(32, 11, 12, 9), inst(32, 12, 12, 10), inst(32, 13, 12, 3), inst(33, 14, 1),
    ] { words += instruction }
    for index: UInt32 in 0..<12 { words += inst(43, 2, 30 + index, index) }
    for (index, value) in [Float(2), 7, 8, 9, 10, 11].enumerated() {
        words += inst(43, 3, 50 + UInt32(index), value.bitPattern)
    }
    words += inst(44, 4, 56, 51, 52) + inst(44, 5, 57, 53, 54, 55)
    words += inst(59, 11, 16, 12) + inst(59, 12, 17, 12) + inst(54, 1, 18, 0, 14) + inst(248, 19)
    for index: UInt32 in 0..<6 {
        let address = 60 + 2 * index
        words += inst(65, 13, address, 16, 30 + index) + inst(61, 3, address + 1, address)
    }
    for instruction in [
        inst(80, 5, 72, 61, 63, 65), inst(80, 5, 73, 67, 69, 71), inst(80, 6, 74, 72, 73),
        inst(84, 7, 75, 74), inst(142, 4, 76, 56, 50), inst(143, 6, 77, 74, 50),
        inst(144, 4, 78, 57, 74), inst(145, 5, 79, 74, 56), inst(146, 8, 80, 74, 75),
        inst(147, 6, 81, 57, 56), inst(81, 4, 82, 75, 0), inst(148, 3, 83, 56, 82),
    ] { words += instruction }
    let extracts: [(UInt32, [UInt32])] = [
        (75, [1, 0]), (75, [1, 1]), (76, [1]), (77, [1, 2]), (78, [0]), (78, [1]),
        (79, [0]), (79, [1]), (79, [2]), (80, [1, 2]), (81, [1, 2]),
    ]
    for (index, (value, path)) in extracts.enumerated() {
        let result = 90 + UInt32(index)
        let address = 110 + UInt32(index)
        words += [UInt32(path.count + 4) << 16 | 81, 3, result, value] + path
        words += inst(65, 13, address, 17, 30 + UInt32(index)) + inst(62, address, result)
    }
    return words + inst(65, 13, 121, 17, 41) + inst(62, 121, 83) + inst(253) + inst(56)
}

private func inverseMatrixSPIRV(scale: Float) -> [UInt32] {
    func inst(_ code: UInt32, _ operands: UInt32...) -> [UInt32] {
        [UInt32(operands.count + 1) << 16 | code] + operands
    }
    var words: [UInt32] = [0x07230203, 0x00010300, 0, 1024, 0]
    words += inst(17, 1) + inst(11, 80, 0x4c534c47, 0x6474732e, 0x3035342e, 0) + inst(14, 0, 1)
    words += inst(15, 5, 15, 0x6e69616d, 0) + inst(16, 15, 17, 1, 1, 1)
    words += inst(71, 10, 2) + inst(71, 14, 34, 0) + inst(71, 14, 33, 0)
    for index: UInt32 in 0..<29 { words += inst(72, 10, index, 35, index * 4) }
    func op(_ opcode: UInt32, _ operands: [UInt32]) -> [UInt32] {
        [UInt32(operands.count + 1) << 16 | opcode] + operands
    }
    for instruction in [
        inst(19, 1), inst(21, 2, 32, 0), inst(22, 3, 32), inst(23, 4, 3, 2), inst(23, 5, 3, 3), inst(23, 6, 3, 4),
        inst(24, 7, 4, 2), inst(24, 8, 5, 3), inst(24, 9, 6, 4),
        op(30, [10] + Array(repeating: 3, count: 29)),
        inst(32, 11, 12, 10), inst(32, 12, 12, 3), inst(33, 13, 1),
    ] {
        words += instruction
    }
    let matrices: [[[Float]]] = [
        [[2, 0], [1, 4]],
        [[1, 0, 5], [2, 1, 6], [3, 4, 0]],
        [[2, 0, 0, 0], [1, 4, 0, 0], [0, 2, 8, 0], [0, 0, 4, 16]],
    ]
    var next: UInt32 = 100
    var columns: [[UInt32]] = []
    for (matrixIndex, matrix) in matrices.enumerated() {
        var columnIDs: [UInt32] = []
        for column in matrix {
            var elements: [UInt32] = []
            for value in column {
                words += inst(43, 3, next, (value * scale).bitPattern)
                elements.append(next)
                next += 1
            }
            words += op(44, [UInt32(4 + matrixIndex), next] + elements)
            columnIDs.append(next)
            next += 1
        }
        columns.append(columnIDs)
    }
    let smallMatrix = next
    next += 1
    words += op(44, [7, smallMatrix] + columns[0])
    let indexBase = next
    for index: UInt32 in 0..<29 { words += inst(43, 2, indexBase + index, index) }
    next += 29
    words += inst(59, 11, 14, 12) + inst(54, 1, 15, 0, 13) + inst(248, 16)
    var outputIndex: UInt32 = 0
    for (index, columnIDs) in columns.enumerated() {
        let type = UInt32(7 + index)
        let matrix: UInt32
        if index == 0 {
            matrix = smallMatrix
        } else {
            matrix = next
            next += 1
            words += op(80, [type, matrix] + columnIDs)
        }
        let inverted = next
        next += 1
        words += inst(12, type, inverted, 80, 34, matrix)
        let size = UInt32(columnIDs.count)
        for column: UInt32 in 0..<size {
            for row: UInt32 in 0..<size {
                words += inst(81, 3, next, inverted, column, row)
                words += inst(65, 12, next + 1, 14, indexBase + outputIndex)
                words += inst(62, next + 1, next)
                outputIndex += 1
                next += 2
            }
        }
    }
    return words + inst(253) + inst(56)
}

private func derivativeDiagnostic(_ device: Device) async throws -> [UInt32] {

    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl("""
        struct Outputs {
            @location(0) regular: vec4u,
            @location(1) fine: vec4u,
            @location(2) coarse: vec4u,
        }
        @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
            if i == 0u { return vec4f(-1, -1, 0, 1); }
            if i == 1u { return vec4f(3, -1, 0, 1); }
            return vec4f(-1, 3, 0, 1);
        }
        fn encode(dx: f32, dy: f32, width: vec2f) -> vec4u {
            return vec4u(u32(dx), u32(8.0 + dy), u32(width.x), u32(width.y));
        }
        fn derivatives(p: vec2f) -> Outputs {
            let q = vec2f(2.0 * p.x - 3.0 * p.y, 5.0 * p.y - 4.0 * p.x);
            let regular = encode(dpdx(q.x), dpdy(q.x), fwidth(q));
            let fine = encode(dpdxFine(q).x, dpdyFine(q).x, fwidthFine(q));
            let coarse = encode(dpdxCoarse(q).x, dpdyCoarse(q).x, fwidthCoarse(q));
            let scalarPositive = all(p.x > 0.0) && any(p.y > 0.0);
            let scalarNegative = all(p.x < 0.0) || any(p.y < 0.0);
            let mixed = vec2<bool>(p.x > 0.0, p.y < 0.0);
            let vectors = all(p > vec2f(0.0)) && any(mixed) && !all(mixed) && !any(p < vec2f(0.0));
            if scalarPositive && !scalarNegative && vectors { return Outputs(regular, fine, coarse); }
            return Outputs(vec4u(0), vec4u(0), vec4u(0));
        }
        @fragment fn fs(@builtin(position) p: vec4f) -> Outputs { return derivatives(p.xy); }
        """)
    _ = try compiler.compileRenderStage(source, entryPoint: "vs", to: .wgsl)
    _ = try compiler.compileRenderStage(source, entryPoint: "fs", to: .wgsl)
    guard case .spirv(let binary) = try compiler.compile(source, to: .spirv) else {
        throw GPUError.backendFailure("Expected binary SPIR-V for fragment derivatives")
    }
    let roundTrip = ShaderInput.spirv(binary)
    let pipeline = try await device.createRenderPipeline(
        vertex: compiler.compileWebGPURenderStage(roundTrip, entryPoint: "vs"),
        fragment: compiler.compileWebGPURenderStage(roundTrip, entryPoint: "fs"),
        colorFormat: .rgba8Uint, additionalColorTargets: [1: .init(format: .rgba8Uint), 2: .init(format: .rgba8Uint)])
    let descriptor = TextureDescriptor(width: 2, height: 2, format: .rgba8Uint, usage: [.renderAttachment, .copySource])
    let regular = try await device.createTexture(descriptor)
    let fine = try await device.createTexture(descriptor)
    let coarse = try await device.createTexture(descriptor)
    let read = try await device.createBuffer(.init(size: 264, usage: [.mapRead, .copyDestination]))
    var commands: [BufferCommand] = [
        .render(pipeline: pipeline, target: regular, clear: .transparentBlack, vertexCount: 3,
                additionalColors: [1: .init(texture: fine, loadAction: .clear(.transparentBlack)),
                                   2: .init(texture: coarse, loadAction: .clear(.transparentBlack))]),
    ]
    for texture in [regular, fine, coarse] {
        commands += [.copyTextureToBuffer(source: texture, destination: read, bytesPerRow: 256), .read(buffer: read, byteCount: 264)]
    }
    let ticket = try await device.submitCommands(commands)
    let pixels = try await ticket.wait().flatMap { snapshot in [0, 1, 64, 65].map { snapshot[$0] } }
    guard pixels == Array(repeating: 0x09050502, count: 12) else {
        throw GPUError.backendFailure("Unexpected fragment derivative readback: \(pixels)")
    }
    return pixels
}

private func derivativeFilterDiagnostic(_ device: Device) async throws -> [UInt32] {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl("""
        diagnostic(off, derivative_uniformity);
        @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
            if i == 0u { return vec4f(-1, -1, 0, 1); }
            if i == 1u { return vec4f(3, -1, 0, 1); }
            return vec4f(-1, 3, 0, 1);
        }
        @diagnostic(info, derivative_uniformity)
        fn helper(x: f32) -> f32 {
            if x < 0.0 { return dpdx(x); }
            return 0.0;
        }
        @fragment fn fs(@builtin(position) p: vec4f) -> @location(0) vec4f {
            let red = helper(p.x);
            if p.y < 0.0 { return vec4f(dpdy(p.x)); }
            return vec4f(red, 1, 0, 1);
        }
        """)
    let pipeline = try await device.createRenderPipeline(
        vertex: compiler.compileRenderStage(source, entryPoint: "vs", to: .wgsl),
        fragment: compiler.compileRenderStage(source, entryPoint: "fs", to: .wgsl), colorFormat: .rgba8Unorm)
    let texture = try await device.createTexture(.init(width: 2, height: 2, format: .rgba8Unorm,
        usage: [.renderAttachment, .copySource]))
    let read = try await device.createBuffer(.init(size: 264, usage: [.mapRead, .copyDestination]))
    let ticket = try await device.submitCommands([
        .render(pipeline: pipeline, target: texture, clear: .transparentBlack, vertexCount: 3),
        .copyTextureToBuffer(source: texture, destination: read, bytesPerRow: 256),
        .read(buffer: read, byteCount: 264),
    ])
    guard let pixels = try await ticket.wait().first else {
        throw GPUError.backendFailure("Missing diagnostic-filter readback")
    }
    let result = [0, 1, 64, 65].map { pixels[$0] }
    guard result == Array(repeating: 0xff00ff00, count: 4) else {
        throw GPUError.backendFailure("Unexpected diagnostic-filter readback: \(result)")
    }
    return result
}

private func numericBuiltinDiagnostic(_ device: Device) async throws -> [UInt32] {

    let shader = try IndependentShaderCompiler().compileCompute(.wgsl("""
        const_assert abs(vec2i(-2147483647i - 1i, -7i)).x == bitcast<i32>(0x80000000u);
        const_assert max(vec2(4294967295, 1), vec2u(0)).x == 0xffffffffu;
        const_assert round(vec4f(-2.5, -1.5, 1.5, 2.5)).w == 2.0f;
        const_assert fract(-1.25f) == 0.75f;
        const_assert sign(vec2(-4, 0)).x == -1;
        const_assert clamp(vec2f(-4, 5), vec2f(-1, 0), vec2f(2, 3)).y == 3.0f;
        @group(0) @binding(0) var<storage, read> input: array<vec4u, 6>;
        @group(0) @binding(1) var<storage, read_write> output: array<vec4u, 13>;
        @compute @workgroup_size(1) fn run() {
            let signed = bitcast<vec4i>(input[0]);
            let unsigned = input[1];
            let values = bitcast<vec4f>(input[2]);
            let scales = bitcast<vec4f>(input[5]);
            output[0] = bitcast<vec4u>(abs(signed));
            output[1] = bitcast<vec4u>(sign(signed));
            output[2] = min(unsigned, vec4(3));
            output[3] = max(unsigned, vec4(3));
            output[4] = clamp(unsigned, input[3], input[4]);
            output[5] = bitcast<vec4u>(round(values));
            output[6] = bitcast<vec4u>(floor(values));
            output[7] = bitcast<vec4u>(ceil(values));
            output[8] = bitcast<vec4u>(trunc(values));
            output[9] = bitcast<vec4u>(fract(values));
            output[10] = bitcast<vec4u>(saturate(values));
            output[11] = bitcast<vec4u>(vec4f(sqrt(scales.x), inverseSqrt(scales.x), sin(scales.y), cos(scales.y)));
            output[12] = bitcast<vec4u>(vec4f(abs(values.x), sign(values.x), min(values.x, values.w), max(values.x, values.w)));
        }
        """), to: .wgsl)
    let pipeline = try await device.createComputePipeline(shader)
    let input = try await device.createBuffer(.init(size: 96, usage: [.storage, .copyDestination]))
    let output = try await device.createBuffer(.init(size: 240, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 240, usage: [.mapRead, .copyDestination]))
    let numericInput: [UInt32] = [
        0x80000000, 0xfffffff9, 0, 9, 0xffffffff, 1, 8, 2,
        0xc0200000, 0xbfc00000, 0x3fc00000, 0x40200000, 0, 4, 0, 5, 4, 2, 3, 1, 0x40800000, 0, 0, 0,
    ]
    let spirv = try IndependentShaderCompiler().compileWebGPUCompute(.spirv(glslMathSPIRV()))
    let spirvPipeline = try await device.createComputePipeline(spirv)
    let compiler = IndependentShaderCompiler()
    let comparison = try compiler.compileWebGPUCompute(.spirv(floatComparisonSPIRV()))
    let comparisonPipeline = try await device.createComputePipeline(comparison)
    let remainder = try compiler.compileWebGPUCompute(.spirv(specialRemainderSPIRV()))
    let remainderPipeline = try await device.createComputePipeline(remainder)
    let roundTripSource = ShaderInput.wgsl("""
        @group(0) @binding(0) var<storage, read> input: array<vec4u, 3>;
        @group(0) @binding(1) var<storage, read_write> output: array<vec4u, 6>;
        @compute @workgroup_size(1) fn round_trip() {
            let signed = bitcast<vec4i>(input[0]);
            let unsigned = input[1];
            let values = bitcast<vec4f>(input[2]);
            output[0] = bitcast<vec4u>(round(values));
            output[1] = bitcast<vec4u>(fract(values));
            output[2] = bitcast<vec4u>(saturate(values));
            output[3] = bitcast<vec4u>(abs(signed));
            output[4] = bitcast<vec4u>(sign(signed));
            let signedDot = bitcast<u32>(dot(signed, vec4i(2, -3, 4, 5)));
            let unsignedDot = dot(unsigned, vec4u(2, 3, 4, 5));
            output[5] = vec4u(signedDot, unsignedDot, bitcast<u32>(abs(values.x)), bitcast<u32>(sign(values.z)));
        }
        """)
    guard case .spirv(let binary) = try compiler.compile(roundTripSource, to: .spirv) else {
        throw GPUError.backendFailure("Expected binary SPIR-V for numeric builtins")
    }
    let roundTripShader = try compiler.compileWebGPUCompute(.spirv(binary), entryPoint: "round_trip")
    let roundTrip = try await device.createComputePipeline(roundTripShader)
    let comparisonInput: [UInt32] = [
        0, 0xff800000, 0x7f800000, 0x7fc01234,
        0x80000000, 0x3f800000, 0xff800000, 0x7f800001,
    ]
    let remainderInput = [Int32(5), 5, -5, -5, 3, -3, 3, -3].map { UInt32(bitPattern: $0) }
        + [Float(-5.5), 5.5, -5.5, 5.5, 3, -3, -3, 3].map(\.bitPattern)
    let ticket = try await device.submitCommands([
        .write(buffer: input, offset: 0, words: numericInput),
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): input, .init(binding: 1): output],
                         workgroups: .init(x: 1)),
        .copy(source: output, destination: read, byteCount: 208), .read(buffer: read, byteCount: 208),
        .dispatchCompute(pipeline: spirvPipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
        .copy(source: output, destination: read, byteCount: 32), .read(buffer: read, byteCount: 32),
        .write(buffer: input, offset: 0, words: comparisonInput),
        .dispatchCompute(pipeline: comparisonPipeline, bindings: [.init(binding: 0): input, .init(binding: 1): output],
                         workgroups: .init(x: 1)),
        .copy(source: output, destination: read, byteCount: 240), .read(buffer: read, byteCount: 240),
        .write(buffer: input, offset: 0, words: remainderInput),
        .dispatchCompute(pipeline: remainderPipeline, bindings: [.init(binding: 0): input, .init(binding: 1): output],
                         workgroups: .init(x: 1)),
        .copy(source: output, destination: read, byteCount: 64), .read(buffer: read, byteCount: 64),
        .write(buffer: input, offset: 0, words: Array(numericInput.prefix(12))),
        .dispatchCompute(pipeline: roundTrip, bindings: [.init(binding: 0): input, .init(binding: 1): output],
                         workgroups: .init(x: 1)),
        .copy(source: output, destination: read, byteCount: 96), .read(buffer: read, byteCount: 96),
    ])
    let expectedIntegers: [UInt32] = [
        0x80000000, 7, 0, 9, 0xffffffff, 0xffffffff, 0, 1,
        3, 1, 3, 2, 0xffffffff, 3, 8, 3, 4, 2, 3, 1,
    ]
    let expectedFloats: [Float] = [
        -2, -2, 2, 2, -3, -2, 1, 2, -2, -1, 2, 3, -2, -1, 1, 2,
        0.5, 0.5, 0.5, 0.5, 0, 0, 1, 1, 2, 0.5, 0, 1, 2.5, -1, -2.5, 2.5,
    ]
    let expectedComparisons: [UInt32] = [
        1, 0, 0, 0, 1, 0, 0, 1,
        0, 1, 1, 0, 0, 1, 1, 1,
        0, 1, 0, 0, 0, 1, 0, 1,
        0, 0, 1, 0, 0, 0, 1, 1,
        1, 1, 0, 0, 1, 1, 0, 1,
        1, 0, 1, 0, 1, 0, 1, 1,
        0, 0, 0, 1, 0, 1, 1, 0,
        1, 0, 1, 0,
    ]
    let vectorRemainders = [Float(-2.5), 2.5, -2.5, 2.5, 0.5, -0.5, -2.5, 2.5].map(\.bitPattern)
    let expectedRemainders: [UInt32] = [2, .max, 1, .max - 1] + vectorRemainders
        + [2, Float(-2.5).bitPattern, Float(0.5).bitPattern, 0]
    let roundTripFloats: [Float] = [-2, -2, 2, 2, 0.5, 0.5, 0.5, 0.5, 0, 0, 1, 1]
    let roundTripExpected = roundTripFloats.map(\.bitPattern) + expectedIntegers.prefix(8)
        + [66, 43, Float(2.5).bitPattern, Float(1).bitPattern]
    let snapshots = try await ticket.wait()
    guard snapshots.count == 5, let words = snapshots.first, Array(words.prefix(20)) == expectedIntegers,
          words.dropFirst(20).map({ Float(bitPattern: $0) }) == expectedFloats,
          snapshots[1].count == 8,
          abs(Float(bitPattern: snapshots[1][0]) - 90) < 0.001,
          abs(Float(bitPattern: snapshots[1][1]) - 0.54930615) < 0.00001,
          Array(snapshots[1][2...]) == [5, 0xff00ff00, Float(5).bitPattern, Float(2).bitPattern, Float(-3).bitPattern, 2],
          snapshots[2] == expectedComparisons, snapshots[3] == expectedRemainders,
          snapshots[4] == roundTripExpected else {
        throw GPUError.backendFailure("Unexpected numeric builtin readback: \(snapshots)")
    }
    return Array(words.prefix(20)) + words.dropFirst(20).map { $0 == 0x80000000 ? 0 : $0 }
        + snapshots[1..<4].flatMap { $0 }
}

private func floatComparisonSPIRV() -> [UInt32] {
    func op(_ code: UInt32, _ operands: UInt32...) -> [UInt32] {
        [UInt32(operands.count + 1) << 16 | code] + operands
    }
    let firstConstant: UInt32 = 350
    var words: [UInt32] = [0x07230203, 0x00010300, 0, 450, 0]
    for instruction in [
        op(17, 1), op(14, 0, 1), op(15, 5, 40, 0x6e69616d, 0), op(16, 40, 17, 1, 1, 1),
        op(71, 10, 2), op(72, 10, 0, 35, 0), op(72, 10, 1, 35, 16),
        op(71, 11, 2), op(72, 11, 0, 35, 0), op(71, 12, 6, 4),
        op(71, 20, 34, 0), op(71, 20, 33, 0), op(71, 21, 34, 0), op(71, 21, 33, 1),
        op(19, 1), op(21, 2, 32, 0), op(22, 3, 32), op(23, 4, 2, 4), op(23, 5, 3, 4),
        op(20, 6), op(23, 7, 6, 4), op(43, 2, 22, 60), op(28, 12, 2, 22), op(30, 10, 4, 4), op(30, 11, 12),
        op(32, 13, 12, 10), op(32, 14, 12, 11), op(32, 15, 12, 4), op(32, 16, 12, 2), op(33, 17, 1),
    ] { words += instruction }
    for word: UInt32 in 0..<60 { words += op(43, 2, firstConstant + word, word) }
    words += op(59, 13, 20, 12) + op(59, 14, 21, 12) + op(54, 1, 40, 0, 17) + op(248, 41)
    words += op(65, 15, 100, 20, firstConstant) + op(61, 4, 101, 100) + op(124, 5, 102, 101)
    words += op(65, 15, 103, 20, firstConstant + 1) + op(61, 4, 104, 103) + op(124, 5, 105, 104)
    for offset: UInt32 in 0..<12 {
        let predicate = 110 + offset
        words += op(180 + offset, 7, predicate, 102, 105)
        for lane: UInt32 in 0..<4 {
            let word = offset * 4 + lane
            words += op(81, 6, 150 + word, predicate, lane)
            words += op(169, 2, 200 + word, 150 + word, firstConstant + 1, firstConstant)
            words += op(65, 16, 250 + word, 21, firstConstant, firstConstant + word) + op(62, 250 + word, 200 + word)
        }
    }
    words += op(156, 7, 300, 102) + op(157, 7, 301, 102) + op(154, 6, 302, 300) + op(155, 6, 303, 301)
    words += op(181, 7, 304, 102, 102) + op(155, 6, 305, 304)
    words += op(182, 7, 306, 102, 102) + op(154, 6, 307, 306)
    for lane: UInt32 in 0..<4 {
        words += op(81, 6, 310 + lane, 300, lane) + op(81, 6, 314 + lane, 301, lane)
    }
    let predicates: [UInt32] = [310, 311, 312, 313, 314, 315, 316, 317, 302, 303, 305, 307]
    for (index, predicate) in predicates.enumerated() {
        let offset = UInt32(index)
        words += op(169, 2, 420 + offset, predicate, firstConstant + 1, firstConstant)
        words += op(65, 16, 320 + offset, 21, firstConstant, firstConstant + 48 + offset)
        words += op(62, 320 + offset, 420 + offset)
    }
    return words + op(253) + op(56)
}

private func specialRemainderSPIRV() -> [UInt32] {
    func op(_ code: UInt32, _ operands: UInt32...) -> [UInt32] {
        [UInt32(operands.count + 1) << 16 | code] + operands
    }
    var words: [UInt32] = [0x07230203, 0x00010300, 0, 100, 0]
    words += op(17, 1) + op(14, 0, 1) + op(15, 5, 40, 0x6e69616d, 0) + op(16, 40, 17, 1, 1, 1)
    for block: UInt32 in [8, 9] {
        words += op(71, block, 2)
        for member: UInt32 in 0..<4 { words += op(72, block, member, 35, member * 16) }
    }
    for instruction in [
        op(71, 20, 34, 0), op(71, 20, 33, 0), op(71, 21, 34, 0), op(71, 21, 33, 1),
        op(19, 1), op(21, 2, 32, 0), op(21, 3, 32, 1), op(22, 4, 32),
        op(23, 5, 2, 4), op(23, 6, 3, 4), op(23, 7, 4, 4), op(30, 8, 5, 5, 7, 7), op(30, 9, 5, 7, 7, 5),
        op(32, 10, 12, 8), op(32, 11, 12, 9), op(32, 12, 12, 5), op(32, 13, 12, 7), op(33, 14, 1),
        op(43, 2, 15, 0), op(43, 2, 16, 1), op(43, 2, 17, 2), op(43, 2, 18, 3),
        op(59, 10, 20, 12), op(59, 11, 21, 12), op(54, 1, 40, 0, 14), op(248, 41),
        op(65, 12, 44, 20, 15), op(61, 5, 45, 44), op(65, 12, 46, 20, 16), op(61, 5, 47, 46), op(124, 6, 48, 47),
        op(65, 13, 50, 20, 17), op(61, 7, 51, 50), op(65, 13, 52, 20, 18), op(61, 7, 53, 52),
        op(139, 5, 54, 45, 48), op(140, 7, 55, 51, 53), op(141, 7, 56, 51, 53),
        op(81, 2, 57, 45, 0), op(81, 3, 58, 48, 0), op(139, 2, 59, 57, 58),
        op(81, 4, 60, 51, 0), op(81, 4, 61, 53, 0), op(140, 4, 62, 60, 61), op(141, 4, 63, 60, 61),
        op(124, 2, 64, 62), op(124, 2, 65, 63), op(80, 5, 66, 59, 64, 65, 15),
        op(65, 12, 70, 21, 15), op(62, 70, 54), op(65, 13, 71, 21, 16), op(62, 71, 55),
        op(65, 13, 72, 21, 17), op(62, 72, 56), op(65, 12, 73, 21, 18), op(62, 73, 66), op(253), op(56),
    ] { words += instruction }
    return words
}

private func glslMathSPIRV() -> [UInt32] {
    func op(_ code: UInt32, _ operands: UInt32...) -> [UInt32] {
        [UInt32(operands.count + 1) << 16 | code] + operands
    }
    var words: [UInt32] = [0x07230203, 0x00010300, 0, 100, 0] + op(17, 1) + op(11, 80, 0x4c534c47, 0x6474732e, 0x3035342e, 0) + op(14, 0, 1)
    for instruction in [
        op(15, 5, 10, 0x6e69616d, 0), op(16, 10, 17, 1, 1, 1),
        op(71, 6, 2), op(72, 6, 0, 35, 0), op(72, 6, 1, 35, 4), op(72, 6, 2, 35, 8),
        op(72, 6, 3, 35, 12), op(72, 6, 4, 35, 16), op(72, 6, 5, 35, 20),
        op(72, 6, 6, 35, 24), op(72, 6, 7, 35, 28), op(71, 11, 34, 0), op(71, 11, 33, 0),
        op(19, 1), op(21, 2, 32, 0), op(22, 3, 32), op(23, 4, 3, 4), op(23, 5, 3, 3),
        op(30, 6, 3, 3, 2, 2, 3, 3, 3, 2), op(32, 7, 12, 6), op(32, 8, 12, 3), op(32, 9, 12, 2), op(33, 12, 1),
        op(43, 3, 20, Float(90).bitPattern), op(43, 3, 21, 0), op(43, 3, 22, Float(1).bitPattern),
        op(43, 3, 23, Float(0.5).bitPattern), op(43, 3, 24, Float(3).bitPattern), op(43, 3, 25, Float(4).bitPattern),
        op(43, 2, 26, UInt32(bitPattern: -5)), op(44, 4, 27, 21, 22, 21, 22), op(44, 5, 28, 24, 25, 21),
        op(43, 2, 30, 0), op(43, 2, 31, 1), op(43, 2, 32, 2), op(43, 2, 33, 3),
        op(43, 2, 34, 4), op(43, 2, 35, 5), op(43, 2, 36, 6), op(43, 2, 37, 7),
        op(59, 7, 11, 12), op(54, 1, 10, 0, 12), op(248, 40),
        op(12, 3, 41, 80, 11, 20),
        op(12, 3, 42, 80, 12, 41), op(12, 3, 43, 80, 22, 21), op(12, 3, 44, 80, 23, 22), op(12, 3, 45, 80, 24, 23),
        op(129, 3, 46, 43, 44), op(129, 3, 47, 46, 45), op(12, 2, 48, 80, 5, 26),
        op(12, 2, 49, 80, 55, 27), op(12, 3, 50, 80, 66, 28), op(12, 3, 51, 80, 53, 23, 32),
        op(12, 3, 58, 80, 71, 24, 22), op(12, 2, 59, 80, 74, 26),
        op(65, 8, 52, 11, 30), op(62, 52, 42), op(65, 8, 53, 11, 31), op(62, 53, 47),
        op(65, 9, 54, 11, 32), op(62, 54, 48), op(65, 9, 55, 11, 33), op(62, 55, 49),
        op(65, 8, 56, 11, 34), op(62, 56, 50), op(65, 8, 57, 11, 35), op(62, 57, 51),
        op(65, 8, 60, 11, 36), op(62, 60, 58), op(65, 9, 61, 11, 37), op(62, 61, 59), op(253), op(56),
    ] {
        words += instruction
    }
    return words
}

private func transcendentalDiagnostic(_ device: Device) async throws -> [UInt32] {

    let shader = try IndependentShaderCompiler().compileCompute(.wgsl("""
        const_assert exp2(4.0) == 16.0;
        const_assert log2(0.125) == -3.0;
        const_assert log2(4.9406564584124654e-324) == -1074.0;
        const_assert log2(exp2(-1074.0)) == -1074.0;
        const_assert abs(exp2(1023.5) / exp2(1023.0) - 1.4142135623730951) < 1e-14;
        const_assert abs(log(exp(0.5)) - 0.5) < 1e-14;
        const_assert abs(atan2(1.0, -1.0) - 2.356194490192345) < 1e-14;
        const_assert abs(sin(1e300) + 0.8178819121159085) < 1e-14;
        const_assert abs(cos(1e30f) + 0.6116048f) < 2e-6f;
        const_assert abs(tan(1e100) + 0.41162296288324984) < 1e-14;
        const_assert abs(asinh(sinh(-2.0)) + 2.0) < 1e-14;
        const_assert abs(acosh(cosh(2.0)) - 2.0) < 1e-14;
        const_assert abs(atanh(tanh(0.5)) - 0.5) < 1e-14;
        const_assert abs(pow(vec2f(9, 16), vec2f(0.5)).y - 4.0f) < 1e-6f;
        @group(0) @binding(0) var<storage, read> input: array<vec4f, 3>;
        @group(0) @binding(1) var<storage, read_write> output: array<vec4f, 6>;
        @compute @workgroup_size(1) fn run() {
            let x = input[0];
            let y = input[1];
            let angle = atan2(x.z, x.w);
            output[0] = vec4f(sin(x.x), cos(x.y), tan(x.x), angle);
            output[1] = vec4f(asin(x.x), acos(x.x), atan(x.z), degrees(angle));
            output[2] = vec4f(sinh(x.x), cosh(x.x), tanh(x.y), radians(y.w));
            output[3] = vec4f(asinh(x).y, acosh(y).x, atanh(x.x), pow(y.z, x.x));
            output[4] = vec4f(exp(x.x), exp2(y.x), log(y.x), log2(y.y));
            output[5] = exp2(input[2]);
        }
        """), to: .wgsl)
    let pipeline = try await device.createComputePipeline(shader)
    let input = try await device.createBuffer(.init(size: 48, usage: [.storage, .copyDestination]))
    let output = try await device.createBuffer(.init(size: 96, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 96, usage: [.mapRead, .copyDestination]))
    let transcendentalInput: [Float] = [0.5, -0.5, 1, -1, 2, 8, 9, 180, 1, 2, 3, 4]
    let ticket = try await device.submitCommands([
        .write(buffer: input, offset: 0, words: transcendentalInput.map(\.bitPattern)),
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): input, .init(binding: 1): output],
                         workgroups: .init(x: 1)),
        .copy(source: output, destination: read, byteCount: 96), .read(buffer: read, byteCount: 96),
    ])
    let expected: [Float] = [
        0.47942555, 0.87758255, 0.5463025, 2.3561945,
        0.5235988, 1.0471976, 0.7853982, 135,
        0.5210953, 1.127626, -0.46211717, 3.1415927,
        -0.48121184, 1.316958, 0.54930615, 3,
        1.6487213, 4, 0.6931472, 3, 2, 4, 8, 16,
    ]
    let snapshots = try await ticket.wait()
    guard let words = snapshots.first, words.count == expected.count,
          zip(words, expected).allSatisfy({ abs(Float(bitPattern: $0) - $1) <= max(1, abs($1)) * 0.0004 }) else {
        throw GPUError.backendFailure("Unexpected transcendental builtin readback: \(snapshots)")
    }
    return words
}

private func decompositionDiagnostic(_ device: Device) async throws -> [UInt32] {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl("""
        const abstractParts = frexp(-0.125);
        const_assert abstractParts.fract == -0.5 && abstractParts.exp == -2;
        const_assert frexp(vec3(8, -1, 3)).exp.z == 2;
        const_assert modf(vec4f(-1, -0.25, 2.5, 3.75)).fract.w == 0.75f;
        const_assert modf(1e300).whole == 1e300;
        const_assert modf(-3.25f).fract == -0.25f;
        const_assert modf(vec2f(-3.25, 2.75)).whole.y == 2.0f;
        const_assert frexp(vec2f(8, -0.125)).exp.y == -2i;
        const_assert ldexp(0.5, 1024) == 0x1p1023;
        const_assert ldexp(0.5f, 128) == 0x1p127f;
        @group(0) @binding(0) var<storage, read> input: array<vec4f, 2>;
        @group(0) @binding(1) var<storage, read_write> output: array<vec4f, 6>;
        @compute @workgroup_size(1) fn run() {
            const parts = modf(vec3(-1.75, 5.25, 3.0));
            const_assert parts.whole.x == -1.0;
            let x = input[0];
            let fraction = frexp(x);
            let whole = modf(x.zw);
            var scalar = frexp(x.y);
            let huge = frexp(1e300);
            scalar.exp += huge.exp - 997i;
            let triple = modf(x.xyz);
            output[0] = fraction.fract;
            output[1] = vec4f(fraction.exp);
            output[2] = vec4f(whole.fract, whole.whole);
            output[3] = vec4f(scalar.fract, f32(scalar.exp), triple.fract.z, parts.fract.y);
            let boundary = input[1];
            let split = frexp(boundary);
            output[4] = ldexp(split.fract, split.exp);
            output[5] = vec4f(ldexp(boundary.y, -4), ldexp(boundary.z, 3i), ldexp(0.5, split.exp.x), ldexp(0.5f, 128));
        }
        """), to: .wgsl)
    let pipeline = try await device.createComputePipeline(shader)
    let input = try await device.createBuffer(.init(size: 32, usage: [.storage, .copyDestination]))
    let output = try await device.createBuffer(.init(size: 96, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 96, usage: [.mapRead, .copyDestination]))
    let values: [Float] = [8, -0.125, -3.25, 2.75, .greatestFiniteMagnitude, -3.25, 0.125, .leastNormalMagnitude]
    let spirv = try compiler.compileWebGPUCompute(.spirv(glslDecompositionSPIRV()))
    let spirvPipeline = try await device.createComputePipeline(spirv)
    let roundTripSource = ShaderInput.wgsl("""
        @group(0) @binding(0) var<storage, read> input: array<vec4f, 2>;
        @group(0) @binding(1) var<storage, read_write> output: array<vec4f, 6>;
        @compute @workgroup_size(1) fn round_trip() {
            let x = input[0];
            let split = frexp(x);
            let parts = modf(x.zw);
            output[0] = split.fract;
            output[1] = vec4f(split.exp);
            output[2] = vec4f(parts.fract, parts.whole);
            output[3] = ldexp(split.fract, split.exp);

            let a = vec3f(x.x * 0.375, x.x * 0.5, 0.0);
            let b = vec3f(0.0, 0.0, -8.0 * x.y);
            let incoming = vec3f(0.0, 0.0, -b.z);
            let normal = normalize(vec3f(0.0, a.y, 0.0));
            output[4] = vec4f(cross(a, b).xy, length(a), distance(a.xy, b.xy));
            output[5] = vec4f(normal.y, reflect(incoming, b).z,
                              refract(incoming, b, 0.5).z, faceForward(b, incoming, b).z);
        }
        """)
    guard case .spirv(let binary) = try compiler.compile(roundTripSource, to: .spirv) else {
        throw GPUError.backendFailure("Expected binary SPIR-V for geometry and decomposition")
    }
    let roundTripShader = try compiler.compileWebGPUCompute(.spirv(binary), entryPoint: "round_trip")
    let roundTrip = try await device.createComputePipeline(roundTripShader)
    let ticket = try await device.submitCommands([
        .write(buffer: input, offset: 0, words: values.map(\.bitPattern)),
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): input, .init(binding: 1): output],
                         workgroups: .init(x: 1)),
        .copy(source: output, destination: read, byteCount: 96), .read(buffer: read, byteCount: 96),
        .dispatchCompute(pipeline: spirvPipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
        .copy(source: output, destination: read, byteCount: 48), .read(buffer: read, byteCount: 48),
        .dispatchCompute(pipeline: roundTrip, bindings: [.init(binding: 0): input, .init(binding: 1): output],
                         workgroups: .init(x: 1)),
        .copy(source: output, destination: read, byteCount: 96), .read(buffer: read, byteCount: 96),
    ])
    let snapshots = try await ticket.wait()
    let expected: [Float] = [-0.25, 0.125, 0.5, -3, 0, 8, -0.8125, 0.5, 0.53125]
    let roundTripExpected: [Float] = [
        0.5, -0.5, -0.8125, 0.6875, 4, -2, 2, 2,
        -0.25, 0.75, -3, 2, 8, -0.125, -3.25, 2.75,
        4, -3, 5, 5, 1, 1, -1, 1,
    ]
    guard snapshots.count == 3, let words = snapshots.first, words.count == 24,
          snapshots[1] == expected.map(\.bitPattern) + [2, UInt32(bitPattern: -2), 4],
          snapshots[2] == roundTripExpected.map(\.bitPattern) else {
        throw GPUError.backendFailure("Unexpected decomposition readback: \(snapshots)")
    }
    return words + snapshots[1]
}

private func glslDecompositionSPIRV() -> [UInt32] {
    func op(_ code: UInt32, _ operands: UInt32...) -> [UInt32] {
        [UInt32(operands.count + 1) << 16 | code] + operands
    }
    var words: [UInt32] = [0x07230203, 0x00010300, 0, 100, 0] + op(17, 1) + op(11, 80, 0x4c534c47, 0x6474732e, 0x3035342e, 0) + op(14, 0, 1)
    words += op(15, 5, 10, 0x6e69616d, 0) + op(16, 10, 17, 1, 1, 1)
    words += op(6, 5, 1, 0x74726170, 0) + op(71, 11, 34, 0) + op(71, 11, 33, 0) + op(71, 7, 2)
    for index: UInt32 in 0..<12 { words += op(72, 7, index, 35, index * 4) }
    for instruction in [
        op(19, 1), op(21, 2, 32, 0), op(22, 3, 32), op(23, 4, 3, 3), op(23, 13, 2, 3),
        op(30, 5, 4, 4), op(30, 6, 4, 13),
        op(30, 7, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2),
        op(32, 8, 12, 7), op(32, 9, 12, 3), op(32, 12, 12, 2), op(33, 14, 1),
        op(43, 3, 20, Float(-3.25).bitPattern), op(43, 3, 21, Float(0.125).bitPattern), op(43, 3, 22, Float(8.5).bitPattern),
        op(44, 4, 23, 20, 21, 22),
    ] {
        words += instruction
    }
    for index: UInt32 in 0..<12 { words += op(43, 2, 24 + index, index) }
    words += op(59, 8, 11, 12) + op(54, 1, 10, 0, 14) + op(248, 40)
    words += op(12, 5, 41, 80, 36, 23) + op(12, 6, 42, 80, 52, 23) + op(12, 5, 43, 80, 36, 23)
    for index: UInt32 in 0..<12 {
        let type: UInt32 = index < 9 ? 3 : 2
        let result: UInt32 = index < 6 ? 41 : 42
        words += op(81, type, 44 + index, result, (index / 3) % 2, index % 3)
        words += op(65, index < 9 ? 9 : 12, 56 + index, 11, 24 + index)
        words += op(62, 56 + index, 44 + index)
    }
    return words + op(253) + op(56)
}

private func halfPrecisionDiagnostic(_ device: Device) async throws -> [UInt32] {
    guard device.capabilities.features.contains(.shaderF16) else { return [] }
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl("""
        enable f16;
        alias HalfPair = vec2h;
        const half: f16 = 1.00048828125;
        const_assert half == 1.0h;
        const_assert f16(0x1.002p0) == 0x1p0h;
        const_assert f16(2i) == 2h;
        const_assert f32(f16(1.0009765625)) == 1.0009765625f;
        const_assert modf(vec2h(-3.25, 2.75)).whole.x == -3h;
        const_assert frexp(vec3h(4, -0.125, 65504)).exp.z == 16i;
        const_assert ldexp(0.5h, 16) == 32768h;
        const_assert ldexp(0.5h, -23i) == 0x1p-24h;
        const_assert ldexp(0.5h, 1i) + 0x1p-11h == 1h;
        @group(0) @binding(0) var<storage, read> input: array<vec4f, 1>;
        @group(0) @binding(1) var<storage, read_write> output: array<vec4f, 3>;
        fn splitPair(x: HalfPair) -> vec4f {
            var parts = modf(x);
            parts.whole = -parts.whole;
            return vec4f(vec2f(parts.fract), vec2f(parts.whole));
        }
        @compute @workgroup_size(1) fn run() {
            let x = vec4h(input[0]);
            let parts = frexp(x);
            let triple = frexp(x.xyz);
            let scalar = frexp(x.y);
            output[0] = vec4f(ldexp(parts.fract, parts.exp));
            output[1] = splitPair(x.yz);
            output[2] = vec4f(f32(triple.fract.x), f32(triple.exp.z), f32(scalar.fract), f32(scalar.exp));
        }
        """), to: .wgsl)
    let pipeline = try await device.createComputePipeline(shader)
    let input = try await device.createBuffer(.init(size: 16, usage: [.storage, .copyDestination]))
    let output = try await device.createBuffer(.init(size: 48, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 48, usage: [.mapRead, .copyDestination]))
    let values: [Float] = [65504, -3.25, 2.75, 0x1p-14]
    let ticket = try await device.submitCommands([
        .write(buffer: input, offset: 0, words: values.map(\.bitPattern)),
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): input, .init(binding: 1): output],
                         workgroups: .init(x: 1)),
        .copy(source: output, destination: read, byteCount: 48), .read(buffer: read, byteCount: 48),
    ])
    let snapshots = try await ticket.wait()
    guard let words = snapshots.first, words.count == 12 else {
        throw GPUError.backendFailure("Unexpected half-precision readback: \(snapshots)")
    }
    return words
}

private func halfBufferDiagnostic(_ device: Device) async throws -> [UInt32] {
    guard device.capabilities.features.contains(.shaderF16) else { return [] }
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl("""
        enable f16;
        struct Input {
            low: f16,
            high: f16,
            @align(8) values: array<vec3h>,
        }
        struct Output {
            @align(4) @size(4) mark: f16,
            values: array<f16>,
        }
        struct Settings {
            low: f16,
            bias: f16,
            tag: u32,
            lanes: vec3h,
            tail: f16,
        }
        @group(0) @binding(0) var<storage, read> input: Input;
        @group(0) @binding(1) var<storage, read_write> output: Output;
        @group(0) @binding(2) var<storage, read_write> vectors: array<vec3h>;
        @group(0) @binding(3) var<uniform> settings: Settings;
        fn sample(i: u32) -> vec3h { return input.values[i]; }
        @compute @workgroup_size(1)
        fn run(@builtin(global_invocation_id) id: vec3u) {
            let v = sample(id.x);
            vectors[id.x] = v;
            output.values[id.x] = v.x + settings.bias;
            if id.x == 0u {
                output.mark = f16(arrayLength(&input.values) + arrayLength(&output.values));
                vectors[0u].z = settings.tail;
            }
        }
        """), to: .wgsl)
    let pipeline = try await device.createComputePipeline(shader)
    let roundTripSource = ShaderInput.wgsl("""
        enable f16;
        struct Input { low: f16, high: f16, reserved: u32, values: array<vec3h> }
        struct Output { mark: f16, sentinel: f16, values: array<f16, 6> }
        @group(0) @binding(0) var<storage, read> input: Input;
        @group(0) @binding(1) var<storage, read_write> output: Output;
        @group(0) @binding(2) var<storage, read_write> vectors: array<vec3h, 3>;
        @compute @workgroup_size(1)
        fn round_trip(@builtin(global_invocation_id) id: vec3u) {
            let v = input.values[id.x];
            let transformed = v * vec3h(2.0h, -1.0h, 0.5h) + vec3h(0.5h, 0.25h, -0.5h);
            vectors[id.x] = transformed;
            output.values[id.x] = transformed.x + transformed.z;
            output.values[id.x + 3u] = transformed.y;
            if id.x == 0u { output.mark = input.high - input.low; }
        }
        """)
    guard case .spirv(let binary) = try compiler.compile(roundTripSource, to: .spirv) else {
        throw GPUError.backendFailure("Expected binary SPIR-V for f16 storage")
    }
    let roundTripShader = try compiler.compileWebGPUCompute(.spirv(binary), entryPoint: "round_trip")
    guard roundTripShader.requiresShaderF16 else {
        throw GPUError.backendFailure("Expected shader-f16 after the SPIR-V round trip")
    }
    let roundTrip = try await device.createComputePipeline(roundTripShader)
    let input = try await device.createBuffer(.init(size: 32, usage: [.storage, .copyDestination]))
    let output = try await device.createBuffer(.init(size: 16, usage: [.storage, .copyDestination, .copySource]))
    let vectors = try await device.createBuffer(.init(size: 24, usage: [.storage, .copyDestination, .copySource]))
    let settings = try await device.createBuffer(.init(size: 16, usage: [.uniform, .copyDestination]))
    let readScalars = try await device.createBuffer(.init(size: 16, usage: [.mapRead, .copyDestination]))
    let readVectors = try await device.createBuffer(.init(size: 24, usage: [.mapRead, .copyDestination]))
    func pair(_ low: Double, _ high: Double) -> UInt32 {
        UInt32(Binary16.encode(low)) | (UInt32(Binary16.encode(high)) << 16)
    }
    let ticket = try await device.submitCommands([
        .write(buffer: input, offset: 0, words: [
            pair(50, 60), 0, pair(1, 2), pair(3, -9), pair(4, 5), pair(6, -9), pair(7, 8), pair(9, -9),
        ]),
        .write(buffer: output, offset: 0, words: [pair(99, -7), 0, 0, 0]),
        .write(buffer: vectors, offset: 0, words: Array(repeating: [0, pair(0, -7)], count: 3).flatMap { $0 }),
        .write(buffer: settings, offset: 0, words: [pair(-5, 0.5), 0, pair(11, 12), pair(13, 2.5)]),
        .dispatchCompute(pipeline: pipeline, bindings: [
            .init(binding: 0): input, .init(binding: 1): output, .init(binding: 2): vectors, .init(binding: 3): settings,
        ], workgroups: .init(x: 8)),
        .copy(source: output, destination: readScalars, byteCount: 16), .read(buffer: readScalars, byteCount: 16),
        .copy(source: vectors, destination: readVectors, byteCount: 24), .read(buffer: readVectors, byteCount: 24),
        .dispatchCompute(pipeline: roundTrip, bindings: [
            .init(binding: 0): input, .init(binding: 1): output, .init(binding: 2): vectors,
        ], workgroups: .init(x: 3)),
        .copy(source: output, destination: readScalars, byteCount: 16), .read(buffer: readScalars, byteCount: 16),
        .copy(source: vectors, destination: readVectors, byteCount: 24), .read(buffer: readVectors, byteCount: 24),
    ])
    let snapshots = try await ticket.wait()
    let roundTripExpected: [[UInt32]] = [
        [pair(10, -7), pair(3.5, 11), pair(18.5, -1.75), pair(-4.75, -7.75)],
        [pair(2.5, -1.75), pair(1, -7), pair(8.5, -4.75), pair(2.5, -7), pair(14.5, -7.75), pair(4, -7)],
    ]
    guard snapshots.map(\.count) == [4, 6, 4, 6], snapshots.suffix(2).elementsEqual(roundTripExpected) else {
        throw GPUError.backendFailure("Unexpected half-precision buffer readback: \(snapshots)")
    }
    return snapshots.flatMap { $0 }
}

private func halfMatrixDiagnostic(_ device: Device) async throws -> [UInt32] {
    guard device.capabilities.features.contains(.shaderF16) else { return [] }
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl("""
        enable f16;
        const rounded = mat2x2h(mat2x2f(1.00048828125, 2, 3, 4));
        const_assert rounded[0][0] == 1h;
        const_assert determinant(mat2x2<f16>(1, 2, 3, 4)) == -2h;
        const_assert transpose(mat2x3(1h, 2, 3, 4, 5, 6))[2].y == 6h;
        const_assert (mat2x2h(1, 2, 3, 4) * vec2(2, 3)).y == 16h;
        struct Data { basis: mat2x3h, transforms: array<mat2x3h> }
        @group(0) @binding(0) var<storage, read_write> data: Data;
        @group(0) @binding(1) var<uniform> scale: mat2x2<f16>;
        @group(0) @binding(2) var<storage, read_write> output: array<vec4f, 4>;
        fn flip(value: mat2x3h) -> mat3x2h { return transpose(value); }
        @compute @workgroup_size(1)
        fn run(@builtin(global_invocation_id) id: vec3u) {
            let saved = data.basis;
            let turned = flip(saved);
            let gram = turned * saved;
            let widened = mat2x3f(saved);
            let adjusted = 2 * (mat2x3h(widened) + saved - saved);
            data.basis = saved * scale;
            let i = id.x + 1u;
            data.transforms[i] = data.transforms[0u] * scale;
            data.transforms[i][id.x] = vec3h(21, 22, 23);
            data.transforms[i][1u][1u] = 55h;
            data.transforms[id.x + 2u] = saved;
            let missing = data.transforms[id.x + 2u];
            output[0] = vec4f(vec3f(saved * vec2h(2, 3)), f32(determinant(scale)));
            output[1] = vec4f(vec2f(vec3h(1, 2, 3) * saved), vec2f(turned[2]));
            output[2] = vec4f(f32(missing[0].x), f32(arrayLength(&data.transforms)), f32(data.transforms[i][1].y), f32(adjusted[1].z));
            output[3] = vec4f(vec2f(gram[0]), vec2f(gram[1]));
        }
        """), to: .wgsl)
    let pipeline = try await device.createComputePipeline(shader)
    let data = try await device.createBuffer(.init(size: 52, usage: [.storage, .copyDestination, .copySource]))
    let scale = try await device.createBuffer(.init(size: 8, usage: [.uniform, .copyDestination]))
    let output = try await device.createBuffer(.init(size: 64, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 64, usage: [.copyDestination, .mapRead]))
    func pair(_ x: Double, _ y: Double) -> UInt32 {
        UInt32(Binary16.encode(x)) | (UInt32(Binary16.encode(y)) << 16)
    }
    let initial: [UInt32] = [
        pair(1, 2), pair(3, -1), pair(4, 5), pair(6, -2),
        pair(7, 8), pair(9, -3), pair(10, 11), pair(12, -4),
        pair(13, 14), pair(15, -5), pair(16, 17), pair(18, -6), 999,
    ]
    let ticket = try await device.submitCommands([
        .write(buffer: data, offset: 0, words: initial),
        .write(buffer: scale, offset: 0, words: [pair(2, 0), pair(0, 3)]),
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): data, .init(binding: 1): scale,
                         .init(binding: 2): output], workgroups: .init(x: 1)),
        .copy(source: output, destination: read, byteCount: 64), .read(buffer: read, byteCount: 64),
        .copy(source: data, destination: read, byteCount: 52), .read(buffer: read, byteCount: 52),
    ])
    let snapshots = try await ticket.wait()
    guard snapshots.map(\.count) == [16, 13] else {
        throw GPUError.backendFailure("Unexpected half-precision matrix readback: \(snapshots)")
    }
    return snapshots.flatMap { $0 }
}

private func halfGeometryDiagnostic(_ device: Device) async throws -> [UInt32] {
    guard device.capabilities.features.contains(.shaderF16) else { return [] }
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl("""
        enable f16;
        const_assert dot(vec2h(1, 0x1p-11h), vec2(1.0, 1.0)) == 1h;
        const_assert length(vec2h(60000, 0)) == 60000h;
        const_assert normalize(vec2h(0x1p-14h, 0)).x == 1h;
        const_assert distance(3h, -2) == 5h;
        const_assert cross(vec3h(1, 0, 0), vec3(0.0, 1.0, 0.0)).z == 1h;
        const_assert reflect(vec2h(0, -1), vec2h(0, 1)).y == 1h;
        const_assert refract(vec2h(1, 0), vec2h(0, 1), 2.0).x == 0h;
        const_assert faceForward(vec2h(0, 1), vec2h(0, -1), vec2h(0, 1)).y == 1h;
        @group(0) @binding(0) var<storage, read> input: array<vec4h, 4>;
        @group(0) @binding(1) var<storage, read_write> output: array<vec4f, 7>;
        @compute @workgroup_size(1) fn run() {
            let a = input[0];
            let b = input[1];
            let axis = input[2].xyz;
            let settings = input[3];
            let incidence = vec3h(0, -axis.y, 0);
            let tangent = vec3h(1, 0, 0);
            output[0] = vec4f(f32(dot(a.xy, a.xy)), f32(length(a.xy)), f32(distance(a.xy, b.xy)), f32(distance(a.x, -2.0)));
            output[1] = vec4f(vec3f(cross(a.xyz, b.xyz)), f32(length(-a.y)));
            output[2] = vec4f(vec3f(normalize(vec3h(0, a.y, 0))), f32(dot(a.xyz, b.xyz)));
            output[3] = vec4f(vec3f(faceForward(axis, incidence, axis)), f32(faceForward(axis, axis, axis).y));
            output[4] = vec4f(vec3f(reflect(incidence, axis)), f32(refract(incidence, axis, settings.z).y));
            output[5] = vec4f(vec3f(refract(tangent, axis, settings.y)), f32(dot(a, b)));
            let wide = vec3h(settings.x, 0, 0);
            let tiny = vec3h(0, settings.w, 0);
            output[6] = vec4f(f32(length(wide)), f32(normalize(wide).x),
                              f32(normalize(tiny).y), f32(distance(wide, vec3h(0))));
        }
        """), to: .wgsl)
    let pipeline = try await device.createComputePipeline(shader)
    let input = try await device.createBuffer(.init(size: 32, usage: [.storage, .copyDestination]))
    let output = try await device.createBuffer(.init(size: 112, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 112, usage: [.mapRead, .copyDestination]))
    let data: [Double] = [3, 4, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 60000, 2, 0.5, 0x1p-14]
    let words = stride(from: 0, to: data.count, by: 2).map { i in
        UInt32(Binary16.encode(data[i])) | (UInt32(Binary16.encode(data[i + 1])) << 16)
    }
    let ticket = try await device.submitCommands([
        .write(buffer: input, offset: 0, words: words),
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): input, .init(binding: 1): output],
                         workgroups: .init(x: 1)),
        .copy(source: output, destination: read, byteCount: 112), .read(buffer: read, byteCount: 112),
    ])
    let snapshots = try await ticket.wait()
    guard let snapshot = snapshots.first, snapshot.count == 28 else {
        throw GPUError.backendFailure("Unexpected half-precision geometry readback: \(snapshots)")
    }
    return snapshot
}

private func halfBitcastDiagnostic(_ device: Device) async throws -> [UInt32] {
    guard device.capabilities.features.contains(.shaderF16) else { return [] }
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl("""
        enable f16;
        alias HalfPair = vec2h;
        const PAIR = bitcast<HalfPair>(0xc0003c00u);
        const_assert PAIR.x == 1h && PAIR.y == -2h;
        const_assert bitcast<u32>(vec2h(-0.0h, 0x1p-24h)) == 0x00018000u;
        const_assert bitcast<vec2u>(bitcast<vec4h>(vec2u(0xc0003c00u, 0x04008000u))).y == 0x04008000u;
        const_assert bitcast<u32>(bitcast<vec2h>(1.0)) == 0x3f800000u;
        const_assert bitcast<u32>(vec2(1, 2)) == 0x40003c00u;
        const_assert bitcast<f16>(0.5) == 0.5h;
        @group(0) @binding(0) var<storage, read> input: array<vec4u, 1>;
        @group(0) @binding(1) var<storage, read_write> output: array<vec4u, 5>;
        struct UniformBits { basis: mat2x2h, halfWords: vec4h }
        @group(0) @binding(2) var<uniform> settings: UniformBits;
        @compute @workgroup_size(1) fn run() {
            let words = input[0];
            let halves = bitcast<vec4h>(words.xy);
            let fromSigned = bitcast<vec4h>(bitcast<vec2i>(words.zw));
            let fromFloats = bitcast<vec4h>(bitcast<vec2f>(words.zw));
            output[0] = vec4u(bitcast<vec2u>(halves), bitcast<vec2u>(fromSigned));
            output[1] = vec4u(bitcast<u32>(bitcast<vec2h>(words.x)), bitcast<u32>(bitcast<i32>(halves.xy)),
                              bitcast<u32>(bitcast<f32>(fromFloats.xy)), bitcast<u32>(bitcast<vec2h>(bitcast<f32>(words.w))));
            output[2] = vec4u(bitcast<vec2u>(bitcast<vec2f>(fromFloats)), bitcast<vec2u>(bitcast<vec2i>(halves)));
            let same = bitcast<vec3h>(halves.xyz);
            output[3] = vec4u(bitcast<u32>(vec2h(same.x, bitcast<f16>(same.y))), bitcast<u32>(halves.zw),
                              bitcast<u32>(PAIR), bitcast<u32>(vec2h(PAIR.y, PAIR.x)));
            output[4] = vec4u(bitcast<vec2u>(settings.halfWords), bitcast<u32>(settings.basis[0]), bitcast<u32>(settings.basis[1]));
        }
        """), to: .wgsl)
    let pipeline = try await device.createComputePipeline(shader)
    let normalizedBoundary = try IndependentShaderCompiler(wgslSourceProfile: .nagaNative).compile(.wgsl("""
        enable f16;
        const positive: f32 = 3.40282347e38;
        const halfMaximum: f16 = 65519.0;
        const width: u32 = 1u;
        const height: i32 = 1i;
        @group(0) @binding(0) var<storage, read_write> output: array<u32, 4>;

        @compute @workgroup_size(width, height)
        fn run() {
            let negative: f32 = -3.40282347e38;
            let halfMinimum: f16 = -65519;
            output[0] = bitcast<u32>(positive);
            output[1] = bitcast<u32>(negative);
            output[2] = bitcast<u32>(vec2h(halfMaximum, halfMinimum));
            let consistent = (1.5 + 2) == (2 + 1.5);
            output[3] = select(0u, bitcast<u32>(f32(3.40282347e38)), consistent);
        }
        """), to: .wgsl)
    guard case .text(.wgsl, let boundarySource) = normalizedBoundary else {
        throw GPUError.backendFailure("Expected normalized WGSL for the float boundary diagnostic")
    }
    let boundaryShader = try IndependentShaderCompiler().compileWebGPUCompute(.wgsl(boundarySource))
    let boundaryPipeline = try await device.createComputePipeline(boundaryShader)
    let input = try await device.createBuffer(.init(size: 16, usage: [.storage, .copyDestination]))
    let output = try await device.createBuffer(.init(size: 80, usage: [.storage, .copySource]))
    let settings = try await device.createBuffer(.init(size: 16, usage: [.uniform, .copyDestination]))
    let read = try await device.createBuffer(.init(size: 80, usage: [.mapRead, .copyDestination]))
    let words: [UInt32] = [0xc0003c00, 0x04008000, 0xbc004400, 0x3c003800]
    let ticket = try await device.submitCommands([
        .write(buffer: input, offset: 0, words: words),
        .write(buffer: settings, offset: 0, words: [0x00018000, 0x04003c00, 0x80000001, 0x4400c000]),
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): input, .init(binding: 1): output,
                         .init(binding: 2): settings],
                         workgroups: .init(x: 1)),
        .copy(source: output, destination: read, byteCount: 80), .read(buffer: read, byteCount: 80),
        .dispatchCompute(pipeline: boundaryPipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
        .copy(source: output, destination: read, byteCount: 16), .read(buffer: read, byteCount: 16),
    ])
    let snapshots = try await ticket.wait()
    guard snapshots.map(\.count) == [20, 4] else {
        throw GPUError.backendFailure("Unexpected half bitcast readback: \(snapshots)")
    }
    return snapshots.flatMap { $0 }
}

private func halfNumericDiagnostic(_ device: Device) async throws -> [UInt32] {
    guard device.capabilities.features.contains(.shaderF16) else { return [] }
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl("""
        enable f16;
        const_assert fma(0x1.004p0h, 0x1.ff8p-1h, -1h) == -0x1p-20h;
        const_assert fma(60000h, 2h, -60000h) == 60000h;
        const_assert mix(vec2h(1, 2), vec2h(5, 6), 0.25).y == 3h;
        const_assert mix(vec2f(1, 2), vec2f(5, 6), 0.25).x == 2.0f;
        const_assert smoothstep(1h, 0h, 0.25h) == 0.84375h;
        const_assert step(vec2h(0), vec2h(-1, 0)).y == 1h;
        const_assert exp2(4h) == 16h;
        @group(0) @binding(0) var<storage, read> input: array<vec4h, 4>;
        @group(0) @binding(1) var<storage, read_write> output: array<vec4f, 16>;
        @compute @workgroup_size(1) fn run() {
            let a = input[0];
            let b = input[1];
            let c = input[2];
            let d = input[3];
            output[0] = vec4f(round(a));
            output[1] = vec4f(clamp(a, vec4(-1.0), vec4(1.0)));
            output[2] = vec4f(saturate(a));
            output[3] = vec4f(fract(a));
            output[4] = vec4f(fma(b, vec4h(2), vec4h(-1)));
            output[5] = vec4f(mix(vec4h(0), b, c.x));
            output[6] = vec4f(mix(vec4h(0), b, vec4h(c.x, 0.5h, 0.75h, 1h)));
            output[7] = vec4f(f32(step(c.z, c.x)), f32(step(c.x, c.z)),
                              f32(smoothstep(c.y, b.w, c.z)), f32(smoothstep(c.z, c.w, c.x)));
            let blend = smoothstep(vec2h(c.y, c.z), vec2h(b.w, c.w), vec2h(c.z, c.x));
            output[8] = vec4f(vec2f(blend), f32(max(a.z, b.x)), f32(min(a.w, b.x)));
            output[9] = vec4f(f32(sqrt(b.w)), f32(inverseSqrt(c.y)), f32(log2(b.z)), f32(exp2(b.y)));
            output[10] = vec4f(f32(sin(b.x)), f32(cos(b.x)), f32(atan2(c.z, c.w)), f32(asinh(b.y)));
            output[11] = vec4f(f32(acosh(b.y)), f32(atanh(b.x)), f32(degrees(atan2(c.z, c.z))), f32(radians(d.x)));
            output[12] = vec4f(f32(exp(b.x)), f32(log(b.y)), f32(pow(b.w, b.x)), f32(sign(c.w)));
            output[13] = vec4f(f32(tan(b.x)), f32(asin(b.x)), f32(acos(b.x)), f32(atan(c.z)));
            output[14] = vec4f(f32(sinh(b.x)), f32(cosh(b.x)), f32(tanh(b.x)), f32(abs(a.x)));
            output[15] = vec4f(f32(ceil(a.x)), f32(floor(a.x)), f32(trunc(a.y)), f32(min(a.w, b.x)));
        }
        """), to: .wgsl)
    let pipeline = try await device.createComputePipeline(shader)
    let input = try await device.createBuffer(.init(size: 32, usage: [.storage, .copyDestination]))
    let output = try await device.createBuffer(.init(size: 256, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 256, usage: [.mapRead, .copyDestination]))
    let samples: [Double] = [-2.5, -1.5, 1.5, 2.5, 0.5, 2, 8, 9, 0.25, 4, 1, -1, 180, 0, 1, 2]
    let words = stride(from: 0, to: samples.count, by: 2).map { i in
        UInt32(Binary16.encode(samples[i])) | (UInt32(Binary16.encode(samples[i + 1])) << 16)
    }
    let ticket = try await device.submitCommands([
        .write(buffer: input, offset: 0, words: words),
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): input, .init(binding: 1): output],
                         workgroups: .init(x: 1)),
        .copy(source: output, destination: read, byteCount: 256), .read(buffer: read, byteCount: 256),
    ])
    let snapshots = try await ticket.wait()
    guard let snapshot = snapshots.first, snapshot.count == 64 else {
        throw GPUError.backendFailure("Unexpected half-precision numeric readback: \(snapshots)")
    }
    return snapshot
}

private func quantizeDiagnostic(_ device: Device) async throws -> [UInt32] {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl("""
        const_assert quantizeToF16(65504) == 65504.0f;
        const_assert quantizeToF16(0x1.002p0f) == 1.0f;
        const_assert quantizeToF16(0x1.006p0f) == 0x1.008p0f;
        const_assert quantizeToF16(0x1p-24f) == 0x1p-24f;
        const_assert bitcast<u32>(quantizeToF16(-0x1p-26f)) == 0x80000000u;
        const_assert bitcast<u32>(quantizeToF16(-0.0)) == 0x80000000u;
        const_assert quantizeToF16(vec2(0.5, -1.0)).y == -1.0f;
        const_assert quantizeToF16(vec4f(0.125, 0.25, 0.5, 1)).z == 0.5f;
        @group(0) @binding(0) var<storage, read> input: array<vec4f, 3>;
        @group(0) @binding(1) var<storage, read_write> output: array<vec4u, 3>;
        @compute @workgroup_size(1) fn run() {
            let b = input[1];
            let c = input[2];
            output[0] = bitcast<vec4u>(quantizeToF16(input[0]));
            output[1] = bitcast<vec4u>(vec4f(quantizeToF16(b.xy), quantizeToF16(b.z), quantizeToF16(b.w)));
            output[2] = bitcast<vec4u>(vec4f(quantizeToF16(c.xyz), quantizeToF16(c.w)));
        }
        """), to: .wgsl)
    guard !shader.requiresShaderF16 else { throw GPUError.backendFailure("Quantize must use core f32 types") }
    let pipeline = try await device.createComputePipeline(shader)
    let imported = try compiler.compileWebGPUCompute(.spirv(quantizeSPIRV()))
    guard !imported.requiresShaderF16 else { throw GPUError.backendFailure("SPIR-V quantize must use core f32 types") }
    let spirv = try await device.createComputePipeline(imported)
    let input = try await device.createBuffer(.init(size: 48, usage: [.storage, .copyDestination]))
    let output = try await device.createBuffer(.init(size: 48, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 48, usage: [.mapRead, .copyDestination]))
    let words: [UInt32] = [
        0x3f800800, 0x3f803800, 0x80000000, 0x477fe000,
        0x38800000, 0xb2800000, 0x33800000, 0xbf802000,
        0x402001a3, 0x3eaaaaab, 0xba83126f, 0xc77fe000,
    ]
    var commands: [BufferCommand] = [
        .write(buffer: input, offset: 0, words: words),
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): input, .init(binding: 1): output],
                         workgroups: .init(x: 1)),
        .copy(source: output, destination: read, byteCount: 48), .read(buffer: read, byteCount: 48),
    ]
    let cases: [[UInt32]] = [
        [0x3f801000, 0x3f803000, 0x3ffff000, 0x477fe000],
        [0x38800000, 0x387fffff, 0xb3800000, Float(65505).bitPattern],
        [Float(-65505).bitPattern, 0x7f800000, 0xff800000, 0x7fc01234],
    ]
    for words in cases {
        commands += [
            .write(buffer: input, offset: 0, words: words),
            .dispatchCompute(pipeline: spirv, bindings: [.init(binding: 0): input, .init(binding: 1): output],
                             workgroups: .init(x: 1)),
            .copy(source: output, destination: read, byteCount: 16), .read(buffer: read, byteCount: 16),
        ]
    }
    let ticket = try await device.submitCommands(commands)
    let snapshots = try await ticket.wait()
    let expected: [[UInt32]] = [
        [0x3f800000, 0x3f804000, 0x40000000, 0x477fe000],
        [0x38800000, 0, 0x80000000, 0x7f800000],
        [0xff800000, 0x7f800000, 0xff800000, 0x7fc01234],
    ]
    guard snapshots.first?.count == 12, Array(snapshots.dropFirst()) == expected else {
        throw GPUError.backendFailure("Unexpected quantizeToF16 readback: \(snapshots)")
    }
    return snapshots.flatMap { $0 }
}

private func quantizeSPIRV() -> [UInt32] {
    func instruction(_ opcode: UInt32, _ operands: UInt32...) -> [UInt32] {
        [UInt32(operands.count + 1) << 16 | opcode] + operands
    }
    var words: [UInt32] = [0x07230203, 0x00010300, 0, 40, 0]
    for item in [
        instruction(17, 1), instruction(14, 0, 1), instruction(15, 5, 20, 0x6e69616d, 0), instruction(16, 20, 17, 1, 1, 1),
        instruction(71, 6, 2), instruction(72, 6, 0, 35, 0), instruction(71, 12, 34, 0), instruction(71, 12, 33, 0),
        instruction(71, 13, 34, 0), instruction(71, 13, 33, 1), instruction(19, 1), instruction(21, 2, 32, 0),
        instruction(22, 3, 32), instruction(23, 4, 2, 4), instruction(23, 5, 3, 4), instruction(30, 6, 4),
        instruction(32, 7, 12, 6), instruction(32, 8, 12, 4), instruction(33, 10, 1), instruction(43, 2, 14, 0),
        instruction(59, 7, 12, 12), instruction(59, 7, 13, 12), instruction(54, 1, 20, 0, 10), instruction(248, 21),
        instruction(65, 8, 30, 12, 14), instruction(61, 4, 31, 30), instruction(124, 5, 32, 31), instruction(116, 5, 33, 32),
        instruction(124, 4, 34, 33), instruction(65, 8, 35, 13, 14), instruction(62, 35, 34), instruction(253), instruction(56),
    ] { words += item }
    return words
}

private func floatingVectorDiagnostic(_ device: Device) async throws -> [UInt32] {
    guard device.capabilities.features.contains(.shaderF16) else { return [] }
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl("""
        enable f16;
        const_assert (vec2(1.5, 2.5) + 0.5).x == 2.0;
        const_assert (vec2h(-5.5, 7.5) % 2h).x == -1.5h;
        @group(0) @binding(0) var<storage, read> input: array<vec4f, 2>;
        @group(0) @binding(1) var<storage, read_write> output: array<vec4f, 8>;
        @compute @workgroup_size(1) fn run() {
            let a = input[0];
            let b = input[1];
            var adjusted = -a;
            adjusted += 2.0;
            adjusted *= 0.5;
            output[0] = quantizeToF16(adjusted);
            output[1] = a / b;
            output[2] = a % b;
            let mask = a < b;
            output[3] = vec4f(a.x / b.x, a.y % b.y, select(0f, 1f, mask.x), select(0f, 1f, mask.y));
            let ah = vec4h(a);
            let bh = vec4h(b);
            output[4] = vec4f((ah + 2.0) * bh);
            output[5] = vec4f((ah % 2h) / (1h + bh));
            output[6] = vec4f(select(vec4h(0), vec4h(1), ah < bh));
            output[7] = 2.0 - a;
        }
        """), to: .wgsl)
    let pipeline = try await device.createComputePipeline(shader)
    let input = try await device.createBuffer(.init(size: 32, usage: [.storage, .copyDestination]))
    let output = try await device.createBuffer(.init(size: 128, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 128, usage: [.mapRead, .copyDestination]))
    let values: [Float] = [-5.5, 7.5, -4, 8, 2, 2, 2, 4]
    let ticket = try await device.submitCommands([
        .write(buffer: input, offset: 0, words: values.map(\.bitPattern)),
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): input, .init(binding: 1): output],
                         workgroups: .init(x: 1)),
        .copy(source: output, destination: read, byteCount: 128), .read(buffer: read, byteCount: 128),
    ])
    let snapshots = try await ticket.wait()
    guard let snapshot = snapshots.first, snapshot.count == 32 else {
        throw GPUError.backendFailure("Unexpected floating vector arithmetic readback: \(snapshots)")
    }
    return snapshot
}

private func overrideDiagnostic(_ device: Device) async throws -> [UInt32] {
    let source = ShaderInput.wgsl("""
        @id(7) override width = 2u;
        override offset = base;
        override base = 3u;
        override addLane: bool = true;
        override unused: u32;
        @group(0) @binding(0) var<storage, read_write> output: array<u32, 4>;
        fn ignored() -> u32 { return unused; }
        fn bias() -> u32 { return offset; }
        @compute @workgroup_size(width)
        fn run(@builtin(local_invocation_index) i: u32) {
            let initial = bias();
            output[i] = select(initial, initial + i, addLane);
        }
        """)
    let compiler = IndependentShaderCompiler()
    let defaultShader = try compiler.compileCompute(source, to: .wgsl)
    let specializedShader = try compiler.compileCompute(source, to: .wgsl, constants: ["7": 4, "base": 8, "addLane": 0])
    guard defaultShader.workgroupSize.x == 2, specializedShader.workgroupSize.x == 4 else {
        throw GPUError.backendFailure("Unexpected specialized workgroup sizes")
    }
    let first = try await device.createComputePipeline(defaultShader)
    let second = try await device.createComputePipeline(specializedShader)
    let binary = ShaderInput.spirv(vulkanAtomicSpecializationSPIRV())
    let defaultBinary = try compiler.compileWebGPUCompute(binary)
    let specializedBinary = try compiler.compileWebGPUCompute(binary, constants: ["100": 25, "101": 0, "102": 1])
    let third = try await device.createComputePipeline(defaultBinary)
    let fourth = try await device.createComputePipeline(specializedBinary)
    let a = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
    let b = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 16, usage: [.copyDestination, .mapRead]))
    let ticket = try await device.submitCommands([
        .dispatchCompute(pipeline: first, bindings: [.init(binding: 0): a], workgroups: .init(x: 1)),
        .dispatchCompute(pipeline: second, bindings: [.init(binding: 0): b], workgroups: .init(x: 1)),
        .copy(source: a, destination: read, byteCount: 16), .read(buffer: read, byteCount: 16),
        .copy(source: b, destination: read, byteCount: 16), .read(buffer: read, byteCount: 16),
        .dispatchCompute(pipeline: third, bindings: [.init(binding: 0): a], workgroups: .init(x: 1)),
        .copy(source: a, destination: read, byteCount: 8), .read(buffer: read, byteCount: 8),
        .dispatchCompute(pipeline: fourth, bindings: [.init(binding: 0): a], workgroups: .init(x: 1)),
        .copy(source: a, destination: read, byteCount: 8), .read(buffer: read, byteCount: 8),
    ])
    let values = try await ticket.wait()
    guard values == [[3, 4, 0, 0], [8, 8, 8, 8], [7, 19], [3, 25]] else {
        throw GPUError.backendFailure("Unexpected override readback: \(values)")
    }
    return values.flatMap { $0 }
}

private func vulkanAtomicSpecializationSPIRV() -> [UInt32] {
    func op(_ code: UInt32, _ operands: UInt32...) -> [UInt32] {
        [UInt32(operands.count + 1) << 16 | code] + operands
    }
    var words: [UInt32] = [0x07230203, 0x00010500, 0, 64, 0]
    // Logical Vulkan, three scalar specialization constants, and QueueFamily relaxed exchanges.
    for instruction in [
        op(17, 1), op(17, 5345), op(14, 0, 3), op(15, 5, 40, 0x6e69616d, 0, 20), op(16, 40, 17, 1, 1, 1),
        op(71, 10, 1, 100), op(71, 11, 1, 101), op(71, 12, 1, 102), op(71, 5, 2),
        op(72, 5, 0, 35, 0), op(72, 5, 1, 35, 4), op(71, 20, 33, 0), op(71, 20, 34, 0),
        op(19, 1), op(21, 2, 32, 0), op(20, 3), op(30, 5, 2, 2),
        op(32, 6, 12, 5), op(32, 7, 12, 2), op(33, 8, 1),
        op(50, 2, 10, 7), op(48, 3, 11), op(49, 3, 12),
        op(43, 2, 13, 0), op(43, 2, 14, 1), op(43, 2, 15, 3), op(43, 2, 16, 19), op(43, 2, 17, 5),
        op(59, 6, 20, 12), op(54, 1, 40, 0, 8), op(248, 41),
        op(169, 2, 42, 11, 10, 15), op(169, 2, 43, 12, 10, 16),
        op(65, 7, 44, 20, 13), op(65, 7, 45, 20, 14),
        op(229, 2, 46, 44, 17, 13, 42), op(229, 2, 47, 45, 17, 13, 43), op(253), op(56),
    ] { words += instruction }
    return words
}

private func computedOverrideDiagnostic(_ device: Device) async throws -> [UInt32] {
    let source = ShaderInput.wgsl("""
        @id(21) override lanes: u32 = 2u;
        override divisor = 2u;
        const numerator = 8u;
        override bias = numerator / divisor + lanes;
        override enableOffsets = lanes > 1u && (10u / divisor > 2u);
        @group(0) @binding(0) var<storage, read_write> output: array<u32, 4>;
        @compute @workgroup_size(min(lanes + 1u, 4u), 1u, 1u)
        fn run(@builtin(local_invocation_index) i: u32) {
            output[i] = select(bias, bias + i, enableOffsets);
        }
        """)
    let compiler = IndependentShaderCompiler()
    let defaultShader = try compiler.compileCompute(source, to: .wgsl)
    let specializedShader = try compiler.compileCompute(source, to: .wgsl, constants: ["21": 1, "divisor": 4])
    guard defaultShader.workgroupSize.x == 3, specializedShader.workgroupSize.x == 2 else {
        throw GPUError.backendFailure("Unexpected specialized workgroup sizes")
    }
    let first = try await device.createComputePipeline(defaultShader)
    let second = try await device.createComputePipeline(specializedShader)
    let multiple = ShaderInput.wgsl("""
        override width = 2u;
        override start: u32;
        @group(0) @binding(0) var<storage, read_write> products: array<u32, 4>;
        @group(0) @binding(0) var<storage, read_write> sequence: array<u32, 3>;
        @group(2) @binding(2) var<uniform> color: vec4f;

        fn product(index: u32) -> u32 { return (index + 1u) * width; }

        @compute @workgroup_size(width)
        fn scale(@builtin(local_invocation_index) lane: u32) {
            products[lane] = product(lane);
        }

        @compute @workgroup_size(3)
        fn fill(@builtin(local_invocation_index) lane: u32) {
            sequence[lane] = start + lane;
        }

        @fragment fn shade() -> @location(0) vec4f { return color; }
        """)
    let scale = try compiler.compileWebGPUCompute(multiple, entryPoint: "scale", constants: ["width": 4])
    let fill = try compiler.compileWebGPUCompute(multiple, entryPoint: "fill", constants: ["start": 70])
    let scalePipeline = try await device.createComputePipeline(scale)
    let fillPipeline = try await device.createComputePipeline(fill)
    let a = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
    let b = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 16, usage: [.copyDestination, .mapRead]))
    let ticket = try await device.submitCommands([
        .dispatchCompute(pipeline: first, bindings: [.init(binding: 0): a], workgroups: .init(x: 1)),
        .dispatchCompute(pipeline: second, bindings: [.init(binding: 0): b], workgroups: .init(x: 1)),
        .copy(source: a, destination: read, byteCount: 16), .read(buffer: read, byteCount: 16),
        .copy(source: b, destination: read, byteCount: 16), .read(buffer: read, byteCount: 16),
        .dispatchCompute(pipeline: scalePipeline, bindings: [.init(binding: 0): a], workgroups: .init(x: 1)),
        .dispatchCompute(pipeline: fillPipeline, bindings: [.init(binding: 0): b], workgroups: .init(x: 1)),
        .copy(source: a, destination: read, byteCount: 16), .read(buffer: read, byteCount: 16),
        .copy(source: b, destination: read, byteCount: 12), .read(buffer: read, byteCount: 12),
    ])
    let values = try await ticket.wait()
    guard values.map(\.count) == [4, 4, 4, 3] else {
        throw GPUError.backendFailure("Unexpected override readback: \(values)")
    }
    return values.flatMap { $0 }
}

private func overrideArrayDiagnostic(_ device: Device) async throws -> [UInt32] {
    let source = ShaderInput.wgsl("""
        @id(4) override lanes = 2u;
        override count = lanes * 2u + 1u;
        override unused: u32;
        alias Scratch = array<u32, count>;
        var<workgroup> scratch: Scratch;
        var<workgroup> extra: array<u32, lanes + 1u>;
        var<workgroup> uncalled: array<u32, unused>;
        @group(0) @binding(0) var<storage, read_write> output: array<u32, 4>;
        fn neverCalled() { uncalled[7u] = 100u; }
        fn readNext(i: u32) -> u32 { return scratch[(i + 1u) % lanes]; }
        @compute @workgroup_size(lanes)
        fn run(@builtin(local_invocation_index) i: u32) {
            let initial = scratch[lanes + i] + extra[i];
            workgroupBarrier();
            scratch[i] = 10u + i;
            extra[i] = 20u + i;
            workgroupBarrier();
            output[i] = initial * 1000u + readNext(i) + extra[i];
        }
        """)
    let compiler = IndependentShaderCompiler()
    let defaultShader = try compiler.compileCompute(source, to: .wgsl)
    let specializedShader = try compiler.compileCompute(source, to: .wgsl, constants: ["4": 4])
    guard defaultShader.workgroupStorageByteSize == 48, specializedShader.workgroupStorageByteSize == 80 else {
        throw GPUError.backendFailure("Unexpected specialized workgroup array footprints")
    }
    let first = try await device.createComputePipeline(defaultShader)
    let second = try await device.createComputePipeline(specializedShader)
    let a = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
    let b = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 16, usage: [.copyDestination, .mapRead]))
    let ticket = try await device.submitCommands([
        .dispatchCompute(pipeline: first, bindings: [.init(binding: 0): a], workgroups: .init(x: 1)),
        .dispatchCompute(pipeline: second, bindings: [.init(binding: 0): b], workgroups: .init(x: 1)),
        .copy(source: a, destination: read, byteCount: 16), .read(buffer: read, byteCount: 16),
        .copy(source: b, destination: read, byteCount: 16), .read(buffer: read, byteCount: 16),
    ])
    let values = try await ticket.wait()
    guard values.map(\.count) == [4, 4] else {
        throw GPUError.backendFailure("Unexpected override readback: \(values)")
    }
    return values.flatMap { $0 }
}

private func pipelineBodyDiagnostic(_ device: Device) async throws -> [UInt32] {
    let source = ShaderInput.wgsl("""
        override divisor = 2u;
        override lane = 0u;
        @group(0) @binding(0) var<storage, read> input: array<u32, 4>;
        @group(0) @binding(1) var<storage, read_write> output: array<u32, 4>;
        fn unused() -> u32 { return 4u / (divisor - divisor); }
        @compute @workgroup_size(1)
        fn run(@builtin(global_invocation_id) id: vec3u) {
            output[0] = input[lane + 1u] / divisor;
            output[1] = (12u / divisor) + id.x;
            let runtimeIndex = lane + 1u;
            output[2] = input[runtimeIndex];
        }
        """)
    let shader = try IndependentShaderCompiler().compileCompute(source, to: .wgsl,
                                                                 constants: ["divisor": 4, "lane": 1])
    let pipeline = try await device.createComputePipeline(shader)
    let input = try await device.createBuffer(.init(size: 16, usage: [.storage, .copyDestination]))
    let output = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 16, usage: [.copyDestination, .mapRead]))
    let ticket = try await device.submitCommands([
        .write(buffer: input, offset: 0, words: [6, 8, 10, 12]),
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): input, .init(binding: 1): output],
                         workgroups: .init(x: 1)),
        .copy(source: output, destination: read, byteCount: 16), .read(buffer: read, byteCount: 16),
    ])
    let values = try await ticket.wait()
    guard let result = values.first, result.count == 4 else {
        throw GPUError.backendFailure("Unexpected pipeline body readback: \(values)")
    }
    return result
}

private func pipelineAggregateDiagnostic(_ device: Device) async throws -> [UInt32] {
    let source = ShaderInput.wgsl("""
        override scale = 2.0f;
        override lane = 0u;
        override divisor = 2u;
        override width = u32(dot(vec2f(scale, 1.0f), vec2f(1.0f)));
        @group(0) @binding(0) var<storage, read_write> output: array<vec4f, 2>;
        @compute @workgroup_size(width)
        fn run(@builtin(local_invocation_index) i: u32) {
            if i != 0u { return; }
            let lanes = quantizeToF16(vec2f(scale, scale + 1.0f));
            let matrix = transpose(mat2x2f(scale, 1, 2, 3));
            let pair = matrix * vec2f(2, 1);
            let picked = vec4u(11, 22, 33, 44).wzyx[lane + 1u];
            let quotient = vec2u(8, 9) / vec2u(divisor, 3);
            output[0] = vec4f(lanes, pair);
            output[1] = vec4f(f32(picked), f32(quotient.x), f32(quotient.y), f32(width));
        }
        """)
    let shader = try IndependentShaderCompiler().compileCompute(source, to: .wgsl,
                                                                 constants: ["scale": 3, "lane": 1, "divisor": 4])
    let pipeline = try await device.createComputePipeline(shader)
    let output = try await device.createBuffer(.init(size: 32, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 32, usage: [.copyDestination, .mapRead]))
    let ticket = try await device.submitCommands([
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
        .copy(source: output, destination: read, byteCount: 32), .read(buffer: read, byteCount: 32),
    ])
    let values = try await ticket.wait()
    guard let result = values.first, result.count == 8 else {
        throw GPUError.backendFailure("Unexpected pipeline aggregate readback: \(values)")
    }
    return result
}

private func pipelineCompositeDiagnostic(_ device: Device) async throws -> [UInt32] {
    let source = ShaderInput.wgsl("""
        struct Pair { count: u32, values: array<u32, 2>, }
        override first = 3u;
        override lane = 0i;
        override width = Pair(2u, array<u32, 2>(first, first + 1u)).values[lane];
        @group(0) @binding(0) var<storage, read_write> output: array<u32, 4>;
        @compute @workgroup_size(width)
        fn run(@builtin(local_invocation_index) id: u32) {
            if id >= 2u { return; }
            let pair = Pair(first, array<u32, 2>(width, first + 10u));
            output[id] = pair.values[id] + pair.count;
            output[id + 2u] = array<Pair, 2>(Pair(), Pair(first, array<u32, 2>()))[lane].count;
        }
        """)
    let shader = try IndependentShaderCompiler().compileCompute(source, to: .wgsl,
                                                                 constants: ["first": 4, "lane": 1])
    let pipeline = try await device.createComputePipeline(shader)
    let output = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 16, usage: [.copyDestination, .mapRead]))
    let ticket = try await device.submitCommands([
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
        .copy(source: output, destination: read, byteCount: 16), .read(buffer: read, byteCount: 16),
    ])
    let values = try await ticket.wait()
    guard let result = values.first, result.count == 4 else {
        throw GPUError.backendFailure("Unexpected pipeline composite readback: \(values)")
    }
    return result
}

private func arrayConstantDiagnostic(_ device: Device) async throws -> [UInt32] {
    let source = ShaderInput.wgsl("""
        alias Row = array<u32, 2>;
        struct Record { row: Row, tag: u32, }
        const empty = Record();
        const table = array(Record(Row(2u, 3u), 8u), Record(array(4u, 5), 10u));
        const_assert table[1].row[0] == 4u && empty.row[1] == 0u;
        override lane = 0u;
        override width = table[lane].row[1];
        @group(0) @binding(0) var<storage, read_write> output: array<u32, 2>;
        @compute @workgroup_size(width)
        fn run(@builtin(local_invocation_index) id: u32) {
            if id >= 2u { return; }
            const local: Row = array(6, 7);
            const_assert local[1] == 7u;
            let values = array(table[lane].tag, local[1] + id);
            output[id] = values[id];
        }
        """)
    let shader = try IndependentShaderCompiler().compileCompute(source, to: .wgsl, constants: ["lane": 1])
    let pipeline = try await device.createComputePipeline(shader)
    let zeroShader = try IndependentShaderCompiler().compileWebGPUCompute(.wgsl("""
        var<private> globalZero: vec2u = vec2();
        @group(0) @binding(0) var<storage, read_write> output: array<u32, 4>;

        fn sum(value: vec2u) -> u32 { return value.x + value.y + 3u; }
        fn zeroMatrix() -> mat2x2f { return mat2x2(vec2(), vec2()); }

        @compute @workgroup_size(1) fn run() {
            let inferred = vec2();
            let signed: vec2i = vec2();
            let unsigned: vec2u = vec2();
            let floating: vec2f = vec2();
            let nested: array<vec2f, 2> = array(vec2(), vec2(0, 3));
            var replaced = vec2u(17u, 19u);
            replaced = vec2();
            let matrix: mat2x2f = mat2x2();
            let returned = zeroMatrix();
            let spread: vec3u = vec3(vec2(), 7);

            output[0] = sum(vec2()) + globalZero.x + unsigned.x + replaced.x;
            output[1] = u32(nested[0].x + nested[1].y + floating.y);
            output[2] = u32(matrix[1].y + returned[0].x) + u32(inferred.x + signed.y);
            output[3] = spread.x + spread.y + spread.z;
        }
        """))
    let zeroPipeline = try await device.createComputePipeline(zeroShader)
    let indexShader = try IndependentShaderCompiler().compileWebGPUCompute(.wgsl("""
        const lanes = vec3(9, -4, 7);
        const pairs = array(vec2(21, 22), vec2(33, 34));
        const transform = mat3x2(1, 2, 3, 4, 5, 6);
        const large = vec2(2147483648, 7);
        const last = 2u;
        @id(4) override selected = 1u;
        @group(0) @binding(0) var<storage, read_write> output: array<vec4u, 5>;

        @compute @workgroup_size(5)
        fn run(@builtin(local_invocation_index) lane: u32) {
            let i = i32(lane) - 1i;
            let value: i32 = lanes[i];
            let pair: vec2i = pairs[i];
            let column: vec2f = transform[i];
            let named: u32 = lanes[last];
            let folded: u32 = large[last - 1u];
            output[lane] = vec4u(bitcast<u32>(value), u32(pair.y), u32(column.y),
                                named + folded + u32(lanes[selected]));
        }
        """), constants: ["4": 0])
    let indexPipeline = try await device.createComputePipeline(indexShader)
    let templateShader = try IndependentShaderCompiler().compileWebGPUCompute(.wgsl("""
        const A = 1u;
        const B = 2u;
        const C = 3u;
        const D = 4u;
        const_assert A<B || C>D;
        const counted: array<u32, (8u >> 2u)>=array(4u, 9u);
        const nested: array<u32, array<u32, 2>(2u, 3u)[0]>=array(5u, 11u);
        @group(0) @binding(0) var<storage, read_write> output: array<u32, 4>;

        @compute @workgroup_size(array<u32, 1>(1u)[0])
        fn run() {
            var sized: array<u32, 1u << 1u>=array(13u, 17u);
            let pointer: ptr<function, array<u32, 2>>=&sized;
            let pair: vec2<u32>=vec2u(19, 23);
            let bits = bitcast<vec2<u32>>(vec2f(1.0, 2.0));
            var shifted = 8u;
            shifted <<= 1u;
            shifted >>= 3u;
            var compared: array<u32, select(1u, 2u, B>=A)>=array(29u, 31u);

            output[0] = counted[1] + nested[0];
            output[1] = pointer[1] + pair.x;
            output[2] = select(0u, compared[1] + shifted, shifted>=B);
            output[3] = bits.x;
        }
        """))
    let templatePipeline = try await device.createComputePipeline(templateShader)
    let output = try await device.createBuffer(.init(size: 80, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 80, usage: [.copyDestination, .mapRead]))
    let ticket = try await device.submitCommands([
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
        .copy(source: output, destination: read, byteCount: 8), .read(buffer: read, byteCount: 8),
        .dispatchCompute(pipeline: zeroPipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
        .copy(source: output, destination: read, byteCount: 16), .read(buffer: read, byteCount: 16),
        .dispatchCompute(pipeline: indexPipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
        .copy(source: output, destination: read, byteCount: 80), .read(buffer: read, byteCount: 80),
        .dispatchCompute(pipeline: templatePipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
        .copy(source: output, destination: read, byteCount: 16), .read(buffer: read, byteCount: 16),
    ])
    let values = try await ticket.wait()
    guard values.map(\.count) == [2, 4, 20, 4] else {
        throw GPUError.backendFailure("Unexpected array constant readback: \(values)")
    }
    return values.flatMap { $0 }
}

private func pipelineBuiltinDiagnostic(_ device: Device) async throws -> [UInt32] {
    let source = ShaderInput.wgsl("""
        override amplitude = 3.5f;
        override active = true;
        override width = select(1u, u32(frexp(amplitude).exp + i32(modf(amplitude).whole)),
                                all(vec2<bool>(active, amplitude > 0.0f)));
        @group(0) @binding(0) var<storage, read_write> output: array<vec4f, 2>;
        @compute @workgroup_size(width)
        fn run(@builtin(local_invocation_index) id: u32) {
            if id != 0u { return; }
            let split = frexp(vec2f(amplitude, -amplitude));
            let parts = modf(vec2f(amplitude, -amplitude));
            let enabled = any(vec2<bool>(false, active));
            output[0] = vec4f(split.fract, parts.fract);
            output[1] = vec4f(f32(split.exp.x), parts.whole.y,
                              select(0.0f, 1.0f, all(all(vec2<bool>(id == 0u, enabled)))),
                              select(0.0f, 1.0f, any(vec2<bool>(id != 0u, false))));
        }
        """)
    let shader = try IndependentShaderCompiler().compileCompute(source, to: .wgsl, constants: ["amplitude": 6.75])
    let pipeline = try await device.createComputePipeline(shader)
    let output = try await device.createBuffer(.init(size: 32, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 32, usage: [.copyDestination, .mapRead]))
    let ticket = try await device.submitCommands([
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
        .copy(source: output, destination: read, byteCount: 32), .read(buffer: read, byteCount: 32),
    ])
    let values = try await ticket.wait()
    guard let result = values.first, result.count == 8 else {
        throw GPUError.backendFailure("Unexpected pipeline builtin readback: \(values)")
    }
    return result
}

private func conversionDiagnostic(_ device: Device) async throws -> [UInt32] {
    let source = ShaderInput.wgsl("""
        override gain = 0.0f;
        override width = 1u + u32(bool(gain));
        @group(0) @binding(0) var<storage, read> input: array<vec4f, 1>;
        @group(0) @binding(1) var<storage, read_write> output: array<vec4u, 4>;
        @compute @workgroup_size(width)
        fn run(@builtin(local_invocation_index) id: u32) {
            if id != 0u { return; }
            let samples = input[0];
            let mask = vec4<bool>(samples);
            output[0] = bitcast<vec4u>(vec4i(samples));
            output[1] = vec4u(samples);
            output[2] = vec4u(mask);
            output[3] = vec4u(u32(bool(samples.w)), u32(bool(samples.z)),
                              bitcast<u32>(f32(bool(samples.y))), bitcast<u32>(i32(bool(samples.x))));
        }
        """)
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(source, to: .wgsl, constants: ["gain": -3])
    let pipeline = try await device.createComputePipeline(shader)
    let spirv = try await device.createComputePipeline(compiler.compileWebGPUCompute(.spirv(numericConversionSPIRV())))
    func roundTripPipeline(half: Bool) async throws -> ComputePipeline {
        let source = ShaderInput.wgsl("""
            \(half ? "enable f16;" : "")
            @group(0) @binding(0) var<storage, read> input: array<\(half ? "vec4h" : "vec4f"), 1>;
            @group(0) @binding(1) var<storage, read_write> output: array<vec4u, 2>;
            @compute @workgroup_size(1) fn round_trip() {
                let v = input[0];
                output[0] = bitcast<vec4u>(vec4i(v));
                output[1] = vec4u(u32(v.x), u32(v.y), u32(v.z), u32(v.w));
            }
            """)
        guard case .spirv(let words) = try compiler.compile(source, to: .spirv) else {
            throw GPUError.backendFailure("Expected binary SPIR-V for numeric conversions")
        }
        let shader = try compiler.compileWebGPUCompute(.spirv(words), entryPoint: "round_trip")
        return try await device.createComputePipeline(shader)
    }
    let f32RoundTrip = try await roundTripPipeline(half: false)
    var roundTrips: [(pipeline: ComputePipeline, input: [UInt32], expected: [UInt32])] = [
        (f32RoundTrip, [0xff800000, 0x7f800000, 0xc0700000, 0x40700000],
         [0x80000000, 0x7fffff80, 0xfffffffd, 3, 0, 0xffffff00, 0, 3]),
        (f32RoundTrip, [0x4effffff, 0x4f000000, 0x4f7fffff, 0x4f800000],
         [0x7fffff80, 0x7fffff80, 0x7fffff80, 0x7fffff80, 0x7fffff80, 0x80000000, 0xffffff00, 0xffffff00]),
    ]
    if device.capabilities.features.contains(.shaderF16) {
        let f16RoundTrip = try await roundTripPipeline(half: true)
        roundTrips += [
            (f16RoundTrip, [0x7c00fc00, 0x4180c180], [0xffff0020, 0xffe0, 0xfffffffe, 2, 0, 0xffe0, 0, 2]),
            (f16RoundTrip, [0xfbff7bff, 0x80000000], [0xffe0, 0xffff0020, 0, 0, 0xffe0, 0, 0, 0]),
        ]
    }
    let input = try await device.createBuffer(.init(size: 16, usage: [.storage, .copyDestination]))
    let output = try await device.createBuffer(.init(size: 64, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 64, usage: [.copyDestination, .mapRead]))
    let samples: [Float] = [-1e20, 1e20, -3.75, -0.0]
    var commands: [BufferCommand] = [
        .write(buffer: input, offset: 0, words: samples.map(\.bitPattern)),
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): input, .init(binding: 1): output],
                         workgroups: .init(x: 1)),
        .copy(source: output, destination: read, byteCount: 64), .read(buffer: read, byteCount: 64),
        .dispatchCompute(pipeline: spirv, bindings: [.init(binding: 1): output], workgroups: .init(x: 1)),
        .copy(source: output, destination: read, byteCount: 8), .read(buffer: read, byteCount: 8),
    ]
    for roundTrip in roundTrips {
        commands += [
            .write(buffer: input, offset: 0, words: roundTrip.input),
            .dispatchCompute(pipeline: roundTrip.pipeline, bindings: [.init(binding: 0): input, .init(binding: 1): output],
                             workgroups: .init(x: 1)),
            .copy(source: output, destination: read, byteCount: 32), .read(buffer: read, byteCount: 32),
        ]
    }
    let values = try await device.submitCommands(commands).wait()
    guard values.count == 2 + roundTrips.count, values[0].count == 16, values[1] == [0xfffffffd, 9],
          values.dropFirst(2).elementsEqual(roundTrips.map { $0.expected }) else {
        throw GPUError.backendFailure("Unexpected conversion readback: \(values)")
    }
    return values.flatMap { $0 }
}

private func numericConversionSPIRV() -> [UInt32] {
    func instruction(_ opcode: UInt32, _ operands: UInt32...) -> [UInt32] {
        [UInt32(operands.count + 1) << 16 | opcode] + operands
    }
    var words: [UInt32] = [0x07230203, 0x00010300, 0, 40, 0]
    for item in [
        instruction(17, 1), instruction(14, 0, 1), instruction(15, 5, 20, 0x6e69616d, 0), instruction(16, 20, 17, 1, 1, 1),
        instruction(71, 4, 2), instruction(72, 4, 0, 35, 0), instruction(72, 4, 1, 35, 4),
        instruction(71, 11, 34, 0), instruction(71, 11, 33, 1), instruction(19, 1), instruction(21, 2, 32, 0),
        instruction(22, 3, 32), instruction(30, 4, 2, 2), instruction(32, 5, 12, 4), instruction(32, 6, 12, 2),
        instruction(33, 7, 1), instruction(43, 2, 8, 0), instruction(43, 2, 9, 1),
        instruction(43, 3, 12, Float(-3.75).bitPattern), instruction(43, 3, 13, Float(9.75).bitPattern), instruction(59, 5, 11, 12),
        instruction(54, 1, 20, 0, 7), instruction(248, 21), instruction(110, 2, 22, 12), instruction(109, 2, 23, 13),
        instruction(65, 6, 24, 11, 8), instruction(62, 24, 22), instruction(65, 6, 25, 11, 9), instruction(62, 25, 23),
        instruction(253), instruction(56),
    ] { words += item }
    return words
}

private func halfConversionDiagnostic(_ device: Device) async throws -> [UInt32] {
    guard device.capabilities.features.contains(.shaderF16) else { return [] }
    let source = ShaderInput.wgsl("""
        enable f16;
        override value = 4.5h;
        override width = u32(vec2i(vec2h(value)).x) + u32(vec2<bool>(vec2h(value)).y);
        @group(0) @binding(0) var<storage, read_write> output: array<vec4i, 1>;
        @compute @workgroup_size(width)
        fn run(@builtin(local_invocation_index) id: u32) {
            if id == 0u {
                let samples = vec2h(f16(id), -2.75h);
                let mask = vec2<bool>(id == 0u, id != 0u);
                output[0] = vec4i(vec2i(samples), vec2i(vec2h(mask)));
            }
        }
        """)
    let shader = try IndependentShaderCompiler().compileCompute(source, to: .wgsl, constants: ["value": 6.75])
    let pipeline = try await device.createComputePipeline(shader)
    let output = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 16, usage: [.copyDestination, .mapRead]))
    let ticket = try await device.submitCommands([
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
        .copy(source: output, destination: read, byteCount: 16), .read(buffer: read, byteCount: 16),
    ])
    let values = try await ticket.wait()
    guard let result = values.first, result.count == 4 else {
        throw GPUError.backendFailure("Unexpected f16 conversion readback: \(values)")
    }
    return result
}

private func localArrayDiagnostic(_ device: Device) async throws -> [UInt32] {
    let source = ShaderInput.wgsl("""
        struct Cell { data: array<vec2u, 2>, stamp: u32, }
        @group(0) @binding(0) var<storage, read_write> output: array<vec4u, 4>;

        fn update(outer: i32, inner: i32) -> vec4u {
            var cells: array<Cell, 2>;
            cells[0].data[0] = vec2u(3u, 5u);
            cells[1].data[1] = vec2u(7u, 11u);
            cells[0].stamp = 41u;
            cells[1].stamp = 101u;
            let before = cells;
            cells[outer].data[inner].y = 19u;
            cells[outer].stamp += 1u;
            var reset = 0u;
            for (var i = 0u; i < 3u; i += 1u) {
                var frame: array<u32, 2>;
                let slot = i & 1u;
                reset += frame[slot];
                frame[slot] = 123u;
            }
            return vec4u(before[outer].data[inner].y, cells[outer].data[inner].y,
                         cells[0].stamp + reset, cells[1].stamp);
        }

        @compute @workgroup_size(4)
        fn run(@builtin(local_invocation_index) id: u32) {
            let outer = i32(id) - 1i;
            let inner = select(1i, 2i, id == 1u);
            output[id] = update(outer, inner);
        }
        """)
    let shader = try IndependentShaderCompiler().compileCompute(source, to: .wgsl)
    let pipeline = try await device.createComputePipeline(shader)
    let output = try await device.createBuffer(.init(size: 64, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 64, usage: [.copyDestination, .mapRead]))
    let ticket = try await device.submitCommands([
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
        .copy(source: output, destination: read, byteCount: 64), .read(buffer: read, byteCount: 64),
    ])
    let values = try await ticket.wait()
    guard let result = values.first, result.count == 16 else {
        throw GPUError.backendFailure("Unexpected mutable local array readback: \(values)")
    }
    return result
}

private func workgroupAggregateDiagnostic(_ device: Device) async throws -> [UInt32] {
    let source = ShaderInput.wgsl("""
        struct Slot { bins: array<vec2u, 2>, stamp: u32, }
        struct Counters { hits: atomic<u32>, parity: array<atomic<u32>, 2>, }
        override count = 4u;
        var<workgroup> slots: array<Slot, count>;
        var<workgroup> counters: Counters;
        @group(0) @binding(0) var<storage, read_write> output: array<vec4u, 8>;

        fn fill(lane: u32, base: u32) {
            slots[lane].bins[1] = vec2u(base + lane + 1u, 2u);
            slots[lane].stamp = 31u + lane;
            atomicAdd(&counters.hits, 1u);
            atomicAdd(&counters.parity[lane % 2u], 1u);
        }

        @compute @workgroup_size(4)
        fn run(@builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) group: vec3u) {
            let initial = slots[lane].bins[1].x + slots[lane].stamp
                        + atomicLoad(&counters.hits) + atomicLoad(&counters.parity[lane % 2u]);
            workgroupBarrier();
            fill(lane, group.x * 10u);
            workgroupBarrier();
            let next = slots[(lane + 1u) % 4u];
            let shared = workgroupUniformLoad(&slots[group.x]);
            let missing = workgroupUniformLoad(&slots[count + group.x]);
            let uniformBin = workgroupUniformLoad(&slots[group.x].bins[group.x + 1u]);
            if shared.stamp != 0u { workgroupBarrier(); }
            let invalid = lane + 2u;
            slots[lane].bins[invalid].x = 999u;
            output[group.x * 4u + lane] = vec4u(next.bins[1].x + initial * 1000u, slots[lane].stamp,
                                              atomicLoad(&counters.hits),
                                              atomicLoad(&counters.parity[lane % 2u]) + slots[lane].bins[invalid].x
                                                  + shared.stamp + missing.stamp + uniformBin.y);
        }
        """)
    let shader = try IndependentShaderCompiler().compileCompute(source, to: .wgsl, constants: ["count": 5])
    let pipeline = try await device.createComputePipeline(shader)
    let output = try await device.createBuffer(.init(size: 128, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 128, usage: [.copyDestination, .mapRead]))
    let ticket = try await device.submitCommands([
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 2)),
        .copy(source: output, destination: read, byteCount: 128), .read(buffer: read, byteCount: 128),
    ])
    let values = try await ticket.wait()
    guard let result = values.first, result.count == 32 else {
        throw GPUError.backendFailure("Unexpected workgroup aggregate readback: \(values)")
    }
    return result
}

private func privateVariableDiagnostic(_ device: Device) async throws -> [UInt32] {
    let source = ShaderInput.wgsl("""
        struct State { samples: array<vec2u, 2>, counter: u32, }
        const seed = array(vec2u(3u, 7u), vec2u(11u, 13u));
        override base = 17u;
        var<private> state = State(seed, base);
        var<private> baseline: vec2u = vec2u(1u, 2u);
        var<private> ticks: u32;
        @group(0) @binding(0) var<storage, read_write> output: array<vec4u, 8>;

        fn bump(index: i32, id: u32) -> u32 {
            let old = ticks;
            ticks += 1u;
            state.samples[index].y = 41u + id;
            state.counter += id;
            return old;
        }
        fn forward(index: i32, id: u32) -> u32 { return bump(index, id); }

        @compute @workgroup_size(4)
        fn run(@builtin(global_invocation_id) gid: vec3u) {
            let id = gid.x;
            let index = i32(id) - 1i;
            let initial = state.counter + baseline.x;
            let first = forward(index, id);
            let second = forward(index, id);
            output[id] = vec4u(initial + first * 100u, state.counter + second * 100u,
                              state.samples[index].y, state.samples[1].x);
        }
        """)
    let shader = try IndependentShaderCompiler().compileCompute(source, to: .wgsl, constants: ["base": 29])
    let pipeline = try await device.createComputePipeline(shader)
    let output = try await device.createBuffer(.init(size: 128, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 128, usage: [.copyDestination, .mapRead]))
    let ticket = try await device.submitCommands([
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 2)),
        .copy(source: output, destination: read, byteCount: 128), .read(buffer: read, byteCount: 128),
    ])
    let values = try await ticket.wait()
    guard let result = values.first, result.count == 32 else {
        throw GPUError.backendFailure("Unexpected private variable readback: \(values)")
    }
    return result
}

private func privateRenderDiagnostic(_ device: Device) async throws -> [UInt32] {
    let source = ShaderInput.wgsl("""
        override red = 0.25f;
        var<private> state = vec4f(red, 0.0f, 0.0f, 1.0f);
        fn take() -> vec4f {
            state.x += 0.25f;
            return state;
        }
        @vertex fn vertexMain(@builtin(vertex_index) id: u32) -> @builtin(position) vec4f {
            let reach = 1.0f + 4.0f * take().x;
            if id == 0u { return vec4f(-1.0f, -1.0f, 0.0f, 1.0f); }
            if id == 1u { return vec4f(reach, -1.0f, 0.0f, 1.0f); }
            return vec4f(-1.0f, reach, 0.0f, 1.0f);
        }
        @fragment fn fragmentMain() -> @location(0) vec4f {
            let color = take();
            return vec4f(select(0.0f, 1.0f, color.x == 0.75f), color.yzw);
        }
        """)
    let compiler = IndependentShaderCompiler()
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vertexMain", to: .wgsl, constants: ["red": 0.5])
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fragmentMain", to: .wgsl, constants: ["red": 0.5])
    let pipeline = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm)
    let target = try await device.createTexture(.init(width: 2, height: 2, format: .rgba8Unorm,
                                                       usage: [.renderAttachment, .copySource]))
    let read = try await device.createBuffer(.init(size: 512, usage: [.copyDestination, .mapRead]))
    let ticket = try await device.submitCommands([
        .render(pipeline: pipeline, target: target, clear: .init(red: 0, green: 0, blue: 1, alpha: 1), vertexCount: 3),
        .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256),
        .read(buffer: read, byteCount: 512),
    ])
    let values = try await ticket.wait()
    guard let pixels = values.first, pixels.count == 128 else {
        throw GPUError.backendFailure("Unexpected private render readback: \(values)")
    }
    return [pixels[0], pixels[1], pixels[64], pixels[65]]
}

private func pointerDiagnostic(_ device: Device) async throws -> [UInt32] {
    let source = ShaderInput.wgsl("""
        struct Packet { values: array<u32, 2>, stamp: u32, }
        var<private> seed = 4u;
        @group(0) @binding(0) var<storage, read_write> output: array<vec4u>;

        fn mutate(packet: ptr<function, Packet>, increment: ptr<private, u32>, index: i32) -> u32 {
            let target = &(*packet).values[index];
            let old = *target;
            *target = old + *increment;
            *increment += 1u;
            (*packet).stamp += *increment;
            return old;
        }

        fn forward(packet: ptr<function, Packet>, increment: ptr<private, u32>, index: i32) -> u32 {
            let again: ptr<function, Packet> = &*packet;
            return mutate(again, increment, index);
        }

        fn readBoth(a: ptr<function, u32>, b: ptr<function, u32>) -> u32 { return *a + *b; }

        @compute @workgroup_size(4)
        fn run(@builtin(local_invocation_index) lane: u32) {
            var packet = Packet(array(10u, 20u), 100u);
            var index = i32(lane) - 1i;
            let chosen = &packet.values[index];
            index = 0i;
            let before = *chosen;
            *chosen = before + 1u;
            let previous = forward(&packet, &seed, i32(lane) - 1i);
            var tag = packet.stamp;
            let outputAddress = &output;
            output[lane] = vec4u(before, previous, packet.values[0] + packet.values[1],
                                 readBoth(&tag, &tag) + seed + arrayLength(outputAddress));
        }
        """)
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(source, to: .wgsl)
    let pipeline = try await device.createComputePipeline(shader)
    let roundTripSource = ShaderInput.wgsl("""
        requires unrestricted_pointer_parameters;
        struct Packet { value: u32, stamp: u32 }
        @group(0) @binding(0) var<storage, read_write> output: array<vec4u, 4>;

        fn mutate(packet: ptr<function, Packet>, delta: u32) -> u32 {
            let old = (*packet).value;
            (*packet).value = old + delta;
            (*packet).stamp += (*packet).value;
            return old;
        }
        fn forward(packet: ptr<function, Packet>, delta: u32) -> u32 {
            return mutate(&*packet, delta);
        }
        fn publish(destination: ptr<storage, vec4u, read_write>, value: vec4u) {
            *destination = value;
        }
        fn relay(destination: ptr<storage, vec4u, read_write>, value: vec4u) {
            publish(&*destination, value);
        }
        @compute @workgroup_size(4)
        fn round_trip(@builtin(local_invocation_index) lane: u32) {
            var packet = Packet(40u + lane, 100u);
            let previous = forward(&packet, 7u + lane);
            relay(&output[lane], vec4u(previous, packet.value, packet.stamp, 300u + lane));
        }
        """)
    guard case .spirv(let binary) = try compiler.compile(roundTripSource, to: .spirv) else {
        throw GPUError.backendFailure("Expected binary SPIR-V for pointer helpers")
    }
    let roundTripShader = try compiler.compileWebGPUCompute(.spirv(binary), entryPoint: "round_trip")
    let roundTrip = try await device.createComputePipeline(roundTripShader)
    let output = try await device.createBuffer(.init(size: 64, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 64, usage: [.copyDestination, .mapRead]))
    let ticket = try await device.submitCommands([
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
        .copy(source: output, destination: read, byteCount: 64), .read(buffer: read, byteCount: 64),
        .dispatchCompute(pipeline: roundTrip, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
        .copy(source: output, destination: read, byteCount: 64), .read(buffer: read, byteCount: 64),
    ])
    let values = try await ticket.wait()
    let expected: [UInt32] = [40, 47, 147, 300, 41, 49, 149, 301, 42, 51, 151, 302, 43, 53, 153, 303]
    guard values.map(\.count) == [16, 16], values[1] == expected else {
        throw GPUError.backendFailure("Unexpected pointer readback: \(values)")
    }
    return values.flatMap { $0 }
}

private func pointerSubviewDiagnostic(_ device: Device) async throws -> [UInt32] {
    let source = ShaderInput.wgsl("""
        struct Packet { samples: array<u32, 2>, sentinel: u32, }
        override base = 30u;
        var<private> state = Packet(array(base, 40u), 222u);
        var<private> calls: u32;
        @group(0) @binding(0) var<storage, read_write> output: array<vec4u, 4>;

        fn modify(p: ptr<function, u32>, q: ptr<private, u32>) -> vec2u {
            calls += 1u;
            let previous = *p + *q;
            *p += 3u;
            *q += 7u;
            return vec2u(previous, *p + *q);
        }

        fn forward(p: ptr<function, u32>, q: ptr<private, u32>) -> vec2u {
            return modify(&*p, &*q);
        }

        fn touch(unused: ptr<private, u32>, lane: u32) {
            state.samples[lane] += 1u;
        }

        @compute @workgroup_size(4)
        fn run(@builtin(local_invocation_index) lane: u32) {
            var packet = Packet(array(10u, 20u), 111u);
            var index = i32(lane) - 1i;
            let packetRef = &packet;
            let samples = &packetRef.samples;
            let p = &samples[index];
            let stateRef = &state;
            let q = &stateRef.samples[lane];
            index = 0i;
            let result = forward(p, q);
            touch(q, lane);
            output[lane] = vec4u(result, packet.samples[0] + packet.samples[1] + packet.sentinel + 1000u * calls,
                                state.samples[0] + state.samples[1] + state.sentinel);
        }
        """)
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileWebGPUCompute(source, constants: ["base": 50])
    let pipeline = try await device.createComputePipeline(shader)
    let roundTripSource = ShaderInput.wgsl("""
        requires unrestricted_pointer_parameters;
        struct Packet { values: array<u32, 2>, marker: u32 }
        var<private> state = Packet(array(30u, 40u), 200u);
        @group(0) @binding(0) var<storage, read_write> output: array<vec4u, 4>;

        fn update(p: ptr<function, u32>, q: ptr<private, u32>, delta: u32) -> vec2u {
            let old = vec2u(*p, *q);
            *p = old.x + delta;
            *q = old.y + 2u * delta;
            return old;
        }
        fn relay(p: ptr<function, u32>, q: ptr<private, u32>, delta: u32) -> vec2u {
            return update(&*p, &*q, delta);
        }
        @compute @workgroup_size(4)
        fn round_trip(@builtin(local_invocation_index) lane: u32) {
            var packet = Packet(array(10u, 20u), 100u);
            var index = lane & 1u;
            let p = &packet.values[index];
            let q = &state.values[index ^ 1u];
            index = index ^ 1u;
            let old = relay(p, q, lane + 3u);
            let local = packet.values[0] * 100u + packet.values[1] + packet.marker;
            let privateValue = state.values[0] * 100u + state.values[1] + state.marker;
            output[lane] = vec4u(old, local, privateValue);
        }
        """)
    guard case .spirv(let binary) = try compiler.compile(roundTripSource, to: .spirv) else {
        throw GPUError.backendFailure("Expected binary SPIR-V for derived pointer helpers")
    }
    let roundTripShader = try compiler.compileWebGPUCompute(.spirv(binary), entryPoint: "round_trip")
    let roundTrip = try await device.createComputePipeline(roundTripShader)
    let output = try await device.createBuffer(.init(size: 64, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 64, usage: [.copyDestination, .mapRead]))
    let ticket = try await device.submitCommands([
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
        .copy(source: output, destination: read, byteCount: 64), .read(buffer: read, byteCount: 64),
        .dispatchCompute(pipeline: roundTrip, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
        .copy(source: output, destination: read, byteCount: 64), .read(buffer: read, byteCount: 64),
    ])
    let values = try await ticket.wait()
    let expected: [UInt32] = [10, 40, 1420, 3246, 20, 30, 1124, 4040, 10, 40, 1620, 3250, 20, 30, 1126, 4440]
    guard values.map(\.count) == [16, 16], values[1] == expected else {
        throw GPUError.backendFailure("Unexpected pointer subview readback: \(values)")
    }
    return values.flatMap { $0 }
}

private func sharedPointerDiagnostic(_ device: Device) async throws -> [UInt32] {
    let source = ShaderInput.wgsl("""
        struct Config { values: vec4u, }
        @group(0) @binding(0) var<uniform> config: Config;
        @group(0) @binding(1) var<storage, read> input: array<u32>;
        @group(0) @binding(2) var<storage, read_write> output: array<vec4u, 8>;
        var<workgroup> totals: array<atomic<u32>, 2>;
        var<workgroup> counts: array<atomic<u32>, 2>;

        fn sample(values: ptr<uniform, vec4u>, source: ptr<storage, array<u32>, read>, lane: u32) -> u32 {
            return (*values)[lane] + (*source)[lane] + arrayLength(source);
        }

        fn accumulate(cell: ptr<workgroup, atomic<u32>>, value: u32) -> u32 {
            atomicAdd(cell, value);
            workgroupBarrier();
            return atomicLoad(cell);
        }

        fn relay(cell: ptr<workgroup, atomic<u32>>, value: u32) -> u32 {
            return accumulate(cell, value);
        }

        fn sumAt(cells: ptr<workgroup, array<atomic<u32>, 2>>, index: u32, value: u32) -> u32 {
            return relay(&(*cells)[index], value);
        }

        fn snapshot(cell: ptr<workgroup, atomic<u32>>) -> u32 {
            return workgroupUniformLoad(cell);
        }

        fn publish(destination: ptr<storage, vec4u, read_write>, value: vec4u) {
            *destination = value;
        }

        @compute @workgroup_size(4)
        fn run(@builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) group: vec3u) {
            let value = sample(&config.values, &input, lane);
            let total = sumAt(&totals, group.x, value);
            let count = sumAt(&counts, 1u - group.x, lane + 1u);
            let invalid = sumAt(&totals, group.x + 2u, 999u);
            let synchronized = snapshot(&totals[group.x]);
            let index = group.x * 4u + lane;
            publish(&output[index], vec4u(value, total, count, lane + invalid + synchronized));
            publish(&output[index + 8u], vec4u(999u));
        }
        """)
    let shader = try IndependentShaderCompiler().compileCompute(source, to: .wgsl)
    let pipeline = try await device.createComputePipeline(shader)
    let uniform = try await device.createBuffer(.init(size: 16, usage: [.uniform, .copyDestination]))
    let input = try await device.createBuffer(.init(size: 16, usage: [.storage, .copyDestination]))
    let output = try await device.createBuffer(.init(size: 128, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 128, usage: [.copyDestination, .mapRead]))
    let ticket = try await device.submitCommands([
        .write(buffer: uniform, offset: 0, words: [10, 20, 30, 40]),
        .write(buffer: input, offset: 0, words: [1, 2, 3, 4]),
        .dispatchCompute(pipeline: pipeline,
                         bindings: [.init(binding: 0): uniform, .init(binding: 1): input, .init(binding: 2): output],
                         workgroups: .init(x: 2)),
        .copy(source: output, destination: read, byteCount: 128), .read(buffer: read, byteCount: 128),
    ])
    let values = try await ticket.wait()
    guard let result = values.first, result.count == 32 else {
        throw GPUError.backendFailure("Unexpected shared pointer readback: \(values)")
    }
    return result
}

private func swizzleAssignmentDiagnostic(_ device: Device) async throws -> [UInt32] {
    let source = ShaderInput.wgsl("""
        requires swizzle_assignment, pointer_composite_access;

        var<private> values: array<vec4u, 1>;
        var<private> events: u32;
        @group(0) @binding(0) var<storage, read_write> output: array<vec4u, 6>;

        fn updateIndex() -> u32 {
            events = events * 10u + 1u;
            values[0] = values[0] + vec4u(2u, 4u, 8u, 10u);
            return 0u;
        }

        fn updateRHS() -> vec2u {
            events = events * 10u + 2u;
            values[0].x = 900u;
            values[0].z = 100u;
            return vec2u(55u, 66u);
        }

        @compute @workgroup_size(1) fn run() {
            values[updateIndex()].yx = updateRHS();
            output[0] = values[0];
            let simpleEvents = events;
            values[0] = vec4u();
            events = 0u;
            values[updateIndex()].yx += updateRHS();
            output[1] = values[0];
            output[2] = vec4u(simpleEvents, events, 0u, 0u);

            var local = vec4u(1u, 2u, 3u, 4u);
            let p = &local;
            p.wzyx.xy = p.xy;
            p.yxz[1u] = 9u;
            p.yxz.y += 2u;
            var invalid = -1i;
            p.yxz[invalid] = 999u;
            p.zy <<= vec2u(1u, 2u);
            p.yx = p.xy;
            output[3] = local;
            let destination = &output;
            destination[3u].yx = destination[3u].xy;

            var f = vec4f(1.0f, 2.0f, 3.0f, 4.0f);
            f.wy *= vec2f(2.0f, 3.0f);
            output[4] = bitcast<vec4u>(f);
            var b = vec4<bool>(true, true, false, true);
            b.yx &= vec2<bool>(true, false);
            b.zx |= vec2<bool>(true, false);
            b.wy.y &= false;
            output[5] = vec4u(b);
        }
        """)
    let shader = try IndependentShaderCompiler().compileCompute(source, to: .wgsl)
    let pipeline = try await device.createComputePipeline(shader)
    let output = try await device.createBuffer(.init(size: 96, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 96, usage: [.copyDestination, .mapRead]))
    let ticket = try await device.submitCommands([
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
        .copy(source: output, destination: read, byteCount: 96), .read(buffer: read, byteCount: 96),
    ])
    let values = try await ticket.wait()
    guard let result = values.first, result.count == 24 else {
        throw GPUError.backendFailure("Unexpected swizzle assignment readback: \(values)")
    }
    return result
}

private func nestedStorageDiagnostic(_ device: Device) async throws -> [UInt32] {
    let source = ShaderInput.wgsl("""
        struct Row { values: array<vec3u, 2>, @size(16) marker: u32, }
        struct Data { @size(16) generation: u32, rows: array<Row>, }
        @group(0) @binding(0) var<storage, read_write> data: Data;
        @group(0) @binding(1) var<storage, read_write> output: array<vec4u, 4>;

        @compute @workgroup_size(4) fn run(@builtin(local_invocation_index) lane: u32) {
            let outer = i32(lane) - 1i;
            let inner = select(1u, 2u, lane == 2u);
            let before = data.rows[outer].values[inner];
            data.rows[outer].values[inner].z = 70u + lane;
            var row = data.rows[outer];
            row.marker += lane;
            data.rows[outer] = row;
            output[lane] = vec4u(before, arrayLength(&data.rows) + data.rows[outer].marker);
        }
        """)
    let shader = try IndependentShaderCompiler().compileCompute(source, to: .wgsl)
    let pipeline = try await device.createComputePipeline(shader)
    let aggregateShader = try IndependentShaderCompiler().compileWebGPUCompute(.wgsl("""
        struct Record {
            @size(16) tag: u32,
            points: array<vec3u, 2>,
            @size(16) marker: u32,
        }
        struct Config { bias: u32, matrix: mat2x2f, }
        @group(0) @binding(0) var<storage, read> source: Record;
        @group(0) @binding(1) var<storage, read_write> destination: Record;
        @group(0) @binding(2) var<storage, read_write> words: array<u32, 4>;
        @group(0) @binding(3) var<uniform> config: Config;

        fn readRecord() -> Record { return source; }

        @compute @workgroup_size(1) fn run() {
            var snapshot = readRecord();
            let settings = config;
            let previous = words;
            words = array<u32, 4>(previous[3], snapshot.tag, u32(settings.matrix[1].y), settings.bias);
            snapshot.tag += 4u;
            snapshot.points[0].y += settings.bias;
            destination = snapshot;
        }
        """))
    let aggregatePipeline = try await device.createComputePipeline(aggregateShader)
    let words = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource, .copyDestination]))
    let config = try await device.createBuffer(.init(size: 24, usage: [.uniform, .copyDestination]))
    let matrix = [Float(10), 20, 30, 40].map(\.bitPattern)
    let data = try await device.createBuffer(.init(size: 112, usage: [.storage, .copySource, .copyDestination]))
    let output = try await device.createBuffer(.init(size: 64, usage: [.storage, .copySource, .copyDestination]))
    let read = try await device.createBuffer(.init(size: 112, usage: [.copyDestination, .mapRead]))
    let ticket = try await device.submitCommands([
        .write(buffer: data, offset: 0, words: [
            7, 700, 701, 702,
            1, 2, 3, 703, 4, 5, 6, 704, 100, 705, 706, 707,
            11, 12, 13, 708, 14, 15, 16, 709, 200, 710, 711, 712,
        ]),
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): data, .init(binding: 1): output],
                         workgroups: .init(x: 1)),
        .copy(source: output, destination: read, byteCount: 64), .read(buffer: read, byteCount: 64),
        .copy(source: data, destination: read, byteCount: 112), .read(buffer: read, byteCount: 112),
        .write(buffer: data, offset: 0, words: [11, 1001, 1002, 1003, 1, 2, 3, 1004, 4, 5, 6, 1005, 17, 1006, 1007, 1008]),
        .write(buffer: output, offset: 0, words: Array(repeating: 900, count: 16)),
        .write(buffer: words, offset: 0, words: [21, 22, 23, 24]),
        .write(buffer: config, offset: 0, words: [7, 999] + matrix),
        .dispatchCompute(pipeline: aggregatePipeline, bindings: [
            .init(binding: 0): data, .init(binding: 1): output,
            .init(binding: 2): words, .init(binding: 3): config,
        ], workgroups: .init(x: 1)),
        .copy(source: output, destination: read, byteCount: 64), .read(buffer: read, byteCount: 64),
        .copy(source: words, destination: read, byteCount: 16), .read(buffer: read, byteCount: 16),
    ])
    let snapshots = try await ticket.wait()
    guard snapshots.map(\.count) == [16, 28, 16, 4] else {
        throw GPUError.backendFailure("Unexpected nested storage readback: \(snapshots)")
    }
    return snapshots.flatMap { $0 }
}

private func nestedHalfStorageDiagnostic(_ device: Device) async throws -> [UInt32] {
    guard device.capabilities.features.contains(.shaderF16) else { return [] }
    let source = ShaderInput.wgsl("""
        enable f16;
        struct Cell { halves: array<f16, 2>, @size(4) marker: f16, }
        @group(0) @binding(0) var<storage, read_write> data: array<Cell>;

        @compute @workgroup_size(4) fn run(@builtin(local_invocation_index) lane: u32) {
            let row = lane / 2u;
            let item = lane % 2u;
            data[row].halves[item] += f16(lane + 1u);
        }
        """)
    let shader = try IndependentShaderCompiler().compileCompute(source, to: .wgsl)
    let pipeline = try await device.createComputePipeline(shader)
    let data = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource, .copyDestination]))
    let read = try await device.createBuffer(.init(size: 16, usage: [.copyDestination, .mapRead]))
    func pair(_ low: Double, _ high: Double) -> UInt32 {
        UInt32(Binary16.encode(low)) | (UInt32(Binary16.encode(high)) << 16)
    }
    let ticket = try await device.submitCommands([
        .write(buffer: data, offset: 0, words: [pair(1, 2), pair(9, -7), pair(3, 4), pair(10, -8)]),
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): data], workgroups: .init(x: 1)),
        .copy(source: data, destination: read, byteCount: 16), .read(buffer: read, byteCount: 16),
    ])
    let snapshots = try await ticket.wait()
    guard let result = snapshots.first, result.count == 4 else {
        throw GPUError.backendFailure("Unexpected nested half storage readback: \(snapshots)")
    }
    return result
}

private func nestedAtomicDiagnostic(_ device: Device) async throws -> [UInt32] {
    let source = ShaderInput.wgsl("""
        requires unrestricted_pointer_parameters;
        struct Cell { @size(16) signedValues: array<atomic<i32>, 2>, count: atomic<u32>, @size(12) tag: u32, }
        struct Data { @size(8) header: u32, cells: array<Cell>, }
        @group(0) @binding(0) var<storage, read_write> data: Data;
        @group(0) @binding(1) var<storage, read_write> output: array<vec4u, 8>;

        fn update(p: ptr<storage, atomic<i32>, read_write>) -> vec4i {
            let beforeMin = atomicMin(p, -20i);
            let beforeMax = atomicMax(p, -8i);
            let mismatch = atomicCompareExchangeWeak(p, -99i, 100i);
            var succeeded = false;
            loop {
                let attempt = atomicCompareExchangeWeak(p, -8i, -2i);
                succeeded = attempt.exchanged;
                if succeeded || attempt.old_value != -8i { break; }
            }
            let beforeSub = atomicSub(p, -4i);
            return vec4i(beforeMin, beforeMax, mismatch.old_value + select(0i, 100i, mismatch.exchanged),
                         beforeSub + select(0i, 100i, succeeded));
        }

        @compute @workgroup_size(4) fn run(@builtin(local_invocation_index) lane: u32) {
            let row = i32(lane) - 1i;
            let index = select(1u, 2u, lane == 2u);
            let values = update(&data.cells[row].signedValues[index]);
            let old = atomicAdd(&data.cells[row].count, lane + 1u);
            let beforeExchange = atomicExchange(&data.cells[row].count, old + 20u);
            output[lane * 2u] = bitcast<vec4u>(values);
            output[lane * 2u + 1u] = vec4u(old, beforeExchange, data.cells[row].tag, arrayLength(&data.cells));
            storageBarrier();
            atomicAdd(&data.cells[0].count, 1u);
        }
        """)
    let shader = try IndependentShaderCompiler().compileCompute(source, to: .wgsl)
    let pipeline = try await device.createComputePipeline(shader)
    let data = try await device.createBuffer(.init(size: 72, usage: [.storage, .copySource, .copyDestination]))
    let output = try await device.createBuffer(.init(size: 128, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 128, usage: [.copyDestination, .mapRead]))
    let ticket = try await device.submitCommands([
        .write(buffer: data, offset: 0, words: [
            7, 800, UInt32(bitPattern: -4), 5, 801, 802, 10, 100, 803, 804,
            UInt32(bitPattern: -5), UInt32(bitPattern: -6), 805, 806, 20, 200, 807, 808,
        ]),
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): data, .init(binding: 1): output],
                         workgroups: .init(x: 1)),
        .copy(source: output, destination: read, byteCount: 128), .read(buffer: read, byteCount: 128),
        .copy(source: data, destination: read, byteCount: 72), .read(buffer: read, byteCount: 72),
    ])
    let snapshots = try await ticket.wait()
    guard snapshots.map(\.count) == [32, 18] else {
        throw GPUError.backendFailure("Unexpected nested atomic readback: \(snapshots)")
    }
    return snapshots.flatMap { $0 }
}

private func nestedUniformDiagnostic(_ device: Device) async throws -> [UInt32] {
    let declarations = """
        struct Pair { dims: vec2u, @size(8) tag: u32, }
        struct Cell { entries: array<Pair, 2>, transform: mat2x2f, }
        struct Settings { @size(16) flags: u32, cells: array<Cell, 2>, @size(16) tail: u32, }
        @group(0) @binding(0) var<uniform> settings: Settings;

        """
    let computeSource = ShaderInput.wgsl(declarations + """
        @group(0) @binding(1) var<storage, read_write> output: array<vec4u, 4>;
        @compute @workgroup_size(4) fn run(@builtin(local_invocation_index) lane: u32) {
            let outer = i32(lane) - 1i;
            let inner = select(1u, 2u, lane == 2u);
            let cell = settings.cells[outer];
            let entry = cell.entries[inner];
            output[lane] = vec4u(entry.dims, entry.tag + settings.flags, u32(cell.transform[1].y) + settings.tail);
        }
        """)
    let renderSource = ShaderInput.wgsl(declarations + """
        @vertex fn vertexMain(@builtin(vertex_index) id: u32) -> @builtin(position) vec4f {
            let reach = f32(settings.cells[0].entries[1].dims.x);
            if id == 0u { return vec4f(-1.0f, -1.0f, 0.0f, 1.0f); }
            if id == 1u { return vec4f(reach, -1.0f, 0.0f, 1.0f); }
            return vec4f(-1.0f, reach, 0.0f, 1.0f);
        }
        @fragment fn fragmentMain() -> @location(0) vec4f {
            let cell = settings.cells[1];
            let correct = cell.entries[0].dims.y == 8u && settings.tail == 9u;
            return select(vec4f(1.0f, 0.0f, 0.0f, 1.0f), vec4f(0.0f, 1.0f, 0.0f, 1.0f), correct);
        }
        """)
    let compiler = IndependentShaderCompiler()
    let compute = try await device.createComputePipeline(compiler.compileCompute(computeSource, to: .wgsl))
    let vertex = try compiler.compileRenderStage(renderSource, entryPoint: "vertexMain", to: .wgsl)
    let fragment = try compiler.compileRenderStage(renderSource, entryPoint: "fragmentMain", to: .wgsl)
    let render = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm)
    guard case .spirv(let renderBinary) = try compiler.compile(renderSource, to: .spirv) else {
        throw GPUError.backendFailure("Expected binary SPIR-V for nested uniform render")
    }
    let roundTripVertex = try compiler.compileWebGPURenderStage(.spirv(renderBinary), entryPoint: "vertexMain")
    let roundTripFragment = try compiler.compileWebGPURenderStage(.spirv(renderBinary), entryPoint: "fragmentMain")
    let uniformMinima = [vertex, fragment, roundTripVertex, roundTripFragment].map {
        $0.uniformBindings.map(\.minimumByteSize)
    }
    guard uniformMinima == [[128], [128], [128], [128]] else {
        throw GPUError.backendFailure("Unexpected nested uniform render minimum sizes: \(uniformMinima)")
    }
    let roundTripRender = try await device.createRenderPipeline(
        vertex: roundTripVertex, fragment: roundTripFragment, colorFormat: .rgba8Unorm)
    let roundTripSource = ShaderInput.wgsl("""
        struct Row { dims: vec2u, tag: u32, reserved: u32 }
        struct Cell { rows: array<Row, 2>, transform: mat2x3f }
        struct Settings { cells: array<Cell, 2> }
        @group(0) @binding(0) var<uniform> settings: Settings;
        @group(0) @binding(1) var<storage, read_write> output: array<vec4u, 4>;
        @compute @workgroup_size(4)
        fn round_trip(@builtin(local_invocation_index) lane: u32) {
            let outer = lane >> 1u;
            let inner = lane & 1u;
            let cell = settings.cells[outer];
            let row = cell.rows[inner];
            let applied = cell.transform * vec2f(2.0, -1.0);
            let bits = bitcast<vec3u>(applied);
            output[lane] = vec4u(row.dims.x + row.dims.y + row.tag, bits.x, bits.y, bits.z);
        }
        """)
    guard case .spirv(let binary) = try compiler.compile(roundTripSource, to: .spirv) else {
        throw GPUError.backendFailure("Expected binary SPIR-V for nested uniform storage")
    }
    let roundTripShader = try compiler.compileWebGPUCompute(.spirv(binary), entryPoint: "round_trip")
    guard roundTripShader.uniformBindings.count == 1, let binding = roundTripShader.uniformBindings.first,
          binding.binding == .init(group: 0, binding: 0), binding.minimumByteSize == 128,
          !roundTripShader.requiresUniformBufferStandardLayout else {
        throw GPUError.backendFailure("Unexpected nested uniform round-trip layout")
    }
    let roundTrip = try await device.createComputePipeline(roundTripShader)
    let uniform = try await device.createBuffer(.init(size: 384, usage: [.uniform, .copyDestination]))
    let output = try await device.createBuffer(.init(size: 64, usage: [.storage, .copySource]))
    let target = try await device.createTexture(.init(width: 2, height: 2, format: .rgba8Unorm,
                                                       usage: [.renderAttachment, .copySource]))
    let read = try await device.createBuffer(.init(size: 512, usage: [.copyDestination, .mapRead]))
    let range = ComputeBufferBinding(buffer: uniform, offset: 256, size: 128)
    let words: [UInt32] =
        [3, 901, 902, 903, 1, 2, 3, 904, 4, 5, 6, 905] + [Float(1), 2, 3, 4].map(\.bitPattern) +
        [7, 8, 9, 906, 10, 11, 12, 907] + [Float(5), 6, 7, 8].map(\.bitPattern) + [9, 908, 909, 910]
    let roundTripInput: [UInt32] = [
        2, 3, 5, 901, 7, 11, 13, 902,
        Float(1).bitPattern, Float(2).bitPattern, Float(3).bitPattern, 903,
        Float(4).bitPattern, Float(5).bitPattern, Float(7).bitPattern, 904,
        17, 19, 23, 905, 29, 31, 37, 906,
        Float(-2).bitPattern, Float(3).bitPattern, Float(1).bitPattern, 907,
        Float(5).bitPattern, Float(-1).bitPattern, Float(4).bitPattern, 908,
    ]
    let clear = RenderClearColor(red: 0, green: 0, blue: 1, alpha: 1)
    let ticket = try await device.submitCommands([
        .write(buffer: uniform, offset: 256, words: words),
        .dispatchComputeRanges(pipeline: compute, bindings: [.init(binding: 0): range, .init(binding: 1): .init(buffer: output)],
                               workgroups: .init(x: 1)),
        .copy(source: output, destination: read, byteCount: 64), .read(buffer: read, byteCount: 64),
        .render(pipeline: render, target: target, clear: clear,
                vertexCount: 3, uniforms: [.init(binding: 0): range]),
        .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256),
        .read(buffer: read, byteCount: 512),
        .render(pipeline: roundTripRender, target: target, clear: clear,
                vertexCount: 3, uniforms: [.init(binding: 0): range]),
        .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256),
        .read(buffer: read, byteCount: 512),
        .write(buffer: uniform, offset: 256, words: roundTripInput),
        .dispatchComputeRanges(pipeline: roundTrip, bindings: [.init(binding: 0): range, .init(binding: 1): .init(buffer: output)],
                               workgroups: .init(x: 1)),
        .copy(source: output, destination: read, byteCount: 64), .read(buffer: read, byteCount: 64),
    ])
    let snapshots = try await ticket.wait()
    let roundTripExpected: [UInt32] = [
        10, 0xc0000000, 0xbf800000, 0xbf800000, 31, 0xc0000000, 0xbf800000, 0xbf800000,
        59, 0xc1100000, 0x40e00000, 0xc0000000, 97, 0xc1100000, 0x40e00000, 0xc0000000,
    ]
    guard snapshots.map(\.count) == [16, 128, 128, 16], snapshots[3] == roundTripExpected else {
        throw GPUError.backendFailure("Unexpected nested uniform readback: \(snapshots)")
    }
    let pixelIndices = [0, 1, 64, 65]
    let pixels = pixelIndices.map { snapshots[1][$0] }
    let roundTripPixels = pixelIndices.map { snapshots[2][$0] }
    guard pixels == roundTripPixels else {
        throw GPUError.backendFailure("Unexpected nested uniform render round-trip pixels: \(roundTripPixels)")
    }
    return snapshots[0] + pixels + snapshots[3] + roundTripPixels
}

private func nestedHalfUniformDiagnostic(_ device: Device) async throws -> [UInt32] {
    guard device.capabilities.features.contains(.shaderF16) else { return [] }
    let source = ShaderInput.wgsl("""
        enable f16;
        struct Packed { low: f16, high: f16, @size(12) tail: f16, }
        struct Settings { pairs: array<Packed, 2>, }
        @group(0) @binding(0) var<uniform> settings: Settings;
        @group(0) @binding(1) var<storage, read_write> output: array<vec4f, 2>;
        @compute @workgroup_size(2) fn run(@builtin(local_invocation_index) lane: u32) {
            let packed = settings.pairs[lane];
            output[lane] = vec4f(f32(packed.low), f32(packed.high), f32(packed.tail), 0.0f);
        }
        """)
    let shader = try IndependentShaderCompiler().compileCompute(source, to: .wgsl)
    let pipeline = try await device.createComputePipeline(shader)
    let uniform = try await device.createBuffer(.init(size: 32, usage: [.uniform, .copyDestination]))
    let output = try await device.createBuffer(.init(size: 32, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 32, usage: [.copyDestination, .mapRead]))
    let ticket = try await device.submitCommands([
        .write(buffer: uniform, offset: 0, words: [0xc0003c00, 0xdead4300, 701, 702, 0x4500c400, 0xbeef4600, 703, 704]),
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): uniform, .init(binding: 1): output],
                         workgroups: .init(x: 1)),
        .copy(source: output, destination: read, byteCount: 32), .read(buffer: read, byteCount: 32),
    ])
    let snapshots = try await ticket.wait()
    guard let result = snapshots.first, result.count == 8 else {
        throw GPUError.backendFailure("Unexpected nested f16 uniform readback: \(snapshots)")
    }
    return result
}

private func standardUniformDiagnostic(_ device: Device) async throws -> [UInt32] {
    guard device.capabilities.features.contains(.uniformBufferStandardLayout) else { return [] }
    let declarations = """
        requires uniform_buffer_standard_layout;
        requires unrestricted_pointer_parameters;
        struct Payload { bias: u32, }
        struct Settings { compact: array<vec2u, 3>, scale: Payload, tail: u32, }
        @group(0) @binding(0) var<uniform> settings: Settings;
        @group(0) @binding(2) var<uniform> dense: array<u32, 4>;
        override increment = 1u;

        """
    let computeSource = ShaderInput.wgsl(declarations + """
        @group(0) @binding(1) var<storage, read_write> output: array<vec4u, 4>;
        fn readDense(values: ptr<uniform, array<u32, 4>>, lane: u32) -> u32 { return (*values)[lane]; }
        @compute @workgroup_size(4) fn run(@builtin(local_invocation_index) lane: u32) {
            let config = settings;
            let item = config.compact[i32(lane) - 1i];
            output[lane] = vec4u(item, config.scale.bias + readDense(&dense, lane), config.tail + increment);
        }
        """)
    let renderSource = ShaderInput.wgsl(declarations + """
        @vertex fn vertexMain(@builtin(vertex_index) id: u32) -> @builtin(position) vec4f {
            let reach = f32(settings.compact[1].x);
            if id == 0u { return vec4f(-1.0f, -1.0f, 0.0f, 1.0f); }
            if id == 1u { return vec4f(reach, -1.0f, 0.0f, 1.0f); }
            return vec4f(-1.0f, reach, 0.0f, 1.0f);
        }
        @fragment fn fragmentMain() -> @location(0) vec4f {
            let valid = settings.scale.bias == 7u && settings.tail + increment == 12u;
            return select(vec4f(1.0f, 0.0f, 0.0f, 1.0f), vec4f(0.0f, 1.0f, 0.0f, 1.0f), valid);
        }
        """)
    let compiler = IndependentShaderCompiler()
    let values = ["increment": 3.0]
    let compute = try await device.createComputePipeline(compiler.compileCompute(computeSource, to: .wgsl, constants: values))
    let vertex = try compiler.compileRenderStage(renderSource, entryPoint: "vertexMain", to: .wgsl, constants: values)
    let fragment = try compiler.compileRenderStage(renderSource, entryPoint: "fragmentMain", to: .wgsl, constants: values)
    let render = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm)
    let config = try await device.createBuffer(.init(size: 32, usage: [.uniform, .copyDestination]))
    let dense = try await device.createBuffer(.init(size: 16, usage: [.uniform, .copyDestination]))
    let output = try await device.createBuffer(.init(size: 64, usage: [.storage, .copySource]))
    let target = try await device.createTexture(.init(width: 2, height: 2, format: .rgba8Unorm,
                                                       usage: [.renderAttachment, .copySource]))
    let read = try await device.createBuffer(.init(size: 512, usage: [.copyDestination, .mapRead]))
    let ticket = try await device.submitCommands([
        .write(buffer: config, offset: 0, words: [10, 20, 30, 40, 50, 60, 7, 9]),
        .write(buffer: dense, offset: 0, words: [100, 200, 300, 400]),
        .dispatchCompute(pipeline: compute, bindings: [.init(binding: 0): config, .init(binding: 1): output, .init(binding: 2): dense],
                         workgroups: .init(x: 1)),
        .copy(source: output, destination: read, byteCount: 64), .read(buffer: read, byteCount: 64),
        .render(pipeline: render, target: target, clear: .init(red: 0, green: 0, blue: 1, alpha: 1),
                vertexCount: 3, uniforms: [.init(binding: 0): .init(buffer: config)]),
        .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256),
        .read(buffer: read, byteCount: 512),
    ])
    let snapshots = try await ticket.wait()
    guard snapshots.map(\.count) == [16, 128] else {
        throw GPUError.backendFailure("Unexpected standard uniform layout readback: \(snapshots)")
    }
    let pixels = snapshots[1]
    return snapshots[0] + [pixels[0], pixels[1], pixels[64], pixels[65]]
}

private func renderStorageBufferDiagnostic(_ device: Device) async throws -> [UInt32] {
    let source = ShaderInput.wgsl("""
        struct Cell { @size(8) count: atomic<u32>, marker: u32, }
        struct Feedback { header: u32, cells: array<Cell>, }
        @group(0) @binding(0) var<storage, read> positions: array<vec2f>;
        @group(0) @binding(1) var<storage, read_write> feedback: Feedback;

        fn position(index: u32) -> vec2f {
            if arrayLength(&positions) != 3u { return vec2f(4.0f); }
            return positions[index];
        }

        @vertex fn vertexMain(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
            return vec4f(position(index), 0.0f, 1.0f);
        }

        fn storeFeedback(index: u32) {
            atomicStore(&feedback.cells[index].count, arrayLength(&feedback.cells) * 100u + index);
        }

        @fragment fn fragmentMain(@builtin(position) point: vec4f) -> @location(0) vec4f {
            let pixel = vec2u(point.xy);
            storeFeedback(pixel.y * 2u + pixel.x);
            return vec4f(0.0f, 1.0f, 0.0f, 1.0f);
        }
        """)
    let compiler = IndependentShaderCompiler()
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vertexMain", to: .wgsl)
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fragmentMain", to: .wgsl)
    let pipeline = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm)
    let positions = try await device.createBuffer(.init(size: 280, usage: [.storage, .copyDestination]))
    let feedback = try await device.createBuffer(.init(size: 308, usage: [.storage, .copyDestination, .copySource]))
    let indices = try await device.createBuffer(.init(size: 8, usage: [.index, .copyDestination]))
    let target = try await device.createTexture(.init(width: 2, height: 2, format: .rgba8Unorm,
                                                       usage: [.renderAttachment, .copySource]))
    let read = try await device.createBuffer(.init(size: 512, usage: [.copyDestination, .mapRead]))
    let clear = RenderClearColor(red: 0, green: 0, blue: 1, alpha: 1)
    let bindings: [ComputeBindingSlot: ComputeBufferBinding] = [
        .init(binding: 0): .init(buffer: positions, offset: 256, size: 24),
        .init(binding: 1): .init(buffer: feedback, offset: 256, size: 52),
    ]
    let ticket = try await device.submitCommands([
        .write(buffer: positions, offset: 256, words: [Float(-1), -1, 3, -1, -1, 3].map(\.bitPattern)),
        .write(buffer: feedback, offset: 256, words: [7, 0, 801, 10, 0, 802, 20, 0, 803, 30, 0, 804, 40]),
        .write(buffer: indices, offset: 0, words: [0x00010000, 2]),
        .render(pipeline: pipeline, target: target, clear: clear, vertexCount: 3, storageBuffers: bindings),
        .renderIndexed(pipeline: pipeline, target: target, clear: clear,
                       indexBuffer: .init(buffer: indices, format: .uint16), draw: .init(indexCount: 3),
                       storageBuffers: bindings),
        .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256),
        .read(buffer: read, byteCount: 512),
        .copy(source: feedback, destination: read, byteCount: 308),
        .read(buffer: read, byteCount: 308),
    ])
    let snapshots = try await ticket.wait()
    guard snapshots.map(\.count) == [128, 77] else {
        throw GPUError.backendFailure("Unexpected render storage buffer readback: \(snapshots)")
    }
    let pixels = snapshots[0]
    return [pixels[0], pixels[1], pixels[64], pixels[65]] + Array(snapshots[1][64..<77])
}

private func renderDiagnostic(
    _ device: Device
) async throws -> (checked: Int, probe: [UInt32], vertexStreams: Int, indexedVertices: Int, instancesPerDraw: Int) {
    let source = ShaderInput.wgsl("""
        struct VertexInput {
            @location(4) clip: vec4f,
            @location(1) tint: vec4f,
            @location(6) gain: i32,
            @builtin(instance_index) instance: u32,
        }
        struct VertexOutput {
            @location(2) tint: vec4f,
            @builtin(position) clip: vec4f,
            @location(5) @interpolate(perspective, centroid) gain: f32,
        }
        struct FragmentInput {
            @location(5) @interpolate(perspective, centroid) gain: f32,
            @location(2) tint: vec4f,
        }
        fn makeVertex(tint: vec4f, clip: vec4f, gain: f32) -> VertexOutput {
            var output: VertexOutput;
            output.tint = tint;
            output.clip = clip;
            output.gain = gain;
            return output;
        }
        @vertex fn vertexMain(input: VertexInput) -> VertexOutput {
            if input.gain == -1i && input.instance == 2u {
                return makeVertex(input.tint, input.clip, 1.0);
            }
            return makeVertex(input.tint, vec4f(0.0, 0.0, 0.0, 1.0), 0.0);
        }
        @fragment fn fragmentMain(input: FragmentInput) -> @location(0) vec4f {
            return shade(input.tint, input.gain);
        }
        fn shade(tint: vec4f, gain: f32) -> vec4f {
            var color = tint;
            let before = color;
            color.r = 0.0;
            return vec4f(before.r * gain, color.g, color.b, before.a);
        }
        """)
    let compiler = IndependentShaderCompiler()
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vertexMain", to: .wgsl)
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fragmentMain", to: .wgsl)
    let layouts: [VertexBufferLayout] = [
        .init(stride: 32, attributes: [.init(location: 4, offset: 0, format: .float32x4),
                                       .init(location: 1, offset: 16, format: .float32x3)]),
        .init(stride: 8, stepMode: .instance, attributes: [.init(location: 6, offset: 4, format: .sint32)]),
    ]
    let pipeline = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm,
                                                         vertexBuffers: layouts)
    let geometry = try await device.createBuffer(.init(size: 128, usage: [.vertex, .copyDestination]))
    let gain = try await device.createBuffer(.init(size: 24, usage: [.vertex, .copyDestination]))
    let shortIndices = try await device.createBuffer(.init(size: 12, usage: [.index, .copyDestination]))
    let wordIndices = try await device.createBuffer(.init(size: 24, usage: [.index, .copyDestination]))
    let vertices: [Float] = [
        99, 4, 4, 0, 1, 0, 0, 0,
        99, -1, 1, 0.25, 1, 1, 0, 0,
        99, 2, 2, 0.5, 2, 0, 1, 0,
        99, -4, -4, 1, 4, 0, 0, 1,
    ]
    let clear = RenderClearColor(red: 0, green: 0, blue: 1, alpha: 1)
    let texture = try await device.createTexture(.init(width: 8, height: 8, format: .rgba8Unorm,
                                                       usage: [.renderAttachment, .copySource]))
    let byteCount: UInt64 = 1824
    let buffer = try await device.createBuffer(.init(size: byteCount, usage: [.copyDestination, .mapRead]))
    let vertexBindings: [VertexBufferBinding] = [.init(buffer: geometry, offset: 4), .init(buffer: gain)]
    let instances = DrawInstances(count: 2, first: 1)
    let draw = IndexedDraw(indexCount: 3, firstIndex: 1, baseVertex: -1, instances: instances)
    if device.supportsPresentation {
        let initial = try await device.createTexture(.init(width: 4, height: 4, format: .rgba8Unorm, usage: .copySource))
        let frame = try await device.present(initial)
        try await device.destroyTexture(initial)
        try await frame.wait()
    }
    do {
        _ = try await device.submitCommands([.copyTextureToBuffer(source: texture, destination: buffer, bytesPerRow: 32)])
        throw GPUError.backendFailure("Swift accepted an invalid texture copy stride")
    } catch GPUError.validation {}
    do {
        _ = try await device.submitCommands([
            .render(pipeline: pipeline, target: texture, clear: clear, vertexCount: 4,
                    vertexBuffers: [.init(buffer: geometry, offset: 8), .init(buffer: gain)]),
        ])
        throw GPUError.backendFailure("Swift accepted an out-of-range vertex fetch")
    } catch GPUError.validation {}
    do {
        _ = try await device.submitCommands([
            .renderIndexed(pipeline: pipeline, target: texture, clear: clear,
                           indexBuffer: .init(buffer: shortIndices, format: .uint16, offset: 2, size: 6),
                           draw: draw, vertexBuffers: vertexBindings),
        ])
        throw GPUError.backendFailure("Swift accepted an out-of-range index slice")
    } catch GPUError.validation {}
    do {
        _ = try await device.submitCommands([
            .renderIndexed(pipeline: pipeline, target: texture, clear: clear,
                           indexBuffer: .init(buffer: wordIndices, format: .uint32, offset: 4),
                           draw: .init(indexCount: 3, firstIndex: 1, baseVertex: -1,
                                       instances: .init(count: 3, first: 1)),
                           vertexBuffers: vertexBindings),
        ])
        throw GPUError.backendFailure("Swift accepted an out-of-range instance fetch")
    } catch GPUError.validation {}
    let ticket = try await device.submitCommands([
        .write(buffer: geometry, offset: 0, words: vertices.map(\.bitPattern)),
        .write(buffer: gain, offset: 0, words: [99, 0, 99, 0, 99, .max]),
        .write(buffer: shortIndices, offset: 0, words: [0x0000ffff, 0x00030002, 0xffff0004]),
        .write(buffer: wordIndices, offset: 0, words: [.max, 0, 2, 3, 4, .max]),
        .copyTextureToBuffer(source: texture, destination: buffer, bytesPerRow: 256),
        .read(buffer: buffer, byteCount: byteCount),
        .renderIndexed(pipeline: pipeline, target: texture, clear: clear,
                       indexBuffer: .init(buffer: shortIndices, format: .uint16, offset: 2, size: 8),
                       draw: draw, vertexBuffers: vertexBindings),
        .copyTextureToBuffer(source: texture, destination: buffer, bytesPerRow: 256),
        .read(buffer: buffer, byteCount: byteCount),
        .renderIndexed(pipeline: pipeline, target: texture, clear: clear,
                       indexBuffer: .init(buffer: wordIndices, format: .uint32, offset: 4, size: 16),
                       draw: draw, vertexBuffers: vertexBindings),
        .copyTextureToBuffer(source: texture, destination: buffer, bytesPerRow: 256),
        .read(buffer: buffer, byteCount: byteCount),
        .render(pipeline: pipeline, target: texture, clear: clear, vertexCount: 3, vertexBuffers: vertexBindings,
                firstVertex: 1, instances: instances),
        .copyTextureToBuffer(source: texture, destination: buffer, bytesPerRow: 256),
        .read(buffer: buffer, byteCount: byteCount),
    ])
    let frame = device.supportsPresentation ? try await device.present(texture) : nil
    try await device.destroyBuffer(geometry)
    try await device.destroyBuffer(shortIndices)
    try await device.destroyTexture(texture)
    let snapshots = try await ticket.wait()
    try await frame?.wait()
    guard snapshots.count == 4, snapshots.allSatisfy({ $0.count == Int(byteCount / 4) }),
          snapshots[0].allSatisfy({ $0 == 0 }), snapshots[1] == snapshots[2],
          snapshots[1] == snapshots[3] else {
        throw GPUError.backendFailure("Unexpected WebGPU texture snapshots for instanced rendering")
    }
    var checked = 0
    let pixels = snapshots[1]
    for y in 0..<8 {
        for x in 0..<8 where x + y != 7 {
            let pixel = pixels[y * 64 + x]
            if x + y < 7 {
                let u = (Double(x) + 0.5) / 8, v = (Double(y) + 0.5) / 8
                let weights = [1 - u - v, u / 2, v / 4]
                let sum = weights.reduce(0, +)
                guard pixel >> 24 == 255, weights.enumerated().allSatisfy({ channel, weight in
                    let expected = Int((255 * weight / sum).rounded())
                    return abs(Int((pixel >> (8 * channel)) & 255) - expected) <= 1
                }) else { throw GPUError.backendFailure("Unexpected WebGPU interpolation at \(x), \(y)") }
            } else if pixel != 0xffff0000 {
                throw GPUError.backendFailure("Unexpected WebGPU clear pixel at \(x), \(y)")
            }
            checked += 1
        }
    }
    return (checked, (0..<3).map { (pixels[0] >> (8 * $0)) & 255 }, pipeline.vertexBuffers.count,
            2 * Int(draw.indexCount), Int(instances.count))
}

private func indirectDispatchDiagnostic(_ device: Device) async throws -> [UInt32] {
    let limit = device.capabilities.maxComputeWorkgroupsPerDimension
    let invalidCount = limit == .max ? 0 : limit + 1
    let compiler = IndependentShaderCompiler()
    let producer = try await device.createComputePipeline(compiler.compileCompute(.wgsl("""
        @group(0) @binding(0) var<storage, read_write> args: array<u32, 10>;
        @compute @workgroup_size(1)
        fn write_args() {
            args[1] = 2u; args[2] = 3u; args[3] = 2u;
            args[4] = 1u; args[5] = \(invalidCount)u; args[6] = 1u;
            args[7] = 1u; args[8] = 1u; args[9] = 0u;
        }
        """), to: .wgsl))
    let consumer = try await device.createComputePipeline(compiler.compileCompute(.wgsl("""
        @group(0) @binding(0) var<storage, read_write> counter: atomic<u32>;
        @compute @workgroup_size(2)
        fn count(@builtin(num_workgroups) groups: vec3<u32>) {
            atomicAdd(&counter, 1u + groups.x + groups.y * 10u + groups.z * 100u);
        }
        """), to: .wgsl))
    let arguments = try await device.createBuffer(.init(size: 40, usage: [.indirect, .storage, .copySource]))
    let counter = try await device.createBuffer(.init(size: 4, usage: [.storage, .copySource]))
    let readback = try await device.createBuffer(.init(size: 40, usage: [.copyDestination, .mapRead]))
    let slot = ComputeBindingSlot(binding: 0)
    let bindings = [slot: StorageBufferBinding(buffer: counter)]
    let ticket = try await device.submitCommands([
        .dispatchCompute(pipeline: producer, bindings: [slot: arguments], workgroups: .init(x: 1)),
        .dispatchComputeIndirect(pipeline: consumer, bindings: bindings, indirectBuffer: arguments, indirectOffset: 16),
        .copy(source: counter, destination: readback, byteCount: 4),
        .read(buffer: readback, byteCount: 4),
        .dispatchComputeIndirect(pipeline: consumer, bindings: bindings, indirectBuffer: arguments, indirectOffset: 4),
        .dispatchComputeIndirect(pipeline: consumer, bindings: bindings, indirectBuffer: arguments, indirectOffset: 28),
        .copy(source: counter, destination: readback, byteCount: 4),
        .read(buffer: readback, byteCount: 4),
        .copy(source: arguments, destination: readback, byteCount: 40),
        .read(buffer: readback, byteCount: 40),
    ])
    return try await ticket.wait().flatMap { $0 }
}

private func indirectDrawDiagnostic(_ device: Device) async throws -> [UInt32] {
    let firstInstance = device.capabilities.features.contains(.indirectFirstInstance) ? 1 : 0
    let compiler = IndependentShaderCompiler()
    let producer = try await device.createComputePipeline(compiler.compileCompute(.wgsl("""
        @group(0) @binding(0) var<storage, read_write> args: array<u32, 29>;
        @compute @workgroup_size(1)
        fn prepare() {
            args[1] = 3u; args[2] = 1u; args[3] = 1u; args[4] = \(firstInstance)u;
            args[5] = 3u; args[6] = 1u; args[7] = 1u; args[8] = 0xffffffffu; args[9] = \(firstInstance)u;
            args[10] = 3u; args[11] = 1u; args[12] = 2u; args[13] = 0xffffffffu; args[14] = 1u;
            args[15] = 3u; args[16] = 0u; args[17] = 1u; args[18] = 1u;
            args[19] = 3u; args[20] = 1u; args[21] = 2u; args[22] = 1u;
            args[23] = 3u; args[24] = 1u; args[25] = 1u; args[26] = 2u;
            args[27] = 0xabcdefu; args[28] = 0x123456u;
        }
        """), to: .wgsl))
    let source = ShaderInput.wgsl("""
        struct Raster { @builtin(position) position: vec4f, @location(0) tint: f32, }
        @vertex fn vs(@location(0) position: vec2f, @location(1) tint: f32) -> Raster {
            return Raster(vec4f(position, 0.0, 1.0), tint);
        }
        @fragment fn fs(in: Raster) -> @location(0) vec4f {
            return vec4f(1.0 - in.tint, in.tint, 0.0, 1.0);
        }
        """)
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .wgsl)
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: .wgsl)
    let pipeline = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm,
        vertexBuffers: [
            .init(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float32x2)]),
            .init(stride: 4, stepMode: .instance, attributes: [.init(location: 1, offset: 0, format: .float32)]),
        ])
    let arguments = try await device.createBuffer(.init(size: 116, usage: [.indirect, .storage, .copySource]))
    let plainArguments = try await device.createBuffer(.init(size: 16, usage: [.indirect, .copyDestination]))
    let positions = try await device.createBuffer(.init(size: 32, usage: [.vertex, .copyDestination]))
    let instances = try await device.createBuffer(.init(size: 8, usage: [.vertex, .copyDestination]))
    let indices = try await device.createBuffer(.init(size: 12, usage: [.index, .copyDestination]))
    let wideIndices = try await device.createBuffer(.init(size: 24, usage: [.index, .copyDestination]))
    let target = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm,
                                                       usage: [.renderAttachment, .copySource]))
    let readback = try await device.createBuffer(.init(size: 256, usage: [.copyDestination, .mapRead]))
    let index = IndexBufferBinding(buffer: indices, format: .uint16, offset: 2, size: 8)
    let wideIndex = IndexBufferBinding(buffer: wideIndices, format: .uint32, offset: 4, size: 16)
    let streams = [VertexBufferBinding(buffer: positions), VertexBufferBinding(buffer: instances)]
    let clear = RenderColorLoadAction.clear(.init(red: 0, green: 0, blue: 1, alpha: 1))
    func draw(_ offset: UInt64, index: IndexBufferBinding? = nil, buffer: ResourceID? = nil) -> BufferCommand {
        .renderIndirect(pipeline: pipeline, target: target, colorLoad: clear, indirectBuffer: buffer ?? arguments,
                        indirectOffset: offset, indexBuffer: index, vertexBuffers: streams)
    }
    let indexWords: [UInt32] = [0x00630063, 0x00030002, 0x00630004]
    let capture: [BufferCommand] = [
        .copyTextureToBuffer(source: target, destination: readback, bytesPerRow: 256),
        .read(buffer: readback, byteCount: 4),
    ]
    let prepare: [BufferCommand] = [
        .write(buffer: positions, offset: 0, words: [Float(4), 4, -1, -1, 3, -1, -1, 3].map(\.bitPattern)),
        .write(buffer: instances, offset: 0, words: [0, Float(1).bitPattern]),
        .write(buffer: plainArguments, offset: 0, words: [3, 1, 1, 1]),
        .write(buffer: indices, offset: 0, words: indexWords),
        .write(buffer: wideIndices, offset: 0, words: [99, 99, 2, 3, 4, 99]),
        .dispatchCompute(pipeline: producer, bindings: [.init(binding: 0): arguments], workgroups: .init(x: 1)),
    ]
    let ticket = try await device.submitCommands(prepare + [
        draw(4), draw(20, index: index), draw(40, index: index), draw(60), draw(76), draw(92),
        draw(0, buffer: plainArguments), draw(20, index: wideIndex),
    ].flatMap { [$0] + capture } + [
        .copy(source: arguments, destination: readback, byteCount: 116),
        .read(buffer: readback, byteCount: 116),
    ])
    return try await ticket.wait().flatMap { $0 }
}

private func clipShader(count: Int) -> String {
    """
    enable clip_distances;
    struct Output {
        @builtin(clip_distances) planes: array<f32, \(count)>,
        @location(0) tint: f32,
        @builtin(position) position: vec4f,
    }

    fn projected(p: vec2f) -> Output {
        var result: Output;
        result.position = vec4f(p, 0.0, 1.0);
        result.tint = 1.0;
        result.planes[0] = p.x;
        \(count > 1 ? "result.planes[\(count - 1)] = p.y;" : "")
        return result;
    }

    @vertex fn vs(@location(0) point: vec2f) -> Output {
        return projected(point);
    }

    @fragment fn fs(@location(0) tint: f32) -> @location(0) vec4f {
        return vec4f(0.0, tint, 0.0, 1.0);
    }
    """
}


private func clipDistanceDiagnostic(_ device: Device) async throws -> [UInt32] {
    let compiler = IndependentShaderCompiler()
    let eightPlaneSource = ShaderInput.wgsl(clipShader(count: 8))
    let vertex = try compiler.compileRenderStage(eightPlaneSource, entryPoint: "vs", to: .wgsl)
    let fragment = try compiler.compileRenderStage(eightPlaneSource, entryPoint: "fs", to: .wgsl)
    let layout = VertexBufferLayout(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float32x2)])
    if !device.capabilities.features.contains(.clipDistances) {
        do {
            _ = try await device.createRenderPipeline(vertex: vertex, fragment: fragment,
                                                       colorFormat: .rgba8Unorm, vertexBuffers: [layout])
        } catch GPUError.validation("Feature not advertised") {
            return []
        }
        throw GPUError.backendFailure("Pipeline accepted clip distances without the device feature")
    }
    let onePlane = try compiler.compileRenderStage(.wgsl(clipShader(count: 1)), entryPoint: "vs", to: .wgsl)
    let onePlanePipeline = try await device.createRenderPipeline(vertex: onePlane, fragment: fragment,
                                                                 colorFormat: .rgba8Unorm, vertexBuffers: [layout])
    let eightPlanePipeline = try await device.createRenderPipeline(vertex: vertex, fragment: fragment,
                                                                   colorFormat: .rgba8Unorm, vertexBuffers: [layout])
    let spirv = try compiler.compileWebGPURenderStage(.spirv(clipDistanceSPIRV()), entryPoint: "main")
    let spirvPipeline = try await device.createRenderPipeline(vertex: spirv, fragment: fragment,
                                                              colorFormat: .rgba8Unorm, vertexBuffers: [layout])
    guard case .spirv(let binary) = try compiler.compile(eightPlaneSource, to: .spirv) else {
        throw GPUError.backendFailure("Expected binary SPIR-V for clip distances")
    }
    let roundTrip = ShaderInput.spirv(binary)
    let importedPipeline = try await device.createRenderPipeline(
        vertex: compiler.compileWebGPURenderStage(roundTrip, entryPoint: "vs"),
        fragment: compiler.compileWebGPURenderStage(roundTrip, entryPoint: "fs"),
        colorFormat: .rgba8Unorm, vertexBuffers: [layout])
    let vertices = try await device.createBuffer(.init(size: 24, usage: [.vertex, .copyDestination]))
    let target = try await device.createTexture(.init(width: 2, height: 2, format: .rgba8Unorm,
                                                       usage: [.renderAttachment, .copySource]))
    let readback = try await device.createBuffer(.init(size: 512, usage: [.copyDestination, .mapRead]))
    let clear = RenderClearColor(red: 0, green: 0, blue: 1, alpha: 1)
    let capture: [BufferCommand] = [
        .copyTextureToBuffer(source: target, destination: readback, bytesPerRow: 256),
        .read(buffer: readback, byteCount: 512),
    ]
    let commands: [BufferCommand] = [
        .write(buffer: vertices, offset: 0, words: [Float(-1), -1, 3, -1, -1, 3].map(\.bitPattern)),
    ] + [onePlanePipeline, eightPlanePipeline, importedPipeline].flatMap { pipeline in
        [.render(pipeline: pipeline, target: target, clear: clear, vertexCount: 3, vertexBuffers: [.init(buffer: vertices)])] + capture
    } + [
        .write(buffer: vertices, offset: 0, words: [Float(-1), 1, 3, 1, -1, -3].map(\.bitPattern)),
        .render(pipeline: spirvPipeline, target: target, clear: clear, vertexCount: 3, vertexBuffers: [.init(buffer: vertices)]),
    ] + capture
    let pixels = try await device.submitCommands(commands).wait()
    guard pixels.map(\.count) == [128, 128, 128, 128] else {
        throw GPUError.backendFailure("Unexpected clipping readback")
    }
    let renderedPixels = pixels.map { [$0[0], $0[1], $0[64], $0[65]] }
    guard renderedPixels[1] == renderedPixels[2] else {
        throw GPUError.backendFailure("SPIR-V clip distances differed from direct WGSL: \(renderedPixels)")
    }
    return (renderedPixels.prefix(2) + renderedPixels.suffix(1)).flatMap { $0 }
}

private func clipDistanceSPIRV() -> [UInt32] {
    func op(_ code: UInt32, _ operands: UInt32...) -> [UInt32] {
        [UInt32(operands.count + 1) << 16 | code] + operands
    }
    var words: [UInt32] = [0x07230203, 0x00010300, 0, 100, 0]
    for instruction in [
        op(17, 1), op(17, 32), op(14, 0, 1), op(15, 0, 14, 0x6e69616d, 0, 15, 16, 18),
        op(71, 9, 2), op(72, 9, 0, 11, 0), op(72, 9, 1, 11, 3), op(71, 15, 30, 0), op(71, 18, 30, 0),
        op(19, 1), op(21, 2, 32, 0), op(22, 3, 32), op(23, 4, 3, 2), op(23, 5, 3, 4),
        op(43, 2, 20, 0), op(43, 2, 21, 1), op(43, 2, 22, 5), op(43, 2, 23, 4),
        op(43, 3, 24, 0), op(43, 3, 25, Float(1).bitPattern),
        op(28, 8, 3, 22), op(32, 6, 1, 4), op(32, 7, 3, 5), op(32, 12, 3, 3), op(33, 13, 1),
        op(30, 9, 5, 8), op(32, 11, 3, 9), op(59, 6, 15, 1), op(59, 11, 16, 3), op(59, 12, 18, 3),
        op(54, 1, 14, 0, 13), op(248, 30), op(61, 4, 31, 15), op(81, 3, 32, 31, 0), op(81, 3, 33, 31, 1),
        op(80, 5, 34, 32, 33, 24, 25), op(65, 7, 35, 16, 20), op(62, 35, 34), op(62, 18, 25),
        op(65, 12, 37, 16, 21, 20), op(62, 37, 32),
        op(127, 3, 38, 33), op(65, 12, 39, 16, 21, 23), op(62, 39, 38), op(253), op(56),
    ] { words += instruction }
    return words
}

private let primitiveIndexShader = """
    enable primitive_index;
    struct VertexOutput {
        @builtin(position) position: vec4f,
        @location(0) tint: f32,
    }

    @vertex fn vs(@location(0) point: vec2f, @builtin(instance_index) instance: u32) -> VertexOutput {
        return VertexOutput(vec4f(point.x, point.y - f32(instance), 0.0, 1.0), 1.0);
    }

    struct FragmentInput {
        @location(0) tint: f32,
        @builtin(primitive_index) index: u32,
        @builtin(position) pixel: vec4f,
        @builtin(front_facing) front: bool,
    }

    fn encode(index: u32, tint: f32) -> vec4u {
        return vec4u(index + u32(tint), 0u, 0u, 255u);
    }

    @fragment fn fs(input: FragmentInput) -> @location(0) vec4u {
        return encode(input.index, input.tint);
    }

    @fragment fn fs_direct(@builtin(primitive_index) index: u32) -> @location(0) vec4u {
        return encode(index, 1.0);
    }
    """

private func verifyNativeBarycentricsAreUnavailable(_ device: Device) async throws {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl("""
        @vertex fn vs() -> @builtin(position) vec4f { return vec4f(0, 0, 0, 1); }
        @fragment fn fs(@builtin(barycentric) weights: vec3f) -> @location(0) vec4f {
            return vec4f(weights, 1);
        }
        """)
    let vertex = try compiler.compileWebGPURenderStage(source, entryPoint: "vs")
    let glsl = ShaderInput.glsl("""
        #version 450
        #extension GL_EXT_fragment_shader_barycentric : require
        layout(location = 0) out vec4 color;
        void main() { color = vec4(gl_BaryCoordNoPerspEXT, 1.0); }
        """, stage: .fragment, defines: [:])
    // glslang output for the same fragment using gl_BaryCoordEXT.
    let spirv = ShaderInput.spirv([
        0x07230203, 0x00010300, 0x0008000b, 0x00000013, 0x00000000, 0x00020011, 0x00000001, 0x00020011,
        0x000014a4, 0x000a000a, 0x5f565053, 0x5f52484b, 0x67617266, 0x746e656d, 0x6168735f, 0x5f726564,
        0x79726162, 0x746e6563, 0x00636972, 0x0003000e, 0x00000000, 0x00000001, 0x0007000f, 0x00000004,
        0x00000004, 0x6e69616d, 0x00000000, 0x00000009, 0x0000000c, 0x00030010, 0x00000004, 0x00000007,
        0x00040047, 0x00000009, 0x0000001e, 0x00000000, 0x00040047, 0x0000000c, 0x0000000b, 0x000014a6,
        0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00030016, 0x00000006, 0x00000020,
        0x00040017, 0x00000007, 0x00000006, 0x00000004, 0x00040020, 0x00000008, 0x00000003, 0x00000007,
        0x0004003b, 0x00000008, 0x00000009, 0x00000003, 0x00040017, 0x0000000a, 0x00000006, 0x00000003,
        0x00040020, 0x0000000b, 0x00000001, 0x0000000a, 0x0004003b, 0x0000000b, 0x0000000c, 0x00000001,
        0x0004002b, 0x00000006, 0x0000000e, 0x3f800000, 0x00050036, 0x00000002, 0x00000004, 0x00000000,
        0x00000003, 0x000200f8, 0x00000005, 0x0004003d, 0x0000000a, 0x0000000d, 0x0000000c, 0x00050051,
        0x00000006, 0x0000000f, 0x0000000d, 0x00000000, 0x00050051, 0x00000006, 0x00000010, 0x0000000d,
        0x00000001, 0x00050051, 0x00000006, 0x00000011, 0x0000000d, 0x00000002, 0x00070050, 0x00000007,
        0x00000012, 0x0000000f, 0x00000010, 0x00000011, 0x0000000e, 0x0003003e, 0x00000009, 0x00000012,
        0x000100fd, 0x00010038,
    ])
    for input in [glsl, spirv] {
        let native = try compiler.compileRenderStage(input, entryPoint: "main", to: .wgsl)
        guard native.requiresBarycentrics else { throw GPUError.backendFailure("Imported barycentrics lost their device feature") }
        do {
            _ = try compiler.compileWebGPURenderStage(input, entryPoint: "main")
        } catch ShaderFailure.unsupported {
            continue
        }
        throw GPUError.backendFailure("Browser profile accepted imported native barycentrics")
    }
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: .wgsl)
    do {
        _ = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm)
    } catch GPUError.validation("Feature not advertised") {
        return
    }
    throw GPUError.backendFailure("Browser pipeline accepted native barycentrics")
}

private func primitiveIndexDiagnostic(_ device: Device) async throws -> [UInt32] {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl(primitiveIndexShader)
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .wgsl)
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: .wgsl)
    let layout = VertexBufferLayout(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float32x2)])
    if !device.capabilities.features.contains(.primitiveIndex) {
        do {
            _ = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Uint,
                                                       vertexBuffers: [layout])
        } catch GPUError.validation("Feature not advertised") {
            return []
        }
        throw GPUError.backendFailure("Pipeline accepted primitive_index without the device feature")
    }
    let directFragment = try compiler.compileRenderStage(source, entryPoint: "fs_direct", to: .wgsl)
    guard case .spirv(let binary) = try compiler.compile(source, to: .spirv) else {
        throw GPUError.backendFailure("Expected binary SPIR-V for primitive index")
    }
    let roundTrip = ShaderInput.spirv(binary)
    let importedVertex = try compiler.compileWebGPURenderStage(roundTrip, entryPoint: "vs")
    let importedFragment = try compiler.compileWebGPURenderStage(roundTrip, entryPoint: "fs")
    let importedDirectFragment = try compiler.compileWebGPURenderStage(roundTrip, entryPoint: "fs_direct")
    guard importedFragment.readsPrimitiveIndex, importedDirectFragment.readsPrimitiveIndex else {
        throw GPUError.backendFailure("SPIR-V round trip lost primitive index reflection")
    }
    var pipelines: [(list: RenderPipeline, strip: RenderPipeline)] = []
    for (stageVertex, listFragment, stripFragment) in [
        (vertex, fragment, directFragment), (importedVertex, importedFragment, importedDirectFragment),
    ] {
        let list = try await device.createRenderPipeline(vertex: stageVertex, fragment: listFragment,
                                                         colorFormat: .rgba8Uint, vertexBuffers: [layout])
        let strip = try await device.createRenderPipeline(vertex: stageVertex, fragment: stripFragment,
            colorFormat: .rgba8Uint, vertexBuffers: [layout],
            primitive: .init(topology: .triangleStrip, stripIndexFormat: .uint16))
        pipelines.append((list, strip))
    }
    let positions = try await device.createBuffer(.init(size: 48, usage: [.vertex, .copyDestination]))
    let indices = try await device.createBuffer(.init(size: 16, usage: [.index, .copyDestination]))
    let target = try await device.createTexture(.init(width: 4, height: 4, format: .rgba8Uint,
                                                       usage: [.renderAttachment, .copySource]))
    let readback = try await device.createBuffer(.init(size: 1024, usage: [.copyDestination, .mapRead]))
    let streams = [VertexBufferBinding(buffer: positions)]
    let capture: [BufferCommand] = [
        .copyTextureToBuffer(source: target, destination: readback, bytesPerRow: 256),
        .read(buffer: readback, byteCount: 1024),
    ]
    var commands: [BufferCommand] = [
        .write(buffer: positions, offset: 0,
               words: [Float(-1), 0, 0, 0, -1, 1, 0, 0, 1, 0, 1, 1].map(\.bitPattern)),
        .write(buffer: indices, offset: 0, words: [0x00010000, 0xffff0002, 0x00040003, 0x00000005]),
    ]
    for (list, strip) in pipelines {
        commands += [
            .render(pipeline: list, target: target, clear: .transparentBlack, vertexCount: 6,
                    vertexBuffers: streams, instances: .init(count: 2)),
        ] + capture + [
            .renderIndexed(pipeline: strip, target: target, clear: .transparentBlack,
                           indexBuffer: .init(buffer: indices, format: .uint16),
                           draw: .init(indexCount: 7, instances: .init(count: 2)), vertexBuffers: streams),
        ] + capture
    }
    let words = try await device.submitCommands(commands).wait()
    guard words.map(\.count) == [256, 256, 256, 256] else {
        throw GPUError.backendFailure("Unexpected primitive index readback")
    }
    let pixels = words.map { [$0[64], $0[67], $0[192], $0[195]] }
    guard pixels[0] == pixels[2], pixels[1] == pixels[3] else {
        throw GPUError.backendFailure("SPIR-V primitive indices differed from direct WGSL: \(pixels)")
    }
    return pixels.prefix(2).flatMap { $0 }
}

private let invariantPositionShader = """
    struct Raster {
        @location(0) tint: f32,
        @builtin(position) @invariant point: vec4f,
    }

    fn project(value: vec2f) -> vec4f {
        let xy = value * vec2f(0.79, 0.81) + vec2f(0.12, 0.11);
        let z = value.x * 0.123 + value.y * 0.077 + 0.45;
        return vec4f(xy, z, 1.0);
    }

    @vertex fn direct(@location(0) vertex: vec2f) -> @invariant @builtin(position) vec4f {
        return project(vertex);
    }

    @vertex fn structured(@location(0) vertex: vec2f) -> Raster {
        return Raster(1.0, project(vertex));
    }

    @fragment fn fill(@builtin(position) @invariant point: vec4f) -> @location(0) vec4f {
        return vec4f(0.0, select(0.0, 1.0, point.w > 0.0), 0.0, 1.0);
    }

    @fragment fn shade(input: Raster) -> @location(0) vec4f {
        return vec4f(0.0, input.tint, 0.0, 1.0);
    }
    """

private func invariantPositionDiagnostic(_ device: Device) async throws -> [UInt32] {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl(invariantPositionShader)
    let layout = VertexBufferLayout(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float32x2)])
    let direct = try await device.createRenderPipeline(
        vertex: compiler.compileRenderStage(source, entryPoint: "direct", to: .wgsl),
        fragment: compiler.compileRenderStage(source, entryPoint: "fill", to: .wgsl),
        colorFormat: .rgba8Unorm, vertexBuffers: [layout], depthState: .init(compare: .always))
    let structured = try await device.createRenderPipeline(
        vertex: compiler.compileRenderStage(source, entryPoint: "structured", to: .wgsl),
        fragment: compiler.compileRenderStage(source, entryPoint: "shade", to: .wgsl),
        colorFormat: .rgba8Unorm, vertexBuffers: [layout], depthState: .init(compare: .equal, writeEnabled: false))
    let vertices = try await device.createBuffer(.init(size: 24, usage: [.vertex, .copyDestination]))
    let color = try await device.createTexture(.init(width: 4, height: 4, format: .rgba8Unorm,
                                                      usage: [.renderAttachment, .copySource]))
    let depth = try await device.createTexture(.init(width: 4, height: 4, format: .depth32Float, usage: .renderAttachment))
    let readback = try await device.createBuffer(.init(size: 1024, usage: [.copyDestination, .mapRead]))
    let clear = RenderClearColor(red: 0, green: 0, blue: 1, alpha: 1)
    let capture: [BufferCommand] = [
        .copyTextureToBuffer(source: color, destination: readback, bytesPerRow: 256),
        .read(buffer: readback, byteCount: 1024),
    ]
    let ticket = try await device.submitCommands([
        .write(buffer: vertices, offset: 0, words: [Float(-1), -1, 3, -1, -1, 3].map(\.bitPattern)),
        .render(pipeline: direct, target: color, clear: clear, vertexCount: 3, vertexBuffers: [.init(buffer: vertices)],
                depth: .init(texture: depth, loadAction: .clear(1))),
    ] + capture + [
        .render(pipeline: structured, target: color, clear: clear, vertexCount: 3, vertexBuffers: [.init(buffer: vertices)],
                depth: .init(texture: depth, loadAction: .load)),
    ] + capture)
    let snapshots = try await ticket.wait()
    guard snapshots.map(\.count) == [256, 256] else { throw GPUError.backendFailure("Unexpected invariant raster readback") }
    return snapshots.flatMap { snapshot in
        (0..<4).flatMap { row in Array(snapshot[(row * 64)..<(row * 64 + 4)]) }
    }
}

private let voidFragmentShader = """
    @vertex fn vs(@location(0) point: vec2f) -> @builtin(position) vec4f {
        return vec4f(point, 0.5, 1.0);
    }

    @fragment fn depth_only(@invariant @builtin(position) point: vec4f) {
        if point.x < 1.0 { discard; }
        if point.y < 1.0 { return; }
    }

    @fragment fn shade() -> @location(0) vec4f { return vec4f(0.0, 1.0, 0.0, 1.0); }
    """

private func voidFragmentDiagnostic(_ device: Device) async throws -> [UInt32] {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl(voidFragmentShader)
    let layout = VertexBufferLayout(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float32x2)])
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .wgsl)
    let depthFragment = try compiler.compileRenderStage(source, entryPoint: "depth_only", to: .wgsl)
    let shade = try compiler.compileRenderStage(source, entryPoint: "shade", to: .wgsl)
    let write = try await device.createRenderPipeline(vertex: vertex, fragment: depthFragment, colorFormat: .rgba8Unorm,
        vertexBuffers: [layout], depthState: .init(compare: .always))
    let masked = try await device.createRenderPipeline(vertex: vertex, fragment: depthFragment, colorFormat: .rgba8Unorm,
        vertexBuffers: [layout], depthState: .init(compare: .always), sampleMask: 0)
    let testDepth = try await device.createRenderPipeline(vertex: vertex, fragment: shade, colorFormat: .rgba8Unorm,
        vertexBuffers: [layout], depthState: .init(compare: .equal, writeEnabled: false))
    let vertices = try await device.createBuffer(.init(size: 24, usage: [.vertex, .copyDestination]))
    let target = try await device.createTexture(.init(width: 2, height: 2, format: .rgba8Unorm,
                                                       usage: [.renderAttachment, .copySource]))
    let depth = try await device.createTexture(.init(width: 2, height: 2, format: .depth32Float, usage: .renderAttachment))
    let readback = try await device.createBuffer(.init(size: 512, usage: [.copyDestination, .mapRead]))
    let clear = RenderClearColor(red: 0, green: 0, blue: 1, alpha: 1)
    let capture: [BufferCommand] = [
        .copyTextureToBuffer(source: target, destination: readback, bytesPerRow: 256),
        .read(buffer: readback, byteCount: 512),
    ]
    var commands: [BufferCommand] = [
        .write(buffer: vertices, offset: 0, words: [Float(-1), -1, 3, -1, -1, 3].map(\.bitPattern)),
    ]
    for pipeline in [write, masked] {
        commands.append(.render(pipeline: pipeline, target: target, clear: clear, vertexCount: 3,
                                vertexBuffers: [.init(buffer: vertices)], depth: .init(texture: depth, loadAction: .clear(1))))
        commands += capture
        commands.append(.render(pipeline: testDepth, target: target, colorLoad: .load, vertexCount: 3,
                                vertexBuffers: [.init(buffer: vertices)], depth: .init(texture: depth, loadAction: .load)))
        commands += capture
    }
    let snapshots = try await device.submitCommands(commands).wait()
    guard snapshots.map(\.count) == [128, 128, 128, 128] else { throw GPUError.backendFailure("Unexpected depth-only readback") }
    return snapshots.flatMap { [$0[0], $0[1], $0[64], $0[65]] }
}

private let discardContinuationShader = """
    @group(0) @binding(0) var<storage, read_write> counts: array<atomic<u32>, 4>;

    fn record_and_discard(lane: u32, x: f32) -> f32 {
        atomicAdd(&counts[lane], 1u);
        discard;
        atomicAdd(&counts[lane], 4u);
        return x * 4.0;
    }

    @vertex fn vs(@location(0) point: vec2f) -> @builtin(position) vec4f {
        return vec4f(point, 0.5, 1.0);
    }

    @fragment fn fs(@builtin(position) point: vec4f) -> @location(0) vec4f {
        let lane = u32(point.y) * 2u + u32(point.x);
        atomicAdd(&counts[lane], 2u);
        var derivativeInput = point.x * 2.0;
        if point.x < 1.0 { derivativeInput = record_and_discard(lane, point.x); }
        let delta = dpdxFine(derivativeInput);
        atomicAdd(&counts[lane], 7u + u32(delta));
        return vec4f(0.0, delta, 0.0, 1.0);
    }
    """

private func discardContinuationDiagnostic(_ device: Device) async throws -> [UInt32] {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl(discardContinuationShader)
    let layout = VertexBufferLayout(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float32x2)])
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .wgsl)
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: .wgsl)
    let pipeline = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm,
        vertexBuffers: [layout])
    guard case .spirv(let binary) = try compiler.compile(source, to: .spirv) else {
        throw GPUError.backendFailure("Expected binary SPIR-V for fragment discard continuation")
    }
    let roundTrip = ShaderInput.spirv(binary)
    let roundTripPipeline = try await device.createRenderPipeline(
        vertex: compiler.compileWebGPURenderStage(roundTrip, entryPoint: "vs"),
        fragment: compiler.compileWebGPURenderStage(roundTrip, entryPoint: "fs"),
        colorFormat: .rgba8Unorm, vertexBuffers: [layout])
    let vertices = try await device.createBuffer(.init(size: 24, usage: [.vertex, .copyDestination]))
    let counts = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource, .copyDestination]))
    let color = try await device.createTexture(.init(width: 2, height: 2, format: .rgba8Unorm,
                                                      usage: [.renderAttachment, .copySource]))
    let readback = try await device.createBuffer(.init(size: 512, usage: [.copyDestination, .mapRead]))
    let clear = RenderClearColor(red: 0, green: 0, blue: 1, alpha: 1)
    var commands: [BufferCommand] = [
        .write(buffer: vertices, offset: 0, words: [Float(-1), -1, 3, -1, -1, 3].map(\.bitPattern)),
    ]
    for render in [pipeline, roundTripPipeline] {
        commands += [
            .write(buffer: counts, offset: 0, words: [0, 0, 0, 0]),
            .render(pipeline: render, target: color, clear: clear, vertexCount: 3,
                    vertexBuffers: [.init(buffer: vertices)], storageBuffers: [.init(binding: 0): .init(buffer: counts)]),
            .copyTextureToBuffer(source: color, destination: readback, bytesPerRow: 256),
            .read(buffer: readback, byteCount: 512),
            .copy(source: counts, destination: readback, byteCount: 16),
            .read(buffer: readback, byteCount: 16),
        ]
    }
    let result = try await device.submitCommands(commands).wait()
    guard result.map(\.count) == [128, 4, 128, 4] else {
        throw GPUError.backendFailure("Unexpected discard readback")
    }
    let pixelOffsets = [0, 1, 64, 65]
    let direct = pixelOffsets.map { result[0][$0] } + result[1]
    let imported = pixelOffsets.map { result[2][$0] } + result[3]
    guard imported == direct else {
        throw GPUError.backendFailure("SPIR-V discard readback \(imported) differs from direct WGSL \(direct)")
    }
    return imported
}

private let depthClipControlShader = """
    @vertex fn vs(@location(0) point: vec4f) -> @builtin(position) vec4f { return point; }
    @fragment fn fs() -> @location(0) vec4f { return vec4f(0.0, 1.0, 0.0, 1.0); }
    """

private func depthClipControlDiagnostic(_ device: Device) async throws -> [UInt32] {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl(depthClipControlShader)
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .wgsl)
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: .wgsl)
    let layout = VertexBufferLayout(stride: 16, attributes: [.init(location: 0, offset: 0, format: .float32x4)])
    let depthState = RenderDepthState(compare: .always)
    if !device.capabilities.features.contains(.depthClipControl) {
        do {
            _ = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm,
                vertexBuffers: [layout], depthState: depthState, primitive: .init(unclippedDepth: true))
        } catch GPUError.validation("Feature not advertised") {
            return []
        }
        throw GPUError.backendFailure("Pipeline accepted unclippedDepth without the device feature")
    }
    let clip = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm,
                                                      vertexBuffers: [layout], depthState: depthState)
    let clamp = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm,
        vertexBuffers: [layout], depthState: depthState, primitive: .init(unclippedDepth: true))
    let points: [Float] = [
        -1, -1, -1, 1,   3, -1, -1, 1,   -1, 3, -1, 1,
        -1, -1,  2, 1,   3, -1,  2, 1,   -1, 3,  2, 1,
         2, -1, -1, 1,   4, -1, -1, 1,    2, 3, -1, 1,
        -1, -1, -1, 1,   1, -1, 1.5, 1,   0, 1, 0.5, 1,
    ]
    let vertices = try await device.createBuffer(.init(size: UInt64(points.count * 4), usage: [.vertex, .copyDestination]))
    let color = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm,
                                                      usage: [.renderAttachment, .copySource]))
    let depth = try await device.createTexture(.init(width: 1, height: 1, format: .depth32Float,
                                                      usage: [.renderAttachment, .copySource]))
    let readback = try await device.createBuffer(.init(size: 256, usage: [.copyDestination, .mapRead]))
    let clear = RenderClearColor(red: 0, green: 0, blue: 1, alpha: 1)
    let raster = RenderRasterState(viewport: .init(x: 0, y: 0, width: 1, height: 1, minDepth: 0.25, maxDepth: 0.75))
    let draws: [(RenderPipeline, UInt32)] = [(clip, 0), (clamp, 0), (clamp, 3), (clip, 3), (clamp, 6), (clamp, 9)]
    let capture: [BufferCommand] = [
        .copyTextureToBuffer(source: color, destination: readback, bytesPerRow: 256),
        .read(buffer: readback, byteCount: 4),
        .copyTextureToBuffer(source: depth, destination: readback, bytesPerRow: 256),
        .read(buffer: readback, byteCount: 4),
    ]
    var commands: [BufferCommand] = [.write(buffer: vertices, offset: 0, words: points.map(\.bitPattern))]
    for (pipeline, firstVertex) in draws {
        commands.append(.render(pipeline: pipeline, target: color, clear: clear, vertexCount: 3,
                                vertexBuffers: [.init(buffer: vertices)], firstVertex: firstVertex,
                                depth: .init(texture: depth, loadAction: .clear(0.5)), raster: raster))
        commands += capture
    }
    return try await device.submitCommands(commands).wait().flatMap { $0 }
}

private let timestampCompute = """
    @group(0) @binding(0) var<storage, read_write> result: array<u32, 64>;
    @compute @workgroup_size(64) fn run(@builtin(local_invocation_index) lane: u32) {
        result[lane] = lane * 7u;
    }
    """

private let timestampRender = """
    @vertex fn vs(@location(0) point: vec2f) -> @builtin(position) vec4f { return vec4f(point, 0.5, 1.0); }
    @fragment fn fs() -> @location(0) vec4f { return vec4f(0.0, 1.0, 0.0, 1.0); }
    """

private func timestampQueryDiagnostic(_ device: Device) async throws -> [UInt32] {
    if !device.capabilities.features.contains(.timestampQuery) {
        do {
            _ = try await device.createQuerySet(.init(type: .timestamp, count: 1))
        } catch GPUError.validation("Feature not advertised") {
            return []
        }
        throw GPUError.backendFailure("Query set accepted without the timestamp feature")
    }
    let compiler = IndependentShaderCompiler()
    let compute = try await device.createComputePipeline(compiler.compileCompute(.wgsl(timestampCompute), to: .wgsl))
    let source = ShaderInput.wgsl(timestampRender)
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .wgsl)
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: .wgsl)
    let render = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm,
        vertexBuffers: [.init(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float32x2)])])
    let query = try await device.createQuerySet(.init(type: .timestamp, count: 6))
    let result = try await device.createBuffer(.init(size: 304, usage: [.queryResolve, .copyDestination, .copySource]))
    let readback = try await device.createBuffer(.init(size: 304, usage: [.copyDestination, .mapRead]))
    let output = try await device.createBuffer(.init(size: 256, usage: .storage))
    let positions = try await device.createBuffer(.init(size: 24, usage: [.vertex, .copyDestination]))
    let target = try await device.createTexture(.init(width: 2, height: 2, format: .rgba8Unorm, usage: .renderAttachment))
    let capture: [BufferCommand] = [
        .resolveQuerySet(querySet: query, firstQuery: 0, queryCount: 6, destination: result, destinationOffset: 256),
        .copy(source: result, destination: readback, byteCount: 304),
        .read(buffer: readback, byteCount: 304),
    ]
    let initialize: [BufferCommand] = [
        .write(buffer: result, offset: 0, words: [0x12345678]),
        .write(buffer: result, offset: 256, words: Array(repeating: 0xffffffff, count: 12)),
        .write(buffer: positions, offset: 0, words: [Float(-1), -1, 3, -1, -1, 3].map(\.bitPattern)),
    ]
    let computePass = BufferCommand.dispatchCompute(pipeline: compute, bindings: [.init(binding: 0): output], workgroups: .init(x: 1))
    let renderPass = BufferCommand.render(pipeline: render, target: target, clear: .transparentBlack,
                                          vertexCount: 3, vertexBuffers: [.init(buffer: positions)])
    let timestamps: [BufferCommand] = [
        .timestamped(pass: computePass, writes: .init(querySet: query, beginningOfPassWriteIndex: 1, endOfPassWriteIndex: 2)),
        .timestamped(pass: renderPass, writes: .init(querySet: query, beginningOfPassWriteIndex: 4, endOfPassWriteIndex: 5)),
    ]
    let snapshots = try await device.submitCommands(initialize + capture + timestamps + capture).wait()
    let repeated = try await device.submitCommands(capture).wait()
    let emptyCompute = BufferCommand.dispatchCompute(pipeline: compute, bindings: [.init(binding: 0): output],
                                                      workgroups: .init(x: 0))
    let emptyRender = BufferCommand.render(pipeline: render, target: target, clear: .transparentBlack,
                                           vertexCount: 0, vertexBuffers: [.init(buffer: positions)])
    let emptyPasses: [BufferCommand] = [
        .timestamped(pass: emptyCompute, writes: .init(querySet: query, beginningOfPassWriteIndex: 0)),
        .timestamped(pass: emptyRender, writes: .init(querySet: query, endOfPassWriteIndex: 3)),
    ]
    let emptySnapshots = try await device.submitCommands(emptyPasses + capture).wait()
    let arguments = try await device.createBuffer(.init(size: 32, usage: [.indirect, .copyDestination]))
    let indirectCompute = BufferCommand.dispatchComputeIndirect(pipeline: compute,
        bindings: [.init(binding: 0): .init(buffer: output)], indirectBuffer: arguments)
    let indirectRender = BufferCommand.renderIndirect(pipeline: render, target: target, colorLoad: .clear(.transparentBlack),
        indirectBuffer: arguments, indirectOffset: 16, vertexBuffers: [.init(buffer: positions)])
    let indirectPasses: [BufferCommand] = [
        .write(buffer: arguments, offset: 0, words: [0, 1, 1, 0, 0, 1, 0, 0]),
        .timestamped(pass: indirectCompute, writes: .init(querySet: query, beginningOfPassWriteIndex: 0)),
        .timestamped(pass: indirectRender, writes: .init(querySet: query, endOfPassWriteIndex: 3)),
    ]
    let indirectSnapshots = try await device.submitCommands(indirectPasses + capture).wait()
    let all = snapshots + repeated + emptySnapshots + indirectSnapshots
    guard all.map(\.count) == [76, 76, 76, 76, 76] else { throw GPUError.backendFailure("Unexpected timestamp readback") }
    return all.flatMap { [$0[0]] + Array($0[64..<76]) }
}

private let occlusionQueryShader = """
    @vertex fn vs(@location(0) point: vec2f) -> @builtin(position) vec4f {
        return vec4f(point, 0.5, 1.0);
    }
    @fragment fn fs() -> @location(0) vec4f { return vec4f(0.0, 1.0, 0.0, 1.0); }
    """

private func occlusionQueryDiagnostic(_ device: Device) async throws -> [UInt32] {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl(occlusionQueryShader)
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .wgsl)
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: .wgsl)
    let render = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm,
        vertexBuffers: [.init(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float32x2)])],
        depthState: .init(compare: .less))
    let query = try await device.createQuerySet(.init(type: .occlusion, count: 4))
    let result = try await device.createBuffer(.init(size: 288, usage: [.queryResolve, .copyDestination, .copySource]))
    let readback = try await device.createBuffer(.init(size: 288, usage: [.copyDestination, .mapRead]))
    let vertices = try await device.createBuffer(.init(size: 24, usage: [.vertex, .copyDestination]))
    let indices = try await device.createBuffer(.init(size: 8, usage: [.index, .copyDestination]))
    let indirect = try await device.createBuffer(.init(size: 16, usage: [.indirect, .copyDestination]))
    let color = try await device.createTexture(.init(width: 2, height: 2, format: .rgba8Unorm, usage: .renderAttachment))
    let depth = try await device.createTexture(.init(width: 2, height: 2, format: .depth32Float, usage: .renderAttachment))
    let clear = RenderColorLoadAction.clear(.transparentBlack)
    let streams = [VertexBufferBinding(buffer: vertices)]
    let depthClear = RenderDepthAttachment(texture: depth, loadAction: .clear(1))
    let depthLoad = RenderDepthAttachment(texture: depth, loadAction: .load)
    let visible = BufferCommand.render(pipeline: render, target: color, colorLoad: clear, vertexCount: 3,
                                       vertexBuffers: streams, depth: depthClear)
    let occluded = BufferCommand.renderIndexed(pipeline: render, target: color, colorLoad: clear,
        indexBuffer: .init(buffer: indices, format: .uint16), draw: .init(indexCount: 3), vertexBuffers: streams, depth: depthLoad)
    let indirectDraw = BufferCommand.renderIndirect(pipeline: render, target: color, colorLoad: clear,
        indirectBuffer: indirect, vertexBuffers: streams, depth: depthClear)
    let empty = BufferCommand.render(pipeline: render, target: color, colorLoad: clear, vertexCount: 0,
                                     vertexBuffers: streams, depth: depthClear)
    let capture: [BufferCommand] = [
        .resolveQuerySet(querySet: query, firstQuery: 4, queryCount: 0, destination: result, destinationOffset: 256),
        .resolveQuerySet(querySet: query, firstQuery: 0, queryCount: 4, destination: result, destinationOffset: 256),
        .copy(source: result, destination: readback, byteCount: 288),
        .read(buffer: readback, byteCount: 288),
    ]
    let initial: [BufferCommand] = [
        .write(buffer: result, offset: 0, words: [0x11223344]),
        .write(buffer: result, offset: 256, words: Array(repeating: 0xffffffff, count: 8)),
        .write(buffer: vertices, offset: 0, words: [Float(-1), -1, 3, -1, -1, 3].map(\.bitPattern)),
        .write(buffer: indices, offset: 0, words: [0x00010000, 2]),
        .write(buffer: indirect, offset: 0, words: [3, 1, 0, 0]),
    ]
    var queries: [BufferCommand] = [
        .occlusionQueried(pass: visible, query: .init(querySet: query, queryIndex: 1)),
        .occlusionQueried(pass: occluded, query: .init(querySet: query, queryIndex: 2)),
        .occlusionQueried(pass: indirectDraw, query: .init(querySet: query, queryIndex: 3)),
    ]
    var captureTimestamp: [BufferCommand] = []
    if device.capabilities.features.contains(.timestampQuery) {
        let timestamps = try await device.createQuerySet(.init(type: .timestamp, count: 1))
        queries[0] = .occlusionQueried(
            pass: .timestamped(pass: visible, writes: .init(querySet: timestamps, endOfPassWriteIndex: 0)),
            query: .init(querySet: query, queryIndex: 1))
        captureTimestamp = [
            .resolveQuerySet(querySet: timestamps, firstQuery: 0, queryCount: 1, destination: result, destinationOffset: 256),
            .copy(source: result, destination: readback, byteCount: 264),
            .read(buffer: readback, byteCount: 264),
        ]
    }
    let overwrite = BufferCommand.occlusionQueried(pass: empty, query: .init(querySet: query, queryIndex: 1))
    let commands = initial + capture + queries + capture + [overwrite] + capture + captureTimestamp
    let snapshots = try await device.submitCommands(commands).wait()
    let repeated = try await device.submitCommands(capture).wait()
    let occlusionSnapshots = Array(snapshots.prefix(3)) + repeated
    guard occlusionSnapshots.map(\.count) == [72, 72, 72, 72] else {
        throw GPUError.backendFailure("Unexpected occlusion readback")
    }
    var words = occlusionSnapshots.flatMap { [$0[0]] + Array($0[64..<72]) }
    if !captureTimestamp.isEmpty {
        guard snapshots.count == 4, let timestamp = snapshots.last, timestamp.count == 66 else {
            throw GPUError.backendFailure("Unexpected timestamp readback from an occlusion pass")
        }
        words += timestamp[64..<66]
    }
    return words
}


private func mappedBufferDiagnostic(_ device: Device) async throws -> [UInt32] {
    let upload = try await device.createBuffer(.init(size: 16, usage: [.mapWrite, .copySource], mappedAtCreation: true))
    let readback = try await device.createBuffer(.init(size: 16, usage: [.mapRead, .copyDestination]))
    let initial = try await device.mappedBuffer(upload)
    guard try await device.readMappedBytes(initial) == Array(repeating: 0, count: 16) else {
        throw GPUError.backendFailure("Unexpected initial mapped bytes")
    }
    try await device.writeMappedBytes(initial, offset: 1, bytes: [1, 2, 3, 4, 5, 6, 7, 8])
    do {
        _ = try await device.submitCommands([.copy(source: upload, destination: readback, byteCount: 16)])
        throw GPUError.backendFailure("Mapped buffer was accepted by the GPU queue")
    } catch GPUError.validation("Buffer is mapped or has a pending mapping") {}
    try await device.unmapBuffer(upload)
    let second = try await device.mapBuffer(upload, mode: .write, offset: 8, size: 8)
    guard try await device.readMappedBytes(second) == [8, 0, 0, 0, 0, 0, 0, 0] else {
        throw GPUError.backendFailure("Mapped writes did not persist")
    }
    try await device.writeMappedBytes(second, offset: 9, bytes: [9, 10, 11])
    try await device.unmapBuffer(upload)

    _ = try await device.submitCommands([.copy(source: upload, destination: readback, byteCount: 16)])
    let result = try await device.mapBuffer(readback, mode: .read, offset: 8)
    let tail = try await device.readMappedBytes(result)
    let subrange = try await device.readMappedBytes(result, offset: 9, size: 3)
    try await device.unmapBuffer(readback)

    let initialized = try await device.createBuffer(.init(size: 4, usage: .copySource, mappedAtCreation: true))
    let creation = try await device.mappedBuffer(initialized)
    try await device.writeMappedBytes(creation, offset: 0, bytes: [21, 22, 23, 24])
    try await device.unmapBuffer(initialized)
    _ = try await device.submitCommands([.copy(source: initialized, destination: readback, byteCount: 4)])
    let prefix = try await device.mapBuffer(readback, mode: .read, size: 4)
    let prefixBytes = try await device.readMappedBytes(prefix)
    try await device.unmapBuffer(readback)

    let empty = try await device.mapBuffer(readback, mode: .read, offset: 16, size: 0)
    guard try await device.readMappedBytes(empty).isEmpty else {
        throw GPUError.backendFailure("Unexpected empty mapped range")
    }
    try await device.unmapBuffer(readback)

    let largeSize = BrowserRPC.packetLimit + 8
    let large = try await device.createBuffer(.init(size: UInt64(largeSize), usage: .mapWrite, mappedAtCreation: true))
    let largeMapping = try await device.mappedBuffer(large)
    var payload = [UInt8](repeating: 0x5a, count: largeSize)
    payload[0] = 1
    payload[largeSize - 1] = 2
    try await device.writeMappedBytes(largeMapping, offset: 0, bytes: payload)
    let echoed = try await device.readMappedBytes(largeMapping)
    guard echoed == payload else { throw GPUError.backendFailure("Chunked browser mapped transfer differs") }
    try await device.unmapBuffer(large)
    try await device.destroyBuffer(large)
    return (tail + subrange + prefixBytes).map(UInt32.init)
}

private func bufferRangeDiagnostic(_ device: Device) async throws -> [UInt32] {
    let source = try await device.createBuffer(.init(size: 20, usage: [.copySource, .copyDestination]))
    let destination = try await device.createBuffer(.init(size: 24, usage: [.copySource, .copyDestination]))
    let readback = try await device.createBuffer(.init(size: 24, usage: [.mapRead, .copyDestination]))
    let payload = [UInt8.max] + Array(UInt8(1)...UInt8(16)) + [UInt8.max]
    _ = try await device.writeBuffer(source, offset: 4, bytes: payload, dataOffset: 1, byteCount: 16)
    _ = try await device.submitCommands([
        .writeBytes(buffer: destination, offset: 0, bytes: Array(repeating: 0xee, count: 24)),
        .copy(source: source, sourceOffset: 8, destination: destination, destinationOffset: 4, byteCount: 8),
        .clearBuffer(buffer: destination, offset: 8, byteCount: 4),
        .copy(source: source, sourceOffset: 16, destination: destination, destinationOffset: 16),
        .clearBuffer(buffer: destination, offset: 20),
        .copy(source: source, sourceOffset: 20, destination: destination, destinationOffset: 24, byteCount: 0),
        .clearBuffer(buffer: destination, offset: 24),
        .writeBytes(buffer: destination, offset: 24, bytes: []),
        .copy(source: destination, destination: readback),
    ])
    let mapping = try await device.mapBuffer(readback, mode: .read)
    let bytes = try await device.readMappedBytes(mapping)
    try await device.unmapBuffer(readback)

    let largeSize = BrowserRPC.packetLimit + 8
    let large = try await device.createBuffer(.init(size: UInt64(largeSize), usage: [.copySource, .copyDestination]))
    let probe = try await device.createBuffer(.init(size: 136, usage: [.mapRead, .copyDestination]))
    let middle = Array(UInt8(0)...UInt8(127))
    let tail: [UInt8] = [200, 201, 202, 203, 204, 205, 206, 207]
    var bulk = [UInt8](repeating: 0x5a, count: largeSize)
    bulk.replaceSubrange((BrowserRPC.packetLimit - middle.count)..<BrowserRPC.packetLimit, with: middle)
    bulk.replaceSubrange(BrowserRPC.packetLimit..<largeSize, with: tail)
    _ = try await device.writeBuffer(large, bytes: bulk)
    _ = try await device.submitCommands([
        .copy(source: large, sourceOffset: UInt64(BrowserRPC.packetLimit - middle.count), destination: probe),
    ])
    let probeMapping = try await device.mapBuffer(probe, mode: .read)
    guard try await device.readMappedBytes(probeMapping) == middle + tail else {
        throw GPUError.backendFailure("Staged browser upload lost bytes near the packet boundary")
    }
    try await device.unmapBuffer(probe)
    try await device.destroyBuffer(large)
    try await device.destroyBuffer(probe)
    return bytes.map(UInt32.init)
}

private let uploadTexels: [UInt32] = [0xff112233, 0xff445566, 0xff778899, 0xffaabbcc]

private func paddedTextureBytes() -> [UInt8] {
    var bytes = Array(repeating: UInt8(0xee), count: 25)
    for (index, offset) in [1, 6, 16, 21].enumerated() {
        let word = uploadTexels[index]
        for lane in 0..<4 { bytes[offset + lane] = UInt8(truncatingIfNeeded: word >> (lane * 8)) }
    }
    return bytes
}

private func queueTextureDiagnostic(_ device: Device) async throws -> [UInt32] {
    let array = try await device.createTexture(.init(width: 4, height: 4, format: .rgba8Unorm,
        usage: [.copySource, .copyDestination], mipLevelCount: 2, arrayLayerCount: 3))
    let volume = try await device.createTexture(.init(width: 2, height: 2, format: .rgba8Unorm,
        usage: [.copySource, .copyDestination], dimension: .d3, depth: 3))
    let stencil = try await device.createTexture(.init(width: 3, height: 2, format: .depth24PlusStencil8,
        usage: [.copySource, .copyDestination]))
    let readback = try await device.createBuffer(.init(size: 784, usage: [.mapRead, .copyDestination]))
    let bytes = paddedTextureBytes()
    let layout = TextureDataLayout(offset: 1, bytesPerRow: 5, rowsPerImage: 3)
    _ = try await device.writeTexture(volume, data: bytes, layout: layout,
        region: .init(x: 1, width: 1, height: 2, z: 1, depth: 2))
    func capture(_ texture: ResourceID, mipLevel: UInt32 = 0, layer: UInt32 = 0, z: UInt32 = 0) -> [BufferCommand] {
        [
            .copyTextureToBuffer(source: texture, destination: readback, bytesPerRow: 256,
                                 mipLevel: mipLevel, arrayLayer: layer, z: z),
            .read(buffer: readback, byteCount: mipLevel == 0 && texture == array ? 784 : 264),
        ]
    }
    let upload = BufferCommand.writeTexture(destination: array, data: bytes, layout: layout,
        region: .init(x: 1, width: 1, height: 2, mipLevel: 1, arrayLayer: 1, depth: 2))
    var commands = capture(array, mipLevel: 1, layer: 1) + [upload]
    for layer: UInt32 in [0, 1, 2] { commands += capture(array, mipLevel: 1, layer: layer) }
    for plane: UInt32 in [0, 1, 2] { commands += capture(volume, z: plane) }
    commands += capture(array, layer: 1)
    commands += [
        .writeTexture(destination: stencil, data: [0xee, 1, 128, 240, 0xee, 0xee, 17, 7, 165],
                      layout: .init(offset: 1, bytesPerRow: 5), region: .init(width: 3, height: 2), aspect: .stencilOnly),
        .copyTextureToBuffer(source: stencil, destination: readback, bytesPerRow: 256, aspect: .stencilOnly),
        .read(buffer: readback, byteCount: 260),
    ]
    let ticket = try await device.submitCommands(commands)
    let snapshots = try await ticket.wait()
    guard snapshots.map(\.count) == [66, 66, 66, 66, 66, 66, 66, 196, 65],
          snapshots[7].allSatisfy({ $0 == 0 }) else {
        throw GPUError.backendFailure("Unexpected queue texture upload readback")
    }
    return snapshots.prefix(7).flatMap { [$0[0], $0[1], $0[64], $0[65]] } +
        [snapshots[8][0] & 0x00ffffff, snapshots[8][64] & 0x00ffffff]
}

private let subgroupGatherShader = """
    enable subgroups;
    struct Inputs {
        @builtin(subgroup_invocation_id) lane: u32,
        @builtin(subgroup_size) size: u32,
    }
    @group(0) @binding(0) var<storage, read_write> out: array<vec4u, 256>;
    fn paired(lane: u32) -> u32 { return subgroupShuffle(lane, lane ^ 1u); }
    @compute @workgroup_size(128) fn run(input: Inputs, @builtin(local_invocation_index) index: u32) {
        let lane = input.lane;
        let size = input.size;
        let down = select(lane, subgroupShuffleDown(lane, 1u), lane + 1u < size);
        let up = select(lane, subgroupShuffleUp(lane, 1u), lane != 0u);
        out[index * 2u] = vec4u(lane, size, subgroupBroadcast(lane, 0u), subgroupBroadcastFirst(lane));
        out[index * 2u + 1u] = vec4u(paired(lane), subgroupShuffleXor(lane, size - 1u), down, up);
    }
    """

private let fragmentSubgroupShader = """
    enable subgroups;
    @vertex fn vs(@location(0) point: vec2f) -> @builtin(position) vec4f { return vec4f(point, 0.5, 1.0); }
    @fragment fn fs(@builtin(subgroup_size) size: u32, @builtin(subgroup_invocation_id) lane: u32) -> @location(0) vec4u {
        return vec4u(size, lane, subgroupBroadcastFirst(lane), 255u);
    }
    """


// Subgroup shuffle, ballot and reduction with four active compute lanes.
private let spirvSubgroupWords: [UInt32] = [
    0x07230203, 0x00010300, 0x00000000, 0x00000064, 0x00000000, 0x00020011, 0x00000001, 0x00020011,
    0x0000003d, 0x00020011, 0x0000003e, 0x00020011, 0x0000003f, 0x00020011, 0x00000040, 0x00020011,
    0x00000042, 0x00020011, 0x00000044, 0x00020011, 0x00000041, 0x0003000e, 0x00000000, 0x00000001,
    0x0006000f, 0x00000005, 0x00000014, 0x6e69616d, 0x00000000, 0x00000012, 0x00060010, 0x00000014,
    0x00000011, 0x00000004, 0x00000001, 0x00000001, 0x00030047, 0x00000012, 0x00000000, 0x00030047,
    0x00000028, 0x00000000, 0x00040047, 0x00000012, 0x0000000b, 0x00000029, 0x00040047, 0x00000005,
    0x00000006, 0x00000010, 0x00030047, 0x00000006, 0x00000002, 0x00050048, 0x00000006, 0x00000000,
    0x00000023, 0x00000000, 0x00040047, 0x00000011, 0x00000022, 0x00000000, 0x00040047, 0x00000011,
    0x00000021, 0x00000000, 0x00020013, 0x00000001, 0x00040015, 0x00000002, 0x00000020, 0x00000000,
    0x00020014, 0x00000003, 0x00040017, 0x00000004, 0x00000002, 0x00000004, 0x0004002b, 0x00000002,
    0x0000001e, 0x00000000, 0x0004002b, 0x00000002, 0x0000001f, 0x00000001, 0x0004002b, 0x00000002,
    0x00000020, 0x00000002, 0x0004002b, 0x00000002, 0x00000021, 0x00000003, 0x0004002b, 0x00000002,
    0x00000022, 0x00000004, 0x0004002b, 0x00000002, 0x00000023, 0x00000064, 0x0004001c, 0x00000005,
    0x00000004, 0x00000022, 0x0003001e, 0x00000006, 0x00000005, 0x00040020, 0x00000007, 0x0000000c,
    0x00000006, 0x00040020, 0x00000008, 0x0000000c, 0x00000004, 0x00040020, 0x00000009, 0x00000001,
    0x00000002, 0x00030021, 0x0000000a, 0x00000001, 0x0004003b, 0x00000007, 0x00000011, 0x0000000c,
    0x0004003b, 0x00000009, 0x00000012, 0x00000001, 0x00050036, 0x00000001, 0x00000014, 0x00000000,
    0x0000000a, 0x000200f8, 0x00000015, 0x0004003d, 0x00000002, 0x00000028, 0x00000012, 0x00050080,
    0x00000002, 0x00000029, 0x00000028, 0x00000023, 0x0006015a, 0x00000002, 0x0000002a, 0x00000021,
    0x00000029, 0x0000001f, 0x000500b0, 0x00000003, 0x0000002b, 0x00000028, 0x00000020, 0x00050153,
    0x00000004, 0x0000002c, 0x00000021, 0x0000002b, 0x00050051, 0x00000002, 0x0000002d, 0x0000002c,
    0x00000000, 0x0006015d, 0x00000002, 0x0000002e, 0x00000021, 0x00000000, 0x00000029, 0x00070050,
    0x00000004, 0x0000002f, 0x0000002a, 0x0000002d, 0x0000002e, 0x00000028, 0x00060041, 0x00000008,
    0x00000030, 0x00000011, 0x0000001e, 0x00000028, 0x0003003e, 0x00000030, 0x0000002f, 0x000100fd,
    0x00010038,
]

// SPIR-V 1.5 broadcast and quad broadcast with selectors derived from the workgroup ID.
private let spirvDynamicBroadcastWords: [UInt32] = [
    0x07230203, 0x00010500, 0x00000000, 0x00000064, 0x00000000, 0x00020011, 0x00000001, 0x00020011,
    0x0000003d, 0x00020011, 0x00000040, 0x00020011, 0x00000044, 0x0003000e, 0x00000000, 0x00000001,
    0x0008000f, 0x00000005, 0x00000014, 0x6e69616d, 0x00000000, 0x00000012, 0x00000013, 0x00000011,
    0x00060010, 0x00000014, 0x00000011, 0x00000004, 0x00000001, 0x00000001, 0x00040047, 0x00000012,
    0x0000000b, 0x00000029, 0x00040047, 0x00000013, 0x0000000b, 0x0000001a, 0x00040047, 0x00000005,
    0x00000006, 0x00000010, 0x00030047, 0x00000006, 0x00000002, 0x00050048, 0x00000006, 0x00000000,
    0x00000023, 0x00000000, 0x00040047, 0x00000011, 0x00000022, 0x00000000, 0x00040047, 0x00000011,
    0x00000021, 0x00000000, 0x00020013, 0x00000001, 0x00040015, 0x00000002, 0x00000020, 0x00000000,
    0x00040017, 0x00000003, 0x00000002, 0x00000003, 0x00040017, 0x00000004, 0x00000002, 0x00000004,
    0x00040017, 0x0000000c, 0x00000002, 0x00000002, 0x0004002b, 0x00000002, 0x0000001e, 0x00000000,
    0x0004002b, 0x00000002, 0x0000001f, 0x00000001, 0x0004002b, 0x00000002, 0x00000021, 0x00000003,
    0x0004002b, 0x00000002, 0x00000022, 0x00000004, 0x0004002b, 0x00000002, 0x00000023, 0x00000064,
    0x0004002b, 0x00000002, 0x00000024, 0x000003e8, 0x0004002b, 0x00000002, 0x00000025, 0x00000008,
    0x0004001c, 0x00000005, 0x00000004, 0x00000025, 0x0003001e, 0x00000006, 0x00000005, 0x00040020,
    0x00000007, 0x0000000c, 0x00000006, 0x00040020, 0x00000008, 0x0000000c, 0x00000004, 0x00040020,
    0x00000009, 0x00000001, 0x00000002, 0x00040020, 0x0000000b, 0x00000001, 0x00000003, 0x00030021,
    0x0000000a, 0x00000001, 0x0004003b, 0x00000007, 0x00000011, 0x0000000c, 0x0004003b, 0x00000009,
    0x00000012, 0x00000001, 0x0004003b, 0x0000000b, 0x00000013, 0x00000001, 0x00050036, 0x00000001,
    0x00000014, 0x00000000, 0x0000000a, 0x000200f8, 0x00000015, 0x0004003d, 0x00000002, 0x00000028,
    0x00000012, 0x00050080, 0x00000002, 0x00000029, 0x00000028, 0x00000023, 0x00050080, 0x00000002,
    0x0000002a, 0x00000028, 0x00000024, 0x00050050, 0x0000000c, 0x0000002b, 0x00000029, 0x0000002a,
    0x0004003d, 0x00000003, 0x0000002c, 0x00000013, 0x00050051, 0x00000002, 0x0000002d, 0x0000002c,
    0x00000000, 0x00050080, 0x00000002, 0x0000002e, 0x0000002d, 0x0000001f, 0x00060151, 0x00000002,
    0x0000002f, 0x00000021, 0x00000029, 0x0000002e, 0x0006016d, 0x0000000c, 0x00000030, 0x00000021,
    0x0000002b, 0x0000002e, 0x00050051, 0x00000002, 0x00000031, 0x00000030, 0x00000000, 0x00050051,
    0x00000002, 0x00000032, 0x00000030, 0x00000001, 0x00050084, 0x00000002, 0x00000033, 0x0000002d,
    0x00000022, 0x00050080, 0x00000002, 0x00000034, 0x00000033, 0x00000028, 0x00070050, 0x00000004,
    0x00000035, 0x0000002f, 0x00000031, 0x00000032, 0x00000028, 0x00060041, 0x00000008, 0x00000036,
    0x00000011, 0x0000001e, 0x00000034, 0x0003003e, 0x00000036, 0x00000035, 0x000100fd, 0x00010038,
]

// Two nested helpers read SPIR-V subgroup inputs in the fragment stage.
private let spirvFragmentSubgroupWords: [UInt32] = [
    0x07230203, 0x00010300, 0x00000000, 0x00000064, 0x00000000, 0x00020011, 0x00000001, 0x00020011,
    0x0000003d, 0x00020011, 0x00000040, 0x0003000e, 0x00000000, 0x00000001, 0x0008000f, 0x00000004,
    0x00000014, 0x6e69616d, 0x00000000, 0x00000010, 0x00000011, 0x00000012, 0x00030010, 0x00000014,
    0x00000007, 0x00040047, 0x00000010, 0x0000000b, 0x00000024, 0x00030047, 0x00000010, 0x0000000e,
    0x00040047, 0x00000011, 0x0000000b, 0x00000029, 0x00030047, 0x00000011, 0x0000000e, 0x00040047,
    0x00000012, 0x0000001e, 0x00000000, 0x00020013, 0x00000001, 0x00040015, 0x00000002, 0x00000020,
    0x00000000, 0x00040015, 0x00000003, 0x00000020, 0x00000001, 0x00040017, 0x00000004, 0x00000002,
    0x00000004, 0x00040020, 0x00000005, 0x00000001, 0x00000002, 0x00040020, 0x00000006, 0x00000001,
    0x00000003, 0x00040020, 0x00000007, 0x00000003, 0x00000004, 0x00030021, 0x00000008, 0x00000001,
    0x00030021, 0x00000009, 0x00000004, 0x0004002b, 0x00000002, 0x0000000a, 0x00000003, 0x0004002b,
    0x00000002, 0x0000000b, 0x000000ad, 0x0004002b, 0x00000003, 0x0000000c, 0x00000000, 0x0004003b,
    0x00000005, 0x00000010, 0x00000001, 0x0004003b, 0x00000006, 0x00000011, 0x00000001, 0x0004003b,
    0x00000007, 0x00000012, 0x00000003, 0x00050036, 0x00000001, 0x00000014, 0x00000000, 0x00000008,
    0x000200f8, 0x00000015, 0x00040039, 0x00000004, 0x00000016, 0x0000001e, 0x0003003e, 0x00000012,
    0x00000016, 0x000100fd, 0x00010038, 0x00050036, 0x00000004, 0x0000001e, 0x00000000, 0x00000009,
    0x000200f8, 0x0000001f, 0x00040039, 0x00000004, 0x00000020, 0x00000028, 0x000200fe, 0x00000020,
    0x00010038, 0x00050036, 0x00000004, 0x00000028, 0x00000000, 0x00000009, 0x000200f8, 0x00000029,
    0x0004003d, 0x00000002, 0x0000002a, 0x00000010, 0x0004003d, 0x00000003, 0x0000002b, 0x00000011,
    0x0004007c, 0x00000002, 0x0000002c, 0x0000002b, 0x00050152, 0x00000002, 0x0000002d, 0x0000000a,
    0x0000002c, 0x00070050, 0x00000004, 0x0000002e, 0x0000002a, 0x0000002c, 0x0000002d, 0x0000000b,
    0x000200fe, 0x0000002e, 0x00010038,
]

private func subgroupGatherDiagnostic(_ device: Device) async throws -> [UInt32] {
    let compiler = IndependentShaderCompiler()
    let compute = try compiler.compileCompute(.wgsl(subgroupGatherShader), to: .wgsl)
    if !device.capabilities.features.contains(.subgroups) {
        do {
            _ = try await device.createComputePipeline(compute)
        } catch GPUError.validation("Feature not advertised") {
            return []
        }
        throw GPUError.backendFailure("Subgroup shader accepted without the device feature")
    }
    let pipeline = try await device.createComputePipeline(compute)
    let spirv = try await device.createComputePipeline(compiler.compileCompute(.spirv(spirvSubgroupWords), to: .wgsl))
    let dynamicBroadcast = try await device.createComputePipeline(
        compiler.compileCompute(.spirv(spirvDynamicBroadcastWords), to: .wgsl))
    let roundTripSource = ShaderInput.wgsl("""
        enable subgroups;
        @group(0) @binding(0) var<storage, read_write> out: array<vec4u, 256>;
        @compute @workgroup_size(128)
        fn round_trip(@builtin(subgroup_invocation_id) lane: u32, @builtin(subgroup_size) size: u32,
                      @builtin(local_invocation_index) index: u32) {
            let paired = subgroupShuffle(lane + 11u, lane ^ 1u);
            let sum = subgroupAdd(lane + 1u);
            let prefix = subgroupInclusiveAdd(lane + 1u);
            out[index * 2u] = vec4u(lane | (index << 8u), size, paired, (sum << 16u) | prefix);
            out[index * 2u + 1u] = subgroupBallot((lane & 1u) == 0u);
        }
        """)
    guard case .spirv(let binary) = try compiler.compile(roundTripSource, to: .spirv) else {
        throw GPUError.backendFailure("Expected binary SPIR-V for subgroup operations")
    }
    let roundTripShader = try compiler.compileWebGPUCompute(.spirv(binary), entryPoint: "round_trip")
    let roundTrip = try await device.createComputePipeline(roundTripShader)
    let source = ShaderInput.wgsl(fragmentSubgroupShader)
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .wgsl)
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: .wgsl)
    let render = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Uint,
        vertexBuffers: [.init(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float32x2)])])
    let spirvFragment = try compiler.compileRenderStage(.spirv(spirvFragmentSubgroupWords), entryPoint: "main", to: .wgsl)
    let spirvRender = try await device.createRenderPipeline(vertex: vertex, fragment: spirvFragment, colorFormat: .rgba8Uint,
        vertexBuffers: [.init(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float32x2)])])
    let output = try await device.createBuffer(.init(size: 4096, usage: [.storage, .copySource]))
    let readback = try await device.createBuffer(.init(size: 4096, usage: [.copyDestination, .mapRead]))
    let vertices = try await device.createBuffer(.init(size: 24, usage: [.vertex, .copyDestination]))
    let target = try await device.createTexture(.init(width: 2, height: 2, format: .rgba8Uint,
        usage: [.renderAttachment, .copySource]))
    let snapshots = try await device.submitCommands([
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
        .copy(source: output, destination: readback, byteCount: 4096),
        .read(buffer: readback, byteCount: 4096),
        .write(buffer: vertices, offset: 0, words: [Float(-1), -1, 3, -1, -1, 3].map(\.bitPattern)),
        .render(pipeline: render, target: target, clear: .transparentBlack, vertexCount: 3,
                vertexBuffers: [.init(buffer: vertices)]),
        .copyTextureToBuffer(source: target, destination: readback, bytesPerRow: 256),
        .read(buffer: readback, byteCount: 512),
        .dispatchCompute(pipeline: spirv, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
        .copy(source: output, destination: readback, byteCount: 64),
        .read(buffer: readback, byteCount: 64),
        .dispatchCompute(pipeline: dynamicBroadcast, bindings: [.init(binding: 0): output], workgroups: .init(x: 2)),
        .copy(source: output, destination: readback, byteCount: 128),
        .read(buffer: readback, byteCount: 128),
        .render(pipeline: spirvRender, target: target, clear: .transparentBlack, vertexCount: 3,
                vertexBuffers: [.init(buffer: vertices)]),
        .copyTextureToBuffer(source: target, destination: readback, bytesPerRow: 256),
        .read(buffer: readback, byteCount: 512),
        .dispatchCompute(pipeline: roundTrip, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
        .copy(source: output, destination: readback, byteCount: 4096),
        .read(buffer: readback, byteCount: 4096),
    ]).wait()
    guard snapshots.map(\.count) == [1024, 128, 16, 32, 128, 1024] else {
        throw GPUError.backendFailure("Unexpected subgroup readback")
    }
    let roundTripWords = snapshots[5]
    for index in 0..<128 {
        let base = index * 8
        let lane = roundTripWords[base] & 255
        let size = roundTripWords[base + 1]
        guard (4...128).contains(size), size.nonzeroBitCount == 1, lane < size else {
            throw GPUError.backendFailure("Unexpected SPIR-V subgroup lane or size: \(lane)/\(size)")
        }
        let sum = size * (size + 1) / 2
        let prefix = (lane + 1) * (lane + 2) / 2
        let ballot = (0..<4).map { word -> UInt32 in
            let active = min(32, max(0, Int(size) - word * 32))
            let mask = active == 32 ? UInt32.max : (UInt32(1) << active) - 1
            return mask & 0x55555555
        }
        let expected = [lane | (UInt32(index) << 8), size, (lane ^ 1) + 11, (sum << 16) | prefix] + ballot
        guard roundTripWords[base..<(base + 8)].elementsEqual(expected) else {
            throw GPUError.backendFailure("Unexpected SPIR-V subgroup round-trip lane \(lane): \(roundTripWords[base..<(base + 8)])")
        }
    }
    let pixelOffsets = [0, 1, 64, 65]
    let wgslPixels = pixelOffsets.map { snapshots[1][$0] }
    let spirvPixels = pixelOffsets.map { snapshots[4][$0] }
    return snapshots[0] + wgslPixels + snapshots[2] + snapshots[3] + spirvPixels
}

private let subgroupCollectiveShader = """
    enable subgroups;
    @group(0) @binding(0) var<storage, read_write> out: array<vec4u, 896>;

    fn sum(lane: u32) -> u32 { return subgroupAdd(lane); }

    @compute @workgroup_size(128)
    fn run(@builtin(subgroup_size) size: u32, @builtin(subgroup_invocation_id) lane: u32,
           @builtin(local_invocation_index) index: u32) {
        let base = index * 7u;
        let factor = select(1u, 2u, lane < 3u);
        let leader = subgroupElect();
        out[base] = vec4u(size, lane, u32(subgroupAll(lane < size)), u32(subgroupAny(lane == size - 1u)));
        out[base + 1u] = subgroupBallot((lane & 1u) == 0u);
        out[base + 2u] = vec4u(sum(lane), subgroupMul(factor), subgroupMin(lane + 7u), subgroupMax(lane));
        out[base + 3u] = vec4u(subgroupAnd(lane | 0x80000000u), subgroupOr(lane),
                              subgroupXor(lane), subgroupInclusiveAdd(lane));
        out[base + 4u] = vec4u(subgroupExclusiveAdd(lane), subgroupExclusiveMul(factor),
                              subgroupInclusiveMul(factor), u32(subgroupAll(lane != 0u)) + u32(leader));
        out[base + 5u] = subgroupBallot();
        let q = quadBroadcast(lane, 0u);
        let x = quadSwapX(lane);
        let y = quadSwapY(lane);
        let d = quadSwapDiagonal(lane);
        out[base + 6u] = vec4u(u32(quadSwapX(x) == lane), u32(quadSwapY(y) == lane),
                              u32(quadSwapY(x) == d), u32(quadSwapDiagonal(q) == q));
    }
    """

private let subgroupCollectiveFragment = """
    enable subgroups;
    @vertex fn vs(@location(0) point: vec2f) -> @builtin(position) vec4f { return vec4f(point, 0.5, 1.0); }
    @fragment fn fs(@builtin(position) point: vec4f) -> @location(0) vec4u {
        let pixel = u32(point.x) + 2u * u32(point.y);
        let visible = u32(subgroupAny(pixel < 4u));
        return vec4u(quadBroadcast(visible, 0u), quadSwapX(pixel), quadSwapY(pixel), quadSwapDiagonal(pixel));
    }
    """

// Workgroup of four lanes, with a bvec2 predicate (lane < 3, lane == 1).
private func spirvLogicalSubgroupWords(opcode: UInt32, operation: UInt32) -> [UInt32] {
    var words: [UInt32] = [
        0x07230203, 0x00010300, 0x00000000, 0x00000064, 0x00000000, 0x00020011, 0x00000001, 0x00020011,
        0x0000003d, 0x00020011, 0x0000003f, 0x0003000e, 0x00000000, 0x00000001, 0x0006000f, 0x00000005,
        0x00000014, 0x6e69616d, 0x00000000, 0x00000012, 0x00060010, 0x00000014, 0x00000011, 0x00000004,
        0x00000001, 0x00000001, 0x00040047, 0x00000012, 0x0000000b, 0x00000029, 0x00040047, 0x00000005,
        0x00000006, 0x00000004, 0x00030047, 0x00000006, 0x00000002, 0x00050048, 0x00000006, 0x00000000,
        0x00000023, 0x00000000, 0x00040047, 0x00000011, 0x00000022, 0x00000000, 0x00040047, 0x00000011,
        0x00000021, 0x00000000, 0x00020013, 0x00000001, 0x00040015, 0x00000002, 0x00000020, 0x00000000,
        0x00020014, 0x00000003, 0x00040017, 0x00000004, 0x00000003, 0x00000002, 0x0004002b, 0x00000002,
        0x0000001e, 0x00000000, 0x0004002b, 0x00000002, 0x0000001f, 0x00000001, 0x0004002b, 0x00000002,
        0x00000020, 0x00000002, 0x0004002b, 0x00000002, 0x00000021, 0x00000003, 0x0004002b, 0x00000002,
        0x00000022, 0x00000008, 0x0004001c, 0x00000005, 0x00000002, 0x00000022, 0x0003001e, 0x00000006,
        0x00000005, 0x00040020, 0x00000007, 0x0000000c, 0x00000006, 0x00040020, 0x00000008, 0x0000000c,
        0x00000002, 0x00040020, 0x00000009, 0x00000001, 0x00000002, 0x00030021, 0x0000000a, 0x00000001,
        0x0004003b, 0x00000007, 0x00000011, 0x0000000c, 0x0004003b, 0x00000009, 0x00000012, 0x00000001,
        0x00050036, 0x00000001, 0x00000014, 0x00000000, 0x0000000a, 0x000200f8, 0x00000015, 0x0004003d,
        0x00000002, 0x00000028, 0x00000012, 0x000500b0, 0x00000003, 0x00000029, 0x00000028, 0x00000021,
        0x000500aa, 0x00000003, 0x0000002a, 0x00000028, 0x0000001f, 0x00050050, 0x00000004, 0x0000002b,
        0x00000029, 0x0000002a,
    ]
    words += [0x00060000 | opcode, 4, 44, 33, operation, 43]
    words += [
        0x00050051, 0x00000003, 0x0000002d, 0x0000002c, 0x00000000, 0x00050051, 0x00000003, 0x0000002e,
        0x0000002c, 0x00000001, 0x000600a9, 0x00000002, 0x0000002f, 0x0000002d, 0x0000001f, 0x0000001e,
        0x000600a9, 0x00000002, 0x00000030, 0x0000002e, 0x0000001f, 0x0000001e, 0x00050084, 0x00000002,
        0x00000031, 0x00000028, 0x00000020, 0x00050080, 0x00000002, 0x00000032, 0x00000031, 0x0000001f,
        0x00060041, 0x00000008, 0x00000033, 0x00000011, 0x0000001e, 0x00000031, 0x00060041, 0x00000008,
        0x00000034, 0x00000011, 0x0000001e, 0x00000032, 0x0003003e, 0x00000033, 0x0000002f, 0x0003003e,
        0x00000034, 0x00000030, 0x000100fd, 0x00010038,
    ]
    return words
}

private func subgroupCollectiveDiagnostic(_ device: Device) async throws -> [UInt32] {
    guard device.capabilities.features.contains(.subgroups) else { return [] }
    let compiler = IndependentShaderCompiler()
    let pipeline = try await device.createComputePipeline(compiler.compileCompute(.wgsl(subgroupCollectiveShader), to: .wgsl))
    let source = ShaderInput.wgsl(subgroupCollectiveFragment)
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .wgsl)
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: .wgsl)
    let render = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Uint,
        vertexBuffers: [.init(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float32x2)])])
    let output = try await device.createBuffer(.init(size: 14336, usage: [.storage, .copySource]))
    let readback = try await device.createBuffer(.init(size: 14336, usage: [.copyDestination, .mapRead]))
    let vertices = try await device.createBuffer(.init(size: 24, usage: [.vertex, .copyDestination]))
    let target = try await device.createTexture(.init(width: 2, height: 2, format: .rgba8Uint,
        usage: [.renderAttachment, .copySource]))
    var commands: [BufferCommand] = [
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
        .copy(source: output, destination: readback, byteCount: 14336),
        .read(buffer: readback, byteCount: 14336),
        .write(buffer: vertices, offset: 0, words: [Float(-1), -1, 3, -1, -1, 3].map(\.bitPattern)),
        .render(pipeline: render, target: target, clear: .transparentBlack, vertexCount: 3,
                vertexBuffers: [.init(buffer: vertices)]),
        .copyTextureToBuffer(source: target, destination: readback, bytesPerRow: 256),
        .read(buffer: readback, byteCount: 512),
    ]
    for opcode: UInt32 in 362...364 {
        for operation: UInt32 in 0...2 {
            let input = ShaderInput.spirv(spirvLogicalSubgroupWords(opcode: opcode, operation: operation))
            let logical = try await device.createComputePipeline(compiler.compileCompute(input, to: .wgsl))
            commands.append(.dispatchCompute(pipeline: logical, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)))
            commands.append(.copy(source: output, destination: readback, byteCount: 32))
            commands.append(.read(buffer: readback, byteCount: 32))
        }
    }
    let snapshots = try await device.submitCommands(commands).wait()
    guard snapshots.map(\.count) == [3584, 128] + Array(repeating: 8, count: 9) else {
        throw GPUError.backendFailure("Unexpected subgroup collective and quad readback")
    }
    let pixels = snapshots[1]
    return snapshots[0] + [pixels[0], pixels[1], pixels[64], pixels[65]] + snapshots.dropFirst(2).flatMap { $0 }
}

private let subgroupIndexShader = """
    requires subgroup_id;
    enable subgroups;

    struct Subgroup {
        @builtin(subgroup_id) group: u32,
        @builtin(num_subgroups) count: u32,
    }
    @group(0) @binding(0) var<storage, read_write> records: array<vec4u, 260>;

    @compute @workgroup_size(130)
    fn run(info: Subgroup, @builtin(global_invocation_id) global: vec3u,
           @builtin(subgroup_invocation_id) lane: u32, @builtin(subgroup_size) size: u32) {
        if info.count > 1u { workgroupBarrier(); }
        records[global.x] = vec4u(info.group, info.count, lane, size);
    }
    """

private func subgroupUniformityDiagnostic(_ device: Device) async throws -> [UInt32] {
    guard device.capabilities.features.isSuperset(of: [.subgroupID, .subgroupUniformity]) else { return [] }
    let source = ShaderInput.wgsl("""
        requires subgroup_id, subgroup_uniformity;
        enable subgroups;
        @group(0) @binding(0) var<storage, read_write> output: array<vec2u>;
        fn selected(group: u32) -> bool {
            var flag: bool;
            if group == 0u { flag = true; } else { flag = false; }
            return flag;
        }
        fn count_lanes(group: u32) -> u32 {
            if selected(group) { return subgroupAdd(1u); }
            return 0u;
        }
        fn forward(group: u32) -> u32 { return count_lanes(group); }
        @compute @workgroup_size(128)
        fn run(@builtin(subgroup_id) group: u32, @builtin(local_invocation_index) index: u32) {
            var selector = subgroupBroadcastFirst(group);
            let previous = selector;
            selector = index;
            output[index] = vec2u(group, forward(previous));
        }
        """)
    let shader = try IndependentShaderCompiler().compileWebGPUCompute(source)
    let pipeline = try await device.createComputePipeline(shader)
    let output = try await device.createBuffer(.init(size: 1024, usage: [.storage, .copySource]))
    let readback = try await device.createBuffer(.init(size: 1024, usage: [.mapRead, .copyDestination]))
    let result = try await device.submitCommands([
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
        .copy(source: output, destination: readback, byteCount: 1024),
        .read(buffer: readback, byteCount: 1024),
    ]).wait()
    guard result.count == 1, let words = result.first, words.count == 256 else {
        throw GPUError.backendFailure("Missing subgroup uniformity readback")
    }
    let indices = stride(from: 0, to: words.count, by: 2)
    let firstGroupSize = UInt32(indices.filter { words[$0] == 0 }.count)
    guard firstGroupSize > 0,
          indices.allSatisfy({ words[$0 + 1] == (words[$0] == 0 ? firstGroupSize : 0) }) else {
        throw GPUError.backendFailure("Subgroup-uniform branch executed on the wrong lanes")
    }
    return [firstGroupSize, UInt32(Set(indices.map { words[$0] }).count)]
}

private func subgroupSizeControlDiagnostic(_ device: Device) async throws -> UInt32 {
    guard device.capabilities.features.contains(.subgroupSizeControl),
          let size = device.capabilities.subgroupSizeRange?.lowerBound else { return 0 }
    let source = ShaderInput.wgsl("""
        enable subgroup_size_control;
        override size: u32;
        @group(0) @binding(0) var<storage, read_write> output: array<vec2u>;
        @compute @workgroup_size(2u * size) @subgroup_size(size)
        fn run(@builtin(local_invocation_index) index: u32,
               @builtin(subgroup_size) width: u32, @builtin(subgroup_invocation_id) lane: u32) {
            output[index] = vec2u(width, lane);
        }
        """)
    let shader = try IndependentShaderCompiler().compileWebGPUCompute(source, constants: ["size": Double(size)])
    let pipeline = try await device.createComputePipeline(shader)
    let byteCount = UInt64(shader.workgroupSize.x) * 8
    let output = try await device.createBuffer(.init(size: byteCount, usage: [.storage, .copySource]))
    let readback = try await device.createBuffer(.init(size: byteCount, usage: [.mapRead, .copyDestination]))
    let snapshots = try await device.submitCommands([
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
        .copy(source: output, destination: readback, byteCount: byteCount),
        .read(buffer: readback, byteCount: byteCount),
    ]).wait()
    guard snapshots.count == 1, let words = snapshots.first, words.count == Int(byteCount / 4) else {
        throw GPUError.backendFailure("Missing subgroup size readback")
    }
    var lanes = Array(repeating: 0, count: Int(size))
    for offset in stride(from: 0, to: words.count, by: 2) {
        let lane = words[offset + 1]
        guard words[offset] == size, lane < size else {
            throw GPUError.backendFailure("Unexpected width or lane for explicit subgroup size")
        }
        lanes[Int(lane)] += 1
    }
    guard lanes.allSatisfy({ $0 == 2 }) else { throw GPUError.backendFailure("Expected two full subgroups") }
    return size
}

private func subgroupIndexDiagnostic(_ device: Device) async throws -> [UInt32] {
    let compiler = IndependentShaderCompiler()
    let compute = try compiler.compileCompute(.wgsl(subgroupIndexShader), to: .wgsl)
    if !device.capabilities.features.contains(.subgroupID) {
        do {
            _ = try await device.createComputePipeline(compute)
        } catch GPUError.validation("Feature not advertised") {
            return []
        }
        throw GPUError.backendFailure("Subgroup ID accepted without the WGSL language feature")
    }
    let pipeline = try await device.createComputePipeline(compute)
    let records = try await device.createBuffer(.init(size: 4160, usage: [.storage, .copySource]))
    let readback = try await device.createBuffer(.init(size: 4160, usage: [.copyDestination, .mapRead]))
    let snapshots = try await device.submitCommands([
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): records], workgroups: .init(x: 2)),
        .copy(source: records, destination: readback, byteCount: 4160),
        .read(buffer: readback, byteCount: 4160),
    ]).wait()
    guard snapshots.count == 1, snapshots[0].count == 1040 else {
        throw GPUError.backendFailure("Unexpected subgroup index readback")
    }
    return snapshots[0]
}

private let dualSourceShader = """
    enable dual_source_blending;
    struct Outputs {
        @location(0) @blend_src(1) weight: vec4f,
        @blend_src(0) @location(0) color: vec4f,
    }
    @vertex fn vs(@location(0) point: vec2f) -> @builtin(position) vec4f { return vec4f(point, 0.5, 1.0); }
    @fragment fn fs() -> Outputs {
        return Outputs(vec4f(0.5, 0.25, 0.75, 0.25), vec4f(1.0, 0.5, 0.25, 1.0));
    }
    """

private func dualSourceDiagnostic(_ device: Device) async throws -> [UInt32] {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl(dualSourceShader)
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .wgsl)
    let fragment = try compiler.compileWebGPURenderStage(.glsl("""
        #version 450
        layout(location=0, index=1) out vec4 weight;
        layout(location=0, index=0) out vec4 color;
        void main() {
            weight = vec4(0.5, 0.25, 0.75, 0.25);
            color = vec4(1.0, 0.5, 0.25, 1.0);
        }
        """, stage: .fragment, defines: [:]), entryPoint: "main")
    let layout = VertexBufferLayout(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float32x2)])
    let blend = RenderBlendState(
        color: .init(sourceFactor: .source1Color, destinationFactor: .oneMinusSource1Color),
        alpha: .init(sourceFactor: .source1Alpha, destinationFactor: .oneMinusSource1Alpha))
    let inverse = RenderBlendState(
        color: .init(sourceFactor: .oneMinusSource1Color, destinationFactor: .source1Color),
        alpha: .init(sourceFactor: .oneMinusSource1Alpha, destinationFactor: .source1Alpha))
    if !device.capabilities.features.contains(.dualSourceBlending) {
        do {
            _ = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm,
                vertexBuffers: [layout], colorState: .init(blend: blend))
        } catch GPUError.validation("Feature not advertised") {
            return []
        }
        throw GPUError.backendFailure("Dual-source blending accepted without the device feature")
    }
    guard case .spirv(let binary) = try compiler.compile(source, to: .spirv) else {
        throw GPUError.backendFailure("Expected binary SPIR-V for dual-source blending")
    }
    let roundTrip = ShaderInput.spirv(binary)
    let importedVertex = try compiler.compileWebGPURenderStage(roundTrip, entryPoint: "vs")
    let importedFragment = try compiler.compileWebGPURenderStage(roundTrip, entryPoint: "fs")
    guard importedFragment.dualSourceComponentCount == 4 else {
        throw GPUError.backendFailure("Expected a four-component second blend source after SPIR-V import")
    }
    let blendStates: [RenderBlendState?] = [blend, inverse, nil]
    var pipelines: [RenderPipeline] = []
    for (stageVertex, stageFragment) in [(vertex, fragment), (importedVertex, importedFragment)] {
        for state in blendStates {
            pipelines.append(try await device.createRenderPipeline(
                vertex: stageVertex, fragment: stageFragment, colorFormat: .rgba8Unorm,
                vertexBuffers: [layout], colorState: .init(blend: state)))
        }
    }
    let vertices = try await device.createBuffer(.init(size: 24, usage: [.vertex, .copyDestination]))
    let target = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm,
        usage: [.renderAttachment, .copySource]))
    let readback = try await device.createBuffer(.init(size: 256, usage: [.mapRead, .copyDestination]))
    let clear = RenderClearColor(red: 0, green: 1, blue: 0, alpha: 0)
    let capture: [BufferCommand] = [
        .copyTextureToBuffer(source: target, destination: readback, bytesPerRow: 256),
        .read(buffer: readback, byteCount: 4),
    ]
    var commands: [BufferCommand] = [
        .write(buffer: vertices, offset: 0, words: [Float(-1), -1, 3, -1, -1, 3].map(\.bitPattern)),
    ]
    for pipeline in pipelines {
        commands.append(.render(pipeline: pipeline, target: target, clear: clear, vertexCount: 3,
                                vertexBuffers: [.init(buffer: vertices)]))
        commands += capture
    }
    let pixels = try await device.submitCommands(commands).wait().flatMap { $0 }
    let direct = Array(pixels.prefix(blendStates.count))
    guard pixels.count == pipelines.count, pixels.suffix(blendStates.count).elementsEqual(direct) else {
        throw GPUError.backendFailure("SPIR-V dual-source blending differed from GLSL import: \(pixels)")
    }
    return direct
}

private func validateNativeEarlyDepthFeature(_ device: Device) async throws {
    let source = ShaderInput.wgsl("""
        @vertex fn vs() -> @builtin(position) vec4f { return vec4f(0, 0, 0, 1); }
        @early_depth_test(force) @fragment fn fs() -> @location(0) vec4f { return vec4f(1); }
        """)
    let compiler = IndependentShaderCompiler()
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .wgsl)
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: .wgsl)
    let scope = try await device.pushErrorScope(.validation)
    do {
        _ = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm)
    } catch GPUError.validation("Feature not advertised") {
        guard case .validation("Feature not advertised") = try await device.popErrorScope(scope) else {
            throw GPUError.backendFailure("Early depth feature failure did not reach the error scope")
        }
        return
    }
    throw GPUError.backendFailure("Browser accepted the native early depth feature")
}

private let bcSampleShader = """
    @group(0) @binding(0) var linear: texture_2d<f32>;
    @group(0) @binding(1) var srgb: texture_2d<f32>;
    @group(0) @binding(2) var<storage, read_write> out: vec4f;
    @compute @workgroup_size(1) fn run() {
        let a = textureLoad(linear, vec2u(1u), 0u);
        let b = textureLoad(srgb, vec2u(1u), 0u);
        out = vec4f(a.x, b.x, a.w, b.w);
    }
    """

private func bcTextureDiagnostic(_ device: Device) async throws -> [UInt32] {
    guard device.capabilities.features.contains(.textureCompressionBC) else {
        do {
            _ = try await device.createTexture(.init(width: 4, height: 4, format: .bc1RGBAUnorm, usage: .textureBinding))
        } catch GPUError.validation("Feature not advertised") {
            return []
        }
        throw GPUError.validation("Accepted BC texture without the device feature")
    }
    let usage: TextureUsage = [.copySource, .copyDestination, .textureBinding]
    let linear = try await device.createTexture(.init(width: 8, height: 8, format: .bc1RGBAUnorm, usage: usage, mipLevelCount: 3))
    let srgb = try await device.createTexture(.init(width: 4, height: 4, format: .bc1RGBAUnormSrgb, usage: usage, mipLevelCount: 2))
    let bc3 = try await device.createTexture(.init(width: 4, height: 4, format: .bc3RGBAUnorm, usage: usage))
    let linearView = try await device.createTextureView(texture: linear, .init(baseMipLevel: 2, mipLevelCount: 1))
    let srgbView = try await device.createTextureView(texture: srgb, .init(baseMipLevel: 1, mipLevelCount: 1))
    let source = try await device.createBuffer(.init(size: 16, usage: [.copySource, .copyDestination]))
    let result = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
    let readback = try await device.createBuffer(.init(size: 272, usage: [.mapRead, .copyDestination]))
    let compiler = IndependentShaderCompiler()
    let pipeline = try await device.createComputePipeline(compiler.compileCompute(.wgsl(bcSampleShader), to: .wgsl))
    // Equal endpoints decode to a neutral mid-gray. Selector bits are zero.
    let gray: [UInt8] = [0xef, 0x7b, 0xef, 0x7b, 0, 0, 0, 0]
    let tail = TextureCopyRegion(width: 4, height: 4, mipLevel: 2)
    let commands: [BufferCommand] = [
        .copyTextureToBuffer(source: linear, destination: readback, bytesPerRow: 256, mipLevel: 2),
        .read(buffer: readback, byteCount: 8),
        .writeTexture(destination: linear, data: gray, region: .init(x: 4, y: 4, width: 4, height: 4)),
        .copyTextureToBuffer(source: linear, destination: readback, bytesPerRow: 256),
        .read(buffer: readback, byteCount: 272),
        .writeTexture(destination: linear, data: [0xee] + gray, layout: .init(offset: 1), region: tail),
        .copyTextureToTexture(source: linear, destination: srgb, sourceRegion: tail, destinationOrigin: .init(mipLevel: 1)),
        .copyTextureToBuffer(source: srgb, destination: readback, bytesPerRow: 256, mipLevel: 1),
        .read(buffer: readback, byteCount: 8),
        .write(buffer: source, offset: 0, words: [0x1234ffff, 0, 0x7bef7bef, 0]),
        .copyBufferToTexture(source: source, layout: .init(), destination: bc3, region: .init(width: 4, height: 4)),
        .copyTextureToBuffer(source: bc3, destination: readback, bytesPerRow: 256),
        .read(buffer: readback, byteCount: 16),
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 2): result], workgroups: .init(x: 1),
                         textures: [.init(binding: 0): linearView, .init(binding: 1): srgbView]),
        .copy(source: result, destination: readback, byteCount: 16),
        .read(buffer: readback, byteCount: 16),
    ]
    let snapshots = try await device.submitCommands(commands).wait()
    guard snapshots.map(\.count) == [2, 68, 2, 4, 4] else {
        throw GPUError.validation("Unexpected BC readback size")
    }
    return snapshots[0] + Array(snapshots[1][0..<4]) + Array(snapshots[1][64..<68]) +
        snapshots[2] + snapshots[3] + snapshots[4]
}

private let etc2ReadShader = """
    @group(0) @binding(0) var linear: texture_2d<f32>;
    @group(0) @binding(1) var srgb: texture_2d<f32>;
    @group(0) @binding(2) var eac: texture_2d<f32>;
    @group(0) @binding(3) var<storage, read_write> out: vec4f;
    @compute @workgroup_size(1) fn run() {
        let a = textureLoad(linear, vec2u(1u), 0u);
        let b = textureLoad(srgb, vec2u(1u), 0u);
        let c = textureLoad(eac, vec2u(1u), 0u);
        out = vec4f(a.x, b.x, c.xy);
    }
    """

private func etc2TextureDiagnostic(_ device: Device) async throws -> [UInt32] {
    guard device.capabilities.features.contains(.textureCompressionETC2) else {
        do {
            _ = try await device.createTexture(.init(width: 4, height: 4, format: .etc2RGB8Unorm, usage: .textureBinding))
        } catch GPUError.validation("Feature not advertised") {
            return []
        }
        throw GPUError.validation("Accepted ETC2 texture without the device feature")
    }
    let formats: [TextureFormat] = [
        .etc2RGB8Unorm, .etc2RGB8UnormSrgb, .etc2RGB8A1Unorm, .etc2RGB8A1UnormSrgb,
        .etc2RGBA8Unorm, .etc2RGBA8UnormSrgb, .eacR11Unorm, .eacR11Snorm, .eacRG11Unorm, .eacRG11Snorm,
    ]
    for format in formats {
        let image = try await device.createTexture(.init(width: 4, height: 4, format: format, usage: .textureBinding))
        try await device.destroyTexture(image)
    }
    let usage: TextureUsage = [.copySource, .copyDestination, .textureBinding]
    let linear = try await device.createTexture(.init(width: 8, height: 8, format: .etc2RGB8Unorm, usage: usage,
                                                     mipLevelCount: 3, arrayLayerCount: 2))
    let srgb = try await device.createTexture(.init(width: 4, height: 4, format: .etc2RGB8UnormSrgb, usage: usage, mipLevelCount: 2))
    let eac = try await device.createTexture(.init(width: 4, height: 4, format: .eacRG11Unorm, usage: usage))
    let linearView = try await device.createTextureView(texture: linear, .init(baseMipLevel: 2, mipLevelCount: 1,
                                                                              dimension: .d2, arrayLayerCount: 1))
    let srgbView = try await device.createTextureView(texture: srgb, .init(baseMipLevel: 1, mipLevelCount: 1))
    let source = try await device.createBuffer(.init(size: 16, usage: [.copySource, .copyDestination]))
    let result = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
    let readback = try await device.createBuffer(.init(size: 16, usage: [.mapRead, .copyDestination]))
    let compiler = IndependentShaderCompiler()
    let pipeline = try await device.createComputePipeline(compiler.compileCompute(.wgsl(etc2ReadShader), to: .wgsl))
    // ETC2 individual mode with the same 4-bit RGB base in both subblocks and zero selectors.
    let gray: [UInt8] = [0x66, 0x66, 0x66, 0, 0, 0, 0, 0]
    let bright: [UInt8] = [0xaa, 0xaa, 0xaa, 0, 0, 0, 0, 0]
    let data = [UInt8(0xee)] + gray + Array(repeating: 0xee, count: 14) + bright
    let commands: [BufferCommand] = [
        .writeTexture(destination: linear, data: data, layout: .init(offset: 1, bytesPerRow: 11, rowsPerImage: 2),
                      region: .init(width: 4, height: 4, mipLevel: 2, depth: 2)),
        .copyTextureToBuffer(source: linear, destination: readback, bytesPerRow: 256, mipLevel: 2, arrayLayer: 1),
        .read(buffer: readback, byteCount: 8),
        .copyTextureToTexture(source: linear, destination: srgb, sourceRegion: .init(width: 4, height: 4, mipLevel: 2),
                              destinationOrigin: .init(mipLevel: 1)),
        .copyTextureToBuffer(source: srgb, destination: readback, bytesPerRow: 256, mipLevel: 1),
        .read(buffer: readback, byteCount: 8),
        .write(buffer: source, offset: 0, words: [0x80, 0, 0x40, 0]),
        .copyBufferToTexture(source: source, layout: .init(), destination: eac, region: .init(width: 4, height: 4)),
        .copyTextureToBuffer(source: eac, destination: readback, bytesPerRow: 256),
        .read(buffer: readback, byteCount: 16),
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 3): result], workgroups: .init(x: 1),
                         textures: [.init(binding: 0): linearView, .init(binding: 1): srgbView, .init(binding: 2): eac]),
        .copy(source: result, destination: readback, byteCount: 16),
        .read(buffer: readback, byteCount: 16),
    ]
    let snapshots = try await device.submitCommands(commands).wait()
    guard snapshots.map(\.count) == [2, 2, 4, 4] else {
        throw GPUError.validation("Unexpected ETC2 and EAC readback size")
    }
    return snapshots.flatMap { $0 }
}

private let astcReadShader = """
    @group(0) @binding(0) var linear: texture_2d<f32>;
    @group(0) @binding(1) var srgb: texture_2d<f32>;
    @group(0) @binding(2) var wide: texture_2d<f32>;
    @group(0) @binding(3) var<storage, read_write> out: vec4f;
    @compute @workgroup_size(1) fn run() {
        let a = textureLoad(linear, vec2u(6u, 5u), 0u);
        let b = textureLoad(srgb, vec2u(6u, 5u), 0u);
        let c = textureLoad(wide, vec2u(4u, 3u), 0u);
        out = vec4f(a.x, b.x, c.x, c.w);
    }
    """

private func astcTextureDiagnostic(_ device: Device) async throws -> [UInt32] {
    guard device.capabilities.features.contains(.textureCompressionASTC) else {
        do {
            _ = try await device.createTexture(.init(width: 4, height: 4, format: .astc4x4Unorm, usage: .textureBinding))
        } catch GPUError.validation("Feature not advertised") {
            return []
        }
        throw GPUError.validation("Accepted ASTC texture without the device feature")
    }
    let shapes: [(UInt32, UInt32, TextureFormat, TextureFormat)] = [
        (4, 4, .astc4x4Unorm, .astc4x4UnormSrgb),
        (5, 4, .astc5x4Unorm, .astc5x4UnormSrgb),
        (5, 5, .astc5x5Unorm, .astc5x5UnormSrgb),
        (6, 5, .astc6x5Unorm, .astc6x5UnormSrgb),
        (6, 6, .astc6x6Unorm, .astc6x6UnormSrgb),
        (8, 5, .astc8x5Unorm, .astc8x5UnormSrgb),
        (8, 6, .astc8x6Unorm, .astc8x6UnormSrgb),
        (8, 8, .astc8x8Unorm, .astc8x8UnormSrgb),
        (10, 5, .astc10x5Unorm, .astc10x5UnormSrgb),
        (10, 6, .astc10x6Unorm, .astc10x6UnormSrgb),
        (10, 8, .astc10x8Unorm, .astc10x8UnormSrgb),
        (10, 10, .astc10x10Unorm, .astc10x10UnormSrgb),
        (12, 10, .astc12x10Unorm, .astc12x10UnormSrgb),
        (12, 12, .astc12x12Unorm, .astc12x12UnormSrgb),
    ]
    for (width, height, linear, srgb) in shapes {
        for format in [linear, srgb] {
            let image = try await device.createTexture(.init(width: width, height: height, format: format, usage: .textureBinding))
            try await device.destroyTexture(image)
        }
    }
    let usage: TextureUsage = [.copySource, .copyDestination, .textureBinding]
    let linear = try await device.createTexture(.init(width: 15, height: 12, format: .astc5x4Unorm, usage: usage, mipLevelCount: 2))
    let srgb = try await device.createTexture(.init(width: 15, height: 12, format: .astc5x4UnormSrgb, usage: usage, mipLevelCount: 2))
    let wide = try await device.createTexture(.init(width: 20, height: 16, format: .astc10x8Unorm,
                                                   usage: usage, mipLevelCount: 3))
    let linearView = try await device.createTextureView(texture: linear, .init(baseMipLevel: 1, mipLevelCount: 1))
    let srgbView = try await device.createTextureView(texture: srgb, .init(baseMipLevel: 1, mipLevelCount: 1))
    let wideView = try await device.createTextureView(texture: wide, .init(baseMipLevel: 2, mipLevelCount: 1))
    let source = try await device.createBuffer(.init(size: 16, usage: [.copySource, .copyDestination]))
    let output = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
    let readback = try await device.createBuffer(.init(size: 288, usage: [.mapRead, .copyDestination]))
    let compiler = IndependentShaderCompiler()
    let pipeline = try await device.createComputePipeline(compiler.compileCompute(.wgsl(astcReadShader), to: .wgsl))
    // Void-extent blocks with all coordinates set to one encode constant colors.
    let gray: [UInt8] = [0xfc, 0xfd, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0xff, 0xff]
    let wideWords: [UInt32] = [0xfffffdfc, 0xffffffff, 0x0000bfbf, 0xffff0000]
    let corner = TextureCopyRegion(x: 5, y: 4, width: 5, height: 4, mipLevel: 1)
    let commands: [BufferCommand] = [
        .writeTexture(destination: linear, data: [0xee] + gray, layout: .init(offset: 1), region: corner),
        .copyTextureToTexture(source: linear, destination: srgb, sourceRegion: corner,
                              destinationOrigin: .init(x: 5, y: 4, mipLevel: 1)),
        .copyTextureToBuffer(source: srgb, destination: readback, bytesPerRow: 256, mipLevel: 1),
        .read(buffer: readback, byteCount: 288),
        .write(buffer: source, offset: 0, words: wideWords),
        .copyBufferToTexture(source: source, layout: .init(), destination: wide, region: .init(width: 10, height: 8, mipLevel: 2)),
        .copyTextureToBuffer(source: wide, destination: readback, bytesPerRow: 256, mipLevel: 2),
        .read(buffer: readback, byteCount: 16),
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 3): output], workgroups: .init(x: 1),
                         textures: [.init(binding: 0): linearView, .init(binding: 1): srgbView, .init(binding: 2): wideView]),
        .copy(source: output, destination: readback, byteCount: 16),
        .read(buffer: readback, byteCount: 16),
    ]
    let snapshots = try await device.submitCommands(commands).wait()
    guard snapshots.map(\.count) == [72, 4, 4] else {
        throw GPUError.validation("Unexpected ASTC readback size")
    }
    return Array(snapshots[0][0..<8]) + Array(snapshots[0][64..<72]) + snapshots[1] + snapshots[2]
}

private let compressedVolumeReadShader = """
    @group(0) @binding(0) var input: texture_3d<f32>;
    @group(0) @binding(1) var copied: texture_3d<f32>;
    @group(0) @binding(2) var<storage, read_write> out: vec4f;
    @compute @workgroup_size(1) fn run() {
        let a = textureLoad(copied, vec3u(5u, 4u, 1u), 0u);
        let b = textureLoad(copied, vec3u(5u, 4u, 2u), 0u);
        let c = textureLoad(input, vec3u(5u, 4u, 2u), 0u);
        out = vec4f(a.x, b.x, c.x, a.w);
    }
    """

private func compressedVolumeDiagnostic(_ device: Device, format: TextureFormat, feature: GPUFeature) async throws -> [UInt32] {
    let bc = format == .bc1RGBAUnorm
    let blockWidth: UInt32 = bc ? 4 : 5
    let texture = TextureDescriptor(width: blockWidth * 3, height: 12, format: format,
                                    usage: [.copySource, .copyDestination, .textureBinding],
                                    mipLevelCount: 3, dimension: .d3, depth: 6)
    guard device.capabilities.features.contains(feature) else {
        do {
            _ = try await device.createTexture(texture)
        } catch GPUError.validation("Feature not advertised") {
            return []
        }
        throw GPUError.validation("Accepted compressed volume without the device feature")
    }
    let blocks: [[UInt32]] = bc ? [[0x40004000, 0], [0x80008000, 0], [0xc000c000, 0]] :
        [0x4040, 0x8080, 0xc0c0].map { [0xfffffdfc, 0xffffffff, $0, 0xffff0000] }
    let blockBytes = blocks[0].count * 4
    let stride = blockBytes + 3
    let bytes = [UInt8(0xee)] + blocks[0].withUnsafeBytes { Array($0) } +
        [UInt8](repeating: 0xee, count: 2 * stride - blockBytes) + blocks[1].withUnsafeBytes { Array($0) }
    let input = try await device.createTexture(texture)
    let copied = try await device.createTexture(texture)
    let inputView = try await device.createTextureView(texture: input, .init(baseMipLevel: 1, mipLevelCount: 1))
    let copiedView = try await device.createTextureView(texture: copied, .init(baseMipLevel: 1, mipLevelCount: 1))
    let staging = try await device.createBuffer(.init(size: UInt64(blockBytes), usage: [.copySource, .copyDestination]))
    let layeredStaging = try await device.createBuffer(.init(size: UInt64(256 + blockBytes), usage: [.copySource, .copyDestination]))
    let layered = try await device.createTexture(texture)
    let output = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
    let readbackSize = UInt64(1280 + 2 * blockBytes)
    let readback = try await device.createBuffer(.init(size: readbackSize, usage: [.copyDestination, .mapRead]))
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(compressedVolumeReadShader), to: .wgsl)
    let pipeline = try await device.createComputePipeline(shader)
    let pair = TextureCopyRegion(x: blockWidth, y: 4, width: blockWidth, height: 4, mipLevel: 1, depth: 2)
    let commands: [BufferCommand] = [
        .writeTexture(destination: input, data: bytes, layout: .init(offset: 1, bytesPerRow: UInt32(stride), rowsPerImage: 2),
                      region: pair),
        .write(buffer: staging, offset: 0, words: blocks[2]),
        .copyBufferToTexture(source: staging, layout: .init(), destination: input,
                             region: .init(x: blockWidth, y: 4, width: blockWidth, height: 4, mipLevel: 1, z: 2)),
        .copyTextureToTexture(source: input, destination: copied, sourceRegion: pair,
                              destinationOrigin: .init(x: blockWidth, y: 4, mipLevel: 1, z: 1)),
        .copyTextureToBuffer(source: copied, destination: readback, bytesPerRow: 256, mipLevel: 1, depth: 3),
        .read(buffer: readback, byteCount: readbackSize),
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 2): output], workgroups: .init(x: 1),
                         textures: [.init(binding: 0): inputView, .init(binding: 1): copiedView]),
        .copy(source: output, destination: readback, byteCount: 16),
        .read(buffer: readback, byteCount: 16),
        .write(buffer: layeredStaging, offset: 0, words: blocks[0]),
        .write(buffer: layeredStaging, offset: 256, words: blocks[1]),
        .copyBufferToTexture(source: layeredStaging, layout: .init(bytesPerRow: 256), destination: layered, region: pair),
        .copyTextureToBuffer(source: layered, destination: readback, bytesPerRow: 256, mipLevel: 1, depth: 2),
        .read(buffer: readback, byteCount: readbackSize),
    ]
    let snapshots = try await device.submitCommands(commands).wait()
    guard snapshots.map(\.count) == [Int(readbackSize / 4), 4, Int(readbackSize / 4)] else {
        throw GPUError.validation("Unexpected compressed volume readback size")
    }
    let blockWords = blocks[0].count
    guard Array(snapshots[2][(64 + blockWords)..<(64 + 2 * blockWords)]) == blocks[0],
          Array(snapshots[2][(192 + blockWords)..<(192 + 2 * blockWords)]) == blocks[1] else {
        throw GPUError.validation("Compressed multi-layer buffer upload returned incorrect blocks")
    }
    let wordsPerRow = 2 * blocks[0].count
    return [0, 64, 128, 192, 256, 320].flatMap { offset in
        snapshots[0][offset..<(offset + wordsPerRow)]
    } + snapshots[1]
}

private let callStatementShader = """
    @group(0) @binding(0) var<storage, read_write> out: array<u32, 4>;
    var<workgroup> cell: atomic<u32>;

    @compute @workgroup_size(1) fn run() {
        let value = required(3u);
        _ = required(4u);
        update(9u);
        for (mark(); out[0] < 23u; _ = required(2u)) { update(5u); }
        for (set(0u); out[2] < 2u; tick()) {}
        atomicStore(&cell, 1u);
        atomicLoad(&cell);
        atomicAdd(&cell, 4u);
        out[3] = value + atomicLoad(&cell);
    }
    @must_use fn required(delta: u32) -> u32 {
        update(delta);
        return delta * 2u;
    }
    fn update(delta: u32) -> u32 {
        out[0] += delta;
        return out[0];
    }
    fn mark() { out[1] = out[0]; }
    fn set(value: u32) { out[2] = value; }
    fn tick() { out[2] += 1u; }
    """

private func callStatementDiagnostic(_ device: Device) async throws -> [UInt32] {
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(callStatementShader), to: .wgsl)
    let pipeline = try await device.createComputePipeline(shader)
    let unreachableShader = try IndependentShaderCompiler().compileWebGPUCompute(.wgsl("""
        @group(0) @binding(0) var<storage, read_write> output: array<vec4u, 2>;

        fn selected(lane: u32) -> u32 {
            switch lane {
                case 0u: {
                    return 7u;
                    output[0].x = 999u;
                    break;
                }
                default: { return 11u; }
            }
            output[1].x = 999u;
        }

        fn loopValue() -> u32 {
            loop {
                return 23u;
                break;
            }
        }

        fn deadBarrier() {
            return;
            workgroupBarrier();
        }

        fn loopCounter() -> u32 {
            var counter = 0u;
            loop {
                counter = 17u;
                break;
                counter = 99u;
                continuing { counter = 123u; }
            }
            return counter;
        }

        @compute @workgroup_size(2)
        fn run(@builtin(local_invocation_index) lane: u32) {
            if lane == 0u { deadBarrier(); }
            output[lane] = vec4u(selected(lane), loopValue(), loopCounter(), 5u);
        }
        """))
    let unreachablePipeline = try await device.createComputePipeline(unreachableShader)
    let output = try await device.createBuffer(.init(size: 32, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 32, usage: [.mapRead, .copyDestination]))
    let snapshots = try await device.submitCommands([
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
        .copy(source: output, destination: read, byteCount: 16),
        .read(buffer: read, byteCount: 16),
        .dispatchCompute(pipeline: unreachablePipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
        .copy(source: output, destination: read, byteCount: 32),
        .read(buffer: read, byteCount: 32),
    ]).wait()
    guard snapshots.map(\.count) == [4, 8] else {
        throw GPUError.validation("Unexpected call-statement readback")
    }
    return snapshots.flatMap { $0 }
}

private let matrixCompoundShader = """
    @group(0) @binding(0) var<storage, read_write> out: array<u32, 14>;
    var<private> matrices: array<mat2x3f, 2>;
    var<private> index: u32;

    fn lhsIndex() -> u32 {
        out[0] += 1u;
        return index;
    }
    fn rhsMatrix() -> mat2x3f {
        out[1] = out[0];
        matrices[0] = mat2x3f(vec3f(99), vec3f(99));
        index = 1u;
        return mat2x3f(vec3f(1), vec3f(2));
    }
    @compute @workgroup_size(1) fn run() {
        matrices[0] = mat2x3f(vec3f(1, 2, 3), vec3f(4, 5, 6));
        matrices[1] = mat2x3f(vec3f(9), vec3f(8));
        matrices[lhsIndex()] += rhsMatrix();
        matrices[0] -= mat2x3(vec3(1.0), vec3(1.0));
        matrices[0] *= 2;
        matrices[0] *= mat2x2f(vec2f(2, 1), vec2f(1, 3));
        var row = vec3f(1, 2, 3);
        row *= mat3x3f(vec3f(1, 0, 0), vec3f(2, 1, 0), vec3f(0, 1, 2));
        out[2] = index;
        out[3] = bitcast<u32>(matrices[0][0].x);
        out[4] = bitcast<u32>(matrices[0][0].y);
        out[5] = bitcast<u32>(matrices[0][0].z);
        out[6] = bitcast<u32>(matrices[0][1].x);
        out[7] = bitcast<u32>(matrices[0][1].y);
        out[8] = bitcast<u32>(matrices[0][1].z);
        out[9] = bitcast<u32>(row.x);
        out[10] = bitcast<u32>(row.y);
        out[11] = bitcast<u32>(row.z);
        out[12] = bitcast<u32>(matrices[1][0].x);
        out[13] = bitcast<u32>(matrices[1][1].x);
    }
    """

private func matrixCompoundDiagnostic(_ device: Device) async throws -> [UInt32] {
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(matrixCompoundShader), to: .wgsl)
    let pipeline = try await device.createComputePipeline(shader)
    let output = try await device.createBuffer(.init(size: 56, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 56, usage: [.mapRead, .copyDestination]))
    let snapshots = try await device.submitCommands([
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
        .copy(source: output, destination: read, byteCount: 56),
        .read(buffer: read, byteCount: 56),
    ]).wait()
    guard snapshots.count == 1, let words = snapshots.first else {
        throw GPUError.validation("Unexpected matrix compound assignment readback")
    }
    return words
}

private func float32TextureShader(filter: Bool, coordinates: String = "vec2f(0.5)") -> ShaderInput {
    let sampler = filter ? "@group(0) @binding(3) var state: sampler;" : ""
    func read(_ name: String) -> String {
        filter ? "textureSampleLevel(\(name), state, \(coordinates), 0.0)" : "textureLoad(\(name), vec2u(0), 0u)"
    }
    return .wgsl("""
        @group(0) @binding(0) var single: texture_2d<f32>;
        @group(0) @binding(1) var pair: texture_2d<f32>;
        @group(0) @binding(2) var color: texture_2d<f32>;
        \(sampler)
        @group(0) @binding(4) var<storage, read_write> out: vec4f;
        fn readAll() -> vec4f {
            return vec4f(\(read("single")).x, \(read("pair")).y, \(read("color")).w, f32(textureDimensions(single).x));
        }
        @compute @workgroup_size(1) fn run() { out = readAll(); }
        """)
}

private func float32TextureDiagnostic(_ device: Device) async throws -> [UInt32] {
    let formats: [TextureFormat] = [.r32Float, .rg32Float, .rgba32Float]
    let values: [[Float]] = [[1, 3], [1, 10, 3, 14], [1, 2, 3, 0.25, 4, 5, 6, 0.75]]
    var textures: [ComputeBindingSlot: ResourceID] = [:]
    var uploads: [BufferCommand] = []
    for (index, format) in formats.enumerated() {
        let image = try await device.createTexture(.init(width: 2, height: 1, format: format,
                                                         usage: [.copyDestination, .textureBinding]))
        textures[.init(binding: UInt32(index))] = image
        let bytes = values[index].withUnsafeBytes { Array($0) }
        uploads.append(.writeTexture(destination: image, data: bytes, region: .init(width: 2, height: 1)))
    }
    let output = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
    let readback = try await device.createBuffer(.init(size: 16, usage: [.mapRead, .copyDestination]))
    let sampler = try await device.createSampler(.init(minFilter: .linear, magFilter: .linear))
    let compiler = IndependentShaderCompiler()
    var results: [[UInt32]] = []
    for filter in [false, true] {
        let shader = try compiler.compileCompute(float32TextureShader(filter: filter), to: .wgsl)
        let pipeline = try await device.createComputePipeline(shader)
        let dispatch = BufferCommand.dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 4): output],
            workgroups: .init(x: 1), textures: textures, samplers: filter ? [.init(binding: 3): sampler] : [:])
        if filter && !device.capabilities.features.contains(.float32Filterable) {
            let scope = try await device.pushErrorScope(.validation)
            do {
                _ = try await device.submitCommands([dispatch])
            } catch GPUError.validation("Float32 sampling requires float32-filterable") {
                guard case .validation("Float32 sampling requires float32-filterable") = try await device.popErrorScope(scope) else {
                    throw GPUError.backendFailure("Float32 filterability error did not reach the error scope")
                }
                break
            }
            throw GPUError.backendFailure("Accepted filtered float32 texture without the device feature")
        }
        let capture: [BufferCommand] = [
            dispatch, .copy(source: output, destination: readback, byteCount: 16), .read(buffer: readback, byteCount: 16),
        ]
        results += try await device.submitCommands(filter ? capture : capture + uploads + capture).wait()
    }
    let nearestShader = try compiler.compileCompute(
        float32TextureShader(filter: true, coordinates: "vec2f(0.75, 0.5)"), to: .wgsl)
    let samplerSlot = ComputeBindingSlot(binding: 3)
    let nearest = try await device.createComputePipeline(nearestShader, nonFilteringSamplers: [samplerSlot])
    let nearestSampler = try await device.createSampler(.init())
    func dispatch(sampler: ResourceID) -> BufferCommand {
        .dispatchCompute(pipeline: nearest, bindings: [.init(binding: 4): output], workgroups: .init(x: 1),
                         textures: textures, samplers: [samplerSlot: sampler])
    }
    let scope = try await device.pushErrorScope(.validation)
    do {
        _ = try await device.submitCommands([dispatch(sampler: sampler)])
        throw GPUError.backendFailure("Accepted a filtering sampler in a non-filtering slot")
    } catch GPUError.validation("Invalid sampler binding") {
        guard case .validation("Invalid sampler binding") = try await device.popErrorScope(scope) else {
            throw GPUError.backendFailure("Invalid sampler error did not reach the error scope")
        }
    }
    results += try await device.submitCommands([
        dispatch(sampler: nearestSampler), .copy(source: output, destination: readback, byteCount: 16),
        .read(buffer: readback, byteCount: 16),
    ]).wait()
    guard let image = textures[.init(binding: 0)] else {
        throw GPUError.backendFailure("Missing float32 texture for render sampling")
    }
    let pixels = try await renderNearestFloat32(device, image: image, sampler: nearestSampler)
    return results.flatMap { $0 } + pixels
}

private func renderNearestFloat32(_ device: Device, image: ResourceID, sampler: ResourceID) async throws -> [UInt32] {
    let source = ShaderInput.wgsl("""
        @group(0) @binding(0) var image: texture_2d<f32>;
        @group(0) @binding(1) var state: sampler;
        @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
            let z = 0.25 * textureLoad(image, vec2u(0), 0u).x;
            if i == 0u { return vec4f(-1, -1, z, 1); }
            if i == 1u { return vec4f(3, -1, z, 1); }
            return vec4f(-1, 3, z, 1);
        }
        @fragment fn fs() -> @location(0) vec4f {
            let green = 0.25 * textureSampleLevel(image, state, vec2f(0.75, 0.5), 0.0).x;
            return vec4f(0, green, 0, 1);
        }
        """)
    let compiler = IndependentShaderCompiler()
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .wgsl)
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: .wgsl)
    let samplerSlot = ComputeBindingSlot(binding: 1)
    let pipeline = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm,
                                                         nonFilteringSamplers: [samplerSlot])
    let target = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm,
                                                      usage: [.renderAttachment, .copySource]))
    let readback = try await device.createBuffer(.init(size: 4, usage: [.mapRead, .copyDestination]))
    return try await device.submitCommands([
        .render(pipeline: pipeline, target: target, clear: .transparentBlack, vertexCount: 3,
                textures: [.init(binding: 0): image], samplers: [samplerSlot: sampler]),
        .copyTextureToBuffer(source: target, destination: readback, bytesPerRow: 256),
        .read(buffer: readback, byteCount: 4),
    ]).wait().flatMap { $0 }
}

private let float32ColorShader = """
    struct Outputs {
        @location(0) red: f32,
        @location(1) pair: vec2f,
        @location(2) color: vec4f,
    }
    @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
        if i == 0u { return vec4f(-1, -1, 0, 1); }
        if i == 1u { return vec4f(3, -1, 0, 1); }
        return vec4f(-1, 3, 0, 1);
    }
    @fragment fn fs() -> Outputs {
        return Outputs(2.5, vec2f(-1, 3), vec4f(0.5, -0.5, 4, 0.25));
    }
    @fragment fn one() -> @location(0) f32 { return 2.5; }
    @fragment fn two() -> @location(0) vec2f { return vec2f(-1, 3); }
    @fragment fn four() -> @location(0) vec4f { return vec4f(0.5, -0.5, 4, 0.25); }
    """

private func float32ColorDiagnostic(_ device: Device) async throws -> [UInt32] {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl(float32ColorShader)
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .wgsl)
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: .wgsl)
    let formats: [TextureFormat] = [.r32Float, .rg32Float, .rgba32Float]
    var images: [ResourceID] = []
    for format in formats {
        images.append(try await device.createTexture(.init(width: 1, height: 1, format: format,
                                                            usage: [.renderAttachment, .copySource])))
    }
    let readback = try await device.createBuffer(.init(size: 16, usage: [.copyDestination, .mapRead]))
    let clear = RenderClearColor(red: 1, green: -2, blue: 3, alpha: 0.5)
    let additional: [UInt32: RenderColorAttachment] = [
        1: .init(texture: images[1], loadAction: .clear(clear)),
        2: .init(texture: images[2], loadAction: .clear(clear)),
    ]
    let additive = RenderBlendComponent(sourceFactor: .one, destinationFactor: .one)
    var snapshots: [[UInt32]] = []
    for blend in [false, true] {
        let state = RenderColorState(blend: blend ? .init(color: additive, alpha: additive) : nil)
        func makePipeline() async throws -> RenderPipeline {
            try await device.createRenderPipeline(
                vertex: vertex, fragment: fragment, colorFormat: formats[0], colorState: state,
                additionalColorTargets: [1: .init(format: formats[1], colorState: state),
                                         2: .init(format: formats[2], colorState: state)])
        }
        if blend && !device.capabilities.features.contains(.float32Blendable) {
            let scope = try await device.pushErrorScope(.validation)
            do {
                _ = try await makePipeline()
                throw GPUError.backendFailure("Accepted float32 blending without the device feature")
            } catch GPUError.validation("Float32 blending requires float32-blendable") {
                guard case .validation("Float32 blending requires float32-blendable") = try await device.popErrorScope(scope) else {
                    throw GPUError.backendFailure("Float32 blending error did not reach the error scope")
                }
                break
            }
        }
        let pipeline = try await makePipeline()
        var commands: [BufferCommand] = [
            .render(pipeline: pipeline, target: images[0], clear: clear, vertexCount: 3, additionalColors: additional),
        ]
        for (image, bytes) in zip(images, [UInt64(4), 8, 16]) {
            commands += [.copyTextureToBuffer(source: image, destination: readback, bytesPerRow: 256),
                         .read(buffer: readback, byteCount: bytes)]
        }
        snapshots += try await device.submitCommands(commands).wait()
    }
    let formatsAndEntries: [(TextureFormat, String)] = [(.r32Float, "one"), (.rg32Float, "two"), (.rgba32Float, "four")]
    var words = snapshots.flatMap { $0 }
    for (format, entry) in formatsAndEntries {
        if device.capabilities.multisampledFloat32Formats.contains(format) {
            words += try await renderMultisampledFloat32(device, format: format, entry: entry)
        } else {
            let descriptor = TextureDescriptor(width: 1, height: 1, format: format, usage: .renderAttachment, sampleCount: 4)
            do {
                _ = try await device.createTexture(descriptor)
                throw GPUError.backendFailure("Accepted an unsupported float32 multisample format")
            } catch GPUError.validation("Invalid multisampled texture format, mip levels or usage") {}
        }
    }
    return words
}

private let float32MultisampleShader = """
    @group(0) @binding(0) var image: texture_multisampled_2d<f32>;
    @group(0) @binding(1) var<storage, read_write> out: vec4f;
    @compute @workgroup_size(1) fn run() {
        let texel = textureLoad(image, vec2u(0), 3u);
        out = vec4f(texel.xyz, f32(textureNumSamples(image)));
    }
    """

private func renderMultisampledFloat32(_ device: Device, format: TextureFormat, entry: String) async throws -> [UInt32] {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl(float32ColorShader)
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .wgsl)
    let fragment = try compiler.compileRenderStage(source, entryPoint: entry, to: .wgsl)
    let render = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: format, sampleCount: 4)
    let compute = try await device.createComputePipeline(compiler.compileCompute(.wgsl(float32MultisampleShader), to: .wgsl))
    let image = try await device.createTexture(.init(width: 1, height: 1, format: format,
                                                     usage: [.renderAttachment, .textureBinding], sampleCount: 4))
    let output = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 16, usage: [.mapRead, .copyDestination]))
    return try await device.submitCommands([
        .render(pipeline: render, target: image, clear: .transparentBlack, vertexCount: 3),
        .dispatchCompute(pipeline: compute, bindings: [.init(binding: 1): output], workgroups: .init(x: 1),
                         textures: [.init(binding: 0): image]),
        .copy(source: output, destination: read, byteCount: 16), .read(buffer: read, byteCount: 16),
    ]).wait().flatMap { $0 }
}

private let halfColorShader = """
    struct Outputs {
        @location(0) red: f32,
        @location(1) pair: vec2f,
        @location(2) color: vec4f,
    }
    @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
        if i == 0u { return vec4f(-1, -1, 0, 1); }
        if i == 1u { return vec4f(3, -1, 0, 1); }
        return vec4f(-1, 3, 0, 1);
    }
    @fragment fn fs() -> Outputs {
        return Outputs(2.5, vec2f(-1, 3), vec4f(0.5, -0.5, 4, 0.25));
    }
    """

private let halfSampleShader = """
    @group(0) @binding(0) var image: texture_2d<f32>;
    @group(0) @binding(1) var state: sampler;
    @group(0) @binding(2) var<storage, read_write> out: f32;
    @compute @workgroup_size(1) fn run() { out = textureSampleLevel(image, state, vec2f(0.5), 0.0).x; }
    """

private func halfTexelTransferDiagnostic(_ device: Device) async throws -> [[UInt32]] {
    let usage: TextureUsage = [.copySource, .copyDestination, .textureBinding]
    let line = try await device.createTexture(.init(width: 1, height: 1, format: .r16Float, usage: usage, dimension: .d1))
    let image = try await device.createTexture(.init(width: 2, height: 1, format: .r16Float, usage: usage))
    let upload = try await device.createBuffer(.init(size: 4, usage: [.copySource, .copyDestination]))
    let output = try await device.createBuffer(.init(size: 4, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 4, usage: [.mapRead, .copyDestination]))
    let linear = try await device.createSampler(.init(minFilter: .linear, magFilter: .linear))
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(halfSampleShader), to: .wgsl)
    let pipeline = try await device.createComputePipeline(shader)
    return try await device.submitCommands([
        .write(buffer: read, offset: 0, words: [0xdead1234]),
        .copyTextureToBuffer(source: line, destination: read, bytesPerRow: 256, destinationOffset: 2),
        .read(buffer: read, byteCount: 4),
        .write(buffer: upload, offset: 0, words: [0x4200ffff]),
        .copyBufferToTexture(source: upload, layout: .init(offset: 2), destination: line, region: .init(width: 1, height: 1)),
        .copyTextureToBuffer(source: line, destination: read, bytesPerRow: 256, destinationOffset: 2),
        .read(buffer: read, byteCount: 4),
        .writeTexture(destination: line, data: [0x99, 0x00, 0x3c], layout: .init(offset: 1),
                      region: .init(width: 1, height: 1)),
        .copyTextureToBuffer(source: line, destination: read, bytesPerRow: 256, destinationOffset: 2),
        .read(buffer: read, byteCount: 4),
        .writeTexture(destination: image, data: [0x00, 0x3c, 0x00, 0x42], region: .init(width: 2, height: 1)),
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 2): output], workgroups: .init(x: 1),
                         textures: [.init(binding: 0): image], samplers: [.init(binding: 1): linear]),
        .copy(source: output, destination: read, byteCount: 4), .read(buffer: read, byteCount: 4),
    ]).wait()
}

private func halfColorResolveDiagnostic(_ device: Device) async throws -> [[UInt32]] {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl(halfColorShader)
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .wgsl)
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: .wgsl)
    let formats: [TextureFormat] = [.r16Float, .rg16Float, .rgba16Float]
    let additive = RenderBlendComponent(sourceFactor: .one, destinationFactor: .one)
    let state = RenderColorState(blend: .init(color: additive, alpha: additive))
    let pipeline = try await device.createRenderPipeline(
        vertex: vertex, fragment: fragment, colorFormat: formats[0], sampleCount: 4, colorState: state,
        additionalColorTargets: [1: .init(format: formats[1], colorState: state),
                                 2: .init(format: formats[2], colorState: state)])
    var targets: [ResourceID] = []
    var resolves: [ResourceID] = []
    for format in formats {
        targets.append(try await device.createTexture(.init(width: 1, height: 1, format: format,
                                                             usage: .renderAttachment, sampleCount: 4)))
        resolves.append(try await device.createTexture(.init(width: 1, height: 1, format: format,
                                                              usage: [.renderAttachment, .copySource])))
    }
    let read = try await device.createBuffer(.init(size: 8, usage: [.mapRead, .copyDestination]))
    let clear = RenderClearColor(red: 1, green: -2, blue: 3, alpha: 0.5)
    var commands: [BufferCommand] = [
        .render(pipeline: pipeline, target: targets[0], clear: clear, vertexCount: 3, resolve: .init(texture: resolves[0]),
                additionalColors: [
                    1: .init(texture: targets[1], loadAction: .clear(clear), resolve: .init(texture: resolves[1])),
                    2: .init(texture: targets[2], loadAction: .clear(clear), resolve: .init(texture: resolves[2])),
                ]),
        .copyTextureToBuffer(source: resolves[0], destination: read, bytesPerRow: 256, destinationOffset: 2),
        .read(buffer: read, byteCount: 4),
        .copyTextureToBuffer(source: resolves[1], destination: read, bytesPerRow: 256),
        .read(buffer: read, byteCount: 4),
    ]
    commands += [.copyTextureToBuffer(source: resolves[2], destination: read, bytesPerRow: 256),
                 .read(buffer: read, byteCount: 8)]
    if device.capabilities.features.contains(.shaderF16) {
        let native = ShaderInput.wgsl("""
            enable f16;
            struct VertexOutput {
                @builtin(position) position: vec4f,
                @location(0) tint: vec2h,
            }
            struct Colors {
                @location(0) red: f16,
                @location(1) rgba: vec4h,
            }
            struct VertexInput {
                @location(0) data: vec4h,
                @builtin(vertex_index) index: u32,
            }
            struct FragmentInput {
                @builtin(front_facing) front: bool,
                @location(0) tint: vec2h,
            }
            struct ColorValue { @location(0) rgba: vec4h, }

            @vertex fn vs(input: VertexInput) -> VertexOutput {
                let snapshot = input;
                let vertex = snapshot.data;
                return VertexOutput(vec4f(vec2f(vertex.xy), 0.0, 1.0), vertex.zw);
            }
            fn forward(input: FragmentInput) -> FragmentInput { return input; }
            @fragment fn fs(input: FragmentInput) -> Colors {
                var snapshot = forward(input);
                snapshot.tint.x += 0.5h;
                let alpha = select(0.0h, 1.0h, snapshot.front == input.front);
                return Colors(input.tint.x, vec4h(input.tint, snapshot.tint.x, alpha));
            }
            @fragment fn fsSingle(input: FragmentInput) -> @location(0) vec2h {
                var snapshot = input;
                snapshot.tint.y = 0.125h;
                return snapshot.tint;
            }
            @fragment fn fsSameType(input: ColorValue) -> ColorValue { return input; }
            """)
        let nativeVertex = try compiler.compileWebGPURenderStage(native, entryPoint: "vs")
        let nativeFragment = try compiler.compileWebGPURenderStage(native, entryPoint: "fs")
        let layout = VertexBufferLayout(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float16x4)])
        let nativePipeline = try await device.createRenderPipeline(
            vertex: nativeVertex, fragment: nativeFragment, colorFormat: .r16Float, vertexBuffers: [layout],
            additionalColorTargets: [1: .init(format: .rgba16Float)])
        guard case .spirv(let binary) = try compiler.compile(native, to: .spirv) else {
            throw GPUError.backendFailure("Expected binary SPIR-V for f16 render stages")
        }
        let roundTripVertex = try compiler.compileWebGPURenderStage(.spirv(binary), entryPoint: "vs")
        let roundTripFragment = try compiler.compileWebGPURenderStage(.spirv(binary), entryPoint: "fs")
        guard roundTripVertex.requiresShaderF16, roundTripFragment.requiresShaderF16 else {
            throw GPUError.backendFailure("Expected shader-f16 after the SPIR-V render round trip")
        }
        let roundTripPipeline = try await device.createRenderPipeline(
            vertex: roundTripVertex, fragment: roundTripFragment, colorFormat: .r16Float, vertexBuffers: [layout],
            additionalColorTargets: [1: .init(format: .rgba16Float)])
        let vertices = try await device.createBuffer(.init(size: 24, usage: [.vertex, .copyDestination]))
        let zero = RenderClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        commands.append(.write(buffer: vertices, offset: 0, words: [
            0xbc00bc00, 0x3a003400, 0xbc004200, 0x3a003400, 0x4200bc00, 0x3a003400,
        ]))
        for pipeline in [nativePipeline, roundTripPipeline] {
            commands += [
                .render(pipeline: pipeline, target: resolves[0], clear: zero, vertexCount: 3,
                        vertexBuffers: [.init(buffer: vertices)],
                        additionalColors: [1: .init(texture: resolves[2], loadAction: .clear(zero))]),
                .write(buffer: read, offset: 0, words: [0]),
                .copyTextureToBuffer(source: resolves[0], destination: read, bytesPerRow: 256),
                .read(buffer: read, byteCount: 4),
                .copyTextureToBuffer(source: resolves[2], destination: read, bytesPerRow: 256),
                .read(buffer: read, byteCount: 8),
            ]
        }
    }
    return try await device.submitCommands(commands).wait()
}

private let halfStorageWriteShader = """
    requires texture_formats_tier1;
    @group(0) @binding(0) var single: texture_storage_2d<r16float, write>;
    @group(0) @binding(1) var pair: texture_storage_2d<rg16float, write>;
    @compute @workgroup_size(2) fn run(@builtin(local_invocation_index) x: u32) {
        let p = vec2u(x, 0u);
        textureStore(single, p, vec4f(0.5 + f32(x), 99, 99, 99));
        textureStore(pair, p, vec4f(-2 + f32(x), 0.25 + f32(x), 99, 99));
    }
    """

private let halfStorageReadShader = """
    requires texture_formats_tier1, readonly_and_readwrite_storage_textures;
    @group(0) @binding(0) var single: texture_storage_2d<r16float, read>;
    @group(0) @binding(1) var pair: texture_storage_2d<rg16float, read>;
    @group(0) @binding(2) var<storage, read_write> out: array<vec4f, 2>;
    @compute @workgroup_size(2) fn run(@builtin(local_invocation_index) x: u32) {
        let p = vec2u(x, 0u);
        let rg = textureLoad(pair, p);
        out[x] = vec4f(textureLoad(single, p).x, rg.xy, rg.w);
    }
    """

private let halfStorageReadWriteShader = """
    requires texture_formats_tier1, readonly_and_readwrite_storage_textures;
    @group(0) @binding(0) var single: texture_storage_2d<r16float, read_write>;
    @compute @workgroup_size(2) fn run(@builtin(local_invocation_index) x: u32) {
        let p = vec2u(x, 0u);
        textureStore(single, p, textureLoad(single, p) + vec4f(2, 0, 0, 0));
    }
    """

private func checkMissingStorageFeature(_ device: Device, shader: ComputeShaderArtifact) async throws {
    let scope = try await device.pushErrorScope(.validation)
    do {
        _ = try await device.createComputePipeline(shader)
        throw GPUError.backendFailure("Accepted a storage texture pipeline without its required device feature")
    } catch GPUError.validation("Feature not advertised") {
        guard case .validation("Feature not advertised") = try await device.popErrorScope(scope) else {
            throw GPUError.backendFailure("Storage texture feature error did not reach the error scope")
        }
    }
}

private func halfStorageDiagnostic(_ device: Device) async throws -> [UInt32] {
    let compiler = IndependentShaderCompiler()
    let writeShader = try compiler.compileCompute(.wgsl(halfStorageWriteShader), to: .wgsl)
    let readShader = try compiler.compileCompute(.wgsl(halfStorageReadShader), to: .wgsl)
    guard device.capabilities.features.contains(.halfFloatStorageTextures) else {
        try await checkMissingStorageFeature(device, shader: writeShader)
        return []
    }
    guard device.capabilities.features.contains(.storageTextureRead) else {
        try await checkMissingStorageFeature(device, shader: readShader)
        return []
    }
    let write = try await device.createComputePipeline(writeShader)
    let read = try await device.createComputePipeline(readShader)
    let single = try await device.createTexture(.init(width: 2, height: 1, format: .r16Float, usage: .storageBinding))
    let pair = try await device.createTexture(.init(width: 2, height: 1, format: .rg16Float, usage: .storageBinding))
    let output = try await device.createBuffer(.init(size: 32, usage: [.storage, .copySource]))
    let readback = try await device.createBuffer(.init(size: 32, usage: [.mapRead, .copyDestination]))
    let images: [ComputeBindingSlot: ResourceID] = [.init(binding: 0): single, .init(binding: 1): pair]
    let groups = DispatchWorkgroups(x: 1)
    let capture: [BufferCommand] = [
        .dispatchCompute(pipeline: read, bindings: [.init(binding: 2): output], workgroups: groups, textures: images),
        .copy(source: output, destination: readback, byteCount: 32), .read(buffer: readback, byteCount: 32),
    ]
    var commands: [BufferCommand] = [
        .dispatchCompute(pipeline: write, bindings: [:], workgroups: groups, textures: images),
    ] + capture
    let updateShader = try compiler.compileCompute(.wgsl(halfStorageReadWriteShader), to: .wgsl)
    if device.capabilities.features.contains(.r16FloatStorageReadWrite) {
        let update = try await device.createComputePipeline(updateShader)
        commands += [.dispatchCompute(pipeline: update, bindings: [:], workgroups: groups, textures: [.init(binding: 0): single])]
        commands += capture
    } else {
        try await checkMissingStorageFeature(device, shader: updateShader)
    }
    return try await device.submitCommands(commands).wait().flatMap { $0 }
}

private let integer16Formats: [TextureFormat] = [.r16Uint, .rg16Uint, .rgba16Uint, .r16Sint, .rg16Sint, .rgba16Sint]

private let integer16ColorShader = """
    struct UnsignedOut { @location(0) r: u32, @location(1) rg: vec2u, @location(2) rgba: vec4u }
    struct SignedOut { @location(0) r: i32, @location(1) rg: vec2i, @location(2) rgba: vec4i }
    @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
        if i == 0u { return vec4f(-1, -1, 0, 1); }
        if i == 1u { return vec4f(3, -1, 0, 1); }
        return vec4f(-1, 3, 0, 1);
    }
    @fragment fn unsignedColor() -> UnsignedOut {
        return UnsignedOut(60000u, vec2u(65535, 300), vec4u(1, 257, 65534, 1234));
    }
    @fragment fn signedColor() -> SignedOut {
        return SignedOut(-20000i, vec2i(-32768, 300), vec4i(-1, 0, 32767, -1234));
    }
    """

private func integer16LoadShader(multisampled: Bool) -> String {
    let texture = multisampled ? "texture_multisampled_2d" : "texture_2d"
    let index = multisampled ? "3u" : "0u"
    return """
        @group(0) @binding(0) var ur: \(texture)<u32>;
        @group(0) @binding(1) var urg: \(texture)<u32>;
        @group(0) @binding(2) var urgba: \(texture)<u32>;
        @group(0) @binding(3) var sr: \(texture)<i32>;
        @group(0) @binding(4) var srg: \(texture)<i32>;
        @group(0) @binding(5) var srgba: \(texture)<i32>;
        @group(0) @binding(6) var<storage, read_write> out: array<vec4u, 6>;
        @compute @workgroup_size(1) fn run() {
            let p = vec2u(0);
            out[0] = textureLoad(ur, p, \(index));
            out[1] = textureLoad(urg, p, \(index));
            out[2] = textureLoad(urgba, p, \(index));
            out[3] = bitcast<vec4u>(textureLoad(sr, p, \(index)));
            out[4] = bitcast<vec4u>(textureLoad(srg, p, \(index)));
            out[5] = bitcast<vec4u>(textureLoad(srgba, p, \(index)));
        }
        """
}

private let integer16WriteShader = """
    requires texture_formats_tier1;
    @group(0) @binding(0) var ur: texture_storage_2d<r16uint, write>;
    @group(0) @binding(1) var urg: texture_storage_2d<rg16uint, write>;
    @group(0) @binding(2) var sr: texture_storage_2d<r16sint, write>;
    @group(0) @binding(3) var srg: texture_storage_2d<rg16sint, write>;
    @compute @workgroup_size(1) fn run() {
        let p = vec2u(0);
        textureStore(ur, p, vec4u(60000, 7, 8, 9));
        textureStore(urg, p, vec4u(65535, 300, 8, 9));
        textureStore(sr, p, vec4i(-20000, 7, 8, 9));
        textureStore(srg, p, vec4i(-32768, 300, 8, 9));
    }
    """

private let integer16ReadShader = """
    requires texture_formats_tier1, readonly_and_readwrite_storage_textures;
    @group(0) @binding(0) var ur: texture_storage_2d<r16uint, read>;
    @group(0) @binding(1) var urg: texture_storage_2d<rg16uint, read>;
    @group(0) @binding(2) var sr: texture_storage_2d<r16sint, read>;
    @group(0) @binding(3) var srg: texture_storage_2d<rg16sint, read>;
    @group(0) @binding(4) var<storage, read_write> out: array<vec4u, 4>;
    @compute @workgroup_size(1) fn run() {
        let p = vec2u(0);
        out[0] = textureLoad(ur, p);
        out[1] = textureLoad(urg, p);
        out[2] = bitcast<vec4u>(textureLoad(sr, p));
        out[3] = bitcast<vec4u>(textureLoad(srg, p));
    }
    """

private let integer16UpdateShader = """
    requires texture_formats_tier1, readonly_and_readwrite_storage_textures;
    @group(0) @binding(0) var ur: texture_storage_2d<r16uint, read_write>;
    @group(0) @binding(1) var sr: texture_storage_2d<r16sint, read_write>;
    @group(0) @binding(2) var urgba: texture_storage_2d<rgba16uint, read_write>;
    @group(0) @binding(3) var srgba: texture_storage_2d<rgba16sint, read_write>;
    @compute @workgroup_size(1) fn run() {
        let p = vec2u(0);
        textureStore(ur, p, textureLoad(ur, p) + vec4u(1000));
        textureStore(sr, p, textureLoad(sr, p) - vec4i(1000));
        textureStore(urgba, p, textureLoad(urgba, p) + vec4u(1, 2, 3, 4));
        textureStore(srgba, p, textureLoad(srgba, p) - vec4i(1, 2, 3, 4));
    }
    """

private func integer16ColorDiagnostic(_ device: Device, sampleCount: UInt32 = 1) async throws -> [UInt32] {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl(integer16ColorShader)
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .wgsl)
    let compute = try await device.createComputePipeline(compiler.compileCompute(.wgsl(integer16LoadShader(multisampled: sampleCount > 1)), to: .wgsl))
    let output = try await device.createBuffer(.init(size: 96, usage: [.storage, .copySource]))
    let readback = try await device.createBuffer(.init(size: 96, usage: [.mapRead, .copyDestination]))
    var usage: TextureUsage = [.renderAttachment, .textureBinding]
    if sampleCount == 1 { usage.insert(.copySource) }
    var textures: [ResourceID] = []
    for format in integer16Formats {
        textures.append(try await device.createTexture(.init(width: 1, height: 1, format: format,
                                                               usage: usage, sampleCount: sampleCount)))
    }
    var commands: [BufferCommand] = []
    for (base, entry) in [(0, "unsignedColor"), (3, "signedColor")] {
        let fragment = try compiler.compileRenderStage(source, entryPoint: entry, to: .wgsl)
        let pipeline = try await device.createRenderPipeline(
            vertex: vertex, fragment: fragment, colorFormat: integer16Formats[base], sampleCount: sampleCount,
            additionalColorTargets: [1: .init(format: integer16Formats[base + 1]),
                                     2: .init(format: integer16Formats[base + 2])])
        commands.append(.render(pipeline: pipeline, target: textures[base], clear: .transparentBlack, vertexCount: 3,
                                additionalColors: [1: .init(texture: textures[base + 1]), 2: .init(texture: textures[base + 2])]))
        if sampleCount == 1 {
            commands += [
                .copyTextureToBuffer(source: textures[base], destination: readback, bytesPerRow: 256, destinationOffset: 2),
                .read(buffer: readback, byteCount: 4),
            ]
        }
    }
    let bindings = Dictionary(uniqueKeysWithValues: textures.enumerated().map { (ComputeBindingSlot(binding: UInt32($0.offset)), $0.element) })
    commands += [
        .dispatchCompute(pipeline: compute, bindings: [.init(binding: 6): output], workgroups: .init(x: 1), textures: bindings),
        .copy(source: output, destination: readback, byteCount: 96), .read(buffer: readback, byteCount: 96),
    ]
    return try await device.submitCommands(commands).wait().flatMap { $0 }
}

private func integer16StorageDiagnostic(_ device: Device) async throws -> [UInt32] {
    let compiler = IndependentShaderCompiler()
    let writeShader = try compiler.compileCompute(.wgsl(integer16WriteShader), to: .wgsl)
    let readShader = try compiler.compileCompute(.wgsl(integer16ReadShader), to: .wgsl)
    guard device.capabilities.features.contains(.narrowInteger16StorageTextures) else {
        try await checkMissingStorageFeature(device, shader: writeShader)
        return []
    }
    guard device.capabilities.features.contains(.storageTextureRead) else {
        try await checkMissingStorageFeature(device, shader: readShader)
        return []
    }
    let write = try await device.createComputePipeline(writeShader)
    let read = try await device.createComputePipeline(readShader)
    var images: [ResourceID] = []
    for format: TextureFormat in [.r16Uint, .rg16Uint, .r16Sint, .rg16Sint, .rgba16Uint, .rgba16Sint] {
        images.append(try await device.createTexture(.init(width: 1, height: 1, format: format, usage: [.storageBinding, .copySource])))
    }
    let output = try await device.createBuffer(.init(size: 64, usage: [.storage, .copySource]))
    let readback = try await device.createBuffer(.init(size: 64, usage: [.mapRead, .copyDestination]))
    let bindings = Dictionary(uniqueKeysWithValues: images.prefix(4).enumerated().map {
        (ComputeBindingSlot(binding: UInt32($0.offset)), $0.element)
    })
    var commands: [BufferCommand] = [
        .dispatchCompute(pipeline: write, bindings: [:], workgroups: .init(x: 1), textures: bindings),
        .dispatchCompute(pipeline: read, bindings: [.init(binding: 4): output], workgroups: .init(x: 1), textures: bindings),
        .copy(source: output, destination: readback, byteCount: 64), .read(buffer: readback, byteCount: 64),
    ]
    let updateShader = try compiler.compileCompute(.wgsl(integer16UpdateShader), to: .wgsl)
    if device.capabilities.features.contains(.integer16StorageReadWrite) {
        let update = try await device.createComputePipeline(updateShader)
        commands += [
            .dispatchCompute(pipeline: update, bindings: [:], workgroups: .init(x: 1),
                             textures: [.init(binding: 0): images[0], .init(binding: 1): images[2],
                                        .init(binding: 2): images[4], .init(binding: 3): images[5]]),
            .write(buffer: readback, offset: 0, words: [0]),
        ]
        for index in [0, 2, 4, 5] {
            commands += [
                .copyTextureToBuffer(source: images[index], destination: readback, bytesPerRow: 256,
                                     destinationOffset: index < 4 ? 2 : 0),
                .read(buffer: readback, byteCount: index < 4 ? 4 : 8),
            ]
        }
    } else {
        try await checkMissingStorageFeature(device, shader: updateShader)
    }
    return try await device.submitCommands(commands).wait().flatMap { $0 }
}

private let normalized16Formats: [(TextureFormat, String)] = [
    (.r16Unorm, "r16unorm"), (.rg16Unorm, "rg16unorm"), (.rgba16Unorm, "rgba16unorm"),
    (.r16Snorm, "r16snorm"), (.rg16Snorm, "rg16snorm"), (.rgba16Snorm, "rgba16snorm"),
]

private func normalized16StorageShader(read: Bool, signed: Bool) -> String {
    let base = signed ? 3 : 0
    let declarations = normalized16Formats[base..<(base + 3)].enumerated().map { index, format in
        "@group(0) @binding(\(index)) var t\(index): texture_storage_2d<\(format.1), \(read ? "read" : "write")>;"
    }.joined(separator: "\n")
    let output = read ? "@group(0) @binding(3) var<storage, read_write> out: array<vec4f, 6>;" : ""
    let values = [
        "f32(x), 0, 0, 0", "0.25, 0.75, 0, 0", "-0.25, 0.25, 0.75, 2",
        "-0.5, 0, 0, 0", "-1, 1, 0, 0", "-2, -0.25, 0.25, 2",
    ]
    let statements = values[base..<(base + 3)].enumerated().map { index, value in
        read ? "out[\(base + index)] = textureLoad(t\(index), vec2u(1, 0));" : "textureStore(t\(index), vec2u(x, 0), vec4f(\(value)));"
    }.joined(separator: "\n")
    return """
        requires texture_formats_tier1;
        \(declarations)
        \(output)
        @compute @workgroup_size(\(read ? 1 : 2)) fn run(@builtin(local_invocation_index) x: u32) {
            \(statements)
        }
        """
}

private func normalized16SampleShader(linear: Bool) -> String {
    let declarations = normalized16Formats.indices.map { index in
        "@group(0) @binding(\(index)) var t\(index): texture_2d<f32>;"
    }.joined(separator: "\n")
    let x = linear ? "0.5" : "0.75"
    let statements = normalized16Formats.indices.map { index in
        "out[\(index)] = textureSampleLevel(t\(index), state, vec2f(\(x), 0.5), 0.0);"
    }.joined(separator: "\n")
    return """
        \(declarations)
        @group(0) @binding(6) var state: sampler;
        @group(0) @binding(7) var<storage, read_write> out: array<vec4f, 6>;
        @compute @workgroup_size(1) fn run() { \(statements) }
        """
}

private let normalized16ColorShader = """
    struct Output { @location(0) red: f32, @location(1) rg: vec2f, @location(2) color: vec4f }
    @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
        if i == 0u { return vec4f(-1, -1, 0, 1); }
        if i == 1u { return vec4f(3, -1, 0, 1); }
        return vec4f(-1, 3, 0, 1);
    }
    @fragment fn fs() -> Output {
        return Output(0.5, vec2f(-0.5, 0.75), vec4f(0.25, 0.5, 0.5, 0.5));
    }
    """

private func normalized16StorageDiagnostic(_ device: Device) async throws -> [UInt32] {
    let compiler = IndependentShaderCompiler()
    guard device.capabilities.features.contains(.normalized16Textures) else {
        let shader = try compiler.compileCompute(.wgsl(normalized16StorageShader(read: false, signed: false)), to: .wgsl)
        try await checkMissingStorageFeature(device, shader: shader)
        return []
    }
    guard device.capabilities.features.contains(.storageTextureRead) else {
        let shader = try compiler.compileCompute(.wgsl(normalized16StorageShader(read: true, signed: false)), to: .wgsl)
        try await checkMissingStorageFeature(device, shader: shader)
        return []
    }
    var textures: [ResourceID] = []
    for (format, _) in normalized16Formats {
        textures.append(try await device.createTexture(.init(width: 2, height: 1, format: format,
                                                               usage: [.storageBinding, .textureBinding, .copySource])))
    }
    let images = Dictionary(uniqueKeysWithValues: textures.enumerated().map { (ComputeBindingSlot(binding: UInt32($0.offset)), $0.element) })
    let output = try await device.createBuffer(.init(size: 96, usage: [.storage, .copySource]))
    let readback = try await device.createBuffer(.init(size: 96, usage: [.mapRead, .copyDestination]))
    let capture: [BufferCommand] = [.copy(source: output, destination: readback, byteCount: 96), .read(buffer: readback, byteCount: 96)]
    var commands: [BufferCommand] = []
    for signed in [false, true] {
        let writeShader = try compiler.compileCompute(.wgsl(normalized16StorageShader(read: false, signed: signed)), to: .wgsl)
        let readShader = try compiler.compileCompute(.wgsl(normalized16StorageShader(read: true, signed: signed)), to: .wgsl)
        let write = try await device.createComputePipeline(writeShader)
        let read = try await device.createComputePipeline(readShader)
        let base = signed ? 3 : 0
        let storage = Dictionary(uniqueKeysWithValues: textures[base..<(base + 3)].enumerated().map {
            (ComputeBindingSlot(binding: UInt32($0.offset)), $0.element)
        })
        commands += [
            .dispatchCompute(pipeline: write, bindings: [:], workgroups: .init(x: 1), textures: storage),
            .dispatchCompute(pipeline: read, bindings: [.init(binding: 3): output], workgroups: .init(x: 1), textures: storage),
        ]
    }
    commands += [
        .copyTextureToBuffer(source: textures[0], destination: readback, bytesPerRow: 256),
        .read(buffer: readback, byteCount: 4),
    ] + capture
    for linear in [false, true] {
        let shader = try compiler.compileCompute(.wgsl(normalized16SampleShader(linear: linear)), to: .wgsl)
        let samplerSlot = ComputeBindingSlot(binding: 6)
        let pipeline = try await device.createComputePipeline(shader, nonFilteringSamplers: linear ? [] : [samplerSlot])
        let sampler = try await device.createSampler(.init(minFilter: linear ? .linear : .nearest, magFilter: linear ? .linear : .nearest))
        let dispatch = BufferCommand.dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 7): output],
                                                      workgroups: .init(x: 1), textures: images, samplers: [samplerSlot: sampler])
        if linear && !device.capabilities.features.contains(.normalized16Filterable) {
            let scope = try await device.pushErrorScope(.validation)
            do {
                _ = try await device.submitCommands([dispatch])
                throw GPUError.backendFailure("Accepted filtered normalized16 texture without device support")
            } catch GPUError.validation("Normalized 16-bit sampling requires normalized16Filterable") {
                guard case .validation = try await device.popErrorScope(scope) else {
                    throw GPUError.backendFailure("Normalized16 filtering error did not reach the error scope")
                }
                continue
            }
        }
        commands += [dispatch] + capture
    }
    return try await device.submitCommands(commands).wait().flatMap { $0 }
}

private func normalized16ColorDiagnostic(_ device: Device) async throws -> [UInt32] {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl(normalized16ColorShader)
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .wgsl)
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: .wgsl)
    let resolve = device.capabilities.features.contains(.normalized16Resolve)
    let sampleCount: UInt32 = resolve ? 4 : 1
    let formats: [TextureFormat] = [.r16Unorm, .rg16Snorm, .rgba16Unorm]
    let additive = RenderBlendComponent(sourceFactor: .one, destinationFactor: .one)
    let blend = RenderColorState(blend: .init(color: additive, alpha: additive))
    func makePipeline() async throws -> RenderPipeline {
        try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: formats[0],
            sampleCount: sampleCount, colorState: blend, additionalColorTargets: [
                1: .init(format: formats[1], colorState: blend), 2: .init(format: formats[2], colorState: blend),
            ])
    }
    if !device.capabilities.features.contains(.normalized16Textures) {
        let scope = try await device.pushErrorScope(.validation)
        do {
            _ = try await makePipeline()
            throw GPUError.backendFailure("Accepted a normalized16 color pipeline without its device feature")
        } catch GPUError.validation("Feature not advertised") {
            guard case .validation = try await device.popErrorScope(scope) else {
                throw GPUError.backendFailure("Normalized16 color error did not reach the error scope")
            }
            return []
        }
    }
    let pipeline = try await makePipeline()
    var targets: [ResourceID] = []
    var outputs: [ResourceID] = []
    for format in formats {
        let target = try await device.createTexture(.init(width: 1, height: 1, format: format,
            usage: resolve ? .renderAttachment : [.renderAttachment, .copySource], sampleCount: sampleCount))
        targets.append(target)
        outputs.append(resolve ? try await device.createTexture(.init(width: 1, height: 1, format: format,
                                      usage: [.renderAttachment, .copySource])) : target)
    }
    let readback = try await device.createBuffer(.init(size: 8, usage: [.mapRead, .copyDestination]))
    let clear = RenderClearColor(red: 0.25, green: 0, blue: 0.5, alpha: 0.25)
    return try await device.submitCommands([
        .render(pipeline: pipeline, target: targets[0], clear: clear, vertexCount: 3,
                resolve: resolve ? .init(texture: outputs[0]) : nil, additionalColors: [
                    1: .init(texture: targets[1], loadAction: .clear(clear), resolve: resolve ? .init(texture: outputs[1]) : nil),
                    2: .init(texture: targets[2], loadAction: .clear(clear), resolve: resolve ? .init(texture: outputs[2]) : nil),
                ]),
        .copyTextureToBuffer(source: outputs[0], destination: readback, bytesPerRow: 256, destinationOffset: 2),
        .read(buffer: readback, byteCount: 4),
        .copyTextureToBuffer(source: outputs[1], destination: readback, bytesPerRow: 256), .read(buffer: readback, byteCount: 4),
        .copyTextureToBuffer(source: outputs[2], destination: readback, bytesPerRow: 256), .read(buffer: readback, byteCount: 8),
    ]).wait().flatMap { $0 }
}

private let packedFormats: [TextureFormat] = [.rgb10a2Unorm, .rgb10a2Uint, .rg11b10Ufloat]

private let packedColorShader = """
    @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
        if i == 0u { return vec4f(-1, -1, 0, 1); }
        if i == 1u { return vec4f(3, -1, 0, 1); }
        return vec4f(-1, 3, 0, 1);
    }
    @fragment fn normalized() -> @location(0) vec4f { return vec4f(0, 0, 1, 2.0 / 3.0); }
    @fragment fn integer() -> @location(0) vec4u { return vec4u(1023, 512, 3, 2); }
    @fragment fn hdr() -> @location(0) vec3f { return vec3f(1, 2, 4); }
    """

private let packedSampleShader = """
    @group(0) @binding(0) var image: texture_2d<f32>;
    @group(0) @binding(1) var state: sampler;
    @group(0) @binding(2) var<storage, read_write> out: vec4f;
    @compute @workgroup_size(1) fn run() { out = textureSampleLevel(image, state, vec2f(0.5), 0.0); }
    """

private let packedWriteShader = """
    requires texture_formats_tier1;
    @group(0) @binding(0) var normalized: texture_storage_2d<rgb10a2unorm, write>;
    @group(0) @binding(1) var integer: texture_storage_2d<rgb10a2uint, write>;
    @group(0) @binding(2) var hdr: texture_storage_2d<rg11b10ufloat, write>;
    @compute @workgroup_size(1) fn run() {
        let p = vec2u(0);
        textureStore(normalized, p, vec4f(1, 0, 1, 2.0 / 3.0));
        textureStore(integer, p, vec4u(1023, 512, 3, 2));
        textureStore(hdr, p, vec4f(1, 2, 4, 99));
    }
    """

private let packedReadShader = """
    requires texture_formats_tier1;
    @group(0) @binding(0) var normalized: texture_storage_2d<rgb10a2unorm, read>;
    @group(0) @binding(1) var integer: texture_storage_2d<rgb10a2uint, read>;
    @group(0) @binding(2) var hdr: texture_storage_2d<rg11b10ufloat, read>;
    @group(0) @binding(3) var<storage, read_write> out: array<vec4u, 3>;
    @compute @workgroup_size(1) fn run() {
        let p = vec2u(0);
        out[0] = bitcast<vec4u>(textureLoad(normalized, p));
        out[1] = textureLoad(integer, p);
        out[2] = bitcast<vec4u>(textureLoad(hdr, p));
    }
    """

private func packedColorDiagnostic(_ device: Device) async throws -> [UInt32] {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl(packedColorShader)
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .wgsl)
    let readback = try await device.createBuffer(.init(size: 16, usage: [.mapRead, .copyDestination]))
    let additive = RenderBlendComponent(sourceFactor: .one, destinationFactor: .one)
    let blend = RenderColorState(blend: .init(color: additive, alpha: additive))
    var commands: [BufferCommand] = []
    for (format, entry) in zip(packedFormats, ["normalized", "integer", "hdr"]) {
        if format == .rg11b10Ufloat && !device.capabilities.features.contains(.rg11b10UfloatRenderable) {
            let scope = try await device.pushErrorScope(.validation)
            do {
                _ = try await device.createTexture(.init(width: 1, height: 1, format: format, usage: .renderAttachment))
                throw GPUError.backendFailure("Accepted RG11B10 render attachment without the device feature")
            } catch GPUError.validation("Feature not advertised") {
                guard case .validation = try await device.popErrorScope(scope) else {
                    throw GPUError.backendFailure("RG11B10 renderability error did not reach the error scope")
                }
                continue
            }
        }
        let isInteger = format == .rgb10a2Uint
        let sampleCount: UInt32 = isInteger ? 1 : 4
        let fragment = try compiler.compileRenderStage(source, entryPoint: entry, to: .wgsl)
        let pipeline = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: format,
                                                             sampleCount: sampleCount, colorState: isInteger ? .init() : blend)
        let output = try await device.createTexture(.init(width: 1, height: 1, format: format, usage: [.renderAttachment, .copySource]))
        let target = isInteger ? output : try await device.createTexture(.init(
            width: 1, height: 1, format: format, usage: .renderAttachment, sampleCount: sampleCount))
        let clear = isInteger ? .transparentBlack :
            RenderClearColor(red: format == .rgb10a2Unorm ? 1 : 0.25, green: 0, blue: 0, alpha: 0)
        commands += [
            .render(pipeline: pipeline, target: target, clear: clear, vertexCount: 3,
                    resolve: isInteger ? nil : .init(texture: output)),
            .copyTextureToBuffer(source: output, destination: readback, bytesPerRow: 256), .read(buffer: readback, byteCount: 4),
        ]
    }
    let hdr = try await device.createTexture(.init(width: 1, height: 1, format: .rg11b10Ufloat,
                                                     usage: [.copyDestination, .textureBinding]))
    let color = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
    let sampler = try await device.createSampler(.init(minFilter: .linear, magFilter: .linear))
    let compute = try await device.createComputePipeline(compiler.compileCompute(.wgsl(packedSampleShader), to: .wgsl))
    let capture: [BufferCommand] = [
        .dispatchCompute(pipeline: compute, bindings: [.init(binding: 2): color], workgroups: .init(x: 1),
                         textures: [.init(binding: 0): hdr], samplers: [.init(binding: 1): sampler]),
        .copy(source: color, destination: readback, byteCount: 16), .read(buffer: readback, byteCount: 16),
    ]
    commands += capture + [
        .writeTexture(destination: hdr, data: [0xc0, 0x03, 0x20, 0x88], region: .init(width: 1, height: 1)),
    ] + capture
    return try await device.submitCommands(commands).wait().flatMap { $0 }
}

private func packedStorageDiagnostic(_ device: Device) async throws -> [UInt32] {
    let compiler = IndependentShaderCompiler()
    let writeShader = try compiler.compileCompute(.wgsl(packedWriteShader), to: .wgsl)
    let readShader = try compiler.compileCompute(.wgsl(packedReadShader), to: .wgsl)
    guard device.capabilities.features.contains(.packedColorStorageTextures) else {
        try await checkMissingStorageFeature(device, shader: writeShader)
        return []
    }
    guard device.capabilities.features.contains(.storageTextureRead) else {
        try await checkMissingStorageFeature(device, shader: readShader)
        return []
    }
    let write = try await device.createComputePipeline(writeShader)
    let read = try await device.createComputePipeline(readShader)
    var images: [ComputeBindingSlot: ResourceID] = [:]
    for (index, format) in packedFormats.enumerated() {
        images[.init(binding: UInt32(index))] = try await device.createTexture(.init(width: 1, height: 1, format: format,
                                                                        usage: [.storageBinding, .copySource]))
    }
    let output = try await device.createBuffer(.init(size: 48, usage: [.storage, .copySource]))
    let readback = try await device.createBuffer(.init(size: 48, usage: [.mapRead, .copyDestination]))
    var commands: [BufferCommand] = [.dispatchCompute(pipeline: write, bindings: [:], workgroups: .init(x: 1), textures: images)]
    for index in packedFormats.indices {
        let image = images[.init(binding: UInt32(index))]!
        commands += [.copyTextureToBuffer(source: image, destination: readback, bytesPerRow: 256),
                     .read(buffer: readback, byteCount: 4)]
    }
    commands += [
        .dispatchCompute(pipeline: read, bindings: [.init(binding: 3): output], workgroups: .init(x: 1), textures: images),
        .copy(source: output, destination: readback, byteCount: 48), .read(buffer: readback, byteCount: 48),
    ]
    return try await device.submitCommands(commands).wait().flatMap { $0 }
}

private let srgbReadShader = """
    @group(0) @binding(0) var rgba: texture_2d<f32>;
    @group(0) @binding(1) var bgra: texture_2d<f32>;
    @group(0) @binding(2) var linear: texture_2d<f32>;
    @group(0) @binding(3) var<storage, read_write> out: vec4f;
    @compute @workgroup_size(1) fn run() {
        let p = vec2u(0);
        let a = textureLoad(rgba, p, 0u);
        let b = textureLoad(bgra, p, 0u);
        let c = textureLoad(linear, p, 0u);
        out = vec4f(a.x, b.y, a.w, c.x);
    }
    """

private let srgbRenderShader = """
    struct Colors { @location(0) rgba: vec4f, @location(1) bgra: vec4f }
    @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
        if i == 0u { return vec4f(-1, -1, 0, 1); }
        if i == 1u { return vec4f(3, -1, 0, 1); }
        return vec4f(-1, 3, 0, 1);
    }
    @fragment fn srgb() -> Colors {
        let color = vec4f(0.25, 0.5, 0.75, 0.5);
        return Colors(color, color);
    }
    @fragment fn hdr() -> @location(0) vec3f { return vec3f(1.015625, 2.015625, 4.015625); }
    """

private let sharedExponentSampleShader = """
    @group(0) @binding(0) var image: texture_2d<f32>;
    @group(0) @binding(1) var state: sampler;
    @group(0) @binding(2) var<storage, read_write> out: vec4f;
    @compute @workgroup_size(1) fn run() { out = textureSampleLevel(image, state, vec2f(0.5), 0.0); }
    """

private func srgbDiagnostic(_ device: Device) async throws -> [UInt32] {
    let compiler = IndependentShaderCompiler()
    let readback = try await device.createBuffer(.init(size: 16, usage: [.mapRead, .copyDestination]))
    let output = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
    let usage: TextureUsage = [.copySource, .copyDestination, .textureBinding]
    var images: [ResourceID] = []
    var commands: [BufferCommand] = []
    for (linear, srgb, bytes) in [(TextureFormat.rgba8Unorm, TextureFormat.rgba8UnormSrgb, [UInt8(128), 64, 32, 128]),
                                  (.bgra8Unorm, .bgra8UnormSrgb, [32, 64, 128, 128])] {
        let source = try await device.createTexture(.init(width: 1, height: 1, format: linear, usage: usage))
        let encoded = try await device.createTexture(.init(width: 1, height: 1, format: srgb, usage: usage))
        let copy = try await device.createTexture(.init(width: 1, height: 1, format: linear, usage: usage))
        images += [encoded, copy]
        let region = TextureCopyRegion(width: 1, height: 1)
        commands += [
            .writeTexture(destination: source, data: bytes, region: region),
            .copyTextureToTexture(source: source, destination: encoded, sourceRegion: region),
            .copyTextureToTexture(source: encoded, destination: copy, sourceRegion: region),
            .copyTextureToBuffer(source: encoded, destination: readback, bytesPerRow: 256), .read(buffer: readback, byteCount: 4),
            .copyTextureToBuffer(source: copy, destination: readback, bytesPerRow: 256), .read(buffer: readback, byteCount: 4),
        ]
    }
    let compute = try await device.createComputePipeline(compiler.compileCompute(.wgsl(srgbReadShader), to: .wgsl))
    commands += [
        .dispatchCompute(pipeline: compute, bindings: [.init(binding: 3): output], workgroups: .init(x: 1),
                         textures: [.init(binding: 0): images[0], .init(binding: 1): images[2], .init(binding: 2): images[1]]),
        .copy(source: output, destination: readback, byteCount: 16), .read(buffer: readback, byteCount: 16),
    ]
    let source = ShaderInput.wgsl(srgbRenderShader)
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .wgsl)
    let fragment = try compiler.compileRenderStage(source, entryPoint: "srgb", to: .wgsl)
    let blend = RenderColorState(blend: .init())
    let render = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8UnormSrgb,
        sampleCount: 4, colorState: blend, additionalColorTargets: [1: .init(format: .bgra8UnormSrgb, colorState: blend)])
    let rgba = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8UnormSrgb, usage: .renderAttachment, sampleCount: 4))
    let bgra = try await device.createTexture(.init(width: 1, height: 1, format: .bgra8UnormSrgb, usage: .renderAttachment, sampleCount: 4))
    let resolvedRGBA = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8UnormSrgb, usage: [.renderAttachment, .copySource]))
    let resolvedBGRA = try await device.createTexture(.init(width: 1, height: 1, format: .bgra8UnormSrgb, usage: [.renderAttachment, .copySource]))
    commands += [
        .render(pipeline: render, target: rgba, clear: .transparentBlack, vertexCount: 3, resolve: .init(texture: resolvedRGBA),
                additionalColors: [1: .init(texture: bgra, loadAction: .clear(.transparentBlack), resolve: .init(texture: resolvedBGRA))]),
        .copyTextureToBuffer(source: resolvedRGBA, destination: readback, bytesPerRow: 256), .read(buffer: readback, byteCount: 4),
        .copyTextureToBuffer(source: resolvedBGRA, destination: readback, bytesPerRow: 256), .read(buffer: readback, byteCount: 4),
    ]
    return try await device.submitCommands(commands).wait().flatMap { $0 }
}

private func sharedExponentDiagnostic(_ device: Device) async throws -> [UInt32] {
    let compiler = IndependentShaderCompiler()
    let usage: TextureUsage = [.textureBinding, .copySource, .copyDestination]
    let image = try await device.createTexture(.init(width: 2, height: 1, format: .rgb9e5Ufloat, usage: usage))
    let clone = try await device.createTexture(.init(width: 2, height: 1, format: .rgb9e5Ufloat, usage: usage))
    let data = [UInt32(0x94050241), 0xa5054221].withUnsafeBytes { Array($0) }
    let readback = try await device.createBuffer(.init(size: 16, usage: [.mapRead, .copyDestination]))
    let output = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
    let sampler = try await device.createSampler(.init(minFilter: .linear, magFilter: .linear))
    let compute = try await device.createComputePipeline(compiler.compileCompute(.wgsl(sharedExponentSampleShader), to: .wgsl))
    let sample: [BufferCommand] = [
        .dispatchCompute(pipeline: compute, bindings: [.init(binding: 2): output], workgroups: .init(x: 1),
                         textures: [.init(binding: 0): clone], samplers: [.init(binding: 1): sampler]),
        .copy(source: output, destination: readback, byteCount: 16), .read(buffer: readback, byteCount: 16),
    ]
    let region = TextureCopyRegion(width: 2, height: 1)
    let commands = sample + [
        .writeTexture(destination: image, data: data, region: region),
        .copyTextureToTexture(source: image, destination: clone, sourceRegion: region),
        .copyTextureToBuffer(source: clone, destination: readback, bytesPerRow: 256), .read(buffer: readback, byteCount: 8),
    ] + sample
    let scope = try await device.pushErrorScope(.validation)
    do {
        _ = try await device.createTexture(.init(width: 1, height: 1, format: .rgb9e5Ufloat, usage: .renderAttachment))
        throw GPUError.backendFailure("Browser accepted RGB9E5 as a render attachment")
    } catch GPUError.validation("Feature not advertised") {
        guard case .validation = try await device.popErrorScope(scope) else {
            throw GPUError.backendFailure("RGB9E5 renderability error did not reach the error scope")
        }
    }
    return try await device.submitCommands(commands).wait().flatMap { $0 }
}

private struct ByteTextureCase: Sendable {
    let format: TextureFormat
    let name: String
    let scalar: String
    let vector: String
    let bytes: [UInt8]
    let value: String
}

private let byteTextureCases: [ByteTextureCase] = [
    .init(format: .r8Unorm, name: "r8unorm", scalar: "f32", vector: "vec4f", bytes: [128], value: "128.0/255.0, 0, 0, 0"),
    .init(format: .rg8Unorm, name: "rg8unorm", scalar: "f32", vector: "vec4f", bytes: [64, 192], value: "64.0/255.0, 192.0/255.0, 0, 0"),
    .init(format: .r8Snorm, name: "r8snorm", scalar: "f32", vector: "vec4f", bytes: [192], value: "-64.0/127.0, 0, 0, 0"),
    .init(format: .rg8Snorm, name: "rg8snorm", scalar: "f32", vector: "vec4f", bytes: [193, 127], value: "-63.0/127.0, 1, 0, 0"),
    .init(format: .r8Uint, name: "r8uint", scalar: "u32", vector: "vec4u", bytes: [250], value: "250, 0, 0, 0"),
    .init(format: .rg8Uint, name: "rg8uint", scalar: "u32", vector: "vec4u", bytes: [4, 200], value: "4, 200, 0, 0"),
    .init(format: .r8Sint, name: "r8sint", scalar: "i32", vector: "vec4i", bytes: [129], value: "-127, 0, 0, 0"),
    .init(format: .rg8Sint, name: "rg8sint", scalar: "i32", vector: "vec4i", bytes: [128, 100], value: "-128, 100, 0, 0"),
]

private func byteReadShader(storage: Bool, base: Int, count: Int) -> String {
    let declarations = byteTextureCases[base..<(base + count)].enumerated().map { index, entry in
        let type = storage ? "texture_storage_2d<\(entry.name), read>" : "texture_2d<\(entry.scalar)>"
        return "@group(0) @binding(\(index)) var t\(index): \(type);"
    }.joined(separator: "\n")
    let expressions = (0..<count).map { index in
        "out[\(base + index)] = vec4f(textureLoad(t\(index), vec2u(0)\(storage ? "" : ", 0u")));"
    }.joined(separator: "\n")
    return """
        \(storage ? "requires texture_formats_tier1;" : "")
        \(declarations)
        @group(0) @binding(\(count)) var<storage, read_write> out: array<vec4f, 8>;
        @compute @workgroup_size(1) fn run() { \(expressions) }
        """
}

private func byteWriteShader(base: Int) -> String {
    let cases = byteTextureCases[base..<(base + 4)]
    let declarations = cases.enumerated().map { index, entry in
        "@group(0) @binding(\(index)) var t\(index): texture_storage_2d<\(entry.name), write>;"
    }.joined(separator: "\n")
    let statements = cases.enumerated().map { index, entry in
        "textureStore(t\(index), vec2u(0), \(entry.vector)(\(entry.value)));"
    }.joined(separator: "\n")
    return """
        requires texture_formats_tier1;
        \(declarations)
        @compute @workgroup_size(1) fn run() { \(statements) }
        """
}

private let byteUpdateShader = """
    requires texture_formats_tier1;
    @group(0) @binding(0) var narrow: texture_storage_2d<r8uint, read_write>;
    @group(0) @binding(1) var rgba: texture_storage_2d<rgba8unorm, read_write>;
    @compute @workgroup_size(1) fn run() {
        let p = vec2u(0);
        textureStore(narrow, p, textureLoad(narrow, p) + vec4u(3, 0, 0, 0));
        textureStore(rgba, p, textureLoad(rgba, p) + vec4f(1, 0, 0, 1));
    }
    """

private let byteColorShader = """
    struct Targets { @location(0) r: f32, @location(1) rg: vec2f, @location(2) rgba: vec4f }
    @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
        if i == 0u { return vec4f(-1, -1, 0, 1); }
        if i == 1u { return vec4f(3, -1, 0, 1); }
        return vec4f(-1, 3, 0, 1);
    }
    @fragment fn unorm() -> Targets {
        return Targets(128.0/255.0, vec2f(64.0/255.0, 192.0/255.0), vec4f(1, 0, 0, 1));
    }
    @fragment fn snorm() -> Targets {
        return Targets(-64.0/127.0, vec2f(-63.0/127.0, 1), vec4f(-64.0/127.0, 1, 0, 1));
    }
    """

private func byteTransferDiagnostic(_ device: Device) async throws -> [UInt32] {
    let usage: TextureUsage = [.copySource, .copyDestination, .textureBinding]
    let upload = try await device.createBuffer(.init(size: 4, usage: [.copySource, .copyDestination]))
    let output = try await device.createBuffer(.init(size: 128, usage: [.storage, .copySource]))
    let readback = try await device.createBuffer(.init(size: 128, usage: [.mapRead, .copyDestination]))
    let shader = try IndependentShaderCompiler().compileCompute(
        .wgsl(byteReadShader(storage: false, base: 0, count: 8)), to: .wgsl)
    let compute = try await device.createComputePipeline(shader)
    var images: [ComputeBindingSlot: ResourceID] = [:]
    var commands: [BufferCommand] = [.write(buffer: upload, offset: 0, words: [0x00008000])]
    for (index, entry) in byteTextureCases.enumerated() {
        let image = try await device.createTexture(.init(width: 1, height: 1, format: entry.format, usage: usage))
        images[.init(binding: UInt32(index))] = image
        let region = TextureCopyRegion(width: 1, height: 1)
        if index == 0 {
            commands += [
                .copyBufferToTexture(source: upload, layout: .init(offset: 1), destination: image, region: region),
                .write(buffer: readback, offset: 0, words: [0xaabbccdd]),
                .copyTextureToBuffer(source: image, destination: readback, bytesPerRow: 256, destinationOffset: 1),
                .read(buffer: readback, byteCount: 4),
            ]
        } else {
            commands.append(.writeTexture(destination: image, data: [0xee] + entry.bytes, layout: .init(offset: 1), region: region))
        }
    }
    commands += [
        .dispatchCompute(pipeline: compute, bindings: [.init(binding: 8): output], workgroups: .init(x: 1), textures: images),
        .copy(source: output, destination: readback, byteCount: 128), .read(buffer: readback, byteCount: 128),
    ]
    return try await device.submitCommands(commands).wait().flatMap { $0 }
}

private func byteColorDiagnostic(_ device: Device) async throws -> [UInt32] {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl(byteColorShader)
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .wgsl)
    let readback = try await device.createBuffer(.init(size: 4, usage: [.mapRead, .copyDestination]))
    var commands: [BufferCommand] = []
    for snorm in [false, true] {
        if snorm && !device.capabilities.features.contains(.snorm8Renderable) {
            let scope = try await device.pushErrorScope(.validation)
            do {
                _ = try await device.createTexture(.init(width: 1, height: 1, format: .r8Snorm, usage: .renderAttachment))
                throw GPUError.backendFailure("Accepted an SNORM8 render attachment without its device feature")
            } catch GPUError.validation("Feature not advertised") {
                guard case .validation = try await device.popErrorScope(scope) else {
                    throw GPUError.backendFailure("SNORM8 renderability error did not reach the error scope")
                }
                continue
            }
        }
        let formats: [TextureFormat] = snorm ? [.r8Snorm, .rg8Snorm, .rgba8Snorm] : [.r8Unorm, .rg8Unorm, .rgba8Unorm]
        let fragment = try compiler.compileRenderStage(source, entryPoint: snorm ? "snorm" : "unorm", to: .wgsl)
        let blend = RenderColorState(blend: .init())
        let pipeline = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: formats[0],
            sampleCount: 4, colorState: blend, additionalColorTargets: [
                1: .init(format: formats[1], colorState: blend), 2: .init(format: formats[2], colorState: blend),
            ])
        var targets: [ResourceID] = []
        var resolves: [ResourceID] = []
        for format in formats {
            targets.append(try await device.createTexture(.init(width: 1, height: 1, format: format,
                                                                usage: .renderAttachment, sampleCount: 4)))
            resolves.append(try await device.createTexture(.init(width: 1, height: 1, format: format,
                                                                 usage: [.renderAttachment, .copySource])))
        }
        commands.append(.render(pipeline: pipeline, target: targets[0], clear: .transparentBlack, vertexCount: 3,
            resolve: .init(texture: resolves[0]), additionalColors: [
                1: .init(texture: targets[1], loadAction: .clear(.transparentBlack), resolve: .init(texture: resolves[1])),
                2: .init(texture: targets[2], loadAction: .clear(.transparentBlack), resolve: .init(texture: resolves[2])),
            ]))
        for (image, offset) in zip(resolves, [UInt64(3), 2, 0]) {
            commands += [
                .write(buffer: readback, offset: 0, words: [0]),
                .copyTextureToBuffer(source: image, destination: readback, bytesPerRow: 256, destinationOffset: offset),
                .read(buffer: readback, byteCount: 4),
            ]
        }
    }
    return try await device.submitCommands(commands).wait().flatMap { $0 }
}

private func byteStorageDiagnostic(_ device: Device) async throws -> [UInt32] {
    let compiler = IndependentShaderCompiler()
    guard device.capabilities.features.contains(.narrow8BitStorageTextures) else {
        let shader = try compiler.compileCompute(.wgsl(byteWriteShader(base: 0)), to: .wgsl)
        try await checkMissingStorageFeature(device, shader: shader)
        return []
    }
    guard device.capabilities.features.contains(.storageTextureRead) else {
        let shader = try compiler.compileCompute(.wgsl(byteReadShader(storage: true, base: 0, count: 4)), to: .wgsl)
        try await checkMissingStorageFeature(device, shader: shader)
        return []
    }
    var images: [ResourceID] = []
    for entry in byteTextureCases {
        images.append(try await device.createTexture(.init(width: 1, height: 1, format: entry.format,
                                                             usage: [.storageBinding, .copySource])))
    }
    let output = try await device.createBuffer(.init(size: 128, usage: [.storage, .copySource]))
    let readback = try await device.createBuffer(.init(size: 128, usage: [.mapRead, .copyDestination]))
    var commands: [BufferCommand] = []
    for base in [0, 4] {
        let write = try await device.createComputePipeline(compiler.compileCompute(.wgsl(byteWriteShader(base: base)), to: .wgsl))
        let read = try await device.createComputePipeline(compiler.compileCompute(
            .wgsl(byteReadShader(storage: true, base: base, count: 4)), to: .wgsl))
        let bindings = Dictionary(uniqueKeysWithValues: images[base..<(base + 4)].enumerated().map {
            (ComputeBindingSlot(binding: UInt32($0.offset)), $0.element)
        })
        commands += [
            .dispatchCompute(pipeline: write, bindings: [:], workgroups: .init(x: 1), textures: bindings),
            .dispatchCompute(pipeline: read, bindings: [.init(binding: 4): output], workgroups: .init(x: 1), textures: bindings),
        ]
    }
    commands += [.copy(source: output, destination: readback, byteCount: 128), .read(buffer: readback, byteCount: 128)]
    let updateShader = try compiler.compileCompute(.wgsl(byteUpdateShader), to: .wgsl)
    if device.capabilities.features.contains(.storage8BitReadWrite) {
        let rgba = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm, usage: [.storageBinding, .copySource]))
        let update = try await device.createComputePipeline(updateShader)
        commands += [
            .dispatchCompute(pipeline: update, bindings: [:], workgroups: .init(x: 1),
                             textures: [.init(binding: 0): images[4], .init(binding: 1): rgba]),
            .write(buffer: readback, offset: 0, words: [0]),
            .copyTextureToBuffer(source: images[4], destination: readback, bytesPerRow: 256, destinationOffset: 3),
            .read(buffer: readback, byteCount: 4),
            .copyTextureToBuffer(source: rgba, destination: readback, bytesPerRow: 256), .read(buffer: readback, byteCount: 4),
        ]
    } else {
        try await checkMissingStorageFeature(device, shader: updateShader)
    }
    return try await device.submitCommands(commands).wait().flatMap { $0 }
}

private let depthOnlyRenderShader = """
    @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
        if i == 0u { return vec4f(-1, -1, 0.25, 1); }
        if i == 1u { return vec4f(3, -1, 0.25, 1); }
        return vec4f(-1, 3, 0.25, 1);
    }
    @fragment fn fs() -> @location(0) vec4f { return vec4f(0, 1, 0, 1); }
    """

private let depthOnlyReadShader = """
    @group(0) @binding(0) var quantized: texture_depth_2d;
    @group(0) @binding(1) var opaqueDepth: texture_depth_2d;
    @group(0) @binding(2) var compare: sampler_comparison;
    @group(0) @binding(3) var<storage, read_write> out: vec4f;
    @compute @workgroup_size(1) fn run() {
        let a = textureLoad(quantized, vec2u(0, 0), 0u);
        let b = textureLoad(quantized, vec2u(1, 0), 0u);
        let c = textureLoad(opaqueDepth, vec2u(0, 0), 0u);
        let d = textureSampleCompareLevel(opaqueDepth, compare, vec2f(0.25, 0.5), 0.2);
        out = vec4f(a, b, c, d);
    }
    """

private func depthOnlyFormatDiagnostic(_ device: Device) async throws -> [UInt32] {
    let compiler = IndependentShaderCompiler()
    let usage: TextureUsage = [.renderAttachment, .copySource, .copyDestination, .textureBinding]
    let depth16 = try await device.createTexture(.init(width: 2, height: 1, format: .depth16Unorm, usage: usage))
    let depth24 = try await device.createTexture(.init(width: 2, height: 1, format: .depth24Plus, usage: usage))
    let opaqueCopy = try await device.createTexture(.init(width: 2, height: 1, format: .depth24Plus,
                                                           usage: [.copyDestination, .textureBinding]))
    let depthView = try await device.createTextureView(texture: opaqueCopy, .init(aspect: .depthOnly))
    let color = try await device.createTexture(.init(width: 2, height: 1, format: .rgba8Unorm, usage: .renderAttachment))
    let readback = try await device.createBuffer(.init(size: 16, usage: [.mapRead, .copyDestination]))
    let upload = try await device.createBuffer(.init(size: 8, usage: [.copySource, .copyDestination]))
    let out = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
    let sampler = try await device.createSampler(.init(compare: .less))
    let source = ShaderInput.wgsl(depthOnlyRenderShader)
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .wgsl)
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: .wgsl)
    let render16 = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm,
        depthState: .init(format: .depth16Unorm, compare: .less))
    let render24 = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm,
        depthState: .init(format: .depth24Plus, compare: .less))
    let compute = try await device.createComputePipeline(compiler.compileCompute(.wgsl(depthOnlyReadShader), to: .wgsl))
    let full = TextureCopyRegion(width: 2, height: 1)
    let raw16: [BufferCommand] = [
        .copyTextureToBuffer(source: depth16, destination: readback, bytesPerRow: 256), .read(buffer: readback, byteCount: 4),
    ]
    let commands: [BufferCommand] = [
        .writeTexture(destination: depth16, data: [0x99, 0, 64, 255, 191], layout: .init(offset: 1), region: full),
    ] + raw16 + [
        .write(buffer: upload, offset: 0, words: [0xdeadbeef, 0xffff8000]),
        .copyBufferToTexture(source: upload, layout: .init(offset: 4), destination: depth16, region: full, aspect: .depthOnly),
    ] + raw16 + [
        .render(pipeline: render16, target: color, clear: .transparentBlack, vertexCount: 3,
                depth: .init(texture: depth16, loadAction: .load)),
    ] + raw16 + [
        .render(pipeline: render24, target: color, clear: .transparentBlack, vertexCount: 3,
                depth: .init(texture: depth24, loadAction: .clear(0.8))),
        .copyTextureToTexture(source: depth24, destination: opaqueCopy, sourceRegion: full),
        .dispatchCompute(pipeline: compute, bindings: [.init(binding: 3): out], workgroups: .init(x: 1),
                         textures: [.init(binding: 0): depth16, .init(binding: 1): depthView], samplers: [.init(binding: 2): sampler]),
        .copy(source: out, destination: readback, byteCount: 16), .read(buffer: readback, byteCount: 16),
    ]
    return try await device.submitCommands(commands).wait().flatMap { $0 }
}

private let depth32StencilRenderShader = """
    @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
        if i == 0u { return vec4f(-1, -1, 0.25, 1); }
        if i == 1u { return vec4f(3, -1, 0.25, 1); }
        return vec4f(-1, 3, 0.25, 1);
    }
    @fragment fn fs() -> @location(0) vec4f { return vec4f(0, 1, 0, 1); }
    """

private let depth32StencilReadShader = """
    @group(0) @binding(0) var depth: texture_depth_2d;
    @group(0) @binding(1) var stencil: texture_2d<u32>;
    @group(0) @binding(2) var<storage, read_write> out: vec4f;
    @compute @workgroup_size(1) fn run() {
        let a = vec2u(0, 0);
        let b = vec2u(1, 0);
        out = vec4f(textureLoad(depth, a, 0u), textureLoad(depth, b, 0u),
                    f32(textureLoad(stencil, a, 0u).x), f32(textureLoad(stencil, b, 0u).x));
    }
    """

private func depth32StencilDiagnostic(_ device: Device) async throws -> [UInt32] {
    guard device.capabilities.features.contains(.depth32FloatStencil8) else {
        let scope = try await device.pushErrorScope(.validation)
        do {
            _ = try await device.createTexture(.init(width: 2, height: 1, format: .depth32FloatStencil8, usage: .renderAttachment))
            throw GPUError.backendFailure("Accepted depth32float-stencil8 without its device feature")
        } catch GPUError.validation("Feature not advertised") {
            guard case .validation = try await device.popErrorScope(scope) else {
                throw GPUError.backendFailure("Depth32 stencil feature error did not reach the error scope")
            }
            return []
        }
    }
    let compiler = IndependentShaderCompiler()
    let original = try await device.createTexture(.init(width: 2, height: 1, format: .depth32FloatStencil8,
        usage: [.renderAttachment, .copySource, .copyDestination]))
    let clone = try await device.createTexture(.init(width: 2, height: 1, format: .depth32FloatStencil8,
        usage: [.copySource, .copyDestination, .textureBinding]))
    let depth = try await device.createTextureView(texture: clone, .init(aspect: .depthOnly))
    let stencil = try await device.createTextureView(texture: clone, .init(aspect: .stencilOnly))
    let color = try await device.createTexture(.init(width: 2, height: 1, format: .rgba8Unorm, usage: [.renderAttachment, .copySource]))
    let readback = try await device.createBuffer(.init(size: 16, usage: [.mapRead, .copyDestination]))
    let out = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
    let source = ShaderInput.wgsl(depth32StencilRenderShader)
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .wgsl)
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: .wgsl)
    let face = RenderStencilFaceState(compare: .equal, pass: .incrementClamp)
    let render = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm,
        depthState: .init(format: .depth32FloatStencil8, compare: .less, stencil: .init(front: face, back: face)))
    let compute = try await device.createComputePipeline(compiler.compileCompute(.wgsl(depth32StencilReadShader), to: .wgsl))
    let region = TextureCopyRegion(width: 2, height: 1)
    return try await device.submitCommands([
        .writeTexture(destination: original, data: [0xee, 3, 7], layout: .init(offset: 1), region: region, aspect: .stencilOnly),
        .copyTextureToBuffer(source: original, destination: readback, bytesPerRow: 256, aspect: .stencilOnly),
        .read(buffer: readback, byteCount: 4),
        .render(pipeline: render, target: color, clear: .transparentBlack, vertexCount: 3,
                depth: .init(texture: original, loadAction: .clear(0.75), stencil: .init(loadAction: .load, reference: 3))),
        .copyTextureToBuffer(source: color, destination: readback, bytesPerRow: 256), .read(buffer: readback, byteCount: 8),
        .copyTextureToTexture(source: original, destination: clone, sourceRegion: region),
        .copyTextureToBuffer(source: clone, destination: readback, bytesPerRow: 256, aspect: .depthOnly),
        .read(buffer: readback, byteCount: 8),
        .write(buffer: readback, offset: 0, words: [0]),
        .copyTextureToBuffer(source: clone, destination: readback, bytesPerRow: 256, aspect: .stencilOnly),
        .read(buffer: readback, byteCount: 4),
        .dispatchCompute(pipeline: compute, bindings: [.init(binding: 2): out], workgroups: .init(x: 1),
                         textures: [.init(binding: 0): depth, .init(binding: 1): stencil]),
        .copy(source: out, destination: readback, byteCount: 16), .read(buffer: readback, byteCount: 16),
    ]).wait().flatMap { $0 }
}

private let stencilOnlyRenderShader = """
    @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
        if i == 0u { return vec4f(-1, -1, 0, 1); }
        if i == 1u { return vec4f(3, -1, 0, 1); }
        return vec4f(-1, 3, 0, 1);
    }
    @fragment fn fs() -> @location(0) vec4f { return vec4f(0, 1, 0, 1); }
    @fragment fn writesDepth() -> @builtin(frag_depth) f32 { return 0.5; }
    """

private let stencilOnlyReadShader = """
    @group(0) @binding(0) var allAspects: texture_2d<u32>;
    @group(0) @binding(1) var stencilView: texture_2d<u32>;
    @group(0) @binding(2) var<storage, read_write> out: vec4u;
    @compute @workgroup_size(1) fn run() {
        let a = vec2u(0, 0);
        let b = vec2u(1, 0);
        out = vec4u(textureLoad(allAspects, a, 0u).x, textureLoad(allAspects, b, 0u).x,
                    textureLoad(stencilView, a, 0u).x, textureLoad(stencilView, b, 0u).x);
    }
    """

private func stencilOnlyDiagnostic(_ device: Device) async throws -> [UInt32] {
    let compiler = IndependentShaderCompiler()
    let original = try await device.createTexture(.init(width: 2, height: 1, format: .stencil8,
        usage: [.renderAttachment, .copySource, .copyDestination]))
    let clone = try await device.createTexture(.init(width: 2, height: 1, format: .stencil8,
        usage: [.copySource, .copyDestination, .textureBinding]))
    let view = try await device.createTextureView(texture: clone, .init(aspect: .stencilOnly))
    let color = try await device.createTexture(.init(width: 2, height: 1, format: .rgba8Unorm, usage: [.renderAttachment, .copySource]))
    let readback = try await device.createBuffer(.init(size: 16, usage: [.mapRead, .copyDestination]))
    let output = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
    let source = ShaderInput.wgsl(stencilOnlyRenderShader)
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .wgsl)
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: .wgsl)
    let face = RenderStencilFaceState(compare: .equal, pass: .incrementClamp)
    let render = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm,
        depthState: .init(format: .stencil8, compare: .always, writeEnabled: false, stencil: .init(front: face, back: face)))
    let compute = try await device.createComputePipeline(compiler.compileCompute(.wgsl(stencilOnlyReadShader), to: .wgsl))
    let region = TextureCopyRegion(width: 2, height: 1)
    return try await device.submitCommands([
        .copyTextureToBuffer(source: original, destination: readback, bytesPerRow: 256), .read(buffer: readback, byteCount: 4),
        .writeTexture(destination: original, data: [0xee, 3, 7], layout: .init(offset: 1), region: region),
        .copyTextureToBuffer(source: original, destination: readback, bytesPerRow: 256, aspect: .stencilOnly),
        .read(buffer: readback, byteCount: 4),
        .render(pipeline: render, target: color, clear: .transparentBlack, vertexCount: 3,
                depth: .init(texture: original, loadAction: .unused, stencil: .init(loadAction: .load, reference: 3))),
        .copyTextureToBuffer(source: color, destination: readback, bytesPerRow: 256), .read(buffer: readback, byteCount: 8),
        .copyTextureToTexture(source: original, destination: clone, sourceRegion: region),
        .write(buffer: readback, offset: 0, words: [0]),
        .copyTextureToBuffer(source: clone, destination: readback, bytesPerRow: 256), .read(buffer: readback, byteCount: 4),
        .dispatchCompute(pipeline: compute, bindings: [.init(binding: 2): output], workgroups: .init(x: 1),
                         textures: [.init(binding: 0): clone, .init(binding: 1): view]),
        .copy(source: output, destination: readback, byteCount: 16), .read(buffer: readback, byteCount: 16),
    ]).wait().flatMap { $0 }
}

private let bgraStorageShader = """
    @group(0) @binding(0) var image: texture_storage_2d<bgra8unorm, write>;
    @compute @workgroup_size(1) fn run() {
        textureStore(image, vec2u(0), vec4f(0.25, 0.5, 0.75, 1.0));
    }
    """

private let bgraSampleShader = """
    @group(0) @binding(0) var image: texture_2d<f32>;
    @group(0) @binding(1) var<storage, read_write> out: vec4f;
    @compute @workgroup_size(1) fn run() { out = textureLoad(image, vec2u(0), 0u); }
    """

private let wideStorageShader = """
    @group(0) @binding(0) var halfImage: texture_storage_2d<rgba16float, read_write>;
    @group(0) @binding(1) var fullImage: texture_storage_2d<rgba32float, read_write>;
    @group(0) @binding(2) var uintImage: texture_storage_2d<rgba32uint, read_write>;
    @group(0) @binding(3) var sintImage: texture_storage_2d<rgba32sint, read_write>;
    @compute @workgroup_size(1) fn run() {
        let p = vec2u(0);
        textureStore(halfImage, p, textureLoad(halfImage, p) + vec4f(0.5, -1, 2, -2));
        textureStore(fullImage, p, textureLoad(fullImage, p) - vec4f(0.5, -0.25, 0.75, -1.25));
        textureStore(uintImage, p, textureLoad(uintImage, p) + vec4u(1, 2, 3, 4));
        textureStore(sintImage, p, textureLoad(sintImage, p) + vec4i(-1, 2, -3, 4));
    }
    """

private func bgraStorageDiagnostic(_ device: Device) async throws -> [UInt32] {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(bgraStorageShader), to: .wgsl)
    guard device.capabilities.features.contains(.bgra8UnormStorage) else {
        try await checkMissingStorageFeature(device, shader: shader)
        return []
    }
    let write = try await device.createComputePipeline(shader)
    let sample = try await device.createComputePipeline(compiler.compileCompute(.wgsl(bgraSampleShader), to: .wgsl))
    let image = try await device.createTexture(.init(width: 1, height: 1, format: .bgra8Unorm,
                                                     usage: [.storageBinding, .textureBinding, .copySource]))
    let output = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
    let readback = try await device.createBuffer(.init(size: 16, usage: [.mapRead, .copyDestination]))
    let bindings: [ComputeBindingSlot: ResourceID] = [.init(binding: 0): image]
    return try await device.submitCommands([
        .dispatchCompute(pipeline: write, bindings: [:], workgroups: .init(x: 1), textures: bindings),
        .copyTextureToBuffer(source: image, destination: readback, bytesPerRow: 256), .read(buffer: readback, byteCount: 4),
        .dispatchCompute(pipeline: sample, bindings: [.init(binding: 1): output], workgroups: .init(x: 1), textures: bindings),
        .copy(source: output, destination: readback, byteCount: 16), .read(buffer: readback, byteCount: 16),
    ]).wait().flatMap { $0 }
}

private func wideStorageDiagnostic(_ device: Device) async throws -> [UInt32] {
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(wideStorageShader), to: .wgsl)
    guard device.capabilities.features.isSuperset(of: [.wideRGBAStorageReadWrite, .storageTextureReadWrite]) else {
        try await checkMissingStorageFeature(device, shader: shader)
        return []
    }
    let compute = try await device.createComputePipeline(shader)
    let formats: [TextureFormat] = [.rgba16Float, .rgba32Float, .rgba32Uint, .rgba32Sint]
    let data = [
        [UInt32(0x40003c00), 0x44004200],
        [Float(-1), 2, -3, 4].map(\.bitPattern),
        [100, 200, 300, 400],
        [Int32(-10), 20, -30, 40].map(UInt32.init(bitPattern:)),
    ]
    let readback = try await device.createBuffer(.init(size: 16, usage: [.mapRead, .copyDestination]))
    var textures: [ResourceID] = []
    var commands: [BufferCommand] = []
    for (format, words) in zip(formats, data) {
        let image = try await device.createTexture(.init(width: 1, height: 1, format: format,
            usage: [.storageBinding, .copySource, .copyDestination]))
        textures.append(image)
        commands.append(.writeTexture(destination: image, data: words.withUnsafeBytes { Array($0) },
                                      region: .init(width: 1, height: 1)))
    }
    let bindings = Dictionary(uniqueKeysWithValues: textures.enumerated().map {
        (ComputeBindingSlot(binding: UInt32($0.offset)), $0.element)
    })
    commands.append(.dispatchCompute(pipeline: compute, bindings: [:], workgroups: .init(x: 1), textures: bindings))
    for (index, image) in textures.enumerated() {
        commands += [.copyTextureToBuffer(source: image, destination: readback, bytesPerRow: 256),
                     .read(buffer: readback, byteCount: index == 0 ? 8 : 16)]
    }
    return try await device.submitCommands(commands).wait().flatMap { $0 }
}

private let forwardedResourceBindings = """
    enable wgpu_binding_array;
    @group(0) @binding(0) var images: binding_array<texture_2d<f32>>;
    @group(1) @binding(1) var states: binding_array<sampler, 2>;
    @group(0) @binding(2) var<storage, read_write> out: array<f32, 4>;
    @group(2) @binding(0) var targets: binding_array<texture_storage_2d<rgba8unorm, write>, 2>;
    """

private let forwardedResourceShader = forwardedResourceBindings + "\n" + """
    fn read(image: texture_2d<f32>, state: sampler, x: f32) -> f32 {
        let width = textureDimensions(image).x;
        return textureSampleLevel(image, state, vec2f(x, 0.5), 0.0).x * f32(width) * 0.5;
    }
    fn forward(x: f32, state: sampler, image: texture_2d<f32>) -> f32 {
        return read(image, state, x);
    }
    fn save(target: texture_storage_2d<rgba8unorm, write>, red: f32) {
        textureStore(target, vec2u(0), vec4f(red, 0, 0, 1));
    }
    @compute @workgroup_size(4) fn run(@builtin(local_invocation_index) lane: u32) {
        let image = lane / 2u;
        let state = lane % 2u;
        let red = forward(1.25, states[state], images[image]);
        out[lane] = red;
        if state == 0u { save(targets[image], red); }
    }
    """

private let forwardedResourceRenderShader = forwardedResourceBindings + "\n" + """
    @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
        let xy = array<vec2f, 3>(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
        return vec4f(xy[i], 0.25 * f32(textureDimensions(images[0]).x), 1);
    }
    @fragment fn fs(@builtin(position) pixel: vec4f) -> @location(0) vec4f {
        let index = u32(pixel.x);
        let red = textureSample(images[index], states[index], vec2f(1.25, 0.5)).x;
        return vec4f(red, 1, 0, 1);
    }
    """

private func forwardedResourcesDiagnostic(_ device: Device) async throws -> [UInt32] {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl(forwardedResourceShader)
    let graphics = ShaderInput.wgsl(forwardedResourceRenderShader)
    let lengths: [IRResourceBinding: UInt32] = [.init(group: 0, binding: 0): 2]
    let shader = try compiler.compileWebGPUCompute(source, bindingArrayLengths: lengths)
    let pipeline = try await device.createComputePipeline(shader)
    let render = try await device.createRenderPipeline(
        vertex: compiler.compileWebGPURenderStage(graphics, entryPoint: "vs", bindingArrayLengths: lengths),
        fragment: compiler.compileWebGPURenderStage(graphics, entryPoint: "fs", bindingArrayLengths: lengths),
        colorFormat: .rgba8Unorm)
    let roundTripSource = ShaderInput.wgsl("""
        enable wgpu_binding_array;
        @group(0) @binding(0) var images: binding_array<texture_2d<f32>, 2>;
        @group(1) @binding(1) var states: binding_array<sampler, 2>;
        @group(0) @binding(2) var<storage, read_write> out: array<f32, 4>;
        @group(2) @binding(0) var targets: binding_array<texture_storage_2d<rgba8unorm, write>, 2>;
        @compute @workgroup_size(4) fn round_trip(@builtin(local_invocation_index) lane: u32) {
            let image = select(1u, 0u, lane >= 2u);
            let state = 1u - (lane & 1u);
            let red = textureSampleLevel(images[image], states[state], vec2f(1.25, 0.5), 0.0).x;
            out[lane] = red;
            if state == 1u {
                textureStore(targets[image], vec2u(0u), vec4f(red, 1.0, 0.0, 1.0));
            }
        }
        """)
    guard case .spirv(let words) = try compiler.compile(roundTripSource, to: .spirv) else {
        throw GPUError.backendFailure("Expected binary SPIR-V for texture and sampler binding arrays")
    }
    let roundTripShader = try compiler.compileWebGPUCompute(.spirv(words), entryPoint: "round_trip")
    guard roundTripShader.sampledTextures.count == 1, let imageArray = roundTripShader.sampledTextures.first,
          imageArray.binding == .init(group: 0, binding: 0), imageArray.bindingArray?.count == 2,
          roundTripShader.samplerBindings.count == 1, let samplerArray = roundTripShader.samplerBindings.first,
          samplerArray.binding == .init(group: 1, binding: 1), samplerArray.bindingArray?.count == 2,
          roundTripShader.storageTextures.count == 1, let targetArray = roundTripShader.storageTextures.first,
          targetArray.binding == .init(group: 2, binding: 0), targetArray.bindingArray?.count == 2,
          targetArray.format == .rgba8Unorm, targetArray.dimension == .d2, targetArray.access == [.store] else {
        throw GPUError.backendFailure("Unexpected round-trip binding array reflection")
    }
    let roundTrip = try await device.createComputePipeline(roundTripShader)
    let layoutSource = ShaderInput.wgsl("""
        enable wgpu_binding_array;
        @group(0) @binding(0) var images: binding_array<texture_2d<f32>>;
        @group(0) @binding(1) var alternates: binding_array<texture_2d<f32>>;
        @group(1) @binding(1) var states: binding_array<sampler>;
        @group(0) @binding(2) var<storage, read_write> out: array<f32, 4>;
        @compute @workgroup_size(4) fn layout_round_trip(@builtin(local_invocation_index) lane: u32) {
            let index = i32(lane) - 1i;
            let uv = vec2f(1.25, 0.5);
            let first = textureSampleLevel(images[index], states[index], uv, 0.0).x;
            let second = textureSampleLevel(alternates[index], states[index], uv, 0.0).x;
            out[lane] = first + 2.0 * second;
        }
        """)
    let wideBinding = IRResourceBinding(group: 0, binding: 0)
    let narrowBinding = IRResourceBinding(group: 0, binding: 1)
    let layoutLengths: [IRResourceBinding: UInt32] = [
        wideBinding: 2, narrowBinding: 1, .init(group: 1, binding: 1): 2,
    ]
    guard case .spirv(let layoutWords) = try compiler.compile(
        layoutSource, to: .spirv, bindingArrayLengths: layoutLengths
    ) else { throw GPUError.backendFailure("Expected binary SPIR-V for layout-sized binding arrays") }
    let layoutShader = try compiler.compileWebGPUCompute(.spirv(layoutWords), entryPoint: "layout_round_trip")
    let layoutImages = layoutShader.sampledTextures
    guard layoutImages.count == 2,
          layoutImages.contains(where: { $0.binding == wideBinding && $0.bindingArray?.count == 2 }),
          layoutImages.contains(where: { $0.binding == narrowBinding && $0.bindingArray?.count == 1 }),
          layoutShader.samplerBindings.count == 1, let layoutSampler = layoutShader.samplerBindings.first,
          layoutSampler.binding == .init(group: 1, binding: 1), layoutSampler.bindingArray?.count == 2 else {
        throw GPUError.backendFailure("Unexpected layout-sized binding array reflection")
    }
    let layoutRoundTrip = try await device.createComputePipeline(layoutShader)
    guard case .spirv(let runtimeWords) = try compiler.compile(layoutSource, to: .spirv) else {
        throw GPUError.backendFailure("Expected binary SPIR-V for runtime binding arrays")
    }
    var offset = 5 // Skip the SPIR-V module header.
    var runtimeArrayCount = 0
    while offset < runtimeWords.count {
        let header = runtimeWords[offset]
        let wordCount = Int(header >> 16)
        guard wordCount > 0, wordCount <= runtimeWords.count - offset else {
            throw GPUError.backendFailure("Invalid SPIR-V instruction length")
        }
        if header & 0xffff == 29 { runtimeArrayCount += 1 } // OpTypeRuntimeArray
        offset += wordCount
    }
    guard runtimeArrayCount >= 2 else {
        throw GPUError.backendFailure("Expected native runtime image and sampler arrays")
    }
    let runtimeShader = try compiler.compileWebGPUCompute(
        .spirv(runtimeWords), entryPoint: "layout_round_trip", bindingArrayLengths: layoutLengths
    )
    guard runtimeShader.sampledTextures == layoutShader.sampledTextures,
          runtimeShader.samplerBindings == layoutShader.samplerBindings else {
        throw GPUError.backendFailure("Unexpected specialized runtime binding array reflection")
    }
    let runtimeRoundTrip = try await device.createComputePipeline(runtimeShader)
    let sampleUsage: TextureUsage = [.textureBinding, .copyDestination]
    let storeUsage: TextureUsage = [.storageBinding, .copySource]
    let a = try await device.createTexture(.init(width: 2, height: 1, format: .rgba8Unorm, usage: sampleUsage))
    let b = try await device.createTexture(.init(width: 2, height: 1, format: .rgba8Unorm, usage: sampleUsage))
    let first = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm, usage: storeUsage))
    let second = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm, usage: storeUsage))
    let color = try await device.createTexture(.init(width: 2, height: 1, format: .rgba8Unorm,
                                                     usage: [.renderAttachment, .copySource]))
    let clamp = try await device.createSampler(.init())
    let repeating = try await device.createSampler(.init(addressModeU: .repeat))
    let output = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
    let readback = try await device.createBuffer(.init(size: 16, usage: [.mapRead, .copyDestination]))
    let region = TextureCopyRegion(width: 2, height: 1)
    let textures: [ComputeBindingSlot: ResourceID] = [.init(binding: 0, arrayElement: 0): a, .init(binding: 0, arrayElement: 1): b]
    let samplers: [ComputeBindingSlot: ResourceID] = [
        .init(group: 1, binding: 1, arrayElement: 0): clamp, .init(group: 1, binding: 1, arrayElement: 1): repeating,
    ]
    let targets: [ComputeBindingSlot: ResourceID] = [
        .init(group: 2, binding: 0, arrayElement: 0): first, .init(group: 2, binding: 0, arrayElement: 1): second,
    ]
    let computeTextures = textures.merging(targets) { sample, _ in sample }
    let layoutTextures = textures.merging([.init(binding: 1, arrayElement: 0): a]) { sample, _ in sample }
    let runtimeTextures: [ComputeBindingSlot: ResourceID] = [
        .init(binding: 0, arrayElement: 0): b, .init(binding: 0, arrayElement: 1): a,
        .init(binding: 1, arrayElement: 0): b,
    ]
    let snapshots = try await device.submitCommands([
        .writeTexture(destination: a, data: [10, 0, 0, 255, 70, 0, 0, 255], region: region),
        .writeTexture(destination: b, data: [150, 0, 0, 255, 220, 0, 0, 255], region: region),
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 2): output], workgroups: .init(x: 1),
                         textures: computeTextures, samplers: samplers),
        .copy(source: output, destination: readback, byteCount: 16), .read(buffer: readback, byteCount: 16),
        .copyTextureToBuffer(source: first, destination: readback, bytesPerRow: 256), .read(buffer: readback, byteCount: 4),
        .copyTextureToBuffer(source: second, destination: readback, bytesPerRow: 256), .read(buffer: readback, byteCount: 4),
        .render(pipeline: render, target: color, clear: .transparentBlack, vertexCount: 3, textures: textures, samplers: samplers),
        .copyTextureToBuffer(source: color, destination: readback, bytesPerRow: 256), .read(buffer: readback, byteCount: 8),
        .dispatchCompute(pipeline: roundTrip, bindings: [.init(binding: 2): output], workgroups: .init(x: 1),
                         textures: computeTextures, samplers: samplers),
        .copy(source: output, destination: readback, byteCount: 16), .read(buffer: readback, byteCount: 16),
        .copyTextureToBuffer(source: first, destination: readback, bytesPerRow: 256), .read(buffer: readback, byteCount: 4),
        .copyTextureToBuffer(source: second, destination: readback, bytesPerRow: 256), .read(buffer: readback, byteCount: 4),
        .dispatchCompute(pipeline: layoutRoundTrip, bindings: [.init(binding: 2): output], workgroups: .init(x: 1),
                         textures: layoutTextures, samplers: samplers),
        .copy(source: output, destination: readback, byteCount: 16), .read(buffer: readback, byteCount: 16),
        .dispatchCompute(pipeline: runtimeRoundTrip, bindings: [.init(binding: 2): output], workgroups: .init(x: 1),
                         textures: runtimeTextures, samplers: samplers),
        .copy(source: output, destination: readback, byteCount: 16), .read(buffer: readback, byteCount: 16),
    ]).wait()
    guard snapshots.count == 9,
          snapshots[4].map({ (Float(bitPattern: $0) * 255).rounded() }) == [150, 220, 10, 70],
          snapshots[5] == [0xff00ff0a], snapshots[6] == [0xff00ff96],
          snapshots[7].map({ (Float(bitPattern: $0) * 255).rounded() }) == [210, 210, 170, 170],
          snapshots[8].map({ (Float(bitPattern: $0) * 255).rounded() }) == [660, 660, 310, 310] else {
        throw GPUError.backendFailure("Unexpected round-trip binding array results: \(snapshots)")
    }
    return snapshots.flatMap { $0 }
}

private let webBufferArrayComputeShader = """
    enable wgpu_binding_array;
    requires unrestricted_pointer_parameters;
    struct Record { counter: atomic<u32>, tail: array<u32> }
    struct Settings { values: vec4u }
    @group(0) @binding(0) var<storage, read_write> records: binding_array<Record>;
    @group(1) @binding(0) var<uniform> settings: binding_array<Settings, 2>;
    @group(0) @binding(2) var<storage, read_write> output: array<vec4u, 4>;

    fn increment(counter: ptr<storage, atomic<u32>, read_write>) -> u32 {
        return atomicAdd(counter, 1u);
    }
    @compute @workgroup_size(4) fn run(@builtin(local_invocation_index) lane: u32) {
        let index = i32(lane) - 1;
        let old = increment(&records[index].counter);
        let new = atomicLoad(&records[index].counter);
        let length = arrayLength(&records[index].tail);
        output[lane] = vec4u(old, new, length, settings[index].values.x);
        records[index].tail[0] = settings[index].values.y;
        records[7].tail[0] = 99u;
    }
    """

private let webBufferArrayRenderShader = """
    enable wgpu_binding_array;
    struct Record { counter: u32, tail: array<u32> }
    struct Settings { values: vec4u }
    struct Varyings {
        @builtin(position) position: vec4f,
        @location(0) @interpolate(flat) red: f32,
    }
    @group(0) @binding(0) var<storage, read> records: binding_array<Record>;
    @group(1) @binding(0) var<uniform> settings: binding_array<Settings, 2>;
    @vertex fn vs(@builtin(vertex_index) i: u32) -> Varyings {
        let xy = array<vec2f, 3>(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
        return Varyings(vec4f(xy[i], 0, 1), f32(settings[0].values.x));
    }
    @fragment fn fs(input: Varyings) -> @location(0) vec4f {
        let index = u32(input.position.x);
        let length = arrayLength(&records[index].tail);
        return vec4f(vec3f(input.red, f32(length), f32(records[index].tail[0])) / 255.0, 1);
    }
    """

private func webBufferArraysDiagnostic(_ device: Device) async throws -> [UInt32] {
    let compiler = IndependentShaderCompiler()
    let lengths: [IRResourceBinding: UInt32] = [.init(group: 0, binding: 0): 2]
    let compute = try await device.createComputePipeline(
        compiler.compileWebGPUCompute(.wgsl(webBufferArrayComputeShader), bindingArrayLengths: lengths))
    let graphics = ShaderInput.wgsl(webBufferArrayRenderShader)
    let render = try await device.createRenderPipeline(
        vertex: compiler.compileWebGPURenderStage(graphics, entryPoint: "vs", bindingArrayLengths: lengths),
        fragment: compiler.compileWebGPURenderStage(graphics, entryPoint: "fs", bindingArrayLengths: lengths),
        colorFormat: .rgba8Unorm)
    let runtimeSource = ShaderInput.wgsl("""
        enable wgpu_binding_array;
        struct Record { counter: u32, tail: array<u32> }
        struct Settings { values: vec4u }
        @group(0) @binding(0) var<storage, read_write> records: binding_array<Record>;
        @group(1) @binding(0) var<uniform> settings: binding_array<Settings>;
        @group(0) @binding(2) var<storage, read_write> output: array<vec4u, 4>;
        @compute @workgroup_size(2) fn runtime_buffers(@builtin(local_invocation_index) lane: u32) {
            let length = arrayLength(&records[lane].tail);
            let first = records[lane].tail[0u];
            let last = records[lane].tail[length - 1u];
            let data = settings[lane].values;
            output[lane] = vec4u(length, first, last, records[lane].counter + data.x);
            records[lane].tail[0u] = last + data.y;
        }
        """)
    guard case .spirv(let runtimeWords) = try compiler.compile(runtimeSource, to: .spirv) else {
        throw GPUError.backendFailure("Expected binary SPIR-V for runtime buffer binding arrays")
    }
    let runtimeLengths = lengths.merging([.init(group: 1, binding: 0): 2]) { original, _ in original }
    let runtimeShader = try compiler.compileWebGPUCompute(
        .spirv(runtimeWords), entryPoint: "runtime_buffers", bindingArrayLengths: runtimeLengths
    )
    guard let recordArray = runtimeShader.storageBindings.first(where: { $0.binding == .init(group: 0, binding: 0) }),
          recordArray.bindingArray?.count == 2, !recordArray.isReadOnly,
          recordArray.runtimeArrayOffset == 4, recordArray.runtimeArrayStride == 4,
          runtimeShader.uniformBindings.count == 1, let settingsArray = runtimeShader.uniformBindings.first,
          settingsArray.binding == .init(group: 1, binding: 0), settingsArray.bindingArray?.count == 2,
          settingsArray.minimumByteSize == 16 else {
        throw GPUError.backendFailure("Unexpected runtime buffer binding array reflection")
    }
    let runtimeCompute = try await device.createComputePipeline(runtimeShader)
    let layoutSource = ShaderInput.wgsl("""
        enable wgpu_binding_array;
        struct Record { counter: u32, tail: array<u32> }
        struct Settings { values: vec4u }
        @group(0) @binding(0) var<storage, read_write> records: binding_array<Record>;
        @group(1) @binding(0) var<uniform> settings: binding_array<Settings>;
        @group(0) @binding(2) var<storage, read_write> output: array<vec4u, 4>;
        @compute @workgroup_size(4) fn layout_buffers(@builtin(local_invocation_index) lane: u32) {
            let signedIndex = i32(lane) - 1i;
            let unsignedIndex = select(lane, 0xffffffffu, lane == 3u);
            let signedLength = arrayLength(&records[signedIndex].tail);
            let unsignedLength = arrayLength(&records[unsignedIndex].tail);
            let signedRead = records[signedIndex].tail[0u] + settings[signedIndex].values.x;
            let unsignedRead = records[unsignedIndex].tail[0u] + settings[unsignedIndex].values.x;
            output[lane] = vec4u(signedLength, unsignedLength, signedRead, 1000u + unsignedRead);
            if lane == 0u { records[signedIndex].tail[0u] = 333u; }
            if lane == 3u { records[unsignedIndex].tail[0u] = 444u; }
        }
        """)
    let layoutLengths = lengths.merging([.init(group: 1, binding: 0): 1]) { original, _ in original }
    guard case .spirv(let layoutWords) = try compiler.compile(
        layoutSource, to: .spirv, bindingArrayLengths: layoutLengths
    ) else { throw GPUError.backendFailure("Expected binary SPIR-V for layout-specialized buffer binding arrays") }
    let layoutShader = try compiler.compileWebGPUCompute(.spirv(layoutWords), entryPoint: "layout_buffers")
    guard layoutShader.storageBindings.first(where: { $0.binding == .init(group: 0, binding: 0) }) == recordArray,
          layoutShader.uniformBindings.count == 1, let layoutSettings = layoutShader.uniformBindings.first,
          layoutSettings.binding == settingsArray.binding, layoutSettings.bindingArray?.count == 1,
          layoutSettings.minimumByteSize == settingsArray.minimumByteSize else {
        throw GPUError.backendFailure("Unexpected layout-specialized buffer binding array reflection")
    }
    let layoutCompute = try await device.createComputePipeline(layoutShader)
    let atomicSource = ShaderInput.wgsl("""
        enable wgpu_binding_array;
        struct Record { counter: atomic<u32>, sentinel: u32 }
        @group(0) @binding(0) var<storage, read_write> records: binding_array<Record>;
        @group(0) @binding(2) var<storage, read_write> output: array<vec4u, 2>;
        @compute @workgroup_size(1) fn guarded_atomic(@builtin(local_invocation_index) lane: u32) {
            let index = i32(lane) - 1i;
            let below = atomicCompareExchangeWeak(&records[index].counter, 0u, 333u);
            let above = atomicCompareExchangeWeak(&records[index + 3i].counter, 0u, 444u);
            let first = atomicCompareExchangeWeak(&records[index + 1i].counter, 0u, 17u);
            let second = atomicCompareExchangeWeak(&records[index + 2i].counter, 0u, 23u);
            output[0] = vec4u(below.old_value, select(0u, 1u, below.exchanged),
                              above.old_value, select(0u, 1u, above.exchanged));
            output[1] = vec4u(first.old_value, select(0u, 1u, first.exchanged),
                              second.old_value, select(0u, 1u, second.exchanged));
        }
        """)
    guard case .spirv(let atomicWords) = try compiler.compile(
        atomicSource, to: .spirv, bindingArrayLengths: lengths
    ) else { throw GPUError.backendFailure("Expected binary SPIR-V for guarded buffer atomics") }
    let atomicShader = try compiler.compileWebGPUCompute(.spirv(atomicWords), entryPoint: "guarded_atomic")
    let atomicCompute = try await device.createComputePipeline(atomicShader)
    let records = try await device.createBuffer(.init(size: 512, usage: [.storage, .copyDestination, .copySource]))
    let settings = try await device.createBuffer(.init(size: 512, usage: [.uniform, .copyDestination]))
    let output = try await device.createBuffer(.init(size: 64, usage: [.storage, .copySource]))
    let readback = try await device.createBuffer(.init(size: 64, usage: [.mapRead, .copyDestination]))
    let color = try await device.createTexture(.init(width: 2, height: 1, format: .rgba8Unorm,
                                                     usage: [.renderAttachment, .copySource]))
    let storage: [ComputeBindingSlot: ComputeBufferBinding] = [
        .init(binding: 0, arrayElement: 0): .init(buffer: records, size: 8),
        .init(binding: 0, arrayElement: 1): .init(buffer: records, offset: 256, size: 16),
    ]
    let uniforms: [ComputeBindingSlot: ComputeBufferBinding] = [
        .init(group: 1, binding: 0, arrayElement: 0): .init(buffer: settings, size: 16),
        .init(group: 1, binding: 0, arrayElement: 1): .init(buffer: settings, offset: 256, size: 16),
    ]
    var bindings = storage.merging(uniforms) { storage, _ in storage }
    bindings[.init(binding: 2)] = .init(buffer: output)
    var layoutBindings = bindings
    layoutBindings[.init(group: 1, binding: 0, arrayElement: 0)] = .init(buffer: settings, offset: 256, size: 16)
    layoutBindings[.init(group: 1, binding: 0, arrayElement: 1)] = nil
    var atomicBindings = storage
    atomicBindings[.init(binding: 2)] = .init(buffer: output)
    let commands: [BufferCommand] = [
        .write(buffer: records, offset: 0, words: [100, 5, 41, 42]),
        .write(buffer: records, offset: 256, words: [200, 7, 8, 9]),
        .write(buffer: settings, offset: 0, words: [11, 55, 0, 0]),
        .write(buffer: settings, offset: 256, words: [22, 77, 0, 0]),
        .dispatchComputeRanges(pipeline: compute, bindings: bindings, workgroups: .init(x: 1)),
        .copy(source: output, destination: readback, byteCount: 64), .read(buffer: readback, byteCount: 64),
        .copy(source: records, destination: readback, byteCount: 16), .read(buffer: readback, byteCount: 16),
        .copy(source: records, sourceOffset: 256, destination: readback, byteCount: 16), .read(buffer: readback, byteCount: 16),
        .render(pipeline: render, target: color, clear: .transparentBlack, vertexCount: 3,
                uniforms: uniforms, storageBuffers: storage),
        .copyTextureToBuffer(source: color, destination: readback, bytesPerRow: 256), .read(buffer: readback, byteCount: 8),
        .dispatchComputeRanges(pipeline: runtimeCompute, bindings: bindings, workgroups: .init(x: 1)),
        .copy(source: output, destination: readback, byteCount: 32), .read(buffer: readback, byteCount: 32),
        .copy(source: records, destination: readback, byteCount: 16), .read(buffer: readback, byteCount: 16),
        .copy(source: records, sourceOffset: 256, destination: readback, byteCount: 16), .read(buffer: readback, byteCount: 16),
        .dispatchComputeRanges(pipeline: layoutCompute, bindings: layoutBindings, workgroups: .init(x: 1)),
        .copy(source: output, destination: readback, byteCount: 64), .read(buffer: readback, byteCount: 64),
        .copy(source: records, destination: readback, byteCount: 16), .read(buffer: readback, byteCount: 16),
        .copy(source: records, sourceOffset: 256, destination: readback, byteCount: 16), .read(buffer: readback, byteCount: 16),
        .write(buffer: records, offset: 0, words: [0]),
        .write(buffer: records, offset: 256, words: [0]),
        .dispatchComputeRanges(pipeline: atomicCompute, bindings: atomicBindings, workgroups: .init(x: 1)),
        .copy(source: output, destination: readback, byteCount: 32), .read(buffer: readback, byteCount: 32),
        .copy(source: records, destination: readback, byteCount: 16), .read(buffer: readback, byteCount: 16),
        .copy(source: records, sourceOffset: 256, destination: readback, byteCount: 16), .read(buffer: readback, byteCount: 16),
    ]
    let snapshots = try await device.submitCommands(commands).wait()
    let runtimeExpected: [[UInt32]] = [
        [1, 55, 55, 112, 3, 77, 9, 223], [101, 110, 41, 42], [201, 86, 8, 9],
    ]
    let layoutExpected: [UInt32] = [0, 1, 0, 1132, 1, 3, 132, 1086, 3, 0, 86, 1000, 0, 0, 0, 1000]
    guard snapshots.count == 13, snapshots[4..<7].elementsEqual(runtimeExpected),
          snapshots[7] == layoutExpected, snapshots[8] == runtimeExpected[1], snapshots[9] == runtimeExpected[2] else {
        throw GPUError.backendFailure("Unexpected SPIR-V buffer binding array results: \(snapshots)")
    }
    let atomicExpected: [[UInt32]] = [[0, 0, 0, 0, 0, 1, 0, 1], [17, 110, 41, 42], [23, 86, 8, 9]]
    guard snapshots.suffix(3).elementsEqual(atomicExpected) else {
        throw GPUError.backendFailure("Unexpected guarded buffer atomic results: \(snapshots.suffix(3))")
    }
    return snapshots.flatMap { $0 }
}

private let webExternalTextureCompute = """
    @group(0) @binding(0) var first: texture_external;
    @group(1) @binding(0) var second: texture_external;
    @group(0) @binding(1) var state: sampler;
    @group(0) @binding(2) var<storage, read_write> output: array<vec4f, 4>;
    fn sample(image: texture_external, s: sampler, x: f32) -> vec4f {
        return textureSampleBaseClampToEdge(image, s, vec2f(x, 0.5));
    }
    @compute @workgroup_size(1) fn run() {
        output[0] = sample(first, state, -0.25);
        output[1] = sample(second, state, 1.25);
        output[2] = textureLoad(first, vec2u(1, 0));
        output[3] = vec4f(vec2f(textureDimensions(first)), vec2f(textureDimensions(second)));
    }
    """

private let webExternalTextureRender = """
    @group(0) @binding(0) var frame: texture_external;
    @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
        let xy = array<vec2f, 3>(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
        return vec4f(xy[i], 0, 1);
    }
    @fragment fn fs(@builtin(position) pixel: vec4f) -> @location(0) vec4f {
        return textureLoad(frame, vec2i(i32(pixel.x), 0));
    }
    """

private func webExternalTextureDiagnostic(
    _ device: Device, frames: (first: UInt64, second: UInt64, closed: UInt64)
) async throws -> [UInt32] {
    let compiler = IndependentShaderCompiler()
    let compute = try await device.createComputePipeline(compiler.compileWebGPUCompute(.wgsl(webExternalTextureCompute)))
    let graphics = ShaderInput.wgsl(webExternalTextureRender)
    let render = try await device.createRenderPipeline(
        vertex: compiler.compileWebGPURenderStage(graphics, entryPoint: "vs"),
        fragment: compiler.compileWebGPURenderStage(graphics, entryPoint: "fs"), colorFormat: .rgba8Unorm)
    do {
        _ = try await device.importExternalTexture(.init(videoFrame: frames.closed))
        throw GPUError.backendFailure("A closed VideoFrame was accepted")
    } catch GPUError.validation("Unknown or closed VideoFrame") {}
    let first = try await device.importExternalTexture(.init(videoFrame: frames.first))
    let second = try await device.importExternalTexture(.init(videoFrame: frames.second))
    let sampler = try await device.createSampler(.init(addressModeU: .repeat, addressModeV: .repeat))
    let output = try await device.createBuffer(.init(size: 64, usage: [.storage, .copySource]))
    let readback = try await device.createBuffer(.init(size: 64, usage: [.mapRead, .copyDestination]))
    let color = try await device.createTexture(.init(width: 2, height: 1, format: .rgba8Unorm,
                                                     usage: [.renderAttachment, .copySource]))
    let commands: [BufferCommand] = [
        .dispatchCompute(pipeline: compute, bindings: [.init(binding: 2): output], workgroups: .init(x: 1),
                         textures: [.init(binding: 0): first, .init(group: 1, binding: 0): second],
                         samplers: [.init(binding: 1): sampler]),
        .copy(source: output, destination: readback, byteCount: 64), .read(buffer: readback, byteCount: 64),
        .render(pipeline: render, target: color, clear: .transparentBlack, vertexCount: 3, textures: [.init(binding: 0): first]),
        .copyTextureToBuffer(source: color, destination: readback, bytesPerRow: 256), .read(buffer: readback, byteCount: 8),
    ]
    let words = try await device.submitCommands(commands).wait().flatMap { $0 }
    try await device.destroyExternalTexture(first)
    return words
}
#endif
