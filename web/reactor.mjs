import { browserWASI } from "./wasi.mjs";
import { openWebGPU } from "./webgpu.mjs";

const utf8 = new TextDecoder();
const encoder = new TextEncoder();
const limit = 16 * 1024 * 1024;
const canvasHosts = new WeakMap();

function withDeadline(promise, milliseconds, onTimeout) {
    let timer;
    const timeout = new Promise((_, reject) => {
        timer = setTimeout(() => {
            onTimeout();
            reject(Object.assign(new Error("WebGPU deadline exceeded"), { status: 4 }));
        }, milliseconds);
    });
    return Promise.race([promise, timeout]).finally(() => clearTimeout(timer));
}

export function closeDiagnosticCanvas(canvas) {
    const owner = canvasHosts.get(canvas);
    if (!owner) return;
    owner.closed = true;
    owner.host?.close();
    if (owner.retained) canvasHosts.delete(canvas);
}

export async function runDiagnostic(wasmURL = new URL("./SAGAWebDiagnostic.wasm", import.meta.url), canvas) {
    if (canvas && canvasHosts.has(canvas)) throw new Error("Canvas already belongs to a diagnostic");
    const owner = canvas ? { host: undefined, closed: false, retained: false } : undefined;
    if (owner) canvasHosts.set(canvas, owner);
    let host;
    const requireOpen = () => {
        if (owner?.closed) throw new Error("Canvas diagnostic closed");
    };
    try {
        const response = await fetch(wasmURL, { signal: AbortSignal.timeout(10_000) });
        requireOpen();
        if (!response.ok) throw new Error(`Swift/Wasm download failed (${response.status})`);
        const module = await WebAssembly.compile(await response.arrayBuffer());
        requireOpen();
        const opening = openWebGPU(canvas);
        host = await withDeadline(opening, 10_000, () => {
            opening.then(lateHost => lateHost.close(), () => {});
        });
        if (owner) owner.host = host;
        requireOpen();
        const result = await runReactor(module, host);
        requireOpen();
        if (owner) owner.retained = true;
        return result;
    } catch (error) {
        requireOpen();
        throw error;
    } finally {
        if (!owner?.retained) {
            if (canvas) canvasHosts.delete(canvas);
            host?.close();
        }
    }
}

async function runReactor(module, host) {
    let instance;
    const memory = () => instance.exports.memory;
    const bytes = (pointer, length) => new Uint8Array(memory().buffer, pointer >>> 0, length >>> 0);
    const replies = new Map();
    let queue = Promise.resolve();
    let failed = false;
    let complete, fail;
    const completion = new Promise((resolve, reject) => { complete = resolve; fail = reject; });
    function deliver(id, status, value) {
        replies.set(id, value);
        try { instance.exports.saga_reply(id, status, value.length); }
        finally { replies.delete(id); }
    }
    const imports = {
        schedule(pointer) {
            queueMicrotask(() => {
                try { instance.exports.saga_run_job(pointer); }
                catch (error) { fail(error); }
            });
        },
        request(id, pointer, length) {
            id >>>= 0;
            const payload = bytes(pointer, length).slice();
            queue = queue.then(async () => {
                try {
                    if (failed) throw Object.assign(new Error("WebGPU bridge closed"), { status: 3 });
                    if (payload.length > limit) throw new Error("Oversized browser bridge request");
                    const result = await withDeadline(host.handle(payload), 10_000, () => {
                        failed = true;
                        host.close();
                    });
                    deliver(id, 0, result);
                } catch (error) {
                    deliver(id, error.status ?? 5, encoder.encode(error.message ?? String(error)));
                }
            }).catch(fail);
        },
        copyReply(id, pointer, length) {
            const reply = replies.get(id >>> 0);
            if (!reply || reply.length !== length) throw new Error("Invalid browser bridge reply");
            bytes(pointer, length).set(reply);
        },
        finish(status, pointer, length) {
            const message = utf8.decode(bytes(pointer, length));
            if (status === 0) complete(message);
            else fail(new Error(message));
        },
    };

    instance = await WebAssembly.instantiate(module, {
        saga_host: imports,
        wasi_snapshot_preview1: browserWASI(memory),
    });
    instance.exports._initialize();
    const frames = [];
    const tokens = [];
    let result;
    try {
        if (typeof VideoFrame === "function") {
            for (const pixels of [[255, 0, 0, 255, 0, 255, 0, 255], [0, 0, 255, 255, 255, 255, 0, 255],
                                  [255, 255, 255, 255, 0, 0, 0, 255]]) {
                const frame = new VideoFrame(new Uint8Array(pixels), { format: "RGBA", codedWidth: 2, codedHeight: 1, timestamp: 0 });
                frames.push(frame);
                tokens.push(host.registerVideoFrame(frame));
            }
            frames[2].close();
        }
        instance.exports.saga_start(BigInt(host.maxBufferSize), host.maxTextureDimension2D,
            Number(host.supportsPresentation), Number(host.supportsReadableStorage), Number(host.supportsShaderF16),
            Number(host.supportsUniformBufferStandardLayout), Number(host.supportsIndirectFirstInstance),
            Number(host.supportsClipDistances), Number(host.supportsPrimitiveIndex), Number(host.supportsDepthClipControl),
            Number(host.supportsTimestampQuery), Number(host.supportsSubgroups), Number(host.supportsSubgroupID),
            Number(host.supportsDualSourceBlending), Number(host.supportsTextureCompressionBC), Number(host.supportsTextureCompressionETC2),
            Number(host.supportsTextureCompressionASTC), Number(host.supportsTextureCompressionBCSliced3D),
            Number(host.supportsTextureCompressionASTCSliced3D), Number(host.supportsFloat32Filtering), Number(host.supportsFloat32Blending), host.multisampledFloat32Mask,
            Number(host.supportsTextureFormatsTier1), Number(host.supportsTextureFormatsTier2), Number(host.supportsRG11B10Render), Number(host.supportsDepth32FloatStencil8),
            Number(host.supportsBGRA8UnormStorage), tokens[0] ?? 0n, tokens[1] ?? 0n, tokens[2] ?? 0n, host.maxImmediateSize,
            Number(host.supportsLinearIndexing), Number(host.supportsSubgroupSizeControl), host.subgroupMinSize, host.subgroupMaxSize,
            Number(host.supportsSubgroupUniformity), host.maxComputeWorkgroupStorageSize,
            host.maxComputeInvocationsPerWorkgroup, host.maxComputeWorkgroupSize.x,
            host.maxComputeWorkgroupSize.y, host.maxComputeWorkgroupSize.z, host.maxComputeWorkgroupsPerDimension,
            host.maxVertexAttributes, host.maxVertexBuffers, host.maxVertexBufferArrayStride,
            Number(host.supportsVertexFormatSnorm1010102), host.maxInterStageShaderVariables,
            host.maxColorAttachments, host.maxColorAttachmentBytesPerSample);
        result = await withDeadline(completion, 30_000, () => { failed = true; });
    } finally {
        for (const token of tokens) host.unregisterVideoFrame(token);
        for (const frame of frames) frame.close();
    }
    const [validation, first, second, pixels, probe, integerWords, vertexStreams, indexedVertices,
        instancesPerDraw, runtimeArrayWords, storageTailWords, storageVectors, groupedStorage,
        builtinInvocations, workgroupWords, storageBarrierWords, atomicTickets, atomicRMWWords,
        atomicCompareWords, mixedAtomicWords, uniformLoadWords, selectWords, constantWords, uniformWords,
        renderUniformPixels, matrixValues, matrixBufferValues, switchWords, textureUploadWords, textureLoadWords,
        renderTexturePixels, samplerValues, renderSamplerPixels, mipTextureWords, textureViewWords, textureGatherLanes,
        storageTextureWords, fragmentStorageTextureWords, readWriteStorageWords, textureBarrierWords,
        renderReadableStorageWords, depthAttachmentWords, depthTextureWords, depthComparisonWords, depthGatherLanes, rawDepthGather, multisampleWords, integerTextureWords, integerGatherWords, sampleCoverageWords, fragmentDepthWords, fragmentRasterWords, interpolationWord, topologyWords, blendWords, alphaCoverageWords, multipleColorWords, viewportWords, depthBiasWords, stencilWords, readOnlyDepthWords, aspectWords, stencilCopyWords, textureCopyWords, arrayWords, layerViewWords, arraySamplingWords, storageArrayWords, storageFormats, cubeWords, volumeWords, volumeStorageWords, volumeRenderWords, lineTextureWords, bitcastWords, integerBitWords, packingWords, geometricWords, matrixBuiltinWords, derivativeWords, derivativeFilterWords, numericBuiltinWords, transcendentalWords, decompositionWords, halfPrecisionWords, halfBufferWords, halfMatrixWords, halfGeometryWords, halfBitcastWords, halfNumericWords, quantizeWords, floatingVectorWords, overrideWords, computedOverrideWords, overrideArrayWords, pipelineBodyWords, pipelineAggregateWords, pipelineCompositeWords, arrayConstantWords, pipelineBuiltinWords, conversionWords, halfConversionWords, localArrayWords, workgroupAggregateWords, privateVariableWords, privateRenderWords, pointerWords, pointerSubviewWords, sharedPointerWords, swizzleAssignmentWords, nestedStorageWords, nestedHalfStorageWords, nestedAtomicWords, nestedUniformWords, nestedHalfUniformWords, standardUniformWords, renderStorageBufferWords, indirectDispatchWords, indirectDrawWords, clipDistancePixels, primitiveIndexPixels, invariantPositionPixels, voidFragmentPixels, discardContinuationWords, depthClipWords, timestampQueryWords, occlusionQueryWords, mappedBufferWords, bufferRangeWords, queueTextureWords, subgroupGatherWords, subgroupCollectiveWords, subgroupIndexWords, dualSourcePixels, bcTextureWords, etc2TextureWords, astcTextureWords, bcVolumeWords, astcVolumeWords, callStatementWords, matrixCompoundWords, float32TextureWords, float32ColorWords, halfTexelWords, halfColorWords, halfStorageWords, integer16ColorWords, integer16MSAAWords, integer16StorageWords, normalized16StorageWords, normalized16ColorWords, packedColorWords, packedStorageWords, srgbWords, sharedExponentWords, byteTransferWords, byteColorWords, byteStorageWords, depthOnlyFormatWords, depth32StencilWords, stencilOnlyWords, bgraStorageWords, wideStorageWords, forwardedResourceWords, webBufferArrayWords, externalTextureWords, immediateWords, renderImmediateWords, linearIndexingInvocations, requestedSubgroupSize, subgroupUniformityWords] = result.split("|");
    const readback = [first, second].map(snapshot => snapshot.split(",").map(Number));
    return {
        ...host.adapter, validation, readback, renderPixels: Number(pixels),
        renderProbe: probe.split(",").map(Number), integerWords: Number(integerWords),
        vertexStreams: Number(vertexStreams), indexedVertices: Number(indexedVertices),
        instancesPerDraw: Number(instancesPerDraw), runtimeArrayWords: Number(runtimeArrayWords),
        storageTailWords: Number(storageTailWords), storageVectors: Number(storageVectors),
        groupedStorage: groupedStorage.split(",").map(Number), builtinInvocations: Number(builtinInvocations),
        workgroupWords: workgroupWords.split(",").map(Number),
        maxComputeWorkgroupStorageSize: host.maxComputeWorkgroupStorageSize,
        maxComputeInvocationsPerWorkgroup: host.maxComputeInvocationsPerWorkgroup,
        maxComputeWorkgroupSize: host.maxComputeWorkgroupSize,
        maxComputeWorkgroupsPerDimension: host.maxComputeWorkgroupsPerDimension,
        maxVertexAttributes: host.maxVertexAttributes,
        maxVertexBuffers: host.maxVertexBuffers,
        maxVertexBufferArrayStride: host.maxVertexBufferArrayStride,
        maxInterStageShaderVariables: host.maxInterStageShaderVariables,
        maxColorAttachments: host.maxColorAttachments,
        maxColorAttachmentBytesPerSample: host.maxColorAttachmentBytesPerSample,
        supportsVertexFormatSnorm1010102: host.supportsVertexFormatSnorm1010102,
        storageBarrierWords: storageBarrierWords.split(",").map(Number), atomicTickets: Number(atomicTickets),
        atomicRMWWords: Number(atomicRMWWords), atomicCompareWords: Number(atomicCompareWords),
        mixedAtomicWords: Number(mixedAtomicWords),
        uniformLoadWords: uniformLoadWords.split(",").map(Number),
        selectWords: selectWords.split(",").map(Number),
        constantWords: constantWords.split(",").map(Number),
        uniformWords: uniformWords.split(",").map(Number),
        renderUniformPixels: renderUniformPixels.split(",").map(Number),
        matrixValues: matrixValues.split(",").map(Number),
        matrixBufferValues: matrixBufferValues.split(",").map(Number),
        switchWords: switchWords.split(",").map(Number),
        textureUploadWords: textureUploadWords.split(",").map(Number),
        textureLoadWords: textureLoadWords.split(",").map(Number),
        renderTexturePixels: renderTexturePixels.split(",").map(Number),
        samplerValues: samplerValues.split(",").map(Number),
        renderSamplerPixels: renderSamplerPixels.split(",").map(Number),
        mipTextureWords: mipTextureWords.split(",").map(Number),
        textureViewWords: textureViewWords.split(",").map(Number),
        textureGatherLanes: textureGatherLanes.split(",").map(Number),
        storageTextureWords: storageTextureWords.split(",").map(Number),
        fragmentStorageTextureWords: fragmentStorageTextureWords.split(",").map(Number),
        readWriteStorageWords: readWriteStorageWords ? readWriteStorageWords.split(",").map(Number) : [],
        textureBarrierWords: textureBarrierWords ? textureBarrierWords.split(",").map(Number) : [],
        renderReadableStorageWords: renderReadableStorageWords ? renderReadableStorageWords.split(",").map(Number) : [],
        depthAttachmentWords: depthAttachmentWords.split(",").map(Number),
        depthTextureWords: depthTextureWords.split(",").map(Number),
        depthComparisonWords: depthComparisonWords.split(",").map(Number),
        depthGatherLanes: depthGatherLanes.split(",").map(Number),
        rawDepthGather: rawDepthGather.split(",").map(Number),
        multisampleWords: multisampleWords.split(",").map(Number),
        integerTextureWords: integerTextureWords.split(",").map(Number),
        integerGatherWords: integerGatherWords.split(",").map(Number),
        sampleCoverageWords: sampleCoverageWords.split(",").map(Number),
        fragmentDepthWords: fragmentDepthWords.split(",").map(Number),
        fragmentRasterWords: fragmentRasterWords.split(",").map(Number),
        interpolationWord: Number(interpolationWord),
        topologyWords: topologyWords.split(",").map(Number),
        blendWords: blendWords.split(",").map(Number),
        alphaCoverageWords: alphaCoverageWords.split(",").map(Number),
        multipleColorWords: multipleColorWords.split(",").map(Number),
        viewportWords: viewportWords.split(",").map(Number),
        depthBiasWords: depthBiasWords.split(",").map(Number),
        stencilWords: stencilWords.split(",").map(Number),
        readOnlyDepthWords: readOnlyDepthWords.split(",").map(Number),
        aspectWords: aspectWords.split(",").map(Number),
        stencilCopyWords: stencilCopyWords.split(",").map(Number),
        textureCopyWords: textureCopyWords.split(",").map(Number),
        arrayWords: arrayWords.split(",").map(Number),
        layerViewWords: layerViewWords.split(",").map(Number),
        arraySamplingWords: arraySamplingWords.split(",").map(Number),
        storageArrayWords: storageArrayWords.split(",").map(Number),
        storageFormats: Number(storageFormats),
        cubeWords: cubeWords.split(",").map(Number),
        volumeWords: volumeWords.split(",").map(Number),
        volumeStorageWords: volumeStorageWords.split(",").map(Number),
        volumeRenderWords: volumeRenderWords.split(",").map(Number),
        lineTextureWords: lineTextureWords.split(",").map(Number),
        bitcastWords: bitcastWords.split(",").map(Number),
        integerBitWords: integerBitWords.split(",").map(Number),
        packingWords: packingWords.split(",").map(Number),
        geometricWords: geometricWords.split(",").map(Number),
        matrixBuiltinWords: matrixBuiltinWords.split(",").map(Number),
        derivativeWords: derivativeWords.split(",").map(Number),
        derivativeFilterWords: derivativeFilterWords.split(",").map(Number),
        numericBuiltinWords: numericBuiltinWords.split(",").map(Number),
        transcendentalWords: transcendentalWords.split(",").map(Number),
        decompositionWords: decompositionWords.split(",").map(Number),
        halfPrecisionWords: host.supportsShaderF16 ? halfPrecisionWords.split(",").map(Number) : null,
        halfBufferWords: host.supportsShaderF16 ? halfBufferWords.split(",").map(Number) : null,
        halfMatrixWords: host.supportsShaderF16 ? halfMatrixWords.split(",").map(Number) : null,
        halfGeometryWords: host.supportsShaderF16 ? halfGeometryWords.split(",").map(Number) : null,
        halfBitcastWords: host.supportsShaderF16 ? halfBitcastWords.split(",").map(Number) : null,
        halfNumericWords: host.supportsShaderF16 ? halfNumericWords.split(",").map(Number) : null,
        quantizeWords: quantizeWords.split(",").map(Number),
        floatingVectorWords: host.supportsShaderF16 ? floatingVectorWords.split(",").map(Number) : null,
        overrideWords: overrideWords.split(",").map(Number),
        computedOverrideWords: computedOverrideWords.split(",").map(Number),
        overrideArrayWords: overrideArrayWords.split(",").map(Number),
        pipelineBodyWords: pipelineBodyWords.split(",").map(Number),
        pipelineAggregateWords: pipelineAggregateWords.split(",").map(Number),
        pipelineCompositeWords: pipelineCompositeWords.split(",").map(Number),
        arrayConstantWords: arrayConstantWords.split(",").map(Number),
        pipelineBuiltinWords: pipelineBuiltinWords.split(",").map(Number),
        conversionWords: conversionWords.split(",").map(Number),
        halfConversionWords: host.supportsShaderF16 ? halfConversionWords.split(",").map(Number) : null,
        localArrayWords: localArrayWords.split(",").map(Number),
        workgroupAggregateWords: workgroupAggregateWords.split(",").map(Number),
        privateVariableWords: privateVariableWords.split(",").map(Number),
        privateRenderWords: privateRenderWords.split(",").map(Number),
        pointerWords: pointerWords.split(",").map(Number),
        pointerSubviewWords: pointerSubviewWords.split(",").map(Number),
        sharedPointerWords: sharedPointerWords.split(",").map(Number),
        swizzleAssignmentWords: swizzleAssignmentWords.split(",").map(Number),
        nestedStorageWords: nestedStorageWords.split(",").map(Number),
        nestedHalfStorageWords: host.supportsShaderF16 ? nestedHalfStorageWords.split(",").map(Number) : null,
        nestedAtomicWords: nestedAtomicWords.split(",").map(Number),
        nestedUniformWords: nestedUniformWords.split(",").map(Number),
        nestedHalfUniformWords: host.supportsShaderF16 ? nestedHalfUniformWords.split(",").map(Number) : null,
        standardUniformWords: host.supportsUniformBufferStandardLayout ? standardUniformWords.split(",").map(Number) : null,
        renderStorageBufferWords: renderStorageBufferWords.split(",").map(Number),
        indirectDispatchWords: indirectDispatchWords.split(",").map(Number),
        indirectDrawWords: indirectDrawWords.split(",").map(Number),
        supportsIndirectFirstInstance: host.supportsIndirectFirstInstance,
        clipDistancePixels: host.supportsClipDistances ? clipDistancePixels.split(",").map(Number) : null,
        primitiveIndexPixels: host.supportsPrimitiveIndex ? primitiveIndexPixels.split(",").map(Number) : null,
        invariantPositionPixels: invariantPositionPixels.split(",").map(Number),
        voidFragmentPixels: voidFragmentPixels.split(",").map(Number),
        discardContinuationWords: discardContinuationWords.split(",").map(Number),
        depthClipWords: host.supportsDepthClipControl ? depthClipWords.split(",").map(Number) : null,
        timestampQueryWords: host.supportsTimestampQuery ? timestampQueryWords.split(",").map(Number) : null,
        occlusionQueryWords: occlusionQueryWords.split(",").map(Number),
        mappedBufferWords: mappedBufferWords.split(",").map(Number),
        bufferRangeWords: bufferRangeWords.split(",").map(Number),
        queueTextureWords: queueTextureWords.split(",").map(Number),
        subgroupGatherWords: host.supportsSubgroups ? subgroupGatherWords.split(",").map(Number) : null,
        subgroupCollectiveWords: host.supportsSubgroups ? subgroupCollectiveWords.split(",").map(Number) : null,
        subgroupIndexWords: host.supportsSubgroupID ? subgroupIndexWords.split(",").map(Number) : null,
        dualSourcePixels: host.supportsDualSourceBlending ? dualSourcePixels.split(",").map(Number) : null,
        bcTextureWords: host.supportsTextureCompressionBC ? bcTextureWords.split(",").map(Number) : null,
        etc2TextureWords: host.supportsTextureCompressionETC2 ? etc2TextureWords.split(",").map(Number) : null,
        astcTextureWords: host.supportsTextureCompressionASTC ? astcTextureWords.split(",").map(Number) : null,
        bcVolumeWords: host.supportsTextureCompressionBCSliced3D ? bcVolumeWords.split(",").map(Number) : null,
        astcVolumeWords: host.supportsTextureCompressionASTCSliced3D ? astcVolumeWords.split(",").map(Number) : null,
        callStatementWords: callStatementWords.split(",").map(Number),
        matrixCompoundWords: matrixCompoundWords.split(",").map(Number),
        float32TextureWords: float32TextureWords.split(",").map(Number),
        float32ColorWords: float32ColorWords.split(",").map(Number),
        halfTexelWords: halfTexelWords.split(",").map(Number),
        halfColorWords: halfColorWords.split(",").map(Number),
        halfStorageWords: halfStorageWords ? halfStorageWords.split(",").map(Number) : [],
        integer16ColorWords: integer16ColorWords.split(",").map(Number),
        integer16MSAAWords: integer16MSAAWords.split(",").map(Number),
        integer16StorageWords: integer16StorageWords ? integer16StorageWords.split(",").map(Number) : [],
        normalized16StorageWords: normalized16StorageWords ? normalized16StorageWords.split(",").map(Number) : [],
        normalized16ColorWords: normalized16ColorWords ? normalized16ColorWords.split(",").map(Number) : [],
        packedColorWords: packedColorWords.split(",").map(Number),
        packedStorageWords: packedStorageWords ? packedStorageWords.split(",").map(Number) : [],
        srgbWords: srgbWords.split(",").map(Number),
        sharedExponentWords: sharedExponentWords.split(",").map(Number),
        byteTransferWords: byteTransferWords.split(",").map(Number),
        byteColorWords: byteColorWords.split(",").map(Number),
        byteStorageWords: byteStorageWords ? byteStorageWords.split(",").map(Number) : [],
        depthOnlyFormatWords: depthOnlyFormatWords.split(",").map(Number),
        depth32StencilWords: depth32StencilWords ? depth32StencilWords.split(",").map(Number) : [],
        stencilOnlyWords: stencilOnlyWords.split(",").map(Number),
        bgraStorageWords: bgraStorageWords ? bgraStorageWords.split(",").map(Number) : [],
        wideStorageWords: wideStorageWords ? wideStorageWords.split(",").map(Number) : [],
        forwardedResourceWords: forwardedResourceWords.split(",").map(Number),
        webBufferArrayWords: webBufferArrayWords.split(",").map(Number),
        externalTextureWords: externalTextureWords ? externalTextureWords.split(",").map(Number) : [],
        maxImmediateSize: host.maxImmediateSize,
        immediateWords: immediateWords ? immediateWords.split(",").map(Number) : [],
        renderImmediateWords: renderImmediateWords ? renderImmediateWords.split(",").map(Number) : [],
        supportsLinearIndexing: host.supportsLinearIndexing,
        linearIndexingInvocations: Number(linearIndexingInvocations),
        supportsSubgroupSizeControl: host.supportsSubgroupSizeControl,
        subgroupMinSize: host.subgroupMinSize,
        requestedSubgroupSize: Number(requestedSubgroupSize),
        supportsSubgroupUniformity: host.supportsSubgroupUniformity,
        subgroupUniformityWords: subgroupUniformityWords ? subgroupUniformityWords.split(",").map(Number) : [],
        multisampledFloat32Mask: host.multisampledFloat32Mask,
        ...host.metrics(),
    };
}
