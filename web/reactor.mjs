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
    instance.exports.saga_start(BigInt(host.maxBufferSize), host.maxTextureDimension2D,
        Number(host.supportsPresentation), Number(host.supportsReadableStorage));
    const result = await withDeadline(completion, 30_000, () => { failed = true; });
    const [validation, first, second, pixels, probe, integerWords, vertexStreams, indexedVertices,
        instancesPerDraw, runtimeArrayWords, storageTailWords, storageVectors, groupedStorage,
        builtinInvocations, workgroupWords, storageBarrierWords, atomicTickets, atomicRMWWords,
        atomicCompareWords, mixedAtomicWords, uniformLoadWords, selectWords, constantWords, uniformWords,
        renderUniformPixels, matrixValues, matrixBufferValues, switchWords, textureUploadWords, textureLoadWords,
        renderTexturePixels, samplerValues, renderSamplerPixels, mipTextureWords, textureViewWords, textureGatherLanes,
        storageTextureWords, fragmentStorageTextureWords, readWriteStorageWords, textureBarrierWords,
        renderReadableStorageWords, depthAttachmentWords, depthTextureWords, depthComparisonWords, depthGatherLanes, rawDepthGather, multisampleWords, integerTextureWords, integerGatherWords, sampleCoverageWords, fragmentDepthWords, fragmentRasterWords, interpolationWord, topologyWords, blendWords, alphaCoverageWords, multipleColorWords, viewportWords, depthBiasWords, stencilWords, readOnlyDepthWords, aspectWords, stencilCopyWords, textureCopyWords, arrayWords, layerViewWords, arraySamplingWords, storageArrayWords, storageFormats, cubeWords, volumeWords] = result.split("|");
    const readback = [first, second].map(snapshot => snapshot.split(",").map(Number));
    return {
        ...host.adapter, validation, readback, renderPixels: Number(pixels),
        renderProbe: probe.split(",").map(Number), integerWords: Number(integerWords),
        vertexStreams: Number(vertexStreams), indexedVertices: Number(indexedVertices),
        instancesPerDraw: Number(instancesPerDraw), runtimeArrayWords: Number(runtimeArrayWords),
        storageTailWords: Number(storageTailWords), storageVectors: Number(storageVectors),
        groupedStorage: groupedStorage.split(",").map(Number), builtinInvocations: Number(builtinInvocations),
        workgroupWords: workgroupWords.split(",").map(Number),
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
        ...host.metrics(),
    };
}
