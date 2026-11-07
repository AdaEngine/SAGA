import { browserWASI } from "./wasi.mjs";
import { openWebGPU } from "./webgpu.mjs";

const utf8 = new TextDecoder();
const encoder = new TextEncoder();
const limit = 16 * 1024 * 1024;

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

export async function runDiagnostic(wasmURL = new URL("./SAGAWebDiagnostic.wasm", import.meta.url)) {
    const response = await fetch(wasmURL, { signal: AbortSignal.timeout(10_000) });
    if (!response.ok) throw new Error(`Swift/Wasm download failed (${response.status})`);
    const module = await WebAssembly.compile(await response.arrayBuffer());
    const opening = openWebGPU();
    const host = await withDeadline(opening, 10_000, () => {
        opening.then(lateHost => lateHost.close(), () => {});
    });
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

    try {
        instance = await WebAssembly.instantiate(module, {
            saga_host: imports,
            wasi_snapshot_preview1: browserWASI(memory),
        });
        instance.exports._initialize();
        instance.exports.saga_start(BigInt(host.maxBufferSize), host.maxTextureDimension2D);
        const result = await withDeadline(completion, 30_000, () => { failed = true; });
        const [validation, first, second, pixels, probe, integerWords, vertexStreams] = result.split("|");
        const snapshots = [first, second];
        const readback = snapshots.map(snapshot => snapshot.split(",").map(Number));
        return {
            ...host.adapter, validation, readback, renderPixels: Number(pixels),
            renderProbe: probe.split(",").map(Number), integerWords: Number(integerWords),
            vertexStreams: Number(vertexStreams), ...host.metrics(),
        };
    } finally {
        host.close();
    }
}
