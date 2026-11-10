import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { createRequire } from "node:module";
const require = createRequire(import.meta.url);
const { chromium } = require(process.env.SAGA_PLAYWRIGHT ?? "playwright-core");
const browser = await chromium.launch({ executablePath: process.env.SAGA_CHROMIUM,
    args: ["--enable-unsafe-webgpu", ...(process.platform === "darwin" ? ["--use-angle=metal"] : [])] });
try {
    const page = await browser.newPage();
    await page.route("http://localhost:8799/**", async route => {
        const path = new URL(route.request().url()).pathname;
        await route.fulfill(path === "/webgpu.mjs"
            ? { contentType: "text/javascript", body: await readFile(new URL("../web/webgpu.mjs", import.meta.url)) }
            : { contentType: "text/html", body: "<!doctype html>" });
    });
    await page.goto("http://localhost:8799/");
    const result = await page.evaluate(async () => {
        const gpu = navigator.gpu, adapter = await gpu.requestAdapter();
        if (!adapter || adapter.info.isFallbackAdapter) throw new Error("No hardware WebGPU adapter");
        const wrap = (target, overrides) => new Proxy(target, { get(object, key) {
            if (key in overrides) return overrides[key];
            const value = Reflect.get(object, key, object);
            return typeof value === "function" ? value.bind(object) : value;
        } });
        let gate, mapStarted, clears = 0, realBuffers = 0;
        const wrappedAdapter = wrap(adapter, { requestDevice: async descriptor => {
            const fresh = await gpu.requestAdapter();
            if (!fresh || fresh.info.isFallbackAdapter) throw new Error("No hardware WebGPU adapter");
            const device = await fresh.requestDevice(descriptor);
            return wrap(device, {
                createBuffer: descriptor => {
                    const buffer = device.createBuffer(descriptor);
                    if (buffer instanceof GPUBuffer) realBuffers++;
                    if (!(descriptor.usage & GPUBufferUsage.MAP_READ)) return buffer;
                    return wrap(buffer, { mapAsync: async (...args) => {
                        const pending = buffer.mapAsync(...args);
                        await pending;
                        mapStarted?.();
                        if (gate) await gate;
                    } });
                },
                createCommandEncoder: descriptor => {
                    const encoder = device.createCommandEncoder(descriptor);
                    return wrap(encoder, { clearBuffer: (...args) => {
                        clears++;
                        return encoder.clearBuffer(...args);
                    } });
                },
            });
        } });
        // Native WebGPU methods require their native GPUBuffer, not a Proxy.
        // Only the MAP_READ buffer needs wrapping; clearBuffer uses the native target.
        Object.defineProperty(navigator, "gpu", { configurable: true, value: wrap(gpu, {
            requestAdapter: async () => wrappedAdapter,
        }) });
        const { openWebGPU } = await import("/webgpu.mjs");
        const host = await openWebGPU();
        class Packet {
            bytes = [];
            word(value) { for (let i = 0; i < 4; i++) this.bytes.push((value >>> (8 * i)) & 255); return this; }
            wide(value) { return this.word(value).word(0); }
            id(value) { return this.wide(1).wide(value); }
            data() { return Uint8Array.from(this.bytes); }
        }
        const allocate = (id, usage) => new Packet().word(1).id(id).wide(4).word(usage).word(0).data();
        const submit = id => new Packet().word(6).word(2).word(4).id(1).wide(4)
            .word(16).id(id).wide(0).word(1).wide(4).data();
        await host.handle(allocate(1, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST));
        await host.handle(allocate(2, GPUBufferUsage.COPY_DST));
        let release, signal;
        gate = new Promise(resolve => { release = resolve; });
        const started = new Promise(resolve => { signal = resolve; });
        mapStarted = signal;
        const stale = host.handle(submit(2)).then(() => null, error => ({ message: error.message, status: error.status }));
        await started;
        await host.handle(new Packet().word(2).id(2).data());
        release();
        const error = await stale;
        const staleClears = clears;
        gate = undefined;
        mapStarted = undefined;
        await host.handle(allocate(3, GPUBufferUsage.COPY_DST));
        await host.handle(submit(3));
        const metrics = host.metrics();
        host.close();
        return { error, staleClears, clears, realBuffers, metrics };
    });
    assert.match(result.error?.message ?? "", /resource.*released/i);
    assert.equal(result.error.status, 1);
    assert.equal(result.staleClears, 0);
    assert.equal(result.clears, 1, "a live buffer must accept a subsequent submission");
    assert.equal(result.realBuffers, 3);
    console.log("WebGPU resource release during submission read passed on hardware adapter");
} finally { await browser.close(); }
