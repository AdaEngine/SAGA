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
        const realGPU = navigator.gpu, adapter = await realGPU.requestAdapter();
        if (!adapter || adapter.info.isFallbackAdapter) throw new Error("No hardware WebGPU adapter");
        const wrap = (target, overrides) => new Proxy(target, { get(object, key) {
            if (key in overrides) return overrides[key];
            const value = Reflect.get(object, key, object);
            return typeof value === "function" ? value.bind(object) : value;
        } });
        const packet = (op, length) => {
            const bytes = new Uint8Array(length), view = new DataView(bytes.buffer);
            view.setUint32(0, op, true); view.setBigUint64(4, 1n, true); view.setBigUint64(12, 1n, true);
            return { bytes, view };
        };
        let host, buffer, device, unmap, initialState;
        let cancelFirst = true;
        const wrappedAdapter = wrap(adapter, { requestDevice: async descriptor => {
            device = await adapter.requestDevice(descriptor);
            return wrap(device, { createBuffer: descriptor => {
                buffer = device.createBuffer(descriptor);
                return wrap(buffer, { mapAsync: (...args) => {
                    const pending = buffer.mapAsync(...args);
                    if (cancelFirst) {
                        cancelFirst = false;
                        initialState = buffer.mapState;
                        // Invoke unmap in the same task, before the native promise can settle.
                        unmap = host.handle(packet(22, 20).bytes);
                    }
                    return pending;
                } });
            } });
        } });
        Object.defineProperty(navigator, "gpu", { configurable: true, value: wrap(realGPU, {
            requestAdapter: async () => wrappedAdapter,
        }) });
        const { openWebGPU } = await import("/webgpu.mjs");
        host = await openWebGPU();
        const create = packet(1, 36);
        create.view.setBigUint64(20, 4n, true);
        create.view.setUint32(28, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST, true);
        await host.handle(create.bytes);
        device.queue.writeBuffer(buffer, 0, new Uint32Array([0x12345678]));
        const map = packet(19, 40);
        map.view.setUint32(20, GPUMapMode.READ, true); map.view.setBigUint64(32, 4n, true);
        let rejected = false;
        try { await host.handle(map.bytes); } catch { rejected = true; }
        await unmap;
        const afterCancel = buffer.mapState;
        await host.handle(map.bytes);
        const read = packet(20, 32); read.view.setUint32(28, 4, true);
        const bytes = Array.from(await host.handle(read.bytes));
        await host.handle(packet(22, 20).bytes);
        host.close();
        return { initialState, rejected, afterCancel, bytes };
    });
    assert.deepEqual(result, { initialState: "pending", rejected: true, afterCancel: "unmapped",
        bytes: [0x78, 0x56, 0x34, 0x12] });
    console.log("WebGPU pending map cancellation and remap readback passed");
} finally { await browser.close(); }
