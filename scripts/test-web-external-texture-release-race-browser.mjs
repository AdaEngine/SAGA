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
        let gate, creations = 0, imports = 0, realImports = 0;
        const wrappedAdapter = wrap(adapter, { requestDevice: async descriptor => {
            const freshAdapter = await gpu.requestAdapter();
            if (!freshAdapter || freshAdapter.info.isFallbackAdapter) throw new Error("No hardware WebGPU adapter");
            const device = await freshAdapter.requestDevice(descriptor);
            return wrap(device, {
                popErrorScope: () => {
                    const pending = device.popErrorScope();
                    return gate ? Promise.all([pending, gate]).then(([error]) => error) : pending;
                },
                createTexture: descriptor => { creations++; return device.createTexture(descriptor); },
                importExternalTexture: descriptor => {
                    const external = device.importExternalTexture(descriptor);
                    imports++;
                    if (external instanceof GPUExternalTexture) realImports++;
                    return external;
                },
            });
        } });
        Object.defineProperty(navigator, "gpu", { configurable: true, value: wrap(gpu, {
            requestAdapter: async () => wrappedAdapter,
        }) });
        const { openWebGPU } = await import("/webgpu.mjs");
        function packet(op, id, values = []) {
            const bytes = new Uint8Array(20 + values.length * 4), data = new DataView(bytes.buffer);
            data.setUint32(0, op, true); data.setBigUint64(4, 1n, true); data.setBigUint64(12, id, true);
            values.forEach((value, index) => data.setUint32(20 + 4 * index, value, true));
            return bytes;
        }
        const frame = () => new VideoFrame(new Uint8Array([255, 0, 0, 255]),
            { format: "RGBA", codedWidth: 1, codedHeight: 1, timestamp: 0 });
        const external = token => packet(26, 2n, [Number(token & 0xffffffffn), Number(token >> 32n), 0]);
        const results = [];
        for (const afterImport of [false, true]) for (const action of ["unregister", "close"]) {
            gate = undefined;
            const host = await openWebGPU();
            const source = frame(), token = host.registerVideoFrame(source);
            let release;
            gate = new Promise(resolve => { release = resolve; });
            let blocker;
            if (!afterImport) {
                const before = creations;
                blocker = host.handle(packet(8, 3n, [1, 1, 1, GPUTextureUsage.TEXTURE_BINDING, 1, 1, 1, 0]));
                while (creations === before) await Promise.resolve();
            }
            const before = imports;
            const stale = host.handle(external(token));
            if (afterImport) while (imports === before) await Promise.resolve();
            if (action === "unregister") host.unregisterVideoFrame(token);
            else source.close();
            release();
            let error;
            try { await stale; } catch (caught) { error = { message: caught.message, status: caught.status }; }
            if (blocker) await blocker;
            const staleImports = imports - before;
            gate = undefined;
            host.unregisterVideoFrame(token);
            source.close();
            const replacement = frame(), replacementToken = host.registerVideoFrame(replacement);
            await host.handle(external(replacementToken));
            const resources = host.metrics().remainingResources;
            await host.handle(packet(27, 2n));
            host.unregisterVideoFrame(replacementToken);
            replacement.close();
            host.close();
            results.push({ error, staleImports, resources });
        }
        return { results, imports, realImports };
    });
    for (const [index, value] of result.results.entries()) {
        assert.match(value.error?.message ?? "", /VideoFrame/);
        assert.equal(value.error.status, 1);
        assert.equal(value.staleImports, index < 2 ? 0 : 1);
        assert.equal(value.resources, index < 2 ? 3 : 2);
    }
    assert.equal(result.imports, 6);
    assert.equal(result.realImports, result.imports);
    console.log("WebGPU external texture frame release races and replacement passed on hardware adapter");
} finally { await browser.close(); }
