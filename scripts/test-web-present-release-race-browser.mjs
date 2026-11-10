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
            : { contentType: "text/html", body: "<!doctype html><canvas></canvas>" });
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
        let gate, realDevice, creations = 0, realTextures = 0, encoders = 0, configurations = 0;
        const wrappedAdapter = wrap(adapter, { requestDevice: async descriptor => {
            const freshAdapter = await gpu.requestAdapter();
            if (!freshAdapter || freshAdapter.info.isFallbackAdapter) throw new Error("No hardware WebGPU adapter");
            realDevice = await freshAdapter.requestDevice(descriptor);
            return wrap(realDevice, {
                popErrorScope: () => {
                    const pending = realDevice.popErrorScope();
                    return gate ? Promise.all([pending, gate]).then(([error]) => error) : pending;
                },
                createTexture: descriptor => {
                    const texture = realDevice.createTexture(descriptor);
                    creations++;
                    if (texture instanceof GPUTexture) realTextures++;
                    return texture;
                },
                createCommandEncoder: descriptor => { encoders++; return realDevice.createCommandEncoder(descriptor); },
            });
        } });
        Object.defineProperty(navigator, "gpu", { configurable: true, value: wrap(gpu, {
            requestAdapter: async () => wrappedAdapter,
        }) });
        const canvas = document.querySelector("canvas"), context = canvas.getContext("webgpu");
        if (!context) throw new Error("No WebGPU canvas context");
        const hostCanvas = {
            get width() { return canvas.width; }, set width(value) { canvas.width = value; },
            get height() { return canvas.height; }, set height(value) { canvas.height = value; },
            getContext: () => wrap(context, { configure: descriptor => {
                configurations++;
                context.configure({ ...descriptor, device: realDevice });
            } }),
        };
        const { openWebGPU } = await import("/webgpu.mjs");
        const host = await openWebGPU(hostCanvas);
        function packet(op, id, values = []) {
            const bytes = new Uint8Array(20 + values.length * 4), data = new DataView(bytes.buffer);
            data.setUint32(0, op, true); data.setBigUint64(4, 1n, true); data.setBigUint64(12, id, true);
            values.forEach((value, index) => data.setUint32(20 + 4 * index, value, true));
            return bytes;
        }
        const texture = id => packet(8, id, [1, 1, 1, GPUTextureUsage.COPY_SRC, 1, 1, 1, 0]);
        await host.handle(texture(1n));
        let release;
        gate = new Promise(resolve => { release = resolve; });
        const before = creations;
        const blocker = host.handle(texture(3n));
        while (creations === before) await Promise.resolve();
        const stale = host.handle(packet(12, 1n));
        await host.handle(packet(9, 1n));
        const replacement = host.handle(texture(1n));
        release();
        let error;
        try { await stale; } catch (caught) { error = { message: caught.message, status: caught.status }; }
        await blocker;
        await replacement;
        const staleEncoders = encoders, staleConfigurations = configurations;
        gate = undefined;
        await host.handle(packet(12, 1n));
        const metrics = host.metrics();
        host.close();
        return { error, staleEncoders, staleConfigurations, encoders, configurations, realTextures, creations, metrics };
    });
    assert.match(result.error?.message ?? "", /texture.*released/i);
    assert.equal(result.error.status, 1);
    assert.equal(result.staleEncoders, 0);
    assert.equal(result.staleConfigurations, 0);
    assert.equal(result.encoders, 1);
    assert.equal(result.configurations, 1);
    assert.equal(result.realTextures, result.creations);
    assert.equal(result.metrics.presentedFrames, 1);
    console.log("WebGPU presentation release race and replacement passed on hardware adapter");
} finally { await browser.close(); }
