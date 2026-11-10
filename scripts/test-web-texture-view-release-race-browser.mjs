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
        let gate, creations = 0, views = 0, realViews = 0;
        const wrappedAdapter = wrap(adapter, { requestDevice: async descriptor => {
            const freshAdapter = await gpu.requestAdapter();
            if (!freshAdapter || freshAdapter.info.isFallbackAdapter) throw new Error("No hardware WebGPU adapter");
            const device = await freshAdapter.requestDevice(descriptor);
            return wrap(device, {
                popErrorScope: () => {
                    const pending = device.popErrorScope();
                    return gate ? Promise.all([pending, gate]).then(([error]) => error) : pending;
                },
                createTexture: descriptor => {
                    const texture = device.createTexture(descriptor);
                    creations++;
                    return wrap(texture, { createView: descriptor => {
                        const view = texture.createView(descriptor);
                        views++;
                        if (view instanceof GPUTextureView) realViews++;
                        return view;
                    } });
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
        const texture = id => packet(8, id, [1, 1, 1, GPUTextureUsage.TEXTURE_BINDING, 1, 1, 1, 0]);
        const view = packet(15, 2n, [1, 0, 1, 0, 0, 0, 0, 0, 0, 0]);
        const results = [];
        for (const afterCreation of [false, true]) {
            const host = await openWebGPU();
            await host.handle(texture(1n));
            let release;
            gate = new Promise(resolve => { release = resolve; });
            let blocker;
            if (!afterCreation) {
                const before = creations;
                blocker = host.handle(texture(3n));
                while (creations === before) await Promise.resolve();
            }
            const before = views;
            const stale = host.handle(view);
            if (afterCreation) while (views === before) await Promise.resolve();
            await host.handle(packet(9, 1n));
            const replacement = host.handle(texture(1n));
            release();
            let error;
            try { await stale; } catch (caught) { error = { message: caught.message, status: caught.status }; }
            if (blocker) await blocker;
            await replacement;
            const staleViews = views - before;
            await host.handle(view);
            results.push({ error, staleViews, resources: host.metrics().remainingResources });
            gate = undefined;
            host.close();
        }
        return { results, views, realViews };
    });
    for (const [index, value] of result.results.entries()) {
        assert.match(value.error?.message ?? "", /texture.*released/i);
        assert.equal(value.error.status, 1);
        assert.equal(value.staleViews, index);
        assert.equal(value.resources, index === 0 ? 3 : 2);
    }
    assert.equal(result.views, 3);
    assert.equal(result.realViews, result.views);
    console.log("WebGPU texture view release races and replacement passed on hardware adapter");
} finally { await browser.close(); }
