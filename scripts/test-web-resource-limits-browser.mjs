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
        const gpu = navigator.gpu;
        if (!gpu) throw new Error("WebGPU unavailable");
        const adapter = await gpu.requestAdapter();
        if (!adapter || adapter.info.isFallbackAdapter) throw new Error("No hardware WebGPU adapter");
        const names = ["maxBufferSize", "maxTextureDimension2D"];
        const limits = object => Object.fromEntries(names.map(name => [name, object[name] ?? null]));
        let requested, deviceLimits;
        const wrap = (target, overrides) => new Proxy(target, { get(object, key) {
            if (key in overrides) return overrides[key];
            const value = Reflect.get(object, key, object);
            return typeof value === "function" ? value.bind(object) : value;
        } });
        const wrappedAdapter = wrap(adapter, { requestDevice: async descriptor => {
            requested = limits(descriptor.requiredLimits);
            const device = await adapter.requestDevice(descriptor);
            deviceLimits = limits(device.limits);
            return device;
        } });
        Object.defineProperty(navigator, "gpu", { configurable: true, value: wrap(gpu, {
            requestAdapter: async () => wrappedAdapter,
        }) });
        const { openWebGPU } = await import("/webgpu.mjs");
        const host = await openWebGPU();
        try { return { adapter: limits(adapter.limits), requested, device: deviceLimits, host: limits(host) }; }
        finally { host.close(); }
    });
    console.log(`WebGPU resource limits probe: ${JSON.stringify(result)}`);
    for (const name of ["maxBufferSize", "maxTextureDimension2D"]) {
        if (result.adapter[name] === null) {
            assert.equal(result.requested[name], null);
            assert.equal(result.host[name], result.device[name]);
            console.log(`UNAVAILABLE: adapter ${name}; host exposes the device default`);
        } else {
            assert.equal(result.requested[name], result.adapter[name], `${name} requested`);
            assert.equal(result.device[name], result.adapter[name], `${name} device`);
            assert.equal(result.host[name], result.device[name], `${name} host`);
            console.log(`PASS: hardware ${name} negotiated and exposed`);
        }
    }
} finally { await browser.close(); }
