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
        const feature = gpu.wgslLanguageFeatures?.has("immediate_address_space") ?? false;
        const adapterLimit = adapter.limits.maxImmediateSize;
        let requested, deviceLimit;
        const wrap = (target, overrides) => new Proxy(target, { get(object, key) {
            if (key in overrides) return overrides[key];
            const value = Reflect.get(object, key, object);
            return typeof value === "function" ? value.bind(object) : value;
        } });
        const wrappedAdapter = wrap(adapter, { requestDevice: async descriptor => {
            requested = descriptor.requiredLimits.maxImmediateSize;
            const device = await adapter.requestDevice(descriptor);
            deviceLimit = device.limits.maxImmediateSize;
            return device;
        } });
        Object.defineProperty(navigator, "gpu", { configurable: true, value: wrap(gpu, {
            requestAdapter: async () => wrappedAdapter,
        }) });
        const { openWebGPU } = await import("/webgpu.mjs");
        const host = await openWebGPU();
        try {
            return { feature, adapterLimit: adapterLimit ?? null, requested: requested ?? null,
                deviceLimit: deviceLimit ?? null, hostLimit: host.maxImmediateSize };
        } finally { host.close(); }
    });
    console.log(`WebGPU immediate limit probe: ${JSON.stringify(result)}`);
    if (!result.feature || result.adapterLimit === null) {
        assert.equal(result.requested, null);
        if (!result.feature) assert.equal(result.hostLimit, 0);
        console.log("UNAVAILABLE: immediate_address_space or adapter maxImmediateSize is unavailable; no hardware limit assertion performed");
    } else {
        assert.equal(result.requested, result.adapterLimit, "host must request the hardware adapter limit");
        assert.equal(result.deviceLimit, result.adapterLimit, "device must expose the requested limit");
        assert.equal(result.hostLimit, result.adapterLimit, "host must expose the negotiated limit");
        console.log("PASS: browser host negotiated the hardware immediate limit");
    }
} finally { await browser.close(); }
