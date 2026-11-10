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
        let gate, realDevice, textures = 0, realTextures = 0, realViews = 0, realBuffers = 0, clears = 0, passes = 0;
        const wrappedAdapter = wrap(adapter, { requestDevice: async descriptor => {
            const fresh = await gpu.requestAdapter();
            if (!fresh || fresh.info.isFallbackAdapter) throw new Error("No hardware WebGPU adapter");
            realDevice = await fresh.requestDevice(descriptor);
            return wrap(realDevice, {
                popErrorScope: () => {
                    const pending = realDevice.popErrorScope();
                    return gate ? Promise.all([pending, gate]).then(([error]) => error) : pending;
                },
                createTexture: descriptor => {
                    textures++;
                    const texture = realDevice.createTexture(descriptor);
                    if (texture instanceof GPUTexture) realTextures++;
                    return wrap(texture, { createView: descriptor => {
                        const view = texture.createView(descriptor);
                        if (view instanceof GPUTextureView) realViews++;
                        return view;
                    } });
                },
                createBuffer: descriptor => {
                    const buffer = realDevice.createBuffer(descriptor);
                    if (buffer instanceof GPUBuffer) realBuffers++;
                    return buffer;
                },
                createCommandEncoder: descriptor => {
                    const encoder = realDevice.createCommandEncoder(descriptor);
                    return wrap(encoder, {
                        clearBuffer: (...args) => { clears++; return encoder.clearBuffer(...args); },
                        beginComputePass: (...args) => { passes++; return encoder.beginComputePass(...args); },
                    });
                },
            });
        } });
        Object.defineProperty(navigator, "gpu", { configurable: true, value: wrap(gpu, {
            requestAdapter: async () => wrappedAdapter,
        }) });
        const { openWebGPU } = await import("/webgpu.mjs");
        const host = await openWebGPU();
        class Packet {
            bytes = [];
            word(value) { for (let i = 0; i < 4; i++) this.bytes.push((value >>> (8 * i)) & 255); return this; }
            wide(value) { this.word(value); this.word(0); return this; }
            id(value) { return this.wide(1).wide(value); }
            string(value) { const bytes = new TextEncoder().encode(value); this.word(bytes.length); this.bytes.push(...bytes); return this; }
            data() { return Uint8Array.from(this.bytes); }
        }
        const allocate = id => new Packet().word(1).id(id).wide(4).word(GPUBufferUsage.COPY_DST).word(0).data();
        const releaseBuffer = id => new Packet().word(2).id(id).data();
        const texture = id => new Packet().word(8).id(id).word(1).word(1).word(1)
            .word(GPUTextureUsage.COPY_SRC).word(1).word(1).word(1).word(0).data();
        const submit = id => new Packet().word(6).word(1).word(16).id(id).wide(0).word(1).wide(4).data();
        const errors = [], staleClears = [];
        for (const [index, replace] of [false, true].entries()) {
            const id = index + 1;
            await host.handle(allocate(id));
            let release;
            gate = new Promise(resolve => { release = resolve; });
            const before = textures;
            const blocker = host.handle(texture(index + 10));
            while (textures === before) await Promise.resolve();
            const stale = host.handle(submit(id)).then(() => null,
                error => ({ message: error.message, status: error.status }));
            await host.handle(releaseBuffer(id));
            // Allocation also uses the scope queue, so replacement is requested
            // before the stale submission runs but published after it rejects.
            const replacement = replace ? host.handle(allocate(id)) : undefined;
            release();
            errors.push(await stale);
            await blocker;
            if (replacement) await replacement;
            staleClears.push(clears);
            gate = undefined;
            if (replace) await host.handle(submit(id));
        }
        const sampledTexture = id => new Packet().word(8).id(id).word(1).word(1).word(1)
            .word(GPUTextureUsage.TEXTURE_BINDING).word(1).word(1).word(1).word(0).data();
        const view = (id, parent) => new Packet().word(15).id(id).id(parent)
            .word(0).word(0).word(0).word(0).word(0).word(0).data();
        const shader = "@group(0) @binding(0) var t: texture_2d<f32>; @compute @workgroup_size(1) fn main() { let size = textureDimensions(t); }";
        await host.handle(new Packet().word(3).id(30).string("main").string(shader).word(0)
            .word(1).word(0).word(0).wide(0).word(3).word(0).word(0).word(0).data());
        const dispatch = id => new Packet().word(6).word(1).word(2).id(30).word(1).word(1).word(1)
            .word(0).word(1).word(0).word(0).id(id).word(0).word(0).data();
        const viewErrors = [], stalePasses = [];
        for (const [index, releaseParent] of [true, false].entries()) {
            const parent = 40 + index, viewId = 50 + index;
            await host.handle(sampledTexture(parent));
            await host.handle(view(viewId, parent));
            let release;
            gate = new Promise(resolve => { release = resolve; });
            const before = textures;
            const blocker = host.handle(texture(60 + index));
            while (textures === before) await Promise.resolve();
            const stale = host.handle(dispatch(viewId)).then(() => null,
                error => ({ message: error.message, status: error.status }));
            await host.handle(new Packet().word(releaseParent ? 9 : 16).id(releaseParent ? parent : viewId).data());
            release();
            viewErrors.push(await stale);
            await blocker;
            stalePasses.push(passes);
            gate = undefined;
        }
        await host.handle(sampledTexture(42));
        await host.handle(view(52, 42));
        await host.handle(dispatch(52));
        const metrics = host.metrics();
        host.close();
        return { errors, staleClears, clears, realBuffers, viewErrors, stalePasses, passes, realTextures, realViews, textures, metrics };
    });
    for (const error of result.errors) {
        assert.match(error?.message ?? "", /resource.*released/i);
        assert.equal(error.status, 1);
    }
    assert.deepEqual(result.staleClears, [0, 0]);
    assert.equal(result.clears, 1, "replacement buffer must accept a new submission");
    assert.equal(result.realBuffers, 3);
    for (const error of result.viewErrors) {
        assert.match(error?.message ?? "", /resource.*released/i);
        assert.equal(error.status, 1);
    }
    assert.deepEqual(result.stalePasses, [0, 0]);
    assert.equal(result.passes, 1, "a live texture view must accept a new dispatch");
    assert.equal(result.realViews, 3);
    assert.equal(result.realTextures, result.textures);
    console.log("WebGPU queued submission release, replacement, and texture views passed on hardware adapter");
} finally { await browser.close(); }
