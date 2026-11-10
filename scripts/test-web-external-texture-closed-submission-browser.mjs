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
    const results = await page.evaluate(async () => {
        const gpu = navigator.gpu, adapter = await gpu.requestAdapter();
        if (!adapter || adapter.info.isFallbackAdapter) throw new Error("No hardware WebGPU adapter");
        const wrap = (target, overrides) => new Proxy(target, { get(object, key) {
            if (key in overrides) return overrides[key];
            const value = Reflect.get(object, key, object);
            return typeof value === "function" ? value.bind(object) : value;
        } });
        let gate, submissions = 0, creations = 0;
        Object.defineProperty(navigator, "gpu", { configurable: true, value: wrap(gpu, {
            requestAdapter: async () => wrap(adapter, { requestDevice: async descriptor => {
                const fresh = await gpu.requestAdapter();
                const device = await fresh.requestDevice(descriptor);
                return wrap(device, {
                    queue: wrap(device.queue, { submit: commands => { submissions++; return device.queue.submit(commands); } }),
                    createTexture: descriptor => { creations++; return device.createTexture(descriptor); },
                    popErrorScope: () => {
                        const pending = device.popErrorScope();
                        return gate ? Promise.all([pending, gate]).then(([error]) => error) : pending;
                    },
                });
            } }),
        }) });
        const { openWebGPU } = await import("/webgpu.mjs");
        class Packet {
            bytes = [];
            word(value) { for (let i = 0; i < 4; i++) this.bytes.push((value >>> (8 * i)) & 255); return this; }
            wide(value) { return this.word(value).word(0); }
            id(value) { return this.wide(1).wide(value); }
            string(value) { const bytes = new TextEncoder().encode(value); this.word(bytes.length); this.bytes.push(...bytes); return this; }
            data() { return Uint8Array.from(this.bytes); }
        }
        const output = [];
        for (const close of [true, false]) {
            gate = undefined;
            const host = await openWebGPU();
            const frame = new VideoFrame(new Uint8Array([255, 0, 0, 255]),
                { format: "RGBA", codedWidth: 1, codedHeight: 1, timestamp: 0 });
            const token = host.registerVideoFrame(frame);
            await host.handle(new Packet().word(26).id(2).wide(Number(token)).word(0).data());
            const shader = "@group(0) @binding(0) var image: texture_external; @compute @workgroup_size(1) fn main() { let x = textureDimensions(image); }";
            await host.handle(new Packet().word(3).id(4).string("main").string(shader).word(0)
                .word(1).word(0).word(0).wide(0).word(6).data());
            let release;
            gate = new Promise(resolve => { release = () => resolve(null); });
            const beforeCreation = creations;
            const blocker = host.handle(new Packet().word(8).id(3).word(1).word(1).word(1)
                .word(GPUTextureUsage.TEXTURE_BINDING).word(1).word(1).word(1).word(0).data());
            while (creations === beforeCreation) await Promise.resolve();
            const before = submissions;
            const pending = host.handle(new Packet().word(6).word(1).word(2).id(4).word(1).word(1).word(1)
                .word(0).word(1).word(0).word(0).id(2).word(0).word(0).data());
            if (close) frame.close();
            release();
            let error;
            try { await pending; } catch (caught) { error = { message: caught.message, status: caught.status }; }
            await blocker;
            output.push({ error, submissions: submissions - before });
            frame.close(); host.close();
        }
        return output;
    });
    assert.match(results[0].error?.message ?? "", /VideoFrame/);
    assert.equal(results[0].error.status, 1);
    assert.equal(results[0].submissions, 0);
    assert.equal(results[1].error, undefined);
    assert.equal(results[1].submissions, 1);
    console.log("WebGPU closed and live VideoFrame submissions passed on hardware adapter");
} finally { await browser.close(); }
