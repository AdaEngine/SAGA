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
    const pixels = await page.evaluate(async () => {
        const realGPU = navigator.gpu;
        const adapter = await realGPU.requestAdapter();
        if (!adapter || adapter.info.isFallbackAdapter) throw new Error("No hardware WebGPU adapter");
        let device, target;
        const wrap = (object, overrides) => new Proxy(object, { get(value, key) {
            if (key in overrides) return overrides[key];
            const member = Reflect.get(value, key, value);
            return typeof member === "function" ? member.bind(value) : member;
        } });
        const wrappedAdapter = wrap(adapter, { requestDevice: async descriptor => {
            device = await adapter.requestDevice(descriptor);
            return wrap(device, { createTexture: descriptor => {
                const texture = device.createTexture(descriptor);
                if (descriptor.size[0] === 4 && descriptor.size[1] === 4 && descriptor.format === "rgba8unorm") target = texture;
                return texture;
            } });
        } });
        Object.defineProperty(navigator, "gpu", { configurable: true, value: wrap(realGPU, {
            requestAdapter: async () => wrappedAdapter,
        }) });
        const { openWebGPU } = await import("/webgpu.mjs");
        class Packet {
            bytes = [];
            word(n) { for (let i = 0; i < 4; i++) this.bytes.push((n >>> (8 * i)) & 255); return this; }
            wide(n) { return this.word(n).word(0); }
            id(n) { return this.wide(1).wide(n); }
            float(n) { const b = new ArrayBuffer(8); new DataView(b).setFloat64(0, n, true); this.bytes.push(...new Uint8Array(b)); return this; }
            string(s) { const b = new TextEncoder().encode(s); this.word(b.length); this.bytes.push(...b); return this; }
            data() { return Uint8Array.from(this.bytes); }
        }
        const host = await openWebGPU();
        // The vertex buffer is four bytes long; binding at offset four is a valid
        // zero-byte range, but cannot contain the float32 vertex attribute.
        const allocate = (id, size, usage) => host.handle(new Packet().word(1).id(id).wide(size).word(usage).word(0).data());
        await allocate(1, 4, GPUBufferUsage.VERTEX);
        await allocate(2, 8, GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST);
        await allocate(3, 20, GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST);
        await host.handle(new Packet().word(8).id(4).word(4).word(4).word(1)
            .word(GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC).word(1).word(1).word(1).word(0).data());
        const vertex = `struct Out { @builtin(position) position: vec4f, @location(0) value: f32 };
            @vertex fn vs(@builtin(vertex_index) i: u32, @location(0) value: f32) -> Out {
                var positions = array<vec2f, 3>(vec2f(-1,-1), vec2f(3,-1), vec2f(-1,3));
                var o: Out; o.position = vec4f(positions[i], 0, 1); o.value = value; return o;
            }`;
        const fragment = `@fragment fn fs(@location(0) value: f32) -> @location(0) vec4f { return vec4f(1.0 + value, 0, 0, 1); }`;
        const p = new Packet().word(10).id(5).word(0).word(1).word(0xffffffff).word(0)
            .word(1).word(0).word(1).word(15).word(0)
            .word(0).word(0).word(3).word(0).word(0)
            .string("vs").string(vertex).word(1).string("fs").string(fragment)
            .word(2).word(0).word(0).word(0) // empty slot 0
            .word(4).word(0).word(1).word(0).word(0).word(1); // attribute in slot 1
        for (let i = 0; i < 7; i++) p.word(0); // six resource lists and depth/stencil
        await host.handle(p.data());
        const upload = (id, words) => {
            const p = new Packet().word(6).word(1).word(1).id(id).wide(0).word(words.length);
            for (const word of words) p.word(word);
            return host.handle(p.data());
        };
        await upload(2, [0x00010000, 2]); // uint16 indices 0, 1, 2
        await upload(3, [3, 1, 0, 0, 0]);
        async function draw(indexed) {
            const p = new Packet().word(6).word(1).word(11).id(3).wide(0).word(indexed ? 1 : 0)
                .id(5).word(0);
            for (let i = 0; i < 4; i++) p.float(0); // blend constant
            // An unused high slot precedes the active sparse slot in binding order.
            p.word(0).word(0).word(2).word(device.limits.maxVertexBuffers - 1).id(1).wide(4)
                .word(1).id(1).wide(4);
            p.word(0).word(0).word(0).word(0); // resources, depth
            p.word(1).word(0).id(4).word(0).word(0).word(1);
            p.float(0).float(0).float(0).float(1).word(0).word(0); // clear, store, no resolve
            if (indexed) p.id(2).word(2).wide(0).word(0);
            await host.handle(p.data());
            const read = device.createBuffer({ size: 1024, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
            const encoder = device.createCommandEncoder();
            encoder.copyTextureToBuffer({ texture: target }, { buffer: read, bytesPerRow: 256 }, [4, 4]);
            device.queue.submit([encoder.finish()]);
            await read.mapAsync(GPUMapMode.READ);
            const pixel = Array.from(new Uint8Array(read.getMappedRange()).slice(0, 4));
            read.unmap(); read.destroy();
            return pixel;
        }
        const indexed = await draw(true);
        const nonindexed = await draw(false);
        host.close();
        return { indexed, nonindexed };
    });
    assert.deepEqual(pixels.indexed, [255, 0, 0, 255]);
    assert.deepEqual(pixels.nonindexed, [0, 0, 0, 255]);
    console.log("WebGPU indexed and non-indexed indirect vertex capacity readback passed");
} finally { await browser.close(); }
