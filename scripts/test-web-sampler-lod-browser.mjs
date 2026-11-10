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
    const values = await page.evaluate(async () => {
        const realGPU = navigator.gpu;
        const adapter = await realGPU.requestAdapter();
        if (!adapter || adapter.info.isFallbackAdapter) throw new Error("No hardware WebGPU adapter");
        let device, sampler;
        const wrap = (target, overrides) => new Proxy(target, { get(object, key) {
            if (key in overrides) return overrides[key];
            const value = Reflect.get(object, key, object);
            return typeof value === "function" ? value.bind(object) : value;
        } });
        const wrappedAdapter = wrap(adapter, { requestDevice: async descriptor => {
            device = await adapter.requestDevice(descriptor);
            return wrap(device, { createSampler: descriptor => (sampler = device.createSampler(descriptor)) });
        } });
        Object.defineProperty(navigator, "gpu", { configurable: true, value: wrap(realGPU, {
            requestAdapter: async () => wrappedAdapter,
        }) });
        const { openWebGPU } = await import("/webgpu.mjs");
        const host = await openWebGPU();
        const packet = new Uint8Array(68), view = new DataView(packet.buffer);
        view.setUint32(0, 13, true); view.setBigUint64(4, 1n, true); view.setBigUint64(12, 1n, true);
        [1, 1, 1].forEach((value, index) => view.setUint32(32 + index * 4, value, true));
        view.setFloat64(44, 1, true); view.setFloat64(52, 1, true);
        view.setUint32(64, 32, true);
        await host.handle(packet);
        const texture = device.createTexture({ size: [4, 4], mipLevelCount: 3, format: "rgba8unorm",
            usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING });
        device.queue.writeTexture({ texture }, new Uint8Array(Array(16).fill([255, 0, 0, 255]).flat()),
            { bytesPerRow: 16 }, [4, 4]);
        device.queue.writeTexture({ texture, mipLevel: 1 }, new Uint8Array(Array(4).fill([0, 255, 0, 255]).flat()),
            { bytesPerRow: 8 }, [2, 2]);
        device.queue.writeTexture({ texture, mipLevel: 2 }, new Uint8Array([0, 0, 255, 255]), {}, [1, 1]);
        const output = device.createBuffer({ size: 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        const read = device.createBuffer({ size: 32, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        const module = device.createShaderModule({ code: `
            @group(0) @binding(0) var image: texture_2d<f32>;
            @group(0) @binding(1) var state: sampler;
            @group(0) @binding(2) var<storage, read_write> result: array<vec4f, 2>;
            @compute @workgroup_size(1) fn main() {
                result[0] = textureSampleLevel(image, state, vec2f(0.5), 0.0);
                result[1] = textureSampleLevel(image, state, vec2f(0.5), 8.0);
            }` });
        const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "main" } });
        const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
            { binding: 0, resource: texture.createView() }, { binding: 1, resource: sampler },
            { binding: 2, resource: { buffer: output } },
        ] });
        const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
        pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(1); pass.end();
        encoder.copyBufferToBuffer(output, 0, read, 0, 32); device.queue.submit([encoder.finish()]);
        await read.mapAsync(GPUMapMode.READ);
        const result = Array.from(new Float32Array(read.getMappedRange()).slice());
        read.unmap(); read.destroy(); output.destroy(); texture.destroy(); host.close();
        return result;
    });
    assert.deepEqual(values, [0, 1, 0, 1, 0, 1, 0, 1]);
    console.log("WebGPU anisotropic sampler LOD clamp mip readback passed");
} finally { await browser.close(); }
