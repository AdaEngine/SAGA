import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";

const wasmPath = process.argv[2];
if (!wasmPath) throw new Error("Pass the built SAGAWebDiagnostic.wasm path");
const require = createRequire(import.meta.url);
const { chromium } = require(process.env.SAGA_PLAYWRIGHT ?? "playwright-core");
const browser = await chromium.launch({
    executablePath: process.env.SAGA_CHROMIUM,
    args: ["--enable-unsafe-webgpu", ...(process.platform === "darwin" ? ["--use-angle=metal"] : [])],
});
try {
    const page = await browser.newPage();
    const root = "http://localhost:8799/";
    const mime = new Map([
        ["/", "text/html"],
        ["/reactor.mjs", "text/javascript"],
        ["/webgpu.mjs", "text/javascript"],
        ["/wasi.mjs", "text/javascript"],
        ["/SAGAWebDiagnostic.wasm", "application/wasm"],
    ]);
    await page.route(`${root}**`, async route => {
        const path = new URL(route.request().url()).pathname;
        const contentType = mime.get(path);
        if (!contentType) return route.fulfill({ status: 404 });
        const file = path.endsWith(".wasm") ? wasmPath : join(dirname(wasmPath), path === "/" ? "index.html" : path.slice(1));
        await route.fulfill({ contentType, body: await readFile(file) });
    });
    await page.goto(root);
    await page.waitForFunction(() => typeof globalThis.sagaDiagnostic?.then === "function");
    const result = await page.evaluate(() => globalThis.sagaDiagnostic);
    assert.equal(result.fallback, false);
    assert.equal(result.validation, "swift");
    assert.deepEqual(result.readback, [[0, 0, 0, 0], [8, 12, 16, 7]]);
    assert.equal(result.renderPixels, 56);
    assert.equal(result.vertexStreams, 2);
    assert.equal(result.renderProbe.length, 3);
    for (const [index, expected] of [242, 9, 4].entries()) {
        assert.ok(Math.abs(result.renderProbe[index] - expected) <= 1);
    }
    assert.equal(result.integerWords, 20);
    assert.equal(result.submissions, 3);
    assert.equal(result.remainingResources, 0);
    await page.route(`${root}SAGAWebDiagnostic.wasm`, route => route.fulfill({ status: 404 }));
    await page.reload();
    await page.waitForFunction(() => typeof globalThis.sagaDiagnostic?.then === "function");
    const withoutWasm = await page.evaluate(() => globalThis.sagaDiagnostic.then(
        () => "unexpected success", error => error.message,
    ));
    assert.match(withoutWasm, /Swift\/Wasm download failed/);
    await page.unroute(`${root}SAGAWebDiagnostic.wasm`);
    await page.addInitScript(() => Object.defineProperty(navigator, "gpu", { value: undefined }));
    await page.reload();
    await page.waitForFunction(() => typeof globalThis.sagaDiagnostic?.then === "function");
    const withoutGPU = await page.evaluate(() => globalThis.sagaDiagnostic.then(
        () => "unexpected success", error => error.message,
    ));
    assert.match(withoutGPU, /No hardware WebGPU adapter available/);
    console.log(JSON.stringify(result));
} finally {
    await browser.close();
}
