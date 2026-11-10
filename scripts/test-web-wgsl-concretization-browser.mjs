import assert from "node:assert/strict";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { chromium } = require(process.env.SAGA_PLAYWRIGHT ?? "playwright-core");
const browser = await chromium.launch({ executablePath: process.env.SAGA_CHROMIUM,
    args: ["--enable-unsafe-webgpu", ...(process.platform === "darwin" ? ["--use-angle=metal"] : [])] });
try {
    const page = await browser.newPage();
    await page.route("http://localhost:8799/**", route => route.fulfill({ contentType: "text/html", body: "<!doctype html>" }));
    await page.goto("http://localhost:8799/");
    const results = await page.evaluate(async () => {
        const adapter = await navigator.gpu?.requestAdapter();
        if (!adapter) throw new Error("No WebGPU adapter");
        const device = await adapter.requestDevice();
        const sources = {
            u32Control: "@compute @workgroup_size(1) fn main() { let x: u32 = 2147483648u; }",
            f32Control: "@compute @workgroup_size(1) fn main() { let x: f32 = 4294967296.0; }",
            incompatibleControl: "@compute @workgroup_size(1) fn main() { let x: i32 = 1; let y: u32 = x; }",
            u32Inferred: "@compute @workgroup_size(1) fn main() { let x = 2147483648; let y: u32 = x; }",
            f32Inferred: "@compute @workgroup_size(1) fn main() { let x = 4294967296; let y: f32 = x; }",
        };
        const results = {};
        for (const [name, code] of Object.entries(sources)) {
            const info = await device.createShaderModule({ code }).getCompilationInfo();
            results[name] = info.messages.map(({ type, message, lineNum, linePos }) => ({ type, message, lineNum, linePos }));
        }
        device.destroy();
        return results;
    });
    console.log(JSON.stringify(results, null, 2));
    const errors = name => results[name].filter(message => message.type === "error");
    assert.equal(errors("u32Control").length, 0, "explicit u32 control must compile");
    assert.equal(errors("f32Control").length, 0, "explicit f32 control must compile");
    assert.ok(errors("incompatibleControl").length > 0, "concrete i32 cannot implicitly convert to u32");
    // The most preferred conversion is selected by rank before checking the
    // value; an AbstractInt outside i32 is a shader-creation error.
    assert.ok(errors("u32Inferred").length > 0, "inferred integer outside i32 must fail");
    assert.ok(errors("f32Inferred").length > 0, "inferred integer outside u32 must fail");
    console.log("WebGPU WGSL concretization diagnostics passed");
} finally { await browser.close(); }
