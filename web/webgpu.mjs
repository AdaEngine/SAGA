const limit = 16 * 1024 * 1024;
const text = new TextDecoder("utf-8", { fatal: true });

class Reader {
    constructor(bytes) {
        this.bytes = bytes;
        this.view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
        this.offset = 0;
    }
    word() {
        const value = this.view.getUint32(this.offset, true);
        this.offset += 4;
        return value;
    }
    wide() {
        const value = this.view.getBigUint64(this.offset, true);
        this.offset += 8;
        return value;
    }
    float() {
        const value = this.view.getFloat64(this.offset, true);
        this.offset += 8;
        return value;
    }
    id() { return `${this.wide()}:${this.wide()}`; }
    size() {
        const value = Number(this.wide());
        if (!Number.isSafeInteger(value)) throw new Error("Invalid WebGPU size");
        return value;
    }
    take(count) {
        const end = this.offset + count;
        if (end > this.bytes.length) throw new Error("Truncated WebGPU packet");
        const value = this.bytes.subarray(this.offset, end);
        this.offset = end;
        return value;
    }
    string() { return text.decode(this.take(this.word())); }
    list(read) {
        const count = this.word();
        if (count > (this.bytes.length - this.offset) / 4) throw new Error("Invalid WebGPU list length");
        return Array.from({ length: count }, read);
    }
    done() {
        if (this.offset !== this.bytes.length) throw new Error("Trailing WebGPU packet data");
    }
}

function textureFormat(value) {
    switch (value) {
    case 1: return "rgba8unorm";
    case 2: return "bgra8unorm";
    case 3: return "r32uint";
    case 4: return "depth32float";
    case 5: return "rgba8uint";
    case 6: return "rgba8sint";
    case 7: return "depth24plus-stencil8";
    case 8: return "rgba8snorm";
    case 9: return "rgba16uint";
    case 10: return "rgba16sint";
    case 11: return "rgba16float";
    case 12: return "r32sint";
    case 13: return "r32float";
    case 14: return "rg32uint";
    case 15: return "rg32sint";
    case 16: return "rg32float";
    case 17: return "rgba32uint";
    case 18: return "rgba32sint";
    case 19: return "rgba32float";
    case 20: return "bc1-rgba-unorm";
    case 21: return "bc1-rgba-unorm-srgb";
    case 22: return "bc2-rgba-unorm";
    case 23: return "bc2-rgba-unorm-srgb";
    case 24: return "bc3-rgba-unorm";
    case 25: return "bc3-rgba-unorm-srgb";
    case 26: return "bc4-r-unorm";
    case 27: return "bc4-r-snorm";
    case 28: return "bc5-rg-unorm";
    case 29: return "bc5-rg-snorm";
    case 30: return "bc6h-rgb-ufloat";
    case 31: return "bc6h-rgb-float";
    case 32: return "bc7-rgba-unorm";
    case 33: return "bc7-rgba-unorm-srgb";
    case 34: return "etc2-rgb8unorm";
    case 35: return "etc2-rgb8unorm-srgb";
    case 36: return "etc2-rgb8a1unorm";
    case 37: return "etc2-rgb8a1unorm-srgb";
    case 38: return "etc2-rgba8unorm";
    case 39: return "etc2-rgba8unorm-srgb";
    case 40: return "eac-r11unorm";
    case 41: return "eac-r11snorm";
    case 42: return "eac-rg11unorm";
    case 43: return "eac-rg11snorm";
    case 44: return "astc-4x4-unorm";
    case 45: return "astc-4x4-unorm-srgb";
    case 46: return "astc-5x4-unorm";
    case 47: return "astc-5x4-unorm-srgb";
    case 48: return "astc-5x5-unorm";
    case 49: return "astc-5x5-unorm-srgb";
    case 50: return "astc-6x5-unorm";
    case 51: return "astc-6x5-unorm-srgb";
    case 52: return "astc-6x6-unorm";
    case 53: return "astc-6x6-unorm-srgb";
    case 54: return "astc-8x5-unorm";
    case 55: return "astc-8x5-unorm-srgb";
    case 56: return "astc-8x6-unorm";
    case 57: return "astc-8x6-unorm-srgb";
    case 58: return "astc-8x8-unorm";
    case 59: return "astc-8x8-unorm-srgb";
    case 60: return "astc-10x5-unorm";
    case 61: return "astc-10x5-unorm-srgb";
    case 62: return "astc-10x6-unorm";
    case 63: return "astc-10x6-unorm-srgb";
    case 64: return "astc-10x8-unorm";
    case 65: return "astc-10x8-unorm-srgb";
    case 66: return "astc-10x10-unorm";
    case 67: return "astc-10x10-unorm-srgb";
    case 68: return "astc-12x10-unorm";
    case 69: return "astc-12x10-unorm-srgb";
    case 70: return "astc-12x12-unorm";
    case 71: return "astc-12x12-unorm-srgb";
    case 86: return "r16float";
    case 87: return "rg16float";
    case 88: return "r16uint";
    case 89: return "r16sint";
    case 90: return "rg16uint";
    case 91: return "rg16sint";
    case 92: return "r16unorm";
    case 93: return "rg16unorm";
    case 94: return "rgba16unorm";
    case 95: return "r16snorm";
    case 96: return "rg16snorm";
    case 97: return "rgba16snorm";
    case 98: return "rgb10a2unorm";
    case 99: return "rgb10a2uint";
    case 100: return "rg11b10ufloat";
    case 101: return "rgba8unorm-srgb";
    case 102: return "bgra8unorm-srgb";
    case 103: return "rgb9e5ufloat";
    case 104: return "r8unorm";
    case 105: return "r8snorm";
    case 106: return "r8uint";
    case 107: return "r8sint";
    case 108: return "rg8unorm";
    case 109: return "rg8snorm";
    case 110: return "rg8uint";
    case 111: return "rg8sint";
    case 112: return "depth16unorm";
    case 113: return "depth24plus";
    case 114: return "depth32float-stencil8";
    case 115: return "stencil8";
    default: throw new Error("Unknown WebGPU texture format");
    }
}

function textureAspect(value) {
    const aspect = ["all", "depth-only", "stencil-only"][value];
    if (!aspect) throw new Error("Invalid WebGPU texture aspect");
    return aspect;
}

function bufferCopyAspect(texture, aspect) {
    if (aspect === "all") {
        if (texture.format === "depth16unorm" || texture.format === "depth32float") return "depth-only";
        if (texture.format === "stencil8") return "stencil-only";
    }
    return aspect;
}

function compareFunction(value) {
    const compare = ["never", "less", "equal", "less-equal", "greater", "not-equal", "greater-equal", "always"][value];
    if (!compare) throw new Error("Invalid WebGPU comparison function");
    return compare;
}

// Stable Swift VertexFormat IDs and their WebGPU names and byte sizes.
const vertexFormats = [
    null,
    ["float32", 4], ["float32x2", 8], ["float32x3", 12], ["float32x4", 16],
    ["uint32", 4], ["uint32x2", 8], ["uint32x3", 12], ["uint32x4", 16],
    ["sint32", 4], ["sint32x2", 8], ["sint32x3", 12], ["sint32x4", 16],
    ["float16", 2], ["float16x2", 4], ["float16x4", 8],
    ["uint8", 1], ["uint8x2", 2], ["uint8x4", 4],
    ["sint8", 1], ["sint8x2", 2], ["sint8x4", 4],
    ["unorm8", 1], ["unorm8x2", 2], ["unorm8x4", 4],
    ["snorm8", 1], ["snorm8x2", 2], ["snorm8x4", 4],
    ["uint16", 2], ["uint16x2", 4], ["uint16x4", 8],
    ["sint16", 2], ["sint16x2", 4], ["sint16x4", 8],
    ["unorm16", 2], ["unorm16x2", 4], ["unorm16x4", 8],
    ["snorm16", 2], ["snorm16x2", 4], ["snorm16x4", 8],
    ["unorm10-10-10-2", 4], ["unorm8x4-bgra", 4], ["snorm10-10-10-2", 4],
];

async function probeSnorm1010102VertexFormat(device) {
    const module = device.createShaderModule({ code: `
        @vertex fn vs(@location(0) value: vec4f) -> @builtin(position) vec4f { return value; }
        @fragment fn fs() -> @location(0) vec4f { return vec4f(0.0); }
    ` });
    try {
        await device.createRenderPipelineAsync({
            layout: "auto",
            vertex: { module, entryPoint: "vs", buffers: [{ arrayStride: 4,
                attributes: [{ format: "snorm10-10-10-2", offset: 0, shaderLocation: 0 }] }] },
            fragment: { module, entryPoint: "fs", targets: [{ format: "rgba8unorm" }] },
        });
        return true;
    } catch (error) {
        if (error instanceof TypeError || error.reason === "validation") return false;
        throw error;
    }
}

export async function openWebGPU(canvas) {
    const context = canvas?.getContext("webgpu");
    if (canvas && !context) throw new Error("Canvas cannot supply a WebGPU surface");
    const adapter = await navigator.gpu?.requestAdapter();
    if (!adapter || adapter.info.isFallbackAdapter) throw new Error("No hardware WebGPU adapter available");
    const requiredFeatures = ["shader-f16", "indirect-first-instance", "clip-distances", "primitive-index", "depth-clip-control",
        "timestamp-query", "subgroups", "subgroup-size-control", "dual-source-blending", "texture-compression-bc", "texture-compression-etc2", "texture-compression-astc",
        "texture-compression-bc-sliced-3d", "texture-compression-astc-sliced-3d", "float32-filterable", "float32-blendable", "texture-formats-tier1", "texture-formats-tier2", "rg11b10ufloat-renderable", "depth32float-stencil8", "bgra8unorm-storage"]
        .filter(feature => adapter.features.has(feature));
    const requiredLimits = {
        maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
        maxComputeInvocationsPerWorkgroup: adapter.limits.maxComputeInvocationsPerWorkgroup,
        maxComputeWorkgroupSizeX: adapter.limits.maxComputeWorkgroupSizeX,
        maxComputeWorkgroupSizeY: adapter.limits.maxComputeWorkgroupSizeY,
        maxComputeWorkgroupSizeZ: adapter.limits.maxComputeWorkgroupSizeZ,
        maxComputeWorkgroupsPerDimension: adapter.limits.maxComputeWorkgroupsPerDimension,
        maxVertexAttributes: adapter.limits.maxVertexAttributes,
        maxVertexBuffers: adapter.limits.maxVertexBuffers,
        maxVertexBufferArrayStride: adapter.limits.maxVertexBufferArrayStride,
        maxInterStageShaderVariables: adapter.limits.maxInterStageShaderVariables,
        maxColorAttachments: adapter.limits.maxColorAttachments,
        maxColorAttachmentBytesPerSample: adapter.limits.maxColorAttachmentBytesPerSample,
    };
    if (adapter.limits.maxBindGroupsPlusVertexBuffers !== undefined) {
        requiredLimits.maxBindGroupsPlusVertexBuffers = adapter.limits.maxBindGroupsPlusVertexBuffers;
    }
    const device = await adapter.requestDevice({ requiredFeatures, requiredLimits });
    let supportsVertexFormatSnorm1010102;
    try {
        supportsVertexFormatSnorm1010102 = await probeSnorm1010102VertexFormat(device);
    } catch (error) {
        device.destroy();
        throw error;
    }
    const indirectFirstInstance = device.features.has("indirect-first-instance");
    const buffers = new Map();
    const mappings = new Map();
    const stagedSubmissions = new Map();
    const querySets = new Map();
    const textures = new Map();
    const externalTextures = new Map();
    const videoFrames = new Map();
    let nextVideoFrame = 0n;
    const textureViews = new Map();
    const samplers = new Map();
    const pipelines = new Map();
    const renderPipelines = new Map();
    let indirectDrawValidator;
    let timestampMarkerPipeline;
    let submissions = 0;
    let presentedFrames = 0;
    let configuredFormat;
    let ownsConfiguration = false;
    let closed = false;
    let lost;
    device.lost.then(info => { lost = info.message || info.reason; });

    function lookup(map, id) {
        const resource = map.get(id);
        if (!resource) throw new Error(`Unknown browser resource ${id}`);
        return resource;
    }

    function registerVideoFrame(frame) {
        if (closed || typeof VideoFrame !== "function" || !(frame instanceof VideoFrame) ||
            frame.codedWidth === 0 || nextVideoFrame === 0xffffffffffffffffn) {
            throw new TypeError("Expected a live VideoFrame and an open WebGPU host");
        }
        const token = ++nextVideoFrame;
        videoFrames.set(token, frame);
        return token;
    }

    function mappedRange(id, offset, length, writable) {
        const mapping = lookup(mappings, id);
        const start = offset - mapping.offset;
        if (start < 0 || start > mapping.bytes.length || length > mapping.bytes.length - start ||
            (writable && !mapping.writable)) {
            throw Object.assign(new Error("Invalid browser mapped buffer range or access"), { status: 1 });
        }
        return mapping.bytes.subarray(start, start + length);
    }

    async function withErrors(operation) {
        for (const filter of ["internal", "out-of-memory", "validation"]) device.pushErrorScope(filter);
        try {
            const result = await operation();
            return result;
        } finally {
            const errors = await Promise.all([device.popErrorScope(), device.popErrorScope(), device.popErrorScope()]);
            if (lost) throw Object.assign(new Error(lost), { status: 3 });
            const error = errors.find(Boolean);
            if (error) {
                const status = error instanceof GPUValidationError ? 1 : error instanceof GPUOutOfMemoryError ? 2 : 5;
                throw Object.assign(new Error(error.message), { status });
            }
        }
    }

    let multisampledFloat32Mask = 0;
    try {
        for (const [index, format] of ["r32float", "rg32float", "rgba32float"].entries()) {
            try {
                const texture = await withErrors(() => device.createTexture({
                    size: [1, 1], format, sampleCount: 4,
                    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
                }));
                texture.destroy();
                multisampledFloat32Mask |= 1 << index;
            } catch (error) {
                if (error.status !== 1) throw error;
            }
        }
    } catch (error) {
        device.destroy();
        throw error;
    }

    function bindingGroups(packet, readEntry, groups = new Map()) {
        for (const [group, entry] of packet.list(() => [packet.word(), readEntry()])) {
            if (group >= device.limits.maxBindGroups) throw new Error("Invalid WebGPU bind group index");
            if (!groups.has(group)) groups.set(group, []);
            groups.get(group).push(entry);
        }
        return groups;
    }

    function pipelineLayout(groups, immediateSize = 0) {
        const length = Math.max(-1, ...groups.keys()) + 1;
        const bindGroupLayouts = Array.from({ length }, (_, group) =>
            device.createBindGroupLayout({ entries: groups.get(group) ?? [] }));
        return device.createPipelineLayout(immediateSize ? { bindGroupLayouts, immediateSize } : { bindGroupLayouts });
    }

    function decodeBufferBindings(packet) {
        return bindingGroups(packet, () => {
            const binding = packet.word();
            const buffer = lookup(buffers, packet.id());
            const offset = packet.size();
            const size = packet.word() ? packet.size() : undefined;
            return { binding, resource: { buffer, offset, size } };
        });
    }

    function decodeResourceBindings(packet) {
        const groups = decodeBufferBindings(packet);
        bindingGroups(packet, () => {
            const binding = packet.word();
            const id = packet.id();
            const external = externalTextures.get(id);
            if (external) return { binding, resource: external };
            const view = textureViews.get(id);
            return view ? { binding, resource: view } : { binding, texture: lookup(textures, id) };
        }, groups);
        return bindingGroups(packet, () => {
            const binding = packet.word();
            return { binding, resource: lookup(samplers, packet.id()) };
        }, groups);
    }

    function sampledTextureLayout(binding, visibility, type, multisampled, dimension) {
        const sampleType = [multisampled ? "unfilterable-float" : "float", "depth", "sint", "uint", "unfilterable-float"][type];
        if (!sampleType) throw new Error("Invalid WebGPU texture sample type");
        const viewDimension = ["2d", "2d-array", "cube", "cube-array", "3d", "1d"][dimension];
        if (!viewDimension) throw new Error("Invalid sampled texture view dimension");
        return { binding, visibility, texture: { sampleType, viewDimension, multisampled } };
    }

    function decodeStorageTextureLayout(packet, binding, visibility) {
        const format = textureFormat(packet.word());
        const access = [undefined, "read-only", "write-only", "read-write"][packet.word()];
        if (!access) throw new Error("Invalid WebGPU storage texture access");
        const viewDimension = ["2d", "2d-array", undefined, undefined, "3d", "1d"][packet.word()];
        if (!viewDimension) throw new Error("Invalid storage texture view dimension");
        return { binding, visibility, storageTexture: { access, format, viewDimension } };
    }

    function samplerLayout(binding, visibility, samplerType) {
        const type = ["filtering", "comparison", "non-filtering"][samplerType];
        if (!type) throw new Error(`Invalid sampler binding type ${samplerType}`);
        return { binding, visibility, sampler: { type } };
    }

    function bindResources(pass, pipeline, bindings) {
        for (const [group, resources] of bindings) {
            const layout = pipeline.getBindGroupLayout(group);
            const entries = resources.map(entry => entry.texture
                ? { binding: entry.binding, resource: entry.texture.createView() }
                : entry);
            pass.setBindGroup(group, device.createBindGroup({ layout, entries }));
        }
    }

    async function allocatePipeline(packet) {
        const id = packet.id();
        const entryPoint = packet.string();
        const code = packet.string();
        const immediateSize = packet.word();
        const groups = bindingGroups(packet, () => {
            const binding = packet.word();
            const minBindingSize = packet.size();
            const code = packet.word();
            if (code === 3) return sampledTextureLayout(binding, GPUShaderStage.COMPUTE, packet.word(), packet.word() !== 0, packet.word());
            if (code === 4) return samplerLayout(binding, GPUShaderStage.COMPUTE, packet.word());
            if (code === 5) return decodeStorageTextureLayout(packet, binding, GPUShaderStage.COMPUTE);
            if (code === 6) return { binding, visibility: GPUShaderStage.COMPUTE, externalTexture: {} };
            const type = ["storage", "read-only-storage", "uniform"][code];
            if (!type) throw new Error("Invalid WebGPU compute binding type");
            return { binding, visibility: GPUShaderStage.COMPUTE, buffer: { type, minBindingSize } };
        });
        packet.done();
        if (pipelines.has(id)) throw new Error("Duplicate WebGPU pipeline");
        const pipeline = await withErrors(async () => {
            return await device.createComputePipelineAsync({
                layout: pipelineLayout(groups, immediateSize),
                compute: { module: device.createShaderModule({ code }), entryPoint },
            });
        });
        pipelines.set(id, pipeline);
    }

    function blendComponent(packet) {
        const operations = ["add", "subtract", "reverse-subtract", "min", "max"];
        const factors = ["zero", "one", "src", "one-minus-src", "dst", "one-minus-dst",
            "src-alpha", "one-minus-src-alpha", "dst-alpha", "one-minus-dst-alpha",
            "src-alpha-saturated", "constant", "one-minus-constant",
            "src1", "one-minus-src1", "src1-alpha", "one-minus-src1-alpha"];
        const operation = operations[packet.word()];
        const srcFactor = factors[packet.word()];
        const dstFactor = factors[packet.word()];
        if (!operation || !srcFactor || !dstFactor) throw new Error("Invalid WebGPU blend component");
        return { operation, srcFactor, dstFactor };
    }

    function colorSlots(packet, read) {
        const colors = [];
        for (const [location, value] of packet.list(() => [packet.word(), read()])) {
            if (location >= device.limits.maxColorAttachments || colors[location]) throw new Error("Invalid WebGPU color attachment location");
            while (colors.length <= location) colors.push(null);
            colors[location] = value;
        }
        return colors;
    }

    function stencilFace(packet) {
        const ops = ["keep", "zero", "replace", "invert", "increment-clamp", "decrement-clamp", "increment-wrap", "decrement-wrap"];
        const compare = compareFunction(packet.word());
        const failOp = ops[packet.word()];
        const depthFailOp = ops[packet.word()];
        const passOp = ops[packet.word()];
        if (!failOp || !depthFailOp || !passOp) throw new Error("Invalid WebGPU stencil operation");
        return { compare, failOp, depthFailOp, passOp };
    }

    async function allocateRenderPipeline(packet) {
        const id = packet.id();
        const immediateSize = packet.word();
        const sampleCount = packet.word();
        const mask = packet.word();
        const alphaToCoverageEnabled = packet.word() !== 0;
        const targets = colorSlots(packet, () => {
            const format = textureFormat(packet.word());
            const writeMask = packet.word();
            const blend = packet.word() ? { color: blendComponent(packet), alpha: blendComponent(packet) } : undefined;
            return { format, writeMask, blend };
        });
        const frontFace = ["ccw", "cw"][packet.word()];
        const cullMode = ["none", "front", "back"][packet.word()];
        const topology = ["point-list", "line-list", "line-strip", "triangle-list", "triangle-strip"][packet.word()];
        const indexFormat = packet.word();
        const unclippedDepth = packet.word() !== 0;
        if (!frontFace || !cullMode || !topology || ![0, 2, 4].includes(indexFormat)) {
            throw new Error("Invalid WebGPU primitive state");
        }
        const stripIndexFormat = indexFormat === 0 ? undefined : indexFormat === 2 ? "uint16" : "uint32";
        const vertex = { entryPoint: packet.string(), code: packet.string() };
        let fragment = packet.word() ? { entryPoint: packet.string(), code: packet.string() } : undefined;
        const activeSamples = (1 << sampleCount) - 1;
        if (!fragment && (mask & activeSamples) !== activeSamples) {
            // Some WebGPU implementations ignore the pipeline mask when the fragment state is absent.
            fragment = { entryPoint: "saga_coverage", code: "@fragment fn saga_coverage() {}" };
        }
        const fetchLayouts = [];
        const vertexBuffers = packet.list(() => {
            const arrayStride = packet.word();
            const step = packet.word();
            if (step !== 0 && step !== 1) throw new Error("Unknown WebGPU vertex step mode");
            const stepMode = step === 0 ? "vertex" : "instance";
            let attributeEnd = 0;
            const attributes = packet.list(() => {
                const shaderLocation = packet.word();
                const offset = packet.word();
                const entry = vertexFormats[packet.word()];
                if (!entry) throw new Error("Unknown WebGPU vertex format");
                const [format, byteSize] = entry;
                attributeEnd = Math.max(attributeEnd, offset + byteSize);
                return { shaderLocation, offset, format };
            });
            fetchLayouts.push({ arrayStride, stepMode, attributeEnd });
            return { arrayStride, stepMode, attributes };
        });
        const groups = bindingGroups(packet, () => {
            const binding = packet.word();
            const minBindingSize = packet.size();
            const visibility = packet.word();
            return { binding, visibility, buffer: { type: "uniform", minBindingSize } };
        });
        bindingGroups(packet, () => {
            const binding = packet.word();
            const minBindingSize = packet.size();
            const visibility = packet.word();
            const type = packet.word() ? "read-only-storage" : "storage";
            return { binding, visibility, buffer: { type, minBindingSize } };
        }, groups);
        bindingGroups(packet, () => sampledTextureLayout(packet.word(), packet.word(), packet.word(), packet.word() !== 0, packet.word()), groups);
        bindingGroups(packet, () => decodeStorageTextureLayout(packet, packet.word(), packet.word()), groups);
        bindingGroups(packet, () => samplerLayout(packet.word(), packet.word(), packet.word()), groups);
        bindingGroups(packet, () => ({ binding: packet.word(), visibility: packet.word(), externalTexture: {} }), groups);
        let depthStencil;
        if (packet.word()) {
            const format = textureFormat(packet.word());
            const depthCompare = compareFunction(packet.word());
            const depthWriteEnabled = packet.word() !== 0;
            const depthBias = packet.word() | 0;
            const depthBiasSlopeScale = packet.float();
            const depthBiasClamp = packet.float();
            depthStencil = format === "stencil8" ? { format } :
                { format, depthCompare, depthWriteEnabled, depthBias, depthBiasSlopeScale, depthBiasClamp };
            if (packet.word()) {
                depthStencil.stencilReadMask = packet.word();
                depthStencil.stencilWriteMask = packet.word();
                depthStencil.stencilFront = stencilFace(packet);
                depthStencil.stencilBack = stencilFace(packet);
            }
        }
        packet.done();
        if (renderPipelines.has(id)) throw new Error("Duplicate WebGPU render pipeline");
        const pipeline = await withErrors(() => device.createRenderPipelineAsync({
            layout: groups.size || immediateSize ? pipelineLayout(groups, immediateSize) : "auto",
            depthStencil,
            multisample: { count: sampleCount, mask, alphaToCoverageEnabled },
            primitive: { frontFace, cullMode, topology, stripIndexFormat, unclippedDepth },
            vertex: { entryPoint: vertex.entryPoint, module: device.createShaderModule({ code: vertex.code }),
                buffers: vertexBuffers },
            fragment: fragment ? {
                entryPoint: fragment.entryPoint, module: device.createShaderModule({ code: fragment.code }),
                targets,
            } : undefined,
        }));
        renderPipelines.set(id, { pipeline, fetchLayouts });
    }

    function decodeRender(packet, indexed, indirect) {
        const { pipeline, fetchLayouts } = lookup(renderPipelines, packet.id());
        const immediates = packet.take(packet.word() * 4);
        const render = { type: "render", pipeline, fetchLayouts, indirect, immediates };
        if (!indirect) {
            render.count = packet.word();
            render.firstVertex = packet.word();
            render.instanceCount = packet.word();
            render.firstInstance = packet.word();
        }
        render.blendConstant = [packet.float(), packet.float(), packet.float(), packet.float()];
        if (packet.word()) render.viewport = Array.from({ length: 6 }, () => packet.float());
        if (packet.word()) render.scissor = Array.from({ length: 4 }, () => packet.word());
        render.vertexBuffers = packet.list(() => ({ buffer: lookup(buffers, packet.id()), offset: packet.size() }));
        render.bindings = decodeResourceBindings(packet);
        function readStoreOp() {
            const action = packet.word();
            if (action > 1) throw new Error("Invalid WebGPU attachment store action");
            return action === 0 ? "store" : "discard";
        }
        if (packet.word()) {
            const depthId = packet.id();
            const depthView = textureViews.get(depthId);
            const mipLevel = packet.word();
            const load = packet.word();
            if (load > 3) throw new Error("Invalid WebGPU depth load action");
            render.depth = {
                view: depthView, texture: depthView ? undefined : lookup(textures, depthId), mipLevel,
                depthReadOnly: load === 2,
                depthLoadOp: load >= 2 ? undefined : load === 1 ? "clear" : "load",
                depthClearValue: load === 1 ? packet.float() : undefined,
            };
            const depthStoreOp = readStoreOp();
            render.depth.depthStoreOp = load < 2 ? depthStoreOp : undefined;
            if (packet.word()) {
                render.depth.stencilReference = packet.word();
                const stencilLoad = packet.word();
                if (stencilLoad > 2) throw new Error("Invalid WebGPU stencil load action");
                render.depth.stencilReadOnly = stencilLoad === 2;
                render.depth.stencilLoadOp = stencilLoad === 2 ? undefined : stencilLoad === 1 ? "clear" : "load";
                render.depth.stencilClearValue = stencilLoad === 1 ? packet.word() & 0xff : undefined;
                const stencilStoreOp = readStoreOp();
                render.depth.stencilStoreOp = stencilLoad < 2 ? stencilStoreOp : undefined;
            }
        }
        function colorView() {
            const id = packet.id();
            const view = textureViews.get(id);
            const mipLevel = packet.word();
            return { view, texture: view ? undefined : lookup(textures, id), mipLevel };
        }
        render.colors = colorSlots(packet, () => {
            const target = colorView();
            const depthSlice = packet.word() ? packet.word() : undefined;
            const load = packet.word();
            if (load !== 0 && load !== 1) throw new Error("Invalid WebGPU color load action");
            const clearValue = load ? [packet.float(), packet.float(), packet.float(), packet.float()] : undefined;
            const storeOp = readStoreOp();
            const resolve = packet.word() ? colorView() : undefined;
            return { target, depthSlice, loadOp: load ? "clear" : "load", storeOp, clearValue, resolve };
        });
        if (indexed) {
            const buffer = lookup(buffers, packet.id());
            const format = packet.word();
            if (format !== 2 && format !== 4) throw new Error("Unknown WebGPU index format");
            const offset = packet.size();
            const size = packet.word() ? packet.size() : undefined;
            render.index = { buffer, format: format === 2 ? "uint16" : "uint32", offset, size };
            if (!indirect) {
                render.index.first = packet.word();
                render.index.baseVertex = packet.word() | 0;
            }
        }
        return render;
    }

    function decodeCommand(packet, wrapperDepth = 0) {
        switch (packet.word()) {
        case 1: return { type: "write", buffer: lookup(buffers, packet.id()),
            offset: packet.size(), bytes: packet.take(packet.word() * 4) };
        case 2: {
            const pipeline = lookup(pipelines, packet.id());
            const groups = [packet.word(), packet.word(), packet.word()];
            const bindings = decodeResourceBindings(packet);
            const immediates = packet.take(packet.word() * 4);
            return { type: "dispatch", pipeline, groups, bindings, immediates };
        }
        case 3: {
            const source = lookup(buffers, packet.id());
            const sourceOffset = packet.size();
            const destination = lookup(buffers, packet.id());
            const destinationOffset = packet.size();
            const size = packet.word() ? packet.size() : source.size - sourceOffset;
            return { type: "copy", source, sourceOffset, destination, destinationOffset, size };
        }
        case 4: return { type: "read", buffer: lookup(buffers, packet.id()), size: packet.size() };
        case 5: return decodeRender(packet, false);
        case 6: return { type: "copyTexture", source: lookup(textures, packet.id()),
            destination: lookup(buffers, packet.id()), bytesPerRow: packet.word(), mipLevel: packet.word(),
            aspect: textureAspect(packet.word()), offset: packet.size(), z: packet.word(), depth: packet.word() };
        case 7: return decodeRender(packet, true);
        case 8: {
            const source = lookup(buffers, packet.id());
            const destination = lookup(textures, packet.id());
            const offset = packet.size();
            const bytesPerRow = packet.word() ? packet.word() : undefined;
            const origin = [packet.word(), packet.word(), packet.word()];
            const size = [packet.word(), packet.word(), packet.word()];
            return { type: "uploadTexture", source, destination, offset, bytesPerRow, origin, size,
                mipLevel: packet.word(), aspect: bufferCopyAspect(destination, textureAspect(packet.word())) };
        }
        case 9: {
            const source = lookup(textures, packet.id());
            const destination = lookup(textures, packet.id());
            const src = { texture: source, origin: [packet.word(), packet.word(), packet.word()], mipLevel: packet.word() };
            const dst = { texture: destination, origin: [packet.word(), packet.word(), packet.word()], mipLevel: packet.word() };
            return { type: "copyTextureRegion", src, dst, size: [packet.word(), packet.word(), packet.word()] };
        }
        case 10: {
            const pipeline = lookup(pipelines, packet.id());
            const indirect = { buffer: lookup(buffers, packet.id()), offset: packet.size() };
            const bindings = decodeResourceBindings(packet);
            const immediates = packet.take(packet.word() * 4);
            return { type: "dispatch", pipeline, indirect, bindings, immediates };
        }
        case 11: {
            const indirect = { buffer: lookup(buffers, packet.id()), offset: packet.size() };
            return decodeRender(packet, packet.word() !== 0, indirect);
        }
        case 12: {
            if (wrapperDepth >= 2) throw new Error("Invalid query pass nesting");
            const querySet = lookup(querySets, packet.id());
            const beginning = packet.word();
            const end = packet.word();
            const command = decodeCommand(packet, wrapperDepth + 1);
            if (command.timestampWrites || (command.type !== "dispatch" && command.type !== "render")) {
                throw new Error("Timestamp writes require a compute or render pass");
            }
            command.timestampWrites = {
                querySet, beginningOfPassWriteIndex: beginning === 0xffffffff ? undefined : beginning,
                endOfPassWriteIndex: end === 0xffffffff ? undefined : end,
            };
            return command;
        }
        case 13: return { type: "resolveQuerySet", querySet: lookup(querySets, packet.id()),
            first: packet.word(), count: packet.word(), destination: lookup(buffers, packet.id()), offset: packet.size() };
        case 14: {
            if (wrapperDepth >= 2) throw new Error("Invalid query pass nesting");
            const querySet = lookup(querySets, packet.id());
            const index = packet.word();
            const command = decodeCommand(packet, wrapperDepth + 1);
            if (command.type !== "render" || command.occlusion) {
                throw new Error("An occlusion query requires a render pass");
            }
            command.occlusion = { querySet, index };
            return command;
        }
        case 15: return { type: "write", buffer: lookup(buffers, packet.id()),
            offset: packet.size(), bytes: packet.take(packet.word()) };
        case 16: {
            const buffer = lookup(buffers, packet.id());
            const offset = packet.size();
            const size = packet.word() ? packet.size() : buffer.size - offset;
            return { type: "clearBuffer", buffer, offset, size };
        }
        case 17: {
            const texture = lookup(textures, packet.id());
            const mipLevel = packet.word();
            const aspect = bufferCopyAspect(texture, textureAspect(packet.word()));
            const origin = [packet.word(), packet.word(), packet.word()];
            const size = [packet.word(), packet.word(), packet.word()];
            const bytesPerRow = packet.word() ? packet.word() : undefined;
            const rowsPerImage = packet.word() ? packet.word() : undefined;
            return { type: "writeTexture", destination: { texture, mipLevel, aspect, origin },
                size, layout: { bytesPerRow, rowsPerImage }, bytes: packet.take(packet.word()) };
        }
        default: throw new Error("Unknown WebGPU command");
        }
    }

    // Validate before drawing so GPU-authored ranges behave like wgpu even when
    // the browser's native vertex fetch substitutes an in-range attribute.
    function validateIndirectDraw(encoder, command, staging) {
        if (!indirectDrawValidator) {
            const module = device.createShaderModule({ code: `
                struct DrawArguments {
                    words: array<u32, 5>,
                    lastElement: u32,
                    lastInstance: u32,
                    flags: u32,
                    hasCapacity: u32,
                }
                @group(0) @binding(0) var<storage, read_write> args: DrawArguments;
                fn fits(first: u32, count: u32, last: u32) -> bool {
                    return count != 0u && first <= last && count - 1u <= last - first;
                }
                @compute @workgroup_size(1) fn validate_draw() {
                    let firstInstance = args.words[3u + (args.flags & 1u)];
                    let supportsFirstInstance = (args.flags & 2u) != 0u;
                    if args.hasCapacity == 0u || (!supportsFirstInstance && firstInstance != 0u) ||
                        !fits(args.words[2], args.words[0], args.lastElement) ||
                        !fits(firstInstance, args.words[1], args.lastInstance) {
                        args.words = array<u32, 5>();
                    }
                }
            ` });
            indirectDrawValidator = device.createComputePipeline({
                layout: "auto", compute: { module, entryPoint: "validate_draw" },
            });
        }
        const maximumCapacity = 2 ** 32;
        let vertices = maximumCapacity;
        let instances = maximumCapacity;
        for (const [slot, layout] of command.fetchLayouts.entries()) {
            const { buffer, offset } = command.vertexBuffers[slot];
            const remaining = buffer.size - offset - layout.attributeEnd;
            const capacity = remaining < 0 ? 0 : layout.arrayStride === 0 ? maximumCapacity :
                Math.floor(remaining / layout.arrayStride) + 1;
            if (layout.stepMode === "instance") instances = Math.min(instances, capacity);
            else vertices = Math.min(vertices, capacity);
        }
        const index = command.index;
        const indexCount = index ? Math.floor((index.size ?? (index.buffer.size - index.offset)) /
            (index.format === "uint16" ? 2 : 4)) : vertices;
        const count = Math.min(maximumCapacity, indexCount);
        const flags = Number(Boolean(index)) + 2 * Number(indirectFirstInstance);
        const args = device.createBuffer({
            size: 36, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.INDIRECT, mappedAtCreation: true,
        });
        staging.push(args);
        // The first five words are reserved for the indirect command. The rest hold bounds.
        new Uint32Array(args.getMappedRange()).set([
            Math.max(0, count - 1), Math.max(0, instances - 1), flags, Number(vertices > 0 && instances > 0 && count > 0),
        ], 5);
        args.unmap();
        encoder.copyBufferToBuffer(command.indirect.buffer, command.indirect.offset, args, 0, index ? 20 : 16);
        const pass = encoder.beginComputePass();
        pass.setPipeline(indirectDrawValidator);
        pass.setBindGroup(0, device.createBindGroup({
            layout: indirectDrawValidator.getBindGroupLayout(0),
            entries: [{ binding: 0, resource: { buffer: args } }],
        }));
        pass.dispatchWorkgroups(1);
        pass.end();
        return args;
    }

    function encodeTimestampWork(pass, staging) {
        if (!timestampMarkerPipeline) {
            const module = device.createShaderModule({ code: `
                @group(0) @binding(0) var<storage, read_write> scratch: u32;
                @compute @workgroup_size(1) fn mark() { scratch = 1u; }
            ` });
            timestampMarkerPipeline = device.createComputePipeline({
                layout: "auto", compute: { module, entryPoint: "mark" },
            });
        }
        const scratch = device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE });
        staging.push(scratch);
        pass.setPipeline(timestampMarkerPipeline);
        pass.setBindGroup(0, device.createBindGroup({
            layout: timestampMarkerPipeline.getBindGroupLayout(0),
            entries: [{ binding: 0, resource: { buffer: scratch } }],
        }));
        pass.dispatchWorkgroups(1);
    }

    function encodeTimestampMarker(encoder, timestampWrites, staging) {
        const pass = encoder.beginComputePass({ timestampWrites });
        encodeTimestampWork(pass, staging);
        pass.end();
    }

    async function submit(packet) {
        const commands = packet.list(() => decodeCommand(packet));
        packet.done();
        const length = commands.reduce((size, command) => size + (command.type === "read" ? command.size + 4 : 0), 4);
        if (length > limit) throw Object.assign(new Error("WebGPU readback exceeds bridge limit"), { status: 1 });
        const reply = new Uint8Array(length);
        const header = new DataView(reply.buffer);
        const staging = [];
        let readCount = 0;
        let offset = 4;
        let encoder;
        const getEncoder = () => (encoder ??= device.createCommandEncoder());
        function flush() {
            if (!encoder) return;
            device.queue.submit([encoder.finish()]);
            encoder = undefined;
        }

        try {
            await withErrors(async () => {
                for (const command of commands) {
                    switch (command.type) {
                    case "write": {
                        if (!command.bytes.length) break;
                        const upload = device.createBuffer({
                            size: command.bytes.length, usage: GPUBufferUsage.COPY_SRC, mappedAtCreation: true,
                        });
                        staging.push(upload);
                        new Uint8Array(upload.getMappedRange()).set(command.bytes);
                        upload.unmap();
                        getEncoder().copyBufferToBuffer(upload, 0, command.buffer, command.offset, command.bytes.length);
                        break;
                    }
                    case "dispatch": {
                        if (command.groups?.includes(0)) {
                            if (command.timestampWrites) encodeTimestampMarker(getEncoder(), command.timestampWrites, staging);
                            break;
                        }
                        const pass = getEncoder().beginComputePass({ timestampWrites: command.timestampWrites });
                        if (command.indirect && command.timestampWrites) encodeTimestampWork(pass, staging);
                        pass.setPipeline(command.pipeline);
                        bindResources(pass, command.pipeline, command.bindings);
                        if (command.immediates.length) pass.setImmediates(0, command.immediates);
                        if (command.indirect) {
                            pass.dispatchWorkgroupsIndirect(command.indirect.buffer, command.indirect.offset);
                        } else {
                            pass.dispatchWorkgroups(...command.groups);
                        }
                        pass.end();
                        break;
                    }
                    case "copy":
                        if (command.size) {
                            getEncoder().copyBufferToBuffer(command.source, command.sourceOffset,
                                command.destination, command.destinationOffset, command.size);
                        }
                        break;
                    case "clearBuffer":
                        if (command.size) getEncoder().clearBuffer(command.buffer, command.offset, command.size);
                        break;
                    case "resolveQuerySet":
                        if (command.count) {
                            getEncoder().resolveQuerySet(command.querySet, command.first, command.count, command.destination, command.offset);
                        }
                        break;
                    case "render": {
                        const indirect = command.indirect && validateIndirectDraw(getEncoder(), command, staging);
                        const depth = command.depth;
                        const depthStencilAttachment = depth ? {
                            view: depth.view ?? depth.texture.createView({ baseMipLevel: depth.mipLevel, mipLevelCount: 1 }),
                            depthReadOnly: depth.depthReadOnly, stencilReadOnly: depth.stencilReadOnly,
                            depthLoadOp: depth.depthLoadOp, depthClearValue: depth.depthClearValue,
                            depthStoreOp: depth.depthStoreOp,
                            stencilLoadOp: depth.stencilLoadOp, stencilClearValue: depth.stencilClearValue,
                            stencilStoreOp: depth.stencilStoreOp,
                        } : undefined;
                        const view = attachment => attachment.view ?? attachment.texture.createView({
                            baseMipLevel: attachment.mipLevel, mipLevelCount: 1,
                        });
                        const colorAttachments = command.colors.map(color => color ? {
                            view: view(color.target), resolveTarget: color.resolve ? view(color.resolve) : undefined,
                            depthSlice: color.depthSlice,
                            loadOp: color.loadOp, storeOp: color.storeOp, clearValue: color.clearValue,
                        } : null);
                        const timestampWrites = command.timestampWrites;
                        const empty = !command.indirect && (!command.count || !command.instanceCount ||
                            command.scissor?.slice(2).includes(0));
                        if (empty && timestampWrites?.beginningOfPassWriteIndex !== undefined) {
                            encodeTimestampMarker(getEncoder(), {
                                querySet: timestampWrites.querySet,
                                beginningOfPassWriteIndex: timestampWrites.beginningOfPassWriteIndex,
                            }, staging);
                        }
                        const pass = getEncoder().beginRenderPass({ depthStencilAttachment, colorAttachments,
                            timestampWrites: empty ? undefined : timestampWrites, occlusionQuerySet: command.occlusion?.querySet });
                        pass.setPipeline(command.pipeline);
                        pass.setBlendConstant(command.blendConstant);
                        if (command.immediates.length) pass.setImmediates(0, command.immediates);
                        if (depth?.stencilReference !== undefined) pass.setStencilReference(depth.stencilReference);
                        if (command.viewport) pass.setViewport(...command.viewport);
                        if (command.scissor) pass.setScissorRect(...command.scissor);
                        bindResources(pass, command.pipeline, command.bindings);
                        for (const [slot, binding] of command.vertexBuffers.entries()) {
                            pass.setVertexBuffer(slot, binding.buffer, binding.offset);
                        }
                        if (command.occlusion) pass.beginOcclusionQuery(command.occlusion.index);
                        if (indirect) {
                            if (command.index) {
                                const index = command.index;
                                pass.setIndexBuffer(index.buffer, index.format, index.offset, index.size);
                                pass.drawIndexedIndirect(indirect, 0);
                            } else {
                                pass.drawIndirect(indirect, 0);
                            }
                        } else if (command.count && command.instanceCount) {
                            if (command.index) {
                                const { buffer, format, offset, size, first, baseVertex } = command.index;
                                pass.setIndexBuffer(buffer, format, offset, size);
                                pass.drawIndexed(command.count, command.instanceCount, first, baseVertex, command.firstInstance);
                            } else {
                                pass.draw(command.count, command.instanceCount, command.firstVertex, command.firstInstance);
                            }
                        }
                        if (command.occlusion) pass.endOcclusionQuery();
                        pass.end();
                        if (empty && timestampWrites?.endOfPassWriteIndex !== undefined) {
                            encodeTimestampMarker(getEncoder(), {
                                querySet: timestampWrites.querySet, endOfPassWriteIndex: timestampWrites.endOfPassWriteIndex,
                            }, staging);
                        }
                        break;
                    }
                    case "copyTexture": {
                        const { source, destination, bytesPerRow, mipLevel, offset, z, depth } = command;
                        const scale = 2 ** mipLevel;
                        const astc = /^astc-(\d+)x(\d+)-/.exec(source.format);
                        const block = /^(bc|etc2-|eac-)/.test(source.format) ? 4 : 1;
                        const blockShape = astc ? [Number(astc[1]), Number(astc[2])] : [block, block];
                        const size = [source.width, source.height].map((axis, index) => {
                            const logical = Math.max(1, Math.floor(axis / scale));
                            const dimension = blockShape[index];
                            return Math.ceil(logical / dimension) * dimension;
                        });
                        const aspect = bufferCopyAspect(source, command.aspect);
                        getEncoder().copyTextureToBuffer({ texture: source, mipLevel, aspect, origin: [0, 0, z] },
                            { buffer: destination, bytesPerRow, offset, rowsPerImage: size[1] / blockShape[1] }, [...size, depth]);
                        break;
                    }
                    case "copyTextureRegion":
                        if (!command.size.includes(0)) {
                            getEncoder().copyTextureToTexture(command.src, command.dst, command.size);
                        }
                        break;
                    case "uploadTexture":
                        if (command.size.includes(0)) break;
                        getEncoder().copyBufferToTexture(
                            { buffer: command.source, offset: command.offset, bytesPerRow: command.bytesPerRow,
                                rowsPerImage: command.size[2] > 1 ? command.size[1] : undefined },
                            { texture: command.destination, origin: command.origin, mipLevel: command.mipLevel, aspect: command.aspect },
                            command.size,
                        );
                        break;
                    case "writeTexture":
                        if (!command.size.includes(0)) {
                            flush();
                            device.queue.writeTexture(command.destination, command.bytes, command.layout, command.size);
                        }
                        break;
                    case "read":
                        flush();
                        await command.buffer.mapAsync(GPUMapMode.READ, 0, command.size);
                        try {
                            header.setUint32(offset, command.size / 4, true);
                            reply.set(new Uint8Array(command.buffer.getMappedRange(0, command.size)), offset + 4);
                            offset += command.size + 4;
                            readCount += 1;
                        } finally {
                            command.buffer.unmap();
                        }
                        break;
                    }
                }
                flush();
                await device.queue.onSubmittedWorkDone();
            });
            header.setUint32(0, readCount, true);
            return reply;
        } finally {
            for (const buffer of staging) buffer.destroy();
        }
    }

    async function handle(bytes) {
        const packet = new Reader(bytes);
        const operation = packet.word();
        if (lost && ![2, 4, 5, 9, 11, 18, 22, 25].includes(operation)) throw Object.assign(new Error(lost), { status: 3 });
        switch (operation) {
        case 1: {
            const id = packet.id();
            const size = packet.size();
            let usage = packet.word();
            const mappedAtCreation = packet.word() !== 0;
            packet.done();
            if (buffers.has(id)) throw new Error("Duplicate WebGPU buffer");
            if (usage & GPUBufferUsage.INDIRECT) usage |= GPUBufferUsage.COPY_SRC;
            const buffer = await withErrors(() => device.createBuffer({ size, usage, mappedAtCreation }));
            if (mappedAtCreation) {
                try {
                    mappings.set(id, { offset: 0, bytes: new Uint8Array(buffer.getMappedRange()), writable: true });
                } catch (error) {
                    buffer.destroy();
                    throw error;
                }
            }
            buffers.set(id, buffer);
            break;
        }
        case 2: {
            const id = packet.id();
            packet.done();
            buffers.get(id)?.destroy();
            buffers.delete(id);
            mappings.delete(id);
            break;
        }
        case 3: await allocatePipeline(packet); break;
        case 4: {
            const id = packet.id();
            packet.done();
            pipelines.delete(id);
            break;
        }
        case 5: {
            const prefix = `${packet.wide()}:`;
            packet.done();
            for (const resources of [buffers, textures, querySets]) {
                for (const [id, resource] of resources) {
                    if (id.startsWith(prefix)) { resource.destroy(); resources.delete(id); }
                }
            }
            for (const states of [pipelines, renderPipelines, samplers, textureViews, externalTextures]) {
                for (const id of states.keys()) if (id.startsWith(prefix)) states.delete(id);
            }
            for (const id of mappings.keys()) if (id.startsWith(prefix)) mappings.delete(id);
            for (const id of stagedSubmissions.keys()) if (id.startsWith(prefix)) stagedSubmissions.delete(id);
            break;
        }
        case 6:
            submissions += 1;
            return await submit(packet);
        case 7: packet.done(); await withErrors(() => device.queue.onSubmittedWorkDone()); break;
        case 8: {
            const id = packet.id();
            const size = [packet.word(), packet.word()];
            const format = textureFormat(packet.word());
            const usage = packet.word();
            const mipLevelCount = packet.word();
            const sampleCount = packet.word();
            size.push(packet.word());
            const dimension = ["2d", "3d", "1d"][packet.word()];
            packet.done();
            if (!dimension || textures.has(id)) throw new Error("Invalid WebGPU texture dimension or duplicate texture");
            textures.set(id, await withErrors(() => device.createTexture({ size, format, usage, mipLevelCount, sampleCount, dimension })));
            break;
        }
        case 9: {
            const id = packet.id();
            packet.done();
            textures.get(id)?.destroy();
            textures.delete(id);
            break;
        }
        case 10: await allocateRenderPipeline(packet); break;
        case 11: {
            const id = packet.id();
            packet.done();
            renderPipelines.delete(id);
            break;
        }
        case 12: {
            const source = lookup(textures, packet.id());
            packet.done();
            await present(source);
            break;
        }
        case 13: {
            const id = packet.id();
            const addresses = ["clamp-to-edge", "repeat", "mirror-repeat"];
            const filters = ["nearest", "linear"];
            const addressModeU = addresses[packet.word()];
            const addressModeV = addresses[packet.word()];
            const minFilter = filters[packet.word()];
            const magFilter = filters[packet.word()];
            const mipmapFilter = filters[packet.word()];
            const compareMode = packet.word();
            const compare = compareMode ? compareFunction(compareMode - 1) : undefined;
            packet.done();
            if (!addressModeU || !addressModeV || !minFilter || !magFilter || !mipmapFilter) {
                throw new Error("Invalid WebGPU sampler descriptor");
            }
            if (samplers.has(id)) throw new Error("Duplicate WebGPU sampler");
            samplers.set(id, await withErrors(() =>
                device.createSampler({ addressModeU, addressModeV, minFilter, magFilter, mipmapFilter, compare })));
            break;
        }
        case 14: {
            const id = packet.id();
            packet.done();
            samplers.delete(id);
            break;
        }
        case 15: {
            const id = packet.id();
            const texture = lookup(textures, packet.id());
            const baseMipLevel = packet.word();
            const mipLevelCount = packet.word() ? packet.word() : undefined;
            const aspect = textureAspect(packet.word());
            const dimensionCode = packet.word();
            const dimension = [undefined, "2d", "2d-array", "cube", "cube-array", "3d", "1d"][dimensionCode];
            if (dimensionCode !== 0 && !dimension) throw new Error("Invalid WebGPU texture view dimension");
            const baseArrayLayer = packet.word();
            const arrayLayerCount = packet.word() ? packet.word() : undefined;
            packet.done();
            if (textureViews.has(id)) throw new Error("Duplicate WebGPU texture view");
            textureViews.set(id, await withErrors(() => texture.createView({
                baseMipLevel, mipLevelCount, aspect, dimension, baseArrayLayer, arrayLayerCount,
            })));
            break;
        }
        case 16: {
            const id = packet.id();
            packet.done();
            textureViews.delete(id);
            break;
        }
        case 17: {
            const id = packet.id();
            const type = ["timestamp", "occlusion"][packet.word()];
            const count = packet.word();
            const label = packet.string();
            packet.done();
            if (!type || querySets.has(id)) throw new Error("Invalid WebGPU query set");
            querySets.set(id, await withErrors(() => device.createQuerySet({ type, count, label })));
            break;
        }
        case 18: {
            const id = packet.id();
            packet.done();
            querySets.get(id)?.destroy();
            querySets.delete(id);
            break;
        }
        case 19: {
            const id = packet.id();
            const mode = packet.word();
            const offset = packet.size();
            const size = packet.size();
            packet.done();
            const buffer = lookup(buffers, id);
            if (mappings.has(id)) throw Object.assign(new Error("Browser buffer already mapped"), { status: 1 });
            try {
                await withErrors(() => buffer.mapAsync(mode, offset, size));
                mappings.set(id, { offset, bytes: new Uint8Array(buffer.getMappedRange(offset, size)),
                    writable: mode === GPUMapMode.WRITE });
            } catch (error) {
                if (buffer.mapState === "mapped") buffer.unmap();
                throw error;
            }
            break;
        }
        case 20: {
            const id = packet.id();
            const offset = packet.size();
            const size = packet.word();
            packet.done();
            return mappedRange(id, offset, size, false).slice();
        }
        case 21: {
            const id = packet.id();
            const offset = packet.size();
            const bytes = packet.take(packet.word());
            packet.done();
            mappedRange(id, offset, bytes.length, true).set(bytes);
            break;
        }
        case 22: {
            const id = packet.id();
            packet.done();
            if (mappings.has(id)) {
                lookup(buffers, id).unmap();
                mappings.delete(id);
            }
            break;
        }
        case 23: {
            const id = packet.id();
            const total = packet.size();
            const offset = packet.size();
            const chunk = packet.take(packet.word());
            packet.done();
            if (total <= limit || !chunk.length || offset > total || chunk.length > total - offset) {
                throw new Error("Invalid browser submission chunk");
            }
            let staged = stagedSubmissions.get(id);
            if (!staged && offset === 0) {
                staged = { bytes: new Uint8Array(total), received: 0 };
                stagedSubmissions.set(id, staged);
            }
            if (!staged || staged.bytes.length !== total || staged.received !== offset) {
                throw new Error("Out-of-order browser submission chunk");
            }
            staged.bytes.set(chunk, offset);
            staged.received += chunk.length;
            break;
        }
        case 24: {
            const id = packet.id();
            packet.done();
            const staged = lookup(stagedSubmissions, id);
            stagedSubmissions.delete(id);
            const submission = new Reader(staged.bytes);
            if (staged.received !== staged.bytes.length || submission.word() !== 6) {
                throw new Error("Incomplete staged browser submission");
            }
            submissions += 1;
            return await submit(submission);
        }
        case 25: {
            const id = packet.id();
            packet.done();
            stagedSubmissions.delete(id);
            break;
        }
        case 26: {
            const id = packet.id();
            const frame = videoFrames.get(packet.wide());
            const colorSpace = ["srgb", "display-p3"][packet.word()];
            packet.done();
            if (!frame || frame.codedWidth === 0) {
                throw Object.assign(new Error("Unknown or closed VideoFrame"), { status: 1 });
            }
            if (!colorSpace || externalTextures.has(id)) throw new Error("Invalid external texture descriptor");
            const resource = await withErrors(() => device.importExternalTexture({ source: frame, colorSpace }));
            externalTextures.set(id, resource);
            break;
        }
        case 27: {
            const id = packet.id();
            packet.done();
            externalTextures.delete(id);
            break;
        }
        default: throw new Error("Unknown browser bridge operation");
        }
        return new Uint8Array();
    }

    async function present(source) {
        if (!context) throw new Error("Browser host does not have a presentation surface");
        const { width, height, format } = source;
        await withErrors(async () => {
            if (canvas.width !== width) canvas.width = width;
            if (canvas.height !== height) canvas.height = height;
            if (configuredFormat !== format) {
                context.configure({ device, format, alphaMode: "opaque", usage: GPUTextureUsage.COPY_DST });
                ownsConfiguration = true;
            }
            // Acquire and submit in the same turn before the canvas texture can expire.
            const destination = context.getCurrentTexture();
            const encoder = device.createCommandEncoder();
            encoder.copyTextureToTexture({ texture: source }, { texture: destination }, [width, height]);
            device.queue.submit([encoder.finish()]);
            await device.queue.onSubmittedWorkDone();
        });
        configuredFormat = format;
        presentedFrames += 1;
    }

    return {
        maxBufferSize: device.limits.maxBufferSize,
        maxComputeWorkgroupStorageSize: device.limits.maxComputeWorkgroupStorageSize,
        maxComputeInvocationsPerWorkgroup: device.limits.maxComputeInvocationsPerWorkgroup,
        maxComputeWorkgroupSize: {
            x: device.limits.maxComputeWorkgroupSizeX,
            y: device.limits.maxComputeWorkgroupSizeY,
            z: device.limits.maxComputeWorkgroupSizeZ,
        },
        maxComputeWorkgroupsPerDimension: device.limits.maxComputeWorkgroupsPerDimension,
        maxVertexAttributes: device.limits.maxVertexAttributes,
        maxVertexBuffers: device.limits.maxVertexBuffers,
        maxVertexBufferArrayStride: device.limits.maxVertexBufferArrayStride,
        maxInterStageShaderVariables: device.limits.maxInterStageShaderVariables,
        maxColorAttachments: device.limits.maxColorAttachments,
        maxColorAttachmentBytesPerSample: device.limits.maxColorAttachmentBytesPerSample,
        supportsVertexFormatSnorm1010102,
        maxImmediateSize: navigator.gpu.wgslLanguageFeatures?.has("immediate_address_space")
            ? device.limits.maxImmediateSize ?? 0 : 0,
        maxTextureDimension2D: device.limits.maxTextureDimension2D,
        supportsReadableStorage: navigator.gpu.wgslLanguageFeatures?.has("readonly_and_readwrite_storage_textures") ?? false,
        supportsPresentation: context != null,
        supportsShaderF16: device.features.has("shader-f16"),
        supportsIndirectFirstInstance: indirectFirstInstance,
        supportsClipDistances: device.features.has("clip-distances"),
        supportsPrimitiveIndex: device.features.has("primitive-index"),
        supportsDepthClipControl: device.features.has("depth-clip-control"),
        supportsTimestampQuery: device.features.has("timestamp-query"),
        supportsSubgroups: device.features.has("subgroups"),
        supportsSubgroupSizeControl: device.features.has("subgroup-size-control"),
        subgroupMinSize: adapter.info.subgroupMinSize ?? 0,
        subgroupMaxSize: adapter.info.subgroupMaxSize ?? 0,
        supportsSubgroupID: device.features.has("subgroups") && (navigator.gpu.wgslLanguageFeatures?.has("subgroup_id") ?? false),
        supportsSubgroupUniformity: navigator.gpu.wgslLanguageFeatures?.has("subgroup_uniformity") ?? false,
        supportsLinearIndexing: navigator.gpu.wgslLanguageFeatures?.has("linear_indexing") ?? false,
        supportsDualSourceBlending: device.features.has("dual-source-blending"),
        supportsTextureCompressionBC: device.features.has("texture-compression-bc"),
        supportsTextureCompressionETC2: device.features.has("texture-compression-etc2"),
        supportsTextureCompressionASTC: device.features.has("texture-compression-astc"),
        supportsTextureCompressionBCSliced3D: device.features.has("texture-compression-bc-sliced-3d"),
        supportsTextureCompressionASTCSliced3D: device.features.has("texture-compression-astc-sliced-3d"),
        supportsFloat32Filtering: device.features.has("float32-filterable"),
        supportsFloat32Blending: device.features.has("float32-blendable"),
        supportsTextureFormatsTier1: device.features.has("texture-formats-tier1"),
        supportsTextureFormatsTier2: device.features.has("texture-formats-tier2"),
        supportsRG11B10Render: device.features.has("rg11b10ufloat-renderable"),
        supportsDepth32FloatStencil8: device.features.has("depth32float-stencil8"),
        supportsBGRA8UnormStorage: device.features.has("bgra8unorm-storage"),
        multisampledFloat32Mask,
        supportsUniformBufferStandardLayout: navigator.gpu.wgslLanguageFeatures?.has("uniform_buffer_standard_layout") ?? false,
        adapter: { vendor: adapter.info.vendor, fallback: adapter.info.isFallbackAdapter },
        registerVideoFrame,
        unregisterVideoFrame: token => videoFrames.delete(token),
        handle,
        metrics: () => ({
            submissions, remainingResources: buffers.size + textures.size + textureViews.size + samplers.size +
                pipelines.size + renderPipelines.size + querySets.size + mappings.size + stagedSubmissions.size +
                externalTextures.size + videoFrames.size,
            presentedFrames, surfaceSize: presentedFrames > 0 ? [canvas.width, canvas.height] : null,
        }),
        close() {
            if (closed) return;
            closed = true;
            stagedSubmissions.clear();
            externalTextures.clear();
            videoFrames.clear();
            if (ownsConfiguration) context.unconfigure();
            device.destroy();
        },
    };
}
