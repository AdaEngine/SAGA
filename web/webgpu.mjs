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
    default: throw new Error("Unknown WebGPU texture format");
    }
}

function textureAspect(value) {
    const aspect = ["all", "depth-only", "stencil-only"][value];
    if (!aspect) throw new Error("Invalid WebGPU texture aspect");
    return aspect;
}

function compareFunction(value) {
    const compare = ["never", "less", "equal", "less-equal", "greater", "not-equal", "greater-equal", "always"][value];
    if (!compare) throw new Error("Invalid WebGPU comparison function");
    return compare;
}

function vertexFormat(value) {
    switch (value) {
    case 1: return "float32";
    case 2: return "float32x2";
    case 3: return "float32x3";
    case 4: return "float32x4";
    case 5: return "uint32";
    case 6: return "uint32x2";
    case 7: return "uint32x3";
    case 8: return "uint32x4";
    case 9: return "sint32";
    case 10: return "sint32x2";
    case 11: return "sint32x3";
    case 12: return "sint32x4";
    default: throw new Error("Unknown WebGPU vertex format");
    }
}

export async function openWebGPU(canvas) {
    const context = canvas?.getContext("webgpu");
    if (canvas && !context) throw new Error("Canvas cannot supply a WebGPU surface");
    const adapter = await navigator.gpu?.requestAdapter();
    if (!adapter || adapter.info.isFallbackAdapter) throw new Error("No hardware WebGPU adapter available");
    const device = await adapter.requestDevice();
    const buffers = new Map();
    const textures = new Map();
    const textureViews = new Map();
    const samplers = new Map();
    const pipelines = new Map();
    const renderPipelines = new Map();
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

    function bindingGroups(packet, readEntry, groups = new Map()) {
        for (const [group, entry] of packet.list(() => [packet.word(), readEntry()])) {
            if (group >= device.limits.maxBindGroups) throw new Error("Invalid WebGPU bind group index");
            if (!groups.has(group)) groups.set(group, []);
            groups.get(group).push(entry);
        }
        return groups;
    }

    function pipelineLayout(groups) {
        const length = Math.max(-1, ...groups.keys()) + 1;
        const bindGroupLayouts = Array.from({ length }, (_, group) =>
            device.createBindGroupLayout({ entries: groups.get(group) ?? [] }));
        return device.createPipelineLayout({ bindGroupLayouts });
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
            const view = textureViews.get(id);
            return view ? { binding, resource: view } : { binding, texture: lookup(textures, id) };
        }, groups);
        return bindingGroups(packet, () => {
            const binding = packet.word();
            return { binding, resource: lookup(samplers, packet.id()) };
        }, groups);
    }

    function sampledTextureLayout(binding, visibility, type, multisampled, dimension) {
        const sampleType = [multisampled ? "unfilterable-float" : "float", "depth", "sint", "uint"][type];
        if (!sampleType) throw new Error("Invalid WebGPU texture sample type");
        const viewDimension = ["2d", "2d-array", "cube", "cube-array", "3d"][dimension];
        if (!viewDimension) throw new Error("Invalid sampled texture view dimension");
        return { binding, visibility, texture: { sampleType, viewDimension, multisampled } };
    }

    function decodeStorageTextureLayout(packet, binding, visibility) {
        const format = textureFormat(packet.word());
        const access = [undefined, "read-only", "write-only", "read-write"][packet.word()];
        if (!access) throw new Error("Invalid WebGPU storage texture access");
        const viewDimension = packet.word() ? "2d-array" : "2d";
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
        const groups = bindingGroups(packet, () => {
            const binding = packet.word();
            const minBindingSize = packet.size();
            const code = packet.word();
            if (code === 3) return sampledTextureLayout(binding, GPUShaderStage.COMPUTE, packet.word(), packet.word() !== 0, packet.word());
            if (code === 4) return samplerLayout(binding, GPUShaderStage.COMPUTE, packet.word());
            if (code === 5) return decodeStorageTextureLayout(packet, binding, GPUShaderStage.COMPUTE);
            const type = ["storage", "read-only-storage", "uniform"][code];
            if (!type) throw new Error("Invalid WebGPU compute binding type");
            return { binding, visibility: GPUShaderStage.COMPUTE, buffer: { type, minBindingSize } };
        });
        packet.done();
        if (pipelines.has(id)) throw new Error("Duplicate WebGPU pipeline");
        const pipeline = await withErrors(async () => {
            return await device.createComputePipelineAsync({
                layout: pipelineLayout(groups),
                compute: { module: device.createShaderModule({ code }), entryPoint },
            });
        });
        pipelines.set(id, pipeline);
    }

    function blendComponent(packet) {
        const operations = ["add", "subtract", "reverse-subtract", "min", "max"];
        const factors = ["zero", "one", "src", "one-minus-src", "dst", "one-minus-dst",
            "src-alpha", "one-minus-src-alpha", "dst-alpha", "one-minus-dst-alpha",
            "src-alpha-saturated", "constant", "one-minus-constant"];
        const operation = operations[packet.word()];
        const srcFactor = factors[packet.word()];
        const dstFactor = factors[packet.word()];
        if (!operation || !srcFactor || !dstFactor) throw new Error("Invalid WebGPU blend component");
        return { operation, srcFactor, dstFactor };
    }

    function colorSlots(packet, read) {
        const colors = [];
        for (const [location, value] of packet.list(() => [packet.word(), read()])) {
            if (location >= 8 || colors[location]) throw new Error("Invalid WebGPU color attachment location");
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
        if (!frontFace || !cullMode || !topology || ![0, 2, 4].includes(indexFormat)) {
            throw new Error("Invalid WebGPU primitive state");
        }
        const stripIndexFormat = indexFormat === 0 ? undefined : indexFormat === 2 ? "uint16" : "uint32";
        const vertex = { entryPoint: packet.string(), code: packet.string() };
        const fragment = { entryPoint: packet.string(), code: packet.string() };
        const vertexBuffers = packet.list(() => {
            const arrayStride = packet.word();
            const step = packet.word();
            if (step !== 0 && step !== 1) throw new Error("Unknown WebGPU vertex step mode");
            const attributes = packet.list(() => ({
                shaderLocation: packet.word(), offset: packet.word(), format: vertexFormat(packet.word()),
            }));
            return { arrayStride, stepMode: step === 0 ? "vertex" : "instance", attributes };
        });
        const groups = bindingGroups(packet, () => {
            const binding = packet.word();
            const minBindingSize = packet.size();
            const visibility = packet.word();
            return { binding, visibility, buffer: { type: "uniform", minBindingSize } };
        });
        bindingGroups(packet, () => sampledTextureLayout(packet.word(), packet.word(), packet.word(), packet.word() !== 0, packet.word()), groups);
        bindingGroups(packet, () => decodeStorageTextureLayout(packet, packet.word(), packet.word()), groups);
        bindingGroups(packet, () => samplerLayout(packet.word(), packet.word(), packet.word()), groups);
        let depthStencil;
        if (packet.word()) {
            const format = textureFormat(packet.word());
            const depthCompare = compareFunction(packet.word());
            const depthWriteEnabled = packet.word() !== 0;
            const depthBias = packet.word() | 0;
            const depthBiasSlopeScale = packet.float();
            const depthBiasClamp = packet.float();
            depthStencil = { format, depthCompare, depthWriteEnabled, depthBias, depthBiasSlopeScale, depthBiasClamp };
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
            layout: groups.size ? pipelineLayout(groups) : "auto",
            depthStencil,
            multisample: { count: sampleCount, mask, alphaToCoverageEnabled },
            primitive: { frontFace, cullMode, topology, stripIndexFormat },
            vertex: { entryPoint: vertex.entryPoint, module: device.createShaderModule({ code: vertex.code }),
                buffers: vertexBuffers },
            fragment: {
                entryPoint: fragment.entryPoint, module: device.createShaderModule({ code: fragment.code }),
                targets,
            },
        }));
        renderPipelines.set(id, pipeline);
    }

    function decodeRender(packet, indexed) {
        const pipeline = lookup(renderPipelines, packet.id());
        const render = {
            type: "render", pipeline, count: packet.word(),
            firstVertex: packet.word(), instanceCount: packet.word(), firstInstance: packet.word(),
        };
        render.blendConstant = [packet.float(), packet.float(), packet.float(), packet.float()];
        if (packet.word()) render.viewport = Array.from({ length: 6 }, () => packet.float());
        if (packet.word()) render.scissor = Array.from({ length: 4 }, () => packet.word());
        render.vertexBuffers = packet.list(() => ({ buffer: lookup(buffers, packet.id()), offset: packet.size() }));
        render.bindings = decodeResourceBindings(packet);
        if (packet.word()) {
            const depthId = packet.id();
            const depthView = textureViews.get(depthId);
            const mipLevel = packet.word();
            const load = packet.word();
            if (load > 2) throw new Error("Invalid WebGPU depth load action");
            render.depth = {
                view: depthView, texture: depthView ? undefined : lookup(textures, depthId), mipLevel,
                depthReadOnly: load === 2,
                depthLoadOp: load === 2 ? undefined : load === 1 ? "clear" : "load",
                depthClearValue: load === 1 ? packet.float() : undefined,
            };
            if (packet.word()) {
                render.depth.stencilReference = packet.word();
                const stencilLoad = packet.word();
                if (stencilLoad > 2) throw new Error("Invalid WebGPU stencil load action");
                render.depth.stencilReadOnly = stencilLoad === 2;
                render.depth.stencilLoadOp = stencilLoad === 2 ? undefined : stencilLoad === 1 ? "clear" : "load";
                render.depth.stencilClearValue = stencilLoad === 1 ? packet.word() & 0xff : undefined;
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
            const load = packet.word();
            if (load !== 0 && load !== 1) throw new Error("Invalid WebGPU color load action");
            const clearValue = load ? [packet.float(), packet.float(), packet.float(), packet.float()] : undefined;
            const resolve = packet.word() ? colorView() : undefined;
            return { target, loadOp: load ? "clear" : "load", clearValue, resolve };
        });
        if (indexed) {
            const buffer = lookup(buffers, packet.id());
            const format = packet.word();
            if (format !== 2 && format !== 4) throw new Error("Unknown WebGPU index format");
            const offset = packet.size();
            const size = packet.word() ? packet.size() : undefined;
            render.index = { buffer, format: format === 2 ? "uint16" : "uint32", offset, size,
                first: packet.word(), baseVertex: packet.word() | 0 };
        }
        return render;
    }

    function decodeCommand(packet) {
        switch (packet.word()) {
        case 1: return { type: "write", buffer: lookup(buffers, packet.id()),
            offset: packet.size(), bytes: packet.take(packet.word() * 4) };
        case 2: {
            const pipeline = lookup(pipelines, packet.id());
            const groups = [packet.word(), packet.word(), packet.word()];
            return { type: "dispatch", pipeline, groups, bindings: decodeResourceBindings(packet) };
        }
        case 3: return { type: "copy", source: lookup(buffers, packet.id()),
            destination: lookup(buffers, packet.id()), size: packet.size() };
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
                mipLevel: packet.word(), aspect: textureAspect(packet.word()) };
        }
        case 9: {
            const source = lookup(textures, packet.id());
            const destination = lookup(textures, packet.id());
            const src = { texture: source, origin: [packet.word(), packet.word(), packet.word()], mipLevel: packet.word() };
            const dst = { texture: destination, origin: [packet.word(), packet.word(), packet.word()], mipLevel: packet.word() };
            return { type: "copyTextureRegion", src, dst, size: [packet.word(), packet.word(), packet.word()] };
        }
        default: throw new Error("Unknown WebGPU command");
        }
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
                        if (command.groups.includes(0)) break;
                        const pass = getEncoder().beginComputePass();
                        pass.setPipeline(command.pipeline);
                        bindResources(pass, command.pipeline, command.bindings);
                        pass.dispatchWorkgroups(...command.groups);
                        pass.end();
                        break;
                    }
                    case "copy":
                        getEncoder().copyBufferToBuffer(command.source, 0, command.destination, 0, command.size);
                        break;
                    case "render": {
                        const depth = command.depth;
                        const depthStencilAttachment = depth ? {
                            view: depth.view ?? depth.texture.createView({ baseMipLevel: depth.mipLevel, mipLevelCount: 1 }),
                            depthReadOnly: depth.depthReadOnly, stencilReadOnly: depth.stencilReadOnly,
                            depthLoadOp: depth.depthLoadOp, depthClearValue: depth.depthClearValue,
                            depthStoreOp: depth.depthReadOnly ? undefined : "store",
                            stencilLoadOp: depth.stencilLoadOp, stencilClearValue: depth.stencilClearValue,
                            stencilStoreOp: depth.stencilLoadOp ? "store" : undefined,
                        } : undefined;
                        const view = attachment => attachment.view ?? attachment.texture.createView({
                            baseMipLevel: attachment.mipLevel, mipLevelCount: 1,
                        });
                        const colorAttachments = command.colors.map(color => color ? {
                            view: view(color.target), resolveTarget: color.resolve ? view(color.resolve) : undefined,
                            loadOp: color.loadOp, storeOp: "store", clearValue: color.clearValue,
                        } : null);
                        const pass = getEncoder().beginRenderPass({ depthStencilAttachment, colorAttachments });
                        pass.setPipeline(command.pipeline);
                        pass.setBlendConstant(command.blendConstant);
                        if (depth?.stencilReference !== undefined) pass.setStencilReference(depth.stencilReference);
                        if (command.viewport) pass.setViewport(...command.viewport);
                        if (command.scissor) pass.setScissorRect(...command.scissor);
                        bindResources(pass, command.pipeline, command.bindings);
                        for (const [slot, binding] of command.vertexBuffers.entries()) {
                            pass.setVertexBuffer(slot, binding.buffer, binding.offset);
                        }
                        if (command.count && command.instanceCount) {
                            if (command.index) {
                                const { buffer, format, offset, size, first, baseVertex } = command.index;
                                pass.setIndexBuffer(buffer, format, offset, size);
                                pass.drawIndexed(command.count, command.instanceCount, first, baseVertex, command.firstInstance);
                            } else {
                                pass.draw(command.count, command.instanceCount, command.firstVertex, command.firstInstance);
                            }
                        }
                        pass.end();
                        break;
                    }
                    case "copyTexture": {
                        const { source, destination, bytesPerRow, mipLevel, offset, z, depth } = command;
                        const scale = 2 ** mipLevel;
                        const size = [source.width, source.height].map(axis => Math.max(1, Math.floor(axis / scale)));
                        const aspect = source.format === "depth32float" ? "depth-only" : command.aspect;
                        getEncoder().copyTextureToBuffer({ texture: source, mipLevel, aspect, origin: [0, 0, z] },
                            { buffer: destination, bytesPerRow, offset, rowsPerImage: size[1] }, [...size, depth]);
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
        if (lost && ![2, 4, 5, 9, 11].includes(operation)) throw Object.assign(new Error(lost), { status: 3 });
        switch (operation) {
        case 1: {
            const id = packet.id();
            const size = packet.size();
            const usage = packet.word();
            packet.done();
            if (buffers.has(id)) throw new Error("Duplicate WebGPU buffer");
            const buffer = await withErrors(() => device.createBuffer({ size, usage }));
            buffers.set(id, buffer);
            break;
        }
        case 2: {
            const id = packet.id();
            packet.done();
            buffers.get(id)?.destroy();
            buffers.delete(id);
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
            for (const resources of [buffers, textures]) {
                for (const [id, resource] of resources) {
                    if (id.startsWith(prefix)) { resource.destroy(); resources.delete(id); }
                }
            }
            for (const states of [pipelines, renderPipelines, samplers, textureViews]) {
                for (const id of states.keys()) if (id.startsWith(prefix)) states.delete(id);
            }
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
            const dimension = ["2d", "3d"][packet.word()];
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
            const dimension = [undefined, "2d", "2d-array", "cube", "cube-array", "3d"][dimensionCode];
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
        maxTextureDimension2D: device.limits.maxTextureDimension2D,
        supportsReadableStorage: navigator.gpu.wgslLanguageFeatures?.has("readonly_and_readwrite_storage_textures") ?? false,
        supportsPresentation: context != null,
        adapter: { vendor: adapter.info.vendor, fallback: adapter.info.isFallbackAdapter },
        handle,
        metrics: () => ({
            submissions, remainingResources: buffers.size + textures.size + textureViews.size + samplers.size + pipelines.size + renderPipelines.size,
            presentedFrames, surfaceSize: presentedFrames > 0 ? [canvas.width, canvas.height] : null,
        }),
        close() {
            if (closed) return;
            closed = true;
            if (ownsConfiguration) context.unconfigure();
            device.destroy();
        },
    };
}
