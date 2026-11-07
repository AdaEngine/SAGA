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
    default: throw new Error("Unknown WebGPU texture format");
    }
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

export async function openWebGPU() {
    const adapter = await navigator.gpu?.requestAdapter();
    if (!adapter || adapter.info.isFallbackAdapter) throw new Error("No hardware WebGPU adapter available");
    const device = await adapter.requestDevice();
    const buffers = new Map();
    const textures = new Map();
    const pipelines = new Map();
    const renderPipelines = new Map();
    let submissions = 0;
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

    async function allocatePipeline(packet) {
        const id = packet.id();
        const entryPoint = packet.string();
        const code = packet.string();
        const entries = packet.list(() => {
            const group = packet.word();
            if (group !== 0) throw new Error("Expected WebGPU bind group zero");
            const binding = packet.word();
            const minBindingSize = packet.size();
            const type = packet.word() ? "read-only-storage" : "storage";
            return { binding, visibility: GPUShaderStage.COMPUTE, buffer: { type, minBindingSize } };
        });
        packet.done();
        if (pipelines.has(id)) throw new Error("Duplicate WebGPU pipeline");
        const pipeline = await withErrors(async () => {
            const layout = device.createBindGroupLayout({ entries });
            return await device.createComputePipelineAsync({
                layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
                compute: { module: device.createShaderModule({ code }), entryPoint },
            });
        });
        pipelines.set(id, pipeline);
    }

    async function allocateRenderPipeline(packet) {
        const id = packet.id();
        const format = textureFormat(packet.word());
        const vertex = { entryPoint: packet.string(), code: packet.string() };
        const fragment = { entryPoint: packet.string(), code: packet.string() };
        const vertexBuffers = packet.list(() => ({
            arrayStride: packet.word(),
            attributes: packet.list(() => ({
                shaderLocation: packet.word(), offset: packet.word(), format: vertexFormat(packet.word()),
            })),
        }));
        packet.done();
        if (renderPipelines.has(id)) throw new Error("Duplicate WebGPU render pipeline");
        const pipeline = await withErrors(() => device.createRenderPipelineAsync({
            layout: "auto",
            vertex: { entryPoint: vertex.entryPoint, module: device.createShaderModule({ code: vertex.code }),
                buffers: vertexBuffers },
            fragment: {
                entryPoint: fragment.entryPoint, module: device.createShaderModule({ code: fragment.code }),
                targets: [{ format }],
            },
        }));
        renderPipelines.set(id, pipeline);
    }

    function decodeCommand(packet) {
        switch (packet.word()) {
        case 1: return { type: "write", buffer: lookup(buffers, packet.id()),
            offset: packet.size(), bytes: packet.take(packet.word() * 4) };
        case 2: {
            const pipeline = lookup(pipelines, packet.id());
            const groups = [packet.word(), packet.word(), packet.word()];
            const entries = packet.list(() => {
                if (packet.word() !== 0) throw new Error("Expected WebGPU bind group zero");
                return { binding: packet.word(), resource: { buffer: lookup(buffers, packet.id()) } };
            });
            return { type: "dispatch", pipeline, groups, entries };
        }
        case 3: return { type: "copy", source: lookup(buffers, packet.id()),
            destination: lookup(buffers, packet.id()), size: packet.size() };
        case 4: return { type: "read", buffer: lookup(buffers, packet.id()), size: packet.size() };
        case 5: return { type: "render", pipeline: lookup(renderPipelines, packet.id()),
            target: lookup(textures, packet.id()), count: packet.word(),
            clear: [packet.float(), packet.float(), packet.float(), packet.float()],
            vertexBuffers: packet.list(() => ({ buffer: lookup(buffers, packet.id()), offset: packet.size() })) };
        case 6: return { type: "copyTexture", source: lookup(textures, packet.id()),
            destination: lookup(buffers, packet.id()), bytesPerRow: packet.word() };
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
                        if (command.entries.length) {
                            const layout = command.pipeline.getBindGroupLayout(0);
                            pass.setBindGroup(0, device.createBindGroup({ layout, entries: command.entries }));
                        }
                        pass.dispatchWorkgroups(...command.groups);
                        pass.end();
                        break;
                    }
                    case "copy":
                        getEncoder().copyBufferToBuffer(command.source, 0, command.destination, 0, command.size);
                        break;
                    case "render": {
                        const pass = getEncoder().beginRenderPass({ colorAttachments: [{
                            view: command.target.createView(), loadOp: "clear", storeOp: "store",
                            clearValue: command.clear,
                        }] });
                        pass.setPipeline(command.pipeline);
                        for (const [slot, binding] of command.vertexBuffers.entries()) {
                            pass.setVertexBuffer(slot, binding.buffer, binding.offset);
                        }
                        if (command.count) pass.draw(command.count);
                        pass.end();
                        break;
                    }
                    case "copyTexture":
                        getEncoder().copyTextureToBuffer(
                            { texture: command.source },
                            { buffer: command.destination, bytesPerRow: command.bytesPerRow },
                            [command.source.width, command.source.height],
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
            for (const states of [pipelines, renderPipelines]) {
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
            packet.done();
            if (textures.has(id)) throw new Error("Duplicate WebGPU texture");
            textures.set(id, await withErrors(() => device.createTexture({ size, format, usage })));
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
        default: throw new Error("Unknown browser bridge operation");
        }
        return new Uint8Array();
    }

    return {
        maxBufferSize: device.limits.maxBufferSize,
        maxTextureDimension2D: device.limits.maxTextureDimension2D,
        adapter: { vendor: adapter.info.vendor, fallback: adapter.info.isFallbackAdapter },
        handle,
        metrics: () => ({ submissions, remainingResources: buffers.size + textures.size + pipelines.size + renderPipelines.size }),
        close: () => device.destroy(),
    };
}
