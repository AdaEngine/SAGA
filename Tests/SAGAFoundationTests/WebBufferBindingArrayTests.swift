import Testing
import SAGACompiler
import SAGAShaderIR

@Test func webBufferBindingArraysLowerLengthsAndAtomics() throws {
    let source = ShaderInput.wgsl("""
        enable wgpu_binding_array;
        requires unrestricted_pointer_parameters;
        struct Record { counter: atomic<u32>, tail: array<u32> }
        struct Settings { values: vec4u }
        @group(0) @binding(0) var<storage, read_write> records: binding_array<Record>;
        @group(0) @binding(1) var<uniform> settings: binding_array<Settings, 2>;
        @group(0) @binding(2) var<storage, read_write> output: array<vec4u, 4>;
        fn previous(p: ptr<storage, atomic<u32>, read_write>) -> u32 {
            let compared = atomicCompareExchangeWeak(p, 1u, 2u);
            return compared.old_value;
        }
        @compute @workgroup_size(4) fn run(@builtin(local_invocation_index) lane: u32) {
            let index = i32(lane) - 1;
            let old = previous(&records[index].counter);
            let length = arrayLength(&records[index].tail);
            let first = records[index].tail[0];
            output[lane] = vec4u(old, length, first, settings[index].values.x);
            records[index].tail[0] = settings[index].values.y;
            records[7].tail[0] = 99u;
        }
        """)
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileWebGPUCompute(source, bindingArrayLengths: [.init(group: 0, binding: 0): 2])
    let records = try #require(shader.storageBindings.first { $0.binding.binding == 0 })
    #expect(records.bindingArray?.webBindingBase != nil)
    #expect(records.runtimeArrayStride == 4)
    #expect(records.runtimeArrayOffset == 4)
    #expect(shader.uniformBindings.first?.bindingArray?.webBindingBase != nil)
    guard case .text(.wgsl, let text) = shader.artifact else {
        Issue.record("Expected WGSL")
        return
    }
    #expect(!text.contains("wgpu_binding_array"))
    #expect(text.split(separator: "\n").filter { $0.contains("atomicCompareExchangeWeak(") }.count == 2)
    #expect(!text.contains("= 99u;"))
    _ = try compiler.parseAndValidate(.wgsl(text))
}

@Test func webUniformBufferArraysPreserveAggregateUniformity() throws {
    let source = ShaderInput.wgsl("""
        enable wgpu_binding_array;
        struct Inner { value: vec4u }
        struct Uniform { nested: Inner, flags: array<vec4u, 2> }
        @group(0) @binding(0) var<uniform> settings: binding_array<Uniform, 2>;
        @compute @workgroup_size(4) fn run(@builtin(workgroup_id) group: vec3u) {
            let settingsValue = settings[group.x];
            let row = settings[group.x].flags[group.y];
            if settingsValue.nested.value.x + row.x == 0u { workgroupBarrier(); }
        }
        """)
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileWebGPUCompute(source)
    guard case .text(.wgsl, let text) = shader.artifact else {
        Issue.record("Expected WGSL")
        return
    }
    _ = try compiler.parseAndValidate(.wgsl(text))
}

@Test func webFragmentBufferBindingArraysPreserveCompareExchangeResult() throws {
    let source = ShaderInput.wgsl("""
        enable wgpu_binding_array;
        struct Counters { value: atomic<u32> }
        @group(0) @binding(0) var<storage, read_write> counters: binding_array<Counters, 2>;
        @fragment fn fs(@builtin(position) pixel: vec4f) -> @location(0) vec4u {
            let result = atomicCompareExchangeWeak(&counters[u32(pixel.x)].value, 1u, 2u);
            return vec4u(result.old_value, u32(result.exchanged), 0u, 1u);
        }
        """)
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileWebGPURenderStage(source, entryPoint: "fs")
    guard case .text(.wgsl, let text) = shader.artifact else {
        Issue.record("Expected WGSL")
        return
    }
    _ = try compiler.parseAndValidate(.wgsl(text))
    #expect(text.split(separator: "\n").filter { $0.contains("atomicCompareExchangeWeak(") }.count == 2)
}
