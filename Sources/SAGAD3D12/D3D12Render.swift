#if os(Windows)
import WinSDK
import SAGATypes
import SAGABackend
import SAGAShaderIR

// Render pipelines and draws: one color target, optional vertex and index buffers, buffer/texture/sampler bindings.

/// A CPU-only render target view heap that hands out consecutive slots for one submission.
final class D3D12RenderTargetArena: @unchecked Sendable {
    let heap: COMObject
    let depthHeap: COMObject
    let increment: UInt
    let depthIncrement: UInt
    let start: UInt
    let depthStart: UInt
    private let capacity: Int
    private var used = 0
    private var depthUsed = 0

    init(device: COMObject, capacity: Int = 64) throws {
        func makeHeap(_ type: UInt32) throws -> COMObject {
            typealias CreateHeap = @convention(c) (UnsafeMutableRawPointer, UnsafeRawPointer, UnsafeRawPointer,
                                                   UnsafeMutablePointer<UnsafeMutableRawPointer?>) -> HRESULT
            var description: (UInt32, UInt32, UInt32, UInt32) = (type, UInt32(capacity), 0, 0)
            var iid = IID.descriptorHeap
            var object: UnsafeMutableRawPointer?
            try d3dCheck(device.slot(14, CreateHeap.self)(device.pointer, &description, &iid, &object), "render target heap creation")
            return COMObject(object!)
        }
        depthHeap = try makeHeap(3)
        typealias DepthIncrement = @convention(c) (UnsafeMutableRawPointer, UInt32) -> UInt32
        typealias DepthStart = @convention(c) (UnsafeMutableRawPointer, UnsafeMutableRawPointer) -> UnsafeMutableRawPointer
        depthIncrement = UInt(device.slot(15, DepthIncrement.self)(device.pointer, 3))
        var depthHandle: UInt64 = 0
        _ = depthHeap.slot(9, DepthStart.self)(depthHeap.pointer, &depthHandle)
        depthStart = UInt(depthHandle)
        typealias CreateHeap = @convention(c) (UnsafeMutableRawPointer, UnsafeRawPointer, UnsafeRawPointer,
                                               UnsafeMutablePointer<UnsafeMutableRawPointer?>) -> HRESULT
        typealias Increment = @convention(c) (UnsafeMutableRawPointer, UInt32) -> UInt32
        typealias HeapStart = @convention(c) (UnsafeMutableRawPointer, UnsafeMutableRawPointer) -> UnsafeMutableRawPointer
        var description: (UInt32, UInt32, UInt32, UInt32) = (2, UInt32(capacity), 0, 0)
        var iid = IID.descriptorHeap
        var object: UnsafeMutableRawPointer?
        try d3dCheck(device.slot(14, CreateHeap.self)(device.pointer, &description, &iid, &object), "render target heap creation")
        heap = COMObject(object!)
        self.capacity = capacity
        increment = UInt(device.slot(15, Increment.self)(device.pointer, 2))
        var handle: UInt64 = 0
        _ = heap.slot(9, HeapStart.self)(heap.pointer, &handle)
        start = UInt(handle)
    }

    func allocateDepth() throws -> UInt {
        guard depthUsed < capacity else { throw GPUError.outOfMemory }
        defer { depthUsed += 1 }
        return depthStart + UInt(depthUsed) * depthIncrement
    }

    func allocate() throws -> UInt {
        guard used < capacity else { throw GPUError.outOfMemory }
        defer { used += 1 }
        return start + UInt(used) * increment
    }
}

struct D3D12RenderPipelineState {
    var model: RenderPipeline
    var rootSignature: COMObject
    var state: COMObject
    var topology: UInt32
    var immediateRoot: Int?
    var immediateWords: Int
    var viewRoot: Int?
    var samplerRoot: Int?
    var viewLayout: [(slot: ComputeBindingSlot, storage: Bool)]
    var samplerLayout: [ComputeBindingSlot]
    /// Storage buffer array elements, held as raw views after the texture views in the view table.
    var bufferViewLayout: [(slot: ComputeBindingSlot, readOnly: Bool)]
    /// Root descriptor storage buffers with their root parameter indices.
    var rootStorage: [(binding: RenderPipeline.StorageBufferBinding, root: Int)]
    /// Root constants that carry the first vertex and first instance when shader model 6.8 is unavailable.
    var drawParametersRoot: Int?
    /// External texture layouts of both stages with the root parameter of each parameter-word constant buffer.
    var externalLayouts: [(layout: ComputeShaderArtifact.ExternalTextureLayout, root: Int)] = []
    /// Acceleration structure bindings with the root parameter of each root SRV.
    var accelerationRoots: [(slot: ComputeBindingSlot, root: Int)] = []
}

func d3d12VertexFormat(_ format: VertexFormat) -> UInt32? {
    switch format {
    case .float32: 41
    case .float32x2: 16
    case .float32x3: 6
    case .float32x4: 2
    case .uint32: 42
    case .uint32x2: 17
    case .uint32x3: 7
    case .uint32x4: 3
    case .sint32: 43
    case .sint32x2: 18
    case .sint32x3: 8
    case .sint32x4: 4
    case .float16x2: 34
    case .float16x4: 10
    case .unorm8x2: 49
    case .unorm8x4: 28
    case .snorm8x2: 51
    case .snorm8x4: 31
    case .uint8x2: 50
    case .uint8x4: 30
    case .sint8x2: 52
    case .sint8x4: 32
    case .uint16x2: 36
    case .uint16x4: 12
    case .sint16x2: 38
    case .sint16x4: 14
    case .unorm16x2: 35
    case .unorm16x4: 11
    case .snorm16x2: 37
    case .snorm16x4: 13
    default: nil
    }
}

func d3d12BlendFactor(_ factor: RenderBlendFactor) -> UInt32 {
    switch factor {
    case .zero: 1
    case .one: 2
    case .sourceColor: 3
    case .oneMinusSourceColor: 4
    case .sourceAlpha: 5
    case .oneMinusSourceAlpha: 6
    case .destinationAlpha: 7
    case .oneMinusDestinationAlpha: 8
    case .destinationColor: 9
    case .oneMinusDestinationColor: 10
    case .sourceAlphaSaturated: 11
    case .constant: 14
    case .oneMinusConstant: 15
    case .source1Color: 16
    case .oneMinusSource1Color: 17
    case .source1Alpha: 18
    case .oneMinusSource1Alpha: 19
    }
}

extension D3D12Device {
    func allocateRenderPipelineNative(_ pipeline: RenderPipeline, vertex: RenderStageArtifact,
                                      fragment: RenderStageArtifact) throws -> D3D12RenderPipelineState {
        guard case .text(.hlsl, let vertexSource) = vertex.artifact, case .text(.hlsl, let fragmentSource) = fragment.artifact,
              let vertexProfile = vertex.hlslProfile, let fragmentProfile = fragment.hlslProfile else {
            throw GPUError.validation("Direct3D 12 requires HLSL render stages")
        }
        guard let colorFormat = pipeline.colorFormat, let target = d3d12Format(colorFormat) else {
            throw GPUError.unsupported("Direct3D 12 render pipelines require a supported color format")
        }
        var depthFormat: UInt32 = 0
        if let depthState = pipeline.depthState {
            guard let formats = d3d12DepthFormats(depthState.format) else {
                throw GPUError.unsupported("Direct3D 12 does not implement depth format \(depthState.format)")
            }
            depthFormat = formats.dsv
        }
        guard pipeline.multiviewMask == nil,
              pipeline.additionalColorTargets.isEmpty, !pipeline.alphaToCoverageEnabled, !pipeline.primitive.unclippedDepth,
              !pipeline.colorState.usesSecondBlendSource,
              pipeline.uniforms.allSatisfy({ $0.slot.arrayElement == nil }),
              pipeline.sampledTextures.allSatisfy({ !$0.requirement.isMultisampled }) else {
            throw GPUError.unsupported("Direct3D 12 render pipelines support one single-sampled color target without depth in this release")
        }
        if compiler == nil { compiler = try D3D12ShaderCompiler(directories: compilerDirectories) }
        // Shader model 6.8 needs a newer DXIL validator than many machines have. When it is missing, the draw
        // base values move from system values into root constants, which any validator accepts.
        var drawParametersRoot: Int?
        var vertexCode: [UInt8]
        do {
            vertexCode = try compiler!.compile(vertexSource, entryPoint: vertex.entryPoint, profile: vertexProfile)
        } catch let error as GPUError {
            guard vertexProfile == "vs_6_8", case .backendFailure(let message) = error,
                  message.contains("does not support target profile") else { throw error }
            let rewritten = vertexSource
                .replacingAll(", int saga_first_vertex : SV_StartVertexLocation", with: "")
                .replacingAll(", uint saga_first_instance : SV_StartInstanceLocation", with: "")
                .replacingAll("asuint(saga_first_vertex)", with: "saga_draw_first_vertex")
                .replacingAll("saga_first_vertex", with: "int(saga_draw_first_vertex)")
                .replacingAll("saga_first_instance", with: "saga_draw_first_instance")
            let declaration = "cbuffer SagaDrawParameters : register(b15, space5) { uint saga_draw_first_vertex; uint saga_draw_first_instance; };\n"
            vertexCode = try compiler!.compile(declaration + rewritten, entryPoint: vertex.entryPoint, profile: "vs_6_0")
            drawParametersRoot = -1
        }
        let fragmentCode = try compiler!.compile(fragmentSource, entryPoint: fragment.entryPoint, profile: fragmentProfile)

        // Root signature: buffers as root descriptors, immediates as constants, textures and samplers as tables.
        var parameters: [D3D12RootParameter] = []
        for uniform in pipeline.uniforms {
            parameters.append(D3D12RootParameter(parameterType: D3D12.rootParameterCBV, shaderRegister: uniform.slot.binding,
                                                 registerSpace: uniform.slot.group))
        }
        var rootStorage: [(binding: RenderPipeline.StorageBufferBinding, root: Int)] = []
        var arrayBuffers: [(slot: ComputeBindingSlot, readOnly: Bool)] = []
        for storage in pipeline.storageBuffers {
            if storage.handle.slot.arrayElement != nil {
                arrayBuffers.append((storage.handle.slot, storage.isReadOnly))
                continue
            }
            rootStorage.append((storage, parameters.count))
            parameters.append(D3D12RootParameter(parameterType: storage.isReadOnly ? D3D12.rootParameterSRV : D3D12.rootParameterUAV,
                                                 shaderRegister: storage.handle.slot.binding, registerSpace: storage.handle.slot.group))
        }
        var immediateRoot: Int?
        let immediateWords = Int((pipeline.immediateByteSize + 3) / 4)
        if immediateWords > 0 {
            guard let binding = vertex.immediateLayout?.hlslBinding ?? fragment.immediateLayout?.hlslBinding else {
                throw GPUError.unsupported("Direct3D 12 render stages need an HLSL immediate binding")
            }
            immediateRoot = parameters.count
            parameters.append(D3D12RootParameter(parameterType: D3D12.rootParameterConstants, shaderRegister: binding.binding,
                                                 registerSpace: binding.group, unionTail: UInt64(immediateWords)))
        }
        if drawParametersRoot != nil {
            drawParametersRoot = parameters.count
            parameters.append(D3D12RootParameter(parameterType: D3D12.rootParameterConstants, shaderRegister: 15, registerSpace: 5,
                                                 unionTail: 2))
        }
        var accelerationRoots: [(slot: ComputeBindingSlot, root: Int)] = []
        for binding in pipeline.accelerationStructureBindings {
            accelerationRoots.append((binding.slot, parameters.count))
            parameters.append(D3D12RootParameter(parameterType: D3D12.rootParameterSRV, shaderRegister: binding.slot.binding,
                                                 registerSpace: binding.slot.group))
        }
        var externalLayouts: [(layout: ComputeShaderArtifact.ExternalTextureLayout, root: Int)] = []
        for layout in vertex.externalTextureLayouts + fragment.externalTextureLayouts
        where !externalLayouts.contains(where: { $0.layout.binding == layout.binding }) {
            externalLayouts.append((layout, parameters.count))
            parameters.append(D3D12RootParameter(parameterType: D3D12.rootParameterCBV, shaderRegister: layout.parametersBindingIndex,
                                                 registerSpace: layout.binding.group))
        }
        // Binding arrays declare their first register in the artifact of the stage that uses them.
        var arrayInfo: [IRResourceBinding: (count: UInt32, base: UInt32)] = [:]
        for stage in [vertex, fragment] {
            for binding in stage.storageBindings { if let a = binding.bindingArray, let base = a.hlslRegisterBase { arrayInfo[binding.binding] = (a.count, base) } }
            for binding in stage.sampledTextures { if let a = binding.bindingArray, let base = a.hlslRegisterBase { arrayInfo[binding.binding] = (a.count, base) } }
            for binding in stage.storageTextures { if let a = binding.bindingArray, let base = a.hlslRegisterBase { arrayInfo[binding.binding] = (a.count, base) } }
            for binding in stage.samplerBindings { if let a = binding.bindingArray, let base = a.hlslRegisterBase { arrayInfo[binding.binding] = (a.count, base) } }
        }
        func ordered(_ slots: [ComputeBindingSlot]) -> [ComputeBindingSlot] {
            slots.sorted { ($0.group, $0.binding, $0.arrayElement ?? 0) < ($1.group, $1.binding, $1.arrayElement ?? 0) }
        }
        func addRanges(_ slots: [ComputeBindingSlot], type: UInt32, to ranges: inout [D3D12DescriptorRange], used: inout UInt32) {
            var seen = Set<IRResourceBinding>()
            for slot in ordered(slots) {
                if slot.arrayElement == nil {
                    ranges.append(D3D12DescriptorRange(rangeType: type, baseShaderRegister: slot.binding, registerSpace: slot.group,
                                                       offsetInDescriptors: used))
                    used += 1
                } else {
                    let key = IRResourceBinding(group: slot.group, binding: slot.binding)
                    guard seen.insert(key).inserted, let info = arrayInfo[key] else { continue }
                    ranges.append(D3D12DescriptorRange(rangeType: type, descriptorCount: info.count, baseShaderRegister: info.base,
                                                       registerSpace: slot.group, offsetInDescriptors: used))
                    used += info.count
                }
            }
        }
        var viewLayout: [(slot: ComputeBindingSlot, storage: Bool)] = []
        var viewRanges: [D3D12DescriptorRange] = [], samplerRanges: [D3D12DescriptorRange] = []
        var viewUsed: UInt32 = 0, samplerUsed: UInt32 = 0
        addRanges(pipeline.sampledTextures.map(\.handle.slot), type: 0, to: &viewRanges, used: &viewUsed)
        viewLayout += ordered(pipeline.sampledTextures.map(\.handle.slot)).map { ($0, false) }
        addRanges(pipeline.storageTextures.map(\.handle.slot), type: 1, to: &viewRanges, used: &viewUsed)
        viewLayout += ordered(pipeline.storageTextures.map(\.handle.slot)).map { ($0, true) }
        let readArrays = arrayBuffers.filter(\.readOnly).map(\.slot), writeArrays = arrayBuffers.filter { !$0.readOnly }.map(\.slot)
        addRanges(readArrays, type: 0, to: &viewRanges, used: &viewUsed)
        addRanges(writeArrays, type: 1, to: &viewRanges, used: &viewUsed)
        let bufferViewLayout = ordered(readArrays).map { ($0, true) } + ordered(writeArrays).map { ($0, false) }
        // Each external texture takes three consecutive SRVs after every other view.
        for entry in externalLayouts {
            for index in entry.layout.planeBindingIndices {
                viewRanges.append(D3D12DescriptorRange(rangeType: 0, baseShaderRegister: index, registerSpace: entry.layout.binding.group,
                                                       offsetInDescriptors: viewUsed))
                viewUsed += 1
            }
        }
        addRanges(pipeline.samplerBindings.map(\.handle.slot), type: 3, to: &samplerRanges, used: &samplerUsed)
        let samplerLayout = ordered(pipeline.samplerBindings.map(\.handle.slot))
        let viewMemory = UnsafeMutablePointer<D3D12DescriptorRange>.allocate(capacity: max(1, viewRanges.count))
        let samplerMemory = UnsafeMutablePointer<D3D12DescriptorRange>.allocate(capacity: max(1, samplerRanges.count))
        defer { viewMemory.deallocate(); samplerMemory.deallocate() }
        var viewRoot: Int?, samplerRoot: Int?
        if !viewRanges.isEmpty {
            viewMemory.initialize(from: viewRanges, count: viewRanges.count)
            viewRoot = parameters.count
            parameters.append(D3D12RootParameter(parameterType: 0, shaderRegister: UInt32(viewRanges.count),
                                                 unionTail: UInt64(UInt(bitPattern: viewMemory))))
        }
        if !samplerRanges.isEmpty {
            samplerMemory.initialize(from: samplerRanges, count: samplerRanges.count)
            samplerRoot = parameters.count
            parameters.append(D3D12RootParameter(parameterType: 0, shaderRegister: UInt32(samplerRanges.count),
                                                 unionTail: UInt64(UInt(bitPattern: samplerMemory))))
        }
        let (serialized, blob, errorBlob) = Self.serialize(library, parameters: &parameters, flags: 0x1)
        withExtendedLifetime(errorBlob) {}
        try d3dCheck(serialized, "root signature serialization")
        let root = try makeRootSignature(blob!)

        // Input layout. The HLSL emitter names every vertex input TEXCOORD<location>.
        let semantic = UnsafeMutablePointer<CChar>.allocate(capacity: 16)
        "TEXCOORD".withCString { semantic.initialize(from: $0, count: 9) }
        defer { semantic.deallocate() }
        var elements: [UInt32] = []  // eight words per element: name pointer (two words), then six fields
        for (slot, layout) in pipeline.vertexBuffers.enumerated() {
            for attribute in layout.attributes {
                guard let format = d3d12VertexFormat(attribute.format) else {
                    throw GPUError.unsupported("Direct3D 12 does not implement vertex format \(attribute.format)")
                }
                let pointerBits = UInt(bitPattern: semantic)
                elements += [UInt32(truncatingIfNeeded: pointerBits), UInt32(truncatingIfNeeded: UInt64(pointerBits) >> 32),
                             attribute.location, format, UInt32(slot), attribute.offset,
                             layout.stepMode == .instance ? 1 : 0, layout.stepMode == .instance ? 1 : 0]
            }
        }

        // D3D12_GRAPHICS_PIPELINE_STATE_DESC, written field by field at its x64 offsets.
        let size = 656
        let desc = UnsafeMutableRawPointer.allocate(byteCount: size, alignment: 8)
        defer { desc.deallocate() }
        desc.initializeMemory(as: UInt8.self, repeating: 0, count: size)
        func put32(_ offset: Int, _ value: UInt32) { desc.storeBytes(of: value, toByteOffset: offset, as: UInt32.self) }
        func putPointer(_ offset: Int, _ value: UnsafeRawPointer?) {
            desc.storeBytes(of: UInt(bitPattern: value), toByteOffset: offset, as: UInt.self)
        }
        func put64(_ offset: Int, _ value: Int) { desc.storeBytes(of: value, toByteOffset: offset, as: Int.self) }
        putPointer(0, UnsafeRawPointer(root.pointer))
        let state: COMObject = try vertexCode.withUnsafeBytes { vertexBytes in
            try fragmentCode.withUnsafeBytes { fragmentBytes in
                try elements.withUnsafeBytes { elementBytes in
                    putPointer(8, vertexBytes.baseAddress); put64(16, vertexBytes.count)
                    putPointer(24, fragmentBytes.baseAddress); put64(32, fragmentBytes.count)
                    Self.writeFixedFunctionState(desc, pipeline: pipeline, target: target, depthFormat: depthFormat)
                    putPointer(552, elementBytes.baseAddress)
                    put32(560, UInt32(elements.count / 8))
                    let topologyType: UInt32 = switch pipeline.primitive.topology {
                    case .pointList: 1
                    case .lineList, .lineStrip: 2
                    case .triangleList, .triangleStrip: 3
                    }
                    put32(572, topologyType)
                    return try Self.createGraphicsState(device, description: desc,
                                                        create: device.slot(10, CreateGraphics.self))
                }
            }
        }
        let topology: UInt32 = switch pipeline.primitive.topology {
        case .pointList: 1
        case .lineList: 2
        case .lineStrip: 3
        case .triangleList: 4
        case .triangleStrip: 5
        }
        return D3D12RenderPipelineState(model: pipeline, rootSignature: root, state: state, topology: topology,
                                        immediateRoot: immediateRoot, immediateWords: immediateWords, viewRoot: viewRoot,
                                        samplerRoot: samplerRoot, viewLayout: viewLayout,
                                        samplerLayout: samplerLayout, bufferViewLayout: bufferViewLayout,
                                        rootStorage: rootStorage,
                                        drawParametersRoot: drawParametersRoot, externalLayouts: externalLayouts,
                                        accelerationRoots: accelerationRoots)
    }

    /// Writes blend, rasterizer, depth-stencil and target-format state of a graphics pipeline description at its x64 offsets.
    nonisolated static func writeFixedFunctionState(
        _ desc: UnsafeMutableRawPointer, pipeline: RenderPipeline, target: UInt32, depthFormat: UInt32
    ) {
        func put32(_ offset: Int, _ value: UInt32) { desc.storeBytes(of: value, toByteOffset: offset, as: UInt32.self) }
        // Blend state: one render target at offset 128.
        if let blend = pipeline.colorState.blend {
            put32(128, 1)
            put32(136, d3d12BlendFactor(blend.color.sourceFactor))
            put32(140, d3d12BlendFactor(blend.color.destinationFactor))
            put32(144, blend.color.operation.rawValue + 1)
            put32(148, d3d12BlendFactor(blend.alpha.sourceFactor))
            put32(152, d3d12BlendFactor(blend.alpha.destinationFactor))
            put32(156, blend.alpha.operation.rawValue + 1)
        } else {
            put32(136, 2); put32(140, 1); put32(144, 1); put32(148, 2); put32(152, 1); put32(156, 1)
        }
        put32(160, 4)  // logic op: no-op
        desc.storeBytes(of: UInt8(pipeline.colorState.writeMask.rawValue & 0xF), toByteOffset: 164, as: UInt8.self)
        put32(448, pipeline.sampleMask)
        // Rasterizer state.
        put32(452, 3)  // solid
        let cull: UInt32 = switch pipeline.primitive.cullMode { case .none: 1; case .front: 2; case .back: 3 }
        put32(456, cull)
        put32(460, pipeline.primitive.frontFace == .counterClockwise ? 1 : 0)
        put32(476, 1)  // depth clip
        if let depthState = pipeline.depthState {
            put32(496, depthState.format.hasDepth ? 1 : 0)
            put32(500, depthState.writeEnabled ? 1 : 0)
            put32(504, depthState.compare.rawValue + 1)
            if depthState.bias.constant != 0 || depthState.bias.slopeScale != 0 {
                desc.storeBytes(of: depthState.bias.constant, toByteOffset: 464, as: Int32.self)
                desc.storeBytes(of: depthState.bias.clamp, toByteOffset: 468, as: Float.self)
                desc.storeBytes(of: depthState.bias.slopeScale, toByteOffset: 472, as: Float.self)
            }
            if let stencil = depthState.stencil {
                func operation(_ value: RenderStencilOperation) -> UInt32 {
                    switch value {
                    case .keep: 1
                    case .zero: 2
                    case .replace: 3
                    case .incrementClamp: 4
                    case .decrementClamp: 5
                    case .invert: 6
                    case .incrementWrap: 7
                    case .decrementWrap: 8
                    }
                }
                put32(508, 1)
                desc.storeBytes(of: UInt8(truncatingIfNeeded: stencil.readMask), toByteOffset: 512, as: UInt8.self)
                desc.storeBytes(of: UInt8(truncatingIfNeeded: stencil.writeMask), toByteOffset: 513, as: UInt8.self)
                for (offset, face) in [(516, stencil.front), (532, stencil.back)] {
                    put32(offset, operation(face.fail))
                    put32(offset + 4, operation(face.depthFail))
                    put32(offset + 8, operation(face.pass))
                    put32(offset + 12, face.compare.rawValue + 1)
                }
            } else {
                // Stencil disabled: keep the face states at their no-op defaults.
                for offset in [516, 532] { put32(offset, 1); put32(offset + 4, 1); put32(offset + 8, 1); put32(offset + 12, 8) }
            }
        }
        put32(576, 1)
        put32(580, target)
        put32(612, depthFormat)
        put32(616, pipeline.sampleCount)
    }

    typealias CreateGraphics = @convention(c) (UnsafeMutableRawPointer, UnsafeRawPointer, UnsafeRawPointer,
                                               UnsafeMutablePointer<UnsafeMutableRawPointer?>) -> HRESULT

    nonisolated static func createGraphicsState(
        _ device: COMObject, description: UnsafeMutableRawPointer,
        create: @convention(c) (UnsafeMutableRawPointer, UnsafeRawPointer, UnsafeRawPointer, UnsafeMutablePointer<UnsafeMutableRawPointer?>) -> HRESULT
    ) throws -> COMObject {
        var iid = IID.pipelineState
        var object: UnsafeMutableRawPointer?
        try d3dCheck(create(device.pointer, description, &iid, &object), "graphics pipeline creation")
        return COMObject(object!)
    }
}
#endif

#if os(Windows)
extension String {
    func replacingAll(_ target: String, with replacement: String) -> String {
        var result = ""
        var remaining = self[...]
        while let range = remaining.range(of: target) {
            result += remaining[remaining.startIndex..<range.lowerBound] + replacement
            remaining = remaining[range.upperBound...]
        }
        return result + remaining
    }
}

private extension Substring {
    func range(of target: String) -> Range<Index>? {
        guard !target.isEmpty, count >= target.count else { return nil }
        var index = startIndex
        while index < endIndex {
            if self[index...].hasPrefix(target) { return index..<self.index(index, offsetBy: target.count) }
            index = self.index(after: index)
        }
        return nil
    }
}
#endif
