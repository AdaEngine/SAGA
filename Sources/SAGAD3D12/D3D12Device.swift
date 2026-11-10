#if os(Windows)
import WinSDK
import SAGATypes
import SAGABackend
import SAGAShaderIR
import SAGACompiler

/// Direct3D 12 device creation. The runtime comes from d3d12.dll and the shader compiler from dxcompiler.dll.
public enum D3D12Backend {
    /// True when d3d12.dll can create a Direct3D 12 device on the default adapter.
    public static func isAvailable() -> Bool {
        (try? D3D12Device.makeNativeDevice()) != nil
    }

    /// Creates a device on the default hardware adapter.
    /// - Parameter shaderCompilerDirectories: Folders that hold `dxcompiler.dll` and `dxil.dll`, searched before the
    ///   normal DLL path. The `SAGA_DXC_DIRECTORIES` environment variable (semicolon separated) is added.
    public static func makeDevice(
        timeout: Duration = .seconds(10), shaderCompilerDirectories: [String] = [], windowHandle: UInt? = nil
    ) throws -> D3D12Device {
        try D3D12Device(timeout: timeout, shaderCompilerDirectories: shaderCompilerDirectories, windowHandle: windowHandle)
    }
}

/// A compute-capable Direct3D 12 device: buffers, copies, direct dispatch and readback.
public actor D3D12Device: ComputeBackendDevice, PlanarExternalTextureBackendDevice {
    public nonisolated let capabilities: DeviceCapabilities
    public nonisolated let supportsPresentation: Bool
    var presenter: D3D12Presenter?

    struct Buffer {
        var descriptor: BufferDescriptor
        var resource: COMObject
        var address: UInt64
    }

    /// An external texture: its plane resources and an upload-heap constant buffer holding the parameter words.
    struct ExternalTexture {
        var planes: [ResourceID]
        var parameters: COMObject
        var address: UInt64
    }

    struct Pipeline {
        var model: ComputePipeline
        /// Acceleration structure bindings with the root parameter of each root SRV.
        var accelerationRoots: [(slot: ComputeBindingSlot, root: Int)] = []
        /// External texture layouts with the root parameter of each constant buffer of parameter words.
        var externalLayouts: [(layout: ComputeShaderArtifact.ExternalTextureLayout, root: Int)] = []
        var rootSignature: COMObject
        var state: COMObject
        /// Root parameter indices of the 32-bit constant blocks, when the shader uses them.
        var immediateRoot: Int?
        var countRoot: Int?
        var immediateWords = 0
        var viewRoot: Int?
        var samplerRoot: Int?
        var viewLayout: [(slot: ComputeBindingSlot, storage: Bool)] = []
        var samplerLayout: [ComputeBindingSlot] = []
        /// Buffers that are not root descriptors: elements of binding arrays, held as raw views in the view table.
        var bufferViewLayout: [(slot: ComputeBindingSlot, readOnly: Bool)] = []
        /// Root descriptor buffers with the root parameter index of each.
        var rootBuffers: [(slot: ComputeBindingSlot, usage: BufferUsage, isReadOnly: Bool, root: Int)] = []
    }

    /// Objects that must outlive the GPU work of one submission.
    struct Retained {
        var objects: [COMObject] = []
    }

    let library: D3D12Library
    let device: COMObject
    let queue: COMObject
    let fence: COMObject
    let timeout: Duration
    let compilerDirectories: [String]
    var compiler: D3D12ShaderCompiler?
    var validator: Pipeline?
    var meshValidator: Pipeline?
    var meshCommandSignature: COMObject?
    var dispatchCommandSignature: COMObject?
    var drawCommandSignature: COMObject?
    var drawIndexedCommandSignature: COMObject?
    var fenceValue: UInt64 = 0
    var buffers: [ResourceID: Buffer] = [:]
    var pipelines: [ResourceID: Pipeline] = [:]
    var mappings: [ResourceID: Mapping] = [:]
    var querySets: [ResourceID: D3D12QuerySet] = [:]
    var textures: [ResourceID: Texture] = [:]
    var textureViews: [ResourceID: TextureViewState] = [:]
    var renderPipelines: [ResourceID: D3D12RenderPipelineState] = [:]
    var samplers: [ResourceID: SamplerDescriptor] = [:]
    var externalTextures: [ResourceID: ExternalTexture] = [:]
    /// Mesh pipelines (kept in `renderPipelines`), each flagged with whether it has a task stage.
    var meshPipelines: [ResourceID: Bool] = [:]
    var accelerationStructures: [ResourceID: AccelerationStructureState] = [:]
    var indexSanitizer: Pipeline?
    var terminal = false
    var quarantine: [Retained] = []

    static func makeNativeDevice() throws -> (D3D12Library, COMObject) {
        guard let library = D3D12Library.load() else { throw GPUError.adapterUnavailable }
        D3D12DebugLayer.enableIfRequested()
        var iid = IID.device
        var object: UnsafeMutableRawPointer?
        let result = library.createDevice(nil, D3D12.featureLevel12_0, &iid, &object)
        guard result >= 0, let object else { throw GPUError.adapterUnavailable }
        let native = COMObject(object)
        D3D12DebugLayer.device = native
        return (library, native)
    }

    init(timeout: Duration, shaderCompilerDirectories: [String], windowHandle: UInt? = nil) throws {
        (library, device) = try Self.makeNativeDevice()
        self.timeout = timeout
        supportsPresentation = windowHandle != nil
        if let windowHandle { presenter = try D3D12Presenter(window: windowHandle) }
        var directories = shaderCompilerDirectories
        if let value = environmentValue("SAGA_DXC_DIRECTORIES") {
            directories += value.split(separator: ";").map(String.init)
        }
        compilerDirectories = directories

        typealias CreateQueue = @convention(c) (UnsafeMutableRawPointer, UnsafeRawPointer, UnsafeRawPointer,
                                                UnsafeMutablePointer<UnsafeMutableRawPointer?>) -> HRESULT
        var queueDesc = D3D12CommandQueueDesc()
        var queueIID = IID.commandQueue
        var queueObject: UnsafeMutableRawPointer?
        try d3dCheck(device.slot(8, CreateQueue.self)(device.pointer, &queueDesc, &queueIID, &queueObject), "queue creation")
        queue = COMObject(queueObject!)

        typealias CreateFence = @convention(c) (UnsafeMutableRawPointer, UInt64, UInt32, UnsafeRawPointer,
                                                UnsafeMutablePointer<UnsafeMutableRawPointer?>) -> HRESULT
        var fenceIID = IID.fence
        var fenceObject: UnsafeMutableRawPointer?
        try d3dCheck(device.slot(36, CreateFence.self)(device.pointer, 0, 0, &fenceIID, &fenceObject), "fence creation")
        fence = COMObject(fenceObject!)

        typealias Frequency = @convention(c) (UnsafeMutableRawPointer, UnsafeMutablePointer<UInt64>) -> HRESULT
        var frequency: UInt64 = 0
        let hasTimestamps = queue.slot(16, Frequency.self)(queue.pointer, &frequency) >= 0 && frequency > 0
        let meshSupported = Self.supportsMeshShaders(device)
        var features: Set<GPUFeature> = []
        if hasTimestamps { features.insert(.timestampQuery) }
        if meshSupported { features.insert(.meshShaders) }
        if Self.supportsRayQueries(device) { features.insert(.rayQuery) }
        // The HLSL cooperative matrix lowering spreads a tile over a 32-lane wave with wave intrinsics.
        let cooperativeSupported = Self.hasFixedWaveSize32(device)
        var cooperativeProperties: [CooperativeMatrixProperties] = []
        if cooperativeSupported {
            features.insert(.cooperativeMatrix)
            features.formUnion([.subgroups, .subgroupSizeControl])
            for m: UInt32 in [8, 16] {
                for n: UInt32 in [8, 16] {
                    for k: UInt32 in [8, 16] {
                        cooperativeProperties.append(.init(mSize: m, nSize: n, kSize: k, inputType: .float32, accumulatorType: .float32))
                    }
                }
            }
        }
        capabilities = try DeviceCapabilities(
            maxBufferSize: 1 << 30,
            features: features,
            maxTextureDimension2D: 16_384,
            timestampPeriodNanoseconds: hasTimestamps ? 1e9 / Double(frequency) : 1,
            maxImmediateSize: 64,
            subgroupSizeRange: cooperativeSupported ? 32...32 : nil,
            maxComputeWorkgroupStorageSize: 32 * 1024,
            maxComputeInvocationsPerWorkgroup: 1024,
            maxComputeWorkgroupSize: DispatchWorkgroups(x: 1024, y: 1024, z: 64),
            maxComputeWorkgroupsPerDimension: 65_535,
            cooperativeMatrixProperties: cooperativeProperties,
            maxTaskWorkgroupTotalCount: meshSupported ? 4_194_304 : nil,
            maxTaskWorkgroupsPerDimension: meshSupported ? 65_535 : nil,
            maxMeshWorkgroupTotalCount: meshSupported ? 4_194_304 : nil,
            maxMeshWorkgroupsPerDimension: meshSupported ? 65_535 : nil,
            maxTaskInvocationsPerWorkgroup: meshSupported ? 128 : nil,
            maxTaskInvocationsPerDimension: meshSupported ? 128 : nil,
            maxMeshInvocationsPerWorkgroup: meshSupported ? 128 : nil,
            maxMeshInvocationsPerDimension: meshSupported ? 128 : nil,
            maxTaskPayloadSize: meshSupported ? 16_384 : nil,
            maxMeshOutputVertices: 256,
            maxMeshOutputPrimitives: 256)
    }

    func requireActive() throws {
        guard !terminal else { throw GPUError.deviceLost("Direct3D 12 device is terminal after a command failure or timeout") }
    }

    // MARK: Buffers

    public func allocateBuffer(id: ResourceID, descriptor: BufferDescriptor) throws {
        try requireActive()
        try descriptor.validate(against: capabilities)
        guard buffers[id] == nil else { throw GPUError.validation("Duplicate Direct3D 12 buffer") }
        let resource = try makeBuffer(size: descriptor.size, heap: D3D12.heapDefault, state: D3D12.stateCommon,
                                      flags: D3D12.flagAllowUnorderedAccess)
        buffers[id] = Buffer(descriptor: descriptor, resource: resource, address: gpuAddress(resource))
        if descriptor.mappedAtCreation { mappings[id] = Mapping(mode: .write) }
    }

    public func destroyBuffer(id: ResourceID) { buffers[id] = nil }

    public func allocatePlanarExternalTexture(id: ResourceID, descriptor: PlanarExternalTextureDescriptor) throws {
        try requireActive()
        guard externalTextures[id] == nil, descriptor.planes.count == descriptor.format.planeComponents.count else {
            throw GPUError.validation("Invalid Direct3D 12 external texture descriptor")
        }
        var first: (width: UInt32, height: UInt32)?
        for (index, plane) in descriptor.planes.enumerated() {
            let view = textureViews[plane]
            guard plane.device == id.device, let texture = textures[view?.texture ?? plane] else {
                throw GPUError.validation("Unknown Direct3D 12 external texture plane")
            }
            let resource = view.map { TextureViewResource(texture: $0.texture, levels: $0.levels, aspect: $0.aspect, layers: $0.layers, dimension: $0.dimension) }
            let extent = try descriptor.planeExtent(at: index, texture: texture.descriptor, view: resource, capabilities: capabilities)
            if index == 0 { first = extent }
        }
        guard let first else { throw GPUError.validation("External texture requires a first plane") }
        let words = try descriptor.parameterWords(planeWidth: first.width, planeHeight: first.height)
        let size = max(UInt64(ComputeShaderArtifact.ExternalTextureLayout.parameterByteSize), UInt64(words.count * 4))
        let buffer = try makeBuffer(size: size, heap: D3D12.heapUpload, state: D3D12.stateGenericRead)
        try map(buffer, count: words.count * 4) { destination in
            words.withUnsafeBytes { destination.copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
        }
        externalTextures[id] = ExternalTexture(planes: descriptor.planes, parameters: buffer, address: gpuAddress(buffer))
    }

    public func destroyExternalTexture(id: ResourceID) { externalTextures[id] = nil }

    public func close(owner: UInt64) {
        for id in buffers.keys where id.device == owner { buffers[id] = nil }
        for id in textures.keys where id.device == owner { textures[id] = nil }
        for id in samplers.keys where id.device == owner { samplers[id] = nil }
        for id in externalTextures.keys where id.device == owner { externalTextures[id] = nil }
        for id in accelerationStructures.keys where id.device == owner { accelerationStructures[id] = nil }
        for id in querySets.keys where id.device == owner { querySets[id] = nil }
        for id in pipelines.keys where id.device == owner { pipelines[id] = nil }
        for id in renderPipelines.keys where id.device == owner { renderPipelines[id] = nil; meshPipelines[id] = nil }
    }

    func makeBuffer(size: UInt64, heap: UInt32, state: UInt32, flags: UInt32 = 0) throws -> COMObject {
        typealias CreateCommitted = @convention(c) (
            UnsafeMutableRawPointer, UnsafeRawPointer, UInt32, UnsafeRawPointer, UInt32,
            UnsafeRawPointer?, UnsafeRawPointer, UnsafeMutablePointer<UnsafeMutableRawPointer?>) -> HRESULT
        var properties = D3D12HeapProperties(type: heap)
        var description = D3D12ResourceDesc()
        description.width = max(4, (size + 3) & ~3)
        description.flags = flags
        var iid = IID.resource
        var object: UnsafeMutableRawPointer?
        try d3dCheck(device.slot(27, CreateCommitted.self)(device.pointer, &properties, 0, &description, state, nil, &iid, &object),
                     "buffer creation")
        return COMObject(object!)
    }

    func gpuAddress(_ resource: COMObject) -> UInt64 {
        typealias GetAddress = @convention(c) (UnsafeMutableRawPointer) -> UInt64
        return resource.slot(11, GetAddress.self)(resource.pointer)
    }

    // MARK: Compute pipelines

    public func allocateComputePipeline(_ pipeline: ComputePipeline, shader: ComputeShaderArtifact) throws {
        try requireActive()
        guard case .text(.hlsl, let source) = shader.artifact, let profile = shader.hlslProfile else {
            throw GPUError.validation("Direct3D 12 requires a compiled HLSL compute shader")
        }
        let reflected = try ComputePipeline(id: pipeline.id, shader: shader, nonFilteringSamplers: pipeline.nonFilteringSamplers)
        guard pipeline == reflected else {
            throw GPUError.validation("Direct3D 12 compute pipeline layout does not match compiled shader")
        }
        guard pipelines[pipeline.id] == nil else { throw GPUError.validation("Duplicate Direct3D 12 compute pipeline") }
        guard shader.externalTextureLayouts.count == pipeline.externalTextureBindings.count,
              pipeline.immediateByteSize == 0 || shader.immediateLayout?.hlslBinding != nil,
              pipeline.textureBindings.allSatisfy({ !$0.isMultisampled }) else {
            throw GPUError.unsupported("Direct3D 12 compute pipelines do not support acceleration structures or multisampled textures")
        }
        if compiler == nil { compiler = try D3D12ShaderCompiler(directories: compilerDirectories) }
        let dxil = try compiler!.compile(source, entryPoint: shader.entryPoint, profile: profile)

        // Every buffer binding is a root descriptor, so no descriptor heaps are needed.
        var parameters: [D3D12RootParameter] = []
        var rootBuffers: [(slot: ComputeBindingSlot, usage: BufferUsage, isReadOnly: Bool, root: Int)] = []
        var arrayBuffers: [(slot: ComputeBindingSlot, readOnly: Bool)] = []
        for binding in pipeline.bufferBindings {
            if binding.slot.arrayElement != nil {
                guard binding.usage != .uniform else { throw GPUError.unsupported("Direct3D 12 uniform buffer arrays are not implemented") }
                arrayBuffers.append((binding.slot, binding.isReadOnly))
                continue
            }
            let type = binding.usage == .uniform ? D3D12.rootParameterCBV
                : (binding.isReadOnly ? D3D12.rootParameterSRV : D3D12.rootParameterUAV)
            rootBuffers.append((binding.slot, binding.usage, binding.isReadOnly, parameters.count))
            parameters.append(D3D12RootParameter(parameterType: type, shaderRegister: binding.slot.binding,
                                                 registerSpace: binding.slot.group))
        }
        // Binding arrays declare their first register in the artifact.
        var arrayInfo: [IRResourceBinding: (count: UInt32, base: UInt32)] = [:]
        for binding in shader.storageBindings { if let a = binding.bindingArray, let base = a.hlslRegisterBase { arrayInfo[binding.binding] = (a.count, base) } }
        for binding in shader.sampledTextures { if let a = binding.bindingArray, let base = a.hlslRegisterBase { arrayInfo[binding.binding] = (a.count, base) } }
        for binding in shader.storageTextures { if let a = binding.bindingArray, let base = a.hlslRegisterBase { arrayInfo[binding.binding] = (a.count, base) } }
        for binding in shader.samplerBindings { if let a = binding.bindingArray, let base = a.hlslRegisterBase { arrayInfo[binding.binding] = (a.count, base) } }
        func ordered(_ slots: [ComputeBindingSlot]) -> [ComputeBindingSlot] {
            slots.sorted { ($0.group, $0.binding, $0.arrayElement ?? 0) < ($1.group, $1.binding, $1.arrayElement ?? 0) }
        }
        /// One range per scalar binding and one per array, with descriptors laid out in element order.
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
        var immediateRoot: Int?, countRoot: Int?
        let immediateWords = Int((pipeline.immediateByteSize + 3) / 4)
        if immediateWords > 0, let binding = shader.immediateLayout?.hlslBinding {
            immediateRoot = parameters.count
            parameters.append(D3D12RootParameter(parameterType: D3D12.rootParameterConstants, shaderRegister: binding.binding,
                                                 registerSpace: binding.group, unionTail: UInt64(immediateWords)))
        }
        if let binding = shader.hlslWorkgroupCountBinding {
            countRoot = parameters.count
            parameters.append(D3D12RootParameter(parameterType: D3D12.rootParameterConstants, shaderRegister: binding.binding,
                                                 registerSpace: binding.group, unionTail: 4))
        }
        var accelerationRoots: [(slot: ComputeBindingSlot, root: Int)] = []
        for slot in pipeline.accelerationStructureBindings {
            accelerationRoots.append((slot, parameters.count))
            parameters.append(D3D12RootParameter(parameterType: D3D12.rootParameterSRV, shaderRegister: slot.binding, registerSpace: slot.group))
        }
        var externalLayouts: [(layout: ComputeShaderArtifact.ExternalTextureLayout, root: Int)] = []
        for layout in shader.externalTextureLayouts {
            externalLayouts.append((layout, parameters.count))
            parameters.append(D3D12RootParameter(parameterType: D3D12.rootParameterCBV, shaderRegister: layout.parametersBindingIndex,
                                                 registerSpace: layout.binding.group))
        }
        var viewLayout: [(slot: ComputeBindingSlot, storage: Bool)] = []
        var viewRanges: [D3D12DescriptorRange] = [], samplerRanges: [D3D12DescriptorRange] = []
        var viewUsed: UInt32 = 0, samplerUsed: UInt32 = 0
        addRanges(pipeline.textureBindings.map(\.slot), type: 0, to: &viewRanges, used: &viewUsed)
        viewLayout += ordered(pipeline.textureBindings.map(\.slot)).map { ($0, false) }
        addRanges(pipeline.storageTextureBindings.map(\.slot), type: 1, to: &viewRanges, used: &viewUsed)
        viewLayout += ordered(pipeline.storageTextureBindings.map(\.slot)).map { ($0, true) }
        // Array buffers: read-only ones are SRV ranges, writable ones UAV ranges.
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
        addRanges(pipeline.samplerBindings.map(\.slot), type: 3, to: &samplerRanges, used: &samplerUsed)
        let samplerLayout = ordered(pipeline.samplerBindings.map(\.slot))
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
        let (serialized, blob, errorBlob) = Self.serialize(library, parameters: &parameters)
        withExtendedLifetime(errorBlob) {}
        try d3dCheck(serialized, "root signature serialization")
        let blobObject = blob!
        typealias BufferPointer = @convention(c) (UnsafeMutableRawPointer) -> UnsafeRawPointer?
        typealias BufferSize = @convention(c) (UnsafeMutableRawPointer) -> Int
        let blobBytes = blobObject.slot(3, BufferPointer.self)(blobObject.pointer)
        let blobSize = blobObject.slot(4, BufferSize.self)(blobObject.pointer)

        typealias CreateRoot = @convention(c) (UnsafeMutableRawPointer, UInt32, UnsafeRawPointer?, Int, UnsafeRawPointer,
                                               UnsafeMutablePointer<UnsafeMutableRawPointer?>) -> HRESULT
        var rootIID = IID.rootSignature
        var rootObject: UnsafeMutableRawPointer?
        try d3dCheck(device.slot(16, CreateRoot.self)(device.pointer, 0, blobBytes, blobSize, &rootIID, &rootObject), "root signature creation")
        let root = COMObject(rootObject!)

        typealias CreatePipeline = @convention(c) (UnsafeMutableRawPointer, UnsafeRawPointer,
                                                   UnsafeRawPointer, UnsafeMutablePointer<UnsafeMutableRawPointer?>) -> HRESULT
        var stateIID = IID.pipelineState
        let (created, stateObject) = Self.createState(device, root: root, dxil: dxil, stateIID: &stateIID, create: device.slot(11, CreatePipeline.self))
        try d3dCheck(created, "compute pipeline creation")
        pipelines[pipeline.id] = Pipeline(model: pipeline, accelerationRoots: accelerationRoots, externalLayouts: externalLayouts, rootSignature: root, state: stateObject!, immediateRoot: immediateRoot,
                                          countRoot: countRoot, immediateWords: immediateWords, viewRoot: viewRoot,
                                          samplerRoot: samplerRoot, viewLayout: viewLayout,
                                          samplerLayout: samplerLayout, bufferViewLayout: bufferViewLayout,
                                          rootBuffers: rootBuffers)
    }

    nonisolated static func serialize(_ library: D3D12Library, parameters: inout [D3D12RootParameter], flags: UInt32 = 0) -> (HRESULT, COMObject?, COMObject?) {
        var blob: UnsafeMutableRawPointer?
        var errors: UnsafeMutableRawPointer?
        let result: HRESULT = parameters.withUnsafeMutableBufferPointer { pointer in
            var description = D3D12RootSignatureDesc(parameterCount: UInt32(pointer.count),
                                                     parameters: UnsafeRawPointer(pointer.baseAddress), flags: flags)
            return library.serializeRootSignature(&description, 1, &blob, &errors)
        }
        return (result, blob.map(COMObject.init), errors.map(COMObject.init))
    }

    nonisolated static func createState(
        _ device: COMObject, root: COMObject, dxil: [UInt8], stateIID: inout D3DGUID,
        create: @convention(c) (UnsafeMutableRawPointer, UnsafeRawPointer, UnsafeRawPointer, UnsafeMutablePointer<UnsafeMutableRawPointer?>) -> HRESULT
    ) -> (HRESULT, COMObject?) {
        var object: UnsafeMutableRawPointer?
        let result: HRESULT = dxil.withUnsafeBytes { code in
            var description = D3D12ComputePipelineStateDesc(rootSignature: root.pointer, shaderBytecode: code.baseAddress,
                                                            shaderBytecodeSize: code.count)
            return create(device.pointer, &description, &stateIID, &object)
        }
        return (result, object.map(COMObject.init))
    }

    public func destroyComputePipeline(id: ResourceID) { pipelines[id] = nil }

    // MARK: Queue

    public func submit(_ submission: Submission) async throws {
        _ = try await submitCommands(submission, commands: [])
    }

    public func submitCommands(_ submission: Submission, commands: [BufferCommand]) async throws -> [[UInt32]] {
        try await runCommands(submission, commands: commands, validate: true)
    }

    /// Runs commands. Internal callers such as buffer mapping skip usage validation for their own transfers.
    func runCommands(_ submission: Submission, commands: [BufferCommand], validate: Bool) async throws -> [[UInt32]] {
        try requireActive()
        let owner = submission.resources.first?.device ??
            commands.lazy.compactMap { $0.recordedPass.command.computeDispatch?.pipeline.id.device }.first ?? 0
        let descriptors = buffers.mapValues(\.descriptor)
        let textureDescriptors = textures.mapValues(\.descriptor)
        let samplerDescriptors = samplers
        let pinned = Set(submission.resources)
        for id in submission.resources
        where buffers[id] == nil && pipelines[id] == nil && textures[id] == nil && textureViews[id] == nil && samplers[id] == nil && externalTextures[id] == nil
            && renderPipelines[id] == nil && querySets[id] == nil && accelerationStructures[id] == nil {
            throw GPUError.validation("Unknown Direct3D 12 resource")
        }

        typealias CreateAllocator = @convention(c) (UnsafeMutableRawPointer, UInt32, UnsafeRawPointer,
                                                    UnsafeMutablePointer<UnsafeMutableRawPointer?>) -> HRESULT
        typealias CreateList = @convention(c) (UnsafeMutableRawPointer, UInt32, UInt32, UnsafeMutableRawPointer, UnsafeMutableRawPointer?,
                                               UnsafeRawPointer, UnsafeMutablePointer<UnsafeMutableRawPointer?>) -> HRESULT
        var allocatorIID = IID.commandAllocator, listIID = IID.commandList
        var allocatorObject: UnsafeMutableRawPointer?
        try d3dCheck(device.slot(9, CreateAllocator.self)(device.pointer, D3D12.commandListDirect, &allocatorIID, &allocatorObject),
                     "command allocator creation")
        let allocator = COMObject(allocatorObject!)
        var listObject: UnsafeMutableRawPointer?
        try d3dCheck(device.slot(12, CreateList.self)(device.pointer, 0, D3D12.commandListDirect, allocator.pointer, nil,
                                                      &listIID, &listObject), "command list creation")
        let list = COMObject(listObject!)

        typealias Barrier = @convention(c) (UnsafeMutableRawPointer, UInt32, UnsafeRawPointer) -> Void
        typealias CopyRegion = @convention(c) (UnsafeMutableRawPointer, UnsafeMutableRawPointer, UInt64, UnsafeMutableRawPointer, UInt64, UInt64) -> Void
        typealias SetRoot = @convention(c) (UnsafeMutableRawPointer, UnsafeMutableRawPointer) -> Void
        typealias SetAddress = @convention(c) (UnsafeMutableRawPointer, UInt32, UInt64) -> Void
        typealias Dispatch = @convention(c) (UnsafeMutableRawPointer, UInt32, UInt32, UInt32) -> Void

        var retained = Retained(objects: [allocator, list])
        var states: [ResourceID: UInt32] = [:]
        var resources: [ResourceID: COMObject] = [:]
        var reads: [(readback: COMObject, count: Int)] = []
        var arena: D3D12DescriptorArena?
        var heapsSet = false
        var renderTargets: D3D12RenderTargetArena?

        func transition(_ id: ResourceID, to state: UInt32) {
            let id = textureViews[id]?.texture ?? id  // views share the state of their texture
            guard let object = buffers[id]?.resource ?? textures[id]?.resource else { return }
            let current = states[id] ?? D3D12.stateCommon
            resources[id] = object
            guard current != state else { return }
            var barrier = D3D12ResourceBarrier(type: D3D12.barrierTransition, resource: object.pointer,
                                               subresource: 0xFFFF_FFFF, stateBefore: current, stateAfter: state)
            list.slot(26, Barrier.self)(list.pointer, 1, &barrier)
            states[id] = state
        }
        func uavBarrier(_ resource: COMObject) {
            var barrier = D3D12ResourceBarrier(type: D3D12.barrierUAV, resource: resource.pointer)
            list.slot(26, Barrier.self)(list.pointer, 1, &barrier)
        }
        func upload(_ bytes: [UInt8], to id: ResourceID, offset: UInt64) throws {
            guard let destination = buffers[id], !bytes.isEmpty else { return }
            let staging = try makeBuffer(size: UInt64(bytes.count), heap: D3D12.heapUpload, state: D3D12.stateGenericRead)
            try map(staging, count: bytes.count) { $0.copyMemory(from: bytes, byteCount: bytes.count) }
            retained.objects.append(staging)
            transition(id, to: D3D12.stateCopyDestination)
            list.slot(15, CopyRegion.self)(list.pointer, destination.resource.pointer, offset, staging.pointer, 0, UInt64(bytes.count))
        }

        do {
            for recorded in commands {
                guard recorded.resources.allSatisfy(pinned.contains) else { throw GPUError.validation("Unpinned command resource") }
                if validate { try recorded.validate(device: owner, descriptors: descriptors, textures: textureDescriptors, samplerDescriptors: samplerDescriptors,
                                      textureViews: textureViews.mapValues { .init(texture: $0.texture, levels: $0.levels, aspect: $0.aspect, layers: $0.layers, dimension: $0.dimension) },
                                      maximumViewportDimension: capabilities.maxTextureDimension2D,
                                      maxVertexBuffers: capabilities.maxVertexBuffers,
                                      maxBindGroupsPlusVertexBuffers: capabilities.maxBindGroupsPlusVertexBuffers,
                                      querySets: querySets.mapValues { $0.descriptor },
                                      features: capabilities.features, externalTextures: Set(externalTextures.keys),
                                      externalPlanes: externalTextures.mapValues(\.planes),
                                      accelerationStructures: accelerationStructures.mapValues(\.descriptor),
                                      maxComputeWorkgroupsPerDimension: capabilities.maxComputeWorkgroupsPerDimension,
                                      maxTaskWorkgroupTotalCount: capabilities.maxTaskWorkgroupTotalCount,
                                      maxTaskWorkgroupsPerDimension: capabilities.maxTaskWorkgroupsPerDimension,
                                      maxMeshWorkgroupTotalCount: capabilities.maxMeshWorkgroupTotalCount,
                                      maxMeshWorkgroupsPerDimension: capabilities.maxMeshWorkgroupsPerDimension) }
                let (command, timestamps, occlusion) = recorded.recordedPass
                let measurement = timestamps == nil && occlusion == nil ? nil
                    : try planMeasurement(timestamps: timestamps, occlusion: occlusion)
                typealias WriteQuery = @convention(c) (UnsafeMutableRawPointer, UnsafeMutableRawPointer, UInt32, UInt32) -> Void
                if let begin = measurement?.begin { list.slot(53, WriteQuery.self)(list.pointer, begin.heap.pointer, 2, begin.index) }
                if let probe = measurement?.occlusion { list.slot(52, WriteQuery.self)(list.pointer, probe.heap.pointer, 0, probe.index) }
                defer {
                    if let probe = measurement?.occlusion { list.slot(53, WriteQuery.self)(list.pointer, probe.heap.pointer, 0, probe.index) }
                    if let end = measurement?.end { list.slot(53, WriteQuery.self)(list.pointer, end.heap.pointer, 2, end.index) }
                }
                switch command {
                case .resolveQuerySet(let set, let first, let count, let destination, let offset):
                    try recordResolve(querySet: set, first: first, count: count, destination: destination, offset: offset,
                                      list: list, transition: transition, retained: &retained)
                case .write(let id, let offset, let words):
                    var bytes: [UInt8] = []
                    bytes.reserveCapacity(words.count * 4)
                    for word in words {
                        bytes += [UInt8(truncatingIfNeeded: word), UInt8(truncatingIfNeeded: word >> 8),
                                  UInt8(truncatingIfNeeded: word >> 16), UInt8(truncatingIfNeeded: word >> 24)]
                    }
                    try upload(bytes, to: id, offset: offset)
                case .writeBytes(let id, let offset, let bytes, let dataOffset, let byteCount):
                    try upload(Array(try BufferCommand.uploadBytes(bytes, dataOffset: dataOffset, byteCount: byteCount)), to: id, offset: offset)
                case .copy(let source, let sourceOffset, let destination, let destinationOffset, let byteCount):
                    guard let from = buffers[source], let to = buffers[destination] else { throw GPUError.validation("Unknown Direct3D 12 buffer") }
                    let count = byteCount ?? (from.descriptor.size - sourceOffset)
                    guard count > 0 else { continue }
                    transition(source, to: D3D12.stateCopySource)
                    transition(destination, to: D3D12.stateCopyDestination)
                    list.slot(15, CopyRegion.self)(list.pointer, to.resource.pointer, destinationOffset, from.resource.pointer,
                                                   sourceOffset, count)
                case .clearBuffer(let id, let offset, let byteCount):
                    guard let buffer = buffers[id] else { throw GPUError.validation("Unknown Direct3D 12 buffer") }
                    let count = byteCount ?? (buffer.descriptor.size - offset)
                    guard count > 0 else { continue }
                    try upload([UInt8](repeating: 0, count: Int(count)), to: id, offset: offset)
                case .read(let id, let byteCount):
                    guard let source = buffers[id] else { throw GPUError.validation("Unknown Direct3D 12 buffer") }
                    let readback = try makeBuffer(size: byteCount, heap: D3D12.heapReadback, state: D3D12.stateCopyDestination)
                    retained.objects.append(readback)
                    transition(id, to: D3D12.stateCopySource)
                    list.slot(15, CopyRegion.self)(list.pointer, readback.pointer, 0, source.resource.pointer, 0, byteCount)
                    reads.append((readback, Int(byteCount)))
                case .writeTexture(let destination, let data, let layout, let r, _):
                    guard let texture = textures[destination] else { throw GPUError.validation("Unknown Direct3D 12 texture") }
                    let texel = UInt64(d3d12BytesPerTexel(texture.descriptor.format))
                    let rowBytes = UInt64(r.width) * texel
                    let stride = UInt64(layout.bytesPerRow ?? UInt32(rowBytes))
                    let rows = UInt64(r.height)
                    let rowsPerImage = UInt64(layout.rowsPerImage ?? r.height)
                    let pitch = align(rowBytes, 256)
                    let is3D = texture.descriptor.dimension == .d3
                    let planeStride = is3D ? rows * pitch : align(rows * pitch, 512)
                    var staged = [UInt8](repeating: 0, count: Int(planeStride * UInt64(r.depth)))
                    for plane in 0..<UInt64(r.depth) {
                        for row in 0..<rows {
                            let from = Int(layout.offset + plane * rowsPerImage * stride + row * stride)
                            let to = Int(plane * planeStride + row * pitch)
                            staged.replaceSubrange(to..<(to + Int(rowBytes)), with: data[from..<(from + Int(rowBytes))])
                        }
                    }
                    let staging = try makeBuffer(size: UInt64(staged.count), heap: D3D12.heapUpload, state: D3D12.stateGenericRead)
                    try map(staging, count: staged.count) { $0.copyMemory(from: staged, byteCount: staged.count) }
                    retained.objects.append(staging)
                    try recordImageCopy(D3D12ImageCopy(toTexture: true, texture: destination, mip: r.mipLevel, layer: r.arrayLayer,
                                                       planes: r.depth, x: r.x, y: r.y, z: r.z, width: r.width, height: r.height,
                                                       buffer: staging, bufferOffset: 0, rowPitch: UInt32(pitch)),
                                        list: list, transition: transition)
                case .copyTextureToBuffer(let source, let destination, let bytesPerRow, let mip, _, let offset, let layer, let z, let depth, let rows):
                    guard let texture = textures[source], let target = buffers[destination],
                          let extent = texture.descriptor.mipExtent(mip), rows == nil || rows == extent.height else {
                        throw GPUError.unsupported("Unsupported Direct3D 12 texture copy")
                    }
                    transition(destination, to: D3D12.stateCopyDestination)
                    try recordImageCopy(D3D12ImageCopy(toTexture: false, texture: source, mip: mip, layer: layer, planes: depth, z: z,
                                                       width: extent.width, height: extent.height, buffer: target.resource,
                                                       bufferOffset: offset, rowPitch: bytesPerRow),
                                        list: list, transition: transition)
                case .copyTextureRegionToBuffer(let source, let destination, let layout, let r, _):
                    guard let target = buffers[destination], let pitch = layout.bytesPerRow,
                          layout.rowsPerImage == nil || layout.rowsPerImage == r.height else {
                        throw GPUError.unsupported("Unsupported Direct3D 12 texture copy")
                    }
                    transition(destination, to: D3D12.stateCopyDestination)
                    try recordImageCopy(D3D12ImageCopy(toTexture: false, texture: source, mip: r.mipLevel, layer: r.arrayLayer, planes: r.depth,
                                                       x: r.x, y: r.y, z: r.z, width: r.width, height: r.height, buffer: target.resource,
                                                       bufferOffset: layout.offset, rowPitch: pitch),
                                        list: list, transition: transition)
                case .copyBufferToTexture(let source, let layout, let destination, let r, _):
                    guard let from = buffers[source], let pitch = layout.bytesPerRow,
                          layout.rowsPerImage == nil || layout.rowsPerImage == r.height else {
                        throw GPUError.unsupported("Unsupported Direct3D 12 texture copy")
                    }
                    transition(source, to: D3D12.stateCopySource)
                    try recordImageCopy(D3D12ImageCopy(toTexture: true, texture: destination, mip: r.mipLevel, layer: r.arrayLayer, planes: r.depth,
                                                       x: r.x, y: r.y, z: r.z, width: r.width, height: r.height, buffer: from.resource,
                                                       bufferOffset: layout.offset, rowPitch: pitch),
                                        list: list, transition: transition)
                case .render, .renderIndexed, .renderIndirect:
                    try recordDraw(command, list: list, transition: transition, arena: &arena, heapsSet: &heapsSet,
                                   renderTargets: &renderTargets)
                case .buildAccelerationStructure, .buildBoxAccelerationStructure, .buildTopLevelAccelerationStructure:
                    try recordStructureBuild(command, list: list, transition: { transition($0, to: $1) }, retained: &retained)
                case .drawMesh:
                    let (equivalent, groups, indirect) = try equivalentRender(for: command)
                    var validated: COMObject?
                    if let indirect {
                        // GPU-authored group counts go through the mesh dispatch validator into a scratch buffer.
                        guard let source = buffers[indirect.buffer], indirect.offset % 4 == 0, indirect.offset <= source.descriptor.size,
                              source.descriptor.size - indirect.offset >= 12 else {
                            throw GPUError.validation("Invalid Direct3D 12 indirect mesh draw arguments")
                        }
                        let helper = try validatorPipeline(mesh: true)
                        let target = try makeBuffer(size: 16, heap: D3D12.heapDefault, state: D3D12.stateCommon,
                                                    flags: D3D12.flagAllowUnorderedAccess)
                        retained.objects.append(target)
                        func manual(_ resource: COMObject, _ before: UInt32, _ after: UInt32) {
                            var barrier = D3D12ResourceBarrier(type: D3D12.barrierTransition, resource: resource.pointer,
                                                               subresource: 0xFFFF_FFFF, stateBefore: before, stateAfter: after)
                            list.slot(26, Barrier.self)(list.pointer, 1, &barrier)
                        }
                        manual(target, D3D12.stateCommon, D3D12.stateUnorderedAccess)
                        transition(indirect.buffer, to: D3D12.stateNonPixelShaderResource)
                        list.slot(29, SetRoot.self)(list.pointer, helper.rootSignature.pointer)
                        typealias SetPipeline = @convention(c) (UnsafeMutableRawPointer, UnsafeMutableRawPointer) -> Void
                        list.slot(25, SetPipeline.self)(list.pointer, helper.state.pointer)
                        list.slot(39, SetAddress.self)(list.pointer, 0, source.address + indirect.offset)
                        list.slot(41, SetAddress.self)(list.pointer, 1, gpuAddress(target))
                        list.slot(14, Dispatch.self)(list.pointer, 1, 1, 1)
                        uavBarrier(target)
                        manual(target, D3D12.stateUnorderedAccess, D3D12.stateIndirectArgument)
                        validated = target
                    }
                    try recordDraw(equivalent, list: list, transition: transition, arena: &arena, heapsSet: &heapsSet,
                                   renderTargets: &renderTargets, meshGroups: groups, meshIndirect: validated)
                case .dispatchCompute, .dispatchComputeRanges, .dispatchComputeIndirect:
                    guard let dispatch = command.computeDispatch, let pipeline = pipelines[dispatch.pipeline.id],
                          pipeline.model == dispatch.pipeline else {
                        throw GPUError.validation("Unknown Direct3D 12 compute pipeline")
                    }
                    var directGroups: DispatchWorkgroups?
                    var indirectSource: (buffer: Buffer, id: ResourceID, offset: UInt64)?
                    switch dispatch.source {
                    case .direct(let groups): directGroups = groups
                    case .indirect(let id, let offset):
                        guard let source = buffers[id] else { throw GPUError.validation("Unknown Direct3D 12 buffer") }
                        guard pipeline.countRoot == nil else {
                            throw GPUError.unsupported("Direct3D 12 indirect dispatch cannot feed shaders that read the workgroup count")
                        }
                        indirectSource = (source, id, offset)
                    }
                    var scratch: COMObject?
                    if let indirectSource {
                        // GPU-authored counts are validated into a scratch buffer, which becomes the indirect argument.
                        let helper = try validatorPipeline()
                        let target = try makeBuffer(size: 16, heap: D3D12.heapDefault, state: D3D12.stateCommon,
                                                    flags: D3D12.flagAllowUnorderedAccess)
                        retained.objects.append(target)
                        scratch = target
                        func manual(_ resource: COMObject, _ before: UInt32, _ after: UInt32) {
                            var barrier = D3D12ResourceBarrier(type: D3D12.barrierTransition, resource: resource.pointer,
                                                               subresource: 0xFFFF_FFFF, stateBefore: before, stateAfter: after)
                            list.slot(26, Barrier.self)(list.pointer, 1, &barrier)
                        }
                        manual(target, D3D12.stateCommon, D3D12.stateUnorderedAccess)
                        transition(indirectSource.id, to: D3D12.stateNonPixelShaderResource)
                        list.slot(29, SetRoot.self)(list.pointer, helper.rootSignature.pointer)
                        typealias SetPipeline = @convention(c) (UnsafeMutableRawPointer, UnsafeMutableRawPointer) -> Void
                        list.slot(25, SetPipeline.self)(list.pointer, helper.state.pointer)
                        list.slot(39, SetAddress.self)(list.pointer, 0, indirectSource.buffer.address + indirectSource.offset)
                        list.slot(41, SetAddress.self)(list.pointer, 1, gpuAddress(target))
                        list.slot(14, Dispatch.self)(list.pointer, 1, 1, 1)
                        uavBarrier(target)
                        manual(target, D3D12.stateUnorderedAccess, D3D12.stateIndirectArgument)
                    }
                    list.slot(29, SetRoot.self)(list.pointer, pipeline.rootSignature.pointer)
                    typealias SetPipeline = @convention(c) (UnsafeMutableRawPointer, UnsafeMutableRawPointer) -> Void
                    list.slot(25, SetPipeline.self)(list.pointer, pipeline.state.pointer)
                    typealias SetConstants = @convention(c) (UnsafeMutableRawPointer, UInt32, UInt32, UnsafeRawPointer, UInt32) -> Void
                    if let root = pipeline.immediateRoot {
                        var words = dispatch.immediates
                        if words.count < pipeline.immediateWords { words += [UInt32](repeating: 0, count: pipeline.immediateWords - words.count) }
                        words.withUnsafeBytes { list.slot(35, SetConstants.self)(list.pointer, UInt32(root), UInt32(pipeline.immediateWords), $0.baseAddress!, 0) }
                    }
                    if let root = pipeline.countRoot, let groups = directGroups {
                        let counts: [UInt32] = [groups.x, groups.y, groups.z, 0]
                        counts.withUnsafeBytes { list.slot(35, SetConstants.self)(list.pointer, UInt32(root), 4, $0.baseAddress!, 0) }
                    }
                    var writtenArrays: [COMObject] = []
                    if pipeline.viewRoot != nil || pipeline.samplerRoot != nil {
                        if arena == nil { arena = try D3D12DescriptorArena(device: device) }
                        let heaps = arena!
                        if !heapsSet {
                            typealias SetHeaps = @convention(c) (UnsafeMutableRawPointer, UInt32, UnsafeRawPointer) -> Void
                            var heapPointers: [UnsafeMutableRawPointer?] = [heaps.viewHeap.pointer, heaps.samplerHeap.pointer]
                            list.slot(28, SetHeaps.self)(list.pointer, 2, &heapPointers)
                            heapsSet = true
                        }
                        typealias SetTable = @convention(c) (UnsafeMutableRawPointer, UInt32, UInt64) -> Void
                        if let root = pipeline.viewRoot {
                            let base = try heaps.allocateViews(pipeline.viewLayout.count + pipeline.bufferViewLayout.count + 3 * pipeline.externalLayouts.count)
                            for (offset, entry) in pipeline.bufferViewLayout.enumerated() {
                                guard let supplied = dispatch.bindings[entry.slot], let buffer = buffers[supplied.buffer] else {
                                    throw GPUError.validation("Missing Direct3D 12 buffer array binding")
                                }
                                let range = try supplied.range(in: buffer.descriptor, usage: .storage)
                                transition(supplied.buffer, to: entry.readOnly ? D3D12.stateNonPixelShaderResource : D3D12.stateUnorderedAccess)
                                if !entry.readOnly { writtenArrays.append(buffer.resource) }
                                createRawBufferView(buffer.resource, offset: range.lowerBound, size: UInt64(range.count),
                                                    storage: !entry.readOnly,
                                                    at: base.cpu + UInt(pipeline.viewLayout.count + offset) * heaps.viewIncrement)
                            }
                            for (slotIndex, entry) in pipeline.viewLayout.enumerated() {
                                guard let id = dispatch.textures[entry.slot], let texture = texture(for: id),
                                      texture.descriptor.usage.contains(entry.storage ? .storageBinding : .textureBinding) else {
                                    throw GPUError.validation("Missing or invalid Direct3D 12 texture binding")
                                }
                                transition(id, to: entry.storage ? D3D12.stateUnorderedAccess : D3D12.stateNonPixelShaderResource)
                                createView(texture, storage: entry.storage, at: base.cpu + UInt(slotIndex) * heaps.viewIncrement)
                            }
                            let externalBase = pipeline.viewLayout.count + pipeline.bufferViewLayout.count
                            for (index, entry) in pipeline.externalLayouts.enumerated() {
                                let slot = ComputeBindingSlot(group: entry.layout.binding.group, binding: entry.layout.binding.binding)
                                guard let id = dispatch.textures[slot], let external = externalTextures[id] else {
                                    throw GPUError.validation("Missing Direct3D 12 external texture binding")
                                }
                                for plane in 0..<3 {
                                    let planeID = external.planes[plane < external.planes.count ? plane : 0]
                                    guard let texture = texture(for: planeID) else { throw GPUError.validation("Unknown Direct3D 12 external texture plane") }
                                    transition(planeID, to: D3D12.stateNonPixelShaderResource)
                                    createView(texture, storage: false,
                                               at: base.cpu + UInt(externalBase + 3 * index + plane) * heaps.viewIncrement)
                                }
                            }
                            list.slot(31, SetTable.self)(list.pointer, UInt32(root), base.gpu)
                        }
                        if let root = pipeline.samplerRoot {
                            let base = try heaps.allocateSamplers(pipeline.samplerLayout.count)
                            for (slotIndex, slot) in pipeline.samplerLayout.enumerated() {
                                guard let id = dispatch.samplers[slot], let sampler = samplers[id] else {
                                    throw GPUError.validation("Missing Direct3D 12 sampler binding")
                                }
                                createSampler(sampler, at: base.cpu + UInt(slotIndex) * heaps.samplerIncrement)
                            }
                            list.slot(31, SetTable.self)(list.pointer, UInt32(root), base.gpu)
                        }
                    }
                    for entry in pipeline.externalLayouts {
                        let slot = ComputeBindingSlot(group: entry.layout.binding.group, binding: entry.layout.binding.binding)
                        guard let id = dispatch.textures[slot], let external = externalTextures[id] else {
                            throw GPUError.validation("Missing Direct3D 12 external texture binding")
                        }
                        list.slot(37, SetAddress.self)(list.pointer, UInt32(entry.root), external.address)
                    }
                    for entry in pipeline.accelerationRoots {
                        guard let id = dispatch.accelerationStructures[entry.slot], let structure = accelerationStructures[id] else {
                            throw GPUError.validation("Missing Direct3D 12 acceleration structure binding")
                        }
                        list.slot(39, SetAddress.self)(list.pointer, UInt32(entry.root), structure.address)
                    }
                    var written: [COMObject] = writtenArrays
                    for rootBuffer in pipeline.rootBuffers {
                        let index = rootBuffer.root
                        let binding = (slot: rootBuffer.slot, usage: rootBuffer.usage, isReadOnly: rootBuffer.isReadOnly)
                        guard let supplied = dispatch.bindings[binding.slot], let buffer = buffers[supplied.buffer] else {
                            throw GPUError.validation("Missing Direct3D 12 buffer binding")
                        }
                        let range = try supplied.range(in: buffer.descriptor, usage: binding.usage)
                        let address = buffer.address + range.lowerBound
                        if binding.usage == .uniform {
                            transition(supplied.buffer, to: D3D12.stateConstantBuffer)
                            list.slot(37, SetAddress.self)(list.pointer, UInt32(index), address)
                        } else if binding.isReadOnly {
                            transition(supplied.buffer, to: D3D12.stateNonPixelShaderResource)
                            list.slot(39, SetAddress.self)(list.pointer, UInt32(index), address)
                        } else {
                            transition(supplied.buffer, to: D3D12.stateUnorderedAccess)
                            list.slot(41, SetAddress.self)(list.pointer, UInt32(index), address)
                            written.append(buffer.resource)
                        }
                    }
                    if let groups = directGroups {
                        list.slot(14, Dispatch.self)(list.pointer, groups.x, groups.y, groups.z)
                    } else if let scratch {
                        typealias ExecuteIndirect = @convention(c) (UnsafeMutableRawPointer, UnsafeMutableRawPointer, UInt32, UnsafeMutableRawPointer, UInt64,
                                                                    UnsafeMutableRawPointer?, UInt64) -> Void
                        let signature = try dispatchSignature()
                        list.slot(59, ExecuteIndirect.self)(list.pointer, signature.pointer, 1, scratch.pointer, 0, nil, 0)
                        var barrier = D3D12ResourceBarrier(type: D3D12.barrierTransition, resource: scratch.pointer,
                                                           subresource: 0xFFFF_FFFF, stateBefore: D3D12.stateIndirectArgument,
                                                           stateAfter: D3D12.stateCommon)
                        list.slot(26, Barrier.self)(list.pointer, 1, &barrier)
                    }
                    written.forEach(uavBarrier)
                default:
                    throw GPUError.unsupported("Direct3D 12 does not implement this command yet")
                }
            }
            // Buffers return to the common state so the next submission can assume it.
            for id in states.keys.sorted(by: { ($0.device, $0.serial) < ($1.device, $1.serial) }) { transition(id, to: D3D12.stateCommon) }
            typealias Close = @convention(c) (UnsafeMutableRawPointer) -> HRESULT
            try d3dCheck(list.slot(9, Close.self)(list.pointer), "command recording")
        }

        typealias Execute = @convention(c) (UnsafeMutableRawPointer, UInt32, UnsafePointer<UnsafeMutableRawPointer?>) -> Void
        typealias Signal = @convention(c) (UnsafeMutableRawPointer, UnsafeMutableRawPointer, UInt64) -> HRESULT
        if let arena { retained.objects += [arena.viewHeap, arena.samplerHeap] }
        if let renderTargets { retained.objects.append(renderTargets.heap) }
        var lists: [UnsafeMutableRawPointer?] = [list.pointer]
        queue.slot(10, Execute.self)(queue.pointer, 1, &lists)
        fenceValue += 1
        let target = fenceValue
        do {
            try d3dCheck(queue.slot(14, Signal.self)(queue.pointer, fence.pointer, target), "fence signal")
            try await waitForFence(target)
        } catch {
            D3D12DebugLayer.drain(device: device)
            terminal = true
            quarantine.append(retained)  // the GPU may still own these objects; leaking them is safer
            throw error
        }
        D3D12DebugLayer.drain(device: device)
        var results: [[UInt32]] = []
        for read in reads {
            var words = [UInt32](repeating: 0, count: read.count / 4)
            try map(read.readback, count: read.count) { pointer in
                words.withUnsafeMutableBytes { $0.copyMemory(from: UnsafeRawBufferPointer(start: pointer, count: read.count)) }
            }
            results.append(words)
        }
        return results
    }

    func map(_ resource: COMObject, count: Int, _ body: (UnsafeMutableRawPointer) -> Void) throws {
        typealias Map = @convention(c) (UnsafeMutableRawPointer, UInt32, UnsafeRawPointer?, UnsafeMutablePointer<UnsafeMutableRawPointer?>) -> HRESULT
        typealias Unmap = @convention(c) (UnsafeMutableRawPointer, UInt32, UnsafeRawPointer?) -> Void
        var data: UnsafeMutableRawPointer?
        try d3dCheck(resource.slot(8, Map.self)(resource.pointer, 0, nil, &data), "buffer mapping")
        body(data!)
        resource.slot(9, Unmap.self)(resource.pointer, 0, nil)
    }

    func waitForFence(_ value: UInt64) async throws {
        typealias Completed = @convention(c) (UnsafeMutableRawPointer) -> UInt64
        let clock = ContinuousClock()
        let deadline = clock.now + timeout
        var delay = Duration.microseconds(50)
        while fence.slot(8, Completed.self)(fence.pointer) < value {
            guard clock.now < deadline else { throw GPUError.timeout }
            try? await Task.sleep(for: delay)
            delay = min(delay * 2, .milliseconds(2))
        }
    }
}
#endif

#if os(Windows)
extension D3D12Device {
    /// Zeroes GPU-authored workgroup counts that exceed the Direct3D 12 per-dimension limit.
    func validatorPipeline(mesh: Bool = false) throws -> Pipeline {
        if mesh, let meshValidator { return meshValidator }
        if !mesh, let validator { return validator }
        if compiler == nil { compiler = try D3D12ShaderCompiler(directories: compilerDirectories) }
        let hlsl = """
            ByteAddressBuffer source : register(t0);
            RWByteAddressBuffer destination : register(u0);
            [numthreads(1, 1, 1)]
            void validate() {
                uint3 counts = source.Load3(0);
                bool invalid = any(counts > 65535u) || any(counts == 0u);
                \(mesh ? "if (!invalid) { uint xy = counts.x * counts.y; invalid = counts.z > 4194304u / xy || xy > 4194304u; }" : "")
                destination.Store3(0, invalid ? uint3(0, 0, 0) : counts);
            }
            """
        let dxil = try compiler!.compile(hlsl, entryPoint: "validate", profile: "cs_6_0")
        var parameters = [
            D3D12RootParameter(parameterType: D3D12.rootParameterSRV, shaderRegister: 0, registerSpace: 0),
            D3D12RootParameter(parameterType: D3D12.rootParameterUAV, shaderRegister: 0, registerSpace: 0),
        ]
        let (serialized, blob, errorBlob) = Self.serialize(library, parameters: &parameters)
        withExtendedLifetime(errorBlob) {}
        try d3dCheck(serialized, "root signature serialization")
        let root = try makeRootSignature(blob!)
        typealias CreatePipeline = @convention(c) (UnsafeMutableRawPointer, UnsafeRawPointer, UnsafeRawPointer, UnsafeMutablePointer<UnsafeMutableRawPointer?>) -> HRESULT
        var stateIID = IID.pipelineState
        let (created, state) = Self.createState(device, root: root, dxil: dxil, stateIID: &stateIID, create: device.slot(11, CreatePipeline.self))
        try d3dCheck(created, "validator pipeline creation")
        let model = ComputePipeline(id: ResourceID(device: 0, serial: 0), storageBindings: [], uniformBindings: [])
        let pipeline = Pipeline(model: model, rootSignature: root, state: state!)
        if mesh { meshValidator = pipeline } else { validator = pipeline }
        return pipeline
    }

    func makeRootSignature(_ blob: COMObject) throws -> COMObject {
        typealias BufferPointer = @convention(c) (UnsafeMutableRawPointer) -> UnsafeRawPointer?
        typealias BufferSize = @convention(c) (UnsafeMutableRawPointer) -> Int
        typealias CreateRoot = @convention(c) (UnsafeMutableRawPointer, UInt32, UnsafeRawPointer?, Int, UnsafeRawPointer,
                                               UnsafeMutablePointer<UnsafeMutableRawPointer?>) -> HRESULT
        var iid = IID.rootSignature
        var object: UnsafeMutableRawPointer?
        try d3dCheck(device.slot(16, CreateRoot.self)(device.pointer, 0, blob.slot(3, BufferPointer.self)(blob.pointer),
                                                      blob.slot(4, BufferSize.self)(blob.pointer), &iid, &object), "root signature creation")
        return COMObject(object!)
    }

    /// The command signature for a plain `Dispatch` argument.
    func dispatchSignature() throws -> COMObject {
        if let dispatchCommandSignature { return dispatchCommandSignature }
        typealias CreateSignature = @convention(c) (UnsafeMutableRawPointer, UnsafeRawPointer, UnsafeMutableRawPointer?, UnsafeRawPointer,
                                                    UnsafeMutablePointer<UnsafeMutableRawPointer?>) -> HRESULT
        let (result, created) = Self.makeSignature(device, argumentType: 2, stride: 12, create: device.slot(41, CreateSignature.self))
        try d3dCheck(result, "command signature creation")
        let signature = created!
        dispatchCommandSignature = signature
        return signature
    }

    /// The command signature for a `DispatchMesh` argument.
    func meshSignature() throws -> COMObject {
        if let meshCommandSignature { return meshCommandSignature }
        typealias CreateSignature = @convention(c) (UnsafeMutableRawPointer, UnsafeRawPointer, UnsafeMutableRawPointer?, UnsafeRawPointer,
                                                    UnsafeMutablePointer<UnsafeMutableRawPointer?>) -> HRESULT
        let (result, created) = Self.makeSignature(device, argumentType: 10, stride: 12, create: device.slot(41, CreateSignature.self))
        try d3dCheck(result, "command signature creation")
        meshCommandSignature = created!
        return created!
    }

    /// The command signature for `DrawInstanced` (16-byte) or `DrawIndexedInstanced` (20-byte) arguments.
    func drawSignature(indexed: Bool) throws -> COMObject {
        if let existing = indexed ? drawIndexedCommandSignature : drawCommandSignature { return existing }
        typealias CreateSignature = @convention(c) (UnsafeMutableRawPointer, UnsafeRawPointer, UnsafeMutableRawPointer?, UnsafeRawPointer,
                                                    UnsafeMutablePointer<UnsafeMutableRawPointer?>) -> HRESULT
        let (result, created) = Self.makeSignature(device, argumentType: indexed ? 1 : 0, stride: indexed ? 20 : 16,
                                                   create: device.slot(41, CreateSignature.self))
        try d3dCheck(result, "command signature creation")
        if indexed { drawIndexedCommandSignature = created } else { drawCommandSignature = created }
        return created!
    }
}
#endif

#if os(Windows)
private struct IndirectArgumentDesc { var type: UInt32; var padding: (UInt32, UInt32, UInt32) = (0, 0, 0) }
private struct CommandSignatureDesc {
    var byteStride: UInt32 = 12
    var count: UInt32 = 1
    var arguments: UnsafeRawPointer?
    var nodeMask: UInt32 = 0
}

extension D3D12Device {
    nonisolated static func makeSignature(
        _ device: COMObject, argumentType: UInt32, stride: UInt32,
        create: @convention(c) (UnsafeMutableRawPointer, UnsafeRawPointer, UnsafeMutableRawPointer?, UnsafeRawPointer,
                                UnsafeMutablePointer<UnsafeMutableRawPointer?>) -> HRESULT
    ) -> (HRESULT, COMObject?) {
        var argument = IndirectArgumentDesc(type: argumentType)
        var iid = IID.commandSignature
        var object: UnsafeMutableRawPointer?
        let result: HRESULT = withUnsafePointer(to: &argument) { argumentPointer in
            var description = CommandSignatureDesc(byteStride: stride, arguments: UnsafeRawPointer(argumentPointer))
            return create(device.pointer, &description, nil, &iid, &object)
        }
        return (result, object.map(COMObject.init))
    }
}
#endif
