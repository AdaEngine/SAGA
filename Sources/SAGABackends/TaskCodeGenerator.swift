import SAGAShaderIR
import SAGAValidation

/// Device limits used to validate task-to-mesh dispatches in generated HLSL.
public struct HLSLTaskDispatchLimits: Sendable, Equatable {
    public var maxMeshWorkgroupsPerDimension: UInt32
    public var maxMeshWorkgroupsTotal: UInt32

    public init(maxMeshWorkgroupsPerDimension: UInt32, maxMeshWorkgroupsTotal: UInt32) {
        self.maxMeshWorkgroupsPerDimension = maxMeshWorkgroupsPerDimension
        self.maxMeshWorkgroupsTotal = maxMeshWorkgroupsTotal
    }
}

/// Emits a native task shader from a validated, specialized module.
package struct TaskCodeGenerator {
    package let format: ShaderOutputFormat
    private let bindingArrayLengths: [IRResourceBinding: UInt32]
    private let hlslTaskDispatchLimits: HLSLTaskDispatchLimits?

    package init(format: ShaderOutputFormat, bindingArrayLengths: [IRResourceBinding: UInt32] = [:], hlslTaskDispatchLimits: HLSLTaskDispatchLimits? = nil) {
        self.format = format
        self.bindingArrayLengths = bindingArrayLengths
        self.hlslTaskDispatchLimits = hlslTaskDispatchLimits
    }

    package func generate(_ validated: ValidatedModule<IRModule>, entryPoint: String) throws -> ShaderArtifact {
        try generateReflected(validated, entryPoint: entryPoint).artifact
    }

    package func generateReflected(_ validated: ValidatedModule<IRModule>, entryPoint: String) throws -> (artifact: ShaderArtifact, uniforms: [ComputeShaderArtifact.UniformBinding], buffers: [ComputeShaderArtifact.StorageBinding], textures: [ComputeShaderArtifact.SampledTextureBinding], samplers: [ComputeShaderArtifact.SamplerBinding], storageTextures: [ComputeShaderArtifact.StorageTextureBinding], requiresSubgroups: Bool) {
        guard format == .msl || format == .hlsl else { throw ShaderFailure.unsupported("Task shader target") }
        let module = validated.module
        guard let entry = module.entryPoints.first(where: { $0.name == entryPoint && $0.stage == .task }),
              let payload = entry.taskPayload, entry.workgroupOverrides == nil, entry.subgroupSize == nil,
              module.overrides.count == 0 else { throw ShaderFailure.unsupported("Task entry point or unspecialized metadata") }
        let function = try module.functions.value(for: entry.function)
        let threads = try meshWorkgroupThreadCount(entry.workgroupSize)
        if format == .hlsl && threads > 128 {
            throw ShaderFailure.unsupported("HLSL task workgroup size exceeds numthreads limits")
        }
        guard function.result != nil else { throw ShaderFailure.unsupported("Task shader result") }
        var helpers = try ShaderFunctions(module: module, entry: entry.function)
        let sharedIndices = Set(helpers.reachableGlobals.filter { module.globalVariables.values[$0].space == .workgroup })
        guard !helpers.usesCooperativeMatrix else {
            throw ShaderFailure.unsupported("Task shader resources, shared memory, or subgroups")
        }
        let bindings: (declarations: [String], parameters: [String], captures: [[ShaderParameter]], storage: [ShaderStorage], uniforms: [ComputeShaderArtifact.UniformBinding], buffers: [ComputeShaderArtifact.StorageBinding], textures: [ComputeShaderArtifact.SampledTextureBinding], samplers: [ComputeShaderArtifact.SamplerBinding], storageTextures: [ComputeShaderArtifact.StorageTextureBinding])
        if format == .msl {
            let value = try MeshBufferBindings(module: module, entry: entry, helpers: &helpers, excluded: sharedIndices.union([payload.index]), bindingArrayLengths: bindingArrayLengths)
            bindings = (value.declarations, value.parameters, value.captures, value.storage, value.uniforms, value.buffers, value.textures, value.samplers, value.storageTextures)
        } else {
            let handles = Set(helpers.reachableGlobals.filter { module.globalVariables.values[$0].space == .handle })
            let value = try MeshHLSLBufferBindings(module: module, entry: entry, helpers: &helpers, excluded: handles.union(sharedIndices).union([payload.index]), bindingArrayLengths: bindingArrayLengths)
            let textures = try MeshHLSLTextureBindings(module: module, entry: entry, helpers: &helpers, bindingArrayLengths: bindingArrayLengths)
            var storage = value.storage
            for (index, binding) in textures.storage { storage[index] = binding }
            bindings = (value.declarations + textures.declarations, value.parameters, value.captures, storage,
                        value.uniforms, value.buffers, textures.textures, textures.samplers, textures.storageTextures)
        }
        let global = try module.globalVariables.value(for: payload)
        var inputs = try ComputeInputs(arguments: function.arguments, workgroupSize: entry.workgroupSize, module: module, format: format)
        guard !inputs.requiresLinearIndexing, format == .msl || inputs.hlslWorkgroupCountBinding == nil else { throw ShaderFailure.unsupported("Task shader input") }
        guard format != .hlsl || !helpers.controlBarriers.contains(.subgroup) else { throw ShaderFailure.unsupported("HLSL subgroup execution barriers") }
        let lane = inputs.workgroupInitializationIndex()
        let subgroupSetup = inputs.subgroupSetup()
        var structures = inputs.structures
        var declarations = inputs.declarations + bindings.declarations + subgroupSetup.declarations
        let sharedTypes = Set(sharedIndices.map { module.globalVariables.values[$0].type })
        let payloadTypes = format == .msl ? try helpers.metalPayloadTypes(global.type, module: module) : []
        declarations += try helpers.compositeDeclarations(module: module, entry: entry.function, format: format,
                                                          additionalTypes: sharedTypes.union([global.type]),
                                                          metalPayloadTypes: payloadTypes, structures: &structures)
        let shared = try MeshWorkgroupMemory(module: module, indices: format == .msl ? sharedIndices : [], structures: structures)
        let hlslShared = try MeshHLSLWorkgroupMemory(module: module, indices: format == .hlsl ? sharedIndices : [], structures: structures)
        declarations += hlslShared.declarations
        let payloadType = try shaderValueType(global.type, module: module, format: format, structures: structures)
        let payloadName = "saga_payload"
        var storage = bindings.storage
        var sharedCaptures = bindings.captures
        for (index, variable) in shared.variables {
            storage[index] = .init(reference: variable.reference, bound: variable.field?.count.map { "\($0)u" }, writable: true)
            sharedCaptures[index] = [.init(type: "threadgroup \(variable.type)*", name: variable.name)]
        }
        for (index, variable) in hlslShared.variables {
            storage[index] = .init(reference: variable.reference, bound: variable.field?.count.map { "\($0)u" }, writable: true)
        }
        storage[payload.index] = ShaderStorage(reference: payloadName, writable: true)
        if format == .msl {
            var captures = sharedCaptures
            captures[payload.index] = [.init(type: "object_data \(payloadType)&", name: payloadName)]
            try helpers.bindMetalStorage(captures)
        } else {
            declarations.append("groupshared \(payloadType) \(payloadName);")
            declarations.append("groupshared uint3 saga_dispatch_size;")
        }
        let helperLines = try helpers.declarations(module: module, format: format, storage: storage, structures: structures)
        let innerArgs = function.arguments.indices.map { "saga_inner_\($0)" }
        var captureNames: Set<String> = []
        let resourceCaptures = helpers.reachableGlobals.subtracting([payload.index]).sorted()
            .flatMap { sharedCaptures[$0] }.filter { captureNames.insert($0.name).inserted }
        let innerParameters = try function.arguments.enumerated().map { index, arg in
            "\(try shaderValueType(arg.type, module: module, format: format, structures: structures)) \(innerArgs[index])"
        } + (format == .msl ? ["object_data \(payloadType)& \(payloadName)"] + resourceCaptures.map { "\($0.type) \($0.name)" } : [])
        var body = ShaderBodyWriter(module: module, function: function, arguments: innerArgs, storage: storage,
                                    outputFormat: format, resultTarget: .direct, structures: structures, callCaptures: helpers.callCaptures,
                                    pointerWrites: helpers.pointerWrites, budget: helpers.budget)
        let bodyLines = try body.write().map { format == .msl ? taskObjectBarrier($0) : $0 }
        let numeric = try integerPrelude(helpers.integerHelpers.union(body.integerHelpers), vectors: helpers.integerVectorHelpers.union(body.integerVectorHelpers), in: format)
        let inner = ["uint3 saga_task_body(\(innerParameters.joined(separator: ", "))) {"] + bodyLines + ["}"]
        let size = entry.workgroupSize
        let outerParameters = inputs.parameters + (format == .msl ? ["metal::mesh_grid_properties saga_grid", "object_data \(payloadType)& \(payloadName) [[payload]]"] + bindings.parameters : [])
        let callArgs = inputs.references + (format == .msl ? [payloadName] + resourceCaptures.map(\.name) : [])
        let zero = format == .msl ? "{}" : "(\(payloadType))0"
        let barrier = format == .msl ? "metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);\n    metal::threadgroup_barrier(metal::mem_flags::mem_object_data);" : "GroupMemoryBarrierWithGroupSync();"
        // Metal's mesh grid limit depends on GPU family. Native clients can set
        // this macro at compilation; 1024 is the baseline for mesh-capable GPUs.
        let gridLimitMacro = TaskStageArtifact.metalMeshGridLimitMacro
        let header = format == .msl ? ["#include <metal_stdlib>", "using namespace metal;", "#ifndef \(gridLimitMacro)", "#define \(gridLimitMacro) 1024u", "#endif"] : []
        let signature = format == .msl ? "[[object]] void saga_main(\(outerParameters.joined(separator: ", "))) {" : "[numthreads(\(size.x), \(size.y), \(size.z))]\nvoid saga_main(\(outerParameters.joined(separator: ", "))) {"
        let dispatch = format == .msl ? """
        if (\(lane) == 0u) {
            if (saga_size.x > \(gridLimitMacro) || saga_size.y > \(gridLimitMacro) ||
                saga_size.z > \(gridLimitMacro) ||
                metal::mulhi(saga_size.x, saga_size.y) != 0u ||
                metal::mulhi(saga_size.x * saga_size.y, saga_size.z) != 0u ||
                saga_size.x * saga_size.y * saga_size.z > \(gridLimitMacro)) {
                saga_size = uint3(0u);
            }
            saga_grid.set_threadgroups_per_grid(saga_size);
        }
        """ : "DispatchMesh(saga_dispatch_size.x, saga_dispatch_size.y, saga_dispatch_size.z, \(payloadName));"
        let sharedSetup = try shared.setup(lane: lane, threads: threads, module: module, structures: structures)
            + hlslShared.initialize(lane: lane, threads: threads, module: module, structures: structures)
        var saveDispatch: [String] = []
        if format == .hlsl {
            if let limits = hlslTaskDispatchLimits {
                let perDimension = min(limits.maxMeshWorkgroupsPerDimension, 2 << 21)
                let total = limits.maxMeshWorkgroupsTotal
                saveDispatch += [
                    "    if (saga_size.x > \(perDimension)u || saga_size.y > \(perDimension)u || saga_size.z > \(perDimension)u ||",
                    "        (uint64_t)saga_size.x * (uint64_t)saga_size.y > \(total)u ||",
                    "        (uint64_t)saga_size.x * (uint64_t)saga_size.y * (uint64_t)saga_size.z > \(total)u) {",
                    "        saga_size = uint3(0u, 0u, 0u);",
                    "    }"
                ]
            }
            saveDispatch.append("    if (\(lane) == 0u) { saga_dispatch_size = saga_size; }")
        }
        let outer = [signature] + sharedSetup + [ "    if (\(lane) == 0u) { \(payloadName) = \(zero); }", "    \(barrier)"] + subgroupSetup.body + inputs.initializers + ["    uint3 saga_size = saga_task_body(\(callArgs.joined(separator: ", ")));"] + saveDispatch + ["    \(barrier)", "    \(dispatch)", "}", ""]
        return (.text(format: format, source: (header + declarations + numeric + helperLines.map { format == .msl ? taskObjectBarrier($0) : $0 } + inner + outer).joined(separator: "\n")), bindings.uniforms, bindings.buffers, bindings.textures, bindings.samplers, bindings.storageTextures, inputs.usesSubgroups || !helpers.subgroupExtensions.isEmpty || helpers.controlBarriers.contains(.subgroup))
    }
}

private func taskObjectBarrier(_ line: String) -> String {
    guard line.contains("threadgroup_barrier("), line.contains("mem_flags::mem_threadgroup") else { return line }
    return line + "\n    metal::threadgroup_barrier(metal::mem_flags::mem_object_data);"
}
