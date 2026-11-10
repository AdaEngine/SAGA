import SAGAShaderIR
import SAGAValidation

/// Emits a Metal mesh entry point from a validated, specialized module.
package struct MeshMetalCodeGenerator {
    package init() {}

    package func generate(_ validated: ValidatedModule<IRModule>, entryPoint: String, bindingArrayLengths: [IRResourceBinding: UInt32] = [:]) throws -> ShaderArtifact {
        try generateReflected(validated, entryPoint: entryPoint, bindingArrayLengths: bindingArrayLengths).artifact
    }

    package func generateReflected(_ validated: ValidatedModule<IRModule>, entryPoint: String, bindingArrayLengths: [IRResourceBinding: UInt32] = [:]) throws -> (artifact: ShaderArtifact, uniforms: [ComputeShaderArtifact.UniformBinding], buffers: [ComputeShaderArtifact.StorageBinding], textures: [ComputeShaderArtifact.SampledTextureBinding], samplers: [ComputeShaderArtifact.SamplerBinding], storageTextures: [ComputeShaderArtifact.StorageTextureBinding], requiresSubgroups: Bool) {
        let module = validated.module
        guard let entry = module.entryPoints.first(where: { $0.name == entryPoint && $0.stage == .mesh }),
              let info = entry.meshInfo, entry.workgroupOverrides == nil, entry.subgroupSize == nil,
              module.overrides.count == 0, info.maxVerticesOverride == nil, info.maxPrimitivesOverride == nil else {
            throw ShaderFailure.unsupported("Mesh entry point or unspecialized metadata")
        }
        let function = try module.functions.value(for: entry.function)
        let threads = try meshWorkgroupThreadCount(entry.workgroupSize)
        var helpers = try ShaderFunctions(module: module, entry: entry.function)
        let sharedIndices = Set(helpers.reachableGlobals.filter { module.globalVariables.values[$0].space == .workgroup && $0 != info.outputVariable.index })
        let allowed = Set([info.outputVariable.index] + (entry.taskPayload.map { [$0.index] } ?? [])).union(sharedIndices)
        guard !helpers.usesCooperativeMatrix else {
            throw ShaderFailure.unsupported("Metal mesh resources, shared memory, or subgroups")
        }
        let bindings = try MeshBufferBindings(module: module, entry: entry, helpers: &helpers, excluded: allowed, bindingArrayLengths: bindingArrayLengths)
        var inputs = try ComputeInputs(arguments: function.arguments, workgroupSize: entry.workgroupSize, module: module, format: .msl)
        guard !inputs.requiresLinearIndexing else { throw ShaderFailure.unsupported("Metal mesh input") }
        let lane = inputs.workgroupInitializationIndex()
        let output = try module.globalVariables.value(for: info.outputVariable)
        var types: Set<IRHandle<IRType>> = [output.type]
        types.formUnion(sharedIndices.map { module.globalVariables.values[$0].type })
        if let payload = entry.taskPayload { types.insert(try module.globalVariables.value(for: payload).type) }
        var structures = inputs.structures
        var declarations = inputs.declarations + bindings.declarations
        let payloadTypes: Set<IRHandle<IRType>>
        if let payload = entry.taskPayload {
            payloadTypes = try helpers.metalPayloadTypes(module.globalVariables.value(for: payload).type, module: module)
        } else {
            payloadTypes = []
        }
        declarations += try helpers.compositeDeclarations(module: module, entry: entry.function, format: .msl, additionalTypes: types, metalPayloadTypes: payloadTypes, structures: &structures)
        let shared = try MeshWorkgroupMemory(module: module, indices: sharedIndices, structures: structures)
        let outputType = try shaderValueType(output.type, module: module, format: .msl, structures: structures)
        guard case .structure(let outputMembers, _) = try module.types.value(for: output.type).kind,
              let outputLayout = structures[output.type] else { throw ShaderFailure.unsupported("Metal mesh output") }
        func member(_ builtin: IRBuiltIn) -> String? {
            outputMembers.firstIndex(where: { $0.binding == .builtIn(builtin) }).map { "saga_output.\(outputLayout.memberName($0))" }
        }
        guard let vertices = member(.vertices), let primitives = member(.primitives) else { throw ShaderFailure.unsupported("Metal mesh output arrays") }
        guard let vertexCountMember = member(.vertexCount), let primitiveCountMember = member(.primitiveCount) else {
            throw ShaderFailure.unsupported("Metal mesh output counts")
        }
        let vertexCount = "metal::min(\(vertexCountMember), \(info.maxVertices)u)"
        let primitiveCount = "metal::min(\(primitiveCountMember), \(info.maxPrimitives)u)"
        var storage = bindings.storage
        var captures = bindings.captures
        for (index, variable) in shared.variables {
            storage[index] = .init(reference: variable.reference, bound: variable.field?.count.map { "\($0)u" }, writable: true)
            captures[index] = [.init(type: "threadgroup \(variable.type)*", name: variable.name)]
        }
        storage[info.outputVariable.index] = .init(reference: "saga_output", writable: true)
        captures[info.outputVariable.index] = [.init(type: "threadgroup \(outputType)&", name: "saga_output")]
        var extraParameters = ["threadgroup \(outputType)& saga_output"]
        var outerParameters: [String] = []
        var extraArguments = ["saga_output"]
        var capturedNames: Set<String> = []
        for index in helpers.reachableGlobals.subtracting(allowed).sorted() {
            for capture in captures[index] where capturedNames.insert(capture.name).inserted {
                extraParameters.append("\(capture.type) \(capture.name)")
                extraArguments.append(capture.name)
            }
        }
        for (_, variable) in shared.variables {
            extraParameters.append("threadgroup \(variable.type)* \(variable.name)")
            extraArguments.append(variable.name)
        }
        outerParameters += bindings.parameters
        if let payload = entry.taskPayload {
            let global = try module.globalVariables.value(for: payload)
            let type = try shaderValueType(global.type, module: module, format: .msl, structures: structures)
            storage[payload.index] = .init(reference: "saga_payload", writable: false)
            captures[payload.index] = [.init(type: "const object_data \(type)&", name: "saga_payload")]
            extraParameters.append("const object_data \(type)& saga_payload")
            outerParameters.append("const object_data \(type)& saga_payload [[payload]]")
            extraArguments.append("saga_payload")
        }
        try helpers.bindMetalStorage(captures)
        let helperLines = try helpers.declarations(module: module, format: .msl, storage: storage, structures: structures)
        let innerArgs = function.arguments.indices.map { "saga_inner_\($0)" }
        let innerParameters = try function.arguments.enumerated().map { index, arg in
            "\(try shaderValueType(arg.type, module: module, format: .msl, structures: structures)) \(innerArgs[index])"
        } + extraParameters
        var body = ShaderBodyWriter(module: module, function: function, arguments: innerArgs, storage: storage, outputFormat: .msl, resultTarget: .direct, structures: structures, callCaptures: helpers.callCaptures, pointerWrites: helpers.pointerWrites, budget: helpers.budget)
        let bodyLines = try body.write()
        let numeric = try integerPrelude(helpers.integerHelpers.union(body.integerHelpers), vectors: helpers.integerVectorHelpers.union(body.integerVectorHelpers), in: .msl)
        let vertex = try interface(info.vertexOutputType, name: "SagaMeshVertex", module: module, structures: structures, primitive: false)
        let primitive = try interface(info.primitiveOutputType, name: "SagaMeshPrimitive", module: module, structures: structures, primitive: true)
        declarations += vertex.declarations + primitive.declarations
        let topology: String
        let indexCount: Int
        switch info.topology { case .points: (topology, indexCount) = ("point", 1); case .lines: (topology, indexCount) = ("line", 2); case .triangles: (topology, indexCount) = ("triangle", 3) }
        guard let indexField = primitive.indexField else { throw ShaderFailure.unsupported("Metal mesh indices") }
        outerParameters = inputs.parameters + outerParameters + ["metal::mesh<SagaMeshVertex, SagaMeshPrimitive, \(info.maxVertices), \(info.maxPrimitives), metal::topology::\(topology)> saga_mesh"]
        let stride = threads
        let sharedSetup = try shared.setup(lane: lane, threads: threads, module: module, structures: structures)
        var outer = ["[[mesh]] void saga_main(\(outerParameters.joined(separator: ", "))) {", "    threadgroup \(outputType) saga_output;"] + sharedSetup + [ "    if (\(lane) == 0u) { saga_output = {}; }", "    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);"] + inputs.initializers
        outer += ["    saga_mesh_body(\((inputs.references + extraArguments).joined(separator: ", ")));", "    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);", "    uint saga_vertex_count = \(vertexCount);", "    uint saga_primitive_count = saga_vertex_count > 0u ? \(primitiveCount) : 0u;", "    for (uint i = \(lane); i < saga_vertex_count; i += \(stride)u) {", "        SagaMeshVertex value = {};"]
        outer += vertex.copies.map { "        value.\($0.0) = \(vertices)[i].\($0.1);" }
        outer += ["        saga_mesh.set_vertex(i, value);", "    }", "    for (uint i = \(lane); i < saga_primitive_count; i += \(stride)u) {"]
        if !primitive.copies.isEmpty { outer.append("        SagaMeshPrimitive value = {};") }
        for i in 0..<indexCount {
            let suffix = indexCount == 1 ? "" : ".\(["x", "y", "z"][i])"
            outer.append("        uint saga_index_\(i) = \(primitives)[i].\(indexField)\(suffix);")
            outer.append("        saga_mesh.set_index(i * \(indexCount)u + \(i)u, metal::min(saga_index_\(i), saga_vertex_count - 1u));")
        }
        outer += primitive.copies.map { "        value.\($0.0) = \(primitives)[i].\($0.1);" }
        if !primitive.copies.isEmpty { outer.append("        saga_mesh.set_primitive(i, value);") }
        outer += ["    }", "    if (\(lane) == 0u) { saga_mesh.set_primitive_count(saga_primitive_count); }", "}", ""]
        let inner = ["void saga_mesh_body(\(innerParameters.joined(separator: ", "))) {"] + bodyLines + ["}"]
        return (.text(format: .msl, source: (["#include <metal_stdlib>", "using namespace metal;"] + declarations + numeric + helperLines + inner + outer).joined(separator: "\n")), bindings.uniforms, bindings.buffers, bindings.textures, bindings.samplers, bindings.storageTextures, inputs.usesSubgroups || !helpers.subgroupExtensions.isEmpty || helpers.controlBarriers.contains(.subgroup))
    }

    private func interface(_ handle: IRHandle<IRType>, name: String, module: IRModule, structures: [IRHandle<IRType>: ShaderStructure], primitive: Bool) throws -> (declarations: [String], copies: [(String, String)], indexField: String?) {
        guard case .structure(let members, _) = try module.types.value(for: handle).kind, let layout = structures[handle] else { throw ShaderFailure.unsupported("Metal mesh interface") }
        var declarations = ["struct \(name) {"]
        var copies: [(String, String)] = []
        var indexField: String?
        for (index, member) in members.enumerated() {
            let attribute: String
            switch member.binding {
            case .builtIn(.pointIndex), .builtIn(.lineIndices), .builtIn(.triangleIndices): indexField = layout.memberName(index); continue
            case .builtIn(.position): guard !primitive else { throw ShaderFailure.unsupported("Metal mesh position") }; attribute = "position"
            case .builtIn(.primitiveIndex): guard primitive else { throw ShaderFailure.unsupported("Metal mesh primitive index") }; attribute = "primitive_id"
            case .builtIn(.cullPrimitive): guard primitive else { throw ShaderFailure.unsupported("Metal mesh cull") }; attribute = "primitive_culled"
            case .location(let location, let interpolation, let sampling, let blend, _):
                guard blend == nil else { throw ShaderFailure.unsupported("Metal mesh blend source") }
                if primitive {
                    guard interpolation == nil || interpolation == .flat,
                          sampling == nil || sampling == .first || sampling == .either else { throw ShaderFailure.unsupported("Metal mesh primitive interpolation") }
                    attribute = "user(locn\(location))"
                } else if let interpolation {
                    let mode: String
                    switch interpolation {
                    case .flat:
                        guard sampling == nil || sampling == .first || sampling == .either else { throw ShaderFailure.unsupported("Metal mesh flat sampling") }
                        mode = "flat"
                    case .perspective, .linear:
                        let sample = sampling ?? .center
                        guard sample == .center || sample == .centroid || sample == .sample else { throw ShaderFailure.unsupported("Metal mesh interpolation sampling") }
                        mode = "\(sample.rawValue)_\(interpolation == .linear ? "no_perspective" : "perspective")"
                    default: throw ShaderFailure.unsupported("Metal mesh interpolation")
                    }
                    attribute = "user(locn\(location)), \(mode)"
                } else {
                    guard sampling == nil else { throw ShaderFailure.unsupported("Metal mesh interpolation sampling") }
                    attribute = "user(locn\(location))"
                }
            default: throw ShaderFailure.unsupported("Metal mesh interface attribute")
            }
            let field = "saga_field_\(primitive ? "p" : "v")\(index)"
            let type = try shaderValueType(member.type, module: module, format: .msl, structures: structures)
            declarations.append("    \(type) \(field) [[\(attribute)]];")
            copies.append((field, layout.memberName(index)))
        }
        declarations.append("};")
        return (declarations, copies, indexField)
    }
}
