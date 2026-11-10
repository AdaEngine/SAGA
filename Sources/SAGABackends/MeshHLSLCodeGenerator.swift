import SAGAShaderIR
import SAGAValidation

/// Lowers a validated mesh entry to the HLSL mesh shader interface.
package struct MeshHLSLCodeGenerator {
    private let bindingArrayLengths: [IRResourceBinding: UInt32]
    package init(bindingArrayLengths: [IRResourceBinding: UInt32] = [:]) { self.bindingArrayLengths = bindingArrayLengths }

    package func generate(_ validated: ValidatedModule<IRModule>, entryPoint: String) throws -> ShaderArtifact {
        let module = validated.module
        guard let entry = module.entryPoints.first(where: { $0.name == entryPoint && $0.stage == .mesh }),
              let info = entry.meshInfo, entry.workgroupOverrides == nil, entry.subgroupSize == nil,
              info.maxVerticesOverride == nil, info.maxPrimitivesOverride == nil, module.overrides.count == 0 else {
            throw ShaderFailure.unsupported("Mesh entry point or unspecialized metadata")
        }
        let size = entry.workgroupSize
        guard size.x <= 128, size.y <= 128, size.z <= 64,
              UInt64(size.x) * UInt64(size.y) * UInt64(size.z) <= 128 else {
            throw ShaderFailure.unsupported("HLSL mesh workgroup size exceeds numthreads limits")
        }
        let topology: String
        let indexType: String
        let indexBinding: IRBuiltIn
        switch info.topology {
        case .triangles: (topology, indexType, indexBinding) = ("triangle", "uint3", .triangleIndices)
        case .lines: (topology, indexType, indexBinding) = ("line", "uint2", .lineIndices)
        case .points: throw ShaderFailure.unsupported("HLSL mesh point topology")
        }
        let function = try module.functions.value(for: entry.function)
        let output = try module.globalVariables.value(for: info.outputVariable)
        guard case .structure(let outputMembers, _) = try module.types.value(for: output.type).kind else {
            throw ShaderFailure.unsupported("Mesh output structure")
        }
        func member(_ builtin: IRBuiltIn) throws -> Int {
            guard let index = outputMembers.firstIndex(where: { $0.binding == .builtIn(builtin) }) else {
                throw ShaderFailure.unsupported("Mesh output missing \(builtin)")
            }
            return index
        }
        let vertices = try member(.vertices), primitives = try member(.primitives)
        let vertexCount = try member(.vertexCount), primitiveCount = try member(.primitiveCount)
        var helpers = try ShaderFunctions(module: module, entry: entry.function)
        let workgroupIndices = Set(helpers.reachableGlobals.filter {
            $0 != info.outputVariable.index && module.globalVariables.values[$0].space == .workgroup
        })
        let handles = Set(helpers.reachableGlobals.filter { module.globalVariables.values[$0].space == .handle })
        let permitted = workgroupIndices.union(handles).union([info.outputVariable.index]).union(entry.taskPayload.map { [$0.index] } ?? [])
        guard !helpers.usesCooperativeMatrix else {
            throw ShaderFailure.unsupported("Mesh resources, shared memory, or subgroups")
        }
        let bindings = try MeshHLSLBufferBindings(module: module, entry: entry, helpers: &helpers, excluded: permitted, bindingArrayLengths: bindingArrayLengths)
        let textures = try MeshHLSLTextureBindings(module: module, entry: entry, helpers: &helpers, bindingArrayLengths: bindingArrayLengths)
        var inputs = try ComputeInputs(arguments: function.arguments, workgroupSize: entry.workgroupSize, module: module, format: .hlsl)
        guard !inputs.requiresLinearIndexing, inputs.hlslWorkgroupCountBinding == nil else {
            throw ShaderFailure.unsupported("Mesh shader input")
        }
        let lane = inputs.workgroupInitializationIndex()
        let subgroupSetup = inputs.subgroupSetup()
        var structures = inputs.structures
        var declarations = inputs.declarations + bindings.declarations + textures.declarations + subgroupSetup.declarations
        var extraTypes = [output.type, info.vertexOutputType, info.primitiveOutputType] + workgroupIndices.map { module.globalVariables.values[$0].type }
        if let payload = entry.taskPayload { extraTypes.append(try module.globalVariables.value(for: payload).type) }
        declarations += try helpers.compositeDeclarations(module: module, entry: entry.function, format: .hlsl,
                                                          additionalTypes: Set(extraTypes), structures: &structures)
        let shared = try MeshHLSLWorkgroupMemory(module: module, indices: workgroupIndices, structures: structures)
        declarations += shared.declarations
        let outputType = try shaderValueType(output.type, module: module, format: .hlsl, structures: structures)
        declarations.append("groupshared \(outputType) saga_mesh_output;")
        var storage = bindings.storage
        for (index, value) in textures.storage { storage[index] = value }
        for (index, variable) in shared.variables {
            storage[index] = .init(reference: variable.reference, bound: variable.field?.count.map { "\($0)u" }, writable: true)
        }
        storage[info.outputVariable.index] = ShaderStorage(reference: "saga_mesh_output", writable: true)
        var payloadParameter: [String] = []
        var initializePayload: [String] = []
        if let payload = entry.taskPayload {
            let type = try shaderValueType(module.globalVariables.value(for: payload).type, module: module, format: .hlsl, structures: structures)
            declarations.append("groupshared \(type) saga_mesh_payload;")
            storage[payload.index] = ShaderStorage(reference: "saga_mesh_payload", writable: false)
            payloadParameter = ["in payload \(type) saga_input_payload"]
            initializePayload = ["        saga_mesh_payload = saga_input_payload;"]
        }
        func interface(_ type: IRHandle<IRType>, name: String, indexBuiltin: IRBuiltIn?) throws -> (String, [String], Int?) {
            guard case .structure(let members, _) = try module.types.value(for: type).kind else { throw ShaderFailure.unsupported("Mesh interface structure") }
            var lines = ["struct \(name) {"]
            var copies: [String] = []
            var indices: Int?
            for (index, member) in members.enumerated() {
                if member.binding == indexBuiltin.map({ .builtIn($0) }) { indices = index; continue }
                let semantic: String
                var qualifier = ""
                switch member.binding {
                case .builtIn(.position): semantic = "SV_Position"
                case .location(let location, let interpolation, let sampling, nil, let perPrimitive):
                    semantic = "TEXCOORD\(location)"
                    if perPrimitive || indexBuiltin != nil {
                        guard interpolation == nil, sampling == nil else {
                            throw ShaderFailure.unsupported("HLSL mesh primitive interpolation")
                        }
                    } else {
                        switch interpolation {
                        case nil, .perspective, .linear:
                            guard sampling == nil || sampling == .center || sampling == .centroid || sampling == .sample else {
                                throw ShaderFailure.unsupported("HLSL mesh interpolation sampling")
                            }
                            if interpolation == .linear { qualifier += "noperspective " }
                            if sampling == .centroid { qualifier += "centroid " }
                            if sampling == .sample { qualifier += "sample " }
                        case .flat:
                            guard sampling == nil || sampling == .first || sampling == .either else {
                                throw ShaderFailure.unsupported("HLSL mesh flat sampling")
                            }
                            qualifier = "nointerpolation "
                        case .perVertex: throw ShaderFailure.unsupported("HLSL mesh interpolation")
                        }
                    }
                case .builtIn(.primitiveIndex): semantic = "SV_PrimitiveID"
                case .builtIn(.cullPrimitive): semantic = "SV_CullPrimitive"
                default: throw ShaderFailure.unsupported("HLSL mesh output binding")
                }
                let fieldType = try shaderValueType(member.type, module: module, format: .hlsl, structures: structures)
                lines.append("    \(qualifier)\(fieldType) saga_field_\(index) : \(semantic);")
                copies.append("saga_field_\(index)")
            }
            if copies.isEmpty { lines.append("    uint saga_unused : TEXCOORD31;") }
            lines.append("};")
            return (lines.joined(separator: "\n"), copies, indices)
        }
        let (vertexDecl, vertexFields, _) = try interface(info.vertexOutputType, name: "SagaMeshVertex", indexBuiltin: nil)
        let (primitiveDecl, primitiveFields, indexField) = try interface(info.primitiveOutputType, name: "SagaMeshPrimitive", indexBuiltin: indexBinding)
        guard let indexField else { throw ShaderFailure.unsupported("Mesh primitive indices") }
        declarations += [vertexDecl, primitiveDecl]
        let helperLines = try helpers.declarations(module: module, format: .hlsl, storage: storage, structures: structures)
        let innerArgs = function.arguments.indices.map { "saga_inner_\($0)" }
        let innerParams = try function.arguments.enumerated().map { index, arg in
            "\(try shaderValueType(arg.type, module: module, format: .hlsl, structures: structures)) \(innerArgs[index])"
        }
        var body = ShaderBodyWriter(module: module, function: function, arguments: innerArgs, storage: storage,
                                    outputFormat: .hlsl, structures: structures, callCaptures: helpers.callCaptures,
                                    pointerWrites: helpers.pointerWrites, budget: helpers.budget)
        let bodyLines = try body.write()
        let numeric = try integerPrelude(helpers.integerHelpers.union(body.integerHelpers), vectors: helpers.integerVectorHelpers.union(body.integerVectorHelpers), in: .hlsl)
        let inner = ["void saga_mesh_body(\(innerParams.joined(separator: ", "))) {"] + bodyLines + ["}"]
        let outputStruct = structures[output.type]!
        let vertexStruct = structures[info.vertexOutputType]!
        let primitiveStruct = structures[info.primitiveOutputType]!
        let vCount = "saga_mesh_output.\(outputStruct.memberName(vertexCount))"
        let pCount = "saga_mesh_output.\(outputStruct.memberName(primitiveCount))"
        let vArray = "saga_mesh_output.\(outputStruct.memberName(vertices)).elements[saga_vertex_index]"
        let pArray = "saga_mesh_output.\(outputStruct.memberName(primitives)).elements[saga_primitive_index]"
        let params = inputs.parameters + ["out indices \(indexType) saga_indices[\(info.maxPrimitives)]", "out vertices SagaMeshVertex saga_vertices[\(info.maxVertices)]", "out primitives SagaMeshPrimitive saga_primitives[\(info.maxPrimitives)]"] + payloadParameter
        let stride = UInt64(size.x) * UInt64(size.y) * UInt64(size.z)
        var outer = ["[outputtopology(\"\(topology)\")]", "[numthreads(\(size.x), \(size.y), \(size.z))]", "void saga_main(\(params.joined(separator: ", "))) {", "    if (\(lane) == 0u) {", "        saga_mesh_output = (\(outputType))0;"] + initializePayload + ["    }"]
        outer += try shared.initialize(lane: lane, threads: meshWorkgroupThreadCount(size), module: module, structures: structures)
        outer += ["    GroupMemoryBarrierWithGroupSync();"] + subgroupSetup.body + inputs.initializers + ["    saga_mesh_body(\(inputs.references.joined(separator: ", ")));", "    GroupMemoryBarrierWithGroupSync();", "    uint saga_vertex_count = min(\(vCount), \(info.maxVertices)u);", "    uint saga_primitive_count = saga_vertex_count == 0u ? 0u : min(\(pCount), \(info.maxPrimitives)u);", "    SetMeshOutputCounts(saga_vertex_count, saga_primitive_count);", "    for (uint saga_vertex_index = \(lane); saga_vertex_index < saga_vertex_count; saga_vertex_index += \(stride)u) {"]
        for field in vertexFields { let index = Int(field.dropFirst("saga_field_".count))!; outer.append("        saga_vertices[saga_vertex_index].\(field) = \(vArray).\(vertexStruct.memberName(index));") }
        outer += ["    }", "    for (uint saga_primitive_index = \(lane); saga_primitive_index < saga_primitive_count; saga_primitive_index += \(stride)u) {", "        saga_indices[saga_primitive_index] = min(\(pArray).\(primitiveStruct.memberName(indexField)), (\(indexType))(saga_vertex_count - 1u));"]
        for field in primitiveFields { let index = Int(field.dropFirst("saga_field_".count))!; outer.append("        saga_primitives[saga_primitive_index].\(field) = \(pArray).\(primitiveStruct.memberName(index));") }
        outer += ["    }", "}", ""]
        return .text(format: .hlsl, source: (declarations + numeric + helperLines + inner + outer).joined(separator: "\n"))
    }
}
