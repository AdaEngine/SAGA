import SAGAShaderIR
import SAGAValidation

/// Lowers the supported mesh interface to GL_EXT_mesh_shader.
package struct MeshGLSLCodeGenerator {
    private let bindingArrayLengths: [IRResourceBinding: UInt32]
    package init(bindingArrayLengths: [IRResourceBinding: UInt32] = [:]) { self.bindingArrayLengths = bindingArrayLengths }

    package func generate(_ validated: ValidatedModule<IRModule>, entryPoint: String) throws -> ShaderArtifact {
        let module = validated.module
        guard let entry = module.entryPoints.first(where: { $0.name == entryPoint && $0.stage == .mesh }),
              let info = entry.meshInfo, entry.workgroupOverrides == nil, entry.subgroupSize == nil,
              info.maxVerticesOverride == nil, info.maxPrimitivesOverride == nil, module.overrides.count == 0 else {
            throw ShaderFailure.unsupported("GLSL mesh entry point or unspecialized metadata")
        }
        let function = try module.functions.value(for: entry.function)
        let output = try module.globalVariables.value(for: info.outputVariable)
        guard case .structure(let outputMembers, _) = try module.types.value(for: output.type).kind else {
            throw ShaderFailure.unsupported("GLSL mesh output structure")
        }
        func member(_ builtin: IRBuiltIn) throws -> Int {
            guard let index = outputMembers.firstIndex(where: { $0.binding == .builtIn(builtin) }) else {
                throw ShaderFailure.unsupported("GLSL mesh output missing \(builtin)")
            }
            return index
        }
        let vertices = try member(.vertices), primitives = try member(.primitives)
        let vertexCount = try member(.vertexCount), primitiveCount = try member(.primitiveCount)
        let topology: String, indexName: String
        let indexBinding: IRBuiltIn
        switch info.topology {
        case .triangles: (topology, indexName, indexBinding) = ("triangles", "gl_PrimitiveTriangleIndicesEXT", .triangleIndices)
        case .lines: (topology, indexName, indexBinding) = ("lines", "gl_PrimitiveLineIndicesEXT", .lineIndices)
        case .points: (topology, indexName, indexBinding) = ("points", "gl_PrimitivePointIndicesEXT", .pointIndex)
        }
        var helpers = try ShaderFunctions(module: module, entry: entry.function)
        guard !helpers.usesCooperativeMatrix else {
            throw ShaderFailure.unsupported("GLSL mesh cooperative matrices")
        }
        let workgroupIndices = Set(helpers.reachableGlobals.filter {
            $0 != info.outputVariable.index && module.globalVariables.values[$0].space == .workgroup
        })
        let handleIndices = Set(helpers.reachableGlobals.filter { module.globalVariables.values[$0].space == .handle })
        let payloadIndices = Set(entry.taskPayload.map { [$0.index] } ?? [])
        let bindings = try MeshGLSLBufferBindings(module: module, entry: entry, helpers: &helpers,
                                                  excluded: workgroupIndices.union(handleIndices).union(payloadIndices).union([info.outputVariable.index]),
                                                  bindingArrayLengths: bindingArrayLengths)
        let textures = try MeshGLSLTextureBindings(module: module, entry: entry, helpers: &helpers, bindingArrayLengths: bindingArrayLengths)
        var inputs = try ComputeInputs(arguments: function.arguments, workgroupSize: entry.workgroupSize, module: module, format: .glsl)
        guard !inputs.requiresLinearIndexing else { throw ShaderFailure.unsupported("GLSL mesh input") }
        let requiresSubgroups = inputs.usesSubgroups || !helpers.subgroupExtensions.isEmpty ||
            helpers.controlBarriers.contains(.subgroup)
        let lane = inputs.workgroupInitializationIndex()
        var structures = inputs.structures
        var declarations = inputs.declarations + bindings.declarations + textures.declarations
        let additionalTypes = Set([output.type, info.vertexOutputType, info.primitiveOutputType] + (workgroupIndices.union(payloadIndices)).map { module.globalVariables.values[$0].type })
        declarations += try helpers.compositeDeclarations(module: module, entry: entry.function, format: .glsl,
            additionalTypes: additionalTypes, structures: &structures)
        let shared = try MeshGLSLWorkgroupMemory(module: module, indices: workgroupIndices, structures: structures)
        declarations += shared.declarations
        if let payload = entry.taskPayload {
            let type = try shaderValueType(module.globalVariables.value(for: payload).type, module: module, format: .glsl, structures: structures)
            declarations.append("taskPayloadSharedEXT \(type) saga_payload;")
        }
        let outputType = try shaderValueType(output.type, module: module, format: .glsl, structures: structures)
        declarations.append("shared \(outputType) saga_mesh_output;")
        func zero(_ type: IRHandle<IRType>) throws -> String {
            switch try module.types.value(for: type).kind {
            case .scalar(let scalar):
                switch scalar.kind {
                case .bool: return "false"
                case .uint: return scalar.width == 2 ? "uint16_t(0u)" : (scalar.width == 8 ? "uint64_t(0u)" : "0u")
                case .sint: return scalar.width == 2 ? "int16_t(0)" : (scalar.width == 8 ? "int64_t(0)" : "0")
                case .float: return scalar.width == 2 ? "float16_t(0.0)" : (scalar.width == 8 ? "double(0.0)" : "0.0")
                default: throw ShaderFailure.unsupported("GLSL mesh zero scalar")
                }
            case .vector(_, let scalar):
                return "\(try shaderValueType(type, module: module, format: .glsl, structures: structures))(\(scalar.kind == .bool ? "false" : "0"))"
            case .matrix:
                return "\(try shaderValueType(type, module: module, format: .glsl, structures: structures))(0)"
            case .array(let base, .constant, _): return "saga_zero_array_\(type.index)(\(try zero(base)))"
            case .structure(let members, _):
                guard let layout = structures[type] else { throw ShaderFailure.unsupported("GLSL mesh zero structure") }
                return "\(layout.constructor)(\(try members.map { try zero($0.type) }.joined(separator: ", ")))"
            default: throw ShaderFailure.unsupported("GLSL mesh zero value")
            }
        }
        let initial = try zero(output.type)
        var storage = bindings.storage
        if let payload = entry.taskPayload { storage[payload.index] = .init(reference: "saga_payload", writable: false) }
        for (index, value) in textures.storage { storage[index] = value }
        for (index, variable) in shared.variables {
            storage[index] = .init(reference: variable.reference, bound: variable.field?.count.map { "\($0)u" }, writable: true)
        }
        storage[info.outputVariable.index] = ShaderStorage(reference: "saga_mesh_output", writable: true)
        let helperLines = try helpers.declarations(module: module, format: .glsl, storage: storage, structures: structures)
        let args = function.arguments.indices.map { "saga_inner_\($0)" }
        let params = try function.arguments.enumerated().map { index, arg in
            "\(try shaderValueType(arg.type, module: module, format: .glsl, structures: structures)) \(args[index])"
        }
        var body = ShaderBodyWriter(module: module, function: function, arguments: args, storage: storage,
                                    outputFormat: .glsl, structures: structures, callCaptures: helpers.callCaptures,
                                    pointerWrites: helpers.pointerWrites, budget: helpers.budget)
        let bodyLines = try body.write()
        let numeric = try integerPrelude(helpers.integerHelpers.union(body.integerHelpers), vectors: helpers.integerVectorHelpers.union(body.integerVectorHelpers), in: .glsl)
        guard let outputLayout = structures[output.type], let vertexLayout = structures[info.vertexOutputType],
              let primitiveLayout = structures[info.primitiveOutputType],
              case .structure(let vertexMembers, _) = try module.types.value(for: info.vertexOutputType).kind,
              case .structure(let primitiveMembers, _) = try module.types.value(for: info.primitiveOutputType).kind else {
            throw ShaderFailure.unsupported("GLSL mesh interface")
        }
        let vArray = "saga_mesh_output.\(outputLayout.memberName(vertices)).elements[saga_vertex_index]"
        let pArray = "saga_mesh_output.\(outputLayout.memberName(primitives)).elements[saga_primitive_index]"
        var vertexCopies: [String] = [], primitiveCopies: [String] = []
        var hasIndices = false
        for (index, field) in vertexMembers.enumerated() {
            let target: String
            switch field.binding {
            case .builtIn(.position): target = "gl_MeshVerticesEXT[saga_vertex_index].gl_Position"
            case .location(let location, let interpolation, let sampling, nil, false):
                let type = try shaderValueType(field.type, module: module, format: .glsl, structures: structures)
                let qualifier: String
                switch interpolation {
                case .flat: qualifier = "flat "
                case .linear, .perspective, nil:
                    guard sampling == nil || sampling == .center || sampling == .centroid || sampling == .sample else {
                        throw ShaderFailure.unsupported("GLSL mesh interpolation sampling")
                    }
                    qualifier = (interpolation == .linear ? "noperspective " : "") +
                        (sampling == .centroid ? "centroid " : (sampling == .sample ? "sample " : ""))
                default: throw ShaderFailure.unsupported("GLSL mesh interpolation")
                }
                declarations.append("layout(location = \(location)) \(qualifier)out \(type) saga_vertex_\(index)[];")
                target = "saga_vertex_\(index)[saga_vertex_index]"
            default: throw ShaderFailure.unsupported("GLSL mesh vertex output binding")
            }
            vertexCopies.append("        \(target) = \(vArray).\(vertexLayout.memberName(index));")
        }
        for (index, field) in primitiveMembers.enumerated() {
            let target: String
            switch field.binding {
            case .builtIn(let builtin) where builtin == indexBinding:
                hasIndices = true
                let type = try shaderValueType(field.type, module: module, format: .glsl, structures: structures)
                primitiveCopies.append("        \(indexName)[saga_primitive_index] = min(\(pArray).\(primitiveLayout.memberName(index)), \(type)(saga_vertex_count - 1u));")
                continue
            case .builtIn(.cullPrimitive): target = "gl_MeshPrimitivesEXT[saga_primitive_index].gl_CullPrimitiveEXT"
            case .builtIn(.primitiveIndex):
                primitiveCopies.append("        gl_MeshPrimitivesEXT[saga_primitive_index].gl_PrimitiveID = int(\(pArray).\(primitiveLayout.memberName(index))); ")
                continue
            case .location(let location, nil, nil, nil, _):
                let type = try shaderValueType(field.type, module: module, format: .glsl, structures: structures)
                declarations.append("layout(location = \(location)) perprimitiveEXT out \(type) saga_primitive_\(index)[];")
                target = "saga_primitive_\(index)[saga_primitive_index]"
            default: throw ShaderFailure.unsupported("GLSL mesh primitive output binding")
            }
            primitiveCopies.append("        \(target) = \(pArray).\(primitiveLayout.memberName(index));")
        }
        guard hasIndices else { throw ShaderFailure.unsupported("GLSL mesh primitive indices") }
        let size = entry.workgroupSize
        let stride = UInt64(size.x) * UInt64(size.y) * UInt64(size.z)
        var prefix: [String] = ["#version 450 core", "#extension GL_EXT_mesh_shader : require"]
        prefix.append(contentsOf: glslUnsigned64Extensions(module: module, storageTextures: textures.storageTextures))
        if module.types.values.contains(where: { $0.kind.usesFloat16 }) { prefix.append(glslFloat16Extension) }
        if moduleUsesInt16(module) { prefix.append(glslInt16Extension) }
        if bindings.usesGLSL16BitStorage { prefix.append(glslStorage16BitExtension) }
        prefix.append(contentsOf: helpers.glslExtensions)
        prefix.append(contentsOf: helpers.glslSubgroupExtensions(requiresSubgroups: requiresSubgroups))
        prefix.append("layout(local_size_x = \(size.x), local_size_y = \(size.y), local_size_z = \(size.z)) in;")
        prefix.append("layout(\(topology), max_vertices = \(info.maxVertices), max_primitives = \(info.maxPrimitives)) out;")
        let inner = ["void saga_mesh_body(\(params.joined(separator: ", "))) {"] + bodyLines + ["}"]
        var main = ["void main() {", "    if (\(lane) == 0u) { saga_mesh_output = \(initial); }"]
        main += try shared.initialize(lane: lane, threads: meshWorkgroupThreadCount(size), module: module, structures: structures)
        main += ["    barrier();"] + inputs.initializers
        main += ["    saga_mesh_body(\(inputs.references.joined(separator: ", ")));", "    barrier();",
                 "    uint saga_vertex_count = min(saga_mesh_output.\(outputLayout.memberName(vertexCount)), \(info.maxVertices)u);",
                 "    uint saga_primitive_count = saga_vertex_count == 0u ? 0u : min(saga_mesh_output.\(outputLayout.memberName(primitiveCount)), \(info.maxPrimitives)u);",
                 "    SetMeshOutputsEXT(saga_vertex_count, saga_primitive_count);",
                 "    for (uint saga_vertex_index = \(lane); saga_vertex_index < saga_vertex_count; saga_vertex_index += \(stride)u) {"]
        main += vertexCopies + ["    }", "    for (uint saga_primitive_index = \(lane); saga_primitive_index < saga_primitive_count; saga_primitive_index += \(stride)u) {"] + primitiveCopies + ["    }", "}", ""]
        return .text(format: .glsl, source: (prefix + declarations + numeric + helperLines + inner + main).joined(separator: "\n"))
    }
}
