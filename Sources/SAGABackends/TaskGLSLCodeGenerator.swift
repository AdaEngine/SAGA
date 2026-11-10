import SAGAShaderIR
import SAGAValidation

/// Device limits used to validate task-to-mesh dispatches in generated GLSL.
public struct GLSLTaskDispatchLimits: Sendable, Equatable {
        public let maxMeshWorkgroupsPerDimension: UInt32
        public let maxMeshWorkgroupsTotal: UInt32

        public init(maxMeshWorkgroupsPerDimension: UInt32, maxMeshWorkgroupsTotal: UInt32) {
            self.maxMeshWorkgroupsPerDimension = maxMeshWorkgroupsPerDimension
            self.maxMeshWorkgroupsTotal = maxMeshWorkgroupsTotal
        }
}

/// Lowers the supported task interface to GL_EXT_mesh_shader.
package struct TaskGLSLCodeGenerator {
    private let bindingArrayLengths: [IRResourceBinding: UInt32]
    private let dispatchLimits: GLSLTaskDispatchLimits?
    package init(bindingArrayLengths: [IRResourceBinding: UInt32] = [:], dispatchLimits: GLSLTaskDispatchLimits? = nil) {
        self.bindingArrayLengths = bindingArrayLengths
        self.dispatchLimits = dispatchLimits
    }

    package func generate(_ validated: ValidatedModule<IRModule>, entryPoint: String) throws -> ShaderArtifact {
        let module = validated.module
        guard let entry = module.entryPoints.first(where: { $0.name == entryPoint && $0.stage == .task }),
              let payload = entry.taskPayload, entry.workgroupOverrides == nil, entry.subgroupSize == nil,
              module.overrides.count == 0 else {
            throw ShaderFailure.unsupported("GLSL task entry point or unspecialized metadata")
        }
        let function = try module.functions.value(for: entry.function)
        guard function.result != nil else { throw ShaderFailure.unsupported("GLSL task shader result") }
        let global = try module.globalVariables.value(for: payload)
        var helpers = try ShaderFunctions(module: module, entry: entry.function)
        guard !helpers.usesCooperativeMatrix else {
            throw ShaderFailure.unsupported("GLSL task cooperative matrix")
        }
        let workgroupIndices = Set(helpers.reachableGlobals.filter {
            $0 != payload.index && module.globalVariables.values[$0].space == .workgroup
        })
        let handleIndices = Set(helpers.reachableGlobals.filter { module.globalVariables.values[$0].space == .handle })
        let bindings = try MeshGLSLBufferBindings(module: module, entry: entry, helpers: &helpers,
                                                  excluded: workgroupIndices.union(handleIndices).union([payload.index]),
                                                  bindingArrayLengths: bindingArrayLengths)
        let textures = try MeshGLSLTextureBindings(module: module, entry: entry, helpers: &helpers, bindingArrayLengths: bindingArrayLengths)
        var inputs = try ComputeInputs(arguments: function.arguments, workgroupSize: entry.workgroupSize, module: module, format: .glsl)
        guard !inputs.requiresLinearIndexing else { throw ShaderFailure.unsupported("GLSL task input") }
        let requiresSubgroups = inputs.usesSubgroups || !helpers.subgroupExtensions.isEmpty ||
            helpers.controlBarriers.contains(.subgroup)
        let lane = inputs.workgroupInitializationIndex()
        var structures = inputs.structures
        var declarations = inputs.declarations + bindings.declarations + textures.declarations
        let additionalTypes = Set([global.type] + workgroupIndices.map { module.globalVariables.values[$0].type })
        declarations += try helpers.compositeDeclarations(module: module, entry: entry.function, format: .glsl,
                                                          additionalTypes: additionalTypes, structures: &structures)
        let shared = try MeshGLSLWorkgroupMemory(module: module, indices: workgroupIndices, structures: structures)
        declarations += shared.declarations
        let payloadType = try shaderValueType(global.type, module: module, format: .glsl, structures: structures)
        declarations.append("taskPayloadSharedEXT \(payloadType) saga_payload;")
        declarations.append("shared uvec3 saga_dispatch_size;")
        func zero(_ type: IRHandle<IRType>) throws -> String {
            switch try module.types.value(for: type).kind {
            case .scalar(let scalar):
                switch scalar.kind {
                case .bool: return "false"
                case .uint: return scalar.width == 2 ? "uint16_t(0u)" : (scalar.width == 8 ? "uint64_t(0u)" : "0u")
                case .sint: return scalar.width == 2 ? "int16_t(0)" : (scalar.width == 8 ? "int64_t(0)" : "0")
                case .float: return scalar.width == 2 ? "float16_t(0.0)" : (scalar.width == 8 ? "double(0.0)" : "0.0")
                default: throw ShaderFailure.unsupported("GLSL task zero scalar")
                }
            case .vector(_, let scalar):
                return "\(try shaderValueType(type, module: module, format: .glsl, structures: structures))(\(scalar.kind == .bool ? "false" : "0"))"
            case .matrix:
                return "\(try shaderValueType(type, module: module, format: .glsl, structures: structures))(0)"
            case .array(let base, .constant, _): return "saga_zero_array_\(type.index)(\(try zero(base)))"
            case .structure(let members, _):
                guard let layout = structures[type] else { throw ShaderFailure.unsupported("GLSL task zero structure") }
                return "\(layout.constructor)(\(try members.map { try zero($0.type) }.joined(separator: ", ")))"
            default: throw ShaderFailure.unsupported("GLSL task zero value")
            }
        }
        var storage = bindings.storage
        for (index, value) in textures.storage { storage[index] = value }
        storage[payload.index] = ShaderStorage(reference: "saga_payload", writable: true)
        for (index, variable) in shared.variables {
            storage[index] = .init(reference: variable.reference, bound: variable.field?.count.map { "\($0)u" }, writable: true)
        }
        let helperLines = try helpers.declarations(module: module, format: .glsl, storage: storage, structures: structures)
        let args = function.arguments.indices.map { "saga_inner_\($0)" }
        let params = try function.arguments.enumerated().map { index, argument in
            "\(try shaderValueType(argument.type, module: module, format: .glsl, structures: structures)) \(args[index])"
        }
        var body = ShaderBodyWriter(module: module, function: function, arguments: args, storage: storage,
                                    outputFormat: .glsl, resultTarget: .direct, structures: structures,
                                    callCaptures: helpers.callCaptures, pointerWrites: helpers.pointerWrites, budget: helpers.budget)
        let bodyLines = try body.write()
        let numeric = try integerPrelude(helpers.integerHelpers.union(body.integerHelpers), vectors: helpers.integerVectorHelpers.union(body.integerVectorHelpers), in: .glsl)
        let size = entry.workgroupSize
        var prefix: [String] = ["#version 450 core", "#extension GL_EXT_mesh_shader : require"]
        prefix.append(contentsOf: glslUnsigned64Extensions(module: module, storageTextures: textures.storageTextures))
        if module.types.values.contains(where: { $0.kind.usesFloat16 }) { prefix.append(glslFloat16Extension) }
        if moduleUsesInt16(module) { prefix.append(glslInt16Extension) }
        if bindings.usesGLSL16BitStorage { prefix.append(glslStorage16BitExtension) }
        prefix.append(contentsOf: helpers.glslExtensions)
        prefix.append(contentsOf: helpers.glslSubgroupExtensions(requiresSubgroups: requiresSubgroups))
        prefix.append("layout(local_size_x = \(size.x), local_size_y = \(size.y), local_size_z = \(size.z)) in;")
        let inner = ["uvec3 saga_task_body(\(params.joined(separator: ", "))) {"] + bodyLines + ["}"]
        var main = ["void main() {", "    if (\(lane) == 0u) { saga_payload = \(try zero(global.type)); }"]
        main += try shared.initialize(lane: lane, threads: meshWorkgroupThreadCount(size), module: module, structures: structures)
        main += ["    barrier();"] + inputs.initializers + [
            "    uvec3 saga_size = saga_task_body(\(inputs.references.joined(separator: ", ")));"
        ]
        if let limits = dispatchLimits {
            main += [
                "    uint saga_product_high, saga_product_low, saga_total_high, saga_total_low;",
                "    umulExtended(saga_size.x, saga_size.y, saga_product_high, saga_product_low);",
                "    umulExtended(saga_product_low, saga_size.z, saga_total_high, saga_total_low);",
                "    if (saga_size.x > \(limits.maxMeshWorkgroupsPerDimension)u || saga_size.y > \(limits.maxMeshWorkgroupsPerDimension)u || saga_size.z > \(limits.maxMeshWorkgroupsPerDimension)u ||",
                "        saga_product_high != 0u || saga_total_high != 0u || saga_total_low > \(limits.maxMeshWorkgroupsTotal)u) {",
                "        saga_size = uvec3(0u);",
                "    }"
            ]
        }
        main += ["    if (\(lane) == 0u) { saga_dispatch_size = saga_size; }",
            "    barrier();", "    EmitMeshTasksEXT(saga_dispatch_size.x, saga_dispatch_size.y, saga_dispatch_size.z);", "}", ""
        ]
        return .text(format: .glsl, source: (prefix + declarations + numeric + helperLines + inner + main).joined(separator: "\n"))
    }
}
