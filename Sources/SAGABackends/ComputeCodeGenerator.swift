import SAGAShaderIR
import SAGAValidation

/// Initial compute writer. All unsupported IR fails closed before an artifact is returned.
package struct ComputeCodeGenerator: ShaderCodeGenerator {
    package let outputFormat: ShaderOutputFormat

    package init?(_ output: ShaderOutputFormat) {
        switch output {
        case .msl, .hlsl, .glsl, .wgsl: outputFormat = output
        default: return nil
        }
    }

    package func generate(_ validated: ValidatedModule<IRModule>) throws -> ShaderArtifact {
        try generateCompute(validated).artifact
    }

    package func generateCompute(_ validated: ValidatedModule<IRModule>) throws -> ComputeShaderArtifact {
        let module = validated.module
        guard module.entryPoints.count == 1, let entry = module.entryPoints.first,
              entry.stage == .compute, entry.workgroupOverrides == nil,
              module.constants.count == 0, module.overrides.count == 0
        else { throw unsupported("expected one compute entry point with no module constants or overrides") }
        let function = try module.functions.value(for: entry.function)
        guard function.result == nil else {
            throw unsupported("compute return values")
        }

        let size = entry.workgroupSize
        guard size.x > 0, size.y > 0, size.z > 0,
              size.x <= 1024, size.y <= 1024, size.z <= 64,
              size.x * size.y * size.z <= 1024
        else { throw unsupported("workgroup size exceeds the compute profile") }

        var declarations: [String] = []
        var parameters: [String] = []
        var arguments: [String] = []
        for (index, argument) in function.arguments.enumerated() {
            guard argument.binding == .builtIn(.globalInvocationId),
                  try module.types.value(for: argument.type).kind == .vector(size: .three, scalar: .u32)
            else { throw unsupported("compute inputs other than global_invocation_id") }
            let name = "saga_arg_\(index)"
            switch outputFormat {
            case .msl: parameters.append("uint3 \(name) [[thread_position_in_grid]]")
            case .hlsl: parameters.append("uint3 \(name) : SV_DispatchThreadID")
            case .wgsl: parameters.append("@builtin(global_invocation_id) \(name): vec3<u32>")
            default: break
            }
            arguments.append(outputFormat == .glsl ? "gl_GlobalInvocationID" : name)
        }

        var storage: [ShaderStorage] = []
        var metalStorage: [ShaderParameter] = []
        var layout: [ComputeShaderArtifact.StorageBinding] = []
        var usedBindings = Set<UInt32>()
        for (index, global) in module.globalVariables.values.enumerated() {
            guard let binding = global.binding, binding.group == 0, binding.binding < 8,
                  case .storage(let access) = global.space, access == .load || access == [.load, .store],
                  global.initializer == nil, global.memoryDecorations.isEmpty,
                  usedBindings.insert(binding.binding).inserted
            else { throw unsupported("expected unique group 0 storage bindings in 0..<8") }
            let count: UInt32?
            switch try module.types.value(for: global.type).kind {
            case .scalar(.u32):
                count = nil
            case .array(let element, .constant(let length), 4)
                where length > 0 && length <= 32 * 1024 * 1024:
                guard try module.types.value(for: element).kind == .scalar(.u32) else {
                    throw unsupported("storage array element type")
                }
                count = length
            default: throw unsupported("expected a u32 scalar or a fixed-size u32 array")
            }
            let name = "saga_buffer_\(index)"
            let slot = binding.binding
            let writable = access.contains(.store)
            let reference: String
            switch outputFormat {
            case .msl:
                let qualifier = writable ? "device" : "device const"
                parameters.append("\(qualifier) uint* \(name) [[buffer(\(slot))]]")
                metalStorage.append(.init(type: "\(qualifier) uint*", name: name))
                reference = count == nil ? "*\(name)" : name
            case .hlsl:
                let type = writable ? "RWStructuredBuffer" : "StructuredBuffer"
                let register = writable ? "u" : "t"
                declarations.append("\(type)<uint> \(name) : register(\(register)\(slot), space0);")
                reference = count == nil ? "\(name)[0]" : name
            case .wgsl:
                let access = writable ? "read_write" : "read"
                let type = count.map { "array<u32, \($0)>" } ?? "u32"
                declarations.append("@group(0) @binding(\(slot)) var<storage, \(access)> \(name): \(type);")
                reference = name
            default:
                let qualifier = writable ? "buffer" : "readonly buffer"
                let suffix = count.map { "[\($0)]" } ?? ""
                declarations.append("layout(std430, binding = \(slot)) \(qualifier) SagaBuffer\(index) { uint value\(suffix); } \(name);")
                reference = "\(name).value"
            }
            storage.append(ShaderStorage(reference: reference, count: count, writable: writable))
            layout.append(.init(binding: binding, minimumByteSize: UInt64(count ?? 1) * 4, isReadOnly: !writable))
        }

        let parameterList = parameters.joined(separator: ", ")
        let entryPoint = outputFormat == .glsl ? "main" : "saga_main"
        var helpers = try ShaderFunctions(module: module, entry: entry.function, metalStorage: metalStorage)
        let helperDeclarations = try helpers.declarations(module: module, format: outputFormat, storage: storage)
        let signature: String
        let attributes: [String]
        switch outputFormat {
        case .msl:
            declarations.insert(contentsOf: ["#include <metal_stdlib>", "using namespace metal;"], at: 0)
            attributes = []
            signature = "kernel void \(entryPoint)(\(parameterList))"
        case .hlsl:
            attributes = ["[numthreads(\(size.x), \(size.y), \(size.z))]"]
            signature = "void \(entryPoint)(\(parameterList))"
        case .wgsl:
            attributes = ["@compute @workgroup_size(\(size.x), \(size.y), \(size.z))"]
            signature = "fn \(entryPoint)(\(parameterList))"
        default:
            declarations.insert(contentsOf: [
                "#version 450 core",
                "layout(local_size_x = \(size.x), local_size_y = \(size.y), local_size_z = \(size.z)) in;",
            ], at: 0)
            attributes = []
            signature = "void \(entryPoint)()"
        }

        var body = ShaderBodyWriter(module: module, function: function, arguments: arguments,
                                    storage: storage, outputFormat: outputFormat,
                                    callCaptures: helpers.callCaptures, budget: helpers.budget)
        let lines = try body.write()
        let numeric = try integerPrelude(helpers.integerHelpers.union(body.integerHelpers),
                                         vectors: helpers.integerVectorHelpers.union(body.integerVectorHelpers),
                                         in: outputFormat)
        let source = (declarations + numeric + helperDeclarations + attributes + [signature + " {"] + lines + ["}", ""])
            .joined(separator: "\n")
        return .init(
            artifact: .text(format: outputFormat, source: source), sourceEntryPoint: entry.name,
            entryPoint: entryPoint, workgroupSize: size, storageBindings: layout)
    }
}

private func unsupported(_ reason: String) -> ShaderFailure {
    .unsupported("Compute code generation: \(reason)")
}
