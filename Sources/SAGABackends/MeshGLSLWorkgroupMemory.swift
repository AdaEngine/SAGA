import SAGAShaderIR

/// GLSL workgroup globals are declared at module scope and initialized cooperatively.
struct MeshGLSLWorkgroupMemory {
    let variables: [(Int, WorkgroupVariable)]

    init(module: IRModule, indices: Set<Int>, structures: [IRHandle<IRType>: ShaderStructure]) throws {
        var result: [(Int, WorkgroupVariable)] = []
        for index in indices.sorted() {
            let global = module.globalVariables.values[index]
            guard global.space == .workgroup, global.binding == nil, global.initializer == nil,
                  global.memoryDecorations.isEmpty else { throw ShaderFailure.unsupported("Mesh workgroup variable") }
            let name = "saga_shared_\(index)"
            let variable: WorkgroupVariable
            if try workgroupCompositeByteSize(global.type, in: module) != nil {
                let type = try shaderValueType(global.type, module: module, format: .glsl, structures: structures)
                variable = .init(name: name, layout: .composite(global.type), format: .glsl, type: type)
            } else {
                let field = try ComputeStorageField(type: global.type, offset: 0, module: module)
                guard !field.runtime else { throw ShaderFailure.unsupported("Mesh workgroup variable type") }
                variable = .init(name: name, layout: .flat(field), format: .glsl,
                                 type: try field.valueType(in: .glsl, module: module))
            }
            result.append((index, variable))
        }
        variables = result
    }

    var declarations: [String] { variables.map { $0.1.declaration } }

    func initialize(lane: String, threads: UInt32, module: IRModule, structures: [IRHandle<IRType>: ShaderStructure]) throws -> [String] {
        var budget = ShaderBudget()
        var lines: [String] = []
        for (_, variable) in variables {
            lines += try variable.initialize(lane: lane, threads: threads, module: module, structures: structures, budget: &budget)
        }
        return lines
    }
}
