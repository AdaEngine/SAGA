import SAGAShaderIR

func meshWorkgroupThreadCount(_ size: IRWorkgroupSize) throws -> UInt32 {
    let (xy, xyOverflow) = size.x.multipliedReportingOverflow(by: size.y)
    let (count, countOverflow) = xy.multipliedReportingOverflow(by: size.z)
    guard !xyOverflow, !countOverflow, count > 0 else {
        throw ShaderFailure.unsupported("Mesh workgroup invocation count exceeds UInt32")
    }
    return count
}

struct MeshWorkgroupMemory {
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
                let type = try shaderValueType(global.type, module: module, format: .msl, structures: structures)
                variable = .init(name: name, layout: .composite(global.type), format: .msl, type: type)
            } else {
                let field = try ComputeStorageField(type: global.type, offset: 0, module: module)
                guard !field.runtime, !(field.atomic && field.scalar.width == 8) else {
                    throw ShaderFailure.unsupported("Mesh workgroup variable type")
                }
                variable = .init(name: name, layout: .flat(field), format: .msl,
                                 type: try field.valueType(in: .msl, module: module))
            }
            result.append((index, variable))
        }
        variables = result
    }

    func setup(lane: String, threads: UInt32, module: IRModule, structures: [IRHandle<IRType>: ShaderStructure]) throws -> [String] {
        var budget = ShaderBudget()
        var lines = variables.map { $0.1.declaration }
        for (_, variable) in variables {
            lines += try variable.initialize(lane: lane, threads: threads, module: module, structures: structures, budget: &budget)
        }
        return lines
    }
}
