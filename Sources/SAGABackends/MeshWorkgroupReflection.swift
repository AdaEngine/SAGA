import SAGAShaderIR

/// Reflects the threadgroup allocations emitted for the selected Metal stage.
package func meshWorkgroupStorageByteSize(module: IRModule, entry: IREntryPoint) throws -> UInt64 {
    let helpers = try ShaderFunctions(module: module, entry: entry.function)
    var indices = Set(helpers.reachableGlobals.filter { module.globalVariables.values[$0].space == .workgroup })
    // The Metal mesh generator always materializes the output, even if the body
    // does not reference it. Object payloads use a separate address space.
    if let output = entry.meshInfo?.outputVariable { indices.insert(output.index) }
    var total: UInt64 = 0
    for index in indices {
        let type = module.globalVariables.values[index].type
        let bytes: UInt64
        if let composite = try workgroupCompositeByteSize(type, in: module) {
            bytes = composite
        } else {
            let field = try ComputeStorageField(type: type, offset: 0, module: module)
            guard !field.runtime else { throw ShaderFailure.unsupported("Expected a fixed workgroup variable") }
            bytes = field.minimumByteSize
        }
        let (padded, overflow) = bytes.addingReportingOverflow(15)
        let (sum, sumOverflow) = total.addingReportingOverflow(padded & ~UInt64(15))
        guard !overflow, !sumOverflow, sum <= UInt64(UInt32.max) else {
            throw ShaderFailure.unsupported("Workgroup storage exceeds the representable device limit")
        }
        total = sum
    }
    return total
}
