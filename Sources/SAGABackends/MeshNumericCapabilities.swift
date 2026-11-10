import SAGAShaderIR

/// Numeric features needed by the selected stage and its reachable helpers.
package func meshNumericCapabilities(module: IRModule, entry: IREntryPoint) throws -> (f16: Bool, int16: Bool, int64: Bool, atomicMinMax: Bool, atomicAllOps: Bool) {
    var helpers = try ShaderFunctions(module: module, entry: entry.function)
    return try helpers.reachableNumericCapabilities(in: module, entry: entry)
}
