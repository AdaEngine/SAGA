import SAGAShaderIR

extension WGSLParser {
  // WGSL predeclared aliases are fallback names, not reserved user identifiers.
  static let predeclaredTypes: [String: IRTypeKind] = {
    var out: [String: IRTypeKind] = [:]
    for (width, size) in [(2, IRVectorSize.two), (3, .three), (4, .four)] {
      for (suffix, scalar) in [("i", IRScalar.i32), ("u", .u32), ("f", .f32), ("h", .f16)] {
        out["vec\(width)\(suffix)"] = .vector(size: size, scalar: scalar)
      }
      for (rows, rowSize) in [(2, IRVectorSize.two), (3, .three), (4, .four)] {
        out["mat\(width)x\(rows)f"] = .matrix(columns: size, rows: rowSize, scalar: .f32)
        out["mat\(width)x\(rows)h"] = .matrix(columns: size, rows: rowSize, scalar: .f16)
      }
    }
    return out
  }()
  func requireFloat16(at token: WGSLToken) throws {
    guard float16Enabled else { throw wgslError(.unsupportedCapability, "f16 requires enable f16", token) }
  }

  func predeclaredType(_ name: String) -> IRTypeKind? {
    guard declarationNames[wgslKey(name)] == nil else { return nil }
    return Self.predeclaredTypes[name]
  }
}
