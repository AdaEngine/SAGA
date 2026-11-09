import SAGAShaderIR

extension WGSLParser {
  func hexFloat(_ source: String, kind: WGSLFloatKind, at t: WGSLToken) throws -> (
    IRLiteral, IRScalar
  ) {
    // Lexer already separates the number; validate exact concrete representability from significand bits,
    // not from rounded Double(text), which could hide a too-precise suffix literal.
    let text = source.lowercased()
    let parts = text.dropFirst(2).split(separator: "p", omittingEmptySubsequences: false)
    guard parts.count <= 2 else { throw wgslError(.invalidToken, "Malformed hex exponent", t) }
    var exponent = 0
    if parts.count == 2 {
      guard let e = Int(parts[1]), (-100000...100000).contains(e) else {
        throw wgslError(.invalidToken, "Hex exponent out of supported range", t)
      }
      exponent = e
    }
    var fractional = 0
    var dot = false
    var digits: [UInt8] = []
    for c in parts[0].utf8 {
      try budget.consume(.work)
      if c == 46 {
        guard !dot else { throw wgslError(.invalidToken, "Duplicate hex point", t) }
        dot = true
        continue
      }
      let d: UInt8
      if c >= 48 && c <= 57 {
        d = c - 48
      } else if c >= 97 && c <= 102 {
        d = c - 87
      } else {
        throw wgslError(.invalidToken, "Invalid hex significand", t)
      }
      digits.append(d)
      if dot { fractional += 1 }
    }
    guard !digits.isEmpty else { throw wgslError(.invalidToken, "Missing hex significand", t) }
    if kind != .abstract, let first = digits.firstIndex(where: { $0 != 0 }),
      let last = digits.lastIndex(where: { $0 != 0 })
    {
      let leading = UInt32(digits[first]).leadingZeroBitCount - 28
      let trailing = UInt32(digits[last]).trailingZeroBitCount
      let significant = (last - first + 1) * 4 - leading - trailing
      let lowest = exponent - fractional * 4 + (digits.count - 1 - last) * 4 + trailing
      let highest = lowest + significant - 1
      let limits = switch kind {
      case .f16: (11, -24, 15)
      case .f32: (24, -149, 127)
      case .f64, .abstract: (53, -1074, 1023)
      }
      guard significant <= limits.0, lowest >= limits.1, highest <= limits.2 else {
        throw wgslError(.invalidToken, "Hex literal not exactly representable", t)
      }
    }
    let parseText = parts.count == 1 ? text + "p0" : text
    guard let d = Double(parseText), d.isFinite else {
      throw wgslError(.invalidToken, "Hex floating overflow/invalid syntax", t)
    }
    if kind == .f16 { return (.f16Bits(Binary16.encode(d)), .f16) }
    if kind == .f64 { return (.f64(d), .f64) }
    if kind == .f32 {
      guard let f = Float(parseText), f.isFinite else {
        throw wgslError(.invalidToken, "Hex f32 literal overflow", t)
      }
      return (.f32(f), .f32)
    }
    return (.abstractFloat(d), .abstractFloat)
  }
}
