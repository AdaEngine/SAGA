import SAGAShaderIR

extension WGSLParser {
    func debugPrintfStatement(at start: WGSLToken, end: String) throws {
        guard debugPrintfEnabled else {
            throw wgslError(.unsupportedCapability, "debugPrintf requires enable wgpu_debug_printf", start)
        }
        _ = try take()
        try expect("(")
        let literal = try take()
        guard literal.kind == .string else {
            throw wgslError(.typeMismatch, "debugPrintf requires a format string literal", literal)
        }
        let format = try decodeDebugFormat(literal)
        var arguments: [WExpr] = []
        while try eat(","), !isNext(")") {
            let value = try defaultMaterialize(loaded(expression(), at: start), at: start)
            try node()
            arguments.append(value.handle)
        }
        try expect(")")
        try expect(end)
        try statement(.debugPrintf(format: format, arguments: arguments), at: start)
    }

    private func decodeDebugFormat(_ token: WGSLToken) throws -> String {
        var scalars = token.text.unicodeScalars.dropFirst().dropLast().makeIterator()
        var result = ""
        while let scalar = scalars.next() {
            try budget.consume(.work)
            if scalar != "\\" {
                result.unicodeScalars.append(scalar)
                continue
            }
            switch scalars.next() {
            case "\\": result += "\\"
            case "\"": result += "\""
            case "n": result += "\n"
            case "r": result += "\r"
            case "t": result += "\t"
            default: throw wgslError(.invalidToken, "Unsupported debugPrintf string escape", token)
            }
        }
        return result
    }
}
