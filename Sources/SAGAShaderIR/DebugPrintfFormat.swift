package enum DebugPrintfFormat {
    package enum Argument {
        case signed, unsigned, floating, wideSigned, wideUnsigned

        package func accepts(_ type: IRTypeKind) -> Bool {
            guard case .scalar(let scalar) = type else { return false }
            return switch self {
            case .signed: scalar == .i32
            case .unsigned: scalar == .u32
            case .floating: scalar.kind == .float
            case .wideSigned: scalar == .i64
            case .wideUnsigned: scalar == .u64
            }
        }
    }

    package static func arguments(
        in format: String, location: ShaderLocation?, budget: inout ShaderBudget
    ) throws -> [Argument] {
        try budget.consume(.work, amount: format.utf8.count)
        let bytes = Array(format.utf8)
        func invalid(_ message: String) -> ShaderFailure {
            .diagnostics([.init(severity: .error, code: .typeMismatch, message: message, location: location)])
        }
        guard bytes.allSatisfy({ $0 >= 32 && $0 != 127 || [9, 10, 13].contains($0) }) else {
            throw invalid("Invalid control character in debugPrintf format")
        }
        var result: [Argument] = []
        var index = 0
        while index < bytes.count {
            let byte = bytes[index]
            index += 1
            guard byte == 37 else { continue } // %
            if index < bytes.count, bytes[index] == 37 {
                index += 1
                continue
            }
            while index < bytes.count, (48...57).contains(bytes[index]) { index += 1 }
            if index < bytes.count, bytes[index] == 46 {
                index += 1
                while index < bytes.count, (48...57).contains(bytes[index]) { index += 1 }
            }
            guard index < bytes.count else { throw invalid("Incomplete debugPrintf format specifier") }
            let argument: Argument
            switch bytes[index] {
            case 100, 105: argument = .signed // d, i
            case 111, 117, 120, 88: argument = .unsigned // o, u, x, X
            case 101, 69, 102, 70, 103, 71: argument = .floating // e, E, f, F, g, G
            case 108: // l
                index += 1
                guard index < bytes.count else { throw invalid("Incomplete debugPrintf format specifier") }
                switch bytes[index] {
                case 100: argument = .wideSigned
                case 117, 120: argument = .wideUnsigned
                default: throw invalid("Unsupported debugPrintf format specifier")
                }
            default: throw invalid("Unsupported debugPrintf format specifier")
            }
            try budget.consume(.nodes)
            result.append(argument)
            index += 1
        }
        return result
    }

    package static func sourceLiteral(_ format: String) -> String {
        var text = "\""
        for scalar in format.unicodeScalars {
            switch scalar {
            case "\"": text += "\\\""
            case "\\": text += "\\\\"
            case "\n": text += "\\n"
            case "\r": text += "\\r"
            case "\t": text += "\\t"
            default: text.unicodeScalars.append(scalar)
            }
        }
        return text + "\""
    }
}
