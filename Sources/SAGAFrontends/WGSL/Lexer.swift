import SAGAShaderIR

struct WGSLToken {
    enum Kind { case word, number, symbol, templateStart, templateEnd, end }
    let kind: Kind
    let text: String
    let start: Int
    let end: Int
    var location: ShaderLocation? { (try? SourceSpan(startUTF8: start, endUTF8: end)).map(ShaderLocation.source) }
    var opensGroup: Bool { kind == .templateStart || text == "(" || text == "[" }
    var closesGroup: Bool { kind == .templateEnd || text == ")" || text == "]" }
}

func wgslError(_ code: ShaderDiagnosticCode, _ message: String, _ token: WGSLToken) -> ShaderFailure {
    .diagnostics([ShaderDiagnostic(severity: .error, code: code, message: message, location: token.location)])
}

// Independently authored scalar scanner. Offsets are original UTF-8, not Character indices.
struct WGSLScanner {
    let source: String
    var budget: ShaderBudget
    let docComments: Bool
    func scan() throws -> ([WGSLToken], ShaderBudget) {
        var budget = budget
        var count = 0
        for _ in source.utf8 {
            try budget.consume(.work)
            guard count < budget.limits.sourceBytes else { throw ShaderFailure.budgetExceeded(.sourceBytes) }
            count += 1
        }
        let scalars = Array(source.unicodeScalars)
        var offsets = [Int](); offsets.reserveCapacity(scalars.count + 1)
        var offset = 0
        for c in scalars { try budget.consume(.work); offsets.append(offset); offset += c.utf8.count }
        offsets.append(offset)
        var result: [WGSLToken] = []
        var i = 0
        func char(_ n: Int) -> Unicode.Scalar? { n < scalars.count ? scalars[n] : nil }
        func digit(_ c: Unicode.Scalar?) -> Bool { c.map { (48...57).contains($0.value) } ?? false }
        func hex(_ c: Unicode.Scalar?) -> Bool { c.map { (48...57).contains($0.value) || (65...70).contains($0.value) || (97...102).contains($0.value) } ?? false }
        func text(_ a: Int, _ b: Int) -> String { String(String.UnicodeScalarView(scalars[a..<b])) }
        func token(_ a: Int, _ b: Int, _ kind: WGSLToken.Kind) -> WGSLToken {
            WGSLToken(kind: kind, text: text(a,b), start: offsets[a], end: offsets[b])
        }
        while i < scalars.count {
            try budget.consume(.work)
            let begin = i, c = scalars[i]
            if [9,10,11,12,13,32,0x85,0x200e,0x200f,0x2028,0x2029].contains(c.value) { i += 1; continue }
            if c == "/" && char(i+1) == "/" {
                if docComments && (char(i+2) == "/" || char(i+2) == "!") { throw wgslError(.unsupportedConstruct, "Documentation capture is not implemented", token(i,i+3,.symbol)) }
                i += 2
                while i < scalars.count && ![10,13,0x85,0x2028,0x2029].contains(scalars[i].value) { try budget.consume(.work); i += 1 }
                continue
            }
            if c == "/" && char(i+1) == "*" {
                if docComments && (char(i+2) == "*" || char(i+2) == "!") { throw wgslError(.unsupportedConstruct,"Documentation capture is not implemented",token(i,i+3,.symbol)) }
                guard budget.limits.depth > 0 else { throw ShaderFailure.budgetExceeded(.depth) }
                i += 2; var nesting = 1
                while nesting > 0 {
                    try budget.consume(.work)
                    guard i < scalars.count else { throw wgslError(.invalidToken,"Unterminated block comment",token(begin,i,.symbol)) }
                    if char(i) == "/" && char(i+1) == "*" {
                        guard nesting < budget.limits.depth else { throw ShaderFailure.budgetExceeded(.depth) }
                        nesting += 1; i += 2
                    } else if char(i) == "*" && char(i+1) == "/" { nesting -= 1; i += 2 }
                    else { i += 1 }
                }
                continue
            }
            let kind: WGSLToken.Kind
            if c == "_" || c.properties.isXIDStart {
                i += 1
                while i < scalars.count && scalars[i].properties.isXIDContinue { try budget.consume(.work); i += 1 }
                kind = .word
            } else if digit(c) || (c == "." && digit(char(i+1))) {
                var isHex = false
                if c == "0" && (char(i+1) == "x" || char(i+1) == "X") { isHex = true; i += 2 }
                while isHex ? hex(char(i)) : digit(char(i)) { try budget.consume(.work); i += 1 }
                if char(i) == "." {
                    i += 1
                    while isHex ? hex(char(i)) : digit(char(i)) { try budget.consume(.work); i += 1 }
                }
                if (isHex && (char(i) == "p" || char(i) == "P")) || (!isHex && (char(i) == "e" || char(i) == "E")) {
                    i += 1
                    if char(i) == "+" || char(i) == "-" { i += 1 }
                    let expStart = i
                    while digit(char(i)) { try budget.consume(.work); i += 1 }
                    if i == expStart { throw wgslError(.invalidToken,"Missing numeric exponent",token(begin,i,.number)) }
                }
                if char(i) == "l" && (char(i + 1) == "u" || char(i + 1) == "i") { i += 2 }
                else if char(i) == "u" || char(i) == "i" || char(i) == "f" || char(i) == "h" { i += 1 }
                if let next = char(i), next.properties.isXIDContinue { throw wgslError(.invalidToken,"Invalid numeric suffix",token(begin,i+1,.number)) }
                kind = .number
            } else {
                guard "@(){}[]:;,.+-*/%=&|^!~<>".unicodeScalars.contains(c) else {
                    throw wgslError(.invalidToken,"Invalid or unsupported token",token(i,i+1,.symbol))
                }
                i += 1
                // Template discovery may split >=. Expression parsing combines adjacent shift tokens.
                if let next = char(i) {
                    let pair = String(c) + String(next)
                    if ["->","==","!=","<=",">=","&&","||","+=","-=","*=","/=","%=","&=","|=","^=","++","--"].contains(pair) { i += 1 }
                }
                kind = .symbol
            }
            try budget.consume(.tokens)
            try budget.consume(.work, amount: offsets[i]-offsets[begin])
            result.append(token(begin,i,kind))
        }
        try budget.consume(.tokens)
        result.append(WGSLToken(kind: .end,text: "",start:offset,end:offset))
        let tokens = try discoverTemplates(in: result, budget: &budget)
        return (tokens, budget)
    }

    private func discoverTemplates(in raw: [WGSLToken], budget: inout ShaderBudget) throws -> [WGSLToken] {
        // Match candidate angle brackets at the same expression depth before parsing.
        var tokens: [WGSLToken] = []
        tokens.reserveCapacity(raw.count)
        var pending: [(position: Int, depth: Int)] = []
        var depth = 0
        for (index, token) in raw.enumerated() {
            try budget.consume(.work)
            switch token.text {
            case "<" where index > 0 && raw[index - 1].kind == .word:
                let next = raw[index + 1]
                if token.end != next.start || (next.text != "<" && next.text != "<=") {
                    pending.append((tokens.count, depth))
                }
            case ">", ">=":
                if let opener = pending.last, opener.depth == depth {
                    pending.removeLast()
                    let open = tokens[opener.position]
                    tokens[opener.position] = .init(kind: .templateStart, text: "<", start: open.start, end: open.end)
                    tokens.append(.init(kind: .templateEnd, text: ">", start: token.start, end: token.start + 1))
                    if token.text == ">=" {
                        try budget.consume(.tokens)
                        tokens.append(.init(kind: .symbol, text: "=", start: token.start + 1, end: token.end))
                        pending.removeAll(keepingCapacity: true)
                        depth = 0
                    }
                    continue
                }
            case "(", "[":
                depth += 1
            case ")", "]", "&&", "||":
                while let opener = pending.last, opener.depth >= depth {
                    try budget.consume(.work)
                    pending.removeLast()
                }
                if token.text == ")" || token.text == "]" { depth = max(0, depth - 1) }
            case "=", ";", "{", ":":
                pending.removeAll(keepingCapacity: true)
                depth = 0
            default: break
            }
            tokens.append(token)
        }
        return tokens
    }
}

// ASCII encoding avoids Swift canonical-equivalence collapsing distinct WGSL identifiers.
func wgslKey(_ text: String) -> String {
    let digits = Array("0123456789abcdef".utf8)
    var bytes: [UInt8] = []
    for byte in text.utf8 { bytes.append(digits[Int(byte >> 4)]); bytes.append(digits[Int(byte & 15)]) }
    return String(decoding: bytes, as: UTF8.self)
}
