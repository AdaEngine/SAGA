import SAGAShaderIR

struct GLSLMacro {
    let parameters: [String]?
    let replacement: [GLSLToken]
}
struct GLSLPreprocessor {
    static let barycentricExtension = "GL_EXT_fragment_shader_barycentric"
    private var barycentricsEnabled = false
    var macros:[String:GLSLMacro]=[:]
    var version=0
    var nesting=0
    struct Conditional { let parent:Bool; var active:Bool; var taken:Bool; var sawElse=false }
    mutating func process(_ tokens:[GLSLToken], defines:[String:String], budget:inout ShaderBudget) throws -> [GLSLToken] {
        for (key,value) in defines.sorted(by:{$0.key<$1.key}) {
            guard glslIdentifier(key), !key.hasPrefix("__"), !key.hasPrefix("GL_") else { throw ShaderFailure.unsupported("Invalid/reserved GLSL define: \(key)") }
            let replacement=try GLSLScanner.scan(value,budget:&budget)
            guard !replacement.contains(where:{$0.text=="\n" || $0.text=="#"}) else { throw ShaderFailure.unsupported("Define must be a token sequence") }
            macros[key] = .init(parameters:nil,replacement:replacement)
        }
        var stack:[Conditional]=[]; var output:[GLSLToken]=[]; var line:[GLSLToken]=[]; var sawAnything=false
        for token in tokens + [.init(text:"\n",start:tokens.last?.end ?? 0,end:tokens.last?.end ?? 0)] {
            try budget.consume(.work)
            if token.text != "\n" { line.append(token); continue }
            if line.isEmpty { continue }
            let active=stack.last?.active ?? true
            if line[0].text == "#" {
                let anchor=line[0]
                guard line.count>=2 else { throw glslError(anchor,.unexpectedToken,"Empty directive") }
                let directive=line[1].text; let args=Array(line.dropFirst(2))
                switch directive {
                case "if", "ifdef", "ifndef":
                    guard stack.count < budget.limits.depth else { throw ShaderFailure.budgetExceeded(.depth) }
                    let condition:Bool
                    if !active { condition=false }
                    else if directive == "if" { condition = try evaluate(args,budget:&budget) != 0 }
                    else {
                        guard args.count==1, glslIdentifier(args[0].text) else { throw glslError(anchor,.unexpectedToken,"Expected macro name") }
                        condition = (macros[args[0].text] != nil) == (directive == "ifdef")
                    }
                    stack.append(.init(parent:active,active:active && condition,taken:condition))
                case "elif":
                    guard var c=stack.popLast(), !c.sawElse else { throw glslError(anchor,.unexpectedToken,"Unmatched elif") }
                    let condition = c.parent && !c.taken ? try evaluate(args,budget:&budget) != 0 : false
                    c.active=c.parent && !c.taken && condition; c.taken = c.taken || condition; stack.append(c)
                case "else":
                    guard args.isEmpty, var c=stack.popLast(), !c.sawElse else { throw glslError(anchor,.unexpectedToken,"Unmatched/duplicate else") }
                    c.active=c.parent && !c.taken; c.taken=true; c.sawElse=true; stack.append(c)
                case "endif":
                    guard args.isEmpty, stack.popLast() != nil else { throw glslError(anchor,.unexpectedToken,"Unmatched endif") }
                default:
                    if active {
                        switch directive {
                        case "version":
                            guard !sawAnything, version==0, args.count==1 || args.count==2,
                                  let v=Int(args[0].text), [440,450,460].contains(v), args.count==1 || args[1].text=="core" else {
                                throw glslError(anchor,.unsupportedConstruct,"Expected first directive version 440/450/460 core")
                            }
                            version=v
                            let value=GLSLToken(text:String(v),start:anchor.start,end:anchor.end)
                            macros["__VERSION__"] = .init(parameters:nil,replacement:[value])
                            if v >= 450 {
                                let supported = GLSLToken(text: "1", start: anchor.start, end: anchor.end)
                                macros[Self.barycentricExtension] = .init(parameters: nil, replacement: [supported])
                            }
                        case "define":
                            guard version != 0, let name=args.first, glslIdentifier(name.text), !name.text.hasPrefix("__"), !name.text.hasPrefix("GL_") else { throw glslError(anchor,.unexpectedToken,"Invalid define") }
                            var index=1; var parameters:[String]?=nil
                            if index<args.count && args[index].text=="(" && args[index].start==name.end {
                                index+=1; parameters=[]
                                while index<args.count && args[index].text != ")" {
                                    let p=args[index]
                                    guard glslIdentifier(p.text), parameters?.contains(p.text)==false else { throw glslError(p,.unexpectedToken,"Invalid macro parameter") }
                                    parameters?.append(p.text); index+=1
                                    if index<args.count && args[index].text=="," { index+=1; guard index<args.count && args[index].text != ")" else { throw glslError(p,.unexpectedToken,"Trailing macro comma") } }
                                    else { break }
                                }
                                guard index<args.count && args[index].text==")" else { throw glslError(name,.unexpectedToken,"Unterminated macro parameters") }; index+=1
                            }
                            let replacement=Array(args.dropFirst(index))
                            if let bad=replacement.first(where:{$0.text=="#" || $0.text=="##"}) { throw glslError(bad,.unsupportedConstruct,"Stringification/token pasting not implemented") }
                            guard macros[name.text]==nil else { throw glslError(name,.unsupportedConstruct,"Macro redefinition not implemented; undef first") }
                            macros[name.text] = .init(parameters:parameters,replacement:replacement)
                        case "undef":
                            guard args.count==1, !args[0].text.hasPrefix("__") else { throw glslError(anchor,.unexpectedToken,"Invalid undef") }
                            macros.removeValue(forKey:args[0].text)
                        case "error": throw glslError(anchor,.unexpectedToken,"Preprocessor error: \(args.map(\.text).joined(separator:" "))")
                        case "extension":
                            guard args.count == 3, args[1].text == ":" else {
                                throw glslError(anchor, .unexpectedToken, "Expected extension name : behavior")
                            }
                            guard version >= 450, args[0].text == Self.barycentricExtension,
                                  ["require", "enable", "warn", "disable"].contains(args[2].text) else {
                                throw glslError(anchor, .unsupportedConstruct, "Unsupported GLSL extension or behavior")
                            }
                            barycentricsEnabled = args[2].text != "disable"
                        default: throw glslError(anchor,.unsupportedConstruct,"Unsupported directive: \(directive)")
                        }
                    }
                }
                sawAnything=true
            } else if active {
                guard version != 0 else { throw glslError(line[0],.unexpectedToken,"Missing version directive") }
                for var token in try expand(line, disabled: [], budget: &budget) {
                    if token.barycentricPerspective != nil { token.barycentricsEnabled = barycentricsEnabled }
                    output.append(token)
                }
                sawAnything=true
            }
            line=[]
        }
        guard stack.isEmpty else { throw glslError(tokens.last ?? .init(text:"",start:0,end:0),.unexpectedToken,"Unterminated conditional") }
        guard version != 0 else { throw ShaderFailure.unsupported("Missing GLSL version") }
        return output
    }
    mutating func expand(_ tokens:[GLSLToken], disabled:Set<String>, budget:inout ShaderBudget) throws -> [GLSLToken] {
        guard nesting<budget.limits.depth else { throw ShaderFailure.budgetExceeded(.depth) }; nesting+=1; defer { nesting-=1 }
        var out:[GLSLToken]=[]; var i=0
        while i<tokens.count {
            try budget.consume(.work); let t=tokens[i]; i+=1
            try budget.consume(.work,amount:t.text.utf8.count)
            guard let macro=macros[t.text], !disabled.contains(t.text) else { try budget.consume(.tokens); out.append(t); continue }
            try budget.consume(.work,amount:macro.replacement.count)
            try budget.consume(.work,amount:macro.replacement.count)
            var replacement=macro.replacement.map { $0.mapped(to:t) }
            if let params=macro.parameters {
                guard i<tokens.count && tokens[i].text=="(" else { try budget.consume(.tokens); out.append(t); continue }
                i+=1; var args:[[GLSLToken]]=[]; var arg:[GLSLToken]=[]; var depth=0; var closed=false
                while i<tokens.count {
                    try budget.consume(.work); let n=tokens[i]; i+=1
                    if n.text==")" && depth==0 { if !arg.isEmpty || !params.isEmpty { args.append(arg) }; closed=true; break }
                    if n.text=="," && depth==0 { args.append(arg); arg=[]; continue }
                    if n.text=="(" { depth+=1; guard depth<=budget.limits.depth else { throw ShaderFailure.budgetExceeded(.depth) } }
                    if n.text==")" { depth-=1 }
                    arg.append(n)
                }
                guard closed, args.count==params.count else { throw glslError(t,.unexpectedToken,"Macro argument count/parentheses") }
                var values:[String:[GLSLToken]]=[:]
                for (p,a) in zip(params,args) { values[p]=try expand(a,disabled:disabled.union([t.text]),budget:&budget) }
                var substituted:[GLSLToken]=[]
                for token in replacement {
                    let piece=values[token.text] ?? [token]
                    try budget.consume(.work,amount:piece.count)
                    substituted += piece
                }
                replacement=substituted
            }
            try budget.consume(.work,amount:replacement.count)
            out += try expand(replacement,disabled:disabled.union([t.text]),budget:&budget)
        }
        return out
    }
    mutating func evaluate(_ tokens:[GLSLToken], budget:inout ShaderBudget) throws -> Int64 {
        var prepared:[GLSLToken]=[]; var i=0
        while i<tokens.count {
            try budget.consume(.work); let t=tokens[i]; i+=1
            if t.text=="defined" {
                let paren=i<tokens.count && tokens[i].text=="("; if paren { i+=1 }
                guard i<tokens.count, glslIdentifier(tokens[i].text) else { throw glslError(t,.unexpectedToken,"defined expects identifier") }
                let value=macros[tokens[i].text] == nil ? "0":"1"; i+=1
                if paren { guard i<tokens.count && tokens[i].text==")" else { throw glslError(t,.unexpectedToken,"defined closing parenthesis") }; i+=1 }
                prepared.append(.init(text:value,start:t.start,end:t.end))
            } else { prepared.append(t) }
        }
        let expanded=try expand(prepared,disabled:[],budget:&budget)
        var evaluator=GLSLIfExpression(tokens:expanded)
        return try evaluator.run(budget:&budget)
    }
}

// Checked integer expression evaluator; unsupported operators fail rather than guess.
private struct GLSLIfExpression {
    let tokens:[GLSLToken]; var i=0; var depth=0
    mutating func run(budget:inout ShaderBudget) throws -> Int64 {
        let value=try expression(0,budget:&budget)
        guard i==tokens.count else { throw glslError(tokens[i],.unsupportedConstruct,"Unsupported conditional expression") }; return value
    }
    mutating func expression(_ min:Int,evaluating:Bool=true,budget:inout ShaderBudget) throws -> Int64 {
        try budget.consume(.work); guard depth<budget.limits.depth else { throw ShaderFailure.budgetExceeded(.depth) }; depth+=1; defer { depth-=1 }
        guard i<tokens.count else { throw ShaderFailure.unsupported("Incomplete preprocessor expression") }
        let t=tokens[i]; i+=1; var lhs:Int64
        if t.text=="(" { lhs=try expression(0,evaluating:evaluating,budget:&budget); guard i<tokens.count && tokens[i].text==")" else { throw glslError(t,.unexpectedToken,"Unclosed conditional") }; i+=1 }
        else if ["!","~","-","+"].contains(t.text) {
            let rhs=try expression(11,evaluating:evaluating,budget:&budget)
            if t.text=="!" { lhs=rhs==0 ? 1:0 } else if t.text=="~" { lhs = ~rhs }
            else if t.text=="-" { guard !evaluating || rhs != Int64.min else { throw glslError(t,.invalidToken,"Integer overflow") }; lhs = evaluating ? -rhs:0 } else { lhs=rhs }
        } else if let n=Int64(t.text) { lhs=n }
        else if glslIdentifier(t.text) { lhs=0 }
        else { throw glslError(t,.invalidToken,"Invalid conditional integer") }
        let precedence=["||":1,"&&":2,"|":3,"^":4,"&":5,"==":6,"!=":6,"<":7,">":7,"<=":7,">=":7,"<<":8,">>":8,"+":9,"-":9,"*":10,"/":10,"%":10]
        while i<tokens.count, let p=precedence[tokens[i].text],p>=min {
            let op=tokens[i]; i+=1
            let live=evaluating && !(op.text=="&&" && lhs==0) && !(op.text=="||" && lhs != 0)
            let r=try expression(p+1,evaluating:live,budget:&budget)
            if !evaluating { lhs=0; continue }
            switch op.text {
            case "||":lhs=(lhs != 0 || r != 0) ? 1:0
            case "&&":lhs=(lhs != 0 && r != 0) ? 1:0
            case "|":lhs |= r
            case "^":lhs ^= r
            case "&":lhs &= r
            case "==":lhs=lhs==r ? 1:0
            case "!=":lhs=lhs != r ? 1:0
            case "<":lhs=lhs<r ? 1:0
            case ">":lhs=lhs>r ? 1:0
            case "<=":lhs=lhs<=r ? 1:0
            case ">=":lhs=lhs>=r ? 1:0
            case "+","-","*":
                let pair=op.text=="+" ? lhs.addingReportingOverflow(r) : op.text=="-" ? lhs.subtractingReportingOverflow(r):lhs.multipliedReportingOverflow(by:r)
                guard !pair.overflow else { throw glslError(op,.invalidToken,"Conditional overflow") }; lhs=pair.partialValue
            case "/","%":
                guard r != 0, !(lhs==Int64.min && r == -1) else { throw glslError(op,.invalidToken,"Invalid conditional division") }
                lhs=op.text=="/" ? lhs/r:lhs%r
            case "<<", ">>":
                guard r>=0,r<64 else { throw glslError(op,.invalidToken,"Invalid conditional shift") }
                lhs=op.text=="<<" ? Int64(bitPattern:UInt64(bitPattern:lhs)<<UInt64(r)):lhs>>UInt64(r)
            default: throw glslError(op,.unsupportedConstruct,"Conditional operator not implemented")
            }
        }
        return lhs
    }
}
