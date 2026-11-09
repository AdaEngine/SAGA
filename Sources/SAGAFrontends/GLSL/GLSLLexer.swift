import SAGAShaderIR

struct GLSLToken: Sendable {
    var text: String
    let start: Int
    let end: Int
    var barycentricsEnabled = false
    var multiviewEnabled = false
    var float16Enabled = false
    var isViewIndex: Bool { text == "gl_ViewIndex" }
    var barycentricPerspective: Bool? {
        switch text {
        case "gl_BaryCoordEXT": true
        case "gl_BaryCoordNoPerspEXT": false
        default: nil
        }
    }
    var location: ShaderLocation { get throws { .source(try SourceSpan(startUTF8:start,endUTF8:end)) } }
    func mapped(to t: GLSLToken) -> GLSLToken { .init(text:text,start:t.start,end:t.end) }
}
func glslError(_ t: GLSLToken, _ code: ShaderDiagnosticCode, _ message: String) -> ShaderFailure {
    // Token offsets are scanner-produced nonnegative UTF8 bounds.
    do { return .diagnostics([.init(severity:.error,code:code,message:message,location:try t.location)]) }
    catch { return .invalidSpan }
}
func glslIdentifier(_ text: String) -> Bool {
    let b=Array(text.utf8)
    return !b.isEmpty && glslLetter(b[0]) && b.allSatisfy { glslLetter($0) || (48...57).contains($0) }
}
private func glslLetter(_ b: UInt8) -> Bool { (65...90).contains(b) || (97...122).contains(b) || b == 95 }

struct GLSLScanner {
    static func scan(_ source: String, budget: inout ShaderBudget) throws -> [GLSLToken] {
        try budget.consume(.work,amount:source.utf8.count)
        let b=Array(source.utf8); var i=0; var result:[GLSLToken]=[]
        let punct:[String] = ["<<=",">>=","++","--","+=","-=","*=","/=","%=","==","!=","<=",">=","&&","||","^^","<<",">>","&=","|=","^=","##"]
        func append(_ text:String,_ start:Int,_ end:Int) throws {
            try budget.consume(.tokens); result.append(.init(text:text,start:start,end:end))
        }
        while i<b.count {
            try budget.consume(.work)
            let c=b[i]; let s=i
            if c == 32 || c == 9 || c == 13 { i+=1; continue }
            if c == 10 { try append("\n",i,i+1); i+=1; continue }
            if c == 47 && i+1<b.count && b[i+1] == 47 {
                i+=2; while i<b.count && b[i] != 10 { try budget.consume(.work); i+=1 }; continue
            }
            if c == 47 && i+1<b.count && b[i+1] == 42 {
                i+=2; var closed=false
                while i<b.count {
                    try budget.consume(.work)
                    if b[i] == 42 && i+1<b.count && b[i+1] == 47 { i+=2; closed=true; break }
                    if b[i] == 10 { try append("\n",i,i+1) }
                    i+=1
                }
                if !closed { throw glslError(.init(text:"/*",start:s,end:s+2),.invalidToken,"Unterminated comment") }; continue
            }
            if glslLetter(c) {
                i+=1; while i<b.count && (glslLetter(b[i]) || (48...57).contains(b[i])) { try budget.consume(.work); i+=1 }
            } else if (48...57).contains(c) || (c == 46 && i+1<b.count && (48...57).contains(b[i+1])) {
                i+=1
                while i<b.count {
                    let n=b[i]
                    if glslLetter(n) || (48...57).contains(n) || n == 46 { try budget.consume(.work); i+=1 }
                    else if (n == 43 || n == 45) && i>s && !(i>=s+2 && b[s]==48 && (b[s+1]==120 || b[s+1]==88)) && [69,101,80,112].contains(b[i-1]) { i+=1 }
                    else { break }
                }
            } else {
                var found=false
                for p in punct {
                    let bytes=Array(p.utf8)
                    if i+bytes.count<=b.count && b[i..<i+bytes.count].elementsEqual(bytes) { i+=bytes.count; found=true; break }
                }
                if !found {
                    if !Array("{}()[];,:.?~!+-*/%<>=&|^#".utf8).contains(c) {
                        var end=i+1; while end<b.count && b[end]&0xc0 == 0x80 { end+=1 }
                        throw glslError(.init(text:"",start:i,end:end),.invalidToken,"Invalid GLSL character")
                    }
                    i+=1
                }
            }
            try append(String(decoding:b[s..<i],as:UTF8.self),s,i)
        }
        return result
    }
}
