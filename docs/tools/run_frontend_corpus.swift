import Foundation
import SAGAFrontends
import SAGAShaderIR
import SAGAValidation

struct CorpusInput: Decodable {
    let source: String
    let path: String
    let sha256: String
    let format: String
    let stage: String
}
@main struct CorpusRun {
    static func main() throws {
        let args=CommandLine.arguments
        guard args.count == 3 else {throw NSError(domain:"SAGA corpus usage",code:1)}
        let inputs=try JSONDecoder().decode([CorpusInput].self,from:Data(contentsOf:URL(fileURLWithPath:args[1])))
        var rows:[[String:Any]]=[]
        for input in inputs {
            var row:[String:Any] = ["source":input.source,"sha256":input.sha256,"format":input.format,"stage":input.stage,"parse":"not-run","semantic":"not-run","reference_configuration_executed":false]
            do {
                let source=try String(contentsOfFile:input.path,encoding:.utf8)
                let module:IRModule
                if input.format == "wgsl" {module=try WGSLFrontend().parse(.wgsl(source))}
                else {
                    let stage:ShaderStage=input.stage == "vertex" ? .vertex : input.stage == "fragment" ? .fragment : .compute
                    module=try GLSLFrontend(options:.init(stage:IRStage(stage))).parse(.glsl(source,stage:stage,defines:[:]))
                }
                row["parse"]="accepted"
                do {try ShaderSemanticValidator().validate(module);row["semantic"]="accepted"}
                catch {row["semantic"]="rejected";describe(error,&row)}
            } catch {row["parse"]="rejected";describe(error,&row)}
            rows.append(row)
        }
        let report:[String:Any] = ["reference_revision":"a55352911bb26a1b33bc8d5fc03bdebb0a967f5a","scope":"Raw input fixture file executions at explicit default options, NOT configured upstream expected outcomes or parity","wgsl_options":"parseDocComments=false,capabilities=allKnown","glsl_defines":[String:String](),"validator_capabilities":"referenceDefault","limits":"ShaderLimits.default","shader_bytes_copied":0,"rows":rows]
        try JSONSerialization.data(withJSONObject:report,options:[.prettyPrinted,.sortedKeys]).write(to:URL(fileURLWithPath:args[2]))
        print("Executed \(rows.count) pinned shader inputs; acceptance/rejection report, not parity")
    }
    static func describe(_ error:Error,_ row:inout [String:Any]) {
        if case ShaderFailure.diagnostics(let diagnostics)=error,let first=diagnostics.first {
            row["diagnostic_code"]=first.code.rawValue;row["diagnostic_message"]=first.message
            switch first.location {
            case .source(let span):row["location"]=["unit":"utf8","start":span.startUTF8,"end":span.endUTF8] as [String:Any]
            case .binary(let span):row["location"]=["unit":"word","start":span.startWord,"end":span.endWord] as [String:Any]
            case nil:break
            }
        } else {row["failure"]=String(describing:error)}
    }
}
