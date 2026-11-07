import Foundation
import CryptoKit
import SAGAFrontends
import SAGAShaderIR
import SAGAValidation

struct ConfiguredPlan: Decodable {
  let reference_revision: String
  let rows: [ConfiguredInput]
}
struct ConfiguredInput: Decodable {
  let source: String
  let path: String
  let sha256: String
  let format: String
  let stage: String
  let capabilities: UInt64
  let parse_doc_comments: Bool
  let adjust_coordinate_space: Bool
  let strict_capabilities: Bool
  let validation: Bool
  let unavailable_reason: String?
  let config_source: String?
  let config_sha256: String?
}
@main struct ConfiguredCorpusRun {
  static func digest(_ data: Data) -> String { SHA256.hash(data:data).map {String(format:"%02x",$0)}.joined() }
  static func main() {
    do { try run() } catch {
      FileHandle.standardError.write(Data((String(describing:error) + "\n").utf8))
      exit(1)
    }
  }
  static func run() throws {
    let args = CommandLine.arguments
    guard args.count == 3 else {throw NSError(domain:"usage: corpus plan.json actual.json",code:1)}
    let plan = try JSONDecoder().decode(ConfiguredPlan.self,from:Data(contentsOf:URL(fileURLWithPath:args[1])))
    guard plan.reference_revision == "a55352911bb26a1b33bc8d5fc03bdebb0a967f5a" else {throw ShaderFailure.invalidHandle}
    var rows: [[String:Any]] = []
    for input in plan.rows {
      var row: [String:Any] = ["source":input.source,"sha256":input.sha256,"parse":"not-run","semantic":"not-run",
        "capabilities":input.capabilities,"parse_doc_comments":input.parse_doc_comments,"validation":input.validation,
        "format":input.format,"stage":input.stage,"strict_capabilities":input.strict_capabilities,
        "adjust_coordinate_space":input.adjust_coordinate_space,
        "config_source":input.config_source as Any? ?? NSNull(),"config_sha256":input.config_sha256 as Any? ?? NSNull()]
      guard input.strict_capabilities else {throw ShaderFailure.unsupported("Relaxed configured capability validation")}
      guard (input.config_source == nil) == (input.config_sha256 == nil) else {throw ShaderFailure.invalidHandle}
      let bytes = try Data(contentsOf:URL(fileURLWithPath:input.path))
      guard digest(bytes) == input.sha256 else {throw NSError(domain:"source hash drift: " + input.source,code:2)}
      if let name = input.config_source, let hash = input.config_sha256 {
        var root = URL(fileURLWithPath:input.path)
        for _ in input.source.split(separator:"/") {root.deleteLastPathComponent()}
        guard digest(try Data(contentsOf:root.appendingPathComponent(name))) == hash else {
          throw NSError(domain:"config hash drift: " + name,code:3)
        }
      }
      if let unavailable = input.unavailable_reason {
        row["unavailable_reason"] = unavailable
        rows.append(row);continue
      }
      do {
        guard let text = String(data:bytes,encoding:.utf8), Data(text.utf8) == bytes else {
          throw NSError(domain:"not strict UTF8",code:4)
        }
        let module: IRModule
        switch input.format {
        case "wgsl":
          module = try WGSLFrontend(options:.init(parseDocComments:input.parse_doc_comments)).parse(.wgsl(text.replacingOccurrences(of:"\r",with:"")))
        case "glsl":
          let stage:ShaderStage = input.stage == "vertex" ? .vertex : input.stage == "fragment" ? .fragment : .compute
          module = try GLSLFrontend(options:.init(stage:IRStage(stage))).parse(.glsl(text,stage:stage,defines:[:]))
        default: throw ShaderFailure.unsupported("No unconfigured format fallback")
        }
        row["parse"] = "accepted"
        if input.validation {
          do {
            try ShaderSemanticValidator(capabilities:.init(rawValue:input.capabilities)).validate(module)
            row["semantic"] = "accepted"
          } catch {row["semantic"] = "rejected";describe(error,&row)}
        }
      } catch {row["parse"] = "rejected";describe(error,&row)}
      rows.append(row)
    }
    try JSONSerialization.data(withJSONObject:rows,options:[.prettyPrinted,.sortedKeys]).write(to:URL(fileURLWithPath:args[2]))
    print("Executed configured frontend rows: \(rows.count); no backend/diagnostic parity claim")
  }
  static func describe(_ error:Error,_ row:inout [String:Any]) {
    if case ShaderFailure.diagnostics(let ds) = error, let d = ds.first {
      row["diagnostic_code"] = d.code.rawValue;row["diagnostic_message"] = d.message
      switch d.location {
      case .source(let s):row["location"] = ["unit":"utf8","start":s.startUTF8,"end":s.endUTF8] as [String:Any]
      case .binary(let s):row["location"] = ["unit":"word","start":s.startWord,"end":s.endWord] as [String:Any]
      case nil:break
      }
    } else {row["failure"] = String(describing:error)}
  }
}
