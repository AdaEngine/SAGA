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
  let assembly: AssembledInput?
}
struct AssembledInput: Codable {
  let source_sha256: String
  let binary_path: String
  let binary_sha256: String
  let binary_bytes: Int
  let spirv_version: UInt32
  let assembler_path: String
  let assembler_sha256: String
  let assembler_version: String
  let assembler_arguments: [String]
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
      let sourceURL = URL(fileURLWithPath: input.path)
      let bytes: Data
      if input.format == "spirv-assembly" {
        guard (input.assembly != nil) != (input.unavailable_reason != nil) else {
          throw NSError(domain: "missing or ambiguous assembly mapping: " + input.source, code: 5)
        }
        bytes = try boundedBytes(at: sourceURL, limit: ShaderLimits.default.sourceBytes)
      } else {
        guard input.assembly == nil else { throw ShaderFailure.invalidHandle }
        bytes = try Data(contentsOf: sourceURL)
      }
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
      let binary = try input.assembly.map { try assembledWords($0, for: input) }
      if let assembly = input.assembly {
        row["assembly"] = try JSONSerialization.jsonObject(with: JSONEncoder().encode(assembly))
      }
      do {
        let module: IRModule
        switch input.format {
        case "wgsl":
          let text = try strictUTF8(bytes)
          module = try WGSLFrontend(options:.init(parseDocComments:input.parse_doc_comments)).parse(.wgsl(text.replacingOccurrences(of:"\r",with:"")))
        case "glsl":
          let text = try strictUTF8(bytes)
          let stage:ShaderStage = input.stage == "vertex" ? .vertex : input.stage == "fragment" ? .fragment : .compute
          module = try GLSLFrontend(options:.init(stage:IRStage(stage))).parse(.glsl(text,stage:stage,defines:[:]))
        case "spirv-assembly":
          guard let binary else { throw ShaderFailure.invalidHandle }
          let options = SPIRVFrontendOptions(adjustCoordinateSpace: input.adjust_coordinate_space,
                                             strictCapabilities: input.strict_capabilities)
          module = try SPIRVFrontend(options: options).parse(.spirv(binary))
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
  static func strictUTF8(_ bytes: Data) throws -> String {
    guard let text = String(data: bytes, encoding: .utf8), Data(text.utf8) == bytes else {
      throw NSError(domain: "not strict UTF8", code: 4)
    }
    return text
  }
  static func boundedBytes(at url: URL, limit: Int) throws -> Data {
    let values = try url.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey])
    guard values.isRegularFile == true, values.isSymbolicLink == false,
          let size = values.fileSize, size <= limit else {
      throw NSError(domain: "invalid or oversized assembly artifact: " + url.path, code: 6)
    }
    let file = try FileHandle(forReadingFrom: url)
    defer { try? file.close() }
    let bytes = try file.read(upToCount: limit + 1) ?? Data()
    guard bytes.count == size else { throw NSError(domain: "assembly artifact size drift: " + url.path, code: 6) }
    return bytes
  }
  static func assembledWords(_ assembly: AssembledInput, for input: ConfiguredInput) throws -> [UInt32] {
    let byteLimit = ShaderLimits.default.words * MemoryLayout<UInt32>.size
    let toolHash = assembly.assembler_sha256.utf8
    guard input.stage == "module", assembly.source_sha256 == input.sha256,
          assembly.assembler_arguments == [input.path, "-o", "-"], assembly.assembler_path.hasPrefix("/"),
          !assembly.assembler_version.isEmpty, assembly.assembler_version.utf8.count <= 4096,
          toolHash.count == 64, toolHash.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }),
          assembly.binary_path.hasPrefix("/"), (20...byteLimit).contains(assembly.binary_bytes),
          assembly.binary_bytes % MemoryLayout<UInt32>.size == 0 else {
      throw NSError(domain: "invalid assembly provenance: " + input.source, code: 7)
    }
    let bytes = try boundedBytes(at: URL(fileURLWithPath: assembly.binary_path), limit: byteLimit)
    guard bytes.count == assembly.binary_bytes, digest(bytes) == assembly.binary_sha256 else {
      throw NSError(domain: "SPIR-V binary hash or size drift: " + input.source, code: 8)
    }
    let words = bytes.withUnsafeBytes { buffer in
      stride(from: 0, to: buffer.count, by: MemoryLayout<UInt32>.size).map { offset in
        UInt32(littleEndian: buffer.loadUnaligned(fromByteOffset: offset, as: UInt32.self))
      }
    }
    let swap = words[0] == 0x03022307
    let version = swap ? words[1].byteSwapped : words[1]
    guard (swap || words[0] == 0x07230203), version == assembly.spirv_version,
          (0x00010000...0x00010600).contains(version), version & 0xff0000ff == 0,
          words[3] != 0, words[4] == 0 else {
      throw NSError(domain: "SPIR-V binary header drift: " + input.source, code: 9)
    }
    return words
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
