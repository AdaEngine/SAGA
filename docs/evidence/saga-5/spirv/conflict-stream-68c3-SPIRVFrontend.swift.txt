import SAGAShaderIR

/// Independent bounded binary decoder. Returns RAW IR; semantic validation remains mandatory.
public struct SPIRVFrontend: BoundedShaderFrontend {
    public typealias Module = IRModule
    public let options: SPIRVFrontendOptions
    public let limits: ShaderLimits
    public var inputFormat: ShaderInputFormat { .spirv }
    public init(options: SPIRVFrontendOptions = .init(), limits: ShaderLimits = .default) {
        self.options = options; self.limits = limits
    }
    public func parse(_ input: ShaderInput, budget: inout ShaderBudget) throws -> IRModule {
        guard case .spirv(let words) = input else { throw ShaderFailure.unsupported("SPIR-V input required") }
        // The public budget-bearing SPI may be called without the legacy wrapper.
        guard words.count <= limits.words else { throw ShaderFailure.budgetExceeded(.words) }
        var decoder = SPVDecoder(options: options, budget: budget)
        defer { budget = decoder.budget }
        return try decoder.decode(words)
    }
}

struct SPVInstruction {
    let op: UInt32
    let a: [UInt32]
    let start: Int
    let end: Int
    var location: ShaderLocation? { (try? BinaryWordSpan(startWord: start, endWord: end)).map(ShaderLocation.binary) }
    func fail(_ message: String, code: ShaderDiagnosticCode = .invalidBinary) -> ShaderFailure {
        .diagnostics([.init(severity: .error, code: code, message: message, location: location)])
    }
    func count(_ n: Int) throws { guard a.count == n else { throw fail("Op\(op): expected \(n) operands, got \(a.count)") } }
    func minimum(_ n: Int) throws { guard a.count >= n else { throw fail("Op\(op): missing operands") } }
}

struct SPVFunctionRecord { let header: SPVInstruction; var body: [SPVInstruction] }
struct SPVPointer { let storage: UInt32; let base: UInt32 }
struct SPVFunctionType { let result: UInt32; let parameters: [UInt32] }
struct SPVEntry { let inst: SPVInstruction; let model: UInt32; let function: UInt32; let name: String; let interfaces: [UInt32] }
struct SPVConstant { let handle: IRHandle<IRConstant>; let type: UInt32; let unsigned: UInt32? }

struct SPVDecoder {
    let options: SPIRVFrontendOptions
    var budget: ShaderBudget
    var bound: UInt32 = 0
    var definitions: Set<UInt32> = []
    var references: [(UInt32, SPVInstruction)] = []
    var names: [UInt32: String] = [:]
    var decorations: [UInt32: [UInt32: [UInt32]]] = [:]
    var members: [UInt32: [UInt32: [UInt32: [UInt32]]]] = [:]
    var capabilities: Set<UInt32> = []
    var spillPointerTypes: [IRHandle<IRType>: IRHandle<IRType>] = [:]
    var types: [IRType] = []
    var typeMap: [UInt32: IRHandle<IRType>] = [:]
    var voids: Set<UInt32> = []
    var pointers: [UInt32: SPVPointer] = [:]
    var functionTypes: [UInt32: SPVFunctionType] = [:]
    var constants: [IRConstant] = []
    var constantMap: [UInt32: SPVConstant] = [:]
    var globalExpressions: [IRExpression] = []
    var globals: [IRGlobalVariable] = []
    var globalMap: [UInt32: (IRHandle<IRGlobalVariable>, UInt32)] = [:]
    var functions: [SPVFunctionRecord] = []
    var functionMap: [UInt32: IRHandle<IRFunction>] = [:]
    var entries: [SPVEntry] = []
    var modes: [UInt32: [UInt32: [UInt32]]] = [:]
    var memorySeen = false
    var section = 0
    var localPointerTypes: [IRHandle<IRType>: IRHandle<IRType>] = [:]

    mutating func charge(_ amount: Int = 1) throws { try budget.consume(.work, amount: amount) }
    mutating func node() throws { try budget.consume(.nodes) }
    func type(_ id: UInt32, _ i: SPVInstruction) throws -> IRHandle<IRType> {
        guard let t = typeMap[id] else { throw i.fail("Undefined or unsupported type %\(id)") }; return t
    }
    func kind(_ id: UInt32, _ i: SPVInstruction) throws -> IRTypeKind { types[try type(id,i).index].kind }
    mutating func define(_ id: UInt32, _ i: SPVInstruction) throws {
        try charge(); try node()
        guard id != 0, id < bound, definitions.insert(id).inserted else { throw i.fail("Invalid/duplicate result ID %\(id)") }
    }
    mutating func reference(_ id: UInt32, _ i: SPVInstruction) throws {
        try charge()
        guard id != 0, id < bound else { throw i.fail("ID %\(id) outside bound") }
        references.append((id,i))
    }
    mutating func addType(_ id: UInt32, _ k: IRTypeKind, _ i: SPVInstruction) throws {
        try node()
        typeMap[id] = try IRHandle(index: types.count)
        types.append(IRType(name: names[id], kind: k, location: i.location))
    }
    func scalar(_ id: UInt32, _ i: SPVInstruction) throws -> IRScalar {
        guard case .scalar(let s) = try kind(id,i) else { throw i.fail("Expected scalar type") }; return s
    }
    func string(_ a: [UInt32], from start: Int, _ i: SPVInstruction) throws -> (String, Int) {
        var bytes: [UInt8] = []
        for j in start..<a.count {
            let w = a[j]
            for k in 0..<4 {
                let byte = UInt8(truncatingIfNeeded: w >> (k * 8))
                if byte == 0 {
                    let s = String(decoding: bytes, as: UTF8.self)
                    guard k == 3 || w >> ((k + 1) * 8) == 0, Array(s.utf8) == bytes else { throw i.fail("Invalid UTF-8 or string padding") }
                    return (s, j + 1)
                }
                bytes.append(byte)
            }
        }
        throw i.fail("Unterminated SPIR-V string")
    }
    mutating func decode(_ input: [UInt32]) throws -> IRModule {
        try charge(input.count)
        let header = SPVInstruction(op: 0, a: [], start: 0, end: min(5,input.count))
        guard input.count >= 5 else { throw header.fail("Incomplete SPIR-V header") }
        let swapped: Bool
        switch input[0] { case 0x07230203: swapped = false; case 0x03022307: swapped = true; default: throw header.fail("Invalid magic") }
        // Charge before any normalization allocation, proportional to actual input, never ID bound.
        let w = swapped ? input.map(\.byteSwapped) : input
        guard w[1] >= 0x00010000, w[1] <= 0x00010600, w[1] & 0xff0000ff == 0,
              w[3] > 0, w[4] == 0 else { throw header.fail("Invalid version, ID bound or schema") }
        guard options.strictCapabilities else { throw header.fail("strictCapabilities=false mode is not implemented",code:.unsupportedConstruct) }
        bound = w[3]
        var cursor = 5
        var current: SPVFunctionRecord?
        while cursor < w.count {
            try budget.consume(.tokens); try charge()
            let count = Int(w[cursor] >> 16), op = w[cursor] & 0xffff
            let e = min(w.count, cursor + max(count,1))
            let i = SPVInstruction(op: op, a: [], start: cursor, end: e)
            guard count > 0, count <= w.count - cursor else { throw i.fail("Invalid/truncated instruction word count") }
            try charge(count); try node()
            let inst = SPVInstruction(op: op, a: Array(w[(cursor+1)..<(cursor+count)]), start: cursor, end: cursor+count)
            cursor += count
            if current != nil {
                if op == 56 {
                    try inst.count(0); if let f = current { functions.append(f) }; current = nil
                } else {
                    guard op != 54 else { throw inst.fail("Nested OpFunction") }
                    try registerFunctionResult(inst)
                    current?.body.append(inst)
                }
                continue
            }
            if op == 54 {
                guard memorySeen else { throw inst.fail("Function before memory model") }
                section = 10; try inst.count(4); try define(inst.a[1],inst)
                functionMap[inst.a[1]] = try IRHandle(index: functions.count)
                current = SPVFunctionRecord(header: inst, body: [])
                continue
            }
            try moduleInstruction(inst)
        }
        guard current == nil, memorySeen, capabilities.contains(1) else { throw header.fail("Missing memory model/Shader capability or unterminated function") }
        for (id,i) in references { try charge(); guard definitions.contains(id) else { throw i.fail("Unresolved ID %\(id)") } }
        // Synthesize function-space pointers for phi spills; never alter the shared IR API.
        let originalCount = types.count
        for j in 0..<originalCount {
            try charge(); let base = try IRHandle<IRType>(index:j)
            if case .pointer = types[j].kind { continue }
            try node(); let handle = try IRHandle<IRType>(index:types.count)
            localPointerTypes[base] = handle
            types.append(IRType(kind:.pointer(base:base,space:.function)))
        }
        // Pre-intern function spill pointer types before publishing immutable module snapshots.
        for index in types.indices {
            try charge(); try node()
            let base = try IRHandle<IRType>(index: index)
            let pointer = try IRHandle<IRType>(index: types.count)
            spillPointerTypes[base] = pointer
            types.append(IRType(kind: .pointer(base: base, space: .function)))
        }
        var lowered: [IRFunction] = []
        try validateDecorations(header)
        for f in functions {
            var lower = SPVFunctionLowerer(module: self, record: f, budget: budget)
            let value = try lower.lower(); budget = lower.budget; lowered.append(value)
        }
        var entryPoints: [IREntryPoint] = []
        var entryNames: Set<String> = []
        for e in entries {
            try charge()
            guard entryNames.insert("\(e.model):\(e.name)").inserted,
                  let function = functionMap[e.function], lowered.indices.contains(function.index) else { throw e.inst.fail("Duplicate or undefined entry point") }
            let f = lowered[function.index]
            guard f.arguments.isEmpty, f.result == nil else { throw e.inst.fail("Entry point must be void without parameters") }
            guard e.interfaces.isEmpty else { throw e.inst.fail("Entry-point IO wrapper lowering unavailable", code: .unsupportedConstruct) }
            let stage: IRStage
            switch e.model { case 0: stage = .vertex; case 4: stage = .fragment; case 5: stage = .compute; default: throw e.inst.fail("Unsupported execution model \(e.model)", code: .unsupportedConstruct) }
            guard stage == .fragment || (modes[e.function]?[7] == nil && modes[e.function]?[9] == nil) else { throw e.inst.fail("Fragment mode on non-fragment entry point") }
            guard stage != .fragment || modes[e.function]?[7] != nil else { throw e.inst.fail("Fragment entry point requires OriginUpperLeft") }
            guard stage != .fragment || modes[e.function]?[7] != nil else { throw e.inst.fail("Fragment needs OriginUpperLeft execution mode") }
            guard stage == .fragment || (modes[e.function]?[7] == nil && modes[e.function]?[9] == nil) else { throw e.inst.fail("Fragment execution mode on another stage") }
            let local = modes[e.function]?[17]
            guard stage != .compute || local != nil else { throw e.inst.fail("Compute entry point needs LocalSize") }
            guard stage == .compute || local == nil else { throw e.inst.fail("LocalSize on non-compute entry point") }
            let size = local.map { IRWorkgroupSize(x:$0[0],y:$0[1],z:$0[2]) } ?? IRWorkgroupSize()
            entryPoints.append(IREntryPoint(name:e.name,stage:stage,function:function,earlyDepthTest:modes[e.function]?[9] == nil ? nil : .force,workgroupSize:size,location:e.inst.location))
        }
        for (id,_) in modes { guard entries.contains(where: { $0.function == id }) else { throw header.fail("ExecutionMode target is not an entry point") } }
        return IRModule(types: IRTable(types), constants:IRTable(constants), globalVariables:IRTable(globals), globalExpressions:IRTable(globalExpressions), functions:IRTable(lowered), entryPoints:entryPoints)
    }

    mutating func order(_ s: Int, _ i: SPVInstruction) throws {
        guard s >= section else { throw i.fail("Instruction appears in wrong logical layout section") }; section = s
    }
    mutating func moduleInstruction(_ i: SPVInstruction) throws {
        let a = i.a
        switch i.op {
        case 0: try i.count(0) // OpNop has no semantic effect.
        case 17:
            try order(1,i); try i.count(1)
            // Safe implemented declaration surface; strict=false does not promise unknown capabilities.
            guard [0,1,9,10,11,22].contains(a[0]) else { throw i.fail("Capability \(a[0]) is not implemented (strict=\(options.strictCapabilities))",code:.unsupportedCapability) }
            guard capabilities.insert(a[0]).inserted else { throw i.fail("Duplicate capability") }
        case 14:
            try order(3,i); try i.count(2)
            guard !memorySeen, a == [0,1] else { throw i.fail("Only Logical GLSL450 memory model is implemented",code:.unsupportedConstruct) }; memorySeen = true
        case 15:
            try order(4,i); try i.minimum(3); try reference(a[1],i)
            let (name,end) = try string(a,from:2,i)
            guard !name.isEmpty else { throw i.fail("Empty entry name") }
            for id in a.dropFirst(end) { try reference(id,i) }
            entries.append(SPVEntry(inst:i,model:a[0],function:a[1],name:name,interfaces:Array(a.dropFirst(end))))
        case 16:
            try order(5,i); try i.minimum(2); try reference(a[0],i)
            switch a[1] {
            case 17: try i.count(5); guard a[2] > 0, a[3] > 0, a[4] > 0 else { throw i.fail("Zero LocalSize") }
            case 7,9: try i.count(2)
            default: throw i.fail("Unsupported execution mode \(a[1])",code:.unsupportedConstruct)
            }
            guard modes[a[0]]?[a[1]] == nil else { throw i.fail("Duplicate execution mode") }
            modes[a[0],default:[:]][a[1]] = Array(a.dropFirst(2))
        case 5:
            try order(6,i); try i.minimum(2); try reference(a[0],i)
            let (n,e) = try string(a,from:1,i); guard e == a.count, names[a[0]] == nil else { throw i.fail("Malformed/duplicate OpName") }; names[a[0]] = n
        case 71,72:
            try order(7,i); try i.minimum(i.op == 71 ? 2 : 3); try reference(a[0],i)
            let index = i.op == 71 ? 1 : 2, dec = a[index], payload = Array(a.dropFirst(index+1))
            let expected: Int
            switch dec { case 2: expected=0; case 6,33,34,35: expected=1
            default: throw i.fail("Decoration \(dec) not implemented",code:.unsupportedConstruct) }
            guard payload.count == expected else { throw i.fail("Bad decoration operand count") }
            if i.op == 71 {
                guard decorations[a[0]]?[dec] == nil else { throw i.fail("Duplicate decoration") }
                decorations[a[0],default:[:]][dec] = payload
            } else {
                guard members[a[0]]?[a[1]]?[dec] == nil else { throw i.fail("Duplicate member decoration") }
                var v = members[a[0],default:[:]]; v[a[1],default:[:]][dec]=payload; members[a[0]]=v
            }
        case 19...33, 41...44,46,59:
            try order(8,i); guard memorySeen else { throw i.fail("Declaration before memory model") }
            try declaration(i)
        default: throw i.fail("Unsupported module opcode \(i.op)",code:.unsupportedConstruct)
        }
    }

    mutating func declaration(_ i: SPVInstruction) throws {
        let a=i.a
        switch i.op {
        case 19: try i.count(1); try define(a[0],i); voids.insert(a[0])
        case 20: try i.count(1); try define(a[0],i); try addType(a[0],.scalar(.bool),i)
        case 21:
            try i.count(3); try define(a[0],i)
            guard [16,32,64].contains(a[1]), a[2] <= 1 else { throw i.fail("Unsupported integer width/sign",code:.unsupportedConstruct) }
            guard a[1] != 64 || capabilities.contains(11), a[1] != 16 || capabilities.contains(22) else { throw i.fail("Missing integer capability",code:.unsupportedCapability) }
            try addType(a[0],.scalar(.init(kind:a[2] == 0 ? .uint:.sint,width:UInt8(a[1]/8))),i)
        case 22:
            try i.count(2); try define(a[0],i)
            guard [16,32,64].contains(a[1]) else { throw i.fail("Unsupported float width",code:.unsupportedConstruct) }
            guard a[1] != 64 || capabilities.contains(10), a[1] != 16 || capabilities.contains(9) else { throw i.fail("Missing float capability",code:.unsupportedCapability) }
            try addType(a[0],.scalar(.init(kind:.float,width:UInt8(a[1]/8))),i)
        case 23:
            try i.count(3); try define(a[0],i)
            guard let n=UInt8(exactly:a[2]).flatMap(IRVectorSize.init(rawValue:)) else { throw i.fail("Invalid vector size") }
            try addType(a[0],.vector(size:n,scalar:scalar(a[1],i)),i)
        case 24:
            try i.count(3); try define(a[0],i)
            guard capabilities.contains(0), case .vector(let rows,let s) = try kind(a[1],i), s.kind == .float,
                  let columns=UInt8(exactly:a[2]).flatMap(IRVectorSize.init(rawValue:)) else { throw i.fail("Invalid matrix type/capability") }
            try addType(a[0],.matrix(columns:columns,rows:rows,scalar:s),i)
        case 28,29:
            try i.count(i.op == 28 ? 3:2); try define(a[0],i); let base=try type(a[1],i)
            guard let stride=decorations[a[0]]?[6]?.first, stride>0 else { throw i.fail("ArrayStride required; layout inference unavailable",code:.unsupportedConstruct) }
            let size: IRArraySize
            if i.op == 28 { guard let n=constantMap[a[2]]?.unsigned, n>0 else { throw i.fail("Invalid array length") }; size = .constant(n) } else { size = .dynamic }
            try addType(a[0],.array(base:base,size:size,stride:stride),i)
        case 30:
            try i.minimum(1); try define(a[0],i); var fields:[IRStructMember]=[]; var span:UInt32=0
            for (j,id) in a.dropFirst().enumerated() {
                try charge(); let t=try type(id,i)
                guard let off=members[a[0]]?[UInt32(j)]?[35]?.first else { throw i.fail("Explicit member Offset required",code:.unsupportedConstruct) }
                let size=try byteSize(t,i)
                let (end,overflow)=off.addingReportingOverflow(size); guard !overflow, off>=span else { throw i.fail("Invalid/overlapping struct offsets") }
                span=end; fields.append(IRStructMember(type:t,offset:off,location:i.location))
            }
            try addType(a[0],.structure(members:fields,span:span),i)
        case 32:
            try i.count(3); try define(a[0],i); let base=try type(a[2],i); let space=try storage(a[1],i)
            pointers[a[0]]=SPVPointer(storage:a[1],base:a[2]); try addType(a[0],.pointer(base:base,space:space),i)
        case 33:
            try i.minimum(2); try define(a[0],i)
            guard voids.contains(a[1]) || typeMap[a[1]] != nil else { throw i.fail("Undefined function return type") }
            for id in a.dropFirst(2) { _=try type(id,i) }
            functionTypes[a[0]]=SPVFunctionType(result:a[1],parameters:Array(a.dropFirst(2)))
        case 41,42,43,44,46: try constant(i)
        case 59:
            guard a.count == 3 || a.count == 4 else { throw i.fail("Invalid global variable operands") }; try define(a[1],i)
            guard let p=pointers[a[0]], p.storage == a[2], p.storage != 7 else { throw i.fail("Invalid global pointer/storage class") }
            let space=try storage(a[2],i), t=try type(p.base,i)
            var initExpr:IRHandle<IRExpression>?
            if a.count == 4 { guard a[2] == 6, let c=constantMap[a[3]], c.type == p.base else { throw i.fail("Bad global initializer") }; initExpr=constants[c.handle.index].initializer }
            let set=decorations[a[1]]?[34]?.first, binding=decorations[a[1]]?[33]?.first
            guard (set == nil) == (binding == nil) else { throw i.fail("Incomplete descriptor binding") }
            if a[2] == 2 || a[2] == 12 {
                guard set != nil, decorations[p.base]?[2] != nil else { throw i.fail("Resource needs Block type and descriptor binding") }
            } else if set != nil { throw i.fail("Descriptor binding on non-resource") }
            let rb:IRResourceBinding? = if let set,let binding { .init(group:set,binding:binding) } else { nil }
            let h=try IRHandle<IRGlobalVariable>(index:globals.count)
            globals.append(.init(name:names[a[1]],space:space,binding:rb,type:t,initializer:initExpr,location:i.location)); globalMap[a[1]]=(h,a[0])
        default: throw i.fail("Unsupported declaration opcode \(i.op)",code:.unsupportedConstruct)
        }
    }
    func storage(_ id:UInt32,_ i:SPVInstruction) throws -> IRAddressSpace {
        switch id {
        case 4: return .workgroup
        case 6: return .private
        case 7: return .function
        case 2: return .uniform
        case 12: return .storage(access:[.load,.store])
        case 1,3: throw i.fail("Input/Output wrapper and coordinate-space lowering not yet implemented",code:.unsupportedConstruct)
        default: throw i.fail("Unsupported storage class \(id)",code:.unsupportedConstruct)
        }
    }
    func byteSize(_ t:IRHandle<IRType>,_ i:SPVInstruction,depth:Int=0) throws -> UInt32 {
        guard depth < 64 else { throw i.fail("Layout depth limit",code:.unsupportedConstruct) }
        switch types[t.index].kind {
        case .scalar(let s):return UInt32(s.width)
        case .vector(let n,let s):return UInt32(n.rawValue)*UInt32(s.width)
        case .structure(_,let span):return span
        case .array(_, .constant(let n),let stride):let (r,o)=n.multipliedReportingOverflow(by:stride);guard !o else { throw i.fail("Array size overflow") };return r
        default:throw i.fail("Layout calculation for this member type is unavailable",code:.unsupportedConstruct)
        }
    }
    mutating func constant(_ i:SPVInstruction) throws {
        let a=i.a; try i.minimum(2); try define(a[1],i); let t=try type(a[0],i)
        let k=types[t.index].kind; var u:UInt32?; let expr:IRExpressionKind
        if i.op == 41 || i.op == 42 {
            try i.count(2); guard k == .scalar(.bool) else { throw i.fail("Boolean constant type mismatch") }; expr = .literal(.bool(i.op == 41))
        } else if i.op == 46 {
            try i.count(2); expr = .zeroValue(t); if k == .scalar(.u32) {u=0}
        } else if i.op == 44 {
            var components:[IRHandle<IRExpression>]=[]
            let expected:[IRHandle<IRType>]
            switch k {
            case .vector(let n,let s):
                try charge(types.count)
                guard let scalarID=types.indices.first(where:{types[$0].kind == .scalar(s)}) else { throw i.fail("Missing scalar type") }
                expected=Array(repeating:try IRHandle(index:scalarID),count:Int(n.rawValue))
            case .structure(let ms,_):expected=ms.map(\.type)
            case .array(let b,.constant(let n),_):guard UInt64(n) == UInt64(a.count-2) else {throw i.fail("Composite count mismatch")}; try charge(Int(n)); expected=Array(repeating:b,count:Int(n))
            default:throw i.fail("Composite type unsupported",code:.unsupportedConstruct)
            }
            guard a.count-2 == expected.count else {throw i.fail("Composite component count mismatch")}
            for (j,id) in a.dropFirst(2).enumerated() {try charge();guard let c=constantMap[id], typeMap[c.type] == expected[j] else {throw i.fail("Composite type mismatch")};components.append(constants[c.handle.index].initializer)}
            expr = .compose(type:t,components:components)
        } else {
            guard case .scalar(let s)=k else {throw i.fail("Scalar constant expected")}; try i.count(s.width==8 ? 4:3)
            switch (s.kind,s.width) {
            case (.uint,4):expr = .literal(.u32(a[2]));u=a[2]
            case (.sint,4):expr = .literal(.i32(Int32(bitPattern:a[2])))
            case (.float,4):expr = .literal(.f32(Float(bitPattern:a[2])))
            case (.float,2):guard a[2]<=0xffff else {throw i.fail("Bad f16 payload")};expr = .literal(.f16Bits(UInt16(a[2])))
            case (.uint,2):guard a[2]<=0xffff else {throw i.fail("Bad u16 payload")};expr = .literal(.u16(UInt16(a[2])))
            case (.sint,2):expr = .literal(.i16(Int16(truncatingIfNeeded:a[2])))
            case (.uint,8):expr = .literal(.u64(UInt64(a[2]) | UInt64(a[3])<<32))
            case (.sint,8):expr = .literal(.i64(Int64(bitPattern:UInt64(a[2]) | UInt64(a[3])<<32)))
            case (.float,8):expr = .literal(.f64(Double(bitPattern:UInt64(a[2]) | UInt64(a[3])<<32)))
            default:throw i.fail("Unsupported constant type",code:.unsupportedConstruct)
            }
        }
        try node(); let h=try IRHandle<IRExpression>(index:globalExpressions.count); globalExpressions.append(.init(type:t,kind:expr,location:i.location))
        let c=try IRHandle<IRConstant>(index:constants.count);constants.append(.init(name:names[a[1]],type:t,initializer:h,location:i.location));constantMap[a[1]]=SPVConstant(handle:c,type:a[0],unsigned:u)
    }
    mutating func validateDecorations(_ header:SPVInstruction) throws {
        for (id,ds) in decorations {
            for (dec,_) in ds {
                try charge()
                switch dec {
                case 2:
                    guard let t=typeMap[id], case .structure = types[t.index].kind else {throw header.fail("Block decoration must target struct")}
                case 6:
                    guard let t=typeMap[id], case .array = types[t.index].kind else {throw header.fail("ArrayStride target must be array")}
                case 33,34:
                    guard globalMap[id] != nil else {throw header.fail("Descriptor decoration target must be global")}
                default:throw header.fail("Unconsumed decoration",code:.unsupportedConstruct)
                }
            }
        }
        for (id,ms) in members {
            guard let t=typeMap[id], case .structure(let fields,_) = types[t.index].kind else {throw header.fail("Member decoration target must be struct")}
            for (index,ds) in ms {
                try charge(ds.count)
                guard UInt64(index)<UInt64(fields.count),Set(ds.keys)==[35] else {throw header.fail("Invalid member decoration")}
            }
        }
    }
    mutating func registerFunctionResult(_ i:SPVInstruction) throws {
        switch i.op {
        case 55,59,61,65,80,81,83,124,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,182,184,186,188,190,194,195,196,197,198,199,200,245,57:
            try i.minimum(2);try define(i.a[1],i)
        case 248:try i.count(1);try define(i.a[0],i)
        case 0,62,246,247,249,250,251,252,253,254,255:break
        default:throw i.fail("Unsupported function opcode \(i.op)",code:.unsupportedConstruct)
        }
    }
}
