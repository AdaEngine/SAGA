import SAGAShaderIR

enum SPVDemote {
    static let capability: UInt32 = 5379
    static let opcode: UInt32 = 5380
    static let coreVersion: UInt32 = 0x00010600
    static let extensionName = "SPV_EXT_demote_to_helper_invocation"
}

enum SPVMemoryModel: UInt32 {
    case simple = 0
    case glsl450 = 1
    case vulkan = 3
}

enum SPVVulkanMemoryModel {
    static let capability: UInt32 = 5345
    static let deviceScopeCapability: UInt32 = 5346
    static let coreVersion: UInt32 = 0x00010500
    static let extensionName = "SPV_KHR_vulkan_memory_model"
}

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
struct SPVConstant { let handle: IRHandle<IRConstant>; let type: UInt32; let nonnegativeInteger: UInt32? }

struct SPVDecoder {
    let options: SPIRVFrontendOptions
    var budget: ShaderBudget
    var bound: UInt32 = 0
    var version: UInt32 = 0
    var definitions: Set<UInt32> = []
    var references: [(UInt32, SPVInstruction)] = []
    var names: [UInt32: String] = [:]
    var memberNames: [UInt32: [UInt32: String]] = [:]
    var debugStrings: [UInt32: String] = [:]
    var sourceFiles: [(UInt32, SPVInstruction)] = []
    var glslInstructionSets: Set<UInt32> = []
    var nonSemanticDebugSets: Set<UInt32> = []
    var debugPrintfSets: Set<UInt32> = []
    var nonSemanticDebugResults: Set<UInt32> = []
    var glslMathTypes: [SPVGLSLShape: IRHandle<IRType>] = [:]
    var glslDecompositionTypes: [IRPredeclaredTypeBinding] = []
    var atomicCompareTypes: [IRPredeclaredTypeBinding] = []
    var hasGLSLMath = false
    var decorations: [UInt32: [UInt32: [UInt32]]] = [:]
    var storageAccessDecorations = SPVStorageAccessDecorations()
    var storageMemoryTypes: Set<IRHandle<IRType>> = []
    var uniformBufferStandardLayout = false
    var members: [UInt32: [UInt32: [UInt32: [UInt32]]]] = [:]
    var capabilities: Set<UInt32> = []
    var extensions: Set<String> = []
    var storageOnly16BitTypes: Set<IRHandle<IRType>> = []
    var types: [IRType] = []
    var typeMap: [UInt32: IRHandle<IRType>] = [:]
    var matrixColumnTypes: [IRHandle<IRType>: IRHandle<IRType>] = [:]
    var voids: Set<UInt32> = []
    var pointers: [UInt32: SPVPointer] = [:]
    var sampledImageTypes: [UInt32: UInt32] = [:]
    var imageDepths: [UInt32: UInt32] = [:]
    var storageImageSampledTypes: [UInt32: UInt32] = [:]
    var storageImageParameterTypes: [UInt32: IRHandle<IRType>] = [:]
    var descriptorArrayElements: [UInt32: UInt32] = [:]
    var hasDrefSampling = false
    var hasImageTexelPointers = false
    var drefParameterTypes: [UInt32: IRHandle<IRType>] = [:]
    var functionTypes: [UInt32: SPVFunctionType] = [:]
    var constants: [IRConstant] = []
    var constantMap: [UInt32: SPVConstant] = [:]
    var overrides: [IROverride] = []
    var overrideMap: [UInt32: (handle: IRHandle<IROverride>, type: UInt32)] = [:]
    var globalExpressions: [IRExpression] = []
    var globals: [IRGlobalVariable] = []
    var globalMap: [UInt32: (IRHandle<IRGlobalVariable>, UInt32)] = [:]
    var storageAliases: [IRHandle<IRGlobalVariable>: IRHandle<IRGlobalVariable>] = [:]
    var storageAliasInterfaceUses: [Set<IRHandle<IRGlobalVariable>>] = []
    var stageInterfaces: [UInt32: [SPVInterfaceField]] = [:]
    var forwardedInputGlobals: Set<IRHandle<IRGlobalVariable>> = []
    var builtinFunctionInputs: [[IRHandle<IRGlobalVariable>]] = []
    var atomicPaths: [IRHandle<IRGlobalVariable>: Set<[SPVAtomicStep]>] = [:]
    var workgroupSizeBuiltin: IRWorkgroupSize?
    var interfaceTypes: Set<UInt32> = []
    var omittedInterfaceMembers: [IRHandle<IRGlobalVariable>: Set<UInt32>] = [:]
    var interfaceGlobals: Set<IRHandle<IRGlobalVariable>> = []
    var functions: [SPVFunctionRecord] = []
    var functionMap: [UInt32: IRHandle<IRFunction>] = [:]
    var functionValueTypes: [UInt32: UInt32] = [:]
    var entries: [SPVEntry] = []
    var modes: [UInt32: [UInt32: [UInt32]]] = [:]
    var memoryModel: SPVMemoryModel?
    var usesVulkanMemoryModel: Bool { memoryModel == .vulkan }
    var usesSimpleMemoryModel: Bool { memoryModel == .simple }
    var section = 0
    var localPointerTypes: [IRHandle<IRType>: IRHandle<IRType>] = [:]
    var controlBoolType: IRHandle<IRType>?
    var hasLoops = false
    var implicitStructs: Set<IRHandle<IRType>> = []
    var implicitArrays: Set<IRHandle<IRType>> = []
    var incompleteMatrixLayouts: Set<IRHandle<IRType>> = []
    var valueLayouts: [IRHandle<IRType>: SPVValueLayout] = [:]
    var verifiedBufferLayouts: Set<IRHandle<IRType>> = []

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
    mutating func internType(_ kind: IRTypeKind) throws -> IRHandle<IRType> {
        try charge(types.count)
        if let index = types.firstIndex(where: { $0.kind == kind }) { return try IRHandle(index: index) }
        try node()
        let handle = try IRHandle<IRType>(index: types.count)
        types.append(.init(kind: kind))
        return handle
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
        version = w[1]
        bound = w[3]
        var cursor = 5
        var current: SPVFunctionRecord?
        var blockContainsDebug = false
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
                    if op == 248 { blockContainsDebug = false }
                    if try skipNonSemanticDebug(inst) {
                        blockContainsDebug = true
                        continue
                    }
                    if op == 245, blockContainsDebug {
                        throw inst.fail("OpPhi must precede non-semantic debug instructions in the block")
                    }
                    try registerFunctionResult(inst)
                    current?.body.append(inst)
                }
                continue
            }
            if op == 54 {
                guard memoryModel != nil else { throw inst.fail("Function before memory model") }
                if functions.isEmpty { try prepare16BitStorage(header) }
                section = 10; try inst.count(4); try define(inst.a[1],inst)
                functionMap[inst.a[1]] = try IRHandle(index: functions.count)
                current = SPVFunctionRecord(header: inst, body: [])
                blockContainsDebug = false
                continue
            }
            try moduleInstruction(inst)
        }
        guard current == nil, memoryModel != nil, capabilities.contains(1) else { throw header.fail("Missing memory model/Shader capability or unterminated function") }
        if functions.isEmpty { try prepare16BitStorage(header) }
        try validateDescriptorIndexingCapabilities(at: header)
        if version < 0x00010300, !capabilities.isDisjoint(with: SPVSubgroup.capabilities) {
            throw header.fail("Subgroup capabilities require SPIR-V 1.3", code: .unsupportedCapability)
        }
        if capabilities.contains(SPVFloatAtomicAdd.capability), !extensions.contains(SPVFloatAtomicAdd.extensionName) {
            throw header.fail("AtomicFloat32AddEXT requires SPV_EXT_shader_atomic_float_add", code: .unsupportedCapability)
        }
        if version < SPVDemote.coreVersion, capabilities.contains(SPVDemote.capability),
           !extensions.contains(SPVDemote.extensionName) {
            throw header.fail("DemoteToHelperInvocation requires SPIR-V 1.6 or SPV_EXT_demote_to_helper_invocation",
                              code: .unsupportedCapability)
        }
        if capabilities.contains(SPIRVStorageImageFormat.int64ImageCapability),
           !extensions.contains(SPIRVStorageImageFormat.int64ImageExtension) {
            throw header.fail("Int64ImageEXT requires SPV_EXT_shader_image_int64", code: .unsupportedCapability)
        }
        if capabilities.contains(SPVBarycentric.capability), !extensions.contains(SPVBarycentric.extensionName) {
            throw header.fail("FragmentBarycentricKHR requires SPV_KHR_fragment_shader_barycentric", code: .unsupportedCapability)
        }
        if version < 0x00010300, capabilities.contains(SPVMultiview.capability), !extensions.contains(SPVMultiview.extensionName) {
            throw header.fail("MultiView requires SPIR-V 1.3 or SPV_KHR_multiview", code: .unsupportedCapability)
        }
        if version < 0x00010300, capabilities.contains(SPVDrawParameters.capability),
           !extensions.contains(SPVDrawParameters.extensionName) {
            throw header.fail("DrawParameters requires SPIR-V 1.3 or SPV_KHR_shader_draw_parameters",
                              code: .unsupportedCapability)
        }
        for (id,i) in references {
            try charge()
            guard definitions.contains(id) else { throw i.fail("Unresolved ID %\(id)") }
            guard !nonSemanticDebugResults.contains(id) || i.op == 5 else {
                throw i.fail("Shader debug result %\(id) cannot be used by a semantic instruction")
            }
        }
        for (id, instruction) in sourceFiles {
            try charge()
            guard debugStrings[id] != nil else { throw instruction.fail("Source file must reference OpString") }
        }
        try orderFunctions()
        try prepareGLSLMathTypes()
        if hasLoops, controlBoolType == nil {
            try node()
            controlBoolType = try IRHandle(index: types.count)
            types.append(.init(kind: .scalar(.bool)))
        }
        // Synthesize function-space pointers for phi spills; never alter the shared IR API.
        let originalCount = types.count
        for j in 0..<originalCount {
            try charge(); let base = try IRHandle<IRType>(index:j)
            switch types[j].kind {
            case .pointer, .image, .sampler, .bindingArray, .array(_, .dynamic, _): continue
            case .structure(let members, _):
                // An unsized storage block can be referenced but cannot occupy a function spill slot.
                if let tail = members.last, case .array(_, .dynamic, _) = types[tail.type.index].kind { continue }
            default: break
            }
            try node(); let handle = try IRHandle<IRType>(index:types.count)
            localPointerTypes[base] = handle
            types.append(IRType(kind:.pointer(base:base,space:.function)))
        }
        var lowered: [IRFunction] = []
        try validateDecorations(header)
        try prepareStorageMemoryDecorations(header)
        try prepareStorageAccessDecorations(header)
        try prepareStorageAliases(header)
        try prepareImageAtomicResources()
        if hasDrefSampling { try prepareDrefResources() }
        for f in functions {
            var lower = SPVFunctionLowerer(module: self, record: f, budget: budget)
            let value = try lower.lower()
            budget = lower.budget
            types = lower.module.types
            atomicCompareTypes = lower.module.atomicCompareTypes
            lowered.append(value)
            builtinFunctionInputs.append(lower.builtinInputs)
            storageAliasInterfaceUses.append(lower.storageAliasInterfaceUses)
            for (global, paths) in lower.atomicPaths {
                try charge(paths.count)
                atomicPaths[global, default: []].formUnion(paths)
            }
        }
        try upgradeAtomicGlobals(in: &lowered, at: header)
        var entryPoints: [IREntryPoint] = []
        var entryNames: Set<String> = []
        for e in entries {
            try charge()
            guard entryNames.insert("\(e.model):\(e.name)").inserted,
                  let function = functionMap[e.function], lowered.indices.contains(function.index) else { throw e.inst.fail("Duplicate or undefined entry point") }
            let f = lowered[function.index]
            guard f.arguments.count == builtinFunctionInputs[function.index].count, f.result == nil else {
                throw e.inst.fail("Entry point must be void without parameters")
            }
            let stage: IRStage
            switch e.model { case 0: stage = .vertex; case 4: stage = .fragment; case 5: stage = .compute; default: throw e.inst.fail("Unsupported execution model \(e.model)", code: .unsupportedConstruct) }
            let executionModes = modes[e.function] ?? [:]
            let depthTest = try entryDepthTest(executionModes, at: e.inst)
            guard stage == .fragment || (executionModes[7] == nil && executionModes[12] == nil && depthTest == nil) else {
                throw e.inst.fail("Fragment mode on non-fragment entry point")
            }
            guard stage != .fragment || executionModes[7] != nil else { throw e.inst.fail("Fragment entry point requires OriginUpperLeft") }
            let local = executionModes[17]
            guard stage != .compute || local != nil || workgroupSizeBuiltin != nil else {
                throw e.inst.fail("Compute entry point needs a workgroup size")
            }
            guard stage == .compute || local == nil else { throw e.inst.fail("LocalSize on non-compute entry point") }
            let size = (stage == .compute ? workgroupSizeBuiltin : nil)
                ?? local.map { IRWorkgroupSize(x:$0[0],y:$0[1],z:$0[2]) } ?? IRWorkgroupSize()
            let target = try lowerStageInterface(e, stage: stage, function: function, functions: &lowered)
            entryPoints.append(IREntryPoint(name:e.name,stage:stage,function:target,earlyDepthTest:depthTest,workgroupSize:size,location:e.inst.location))
        }
        for (id,_) in modes { guard entries.contains(where: { $0.function == id }) else { throw header.fail("ExecutionMode target is not an entry point") } }
        try compactStorageAliases(in: &lowered, at: header)
        let diagnosticFilters: [IRDiagnosticFilter] = options.allowNonUniformDerivatives
            ? [.init(rule: .derivativeUniformity, severity: .off)] : []
        let diagnosticFilter = diagnosticFilters.isEmpty ? nil : try IRHandle<IRDiagnosticFilter>(index: 0)
        return IRModule(types: IRTable(types), specialTypes: .init(predeclared: glslDecompositionTypes + atomicCompareTypes),
                        constants:IRTable(constants), overrides:IRTable(overrides), globalVariables:IRTable(globals),
                        globalExpressions:IRTable(globalExpressions), functions:IRTable(lowered), entryPoints:entryPoints,
                        diagnosticFilters: IRTable(diagnosticFilters), diagnosticFilter: diagnosticFilter,
                        uniformBufferStandardLayout: uniformBufferStandardLayout)
    }

    mutating func order(_ s: Int, _ i: SPVInstruction) throws {
        guard s >= section else { throw i.fail("Instruction appears in wrong logical layout section") }; section = s
    }

    mutating func orderFunctions() throws {
        try charge(entries.count)
        let entryIDs = Set(entries.map(\.function))
        var active: Set<Int> = []
        var complete: Set<Int> = []
        var ordered: [SPVFunctionRecord] = []
        for root in functions.indices where !complete.contains(root) {
            var pending = [(index: root, location: functions[root].header, finishing: false)]
            while let visit = pending.popLast() {
                try charge()
                if visit.finishing {
                    active.remove(visit.index)
                    complete.insert(visit.index)
                    ordered.append(functions[visit.index])
                    continue
                }
                if complete.contains(visit.index) { continue }
                guard active.insert(visit.index).inserted else {
                    throw visit.location.fail("Recursive function calls are not allowed")
                }
                pending.append((visit.index, visit.location, true))
                for call in functions[visit.index].body.reversed() {
                    try charge()
                    guard call.op == 57 else { continue }
                    try call.minimum(3)
                    guard let callee = functionMap[call.a[2]] else { throw call.fail("Undefined function") }
                    guard !entryIDs.contains(call.a[2]) else { throw call.fail("Cannot call an entry point") }
                    pending.append((callee.index, call, false))
                }
            }
        }
        functions = ordered
        for (index, function) in functions.enumerated() {
            try charge()
            functionMap[function.header.a[1]] = try IRHandle(index: index)
        }
    }

    private func entryDepthTest(_ modes: [UInt32: [UInt32]], at instruction: SPVInstruction) throws -> IREarlyDepthTest? {
        let hints: [(UInt32, IRConservativeDepth)] = [(14, .greaterEqual), (15, .lessEqual), (16, .unchanged)]
        let selected = hints.filter { modes[$0.0] != nil }
        guard selected.count <= 1 else {
            throw instruction.fail("Fragment entry point declares conflicting conservative depth modes")
        }
        // Forced tests already run before shading. A conservative hint only
        // permits early testing, so keep the stronger execution requirement.
        if modes[9] != nil { return .force }
        return selected.first.map { .allow($0.1) }
    }

    mutating func moduleInstruction(_ i: SPVInstruction) throws {
        let a = i.a
        switch i.op {
        case 0: try i.count(0) // OpNop has no semantic effect.
        case 17:
            try order(1,i); try i.count(1)
            // Safe implemented declaration surface; strict=false does not promise unknown capabilities.
            let implemented: [UInt32] = [0, 1, SPVPrimitiveIndex.geometryCapability, SPVPrimitiveIndex.tessellationCapability,
                                         9, 10, 11, 22, 32, 35, SPVFloatAtomicAdd.capability,
                                         SPVImageTexelPointer.int64AtomicsCapability, SPIRVStorageImageFormat.int64ImageCapability,
                                         SPVBarycentric.capability, SPVMultiview.capability, SPVDrawParameters.capability,
                                         SPVDemote.capability, SPVVulkanMemoryModel.capability, SPVVulkanMemoryModel.deviceScopeCapability,
                                         SPVStorageImage.multisampleCapability,
                                         SPVStorageImage.sampled1DCapability, SPVStorageImage.image1DCapability,
                                         SPVStorageImage.extendedFormatsCapability, SPVImageQuery.capability, SPVDerivative.controlCapability,
                                         SPVSampledImage.imageCubeArrayCapability, SPVSampledImage.cubeArrayCapability,
                                         SPVDescriptorIndexing.dynamicSampledArray, SPVDescriptorIndexing.shaderNonUniform,
                                         SPVDescriptorIndexing.runtimeDescriptorArray,
                                         SPVDescriptorIndexing.nonUniformSampledArray, SPVDescriptorIndexing.dynamicStorageArray,
                                         SPVDescriptorIndexing.nonUniformStorageArray, SPVDescriptorIndexing.dynamicUniformBufferArray,
                                         SPVDescriptorIndexing.dynamicStorageBufferArray, SPVDescriptorIndexing.nonUniformUniformBufferArray,
                                         SPVDescriptorIndexing.nonUniformStorageBufferArray,
                                         SPV16BitStorage.storage, SPV16BitStorage.uniform, SPV16BitStorage.pushConstant,
                                         SPV16BitStorage.interface]
            guard implemented.contains(a[0]) || SPVSubgroup.capabilities.contains(a[0]) else {
                throw i.fail("Capability \(a[0]) is not implemented (strict=\(options.strictCapabilities))", code: .unsupportedCapability)
            }
            guard capabilities.insert(a[0]).inserted else { throw i.fail("Duplicate capability") }
        case 10:
            try order(2, i)
            guard glslInstructionSets.isEmpty, nonSemanticDebugSets.isEmpty, debugPrintfSets.isEmpty else {
                throw i.fail("OpExtension must precede OpExtInstImport")
            }
            let (name, end) = try string(a, from: 0, i)
            guard end == a.count else { throw i.fail("Unexpected OpExtension operands") }
            guard [SPVFloatAtomicAdd.extensionName, SPVBarycentric.extensionName, SPVMultiview.extensionName, SPVDemote.extensionName,
                   SPVDrawParameters.extensionName, SPVVulkanMemoryModel.extensionName, SPVNonSemanticDebug.extensionName,
                   SPV16BitStorage.extensionName, SPV16BitStorage.storageClassExtension,
                   SPVDescriptorIndexing.extensionName, SPIRVStorageImageFormat.int64ImageExtension].contains(name) else {
                throw i.fail("SPIR-V extension \(name) is not implemented", code: .unsupportedConstruct)
            }
            guard extensions.insert(name).inserted else { throw i.fail("Duplicate SPIR-V extension") }
        case 11:
            try order(2, i); try i.minimum(2)
            let (name, end) = try string(a, from: 1, i)
            guard end == a.count else { throw i.fail("Unexpected OpExtInstImport operands") }
            switch name {
            case "GLSL.std.450":
                try define(a[0], i)
                glslInstructionSets.insert(a[0])
            case SPVNonSemanticDebug.instructionSet:
                try importNonSemanticDebug(i)
            case SPVNonSemanticDebug.printfInstructionSet:
                try requireNonSemanticInstructions(at: i)
                try define(a[0], i)
                debugPrintfSets.insert(a[0])
            default:
                throw i.fail("Extended instruction set \(name) is not implemented", code: .unsupportedConstruct)
            }
        case 12:
            if section < 8 { try order(8, i) }
            guard memoryModel != nil, try skipNonSemanticDebug(i) else {
                throw i.fail("Only non-semantic extended instructions are supported at module scope", code: .unsupportedConstruct)
            }
        case 14:
            try order(3,i); try i.count(2)
            guard memoryModel == nil else { throw i.fail("Duplicate OpMemoryModel") }
            guard a[0] == 0, let model = SPVMemoryModel(rawValue: a[1]) else {
                throw i.fail("Expected Logical addressing with Simple, GLSL450 or Vulkan memory model", code: .unsupportedConstruct)
            }
            let vulkan = model == .vulkan
            guard capabilities.contains(SPVVulkanMemoryModel.capability) == vulkan else {
                throw i.fail("Vulkan memory model and VulkanMemoryModel capability must be declared together",
                             code: .unsupportedCapability)
            }
            guard vulkan || !capabilities.contains(SPVVulkanMemoryModel.deviceScopeCapability) else {
                throw i.fail("VulkanMemoryModelDeviceScope requires the Vulkan memory model", code: .unsupportedCapability)
            }
            if vulkan, version < SPVVulkanMemoryModel.coreVersion,
               !extensions.contains(SPVVulkanMemoryModel.extensionName) {
                throw i.fail("Vulkan memory model requires SPIR-V 1.5 or SPV_KHR_vulkan_memory_model",
                             code: .unsupportedCapability)
            }
            memoryModel = model
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
            case 7,9,12,14...16: try i.count(2)
            default: throw i.fail("Unsupported execution mode \(a[1])",code:.unsupportedConstruct)
            }
            guard modes[a[0]]?[a[1]] == nil else { throw i.fail("Duplicate execution mode") }
            modes[a[0],default:[:]][a[1]] = Array(a.dropFirst(2))
        case 3:
            try order(6, i); try i.minimum(2)
            if a.count > 2 {
                try reference(a[2], i)
                sourceFiles.append((a[2], i))
            }
            if a.count > 3 {
                let (_, end) = try string(a, from: 3, i)
                guard end == a.count else { throw i.fail("Unexpected OpSource operands") }
            }
        case 4:
            try order(6, i)
            let (_, end) = try string(a, from: 0, i)
            guard end == a.count else { throw i.fail("Unexpected OpSourceExtension operands") }
        case 5:
            try order(6,i); try i.minimum(2); try reference(a[0],i)
            let (name, end) = try string(a, from: 1, i)
            guard end == a.count else { throw i.fail("Unexpected OpName operands") }
            names[a[0]] = name
        case 6:
            try order(6, i); try i.minimum(3); try reference(a[0], i)
            let (name, end) = try string(a, from: 2, i)
            guard end == a.count else { throw i.fail("Unexpected OpMemberName operands") }
            memberNames[a[0], default: [:]][a[1]] = name
        case 7:
            try order(6, i); try i.minimum(2)
            let (text, end) = try string(a, from: 1, i)
            guard end == a.count else { throw i.fail("Unexpected OpString operands") }
            try define(a[0], i)
            debugStrings[a[0]] = text
        case 71,72:
            try order(7,i); try i.minimum(i.op == 71 ? 2 : 3); try reference(a[0],i)
            let index = i.op == 71 ? 1 : 2, dec = a[index], payload = Array(a.dropFirst(index+1))
            let expected: Int
            switch dec {
            case 0 where i.op == 71: expected = 0
            case 1 where i.op == 71: expected = 1 // SpecId
            case SPVDescriptorIndexing.nonUniform where i.op == 71: expected = 0
            case SPVBarycentric.perVertex:
                guard i.op == 71 else {
                    throw i.fail("PerVertexKHR interface block members are not implemented", code: .unsupportedConstruct)
                }
                expected = 0
            case SPVStorageAliasDecoration.restrict where i.op == 71,
                 SPVStorageAliasDecoration.aliased where i.op == 71: expected = 0
            case SPVMatrixMemoryLayout.rowMajor where i.op == 72,
                 SPVMatrixMemoryLayout.columnMajor where i.op == 72: expected = 0
            case SPVMatrixMemoryLayout.stride where i.op == 72: expected = 1
            case SPVLegacyStorage.bufferBlock where i.op == 71:
                guard version <= SPVLegacyStorage.lastVersion else {
                    throw i.fail("BufferBlock requires SPIR-V 1.3 or older")
                }
                expected = 0
            case SPVStorageMemoryDecoration.coherent, SPVStorageMemoryDecoration.volatile:
                guard !usesVulkanMemoryModel else {
                    throw i.fail("Coherent and Volatile decorations are invalid with the Vulkan memory model")
                }
                expected = 0
            case 2, 13, 14, 16, 17, 18,
                 SPVStorageAccessDecorations.nonWritable, SPVStorageAccessDecorations.nonReadable:
                expected = 0
            case 32:
                guard i.op == 71 else { throw i.fail("Index decoration requires a variable") }
                expected = 1
            case 6, 11, 30, 33, 34, 35: expected = 1
            default: throw i.fail("Decoration \(dec) not implemented", code: .unsupportedConstruct)
            }
            guard payload.count == expected else { throw i.fail("Bad decoration operand count") }
            if i.op == 71 {
                guard decorations[a[0]]?[dec] == nil else { throw i.fail("Duplicate decoration") }
                decorations[a[0],default:[:]][dec] = payload
            } else {
                guard members[a[0]]?[a[1]]?[dec] == nil else { throw i.fail("Duplicate member decoration") }
                var v = members[a[0],default:[:]]; v[a[1],default:[:]][dec]=payload; members[a[0]]=v
            }
        case 1,19...33,41...44,46,48...50,59:
            try order(8,i); guard memoryModel != nil else { throw i.fail("Declaration before memory model") }
            try declaration(i)
        default: throw i.fail("Unsupported module opcode \(i.op)",code:.unsupportedConstruct)
        }
    }

    mutating func declaration(_ i: SPVInstruction) throws {
        let a=i.a
        switch i.op {
        case 19: try i.count(1); try define(a[0],i); voids.insert(a[0])
        case 20:
            try i.count(1); try define(a[0],i); try addType(a[0],.scalar(.bool),i)
            controlBoolType = typeMap[a[0]]
        case 21:
            try i.count(3); try define(a[0],i)
            guard [16,32,64].contains(a[1]), a[2] <= 1 else { throw i.fail("Unsupported integer width/sign",code:.unsupportedConstruct) }
            let int64 = capabilities.contains(11) || capabilities.contains(SPVImageTexelPointer.int64AtomicsCapability)
                || capabilities.contains(SPIRVStorageImageFormat.int64ImageCapability)
            guard a[1] != 64 || int64, a[1] != 16 || capabilities.contains(22) || has16BitStorage else {
                throw i.fail("Missing integer capability", code: .unsupportedCapability)
            }
            try addType(a[0],.scalar(.init(kind:a[2] == 0 ? .uint:.sint,width:UInt8(a[1]/8))),i)
        case 22:
            try i.count(2); try define(a[0],i)
            guard [16,32,64].contains(a[1]) else { throw i.fail("Unsupported float width",code:.unsupportedConstruct) }
            guard a[1] != 64 || capabilities.contains(10), a[1] != 16 || capabilities.contains(9) || has16BitStorage else {
                throw i.fail("Missing float capability", code: .unsupportedCapability)
            }
            try addType(a[0],.scalar(.init(kind:.float,width:UInt8(a[1]/8))),i)
        case 23:
            try i.count(3); try define(a[0],i)
            guard let n=UInt8(exactly:a[2]).flatMap(IRVectorSize.init(rawValue:)) else { throw i.fail("Invalid vector size") }
            try addType(a[0],.vector(size:n,scalar:scalar(a[1],i)),i)
        case 24:
            try i.count(3); try define(a[0],i)
            guard (capabilities.contains(0) || capabilities.contains(1)),
                  case .vector(let rows,let s) = try kind(a[1],i), s.kind == .float,
                  let columns=UInt8(exactly:a[2]).flatMap(IRVectorSize.init(rawValue:)) else { throw i.fail("Invalid matrix type/capability") }
            try addType(a[0],.matrix(columns:columns,rows:rows,scalar:s),i)
            matrixColumnTypes[try type(a[0], i)] = try type(a[1], i)
        case 25: try declareImage(i)
        case 26: try declareSampler(i)
        case 27: try declareSampledImage(i)
        case 28,29:
            try i.count(i.op == 28 ? 3:2); try define(a[0],i)
            guard sampledImageTypes[a[1]] == nil else {
                throw i.fail("Combined image-sampler descriptor arrays are not implemented", code: .unsupportedConstruct)
            }
            let base = try type(a[1], i)
            let size: IRArraySize
            if i.op == 28 { guard let n=constantMap[a[2]]?.nonnegativeInteger, n>0 else { throw i.fail("Invalid array length") }; size = .constant(n) } else { size = .dynamic }
            if try declareDescriptorArray(i, base: base, size: size) { return }
            let explicitStride = decorations[a[0]]?[6]?.first
            let stride: UInt32
            if let explicitStride {
                guard explicitStride > 0 else { throw i.fail("Invalid ArrayStride") }
                stride = explicitStride
            } else {
                guard i.op == 28 else { throw i.fail("Runtime array requires ArrayStride", code: .unsupportedConstruct) }
                let layout = try valueLayout(base, i)
                guard let value = UInt32(exactly: layout.aligned(layout.size)), value > 0 else {
                    throw i.fail("Invalid implicit array stride")
                }
                stride = value
            }
            try addType(a[0],.array(base:base,size:size,stride:stride),i)
            if explicitStride == nil { implicitArrays.insert(try type(a[0], i)) }
        case 30:
            try i.minimum(1); try define(a[0],i)
            let declaredSpan = try readStructureSpanName(a[0], i)
            let completeMatrixLayout = try validateMatrixMemberLayouts(i)
            if members[a[0]]?.values.contains(where: { $0[35] != nil }) != true {
                guard declaredSpan == nil else { throw i.fail("SAGA structure span requires explicit member offsets") }
                try addImplicitStruct(i)
                return
            }
            var fields: [IRStructMember] = []
            var span: UInt32 = 0
            var alignment: UInt64 = 1
            for (index, id) in a.dropFirst().enumerated() {
                try charge()
                let type = try self.type(id, i)
                guard let offset = members[a[0]]?[UInt32(index)]?[35]?.first else {
                    throw i.fail("Explicit member Offset required", code: .unsupportedConstruct)
                }
                if case .array(let element, .dynamic, let stride) = types[type.index].kind {
                    guard index == a.count - 2 else { throw i.fail("Runtime array must be the final structure member") }
                    try requireFixedSize(element, i)
                    let elementLayout = try valueLayout(element, i)
                    guard UInt64(stride) >= elementLayout.size, UInt64(stride) % elementLayout.alignment == 0 else {
                        throw i.fail("Invalid runtime array stride")
                    }
                    guard offset >= span, UInt64(offset) % elementLayout.alignment == 0 else {
                        throw i.fail("Misaligned or overlapping runtime array offset")
                    }
                    alignment = max(alignment, elementLayout.alignment)
                    // IR records the minimum binding footprint, including space for one tail element.
                    let end = UInt64(offset) + UInt64(stride)
                    let minimum = SPVValueLayout(alignment: alignment, size: 0).aligned(end)
                    guard minimum <= UInt64(Int32.max) else { throw i.fail("Runtime array minimum span exceeds the size limit") }
                    span = UInt32(minimum)
                } else {
                    let size = try byteSize(type, i)
                    let layout = try valueLayout(type, i)
                    alignment = max(alignment, layout.alignment)
                    let (end, overflow) = offset.addingReportingOverflow(size)
                    guard !overflow, offset >= span else { throw i.fail("Invalid/overlapping struct offsets") }
                    span = end
                }
                fields.append(.init(name: memberNames[a[0]]?[UInt32(index)], type: type, offset: offset, location: i.location))
            }
            let paddedSpan = SPVValueLayout(alignment: alignment, size: 0).aligned(UInt64(span))
            guard paddedSpan <= UInt64(Int32.max) else { throw i.fail("Structure span exceeds the size limit") }
            if let declaredSpan {
                guard UInt64(declaredSpan) >= paddedSpan, declaredSpan <= UInt32(Int32.max),
                      UInt64(declaredSpan) % alignment == 0 else {
                    throw i.fail("SAGA structure span conflicts with explicit member layout")
                }
            }
            try addType(a[0], .structure(members: fields, span: declaredSpan ?? UInt32(paddedSpan)), i)
            if !completeMatrixLayout { incompleteMatrixLayouts.insert(try type(a[0], i)) }
        case 32:
            try i.count(3); try define(a[0],i)
            guard sampledImageTypes[a[2]] == nil else {
                throw i.fail("Combined image-sampler descriptors are not implemented", code: .unsupportedConstruct)
            }
            let base=try type(a[2],i)
            let pointer = SPVPointer(storage: a[1], base: a[2])
            if pointer.storage == SPVImageTexelPointer.storageClass {
                guard [.scalar(.u32), .scalar(.i32), .scalar(.u64)].contains(types[base.index].kind) else {
                    throw i.fail("Image pointer requires an i32, u32, or u64 texel", code: .unsupportedConstruct)
                }
                pointers[a[0]] = pointer
                return
            }
            try validateDescriptorArrayPointer(pointer, at: i)
            let space = try effectivePointerSpace(pointer, i)
            pointers[a[0]] = pointer
            if space == .handle {
                guard types[base.index].kind.isTextureOrSampler || descriptorArrayElements[a[2]] != nil else {
                    throw i.fail("UniformConstant pointers require an image, sampler or descriptor array")
                }
                // SPIR-V loads opaque resources through pointers. IR uses the image handle directly.
                typeMap[a[0]] = base
            } else {
                guard !types[base.index].kind.isTextureOrSampler else {
                    throw i.fail("Images and samplers require UniformConstant pointers")
                }
                try addType(a[0], .pointer(base: base, space: space), i)
            }
        case 33:
            try i.minimum(2); try define(a[0],i)
            guard voids.contains(a[1]) || typeMap[a[1]] != nil else { throw i.fail("Undefined function return type") }
            for id in a.dropFirst(2) { _=try type(id,i) }
            functionTypes[a[0]]=SPVFunctionType(result:a[1],parameters:Array(a.dropFirst(2)))
        case 1,41,42,43,44,46,48...50: try constant(i)
        case 59:
            guard a.count == 3 || a.count == 4 else { throw i.fail("Invalid global variable operands") }; try define(a[1],i)
            guard let p=pointers[a[0]], p.storage == a[2], p.storage != 7 else { throw i.fail("Invalid global pointer/storage class") }
            let space = try effectivePointerSpace(p, i)
            var t = try type(p.base, i)
            var initExpr:IRHandle<IRExpression>?
            if a.count == 4 {
                guard a[2] == 6 else { throw i.fail("Bad global initializer") }
                if let c = constantMap[a[3]], c.type == p.base {
                    initExpr = constants[c.handle.index].initializer
                } else if let value = overrideMap[a[3]], value.type == p.base {
                    try node()
                    initExpr = try IRHandle(index: globalExpressions.count)
                    globalExpressions.append(.init(type: t, kind: .override(value.handle), location: i.location))
                } else {
                    throw i.fail("Bad global initializer")
                }
            }
            let set=decorations[a[1]]?[34]?.first, binding=decorations[a[1]]?[33]?.first
            guard (set == nil) == (binding == nil) else { throw i.fail("Incomplete descriptor binding") }
            if a[2] == 2 || a[2] == 12 {
                let block = isLegacyStoragePointer(p) ? SPVLegacyStorage.bufferBlock : SPVLegacyStorage.block
                let element = descriptorArrayElements[p.base] ?? p.base
                guard set != nil, decorations[element]?[block] != nil else {
                    throw i.fail("Resource needs a matching block decoration and descriptor binding")
                }
                try requireExplicitBufferLayout(t, i)
                if space == .uniform, !uniformBufferStandardLayout {
                    uniformBufferStandardLayout = try requiresStandardUniformLayout(t)
                }
            } else if space == .immediate {
                guard set == nil, decorations[p.base]?[SPVLegacyStorage.block] != nil,
                      case .structure = types[t.index].kind else {
                    throw i.fail("PushConstant requires a block structure without a descriptor binding")
                }
                try requireExplicitBufferLayout(t, i)
                try requireFixedSize(t, i)
            } else if space == .handle {
                guard set != nil else { throw i.fail("Image or sampler requires a descriptor binding") }
                if types[t.index].kind.storageTexture != nil {
                    t = try qualifyStorageImage(t, variable: a[1], at: i)
                } else if case .bindingArray(let element, let size) = types[t.index].kind,
                          types[element.index].kind.storageTexture != nil {
                    let qualified = try qualifyStorageImage(element, variable: a[1], at: i)
                    if qualified != element { t = try internType(.bindingArray(base: qualified, size: size)) }
                } else if !SPVStorageAccessDecorations.prohibited(decorations[a[1]] ?? [:]).isEmpty {
                    throw i.fail("Storage access decorations require a storage image or buffer")
                }
            } else if set != nil { throw i.fail("Descriptor binding on non-resource") }
            let rb:IRResourceBinding? = if let set,let binding { .init(group:set,binding:binding) } else { nil }
            let h=try IRHandle<IRGlobalVariable>(index:globals.count)
            if a[2] == 1 || a[2] == 3 {
                try registerStageInterface(a[1], global: h, pointerType: a[0], valueType: p.base, input: a[2] == 1, at: i)
            }
            globals.append(.init(name:names[a[1]],space:space,binding:rb,type:t,initializer:initExpr,location:i.location)); globalMap[a[1]]=(h,a[0])
        default: throw i.fail("Unsupported declaration opcode \(i.op)",code:.unsupportedConstruct)
        }
    }
    func storage(_ id:UInt32,_ i:SPVInstruction) throws -> IRAddressSpace {
        switch id {
        case SPVStorageImage.uniformConstant: return .handle
        case 4: return .workgroup
        case 1,3,6: return .private
        case 7: return .function
        case 9: return .immediate
        case 2: return .uniform
        case 12: return .storage(access:[.load,.store])
        default: throw i.fail("Unsupported storage class \(id)",code:.unsupportedConstruct)
        }
    }
    private mutating func readStructureSpanName(_ id: UInt32, _ instruction: SPVInstruction) throws -> UInt32? {
        guard let name = names[id], name.hasPrefix(SPIRVStructureSpanName.prefix) else { return nil }
        try charge(name.utf8.count)
        guard let decoded = SPIRVStructureSpanName.decode(name) else {
            throw instruction.fail("Malformed SAGA structure span")
        }
        names[id] = decoded.name
        return decoded.span
    }

    mutating func byteSize(_ t:IRHandle<IRType>,_ i:SPVInstruction) throws -> UInt32 {
        try requireFixedSize(t, i)
        switch types[t.index].kind {
        case .scalar(let s):return UInt32(s.width)
        case .vector(let n,let s):return UInt32(n.rawValue)*UInt32(s.width)
        case .matrix: return UInt32(try valueLayout(t, i).size)
        case .structure(_,let span):return span
        case .array(_, .constant(let n),let stride):let (r,o)=n.multipliedReportingOverflow(by:stride);guard !o else { throw i.fail("Array size overflow") };return r
        default:throw i.fail("Layout calculation for this member type is unavailable",code:.unsupportedConstruct)
        }
    }
    mutating func constant(_ i:SPVInstruction) throws {
        let a=i.a; try i.minimum(2); try define(a[1],i); let t=try type(a[0],i)
        let k=types[t.index].kind; var u:UInt32?; let expr:IRExpressionKind
        if [41, 42, 48, 49].contains(i.op) {
            try i.count(2); guard k == .scalar(.bool) else { throw i.fail("Boolean constant type mismatch") }; expr = .literal(.bool(i.op == 41 || i.op == 48))
        } else if i.op == 46 || i.op == 1 {
            try i.count(2); expr = .zeroValue(t)
            if i.op == 46, k == .scalar(.u32) || k == .scalar(.i32) { u=0 }
        } else if i.op == 44 {
            var components:[IRHandle<IRExpression>]=[]
            let expected:[IRHandle<IRType>]
            switch k {
            case .vector(let n,let s):
                try charge(types.count)
                guard let scalarID=types.indices.first(where:{types[$0].kind == .scalar(s)}) else { throw i.fail("Missing scalar type") }
                expected=Array(repeating:try IRHandle(index:scalarID),count:Int(n.rawValue))
            case .matrix(let columns, _, _):
                guard let column = matrixColumnTypes[t] else { throw i.fail("Missing matrix column type") }
                expected = Array(repeating: column, count: Int(columns.rawValue))
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
            case (.sint,4):
                let value = Int32(bitPattern: a[2])
                expr = .literal(.i32(value))
                u = UInt32(exactly: value)
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
        if let rawID = decorations[a[1]]?[1]?.first {
            guard (48...50).contains(i.op) else { throw i.fail("SpecId requires a scalar specialization constant") }
            guard let id = UInt16(exactly: rawID) else {
                throw i.fail("SpecId exceeds the override ID range", code: .unsupportedConstruct)
            }
            try node()
            let handle = try IRHandle<IROverride>(index: overrides.count)
            overrides.append(.init(name: names[a[1]], id: id, type: t, initializer: h, location: i.location))
            overrideMap[a[1]] = (handle, a[0])
            return
        }
        let c=try IRHandle<IRConstant>(index:constants.count);constants.append(.init(name:names[a[1]],type:t,initializer:h,location:i.location));constantMap[a[1]]=SPVConstant(handle:c,type:a[0],nonnegativeInteger:u)
        if decorations[a[1]]?[11]?.first == 25 {
            guard i.op == 44, a.count == 5, k == .vector(size: .three, scalar: .u32) || k == .vector(size: .three, scalar: .i32),
                  let x = constantMap[a[2]]?.nonnegativeInteger, x > 0,
                  let y = constantMap[a[3]]?.nonnegativeInteger, y > 0,
                  let z = constantMap[a[4]]?.nonnegativeInteger, z > 0 else {
                throw i.fail("WorkgroupSize requires a constant integer vector with positive dimensions")
            }
            guard workgroupSizeBuiltin == nil else {
                throw i.fail("Multiple WorkgroupSize constants are not implemented", code: .unsupportedConstruct)
            }
            workgroupSizeBuiltin = .init(x: x, y: y, z: z)
        }
    }
    private func supportsRelaxedPrecision(_ type: IRHandle<IRType>) -> Bool {
        var kind = types[type.index].kind
        if case .array(let element, _, _) = kind { kind = types[element.index].kind }
        switch kind {
        case .scalar(let scalar), .vector(_, let scalar), .matrix(_, _, let scalar):
            return [.i32, .u32, .f32].contains(scalar)
        default:
            return false
        }
    }

    mutating func validateDecorations(_ header:SPVInstruction) throws {
        var hasBufferBlock = false
        for (id,ds) in decorations {
            for (dec,_) in ds {
                try charge()
                switch dec {
                case 0:
                    // RelaxedPrecision permits the original 32-bit precision.
                    let rawType = functionValueTypes[id] ?? globalMap[id]?.1
                        ?? functionMap[id].map { functions[$0.index].header.a[0] } ?? constantMap[id]?.type ?? overrideMap[id]?.type
                    guard let rawType, let type = typeMap[pointers[rawType]?.base ?? rawType],
                          supportsRelaxedPrecision(type) else {
                        throw header.fail("RelaxedPrecision requires a 32-bit numeric value")
                    }
                case 2:
                    guard let t=typeMap[id], case .structure = types[t.index].kind else {throw header.fail("Block decoration must target struct")}
                case SPVLegacyStorage.bufferBlock:
                    guard let t = typeMap[id], case .structure = types[t.index].kind,
                          ds[SPVLegacyStorage.block] == nil else {
                        throw header.fail("BufferBlock must target a struct without Block")
                    }
                    hasBufferBlock = true
                case 6:
                    try validateArrayStrideDecoration(id, at: header)
                case 33,34:
                    guard globalMap[id] != nil else {throw header.fail("Descriptor decoration target must be global")}
                case 32:
                    guard let field = stageInterfaces[id]?.first, field.member == nil, !field.input,
                          ds[30] != nil, ds[11] == nil else {
                        throw header.fail("Index decoration requires a color output variable with Location")
                    }
                    guard let index = ds[32]?.first, index <= 1 else {
                        throw header.fail("Color output Index must be 0 or 1")
                    }
                case SPVBarycentric.perVertex:
                    guard let field = stageInterfaces[id]?.first, field.member == nil, field.input,
                          ds[30] != nil, ds[11] == nil else {
                        throw header.fail("PerVertexKHR requires an input variable with Location")
                    }
                case SPVDescriptorIndexing.nonUniform:
                    try validateNonUniformDecoration(id, at: header)
                case SPVStorageAliasDecoration.restrict, SPVStorageAliasDecoration.aliased:
                    try validateStorageAliasDecoration(id, ds, at: header)
                case SPVStorageAccessDecorations.nonWritable, SPVStorageAccessDecorations.nonReadable:
                    guard globalMap[id] != nil else { throw header.fail("Storage access decoration must target a global variable") }
                case SPVStorageMemoryDecoration.coherent, SPVStorageMemoryDecoration.volatile:
                    guard let (global, _) = globalMap[id], case .storage = globals[global.index].space else {
                        throw header.fail("Coherent and Volatile require a storage buffer", code: .unsupportedConstruct)
                    }
                case 1:
                    guard overrideMap[id] != nil else { throw header.fail("SpecId requires a scalar specialization constant") }
                case 11 where ds[11]?.first == 25 && constantMap[id] != nil:
                    break
                case 11,13,14,16,17,18,30:
                    guard stageInterfaces[id] != nil else { throw header.fail("Interface decoration requires an interface variable") }
                default:throw header.fail("Unconsumed decoration",code:.unsupportedConstruct)
                }
            }
        }
        for (id,ms) in members {
            guard let t=typeMap[id], case .structure(let fields,_) = types[t.index].kind else {throw header.fail("Member decoration target must be struct")}
            for (index,ds) in ms {
                try charge(ds.count)
                let allowed: Set<UInt32> = interfaceTypes.contains(id) ? [11,13,14,16,17,18,30,35] :
                    [35, SPVStorageAccessDecorations.nonWritable, SPVStorageAccessDecorations.nonReadable,
                     SPVStorageMemoryDecoration.coherent, SPVStorageMemoryDecoration.volatile,
                     SPVMatrixMemoryLayout.rowMajor, SPVMatrixMemoryLayout.columnMajor, SPVMatrixMemoryLayout.stride]
                guard UInt64(index)<UInt64(fields.count), Set(ds.keys).isSubset(of: allowed) else {
                    throw header.fail("Invalid member decoration")
                }
            }
        }
        for (id, names) in memberNames {
            try charge(names.count)
            guard let type = typeMap[id], case .structure(let fields, _) = types[type.index].kind,
                  names.keys.allSatisfy({ UInt64($0) < UInt64(fields.count) }) else {
                throw header.fail("OpMemberName targets an invalid structure member")
            }
        }
        if hasBufferBlock { try validateLegacyStorageClasses(header) }
    }
    mutating func registerFunctionResult(_ i:SPVInstruction) throws {
        try validateStorageOnly16BitResult(i)
        switch i.op {
        case _ where SPVSubgroup.capability(for: i.op) != nil:
            try i.minimum(2); try define(i.a[1], i)
            functionValueTypes[i.a[1]] = i.a[0]
        case 227,229,230,232...242,SPVFloatAtomicAdd.opcode:
            try i.minimum(2); try define(i.a[1], i)
            functionValueTypes[i.a[1]] = i.a[0]
            if i.op == 230 { try prepareAtomicCompareType(at: i) }
        case 12:
            try i.minimum(4); try define(i.a[1], i)
            if debugPrintfSets.contains(i.a[2]) {
                try i.minimum(5)
                guard voids.contains(i.a[0]), i.a[3] == 1, debugStrings[i.a[4]] != nil else {
                    throw i.fail("Invalid NonSemantic.DebugPrintf instruction")
                }
                nonSemanticDebugResults.insert(i.a[1])
            } else {
                functionValueTypes[i.a[1]] = i.a[0]
                hasGLSLMath = true
            }
        case 89, 90, 93, 94, 97:
            try i.minimum(5); try define(i.a[1], i)
            functionValueTypes[i.a[1]] = i.a[0]
            hasDrefSampling = true
        case 60:
            try i.count(5); try define(i.a[1], i)
            functionValueTypes[i.a[1]] = i.a[0]
            hasImageTexelPointers = true
        case 1,55,57,59,61,65,66,68,77...84,86...88,91,92,95,96,98,100,103,104,106,107,109...116,124,126...148,154...157,164...191,194...205,207...215,245:
            try i.minimum(2);try define(i.a[1],i)
            functionValueTypes[i.a[1]] = i.a[0]
        case 248:try i.count(1);try define(i.a[0],i)
        case 246: hasLoops = true
        case 0,62,99,224,225,228,247,249,250,251,252,253,254,255,SPVDemote.opcode:break
        default:throw i.fail("Unsupported function opcode \(i.op)",code:.unsupportedConstruct)
        }
    }
}
