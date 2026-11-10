import SAGAShaderIR
import SAGAValidation

/// A binary writer for validated shader IR. Layout-sized descriptor accesses are bounded before lowering.
public struct SPIRVCodeGenerator: ShaderCodeGenerator {
    public let limits: ShaderLimits
    public var outputFormat: ShaderOutputFormat { .spirv }
    private let bindingArrayLengths: [IRResourceBinding: UInt32]

    public init(limits: ShaderLimits = .default) {
        self.init(bindingArrayLengths: [:], limits: limits)
    }

    /// When set, only this entry point is written, so a module with several stages yields a single-stage binary.
    private let selectedEntry: String?

    package init(bindingArrayLengths: [IRResourceBinding: UInt32], entryPoint: String? = nil, limits: ShaderLimits = .default) {
        self.limits = limits
        self.bindingArrayLengths = bindingArrayLengths
        self.selectedEntry = entryPoint
    }

    public func generate(_ module: ValidatedModule<IRModule>) throws -> ShaderArtifact {
        let ir = module.module
        let hasDescriptorArrays = try ir.globalVariables.values.contains { global in
            if case .bindingArray = try ir.types.value(for: global.type).kind { return true }
            return false
        }
        let nonUniform = hasDescriptorArrays
            ? try ShaderSemanticValidator(capabilities: .allKnown, limits: limits, allowsVertexWritableStorage: true)
                .nonUniformDescriptorAccesses(in: ir)
            : [:]
        var writer = SPIRVBinaryWriter(module: ir, budget: ShaderBudget(limits: limits),
                                       bindingArrayLengths: bindingArrayLengths, nonUniformDescriptors: nonUniform,
                                       selectedEntry: selectedEntry)
        return try .spirv(writer.writeModule())
    }
}

private struct SPIRVFunctionSignature: Hashable {
    let result: UInt32
    let parameters: [UInt32]
}

enum SPIRVEntryPlan {
    case compute(SPIRVComputeInputPlan)
    case vertex(SPIRVRenderInterfacePlan)
    case fragment(SPIRVRenderInterfacePlan)
    case mesh(SPIRVMeshPlan)

    var interfaceIDs: [UInt32] {
        switch self {
        case .compute(let plan): plan.interfaceIDs
        case .vertex(let plan), .fragment(let plan): plan.interfaceIDs
        case .mesh(let plan): plan.interfaceIDs
        }
    }

    var executionModel: UInt32 {
        switch self { case .vertex: 0; case .fragment: 4; case .compute: 5; case .mesh: 5365 }
    }

    var renderInterface: SPIRVRenderInterfacePlan? {
        switch self {
        case .vertex(let plan), .fragment(let plan): plan
        case .compute, .mesh: nil
        }
    }
}

struct SPIRVBinaryWriter {
    let module: IRModule
    let bindingArrayLengths: [IRResourceBinding: UInt32]
    private let selectedEntry: String?
    private let nonUniformDescriptors: [Int: Set<IRHandle<IRExpression>>]
    var budget: ShaderBudget
    var declarationState = SPIRVDeclarationState()

    private var nextID: UInt32 = 1
    var capabilities: Set<UInt32> = [1] // Shader
    var extensions: Set<String> = []
    var usesMeshShading = false
    /// Module-scope variables, listed in every entry point interface when the module targets SPIR-V 1.4.
    private var globalVariableIDs: [UInt32] = []
    private var pendingEntries: [(model: UInt32, id: UInt32, name: [UInt32], interface: [UInt32])] = []
    private var glslInstructions: UInt32?
    private(set) var constantIDs: Set<UInt32> = []
    private(set) var constantWords: [UInt32: UInt32] = [:]
    private(set) var nonUniformResourceIDs: Set<UInt32> = []
    private var entryPoints: [UInt32] = []
    var executionModes: [UInt32] = []
    private var debug: [UInt32] = []
    private var debugStrings: [UInt32] = []
    private var debugFormatIDs: [String: UInt32] = [:]
    private var debugPrintfInstructions: UInt32?
    private var usesCooperativeMatrix = false
    private var usesStorageDeviceScopeAtomics = false
    private var usesCoherentOrVolatileStorage = false
    private var annotations: [UInt32] = []
    private var declarations: [UInt32] = []
    private var functions: [UInt32] = []
    private(set) var currentBlock: UInt32?
    private var signatures: [SPIRVFunctionSignature: UInt32] = [:]
    private var voidID: UInt32?
    private(set) var functionIDs: [UInt32] = []

    init(module: IRModule, budget: ShaderBudget, bindingArrayLengths: [IRResourceBinding: UInt32],
         nonUniformDescriptors: [Int: Set<IRHandle<IRExpression>>], selectedEntry: String? = nil) {
        self.selectedEntry = selectedEntry
        self.module = module
        self.budget = budget
        self.bindingArrayLengths = bindingArrayLengths
        self.nonUniformDescriptors = nonUniformDescriptors
    }

    mutating func writeModule() throws -> [UInt32] {
        guard !module.entryPoints.isEmpty else { throw unsupported("requires a shader entry point") }
        guard module.overrides.count == 0 else { throw unsupported("requires specialized overrides") }
        for _ in module.functions.values { functionIDs.append(try allocateID()) }
        let hasWorkgroupMemory = module.globalVariables.values.contains { $0.space == .workgroup }
        var entryPlans: [Int: (entry: IREntryPoint, plan: SPIRVEntryPlan)] = [:]
        for entry in module.entryPoints where selectedEntry == nil || entry.name == selectedEntry {
            let index = entry.function.index
            let function = try module.functions.value(for: entry.function)
            let id = functionIDs[index]
            let plan: SPIRVEntryPlan
            if let existing = entryPlans[index] {
                guard existing.entry.stage == entry.stage, existing.entry.workgroupSize == entry.workgroupSize,
                      existing.entry.workgroupOverrides == entry.workgroupOverrides,
                      existing.entry.subgroupSize == entry.subgroupSize, existing.entry.earlyDepthTest == entry.earlyDepthTest else {
                    throw unsupported("requires matching execution modes for entry names sharing a function")
                }
                plan = existing.plan
            } else {
                plan = try prepareEntry(entry, function: function, id: id, hasWorkgroupMemory: hasWorkgroupMemory)
                entryPlans[index] = (entry, plan)
            }
            let name = try stringWords(entry.name)
            pendingEntries.append((plan.executionModel, id, name, plan.interfaceIDs))
        }
        for (index, global) in module.globalVariables.values.enumerated() {
            let handle = try IRHandle<IRGlobalVariable>(index: index)
            // Declare private variables on demand to omit unused stage-interface placeholders.
            if global.space == .private { continue }
            if global.space == .handle {
                if case .bindingArray(_, .dynamic) = try module.types.value(for: global.type).kind,
                   let binding = global.binding, bindingArrayLengths[binding] == nil {
                    // Demand a layout only if an emitted function references this declaration.
                    continue
                }
                _ = try resourceGlobal(handle)
            } else if case .bindingArray(_, let size) = try module.types.value(for: global.type).kind {
                if size == .dynamic, !bindingArrayLengths.isEmpty,
                   let binding = global.binding, bindingArrayLengths[binding] == nil {
                    continue // Report a missing layout only for referenced arrays.
                }
                _ = try bufferArrayGlobal(handle)
            } else if global.space == .workgroup {
                _ = try workgroupGlobal(handle)
            } else {
                _ = try bufferGlobal(handle)
            }
        }
        for (index, function) in module.functions.values.enumerated() {
            try writeFunction(function, id: functionIDs[index], entryPlan: entryPlans[index]?.plan,
                              nonUniformDescriptors: nonUniformDescriptors[index] ?? [])
        }
        for entry in pendingEntries {
            // SPIR-V 1.4 requires every referenced global variable in the interface, so list all of them.
            let extra = usesMeshShading ? globalVariableIDs.filter { !entry.interface.contains($0) } : []
            entryPoints += try instruction(15, [entry.model, entry.id] + entry.name + entry.interface + extra) // OpEntryPoint
        }
        if usesCooperativeMatrix && usesStorageDeviceScopeAtomics {
            try requireCapability(5346) // VulkanMemoryModelDeviceScope
        }
        if usesCooperativeMatrix && usesCoherentOrVolatileStorage {
            throw unsupported("does not support coherent or volatile storage decorations with the Vulkan memory model")
        }
        try budget.consume(.words, amount: 5)
        var words: [UInt32] = [0x07230203, usesMeshShading ? 0x00010400 : 0x00010300, 0, nextID, 0]
        for capability in capabilities.sorted() { words += try instruction(17, [capability]) }
        for name in extensions.sorted() { words += try instruction(10, stringWords(name)) }
        if let glslInstructions {
            words += try instruction(11, [glslInstructions] + stringWords("GLSL.std.450")) // OpExtInstImport
        }
        if let debugPrintfInstructions {
            words += try instruction(11, [debugPrintfInstructions] + stringWords("NonSemantic.DebugPrintf"))
        }
        words += try instruction(14, usesCooperativeMatrix ? [0, 3] : [0, 1]) // Logical VulkanKHR / GLSL450
        for section in [entryPoints, executionModes, debugStrings, debug, annotations, declarations, functions] { words += section }
        return words
    }

    private mutating func prepareEntry(
        _ entry: IREntryPoint, function: IRFunction, id: UInt32, hasWorkgroupMemory: Bool
    ) throws -> SPIRVEntryPlan {
        switch entry.stage {
        case .compute:
            guard function.result == nil, entry.workgroupOverrides == nil, entry.subgroupSize == nil else {
                throw unsupported("requires void compute entry points with specialized workgroup sizes and no subgroup override")
            }
            let size = entry.workgroupSize
            executionModes += try instruction(16, [id, 17, size.x, size.y, size.z]) // LocalSize
            var workgroup: [IRHandle<IRGlobalVariable>] = []
            if hasWorkgroupMemory {
                let globals = try ShaderFunctions(module: module, entry: entry.function).reachableGlobals
                try charge(globals.count)
                for index in globals.sorted() where module.globalVariables.values[index].space == .workgroup {
                    workgroup.append(try IRHandle(index: index))
                }
            }
            return .compute(try prepareComputeInputs(function, workgroup: workgroup))
        case .vertex:
            return .vertex(try prepareRenderInterface(function, stage: .vertex))
        case .fragment:
            executionModes += try instruction(16, [id, 7]) // OriginUpperLeft
            if let early = entry.earlyDepthTest {
                let mode: UInt32 = switch early {
                case .force: 9 // EarlyFragmentTests
                case .allow(.greaterEqual): 14 // DepthGreater
                case .allow(.lessEqual): 15 // DepthLess
                case .allow(.unchanged): 16 // DepthUnchanged
                }
                executionModes += try instruction(16, [id, mode])
            }
            let plan = try prepareRenderInterface(function, stage: .fragment)
            if plan.writesDepth { executionModes += try instruction(16, [id, 12]) } // DepthReplacing
            return .fragment(plan)
        case .mesh:
            return .mesh(try prepareMeshEntry(entry, function: function, id: id))
        default: throw unsupported("does not lower the \(entry.stage.rawValue) stage")
        }
    }

    mutating func charge(_ amount: Int = 1) throws { try budget.consume(.work, amount: amount) }

    mutating func requireCapability(_ value: UInt32) throws {
        try charge()
        capabilities.insert(value)
    }

    mutating func requireCooperativeMatrix() throws {
        usesCooperativeMatrix = true
        try requireCapability(6022) // CooperativeMatrixKHR
        try requireCapability(5345) // VulkanMemoryModel
        extensions.insert("SPV_KHR_cooperative_matrix")
        extensions.insert("SPV_KHR_vulkan_memory_model")
    }

    mutating func markStorageDeviceScopeAtomic() {
        usesStorageDeviceScopeAtomics = true
    }

    mutating func markCoherentOrVolatileStorage() {
        usesCoherentOrVolatileStorage = true
    }

    mutating func glslInstructionSetID() throws -> UInt32 {
        if let glslInstructions { return glslInstructions }
        let id = try allocateID()
        glslInstructions = id
        return id
    }

    mutating func writeDebugPrintf(format: String, arguments: [UInt32]) throws {
        let importID: UInt32
        if let existing = debugPrintfInstructions {
            importID = existing
        } else {
            importID = try allocateID()
            debugPrintfInstructions = importID
            extensions.insert("SPV_KHR_non_semantic_info")
        }
        let formatID: UInt32
        if let existing = debugFormatIDs[format] {
            formatID = existing
        } else {
            formatID = try allocateID()
            let words = try stringWords(format)
            debugStrings += try instruction(7, [formatID] + words) // OpString
            debugFormatIDs[format] = formatID
        }
        let type = try voidTypeID()
        let result = try allocateID()
        try emit(12, [type, result, importID, 1, formatID] + arguments) // DebugPrintf
    }

    mutating func requireNonUniformDescriptors(_ capability: UInt32) throws {
        try requireCapability(5301) // ShaderNonUniform
        try requireCapability(capability)
        extensions.insert("SPV_EXT_descriptor_indexing")
    }

    mutating func requireRuntimeDescriptors() throws {
        try requireCapability(5302) // RuntimeDescriptorArray
        extensions.insert("SPV_EXT_descriptor_indexing")
    }

    mutating func requireFloatAtomicAdd() throws {
        try requireCapability(6033) // AtomicFloat32AddEXT
        extensions.insert("SPV_EXT_shader_atomic_float_add")
    }

    mutating func requireInt64StorageImage() throws {
        try requireCapability(SPIRVStorageImageFormat.int64ImageCapability)
        extensions.insert(SPIRVStorageImageFormat.int64ImageExtension)
    }

    mutating func requireHelperDemotion() throws {
        try requireCapability(5379) // DemoteToHelperInvocation
        extensions.insert("SPV_EXT_demote_to_helper_invocation")
    }

    mutating func requireFragmentBarycentrics() throws {
        try requireCapability(5284) // FragmentBarycentricKHR
        extensions.insert("SPV_KHR_fragment_shader_barycentric")
    }

    mutating func markNonUniformResource(_ id: UInt32) throws {
        if nonUniformResourceIDs.insert(id).inserted { try decorate(71, [id, 5300]) } // NonUniform
    }

    mutating func allocateID() throws -> UInt32 {
        try budget.consume(.nodes)
        guard nextID < UInt32.max else { throw unsupported("ID space exhausted") }
        let id = nextID
        nextID += 1
        return id
    }

    mutating func declare(_ opcode: UInt32, _ operands: [UInt32]) throws {
        declarations += try instruction(opcode, operands)
        switch opcode {
        case 41...44, 46: constantIDs.insert(operands[1]) // Module constants
        default: break
        }
        if opcode == 59, operands.count >= 3 { globalVariableIDs.append(operands[1]) }
        if opcode == 43, operands.count == 3 { constantWords[operands[1]] = operands[2] }
        if opcode == 46 { constantWords[operands[1]] = 0 }
    }

    mutating func decorate(_ opcode: UInt32, _ operands: [UInt32]) throws {
        annotations += try instruction(opcode, operands)
    }

    mutating func emit(_ opcode: UInt32, _ operands: [UInt32] = []) throws {
        functions += try instruction(opcode, operands)
        if opcode == 248 { currentBlock = operands.first } // OpLabel
    }

    mutating func debugName(_ id: UInt32, _ name: String?) throws {
        if let name {
            let words = try stringWords(name)
            debug += try instruction(5, [id] + words) // OpName
        }
    }

    mutating func voidTypeID() throws -> UInt32 {
        if let voidID { return voidID }
        let id = try allocateID()
        try declare(19, [id]) // OpTypeVoid
        voidID = id
        return id
    }

    mutating func functionTypeID(result: UInt32, parameters: [UInt32]) throws -> UInt32 {
        let signature = SPIRVFunctionSignature(result: result, parameters: parameters)
        if let existing = signatures[signature] { return existing }
        let id = try allocateID()
        try declare(33, [id, result] + parameters) // OpTypeFunction
        signatures[signature] = id
        return id
    }

    func unsupported(_ reason: String) -> ShaderFailure { .unsupported("SPIR-V writer \(reason)") }

    mutating func instructionWords(_ opcode: UInt32, _ operands: [UInt32]) throws -> [UInt32] {
        try instruction(opcode, operands)
    }

    private mutating func instruction(_ opcode: UInt32, _ operands: [UInt32]) throws -> [UInt32] {
        try charge()
        guard opcode <= UInt16.max, operands.count < UInt16.max else { throw unsupported("instruction word count is too large") }
        let count = operands.count + 1
        try budget.consume(.words, amount: count)
        return [UInt32(count) << 16 | opcode] + operands
    }

    private mutating func stringWords(_ text: String) throws -> [UInt32] {
        try charge(text.utf8.count)
        var words: [UInt32] = [0]
        var byte = 0
        for value in text.utf8 {
            guard value != 0 else { throw unsupported("strings cannot contain NUL") }
            words[words.count - 1] |= UInt32(value) << (byte * 8)
            byte += 1
            if byte == 4 {
                words.append(0)
                byte = 0
            }
        }
        return words
    }
}
