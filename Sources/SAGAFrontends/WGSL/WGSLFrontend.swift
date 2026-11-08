import SAGAShaderIR

/// Independent bounded WGSL frontend; raw IR still requires SAGAValidation.
/// Unsupported families are rejected, never accepted as opaque source or dummy IR.
public struct WGSLFrontend: BoundedShaderFrontend {
    public typealias Module = IRModule
    public let inputFormat: ShaderInputFormat = .wgsl
    public let limits: ShaderLimits
    public let options: WGSLFrontendOptions
    public init(options: WGSLFrontendOptions = WGSLFrontendOptions(), limits: ShaderLimits = .default) {
        self.options = options; self.limits = limits
    }
    public func parse(_ input: ShaderInput, budget: inout ShaderBudget) throws -> IRModule {
        guard case .wgsl(let source) = input else { throw ShaderFailure.unsupported("WGSL input required") }
        let (tokens, scanned) = try WGSLScanner(source: source, budget: budget, docComments: options.parseDocComments).scan()
        let parser = WGSLParser(tokens: tokens, budget: scanned, options: options)
        defer { budget = parser.budget }
        return try parser.module()
    }
}

typealias WType = IRHandle<IRType>
typealias WExpr = IRHandle<IRExpression>
struct WGSLValue {
    let handle: WExpr
    let type: WType // value type, not pointer type
    let pointer: Bool
    let writable: Bool
    let swizzle: WGSLSwizzleView?

    init(handle: WExpr, type: WType, pointer: Bool, writable: Bool, swizzle: WGSLSwizzleView? = nil) {
        self.handle = handle
        self.type = type
        self.pointer = pointer
        self.writable = writable
        self.swizzle = swizzle
    }
}
struct WGSLLoopScope {
    let depth: Int
    let declarations: Int
    var visibleAtFirstContinue: Set<String>?
}
final class WGSLFunctionBody {
    var arguments: [IRFunctionArgument] = []
    var result: IRFunctionResult?
    var locals: [IRLocalVariable] = []
    var expressions: [IRExpression] = []
    var pipelineStages: [IRPipelineEvaluationStage] = []
    var statements: [IRStatement] = []
    var blocks: [IRBlock] = []
    var scopes: [[String: WGSLValue]] = [[:]]
    var constScopes: [[String: WGSLScalarConstant]] = [[:]]
    var current: [IRHandle<IRStatement>] = []
    var loopDepth = 0
    var switchDepth = 0
    var continuing = false
    var loopScopes: [WGSLLoopScope] = []
    var continuingVisibility: [Int: Set<String>] = [:]
}

final class WGSLParser {
    let tokens: [WGSLToken]
    var cursor = 0
    var declarations: [WGSLModuleDeclaration] = []
    var declarationNames: [String: Int] = [:]
    var declarationStates: [UInt8] = []
    var dependencyDepth = 0
    var declarationStack: [Int] = []
    var declarationEdges: [Set<Int>] = []
    var budget: ShaderBudget
    let options: WGSLFrontendOptions
    var nesting = 0
    var types: [IRType] = []
    var typeNames: [String: WType] = [:]
    var structureAlignments: [WType: UInt32] = [:]
    var predeclaredTypes: [IRPredeclaredTypeBinding] = []
    var globals: [IRGlobalVariable] = []
    var constants: [IRConstant] = []
    var overrides: [IROverride] = []
    var overrideNames: [String: IRHandle<IROverride>] = [:]
    var hasOverrideDeclarations = false
    var globalExpressions: [IRExpression] = []
    var functions: [IRFunction] = []
    var entries: [IREntryPoint] = []
    var diagnosticFilters: [IRDiagnosticFilter] = []
    var float16Enabled = false
    var clipDistancesEnabled = false
    var primitiveIndexEnabled = false
    var subgroupsEnabled = false
    var subgroupIDEnabled = false
    var requiresSubgroupUniformity = false
    var subgroupSizeControlEnabled = false
    var linearIndexingEnabled = false
    var dualSourceBlendingEnabled = false
    var bindingArraysEnabled = false
    // Record whether the module needs pointer parameter lowering with subview bounds checks.
    var allowsPointerSubviews = false
    var uniformBufferStandardLayout = false
    var allowsSwizzleAssignment = false
    var moduleDiagnosticFilter: IRHandle<IRDiagnosticFilter>?
    var names: Set<String> = []
    var constantNames: [String: Int] = [:]
    var integerConstants: [String: WGSLIntegerConstant] = [:]
    var matrixConstants: [String:WGSLMatrixConstant] = [:]
    var compositeConstants: [String: WGSLScalarConstant] = [:]
    var decompositionConstants: [String: WGSLDecompositionConstant] = [:]
    var vectorConstants: [String: WGSLVectorConstant] = [:]
    var booleanConstants: [String: Bool] = [:]
    var floatConstants: [String: WGSLFloatingConstant] = [:]
    var abstractFloatConstants: [String: Double] = [:]
    var globalNames: [String: Int] = [:]
    var functionNames: [String: Int] = [:]
    var mustUseFunctions: Set<Int> = []
    var body = WGSLFunctionBody()
    var pipelineExpressionEnd: Int?
    init(tokens: [WGSLToken], budget: ShaderBudget, options: WGSLFrontendOptions) {
        self.tokens = tokens; self.budget = budget; self.options = options
    }
    var token: WGSLToken { tokens[cursor] }
    func take() throws -> WGSLToken {
        try budget.consume(.work)
        let t = token
        if cursor < tokens.count - 1 { cursor += 1 }
        return t
    }
    func isNext(_ text: String) -> Bool {
        switch text {
        case "<": token.kind == .templateStart
        case ">": token.kind == .templateEnd
        default: token.text == text
        }
    }
    @discardableResult func eat(_ text: String) throws -> Bool {
        try budget.consume(.work)
        if isNext(text) { _ = try take(); return true }; return false
    }
    @discardableResult func expect(_ text: String) throws -> WGSLToken {
        guard isNext(text) else { throw wgslError(.unexpectedToken,"Expected '\(text)', found '\(token.text)'",token) }
        return try take()
    }
    func unsupported(_ what: String, at t: WGSLToken? = nil) -> ShaderFailure {
        wgslError(.unsupportedConstruct,"WGSL construct not implemented: \(what)",t ?? token)
    }
    func enter() throws {
        try budget.consume(.work)
        guard nesting < min(budget.limits.depth, 128) else { throw ShaderFailure.budgetExceeded(.depth) }
        nesting += 1
    }
    func node() throws { try budget.consume(.nodes); try budget.consume(.work) }
    func ident() throws -> WGSLToken {
        let t = token
        guard t.kind == .word && t.text != "_" && !t.text.hasPrefix("__") else { throw wgslError(.unexpectedToken,"Expected identifier",t) }
        if Self.reserved.contains(t.text) { throw wgslError(.invalidToken,"Reserved identifier",t) }
        return try take()
    }
    static let reserved: Set<String> = ["asm","bf16","do","enum","f64","i8","i16","i64","typedef","u8","u16","u64","using","void","class","namespace","const_cast","volatile","static","switch","case","default","fn","var","let","const","alias","struct","return","if","else","for","while","loop","break","continue","continuing","discard","override","true","false","enable","requires","diagnostic"]
    func reserve(_ t: WGSLToken) throws {
        try budget.consume(.work,amount:t.text.utf8.count)
        if let i = declarationNames[wgslKey(t.text)], declarations[i].name.start == t.start {return}
        guard names.insert(wgslKey(t.text)).inserted else { throw wgslError(.duplicateName,"Duplicate declaration '\(t.text)'",t) }
    }
    func intern(_ kind: IRTypeKind, name: String? = nil, at t: WGSLToken? = nil) throws -> WType {
        for (i, existing) in types.enumerated() {
            try budget.consume(.work)
            if name == nil && existing.name == nil && existing.kind == kind { return try WType(index:i) }
        }
        try node()
        let h = try WType(index:types.count); types.append(IRType(name:name,kind:kind,location:t?.location)); return h
    }
    func module() throws -> IRModule {
        try parseModuleDirectives()
        try indexModuleDeclarations()
        for i in declarations.indices {
            try resolveDeclaration(i,at:declarations[i].name)
        }
        try budget.checkCancellation()
        try compactAbstractIntermediates()
        return IRModule(types:IRTable(types),specialTypes:.init(predeclared:predeclaredTypes),constants:IRTable(constants), overrides:IRTable(overrides),
                        globalVariables:IRTable(globals),globalExpressions:IRTable(globalExpressions),
                        functions:IRTable(functions),entryPoints:entries,
                        diagnosticFilters:IRTable(diagnosticFilters),diagnosticFilter:moduleDiagnosticFilter,
                        allowsPointerSubviews: allowsPointerSubviews, uniformBufferStandardLayout: uniformBufferStandardLayout,
                        requiresSubgroupUniformity: requiresSubgroupUniformity)
    }
    struct Attribute {
        let name: WGSLToken
        let expressions: [[WGSLToken]]
        var arguments: [WGSLToken] { expressions.flatMap { $0 } }
    }
    func attributes() throws -> [String: Attribute] {
        var out: [String: Attribute] = [:]
        while try eat("@") {
            let n = try isNext("diagnostic") ? take() : ident()
            let key = n.text == "diagnostic" ? "diagnostic@\(n.start)" : n.text
            guard out[key] == nil else { throw wgslError(.duplicateName,"Duplicate attribute",n) }
            let hasParentheses = try eat("(")
            if hasParentheses && (n.text == "invariant" || n.text == "must_use") {
                throw wgslError(.unexpectedToken, "@\(n.text) does not take parentheses", n)
            }
            let expressions: [[WGSLToken]]
            if n.text == "early_depth_test" {
                guard hasParentheses else {
                    throw wgslError(.unexpectedToken, "@early_depth_test requires a mode in parentheses", n)
                }
                expressions = [[try ident()]]
                try expect(")")
            } else {
                expressions = try hasParentheses ? attributeArguments(at: n) : []
            }
            try node(); out[key] = Attribute(name:n,expressions:expressions)
        }
        return out
    }
    func attributeArguments(at name: WGSLToken) throws -> [[WGSLToken]] {
        var expressions: [[WGSLToken]] = []
        var argument: [WGSLToken] = []; var depth = 0
        while !(isNext(")") && depth == 0) {
            guard token.kind != .end else { throw wgslError(.unexpectedToken,"Unterminated attribute",name) }
            if isNext(",") && depth == 0 {
                guard !argument.isEmpty else { throw wgslError(.unexpectedToken,"Empty attribute argument",token) }
                expressions.append(argument); argument=[]; _ = try take(); continue
            }
            if token.opensGroup { try enter(); depth += 1 }
            else if token.closesGroup {
                guard depth > 0 else { throw wgslError(.unexpectedToken, "Unmatched attribute delimiter", token) }
                depth -= 1; nesting -= 1
            }
            try node(); argument.append(try take())
        }
        if !argument.isEmpty { expressions.append(argument) }
        try expect(")")
        return expressions
    }
    func checkAttributes(_ attrs: [String: Attribute], allowed: Set<String>) throws {
        for a in attrs.values { try budget.consume(.work); guard allowed.contains(a.name.text) else { throw unsupported("attribute @\(a.name.text)",at:a.name) } }
    }
    func uint(_ t: WGSLToken) throws -> UInt32 {
        var text = t.text
        if text.hasSuffix("u") || text.hasSuffix("i") { text.removeLast() }
        let value = text.hasPrefix("0x") ? UInt32(text.dropFirst(2),radix:16) : UInt32(text)
        guard let value, t.kind == .number else { throw wgslError(.invalidType,"Expected u32 constant",t) }; return value
    }
    func attributeUInt(_ a: Attribute) throws -> UInt32 {
        let value = try attributeConstant(a)
        return try attributeValue(value,at:a.arguments.first ?? a.name)
    }
    func binding(_ attrs: [String: Attribute]) throws -> IRBinding? {
        try checkAttributes(attrs,allowed:["builtin","location","interpolate","invariant","blend_src"])
        let invariant = attrs["invariant"]
        if let invariant, attrs["builtin"]?.arguments.first?.text != "position" {
            throw wgslError(.invalidBinding,"@invariant requires @builtin(position)",invariant.name)
        }
        var blendSource: UInt32?
        if let blend = attrs["blend_src"] {
            guard dualSourceBlendingEnabled else {
                throw wgslError(.unsupportedCapability,"@blend_src requires enable dual_source_blending",blend.name)
            }
            let source = try attributeUInt(blend)
            guard attrs["location"] != nil, source <= 1 else {
                throw wgslError(.invalidBinding,"@blend_src requires @location and a source index of 0 or 1",blend.name)
            }
            blendSource = source
        }
        guard attrs["builtin"] == nil || attrs["location"] == nil,
              attrs["interpolate"] == nil || attrs["location"] != nil else {
            throw wgslError(.invalidBinding,"Conflicting binding attributes",token)
        }
        if let a = attrs["location"] {
            let index = try attributeUInt(a)
            guard let attribute = attrs["interpolate"] else {
                return .location(index:index,interpolation:nil,sampling:nil,blendSource:blendSource,perPrimitive:false)
            }
            let words = attribute.arguments
            guard (1...2).contains(words.count), attribute.expressions.count == words.count,
                  attribute.expressions.allSatisfy({ $0.count == 1 && $0[0].kind == .word }),
                  let interpolation = IRInterpolation(rawValue: words[0].text), interpolation != .perVertex else {
                throw wgslError(.invalidBinding,"Invalid interpolation type",attribute.name)
            }
            let sampling = words.count == 2 ? IRSampling(rawValue: words[1].text) :
                (interpolation == .flat ? .first : .center)
            let valid = interpolation == .flat ? sampling == .first || sampling == .either :
                sampling == .center || sampling == .centroid || sampling == .sample
            guard valid else { throw wgslError(.invalidBinding,"Invalid interpolation sampling",attribute.name) }
            return .location(index:index,interpolation:interpolation,sampling:sampling,blendSource:blendSource,perPrimitive:false)
        }
        if let a = attrs["builtin"] {
            guard a.expressions.count == 1, a.arguments.count == 1, let t = a.arguments.first, t.kind == .word else { throw wgslError(.invalidBinding,"Expected builtin name",a.name) }
            let builtin: IRBuiltIn
            switch t.text {
            case "global_invocation_id": builtin = .globalInvocationId
            case "global_invocation_index", "workgroup_index":
                guard linearIndexingEnabled else {
                    throw wgslError(.unsupportedCapability, "This builtin requires linear_indexing", t)
                }
                builtin = t.text == "global_invocation_index" ? .globalInvocationIndex : .workgroupIndex
            case "local_invocation_id": builtin = .localInvocationId
            case "local_invocation_index": builtin = .localInvocationIndex
            case "workgroup_id": builtin = .workgroupId
            case "num_workgroups": builtin = .numWorkgroups
            case "vertex_index": builtin = .vertexIndex
            case "instance_index": builtin = .instanceIndex
            case "position": builtin = .position(invariant: invariant != nil)
            case "clip_distances":
                guard clipDistancesEnabled else {
                    throw wgslError(.unsupportedCapability, "clip_distances requires enable clip_distances", t)
                }
                builtin = .clipDistances
            case "front_facing": builtin = .frontFacing
            case "primitive_index":
                guard primitiveIndexEnabled else {
                    throw wgslError(.unsupportedCapability, "primitive_index requires enable primitive_index", t)
                }
                builtin = .primitiveIndex
            case "barycentric", "barycentric_no_perspective":
                builtin = .barycentric(perspective: t.text == "barycentric")
            case "frag_depth": builtin = .fragDepth
            case "sample_index":
                guard options.capabilities.contains(.multisampledShading) else { throw wgslError(.unsupportedCapability,"sample_index requires multisampledShading",t) }
                builtin = .sampleIndex
            case "sample_mask": builtin = .sampleMask
            case "subgroup_size", "subgroup_invocation_id":
                try requireSubgroups(at: t)
                builtin = t.text == "subgroup_size" ? .subgroupSize : .subgroupInvocationId
            case "subgroup_id", "num_subgroups":
                try requireSubgroups(at: t)
                guard subgroupIDEnabled else {
                    throw wgslError(.unsupportedCapability, "This builtin requires subgroup_id", t)
                }
                builtin = t.text == "subgroup_id" ? .subgroupId : .numSubgroups
            default: throw unsupported("builtin '\(t.text)'",at:t)
            }
            return .builtIn(builtin)
        }
        return nil
    }
    func type() throws -> WType {
        try enter(); defer { nesting -= 1 }
        let t = try take()
        try resolveTypeDeclarationIfIndexed(t)
        if let known = typeNames[wgslKey(t.text)] { return known }
        if let kind = predeclaredType(t.text) {
            if kind.usesFloat16 { try requireFloat16(at: t) }
            return try intern(kind)
        }
        let scalar: IRScalar?
        switch t.text {
        case "u32": scalar = .u32
        case "u64": scalar = .u64
        case "i32": scalar = .i32
        case "i64": scalar = .i64
        case "f32": scalar = .f32
        case "f16": try requireFloat16(at: t); scalar = .f16
        case "bool": scalar = .bool
        default: scalar = nil
        }
        if let scalar { return try intern(.scalar(scalar)) }
        if t.text == "ptr" { return try pointerType(at: t) }
        if t.text == "texture_external" {
            return try intern(.image(dimension: .d2, arrayed: false, class: .external))
        }
        if t.text == "sampler" || t.text == "sampler_comparison" {
            return try intern(.sampler(comparison: t.text == "sampler_comparison"))
        }
        if ["texture_storage_1d", "texture_storage_2d", "texture_storage_2d_array", "texture_storage_3d"].contains(t.text) {
            try expect("<")
            let format = try take()
            try expect(",")
            let access = try take()
            _ = try eat(",")
            try expect(">")
            guard let storageFormat = IRStorageFormat.allCases.first(where: { $0.rawValue.lowercased() == format.text }) else {
                throw unsupported("storage texture format", at: format)
            }
            let mode: IRStorageAccess
            switch access.text {
            case "read": mode = .load
            case "write": mode = .store
            case "read_write": mode = [.load, .store]
            case "atomic": mode = [.load, .store, .atomic]
            default: throw unsupported("storage texture access", at: access)
            }
            let dimension: IRImageDimension = t.text == "texture_storage_1d" ? .d1 : t.text == "texture_storage_3d" ? .d3 : .d2
            let kind = IRTypeKind.image(dimension: dimension, arrayed: t.text == "texture_storage_2d_array",
                                        class: .storage(format: storageFormat, access: mode))
            guard kind.storageTexture != nil else { throw unsupported("storage texture format and access combination", at: t) }
            return try intern(kind)
        }
        if t.text == "texture_depth_2d" { return try intern(.sampledDepth2D) }
        if ["texture_depth_2d_array", "texture_depth_cube", "texture_depth_cube_array"].contains(t.text) {
            let dimension: IRImageDimension = t.text.contains("_cube") ? .cube : .d2
            return try intern(.image(dimension: dimension, arrayed: t.text.hasSuffix("_array"), class: .depth(multisampled: false)))
        }
        if t.text == "texture_depth_multisampled_2d" { return try intern(.multisampledDepth2D) }
        if ["texture_1d", "texture_2d", "texture_2d_array", "texture_multisampled_2d", "texture_cube", "texture_cube_array", "texture_3d"].contains(t.text) {
            try expect("<")
            let element = try type()
            try expect(">")
            guard case .scalar(let scalar) = types[element.index].kind,
                  [.f32, .i32, .u32].contains(scalar) else {
                throw wgslError(.typeMismatch, "Sampled texture element type must be f32, i32 or u32", t)
            }
            let dimension: IRImageDimension = switch t.text {
            case "texture_1d": .d1
            case "texture_3d": .d3
            case "texture_cube", "texture_cube_array": .cube
            default: .d2
            }
            return try intern(.image(dimension: dimension, arrayed: t.text.hasSuffix("_array"),
                                     class: .sampled(kind: scalar.kind, multisampled: t.text == "texture_multisampled_2d")))
        }
        if ["vec2","vec3","vec4"].contains(t.text) {
            try expect("<"); let elem = try type(); try expect(">")
            guard case .scalar(let s) = types[elem.index].kind, let size = IRVectorSize(rawValue:UInt8(t.text.last?.wholeNumberValue ?? 0)) else { throw wgslError(.invalidType,"Vector scalar required",t) }
            return try intern(.vector(size:size,scalar:s))
        }
        if ["mat2x2","mat2x3","mat2x4","mat3x2","mat3x3","mat3x4","mat4x2","mat4x3","mat4x4"].contains(t.text) {
            let chars = Array(t.text)
            guard chars[4] == "x", let c = chars[3].wholeNumberValue, let r = chars[5].wholeNumberValue,
                  let columns = IRVectorSize(rawValue:UInt8(c)), let rows = IRVectorSize(rawValue:UInt8(r)) else { throw wgslError(.invalidType,"Invalid matrix type",t) }
            try expect("<"); let elem = try type(); try expect(">")
            guard case .scalar(let scalar) = types[elem.index].kind, scalar == .f32 || scalar == .f16 else {
                throw wgslError(.typeMismatch, "Matrix scalar must be f32 or f16", t)
            }
            return try intern(.matrix(columns: columns, rows: rows, scalar: scalar))
        }
        if t.text == "array" || t.text == "binding_array" {
            let bindingArray = t.text == "binding_array"
            if bindingArray, !bindingArraysEnabled {
                throw wgslError(.unsupportedCapability, "binding_array requires enable wgpu_binding_array", t)
            }
            try expect("<")
            let elem = try type()
            var count = IRArraySize.dynamic
            if try eat(","), !isNext(">") {
                var expression: [WGSLToken] = []; var depth = 0
                while !(depth == 0 && (isNext(">") || isNext(","))) {
                    guard token.kind != .end else { throw wgslError(.unexpectedToken,"Unterminated array count",t) }
                    if token.opensGroup { try enter(); depth += 1 }
                    else if token.closesGroup {
                        guard depth > 0 else { throw wgslError(.unexpectedToken,"Unmatched array count delimiter",token) }
                        depth -= 1; nesting -= 1
                    }
                    try node(); expression.append(try take())
                }
                if let pending = try overrideArrayCount(expression, at: expression.first ?? t) {
                    count = .override(pending)
                } else {
                    let value = try attributeValue(integerConstant(expression, at: t), at: expression.first ?? t)
                    guard value > 0 else { throw wgslError(.invalidType, "Array count must be positive", t) }
                    count = .constant(value)
                }
                _ = try eat(",")
            }
            try expect(">")
            if bindingArray { return try intern(.bindingArray(base: elem, size: count), at: t) }
            guard try isSized(elem) else { throw wgslError(.invalidType,"Unsized array element",t) }
            let l = try layout(elem)
            return try intern(.array(base: elem, size: count, stride: try aligned(l.1, to: l.0, at: t)))
        }
        if t.text == "atomic" {
            try expect("<"); let elem = try type(); try expect(">")
            guard case .scalar(let s) = types[elem.index].kind, [.u32, .i32, .u64, .i64, .f32].contains(s) else {
                throw wgslError(.invalidType, "Atomic requires a supported numeric scalar", t)
            }
            return try intern(.atomic(s))
        }
        throw wgslError(.unresolvedName,"Unresolved type '\(t.text)'",t)
    }
    func aligned(_ value: UInt32, to alignment: UInt32, at t: WGSLToken) throws -> UInt32 {
        guard alignment > 0 else { throw wgslError(.invalidType,"Zero layout alignment",t) }
        let wide = ((UInt64(value) + UInt64(alignment)-1) / UInt64(alignment)) * UInt64(alignment)
        guard wide <= UInt64(Int32.max), let out = UInt32(exactly:wide) else { throw wgslError(.invalidType,"Layout overflow",t) }; return out
    }
    // Sizedness is independent of minimum footprint: a dynamic array occupies at least one stride in IR.
    private func pointerType(at t: WGSLToken) throws -> WType {
        try expect("<")
        let addressSpace = try take()
        try expect(",")
        let base = try type()
        if case .bindingArray = types[base.index].kind {
            throw wgslError(.invalidType, "A binding array cannot be the base of an explicit pointer type", t)
        }
        let space: IRAddressSpace
        switch addressSpace.text {
        case "function": space = .function
        case "private": space = .private
        case "workgroup": space = .workgroup
        case "uniform": space = .uniform
        case "immediate": space = .immediate
        case "storage":
            var access: IRStorageAccess = .load
            if try eat(","), !isNext(">") {
                let mode = try take()
                switch mode.text {
                case "read": break
                case "read_write": access.insert(.store)
                default: throw wgslError(.invalidAddressSpace, "Invalid storage pointer access mode", mode)
                }
            }
            space = .storage(access: access)
        default: throw wgslError(.invalidAddressSpace, "Invalid pointer address space", addressSpace)
        }
        _ = try eat(",")
        try expect(">")
        return try intern(.pointer(base: base, space: space), at: t)
    }

    func isSized(_ h: WType) throws -> Bool {
        try enter(); defer { nesting -= 1 }
        guard types.indices.contains(h.index) else { throw wgslError(.invalidReference,"Invalid layout type",token) }
        switch types[h.index].kind {
        case .array(let base, let count, _):
            guard case .constant = count else { return false }
            return try isSized(base)
        case .structure(let members, _):
            for member in members {
                try budget.consume(.work)
                if try !isSized(member.type) { return false }
            }
            return true
        default: return true
        }
    }
    // WGSL #alignment-and-size differs from Naga's width-based IR bool representation.
    // Keep IRScalar.bool.width == 1; only source-language memory layout uses four bytes.
    func memoryWidth(_ scalar: IRScalar) -> UInt32 {
        scalar.kind == .bool ? 4 : UInt32(scalar.width)
    }
    func layout(_ h: WType) throws -> (UInt32,UInt32) {
        try enter(); defer { nesting -= 1 }
        guard types.indices.contains(h.index) else { throw wgslError(.invalidReference,"Invalid layout type",token) }
        switch types[h.index].kind {
        case .scalar(let s), .atomic(let s): return (memoryWidth(s),memoryWidth(s))
        case .vector(let n,let s): return (UInt32(n == .two ? 2 : 4)*memoryWidth(s),UInt32(n.rawValue)*memoryWidth(s))
        case .matrix(let c,let r,let s):
            let alignment = UInt32(r == .two ? 2 : 4)*UInt32(s.width)
            return (alignment,UInt32(c.rawValue)*alignment)
        case .array(let base,let count,let stride):
            if case .override = count { throw unsupported("Workgroup array footprint requires specialization") }
            let alignment = try layout(base).0
            if case .constant(let n) = count {
                guard UInt64(n)*UInt64(stride) <= UInt64(Int32.max), let size = UInt32(exactly:UInt64(n)*UInt64(stride)) else { throw wgslError(.invalidType,"Array layout overflow",token) }; return (alignment,size)
            }; return (alignment,stride)
        case .structure(let members,let span):
            if let alignment = structureAlignments[h] { return (alignment, span) }
            var align: UInt32 = 1
            for member in members { try budget.consume(.work); align = max(align,try layout(member.type).0) }; return (align,span)
        default: throw unsupported("layout of opaque/pointer type")
        }
    }
    func structure() throws {
        let n = try ident(); try reserve(n); try expect("{")
        var members: [IRStructMember] = [], memberNames: Set<String> = []; var size: UInt32 = 0, alignment: UInt32 = 1
        while !isNext("}") {
            let attrs = try attributes(); let name = try ident()
            guard memberNames.insert(wgslKey(name.text)).inserted else { throw wgslError(.duplicateName,"Duplicate struct member",name) }
            if let last = members.last, try !isSized(last.type) { throw wgslError(.invalidType,"Runtime array must be final member",name) }
            try expect(":"); let ty = try type()
            if try !isSized(ty) {
                guard case .array(_, .dynamic, _) = types[ty.index].kind else {
                    throw wgslError(.invalidType,"Only a direct runtime array may be an unsized member",name)
                }
            }
            try checkAttributes(attrs, allowed: ["align", "size", "builtin", "location", "interpolate", "invariant", "blend_src"])
            let l = try layout(ty)
            var memberAlignment = l.0, memberSize = l.1
            if let a = attrs["align"] {
                let value = try attributeUInt(a)
                guard value > 0, value & (value - 1) == 0, value >= l.0, value <= UInt32(Int32.max) else {
                    throw wgslError(.invalidType,"Member alignment must be a power of two at least the natural alignment",a.arguments.first ?? a.name)
                }
                memberAlignment = value
            }
            if let a = attrs["size"] {
                let value = try attributeUInt(a)
                guard try isSized(ty), value >= l.1, value <= UInt32(Int32.max) else {
                    throw wgslError(.invalidType,"Member size requires fixed footprint and at least natural size",a.arguments.first ?? a.name)
                }
                memberSize = value
            }
            alignment = max(alignment,memberAlignment)
            let off = try aligned(size,to:memberAlignment,at:name)
            guard UInt64(off)+UInt64(memberSize) <= UInt64(Int32.max), let end = UInt32(exactly:UInt64(off)+UInt64(memberSize)) else { throw wgslError(.invalidType,"Struct layout overflow",name) }
            try node(); members.append(IRStructMember(name:name.text,type:ty,binding:try binding(attrs.filter { $0.key != "align" && $0.key != "size" }),offset:off,location:name.location)); size = end
            if !(try eat(",")) { break }
        }
        try expect("}")
        guard !members.isEmpty else { throw wgslError(.invalidType,"Empty structure",n) }
        let handle = try intern(.structure(members:members,span:try aligned(size,to:alignment,at:n)),name:n.text,at:n)
        structureAlignments[handle] = alignment
        typeNames[wgslKey(n.text)] = handle
    }
    func global(_ attrs: [String: Attribute]) throws {
        try checkAttributes(attrs,allowed:["group","binding"])
        var space: IRAddressSpace = .private
        let explicitSpace = try eat("<")
        if explicitSpace {
            let s = try take(); var access: IRStorageAccess = [.load]
            if try eat(",") {
                guard s.text == "storage" else { throw wgslError(.invalidAddressSpace,"Access mode only allowed for storage",s) }
                let a = try take()
                switch a.text { case "read": access = [.load]; case "read_write":access = [.load,.store]; default:throw unsupported("storage access",at:a) }
            }
            try expect(">")
            switch s.text {
            case "storage": space = .storage(access:access)
            case "uniform": space = .uniform
            case "immediate": space = .immediate
            case "private": space = .private
            case "workgroup": space = .workgroup
            default: throw unsupported("global address space",at:s)
            }
        }
        let n = try ident()
        try reserve(n)
        let annotation = try eat(":") ? try type() : nil
        let initializer: WExpr?
        let ty: WType
        if try eat("=") {
            guard space == .private else {
                throw wgslError(.invalidAddressSpace, "Only private globals can have initializers", n)
            }
            let tokens = try constantExpressionTokens(at: n)
            let value = try lowerPrivateInitializer(tokens, annotation: annotation, at: n)
            initializer = value
            ty = globalExpressions[value.index].type
        } else {
            guard let annotation else { throw wgslError(.invalidType, "Global variable needs a type or initializer", n) }
            ty = annotation
            initializer = nil
            try expect(";")
        }
        let kind = types[ty.index].kind
        let resourceKind: IRTypeKind
        if case .bindingArray(let element, _) = kind {
            resourceKind = types[element.index].kind
        } else {
            resourceKind = kind
        }
        if resourceKind.isTextureOrSampler {
            guard !explicitSpace else {
                throw wgslError(.invalidAddressSpace, "Texture and sampler address space is implicit", n)
            }
            space = .handle
        }
        var bind: IRResourceBinding?
        if let group = attrs["group"], let b = attrs["binding"] { bind = IRResourceBinding(group:try attributeUInt(group),binding:try attributeUInt(b)) }
        else if !attrs.isEmpty { throw wgslError(.invalidBinding,"group and binding must occur together",n) }
        if space == .handle || space == .uniform || { if case .storage = space { true } else { false } }() {
            guard bind != nil else { throw wgslError(.invalidBinding,"Resource requires binding",n) }
        } else if bind != nil { throw wgslError(.invalidBinding,"Non-resource cannot have binding",n) }
        try node()
        globalNames[wgslKey(n.text)] = globals.count
        globals.append(.init(name: n.text, space: space, binding: bind, type: ty, initializer: initializer, location: n.location))
    }
    func constant(_ attrs: [String: Attribute]) throws {
        guard attrs.isEmpty else { throw unsupported("constant attributes") }
        let n = try ident(); try reserve(n)
        let annotation = try eat(":") ? try type() : nil
        try expect("="); let t = token
        var expression: [WGSLToken] = []
        while !isNext(";") {
            guard token.kind != .end else { throw wgslError(.unexpectedToken,"Unterminated const declaration",t) }
            try node(); expression.append(try take())
        }
        var value = try scalarConstant(expression,at:t)
        if let annotation {value = try annotatedConstant(value,type:annotation,at:t)}
        try expect(";")
        let key=wgslKey(n.text)
        switch value {
        case .array, .composite: compositeConstants[key] = value; return
        case .decomposition(let parts): decompositionConstants[key] = parts; return
        case .matrix(let m):matrixConstants[key]=m;return
        case .vector(let v):vectorConstants[key]=v;return
        case .integer(let i):
            integerConstants[key]=i
            if i.kind == .abstract {return}
        case .boolean(let b):booleanConstants[key]=b
        case .floating(let f):
            floatConstants[key]=f
            if f.kind == .abstract {abstractFloatConstants[key]=f.value;return}
        }
        let ty = try annotation ?? intern(.scalar(value.scalar))
        try node();let initializer=try WExpr(index:globalExpressions.count)
        globalExpressions.append(.init(type:ty,kind:.literal(try value.literal),location:t.location))
        try node();constantNames[key]=constants.count
        constants.append(.init(name:n.text,type:ty,initializer:initializer,location:n.location))
    }
}
