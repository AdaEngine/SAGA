public enum ShaderBudgetResource: String, Sendable, CaseIterable {
    case sourceBytes, words, tokens, nodes, depth, work
}

public struct ShaderLimits: Sendable, Equatable {
    public let sourceBytes: Int
    public let words: Int
    public let tokens: Int
    public let nodes: Int
    public let depth: Int
    public let work: Int
    public init(sourceBytes: Int = 4_194_304, words: Int = 1_048_576, tokens: Int = 1_048_576,
                nodes: Int = 1_048_576, depth: Int = 128, work: Int = 16_777_216) throws {
        guard sourceBytes >= 0, words >= 0, tokens >= 0, nodes >= 0, depth >= 0, depth <= 512, work >= 0 else {
            throw ShaderFailure.invalidBudget
        }
        self.sourceBytes = sourceBytes; self.words = words; self.tokens = tokens
        self.nodes = nodes; self.depth = depth; self.work = work
    }
    private init() {
        sourceBytes = 4_194_304; words = 1_048_576; tokens = 1_048_576
        nodes = 1_048_576; depth = 128; work = 16_777_216
    }
    public static let `default` = ShaderLimits()
    public subscript(resource: ShaderBudgetResource) -> Int {
        switch resource {
        case .sourceBytes: sourceBytes
        case .words: words
        case .tokens: tokens
        case .nodes: nodes
        case .depth: depth
        case .work: work
        }
    }
}

/// Per-invocation value-semantic accounting. Consumers must charge every scan/expansion/traversal step.
/// Copying a budget starts an independent accounting branch, NOT a shared global quota.
public struct ShaderBudget: Sendable {
    public let limits: ShaderLimits
    private var used: [ShaderBudgetResource: Int] = [:]
    public init(limits: ShaderLimits = .default) { self.limits = limits }
    public func usage(_ resource: ShaderBudgetResource) -> Int { used[resource, default: 0] }
    public func checkCancellation() throws { try Task.checkCancellation() }
    public mutating func consume(_ resource: ShaderBudgetResource, amount: Int = 1) throws {
        try Task.checkCancellation()
        guard resource != .depth, amount >= 0 else { throw ShaderFailure.invalidBudget }
        let previous = usage(resource)
        guard amount <= limits[resource] - previous else { throw ShaderFailure.budgetExceeded(resource) }
        used[resource] = previous + amount
    }
    public mutating func enterDepth() throws {
        try Task.checkCancellation()
        guard usage(.depth) < limits.depth else { throw ShaderFailure.budgetExceeded(.depth) }
        used[.depth] = usage(.depth) + 1
    }
    public mutating func leaveDepth() throws {
        try Task.checkCancellation()
        guard usage(.depth) > 0 else { throw ShaderFailure.invalidBudget }
        used[.depth] = usage(.depth) - 1
    }
    /// Scoped nesting accounting is restored even when the body throws or is cancelled.
    public mutating func withDepth<T>(_ body: (inout ShaderBudget) throws -> T) throws -> T {
        let previous = usage(.depth)
        try enterDepth()
        defer { used[.depth] = previous }
        return try body(&self)
    }
    /// Counts UTF-8 bytes lazily; stops at first exhausted limit. Defines count too.
    /// Call once per invocation. Failed preflight leaves charged work; discard that invocation.
    public mutating func checkInput(_ input: ShaderInput) throws {
        try Task.checkCancellation()
        switch input {
        case .wgsl(let source): try chargeText(source)
        case .glsl(let source, _, let defines):
            try chargeText(source)
            for (name, value) in defines {
                try consume(.work)
                try chargeText(name); try chargeText(value)
            }
        case .spirv(let words):
            try consume(.words, amount: words.count)
            try consume(.work, amount: words.count)
        }
    }
    private mutating func chargeText(_ text: String) throws {
        for _ in text.utf8 {
            try consume(.sourceBytes)
            try consume(.work)
        }
    }
}
