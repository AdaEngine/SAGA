/// Typed table-relative index, not an ownership/security token or validation witness.
/// Context (module vs function arena) is supplied by the owning field, never guessed.
public struct IRHandle<Node: Sendable>: Sendable, Hashable {
    public let index: Int
    public init(index: Int) throws {
        guard index >= 0, UInt64(index) <= UInt64(UInt32.max) else { throw ShaderFailure.invalidHandle }
        self.index = index
    }
    public init(rawValue: UInt32) throws {
        guard let index = Int(exactly: rawValue) else { throw ShaderFailure.invalidHandle }
        self.index = index
    }
    public static func == (lhs: Self, rhs: Self) -> Bool { lhs.index == rhs.index }
    public func hash(into hasher: inout Hasher) { hasher.combine(index) }
}

/// Immutable value-semantic table. No unchecked integer subscript is public.
public struct IRTable<Element: Sendable>: Sendable {
    public let values: [Element]
    public init(_ values: [Element] = []) { self.values = values }
    public var count: Int { values.count }
    public func handle(at index: Int) throws -> IRHandle<Element> {
        guard values.indices.contains(index) else { throw ShaderFailure.invalidHandle }
        return try IRHandle(index: index)
    }
    public subscript(handle: IRHandle<Element>) -> Element? {
        guard values.indices.contains(handle.index) else { return nil }
        return values[handle.index]
    }
    public func value(for handle: IRHandle<Element>) throws -> Element {
        guard let value = self[handle] else { throw ShaderFailure.invalidHandle }
        return value
    }
}
extension IRTable: Equatable where Element: Equatable {}

/// Local mutable assembly state; freeze publishes an immutable snapshot. No global counters.
public struct IRTableBuilder<Element: Sendable>: Sendable {
    private var values: [Element] = []
    public init() {}
    public var count: Int { values.count }
    public mutating func append(_ value: Element, budget: inout ShaderBudget) throws -> IRHandle<Element> {
        let handle = try IRHandle<Element>(index: values.count)
        try budget.consume(.nodes)
        try budget.consume(.work)
        values.append(value)
        return handle
    }
    public func freeze() -> IRTable<Element> { IRTable(values) }
}
