package struct IRBitcastShape {
    package let size: IRVectorSize?
    package let scalar: IRScalar

    package init(size: IRVectorSize?, scalar: IRScalar) {
        self.size = size
        self.scalar = scalar
    }

    package init?(_ type: IRTypeKind) {
        switch type {
        case .scalar(let scalar): self.init(size: nil, scalar: scalar)
        case .vector(let size, let scalar): self.init(size: size, scalar: scalar)
        default: return nil
        }
    }

    package var type: IRTypeKind {
        size.map { .vector(size: $0, scalar: scalar) } ?? .scalar(scalar)
    }

    package func permits(_ target: Self) -> Bool {
        guard scalar.kind != .bool, target.scalar.kind != .bool else { return false }
        return (size == target.size && scalar.width == target.scalar.width) || packedHalfWords(to: target) != nil
    }

    package func packedHalfWords(to target: Self) -> Int? {
        if scalar == .f16 { return Self.packedHalfWords(half: self, word: target) }
        if target.scalar == .f16 { return Self.packedHalfWords(half: target, word: self) }
        return nil
    }

    private static func packedHalfWords(half: Self, word: Self) -> Int? {
        guard [.i32, .u32, .f32].contains(word.scalar) else { return nil }
        switch (half.size, word.size) {
        case (.two?, nil): return 1
        case (.four?, .two?): return 2
        default: return nil
        }
    }
}
