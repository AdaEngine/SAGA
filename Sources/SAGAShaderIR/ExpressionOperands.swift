extension IRExpressionKind {
    package var operands: [IRHandle<IRExpression>] {
      switch self {
      case .unary(_, let x), .load(let x), .runtimeValue(let x), .derivative(_, _, let x), .cast(let x, _),
        .bitcast(let x, _), .relational(_, let x), .arrayLength(let x):
        return [x]
      case .binary(_, let a, let b), .access(let a, let b): return [a, b]
      case .select(let c, let a, let b): return [c, a, b]
      case .accessIndex(let b, _), .splat(_, let b), .swizzle(_, let b, _): return [b]
      case .imageSample(let sample):
        return [sample.image, sample.sampler, sample.coordinate] + sample.level.operands +
          [sample.arrayIndex, sample.offset, sample.depthReference].compactMap { $0 }
      case .imageLoad(let load):
        return [load.image, load.coordinate] + [load.arrayIndex, load.sample, load.level].compactMap { $0 }
      case .imageQuery(let image, .size(let level?)): return [image, level]
      case .imageQuery(let image, _): return [image]
      case .compose(_, let xs): return xs
      case .math(_, let a, let b, let c, let d): return [a] + [b, c, d].compactMap({ $0 })
      default: return []
      }
    }
}
