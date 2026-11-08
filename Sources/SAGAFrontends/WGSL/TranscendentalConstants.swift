import SAGAShaderIR

extension WGSLScalarEvaluator {
  func transcendentalConstant(_ function: IRMathFunction, values: [Double]) throws -> Double {
    try parser.budget.consume(.work, amount: 128)
    let x = values[0]
    switch function {
    case .sin: return ConstantMath.sinCos(x).sin
    case .cos: return ConstantMath.sinCos(x).cos
    case .tan:
      let pair = ConstantMath.sinCos(x)
      return pair.sin / pair.cos
    case .asin, .acos:
      guard abs(x) <= 1 else { return .nan }
      let adjacent = ((1 - abs(x)) * (1 + abs(x))).squareRoot()
      return function == .asin ? ConstantMath.atan2(x, adjacent) : ConstantMath.atan2(adjacent, x)
    case .atan: return ConstantMath.atan2(x, 1)
    case .atan2: return ConstantMath.atan2(x, values[1])
    case .sinh, .cosh:
      let magnitude = abs(x)
      let positive: Double
      if magnitude <= 0.5 {
        let a = ConstantMath.expMinusOne(magnitude)
        let b = ConstantMath.expMinusOne(-magnitude)
        positive = function == .sinh ? (a - b) * 0.5 : 1 + (a + b) * 0.5
      } else {
        let root = ConstantMath.exp(magnitude * 0.5)
        let half = (0.5 * root) * root
        positive = function == .sinh ? half - 0.25 / half : half + 0.25 / half
      }
      return function == .sinh && x.sign == .minus ? -positive : positive
    case .tanh:
      let delta = -ConstantMath.expMinusOne(-2 * abs(x))
      return (x.sign == .minus ? -delta : delta) / (2 - delta)
    case .asinh:
      let a = abs(x)
      let result: Double
      if a > 1e150 {
        result = ConstantMath.log(a) + ConstantMath.ln2
      } else {
        let square = a * a
        result = ConstantMath.logOnePlus(a + square / (1 + (1 + square).squareRoot()))
      }
      return x.sign == .minus ? -result : result
    case .acosh:
      guard x >= 1 else { return .nan }
      if x > 1e150 { return ConstantMath.log(x) + ConstantMath.ln2 }
      return ConstantMath.logOnePlus((x - 1) + ((x - 1) * (x + 1)).squareRoot())
    case .atanh:
      guard abs(x) <= 1 else { return .nan }
      return 0.5 * (ConstantMath.logOnePlus(x) - ConstantMath.logOnePlus(-x))
    case .exp: return ConstantMath.exp(x)
    case .exp2: return ConstantMath.exp2(x)
    case .log: return ConstantMath.log(x)
    case .log2: return ConstantMath.log(x, base2: true)
    case .pow:
      let exponent = values[1]
      guard x >= 0 else { return .nan }
      if exponent == 0 { return 1 }
      if x == 0 { return exponent > 0 ? 0 : .infinity }
      return ConstantMath.exp2(exponent * ConstantMath.log(x, base2: true))
    case .radians: return x * (Double.pi / 180)
    case .degrees: return x * (180 / Double.pi)
    default: throw ShaderFailure.invalidHandle
    }
  }
}

private enum ConstantMath {
  static let ln2 = 0.6931471805599453094
  static let inverseLn2 = 1.4426950408889634074
  private static let ln2High = 0x1.62e42fee00000p-1
  private static let ln2Low = 1.9082149292705877e-10

  // Sum exp(x)-1 on a small interval to retain the low bits when x is near zero.
  private static func expSeries(_ x: Double) -> Double {
    var term = x
    var sum = x
    for n in 2...20 { term *= x / Double(n); sum += term }
    return sum
  }

  static func expMinusOne(_ x: Double) -> Double { abs(x) <= 0.5 ? expSeries(x) : exp(x) - 1 }

  static func exp(_ x: Double) -> Double {
    if x.isNaN { return .nan }
    if x > 710 { return .infinity }
    if x < -746 { return 0 }
    let n = (x * inverseLn2).rounded(.toNearestOrEven)
    let residual = (x - n * ln2High) - n * ln2Low
    return Double(sign: .plus, exponent: Int(n), significand: 1 + expSeries(residual))
  }

  static func exp2(_ x: Double) -> Double {
    if x.isNaN { return .nan }
    if x >= 1024 { return .infinity }
    if x <= -1075 { return 0 }
    let n = x.rounded(.toNearestOrEven)
    return Double(sign: .plus, exponent: Int(n), significand: 1 + expSeries((x - n) * ln2))
  }

  private static func logSeries(_ ratio: Double) -> Double {
    let square = ratio * ratio
    var term = ratio
    var sum = ratio
    for n in 1...22 { term *= square; sum += term / Double(2 * n + 1) }
    return 2 * sum
  }

  static func log(_ x: Double, base2: Bool = false) -> Double {
    if x == 0 { return -.infinity }
    guard x > 0 else { return .nan }
    if x.isInfinite { return .infinity }
    var exponent = x.exponent
    var significand = x.significand
    if significand > 1.4142135623730951 { significand *= 0.5; exponent += 1 }
    let mantissa = logSeries((significand - 1) / (significand + 1))
    let scale = Double(exponent)
    return base2 ? scale + mantissa * inverseLn2 : scale * ln2High + (mantissa + scale * ln2Low)
  }

  static func logOnePlus(_ x: Double) -> Double {
    abs(x) <= 0.5 ? logSeries(x / (2 + x)) : log(1 + x)
  }

  static func atan2(_ y: Double, _ x: Double) -> Double {
    if y == 0 { return x.sign == .minus ? (y.sign == .minus ? -.pi : .pi) : y }
    let a = abs(x), b = abs(y)
    var ratio = min(a, b) / max(a, b)
    let aroundOne = ratio > 0.41421356237309503
    if aroundOne { ratio = (ratio - 1) / (ratio + 1) }
    let square = -(ratio * ratio)
    var term = ratio
    var angle = ratio
    for n in 1...25 { term *= square; angle += term / Double(2 * n + 1) }
    if aroundOne { angle += .pi / 4 }
    if b > a { angle = .pi / 2 - angle }
    if x < 0 { angle = .pi - angle }
    return y.sign == .minus ? -angle : angle
  }

  static func sinCos(_ x: Double) -> (sin: Double, cos: Double) {
    let magnitude = abs(x)
    let (quadrant, angle) = magnitude <= .pi / 4 ? (0, magnitude) : reducedAngle(magnitude)
    let square = -(angle * angle)
    var sinTerm = angle, cosTerm = 1.0
    var sine = angle, cosine = 1.0
    for n in 1...12 {
      sinTerm *= square / Double(2 * n * (2 * n + 1))
      cosTerm *= square / Double((2 * n - 1) * 2 * n)
      sine += sinTerm
      cosine += cosTerm
    }
    let rotated: (Double, Double) = switch quadrant {
    case 0: (sine, cosine)
    case 1: (cosine, -sine)
    case 2: (-sine, -cosine)
    default: (-cosine, sine)
    }
    return (x.sign == .minus ? -rotated.0 : rotated.0, rotated.1)
  }

  // Little-endian limbs of floor((2 / pi) * 2^1280) for binary64 argument reduction.
  private static let twoOverPi: [UInt64] = [
    0xf0cfbc209af4361d, 0x56033046fc7b6bab, 0x6bfb5fb11f8d5d08,
    0x3d0739f78a5292ea, 0x7527bac7ebe5f17b, 0x4f463f669e5fea2d,
    0x6d367ecf27cb09b7, 0xef2f118b5a0a6d1f, 0x1ff897ffde05980f,
    0x9c845f8bbdf9283b, 0x3991d639835339f4, 0xe99c7026b45f7e41,
    0xe88235f52ebb4484, 0xfe1deb1cb129a73e, 0x06492eea09d1921c,
    0xb7246e3a424dd2e0, 0xfe5163abdebbc561, 0xdb6295993c439041,
    0xfc2757d1f534ddc0, 0xa2f9836e4e441529,
  ]

  private static func reducedAngle(_ x: Double) -> (Int, Double) {
    let bits = x.bitPattern
    let significand = (bits & 0x000f_ffff_ffff_ffff) | 0x0010_0000_0000_0000
    let binaryPoint = 1280 - (x.exponent - 52)
    var product: [UInt64] = []
    var carry: UInt64 = 0
    for limb in twoOverPi {
      let full = limb.multipliedFullWidth(by: significand)
      let (word, overflow) = full.low.addingReportingOverflow(carry)
      product.append(word)
      carry = full.high + (overflow ? 1 : 0)
    }
    product.append(carry)
    func word(at bit: Int) -> UInt64 {
      let limb = bit / 64, shift = bit % 64
      let low = product.indices.contains(limb) ? product[limb] : 0
      if shift == 0 { return low }
      let high = product.indices.contains(limb + 1) ? product[limb + 1] : 0
      return (low >> shift) | (high << (64 - shift))
    }
    let integral = word(at: binaryPoint) & 3
    var high = word(at: binaryPoint - 64)
    var low = word(at: binaryPoint - 128)
    let roundUp = high >> 63 != 0
    if roundUp {
      high = ~high &+ (low == 0 ? 1 : 0)
      low = 0 &- low
    }
    let fraction = Double(high) * 0x1p-64 + Double(low) * 0x1p-128
    let remainder = (roundUp ? -fraction : fraction) * (.pi / 2)
    return (Int((integral + (roundUp ? 1 : 0)) & 3), remainder)
  }
}
