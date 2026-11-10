/// Packs bytes into little-endian 32-bit words. A plain loop keeps long shift-and-or chains out of the type checker.
func packLittleEndianWords(_ bytes: [UInt8]) -> [UInt32] {
    var words: [UInt32] = []
    var index = 0
    while index + 3 < bytes.count {
        var word: UInt32 = UInt32(bytes[index])
        word |= UInt32(bytes[index + 1]) << 8
        word |= UInt32(bytes[index + 2]) << 16
        word |= UInt32(bytes[index + 3]) << 24
        words.append(word)
        index += 4
    }
    return words
}
