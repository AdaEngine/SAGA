// Exact production string method extracted for isolated availability/behavior verification.
// Not an integrated frontend test or proof that the blocked package builds.
struct SPVInstruction { func fail(_ message: String) -> Failure { Failure() } }
struct Failure: Error {}
struct Decoder {
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
}
func words(_ bytes: [UInt8]) -> [UInt32] {
    var bytes=bytes; while bytes.count % 4 != 0 {bytes.append(0)}
    return stride(from:0,to:bytes.count,by:4).map { j in
        let a=UInt32(bytes[j]);let b=UInt32(bytes[j+1]) << 8
        let c=UInt32(bytes[j+2]) << 16;let d=UInt32(bytes[j+3]) << 24
        return a | b | c | d
    }
}
let d=Decoder(), i=SPVInstruction()
var passed=0
for text in ["", "main", "é", "😀", "aé😀", "�", String(repeating:"x",count:4096)] {
    let w=words(Array(text.utf8)+[0]);let result=try d.string(w,from:0,i)
    precondition(result.0==text && result.1==w.count);passed+=1
}
for bad: [UInt8] in [[0xc0,0x80,0], [0xed,0xa0,0x80,0], [0xf4,0x90,0x80,0x80,0], [0x80,0], [0xe2,0x82,0], [0,1,0,0], [0x61,0x62,0x63,0x64]] {
    do {_ = try d.string(words(bad),from:0,i);preconditionFailure("invalid UTF8/padding accepted")}
    catch is Failure {passed+=1}
}
print("PASS \(passed) exact-production string checks; target macOS14, running on host OS")
