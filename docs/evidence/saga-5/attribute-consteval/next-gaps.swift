import SAGAShaderIR
import SAGAFrontends
import SAGAValidation
@main struct NextGaps {
    static func main() {
        let cases = [
            ("original-resolved", "const A = 16u; struct S { @align(A) x: f32, }"),
            ("module-forward-reference", "struct S { @align(A) x: f32, } const A = 16u;"),
            ("const-builtin", "const A = min(16u, 32u); struct S { @align(A) x: f32, }"),
            ("abstract-constant-validator", "const A = 16; struct S { @align(A) x: f32, }")
        ]
        for (name,source) in cases {
            do {
                let raw=try WGSLFrontend().parse(.wgsl(source))
                print("\(name): parsed")
                do {try ShaderSemanticValidator().validate(raw);print("\(name): validated")}
                catch {print("\(name): validator \(error)")}
            } catch {print("\(name): frontend \(error)")}
        }
    }
}
