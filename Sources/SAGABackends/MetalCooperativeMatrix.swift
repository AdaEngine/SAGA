import SAGAShaderIR

// A value consists of independently distributed 8x8 simdgroup tiles.
let metalCooperativeMatrixPrelude: String = ["float", "half"].flatMap { scalar in
    [(16, 8), (8, 16), (16, 16)].map { columns, rows in
        let fields = (0..<(rows / 8)).flatMap { row in
            (0..<(columns / 8)).map { column in
                "    simdgroup_\(scalar)8x8 t\(row)_\(column);"
            }
        }.joined(separator: "\n")
        return "struct saga_coop_\(scalar)_\(columns)x\(rows) {\n\(fields)\n};"
    }
}.joined(separator: "\n")
