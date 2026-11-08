import SAGAShaderIR

private struct MatrixMinor: Hashable {
    let rows: [Int]
    let columns: [Int]
}

func inverseMatrixStatements(
    _ argument: String, size: IRVectorSize, scalar: IRScalar, resultType: String,
    named name: String, format: ShaderOutputFormat
) throws -> [String] {
    let indices = Array(0..<Int(size.rawValue))
    let captured = "\(name)_operand"
    let wideVector = try shaderVectorType(size.rawValue, scalar: .f32, in: format)
    let wideScalar = floatType(1, in: format)
    func binding(_ type: String, _ name: String, _ value: String) -> String {
        let declaration = format == .wgsl ? "let \(name)" : "\(type) \(name)"
        return "\(declaration) = \(value);"
    }
    func maximum(_ values: [String]) -> String {
        values.dropFirst().reduce(values[0]) { "max(\($0), \($1))" }
    }

    var statements = [binding(resultType, captured, argument)]
    let columns = indices.map { "\(name)_column_\($0)" }
    for column in indices {
        let value = "\(captured)[\(column)u]"
        statements.append(binding(wideVector, columns[column], scalar == .f16 ? "\(wideVector)(\(value))" : value))
    }
    // Work at unit scale so otherwise well-conditioned matrices do not lose their determinant
    // to underflow or overflow. Keep half inputs in f32 until the result is assembled.
    let magnitudes = "\(name)_magnitudes"
    statements.append(binding(wideVector, magnitudes, maximum(columns.map { "abs(\($0))" })))
    let scale = "\(name)_scale"
    statements.append(binding(wideScalar, scale, maximum(indices.map { "\(magnitudes)[\($0)u]" })))
    let normalized = indices.map { "\(name)_normalized_\($0)" }
    for column in indices {
        let lanes = indices.map { "\(columns[column])[\($0)u] / \(scale)" }.joined(separator: ", ")
        statements.append(binding(wideVector, normalized[column], "\(wideVector)(\(lanes))"))
    }
    func element(_ row: Int, _ column: Int) -> String {
        "\(normalized[column])[\(row)u]"
    }

    var minors: [MatrixMinor: String] = [:]
    func minor(_ rows: [Int], _ columns: [Int]) -> String {
        if rows.count == 1 { return element(rows[0], columns[0]) }
        let key = MatrixMinor(rows: rows, columns: columns)
        if let existing = minors[key] { return existing }
        let expression: String
        if rows.count == 2 {
            expression = "\(element(rows[0], columns[0])) * \(element(rows[1], columns[1])) - \(element(rows[0], columns[1])) * \(element(rows[1], columns[0]))"
        } else {
            let tail = Array(rows.dropFirst())
            let terms = columns.enumerated().map { index, column in
                let term = "\(element(rows[0], column)) * \(minor(tail, columns.filter { $0 != column }))"
                return index.isMultiple(of: 2) ? term : "-(\(term))"
            }
            expression = terms.joined(separator: " + ")
        }
        let temporary = "\(name)_minor_\(minors.count)"
        statements.append(binding(wideScalar, temporary, expression))
        minors[key] = temporary
        return temporary
    }

    let cofactors = indices.map { row in
        indices.map { column in
            let value = minor(indices.filter { $0 != row }, indices.filter { $0 != column })
            return (row + column).isMultiple(of: 2) ? value : "-(\(value))"
        }
    }
    let determinant = "\(name)_determinant"
    let expansion = indices.map { "\(element(0, $0)) * (\(cofactors[0][$0]))" }.joined(separator: " + ")
    statements.append(binding(wideScalar, determinant, expansion))
    let outputVector = try shaderVectorType(size.rawValue, scalar: scalar, in: format)
    let inverseColumns = indices.map { column in
        // A column of the adjugate is a row of the cofactor matrix.
        let values = indices.map { "((\(cofactors[column][$0])) / \(determinant)) / \(scale)" }
        let wide = "\(wideVector)(\(values.joined(separator: ", ")))"
        guard scalar == .f16 else { return wide }
        return format == .hlsl ? "(\(outputVector))(\(wide))" : "\(outputVector)(\(wide))"
    }
    statements.append(binding(resultType, name, "\(resultType)(\(inverseColumns.joined(separator: ", ")))"))
    return statements
}
