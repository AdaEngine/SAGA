import Testing
import SAGATypes

@Test func emptyQueryResolveAcceptsEndOfQuerySet() throws {
    let query = ResourceID(device: 1, serial: 1)
    let buffer = ResourceID(device: 1, serial: 2)
    let descriptors = [buffer: BufferDescriptor(size: 256, usage: .queryResolve)]
    let queries = [query: QuerySetDescriptor(type: .occlusion, count: 4)]
    try BufferCommand.resolveQuerySet(querySet: query, firstQuery: 4, queryCount: 0,
                                      destination: buffer, destinationOffset: 256)
        .validate(device: 1, descriptors: descriptors, querySets: queries)
    #expect(throws: GPUError.self) {
        try BufferCommand.resolveQuerySet(querySet: query, firstQuery: 5, queryCount: 0,
                                          destination: buffer, destinationOffset: 256)
            .validate(device: 1, descriptors: descriptors, querySets: queries)
    }
}
