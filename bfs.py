from pygraphblas import Vector


def bfs(graph, start_vertices):
    current_vertices = start_vertices.nonzero() & Vector.from_1_to_n(start_vertices.size)
    next_vertices = bfs_step(graph, current_vertices)
    while next_vertices.isne(current_vertices):
        current_vertices = next_vertices
        next_vertices = bfs_step(graph, current_vertices)
    return current_vertices


def bfs_step(graph, current_vertices):
    next_vertices = current_vertices.vxm(graph, semiring=current_vertices.type.min_times)
    return current_vertices.eadd(next_vertices, current_vertices.type.first)
