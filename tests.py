import unittest
import numpy as np
import pygraphblas as pb
from bfs import bfs


def matrix_from_edges(edges, size):
    lists = np.array(edges).T.tolist()
    return pb.Matrix.from_lists(lists[0], lists[1], [True] * len(lists[0]), size, size)


def vector_from_vertices(starting_vertices, size):
    return pb.Vector.from_lists(starting_vertices, [True] * len(starting_vertices), size)


def convert_to_vector(expected):
    indices = [i for i in range(len(expected)) if expected[i] is not None]
    values = [expected[i] + 1 for i in indices]
    return pb.Vector.from_lists(indices, values, len(expected))


class TestBFS(unittest.TestCase):
    def init(self, edges, size):
        self.size = size
        self.graph = matrix_from_edges(edges, size)

    def template_test(self, startings, expected):
        start_vector = vector_from_vertices(startings, self.size)
        actual = bfs(self.graph, start_vector)
        self.assertTrue(convert_to_vector(expected).iseq(actual))


class TestSimple(TestBFS):
    def setUp(self):
        size = 2
        edges = [[0, 1]]
        super().init(edges, size)

    def test_1(self):
        startings = [0]
        expected = [0, 0]
        super().template_test(startings, expected)

    def test_2(self):
        startings = [1]
        expected = [None, 1]
        super().template_test(startings, expected)

class TestRegular(TestBFS):
    def setUp(self):
        size = 5
        edges = [[0, 1], [0, 4], [1, 4], [4, 2], [2, 3]]
        super().init(edges, size)

    def test_1(self):
        startings = [0, 1, 2, 3, 4]
        expected = [0, 1, 2, 3, 4]
        super().template_test(startings, expected)

    def test_2(self):
        startings = [0]
        expected = [0, 0, 0, 0, 0]
        super().template_test(startings, expected)

    def test_3(self):
        startings = [0, 4]
        expected = [0, 0, 4, 4, 4]
        super().template_test(startings, expected)

    def test_4(self):
        startings = [2, 3]
        expected = [None, None, 2, 3, None]
        super().template_test(startings, expected)


class TestCycle(TestBFS):
    def setUp(self):
        size = 3
        edges = [[0, 1], [1, 2], [2, 0]]
        super().init(edges, size)

    def test_1(self):
        startings = [0]
        expected = [0, 0, 0]
        super().template_test(startings, expected)

    def test_2(self):
        startings = [1]
        expected = [1, 1, 1]
        super().template_test(startings, expected)

    def test_3(self):
        startings = [0, 1]
        expected = [0, 1, 1]
        super().template_test(startings, expected)


class TestBig(TestBFS):
    def setUp(self) -> None:
        size = 10
        edges = [[0, 8], [1, 7], [6, 4], [5, 1], [9, 3], [3, 2], [8, 9], [6, 4], [2, 4]]
        super().init(edges, size)

    def test_1(self):
        startings = [0]
        expected = [0, None, 0, 0, 0, None, None, None, 0, 0]
        super().template_test(startings, expected)

    def test_2(self):
        startings = [2]
        expected = [None, None, 2, None, 2, None, None, None, None, None]
        super().template_test(startings, expected)

    def test_3(self):
        startings = [5, 6, 7, 8]
        expected = [None, 5, 8, 8, 6, 5, 6, 7, 8, 8]
        super().template_test(startings, expected)


if __name__ == '__main__':
    unittest.main()
