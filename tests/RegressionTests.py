"""
Copyright (c) 2012-2023 Oscar Riveros [https://twitter.com/maxtuno].
Copyright (c) 2023 Balázs Glávits <balazs@glavits.hu>

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import fractions
import functools
import itertools
import math
import operator
import random
import string
import unittest
from functools import lru_cache

import numpy as np

import satx
from tests.SatXTestCase import SatXTestCase


class RegressionTests(SatXTestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_simple_example_1(self):
        self.engine(bits=10)

        x = satx.integer()
        y = satx.integer()

        assert 0 < x <= 3
        assert 0 < y <= 3

        assert x + y > 2

        ground_truth = {(3, 3), (2, 2), (3, 2), (2, 3), (1, 2), (1, 3), (2, 1), (3, 1)}
        self.assert_solutions(lambda: (int(x), int(y)), ground_truth)

    def test_simple_example_2(self):
        self.engine(bits=10)

        x = satx.integer()
        y = satx.integer()

        assert x ** 2 + y ** 2 == 100

        self.assert_solutions(lambda: (int(x), int(y)), {(0, 10), (10, 0), (6, 8), (8, 6)})

    def test_diophantine_equation_1(self):
        # http://www.artofproblemsolving.com/Forum/viewtopic.php?t=66245

        self.engine(bits=16)

        _2 = satx.constant(2)
        n = satx.integer()
        x = satx.integer()

        assert _2 ** n - 7 == x ** 2

        self.assert_solutions(lambda: (int(n), int(x)), {(5, 5), (4, 3), (7, 11), (3, 1), (15, 181)})

    def test_diophantine_equation_2(self):
        # http://www.artofproblemsolving.com/Forum/viewtopic.php?t=103239

        self.engine(bits=16)

        _2 = satx.constant(2)

        n = satx.integer()
        k = satx.integer()

        assert n ** 3 + 10 == _2 ** k + 5 * n

        self.assert_solutions(lambda: (int(n), int(k)), {(2, 3)})

    def test_diophantine_equation_3(self):
        # http://www.artofproblemsolving.com/Forum/viewtopic.php?t=103239

        self.engine(bits=16)

        a = satx.integer()
        b = satx.integer()

        assert a ** 2 == b ** 3 + 1

        self.assert_solutions(lambda: (int(a), int(b)), {(3, 2), (1, 0)})

    def test_diophantine_equation_4(self):
        # http://www.artofproblemsolving.com/Forum/viewtopic.php?t=31947

        self.engine(bits=16)

        _2 = satx.constant(2)
        _3 = satx.constant(3)

        x = satx.integer()
        y = satx.integer()

        assert _3 ** x == y * _2 ** x + 1

        self.assert_solutions(lambda: (int(x), int(y)), {(2, 2), (4, 5), (1, 1), (0, 0)})

    def test_integer_factorization(self):
        """
        In number theory, integer factorization is the decomposition of a composite number into a product of smaller
        integers. If these factors are further restricted to prime numbers, the process is called prime factorization.
        """

        rsa = 3007

        self.engine(bits=rsa.bit_length())

        p = satx.integer()
        q = satx.integer()

        assert p * q == rsa

        self.assert_solutions(lambda: (int(p), int(q)), {(3007, 1), (31, 97), (1, 3007), (97, 31)})

    def test_palindromic_numbers(self):

        self.engine(bits=10)

        x = satx.integer()

        # without "copy" for inplace reverse of bits
        assert x == x.reverse(copy=True)

        ground_truth = {0, 132, 72, 258, 513, 48, 204, 771, 120, 390, 561, 180, 330, 645, 585, 306, 462, 693, 633, 819,
                        252, 843, 438, 717, 378, 903, 975, 510, 765, 891, 951, 1023}

        def solution():
            s = int(x)
            if self.verbose:
                print(s, ''.join('1' if d else '0' for d in x.binary))
            return s

        self.assert_solutions(solution, ground_truth)

    def test_xor_problem(self):
        """
        The XOr, or “exclusive or”, problem is a classic problem in ANN research. It is the problem of using a neural
        network to predict the outputs of XOr logic gates given two binary inputs. An XOr function should return a
        true value if the two inputs are not equal and a false value if they are equal.
        """

        x = [[0, 0], [0, 1], [1, 0], [1, 1]]
        y = [0, 1, 1, 0]

        n, m = len(x), len(x[0])

        self.engine(bits=10)

        w = satx.matrix(dimensions=(n, m))
        b = satx.vector(size=n)

        for i in range(n):
            assert y[i] == satx.dot(x[i], w[i]) + b[i]

        self.assert_satisfiable()
        for i in range(n):
            self.assertEqual(int(satx.dot(x[i], w[i]) + b[i]), y[i])

    def test_absolute_values(self):
        self.engine(bits=4)

        x = satx.integer()
        y = satx.integer()

        assert abs(x - y) == 1

        assert x != satx.oo()
        assert y != satx.oo()

        self.assert_solutions(lambda: (int(x), int(y)), {
            (0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2), (3, 4), (4, 3), (4, 5), (5, 4), (5, 6), (6, 5), (6, 7),
            (7, 6), (7, 8), (8, 7), (8, 9), (9, 8), (9, 10), (10, 9), (10, 11), (11, 10), (11, 12), (12, 11), (12, 13),
            (13, 12), (13, 14), (14, 13)
        })

    def test_fermat_integer_factorization(self):
        # Note: when there is a negative number in the model, increment the bits by 1.

        rsa = 3007

        self.engine(bits=rsa.bit_length() + 1)

        p = satx.integer()
        q = satx.integer()

        assert p ** 2 - q ** 2 == rsa
        assert q < p

        # the factors are (p + q) * (p - q)
        # if it's not feasible then it's a prime

        self.assert_solutions(lambda: (int(p), int(q)), {(64, 33)})

    def test_diophantine_equation_5(self):
        n = 32

        self.engine(bits=n.bit_length())

        _3 = satx.constant(3)
        n = satx.integer()
        x = satx.integer()
        c = satx.integer()

        assert x ** 2 + c == _3 ** n
        assert x > 1
        assert c > 1
        assert n > 1

        ground_truth = {(3, 5, 2), (2, 2, 5), (3, 3, 18), (3, 2, 23), (3, 4, 11)}
        self.assert_solutions(lambda: (int(n), int(x), int(c)), ground_truth)

    def test_tensors_1(self):
        """
        Tensors object are the most advanced concept behind SAT-X, integers are tensors, work like integers,
        but their bits act like a multidimensional matrix of lambda functions.

        Note: [[*]] for access to lambda (bit) functions.
        """
        self.engine(bits=10)

        x = satx.tensor(dimensions=(4,))
        y = satx.tensor(dimensions=(2, 2))

        assert x + y == 10
        assert x[[0]](0, 1) == 1
        assert y[[0, 0]](0, 1) == 1

        def solution():
            if self.verbose:
                print(x, y, x.binary, y.binary)
            return int(x), int(y)

        self.assert_solutions(solution, {(1, 9), (9, 1), (5, 5), (7, 3), (3, 7)})

    def test_tensors_2(self):
        n = 2

        self.engine(bits=4)

        x = satx.tensor(dimensions=(n, n))
        a = satx.integer()
        b = satx.integer()

        assert sum(x[[i, j]](a ** 2 - b ** 3, a ** 3 - b ** 2) for i in range(n) for j in range(n)) == 0

        def solution():
            if self.verbose:
                print(a, b)
                print(np.vectorize(int)(x.binary))
                print()
            return int(x), int(a), int(b)

        ground_truth = {(0, 0, 0), (11, 1, 1), (7, 1, 1), (2, 1, 1), (8, 0, 0), (9, 0, 0), (15, 0, 0), (11, 0, 0),
                        (5, 1, 1), (1, 0, 0), (14, 0, 0), (5, 0, 0), (3, 0, 0), (7, 0, 0), (4, 0, 0), (13, 0, 0),
                        (2, 0, 0), (6, 0, 0), (12, 0, 0), (10, 0, 0), (0, 1, 1), (1, 1, 1), (8, 1, 1), (3, 1, 1),
                        (10, 1, 1), (9, 1, 1), (15, 1, 1), (13, 1, 1), (12, 1, 1), (14, 1, 1), (6, 1, 1), (4, 1, 1)}
        self.assert_solutions(solution, ground_truth)

    def test_integer_factorization_with_tensors(self):
        rsa = 3007

        self.engine(bits=rsa.bit_length())

        p = satx.tensor(dimensions=(satx.bits()))
        q = satx.tensor(dimensions=(satx.bits()))

        assert p * q == rsa
        assert p[[0]](0, 1) == 1
        assert q[[0]](0, 1) == 1
        assert sum(p[[i]](0, 1) for i in range(satx.bits() // 2 + 1, satx.bits())) == 0
        assert sum(q[[i]](0, 1) for i in range(satx.bits() // 2, satx.bits())) == 0

        # if it's not satisfiable then it's prime
        self.assert_solutions(lambda: (int(p), int(q)), {(97, 31)})

    def test_random_subset_sum_problem(self):
        """
        In this problem, there is a given set with some integer elements. And another some value is also provided,
        we have to find a subset of the given set whose sum is the same as the given sum value.

        https://en.wikipedia.org/wiki/Subset_sum_problem
        """
        universe = np.random.randint(1, 2 ** 16, size=100)
        t = np.random.randint(min(universe), sum(universe))

        self.engine(bits=t.bit_length())

        bits, subset = satx.subsets(universe)

        assert sum(subset) == t

        self.assert_satisfiable()
        sub = [universe[i] for i in range(len(universe)) if bits.binary[i]]
        if self.verbose:
            print(sub)
        self.assertEqual(t, sum(sub))

    def test_random_subset_sum_problem_with_tensors(self):
        universe = np.random.randint(1, 2 ** 16, size=100)
        t = np.random.randint(min(universe), sum(universe))

        self.engine(bits=t.bit_length())

        x = satx.tensor(dimensions=(len(universe),))

        assert sum(x[[i]](0, universe[i]) for i in range(len(universe))) == t

        self.assert_satisfiable()
        sub = [universe[i] for i in range(len(universe)) if x.binary[i]]
        if self.verbose:
            print(sub)
        self.assertEqual(t, sum(sub))

    def test_random_multiset_reconstruction(self):
        """
        Given a sorted multiset, their differences and one tip (an element and position for only one arbitrary
        element), is possible recovery the original multiset?
        """

        def generator(n, max_val):
            return sorted([random.randint(1, max_val) for _ in range(n)])

        def differences(lst):
            return [abs(lst[i] - lst[i - 1]) for i in range(1, len(lst))]

        for n in range(1, 10):
            m = random.randint(1, n ** 2)
            original = generator(n, m)
            diffs = differences(original)

            # only one tip
            ith = random.choice(range(n))
            tip = original[ith]

            # Empirical bits necessarily to solve the problem.
            self.engine(bits=sum(diffs).bit_length() + 4)

            # Declare an n-vector of integer variables to store the solution.
            x = satx.vector(size=n)

            # The tip is on x at index ith
            assert tip == satx.index(ith, x)

            # The i-th element of the instance is the absolute difference of two consecutive elements
            for i in range(n - 1):
                assert x[i] <= x[i + 1]
                assert satx.index(i, diffs) == x[i + 1] - x[i]

            # Solve the problem for only one solution
            # Turbo parameter is a destructive simplification
            self.assert_satisfiable()
            self.assertEqual(differences(x), diffs)
            if self.verbose:
                print(differences(x))
            self.assertEqual(len(set(map(int, x)).intersection(set(original))), len(set(original)))

    def test_random_permutation_reconstruction(self):
        """
        https://arxiv.org/pdf/1410.6396.pdf
        """

        def differences(lst):
            return [abs(lst[i] - lst[i - 1]) for i in range(1, len(lst))]

        for n in range(1, 30):
            perm = list(range(1, n + 1))
            random.shuffle(perm)
            perm = tuple(perm)
            diffs = differences(perm)

            self.engine(bits=n.bit_length() + 1, signed=True)
            x = satx.vector(size=n)
            satx.all_different(x)
            satx.apply_single(x, lambda a: 1 <= a <= n)
            for i in range(n - 1):
                assert diffs[i] == satx.one_of([x[i + 1] - x[i], x[i] - x[i + 1]])

            solutions = set()
            while self.satisfy():
                solutions.add(tuple(int(xx) for xx in x))
                if self.verbose:
                    print(perm, differences(x), x)
            if self.verbose:
                print()
            self.assertTrue(all(differences(s) == diffs for s in solutions))
            self.assertTrue(any(s == perm for s in solutions))

    def test_diophantine_equation_6(self):
        self.engine(bits=10)

        x = satx.integer()
        y = satx.integer()

        assert x ** 3 - x + 1 == y ** 2

        assert x != 0
        assert y != 0

        self.assert_solutions(lambda: (int(x), int(y)), {(1, 1), (5, 11), (3, 5)})

    def test_satisfiability(self):
        """
        Study of boolean functions generally is concerned with the set of truth assignments(assignments of 0 or 1 to
        each of the variables) that make the function true.

        https://en.wikipedia.org/wiki/Boolean_satisfiability_problem
        """
        n, m = 10, 24
        sat = [[9, -5, 10, -6, 3],
               [6, 8],
               [8, 4],
               [-10, 5],
               [-9, 8],
               [-9, -3],
               [-2, 5],
               [6, 4],
               [-2, -1],
               [7, -2],
               [-9, 4],
               [-1, -10],
               [-3, 4],
               [7, 5],
               [6, -3],
               [-10, 7],
               [-1, 7],
               [8, -3],
               [-2, -10],
               [-1, 5],
               [-7, 1, 9, -6, 3],
               [-9, 6],
               [-8, 10, -5, -4, 2],
               [-4, -7, 1, -8, 2]]

        self.engine(bits=1)
        x = satx.tensor(dimensions=(n,))
        assert functools.reduce(operator.iand,
                                (functools.reduce(operator.ior, (x[[abs(lit) - 1]](lit < 0, lit > 0) for lit in cls))
                                 for cls in sat)) == 1

        def solution():
            if self.verbose:
                print(' '.join(map(str, [(i + 1) if b else -(i + 1) for i, b in enumerate(x.binary)])) + ' 0')
            return int(x)

        self.assert_solutions(solution, {254, 218, 506})

    def test_k_clique(self):
        """
        Input: Graph G positive integer k. Property: G has a set of mutually adjacent nodes.

        https://en.wikipedia.org/wiki/Clique_problem
        """
        # Ths bits of the clique to search
        k = 3

        # Get the graph, and the dimension for the graph
        n, matrix = 5, [(1, 0), (0, 2), (1, 4), (2, 1), (4, 2), (3, 2)]

        # Ensure the problem can be represented
        self.engine(bits=k.bit_length())

        # Declare an integer of n-bits
        bits = satx.integer(bits=n)

        # The bits integer have "bits"-active bits, i.e, the clique has "bits"-elements
        assert sum(satx.switch(bits, i) for i in range(n)) == k

        # This entangles all elements that are joined together
        for i in range(n - 1):
            for j in range(i + 1, n):
                if (i, j) not in matrix and (j, i) not in matrix:
                    assert satx.switch(bits, i) + satx.switch(bits, j) <= 1

        def solution():
            if self.verbose:
                print(' '.join([str(i) for i in range(n) if not bits.binary[i]]))
            return int(bits)

        self.assert_solutions(solution, {24, 9})

    def test_vertex_cover(self):
        """
        In the mathematical discipline of graph theory, a vertex cover (sometimes node cover) of a graph is a set of
        vertices that includes at least one endpoint of every edge of the graph. The problem of finding a minimum
        vertex cover is a classical optimization problem in computer science and is a typical example of an NP-hard
        optimization problem that has an approximation algorithm. Its decision version, the vertex cover problem,
        was one of Karp's 21 NP-complete problems and is therefore a classical NP-complete problem in computational
        complexity theory. Furthermore, the vertex cover problem is fixed-parameter tractable and a central problem
        in parameterized complexity theory.

        https://en.wikipedia.org/wiki/Vertex_cover
        """
        # Get the graph and dimension, and the bits of the cover.
        n, graph, vertex, k = 5, [(1, 0), (0, 2), (1, 4), (2, 1), (4, 2), (3, 2)], [0, 1, 2, 3, 4], 3

        # Ensure the problem can be represented
        self.engine(bits=n.bit_length() + 1)

        # An integer with n-bits to store the indexes for the cover
        index = satx.integer(bits=n)

        # This entangled the all possible covers
        for i, j in graph:
            assert satx.switch(index, vertex.index(i), neg=True) + satx.switch(index, vertex.index(j), neg=True) >= 1

        # Ensure the cover has bits k
        assert sum(satx.switch(index, vertex.index(i), neg=True) for i in vertex) == k

        def solution():
            if self.verbose:
                print(' '.join([str(vertex[i]) for i in range(n) if index.binary[i]]))
            return int(index)

        self.assert_solutions(solution, {7, 22, 21, 14})

    def test_multidimensional_latin_squares(self) -> None:
        """
        In combinatorics and in experimental design, a Latin square is an n × n array filled with n different symbols,
        each occurring exactly once in each row and exactly once in each column.

        https://en.wikipedia.org/wiki/Latin_square
        """
        shape = (6, 6, 6)
        n = max(shape)

        self.engine(bits=n.bit_length())

        Y_flat = satx.vector(size=np.prod(shape, dtype=int).item())

        satx.apply_single(Y_flat, lambda k: k < n)

        Y = np.reshape(Y_flat, newshape=shape)

        for axis in range(len(shape)):
            all_dimensions: list = [range(d) for d in shape]
            all_dimensions[axis] = [slice(None, None, None)]
            for idx in itertools.product(*all_dimensions):
                satx.all_different(Y[tuple(idx)])

        self.assert_satisfiable()
        Y_int = np.vectorize(int)(Y).reshape(shape)

        if self.verbose:
            print(Y_int)

        def assert_all_different(arr):
            for i in range(len(arr)):
                for j in range(i + 1, len(arr)):
                    self.assertNotEqual(arr[i], arr[j])

        for axis in range(len(shape)):
            all_dimensions = [range(d) for d in shape]
            all_dimensions[axis] = [slice(None, None, None)]
            for idx in itertools.product(*all_dimensions):
                assert_all_different(Y_int[tuple(idx)])

    def test_magic_square(self):
        """
        In recreational mathematics and combinatorial design, a magic square is a $n \\times n$ square grid (where n
        is the number of cells on each side) filled with distinct positive integers in the range
        $1,2,\\mathellipsis,n^2$ such that each cell contains a different integer and the sum of the
        integers in each row, column and diagonal is equal.

        https://en.wikipedia.org/wiki/Magic_square
        """
        n = 3

        self.engine(bits=5)

        c = satx.integer()

        xs = satx.matrix(dimensions=(n, n))
        xs_flat = satx.flatten(xs)

        satx.apply_single(xs_flat, lambda x: x > 0)
        satx.all_different(xs_flat)

        for i in range(n):
            assert sum(xs[i][j] for j in range(n)) == c
        for j in range(n):
            assert sum(xs[i][j] for i in range(n)) == c

        assert sum(xs[i][i] for i in range(n)) == c
        assert sum(xs[i][n - 1 - i] for i in range(n)) == c

        # There are many solutions the solver might spit out. We can't hard code each one.
        # Just check the first and call it a day.

        self.assert_satisfiable()
        if self.verbose:
            print(c)
        xs = np.vectorize(int)(xs)
        if self.verbose:
            print(xs)

        for i in range(n):
            self.assertEqual(sum(xs[i][j] for j in range(n)), c)
        for j in range(n):
            self.assertEqual(sum(xs[i][j] for i in range(n)), c)

        self.assertEqual(sum(xs[i][i] for i in range(n)), c)
        self.assertEqual(sum(xs[i][n - 1 - i] for i in range(n)), c)

    def test_random_schur_triples_problem(self):
        """
        Input: list of 3N distinct positive integers

        Question: Is there a partition of the list into N triples $(a_i, b_i, c_i)$ such that $a_i + b_i = c_i$.

        The condition that all numbers must be distinct makes the problem very interesting and McDiarmid
        calls it a surprisingly troublesome.

        https://cstheory.stackexchange.com/questions/16253/list-of-strongly-np-hard-problems-with-numerical-data
        """
        bits = 7
        size = 3 * 10
        triplets = []
        while len(triplets) < size:
            a = np.random.randint(1, 2 ** bits)
            b = np.random.randint(1, 2 ** bits)
            if a != b and a not in triplets and b not in triplets and a + b not in triplets:
                triplets += [a, b, a + b]
        triplets.sort()
        if self.verbose:
            print(triplets)
        self.engine(bits=max(triplets).bit_length())
        xs, ys = satx.permutations(triplets, size)

        for i in range(0, size, 3):
            assert ys[i] + ys[i + 1] == ys[i + 2]

        # Checking that there are no more solutions is very difficult.
        self.assert_satisfiable()
        for i in range(0, size, 3):
            if self.verbose:
                print('{} == {} + {}'.format(ys[i + 2], ys[i], ys[i + 1]))
            self.assertEqual(int(ys[i + 2]), int(ys[i]) + int(ys[i + 1]))

    def test_hamiltonian_cycle(self):
        """
        In the mathematical field of graph theory, a Hamiltonian path (or traceable path) is a path in an undirected
        or directed graph that visits each vertex exactly once. A Hamiltonian cycle (or Hamiltonian circuit) is a
        Hamiltonian path that is a cycle. Determining whether such paths and cycles exist in graphs is the
        Hamiltonian path problem, which is NP-complete.

        https://en.wikipedia.org/wiki/Hamiltonian_path
        """
        n = 10

        # M = np.random.randint(0, 2, size=(n, n))
        #
        # Be careful when choosing a random graph. It might not be Hamiltonian.

        M = np.array([[0, 0, 1, 0, 1, 0, 1, 1, 0, 0],
                      [1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
                      [1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
                      [0, 0, 1, 1, 0, 1, 1, 1, 0, 1],
                      [0, 1, 0, 1, 1, 0, 0, 0, 1, 0],
                      [1, 0, 1, 0, 1, 0, 0, 0, 1, 0],
                      [0, 1, 1, 0, 1, 1, 1, 0, 0, 0],
                      [0, 0, 1, 0, 1, 0, 1, 1, 1, 0],
                      [1, 0, 0, 1, 0, 1, 0, 1, 0, 0],
                      [0, 1, 1, 0, 0, 1, 1, 1, 1, 0]])

        self.engine(bits=(n ** 2).bit_length())
        ids, elements = satx.matrix_permutation((1 - M).flatten(), n)

        assert sum(elements) == 0

        self.assert_satisfiable()
        perm = [int(i) for i in ids]
        if self.verbose:
            print(ids, elements)
            print(perm)
        for i in range(len(perm) - 1):
            self.assertEqual(M[perm[i], perm[i + 1]], 1)
        self.assertEqual(M[perm[-1], perm[0]], 1)

    def test_non_hamiltonian_graph(self):
        # The Petersen graph does not contain a Hamiltonian cycle.

        n = 10
        M = np.array([[0, 0, 1, 1, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 1, 1, 0, 1, 0, 0, 0],
                      [1, 0, 0, 0, 1, 0, 0, 1, 0, 0],
                      [1, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                      [0, 1, 1, 0, 0, 0, 0, 0, 0, 1],
                      [1, 0, 0, 0, 0, 0, 1, 0, 0, 1],
                      [0, 1, 0, 0, 0, 1, 0, 1, 0, 0],
                      [0, 0, 1, 0, 0, 0, 1, 0, 1, 0],
                      [0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
                      [0, 0, 0, 0, 1, 1, 0, 0, 1, 0]])
        self.engine(bits=(n ** 2).bit_length())
        ids, elements = satx.matrix_permutation((1 - M).flatten(), n)

        assert sum(elements) == 0

        self.assertFalse(self.satisfy())

    def test_bin_packing_problem(self):
        """
        In the bin packing problem, items of different volumes must be packed into a finite number of bins or
        containers each of a fixed given volume in a way that minimizes the number of bins used. In computational
        complexity theory, it is a combinatorial NP-hard problem. The decision problem (deciding if items will fit
        into a specified number of bins) is NP-complete.

        https://en.wikipedia.org/wiki/Bin_packing_problem
        """
        capacity = 50
        # size = 50
        # elements = sorted([np.random.randint(1, capacity // 2 - 1) for _ in range(size)], reverse=True)

        elements = [23, 22, 21, 20, 20, 19, 19, 19, 19, 17, 17, 17, 16, 16, 15, 15, 15, 14, 14, 14, 13, 12, 11, 11, 11,
                    11, 10, 10, 10, 9, 9, 8, 7, 7, 7, 6, 6, 5, 5, 5, 4, 4, 4, 4, 3, 2, 2, 2, 1, 1]
        ground_truth = 12

        if self.verbose:
            print(capacity)
            print(elements)

        bins = int(np.ceil(sum(elements) / capacity))
        while True:
            self.engine(bits=capacity.bit_length() + 1)
            slots = satx.vector(bits=len(elements), size=bins)
            for i in range(len(elements)):
                assert sum(satx.switch(slot, i) for slot in slots) == 1
            for slot in slots:
                assert sum(satx.switch(slot, i) * elements[i] for i in range(len(elements))) <= capacity
            if self.satisfy():
                items_in_bins = [[item for i, item in enumerate(elements) if not slot.binary[i]] for slot in slots]
                if self.verbose:
                    for slot in slots:
                        print(''.join(['_' if boolean else '#' for boolean in slot.binary]))
                    print()
                    print(items_in_bins)
                self.assertEqual(len(items_in_bins), ground_truth)
                self.assertEqual(sorted(itertools.chain.from_iterable(items_in_bins), reverse=True), elements)
                self.assertTrue(all(sum(bin) <= capacity for bin in items_in_bins))
                break
            else:
                # warning: with this example the solution is found on the first try
                bins += 1
                if self.verbose:
                    print(bins)
                self.assertLessEqual(bins, ground_truth)

    def test_random_zero_one_integer_programming(self):
        """
        Zero-one integer programming (which can also be written as 0-1 integer programming) is a mathematical method
        of using a series of binary, yes (1) and no (0) answers to arrive at a solution when there are two mutually
        exclusive options.

        https://en.wikipedia.org/wiki/Integer_programming
        """
        n, m = 10, 5
        cc = np.random.randint(0, 1000, size=(n, m))
        d = np.dot(cc, np.random.randint(0, 2, size=(m,)))

        self.engine(bits=int(np.sum(cc)).bit_length())
        xs = satx.vector(size=m)
        satx.all_binaries(xs)
        assert np.all(np.dot(cc, xs) == d)

        self.assert_satisfiable()
        self.assertTrue(np.all(np.dot(cc, xs) == d))

    def test_queens_problem(self):
        """
        The n-Queens Completion problem is a variant, dating to 1850, in which some queens are already placed and
        the solver is asked to place the rest, if possible. ... The n-Queens problem is to place n chess queens on
        an n by n chessboard so that no two queens are on the same row, column or diagonal.

        https://www.ijcai.org/Proceedings/2018/0794.pdf
        """

        def setup(board_size, pieces, seed):
            """
            http://www.csplib.org/Problems/prob079/data/queens-gen-fast.py.html
            """
            rng = random.Random(seed)

            d1 = [0 for _ in range(2 * board_size - 1)]
            d2 = [0 for _ in range(2 * board_size - 1)]

            valid_rows = [i for i in range(board_size)]
            valid_cols = [j for j in range(board_size)]

            def no_attack(r, c):
                return d1[r + c] == 0 and d2[r - c + board_size - 1] == 0

            pc = []
            queens_left = board_size

            for attempt in range(board_size * board_size):
                i = rng.randrange(queens_left)
                j = rng.randrange(queens_left)
                r = valid_rows[i]
                c = valid_cols[j]
                if no_attack(r, c):
                    pc.append([r, c])
                    d1[r + c] = 1
                    d2[r - c + board_size - 1] = 1
                    valid_rows[i] = valid_rows[queens_left - 1]
                    valid_cols[j] = valid_cols[queens_left - 1]
                    queens_left -= 1
                    if len(pc) == pieces:
                        return [[x + 1, y + 1] for x, y in pc]

        def show_setup(pc):
            table = ''
            for i in range(1, n + 1):
                table += ''
                for j in range(1, n + 1):
                    if [i, j] not in pc:
                        table += '. '
                    else:
                        table += 'Q '
                table += '\n'
            print(table)

        def show_solution(qs):
            for i in range(n):
                print(''.join(['Q ' if qs[i] == j else '. ' for j in range(n)]))
            print('')

        n, m, seed = 30, 15, 0
        placed_queens = setup(n, m, seed)

        if self.verbose:
            show_setup(placed_queens)

        self.engine(bits=n.bit_length() + 1)
        qs = satx.vector(size=n)
        for (x, y) in placed_queens:
            assert qs[x - 1] == y - 1
        satx.apply_single(qs, lambda x: x < n)
        satx.apply_dual(qs, lambda x, y: x != y)
        satx.apply_dual([qs[i] + i for i in range(n)], lambda x, y: x != y)
        satx.apply_dual([qs[i] - i for i in range(n)], lambda x, y: x != y)

        self.assert_satisfiable()

        if self.verbose:
            show_solution(qs)

        board = np.zeros((n, n), dtype=int)
        for x, y in enumerate(qs):
            y = int(y)
            board[x, y] = 1
        self.assertEqual(np.sum(board), n)
        self.assertTrue(np.all(board.sum(axis=0) <= 1))
        self.assertTrue(np.all(board.sum(axis=1) <= 1))
        for k in range(-7, 7):
            self.assertLessEqual(np.sum(np.diag(board, k)), 1)
        board = np.rot90(board)
        for k in range(-7, 7):
            self.assertLessEqual(np.sum(np.diag(board, k)), 1)

    def test_random_partition_problem(self):
        size = 20

        split = random.randint(1, size - 2)
        data1 = [random.randint(0, 1000) for _ in range(split)]
        data2 = [random.randint(0, 1000) for _ in range(size - split - 1)]

        extra = abs(sum(data1) - sum(data2))
        data = data1 + data2 + [extra]
        random.shuffle(data)

        if self.verbose:
            print(data)

        self.engine(bits=int(sum(data)).bit_length())

        T, sub, com = satx.subsets(data, complement=True)

        assert sum(sub) == sum(com)

        self.assert_satisfiable()
        sub_ = [data[i] for i in range(size) if T.binary[i]]
        com_ = [data[i] for i in range(size) if not T.binary[i]]
        if self.verbose:
            print(sub_, com_)
        self.assertCountEqual(sub_ + com_, data)
        self.assertEqual(sum(sub_), sum(com_))

    def test_random_sudoku(self):
        """
        Sudoku is a logic-based, combinatorial number-placement puzzle. The objective is to fill a 9×9 grid with
        digits so that each column, each row, and each of the nine 3×3 subgrids that compose the grid (also called
        "boxes", "blocks", or "regions") contain all the digits from 1 to 9. The puzzle setter provides a
        partially completed grid, which for a well-posed puzzle has a single solution.

        Completed games are always an example of a Latin square which include an additional constraint on the
        contents of individual regions. For example, the same single integer may not appear twice in the same row,
        column, or any of the nine 3×3 subregions of the 9×9 playing board.

        https://en.wikipedia.org/wiki/Sudoku
        """

        def expand_line(line):
            return line[0] + line[5:9].join([line[1:5] * (base - 1)] * base) + line[9:13]

        def show(board):
            line0 = expand_line('╔═══╤═══╦═══╗')
            line1 = expand_line('║ . │ . ║ . ║')
            line2 = expand_line('╟───┼───╫───╢')
            line3 = expand_line('╠═══╪═══╬═══╣')
            line4 = expand_line('╚═══╧═══╩═══╝')

            symbol = ' ' + string.printable.replace(' ', '')
            nums = [[''] + [symbol[n] for n in row] for row in board]
            print(line0)
            for r in range(1, side + 1):
                print("".join(n + s for n, s in zip(nums[r - 1], line1.split('.'))))
                print([line2, line3, line4][(r % side == 0) + (r % base == 0)])

        def generate(base):
            # pattern for a baseline valid solution
            def pattern(r, c):
                return (base * (r % base) + r // base + c) % side

            # randomize rows, columns and numbers (of valid base pattern)

            def shuffle(s):
                return random.sample(s, len(s))

            rBase = range(base)
            rows = [g * base + r for g in shuffle(rBase) for r in shuffle(rBase)]
            cols = [g * base + c for g in shuffle(rBase) for c in shuffle(rBase)]
            nums = shuffle(range(1, base * base + 1))

            # produce board using randomized baseline pattern
            board = [[nums[pattern(r, c)] for c in cols] for r in rows]

            squares = side * side
            empties = (squares * 3) // 4
            for p in map(int, random.sample(range(squares), empties)):
                board[p // side][p % side] = 0

            return board

        base = 4
        side = base * base

        puzzle = np.asarray(generate(base))
        if self.verbose:
            show(puzzle)
            print()

        self.engine(bits=side.bit_length())

        board = np.asarray(satx.matrix(dimensions=(side, side)))
        satx.apply_single(board.flatten(), lambda x: 1 <= x <= side)

        for i in range(side):
            for j in range(side):
                if puzzle[i][j]:
                    assert board[i][j] == puzzle[i][j]

        for c, r in zip(board, board.T):
            satx.all_different(c)
            satx.all_different(r)

        for i in range(base):
            for j in range(base):
                satx.all_different(board[i * base:(i + 1) * base, j * base:(j + 1) * base].flatten())

        self.assert_satisfiable()
        board_int = np.vectorize(int)(board)

        if self.verbose:
            show(board_int)
            print()

        def assert_all_different(arr):
            for i in range(len(arr)):
                for j in range(i + 1, len(arr)):
                    self.assertNotEqual(arr[i], arr[j])

        for i in range(side):
            for j in range(side):
                if puzzle[i][j] != 0:
                    self.assertEqual(board_int[i][j], puzzle[i][j])

        for row, col in zip(board_int, board_int.T):
            assert_all_different(row)
            assert_all_different(col)

        for i in range(base):
            for j in range(base):
                assert_all_different(board_int[i * base:(i + 1) * base, j * base:(j + 1) * base].flatten())

    def test_board_coloration(self):
        """
        All squares of a board of a specified size (specified numbers of rows and columns)
        must be colored with the minimum number of colors. The four corners of any rectangle inside
        the board must not be assigned the same color.
        """

        n, m = 5, 8

        opt = 1
        ground_truth = 3
        while True:
            self.engine(opt.bit_length())

            # x[i][j] is the color at row i and column j
            x = satx.matrix(dimensions=(n, m))

            # at least one corners of different color for any rectangle inside the board
            for i1, i2 in itertools.combinations(range(n), 2):
                for j1, j2 in itertools.combinations(range(m), 2):
                    corners = [x[i1][j1], x[i1][j2], x[i2][j1], x[i2][j2]]
                    # there exists a pair of corners that are not equal
                    assert satx.one_of(corners) != satx.one_of(corners)

            satx.apply_single(satx.flatten(x), lambda t: t < opt)

            if self.satisfy():
                self.assertEqual(opt, ground_truth)
                x_int = np.vectorize(int)(x).reshape((n, m))
                if self.verbose:
                    print(x_int)

                for i1, i2 in itertools.combinations(range(n), 2):
                    for j1, j2 in itertools.combinations(range(m), 2):
                        corners = [x_int[i1][j1], x_int[i1][j2], x_int[i2][j1], x_int[i2][j2]]
                        self.assertTrue(len(set(corners)) > 1)
                break
            else:
                opt += 1
                if self.verbose:
                    print(opt)
                self.assertLessEqual(opt, ground_truth)

    def test_change_making(self):
        """
        See https://en.wikipedia.org/wiki/Change-making_problem
        """

        k = 13
        coins = [1, 5, 10, 20, 50, 100, 200]

        opt = k
        x_int = None
        while True:
            self.engine(sum(coins).bit_length())

            x = satx.vector(size=len(coins))

            assert satx.dot(x, coins) == k

            assert sum(x) < opt

            if self.satisfy():
                self.assertLess(sum(x), opt)
                opt = sum(x)
                self.assertGreaterEqual(opt, 4)
                x_int = [int(xx) for xx in x]
                if self.verbose:
                    print(opt, x_int)
            else:
                break

        self.assertEqual(opt, 4)
        self.assertEqual(x_int, [3, 0, 1, 0, 0, 0, 0])

    def test_dimes(self):
        """
        "Dad wants one-cent, two-cent, three-cent, five-cent, and ten-cent stamps.
        He said to get four each of two sorts and three each of the others, but I've
        forgotten which. He gave me exactly enough to buy them; just these dimes."
        How many stamps of each type does Dad want? A dime is worth ten cents.
        -- J.A.H. Hunter
        """

        self.engine(bits=10)

        # x is the number of dimes
        x = satx.integer()

        # s[i] is the number of stamps of value 1, 2, 3, 5 and 10 according to i
        s = satx.vector(size=5)

        satx.apply_single(s, lambda t: t.is_in([3, 4]))

        # 26 is a safe upper bound
        assert x <= 26

        assert satx.dot(s, [1, 2, 3, 5, 10]) == x * 10

        ground_truth = {(3, 4, 3, 4, 3, 7), (3, 4, 3, 4, 4, 8)}

        self.assert_solutions(lambda: tuple(int(ss) for ss in s) + (int(x),), ground_truth)

    def test_dinner(self):
        """
        My son came to me the other day and said, "Dad, I need help with a math problem."
        The problem went like this:
        - We're going out to dinner taking 1-6 grandparents, 1-10 parents and/or 1-40 children
        - Grandparents cost $3 for dinner, parents $2 and children $0.50
        - There must be 20 total people at dinner, and it must cost $20
        How many grandparents, parents and children are going to dinner?
        """

        self.engine(bits=10)

        # g is the number of grandparents
        g = satx.integer()
        # c is the number of children
        c = satx.integer()
        # p is the number of parents
        p = satx.integer()

        assert 1 <= g <= 7
        assert 1 <= p <= 11
        assert 1 <= c <= 41

        assert g * 6 + p * 2 + c * 1 == 40
        assert g + p + c == 20

        self.assert_solutions(lambda: (int(g), int(p), int(c)), {(2, 10, 8), (3, 5, 12)})

    def test_grocery(self):
        """
        See "Constraint Programming in Oz. A Tutorial" by C. Schulte and G. Smolka, 2001

        A kid goes into a grocery store and buys four items.
        The cashier charges $7.11, the kid pays and is about to leave when the cashier calls the kid back, and says
        ``Hold on, I multiplied the four items instead of adding them;
          I'll try again;
          Hah, with adding them the price still comes to $7.1``.
        What were the prices of the four items?
        """

        # 711 * 100 * 100 * 100 -> 30 bits
        self.engine(bits=30)

        # x[i] is the price (multiplied by 100) of the ith item
        x = satx.vector(size=4)

        satx.apply_single(x, lambda t: t < 711)

        # adding the prices of items corresponds to 711 cents
        assert sum(x) == 711

        # multiplying the prices of items corresponds to 711 cents (times 1000000)
        assert x[0] * x[1] * x[2] * x[3] == 711 * 100 * 100 * 100

        # There are many solutions. There is no need to list them.
        self.assert_satisfiable()
        x_int = [int(xx) for xx in x]
        if self.verbose:
            print(x_int)

        self.assertTrue(all(xx < 711 for xx in x_int))
        self.assertEqual(sum(x_int), 711)
        self.assertEqual(x_int[0] * x_int[1] * x_int[2] * x_int[3], 711 * 100 * 100 * 100)

    def test_safe_cracking(self):
        """
        From the Oz Primer. See http://www.comp.nus.edu.sg/~henz/projects/puzzles/digits/index.html

        The code of Professor Smart's safe is a sequence of 9 distinct
        nonzero digits d1 .. d9 such that the following equations and
        inequalities are satisfied:

               d4 - d6   =   d7
          d1 * d2 * d3   =   d8 + d9
          d2 + d3 + d6   <   d8
                    d9   <   d8
          d1 <> 1, d2 <> 2, ..., d9 <> 9

        Can you find the correct combination?
        """
        self.engine(bits=10)

        # x[i] is the i(+1)th digit
        x = satx.vector(size=9)

        satx.apply_single(x, lambda t: t.is_in(range(10)))

        satx.all_different(x)

        assert x[3] - x[5] == x[6]
        assert x[0] * x[1] * x[2] == x[7] + x[8]
        assert x[1] + x[2] + x[5] < x[7]
        assert x[8] < x[7]

        # There are many solutions the solver might spit out. We can't hard code each one.
        # Just check the first and call it a day.
        self.assert_satisfiable()
        x_int = np.vectorize(int)(x)
        if self.verbose:
            print(x_int)
        self.assertTrue(all(0 <= xx < 10 for xx in x_int))
        self.assertEqual(len(set(x_int)), len(x_int))
        self.assertEqual(x_int[3] - x_int[5], x_int[6])
        self.assertEqual(x_int[0] * x_int[1] * x_int[2], x_int[7] + x_int[8])
        self.assertLess(x_int[1] + x_int[2] + x_int[5], x_int[7])
        self.assertLess(x_int[8], x_int[7])

    def test_send_more(self):
        """
        See https://en.wikipedia.org/wiki/Verbal_arithmetic
        """

        self.engine(bits=16)

        # letters[i] is the digit of the ith letter involved in the equation
        s, e, n, d, m, o, r, y = letters = satx.vector(size=8)

        satx.apply_single(letters, lambda t: t < 10)

        # letters are given different values
        satx.all_different(letters)

        # words cannot start with 0
        assert s > 0
        assert m > 0

        # respecting the mathematical equation
        assert satx.dot([s, e, n, d], [1000, 100, 10, 1]) + satx.dot([m, o, r, e], [1000, 100, 10, 1]) == \
               satx.dot([m, o, n, e, y], [10000, 1000, 100, 10, 1])

        self.assert_satisfiable()
        letters_int = [int(l) for l in letters]
        if self.verbose:
            print(letters_int)
        self.assertEqual(letters_int, [9, 5, 6, 7, 1, 0, 8, 2])

    def test_square(self):
        """
        See http://en.wikibooks.org/wiki/Puzzles/Arithmetical_puzzles/Digits_of_the_Square

        There is one four-digit whole number x, such that the last four digits of x^2
        are in fact the original number x. What is it?
        """

        self.engine(bits=30)

        # x is the number we look for
        x = satx.integer()

        # d[i] is the ith digit of x
        d = satx.vector(size=4)

        satx.apply_single(d, lambda t: t.is_in(range(10)))

        assert 1000 <= x < 10000
        assert satx.dot(d, [1000, 100, 10, 1]) == x
        assert (x * x) % 10000 == x

        self.assert_satisfiable()
        if self.verbose:
            print(x)
        self.assertEqual([int(x)] + [int(dd) for dd in d], [9376, 9, 3, 7, 6])


class SlowRegressionTests(SatXTestCase):
    """
    These regression tests may sometimes give SAT solvers a hard time or otherwise take long.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_exponential_diophantine_equation(self):
        self.engine(bits=16)

        x = satx.integer()
        y = satx.integer()
        z = satx.integer()

        deep = 4
        assert y <= deep
        assert x < 2 ** deep

        satx.apply_single([x, y, z], lambda t: t != 0)

        assert x ** y == z

        ground_truth = {(2, 4, 16), (1, 4, 1), (1, 2, 1), (4, 3, 64), (4, 2, 16), (4, 4, 256), (4, 1, 4), (8, 1, 8),
                        (8, 2, 64), (8, 4, 4096), (15, 1, 15), (1, 1, 1), (3, 1, 3), (3, 2, 9), (1, 3, 1), (6, 3, 216),
                        (2, 1, 2), (8, 3, 512), (9, 4, 6561), (9, 3, 729), (9, 1, 9), (9, 2, 81), (10, 2, 100),
                        (10, 4, 10000), (10, 3, 1000), (10, 1, 10), (2, 3, 8), (2, 2, 4), (3, 4, 81), (3, 3, 27),
                        (11, 1, 11), (11, 4, 14641), (11, 2, 121), (11, 3, 1331), (5, 3, 125), (5, 4, 625), (5, 1, 5),
                        (5, 2, 25), (7, 3, 343), (15, 3, 3375), (13, 3, 2197), (13, 4, 28561), (13, 1, 13),
                        (13, 2, 169), (14, 3, 2744), (14, 4, 38416), (14, 1, 14), (14, 2, 196), (15, 2, 225),
                        (15, 4, 50625), (7, 4, 2401), (7, 2, 49), (6, 4, 1296), (6, 2, 36), (6, 1, 6), (7, 1, 7),
                        (12, 4, 20736), (12, 3, 1728), (12, 2, 144), (12, 1, 12)}

        self.assert_solutions(lambda: (int(x), int(y), int(z)), ground_truth)

    def test_centroid(self, plot=False):
        # n = 15
        # np.random.seed(0)
        # data = np.random.randint(0, 10, size=(n, 2))

        data = [[5, 0], [3, 3], [7, 9], [3, 5], [2, 4], [7, 6], [8, 8],
                [1, 6], [7, 7], [8, 1], [5, 9], [8, 9], [4, 3], [0, 3], [5, 0]]

        opt = 1
        ground_truth = 73
        while True:
            self.engine(bits=10)

            x = satx.integer()
            y = satx.integer()

            assert sum(abs(xy[0] - x) + abs(xy[1] - y) for xy in data) < opt

            assert x != satx.oo()
            assert y != satx.oo()

            if self.satisfy():
                self.assertEqual((int(x), int(y)), (5, 5))
                self.assertEqual(opt, ground_truth)
                if plot:
                    import matplotlib.pyplot as plt  # type: ignore
                    a, b = zip(*data)
                    plt.plot(a, b, '.')
                    plt.plot(x, y, 'x')
                    plt.show()

                break
            else:
                if self.verbose:
                    print(opt)
                opt += 1
                self.assertLessEqual(opt, ground_truth)

    def test_travelling_salesman_problem(self, plot=False):
        """
        The travelling salesman problem asks the following question: "Given a list of cities and the distances
        between each pair of cities, what is the shortest possible route that visits each city and returns to the
        origin city?" It is an NP-hard problem in combinatorial optimization, important in operations research and
        theoretical computer science.

        https://en.wikipedia.org/wiki/Travelling_salesman_problem
        """
        n = 10

        # data = np.random.randint(0, 10, size=(n, 2))
        # Be careful when generating random instances. A solution might not be unique or might not exist.
        data = np.array([[9, 3], [4, 7], [8, 8], [8, 2], [7, 9], [7, 0], [4, 8], [5, 4], [3, 1], [3, 9]], dtype=int)
        ground_truth = 25

        distances = np.zeros(shape=(n, n))
        for i in range(n):
            for j in range(n):
                distances[i][j] = int(np.linalg.norm(data[i] - data[j]))

        x_int, y_int = None, None
        opt = 2 ** n - 1
        while True:
            self.engine(bits=int(sum(distances.flatten())).bit_length() + 1)
            x, y = satx.matrix_permutation(distances.flatten(), n)
            assert sum(y) < opt
            if self.satisfy():
                x_int, y_int = [int(x_item) for x_item in x], [int(y_item) for y_item in y]
                self.assertLess(sum(y_int), opt)
                self.assertGreaterEqual(sum(y_int), ground_truth)
                opt = sum(y_int)
                if self.verbose:
                    print(opt, x_int, y_int)
                if plot:
                    import matplotlib.pyplot as plt  # type: ignore
                    a, b = zip(*[data[i] for i in x_int + [x_int[0]]])
                    plt.plot(a, b, 'ro')
                    plt.plot(a, b, 'k-')
                    plt.show()
            else:
                break
        self.assertEqual(opt, ground_truth)
        y_from_x = [distances[x_int[i], x_int[i + 1]] for i in range(len(x_int) - 1)] + [distances[x_int[-1], x_int[0]]]
        self.assertEqual(y_int, y_from_x)
        self.assertEqual(set(x_int), set(range(n)))

    def test_factorial(self):
        self.engine(bits=22)  # be sure to give it enough bits

        x = satx.integer()

        assert satx.factorial(x) == math.factorial(10)

        # Don't bother asserting that there are no more solutions. It takes too long.
        self.assert_satisfiable()
        self.assertEqual(x, 10)

    def test_sigma(self):
        self.engine(bits=16)

        x = satx.integer()
        n = satx.integer()

        assert satx.sigma(lambda k: k ** 2, 1, n) == x

        ground_truth = {(0, 0), (385, 10), (140, 7), (14, 3), (91, 6), (204, 8), (55, 5), (30, 4),
                        (819, 13), (506, 11), (285, 9), (1015, 14), (650, 12), (1, 1), (5, 2)}

        self.assert_solutions(lambda: (int(x), int(n)), ground_truth)

        for x, n in ground_truth:
            self.assertEqual(x, sum(k ** 2 for k in range(1, n + 1)))

    def test_pi(self):
        self.engine(bits=32)

        x = satx.integer()
        n = satx.integer()

        assert satx.pi(lambda k: k ** 2, 1, n) == x
        assert 0 < x <= 2 ** math.log(satx.oo())  # limit the CNF overflow
        assert n > 0

        ground_truth = {(4, 2), (576, 4), (518400, 6), (14400, 5), (1, 1), (36, 3)}
        self.assert_solutions(lambda: (int(x), int(n)), ground_truth)

        for x, n in ground_truth:
            self.assertEqual(x, functools.reduce(operator.mul, (k ** 2 for k in range(1, n + 1))))

    def test_maximum_constrained_partition(self):
        # http://www.csc.kth.se/~viggo/wwwcompendium/node152.html
        # However this implementation does not find the maximum.

        bits = 10
        n = 2 * 100

        # D = [random.randint(1, 2 ** bits) for _ in range(n)]
        # Be careful, a solution might not exist.

        D = [588, 685, 114, 74, 986, 856, 289, 1008, 168, 311, 723, 843, 73, 955, 792, 940, 97, 208, 965, 311, 42, 67,
             272, 664, 216, 710, 400, 786, 1005, 228, 124, 957, 692, 255, 607, 261, 794, 602, 249, 388, 79, 803, 911,
             761, 391, 933, 731, 155, 92, 82, 996, 524, 55, 443, 471, 192, 861, 626, 233, 299, 873, 865, 173, 215, 852,
             129, 204, 851, 320, 63, 915, 883, 855, 62, 1018, 665, 518, 161, 722, 145, 249, 736, 61, 708, 713, 365, 21,
             473, 750, 145, 294, 426, 7, 420, 253, 15, 601, 756, 51, 477, 291, 383, 930, 231, 977, 706, 529, 267, 58,
             427, 742, 687, 970, 600, 607, 670, 377, 166, 211, 631, 321, 772, 301, 257, 457, 647, 498, 485, 377, 597,
             763, 860, 95, 271, 43, 807, 160, 150, 271, 861, 614, 854, 292, 865, 611, 727, 174, 509, 911, 757, 119, 771,
             837, 18, 855, 657, 904, 418, 762, 601, 965, 187, 380, 223, 568, 230, 316, 914, 817, 380, 864, 885, 358,
             508, 929, 698, 292, 728, 948, 178, 990, 418, 604, 4, 920, 947, 16, 448, 612, 235, 617, 320, 869, 966, 190,
             1020, 476, 831, 574, 45]
        if self.verbose:
            print(D)

        self.engine(bits=sum(D).bit_length())

        bins, sub, com = satx.subsets(D, n // 2, complement=True)

        assert sum(sub) == sum(com)

        self.assert_satisfiable()
        sub = [D[i] for i in range(n) if bins.binary[i]]
        com = [D[i] for i in range(n) if not bins.binary[i]]
        if self.verbose:
            print(sub, com)
        self.assertEqual(sum(sub), sum(com))

    def test_vectors(self, plot=False) -> None:
        dim = 2

        self.engine(bits=6)

        ps: list[satx.Rational] = satx.vector(size=dim, is_rational=True)

        assert sum(p ** 2 for p in ps) <= 1  # type: ignore

        ground_truth = {((2, 6), (0, 1)), ((0, 1), (4, 7)), ((0, 2), (0, 1)), ((0, 2), (0, 3)), ((0, 3), (2, 2)),
                        ((1, 1), (0, 7)), ((0, 1), (0, 3)), ((0, 1), (5, 7)), ((0, 5), (1, 1)), ((0, 1), (0, 7)),
                        ((0, 1), (3, 7)), ((1, 1), (0, 6)), ((2, 3), (0, 1)), ((1, 2), (0, 2)), ((1, 2), (0, 3)),
                        ((0, 1), (1, 5)), ((0, 1), (5, 5)), ((0, 1), (4, 5)), ((0, 1), (5, 6)), ((0, 1), (4, 6)),
                        ((0, 1), (1, 6)), ((0, 1), (1, 2)), ((0, 2), (1, 1)), ((2, 2), (0, 2)), ((1, 2), (0, 1)),
                        ((1, 7), (0, 1)), ((0, 1), (2, 3)), ((1, 1), (0, 3)), ((0, 2), (3, 3)), ((4, 6), (0, 1)),
                        ((0, 3), (0, 1)), ((5, 7), (0, 1)), ((0, 3), (1, 1)), ((0, 1), (3, 6)), ((3, 7), (0, 1)),
                        ((0, 1), (2, 4)), ((0, 1), (0, 2)), ((1, 1), (0, 2)), ((0, 1), (2, 2)), ((0, 7), (0, 1)),
                        ((4, 7), (0, 1)), ((7, 7), (0, 1)), ((1, 1), (0, 5)), ((0, 1), (2, 5)), ((0, 6), (0, 1)),
                        ((0, 1), (3, 5)), ((0, 1), (1, 4)), ((0, 1), (0, 4)), ((0, 1), (4, 4)), ((1, 1), (0, 4)),
                        ((0, 3), (0, 2)), ((1, 5), (0, 1)), ((2, 5), (0, 1)), ((0, 1), (2, 7)), ((0, 1), (6, 7)),
                        ((1, 6), (0, 1)), ((1, 4), (0, 1)), ((3, 4), (0, 1)), ((0, 4), (0, 1)), ((4, 4), (0, 1)),
                        ((2, 4), (0, 1)), ((0, 1), (0, 5)), ((0, 7), (1, 1)), ((1, 2), (2, 3)), ((0, 1), (1, 3)),
                        ((0, 2), (1, 3)), ((2, 3), (0, 2)), ((0, 1), (3, 4)), ((0, 2), (2, 2)), ((0, 1), (2, 6)),
                        ((3, 3), (0, 2)), ((6, 6), (0, 1)), ((1, 2), (1, 3)), ((2, 2), (0, 3)), ((0, 1), (0, 6)),
                        ((1, 3), (0, 2)), ((1, 3), (1, 2)), ((0, 4), (1, 1)), ((0, 6), (1, 1)), ((0, 2), (0, 2)),
                        ((0, 1), (1, 7)), ((0, 3), (1, 2)), ((0, 2), (2, 3)), ((0, 2), (1, 2)), ((2, 2), (0, 1)),
                        ((0, 5), (0, 1)), ((0, 1), (1, 1)), ((0, 1), (3, 3)), ((0, 1), (6, 6)), ((4, 5), (0, 1)),
                        ((5, 5), (0, 1)), ((1, 1), (0, 1)), ((1, 2), (1, 2)), ((0, 1), (0, 1)), ((1, 3), (0, 1)),
                        ((2, 7), (0, 1)), ((3, 5), (0, 1)), ((5, 6), (0, 1)), ((3, 3), (0, 1)), ((3, 6), (0, 1)),
                        ((0, 1), (7, 7)), ((6, 7), (0, 1)), ((2, 3), (1, 2))}

        self.assert_solutions(lambda: tuple((int(x.numerator), int(x.denominator)) for x in ps), ground_truth)
        if plot:
            import matplotlib.pyplot as plt  # type: ignore
            dots = {tuple(num / denom for num, denom in s) for s in ground_truth}
            x, y = zip(*dots)
            plt.axis('equal')
            plt.plot(x, y, 'r.')
            plt.show()

    def test_fibonacci(self):

        @lru_cache
        def fibonacci(k: int):
            if k == 0 or k == 1:
                return k
            return fibonacci(k - 1) + fibonacci(k - 2)

        for n in range(2, 100):
            self.engine(bits=n)

            x = satx.vector(size=n + 1)

            assert x[0] == 0
            assert x[1] == 1
            for i in range(2, n + 1):
                assert x[i - 1] + x[i - 2] == x[i]

            self.assert_satisfiable()
            self.assertEqual(int(x[n]), fibonacci(n))

    def test_rational_diophantine_equation(self):
        self.engine(bits=10)

        x = satx.rational()
        y = satx.rational()

        assert x ** 3 + x * y == y ** 2
        assert x != 0
        assert y != 0

        solutions = set()
        while self.satisfy():
            solutions.add((fractions.Fraction(int(x.numerator), int(x.denominator)),
                           fractions.Fraction(int(y.numerator), int(y.denominator))))

        self.assertEqual(solutions, {(fractions.Fraction(2, 1), fractions.Fraction(4, 1)),
                                     (fractions.Fraction(6, 1), fractions.Fraction(18, 1))})

    def test_semi_magic_square_of_squares(self):
        self.engine(bits=22)

        p = satx.integer()
        q = satx.integer()
        r = satx.integer()
        s = satx.integer()

        satx.apply_single([p, q, r, s], lambda x: x > 0)

        A = (p ** 2 + q ** 2 - r ** 2 - s ** 2) ** 2
        B = (2 * (q * r + p * s)) ** 2
        C = (2 * (p * r - q * s)) ** 2

        D = (2 * (q * r - p * s)) ** 2
        E = (p ** 2 - q ** 2 + r ** 2 - s ** 2) ** 2
        F = (2 * (r * s + p * q)) ** 2

        G = (2 * (q * s + p * r)) ** 2
        H = (2 * (p * q - r * s)) ** 2
        I = (p ** 2 - q ** 2 - r ** 2 + s ** 2) ** 2

        assert E + I == B + C
        # assert G + E == F + I # perfect magic

        self.assert_satisfiable()

        self.assertIn((int(p), int(q), int(r), int(s)), {(38, 21, 16, 5), (38, 16, 21, 5)})

    def test_dudeney(self):
        """
        See https://en.wikipedia.org/wiki/Dudeney_number

        In number theory, a Dudeney number in a given number base b is a natural number
        equal to the perfect cube of another natural number such that the digit sum
        of the first natural number is equal to the second.
        The name derives from Henry Dudeney, who noted the existence of these numbers in one of his puzzles.

        There are 5 non-trivial numbers for base 10, and the highest such number is formed of 5 digits.
        Below, the model is given for base 10.
        """

        # for base 10
        n_digits = 5

        self.engine((10 ** n_digits).bit_length())

        # n is a (non-trivial) Dudeney number
        n = satx.integer()
        # s is the perfect cubic root of n
        s = satx.integer()
        # d[i] is the ith digit of the Dudeney number
        d = satx.vector(size=n_digits)

        satx.apply_single(d, lambda t: t < 10)

        assert 2 <= n < 10 ** n_digits
        assert s < math.ceil((10 ** n_digits) ** (1 / 3)) + 1
        assert n == s * s * s
        assert sum(d) == s
        assert satx.dot(d, [10 ** (n_digits - i - 1) for i in range(n_digits)]) == n

        ground_truth = {(4913, 17, 0, 4, 9, 1, 3), (5832, 18, 0, 5, 8, 3, 2), (19683, 27, 1, 9, 6, 8, 3),
                        (17576, 26, 1, 7, 5, 7, 6), (512, 8, 0, 0, 5, 1, 2)}

        self.assert_solutions(lambda: (int(n), int(s)) + tuple(int(dd) for dd in d), ground_truth)

    def test_magic_modulo_number(self):
        """
        See model in OscaR

        A number with an interesting property: when I divide it by v, the remainder is v-1,
        and this from v ranging from 2 to 9.
        It's not a small number, but it's not really big, either.
        When I looked for a smaller number with this property I couldn't find one.
        Can you find it?
        """

        self.engine(bits=14)

        x = satx.integer()

        for i in range(2, 10):
            assert x % i == i - 1

        self.assert_solutions(lambda: int(x), {5039, 10079, 15119, 7559, 12599, 2519})

    def test_n_fractions(self):
        """
        Problem 041 on CSPLib

        https://github.com/csplib/csplib/blob/master/Problems/prob041/specification.md
        """

        self.engine(bits=16)

        digits = satx.vector(size=9)

        satx.apply_single(digits, lambda t: 0 < t < 10)

        a, b, c, d, e, f, g, h, i = digits

        satx.all_different(digits)

        assert a * (10 * e + f) * (10 * h + i) + d * (10 * b + c) * (10 * h + i) + g * (10 * b + c) * (
                10 * e * f) == \
               (10 * b + c) * (10 * e + f) * (10 * h + i)

        ground_truth = {(7, 1, 6, 8, 3, 2, 9, 5, 4), (9, 5, 2, 7, 1, 3, 8, 6, 4), (9, 1, 8, 4, 2, 3, 7, 5, 6),
                        (2, 1, 9, 4, 3, 8, 7, 5, 6), (9, 2, 1, 8, 6, 3, 7, 4, 5), (7, 6, 9, 4, 2, 3, 5, 1, 8),
                        (9, 1, 3, 4, 7, 8, 2, 5, 6), (6, 5, 2, 9, 1, 3, 7, 8, 4), (5, 1, 3, 4, 2, 9, 6, 7, 8),
                        (6, 3, 8, 2, 1, 9, 7, 4, 5), (7, 4, 9, 3, 1, 6, 5, 2, 8), (9, 5, 2, 7, 1, 3, 6, 4, 8),
                        (7, 1, 2, 5, 9, 3, 6, 4, 8), (9, 1, 8, 2, 3, 4, 7, 5, 6), (7, 8, 4, 9, 3, 6, 2, 1, 5)}

        self.assert_solutions(lambda: tuple(int(d) for d in digits), ground_truth)

    def test_prime_looking(self):
        """
        See Model in OscaR

        Martin Gardner Problem:
         * Call a number "prime-looking" if it is composite but not divisible by 2,3 or 5.
         * The three smallest prime-looking numbers are 49, 77 and 91.
         * There are 168 prime numbers less than 1000.
         * How many prime-looking numbers are there less than 1000?
        """

        self.engine(bits=10)

        # the number we're looking for
        x = satx.integer()
        # a first divisor
        d1 = satx.integer()
        # a second divisor
        d2 = satx.integer()

        assert x < 1000
        assert 2 <= d1 < 1000
        assert 2 <= d2 < 1000

        assert x == d1 * d2
        assert x % 2 != 0
        assert x % 3 != 0
        assert x % 5 != 0
        assert d1 <= d2

        ground_truth = {(427, 7, 61), (959, 7, 137), (917, 7, 131), (949, 13, 73), (583, 11, 53), (737, 11, 67),
                        (871, 13, 67), (791, 7, 113), (889, 7, 127), (851, 23, 37), (731, 17, 43), (667, 23, 29),
                        (203, 7, 29), (259, 7, 37), (403, 13, 31), (299, 13, 23), (413, 7, 59), (913, 11, 83),
                        (539, 11, 49), (671, 11, 61), (553, 7, 79), (497, 7, 71), (119, 7, 17), (121, 11, 11),
                        (869, 11, 79), (511, 7, 73), (481, 13, 37), (451, 11, 41), (847, 11, 77), (91, 7, 13),
                        (77, 7, 11), (473, 11, 43), (847, 7, 121), (833, 7, 119), (469, 7, 67), (221, 13, 17),
                        (637, 13, 49), (247, 13, 19), (973, 7, 139), (287, 7, 41), (559, 13, 43), (533, 13, 41),
                        (539, 7, 77), (793, 13, 61), (767, 13, 59), (611, 13, 47), (217, 7, 31), (329, 7, 47),
                        (377, 13, 29), (169, 13, 13), (301, 7, 43), (623, 7, 89), (749, 7, 107), (763, 7, 109),
                        (637, 7, 91), (841, 29, 29), (961, 31, 31), (899, 29, 31), (713, 23, 31), (943, 23, 41),
                        (989, 23, 43), (817, 19, 43), (893, 19, 47), (517, 11, 47), (781, 11, 71), (803, 11, 73),
                        (931, 19, 49), (343, 7, 49), (371, 7, 53), (319, 11, 29), (551, 19, 29), (161, 7, 23),
                        (529, 23, 23), (437, 19, 23), (589, 19, 31), (341, 11, 31), (253, 11, 23), (49, 7, 7),
                        (209, 11, 19), (703, 19, 37), (361, 19, 19), (187, 11, 17), (407, 11, 37), (779, 19, 41),
                        (143, 11, 13), (649, 11, 59), (979, 11, 89), (679, 7, 97), (931, 7, 133), (923, 13, 71),
                        (689, 13, 53), (707, 7, 101), (721, 7, 103), (697, 17, 41), (527, 17, 31), (581, 7, 83),
                        (629, 17, 37), (133, 7, 19), (901, 17, 53), (833, 17, 49), (493, 17, 29), (289, 17, 17),
                        (391, 17, 23), (323, 17, 19), (799, 17, 47)}

        self.assert_solutions(lambda: (int(x), int(d1), int(d2)), ground_truth)

    def test_traffic_lights(self):
        """
        Problem 016 on CSPLib

        https://github.com/csplib/csplib/blob/master/Problems/prob016/specification.md
        """

        self.engine(bits=13)

        mapping = {1: 'r', 2: 'ry', 3: 'g', 4: 'y'}

        R, RY, G, Y = 1, 2, 3, 4

        table = [(R, R, G, G), (RY, R, Y, R), (G, G, R, R), (Y, R, RY, R)]

        # v[i] is the color for the ith vehicle traffic light
        v = satx.vector(size=4)
        # p[i] is the color for the ith pedestrian traffic light
        p = satx.vector(size=4)

        satx.apply_single(v, lambda t: t.is_in([R, RY, G, Y]))
        satx.apply_single(p, lambda t: t.is_in([R, G]))

        for i in range(4):
            assert satx.dot([v[i], p[i], v[(i + 1) % 4], p[(i + 1) % 4]], [1, 10, 100, 1000]) == \
                   satx.one_of([satx.dot(t, [1, 10, 100, 1000]) for t in table])

        ground_truth = {((1, 3, 1, 3), (1, 3, 1, 3)), ((4, 2, 4, 2), (1, 1, 1, 1)),
                        ((2, 4, 2, 4), (1, 1, 1, 1)), ((3, 1, 3, 1), (3, 1, 3, 1))}

        def solution():
            if self.verbose:
                vv = [mapping[t.value] for t in v]
                pp = [mapping[t.value] for t in p]
                for a, b in zip(vv, pp):
                    print(a, b, end=', ')
                print()
            return tuple(int(vv) for vv in v), tuple(int(pv) for pv in p)

        self.assert_solutions(solution, ground_truth)


if __name__ == "__main__":
    unittest.main()
