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

from __future__ import annotations

import functools
import operator
from typing import Sequence, Union, Tuple, SupportsInt, Callable, Optional, cast, Iterable

from satx.engine import Engine
from satx.gaussian import Gaussian
from satx.rational import Rational
from satx.unit import Unit


class SatXEngine(Engine):
    """
    Adds a few convenience and utility methods to the Engine.
    """

    def __init__(self, bits: int, cnf_path=None, signed=False, simplify=False):
        """
        Initialize or reset the SAT-X system.
        :param bits: Implies an $[-2^{bits}, 2^{bits})$ search space.
        :param cnf_path: Path to render the generated CNF.
        :param signed: Indicates use of signed integer engine
        """
        super().__init__(bits, cnf_path, signed, simplify)

    @property
    def zero(self):
        return self.constant(0)

    @property
    def one(self):
        return self.constant(1)

    def subsets(self, lst: Sequence[Unit | SupportsInt], k: Unit | SupportsInt | None = None, complement=False) \
            -> Union[Tuple[Unit, list[Unit]], Tuple[Unit, list[Unit], list[Unit]]]:
        """
        Generate all subsets for a specific universe of data.
        :param lst: The universe of data.
        :param k: The cardinality of the subsets.
        :param complement: True if include the complement in return .
        :return: (binary representation of subsets, the generic subset representation,
                  the complement of subset if complement=True)
        """

        bits = self.integer(bits=len(lst))
        if k is not None:
            assert sum(self.zero.iff(~bits[i], self.one) for i in range(len(lst))) == k
        subset_ = [self.zero.iff(~bits[i], lst[i]) for i in range(len(lst))]
        if complement:
            complement_ = [self.zero.iff(bits[i], lst[i]) for i in range(len(lst))]
            return bits, subset_, complement_
        else:
            return bits, subset_

    def _array(self, dimension: int, bits: int | None = None) -> list[Unit]:
        return [self.integer(bits) for _ in range(dimension)]

    def _subset(self, k: int, data: Sequence[Unit | SupportsInt],
                empty: Unit | SupportsInt | None = None,
                complement=False) -> list[Unit] | Tuple[list[Unit], list[Unit]]:
        x = self.integer(bits=len(data))
        self.at_most_k(x, k)
        y = self._array(dimension=len(data))
        z = []
        if complement:
            z = self._array(dimension=len(data))
        for i in range(len(data)):
            assert self.zero.iff(x[i], data[i]) == self.zero.iff(x[i], y[i])
            assert self.zero.iff(~x[i], self.zero if empty is None else empty) == self.zero.iff(~x[i], y[i])
            if complement:
                assert self.zero.iff(~x[i], data[i]) == self.zero.iff(~x[i], z[i])
                assert self.zero.iff(x[i], self.zero if empty is None else empty) == self.zero.iff(x[i], z[i])
        if complement:
            return y, z
        return y

    def subset(self, lst: Sequence[Unit | SupportsInt], k: int, empty=None, complement=False) \
            -> list[Unit] | Tuple[list[Unit], list[Unit]]:
        """
        An operative structure (like integer ot constant) that represent a subset of at most k elements.
        :param lst: The data for the subsets.
        :param k: The maximal bits for subsets.
        :param empty: The empty element, 0, by default.
        :param complement: True if include in return the complement.
        :return: An instance of subset or (subset, complement) if complement=True.
        """
        complement_: list[Unit] = []
        if complement:
            res = cast(Tuple[list[Unit], list[Unit]], self._subset(k, lst, empty, complement=complement))
            subset_, complement_ = res
        else:
            subset_ = cast(list[Unit], self._subset(k, lst, empty))
        if complement:
            return subset_, complement_
        return subset_

    def vector(self, size: int, bits: int | None = None, is_gaussian=False, is_rational=False):
        """
        A vector of integers.
        :param bits: The bit bits for each integer.
        :param size: The bits of the vector.
        :param is_gaussian: Indicate of is a Gaussian Integers vector.
        :param is_rational: Indicate of is a Rational vector.
        :return: An instance of vector.
        """

        if is_rational:
            return [self.rational() for _ in range(size)]
        if is_gaussian:
            return [self.gaussian() for _ in range(size)]
        array_ = self._array(bits=bits, dimension=size)
        return array_

    def matrix(self, dimensions: Tuple[int, int], bits: int | None = None, is_gaussian=False, is_rational=False):
        """
        A matrix of integers.
        :param bits: The bit bits for each integer.
        :param dimensions: A tuple with the dimensions for the Matrix (n, m).
        :param is_gaussian: Indicate of is a Gaussian Integers vector.
        :param is_rational: Indicate of is a Rational Matrix.
        :return: An instance of Matrix.
        """

        matrix_: list[list[Unit | Rational | Gaussian]] = []
        for i in range(dimensions[0]):
            row = []
            for j in range(dimensions[1]):
                element: Unit | Rational | Gaussian
                if is_rational:
                    x = self.integer(bits=bits)
                    y = self.integer(bits=bits)
                    element = self.rational(x, y)
                elif is_gaussian:
                    x = self.integer(bits=bits)
                    y = self.integer(bits=bits)
                    element = self.gaussian(x, y)
                else:
                    element = self.integer(bits=bits)
                row.append(element)
            matrix_.append(row)
        return matrix_

    def matrix_permutation(self, lst: Sequence[Unit | int], n: int):
        """
        This generates the permutations for a square matrix.
        :param lst: The flattened matrix of data, i.e. a vector.
        :param n: The dimension for the square nxn-matrix.
        :return: A tuple with (index for the elements, the elements that represent the indexes)
        """

        xs: list[Unit] = self.vector(size=n)
        ys: list[Unit] = self.vector(size=n)
        self._apply(xs, single=lambda x: 0 <= x < n)
        self._apply(xs, dual=lambda a, b: a != b)
        self._indexing(xs, ys, lst)
        return xs, ys

    def permutations(self, lst: Sequence[Unit | SupportsInt], n: int):
        """
        Entangle all permutations of size n of a list.
        :param lst: The list to entangle.
        :param n: The bits of entanglement.
        :return: (indexes, values)
        """

        xs: list[Unit] = self.vector(size=n)
        ys: list[Unit] = self.vector(size=n)
        for i in range(n):
            assert self.element(ys[i], lst) == xs[i]
        self.apply_single(xs, lambda a: 0 <= a < n)
        self.apply_dual(xs, lambda a, b: a != b)
        return xs, ys

    def combinations(self, lst: Sequence[Unit | SupportsInt], n: int):
        """
        Entangle all combinations of bits n for the vector lst.
        :param lst: The list to entangle.
        :param n: The bits of entanglement.
        :return: (indexes, values)
        """

        xs: list[Unit] = self.vector(size=n)
        ys: list[Unit] = self.vector(size=n)
        for i in range(n):
            assert self.element(ys[i], lst) == xs[i]
        return xs, ys

    def all_binaries(self, lst: Sequence[Unit | SupportsInt]):
        """
        This say that, the vector of integer are all binaries.
        :param lst: The vector of integers.
        :return:
        """

        self._apply(lst, single=lambda arg: 0 <= arg <= 1)

    def switch(self, x: Unit, ith: int, neg=False):
        """
        This conditionally flip the internal bit for an integer.
        :param x: The integer.
        :param ith: Indicate the ith bit.
        :param neg: indicate if the condition is inverted.
        :return: 0 if the uth bit for the argument collapse to true else return 1, if neg is active exchange 1 by 0.
        """
        return self.alu.iff(0, 1, ~x[ith] if neg else x[ith])

        # return self.zero.iff(-x[ith] if neg else x[ith], self.one)

    def one_of(self, lst: Sequence[Unit | SupportsInt]):
        """
        This indicates that at least one of the instruction on the array is active for the current problem.
        :param lst: A list of instructions.
        :return: The entangled structure.
        """

        bits = self.integer(bits=len(lst))
        assert sum(bits[[i]](self.zero, self.one) for i in range(len(lst))) == self.one
        return sum(bits[[i]](self.zero, lst[i]) for i in range(len(lst)))

    def factorial(self, x: Unit | int):
        """
        The factorial for the integer.
        :param x: The integer.
        :return: The factorial.
        """

        sub = self.integer()
        assert sum([self.zero.iff(sub[i], self.one) for i in range(self.bits)]) == self.one
        assert sum([self.zero.iff(sub[i], i) for i in range(self.bits)]) == x
        return sum([self.zero.iff(sub[i], functools.reduce(operator.mul, [x - j for j in range(i)]))
                    for i in range(1, self.bits)])

    def sigma(self, f: Callable[[int], Unit | SupportsInt], i: int, n: Unit | SupportsInt):
        """
        The Sum for i to n, for the lambda f,
        :param f: A lambda f with a standard int parameter.
        :param i: The start for the Sum, a standard int.
        :param n: The integer that represent the end of the Sum.
        :return: The entangled structure.
        """

        def __sum(xs):
            if xs:
                return functools.reduce(operator.add, xs)
            return self.zero

        sub = self.integer()
        assert sum([self.zero.iff(sub[j], self.one) for j in range(self.bits)]) == self.one
        assert sum([self.zero.iff(sub[j], j) for j in range(self.bits)]) == n + self.one
        return sum([self.zero.iff(sub[j], __sum([f(j) for j in range(i, j)])) for j in range(i, self.bits)])

    def pi(self, f: Callable[[int], Unit | SupportsInt], i: int, n: Unit | SupportsInt):
        """
        The Pi for i to n, for the lambda f,
        :param f: A lambda f with a standard int parameter.
        :param i: The start for the Pi, a standard int.
        :param n: The integer that represent the end of the Pi.
        :return: The entangled structure.
        """

        def __pi(xs):
            if xs:
                return functools.reduce(operator.mul, xs)
            return self.one

        sub = self.integer()
        assert sum([self.zero.iff(sub[j], self.one) for j in range(self.bits)]) == self.one
        assert sum([self.zero.iff(sub[j], j) for j in range(self.bits)]) == n + self.one
        return sum([self.zero.iff(sub[j], __pi([f(j) for j in range(i, j)])) for j in range(i, self.bits)])

    @staticmethod
    def dot(xs: Iterable, ys: Iterable):
        """
        The dot product of two compatible Vectors.
        :param xs: The first vector.
        :param ys: The second vector.
        :return: The dot product.
        """
        return sum([x * y for x, y in zip(xs, ys)])

    @staticmethod
    def mul(xs: Iterable, ys: Iterable):
        """
        The element wise product of two Vectors.
        :param xs: The first vector.
        :param ys: The second vector.
        :return: The product.
        """
        return [x * y for x, y in zip(xs, ys)]

    @staticmethod
    def values(lst: Sequence[Unit], cleaner: Optional[Callable] = None):
        """
        Convert to standard values
        :param lst: List with elements.
        :param cleaner: Filter for elements.
        :return: Standard (filtered) values.
        """
        if cleaner is not None:
            return list(filter(cleaner, [x.value for x in lst]))
        return [x.value for x in lst]

    @staticmethod
    def _apply(xs: Sequence, single: Optional[Callable] = None, dual: Optional[Callable] = None,
               different: Optional[Callable] = None):
        for i in range(len(xs)):
            if single is not None:
                single(xs[i])
            if dual is not None:
                for j in range(i + 1, len(xs)):
                    dual(xs[i], xs[j])
            if different is not None:
                for j in range(len(xs)):
                    if i != j:
                        different(xs[i], xs[j])

    @staticmethod
    def _apply_indexed(xs: Sequence, single: Optional[Callable] = None, dual: Optional[Callable] = None,
                       different: Optional[Callable] = None):
        for i in range(len(xs)):
            if single is not None:
                single(i, xs[i])
            if dual is not None:
                for j in range(i + 1, len(xs)):
                    dual(i, j, xs[i], xs[j])
            if different is not None:
                for j in range(len(xs)):
                    if i != j:
                        different(i, j, xs[i], xs[j])

    @staticmethod
    def apply_single(lst: Sequence, f: Callable, indexed=False):
        """
        A sequential operation over a vector.
        :param lst: The vector.
        :param f: The lambda f of one integer variable.
        :param indexed: The lambda f of two integer variable, the first is an index.
        :return: The entangled structure.
        """
        if indexed:
            SatXEngine._apply_indexed(lst, single=f)
        else:
            SatXEngine._apply(lst, single=f)

    @staticmethod
    def apply_dual(lst: Sequence, f: Callable, indexed=False):
        """
        A cross operation over a vector on all pairs i, j such that i < j elements.
        :param lst: The vector.
        :param f: The lambda f of two integer variables.
        :param indexed: The lambda f of four integer variable, the 2 firsts are indexes.
        :return: The entangled structure.
        """
        if indexed:
            SatXEngine._apply_indexed(lst, dual=f)
        else:
            SatXEngine._apply(lst, dual=f)

    @staticmethod
    def apply_different(lst: Sequence, f: Callable, indexed=False):
        """
        A cross operation over a vector on all pairs i, j such that i != j elements.
        :param lst: The vector.
        :param f: The lambda f of two integer variables.
        :param indexed: The lambda f of four integer variable, the 2 firsts are indexes.
        :return: The entangled structure.
        """
        if indexed:
            SatXEngine._apply_indexed(lst, different=f)
        else:
            SatXEngine._apply(lst, different=f)

    @staticmethod
    def all_different(args: Sequence):
        """
        The all different global constraint.
        :param args: A vector of integers.
        :return:
        """
        SatXEngine._apply(args, dual=lambda x, y: x != y)

    @staticmethod
    def all_out(args: Sequence, values: Sequence):
        """
        The all different to values global constraint.
        :param args: A vector of integers.
        :param values: The values excluded.
        :return:
        """
        SatXEngine._apply(args, single=lambda x: [x != v for v in values])

    def all_in(self, args: Sequence, values: Sequence[Unit | SupportsInt]):
        """
        The all in values global constraint.
        :param args: A vector of integers.
        :param values: The values included.
        :return:
        """
        SatXEngine._apply(args, single=lambda x: x == self.one_of(values))

    @staticmethod
    def flatten(mtx):
        """
        Flatten a matrix into list.
        :param mtx: The matrix.
        :return: The entangled structure.
        """
        return [item for sublist in mtx for item in sublist]

    @property
    def bits(self):
        """
        The current bits for the engine.
        :return: The bits
        """

        return self.alu.default_bits

    @property
    def oo(self):
        """
        The infinite for rhe system, the maximal value for the current engine.
        :return: 2 ** bits - 1
        """

        return 2 ** self.bits - 1

    def assert_element(self, index: Unit | SupportsInt, lst: Sequence[Unit | SupportsInt], item: Unit | SupportsInt):
        """
        assert lst[index] == item
        """
        index_oh = self.alu.one_hot(index, len(lst))
        lst_item = self.alu.mux_array(lst, index_oh)
        assert lst_item == item
        # idx = self.integer(bits=len(lst))
        # self.at_most_k(idx, 1)
        # for i in range(len(lst)):
        #     assert self.zero.iff(idx[i], i) == self.zero.iff(idx[i], index)
        #     assert self.zero.iff(idx[i], lst[i]) == self.zero.iff(idx[i], item)
        pass

    def element(self, item: Unit | SupportsInt, data: Sequence[Unit | SupportsInt]):
        """
        Ensure that the element i is on the data, on the position index.
        :param item: The element
        :param data: The data
        :return: The position of element
        """

        ith = self.integer()
        self.assert_element(ith, data, item)
        return ith

    def index(self, ith: Unit | SupportsInt, data: Sequence[Unit | SupportsInt]):
        """
        Ensure that the element i is on the data, on the position index.
        :param ith: The element
        :param data: The data
        :return: The position of element
        """

        item = self.integer()
        self.assert_element(ith, data, item)
        return item

    def _indexing(self, xs: Sequence[Unit | int], ys: Sequence[Unit | int],
                  lst: Sequence[Unit | int]):
        n = len(xs)
        for i in range(n):
            self.assert_element(n * xs[i] + xs[(i + 1) % n], lst, ys[i])

    def _sequencing(self, xs: Sequence[Unit | SupportsInt], ys: Sequence[Unit | SupportsInt],
                    lst: Sequence[Unit | SupportsInt]):
        n = len(xs)
        zs = self._array(n)
        for i in range(n):
            self.assert_element(zs[i], lst, ys[i])
            assert xs[i] == zs[i]

    def _permutations(self, xs: Sequence[Unit | SupportsInt], lst: Sequence[Unit | SupportsInt]):
        n = len(xs)
        zs = self._array(n)
        for i in range(n):
            self.assert_element(zs[i], lst, xs[i])
        self._apply(zs, single=lambda a: a < n)
        self._apply(zs, dual=lambda a, b: a != b)

    def _combinations(self, xs: Sequence[Unit | SupportsInt], lst: Sequence[Unit | SupportsInt]):
        n = len(xs)
        zs = self._array(n)
        for i in range(n):
            self.assert_element(zs[i], lst, xs[i])

    def at_most_k(self, x: Unit, k: int) -> Unit:
        """
        At most k bits can be activated for this integer.
        :param x: An integer.
        :param k: k elements
        :return: The encoded variable
        """

        self.alu.at_most_k(x, k)
        return x

    def sqrt(self, x: Unit | SupportsInt):
        """
        Define x as a perfect square.
        :param x: The integer
        :return: The square of this integer.
        """

        y = self.integer()
        assert x == y * y
        return y

    @staticmethod
    def reshape(lst, dimensions):
        """
        Reshape a list
        :param lst: The coherent list to reshape
        :param dimensions:  The list of dimensions
        :return: The reshaped list
        """
        if len(dimensions) == 1:
            return lst
        n = functools.reduce(operator.mul, dimensions[1:])
        return [SatXEngine.reshape(lst[i * n:(i + 1) * n], dimensions[1:]) for i in range(len(lst) // n)]

    @staticmethod
    def clear(lst: Sequence[Unit]):
        """
        Clear a list of integers, used with optimization routines.
        :param lst: The coherent list of integers to clear.
        """
        for x in lst:
            x.clear()

    def rotate(self, x: Unit, k: Unit | int):
        """
        Rotate an integer k places
        :param x: the integer.
        :param k: k-places.
        :return: a rotated integer.
        """
        v = self.integer()
        for i in range(self.bits):
            assert x[[(i + k) % self.bits]](0, 1) == v[[i]](0, 1)
        return v

    def is_prime(self, p: Unit | SupportsInt):
        """
        Indicate that p is prime.
        :param p: the integer.
        """

        assert pow(self.constant(2), p, p) == 2

    def is_not_prime(self, p: Unit | SupportsInt):
        """
        Indicate that p is not prime.
        :param p: the integer.
        """

        assert p != 2
        assert pow(self.constant(2), p, p) != 2

    @staticmethod
    def abs_val(x, y):
        """
        Enforce the fact that the first variable is equal to the absolute value of the second variable.
        """
        assert y >= 0
        assert abs(x) == y

    def all_differ_from_at_least_k_pos(self, k: Unit | SupportsInt, lst: Sequence[Sequence[Unit | SupportsInt]]):
        """
        Enforce all pairs of distinct vectors of the VECTORS collection to differ from at least K positions.
        """
        nil1 = self.integer()
        nil2 = self.integer()
        for V in lst:
            nil1.is_not_in(V)
            nil2.is_not_in(V)
        assert nil1 != nil2
        t = self.tensor(dimensions=(len(lst[0]),))
        assert sum(t[[i]](0, 1) for i in range(len(lst[0]))) >= k
        for i1 in range(len(lst) - 1):
            for i2 in range(i1 + 1, len(lst)):
                for j in range(len(lst[0])):
                    assert t[[j]](nil1, lst[i1][j]) != t[[j]](nil2, lst[i2][j])

    def all_differ_from_at_most_k_pos(self, k: Unit | SupportsInt, lst: Sequence[Sequence[Unit | SupportsInt]]):
        """
        Enforce all pairs of distinct vectors of the VECTORS collection to differ from at most K positions.
        """
        nil1 = self.integer()
        nil2 = self.integer()
        for V in lst:
            nil1.is_not_in(V)
            nil2.is_not_in(V)
        assert nil1 == nil2
        t = self.tensor(dimensions=(len(lst[0]),))
        assert sum(t[[i]](0, 1) for i in range(len(lst[0]))) >= self.constant(len(lst[0])) - k
        for i1 in range(len(lst) - 1):
            for i2 in range(i1 + 1, len(lst)):
                for j in range(len(lst[0])):
                    assert t[[j]](nil1, lst[i1][j]) == t[[j]](nil2, lst[i2][j])

    def all_differ_from_exactly_k_pos(self, k: Unit | SupportsInt, lst: Sequence[Sequence[Unit | SupportsInt]]):
        """
        Enforce all pairs of distinct vectors of the VECTORS collection to differ from exactly K positions.
        Enforce K = 0 when |VECTORS| < 2.
        """
        self.all_differ_from_at_least_k_pos(k, lst)
        self.all_differ_from_at_most_k_pos(k, lst)

    @staticmethod
    def all_equal(lst: Sequence):
        """
        Enforce all variables of the collection
        """
        SatXEngine.apply_dual(lst, lambda x, y: x == y)

    @staticmethod
    def gcd(x: Unit | int, y: Unit | int, z: Unit | int):
        """
        Enforce the fact that 𝚉 is the greatest common divisor of 𝚇 and 𝚈. (assume X <= Y)
        """
        assert 0 < x <= y
        assert z > 0
        assert z == y % x
        assert (x // z) % (y % z) == 0

    def sort(self, lst1: Sequence[Unit | SupportsInt], lst2: Sequence[Unit | SupportsInt]):
        """
        First, the variables of the collection 𝚅𝙰𝚁𝙸𝙰𝙱𝙻𝙴𝚂 2 correspond to a permutation of the variables of
        𝚅𝙰𝚁𝙸𝙰𝙱𝙻𝙴𝚂 1. Second, the variables of 𝚅𝙰𝚁𝙸𝙰𝙱𝙻𝙴𝚂 2 are sorted in increasing order.
        """
        _, ys = self.permutations(lst1, len(lst1))
        SatXEngine.apply_single(lst2, lambda i, t: t == ys[i], indexed=True)
        SatXEngine.apply_dual(lst2, lambda a, b: a <= b)

    def sort_permutation(self, lst_from: Sequence[Unit | SupportsInt], lst_per: Sequence[Unit | SupportsInt],
                         lst_to: Sequence[Unit | SupportsInt]):
        """
        The variables of collection 𝙵𝚁𝙾𝙼 correspond to the variables of collection 𝚃𝙾 according to the
        permutation 𝙿𝙴𝚁𝙼𝚄𝚃𝙰𝚃𝙸𝙾𝙽 (i.e., 𝙵𝚁𝙾𝙼[i].𝚟𝚊𝚛=𝚃𝙾[𝙿𝙴𝚁𝙼𝚄𝚃𝙰𝚃𝙸𝙾𝙽[i].𝚟𝚊𝚛].𝚟𝚊𝚛).
        The variables of collection 𝚃𝙾 are also sorted in increasing order.
        """
        SatXEngine.apply_dual(lst_to, lambda a, b: a <= b)
        xs1, ys1 = self.permutations(lst_from, len(lst_from))
        assert ys1 == lst_to
        assert lst_per == xs1

    def count(self, val: Unit | int, lst: Sequence[Unit | int], rel: Callable, lim: Unit | SupportsInt):
        """
        Let N be the number of variables of the 𝚅𝙰𝚁𝙸𝙰𝙱𝙻𝙴𝚂 collection assigned to value 𝚅𝙰𝙻𝚄𝙴; Enforce
        condition N 𝚁𝙴𝙻𝙾𝙿 𝙻𝙸𝙼𝙸𝚃 to hold.
        """
        t = self.tensor(dimensions=(len(lst),))
        for i in range(len(lst)):
            assert t[[i]](0, lst[i] - val) == 0
        assert rel(sum(t[[i]](0, 1) for i in range(len(lst))), lim)
