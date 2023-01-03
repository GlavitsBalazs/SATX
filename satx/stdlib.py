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

from typing import Sequence, Union, Tuple, SupportsInt, Callable, Iterable, Optional

from satx.gaussian import Gaussian
from satx.rational import Rational
from satx.satxengine import SatXEngine
from satx.unit import Unit

"""
The standard high level library for the SAT-X system.
"""

_engine: SatXEngine | None = None

__all__ = ['Unit', 'Rational', 'Gaussian', 'engine', 'get_engine', 'integer', 'constant', 'subsets', 'subset', 'vector',
           'matrix', 'matrix_permutation', 'permutations', 'combinations', 'all_binaries', 'switch', 'one_of',
           'factorial', 'sigma', 'pi', 'dot', 'mul', 'values', 'apply_single', 'apply_dual', 'apply_different',
           'all_different', 'all_out', 'all_in', 'flatten', 'bits', 'oo', 'element', 'index', 'gaussian', 'rational',
           'at_most_k', 'sqrt', 'reshape', 'tensor', 'clear', 'rotate', 'is_prime', 'is_not_prime', 'satisfy', 'reset']


def engine(bits: int, deep: int | None = None, info=False, cnf_path='', signed=False, simplify=False):
    """
    Initialize or reset the SAT-X system.
    :param bits: Implies an $[-2^{bits}, 2^{bits})$ search space.
    :param deep: Ignored.
    :param info: Print the information about the system.
    :param cnf_path: Path to render the generated CNF.
    :param signed: Indicates use of signed integer engine
    """
    global _engine

    if info:
        print('SAT-X The constraint modeling language for SAT solvers')
        print('Copyright (c) 2012-2022 Oscar Riveros. all rights reserved.')
        print('[SAT-X]')

    if _engine is not None:
        _engine.close()
    _engine = SatXEngine(bits, cnf_path, signed, simplify)


def get_engine():
    global _engine
    if _engine is None:
        raise Exception('The SAT-X system is not initialized.')
    return _engine


def integer(bits: int | None = None) -> Unit:
    """
    Correspond to an integer.
    :param bits: The bits for the integer.
    :return: An instance of Integer.
    """
    return get_engine().integer(bits)


def constant(value: int, bits: int | None = None) -> Unit:
    """
    Correspond to a constant.
    :param bits: The bits for the constant.
    :param value: The value of the constant.
    :return: An instance of Constant.
    """
    return get_engine().constant(value, bits)


def subsets(lst: Sequence[Unit | SupportsInt], k: Unit | SupportsInt | None = None, complement=False) \
        -> Union[Tuple[Unit, list[Unit]], Tuple[Unit, list[Unit], list[Unit]]]:
    """
    Generate all subsets for a specific universe of data.
    :param lst: The universe of data.
    :param k: The cardinality of the subsets.
    :param complement: True if include the complement in return .
    :return: (binary representation of subsets, the generic subset representation, the complement of subset if complement=True)
    """
    return get_engine().subsets(lst, k, complement)


def subset(lst: Sequence[Unit | SupportsInt], k: int, empty=None, complement=False):
    """
    An operative structure (like integer ot constant) that represent a subset of at most k elements.
    :param lst: The data for the subsets.
    :param k: The maximal bits for subsets.
    :param empty: The empty element, 0, by default.
    :param complement: True if include in return the complement.
    :return: An instance of subset or (subset, complement) if complement=True.
    """
    return get_engine().subset(lst, k, empty, complement)


def vector(bits: int | None = None, size: int | None = None, is_gaussian=False, is_rational=False):
    """
    A vector of integers.
    :param bits: The bit bits for each integer.
    :param size: The bits of the vector.
    :param is_gaussian: Indicate of is a Gaussian Integers vector.
    :param is_rational: Indicate of is a Rational vector.
    :return: An instance of vector.
    """
    assert size is not None
    return get_engine().vector(size, bits, is_gaussian, is_rational)


def matrix(bits: int | None = None, dimensions: Tuple[int, int] | None = None, is_gaussian=False, is_rational=False) \
        -> list[list[Unit | Rational | Gaussian]]:
    """
    A matrix of integers.
    :param bits: The bit bits for each integer.
    :param dimensions: A tuple with the dimensions for the Matrix (n, m).
    :param is_gaussian: Indicate of is a Gaussian Integers vector.
    :param is_rational: Indicate of is a Rational Matrix.
    :return: An instance of Matrix.
    """
    assert dimensions is not None
    return get_engine().matrix(dimensions, bits, is_gaussian, is_rational)


def matrix_permutation(lst: Sequence[Unit | int], n: int):
    """
    This generates the permutations for a square matrix.
    :param lst: The flattened matrix of data, i.e. a vector.
    :param n: The dimension for the square nxn-matrix.
    :return: A tuple with (index for the elements, the elements that represent the indexes)
    """
    return get_engine().matrix_permutation(lst, n)


def permutations(lst: Sequence[Unit | int], n: int):
    """
    Entangle all permutations of size n of a list.
    :param lst: The list to entangle.
    :param n: The bits of entanglement.
    :return: (indexes, values)
    """
    return get_engine().permutations(lst, n)


def combinations(lst: Sequence[Unit | int], n: int):
    """
    Entangle all combinations of bits n for the vector lst.
    :param lst: The list to entangle.
    :param n: The bits of entanglement.
    :return: (indexes, values)
    """
    return get_engine().combinations(lst, n)


def all_binaries(lst: Sequence[Unit | SupportsInt]):
    """
    This says that, the vector of integer are all binaries.
    :param lst: The vector of integers.
    :return:
    """
    return get_engine().all_binaries(lst)


def switch(x: Unit, ith: int, neg=False):
    """
    This conditionally flip the internal bit for an integer.
    :param x: The integer.
    :param ith: Indicate the ith bit.
    :param neg: indicate if the condition is inverted.
    :return: 0 if the uth bit for the argument collapse to true else return 1, if neg is active exchange 1 by 0.
    """
    return get_engine().switch(x, ith, neg)


def one_of(lst: Sequence[Unit | SupportsInt]):
    """
    This indicates that at least one of the instruction on the array is active for the current problem.
    :param lst: A list of instructions.
    :return: The entangled structure.
    """
    return get_engine().one_of(lst)


def factorial(x: Unit | int):
    """
    The factorial for the integer.
    :param x: The integer.
    :return: The factorial.
    """
    return get_engine().factorial(x)


def sigma(f: Callable[[int], Unit | SupportsInt], i: int, n: Unit | SupportsInt):
    """
    The Sum for i to n, for the lambda f,
    :param f: A lambda f with a standard int parameter.
    :param i: The start for the Sum, a standard int.
    :param n: The integer that represent the end of the Sum.
    :return: The entangled structure.
    """
    return get_engine().sigma(f, i, n)


def pi(f: Callable[[int], Unit | SupportsInt], i: int, n: Unit | SupportsInt):
    """
    The Pi for i to n, for the lambda f,
    :param f: A lambda f with a standard int parameter.
    :param i: The start for the Pi, a standard int.
    :param n: The integer that represent the end of the Pi.
    :return: The entangled structure.
    """
    return get_engine().pi(f, i, n)


def dot(xs: Iterable, ys: Iterable):
    """
    The dot product of two compatible Vectors.
    :param xs: The first vector.
    :param ys: The second vector.
    :return: The dot product.
    """
    return SatXEngine.dot(xs, ys)


def mul(xs: Iterable, ys: Iterable):
    """
    The element wise product of two Vectors.
    :param xs: The first vector.
    :param ys: The second vector.
    :return: The product.
    """
    return SatXEngine.mul(xs, ys)


def values(lst: Sequence[Unit], cleaner: Optional[Callable] = None):
    """
    Convert to standard values
    :param lst: List with elements.
    :param cleaner: Filter for elements.
    :return: Standard (filtered) values.
    """
    return SatXEngine.values(lst, cleaner)


def apply_single(lst: Sequence, f: Callable, indexed=False):
    """
    A sequential operation over a vector.
    :param lst: The vector.
    :param f: The lambda f of one integer variable.
    :param indexed: The lambda f of two integer variable, the first is an index.
    :return:
    """
    SatXEngine.apply_single(lst, f, indexed)


def apply_dual(lst: Sequence, f: Callable, indexed=False):
    """
    A cross operation over a vector on all pairs i, j such that i < j elements.
    :param lst: The vector.
    :param f: The lambda f of two integer variables.
    :param indexed: The lambda f of four integer variable, the 2 firsts are indexes.
    :return:
    """
    SatXEngine.apply_dual(lst, f, indexed)


def apply_different(lst: Sequence, f: Callable, indexed=False):
    """
    A cross operation over a vector on all pairs i, j such that i != j elements.
    :param lst: The vector.
    :param f: The lambda f of two integer variables.
    :param indexed: The lambda f of four integer variable, the 2 firsts are indexes.
    :return:
    """
    SatXEngine.apply_different(lst, f, indexed)


def all_different(args: Sequence):
    """
    The all different global constraint.
    :param args: A vector of integers.
    :return:
    """
    SatXEngine.all_different(args)


def all_out(args: Sequence, values: Sequence):
    """
    The all different to values global constraint.
    :param args: A vector of integers.
    :param values: The values excluded.
    :return:
    """
    SatXEngine.all_out(args, values)


def all_in(args: Sequence, values: Sequence[Unit | SupportsInt]):
    """
    The all in values global constraint.
    :param args: A vector of integers.
    :param values: The values included.
    :return:
    """
    get_engine().all_in(args, values)


def flatten(mtx):
    """
    Flatten a matrix into list.
    :param mtx: The matrix.
    :return: The entangled structure.
    """
    return SatXEngine.flatten(mtx)


def bits():
    """
    The current bits for the engine.
    :return: The bits
    """
    return get_engine().bits


def oo():
    """
    The infinite for the system, the maximal value for the current engine.
    :return: 2 ** bits - 1
    """
    return get_engine().oo


def element(item: Unit | SupportsInt, data: Sequence[Unit | SupportsInt]):
    """
    Ensure that the element i is on the data, on the position index.
    :param item: The element
    :param data: The data
    :return: The position of element
    """
    return get_engine().element(item, data)


def index(ith: Unit | SupportsInt, data: Sequence[Unit | SupportsInt]):
    """
    Ensure that the element i is on the data, on the position index.
    :param ith: The element
    :param data: The data
    :return: The position of element
    """
    return get_engine().index(ith, data)


def gaussian(x: Unit | None = None, y: Unit | None = None) -> Gaussian:
    """
    Create a gaussian integer from (x+yj).
    :param x: real
    :param y: imaginary
    :return: (x+yj)
    """
    return get_engine().gaussian(x, y)


def rational(x: Unit | None = None, y: Unit | None = None) -> Rational:
    """
    Create a rational x / y.
    :param x: numerator
    :param y: denominator
    :return: x / y
    """
    return get_engine().rational(x, y)


def at_most_k(x: Unit, k: int):
    """
    At most k bits can be activated for this integer.
    :param x: An integer.
    :param k: k elements
    :return: The encoded variable
    """
    return get_engine().at_most_k(x, k)


def sqrt(x: Unit | SupportsInt):
    """
    Define x as a perfect square.
    :param x: The integer
    :return: The square of this integer.
    """
    return get_engine().sqrt(x)


def reshape(lst, dimensions):
    """
    Reshape a list
    :param lst: The coherent list to reshape
    :param dimensions:  The list of dimensions
    :return: The reshaped list
    """
    return SatXEngine.reshape(lst, dimensions)


def tensor(dimensions: Sequence[int]):
    """
    Create a tensor
    :param dimensions: The list of dimensions
    :return: A tensor
    """
    return get_engine().tensor(dimensions)


def clear(lst: Sequence[Unit]):
    """
    Clear a list of integers, used with optimization routines.
    :param lst: The coherent list of integers to clear.
    """
    SatXEngine.clear(lst)


def rotate(x: Unit, k: Unit | int):
    """
    Rotate an integer k places
    :param x: the integer.
    :param k: k-places.
    :return: a rotated integer.
    """
    return get_engine().rotate(x, k)


def is_prime(p: Unit | SupportsInt):
    """
    Indicate that p is prime.
    :param p: the integer.
    """
    get_engine().is_prime(p)


def is_not_prime(p: Unit | SupportsInt):
    """
    Indicate that p is not prime.
    :param p: the integer.
    """
    get_engine().is_not_prime(p)


def satisfy(solver: str, params='', log=False):
    """
    Solve with external solver.
    :param solver: The external solver.
    :param params: Parameters passed to external solver.
    :return: True if SAT else False.
    """
    return get_engine().satisfy(solver, params, log)


def reset():
    get_engine().reset()
