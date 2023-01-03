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

from typing import Sequence, SupportsInt, Callable

import satx
from satx.unit import Unit

"""
The Global Constraint Catalog implementation for SAT-X.
"""


def abs_val(x, y):
    """
    Enforce the fact that the first variable is equal to the absolute value of the second variable.
    """
    satx.SatXEngine.abs_val(x, y)


def all_differ_from_at_least_k_pos(k: Unit | SupportsInt, lst: Sequence[Sequence[Unit | SupportsInt]]):
    """
    Enforce all pairs of distinct vectors of the VECTORS collection to differ from at least K positions.
    """
    return satx.get_engine().all_differ_from_at_least_k_pos(k, lst)


def all_differ_from_at_most_k_pos(k: Unit | SupportsInt, lst: Sequence[Sequence[Unit | SupportsInt]]):
    """
    Enforce all pairs of distinct vectors of the VECTORS collection to differ from at most K positions.
    """
    return satx.get_engine().all_differ_from_at_least_k_pos(k, lst)


def all_differ_from_exactly_k_pos(k: Unit | SupportsInt, lst: Sequence[Sequence[Unit | SupportsInt]]):
    """
    Enforce all pairs of distinct vectors of the VECTORS collection to differ from exactly K positions.
    Enforce K = 0 when |VECTORS| < 2.
    """
    satx.get_engine().all_differ_from_exactly_k_pos(k, lst)


def all_equal(lst: Sequence):
    """
    Enforce all variables of the collection
    """
    satx.SatXEngine.all_equal(lst)


def all_different(lst: Sequence):
    """
    Enforce all variables of the collection 𝚅𝙰𝚁𝙸𝙰𝙱𝙻𝙴𝚂 to take distinct values.
    """
    satx.SatXEngine.all_different(lst)


def element(idx: Unit | SupportsInt, lst: Sequence[Unit | SupportsInt], val: Unit | SupportsInt):
    """
    𝚅𝙰𝙻𝚄𝙴 is equal to the 𝙸𝙽𝙳𝙴𝚇-th item of 𝚃𝙰𝙱𝙻𝙴, i.e. 𝚅𝙰𝙻𝚄𝙴 = 𝚃𝙰𝙱𝙻𝙴[𝙸𝙽𝙳𝙴𝚇].
    """
    satx.get_engine().assert_element(idx, lst, val)


def gcd(x: Unit | int, y: Unit | int, z: Unit | int):
    """
    Enforce the fact that 𝚉 is the greatest common divisor of 𝚇 and 𝚈. (assume X <= Y)
    """
    satx.SatXEngine.gcd(x, y, z)


def sort(lst1: Sequence[Unit | SupportsInt], lst2: Sequence[Unit | SupportsInt]):
    """
    First, the variables of the collection 𝚅𝙰𝚁𝙸𝙰𝙱𝙻𝙴𝚂 2 correspond to a permutation of the variables of 𝚅𝙰𝚁𝙸𝙰𝙱𝙻𝙴𝚂 1.
    Second, the variables of 𝚅𝙰𝚁𝙸𝙰𝙱𝙻𝙴𝚂 2 are sorted in increasing order.
    """
    satx.get_engine().sort(lst1, lst2)


def sort_permutation(lst_from: Sequence[Unit | SupportsInt], lst_per: Sequence[Unit | SupportsInt],
                     lst_to: Sequence[Unit | SupportsInt]):
    """
    The variables of collection 𝙵𝚁𝙾𝙼 correspond to the variables of collection 𝚃𝙾 according to the permutation
    𝙿𝙴𝚁𝙼𝚄𝚃𝙰𝚃𝙸𝙾𝙽 (i.e., 𝙵𝚁𝙾𝙼[i].𝚟𝚊𝚛=𝚃𝙾[𝙿𝙴𝚁𝙼𝚄𝚃𝙰𝚃𝙸𝙾𝙽[i].𝚟𝚊𝚛].𝚟𝚊𝚛). The variables of
    collection 𝚃𝙾 are also sorted in increasing order.
    """
    satx.get_engine().sort_permutation(lst_from, lst_per, lst_to)


def count(val: Unit | int, lst: Sequence[Unit | int], rel: Callable, lim: Unit | SupportsInt):
    """
    Let N be the number of variables of the 𝚅𝙰𝚁𝙸𝙰𝙱𝙻𝙴𝚂 collection assigned to value 𝚅𝙰𝙻𝚄𝙴;
    Enforce condition N 𝚁𝙴𝙻𝙾𝙿 𝙻𝙸𝙼𝙸𝚃 to hold.
    """
    satx.get_engine().count(val, lst, rel, lim)
