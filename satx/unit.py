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

import numbers
from typing import Union, Sequence, SupportsInt, SupportsIndex, Tuple, cast, TYPE_CHECKING

import numpy as np

from satx.formula import Atom

if TYPE_CHECKING:
    from satx.alu import ALU


class Unit(numbers.Integral):
    """
    A Unit is an integer made from a block of atoms. The i-th atom encodes the i-th bit: True = 1, False = 0.
    It behaves similarly to an int, except that all operations done with it are proxied to an ALU object.
    The ALU will determine the exact value of the atoms and so the Unit.
    """

    __slots__ = ('alu', 'block_arr', 'signed')

    def __init__(self, alu: 'ALU', block_arr: np.ndarray, signed=True):
        """
        :param alu: The element
        :param data: The data
        :return: The position of element
        """
        self.alu = alu
        self.block_arr = block_arr
        self.signed = signed

    # We don't have equality comparisons in the traditional sense. There is no reason to allow hashing.
    __hash__ = None  # type: ignore

    @property
    def block(self) -> Sequence[Atom]:
        return list(self.block_arr.flat)  # todo: do we really need to copy this?

    @property
    def bits(self) -> int:
        return len(self.block)

    def __int__(self) -> int:
        val = self.alu.maybe_int(self)
        if val is None:
            raise ValueError("Unit without a model has no value.")
        return val

    @property
    def value(self) -> int | None:
        return self.alu.maybe_int(self)

    # Binary operations:

    def __add__(self, other: Union['Unit', SupportsInt]) -> 'Unit':
        return self.alu.add(self, other)

    def __radd__(self, other: Union['Unit', SupportsInt]) -> 'Unit':
        return self.alu.add(other, self)

    def __sub__(self, other: Union['Unit', SupportsInt]) -> 'Unit':
        return self.alu.sub(self, other)

    def __rsub__(self, other: Union['Unit', SupportsInt]) -> 'Unit':
        return self.alu.sub(other, self)

    def __mul__(self, other: Union['Unit', SupportsInt]) -> 'Unit':
        return self.alu.mul(self, other)

    def __rmul__(self, other: Union['Unit', SupportsInt]) -> 'Unit':
        return self.alu.mul(other, self)

    # numbers.pyi says that floordiv should return an int. This is wrong. Any Integral may be returned.
    def __floordiv__(self, other: Union['Unit', SupportsInt]) -> 'Unit':  # type: ignore
        return self.alu.floordiv(self, other)

    def __rfloordiv__(self, other: Union['Unit', SupportsInt]) -> 'Unit':  # type: ignore
        return self.alu.floordiv(other, self)

    def __truediv__(self, other: Union['Unit', SupportsInt]) -> 'Unit':
        return self.alu.truediv(self, other)

    def __rtruediv__(self, other: Union['Unit', SupportsInt]) -> 'Unit':
        return self.alu.truediv(other, self)

    def __mod__(self, other: Union['Unit', SupportsInt]) -> 'Unit':
        return self.alu.mod(self, other)

    def __rmod__(self, other: Union['Unit', SupportsInt]) -> 'Unit':
        return self.alu.mod(other, self)

    def __lshift__(self, other: Union['Unit', SupportsInt]) -> 'Unit':
        return self.alu.lshift(self, other)

    def __rlshift__(self, other: Union['Unit', SupportsInt]) -> 'Unit':
        return self.alu.lshift(other, self)

    def __rshift__(self, other: Union['Unit', SupportsInt]) -> 'Unit':
        return self.alu.rshift(self, other)

    def __rrshift__(self, other: Union['Unit', SupportsInt]) -> 'Unit':
        return self.alu.rshift(other, self)

    def __and__(self, other: Union['Unit', SupportsInt]) -> 'Unit':
        return self.alu.and_(self, other)

    def __rand__(self, other: Union['Unit', SupportsInt]) -> 'Unit':
        return self.alu.and_(other, self)

    def __xor__(self, other: Union['Unit', SupportsInt]) -> 'Unit':
        return self.alu.xor(self, other)

    def __rxor__(self, other: Union['Unit', SupportsInt]) -> 'Unit':
        return self.alu.xor(other, self)

    def __or__(self, other: Union['Unit', SupportsInt]) -> 'Unit':
        return self.alu.or_(self, other)

    def __ror__(self, other: Union['Unit', SupportsInt]) -> 'Unit':
        return self.alu.or_(other, self)

    # Other operations:

    def __pow__(self, exponent: Union['Unit', SupportsInt],
                modulus: Union['Unit', SupportsInt] | None = None) -> 'Unit':
        if modulus is None:
            return self.alu.pow(self, exponent)
        else:
            return self.alu.mod_pow(self, exponent, modulus)

    def __rpow__(self, base: Union['Unit', SupportsInt]) -> 'Unit':
        return self.alu.pow(base, self)

    def __abs__(self) -> 'Unit':
        return self.alu.abs(self)

    def __neg__(self) -> 'Unit':
        return self.alu.neg(self)

    def __invert__(self) -> 'Unit':
        return self.alu.invert(self)

    def __divmod__(self, other: Union['Unit', SupportsInt]) -> Tuple['Unit', 'Unit']:
        return self.alu.divmod(self, other)

    def __rdivmod__(self, other: Union['Unit', SupportsInt]) -> Tuple['Unit', 'Unit']:
        return self.alu.divmod(other, self)

    # Comparisons.
    # Note that they always return True on Units without a model.

    def __bool__(self) -> bool:
        val = self.alu.maybe_int(self)
        if val is None:
            return True
        else:
            return bool(val)

    def __lt__(self, other: Union['Unit', SupportsInt]) -> bool:
        return self.alu.lt(self, other)

    def __gt__(self, other: Union['Unit', SupportsInt]) -> bool:
        return self.alu.gt(self, other)

    def __le__(self, other: Union['Unit', SupportsInt]) -> bool:
        return self.alu.le(self, other)

    def __ge__(self, other: Union['Unit', SupportsInt]) -> bool:
        return self.alu.ge(self, other)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, (Unit, SupportsInt)):
            return False
        return self.alu.eq(self, cast(Union['Unit', SupportsInt], other))

    def __ne__(self, other: object) -> bool:
        if not isinstance(other, (Unit, SupportsInt)):
            return True
        return self.alu.ne(self, cast(Union['Unit', SupportsInt], other))

    # Boilerplate:

    def __pos__(self) -> 'Unit':
        return self

    def __index__(self) -> int:
        return self.__int__()

    # numbers.pyi says that these methods should return ints. This is wrong. Any Integral may be returned.

    def __round__(self, ndigits: int | None = None) -> 'Unit':  # type: ignore
        return self

    def __trunc__(self) -> 'Unit':  # type: ignore
        return self

    def __floor__(self) -> 'Unit':  # type: ignore
        return self

    def __ceil__(self) -> 'Unit':  # type: ignore
        return self

    @property
    def numerator(self) -> 'Unit':  # type: ignore
        return self

    @property
    def denominator(self) -> 'Unit':  # type: ignore
        return self.alu.constant(1, self.bits)

    def conjugate(self) -> 'Unit':
        return self

    def __float__(self) -> float:
        return float(self.__int__())

    def __complex__(self) -> complex:
        return complex(self.__int__())

    @property
    def real(self) -> 'Unit':
        return self

    @property
    def imag(self) -> 'Unit':
        return self.alu.constant(0, self.bits)

    # Convenience methods, not required by numbers.Integral:

    def is_in(self, items: Sequence[Union['Unit', SupportsInt]]):
        bits = self.alu.integer(bits=len(items))
        assert sum(self.alu.iff(0, 1, bits[i]) for i in range(len(items))) == 1
        assert sum(self.alu.iff(0, items[i], bits[i]) for i in range(len(items))) == self
        return self

    def is_not_in(self, items: Sequence[Union['Unit', SupportsInt]]):
        for element in items:
            assert self != element
        return self

    def iff(self, bit: Union['Unit', SupportsInt], other: Union['Unit', SupportsInt]) -> 'Unit':
        return self.alu.iff(lhs=self, rhs=other, lhs_condition=bit)

    def __getitem__(self, *args) -> 'Unit':
        if len(args) == 1 and isinstance(args[0], SupportsIndex) and not isinstance(args[0], Unit):
            item: Atom = self.block[args[0].__index__()]
            sub_block = np.array([item], dtype=int)
        elif len(args) == 1 and isinstance(args[0], list):
            item = self.block[np.ravel_multi_index(args[0], self.block_arr.shape).__index__()]
            sub_block = np.array([item], dtype=int)
        else:
            sub_block = self.block_arr.__getitem__(*args)
        return Unit(self.alu, sub_block, self.signed)

    def __call__(self, other: Union['Unit', SupportsInt], value: Union['Unit', SupportsInt]):
        return self.alu.iff(lhs=value, rhs=other, lhs_condition=self)

    @property
    def binary(self):
        res = self.alu.binary(self)
        if res is None:
            return None
        return np.array(res).reshape(self.block_arr.shape)

    def __repr__(self):
        return str(self.value)

    def __str__(self):
        return self.__repr__()

    def clear(self):
        raise NotImplementedError("Units are immutable.")

    def reverse(self, copy=False):
        if copy:
            # np.flip doesn't actually copy the underlying memory
            return Unit(self.alu, np.flip(self.block_arr), self.signed)
        else:
            raise NotImplementedError("Units must be copied when reversing.")
