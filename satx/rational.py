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
import operator
from abc import ABCMeta, abstractmethod
from typing import TypeVar, Generic, cast, SupportsInt

from satx.unit import Unit

_TZ = TypeVar('_TZ', bound=numbers.Integral)
_TQ = TypeVar('_TQ', bound=numbers.Rational, covariant=True)


class RationalBase(numbers.Rational, Generic[_TZ, _TQ], metaclass=ABCMeta):
    __slots__ = ('_numerator', '_denominator')

    def __init__(self, numerator: _TZ, denominator: _TZ):
        assert denominator != 0
        self._numerator = numerator
        self._denominator = denominator

    @classmethod
    def divide(cls, numerator: _TZ, denominator: _TZ):
        # subclasses may normalize by the GCD.
        return cls(numerator, denominator)

    @property
    def numerator(self) -> _TZ:  # type: ignore
        return self._numerator

    @property
    def denominator(self) -> _TZ:  # type: ignore
        return self._denominator

    def __trunc__(self) -> _TZ:  # type: ignore
        return self.__ceil__() if self.numerator < 0 else self.__floor__()

    def __floor__(self) -> _TZ:  # type: ignore
        return self.numerator // self.denominator  # type: ignore

    def __ceil__(self) -> _TZ:  # type: ignore
        return -((-self.numerator) // self.denominator)

    def __round__(self, ndigits: int | None = None) -> _TQ:  # type: ignore
        raise NotImplementedError  # todo

    def __floordiv__(self, other: numbers.Rational) -> _TZ:  # type: ignore
        return (self.numerator * other.denominator) // (other.numerator * self.denominator)

    def __rfloordiv__(self, other: numbers.Rational) -> _TZ:  # type: ignore
        return (other.numerator * self.denominator) // (self.numerator * other.denominator)

    def __mod__(self, other: numbers.Rational) -> _TQ:
        return self.divide((self.numerator * other.denominator) % (other.numerator * self.denominator),
                           self.denominator * other.denominator)

    def __rmod__(self, other: numbers.Rational) -> _TQ:
        return self.divide((other.numerator * self.denominator) % (self.numerator * other.denominator),
                           other.denominator * self.denominator)

    def __add__(self, other: numbers.Rational) -> _TQ:
        return self.divide(self.denominator * other.numerator + self.numerator * other.denominator,
                           self.denominator * other.denominator)

    def __radd__(self, other: numbers.Rational) -> _TQ:
        return self.divide(other.denominator * self.numerator + other.numerator * self.denominator,
                           other.denominator * self.denominator)

    def __sub__(self, other: numbers.Rational) -> _TQ:
        return self.divide(self.denominator * other.numerator - self.numerator * other.denominator,
                           self.denominator * other.denominator)

    def __rsub__(self, other: numbers.Rational) -> _TQ:
        return self.divide(other.denominator * self.numerator - other.numerator * self.denominator,
                           other.denominator * self.denominator)

    def __neg__(self) -> _TQ:
        return self.divide(-self.numerator, self.denominator)

    def __pos__(self) -> _TQ:
        return self.divide(self.numerator, self.denominator)

    def __mul__(self, other: numbers.Rational) -> _TQ:
        return self.divide(self.numerator * other.numerator, self.denominator * other.denominator)

    def __rmul__(self, other: numbers.Rational) -> _TQ:
        return self.divide(other.numerator * self.numerator, other.denominator * self.denominator)

    def __truediv__(self, other: numbers.Rational) -> _TQ:
        return self.divide(self.numerator * other.denominator, other.numerator * self.denominator)

    def __rtruediv__(self, other: numbers.Rational) -> _TQ:
        return self.divide(other.numerator * self.denominator, self.numerator * other.denominator)

    def __pow__(self, exponent: numbers.Rational) -> numbers.Complex:
        return self._pow(self, exponent)

    def __rpow__(self, base: numbers.Rational) -> numbers.Complex:
        return self._pow(base, self)

    def _pow(self, base: numbers.Rational, exponent: numbers.Rational) -> numbers.Complex:
        int_exponent: int | None = None
        try:
            int_exponent = int(cast(SupportsInt, exponent))
        except TypeError:
            pass
        if int_exponent is not None:
            res = base
            for _ in range(int_exponent - 1):
                res *= base
            return res
        else:
            return self._complex_pow(base, exponent)

    @abstractmethod
    def _complex_pow(self, base: numbers.Rational, exponent: numbers.Rational) -> numbers.Complex:
        raise NotImplementedError

    def __abs__(self) -> _TQ:
        return self.divide(cast(_TZ, abs(self.numerator)), cast(_TZ, abs(self.denominator)))

    @staticmethod
    def _compare(cmp, lhs: numbers.Rational, rhs: numbers.Rational) -> bool:
        if not isinstance(lhs, numbers.Rational) or not isinstance(rhs, numbers.Rational):
            return NotImplemented
        return cmp(lhs.numerator * rhs.denominator, rhs.numerator * lhs.denominator)

    def __lt__(self, other: numbers.Rational) -> bool:
        return self._compare(operator.lt, self, other)

    def __gt__(self, other: numbers.Rational) -> bool:
        return self._compare(operator.gt, self, other)

    def __le__(self, other: numbers.Rational) -> bool:
        return self._compare(operator.le, self, other)

    def __ge__(self, other: numbers.Rational) -> bool:
        return self._compare(operator.ge, self, other)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, numbers.Rational):
            return False
        return self._compare(operator.eq, self, other)

    def __ne__(self, other: object) -> bool:
        if not isinstance(other, numbers.Rational):
            return True
        return self._compare(operator.ne, self, other)

    def __hash__(self) -> int:
        return hash((self.numerator, self.denominator))


class Rational(RationalBase[Unit, 'Rational']):
    def __trunc__(self) -> Unit:  # type: ignore
        ceil = self.__ceil__()
        floor = self.__floor__()
        alu = self.numerator.alu
        return alu.iff(ceil, floor, alu.is_lt(self.numerator, 0))

    def __round__(self, ndigits: int | None = None) -> 'Rational':  # type: ignore
        raise NotImplementedError  # todo

    def __pow__(self, exponent: int) -> 'Rational':  # type: ignore
        return cast(Rational, super().__pow__(cast(numbers.Rational, exponent)))

    def _complex_pow(self, base: numbers.Rational, exponent: numbers.Rational) -> numbers.Complex:
        raise NotImplementedError("Irrational numbers are not supported.")
