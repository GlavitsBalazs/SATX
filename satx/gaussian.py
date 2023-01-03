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
from abc import abstractmethod, ABCMeta
from typing import TypeVar, Generic, cast, SupportsInt

from satx.unit import Unit

_TR = TypeVar('_TR', bound=numbers.Real)
_TC = TypeVar('_TC', bound=numbers.Complex, covariant=True)


class ComplexBase(numbers.Complex, Generic[_TR, _TC], metaclass=ABCMeta):
    __slots__ = ('_real', '_imag')

    def __init__(self, real: _TR, imag: _TR):
        self._real = real
        self._imag = imag

    @classmethod
    def from_re_im(cls, real: _TR, imag: _TR):
        return cls(real, imag)

    @property
    def real(self) -> _TR:
        return self._real

    @property
    def imag(self) -> _TR:
        return self._imag

    def __add__(self, other: numbers.Complex) -> _TC:
        return self.from_re_im(self.real + other.real, self.imag + other.imag)

    def __radd__(self, other: numbers.Complex) -> _TC:
        return self.from_re_im(other.real + self.real, other.imag + self.imag)

    def __neg__(self) -> _TC:
        return self.from_re_im(-self.real, -self.imag)

    def __pos__(self) -> _TC:
        return self.from_re_im(self.real, self.imag)

    def __mul__(self, other: numbers.Complex) -> _TC:
        return self.from_re_im((self.real * other.real) - (self.imag * other.imag),
                               ((self.real * other.imag) + (self.imag * other.real)))

    def __rmul__(self, other: numbers.Complex) -> _TC:
        return self.from_re_im((other.real * self.real) - (other.imag * self.imag),
                               ((other.real * self.imag) + (other.imag * self.real)))

    def __truediv__(self, other: numbers.Complex) -> _TC:
        divisor_abs_squared = other.real * other.real + other.imag * other.imag
        return self.from_re_im(((self.real * other.real) + (self.imag * other.imag)) / divisor_abs_squared,
                               ((self.imag * other.real) - (self.real * other.imag)) / divisor_abs_squared)

    def __rtruediv__(self, other: numbers.Complex) -> _TC:
        divisor_abs_squared = self.real * self.real + self.imag * self.imag
        return self.from_re_im(((other.real * self.real) + (other.imag * self.imag)) / divisor_abs_squared,
                               ((other.imag * self.real) - (other.real * self.imag)) / divisor_abs_squared)

    def __pow__(self, exponent: numbers.Complex) -> _TC:
        return self._pow(self, exponent)

    def __rpow__(self, base: numbers.Complex) -> _TC:
        return self._pow(base, self)

    def _pow(self, base: numbers.Complex, exponent: numbers.Complex) -> _TC:
        int_exponent: int | None = None
        try:
            int_exponent = int(cast(SupportsInt, exponent))
        except (TypeError, ValueError):
            pass
        if int_exponent is not None:
            res = base
            for _ in range(int_exponent - 1):
                res *= base
            return self.from_re_im(res.real, res.imag)
        else:
            return self._complex_pow(base, exponent)

    @abstractmethod
    def _complex_pow(self, base: numbers.Complex, exponent: numbers.Complex) -> _TC:
        raise NotImplementedError

    @abstractmethod
    def _real_sqrt(self, arg: _TR) -> _TR:
        raise NotImplementedError

    def __abs__(self) -> _TR:
        return self._real_sqrt((self.real * self.real + self.imag * self.imag).real)

    def conjugate(self) -> _TC:
        return self.from_re_im(self.real, -self.imag)

    def __complex__(self) -> complex:
        return complex(float(self.real), float(self.imag))

    def __repr__(self):
        return f'{self.real}+{self.imag}j'

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, numbers.Complex):
            return False
        other_complex = cast(numbers.Complex, other)
        real_eq = self.real == other_complex.real
        imag_eq = self.imag == other_complex.imag
        return real_eq and imag_eq  # do both comparisons, to prevent short-circuiting

    def __ne__(self, other: object) -> bool:
        if not isinstance(other, numbers.Complex):
            return True
        other_complex = cast(numbers.Complex, other)
        real_ne = self.real != other_complex.real
        imag_ne = self.imag != other_complex.imag
        return real_ne or imag_ne  # do both comparisons, to prevent short-circuiting

    def __hash__(self) -> int:
        return hash((self.real, self.imag))


class Gaussian(ComplexBase[Unit, 'Gaussian']):
    def _complex_pow(self, base: numbers.Complex, exponent: numbers.Complex):
        raise NotImplementedError("Irrational numbers are not supported.")

    def __pow__(self, exponent: int) -> 'Gaussian':  # type: ignore
        return cast(Gaussian, super().__pow__(cast(numbers.Complex, exponent)))

    def _real_sqrt(self, arg: Unit) -> Unit:
        y = arg.alu.integer()
        assert arg == y * y
        return y
