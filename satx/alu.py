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
import warnings
from typing import Any, Callable, Sequence, SupportsInt, Tuple, cast

import numpy as np

from satx.formula import Atom, Literal
from satx.logicgate import LogicGateWriter
from satx.model import Model
from satx.unit import Unit


class ALU:
    """
    Arithmetic logic unit.

    Converts operations on Units and ints into logic gates.
    It may hold a Model, which gives Units concrete int values.
    """

    def __init__(self, gate_writer: LogicGateWriter, default_bits: int, default_signed=False):
        self.gate: LogicGateWriter = gate_writer
        self.default_bits: int = default_bits
        self.default_signed: bool = default_signed
        self.model: Model | None = None

    def maybe_int(self, arg: Unit | SupportsInt) -> int | None:
        if isinstance(arg, Unit):
            if self.model is not None:
                maybe_bits = [self.model.get(lit) for lit in arg.block]
                if any(b is None for b in maybe_bits):
                    return None
                bits = cast(list[int], maybe_bits)
                if arg.signed:
                    if len(bits) == 1:
                        # It makes no sense for a single bit to be signed.
                        return bits[0]
                    msb = bits.pop()
                    if msb == 1:
                        neg = [1 - b for b in bits]
                        binary = ''.join(map(str, reversed(neg)))
                        return -int(binary, 2) - 1
                binary = ''.join(map(str, reversed(bits)))
                return int(binary, 2)
            else:
                return None
        else:
            return int(arg)

    def integer(self, bits: int | None = None, signed: bool | None = None) -> Unit:
        if bits is None:
            bits = self.default_bits
        if signed is None:
            signed = self.default_signed
        block = self.gate.block(bits)
        return Unit(self, np.array(block, dtype=int), signed)

    def tensor(self, shape: Sequence[int], signed: bool | None = None) -> Unit:
        if signed is None:
            signed = self.default_signed
        block = self.gate.block(np.prod(shape, dtype=int))
        return Unit(self, np.array(block, dtype=int).reshape(shape), signed)

    def constant(self, value: int, bits: int | None = None, signed: bool | None = None) -> Unit:
        if bits is None:
            bits = self.default_bits
        if signed is None:
            signed = self.default_signed
        if value < 0 and not signed:
            warnings.warn("Coercion of a negative constant to an unsigned Unit.")
        bl = value.bit_length() if value >= 0 else (~value).bit_length()
        if (signed and bl > bits - 1) or (not signed and bl > bits):
            warnings.warn("Not enough bits to represent the constant")
        block = self.gate.constant(value, bits)
        return Unit(self, np.array(block, dtype=int), signed)

    def _func_with_int_fallback(self, block_func: Callable, int_func: Callable, *args: Unit | SupportsInt) -> Any:
        """
        Computes a function either on Units or ints.
        """
        args_as_ints = [a for a in args]
        for i, a in enumerate(args_as_ints):
            if isinstance(a, Unit):
                val = self.maybe_int(a)
                if val is not None:
                    args_as_ints[i] = val
        if all(isinstance(a, numbers.Integral) and not isinstance(a, Unit) for a in args_as_ints):
            return int_func(*(int(a) for a in args))
        args_maybe_as_units: list[Unit | SupportsInt] = [a for a in args]
        for i, a in enumerate(args_maybe_as_units):
            if isinstance(a, SupportsInt) and not isinstance(a, Unit):
                args_maybe_as_units[i] = self.constant(int(a))
        if not all(isinstance(a, Unit) for a in args_maybe_as_units):
            return NotImplemented
        args_as_units = cast(Sequence[Unit], args_maybe_as_units)
        return block_func(*(a.block for a in args_as_units))

    def _op_with_int_fallback(self, block_op: Callable, int_op: Callable,
                              *args: Unit | SupportsInt) -> Unit:
        """
        Computes an operation either on Units or ints.
        """

        def block_op_with_result(*block_args: Sequence[Literal]) -> Unit:
            result = self.integer(bits=len(block_args[0])) if len(block_args) == 1 else self.integer()
            block_op(*block_args, result.block)
            return result

        def int_op_to_const(*int_args: int) -> Unit:
            return self.constant(int_op(*int_args))

        return self._func_with_int_fallback(block_op_with_result, int_op_to_const, *args)

    def _cmp_with_int_fallback(self, assert_block_cmp: Callable, int_cmp: Callable,
                               *args: Unit | SupportsInt) -> bool:
        """
        Asserts a comparison on Units or computes a comparison ints.
        """

        def block_cmp_as_true(*block_args: Sequence[Literal]) -> bool:
            assert_block_cmp(*block_args)  # The gate has no result. It only asserts.
            return True  # Comparisons always return True on Units without a model.

        return self._func_with_int_fallback(block_cmp_as_true, int_cmp, *args)

    def _cmp_op_with_int_fallback(self, block_cmp: Callable, int_cmp: Callable,
                                  *args: Unit | SupportsInt) -> Unit:
        """
        Computes a comparison operation on Units or ints.
        """

        def int_cmp_to_bit(*int_args: int) -> Unit:
            res = int_cmp(*int_args)
            assert isinstance(res, bool)  # it's not NotImplemented
            return self.constant(1, bits=1) if res else self.constant(0, bits=1)

        def block_cmp_with_result(*block_args: Sequence[Literal]) -> Unit:
            result_bit = self.integer(bits=1)
            block_cmp(*block_args, result_bit)
            return result_bit

        return self._func_with_int_fallback(block_cmp_with_result, int_cmp_to_bit, *args)

    # binary operations:

    def add(self, lhs: Unit | SupportsInt, rhs: Unit | SupportsInt) -> Unit:
        return self._op_with_int_fallback(self.gate.add, operator.add, lhs, rhs)

    def sub(self, lhs: Unit | SupportsInt, rhs: Unit | SupportsInt) -> Unit:
        return self._op_with_int_fallback(self.gate.sub, operator.sub, lhs, rhs)

    def mul(self, lhs: Unit | SupportsInt, rhs: Unit | SupportsInt) -> Unit:
        return self._op_with_int_fallback(self.gate.mul, operator.mul, lhs, rhs)

    def pow(self, lhs: Unit | SupportsInt, rhs: Unit | SupportsInt) -> Unit:
        return self._op_with_int_fallback(self.gate.pow, operator.pow, lhs, rhs)

    def lshift(self, lhs: Unit | SupportsInt, rhs: Unit | SupportsInt) -> Unit:
        return self._op_with_int_fallback(self.gate.lshift, operator.lshift, lhs, rhs)

    def rshift(self, lhs: Unit | SupportsInt, rhs: Unit | SupportsInt) -> Unit:
        return self._op_with_int_fallback(self.gate.rshift, operator.rshift, lhs, rhs)

    def and_(self, lhs: Unit | SupportsInt, rhs: Unit | SupportsInt) -> Unit:
        return self._op_with_int_fallback(self.gate.and_, operator.and_, lhs, rhs)

    def xor(self, lhs: Unit | SupportsInt, rhs: Unit | SupportsInt) -> Unit:
        return self._op_with_int_fallback(self.gate.xor, operator.xor, lhs, rhs)

    def or_(self, lhs: Unit | SupportsInt, rhs: Unit | SupportsInt) -> Unit:
        return self._op_with_int_fallback(self.gate.or_, operator.or_, lhs, rhs)

    def floordiv(self, lhs: Unit | SupportsInt, rhs: Unit | SupportsInt) -> Unit:
        return self._op_with_int_fallback(self.gate.floordiv, operator.floordiv, lhs, rhs)

    def mod(self, lhs: Unit | SupportsInt, rhs: Unit | SupportsInt) -> Unit:
        return self._op_with_int_fallback(self.gate.mod, operator.mod, lhs, rhs)

    # other operations

    def neg(self, arg: Unit | SupportsInt) -> Unit:
        return self._op_with_int_fallback(self.gate.neg, operator.neg, arg)

    def abs(self, arg: Unit | SupportsInt) -> Unit:
        return self._op_with_int_fallback(self.gate.abs, operator.abs, arg)

    def invert(self, arg: Unit | SupportsInt) -> Unit:
        return self._op_with_int_fallback(self.gate.invert, operator.invert, arg)

    def mod_pow(self, base: Unit | SupportsInt, exponent: Unit | SupportsInt, modulus: Unit | SupportsInt) -> Unit:
        return self._op_with_int_fallback(self.gate.mod_pow, lambda b, e, m: pow(b, e, m), base, exponent, modulus)

    def truediv(self, lhs: Unit | SupportsInt, rhs: Unit | SupportsInt) -> Unit:
        def int_truediv(lhs_int: int, rhs_int: int) -> Unit:
            if lhs_int % rhs_int != 0:
                raise AssertionError
            return self.constant(lhs_int // rhs_int)

        def block_truediv(lhs_block: Sequence[Atom], rhs_block: Sequence[Atom]) -> Unit:
            result = self.integer()
            remainder = self.constant(0)
            self.gate.divmod(lhs_block, rhs_block, result.block, remainder.block)
            return result

        return self._func_with_int_fallback(block_truediv, int_truediv, lhs, rhs)

    def divmod(self, lhs: Unit | SupportsInt, rhs: Unit | SupportsInt) -> Tuple[Unit, Unit]:
        def int_divmod(lhs_int: int, rhs_int: int) -> Tuple[Unit, Unit]:
            return self.constant(lhs_int // rhs_int), self.constant(lhs_int % rhs_int)

        def block_divmod(lhs_block: Sequence[Atom], rhs_block: Sequence[Atom]) -> Tuple[Unit, Unit]:
            result = self.integer()
            remainder = self.integer()
            self.gate.divmod(lhs_block, rhs_block, result.block, remainder.block)
            return result, remainder

        return self._func_with_int_fallback(block_divmod, int_divmod, lhs, rhs)

    # comparison:

    def eq(self, lhs: Unit | SupportsInt, rhs: Unit | SupportsInt) -> bool:
        return self._cmp_with_int_fallback(self.gate.assert_eq, operator.eq, lhs, rhs)

    def ne(self, lhs: Unit | SupportsInt, rhs: Unit | SupportsInt) -> bool:
        return self._cmp_with_int_fallback(self.gate.assert_ne, operator.ne, lhs, rhs)

    def lt(self, lhs: Unit | SupportsInt, rhs: Unit | SupportsInt) -> bool:
        return self._cmp_with_int_fallback(self.gate.assert_gt, operator.gt, rhs, lhs)

    def le(self, lhs: Unit | SupportsInt, rhs: Unit | SupportsInt) -> bool:
        return self._cmp_with_int_fallback(self.gate.assert_le, operator.le, lhs, rhs)

    def gt(self, lhs: Unit | SupportsInt, rhs: Unit | SupportsInt) -> bool:
        return self._cmp_with_int_fallback(self.gate.assert_gt, operator.gt, lhs, rhs)

    def ge(self, lhs: Unit | SupportsInt, rhs: Unit | SupportsInt) -> bool:
        return self._cmp_with_int_fallback(self.gate.assert_le, operator.le, rhs, lhs)

    # extra stuff:

    def is_eq(self, lhs: Unit | SupportsInt, rhs: Unit | SupportsInt) -> Unit:
        return self._cmp_op_with_int_fallback(self.gate.eq, operator.eq, lhs, rhs)

    def is_ne(self, lhs: Unit | SupportsInt, rhs: Unit | SupportsInt) -> Unit:
        return self._cmp_op_with_int_fallback(self.gate.ne, operator.ne, lhs, rhs)

    def is_lt(self, lhs: Unit | SupportsInt, rhs: Unit | SupportsInt) -> Unit:
        return self._cmp_op_with_int_fallback(self.gate.gt, operator.gt, rhs, lhs)

    def is_le(self, lhs: Unit | SupportsInt, rhs: Unit | SupportsInt) -> Unit:
        return self._cmp_op_with_int_fallback(self.gate.le, operator.le, lhs, rhs)

    def is_gt(self, lhs: Unit | SupportsInt, rhs: Unit | SupportsInt) -> Unit:
        return self._cmp_op_with_int_fallback(self.gate.gt, operator.gt, lhs, rhs)

    def is_ge(self, lhs: Unit | SupportsInt, rhs: Unit | SupportsInt) -> Unit:
        return self._cmp_op_with_int_fallback(self.gate.le, operator.le, rhs, lhs)

    def at_most_k(self, arg: Unit, k: int):
        self.gate.at_most_k(arg.block, k)

    def binary(self, arg: Unit) -> list[bool] | None:
        if self.model is None:
            return None
        maybe_bits: list[int | None] = [self.model.get(lit) for lit in arg.block]
        if any(b is None for b in maybe_bits):
            return None
        bits = cast(list[bool], maybe_bits)
        for i, b in enumerate(bits):
            bits[i] = b != 0
        return bits

    def iff(self, lhs: Unit | SupportsInt, rhs: Unit | SupportsInt, lhs_condition: Unit | SupportsInt):
        def int_fallback(a, b, c):
            return a if c else b

        return self._op_with_int_fallback(self.gate.iff, int_fallback, lhs, rhs, lhs_condition)

    def one_hot(self, binary: Unit | SupportsInt, n: int | None = None):
        if n is None:
            n = 2 ** binary.bits if isinstance(binary, Unit) else 2 ** (int(binary).bit_length())
        n_not_none = cast(int, n)
        self.le(binary, n_not_none)

        def block_op_with_result(block: Sequence[Atom]) -> Unit:
            result = self.integer(n)
            self.gate.one_hot(block, result.block)
            return result

        def int_op_to_const(bin_as_int: int) -> Unit:
            return self.constant(1 << bin_as_int, n) if bin_as_int < n_not_none else self.constant(0, n)

        return self._func_with_int_fallback(block_op_with_result, int_op_to_const, binary)

    def mux_array(self, values: Sequence[Unit | SupportsInt], one_hot: Unit):
        value_blocks = tuple(v.block if isinstance(v, Unit) else self.constant(int(v)).block for v in values)
        result = self.integer()
        self.gate.mux_array(value_blocks, one_hot.block, result.block)
        return result
