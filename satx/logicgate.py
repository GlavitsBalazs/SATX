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
import itertools
import operator
from typing import Sequence, Iterable, Callable, Tuple, cast

from satx.formula import FormulaWriter, Atom, Literal


class LogicGateWriter:
    """
    Converts logic gates into CNF clauses.
    """

    def __init__(self, form: FormulaWriter):
        self.form = form
        self.true = self.form.add_atom()
        self.false = self.form.add_atom()
        self.form.add_clause((self.true,))
        self.form.add_clause((Literal(-self.false),))

    def block(self, n: int) -> Tuple[Atom, ...]:
        return tuple(self.form.add_atom() for _ in range(n))

    @staticmethod
    def int2bin(value: int, bits: int) -> list[int]:
        res = list(0 for _ in range(bits))
        for i in range(bits):
            if value == 0:
                break
            if value % 2 != 0:
                res[i] = 1
            value //= 2  # >>= 1
        return res

    def constant(self, value: int, bits: int) -> Tuple[Atom, ...]:
        return tuple(self.true if b != 0 else self.false for b in self.int2bin(value, bits))

    # There are only four "primitive" logic gates that add clauses to the formula: or, and, xor, full adder.
    # All others are derived from these.

    def bit_or(self, *args: Literal, result: Literal):
        if any(self.form.get_constant(lit) is True for lit in args):
            self.form.add_clause((result,))
            return
        if len(args) == 0:
            self.form.add_clause((result,))
            return
        self.form.add_clause(args + (Literal(-result),))
        for lit in args:
            self.form.add_clause((Literal(-lit), result))

    def bit_and(self, *args: Literal, result: Literal):
        if any(self.form.get_constant(lit) is False for lit in args):
            self.form.add_clause((Literal(-result),))
            return
        if len(args) == 0:
            self.form.add_clause((Literal(-result),))
            return
        neg_args = tuple(Literal(-b) for b in args)
        self.form.add_clause(neg_args + (result,))
        for lit in args:
            self.form.add_clause((lit, Literal(-result)))

    def bit_xor(self, *args: Literal, result: Literal):
        variable_args: list[Literal] = list()
        constant_args: list[Literal] = list()
        for arg in args:
            (constant_args if self.form.get_constant(arg) is not None else variable_args).append(arg)
        constant_xor = functools.reduce(operator.xor, (self.form.get_constant(lit) for lit in constant_args), False)
        if constant_xor:
            result = Literal(-result)
        args = tuple(variable_args)
        for n in range(2 ** len(args)):
            bits = self.int2bin(n, len(args))
            pop_cnt = sum(bits)
            clause = tuple(Literal(-lit) if b != 0 else lit for b, lit in zip(bits, args))
            self.form.add_clause(clause + (result if pop_cnt % 2 != 0 else Literal(-result),))

    def full_adder(self, lhs: Literal, rhs: Literal, carry_in: Literal, carry_out: Literal, result: Literal):
        clauses = [
            (lhs, rhs, -carry_out),
            (lhs, carry_in, -carry_out),
            (lhs, -carry_out, -result),
            (rhs, carry_in, -carry_out),
            (rhs, -carry_out, -result),
            (-lhs, carry_out, result),
            (-lhs, -rhs, carry_out),
            (-lhs, -carry_in, carry_out),
            (-rhs, carry_out, result),
            (-rhs, -carry_in, carry_out),
            (carry_in, -carry_out, -result),
            (-carry_in, carry_out, result),
            (lhs, rhs, carry_in, -result),
            (-lhs, -rhs, -carry_in, result)
        ]
        for c in clauses:
            self.form.add_clause(tuple(cast(Literal, lit) for lit in c))

    # Ideas for more primitive logic gates: equivalent, majority,
    # https://graphics.stanford.edu/~seander/bithacks.html,
    # at least k, at most k, exactly k, k consecutive, pop count, etc.

    @staticmethod
    def block_gate(bit_gate: Callable, lhs: Sequence[Literal], rhs: Sequence[Literal],
                   result: Sequence[Literal]):
        for lhs_lit, rhs_lit, ol in zip(lhs, rhs, result):
            bit_gate(lhs_lit, rhs_lit, result=ol)

    def and_(self, lhs: Sequence[Literal], rhs: Sequence[Literal], result: Sequence[Literal]):
        self.block_gate(self.bit_and, lhs, rhs, result)

    def or_(self, lhs: Sequence[Literal], rhs: Sequence[Literal], result: Sequence[Literal]):
        self.block_gate(self.bit_or, lhs, rhs, result)

    def xor(self, lhs: Sequence[Literal], rhs: Sequence[Literal], result: Sequence[Literal]):
        self.block_gate(self.bit_xor, lhs, rhs, result)

    def add(self, lhs: Sequence[Literal], rhs: Sequence[Literal], result: Sequence[Literal],
            carry_in_lit: Literal | None = None, carry_out_lit: Literal | None = None):

        if carry_in_lit is None:
            carry_in_lit = self.false
        if carry_out_lit is None:
            carry_out_lit = self.false

        # ripple-carry adder
        width = min(len(lhs), len(rhs))
        carry: list[Literal] = list(self.block(width - 1))
        carry.insert(0, carry_in_lit)
        carry.append(carry_out_lit)
        for i in range(0, width):
            self.full_adder(lhs[i], rhs[i], carry[i], carry[i + 1], result[i])

    def sub(self, lhs: Sequence[Literal], rhs: Sequence[Literal], result: Sequence[Literal]):
        non_negative = self.form.add_atom()  # the sign of the output
        inv_rhs = [Literal(-x) for x in rhs]
        self.add(lhs=lhs, rhs=inv_rhs, result=result,
                 carry_in_lit=self.true, carry_out_lit=non_negative)

    def mul_with_overflow(self, lhs: Sequence[Literal], rhs: Sequence[Literal],
                          result: Sequence[Literal], overflow: Literal | None = None):
        # todo: make this do signed multiplication
        width = len(lhs)
        partial_products = [(result[0],) + self.block(width - 1)]
        self.and_(rhs, [lhs[0]] * width, partial_products[0])
        for i in range(1, width):
            pp_i = self.block(width)
            if overflow is not None:
                self.and_(rhs, [lhs[i]] * width, result=pp_i)
            else:
                self.and_(rhs[0:width - i], [lhs[i]] * (width - i), result=pp_i)
            partial_products.append(pp_i)

        partial_sums = [(result[i],) + self.block(width - i - 1) for i in range(1, width)]
        csc = self.block(width - 1)
        cps = partial_products[0][1:width]
        for i in range(1, width):
            cpp = partial_products[i][0:width - i]
            psa = partial_sums[i - 1]
            assert len(cps) == width - i
            self.add(lhs=cps, rhs=cpp, result=psa, carry_in_lit=self.false, carry_out_lit=csc[i - 1])
            cps = psa[1:]
        if overflow is not None:
            ow: list[Literal] = list(csc)
            for i in range(1, width):
                ow += (partial_products[i][width - i:width])
            self.bit_or(*ow, result=overflow)

    def mul(self, lhs: Sequence[Literal], rhs: Sequence[Literal], result: Sequence[Literal]):
        self.mul_with_overflow(lhs, rhs, result, overflow=self.false)

    def le(self, lhs: Sequence[Literal], rhs: Sequence[Literal], result: Literal):
        if len(lhs) == 1:
            self.bit_and(lhs[0], Literal(-rhs[0]), result=Literal(-result))
            return
        width = len(lhs)
        rl = self.form.add_atom()
        self.le(lhs[:width - 1], rhs[:width - 1], result=rl)
        lhs_msb, rhs_msb = lhs[width - 1], rhs[width - 1]
        msb_is_lt = self.form.add_atom()
        self.bit_and(Literal(-lhs_msb), rhs_msb, result=msb_is_lt)
        msb_is_eq = self.form.add_atom()
        self.bit_xor(lhs_msb, rhs_msb, result=Literal(-msb_is_eq))
        leq_if_first_is_eq = self.form.add_atom()
        self.bit_and(msb_is_eq, rl, result=leq_if_first_is_eq)
        self.bit_or(msb_is_lt, leq_if_first_is_eq, result=result)

    def eq(self, lhs: Sequence[Literal], rhs: Sequence[Literal], result: Literal):
        xor = self.block(len(lhs))
        self.xor(lhs, rhs, result=xor)
        self.bit_or(*xor, result=Literal(-result))

    def mux_binary(self, lhs: Sequence[Literal], rhs: Sequence[Literal],
                   lhs_condition: Literal, result: Sequence[Literal]):
        # multiplexer
        if self.form.get_constant(lhs_condition) is True:
            self.eq(result, lhs, self.true)
            return
        if self.form.get_constant(lhs_condition) is False:
            self.eq(result, rhs, self.true)
            return
        lhs_s = self.block(len(lhs))
        rhs_s = self.block(len(rhs))
        self.and_(lhs=lhs, rhs=[lhs_condition] * len(lhs), result=lhs_s)
        self.and_(lhs=rhs, rhs=[Literal(-lhs_condition)] * len(rhs), result=rhs_s)
        self.or_(lhs=lhs_s, rhs=rhs_s, result=result)

    def iff(self, lhs: Sequence[Literal], rhs: Sequence[Literal],
            lhs_condition: Sequence[Literal], result: Sequence[Literal]):
        lhs_condition_bit = self.form.add_atom()
        self.bit_and(*lhs_condition, result=lhs_condition_bit)
        self.mux_binary(lhs, rhs, lhs_condition_bit, result)

    def _staggered_or_gate(self, *args: Literal, result: Sequence[Literal]):
        width = len(args)
        self.bit_or(args[-1], result=result[-1])
        for idx in reversed(range(width - 1)):
            self.bit_or(args[idx], result[idx + 1], result=result[idx])

    def divmod(self, lhs: Sequence[Literal], rhs: Sequence[Literal],
               result_quotient: Sequence[Literal] | None = None,
               result_remainder: Sequence[Literal] | None = None):
        width = len(lhs)
        dnz = self.block(len(rhs))
        self._staggered_or_gate(*rhs, result=dnz)
        qt = self.block(width)
        rem: Tuple[Literal, ...] = tuple()
        for step_idx in reversed(range(0, width)):
            rem = (lhs[step_idx],) + rem
            if len(rem) == len(rhs):
                self.le(lhs=rhs, rhs=rem, result=qt[step_idx])
            else:
                lbc = self.form.add_atom()
                self.le(lhs=rhs[0:len(rem)], rhs=rem, result=lbc)
                hbc = dnz[len(rem)]
                self.bit_and(lbc, Literal(-hbc), result=qt[step_idx])
            rmd = self.block(len(rem))
            self.sub(rem, rhs[0:len(rem)], rmd)
            rem2 = self.block(len(rem))
            self.mux_binary(lhs=rmd, rhs=rem, lhs_condition=qt[step_idx], result=rem2)
            rem = rem2
        rhs_is_zero = self.form.add_atom()
        self.bit_or(*rhs, result=Literal(-rhs_is_zero))
        if result_quotient is not None:
            self.and_(lhs=([Literal(-rhs_is_zero)] * width), rhs=qt, result=result_quotient)
        if result_remainder is not None:
            self.and_(lhs=([Literal(-rhs_is_zero)] * width), rhs=rem, result=result_remainder)

    def floordiv(self, lhs: Sequence[Literal], rhs: Sequence[Literal], result: Sequence[Literal]):
        self.divmod(lhs, rhs, result_quotient=result, result_remainder=None)

    def mod(self, lhs: Sequence[Literal], rhs: Sequence[Literal], result: Sequence[Literal]):
        self.divmod(lhs, rhs, result_quotient=None, result_remainder=result)

    def one_hot(self, binary: Sequence[Literal], oh: Sequence[Literal]):
        for i, lit in enumerate(oh):
            self.eq(binary, self.constant(i, len(binary)), lit)

    def mux_array(self, values: Sequence[Sequence[Literal]], one_hot: Sequence[Literal], ols: Sequence[Literal]):
        # todo: optimize the constant case
        and_values: list[Tuple[Atom, ...]] = []
        for v, s in zip(values, one_hot):  # when the len(values) != len(one_hot) items will be discarded
            av = self.block(len(v))
            and_values.append(av)
            self.and_(v, [s] * len(v), result=av)
        for av, ol in zip(zip(*and_values), ols):
            self.bit_or(*av, result=ol)

    def pow(self, base: Sequence[Literal], exponent: Sequence[Literal], result: Sequence[Literal],
            max_exponent: int | None = None, checked=True):
        # This may also be used to compute integer logarithms and roots.
        if max_exponent is None:
            # In the worst case, when base is 2 we need to compute powers up to len(result) before running out of bits.
            max_exponent = len(result)
        all_powers = [self.constant(1, len(result))] + [self.block(len(result)) for _ in range(1, max_exponent)]
        # all_powers[i] should equal base ** i before overflowing
        has_overflowed = [self.false] + [self.form.add_atom() for _ in range(1, max_exponent)]
        for n in range(1, max_exponent):
            overflow = self.form.add_atom()
            self.mul_with_overflow(all_powers[n - 1], base, all_powers[n], overflow=overflow)
            self.bit_or(has_overflowed[n - 1], overflow, result=has_overflowed[n])
        exp_selection = self.block(max_exponent)
        self.one_hot(exponent, exp_selection)
        if checked:
            # assert that base**exponent can't overflow
            self.and_(exp_selection, has_overflowed, [self.false for _ in range(max_exponent)])
        self.mux_array(all_powers, exp_selection, result)

    def product(self, n: int, factors: Iterable[Sequence[Literal]], result: Sequence[Literal]):
        factor_bits = [len(f) for f in factors]
        if len(factor_bits) == 0:
            self.assert_eq(result, self.constant(1, len(result)))
            return
        if len(factor_bits) == 1:
            self.assert_eq(result, next(iter(factors)))
            return
        assert len(factor_bits) >= 2
        it = iter(factors)
        res = result
        for i, f in enumerate(it):
            if i == n - 2:
                last_factor = next(it)
                self.mul(f, last_factor, res)
                return
            required_bits = min(sum(factor_bits[i + 1::]), len(result))
            rest = self.block(required_bits)
            self.mul(f, rest, res)
            res = rest

    def pow_by_const(self, base: Sequence[Literal], exponent: int, result: Sequence[Literal]):
        # http://szhorvat.net/pelican/fast-computation-of-powers.html
        assert exponent >= 0
        if exponent == 0:
            self.assert_eq(result, self.constant(1, len(result)))
            return
        if exponent == 1:
            self.assert_eq(base, result)
            return
        if exponent % 2 == 0:
            base_2 = self.block(min(2 * len(base), len(result)))
            self.mul(base, base, base_2)
            self.pow_by_const(base_2, exponent // 2, result)
        elif exponent % 3 == 0:
            base_2 = self.block(min(2 * len(base), len(result)))
            self.mul(base, base, base_2)
            base_3 = self.block(min(3 * len(base), len(result)))
            self.mul(base_2, base, base_3)
            self.pow_by_const(base_3, exponent // 3, result)
        else:
            base_e_minus_1 = self.block(min((exponent - 1) * len(base), len(result)))
            self.pow_by_const(base, exponent - 1, base_e_minus_1)
            self.mul(base_e_minus_1, base, result)

    def mod_pow(self, base: Sequence[Literal], exponent: Sequence[Literal],
                modulus: Sequence[Literal], result: Sequence[Literal]):
        # https://en.wikipedia.org/wiki/Modular_exponentiation
        def mod_mul(lhs: Sequence[Literal], rhs: Sequence[Literal], mul_res: Sequence[Literal]):
            # todo: when modulus is None, the result should be computed modulo 2 ** len(result)
            # what happens when the multiplication overflows?
            if modulus is None:
                self.mul(lhs, rhs, mul_res)
            else:
                mr = self.block(len(mul_res))
                self.mul(lhs, rhs, mr)
                self.mod(mr, modulus, mul_res)

        if modulus is not None:
            base_m = self.block(len(base))
            self.mod(base, modulus, base_m)
            base = base_m

        one = self.constant(1, len(result))
        res = one
        for b in reversed(exponent):
            if res != one:
                r2 = self.block(len(result))
                mod_mul(res, res, r2)
            else:
                r2 = one
            if b != self.false and b != -self.true:
                if r2 != one:
                    rb = self.block(len(result))
                    mod_mul(r2, base, rb)
                else:
                    rb = base
                res = self.block(len(result))
                self.mux_binary(rb, r2, b, res)
            else:
                res = r2
        self.assert_eq(res, result)

    def pow_of_2(self, exponent: Sequence[Literal], result: Sequence[Literal]):
        self.pow(self.constant(2, bits=len(exponent)), exponent, result)  # todo: optimize

    def lshift(self, lhs: Sequence[Literal], rhs: Sequence[Literal], result: Sequence[Literal]):
        # todo: optimize the constant case
        zero = self.constant(0, len(rhs))
        self.assert_gt(rhs, zero)
        shift = self.block(len(lhs))
        self.pow_of_2(rhs, shift)
        self.mul(lhs, shift, result)

    def rshift(self, lhs: Sequence[Literal], rhs: Sequence[Literal], result: Sequence[Literal]):
        # todo: optimize the constant case
        zero = self.constant(0, len(rhs))
        self.assert_gt(rhs, zero)
        shift = self.block(len(lhs))
        self.pow_of_2(rhs, shift)
        self.floordiv(lhs, shift, result)

    def abs(self, val: Sequence[Literal], result: Sequence[Literal]):
        if len(val) == 1:
            self.assert_eq(val, result)
            return
        sgn = val[-1]  # treat MSB as the sign bit
        negated = self.block(len(val))
        self.neg(val, negated)
        self.mux_binary(negated, val, sgn, result)

    # Convenience methods:

    def at_most_k(self, x: Sequence[Literal], k: int):
        self.bit_or(*(Literal(-lit) for lit in x), result=self.true)
        for sub in itertools.combinations(x, k + 1):
            self.bit_or(*sub, result=self.true)

    def invert(self, val: Sequence[Literal], result: Sequence[Literal]):
        self.eq([Literal(-lit) for lit in val], result, self.true)

    def neg(self, val: Sequence[Literal], result: Sequence[Literal]):
        self.sub(self.constant(0, len(val)), val, result)

    def gt(self, lhs: Sequence[Literal], rhs: Sequence[Literal], result: Literal):
        self.le(lhs, rhs, Literal(-result))

    def assert_gt(self, lhs: Sequence[Literal], rhs: Sequence[Literal]):
        self.le(lhs, rhs, self.false)

    def assert_le(self, lhs: Sequence[Literal], rhs: Sequence[Literal]):
        self.le(lhs, rhs, self.true)

    def ne(self, lhs: Sequence[Literal], rhs: Sequence[Literal], result: Literal):
        self.eq(lhs, rhs, Literal(-result))

    def assert_eq(self, lhs: Sequence[Literal], rhs: Sequence[Literal]):
        self.eq(lhs, rhs, self.true)

    def assert_ne(self, lhs: Sequence[Literal], rhs: Sequence[Literal]):
        self.eq(lhs, rhs, self.false)
