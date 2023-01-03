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

import struct
from io import SEEK_SET
from typing import Sequence, IO, TextIO, NewType

Literal = NewType('Literal', int)
"""
A literal is an atom or the negation of an atom.
"""

Atom = NewType('Atom', Literal)
"""
An atomic formula, also known as a variable. Each one is identified by an int greater than zero. Must not be negated.
"""


class FormulaWriter:
    """
    Writes a propositional logic formula in conjunctive normal form.

    First it writes to a binary buffer, which then may be converted to DIMACS format.
    """

    def __init__(self, buffer: IO[bytes], simplify=True):
        self.buffer = buffer
        self.simplify = simplify
        self.number_of_clauses = 0
        self.number_of_atoms = 0
        self._constant_atoms: dict[Atom, bool] = dict()

    def reset(self):
        self.buffer.seek(0, SEEK_SET)
        self.number_of_clauses = 0
        self.number_of_atoms = 0
        self._constant_atoms.clear()

    def add_atom(self) -> Atom:
        """
        An atomic formula, also known as a variable. Each one is identified by an int greater than zero.
        """
        if self.number_of_atoms >= 2 ** 31 - 1:
            raise ValueError  # most SAT solvers use signed 32 bit ints
        self.number_of_atoms += 1
        return Atom(Literal(self.number_of_atoms))

    def add_clause(self, clause: Sequence[Literal], simplify=None):
        """
        A clause is the disjunction of a sequence of literals.
        The formula will be the conjunction of these clauses.
        """
        if simplify is None:
            simplify = self.simplify
        if simplify:
            clause = self._simplify_clause(clause)
        if len(clause) == 0 or Literal(0) in clause:
            return  # Treat the empty clause as true. Treat the 0 atom as true.
        if self.number_of_clauses >= 2 ** 31 - 1:
            raise ValueError
        self.write_int_array(self.buffer, clause)
        self.number_of_clauses += 1

    def get_constant(self, lit: Literal) -> bool | None:
        atom, negated = Atom(Literal(abs(lit))), lit >= 0
        val = self._constant_atoms.get(atom)
        if val is None:
            return val
        else:
            return val if negated else not val

    def _simplify_clause(self, clause: Sequence[Literal]) -> Sequence[Literal]:
        clause_set = set(clause)  # Remove duplicates.
        for lit in tuple(clause_set):
            v: bool | None = self.get_constant(lit)
            if v is True or -lit in clause_set:
                return tuple()  # The clause is valid.
            if v is False:
                clause_set.remove(lit)  # False literals may be removed.
        if len(clause_set) == 1:
            # It's a constant literal.
            lit = next(iter(clause_set))
            atom, value = Atom(Literal(abs(lit))), lit >= 0
            if atom in self._constant_atoms:
                if self._constant_atoms[atom] != value:
                    # Maybe we shouldn't throw an exception here if we want to allow unsatisfiable formulas.
                    raise AssertionError("Inconsistent constant literal assertion.")
                return tuple()  # It's already in the formula.
            else:
                self._constant_atoms[atom] = value
        return sorted(clause_set, key=lambda x: abs(x))

    @staticmethod
    def write_int_array(io: IO[bytes], val: Sequence[int]):
        # Again, signed 32 bit ints are used. No need to have more bits.
        io.write(struct.pack(f'>{len(val) + 1}i', len(val), *val))

    @staticmethod
    def read_int_array(io: IO[bytes]) -> Sequence[int]:
        sizeof_int = struct.calcsize('>i')
        arr_len = struct.unpack('>i', io.read(sizeof_int))[0]
        return struct.unpack(f'>{arr_len}i', io.read(sizeof_int * arr_len))

    def write_dimacs_file(self, output: TextIO, comments: str | None = None):
        # http://www.satcompetition.org/2009/format-benchmarks2009.html
        if comments:
            for line in comments.split('\n'):
                output.write(f"c {line}\n")
        output.write(f'p cnf {self.number_of_atoms} {self.number_of_clauses}\n')
        self.buffer.seek(0, SEEK_SET)
        for _ in range(self.number_of_clauses):
            clause = self.read_int_array(self.buffer)
            output.write(' '.join(map(str, clause)) + ' 0\n')
