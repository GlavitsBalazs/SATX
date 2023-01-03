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

import typing
from enum import Enum
from typing import Sequence, TextIO, Mapping, Iterator

from satx.formula import Atom, Literal


class SatResult(Enum):
    Satisfiable = "SATISFIABLE"
    Unsatisfiable = "UNSATISFIABLE"
    Unknown = "UNKNOWN"

    @classmethod
    def _missing_(cls, value):
        return cls.Unknown


class Model(Mapping[Atom, typing.Literal[0, 1]]):
    def __init__(self, clause: Sequence[Literal]):
        self.clause = clause
        self.index: Mapping[Atom, typing.Literal[0, 1]] = {Atom(Literal(abs(x))): 1 if x >= 0 else 0 for x in clause}

    def __getitem__(self, atom: Atom) -> typing.Literal[0, 1]:
        return self.index[atom]

    def __len__(self) -> int:
        return len(self.index)

    def __iter__(self) -> Iterator[Atom]:
        return iter(self.index)

    def entails(self, atom: Atom) -> bool:
        return atom in self.index and self.index[atom] >= 0

    # todo: call on formula, entails formula...

    @classmethod
    def read_model_file(cls, mod_file: TextIO, log_file: TextIO | None = None):
        # http://www.satcompetition.org/2009/format-solvers2009.html
        result = SatResult.Unknown
        clause: list[Literal] = []
        for line in mod_file:
            line = line.strip('\n')
            if line.startswith('c'):
                if log_file is not None:
                    log_file.write(line.partition(' ')[2] + '\n')
                continue
            if line.startswith('s'):
                res = line.partition(' ')[2]
                result = SatResult(res)
                if log_file is not None:
                    log_file.write(res + '\n')
            if line.startswith('v'):
                clause.extend(Literal(int(lit)) for lit in line.split(sep=None)[1:])
        if result == SatResult.Satisfiable:
            assert len(clause) > 1
            assert clause.pop(-1) == 0
        return result, cls(clause)
