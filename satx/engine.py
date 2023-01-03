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

import os
import shutil
import subprocess
import sys
import warnings
from contextlib import AbstractContextManager
from io import BytesIO
from typing import IO, Sequence

from satx.alu import ALU
from satx.formula import FormulaWriter, Literal
from satx.gaussian import Gaussian
from satx.logicgate import LogicGateWriter
from satx.model import Model, SatResult
from satx.rational import Rational
from satx.unit import Unit


class Engine(AbstractContextManager):
    def __init__(self, bits: int, cnf_path=None, signed=False, simplify=True):
        """
        Initialize or reset the SAT-X system.
        :param bits: Implies an $[-2^{bits}, 2^{bits})$ search space.
        :param cnf_path: Path to render the generated CNF.
        :param signed: Indicates use of signed integer engine
        """
        self._cnf_path = cnf_path
        if os.path.exists(self._cnf_path):
            os.remove(self._cnf_path)
        self._buffer: IO[bytes] = BytesIO()
        self._form = FormulaWriter(self._buffer, simplify=simplify)
        self._gate_writer = LogicGateWriter(self._form)
        self._alu = ALU(self._gate_writer, bits, signed)

    @property
    def alu(self) -> ALU:
        return self._alu

    def integer(self, bits: int | None = None) -> Unit:
        """
        Correspond to an integer.
        :param bits: The bits for the integer.
        :return: An instance of Integer.
        """

        return self.alu.integer(bits)

    def constant(self, value: int, bits: int | None = None) -> Unit:
        """
        Correspond to a constant.
        :param bits: The bits for the constant.
        :param value: The value of the constant.
        :return: An instance of Constant.
        """

        return self.alu.constant(value, bits)

    def gaussian(self, x: Unit | None = None, y: Unit | None = None) -> Gaussian:
        """
        Create a gaussian integer from (x+yj).
        :param x: real
        :param y: imaginary
        :return: (x+yj)
        """
        if x is None:
            x = self.integer()
        if y is None:
            y = self.integer()
        return Gaussian.from_re_im(x, y)

    def rational(self, x: Unit | None = None, y: Unit | None = None) -> Rational:
        """
        Create a rational x / y.
        :param x: numerator
        :param y: denominator
        :return: x / y
        """
        if x is None:
            x = self.integer()
        if y is None:
            y = self.integer()
        return Rational.divide(x, y)

    def tensor(self, dimensions: Sequence[int]):
        """
        Create a tensor
        :param dimensions: The list of dimensions
        :return: A tensor
        """

        return self.alu.tensor(shape=dimensions)

    def reset(self):
        if os.path.exists(self._cnf_path):
            os.remove(self._cnf_path)
        self._buffer.close()
        self._form.reset()
        self._gate_writer = LogicGateWriter(self._form)
        self._alu = ALU(self._gate_writer, self._alu.default_bits, self._alu.default_signed)

    def close(self):
        self._buffer.close()  # If it were a file we should close it.

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def satisfy(self, solver, params='', log=False, cnf_path=None, mod_path=None):
        """
        Solve with external solver.
        :param solver: The external solver.
        :param params: Parameters passed to external solver.
        :return: True if SAT else False.
        """
        if cnf_path is None:
            cnf_path = self._cnf_path
        if mod_path is None:
            name = os.path.splitext(cnf_path)[0]
            mod_path = name + '.mod'

        with open(cnf_path, 'wt') as cnf:
            self._form.write_dimacs_file(cnf)

        # Optimization idea: Don't write to a file. Keep it in a BytesIO.
        with open(mod_path, 'wb') as mod:
            command = f'{solver} {cnf_path} {params}' if params else f'{solver} {cnf_path}'
            with subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as proc:
                shutil.copyfileobj(proc.stdout, mod)
        with open(mod_path, 'rt') as mod:
            result, model = Model.read_model_file(mod, log_file=sys.stdout if log else None)

        if result == SatResult.Unknown:
            warnings.warn("Satisfiability unknown.")
        if result != SatResult.Satisfiable:
            # There is no guarantee that the model is empty.
            return False

        self._alu.model = model  # Load the model. Now Units of the ALU can have values.

        # In order to be able to find multiple models, the negation of this model is asserted.
        neg = [Literal(-lit) for lit in model.clause]
        self._form.add_clause(neg, simplify=False)  # It's important not to simplify it.

        return True
