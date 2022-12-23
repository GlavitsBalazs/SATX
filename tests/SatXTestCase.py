import sys
import unittest
from typing import Callable, AbstractSet, Any, Iterator

import satx


class SatXTestCase(unittest.TestCase):

    def __init__(self, method_name='runTest', cnf_path='aux.cnf', signed_engine=False, simplify_engine=False,
                 solver='kissat', solver_params='', solver_log=False, verbose=False):
        super().__init__(methodName=method_name)
        self.cnf_path = cnf_path
        self.signed_engine = signed_engine
        self.simplify_engine = simplify_engine
        self.solver = solver
        self.solver_params = solver_params
        self.solver_log = solver_log
        self.verbose = verbose

    def engine(self, bits, deep=None, signed=None, simplify=None):
        if signed is None:
            signed = self.signed_engine
        if simplify is None:
            simplify = self.simplify_engine
        satx.engine(bits=bits, deep=deep, info=False, cnf_path=self.cnf_path, signed=signed, simplify=simplify)

    def satisfy(self):
        return satx.satisfy(solver=self.solver, params=self.solver_params, log=self.solver_log)

    def assert_satisfiable(self):
        self.assertTrue(self.satisfy())
        return True

    def assert_not_satisfiable(self):
        self.assertFalse(self.satisfy())
        return False

    def iterate_solutions(self, min_solutions: int = 1, max_solutions: int | None = None,
                          progress: Callable[[int], None] | None = None) -> Iterator[int]:
        """
        Iteratively try to satisfy the formula. Yield whenever a solution is found.
        Assert that the number of solutions N is min_solutions <= N <= max_solutions.
        max_solutions = None means infinity.
        """
        if max_solutions is not None:
            for i in range(max_solutions):
                satisfiable = self.satisfy()
                if progress is not None:
                    progress(i)
                if not satisfiable:
                    self.assertLessEqual(min_solutions, i)
                    break
                yield i
            else:
                # We've found max_solutions number of solutions.
                # There must not be more.
                self.assert_not_satisfiable()
                if progress is not None:
                    progress(max_solutions + 1)
        else:
            for i in range(min_solutions):
                self.assert_satisfiable()
                if progress is not None:
                    progress(i)
                yield i

    def assert_solutions(self, solution_func: Callable[[], Any], ground_truth: AbstractSet, verbose=None) -> None:
        if verbose is None:
            verbose = self.verbose
        solutions = set()

        def progress(found_solutions):
            print(f'{found_solutions} / {len(ground_truth) + 1}', file=sys.stderr)

        for _ in self.iterate_solutions(min_solutions=len(ground_truth), max_solutions=len(ground_truth),
                                        progress=progress if verbose else None):
            solutions.add(solution_func())
        self.assertEqual(solutions, ground_truth)
