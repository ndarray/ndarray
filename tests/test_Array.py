#
# Copyright (c) 2010-2018, Jim Bosch
# All rights reserved.
#
# ndarray is distributed under a simple BSD-like license;
# see the LICENSE file that should be present in the root
# of the source distribution, or alternately available at:
# https://github.com/ndarray/ndarray
#

import unittest
import itertools
from collections import namedtuple
from .compilation import CompilationTestMixin, SnippetContext


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    suite.addTests(ArrayTestCase.makeSuite())
    return suite


class ArrayTestCase(unittest.TestCase, CompilationTestMixin):

    context = SnippetContext(
        """
        #include "ndarray/Array.hpp"

        using namespace ndarray;

        namespace {

        template <typename T, Size N, Offset C>
        bool accept_Array(Array<T, N, C> const & a) {
            return true;
        }

        } // <anonymous>
        """
    )

    default_construct_template = "Array{p.ref}<{p.scalar} {p.const}, {p.n}, {p.c}> {var};"
    accept_template = "accept_Array{p.ref}<{p.scalar} {p.const}, {p.n}, {p.c}>({var});"

    ParameterTuple = namedtuple("ParameterTuple", ("ref", "scalar", "const", "n", "c"))

    @classmethod
    def generate_parameters(cls, base=None, **kwds):
        """Generate ParameterTuples from cartesian products of possible values.

        Parameters
        ----------
        base : `ParameterTuple`
            A tuple of parameter values or ranges to use as defaults.

        Keyword arguments with keys matching any of the fields in
        ParameterTuple are accepted.  THese may be scalar values or sequences
        of values to include in the certesian product.  "c" may have the
        special value `range` (the built-in function), which generates "c"
        values as `range(-n, n+1)` for every "n" value generated.
        """
        param_range_list = []
        for field in cls.ParameterTuple._fields:
            try:
                v = kwds[field]
            except KeyError:
                assert base is not None
                v = getattr(base, field)
            if not isinstance(v, (list, tuple)):
                v = (v,)
            param_range_list.append(v)
        for params in itertools.product(*param_range_list):
            if params[-1] == range:
                n = params[-2]
                for c in range(-n, n + 1):
                    yield cls.ParameterTuple._make(params[:-1] + (c,))
            else:
                yield cls.ParameterTuple._make(params)

    def __init__(self, method, parameters=None):
        super().__init__(method)
        self.parameters = parameters

    def testContiguousConversions(self):
        """Test that we can convert Arrays only when we do not increase
        abs(C), and do not change the sign of C when C > 1.
        """
        valid = []
        invalid = []
        with self.subTest(**self.parameters._asdict()):
            for out_params in self.generate_parameters(self.parameters, c=range):
                if out_params.n == 1:
                    if self.parameters.c == 0 and out_params.c != 0:
                        invalid.append(out_params)
                    else:
                        valid.append(out_params)
                else:
                    if out_params.c > 0 and out_params.c > self.parameters.c:
                        invalid.append(out_params)
                    elif out_params.c < 0 and out_params.c < self.parameters.c:
                        invalid.append(out_params)
                    else:
                        valid.append(out_params)
            # Try compiling expected failures separately, since otherwise they'd hide each other.
            for out_params in invalid:
                with self.subTest(**out_params._asdict()):
                    self.assertDoesNotCompile(
                        [self.default_construct_template.format(p=self.parameters, var="a"),
                         self.accept_template.format(p=out_params, var="a")],
                        stderr_regex="invalid contiguousness conversion",
                        context=self.context
                    )
            # Compile expected successes together to save compile time.
            lines = [self.default_construct_template.format(p=self.parameters, var="a")]
            lines.extend(self.accept_template.format(p=out_params, var="a") for out_params in valid)
            self.assertCompiles("\n".join(lines), context=self.context)

    @classmethod
    def makeSuite(cls):
        suite = unittest.TestSuite()
        for p in cls.generate_parameters(ref="", scalar="float", const=("const", ""), n=(1, 2), c=range):
            suite.addTest(ArrayTestCase("testContiguousConversions", parameters=p))
        return suite
