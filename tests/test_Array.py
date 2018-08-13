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


class ParameterTuple(namedtuple("ParameterTuple", ("scalar", "const", "n", "c"))):

    @classmethod
    def generate(cls, base=None, **kwds):
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
        for field in cls._fields:
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
                    yield cls._make(params[:-1] + (c,))
            else:
                yield cls._make(params)

    def __str__(self):
        return f"<{self.scalar} {self.const}, {self.n}, {self.c}>"


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

    def __init__(self, method, parameters=None):
        super().__init__(method)
        self.parameters = parameters

    def runConversionTest(self, valid, invalid, stderr_regex=None):
        # Try compiling expected failures separately, since otherwise they'd hide each other.
        for out_params in invalid:
            with self.subTest(**out_params._asdict()):
                self.assertDoesNotCompile(
                    ["Array{} a;".format(self.parameters),
                     "accept_Array{}(a);".format(out_params)],
                    stderr_regex=stderr_regex,
                    context=self.context
                )
        # Compile expected successes together to save compile time.
        lines = [f"Array{self.parameters} a;"]
        lines.extend(f"accept_Array{out_params}(a);" for out_params in valid)
        self.assertCompiles(lines, context=self.context)

    def testContiguousConversions(self):
        """Test that we can convert Arrays only when we do not increase
        abs(C), and do not change the sign of C when C > 1.
        """
        valid = []
        invalid = []
        with self.subTest(**self.parameters._asdict()):
            for out_params in ParameterTuple.generate(self.parameters, c=range):
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
            self.runConversionTest(valid, invalid, stderr_regex="invalid contiguousness conversion")

    def testConstConversions(self):
        """Test that we can convert Arrays from T to T const, but not the
        reverse.

        Unlike contiguousness conversions, this should work even when trying
        to match templated signatures, because Array<T, ...> inherits from
        Array<T const, ...>.
        """
        valid = []
        invalid = []
        with self.subTest(**self.parameters._asdict()):
            for out_params in ParameterTuple.generate(self.parameters, const=["const", ""]):
                if self.parameters.const and not out_params.const:
                    invalid.append(out_params)
                else:
                    valid.append(out_params)
            # Tests with no template signature matching.
            self.runConversionTest(valid, invalid)
            # Test success with template signature matching.
            self.assertCompiles([f"Array{self.parameters} a;", "accept_Array(a);"], context=self.context)

    @classmethod
    def makeSuite(cls):
        suite = unittest.TestSuite()
        for p in ParameterTuple.generate(scalar="float", const=("const", ""), n=(1, 2), c=range):
            suite.addTest(ArrayTestCase("testContiguousConversions", parameters=p))
            suite.addTest(ArrayTestCase("testConstConversions", parameters=p))
        return suite
