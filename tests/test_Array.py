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
from .compilation import CompilationTestMixin, SnippetFormatter


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    suite.addTests(ArrayConversionTestCase.makeSuite1d())
    suite.addTests(ArrayConversionTestCase.makeSuite2d())
    return suite


class ArrayConversionTestCase(unittest.TestCase, CompilationTestMixin):

    formatter = SnippetFormatter(
        """
        #include "ndarray/Array.hpp"

        using namespace ndarray;

        namespace {

        template <typename T, Size N, Offset C>
        bool accept_array(Array<T, N, C> const & a) {
            return true;
        }

        } // <anonymous>
        """
    )

    tmpl = """
        Array<float {p.t_in}, {n}, {p.c_in}> a;
        accept_array<float {p.t_out}, {n}, {p.c_out}>(a);
    """

    ParameterTuple = namedtuple("ParameterTuple", ("t_in", "c_in", "t_out", "c_out"))

    @classmethod
    def generate_combinations(cls, n):
        for combo in itertools.product(("const", ""), tuple(range(-n, n + 1)), repeat=2):
            yield cls.ParameterTuple._make(combo)

    def __init__(self, code, should_compile=None, stderr_regex=None, parameters=None):
        if should_compile is None:
            super().__init__()
        else:
            super().__init__("testCompiles" if should_compile else "testDoesNotCompile")
        self.code = code
        self.stderr_regex = stderr_regex
        self.parameters = parameters

    def testCompiles(self):
        with self.subTest(**self.parameters._asdict()):
            self.assertCompiles(self.code, formatter=self.formatter)

    def testDoesNotCompile(self):
        with self.subTest(**self.parameters._asdict()):
            self.assertDoesNotCompile(self.code, stderr_regex=self.stderr_regex, formatter=self.formatter)

    @classmethod
    def makeSuite1d(cls):
        suite = unittest.TestSuite()
        for p in cls.generate_combinations(n=1):
            should_compile = True
            stderr_regex = None
            if p.c_in == 0 and p.c_out != 0:
                should_compile = False
                stderr_regex = "invalid contiguousness conversion"
            if p.t_in == "const" and p.t_out != "const":
                should_compile = False
                stderr_regex = None
            code = cls.tmpl.format(n=1, p=p)
            suite.addTest(ArrayConversionTestCase(code, should_compile, stderr_regex, parameters=p))
        return suite

    @classmethod
    def makeSuite2d(cls):
        suite = unittest.TestSuite()
        for p in cls.generate_combinations(n=2):
            should_compile = True
            stderr_regex = None
            if (p.c_out > 0 and p.c_out > p.c_in) or (p.c_out < 0 and p.c_out < p.c_in):
                should_compile = False
                stderr_regex = "invalid contiguousness conversion"
            if p.t_in == "const" and p.t_out != "const":
                should_compile = False
                stderr_regex = None
            code = cls.tmpl.format(n=2, p=p)
            suite.addTest(ArrayConversionTestCase(code, should_compile, stderr_regex, parameters=p))
        return suite
