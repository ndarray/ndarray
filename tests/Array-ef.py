#!/usr/bin/env python3

import unittest
import test_compile_failure

class ArrayTestCase(unittest.TestCase, test_compile_failure.CompileTest):

    def __init__(self, *args, **kwds):
        unittest.TestCase.__init__(self, *args, **kwds)
        test_compile_failure.CompileTest.__init__(
            self, subdir="Array-ef", template_target="Layout",
            preamble="""
#include "ndarray/Array.hpp"
namespace nd = ndarray;

"""
        )

    def test_const_correctness_1(self):
        self.assertCompileFails(
            "const_correctness_1",
            """
            nd::Array<float,3> a({4, 5, 2});
            nd::Array<float const,3> b(a);
            nd::Array<float,3> c(b);
            """
        )

if __name__ == "__main__":
    unittest.main()
