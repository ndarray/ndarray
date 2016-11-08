#!/usr/bin/env python3

import unittest
import test_compile_failure


class ArrayTestCase(unittest.TestCase, test_compile_failure.CompileTest):

    def __init__(self, *args, **kwds):
        unittest.TestCase.__init__(self, *args, **kwds)
        test_compile_failure.CompileTest.__init__(
            self, subdir="tests/Array-ef", template_target="Layout",
            preamble="""
#include "ndarray/Array.hpp"
namespace nd = ndarray;

"""
        )

    def test_const_correct_copy_construct(self):
        """Test that we can't construct or a non-const array from a copy of
        a const one.
        """
        self.assertCompileFails(
            "const_correct_copy_construct",
            """
            nd::Array<float,3> a({4, 5, 2});
            nd::Array<float const,3> b(a);
            nd::Array<float,3> c(b);
            """
        )

    def test_const_correct_move_construct(self):
        """Test that we can't construct a non-const array from a move of
        a const one.
        """
        self.assertCompileFails(
            "const_correct_move_construct",
            """
            nd::Array<float,3> a({4, 5, 2});
            nd::Array<float const,3> b(a);
            nd::Array<float,3> c(std::move(b));
            """
        )

    def test_const_correct_copy_assign(self):
        """Test that we can't assign to a non-const array from a copy of
        a const one.
        """
        self.assertCompileFails(
            "const_correct_copy_assign",
            """
            nd::Array<float,3> a({4, 5, 2});
            nd::Array<float const,3> b(a);
            nd::Array<float,3> c;
            c = b;
            """
        )

    def test_const_correct_move_assign(self):
        """Test that we can't assign to a non-const array from a move of
        a const one.
        """
        self.assertCompileFails(
            "const_correct_move_assign",
            """
            nd::Array<float,3> a({4, 5, 2});
            nd::Array<float const,3> b(a);
            nd::Array<float,3> c;
            c = std::move(b);
            """
        )

    def test_contiguous_copy_construct(self):
        """Test that we can't construct an array with more contiguous
        dimensions from a copy of one with fewer.
        """
        self.assertCompileFails(
            "contiguous_copy_construct",
            """
            nd::Array<float,3> a({4, 5, 2});
            nd::Array<float,3,2> b(a);
            """,
            stderr_regex="Invalid conversion"
        )

    def test_contiguous_copy_assign(self):
        """Test that we can't assign to an array with more contiguous
        dimensions from a copy of one with fewer.
        """
        self.assertCompileFails(
            "contiguous_copy_assign",
            """
            nd::Array<float,3> a({4, 5, 2});
            nd::Array<float,3,2> b;
            b = a;
            """,
            stderr_regex="Invalid conversion"
        )

    def test_contiguous_move_construct(self):
        """Test that we can't construct an array with more contiguous
        dimensions from a move of one with fewer.
        """
        self.assertCompileFails(
            "contiguous_move_construct",
            """
            nd::Array<float,3> a({4, 5, 2});
            nd::Array<float,3,2> b(std::move(a));
            """,
            stderr_regex="Invalid conversion"
        )

    def test_contiguous_move_assign(self):
        """Test that we can't assign to an array with more contiguous
        dimensions from a move of one with fewer.
        """
        self.assertCompileFails(
            "contiguous_move_assign",
            """
            nd::Array<float,3> a({4, 5, 2});
            nd::Array<float,3,2> b;
            b = std::move(a);
            """,
            stderr_regex="Invalid conversion"
        )

if __name__ == "__main__":
    unittest.main()
