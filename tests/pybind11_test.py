# -*- python -*-
#
#  Copyright (c) 2010-2012, Jim Bosch
#  All rights reserved.
#
#  ndarray is distributed under a simple BSD-like license;
#  see the LICENSE file that should be present in the root
#  of the source distribution, or alternately available at:
#  https://github.com/ndarray/ndarray
#
import numpy
import unittest

import pybind11_test_mod


class TestNumpyPybind11(unittest.TestCase):

    def testArray1(self):
        a1 = pybind11_test_mod.returnArray1()
        a2 = numpy.arange(6, dtype=float)
        self.assertTrue((a1 == a2).all())
        self.assertTrue(pybind11_test_mod.acceptArray1(a2))
        a3 = pybind11_test_mod.returnConstArray1()
        self.assertTrue((a1 == a3).all())
        self.assertFalse(a3.flags["WRITEABLE"])

    def testArray3(self):
        a1 = pybind11_test_mod.returnArray3()
        a2 = numpy.arange(4*3*2, dtype=float).reshape(4, 3, 2)
        self.assertTrue((a1 == a2).all())
        self.assertTrue(pybind11_test_mod.acceptArray3(a2))
        a3 = pybind11_test_mod.returnConstArray3()
        self.assertTrue((a1 == a3).all())
        self.assertFalse(a3.flags["WRITEABLE"])

    def testStrideHandling(self):
        # in NumPy 1.8+ 1- and 0-sized arrays can have arbitrary strides; we should
        # be able to handle those
        array = numpy.zeros(1, dtype=float)
        # just test that these don't throw
        pybind11_test_mod.acceptArray10(array)
        pybind11_test_mod.acceptArray1(array)
        array = numpy.zeros(0, dtype=float)
        pybind11_test_mod.acceptArray10(array)
        pybind11_test_mod.acceptArray1(array)
        # test that we gracefully fail when the strides are no multiples of the itemsize
        dtype = numpy.dtype([("f1", numpy.float64), ("f2", numpy.int16)])
        table = numpy.zeros(3, dtype=dtype)
        self.assertRaises(TypeError, pybind11_test_mod.acceptArray10, table['f1'])

    def testNone(self):
        array = numpy.zeros(10, dtype=float)
        self.assertEqual(pybind11_test_mod.acceptNoneArray(array), 0)
        self.assertEqual(pybind11_test_mod.acceptNoneArray(None), 1)
        self.assertEqual(pybind11_test_mod.acceptNoneArray(), 1)

    def testNonNativeByteOrder(self):
        d1 = numpy.dtype("<f8")
        d2 = numpy.dtype(">f8")
        nonnative = d2 if d1 == numpy.dtype(float) else d1
        a = numpy.zeros(5, dtype=nonnative)
        self.assertRaises(TypeError, pybind11_test_mod.acceptArray10, a)


if __name__ == "__main__":
    unittest.main()
