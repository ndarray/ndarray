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
import pybind11_test_mod
import unittest

class TestNumpyPybind11(unittest.TestCase):

    def testMatrixXd(self):
        m1 = pybind11_test_mod.returnMatrixXd()
        m2 = numpy.matrix(numpy.arange(15, dtype=float).reshape(3,5).transpose())
        self.assert_((m1 == m2).all())
        self.assert_(pybind11_test_mod.acceptMatrixXd(m2))

    def testMatrix2d(self):
        m1 = pybind11_test_mod.returnMatrix2d()
        m2 = numpy.matrix([[0.0, 2.0], [1.0, 3.0]])
        self.assert_((m1 == m2).all())
        self.assert_(pybind11_test_mod.acceptMatrix2d(m2))

    def testArray1(self):
        a1 = pybind11_test_mod.returnArray1()
        a2 = numpy.arange(6, dtype=float)
        self.assert_((a1 == a2).all())
        self.assert_(pybind11_test_mod.acceptArray1(a2))
        a3 = pybind11_test_mod.returnConstArray1()
        self.assert_((a1 == a3).all())
        self.assert_(a3.flags["WRITEABLE"] == False)

    def testArray3(self):
        a1 = pybind11_test_mod.returnArray3()
        a2 = numpy.arange(4*3*2, dtype=float).reshape(4,3,2)
        self.assert_((a1 == a2).all())
        self.assert_(pybind11_test_mod.acceptArray3(a2))
        a3 = pybind11_test_mod.returnConstArray3()
        self.assert_((a1 == a3).all())
        self.assert_(a3.flags["WRITEABLE"] == False)

    def testClass(self):
        a = pybind11_test_mod.MatrixOwner()
        m1 = a.member
        m2 = a.getMember()
        self.assert_((m1 == 0).all())
        self.assert_((m2 == 0).all())
        self.assertEqual(m1.shape, (2,2))
        self.assertEqual(m2.shape, (2,2))

    def testOverloads(self):
        self.assertEqual(pybind11_test_mod.acceptOverload(1), 0)
        self.assertEqual(pybind11_test_mod.acceptOverload(numpy.zeros((2,2), dtype=float)), 2)
        self.assertEqual(pybind11_test_mod.acceptOverload(numpy.zeros((3,3), dtype=float)), 3)

    def testStrideHandling(self):
        # in NumPy 1.8+ 1- and 0-sized arrays can have arbitrary strides; we should
        # be able to handle those
        array = numpy.zeros(1, dtype=float)
        # just test that these don't throw
        pybind11_test_mod.acceptArray10(array)
        pybind11_test_mod.acceptArray10(array)
        array = numpy.zeros(0, dtype=float)
        pybind11_test_mod.acceptArray10(array)
        pybind11_test_mod.acceptArray10(array)
        # test that we gracefully fail when the strides are no multiples of the itemsize
        dtype = numpy.dtype([("f1", numpy.float64), ("f2", numpy.int16)])
        table = numpy.zeros(3, dtype=dtype)
        self.assertRaises(TypeError, pybind11_test_mod.acceptArray10, table['f1'])

    def testNone(self):
        array = numpy.zeros(10, dtype=float)
        self.assertEqual(pybind11_test_mod.acceptNoneArray(array), 0)
        self.assertEqual(pybind11_test_mod.acceptNoneArray(None), 1)
        self.assertEqual(pybind11_test_mod.acceptNoneArray(), 1)

        m1 = pybind11_test_mod.returnMatrixXd()
        self.assertEqual(pybind11_test_mod.acceptNoneMatrixXd(m1), 2)
        self.assertEqual(pybind11_test_mod.acceptNoneMatrixXd(None), 3)
        self.assertEqual(pybind11_test_mod.acceptNoneMatrixXd(), 3)

        m2 = pybind11_test_mod.returnMatrix2d()
        self.assertEqual(pybind11_test_mod.acceptNoneMatrix2d(m2), 4)
        self.assertEqual(pybind11_test_mod.acceptNoneMatrix2d(None), 5)
        self.assertEqual(pybind11_test_mod.acceptNoneMatrix2d(), 5)

    def testFullySpecifiedMatrix(self):
        # Failure on this specific case was not caught by other unit tests
        a = numpy.array([[ 1.,  0.], [ 0.,  1.]])
        b = numpy.array([ 0.,  0.])
        self.assert_(pybind11_test_mod.acceptFullySpecifiedMatrix(a, b))

    def testNonNativeByteOrder(self):
        d1 = numpy.dtype("<f8")
        d2 = numpy.dtype(">f8")
        nonnative = d2 if d1 == numpy.dtype(float) else d1
        a = numpy.zeros(5, dtype=nonnative)
        self.assertRaises(TypeError, pybind11_test_mod.acceptArray10, a)

if __name__ == "__main__":
    unittest.main()
