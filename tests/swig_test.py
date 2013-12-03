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
import swig_test_mod
import unittest

class TestNumpySwig(unittest.TestCase):

    def testMatrixXd(self):
        m1 = swig_test_mod.returnMatrixXd()
        m2 = numpy.matrix(numpy.arange(15, dtype=float).reshape(3,5).transpose())
        self.assert_((m1 == m2).all())
        self.assert_(swig_test_mod.acceptMatrixXd(m2))

    def testMatrix2d(self):
        m1 = swig_test_mod.returnMatrix2d()
        m2 = numpy.matrix([[0.0, 2.0], [1.0, 3.0]])
        self.assert_((m1 == m2).all())
        self.assert_(swig_test_mod.acceptMatrix2d(m2))

    def testArray1(self):
        a1 = swig_test_mod.returnArray1()
        a2 = numpy.arange(6, dtype=float)
        self.assert_((a1 == a2).all())
        self.assert_(swig_test_mod.acceptArray1(a2))
        a3 = swig_test_mod.returnConstArray1()
        self.assert_((a1 == a3).all())
        self.assert_(a3.flags["WRITEABLE"] == False)

    def testArray3(self):
        a1 = swig_test_mod.returnArray3()
        a2 = numpy.arange(4*3*2, dtype=float).reshape(4,3,2)
        self.assert_((a1 == a2).all())
        self.assert_(swig_test_mod.acceptArray3(a2))
        a3 = swig_test_mod.returnConstArray3()
        self.assert_((a1 == a3).all())
        self.assert_(a3.flags["WRITEABLE"] == False)

    def testClass(self):
        a = swig_test_mod.MatrixOwner()
        m1 = a.member
        m2 = a.getMember()
        self.assert_((m1 == 0).all())
        self.assert_((m2 == 0).all())
        self.assertEqual(m1.shape, (2,2))
        self.assertEqual(m2.shape, (2,2))

    def testOverloads(self):
        self.assertEqual(swig_test_mod.acceptOverload(1), 0)
        self.assertEqual(swig_test_mod.acceptOverload(numpy.zeros((2,2), dtype=float)), 2)
        self.assertEqual(swig_test_mod.acceptOverload(numpy.zeros((3,3), dtype=float)), 3)

    def testStrideHandling(self):
        # in NumPy 1.8+ 1- and 0-sized arrays can have arbitrary strides; we should
        # be able to handle those
        array = numpy.zeros(1, dtype=float)
        # just test that these don't throw
        swig_test_mod.acceptArray10(array)
        swig_test_mod.acceptArray10(array)
        array = numpy.zeros(0, dtype=float)
        swig_test_mod.acceptArray10(array)
        swig_test_mod.acceptArray10(array)
        # test that we gracefully fail when the strides are no multiples of the itemsize
        dtype = numpy.dtype([("f1", numpy.float64), ("f2", numpy.int16)])
        table = numpy.zeros(3, dtype=dtype)
        self.assertRaises(TypeError, swig_test_mod.acceptArray10, table['f1'])


if __name__ == "__main__":
    unittest.main()
