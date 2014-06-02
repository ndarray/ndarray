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
import bp_test_mod
import unittest
import numpy

class TestNdArrayWrappers(unittest.TestCase):

    def testReturnArray(self):
        for suffix in ("11","10","22","21","20","33","32","31","30"):
            func = getattr(bp_test_mod, "returnArray_d%s" % suffix)
            cfunc = getattr(bp_test_mod, "returnArray_dc%s" % suffix)
            array = func()
            carray = cfunc()
            self.assert_((array == numpy.arange(0, array.size).reshape(array.shape)).all())
            self.assert_((carray == numpy.arange(0, carray.size).reshape(carray.shape)).all())
            self.assert_(array.flags["WRITEABLE"])
            self.assert_(not carray.flags["WRITEABLE"])

    def testAcceptArray(self):
        for suffix in ("11","10","22","21","20","33","32","31","30"):
            func = getattr(bp_test_mod, "acceptArray_d%s" % suffix)
            cfunc = getattr(bp_test_mod, "acceptArray_dc%s" % suffix)
            nd = int(suffix[0])
            shape = tuple(int(i) for i in numpy.random.randint(low=2,high=5,size=nd))
            array = numpy.zeros(shape, dtype=float)
            array[:] = numpy.arange(0, array.size).reshape(array.shape)
            self.assert_(func(array))
            self.assert_(cfunc(array))

    def testAcceptArrayVal(self):
        for suffix in ("11","10","22","21","20","33","32","31","30"):
            func = getattr(bp_test_mod, "acceptArrayVal_d%s" % suffix)
            cfunc = getattr(bp_test_mod, "acceptArrayVal_dc%s" % suffix)
            nd = int(suffix[0])
            shape = tuple(int(i) for i in numpy.random.randint(low=2,high=5,size=nd))
            array = numpy.zeros(shape, dtype=float)
            array[:] = numpy.arange(0, array.size).reshape(array.shape)
            self.assert_(func(array))
            self.assert_(cfunc(array))

    def testExtractArray(self):
        for suffix in ("11","10","22","21","20","33","32","31","30"):
            func = getattr(bp_test_mod, "extractArray_d%s" % suffix)
            cfunc = getattr(bp_test_mod, "extractArray_dc%s" % suffix)
            nd = int(suffix[0])
            shape = tuple(int(i) for i in numpy.random.randint(low=2,high=5,size=nd))
            array = numpy.zeros(shape, dtype=float)
            array[:] = numpy.arange(0, array.size).reshape(array.shape)
            self.assert_(func(array))
            self.assert_(cfunc(array))

    def testReturnVector(self):
        for suffix in ("2","3"):
            func = getattr(bp_test_mod, "returnVector_d%s" % suffix)
            vector = func()
            self.assertEqual(vector, tuple(numpy.arange(len(vector), dtype=float)))

    def testAcceptVector(self):
        for suffix in ("2","3"):
            func = getattr(bp_test_mod, "acceptVector_d%s" % suffix)
            nd = int(suffix[0])
            vector = tuple(numpy.arange(nd, dtype=float))
            self.assert_(func(vector))

    def testStrideHandling(self):
        # in NumPy 1.8+ 1- and 0-sized arrays can have arbitrary strides; we should
        # be able to handle those
        array = numpy.zeros(1, dtype=float)
        self.assert_(bp_test_mod.acceptArray_d11(array))
        self.assert_(bp_test_mod.acceptArray_d10(array))
        array = numpy.zeros(0, dtype=float)
        self.assert_(bp_test_mod.acceptArray_d11(array))
        self.assert_(bp_test_mod.acceptArray_d10(array))
        # test that we gracefully fail when the strides are not multiples of the itemsize
        dtype = numpy.dtype([("f1", numpy.float64), ("f2", numpy.int16)])
        table = numpy.zeros(3, dtype=dtype)
        self.assertRaises(TypeError, bp_test_mod.acceptArray_d10, table['f1'])

    def _testMemory(self):
        shape = (400, 400, 10)
        for n in range(1000000):
            a = bp_test_mod.makeArray_d33(shape)

if __name__=="__main__":
    unittest.main()
