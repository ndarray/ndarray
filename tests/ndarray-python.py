#!/usr/bin/env python
import numpy
import ndarray_python_test
import swig_test_mod
import unittest

class TestNDArray(unittest.TestCase):

    def testVectorConversion(self):
        a = (3.5,6.7,1.2)
        b = ndarray_python_test.passFloatVector3(a)
        self.assertEqual(a,b)

    def testArrayConversion(self):
        a = numpy.zeros((5,3,4),dtype=float)
        b = ndarray_python_test.passFloatArray33(a)
        self.assertEqual(a.shape,b.shape)
        self.assertEqual(a.strides,b.strides)
        c = a[:,:,:2]
        d = ndarray_python_test.passFloatArray33(c)
        self.assertEqual(c.shape,d.shape)
        self.assertNotEqual(c.strides,d.strides)
        del d
        d = ndarray_python_test.passFloatArray30(c)
        self.assertEqual(c.shape,d.shape)
        self.assertEqual(c.strides,d.strides)
        e = ndarray_python_test.passConstFloatArray33(a)
        self.assertEqual(a.shape,e.shape)
        self.assertEqual(a.strides,e.strides)
        self.assert_(not e.flags.writeable)

    def testArrayCreation(self):
        a = ndarray_python_test.makeFloatArray3((3,4,5))

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

if __name__ == "__main__":
    unittest.main()
