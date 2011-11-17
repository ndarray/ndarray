#!/usr/bin/env python
import numpy
import ndarray_python_test
import swig_test_mod
import unittest

class TestNDArray(unittest.TestCase):

    def testIntArrayConversion(self):
        a1 = numpy.zeros((5,3,4),dtype=numpy.int32)
        a2 = a1[:,:,:2]
        self.assert_(a1.flags["WRITEABLE"])
        self.assert_(a2.flags["WRITEABLE"])

        b1 = ndarray_python_test.passIntArray33(a1)
        self.assert_(b1.flags["WRITEABLE"])
        self.assertEqual(a1.shape,b1.shape)
        self.assertEqual(a1.strides,b1.strides)

        c1 = ndarray_python_test.passIntArray30(a1)
        self.assert_(c1.flags["WRITEABLE"])
        self.assertEqual(a1.shape,c1.shape)
        self.assertEqual(a1.strides,c1.strides)

        d1 = ndarray_python_test.passConstIntArray33(a1)
        self.assertEqual(a1.shape,d1.shape)
        self.assertEqual(a1.strides,d1.strides)
        self.assert_(not d1.flags.writeable)

        self.assertRaises(ValueError, ndarray_python_test.passIntArray33, a2)

        c2 = ndarray_python_test.passIntArray30(a2)
        self.assert_(c2.flags["WRITEABLE"])
        self.assertEqual(a2.shape,c2.shape)
        self.assertEqual(a2.strides,c2.strides)

        # test to be sure a tuple of tuples does not work
        self.assertRaises(TypeError, ndarray_python_test.passIntArray30, ((1,1,1),(2,2,2),(3,3,3)))

        # test to be sure an array of the wrong type does not work
        self.assertRaises(ValueError, ndarray_python_test.passIntArray30, 
                          numpy.zeros((4,3,4),dtype=numpy.float))

    def testVectorConversion(self):
        a = (3.5,6.7,1.2)
        b = ndarray_python_test.passFloatVector3(a)
        self.assertEqual(a,b)

    def testArrayConversion(self):
        a1 = numpy.zeros((5,3,4),dtype=float)
        a2 = a1[:,:,:2]
        self.assert_(a1.flags["WRITEABLE"])
        self.assert_(a2.flags["WRITEABLE"])

        b1 = ndarray_python_test.passFloatArray33(a1)
        self.assert_(b1.flags["WRITEABLE"])
        self.assertEqual(a1.shape,b1.shape)
        self.assertEqual(a1.strides,b1.strides)


        c1 = ndarray_python_test.passFloatArray30(a1)
        self.assert_(c1.flags["WRITEABLE"])
        self.assertEqual(a1.shape,c1.shape)
        self.assertEqual(a1.strides,c1.strides)

        d1 = ndarray_python_test.passConstFloatArray33(a1)
        self.assertEqual(a1.shape,d1.shape)
        self.assertEqual(a1.strides,d1.strides)
        self.assert_(not d1.flags.writeable)

        self.assertRaises(ValueError, ndarray_python_test.passFloatArray33, a2)

        c2 = ndarray_python_test.passFloatArray30(a2)
        self.assert_(c2.flags["WRITEABLE"])
        self.assertEqual(a2.shape,c2.shape)
        self.assertEqual(a2.strides,c2.strides)

        # test to be sure a tuple of tuples does not work
        self.assertRaises(TypeError, ndarray_python_test.passFloatArray30, 
                          ((1.0,1.0,1.0),(2.0,2.0,2.0),(3.0,3.0,3.0)))

        # test to be sure an array of the wrong type does not work
        self.assertRaises(ValueError, ndarray_python_test.passFloatArray30,
                          numpy.zeros((4,3,4),dtype=numpy.int32))

        # test to be sure a tuple of tuples does not work
        self.assertRaises(TypeError, ndarray_python_test.passFloatArray30,
                          ((1.0,1.0,1.0),(2.0,2.0,2.0),(3.0,3.0,3.0)))

        # test to be sure an array of the wrong type does not work
        self.assertRaises(ValueError, ndarray_python_test.passFloatArray30,
                          numpy.zeros((4,3,4),dtype=numpy.int32))

    def testFloatArrayCreation(self):
        a = ndarray_python_test.makeFloatArray3((3,4,5))
    
    def testIntArrayCreation(self):
        a = ndarray_python_test.makeIntArray3((3,4,5))

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
        

if __name__ == "__main__":
    unittest.main()
