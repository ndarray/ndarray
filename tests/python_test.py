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
import python_test_mod
import unittest

class TestNDArray(unittest.TestCase):

    def testIntArrayConversion(self):
        a1 = numpy.zeros((5,3,4),dtype=numpy.int32)
        a2 = a1[:,:,:2]
        self.assert_(a1.flags["WRITEABLE"])
        self.assert_(a2.flags["WRITEABLE"])

        ua1 = numpy.zeros((5,3,4),dtype=numpy.uint32)
        self.assert_(ua1.flags["WRITEABLE"])

        b1 = python_test_mod.passIntArray33(a1)
        self.assert_(b1.flags["WRITEABLE"])
        self.assertEqual(a1.shape,b1.shape)
        self.assertEqual(a1.strides,b1.strides)

        ub1 = python_test_mod.passuIntArray33(ua1)
        self.assert_(ub1.flags["WRITEABLE"])
        self.assertEqual(a1.shape,ub1.shape)
        self.assertEqual(a1.strides,ub1.strides)

        c1 = python_test_mod.passIntArray30(a1)
        self.assert_(c1.flags["WRITEABLE"])
        self.assertEqual(a1.shape,c1.shape)
        self.assertEqual(a1.strides,c1.strides)

        d1 = python_test_mod.passConstIntArray33(a1)
        self.assertEqual(a1.shape,d1.shape)
        self.assertEqual(a1.strides,d1.strides)
        self.assert_(not d1.flags.writeable)

        self.assertRaises(ValueError, python_test_mod.passIntArray33, a2)

        c2 = python_test_mod.passIntArray30(a2)
        self.assert_(c2.flags["WRITEABLE"])
        self.assertEqual(a2.shape,c2.shape)
        self.assertEqual(a2.strides,c2.strides)

        # test to be sure a tuple of tuples does not work
        self.assertRaises(TypeError, python_test_mod.passIntArray30, ((1,1,1),(2,2,2),(3,3,3)))

        # test to be sure an array of the wrong type does not work
        self.assertRaises(ValueError, python_test_mod.passIntArray30,
                          numpy.zeros((4,3,4),dtype=numpy.float))

    def testVectorConversion(self):
        a = (3.5,6.7,1.2)
        b = python_test_mod.passFloatVector3(a)
        self.assertEqual(a,b)

    def testArrayConversion(self):
        a1 = numpy.zeros((5,3,4),dtype=float)
        a2 = a1[:,:,:2]
        self.assert_(a1.flags["WRITEABLE"])
        self.assert_(a2.flags["WRITEABLE"])

        b1 = python_test_mod.passFloatArray33(a1)
        self.assert_(b1.flags["WRITEABLE"])
        self.assertEqual(a1.shape,b1.shape)
        self.assertEqual(a1.strides,b1.strides)


        c1 = python_test_mod.passFloatArray30(a1)
        self.assert_(c1.flags["WRITEABLE"])
        self.assertEqual(a1.shape,c1.shape)
        self.assertEqual(a1.strides,c1.strides)

        d1 = python_test_mod.passConstFloatArray33(a1)
        self.assertEqual(a1.shape,d1.shape)
        self.assertEqual(a1.strides,d1.strides)
        self.assert_(not d1.flags.writeable)

        self.assertRaises(ValueError, python_test_mod.passFloatArray33, a2)

        c2 = python_test_mod.passFloatArray30(a2)
        self.assert_(c2.flags["WRITEABLE"])
        self.assertEqual(a2.shape,c2.shape)
        self.assertEqual(a2.strides,c2.strides)

        # test to be sure a tuple of tuples does not work
        self.assertRaises(TypeError, python_test_mod.passFloatArray30,
                          ((1.0,1.0,1.0),(2.0,2.0,2.0),(3.0,3.0,3.0)))

        # test to be sure an array of the wrong type does not work
        self.assertRaises(ValueError, python_test_mod.passFloatArray30,
                          numpy.zeros((4,3,4),dtype=numpy.int32))

        # test to be sure a tuple of tuples does not work
        self.assertRaises(TypeError, python_test_mod.passFloatArray30,
                          ((1.0,1.0,1.0),(2.0,2.0,2.0),(3.0,3.0,3.0)))

        # test to be sure an array of the wrong type does not work
        self.assertRaises(ValueError, python_test_mod.passFloatArray30,
                          numpy.zeros((4,3,4),dtype=numpy.int32))

    def testFloatArrayCreation(self):
        a = python_test_mod.makeFloatArray3((3,4,5))

    def testIntArrayCreation(self):
        a = python_test_mod.makeIntArray3((3,4,5))

if __name__ == "__main__":
    unittest.main()
