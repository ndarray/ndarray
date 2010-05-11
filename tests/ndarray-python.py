import numpy
import ndarray_python_test
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

if __name__ == "__main__":
    unittest.main()
