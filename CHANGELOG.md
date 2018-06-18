# ndarray change log

## 1.5.1

### Bug fixes

Reverted a non-portable workaround for bad module suffixes with pybind11 2.1.x.  Please just use pybind11 2.2.x instead.

## 1.5.0

### New features

`ndarray::EigenView` is not compatible with Eigen 3.3 and fixing this appears to be more work than it's worth,
so this version deprecates `ndarray::EigenView` and adds new free functions `asEigen`, `asEigenArray`
and `asEigenMatrix` which return an `eigen::Map` view of an ndarray array. Unlike `ndarray::EigenView`,
the returned eigen map does not own the memory, so you must be careful to keep the ndarray array around
until you are done with the map.

Added build option `-DNDARRAY_EIGENVIEW` which controls whether to build ndarray::EigenView. The default is OFF.

Added build option `-DNDARRAY_STDPYBIND11EIGEN` which, if ON, imports `pybind11/eigen.h` into `ndarray/eigen.h`.
The intent is to make it easier to port old pybind11 wrappers that used EigenView. The default is `OFF`.
`-DNDARRAY_EIGENVIEW` and `-DNDARRAY_STDPYBIND11EIGEN` cannot both be `ON`.

### Bug fixes

Fixed a race condition in the cmake build files: ArrayBaseN.h.m4 might not be processed before other
m4 include files that rely on it. This may fix ndarray issue 71.
