This branch is a from-the-ground rewrite of the ndarray library, with the following goals:

 - Simplify the codebase and its dependencies by leveraging C++11/14 features.  Ideally this will include removing Boost as a dependency (aside from optional Boost.Python bindings).

 - Support both Python 2 and Python 3 via pybind11, Boost.Python, and possibly Swig and Cython.

 - Remove internal expression template implementations in favor of delegating more work to Eigen (which may become a required dependency).

 - Add support for record dtypes and non-POD arrays.

When the rewrite reaches a usable state, the current master version will be archived to a stable branch and the devel branch will become the new master.
