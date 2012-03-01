Import("bp_numpy_env")
import scons_tools

targets = {}

scons_tools.LocalConfiguration(
    name="boost.python.ndarray",
    libraries=["boost_python_ndarray"],
    dependencies=("boost.python.numpy", "ndarray", "eigen")
    )

if scons_tools.database["ndarray"].check():
    bp_ndarray_env = bp_numpy_env.Clone()
    bp_ndarray_env.SetupPackages(scons_tools.database["boost.python.ndarray"].dependencies)
    Export("bp_ndarray_env")
    targets["lib"] = SConscript("libs/python/ndarray/src/SConscript")
    targets["install"] = ( 
        bp_ndarray_env.RecursiveInstall(
            "#include/boost/python", 
            "boost/python", 
            regex = "(.*\.hpp)"
            )
        + bp_ndarray_env.Install(bp_ndarray_env["INSTALL_LIB"], targets["lib"])
        )
    targets["test"] = SConscript("libs/python/ndarray/test/SConscript")
else:
    print "ndarray library not found, skipping 'boost.python.ndarray' targets."

Return("targets")
