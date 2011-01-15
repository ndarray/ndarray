Import("bp_numpy_env")
import scons_tools

targets = {}

scons_tools.LocalConfiguration(
    name="boost.python.ndarray",
    dependencies=("boost.python.numpy", "ndarray", "eigen")
    )

if scons_tools.database["ndarray"].check():
    bp_ndarray_env = bp_numpy_env.Clone()
    bp_ndarray_env.SetupPackages(scons_tools.database["boost.python.ndarray"].dependencies)
    Export("bp_ndarray_env")
    targets["install"] = ( 
        bp_ndarray_env.RecursiveInstall(
            "#include/boost/python", 
            "boost/python", 
            regex = "(.*\.hpp)"
            )
        )
    targets["test"] = SConscript("libs/python/ndarray/test/SConscript", 
                                 variant_dir="%s/python/test" % scons_tools.GetBuildDir())
else:
    print "ndarray library not found, skipping 'boost.python.ndarray' targets."

Return("targets")
