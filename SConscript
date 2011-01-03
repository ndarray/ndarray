Import("base_env", "build_dir")
import scons_tools

targets = {}

scons_tools.LocalConfiguration(
    name="boost.python.eigen",
    dependencies=("boost.python.numpy", "eigen")
    )

if scons_tools.database["eigen"].check():
    bp_eigen_env = base_env.Clone()
    bp_eigen_env.SetupPackages(scons_tools.database["boost.python.eigen"].dependencies)
    Export("bp_eigen_env")
    targets["install"] = (
        bp_eigen_env.RecursiveInstall(
            "#include/boost/python", 
            "boost/python", 
            regex = "(.*\.hpp)"
            )
        )
    targets["test"] = SConscript("libs/python/eigen/test/SConscript",
                                 variant_dir="%s/python/eigen/test" % build_dir)
else:
    print "Eigen library not found, skipping 'boost.python.eigen' targets."

Return("targets")
