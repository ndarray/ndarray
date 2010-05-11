import os
import distutils.sysconfig
import sys
import scons_tools

base_env = scons_tools.makeEnvironment(local_include="include")
base_env = scons_tools.configure(base_env,packages=("boost",),)

Export("base_env")

doc = SConscript(os.path.join("doc","SConscript"))

paths = scons_tools.getInstallPaths()

base_env.Append(M4FLAGS="-I%s" % os.path.join(os.path.abspath('.'),'m4'))
generated = ["include/ndarray/ArrayRef.hpp",
             "include/ndarray/operators.hpp",
             "include/ndarray/Vector.hpp",
             "include/ndarray/fft/FFTWTraits.hpp",
             ]
headers = [base_env.M4(filename,"%s.m4" % filename) for filename in generated]
base_env.Depends(headers,Glob("#m4/*.m4"))

install_headers = base_env.InstallSource(paths['include'], "include", patterns=("*.hpp","*.cc"))
AlwaysBuild(install_headers)
install = Alias("install", install_headers)

Default(headers)

if "test" in COMMAND_LINE_TARGETS:
    tests = SConscript(os.path.join("tests","SConscript"))
