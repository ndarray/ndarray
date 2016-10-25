import json
import shlex
import subprocess
import os.path
import re

class CompileTest(object):

    def __init__(self, subdir, template_target, preamble=""):
        with open("compile_commands.json", "r") as stream:
            d = json.load(stream)
        template = None
        for item in d:
            if template_target in item['file']:
                template = item['command']
                self.directory = os.path.join(item['directory'], subdir)
                break
        else:
            raise RuntimeError("Could not find template compile command.")
        def keep(term):
            if template_target in term:
                return False
            if term == "-o":
                return False
            if term == "-c":
                return False
            if term.startswith("-M"):
                return False
            if term.startswith("-fdiagnostics-color"):
                return False
            return True
        self.terms = [term for term in shlex.split(template) if keep(term)]
        self.preamble = preamble

    def tryCompile(self, name, snippet, in_main=True):
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        filename = os.path.join(self.directory, name + ".cpp")
        with open(filename, "w") as stream:
            stream.write(self.preamble)
            if in_main:
                stream.write("int main() {\n")
            stream.write(snippet)
            if in_main:
                stream.write("    return 0;\n}\n")
        return subprocess.run(self.terms + ["-c", filename],
                              universal_newlines=True,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE)

    def assertCompileSucceeds(self, name, snippet, msg=None,
                              print_stderr=True, **kwds):
        r = self.tryCompile(name, snippet, **kwds)
        if r.returncode:
            if msg is None:
                msg = "Snippet '{}' failed to compile.".format(name)
                if print_stderr:
                    msg = "\n".join([msg, r.stderr])
            self.fail(msg=msg)

    def assertCompileFails(self, name, snippet, msg=None,
                           stderr_regex=None, print_stderr=True, **kwds):
        r = self.tryCompile(name, snippet, **kwds)
        if not r.returncode:
            if msg is None:
                msg = "Snippet '{}' did not fail to compile.".format(name)
            self.fail(msg=msg)
        if stderr_regex is not None:
            if re.search(stderr_regex, r.stderr) is None:
                if msg is None:
                    msg = ("Did not match '{}' in compiler error message "
                           "for snippet '{}'.").format(stderr_regex, name)
                    if print_stderr:
                        msg = "\n".join([msg, r.stderr])
                self.fail(msg=msg)
