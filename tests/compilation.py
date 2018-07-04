#
# Copyright (c) 2010-2018, Jim Bosch
# All rights reserved.
#
# ndarray is distributed under a simple BSD-like license;
# see the LICENSE file that should be present in the root
# of the source distribution, or alternately available at:
# https://github.com/ndarray/ndarray
#

import os
import io
import json
import inspect
import subprocess
import re
import textwrap

__all__ = ("Compiler", "SnippetFormatter", "CompilationTestMixin")


COMPILE_COMMAND_CMAKE_TARGET = "tests/main.cpp"

OUTPUT_DIR = "compile_tests"


class Compiler:
    """An object that attempts to compile string code snippets.

    Parameters
    ----------
    args : `list`
        A command and command-line arguments, of the sort accepted as the
        ``args`` argment to `subprocess.run`.
    cwd : `str`
        Working directory used to compile the target file.  Should be passed
        as the ``cwd`` argment to `subprocess.run`.
    source_ext : `str`
        Filename extension to use for source files.
    output_ext : `str`
        Filename extension to use for output files.
    source_opt : `str`
        Command-line option indicating the source file.
    output_opt : `str`
        Command-line option indicating the output file.
    """
    def __init__(self, args, cwd, source_ext=".cpp", output_ext=".o", source_opt='-c', output_opt='-o'):
        self.args = args
        self.cwd = cwd
        self.source_ext = source_ext
        self.output_ext = output_ext
        self.source_opt = source_opt
        self.output_opt = output_opt

    @classmethod
    def fromCMake(cls, filename, source_opt='-c', output_opt='-o'):
        """Create a Compiler that uses CMake's compiler commands.

        This reads CMake's "compile_commands.json" (enabled with
        CMAKE_EXPORT_COMPILE_COMMANDS, only available for Makefile or Ninja
        builds), finds the source filename that ends with the given string (must
        be unique), and removes the source and output arguments so they can be
        specialized by __call__.

        Parameters
        ----------
        filename : `str`, path-like
            Name of a source file built by CMake whose build command we should
            copy.
        source_opt : `str`
            Command-line option indicating the source file.
        output_opt : `str`
            Command-line option indicating the output file.

        Returns
        -------
        compiler : `Compiler`
            New Compiler instance.
        """
        with open("compile_commands.json", "r") as f:
            targets = json.load(f)
        target, = [t for t in targets if t["file"].endswith(filename)]
        args = []
        next_is_source = False
        next_is_output = False
        source_ext = ""
        output_ext = ".o"
        # discard any -o or -c arguments; we'll supply those ourselves
        for arg in target["command"].split():
            if arg.startswith(output_opt):
                if arg == output_opt:
                    next_is_output = True
                _, output_ext = os.path.splitext(arg[2:])
                continue
            if arg.startswith(source_opt):
                if arg == source_opt:
                    next_is_source = True
                _, source_ext = os.path.splitext(arg[2:])
                continue
            if next_is_output:
                next_is_output = False
                _, output_ext = os.path.splitext(arg)
            if next_is_source:
                next_is_source = False
                _, source_ext = os.path.splitext(arg)
                continue
            args.append(arg)
        return cls(args=args, cwd=target["directory"], source_ext=source_ext, output_ext=output_ext,
                   source_opt=source_opt, output_opt=output_opt)

    def compile(self, name, code, output_dir=".", write_stderr=True, write_stdout=True):
        """Attempt to compile the given code snippet.

        Parameters
        ----------
        name : `str`
            Name of the snippet, used to create source and output filenames.
        code : `str`
            Block of source code to compile.
        output_dir : `str`, path-like
            Directory for both source and output files.
        write_stderr : `bool`
            If True, write a stderr to [output_dir]/[name].stderr if it would be
            non-empty.
        write_stdout : `bool`
            If True, write a stdout to [output_dir]/[name].stdout if it would be
            non-empty.

        Returns
        -------
        process : `subprocess.CompletedProcess`
            A struct containing information about the compiler process,
            including ``returncode`` (`int`), ``stdout`` (`str`), and
            ``stderr`` (`str`).
        output_file : `str`, path-like
            Name of the output file that should have been generated if
            the compiler was successful.
        """
        source_file = os.path.join(output_dir, name + self.source_ext)
        output_file = os.path.join(output_dir, name + self.output_ext)
        if not os.path.isdir(os.path.dirname(source_file)):
            os.makedirs(os.path.dirname(source_file))
        with open(source_file, 'w') as f:
            f.write(code)
        args = self.args + [self.source_opt, source_file,
                            self.output_opt, output_file]
        process = subprocess.run(args, cwd=self.cwd, universal_newlines=True, stderr=subprocess.PIPE,
                                 stdout=subprocess.PIPE)

        if process.stderr and write_stderr:
            with open(os.path.join(output_dir, name + ".stderr"), "w") as f:
                f.write(process.stderr)
        if process.stdout and write_stdout:
            with open(os.path.join(output_dir, name + ".stdout"), "w") as f:
                f.write(process.stdout)
        return process, output_file


compiler = Compiler.fromCMake(COMPILE_COMMAND_CMAKE_TARGET)


class SnippetFormatter:
    """A helper for formatting code snippets into compileable files.

    Parameters
    ----------
    preamble : `str` or `bool`
        A block of code to add to the beginning of every snippet.
        Typically #includes, using declarations, and common test classes.
        The preamble is always dedented to make it more natural to write
        as a multi-line string.
    prefix : `str` or `bool`
        A block of code inserted after ``preamble`` and before ``code``.
        Defaults to the opening of the main() function.
    suffix : `str` or `bool`
        A block of code inserted after ``code``.
        Defaults to the opening closing brace of the main() function
        started by ``prefix``.
    indent : `str` or `bool`
        Indentation to add to every line in ``code``.
    base : `SnippetFormatter`
        Another `SnippetFormatter` from which to inherit values.
        If provided, passing `True` (the default) to any other argument
        will cause that argument to be coped from ``base``.
    """

    def __init__(self, preamble=True, prefix=True, suffix=True, indent=True, base=None):
        self.preamble = preamble
        self.prefix = prefix
        self.suffix = suffix
        self.indent = indent
        self._partial = tuple(base._partial) if base is not None else tuple()

        def inherit_arg(arg, default):
            given = getattr(self, arg)
            if given is True:
                if base is not None:
                    setattr(self, arg, getattr(base, arg))
                else:
                    setattr(self, arg, default)
            elif given is False or given is None:
                setattr(self, arg, "")

        inherit_arg("preamble", "")
        inherit_arg("prefix", "int main() {")
        inherit_arg("indent", "    ")
        if isinstance(self.indent, int):
            self.indent = self.indent * " "
        inherit_arg("suffix", self.indent + "return 0;\n}")
        self.preamble = textwrap.dedent(self.preamble)

    def __call__(self, code):
        """Format the given code."""
        lines = list(self._partial)
        lines.extend(textwrap.dedent(code).splitlines())
        b = io.StringIO()
        b.write(self.preamble)
        b.write("\n")
        b.write(self.prefix)
        b.write("\n")
        for line in lines:
            b.write(self.indent)
            b.write(line)
            b.write("\n")
        b.write("\n")
        b.write(self.suffix)
        b.write("\n")
        return b.getvalue()

    def partial(self, code):
        r = SnippetFormatter(base=self)
        r._partial += tuple(textwrap.dedent(code).splitlines())
        return r


def make_snippet_name(frame):
    module, _ = os.path.splitext(os.path.basename(frame.filename))
    return "{}_{}_{}".format(module, frame.lineno, frame.function)


class CompilationTestMixin:
    """A unittest.TestCase mixin for testing code compilation.
    """

    def assertCompiles(self, code, name=None, output_dir=None, formatter=None, **kwds):
        """Assert that a code snippet does not compile.

        Parameters
        ----------
        lines : `str`
            Code to be compiled.
        name : `str`, optional
            Name of the snippet; used to generate filenames for source and
            output files.  If not provided, the calling method's
            module, line number, and name are used to construct a name.
        output_dir : `str`, path-like
            Directory for source and output files.
        formatter : callable, optional
            A callable to apply to the code before compiling it (typically
            a `SnippetFormatter`).   If provided, the original code snippet
            will be included in any test failure messages.

        Additional keyword arguments are passed to Compiler.compile.
        """
        if name is None:
            name = make_snippet_name(inspect.stack()[1])
        if output_dir is None:
            output_dir = OUTPUT_DIR
        if formatter is not None:
            code = formatter(code)
        process, _ = compiler.compile(name, code, output_dir=output_dir, **kwds)
        self.assertEqual(process.returncode, 0, msg="{} did not compile".format(name))

    def assertDoesNotCompile(self, code, name=None, stderr_regex=None, stdout_regex=None,
                             output_dir=None, formatter=None, **kwds):
        """Assert that a code snippet does not compile.

        Parameters
        ----------
        lines : `str`
            Code to be compiled.
        name : `str`, optional
            Name of the snippet; used to generate filenames for source and
            output files.  If not provided, the calling method's
            module, line number, and name are used to construct a name.
        stderr_regex : `str` or compiled regular expression
            A regular expression to search for in the compiler's STDERR.  If
            provided, failure to find this expression will also result in
            an assertion failure.
        stdout_regex : `str` or compiled regular expression
            Like ``stderr_regex``, but applied to STDOUT.
        output_dir : `str`, path-like
            Directory for source and output files.
        formatter : callable, optional
            A callable to apply to the code before compiling it (typically
            a `SnippetFormatter`).

        Additional keyword arguments are passed to Compiler.compile.
        """
        if name is None:
            name = make_snippet_name(inspect.stack()[1])
        if output_dir is None:
            output_dir = OUTPUT_DIR
        if formatter is not None:
            code = formatter(code)
        process, _ = compiler.compile(name, code, output_dir=output_dir, **kwds)
        self.assertNotEqual(process.returncode, 0, msg="{} unexpectedly compiled".format(name))
        if stderr_regex is not None:
            self.assertTrue(
                re.search(stderr_regex, process.stderr) is not None,
                msg="'{}' not found in stderr (see {}.stderr).".format(
                    stderr_regex, os.path.join(output_dir, name)
                )
            )
        if stdout_regex is not None:
            self.assertTrue(
                re.search(stdout_regex, process.stdout) is not None,
                msg="'{}' not found in stdout (see {}.stdout).".format(
                    stdout_regex, os.path.join(output_dir, name)
                )
            )
