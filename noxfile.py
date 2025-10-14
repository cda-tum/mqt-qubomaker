# Copyright (c) 2024 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Nox sessions."""

from __future__ import annotations

import argparse
import contextlib
import os
import shutil
import tempfile
from typing import TYPE_CHECKING

import nox

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence

nox.needs_version = ">=2024.3.2"
nox.options.default_venv_backend = "uv"

nox.options.sessions = ["lint", "tests", "minimums"]

PYTHON_ALL_VERSIONS = ["3.10", "3.11", "3.12", "3.13"]

if os.environ.get("CI", None):
    nox.options.error_on_missing_interpreters = True


@contextlib.contextmanager
def preserve_lockfile() -> Generator[None]:
    """Preserve the lockfile by moving it to a temporary directory."""
    with tempfile.TemporaryDirectory() as temp_dir_name:
        shutil.move("uv.lock", f"{temp_dir_name}/uv.lock")
        try:
            yield
        finally:
            shutil.move(f"{temp_dir_name}/uv.lock", "uv.lock")


@nox.session(reuse_venv=True)
def lint(session: nox.Session) -> None:
    """Run the linter."""
    if shutil.which("pre-commit") is None:
        session.install("pre-commit")

    session.run("pre-commit", "run", "--all-files", *session.posargs, external=True)


def _run_tests(
    session: nox.Session,
    *,
    install_args: Sequence[str] = (),
    extra_command: Sequence[str] = (),
    pytest_run_args: Sequence[str] = (),
) -> None:
    env = {"UV_PROJECT_ENVIRONMENT": session.virtualenv.location}
    if shutil.which("cmake") is None and shutil.which("cmake3") is None:
        session.install("cmake")
    if shutil.which("ninja") is None:
        session.install("ninja")

    # install build and test dependencies on top of the existing environment
    session.run(
        "uv",
        "sync",
        "--inexact",
        "--only-group",
        "build",
        "--only-group",
        "test",
        *install_args,
        env=env,
    )
    print(session.python)
    add_tsplib = [] if session.python == "3.13" else ["--extra", "tsplib"]

    session.run(
        "uv",
        "sync",
        "--inexact",
        "--no-dev",  # do not auto-install dev dependencies
        "--no-build-isolation-package",
        "mqt-qubomaker",  # build the project without isolation
        "--extra",
        "check",
        *add_tsplib,
        *install_args,
        env=env,
    )
    if extra_command:
        session.run(*extra_command, env=env)
    if "--cov" in session.posargs:
        # try to use the lighter-weight `sys.monitoring` coverage core
        env["COVERAGE_CORE"] = "sysmon"
    session.run(
        "uv",
        "run",
        "--no-sync",  # do not sync as everything is already installed
        *install_args,
        "pytest",
        *pytest_run_args,
        *session.posargs,
        "--cov-config=pyproject.toml",
        env=env,
    )


@nox.session(reuse_venv=True, python=PYTHON_ALL_VERSIONS)
def tests(session: nox.Session) -> None:
    """Run the test suite."""
    _run_tests(session)


@nox.session(reuse_venv=True, venv_backend="uv", python=PYTHON_ALL_VERSIONS)
def minimums(session: nox.Session) -> None:
    """Test the minimum versions of dependencies."""
    with preserve_lockfile():
        _run_tests(
            session,
            install_args=["--resolution=lowest-direct"],
            pytest_run_args=["-Wdefault"],
        )
        env = {"UV_PROJECT_ENVIRONMENT": session.virtualenv.location}
        session.run("uv", "tree", "--frozen", env=env)


@nox.session(reuse_venv=True, venv_backend="uv", python=PYTHON_ALL_VERSIONS)
def qiskit(session: nox.Session) -> None:
    """Tests against the latest version of Qiskit."""
    with preserve_lockfile():
        _run_tests(
            session,
            extra_command=["uv", "pip", "install", "qiskit[qasm3-import] @ git+https://github.com/Qiskit/qiskit.git"],
        )
        env = {"UV_PROJECT_ENVIRONMENT": session.virtualenv.location}
        session.run("uv", "pip", "show", "qiskit", env=env)


@nox.session(reuse_venv=True)
def docs(session: nox.Session) -> None:
    """Build the docs. Use "--non-interactive" to avoid serving. Pass "-b linkcheck" to check links."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", dest="builder", default="html", help="Build target (default: html)")
    args, posargs = parser.parse_known_args(session.posargs)

    serve = args.builder == "html" and session.interactive
    if serve:
        session.install("sphinx-autobuild")

    env = {"UV_PROJECT_ENVIRONMENT": session.virtualenv.location}
    # install build and docs dependencies on top of the existing environment
    session.run(
        "uv",
        "sync",
        "--inexact",
        "--only-group",
        "build",
        "--only-group",
        "docs",
        env=env,
    )

    shared_args = [
        "-n",  # nitpicky mode
        "-T",  # full tracebacks
        f"-b={args.builder}",
        "docs",
        f"docs/_build/{args.builder}",
        *posargs,
    ]

    session.run(
        "uv",
        "run",
        "--no-dev",  # do not auto-install dev dependencies
        "--no-build-isolation-package",
        "mqt-qubomaker",  # build the project without isolation
        "sphinx-autobuild" if serve else "sphinx-build",
        *shared_args,
        env=env,
    )
