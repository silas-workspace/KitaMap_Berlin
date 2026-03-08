import nox

nox.options.sessions = ["lint", "typecheck"]


@nox.session
def lint(session: nox.Session) -> None:
    """Run ruff linting and formatting checks."""
    session.install("ruff")
    session.run("ruff", "check", "src/", "run_analysis.py", "main.py")
    session.run("ruff", "format", "--check", "src/", "run_analysis.py", "main.py")


@nox.session
def typecheck(session: nox.Session) -> None:
    """Run pyright type checking."""
    session.install("-r", "requirements.txt")
    session.install("pyright", "pandas-stubs")
    session.run("pyright", "src/", "run_analysis.py", "main.py")
