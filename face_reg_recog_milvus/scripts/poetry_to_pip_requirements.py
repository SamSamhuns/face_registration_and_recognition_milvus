"""Generate requirements.txt from pyproject.toml."""

import tomllib
from pathlib import Path


def poetry_to_pip(name: str, constraint: str) -> str:
    # handle wildcard / no version
    if constraint in ("*", "", None):
        return name

    # caret constraints
    if constraint.startswith("^"):
        base_version = constraint[1:]
        parts = [int(p) for p in base_version.split(".")]
        # expand ^X.Y.Z to >=X.Y.Z,<X+1.0.0 if X > 0
        if parts[0] > 0:
            upper = f"{parts[0] + 1}.0.0"
        elif parts[1] > 0:
            upper = f"0.{parts[1] + 1}.0"
        else:
            upper = f"0.0.{parts[2] + 1}"
        return f"{name}>={base_version},<{upper}"

    # tilde constraints (~X.Y) → >=X.Y,<X.(Y+1)
    if constraint.startswith("~"):
        base_version = constraint[1:]
        parts = [int(p) for p in base_version.split(".")]
        upper = f"{parts[0] + 1}.0.0" if len(parts) == 1 else f"{parts[0]}.{parts[1] + 1}.0"
        return f"{name}>={base_version},<{upper}"

    # already pip-compatible
    return f"{name}{constraint}"


with open("../pyproject.toml", "rb") as f:
    pyproject = tomllib.load(f)

# project deps (strings already formatted like "torch (>=2.8.0,<3.0.0)")
deps = pyproject["project"]["dependencies"]
# test deps (dict of name -> version constraint)
test_deps = pyproject["tool"]["poetry"]["group"]["test"]["dependencies"]
# dev deps (dict of name -> version constraint)
dev_deps = pyproject["tool"]["poetry"]["group"]["dev"]["dependencies"]

# write train requirements
with open("requirements.txt", "w") as f:
    for dep in deps:
        f.write(dep + "\n")

# write test requirements
Path("tests").mkdir(exist_ok=True)
with open("tests/requirements.txt", "w") as f:
    for pkg, constraint in test_deps.items():
        f.write(poetry_to_pip(pkg, constraint) + "\n")
with open("tests/requirements.txt", "a") as f:
    for pkg, constraint in dev_deps.items():
        f.write(poetry_to_pip(pkg, constraint) + "\n")
