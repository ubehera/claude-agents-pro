---
name: packaging-distribution
description: Load when user needs pyproject.toml, Poetry, Hatch, wheels, PyPI publishing, or Python package distribution and dependency management patterns
trigger_keywords: [packaging, pyproject.toml, poetry, hatch, setup.py, setuptools, wheel, sdist, pypi, pip install, package distribution, dependency management, build system]
---

# Packaging & Distribution Skill

Modern Python packaging with pyproject.toml, Poetry, Hatch, and standardized build tools for library and application distribution.

## Overview

Python packaging evolved from setup.py to pyproject.toml (PEP 517, 518, 621). Modern tools like Poetry and Hatch provide dependency management, virtual environments, and publishing workflows.

**When to Use**:
- Creating distributable Python libraries
- Managing application dependencies
- Publishing packages to PyPI
- Building reproducible environments

## Core Concepts

### pyproject.toml Standard (PEP 621)

**Minimal pyproject.toml**:
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "my-package"
version = "0.1.0"
description = "A sample Python package"
authors = [
    {name = "Your Name", email = "you@example.com"}
]
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "requests>=2.31.0",
    "pydantic>=2.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "mypy>=1.5.0",
    "ruff>=0.0.290",
]

[project.urls]
Homepage = "https://github.com/username/my-package"
Documentation = "https://my-package.readthedocs.io"
```

### Build System Comparison

**setuptools (Traditional)**:
```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]
include = ["mypackage*"]
```

**Hatchling (Modern)**:
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/mypackage"]
```

**Poetry (Opinionated)**:
```toml
[build-system]
requires = ["poetry-core>=1.0.0"]
build-backend = "poetry.core.masonry.api"

[tool.poetry]
name = "my-package"
version = "0.1.0"
description = ""
authors = ["Your Name <you@example.com>"]

[tool.poetry.dependencies]
python = "^3.11"
requests = "^2.31.0"

[tool.poetry.group.dev.dependencies]
pytest = "^7.4.0"
```

## Dependency Management

### Poetry Workflow

**Installation**:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

**Project Setup**:
```bash
# Create new project
poetry new my-package
cd my-package

# Or initialize in existing directory
poetry init

# Add dependencies
poetry add requests pydantic

# Add dev dependencies
poetry add --group dev pytest mypy ruff

# Install dependencies
poetry install

# Run commands in virtual env
poetry run python script.py
poetry run pytest

# Update dependencies
poetry update

# Build package
poetry build

# Publish to PyPI
poetry publish
```

**poetry.lock**:
```toml
# Auto-generated, commit to version control
# Ensures reproducible installs across environments
[[package]]
name = "requests"
version = "2.31.0"
description = "Python HTTP for Humans."
```

### Hatch Workflow

**Installation**:
```bash
pip install hatch
```

**Project Setup**:
```bash
# Create new project
hatch new my-package
cd my-package

# Create environment
hatch env create

# Run commands
hatch run python script.py
hatch run pytest

# Build package
hatch build

# Publish to PyPI
hatch publish
```

**Environment Configuration**:
```toml
[tool.hatch.envs.default]
dependencies = [
    "pytest",
    "pytest-cov",
]

[tool.hatch.envs.default.scripts]
test = "pytest {args}"
cov = "pytest --cov-report=term-missing --cov=mypackage {args}"

[tool.hatch.envs.lint]
detached = true
dependencies = [
    "ruff",
    "mypy",
]

[tool.hatch.envs.lint.scripts]
typing = "mypy --install-types --non-interactive {args:src/mypackage}"
style = "ruff check {args:src}"
```

### PDM (Alternative)

**Installation**:
```bash
pip install pdm
```

**Features**:
- PEP 582 support (no virtual env needed)
- Fast dependency resolution
- Lock file compatible with pip

```bash
pdm init
pdm add requests
pdm add -dG test pytest
pdm install
pdm run pytest
```

## Package Structure

### Source Layout (Recommended)

```
my-package/
├── src/
│   └── mypackage/
│       ├── __init__.py
│       ├── core.py
│       └── utils.py
├── tests/
│   ├── __init__.py
│   └── test_core.py
├── docs/
├── pyproject.toml
├── README.md
└── LICENSE
```

**Advantages**:
- Prevents accidental import of development version
- Forces proper installation for testing
- Cleaner namespace

**pyproject.toml**:
```toml
[tool.setuptools.packages.find]
where = ["src"]

[tool.hatch.build.targets.wheel]
packages = ["src/mypackage"]
```

### Flat Layout (Simple Projects)

```
my-package/
├── mypackage/
│   ├── __init__.py
│   ├── core.py
│   └── utils.py
├── tests/
├── pyproject.toml
└── README.md
```

## Version Management

### Static Version

```toml
[project]
version = "1.2.3"
```

### Dynamic Version from Git Tags

```toml
[build-system]
requires = ["hatchling", "hatch-vcs"]

[tool.hatch.version]
source = "vcs"

[project]
dynamic = ["version"]
```

**Git workflow**:
```bash
git tag v1.0.0
git push --tags
hatch build  # Version automatically set to 1.0.0
```

### Dynamic Version from File

```toml
[tool.hatch.version]
path = "src/mypackage/__init__.py"
```

```python
# src/mypackage/__init__.py
__version__ = "1.0.0"
```

## Building & Distribution

### Build Artifacts

```bash
# Build both wheel and sdist
python -m build

# Or with Poetry
poetry build

# Or with Hatch
hatch build

# Output:
# dist/
#   mypackage-1.0.0-py3-none-any.whl  (wheel)
#   mypackage-1.0.0.tar.gz             (source distribution)
```

**Wheel vs Source Distribution**:
- **Wheel (.whl)**: Pre-built, fast installation, platform-specific
- **Sdist (.tar.gz)**: Source code, requires build step, universal

### Publishing to PyPI

**Test PyPI First**:
```bash
# Poetry
poetry config repositories.testpypi https://test.pypi.org/legacy/
poetry publish -r testpypi

# Twine (manual)
pip install twine
twine upload --repository testpypi dist/*

# Install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ mypackage
```

**Production PyPI**:
```bash
# Poetry
poetry publish

# Twine
twine upload dist/*

# Install
pip install mypackage
```

### PyPI Credentials

```bash
# Create ~/.pypirc
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-AgEIcHlwaS5vcmc...

[testpypi]
username = __token__
password = pypi-AgENdGVzdC5weXBp...
```

## Advanced Patterns

### Entry Points (CLI Scripts)

```toml
[project.scripts]
mycommand = "mypackage.cli:main"

[project.gui-scripts]
myapp-gui = "mypackage.gui:main"
```

```python
# src/mypackage/cli.py
def main():
    print("Hello from CLI!")

if __name__ == "__main__":
    main()
```

**After installation**:
```bash
pip install mypackage
mycommand  # Runs mypackage.cli:main()
```

### Plugin System

```toml
[project.entry-points."myapp.plugins"]
csv_reader = "mypackage.plugins.csv:CSVReader"
json_reader = "mypackage.plugins.json:JSONReader"
```

```python
# Discover plugins
from importlib.metadata import entry_points

discovered_plugins = entry_points(group='myapp.plugins')
for plugin in discovered_plugins:
    reader_class = plugin.load()
```

### Data Files

**Include Non-Python Files**:
```toml
[tool.hatch.build.targets.wheel]
include = [
    "/src/mypackage",
]
packages = ["src/mypackage"]

[tool.hatch.build.targets.wheel.force-include]
"data" = "mypackage/data"
```

**Access at Runtime**:
```python
from importlib.resources import files

def load_config():
    config_path = files('mypackage') / 'data' / 'config.json'
    return json.loads(config_path.read_text())
```

### Conditional Dependencies

```toml
[project.optional-dependencies]
# Install with: pip install mypackage[postgres]
postgres = ["psycopg2-binary>=2.9.0"]
mysql = ["mysqlclient>=2.2.0"]
all = ["psycopg2-binary>=2.9.0", "mysqlclient>=2.2.0"]

# Platform-specific
[tool.poetry.dependencies]
pywin32 = {version = "^306", markers = "sys_platform == 'win32'"}
```

## Production-Ready Examples

### Complete pyproject.toml

```toml
[build-system]
requires = ["hatchling>=1.18.0"]
build-backend = "hatchling.build"

[project]
name = "mylib"
version = "1.0.0"
description = "Production-grade Python library"
authors = [
    {name = "Your Name", email = "you@example.com"}
]
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.11"
keywords = ["api", "client", "http"]
classifiers = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]

dependencies = [
    "httpx>=0.25.0",
    "pydantic>=2.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "mypy>=1.5.0",
    "ruff>=0.1.0",
]

[project.urls]
Homepage = "https://github.com/username/mylib"
Documentation = "https://mylib.readthedocs.io"
Repository = "https://github.com/username/mylib.git"
Changelog = "https://github.com/username/mylib/blob/main/CHANGELOG.md"

[project.scripts]
mylib-cli = "mylib.cli:main"

[tool.hatch.build.targets.wheel]
packages = ["src/mylib"]

[tool.hatch.envs.default]
dependencies = [
    "pytest",
    "pytest-cov",
]

[tool.hatch.envs.default.scripts]
test = "pytest {args}"
cov = "pytest --cov-report=term-missing --cov=mylib {args}"

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "--cov=mylib --cov-report=html --cov-report=term-missing"

[tool.mypy]
python_version = "3.11"
strict = true
warn_return_any = true
warn_unused_configs = true

[tool.ruff]
target-version = "py311"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "N", "UP", "B", "A", "C4", "DTZ", "T10", "EM", "ISC", "ICN", "G", "PIE", "T20", "PYI", "PT", "Q", "RSE", "RET", "SLF", "SIM", "TID", "TCH", "ARG", "PTH", "ERA", "PD", "PGH", "PL", "TRY", "NPY", "RUF"]
```

### CI/CD with GitHub Actions

```yaml
# .github/workflows/publish.yml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v4
        with:
          python-version: "3.11"

      - name: Install Hatch
        run: pip install hatch

      - name: Build package
        run: hatch build

      - name: Publish to PyPI
        env:
          HATCH_INDEX_USER: __token__
          HATCH_INDEX_AUTH: ${{ secrets.PYPI_API_TOKEN }}
        run: hatch publish
```

## Best Practices

### 1. Use pyproject.toml (Not setup.py)
```toml
# ✅ Modern standard
[project]
name = "mypackage"
version = "1.0.0"
```

### 2. Pin Dependencies in Lock Files
```bash
# ✅ Commit poetry.lock or pdm.lock
git add poetry.lock
git commit -m "feat: add dependency locking"
```

### 3. Specify Python Version Range
```toml
[project]
requires-python = ">=3.11,<4.0"  # Clear compatibility
```

### 4. Use Semantic Versioning
```
MAJOR.MINOR.PATCH
1.0.0 - Initial release
1.1.0 - New feature (backwards compatible)
1.1.1 - Bug fix
2.0.0 - Breaking change
```

### 5. Include Comprehensive Metadata
```toml
[project]
keywords = ["api", "client"]
classifiers = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
]
```

## Common Pitfalls

❌ **Committing virtual environments**
```bash
# .gitignore
.venv/
venv/
*.egg-info/
dist/
```

❌ **Not specifying minimum Python version**
```toml
# ❌ No version specified
[project]
name = "mypackage"

# ✅ Clear requirement
requires-python = ">=3.11"
```

❌ **Forgetting to include package data**
```toml
# ❌ Data files not included
[tool.hatch.build.targets.wheel]
packages = ["src/mypackage"]

# ✅ Include data
[tool.hatch.build.targets.wheel.force-include]
"data" = "mypackage/data"
```

❌ **Using absolute version pins in libraries**
```toml
# ❌ Too strict (breaks dependency resolution)
dependencies = ["requests==2.31.0"]

# ✅ Flexible range
dependencies = ["requests>=2.31.0,<3.0.0"]
```

## Quality Standards

- **Build System**: Use modern PEP 517/518 backend (Hatchling, Poetry)
- **Dependency Locking**: Commit lock files for applications
- **Version Control**: Follow semantic versioning
- **Metadata**: Complete project metadata in pyproject.toml
- **Testing**: Test package installation in clean environment

---

**Skill Type**: Python - Packaging
**Complexity**: Moderate
**Typical Usage**: Activated when creating distributable Python packages
**Performance**: Modern build tools (Hatch, Poetry) provide fast dependency resolution and builds
