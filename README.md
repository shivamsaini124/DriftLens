# DriftLens

DriftLens is a Python project for training a baseline classifier, generating
data-drift scenarios, and evaluating drift detectors.

## Setup on another computer

Before starting, install:

- [Git](https://git-scm.com/downloads)
- [Python](https://www.python.org/downloads/) 3.10 or newer

The commands below assume that the repository has been pushed to GitHub. Replace
`YOUR_USERNAME` with the GitHub account or organization that owns the
repository.

### Windows (PowerShell)

Open PowerShell and run:

```powershell
git clone https://github.com/YOUR_USERNAME/DriftLens.git
cd DriftLens

py -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation scripts, allow them for the current user and
activate the environment again:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\.venv\Scripts\Activate.ps1
```

To verify the installation:

```powershell
python -m pytest
```

When finished, leave the virtual environment with:

```powershell
deactivate
```

#### Windows Command Prompt

The equivalent activation command in Command Prompt (`cmd.exe`) is:

```bat
.\.venv\Scripts\activate.bat
```

Run the remaining setup and test commands from the PowerShell instructions
after activation.

### Linux

Open a terminal and run:

```bash
git clone https://github.com/YOUR_USERNAME/DriftLens.git
cd DriftLens

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

To verify the installation:

```bash
python -m pytest
```

When finished, leave the virtual environment with:

```bash
deactivate
```

## Development commands

With the virtual environment activated, run the test suite from the repository
root:

```bash
python -m pytest
```

The source code is in [`src/`](src/), and tests are in [`tests/`](tests/).
