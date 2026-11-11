# Creditworthiness Predicitive Modelling: How to run?

## Prerequisites
1. **Python3** (3.10 or newer preferred)
2. **Git** (optional, only needed to clone the repo)

If Python is not installed: 

- **Windows:** Download the installer from https://www.python.org/downloads/ and tick **"Add Python to PATH"** during installation.
- **macOS:** Install with Homebrew (`brew install python`)
- **Linux (Debian/Ubuntu):** 
`sudo apt update && sudo apt install python3 python3-venv python3-pip`
- **Linux (Fedora):** `sudo dnf install python3 python3-pip`

## 1. Clone project

```bash
git clone https://github.com/dungphys/CodeAlpha_CreditworthinessModelling.git
cd CodeAlpha_CreditworthinessModelling
```
Or download the ZIP and extract it.

## 2. Create a virtual env

A virtual environment keeps the project's dependencies isolated from the rest of your system

### Windows (PowerShell)
 
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```
 
If PowerShell blocks the activation script, run this once and try again:
 
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

### Windows (Command Prompt)
 
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```
 
### macOS and Linux
 
```bash
python3 -m venv .venv
source .venv/bin/activate
```
 
When the environment is active, your prompt shows `(.venv)` at the start.

## 3. Install dependencies
 
With the environment active:
 
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 4. Register the environment as a Jupyter kernel
 
This makes the environment selectable inside Jupyter and VS Code.
 
```bash
python -m pip install ipykernel
python -m ipykernel install --user --name project-env --display-name "Python (project-env)"
```
 
If `jupyter` or `notebook` is not listed in `requirements.txt`, install it too:
 
```bash
python -m pip install notebook
```
 
(Use `jupyterlab` instead if you prefer JupyterLab.)

## 5. Launch the notebooks
 
### Option A: Jupyter Notebook
 
```bash
jupyter notebook
```
 
### Option B: JupyterLab
 
```bash
jupyter lab
```
 
A browser tab opens. Open the `.ipynb` file you want and select the **Python (project-env)** kernel via **Kernel > Change kernel** if it is not already chosen.
 
### Option C: VS Code
 
1. Install the **Python** and **Jupyter** extensions.
2. Open the project folder in VS Code.
3. Open a `.ipynb` file.
4. Click **Select Kernel** (top right) and choose the `.venv` environment.

### Order
1. The notebooks can be found in **`/notebooks`**
2. Run `analysis.ipynb` first. This produces the processed datasets from the raw dataset.
3. Others notebooks can be runned after that in any order. It is recommended to run `baseline.ipynb` first to have a baseline.   

## 6. Deactivate and clean up
 
When you are done, deactivate the virtual env using:
 
```bash
deactivate
```
 
To remove the environment entirely, delete the `.venv` folder.