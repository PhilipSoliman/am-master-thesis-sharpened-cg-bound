# Setup instructions for the project
This document provides setup instructions for this thesis repository. It includes information on how to set up the python environment, compile latex and C files, and run the code. In general, this repository is set up to be used with Visual Studio Code (VSCode). It is recommended to use VSCode for the project, as it provides a lot of useful features and extensions that can help with the development process.

## Requirements
Check that you have the following 
- `python` (3.9 or higher)
- `make` (consider using [MSYS2](https://www.msys2.org/) on Windows)
- `gcc` (again, consider using [MSYS2](https://www.msys2.org/) on Windows)
- Full [TeXLive](https://www.tug.org/texlive/windows.html) installation. On Linux this can be done using the following command:
```bash
sudo apt-get install texlive-full
```
If you are on Windows, the above should be made available in your PATH

## Easy setup 
The easiest way to set up the project is to use the provided [setup_env.py](setup_env.py) script. On Linux, you might need to give it rights by running the following command:
```bash
chmod +x setup_env.py
```
The script will automatically set up a virtual environment, install all the required (local) python packages and the Arial font for LaTeX compilation (latter only on Linux). Simply use your python installation to run the script:
```bash
python setup_env.py
```
This will create a virtual environment in the `.venv` folder and install all the required packages specified in the `requirements.txt` file. On Windows the script will also activate the environment for you, while on Linux you will need to explicitly do so by running the following command:
```bash
source .venv/bin/activate
```
Latex file compilation can be done using the provided [compile_latex.py](compile_latex.py) script. This script will check if all figures are generated (generating them if not) and allow you to choose which tex files you want to compile. It will also create a `build` folder in the respective folders, where all the output files will be stored. The script can be run using the following command:
    
```bash
python compile_latex.py
```
## Manual setup (VSCode)
### 1. Running Python files 
In order to run the Python files, one needs to have set up a (virtual) environment with the required packages. The requirements are listed in [requirements.txt](requirements.txt). To install the packages, run the following command in the terminal (after the environment is activated):
```bash
pip install -r requirements.txt
```
 
Next to this, the project root (or workspace folder in VSCode) should be added to the Python path. In VSCode this can be done by adding the following line to the settings.json file:
```json
{
    "terminal.integrated.env.linux": {
        "PYTHONPATH": "${env:PYTHONPATH}:${workspaceFolder}"
    }
}
```
or for Windows:
```json
{
    "terminal.integrated.env.windows": {
        "PYTHONPATH": "${env:PYTHONPATH};${workspaceFolder}"
    }
}
```

Some Python files can accept command line arguments. To make it easier to work with these files in VSCode it is recommended to use the [VSCode Action Buttons](https://marketplace.visualstudio.com/items?itemName=seunlanlege.action-buttons) extension together with the following settings:
```json
{
  "actionButtons": {
    "commands":
      [            
        {
        "name": "$(triangle-right) Run Python (show output)",
        "color": "green",
        "singleInstance": true,
        "command": "python ${file} --show-output",
        },
        {
        "name": "$(triangle-right) Run Python (generate output)",
        "color": "green",
        "singleInstance": true,
        "command": "python ${file} --generate-output",
        },
    ],

    "defaultColor": "white",
    "reloadButton": "↻",
    "loadNpmCommands": false
  },
}
```
Once setup, simply press the white reload button in the bottom left corner of the VSCode window to load the buttons. The buttons will then appear next to the reload button. The first button will run the Python file with the `--show-output` argument, while the second button will run the file with the `--generate-output` argument. This allows for easy testing and debugging of the Python files.

### 2. Generating figures
In this repository, figures are custom-generated using a function in [python utils](utils/utils.py) called ```save_latex_figure```. There is also a convience script [generate_figures.py](generate_figures.py) that can be used to generate all the figures in the project. It simply calls the function for each python file ending in "_fig.py" in [code](code). 

The generated figures are always saved as PDFs in a folder called `figures` in the root of the project. 

**IMPORTANT**: the figures need to be generated before compiling the latex documents, as they are included in the documents using the `\includegraphics` command.

### 3. Compiling Latex files
It is recommended to use the VSCode extension [LaTeX Workshop](https://marketplace.visualstudio.com/items?itemName=James-Yu.latex-workshop) to compile LaTeX files. This extension allows for easy compilation and previewing of the Latex documents. The following settings are necessary to compile the documents with the `lualatexmk` and `biber` tools:
```json
{
    "latex-workshop.intellisense.citation.backend": "biblatex",
    "latex-workshop.kpsewhich.enabled": false,
    "latex-workshop.latex.recipes": [
        {
            "name": "lualatexmk -> biber -> lualatexmk * 2",
            "tools": [
                "lualatexmk",
                "biber",
                "lualatexmk",
                "lualatexmk"
            ]
        },
    ],
    "latex-workshop.latex.tools": [
                {
            "name": "lualatexmk",
            "command": "latexmk",
            "args": [
                "-synctex=1",
                "-cd",
                "-interaction=nonstopmode",
                "-file-line-error",
                "-lualatex",
                "-outdir=%OUTDIR%",
                "%DOC%"
            ],
            "env": {}
        },
                {
            "name": "biber",
            "command": "biber",
            "args": [
                "--input-directory=%OUTDIR%",
                "--output-directory=%OUTDIR%",
                "%DOCFILE%"
            ],
            "env": {}
        },
    ],
    "latex-workshop.latex.outDir": "%DIR%/build",
},
```
These settings should compile allow for compilation of all the main tex files in [final_thesis](final_thesis), [interim_thesis](interim_thesis) and [manuscript](manuscript) folders (with the same name) by using the dedicated button from the Latex Workshop extension. The `build` folder will be created in the respective folders, where all the output files will be stored. Where and if subfiles are configured for the separate chapters, compilation will be possible for these files in the same way. See for instance the [theory](manuscript/chapters/theory/theory.tex) chapter of the manuscript.

Alternatively, one can use the recipe and subsidiary commands outlined above to compile the documents manually in a terminal (with a working Latex installation).

## C file compilation
For this, one can either use gcc or clang directly in combination with the provided [Makefile](clibs/Makefile). To compile the C files, run the following command in the terminal:
```bash
make
```
This will compile all the C files in the project. To clean up the compiled files, use:
```bash
make clean
```

When working in VSCode, it is recommended to use the [Makefile tools](https://marketplace.visualstudio.com/items?itemName=ms-vscode.makefile-tools) extension. This extension allows for easy compilation and management of Makefiles. The following settings are necessary to compile the C files with the `Makefile` tool:
```json
{
  "makefile.makefilePath": "clibs",
  "makefile.buildLog": "",
  "makefile.makePath": "C:\\msys64\\usr\\bin", 
  "makefile.configurations": [],
  "makefile.makeDirectory": "clibs",
}
```
Note; the makePath is just an example, and should be changed to the path where the make command is installed.