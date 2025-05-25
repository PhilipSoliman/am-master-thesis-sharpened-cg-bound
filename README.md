# Master Thesis Project
This document provides [setup instructions](#setup) as well as [general guidelines](#guidelines) for this thesis repository. It includes information on how to set up the python environment, compile latex and C files, and run the code. In general, this repository is set up to be used with Visual Studio Code (VSCode). It is recommended to use VSCode for the project, as it provides a lot of useful features and extensions that can help with the development process.

## Setup
This section outlines the setup process for the project, including the installation of [required software](#requirements) and configuration of the development environment. After ensuring you have the necessary software installed, you can proceed with either the [easy](#easy-setup) or [manual](#manual-setup-vscode) setup process.

### Requirements
Check that you have the following 
- `python` (3.10 or higher)
- `make` (consider using [MSYS2](https://www.msys2.org/) on Windows)
- `gcc` (again, consider using [MSYS2](https://www.msys2.org/) on Windows)
- Full [TeXLive](https://www.tug.org/texlive/windows.html) installation. On Linux this can be done using the following command:
```bash
apt-get install texlive-full
```
possibly using `sudo`. If you are on Windows, the above should be made available in your PATH. Lastly for the LaTeX compilation it is necessary that the path to this repository's root does not contain any spaces. This is a [known issue](https://github.com/James-Yu/LaTeX-Workshop/issues/2910) with the `latexmk` tool, which is used for compiling the LaTeX files.

---

### Easy Setup
The easiest way to set up the project is to use the provided [setup_env.py](setup_env.py) script. On Linux, you might need to give it rights by running the following command (in a bash shell):
```bash
chmod +x setup_env.py
```
The script automatically sets up a virtual environment, installs all the required python packages specified in the `requirements.txt` file as well as the local `lib` module. Simply use your python installation to run the script:
```bash
<python-executable> setup_env.py
```
Note, Linux `<python-executable> = python3`, while on Windows one can use either `python` or `py`.

On Windows the script will also activate the environment for you, while on Linux you will need to explicitly do so by running the following command (in a bash shell):
```bash
source .venv/bin/activate
```
Latex file compilation can be done using the provided [compile_latex.py](compile_latex.py) script. This script will check if a `figure` directory is present in the root of the project (generating it if not) and, subsequently allow you to choose which LaTeX files you want to compile. It will also create a `build` folder in the respective folders, where all the output files will be stored. The script can be run using the following command in the terminal (after the environment is activated):
```bash
python compile_latex.py
```

---
### Manual Setup (VSCode)
#### 1. Running Python Files 
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

#### 2. Generating Figures
In this repository, figures are custom-generated using a function in [python utils](utils/utils.py) called ```save_latex_figure```. There is also a convenience script [generate_figures.py](generate_figures.py) that can be used to generate all the figures in the project. It simply calls the function for each python file ending in "_fig.py" in [code](code). 

The generated figures are always saved as PDFs in a folder called `figures` in the root of the project. 

**IMPORTANT**: the figures need to be generated before compiling the latex documents, as they are included in the documents using the `\includegraphics` command.

#### 3. Compiling LaTeX Files
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
These settings should compile allow for compilation of all the main LaTeX files in [final_thesis](final_thesis), [interim_thesis](interim_thesis) and [manuscript](manuscript) folders (with the same name) by using the dedicated button from the Latex Workshop extension. The `build` folder will be created in the respective folders, where all the output files will be stored. Where and if subfiles are configured for the separate chapters, compilation will be possible for these files in the same way. See for instance the [theory](manuscript/chapters/theory/theory.tex) chapter of the manuscript.

Alternatively, one can use the recipe and subsidiary commands outlined above to compile the documents manually in a terminal (with a working Latex installation).

#### C File Compilation
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
  "makefile.makePath": "C:\\path\\to\\make.exe", // Change this to the path where make is installed
  "makefile.configurations": [],
  "makefile.makeDirectory": "clibs",
}
```
Note; the makePath is just an example, and should be changed to the path where the make command is installed.

---

## Guidelines
This section outlines the key steps, resources, and recommendations necessary for successfully completing this thesis project.

### Supervision Meetings
Regular meetings with your daily supervisors are essential for staying on track. Aim for weekly or bi-weekly meetings. [Schedule a meeting with Alexander Heinlein](https://outlook.office.com/bookwithme/user/8c40d346703344f2bbeb0132b51295f1@tudelft.nl?anonymous&ep=plink). For each meeting:
- **Prepare an agenda**: Outline the topics you want to discuss.
- **Prepare slides**: Summarize your progress and key points.
- **Follow up**: Share brief meeting minutes and outline next steps.

### Timeline

#### Initial Setup
- [ ] Register the project in MaRe.
- [ ] Upload the DCO to MaRe.
- [ ] Prepare and upload the project description in MaRe.

#### Interim Thesis
- [ ] Conduct a literature study.
- [ ] Set up the software environment and begin preliminary implementation and experiments.
- [ ] Formulate research questions.
- [ ] Write the interim thesis and submit it for feedback.
- [ ] Send the interim thesis to all supervisors as well as the committee members before the interim thesis presentation.

*The interim thesis and presentation should be completed after roughly 2-3 months of starting the project.*

#### Interim Presentation
- [ ] Find all committee members (at least one external to the Numerical Analysis group).
- [ ] Schedule the presentation and invite all committee members (attendance by daily supervisors is mandatory; others are optional).
- [ ] Prepare the presentation.

#### Research
- [ ] Perform research based on the formulated research questions.
    - *Note*: Research questions serve as a guideline. During the research process, it may become necessary to adapt or refine them based on new findings or challenges.
- [ ] Continuously document your progress, including experiments, results, and challenges.
- [ ] Regularly discuss your findings with your supervisors to ensure alignment with the project goals.

#### Final Thesis
- [ ] Write the thesis, ensuring it adheres to the structure and guidelines provided in this document.
- [ ] Finalize the thesis after incorporating feedback from supervisors.
- [ ] Send the thesis to committee members for feedback (at least 2 weeks before the defense).

#### Defense
- [ ] Schedule a green light meeting (minimum 6 weeks before the defense).
    - **Green light**: Daily supervisors agree the thesis is ready for defense, except for minor adjustments.
- [ ] Prepare for the defense presentation.
- [ ] Attend the defense and respond to questions from the committee.

*The total duration of the project is flexible and depends on the intensity with which the student works on the project.*

---

### General Tips
- **Continuously collect literature** throughout the project.
- **Write regularly** to avoid last-minute stress.
- **Review the grading rubric** to ensure alignment with evaluation criteria.
- **Access to Resources**: Arrange access to data and/or computing resources early. For instance, familiarize yourself with compute clusters if you plan to use them.
- **Make a Timeline**: Create (at least roughly) a timeline for your thesis project. Since research is not always plannable, update it as needed during the project. Keeping track of your progress is essential.

---

### General Remarks on Thesis Writing

#### Writing Style
- **Clarity and Precision**: Be concise, precise, and mathematically rigorous. Quality is more important than quantity.
- **Figures and Tables**: Ensure all labels and legends are readable (font size similar to the main text, not less than half as large). Captions should make figures and tables self-explanatory.
- **Equations**: Number only those equations referenced later in the text.
- **Abbreviations**: Introduce abbreviations once and use them consistently.
- **Computational Results**: Mention software packages (with versions) and hardware used. Acknowledge computing resources like Delft Blue if applicable.
- **Understandability**: Write at a level of detail understandable to your peers.

#### Referencing
- Reference every source you use. Justify statements or cite relevant work. If a section is based on a reference, mention it at the beginning. Incorporate references grammatically into sentences.

#### Relevant Results
- Include only results used in your discussion. Move other results to an appendix if necessary.

#### AI Disclosure
- If you use AI tools for writing, disclose this in your thesis.

---

### Useful Links and Software

#### Useful Links
- [Google Scholar](https://scholar.google.com)
- [MathSciNet](https://mathscinet.ams.org)
- [ResearchGate](https://www.researchgate.net)

#### Useful Software
- [TexStudio](https://www.texstudio.org)
- [Visual Studio Code (VS Code)](https://code.visualstudio.com)
- [Spyder](https://www.spyder-ide.org)
- [PyCharm](https://www.jetbrains.com/pycharm)

---

### Repository Structure

- **[code](code)**: Contains all project-related code. Use [Git Submodules](https://git-scm.com/book/en/v2/Git-Tools-Submodules) to manage external repositories.
- **[data](data)**: Contains datasets and related files.
- **[defense](defense)**: Contains materials for the defense.
- **[final_thesis](final_thesis)**: Contains the final thesis document.
- **[interim_presentation](interim_presentation)**: Contains materials for the interim presentation.
- **[interim_thesis](interim_thesis)**: Contains the interim thesis document.
- **[manuscript](manuscript)**: Contains the work-in-progress manuscript for the thesis.
- **[meetings](meetings)**: Contains meeting notes and related materials.
- **[project_description](project_description)**: Contains the project description.

*Note*: Use Git submodules to manage external resources or collaborative documents within the same project structure. Refer to the [Git Submodules documentation](https://git-scm.com/book/en/v2/Git-Tools-Submodules) for more information.