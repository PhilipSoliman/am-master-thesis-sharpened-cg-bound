# Master Thesis Project
This document provides [setup instructions](#setup) as well as [general guidelines](#guidelines) for this thesis repository. It includes information on how to set up the python environment, generate figures, run experiments and compile latex files. In general, this repository is set up to be used with Visual Studio Code (VSCode). It is recommended to use VSCode for the project, as it provides a lot of useful features and extensions that can help with the development process.

## Setup
This section outlines the setup process for the project, including the installation of [required software](#requirements) and configuration of the development environment.

### Requirements
Check that you have the following 
- `python` 3.10
- Full [TeXLive](https://www.tug.org/texlive/windows.html) installation. On Linux this can be done using the following command:
```bash
apt-get install texlive-full
```
possibly using `sudo`. If you are on Windows, the above should be made available in your PATH. You can check this by running the following command in a terminal:
```bash
tex --version
```
Lastly for the LaTeX compilation it is necessary that the path to this repository does not contain any spaces and/or hyphens. This is a [known issue](https://github.com/James-Yu/LaTeX-Workshop/issues/2910) with the `latexmk` tool, which is used for compiling the LaTeX files. **UPDATE**: This issue has been fixed as of 08/07/2025. However, it is still recommended to avoid 'bad' file paths, as they might cause the same issue anyway.

---

### Setup Script
Firstly, the [hcmsfem](https://github.com/PhilipSoliman/hcmsfem) repository is set up as a [git submodule](https://git-scm.com/book/en/v2/Git-Tools-Submodules) of this repository. If you have not done so already, run the following command after cloning:
```bash
git submodule update --init
```
This clones hcmsfem repository at the specific commit on the [philip-soliman-am-master-thesis](https://github.com/PhilipSoliman/hcmsfem/tree/philip-soliman-am-master-thesis) branch on which this main repository relies. For more information on how to use git submodules, refer to the [Git Submodules documentation](https://git-scm.com/book/en/v2/Git-Tools-Submodules).

Second, run the provided [setup_env.py](setup_env.py) script. On Linux, you might need to give it appropriate rights by running the following command (in a bash shell):
```bash
chmod +x setup_env.py
```
On Windows, the `setup_env.py` script will also activate the environment for you, while on Linux you will need to explicitly do so by running the following command (in a bash shell) located in the root of this repository:
```bash
source .venv/bin/activate
```

#### VSCode Extensions
After the setup script is run and in case you do not already have some installed, you should be prompted to install the following VSCode extensions:
- [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
- [Pylance](https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance)
- [LaTeX Workshop](https://marketplace.visualstudio.com/items?itemName=James-Yu.latex-workshop)
- [Action Buttons](https://marketplace.visualstudio.com/items?itemName=seunlanlege.action-buttons)

For the Action Buttons to appear, you will need to press the white "Reload" button in the bottom left corner of your VSCode Workspace, after installing the extension.

---

### Running Code
Any python script created in this repository can be run using the python interpreter from the virtual environment created by the setup script. To run a script, simply use the following command in the terminal (after the environment is activated):
```bash
python <script_name>.py
```
Or, select the virtual environment as the current workspace's python interpreter in VSCode and run the script using the "Run Python File in Terminal" command or play button. This will ensure that the script is run with the correct python interpreter and all dependencies are available.

Any script that imports the `hcmsfem` package can be run with the `--help` flag to see all available options. For example, to run the script that [generates all quadrilateral meshes for experiments](hcmsfem/generate_meshes.py) with logging level `info` and visible progress bar(s), use the following command:
```bash
python hcmsfem/generate_meshes.py --loglvl info --show-progress
```
Or, to run the [experiment that calculates approximate eigenspectra](code/model_spectra/approximate_spectra.py) of the GDSW, RGDSW and AMS preconditioners for a diffusion problem with homogeneous DBCs on a unit square for various high-contrast coefficient functions with logging level `debug` and visible progress bar(s), use the following command:
```bash
python code/model_spectra/approximate_spectra.py --loglvl debug --show-progress
```

---

### Showing and Generating Figures
Any file in the [code](code) folder that ends with `_fig.py` is a script that generates a figure. These scripts can be run using the python interpreter from the virtual environment created by the setup script. To run a figure generating script and show its output, simply use the following command in the terminal (after the environment is activated):
```bash
python path_to_script/*_fig.py --show-output
```
Or, to generate an output PDF file without showing it, use the following command:
```bash
python path_to_script/*_fig.py --generate-output
```
For example, to see the performance plot of the improved CG bound for the preconditioners and diffusion problem [described above](#running-code), run the following command:
```bash
python code/model_spectra/absolute_performance_fig.py --show-output
```

The above actions can also be done by clicking on the action buttons that are configured automatically by the [setup script](#vscode-extensions).

Additionally, there is a convenience script [generate_figures.py](generate_figures.py) that can be used to generate figures resulting from all the python files ending in "_fig.py" in the [code](code) directory.

The generated figures are always saved as PDFs in a folder called `figures` in the root of this repository.

**IMPORTANT**: The figures are not included in the repository and need to be generated before compiling the latex documents.

---

### Compiling LaTeX Files

#### VSCode
It is recommended to use the VSCode extension [LaTeX Workshop](https://marketplace.visualstudio.com/items?itemName=James-Yu.latex-workshop) to compile LaTeX files. This extension allows for easy compilation and previewing of the Latex documents. The automated VSCode setup configures this extension with the proper `lualatexmk` and `biber` tools.

All main LaTeX files in this repository can be compiled using the dedicated button from the LaTeX Workshop extension. A `build` folder will be created in the respective *.tex files' folders, in which all the output files will be stored.

#### Terminal
Latex file compilation can be done using the provided [compile_latex.py](compile_latex.py) script. This script will check if a `figure` directory is present in the root of the project (generating it if not) and, subsequently allow you to choose which LaTeX files you want to compile. It will also create a `build` folder in the respective folders, where all the output files will be stored. This way you can choose to use the Latex Workshop extension, the compile script or both; there is no difference in the output.

The compile script can be run using the following command in the terminal (after the environment is activated):
```bash
python compile_latex.py
```

**IMPORTANT**: The compile script will not check if ALL the figures necessary for the compilation are present. To ensure that all figures are generated, run the [generate_figures.py](generate_figures.py) script, as described in [this section](#generating-figures) before compiling the LaTeX files.

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