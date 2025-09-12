# NOTE the structure and functionality of this script and its main class `defense` are
# adopted from https://github.com/jeertmans/jeertmans.github.io/blob/main/_slides/2023-12-07-confirmation/main.py
# next to that of course the manim_slides documentation https://manim-slides.readthedocs.io/en/latest/
import itertools

# import cv2 # install for video playback
import os
from enum import Enum
from json import load
from pathlib import Path
from random import random

from manim_slides import Slide

# from manim import *
# manim_config = config
# FRAME_WIDTH = manim_config.frame_width
# FRAME_HEIGHT = manim_config.frame_height
# TOP = manim_config.frame_y_radius * UP
# BOTTOM = manim_config.frame_y_radius * DOWN
# LEFT_SIDE = manim_config.frame_x_radius * LEFT
# RIGHT_SIDE = manim_config.frame_x_radius * RIGHT
# ShowCreation = Create
from manimlib import *

os.environ["NO_HCMSFEM_CLI_ARGS"] = ""
# add to pythonpath
import sys

from hcmsfem.eigenvalues import eigs
from hcmsfem.logger import LOGGER
from hcmsfem.meshes import DefaultQuadMeshParams
from hcmsfem.plot_utils import CustomColors
from hcmsfem.preconditioners import coarse_space
from hcmsfem.problems import CoefFunc
from hcmsfem.root import get_venv_root
from hcmsfem.solvers import (
    CustomCG,
    classic_cg_iteration_bound,
    multi_cluster_cg_iteration_bound,
    partition_eigenspectrum,
)

sys.path.append((get_venv_root() / "code" / "model_spectra").as_posix())
from approximate_spectra import PRECONDITIONERS, RTOL, get_spectrum_save_path

FIGURE_DIR = get_venv_root() / "figures"


MESHES = DefaultQuadMeshParams
COEF_FUNCS = [
    CoefFunc.THREE_LAYER_VERTEX_INCLUSIONS,
    CoefFunc.EDGE_SLABS_AROUND_VERTICES_INCLUSIONS,
]
LOG_RTOL = np.log(RTOL)
N_ITERATIONS = 300

# Manim render settings
FPS = 30


class QUALITY(Enum):
    FK = (3840, 2160)
    HD = (1920, 1080)
    P720 = (1280, 720)
    P480 = (854, 480)
    P360 = (640, 360)
    P240 = (426, 240)


manim_config.camera.fps = FPS
manim_config.camera.resolution = QUALITY.P240.value
manim_config.background_color = WHITE
manim_config.directories.raster_images = (get_venv_root() / "images").as_posix()
manim_config.camera.background_color = CustomColors.NAVY.value
SCENE_WIDTH_CM = FRAME_WIDTH * 2.54

# font settings
TITLE_FONT_SIZE = 48
CONTENT_FONT_SIZE = 0.6 * TITLE_FONT_SIZE
SOURCE_FONT_SIZE = 0.2 * TITLE_FONT_SIZE
FOOTNOTE_FONT_SIZE = 19

# pgf preamble
PGF_PREAMBLE = r"""
\def\mathdefault#1{#1}
\everymath=\expandafter{\the\everymath\displaystyle}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{siunitx}
\usepackage{pgf}
\usepackage{tikz}
\usepackage{lmodern}
\usepackage[no-math]{fontspec}
\setmainfont{Arial}
\setmathsf{Arial}
\setmathtt{Arial}
\setmathrm{Arial}
\makeatletter\@ifpackageloaded{underscore}{}{\usepackage[strings]{underscore}}\makeatother
"""


# TODO: margin settings
class MARGINS(Enum):
    LEFT = 0.05 * FRAME_WIDTH
    RIGHT = 0.05 * FRAME_WIDTH
    TOP = 0.05 * FRAME_HEIGHT
    BOTTOM = 0.1 * FRAME_HEIGHT


# custom tex environment
class MiniPageTex(Tex):
    tex_environment = "minipage"


class TexText(TexText):
    def __init__(self, *args, additional_preamble="", **kwargs):
        additional_preamble = (
            "\\usepackage{tgheros}"
            + "\n\\renewcommand{\\familydefault}{\\sfdefault}"
            + additional_preamble
        )
        super().__init__(*args, additional_preamble=additional_preamble, **kwargs)


# tex alignment
class ALIGN(Enum):
    LEFT = 1
    RIGHT = 2
    CENTER = 3


# affiliation table
AFFILIATION = TexText(
    r"""
\begin{tabular}{lll}
    Student number:   & 4945255                                                                                \\
    Project duration: & \multicolumn{2}{l}{December 2024 -- September 2025}                                    \\
    Thesis committee: & Prof. H. Schuttelaars,                            & TU Delft, committee \\
                        & Dr. A. Heinlein,                                  & TU Delft, daily supervisor       \\
                        & F. Cumaru,                                         & TU Delft, daily co-supervisor    \\
    Faculty:          & Faculty of Electrical Engineering,                                                     \\
                        & Mathematics and Computer Science                                                       \\
    Department:       & Delft Institute of Applied Mathematics (DIAM)                                          \\
\end{tabular}
    """,
    additional_preamble="\\usepackage{tabularx}\n\\usepackage{array}",
    t2c={
        "Dr. A. Heinlein": CustomColors.RED.value,
        "F. Cumaru": CustomColors.RED.value,
        "Prof. H. Schuttelaars": CustomColors.RED.value,
    },
    alignment=R"\raggedleft",
    font_size=0.25 * TITLE_FONT_SIZE,
)

# default color settings
DEFAULT_MOBJECT_COLOR = CustomColors.RED.value


def set_default_color(func):
    """Sets default color to black"""

    def wrapper(*args, color=DEFAULT_MOBJECT_COLOR, **kwargs):
        return func(*args, color=color, **kwargs)

    return wrapper


Tex = set_default_color(Tex)
Text = set_default_color(Text)
# MathTex = set_default_color(MathTex)
Line = set_default_color(Line)
Dot = set_default_color(Dot)
Brace = set_default_color(Brace)
Arrow = set_default_color(Arrow)
# Angle = set_default_color(Angle)


# handy class for itemized lists
class Item:
    def __init__(self, initial=1):
        self.value = initial

    def __repr__(self):
        s = repr(self.value)
        self.value += 1
        return s


# citations
BIBLIOGRAPHY_FILE = get_venv_root() / "defense/manimations/bibliography.json"
with open(BIBLIOGRAPHY_FILE.as_posix(), "r") as f:
    REFERENCES = load(f)
CITED_REFERENCES = {}


class Reference:
    _counter = 1  # class variable for numbering

    def __init__(self, author, title, year, doi="", url=""):
        self.author = author
        self.title = title
        self.year = year
        self.doi = doi
        self.url = url
        self.number = Reference._counter
        Reference._counter += 1

    def __repr__(self):
        s = f"[{self.number}] {self.author}, {self.title}, {self.year}."
        if self.doi:
            s += f" DOI: {self.doi}"
        if self.url:
            s += f" URL: {self.url}"
        return s

    def short_cite(self):
        """Return short numeric citation style, e.g. [1]"""
        return Text(f"[{self.number}]", font_size=12, color=WHITE)


def cite(ref_key):
    if ref_key not in CITED_REFERENCES:
        data = REFERENCES[ref_key]
        CITED_REFERENCES[ref_key] = Reference(**data)
    return CITED_REFERENCES[ref_key].short_cite()


# global functions


def generate_spectra(
    eigenvalue_range: tuple[float, float],
    cluster_width_limits: tuple[float, float],
    num_clusters: int,
    num_spectra: int,
    problem_size: int,
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    spectra = []
    matrices = []
    n_interior_eigenvalues = problem_size - 2  # reserving 2 for min and max eigenvalues
    for _ in range(num_spectra):
        n_interior_eigenvalues = (
            problem_size - 2
        )  # reserving 2 for min and max eigenvalues
        clusters = []
        for i in range(num_clusters):
            # determine number of eigenvalues in this cluster
            if i < num_clusters - 1:
                n_cluster_eigenvalues = int(n_interior_eigenvalues * rng.random(1))
            else:
                n_cluster_eigenvalues = n_interior_eigenvalues

            # construct cluster
            cluster_width = rng.uniform(*cluster_width_limits)
            min_eig = eigenvalue_range[0] + cluster_width
            max_eig = eigenvalue_range[1] - cluster_width
            cluster_center = rng.uniform(min_eig, max_eig)
            cluster = np.random.normal(
                cluster_center, cluster_width / 8, size=n_cluster_eigenvalues
            )

            # store cluster and update remaining number of eigenvalues
            clusters.append(cluster)
            n_interior_eigenvalues -= n_cluster_eigenvalues

        # construct spectrum
        spectrum = np.sort(
            np.concatenate(([eigenvalue_range[0]], *clusters, [eigenvalue_range[1]]))
        )

        # store spectra and associated diagonal matrices
        spectra.append(spectrum)
        matrices.append(np.diag(spectrum))
    return spectra, matrices


def generate_cg_residual_polys(
    spectra: list[np.ndarray],
    diagonal_matrices: list[np.ndarray],
    problem_size: int,
    rhs: np.ndarray,
    x0: np.ndarray,
) -> list[np.poly1d]:
    residual_polys = []
    for spectrum, matrix in zip(spectra, diagonal_matrices):
        cg = CustomCG(matrix, rhs, x0, tol=1e-8, maxiter=problem_size)
        x_exact = rhs / spectrum
        _, _ = cg.solve(x_exact=x_exact)
        cg.residual_polynomials()
        coeffs = cg.residual_polynomials_coefficients[-1]
        rp = np.poly1d(coeffs)
        residual_polys.append(rp)
    return residual_polys


class defense(Slide):
    RUN_TIME = 0.5

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Colors
        self.BS_COLOR = BLUE_D
        self.UE_COLOR = MAROON_D
        self.SIGNAL_COLOR = BLUE_B
        self.WALL_COLOR = LIGHT_BROWN
        self.INVALID_COLOR = RED
        self.VALID_COLOR = "#28C137"
        self.IMAGE_COLOR = "#636463"
        self.X_COLOR = DARK_BROWN

        # Coordinates
        self.UL = Dot().to_corner(UL).get_center()
        self.UR = Dot().to_corner(UR).get_center()
        self.DL = Dot().to_corner(DL).get_center()
        self.DR = Dot().to_corner(DR).get_center()

        # Margin coordinates
        self.BOTTOM_MARGIN = BOTTOM + 1.1 * UP
        self.TOP_MARGIN = TOP + 0.65 * DOWN
        self.LEFT_MARGIN = LEFT_SIDE + 0.65 * RIGHT
        self.RIGHT_MARGIN = RIGHT_SIDE + 0.65 * LEFT

        # Mutable variables
        self.counter = 0
        self.slide_number_size = 16
        self.slide_number = (
            TexText(f"{self.counter}", font_size=self.slide_number_size)
            .align_to(self.RIGHT_MARGIN, RIGHT)
            .align_to(self.BOTTOM_MARGIN, UP)
        )
        self.content_size = 25

        # index, title, subtitle
        self.slide_index = (
            TexText("Index", font_size=self.slide_number_size)
            .next_to(self.slide_number, DOWN, buff=0.12)
            .align_to(self.RIGHT_MARGIN, RIGHT)
        )
        self.title_size = 28
        self.slide_title = TexText(
            r"\textbf{Title}", font_size=self.title_size
        ).align_to(self.BOTTOM_MARGIN, UP)
        self.subtitle_size = 22
        self.slide_subtitle = TexText(
            "Subcontents", font_size=self.subtitle_size
        ).next_to(self.slide_title, DOWN, buff=0.15)
        self.slide_subtitle_visible = False

        # slide contents (everything except title, subtitle, slide number)
        self.slide_contents: list[Mobject] = []

        # TU delft logo
        self.tu_delft_logo = (
            ImageMobject("TUDelft_logo_white", height=0.1 * FRAME_HEIGHT)
            .align_to(self.LEFT_MARGIN, LEFT)
            .align_to(self.BOTTOM_MARGIN, UP)
        )
        self.tu_delft_logo.set_z_index(10)  # make sure logo is always on top

        # TU Delft affiliation
        self.tu_delft_affiliation = (
            defense.paragraph(
                r"Delft University of Technology",
                "Numerical Analysis",
                r"Philip M. Soliman",
                font_size=self.slide_number_size,
                buff=0.07,
            )
            .align_to(self.LEFT_MARGIN, LEFT)
            .align_to(self.BOTTOM_MARGIN, UP)
        )
        self.add(self.tu_delft_affiliation)

        # equations
        self.strong_equation = (
            r"\begin{equation*}"
            r"\begin{aligned}"
            r"-\nabla\cdot\left(\mathcal{C}\nabla u\right) & = f &\text{in } \Omega,\\"
            r"u& = u_D &\text{on } \partial\Omega"
            r"\end{aligned}"
            r"\end{equation*}"
        )

        self.weak_equation = (
            r"\begin{equation*}"
            r"\begin{aligned}"
            r"a(u, v) = \int_\Omega \mathcal{C}\nabla u\cdot\nabla v\,dx = \int_\Omega f v\,dx = (f,v)"
            r"\end{aligned}"
            r"\end{equation*}"
        )

        self.fem_formulation = (
            r"\begin{equation*}"
            r"A\mathbf{u} = \mathbf{b}, \quad A_{ij} = a(\phi_i, \phi_j), \ b_i = (f, \phi_i) \quad \forall i,j\in\mathcal{N},"
            r"\end{equation*}"
        )

        self.cg_algorithm = (
            r"\begin{algorithmic}"
            r"\STATE $\mathbf{r}_0 = \mathbf{b} - A\mathbf{u}_0$, $\mathbf{p}_0 = \mathbf{r}_0$, $\beta_0 = 0$"
            r"\WHILE{$\epsilon_j < \epsilon$}"
            r"  \STATE $\alpha_j = (\mathbf{r}_j, \mathbf{r}_j) / (A \mathbf{p}_j, \mathbf{p}_j)$"
            r"  \STATE $\mathbf{u}_{j+1} = \mathbf{u}_j + \alpha_j \mathbf{p}_j$"
            r"  \STATE $\mathbf{r}_{j+1} = \mathbf{r}_j - \alpha_j A \mathbf{p}_j$"
            r"  \STATE $\beta_j = (\mathbf{r}_{j+1}, \mathbf{r}_{j+1}) / (\mathbf{r}_j, \mathbf{r}_j)$"
            r"  \STATE $\mathbf{p}_{j+1} = \mathbf{r}_{j+1} + \beta_j \mathbf{p}_j$"
            r"\ENDWHILE"
            r"\end{algorithmic}"
        )

        # main question
        self.main_question = VGroup()
        main_rq_text = defense.paragraph(
            "\\textit{How can we determine the total number of necessary CG iterations?}",
            font_size=1.25 * self.title_size,
            alignment=ALIGN.CENTER,
            t2c={
                "necessary": CustomColors.RED.value,
                "iterations": CustomColors.GOLD.value,
            },
            width=0.5 * FRAME_WIDTH,
        )
        main_rq_background = BackgroundRectangle(
            main_rq_text, fill_color=CustomColors.BLUE.value, fill_opacity=1.0, buff=0.2
        ).round_corners(0.1)
        main_rq_background.set_z_index(-1)
        dot = Dot(
            main_rq_background.get_corner(DL),
            fill_color=CustomColors.GOLD.value,
            radius=0.1,
        )
        self.main_question.add(dot, main_rq_background, main_rq_text)
        self.sub_questions = VGroup()

    # utility functions
    def next_slide(self, additional_animations: list[Animation] = [], **kwargs):
        slide_number_update, new_slide_number = self.update_slide_number()
        self.play(slide_number_update, *additional_animations, run_time=self.RUN_TIME)
        self.slide_number = new_slide_number
        super().next_slide(**kwargs)

    def update_slide_number(self):
        self.counter += 1
        new_slide_number = TexText(
            f"{self.counter}", font_size=self.slide_number_size
        ).move_to(self.slide_number)
        slide_number_update = ReplacementTransform(self.slide_number, new_slide_number)
        return slide_number_update, new_slide_number

    def update_slide_titles(self, title, subtitle, index):
        title_animations = []

        # construct new title
        if title is not None:
            new_title = TexText(
                f"\\textbf{{{title}}}", font_size=self.title_size
            ).move_to(self.slide_title)
            title_animations.append(ReplacementTransform(self.slide_title, new_title))

        if index is not None:
            z_index = self.slide_index.z_index
            new_index = (
                TexText(index, font_size=self.slide_number_size)
                .next_to(self.slide_number, DOWN, buff=0.12)
                .align_to(self.RIGHT_MARGIN, RIGHT)
            ).add_background_rectangle(color=CustomColors.NAVY.value, buff=0.0)
            new_index.set_z_index(z_index+2)
            new_index.background_rectangle.set_z_index(z_index+1)
            title_animations.append(FadeOut(self.slide_index))
            title_animations.append(FadeIn(new_index))

        # check for new subtitle
        if subtitle is not None and self.slide_subtitle_visible:
            new_subtitle = TexText(subtitle, font_size=self.subtitle_size).move_to(
                self.slide_subtitle
            )
            title_animations.append(
                ReplacementTransform(self.slide_subtitle, new_subtitle)
            )
            self.slide_subtitle_visible = True
        elif subtitle is not None and not self.slide_subtitle_visible:
            new_subtitle = TexText(subtitle, font_size=self.subtitle_size).move_to(
                self.slide_subtitle
            )
            title_animations.append(FadeIn(new_subtitle))
            self.slide_subtitle_visible = True
        else:
            # if none, do not delete subtitle, just fade out
            title_animations.append(self.slide_subtitle.animate.set_opacity(0))
            self.slide_subtitle_visible = False

        return (
            title_animations,
            new_title if title else self.slide_title,
            new_subtitle if subtitle else None,
            new_index if index else self.slide_index,
        )

    def update_slide(
        self,
        title=None,
        subtitle=None,
        index=None,
        new_contents: list[Mobject] = [],
        transition_time: float = 0.75,
        notes: str = "",
        additional_animations: list[Animation] = [],
        rate_func=smooth,
        **kwargs,
    ):
        """
        Update the slide with new new_contents. If clean_up is True, remove all existing new_contents from the slide.
        """

        wipe_animation = []

        # make animations for wiping old content out
        wipe_animation += [
            m.animate.move_to(m.get_center() - np.array([FRAME_WIDTH, 0, 0]))
            for m in self.slide_contents
        ]

        # add new content to scene but out of view
        for m in new_contents:
            m.move_to(m.get_center() + np.array([FRAME_WIDTH, 0, 0]))

        # make animations for bringing new content in
        wipe_animation += [
            m.animate.move_to(m.get_center() - np.array([FRAME_WIDTH, 0, 0]))
            for m in new_contents
        ]

        # animate slide number change and optional content wipe
        title_update, new_title, new_subtitle, new_index = self.update_slide_titles(
            title, subtitle, index
        )
        slide_number_update, new_slide_number = self.update_slide_number()

        self.play(
            *title_update,
            slide_number_update,
            *wipe_animation,
            *additional_animations,
            run_time=transition_time,
            rate_func=rate_func,
        )

        # update new_contents
        self.slide_contents = []
        self.slide_title = new_title
        if subtitle:
            self.slide_subtitle = new_subtitle
        self.slide_number = new_slide_number
        self.index = new_index

        # go to next slide
        super().next_slide(notes=notes, **kwargs)

    def update_slide_contents(
        self, new_contents: list[Mobject], notes: str = "", **kwargs
    ):
        """
        Update the slide with new new_contents, without changing title or slide number.
        """

        wipe_animation = []

        # make animations for wiping old content out
        wipe_animation += [
            m.animate.move_to(m.get_center() - np.array([FRAME_WIDTH, 0, 0]))
            for m in self.slide_contents
        ]

        # add new content to scene but out of view
        for m in new_contents:
            m.move_to(m.get_center() + np.array([FRAME_WIDTH, 0, 0]))

        # make animations for bringing new content in
        wipe_animation += [
            m.animate.move_to(m.get_center() - np.array([FRAME_WIDTH, 0, 0]))
            for m in new_contents
        ]

        # update slide number
        slide_number_update, new_slide_number = self.update_slide_number()

        # animate optional content wipe
        self.play(*wipe_animation, slide_number_update)

        # update new_contents
        self.slide_constents = []
        self.slide_number = new_slide_number

        super().next_slide(notes=notes, **kwargs)

    @staticmethod
    def paragraph(
        *strs,
        alignment: ALIGN = ALIGN.LEFT,
        direction=DOWN,
        width=0.5 * FRAME_WIDTH,
        buff=0.1,
        **kwargs,
    ):
        # output list of TexText mobjects
        texts = []

        # determine alignment for tex
        if alignment == ALIGN.CENTER:
            tex_alignment = R"\centering"
            alignment = None
        if alignment == ALIGN.LEFT:
            tex_alignment = R"\raggedright"
            alignment = LEFT
        elif alignment == ALIGN.RIGHT:
            tex_alignment = R"\raggedleft"
            alignment = RIGHT

        # create TexText mobjects
        for s in strs:
            print(s)  # print for debugging
            texts.append(
                TexText(
                    f"\\begin{{minipage}}{{{width}in}}{{{s}}}\\end{{minipage}}",
                    alignment=tex_alignment,
                    **kwargs,
                )
            )

        # arrange mobjects in specified direction
        texts = VGroup(*texts).arrange(direction, buff=buff)

        # align all mobjects to the first one
        if len(strs) > 1 and alignment is not None:
            for text in texts[1:]:
                text.align_to(texts[0], direction=alignment)

        return texts

    # level constructs
    def title_slide(self):
        # Add initial slide (maybe high-contrast coefficient func?)
        cover = ImageMobject("presentation_cover", height=FRAME_HEIGHT).set_z_index(-2)
        rectangle = Rectangle(
            width=FRAME_WIDTH,
            height=FRAME_HEIGHT,
            fill_color=BLACK,
            fill_opacity=0.65,
            stroke_width=0,
        ).set_z_index(-1)
        self.play(FadeIn(cover), FadeIn(rectangle), run_time=0.5)

        # Title slide
        super().next_slide(notes="title slide")
        title = TexText(
            "Sharpened CG Iteration Bound for High-contrast Heterogeneous Scalar Elliptic PDEs",
            font_size=0.8 * TITLE_FONT_SIZE,
            t2c={"High-contrast": CustomColors.RED.value},
            alignment=R"\centering",
        )
        subtitle = TexText(
            "Going Beyond Condition Number",
            font_size=0.5 * TITLE_FONT_SIZE,
            alignment=R"\centering",
        ).next_to(title, DOWN)
        author = TexText(
            "Philip M. Soliman",
            font_size=0.4 * TITLE_FONT_SIZE,
            alignment=R"\centering",
        ).next_to(subtitle, DOWN)
        master_thesis = TexText(
            "Master Thesis Defense",
            font_size=0.3 * TITLE_FONT_SIZE,
            alignment=R"\centering",
        ).next_to(author, DOWN)
        affiliation = AFFILIATION.to_corner(DR)
        self.play(
            Write(title),
            Write(subtitle),
            Write(author),
            Write(master_thesis),
            Write(affiliation),
            FadeIn(self.tu_delft_logo),
        )
        self.slide_contents = [
            cover,
            VGroup(rectangle, title, subtitle, author, master_thesis, affiliation),
            self.tu_delft_logo,
        ]
        super().next_slide()

    def level_0_opening(self):
        image_height = FRAME_HEIGHT - self.BOTTOM_MARGIN - self.TOP_MARGIN - 2.5

        # slide: pcb simulation
        pcb_img = ImageMobject(
            "pcb_temp_sim",
            height=image_height,
        ).align_to(self.TOP_MARGIN, UP)
        pcb_text = (
            TexText("PCB Temperature Simulation", font_size=FOOTNOTE_FONT_SIZE)
            .next_to(pcb_img, 0.01 * DOWN)
            .align_to(pcb_img, LEFT)
            .set_z_index(5)
            .shift(0.3 * UP)
            .shift(0.3 * RIGHT)
        )
        pcb_cite = cite("pcb_simulation").next_to(pcb_text, RIGHT)
        self.update_slide(
            "Opening",
            subtitle="Motivation: PCB Simulation",
            index="Opening",
            new_contents=[pcb_img, pcb_text, pcb_cite],
            notes="What do Printed Circuit Board simulations...",
        )

        # slide: subsurface modelling
        subsurface_img = ImageMobject(
            "subsurface_modelling", height=image_height
        ).align_to(self.TOP_MARGIN, UP)
        subsurface_text = (
            TexText("Subsurface Modelling", font_size=FOOTNOTE_FONT_SIZE)
            .next_to(subsurface_img, 0.5 * DOWN)
            .align_to(subsurface_img, LEFT)
        )
        subsurface_cite = cite("subsurface_modelling").next_to(subsurface_text, RIGHT)
        self.update_slide(
            subtitle="Motivation: Subsurface Modelling",
            new_contents=[subsurface_img, subsurface_text, subsurface_cite],
            notes="subsurface modelling...",
            additional_animations=[
                pcb_img.animate.set_opacity(0.0),
                FadeOut(pcb_text),
                FadeOut(pcb_cite),
            ],
        )

        # slide: eit
        eit_img = ImageMobject("eit", height=image_height).align_to(self.TOP_MARGIN, UP)
        eit_text = (
            TexText("Electrical Impedance Tomography", font_size=FOOTNOTE_FONT_SIZE)
            .next_to(eit_img, 0.5 * DOWN)
            .align_to(eit_img, LEFT)
        )
        eit_cite = cite("eit").next_to(eit_text, RIGHT)
        self.update_slide(
            subtitle="Motivation: Electrical Impedance Tomography",
            new_contents=[eit_img, eit_text, eit_cite],
            notes="Electrical Impedance Tomography...",
            additional_animations=[
                subsurface_img.animate.set_opacity(0.0),
                FadeOut(subsurface_text),
                FadeOut(subsurface_cite),
            ],
        )

        # slide: composite materials
        composite_img = ImageMobject(
            "composite_material", height=image_height
        ).align_to(self.TOP_MARGIN, UP)
        composite_text = (
            TexText("Composite Materials", font_size=FOOTNOTE_FONT_SIZE)
            .next_to(composite_img, 0.5 * DOWN)
            .align_to(composite_img, LEFT)
        )
        composite_cite = cite("composite_materials").next_to(composite_text, RIGHT)
        self.update_slide(
            subtitle="Motivation: Composite Materials",
            new_contents=[composite_img, composite_text, composite_cite],
            notes="Composite materials...",
            additional_animations=[
                eit_img.animate.set_opacity(0.0),
                FadeOut(eit_text),
                FadeOut(eit_cite),
            ],
        )

        # slide: pem
        pem_img = ImageMobject("pem", height=image_height).align_to(self.TOP_MARGIN, UP)
        pem_text = (
            TexText("Proton Exchange Membranes", font_size=FOOTNOTE_FONT_SIZE)
            .next_to(pem_img, 0.5 * DOWN)
            .align_to(pem_img, LEFT)
        )
        pem_cite = cite("pem").next_to(pem_text, RIGHT)
        self.update_slide(
            subtitle="Motivation: Proton Exchange Membranes",
            new_contents=[pem_img, pem_text, pem_cite],
            notes="Proton Exchange Membranes...",
            additional_animations=[
                composite_img.animate.set_opacity(0.0),
                FadeOut(composite_text),
                FadeOut(composite_cite),
            ],
        )
        # return
        # slide: model problem
        images = always_redraw(
            Group,
            pcb_img.set_opacity(0.0),
            subsurface_img.set_opacity(0.0),
            eit_img.set_opacity(0.0),
            composite_img.set_opacity(1.0),
            pem_img.set_opacity(1.0),
        )
        buff = 0.05
        target_width = (FRAME_WIDTH - self.LEFT_MARGIN - self.RIGHT_MARGIN) / len(
            images
        ) - 0.3
        for i, img in enumerate(images):
            img.target.set_width(target_width)
            img.target.align_to(self.TOP_MARGIN, UP).shift(1.5 * DOWN)
            if i == 0:
                img.target.align_to(self.LEFT_MARGIN, LEFT)
            if i > 0:
                img.target.next_to(images[i - 1].target, RIGHT, buff=buff / 2)
            img.target.set_opacity(1.0)
        high_contrast_brace = always_redraw(Brace, images, direction=DOWN)
        high_contrast_label = TexText(
            "High-contrast Elliptic Problems", font_size=self.content_size
        )
        always(high_contrast_label.next_to, high_contrast_brace, DOWN, buff=0.1)
        self.update_slide(
            subtitle="Motivation: High-contrast Coefficient Functions",
            additional_animations=[
                *[MoveToTarget(img) for img in images],
                Write(high_contrast_brace),
                Write(high_contrast_label),
                FadeOut(pem_text),
                FadeOut(pem_cite),
            ],
        )

        # slide: model problem
        model_problem = defense.paragraph(
            r"Let $u_D\in H^{3/2}(\partial\Omega)$, $f\in L^2(\Omega)$, $\mathcal{C}\in L^\infty(\Omega)$. Find $u\in H^2(\Omega)$ such that"
            + self.strong_equation,
            width=0.2 * FRAME_WIDTH,
            font_size=CONTENT_FONT_SIZE,
            alignment=ALIGN.CENTER,
        )
        high_contrast_brace.clear_updaters()
        high_contrast_label.clear_updaters()
        self.update_slide(
            notes="We want to find the solution u. To do so we discretize...",
            subtitle="Model Problem",
            additional_animations=[
                FadeOut(images),
                FadeOut(high_contrast_brace),
                FadeOut(high_contrast_label),
                Write(model_problem),
            ],
            transition_time=self.RUN_TIME,
        )

        # slide: variational formulation & discretization
        self.play(
            model_problem.animate.align_to(self.TOP_MARGIN, UP).shift(0.5 * DOWN),
            run_time=0.5,
        )
        model_problem_weak = defense.paragraph(
            r"Let $u_D\in H^{1/2}(\partial\Omega)$, $f\in L^2(\Omega)$, $\mathcal{C}\in L^\infty(\Omega)$. Find $u\in \{u\in H^1(\Omega) | u_{\partial \Omega} = u_D\}$ such that $\forall v \in H^1_0(\Omega)$"
            + self.weak_equation,
            width=0.2 * FRAME_WIDTH,
            font_size=CONTENT_FONT_SIZE,
            alignment=ALIGN.CENTER,
        ).move_to(model_problem)
        self.play(
            ReplacementTransform(model_problem[0], model_problem_weak[0]),
            run_time=1.0,
        )
        domain_width = model_problem_weak.get_bottom()[1] - self.BOTTOM_MARGIN[1] - 0.6
        domain = Rectangle(
            width=domain_width, height=domain_width, color=WHITE
        ).next_to(model_problem_weak, DOWN, buff=0.3)
        domain_label = TexText(r"$\Omega$", font_size=CONTENT_FONT_SIZE).move_to(
            domain.get_center()
        )
        self.play(
            Write(domain),
            Write(domain_label),
            run_time=1.0,
        )
        num_boxes_per_side = 5
        grid_size = domain_width / num_boxes_per_side
        grid_lines = NumberPlane(
            x_range=[0, domain_width, grid_size],
            y_range=[0, domain_width, grid_size],
            width=domain_width,
            height=domain_width,
            background_line_style={
                "stroke_color": WHITE,
                "stroke_width": 1,
                "stroke_opacity": 1.0,
            },
        ).next_to(model_problem_weak, DOWN, buff=0.3)
        self.play(ShowCreation(grid_lines), run_time=1.0)

        # slide: FEM formulation
        fem_formulation = defense.paragraph(
            r"FEM basis $V_h = \text{span}\{\phi_i\}_{i=1}^{n}$" + self.fem_formulation,
            width=0.2 * FRAME_WIDTH,
            font_size=CONTENT_FONT_SIZE,
            alignment=ALIGN.CENTER,
        ).move_to(model_problem_weak)
        self.update_slide(
            notes="we get a linear system..",
            subtitle="Model Problem",
            additional_animations=[
                ReplacementTransform(model_problem_weak[0], fem_formulation[0]),
            ],
            transition_time=self.RUN_TIME,
        )

        # slide: linear system
        linear_system_text = TexText(
            r"\begin{equation*}A\mathbf{u}=\mathbf{b}\end{equation*}",
            font_size=2.0 * CONTENT_FONT_SIZE,
        ).move_to(ORIGIN)
        self.update_slide(
            notes="we get a linear system..",
            new_contents=[linear_system_text],
            subtitle="Discretized Model Problem",
            additional_animations=[
                FadeOut(domain),
                FadeOut(domain_label),
                FadeOut(grid_lines),
                ReplacementTransform(fem_formulation[0], linear_system_text),
            ],
            transition_time=self.RUN_TIME,
        )

        # slide: CG schematic
        cg_shematic = ImageMobject("cg_schematic", height=1.2 * image_height).align_to(
            self.TOP_MARGIN, UP
        )
        self.update_slide(
            new_contents=[cg_shematic],
            notes="We solve this system using Conjugate Gradient method...",
            subtitle="Conjugate Gradient (CG) Method",
            additional_animations={
                FadeOut(linear_system_text),
            },
        )

        # slide: main research question
        self.slide_contents = [cg_shematic]
        self.update_slide(
            new_contents=[self.main_question.move_to(ORIGIN)],
            subtitle="Main Research Question",
            notes="",
        )
        self.slide_contents = [self.main_question]

    def toc(self):
        item = Item()
        contents = defense.paragraph(
            f"{item}. Introducing CG"
            + f"\\\\{item}. How Does CG Converge? The Role of Eigenvalues"
            + f"\\\\{item}. Preconditioning: Taming High-Contrast Problems"
            + f"\\\\{item}. Towards a Sharper Iteration Bound: Clustered Spectra"
            + f"\\\\{item}. Sharpness \& Practical Use of New Bound: Ritz Values"
            + f"\\\\{item}. Conclusion: Key Takeaways \& Future Directions",
            font_size=1.5 * CONTENT_FONT_SIZE,
            alignment=ALIGN.LEFT,
            width=0.3 * FRAME_WIDTH,
        ).align_to(self.LEFT_MARGIN, LEFT)
        self.update_slide(
            "Contents", new_contents=[contents], notes="Table of Contents"
        )
        self.slide_contents = [contents]

    def level_1_intro_cg(self):
        linear_system_text = TexText(
            r"\begin{equation*}A\mathbf{u}=\mathbf{b}\end{equation*}",
            font_size=2.0 * CONTENT_FONT_SIZE,
        )
        self.update_slide(
            "Introducing CG",
            index="Introducing CG",
            new_contents=[linear_system_text],
            notes="Explain CG as an iterative method for solving Ax=b",
        )

        # slide: initial guess
        initial_guess = TexText(
            r"Initial Guess:\begin{equation*}\mathbf{u}_0\sim 0\end{equation*}",
            font_size=2.0 * CONTENT_FONT_SIZE,
        ).next_to(linear_system_text, LEFT, buff=2.0)
        arrow = Arrow(
            start=initial_guess.get_right(),
            end=linear_system_text.get_left(),
            buff=0.1,
            color=WHITE,
        )
        self.next_slide(notes="We provide an initial guess...")
        self.update_slide(
            subtitle="Initial Guess",
            additional_animations=[
                Write(initial_guess),
                Write(arrow),
            ],
            run_time=self.RUN_TIME,
        )

        # slide: residual
        initial_system = VGroup(linear_system_text, initial_guess, arrow)
        initial_residual = TexText(
            r"\begin{equation*}\mathbf{r}_0 = \mathbf{b} - A\mathbf{u}_0\end{equation*}",
            font_size=2.0 * CONTENT_FONT_SIZE,
        ).move_to(initial_system.get_center())
        self.update_slide(
            subtitle="Initial Residual",
            notes="which gives an initial residual.",
            additional_animations=[
                ReplacementTransform(initial_system, initial_residual),
            ],
            run_time=self.RUN_TIME,
        )

        # slide: desired error tolerance + CG black box
        initial_residual.generate_target()
        initial_residual.target.align_to(self.LEFT_MARGIN, LEFT)
        initial_residual.target.shift(UP)
        self.play(
            MoveToTarget(initial_residual),
            run_time=self.RUN_TIME,
        )
        error_tol = TexText(
            r"Error Tolerance $\epsilon$",
            font_size=2.0 * CONTENT_FONT_SIZE,
        ).next_to(initial_residual, DOWN, buff=0.5)
        inputs = VGroup(initial_residual, error_tol)
        inputs_brace = Brace(inputs, RIGHT)
        cg_rectangle = Rectangle(
            width=3.0,
            height=1.0,
            color=WHITE,
        ).next_to(inputs_brace, RIGHT, buff=1.0)
        cg_text = TexText(
            "CG",
            font_size=2.0 * CONTENT_FONT_SIZE,
        ).move_to(cg_rectangle.get_center())
        arrow2 = Arrow(
            start=inputs_brace.get_right(),
            end=cg_rectangle.get_left(),
            buff=0.1,
            color=WHITE,
        )
        self.update_slide(
            subtitle="Conjugate Gradient (CG) Method",
            notes="We specify a desired error tolerance and feed everything into CG...",
            additional_animations=[
                Write(error_tol),
                Write(inputs_brace),
                Write(arrow2),
                Write(cg_rectangle),
                Write(cg_text),
            ],
            transition_time=self.RUN_TIME,
        )

        # slide: cg algorithm
        cg_algorithm = TexText(
            self.cg_algorithm,
            additional_preamble=r"\usepackage{algorithmic}",
            font_size=2 * CONTENT_FONT_SIZE,
        ).shift(0.5 * UP)
        cg_alg_width = cg_algorithm.get_corner(UR)[0] - cg_algorithm.get_corner(DL)[0]
        cg_alg_height = cg_algorithm.get_corner(UR)[1] - cg_algorithm.get_corner(DL)[1]
        cg_rectangle_og = cg_rectangle.copy()
        cg_text_og = cg_text.copy()
        cg_rectangle.generate_target()
        cg_rectangle.target.move_to(ORIGIN).shift(0.5 * UP)
        cg_rectangle.target.stretch_to_fit_width(cg_alg_width + 0.5)
        cg_rectangle.target.stretch_to_fit_height(cg_alg_height + 0.5)
        cg_rectangle.target.set_fill(CustomColors.BLUE.value)
        cg_rectangle.target.set_opacity(1.0)
        inputs_brace_arrow = VGroup(inputs, inputs_brace, arrow2)
        self.update_slide(
            notes="which runs the following algorithm...",
            additional_animations=[
                MoveToTarget(cg_rectangle),
                ReplacementTransform(cg_text, cg_algorithm),
                inputs_brace_arrow.animate.set_opacity(0.4),
            ],
            subtitle="CG Algorithm",
        )

        # slide: highlight inputs in cg algorithm
        cg_algorithm_colored = TexText(
            self.cg_algorithm,
            additional_preamble=r"\usepackage{algorithmic}",
            font_size=2 * CONTENT_FONT_SIZE,
            t2c={
                r"\mathbf{r}_0": CustomColors.RED.value,
                r"\epsilon": CustomColors.RED.value,
                r"\mathbf{u}_{j+1} = \mathbf{u}_j + \alpha_j \mathbf{p}_j": CustomColors.GOLD.value,
                r"\epsilon_j": CustomColors.GOLD.value,
            },
        ).move_to(cg_rectangle.get_center())
        self.update_slide(
            notes="We recognize the initial residual and error tolerance as inputs. Also note the error at the jth iteration.",
            additional_animations=[
                FadeTransform(cg_algorithm, cg_algorithm_colored),
            ],
            transition_time=2 * self.RUN_TIME,
            subtitle="CG Algorithm: Inputs and Outputs Highlighted",
        )

        # slide: CG output
        self.play(
            ReplacementTransform(cg_rectangle, cg_rectangle_og),
            ReplacementTransform(cg_algorithm, cg_text_og),
            ReplacementTransform(cg_algorithm_colored, cg_text_og),
            run_time=2 * self.RUN_TIME,
        )
        u_j_text = (
            TexText(
                r"$\mathbf{u}_j$",
                font_size=2.0 * CONTENT_FONT_SIZE,
                color=CustomColors.GOLD.value,
            )
            .next_to(cg_rectangle_og, RIGHT, buff=3.0)
            .shift(0.7 * UP)
        )
        epsilon_j_text = TexText(
            r"$\epsilon_j = \frac{||\mathbf{e}_j||_A}{||\mathbf{e}_0||_A}$",
            font_size=2.0 * CONTENT_FONT_SIZE,
            color=CustomColors.GOLD.value,
        ).next_to(u_j_text, DOWN, buff=0.5)
        outputs = VGroup(u_j_text, epsilon_j_text)
        outputs_brace = Brace(outputs, LEFT)
        arrow3 = Arrow(
            start=cg_rectangle_og.get_right(),
            end=outputs_brace.get_left(),
            buff=0.1,
            color=WHITE,
        )
        self.update_slide(
            notes="CG outputs the jth approximation and the corresponding error.",
            additional_animations=[
                Write(outputs),
                Write(outputs_brace),
                Write(arrow3),
                inputs_brace_arrow.animate.set_opacity(1.0),
            ],
            transition_time=self.RUN_TIME,
            subtitle="CG Inputs and Outputs",
        )

        # slide: CG convergence
        succesive_approximations = VGroup()
        succesive_errors = VGroup()
        num_approximations = 5
        for i in range(num_approximations):
            approx = TexText(
                f"$\\mathbf{{u}}_{{{i}}}$,",
                font_size=1.5 * CONTENT_FONT_SIZE,
            )
            error = TexText(
                f"$\\epsilon_{{{i}}}$,",
                font_size=1.5 * CONTENT_FONT_SIZE,
            )
            succesive_approximations.add(approx)
            succesive_errors.add(error)
        succesive_approximations.add(TexText("...", font_size=1.5 * CONTENT_FONT_SIZE))
        succesive_approximations.add(
            TexText(f"$\\mathbf{{u}}_{{m}}$", font_size=1.5 * CONTENT_FONT_SIZE)
        )
        succesive_approximations.arrange(RIGHT, buff=0.8)
        succesive_errors.add(TexText("...", font_size=1.5 * CONTENT_FONT_SIZE))
        succesive_errors.add(
            TexText(f"$\\epsilon_{{m}}$", font_size=1.5 * CONTENT_FONT_SIZE)
        )
        succesive_errors.arrange(RIGHT, buff=0.8)
        succesive_approximations.shift(0.5 * UP)
        succesive_errors.next_to(succesive_approximations, DOWN, buff=0.5)
        self.update_slide(
            subtitle="CG Error Convergence",
            additional_animations=[
                ReplacementTransform(u_j_text, succesive_approximations),
                ReplacementTransform(epsilon_j_text, succesive_errors),
                FadeOut(inputs_brace_arrow),
                FadeOut(cg_text_og),
                FadeOut(cg_rectangle_og),
                FadeOut(outputs_brace),
                FadeOut(arrow3),
            ],
            transition_time=self.RUN_TIME,
        )

        # slide: CG error
        max_scale = 3.0
        min_scale = 0.5
        for i, err in enumerate(succesive_errors):
            center = err.get_center()
            center_on_origin = center - [0, center[1], 0]
            err.generate_target()
            err.target.scale(min_scale + (max_scale - min_scale) / (2**i))
            err.target.move_to(center_on_origin)
        self.update_slide(
            subtitle="CG Error Convergence",
            additional_animations=[
                FadeOut(succesive_approximations),
                *[MoveToTarget(err) for err in succesive_errors],
            ],
            transition_time=self.RUN_TIME,
            notes="with errors decreasing as j increases.",
        )

        # slide: CG classical error bound
        citation = cite("iter_method_saad").next_to(self.slide_title, RIGHT, buff=0.2)
        classical_error_bound = TexText(
            r"$\epsilon_m \leq 2\left(\frac{\sqrt{\kappa(A)}-1}{\sqrt{\kappa(A)}+1}\right)^m$",
            font_size=2.0 * CONTENT_FONT_SIZE,
            t2c={r"\kappa": CustomColors.RED.value, r"m": CustomColors.GOLD.value},
        ).move_to(succesive_errors.get_center())
        classical_error_bound_box = SurroundingRectangle(
            classical_error_bound, buff=0.5, opacity=0, color=WHITE
        )
        classical_error_bound_text = TexText(
            "Classical CG Error Bound", font_size=2.0 * CONTENT_FONT_SIZE
        ).next_to(classical_error_bound_box, UP, buff=0.5)
        self.update_slide(
            subtitle="CG Error Convergence",
            notes="The classical CG error bound relates the error to the condition number kappa(A) and the number of iterations m.",
            additional_animations=[
                Write(citation),
                ReplacementTransform(succesive_errors, classical_error_bound),
                Write(classical_error_bound_box),
                Write(classical_error_bound_text),
            ],
            transition_time=self.RUN_TIME,
        )

        # slide: CG classical iteration bound
        classical_iteration_bound = TexText(
            r"$m \leq \left\lfloor\frac{\sqrt{\kappa(A)}}{2}\ln\left(\frac{2}{\epsilon}\right) + 1\right\rfloor = m_1(\kappa)$",
            font_size=2.0 * CONTENT_FONT_SIZE,
            t2c={
                r"\kappa(A)": CustomColors.RED.value,
                r"\kappa": CustomColors.RED.value,
                r"m": CustomColors.GOLD.value,
            },
        ).move_to(classical_error_bound.get_center())
        classical_iteration_bound_box = SurroundingRectangle(
            classical_iteration_bound, buff=0.5, opacity=0, color=WHITE
        )
        classical_iteration_bound_text = TexText(
            "Classical CG Iteration Bound",
            font_size=2.0 * CONTENT_FONT_SIZE,
        ).move_to(classical_error_bound_text.get_center())
        self.update_slide(
            subtitle="CG Iteration Bound",
            additional_animations=[
                FadeOut(citation),
                ReplacementTransform(classical_error_bound, classical_iteration_bound),
                ReplacementTransform(
                    classical_error_bound_text, classical_iteration_bound_text
                ),
                ReplacementTransform(
                    classical_error_bound_box, classical_iteration_bound_box
                ),
            ],
            transition_time=self.RUN_TIME,
            notes="From this we can derive a classical iteration bound...",
        )

        # slide: current problem
        m_text = TexText(
            r"$m \leq$",
            font_size=2.0 * CONTENT_FONT_SIZE,
        )
        m_1_text = TexText(
            r"$m_1(\kappa)$",
            font_size=2.0 * CONTENT_FONT_SIZE,
            t2c={r"\kappa": CustomColors.RED.value},
        )
        cg_iteration_bound_simplified = always_redraw(VGroup, m_text, m_1_text).arrange(
            RIGHT, buff=0.5
        )
        cg_iteration_bound_simplified_box = always_redraw(
            SurroundingRectangle,
            cg_iteration_bound_simplified,
            buff=0.5,
            opacity=0,
            color=WHITE,
        )
        self.play(
            ReplacementTransform(
                classical_iteration_bound, cg_iteration_bound_simplified
            ),
            ReplacementTransform(
                classical_iteration_bound_box, cg_iteration_bound_simplified_box
            ),
            run_time=self.RUN_TIME,
        )
        m_1_text.generate_target()
        m_1_text.target.scale(2.0)
        m_text_new = TexText(
            r"$m \ll$",
            font_size=2.0 * CONTENT_FONT_SIZE,
        )
        high_contrast_text = TexText(
            "...due to high-contrast in $\mathcal{C}$.",
            font_size=2.0 * CONTENT_FONT_SIZE,
        ).next_to(cg_iteration_bound_simplified, DOWN, buff=0.5)
        always(
            high_contrast_text.next_to,
            cg_iteration_bound_simplified_box,
            DOWN,
            buff=0.5,
        )
        always(high_contrast_text.set_color, CustomColors.RED.value)
        m_text_new.move_to(m_text.get_center() - [0.5, 0, 0])
        self.update_slide(
            subtitle="CG Iteration Bound (Revised)",
            notes="The problem is that for high-contrast problems, kappa is very large. Leading to very pessimistic bounds on m.",
            additional_animations=[
                MoveToTarget(m_1_text),
                ReplacementTransform(m_text, m_text_new),
                Write(high_contrast_text),
            ],
            transition_time=2 * self.RUN_TIME,
        )

        # slide: update research question
        self.slide_contents = [
            cg_iteration_bound_simplified,
            high_contrast_text,
            m_1_text,
            m_text_new,
            classical_iteration_bound_text,
        ]
        self.main_question.move_to(ORIGIN).shift(2 * UP)
        addendum = VGroup()
        addendum_text = (
            defense.paragraph(
                "+ \\textit{That is sharp for high-contrast problems.}",
                font_size=self.title_size,
                alignment=ALIGN.CENTER,
                width=0.5 * FRAME_WIDTH,
            )
            .next_to(self.main_question, DOWN, buff=0.5)
            .align_to(self.main_question, LEFT)
            .shift(0.5*RIGHT)
        )
        addendum_bgr = BackgroundRectangle(
            addendum_text, color=CustomColors.GOLD.value, fill_opacity=1.0, buff=0.2
        ).round_corners(0.1)
        addendum_bgr.set_z_index(-1)
        # arrow = always_redraw(
        #     CurvedArrow, self.main_question[0].get_center(), addendum_bgr.get_left()
        # )
        addendum.add(addendum_bgr, addendum_text)
        self.update_slide(
            notes="This is the main research question of this thesis.",
            title="Research Question (Revised)",
            subtitle="Sharpness for High-Contrast Problems",
            new_contents=[self.main_question],
            additional_animations=[
                FadeIn(addendum, lag_ratio=0.5),
            ],
            transition_time=2 * self.RUN_TIME,
        )
        # self.main_question.add(addendum)

        self.slide_contents = [self.main_question]
        self.sub_questions.add(addendum)
        self.sub_questions.generate_target()
        self.sub_questions.target.set_opacity(0)
        self.play(MoveToTarget(self.sub_questions), run_time=self.RUN_TIME)


    def level_2_cg_convergence(self):
        linear_system_text = TexText(
            r"\begin{equation*}A\mathbf{u}=\mathbf{b}\end{equation*}",
            font_size=2.0 * CONTENT_FONT_SIZE,
        )
        self.update_slide(
            "How Does CG Converge?",
            new_contents=[linear_system_text],
            subtitle="The Role of Eigenvalues",
            notes="Explain CGs dependence on eigenvalues",
            index="How Does CG Converge?",
        )

        plot_resolution = 0.01
        freq_1, ampl_1, phas_1 = 3 * 2 * PI, 0.5, 0.1 * PI
        freq_2, ampl_2, phas_2 = 9 * 2 * PI, 2.5, 0.2 * PI
        freq_3, ampl_3, phas_3 = 15 * 2 * PI, 1.0, 0.3 * PI
        sin_1 = lambda x, t: ampl_1 * np.sin(freq_1 * (t - x + phas_1))
        sin_2 = lambda x, t: ampl_2 * np.sin(freq_2 * (t - x + phas_2))
        sin_3 = lambda x, t: ampl_3 * np.sin(freq_3 * (t - x + phas_3))
        speaker_img = (
            ImageMobject("speaker")
            .stretch_to_fit_width(0.2 * FRAME_WIDTH)
            .align_to(self.LEFT_MARGIN, LEFT)
        )

        def composite_sine_wave(t):
            composite_sine_wave_f = lambda x: sin_1(x, t) + sin_2(x, t) + sin_3(x, t)
            return (
                FunctionGraph(
                    composite_sine_wave_f,
                    x_range=[0, 1, plot_resolution],
                    color=WHITE,
                )
                .stretch_to_fit_width(0.7 * FRAME_WIDTH)
                .stretch_to_fit_height(0.5 * FRAME_HEIGHT)
                .next_to(speaker_img, RIGHT, buff=0)
            )

        plot = composite_sine_wave(0)
        tracker = ValueTracker(0)
        plot.add_updater(lambda m: m.become(composite_sine_wave(tracker.get_value())))

        # slide: see A as a Sine Wave
        self.update_slide(
            title="The Role of Eigenvalues",
            subtitle="Sound Example",
            notes="We can view the matrix A as a signal...",
            additional_animations=[
                ReplacementTransform(linear_system_text, plot),
                FadeIn(speaker_img),
            ],
            transition_time=self.RUN_TIME,
        )

        # slide: moving sine wave
        super().next_slide(notes="That is travelling...", loop=True)
        T = 2 * PI / freq_1
        slow_factor = 5
        self.play(
            tracker.animate.set_value(T), run_time=slow_factor * T, rate_func=linear
        )

        # slide: decomposition
        super().next_slide(
            notes="We can decompose this signal into its frequency components...",
            loop=True,
        )
        max_ampl = max(ampl_1, ampl_2, ampl_3)
        num_cycles = 3
        T = num_cycles * 2 * PI / freq_1

        def wave_1(t):
            wave_1_f = lambda x: sin_1(x, t)
            return (
                FunctionGraph(
                    wave_1_f,
                    x_range=[0, 1, plot_resolution],
                )
                .stretch_to_fit_width(0.7 * FRAME_WIDTH)
                .next_to(speaker_img, RIGHT, buff=0)
                .shift(0.2 * FRAME_HEIGHT * UP)
                .stretch_to_fit_height(0.15 * FRAME_HEIGHT * ampl_1 / max_ampl)
                .set_color(CustomColors.RED.value)
            )

        def wave_2(t):
            wave_2_f = lambda x: sin_2(x, t)
            return (
                FunctionGraph(
                    wave_2_f,
                    x_range=[0, 1, plot_resolution],
                )
                .stretch_to_fit_width(0.7 * FRAME_WIDTH)
                .next_to(speaker_img, RIGHT, buff=0)
                .stretch_to_fit_height(0.15 * FRAME_HEIGHT * ampl_2 / max_ampl)
                .set_color(CustomColors.GOLD.value)
            )

        def wave_3(t):
            wave_3_f = lambda x: sin_3(x, t)
            return (
                FunctionGraph(
                    wave_3_f,
                    x_range=[0, 1, plot_resolution],
                    color=CustomColors.BLUE.value,
                )
                .stretch_to_fit_width(0.7 * FRAME_WIDTH)
                .next_to(speaker_img, RIGHT, buff=0)
                .shift(0.2 * FRAME_HEIGHT * DOWN)
                .stretch_to_fit_height(0.15 * FRAME_HEIGHT * ampl_3 / max_ampl)
                .set_color(CustomColors.BLUE.value)
            )

        new_tracker = ValueTracker(0)
        wave_1_plot = wave_1(0).move_to(plot.get_center()).set_stroke(opacity=0)
        wave_2_plot = wave_2(0).move_to(plot.get_center()).set_stroke(opacity=0)
        wave_3_plot = wave_3(0).move_to(plot.get_center()).set_stroke(opacity=0)
        wave_1_plot.generate_target()
        wave_1_plot.target.move_to(wave_1(0).get_center())
        wave_1_plot.target.set_stroke(opacity=1)
        wave_2_plot.generate_target()
        wave_2_plot.target.move_to(wave_2(0).get_center())
        wave_2_plot.target.set_stroke(opacity=1)
        wave_3_plot.generate_target()
        wave_3_plot.target.move_to(wave_3(0).get_center())
        wave_3_plot.target.set_stroke(opacity=1)
        plot.clear_updaters()
        self.play(
            FadeOut(plot),
            MoveToTarget(wave_1_plot),
            MoveToTarget(wave_2_plot),
            MoveToTarget(wave_3_plot),
            run_time=self.RUN_TIME,
        )
        wave_1_plot.add_updater(lambda m: m.become(wave_1(new_tracker.get_value())))
        wave_2_plot.add_updater(lambda m: m.become(wave_2(new_tracker.get_value())))
        wave_3_plot.add_updater(lambda m: m.become(wave_3(new_tracker.get_value())))
        self.play(
            new_tracker.animate.set_value(T), run_time=slow_factor * T, rate_func=linear
        )
        super().next_slide(notes="")

        # slide: return to linear system
        linear_system_text = TexText(
            r"\begin{equation*}A\mathbf{u}=\mathbf{b}\end{equation*}",
            font_size=2.0 * CONTENT_FONT_SIZE,
        )
        self.slide_contents = [speaker_img, wave_1_plot, wave_2_plot, wave_3_plot]
        self.update_slide(
            subtitle="Eigenvalues of the A",
            new_contents=[linear_system_text],
            notes="Returning to our original system...",
        )

        # slide: eigenvalues of A
        spectrum_A = (
            TexText(
                r"$\sigma(A) = \{\lambda_1, \lambda_2, \ldots, \lambda_n\}$",
                font_size=2.0 * CONTENT_FONT_SIZE,
                t2c={
                    r"\lambda_1": CustomColors.RED.value,
                    r"\lambda_2": CustomColors.GOLD.value,
                    r"\lambda_n": CustomColors.BLUE.value,
                },
            )
            .move_to(linear_system_text.get_center())
            .shift(2 * UP)
        )
        condition_number_text = TexText(
            r"$\kappa(A) = \frac{\lambda_{\text{max}}}{\lambda_{\text{min}}}$",
            font_size=2.0 * CONTENT_FONT_SIZE,
        )
        always(condition_number_text.next_to, spectrum_A, DOWN, buff=0.5)
        self.update_slide(
            subtitle="Eigenvalues of A",
            notes="We can do a similar decomposition of A resulting in its eigenvalues or spectrum...",
            additional_animations=[
                ReplacementTransform(linear_system_text, spectrum_A),
                Write(condition_number_text, lag_ratio=0.5),
            ],
            transition_time=2 * self.RUN_TIME,
        )

        # slide: CG solution polynmomial
        spectrum_A.generate_target()
        spectrum_A.target.align_to(self.LEFT_MARGIN, LEFT)
        cg_rectangle_and_text = VGroup(
            Rectangle(
                width=3.0,
                height=1.0,
                color=WHITE,
            ),
            TexText(
                "CG",
                font_size=2.0 * CONTENT_FONT_SIZE,
            ),
        )
        always(cg_rectangle_and_text.next_to, condition_number_text, DOWN, buff=0.5)
        self.play(
            MoveToTarget(spectrum_A),
            Write(cg_rectangle_and_text),
            run_time=2 * self.RUN_TIME,
        )
        cg_solution = TexText(
            r"$\mathbf{u}_m = \mathbf{u}_0 + q_m(A)\mathbf{r}_0$",
            font_size=2.0 * CONTENT_FONT_SIZE,
            t2c={r"q_m(A)": CustomColors.GOLD.value},
        ).next_to(cg_rectangle_and_text, RIGHT, buff=1.0)
        cg_solution_text = TexText(
            r"Solution Polynomial",
            font_size=2.0 * CONTENT_FONT_SIZE,
            color=CustomColors.GOLD.value,
        ).next_to(cg_solution, UP, buff=0.5)
        arrow = Arrow(
            start=cg_rectangle_and_text.get_right(),
            end=cg_solution.get_left(),
            buff=0.1,
            color=WHITE,
        )
        self.update_slide(
            title="Min-Max Polynomial Approximation",
            subtitle="CG Solution and Residual Polynomials",
            notes="Why is this important? To see this, we look at the CG solution polynomial...",
            additional_animations=[
                Write(arrow),
                Write(cg_solution),
                Write(cg_solution_text),
            ],
            transition_time=2 * self.RUN_TIME,
        )

        # slide: CG residual polynomial
        cg_residual = TexText(
            r"$r_m(A) = I - Aq_m(A)$",
            font_size=2.0 * CONTENT_FONT_SIZE,
            t2c={r"r_m": CustomColors.RED.value, r"q_m": CustomColors.GOLD.value},
        ).next_to(cg_rectangle_and_text, RIGHT, buff=1.0)
        cg_residual_text = TexText(
            r"Residual Polynomial",
            font_size=2.0 * CONTENT_FONT_SIZE,
            color=CustomColors.RED.value,
        ).next_to(cg_residual, UP, buff=0.5)
        self.update_slide(
            subtitle="CG Solution and Residual Polynomials",
            notes="Which is related to the residual polynomial...",
            additional_animations=[
                ReplacementTransform(cg_solution, cg_residual),
                ReplacementTransform(cg_solution_text, cg_residual_text),
            ],
            transition_time=self.RUN_TIME,
        )

        # slide: General CG error bound
        cg_residual.generate_target()
        cg_residual.target.next_to(condition_number_text, DOWN, buff=0.5).align_to(
            self.LEFT_MARGIN, LEFT
        )
        cg_residual_and_spectrum = VGroup(
            spectrum_A, condition_number_text, cg_residual
        )
        brace = Brace(cg_residual_and_spectrum, RIGHT, buff=0.1)
        always(brace.next_to, cg_residual_and_spectrum, RIGHT, buff=0.1)
        cg_error_bound = TexText(
            r"$\epsilon_m \leq \underset{r\in\mathcal{P}_m,\ r(0)=1}{\min} \ \underset{\lambda \in \sigma(A)}{\max}|r(\lambda)|$",
            font_size=1.5 * CONTENT_FONT_SIZE,
            t2c={
                r"r": CustomColors.RED.value,
                r"\sigma(A)": CustomColors.SKY.value,
            },
        )
        always(cg_error_bound.next_to, brace, RIGHT, buff=0.1)
        cg_error_bound_text = TexText(
            "General CG Error Bound",
            font_size=2.0 * CONTENT_FONT_SIZE,
        )
        always(cg_error_bound_text.next_to, cg_error_bound, UP, buff=0.5)
        self.next_slide(
            subtitle="CG Error Bound",
            notes="The CG error can be bounded in terms of the residual polynomial...",
            additional_animations=[
                FadeOut(cg_residual_text),
                FadeOut(cg_rectangle_and_text),
                FadeOut(arrow),
                MoveToTarget(cg_residual),
                Write(brace),
                Write(cg_error_bound),
                Write(cg_error_bound_text),
            ],
            transition_time=2 * self.RUN_TIME,
        )

        # slide: CG recovering classical bound
        cg_error_bound.clear_updaters()
        cg_error_bound_text.clear_updaters()
        cg_error_bound_uniform = (
            TexText(
                r"$\epsilon_m \leq \underset{r\in\mathcal{P}_m,\ r(0)=1}{\min} \ \underset{\lambda \in [\lambda_{\min}, \lambda_{\max}]}{\max}|r(\lambda)|$",
                font_size=2.0 * CONTENT_FONT_SIZE,
                t2c={
                    r"r": CustomColors.RED.value,
                    r"\lambda_{\min}, \lambda_{\max}": CustomColors.SKY.value,
                },
            )
            .move_to(ORIGIN)
            .shift(1.5 * LEFT)
        )
        m_1_text = TexText(
            r"$m_1\left(\frac{\lambda_{\max}}{\lambda_{\min}}\right)$",
            font_size=2.0 * CONTENT_FONT_SIZE,
            t2c={
                r"\lambda_{\max}": CustomColors.SKY.value,
                r"\lambda_{\min}": CustomColors.SKY.value,
            },
        )
        always(m_1_text.next_to, cg_error_bound_uniform, RIGHT, buff=1.0)
        arrow = always_redraw(
            Arrow,
            start=cg_error_bound_uniform.get_right(),
            end=m_1_text.get_left(),
            buff=0.2,
            color=WHITE,
        )
        full_classical_bound = VGroup(cg_error_bound_uniform, m_1_text)
        cg_error_bound_text_uniform = TexText(
            "Classical CG Error Bound",
            font_size=2.0 * CONTENT_FONT_SIZE,
        ).next_to(full_classical_bound, UP, buff=0.5)
        self.update_slide(
            notes="which can recover the classical bound by assuming a uniform distribution of eigenvalues...",
            subtitle="Recovering the Classical Bound",
            additional_animations=[
                FadeOut(brace),
                FadeOut(cg_residual_and_spectrum),
                ReplacementTransform(cg_error_bound, cg_error_bound_uniform),
                ReplacementTransform(cg_error_bound_text, cg_error_bound_text_uniform),
                Write(arrow),
                Write(m_1_text),
            ],
            transition_time=2 * self.RUN_TIME,
        )

        # slide: cg spetra and polynomials
        domain = (0.0, 1.01, 0.2)
        domain_range = domain[1] - domain[0]
        codomain = (-1.0, 1.01, 0.5)
        codomain_range = codomain[1] - codomain[0]
        height = 0.5 * FRAME_HEIGHT
        width = 0.7 * FRAME_WIDTH
        num_points = 100
        spectra_transition_time = 1.5

        # initiate spectra
        num_spectra = 10
        num_clusters = 3
        eigv_sample_range = (
            0.1,
            1.0,
        )
        problem_size = 300
        solve_precision = 1e-8
        rng = np.random.default_rng(95)
        rhs = rng.random(problem_size)
        x0 = np.zeros(problem_size)
        axes = Axes(
            x_range=domain,
            y_range=codomain,
            height=height,
            width=width,
            axis_config={
                "stroke_color": WHITE,
                "stroke_width": 2,
                "include_tip": True,
            },
        ).align_to(self.TOP_MARGIN, UP)
        self.x_label = Tex(r"\lambda").move_to(
            axes.c2p(domain[1] + 0.1 * domain_range, codomain[0] + 0.5 * codomain_range)
        )
        self.y_label = Tex(r"r(\lambda)").move_to(
            axes.c2p(domain[0] - 0.15 * domain_range, 0.8 * codomain[1])
        )
        l_min_label = (
            TexText(
                r"$\lambda_{\min}$", t2c={r"\lambda_{\min}": CustomColors.SKY.value}
            )
            .move_to(
                axes.c2p(eigv_sample_range[0], codomain[0] + 0.35 * codomain_range)
            )
            .set_z_index(10)
        )
        l_max_label = (
            TexText(
                r"$\lambda_{\max}$", t2c={r"\lambda_{\max}": CustomColors.SKY.value}
            )
            .move_to(
                axes.c2p(eigv_sample_range[1], codomain[0] + 0.35 * codomain_range)
            )
            .set_z_index(10)
        )
        axes.add_coordinate_labels(
            font_size=20,
            num_decimal_places=1,
        )
        classical_bound_val = classic_cg_iteration_bound(
            eigv_sample_range[1] / eigv_sample_range[0], np.log(solve_precision)
        )
        self.play(
            FadeOut(cg_error_bound_text_uniform),
            FadeOut(arrow),
            full_classical_bound.animate.to_edge(self.BOTTOM_MARGIN).shift(0.5 * UP),
            ShowCreation(axes),
            ShowCreation(self.x_label),
            ShowCreation(self.y_label),
            ShowCreation(l_min_label),
            ShowCreation(l_max_label),
            run_time=2 * self.RUN_TIME,
        )
        cg_error_bound_uniform_short = (
            (
                TexText(
                    r"$\underset{r\in\mathcal{P}_m,\ r(0)=1}{\min} \ \underset{\lambda \in [\lambda_{\min}, \lambda_{\max}]}{\max}|r(\lambda)|$",
                    font_size=2.0 * CONTENT_FONT_SIZE,
                    t2c={
                        r"r": CustomColors.RED.value,
                        r"\lambda_{\min}, \lambda_{\max}": CustomColors.SKY.value,
                    },
                )
            )
            .move_to(cg_error_bound_uniform.get_center())
            .set_z_index(10)
        )
        self.play(
            ReplacementTransform(cg_error_bound_uniform, cg_error_bound_uniform_short),
            run_time=0.5 * self.RUN_TIME,
        )
        m_1_text.clear_updaters()
        always(m_1_text.next_to, cg_error_bound_uniform_short, RIGHT, buff=1.0)
        m_1_text.set_z_index(10)
        arrow2 = always_redraw(
            Arrow,
            start=cg_error_bound_uniform_short.get_right(),
            end=m_1_text.get_left(),
            buff=0.2,
            color=WHITE,
        ).set_z_index(10)
        self.play(
            cg_error_bound_uniform_short.animate.align_to(cg_error_bound_uniform, LEFT),
            Write(arrow2),
            run_time=0.5 * self.RUN_TIME,
        )

        # generate spectra and polynomials for two clusters
        spectra, matrices = generate_spectra(
            eigv_sample_range,
            (1e-2, 0.2),
            num_clusters=num_clusters,
            num_spectra=num_spectra,
            problem_size=problem_size,
            rng=rng,
        )
        cg_residual_polys = generate_cg_residual_polys(
            spectra, matrices, problem_size, rhs, x0
        )

        # plot the first spectrum and its residual polynomial
        self.plot_spectrum_and_poly(
            axes,
            spectra[0],
            cg_residual_polys[0],
            domain,
            domain_range,
            codomain,
            num_points,
            spectra_transition_time,
            classical_bound_val,
        )

        # slide: loop through spectra and polys
        super().next_slide(loop=True, notes="We can now look at different spectra...")
        for spectrum, cg_residual_poly in zip(spectra[1:], cg_residual_polys[1:]):
            self.plot_spectrum_and_poly(
                axes,
                spectrum,
                cg_residual_poly,
                domain,
                domain_range,
                codomain,
                num_points,
                spectra_transition_time,
                classical_bound_val,
            )
        self.plot_spectrum_and_poly(
            axes,
            spectra[0],
            cg_residual_polys[0],
            domain,
            domain_range,
            codomain,
            num_points,
            spectra_transition_time,
            classical_bound_val,
        )
        super().next_slide(loop=False, notes="...Lets look at a uniform spectrum now.")

        # slide: uniform spectrum
        uniform_spectrum = np.linspace(
            eigv_sample_range[0], eigv_sample_range[1], problem_size
        )
        uniform_cg_residual_poly = generate_cg_residual_polys(
            [uniform_spectrum],
            [np.diag(uniform_spectrum)],
            problem_size,
            rhs,
            x0,
        )[0]
        self.plot_spectrum_and_poly(
            axes,
            uniform_spectrum,
            uniform_cg_residual_poly,
            domain,
            domain_range,
            codomain,
            num_points,
            spectra_transition_time,
            classical_bound_val,
            # return_anims=True
        )
        self.update_slide(
            notes="For a uniform spectrum, the classical bound is sharp.",
            subtitle="Uniform Spectrum",
            # additional_animations=anims,
            transition_time=spectra_transition_time,
        )

        # slide: split spectrum
        half_1 = problem_size // 2
        half_2 = problem_size - half_1
        split_spectrum = np.concatenate(
            [
                np.ones(half_1) * eigv_sample_range[0],
                np.ones(half_2) * eigv_sample_range[1],
            ]
        )
        split_cg_residual_poly = generate_cg_residual_polys(
            [split_spectrum],
            [np.diag(split_spectrum)],
            problem_size,
            rhs,
            x0,
        )[0]
        self.plot_spectrum_and_poly(
            axes,
            split_spectrum,
            split_cg_residual_poly,
            domain,
            domain_range,
            codomain,
            num_points,
            spectra_transition_time,
            classical_bound_val,
            return_anims=False,
        )
        self.update_slide(
            notes="For a split spectrum, the classical bound is pessimistic.",
            subtitle="Split Spectrum",
            # additional_animations=anims2,
            transition_time=spectra_transition_time,
        )
        # del anims2, anims

        # slide: restate research question
        arrow2.clear_updaters()
        self.slide_contents = [
            axes,
            self.x_label,
            self.y_label,
            l_min_label,
            l_max_label,
            self.cg_residual_poly_graph,
            self.spectrum_group,
            self.y_label,
            self.degree_tex,
            cg_error_bound_uniform_short,
            arrow2,
        ]
        self.main_question.move_to(ORIGIN).shift(2 * UP)
        addendum = VGroup()
        addendum_text = (
            defense.paragraph(
                "+ \\textit{Accounts for the full spectrum of A.}",
                font_size=self.title_size,
                alignment=ALIGN.CENTER,
                width=0.5 * FRAME_WIDTH,
            )
            .next_to(self.main_question, DOWN, buff=1.5)
            .align_to(self.main_question, LEFT)
            .shift(0.5*RIGHT)
        )
        addendum_bgr = BackgroundRectangle(
            addendum_text, color=CustomColors.GOLD.value, fill_opacity=1.0, buff=0.2
        ).round_corners(0.1)
        addendum_bgr.set_z_index(-1)
        # arrow = always_redraw(
        #     CurvedArrow, self.main_question[0].get_center(), addendum_bgr.get_left()
        # )
        addendum.add(addendum_bgr, addendum_text)
        self.sub_questions.generate_target()
        self.sub_questions.target.set_opacity(1)
        self.update_slide(
            notes="This is the main research question of this thesis.",
            title="Research Question (Revised)",
            subtitle="Sharpness for High-Contrast Problems",
            new_contents=[self.main_question],
            additional_animations=[
                FadeIn(addendum, lag_ratio=0.5),
                MoveToTarget(self.sub_questions)
            ],
            transition_time=2 * self.RUN_TIME,
        )
        # self.main_question.add(addendum)
        self.slide_contents = [self.main_question]
        self.sub_questions.add(addendum)
        self.sub_questions.generate_target()
        self.sub_questions.target.set_opacity(0)
        self.play(MoveToTarget(self.sub_questions), run_time=self.RUN_TIME)

    def plot_spectrum_and_poly(
        self,
        axes,
        spectrum: np.ndarray,
        cg_residual_poly: np.poly1d,
        domain: tuple,
        domain_range: float,
        codomain: tuple,
        num_points: int,
        spectra_transition_time: float,
        classical_bound_val: int,
        return_anims=False,
    ):
        # create dots from spectrum
        spectrum_dots = [
            Dot(
                axes.c2p(eig, 0),
                fill_color=CustomColors.SKY.value,
                radius=DEFAULT_DOT_RADIUS,
            )
            for eig in spectrum
        ]
        spectrum_group = VGroup(*spectrum_dots)

        # create graph from cg_residual_poly
        cg_residual_poly_graph = axes.get_graph(
            lambda x: cg_residual_poly(x),
            x_range=(domain[0], domain[1], domain[1] / num_points),
            color=CustomColors.GOLD.value,
        )

        # get degree of polynomial (= number of CG iterations)
        degree = cg_residual_poly.coefficients.shape[0] - 1
        y_label = axes.get_y_axis_label(f"r_{{{degree}}}(\\lambda)").move_to(
            self.y_label.get_center()
        )

        # add degree in seperate texbox
        degree_tex = Tex(
            f"m = {degree} \leq m_1 = {classical_bound_val}",
            font_size=2.0 * CONTENT_FONT_SIZE,
        )
        degree_tex.move_to(axes.c2p(domain[0] + 0.5 * domain_range, 0.9 * codomain[1]))

        if not hasattr(self, "cg_residual_poly_graph"):
            if return_anims:
                return [
                    FadeIn(cg_residual_poly_graph, scale=0.5),
                    FadeIn(spectrum_group, scale=0.5),
                    ReplacementTransform(self.y_label, y_label),
                    FadeIn(degree_tex, scale=0.5),
                ]
            self.play(
                FadeIn(cg_residual_poly_graph, scale=0.5),
                FadeIn(spectrum_group, scale=0.5),
                ReplacementTransform(self.y_label, y_label),
                FadeIn(degree_tex, scale=0.5),
                run_time=spectra_transition_time,
            )
        else:
            if return_anims:
                return [
                    ReplacementTransform(
                        self.cg_residual_poly_graph, cg_residual_poly_graph
                    ),
                    ReplacementTransform(self.spectrum_group, spectrum_group),
                    ReplacementTransform(self.y_label, y_label),
                    ReplacementTransform(self.degree_tex, degree_tex),
                ]
            self.play(
                ReplacementTransform(
                    self.cg_residual_poly_graph, cg_residual_poly_graph
                ),
                ReplacementTransform(self.spectrum_group, spectrum_group),
                ReplacementTransform(self.y_label, y_label),
                ReplacementTransform(self.degree_tex, degree_tex),
                run_time=spectra_transition_time,
            )

        # update mobjects
        self.cg_residual_poly_graph = cg_residual_poly_graph
        self.spectrum_group = spectrum_group
        self.y_label = y_label
        self.degree_tex = degree_tex

    def level_3_preconditioning(self):
        # slide: recap high-contrast leading to pessimistic bounds
        high_contrast_text = TexText(
            "High-contrast $\mathcal{C}$",
            font_size=2.0 * CONTENT_FONT_SIZE,
        )
        arrow_hc_to_kappa = Arrow(
            start=ORIGIN,
            end=0.5 * RIGHT,
            buff=0.0,
            color=WHITE,
        )
        condition_number = TexText(
            "$\kappa(A)$",
            font_size=2.0 * CONTENT_FONT_SIZE,
            t2c={r"\kappa": CustomColors.RED.value},
        )
        arrow_to_m_1 = Arrow(
            start=ORIGIN,
            end=0.5 * RIGHT,
            buff=0.0,
            color=WHITE,
        ).set_opacity(0)
        m_1 = TexText(
            "$m_1(\kappa)$",
            font_size=2.0 * CONTENT_FONT_SIZE,
            t2c={r"\kappa": CustomColors.RED.value},
        ).set_opacity(0)
        m = TexText(
            "$\gg m$",
            font_size=2.0 * CONTENT_FONT_SIZE,
        ).set_opacity(0)
        high_contrast_issue = always_redraw(
            VGroup,
            high_contrast_text,
            arrow_hc_to_kappa,
            condition_number,
            arrow_to_m_1,
            m_1,
            m,
        )
        always(high_contrast_issue.arrange, RIGHT, buff=0.5)
        self.update_slide(
            "Preconditioning",
            new_contents=[high_contrast_issue],
            subtitle="Taming High-Contrast Problems",
            notes="Recap: High-contrast leads to",
            index="Preconditioning",
        )
        always(high_contrast_issue.move_to, ORIGIN)

        # slide: continued
        self.update_slide(
            subtitle="Taming High-Contrast Problems",
            notes="high values for kappa and thus pessimistic bound m_1 on m.",
            additional_animations=[
                condition_number.animate.scale(2.0),
                arrow_to_m_1.animate.set_opacity(1),
                m_1.animate.set_opacity(1).scale(2.0),
                m.animate.set_opacity(1),
            ],
            run_time=2 * self.RUN_TIME,
        )

        # slide: preconditioned system
        preconditioner = TexText(
            r"Preconditioner: $M \approx A$",
            font_size=2.0 * CONTENT_FONT_SIZE,
            t2c={
                r"Preoconditioner": CustomColors.GOLD.value,
                r"M": CustomColors.GOLD.value,
            },
        ).next_to(high_contrast_issue, UP, buff=1.0)
        system = TexText(
            r"$A\mathbf{u} = \mathbf{b}$",
            font_size=2.0 * CONTENT_FONT_SIZE,
        ).next_to(high_contrast_issue, DOWN, buff=1.0)
        preconditioned_system = TexText(
            r"$M^{-1}A\mathbf{u} = M^{-1}\mathbf{b}$",
            t2c={r"M": CustomColors.GOLD.value},
            font_size=2.0 * CONTENT_FONT_SIZE,
        ).move_to(system.get_center())
        self.play(Write(system), Write(preconditioner), run_time=2 * self.RUN_TIME)
        self.update_slide(
            notes="The problem of large condition numbers can be mitigated by using preconditioning. We transform the system...",
            subtitle="Effect on Condition Number",
            additional_animations=[ReplacementTransform(system, preconditioned_system)],
            transition_time=self.RUN_TIME,
        )

        # slide: condition number of preconditioned system
        conditioner_number_center = condition_number.get_center()
        preconditioned_condition_number = TexText(
            r"$\kappa(M^{-1}A)$",
            font_size=2.0 * CONTENT_FONT_SIZE,
            t2c={r"\kappa": CustomColors.RED.value, r"M": CustomColors.GOLD.value},
        ).scale(2.0)
        self.play(
            FadeOut(preconditioner),
            # FadeOutToPoint(preconditioned_system, conditioner_number_center),
            FadeOut(preconditioned_system),
            ReplacementTransform(condition_number, preconditioned_condition_number),
            run_time=2 * self.RUN_TIME,
        )
        high_contrast_issue2 = VGroup(
            high_contrast_text,
            arrow_hc_to_kappa,
            preconditioned_condition_number,
            arrow_to_m_1,
            m_1,
            m,
        )
        high_contrast_issue2.arrange(RIGHT, buff=0.5)
        high_contrast_issue2.move_to(ORIGIN)
        high_contrast_issue.align_to(high_contrast_issue2.get_center())
        high_contrast_issue2.add_updater(
            lambda m: m.arrange(RIGHT, buff=0.5).move_to(ORIGIN)
        )
        m_sharp = TexText(
            r"$ \geq m$",
            font_size=2.0 * CONTENT_FONT_SIZE,
            t2c={r"\kappa": CustomColors.RED.value},
        ).move_to(m.get_center())
        self.play(
            ReplacementTransform(high_contrast_issue, high_contrast_issue2),
            run_time=self.RUN_TIME,
        )
        self.update_slide(
            subtitle="Effect on Condition Number",
            additional_animations=[
                preconditioned_condition_number.animate.scale(0.5),
                m_1.animate.scale(0.5),
                ReplacementTransform(m, m_sharp, lag_ratio=0.5),
            ],
            transition_time=2 * self.RUN_TIME,
        )

        # slide: high coefficient functions from Filipe and Alexander
        high_contrast_issue2.clear_updaters(recurse=True)
        coefficient_function_image = (
            ImageMobject("coefficient_functions")
            .set_height(0.6 * FRAME_HEIGHT)
            .set_z_index(1)
            .align_to(self.TOP_MARGIN, UP)
        )
        coefficient_rectangle = (
            BackgroundRectangle(
                coefficient_function_image,
                color=CustomColors.BLUE.value,
                buff=0.1,
            )
            .set_z_index(-1)
            .round_corners(0.1)
            .set_opacity(1.0)
        )
        citations = [
            cite("ams_coarse_space_comp_study_Alves2024"),
            cite("ams_framework_Wang2014"),
            cite("gdsw_coarse_space_Dohrmann2008"),
            cite("rgdsw_coarse_space_Dohrmann2017"),
        ]
        for i, c in enumerate(citations):
            c.next_to(self.slide_subtitle, RIGHT, buff=0.3).shift(i * 0.3 * RIGHT)
        self.update_slide(
            title="Preconditioners for High-Contrast Problems",
            subtitle="Sample Coefficient Functions",
            new_contents=[
                coefficient_function_image,
                citations[0],
                coefficient_rectangle,
            ],
            additional_animations=[
                FadeOut(high_contrast_text),
                FadeOut(arrow_hc_to_kappa),
                FadeOut(preconditioned_condition_number),
                FadeOut(arrow_to_m_1),
                FadeOut(m_1),
                FadeOut(m_sharp),
                FadeOut(high_contrast_issue2),
            ],
            notes="Lets look at two specific examples of coefficient functions.",
        )

        # slide: introduce three preconditioners
        M_1 = TexText(
            r"$M_{\text{2-OAS-GDSW}}$",
            font_size=1.5 * CONTENT_FONT_SIZE,
        )
        M_2 = TexText(
            r"$M_{\text{2-OAS-RGDSW}}$",
            font_size=1.5 * CONTENT_FONT_SIZE,
        )
        M_3 = TexText(
            r"$M_{\text{2-OAS-AMS}}$",
            font_size=1.5 * CONTENT_FONT_SIZE,
        )
        for i, M in enumerate([M_1, M_2, M_3]):
            M.next_to(coefficient_rectangle, DOWN, buff=0).shift((i - 1) * 3.0 * RIGHT)
        self.update_slide(
            title="Preconditioners for High-Contrast Problems",
            subtitle="Three Tailored Preconditioners",
            additional_animations=[
                Write(M_1),
                Write(M_2),
                Write(M_3),
                *[Write(citation) for citation in citations[1:]],
            ],
            transition_time=2 * self.RUN_TIME,
        )

        # slide: simplify labels
        M_1_simple = TexText(
            r"$M_1$",
            font_size=1.5 * CONTENT_FONT_SIZE,
        )
        M_2_simple = TexText(
            r"$M_2$",
            font_size=1.5 * CONTENT_FONT_SIZE,
        )
        M_3_simple = TexText(
            r"$M_3$",
            font_size=1.5 * CONTENT_FONT_SIZE,
        )
        for i, M in enumerate([M_1_simple, M_2_simple, M_3_simple]):
            M.next_to(coefficient_rectangle, DOWN, buff=0).shift((i - 1) * 1.0 * RIGHT)
        self.update_slide(
            title="Preconditioners for High-Contrast Problems",
            subtitle="Three Tailored Preconditioners",
            notes="which we will refer to as M1, M2, and M3.",
            additional_animations=[
                ReplacementTransform(M_1, M_1_simple),
                ReplacementTransform(M_2, M_2_simple),
                ReplacementTransform(M_3, M_3_simple),
            ],
            transition_time=2 * self.RUN_TIME,
        )

        # slide: all preconditioners had similar condition numbers
        M_2_kappa = TexText(
            r"$\sim\kappa(M_2^{-1}A)\sim^*$",
            font_size=1.5 * CONTENT_FONT_SIZE,
        )
        M_1_kappa = TexText(
            r"$\kappa(M_1^{-1}A)$",
            font_size=1.5 * CONTENT_FONT_SIZE,
        ).next_to(M_2_kappa, LEFT, buff=0.1)
        M_3_kappa = TexText(
            r"$\kappa(M_3^{-1}A)$",
            font_size=1.5 * CONTENT_FONT_SIZE,
        ).next_to(M_2_kappa, RIGHT, buff=0.1)
        footnote = (
            TexText(
                r"$^*$ $\sigma(M_3^{-1}A)$ is actually lower for $C_{\text{3layer, vert}}$",
                font_size=self.slide_number_size,
            )
            .align_to(self.BOTTOM_MARGIN, DOWN)
            .align_to(self.LEFT_MARGIN, LEFT)
            .shift(0.1 * RIGHT + 0.1 * UP)
        )
        self.update_slide(
            title="Preconditioner Performance",
            subtitle="The Condition Number as a Measure",
            notes="All three preconditioners resulted in similar condition numbers...",
            additional_animations=[
                *[FadeOut(citation) for citation in citations],
                FadeOut(coefficient_function_image),
                FadeOut(coefficient_rectangle),
                ReplacementTransform(M_1_simple, M_1_kappa),
                ReplacementTransform(M_2_simple, M_2_kappa),
                ReplacementTransform(M_3_simple, M_3_kappa),
                Write(footnote),
            ],
            transition_time=2 * self.RUN_TIME,
        )

        # slide: results from study
        vertex_func_results = (
            ImageMobject("cg_iterations_coefficient_vertex_Alves")
            .stretch_to_fit_width(0.4 * FRAME_WIDTH)
            .align_to(self.TOP_MARGIN, UP)
        )
        edge_func_results = (
            ImageMobject("cg_iterations_coefficient_edges_Alves")
            .stretch_to_fit_width(0.4 * FRAME_WIDTH)
            .next_to(vertex_func_results, RIGHT, buff=0.2)
        )
        vertex_func_plot = [
            vertex_func_results,
            TexText(
                r"$\mathcal{C}_{\text{3layer, vert}}$",
                font_size=CONTENT_FONT_SIZE,
            ).next_to(vertex_func_results, UP, buff=0.2),
            TexText(
                "Problem Size",
                font_size=CONTENT_FONT_SIZE,
            ).next_to(vertex_func_results, DOWN, buff=0.2),
            TexText("Iterations", font_size=CONTENT_FONT_SIZE)
            .rotate(90 * DEGREES)
            .next_to(vertex_func_results, LEFT, buff=0.2),
        ]
        edge_func_plot = [
            edge_func_results,
            TexText(
                r"$\mathcal{C}_{\text{edge slabs, around vertices}}$",
                font_size=CONTENT_FONT_SIZE,
            ).next_to(edge_func_results, UP, buff=0.2),
        ]
        plots = Group(*vertex_func_plot, *edge_func_plot)
        plots.move_to(ORIGIN).align_to(self.TOP_MARGIN, UP)
        legend = (
            ImageMobject("method_legend_Alves")
            .stretch_to_fit_height(0.05 * FRAME_HEIGHT)
            .stretch_to_fit_width(0.4 * FRAME_WIDTH)
            .next_to(plots, DOWN, buff=0.5)
        )
        M_1_legend = TexText(
            r"$M_1$",
            font_size=1.5 * CONTENT_FONT_SIZE,
        )
        M_2_legend = TexText(
            r"$M_2$",
            font_size=1.5 * CONTENT_FONT_SIZE,
        )
        M_3_legend = TexText(
            r"$M_3$",
            font_size=1.5 * CONTENT_FONT_SIZE,
        )
        for i, M in enumerate([M_1_legend, M_2_legend, M_3_legend]):
            M.add_background_rectangle()
            M.background_rectangle.set_fill(opacity=1.0, color=CustomColors.NAVY.value)
            M.background_rectangle.stretch_to_fit_width(1.3)
            M.background_rectangle.stretch_to_fit_height(legend.get_height())
            M.background_rectangle.set_stroke(opacity=0.0)
            M.move_to(legend.get_center()).align_to(legend, LEFT).shift(
                [0.7, 0, 0] + i * 2.0 * RIGHT
            )

        self.update_slide(
            notes="However the number of CG iterations varied significantly...",
            title="Preconditioner Performance",
            subtitle="PCG Iterations",
            additional_animations=[
                FadeOut(footnote),
                FadeOut(M_1_kappa),
                FadeOut(M_2_kappa),
                FadeOut(M_3_kappa),
                *[FadeIn(mobj) for mobj in vertex_func_plot],
                *[FadeIn(mobj) for mobj in edge_func_plot],
                FadeIn(legend),
                FadeIn(M_1_legend),
                FadeIn(M_2_legend),
                FadeIn(M_3_legend),
            ],
            transition_time=2 * self.RUN_TIME,
        )

        # slide: M_2 (RGDSW) took more iterations
        M_2_legend.generate_target()
        M_2_legend.target.scale(2.0)
        M_2_legend.background_rectangle.generate_target()
        M_2_legend.background_rectangle.target.stretch_to_fit_width(1.3)
        M_2_legend.background_rectangle.target.stretch_to_fit_height(
            legend.get_height()
        )
        self.update_slide(
            subtitle="PCG Iterations",
            additional_animations=[
                MoveToTarget(M_2_legend),
                MoveToTarget(M_2_legend.background_rectangle),
            ],
            transition_time=self.RUN_TIME,
        )

        # slide: restate research question
        self.slide_contents = (
            vertex_func_plot
            + edge_func_plot
            + [legend, M_1_legend, M_2_legend, M_3_legend]
        )

        self.main_question.move_to(ORIGIN).shift(2 * UP)
        addendum = VGroup()
        addendum_text = (
            defense.paragraph(
                "+ \\textit{can distinguish between performance of $M_1,M_2,M_3$}",
                font_size=self.title_size,
                alignment=ALIGN.CENTER,
                width=0.5 * FRAME_WIDTH,
            )
            .next_to(self.main_question, DOWN, buff=2.5)
            .align_to(self.main_question, LEFT)
            .shift(0.5*RIGHT)
        )
        addendum_bgr = BackgroundRectangle(
            addendum_text, color=CustomColors.GOLD.value, fill_opacity=1.0, buff=0.2
        ).round_corners(0.1)
        addendum_bgr.set_z_index(-1)
        # arrow = always_redraw(
        #     CurvedArrow, self.main_question[0].get_center(), addendum_bgr.get_left()
        # )
        addendum.add(addendum_bgr, addendum_text)
        self.sub_questions.generate_target()
        self.sub_questions.target.set_opacity(1)
        self.update_slide(
            notes="So we need to refine our research question further.",
            title="Research Question (Revised)",
            subtitle="Predicting Preconditioned CG Performance",
            new_contents=[self.main_question],
            additional_animations=[
                FadeIn(addendum),
                MoveToTarget(self.sub_questions)
            ],
            transition_time=2 * self.RUN_TIME,
        )
        # self.main_question.add(addendum)

        self.slide_contents = [self.main_question]
        self.sub_questions.add(addendum)
        self.sub_questions.generate_target()
        self.sub_questions.target.set_opacity(0)
        self.play(MoveToTarget(self.sub_questions), run_time=self.RUN_TIME)

    def level_4_two_clusters(self):
        # slide: classical general CG error bound
        cg_error_bound_uniform = TexText(
            r"$\epsilon_m \leq \underset{r\in\mathcal{P}_m,\ r(0)=1}{\min} \ \underset{\lambda \in [\lambda_{\min}, \lambda_{\max}]}{\max}|r(\lambda)|$",
            font_size=2.0 * CONTENT_FONT_SIZE,
            t2c={
                r"\lambda_{\min}, \lambda_{\max}": CustomColors.SKY.value,
            },
        )
        self.update_slide(
            "Towards Sharper Iteration Bounds",
            subtitle="Clustered Spectra",
            new_contents=[cg_error_bound_uniform],
            notes="We recap the classical CG error bound...",
            index="Towards a Sharper Iteration Bound",
        )

        # slide: Chebyshev polynomial
        text1 = TexText(
            "We can proof that the classical CG error bound's min-max problem",
            font_size=CONTENT_FONT_SIZE,
            t2c={r"min-max": CustomColors.RED.value},
        ).shift(UP)
        cg_error_bound_uniform_short = TexText(
            r"$\underset{r\in\mathcal{P}_m,\ r(0)=1}{\min} \ \underset{\lambda \in [\lambda_{\min}, \lambda_{\max}]}{\max}|r(\lambda)|$",
            font_size=2.0 * CONTENT_FONT_SIZE,
            t2c={
                r"\min": CustomColors.RED.value,
                r"\max": CustomColors.RED.value,
                r"\lambda_{\min}, \lambda_{\max}": CustomColors.SKY.value,
            },
        )
        text2 = TexText(
            r"is solved by a \textit{Chebyshev polynomial}.",
            font_size=CONTENT_FONT_SIZE,
            t2c={r"\textit{Chebyshev polynomial}": CustomColors.GOLD.value},
        ).shift(DOWN)
        chebyshev_poly = TexText(
            r"\begin{equation*}"
            r"    C_m(x) = \frac{1}{2}\left(\left(x + \sqrt{x^2 - 1}\right)^m + \left(x - \sqrt{x^2 - 1}\right)^m\right)"
            r"\end{equation*}",
            font_size=CONTENT_FONT_SIZE,
            t2c={r"C_m": CustomColors.GOLD.value},
        ).next_to(text2, DOWN, buff=0.5)
        self.update_slide(
            notes="which can be solved using the Chebyshev polynomial...",
            subtitle="Chebyshev Polynomial",
            additional_animations=[
                Write(text1),
                ReplacementTransform(
                    cg_error_bound_uniform, cg_error_bound_uniform_short
                ),
                Write(text2),
                Write(chebyshev_poly),
            ],
            transition_time=self.RUN_TIME,
        )

        # slide: transformed chebyshev polynomial
        trans_chebyshev_poly = TexText(
            r"\begin{equation*}"
            r"r(\lambda) = \hat{C}_m(\lambda) = \frac{C_m\left(\frac{2\lambda - (\lambda_{\min} + \lambda_{\max})}{\lambda_{\max} - \lambda_{\min}}\right)}{C_m\left(\frac{\lambda_{\min} + \lambda_{\max}}{\lambda_{\max} - \lambda_{\min}}\right)}"
            r"\end{equation*}",
            font_size=CONTENT_FONT_SIZE,
            t2c={
                r"\hat{C}_m": CustomColors.GOLD.value,
                r"\lambda_{\min}": CustomColors.SKY.value,
                r"\lambda_{\max}": CustomColors.SKY.value,
            },
        ).move_to(chebyshev_poly.get_center())
        text3 = TexText(
            r"is solved by a \textit{scaled} Chebyshev polynomial",
            font_size=CONTENT_FONT_SIZE,
            t2c={r"\textit{scaled} Chebyshev polynomial": CustomColors.GOLD.value},
        ).shift(DOWN)
        self.update_slide(
            notes="actually, a scaled Chebyshev polynomial...",
            subtitle="Scaled Chebyshev Polynomial",
            additional_animations=[
                ReplacementTransform(text2, text3),
                ReplacementTransform(chebyshev_poly, trans_chebyshev_poly),
            ],
            transition_time=self.RUN_TIME,
        )

        # slide: reintroduce general CG error bound
        text4 = TexText(
            "But what if we want to solve the min-max problem over the full spectrum of A?",
            font_size=CONTENT_FONT_SIZE,
            t2c={
                "spectrum of A": CustomColors.SKY.value,
                "min-max problem": CustomColors.RED.value,
            },
        ).shift(UP)
        cg_error_bound = TexText(
            r"$\underset{r\in\mathcal{P}_m,\ r(0)=1}{\min} \ \underset{\lambda \in \sigma(A)}{\max}|r(\lambda)|$",
            font_size=2.0 * CONTENT_FONT_SIZE,
            t2c={
                r"\sigma(A)": CustomColors.SKY.value,
                r"\min": CustomColors.RED.value,
                r"\max": CustomColors.RED.value,
            },
        )
        text5 = TexText(
            "Can we still use $\hat{C}_m$?",
            font_size=CONTENT_FONT_SIZE,
        ).shift(DOWN)
        self.update_slide(
            notes="But what if we reintroduce the full spectrum of A?",
            title="Clustered Spectra",
            subtitle="General CG Error Bound",
            additional_animations=[
                ReplacementTransform(text1, text4),
                ReplacementTransform(cg_error_bound_uniform_short, cg_error_bound),
                FadeOut(text3),
                FadeOut(trans_chebyshev_poly),
                FadeIn(text5),
            ],
            transition_time=self.RUN_TIME,
        )

        # slide: two clusters spectrum
        self.slide_contents = [
            text4,
            cg_error_bound,
            text5,
        ]
        a = 0.1
        b = 0.2
        c = 0.6
        d = 0.9
        two_cluster_spectrum = self.generate_clustered_spectrum(
            [(a, b), (c, d)],
            ["a", "b", "c", "d"],
        )
        spectrum_label = Tex(
            r"\sigma(A) = [a,b] \cup [c,d]",
            font_size=CONTENT_FONT_SIZE,
            t2c={r"\sigma(A)": CustomColors.SKY.value},
        )
        always(spectrum_label.next_to, two_cluster_spectrum, UP, buff=0.5)
        self.update_slide(
            subtitle="Simplest, Non-Trivial Case: Two-Cluster Spectra",
            additional_animations=[Write(two_cluster_spectrum), Write(spectrum_label)],
            notes="we consider the simplest, non-trivial case of two-cluster spectra.",
        )

        # slide: how can we solve the min-max problem?
        text_minmax = TexText(
            "How can we solve the min-max problem?",
            t2c={"min-max": CustomColors.RED.value},
            font_size=CONTENT_FONT_SIZE,
        ).next_to(two_cluster_spectrum, DOWN, buff=0.5)
        self.update_slide(
            notes="How can we solve the min-max problem in this case?",
            subtitle="Simplest, Non-Trivial Case: Two-Cluster Spectra",
            additional_animations=[
                Write(text_minmax),
            ],
            transition_time=self.RUN_TIME,
        )

        # slide: composite Chebyshev polynomial
        text_axelsson = TexText(
            "We simply use the product of two scaled Chebyshev polynomials!",
            font_size=CONTENT_FONT_SIZE,
            t2c={"two scaled Chebyshev polynomials": CustomColors.GOLD.value},
        ).next_to(two_cluster_spectrum, DOWN, buff=0.5)
        citation = cite("cg_sharpened_convrate_Axelsson1976").next_to(
            text_axelsson, RIGHT, buff=0.1
        )
        left_chebyshev = Tex(
            r"\hat{C}_{p}^{[a,b]}(\lambda)",
            font_size=CONTENT_FONT_SIZE,
            t2c={
                r"p": CustomColors.GOLD.value,
                r"a": CustomColors.SKY.value,
                r"b": CustomColors.SKY.value,
            },
        )
        f_always(
            left_chebyshev.move_to,
            lambda: two_cluster_spectrum.get_left()
            + [(a + b) / 2 * two_cluster_spectrum.get_width(), 0.5, 0],
        )
        right_chebyshev = Tex(
            r"\hat{C}_{p-m}^{[c,d]}(\lambda)",
            font_size=CONTENT_FONT_SIZE,
            t2c={
                r"p-m": CustomColors.GOLD.value,
                r"c": CustomColors.SKY.value,
                r"d": CustomColors.SKY.value,
            },
        )
        f_always(
            right_chebyshev.move_to,
            lambda: two_cluster_spectrum.get_left()
            + [(c + d) / 2 * two_cluster_spectrum.get_width(), 0.5, 0],
        )
        composite_chebyshev_poly = Tex(
            r"r(\lambda) \approx \hat{C}_{p}^{[a,b]}(\lambda) \cdot \hat{C}_{p-m}^{[c,d]}(\lambda) \in \mathcal{P}_m",
            font_size=CONTENT_FONT_SIZE,
            t2c={
                r"\hat{C}_{m - p}": CustomColors.GOLD.value,
                r"\hat{C}_{m}": CustomColors.GOLD.value,
                r"a": CustomColors.SKY.value,
                r"b": CustomColors.SKY.value,
                r"c": CustomColors.SKY.value,
                r"d": CustomColors.SKY.value,
            },
        ).next_to(text_axelsson, DOWN, buff=0.5)
        self.next_slide(
            notes="We use the product of two scaled Chebyshev polynomials.",
            additional_animations=[
                ReplacementTransform(text_minmax, text_axelsson),
                Write(citation),
                Write(composite_chebyshev_poly),
                Write(left_chebyshev),
                Write(right_chebyshev),
            ],
        )

        # slide: two-cluster CG iteration bound
        text_two_cluster = TexText(
            "This leads to the two-cluster CG iteration bound from Axelsson (1976)",
            font_size=CONTENT_FONT_SIZE,
            t2c={r"two-cluster CG iteration bound": CustomColors.GOLD.value},
        )
        two_cluster_cg_bound = TexText(
            r"\["
            r"m \leq m_2(a,b,c,d) = \left\lfloor\frac{1}{2}\sqrt{\frac{d}{c}}\ln\left(\frac{2}{\epsilon}\right) + \left(1+\frac{1}{2}\sqrt{\frac{d}{c}}\ln\left(\frac{4 d}{b}\right)\right) p(a,b)\right\rfloor"
            r"\]",
            font_size=CONTENT_FONT_SIZE,
            t2c={
                r"m_2": CustomColors.GOLD.value,
                r"a": CustomColors.SKY.value,
                r"b": CustomColors.SKY.value,
                r"c": CustomColors.SKY.value,
                r"d": CustomColors.SKY.value,
            },
        ).shift(DOWN)
        self.update_slide(
            title="Cluster Based CG Iteration Bounds",
            subtitle="Two-Cluster",
            additional_animations=[
                two_cluster_spectrum[0].animate.shift(UP),
                Write(text_two_cluster),
                Write(two_cluster_cg_bound),
                FadeOut(text_axelsson),
                FadeOut(citation),
                FadeOut(composite_chebyshev_poly),
            ],
            notes="This leads to the two-cluster CG iteration bound from Axelsson (1976).",
        )

        # slide: but why stop at two clusters?
        text_why_stop = TexText(
            "But why stop at two clusters? We can extend this to multi-cluster spectra!",
            font_size=CONTENT_FONT_SIZE,
            t2c={r"multi-cluster spectra": CustomColors.GOLD.value},
        ).move_to(two_cluster_spectrum.get_center())
        a_1 = 0.1
        b_1 = 0.15
        a_2 = 0.24
        b_2 = 0.3
        a_3 = 0.35
        b_3 = 0.7
        a_4 = 0.75
        b_4 = 0.99
        multi_cluster_spectrum = self.generate_clustered_spectrum(
            [(a_1, b_1), (a_2, b_2), (a_3, b_3), (a_4, b_4)],
            ["a_1", "b_1", "a_2", "b_2", "a_3", "b_3", "a_4", "b_4"],
        )
        cg_general_residual_poly = Tex(
            r"r(\lambda) \approx \hat{C}_{p_1}^{[a_1,b_1]}(\lambda) \cdot \hat{C}_{p_2}^{[a_2,b_2]}(\lambda) \cdot \hat{C}_{p_3}^{[a_3,b_3]}(\lambda) \cdot \hat{C}_{p_4}^{[a_4,b_4]}(\lambda) \in \mathcal{P}_m",
            font_size=CONTENT_FONT_SIZE,
        ).next_to(multi_cluster_spectrum, DOWN, buff=1.0)
        two_cluster_spectrum.clear_updaters()
        self.update_slide(
            subtitle="Multi-Cluster",
            additional_animations=[
                ReplacementTransform(text_two_cluster, text_why_stop),
                ReplacementTransform(two_cluster_spectrum, multi_cluster_spectrum),
                Write(cg_general_residual_poly),
                FadeOut(two_cluster_cg_bound),
                FadeOut(spectrum_label),
                FadeOut(left_chebyshev),
                FadeOut(right_chebyshev),
            ],
            notes="But why stop at two clusters?",
        )

        # slide: multi-cluster CG animation
        super().next_slide(
            notes="We actually want to develop the most general bound...", loop=True
        )
        # NOTE: using updaters
        # self.now = self.time
        # multi_cluster_spectrum.add_updater(
        #     lambda m, dt: self.multi_cluster_spectrum_updater(
        #         m, dt, update_frequency=2.0
        #     )
        # )
        # self.play(multi_cluster_spectrum.animate.scale(1.0), run_time=10.0)
        # NOTE: explicitly generating multiple random spectra
        num_spectra = 5
        spectra, res_polys = self.generate_multiple_random_spectra(num_spectra)
        all_spectra = [
            multi_cluster_spectrum,
            *spectra,
        ]
        all_res_polys = [
            cg_general_residual_poly,
            *res_polys,
        ]
        og_spectrum = multi_cluster_spectrum.copy()
        og_respoly = cg_general_residual_poly.copy()
        for curr_spectrum, new_spectrum, curr_poly, new_poly in zip(
            all_spectra[:-1], all_spectra[1:], all_res_polys[:-1], all_res_polys[1:]
        ):
            self.play(
                FadeOut(curr_spectrum),
                FadeIn(new_spectrum),
                ReplacementTransform(curr_poly, new_poly),
                run_time=3.0 * self.RUN_TIME,
            )
        self.play(
            FadeOut(new_spectrum),
            FadeIn(og_spectrum),
            ReplacementTransform(new_poly, og_respoly),
            run_time=3.0 * self.RUN_TIME,
        )
        super().next_slide()

        # slide: general cg iteration bound
        text_result = TexText(
            "This leads to the \\textit{general CG iteration bound}.",
            font_size=CONTENT_FONT_SIZE,
            t2c={"\\textit{general CG iteration bound}": CustomColors.GOLD.value},
        ).move_to(text_why_stop.get_center()).shift(UP)
        multi_cluster_cg_bound = TexText(
            r"\begin{equation*}"
            r"  m_{N_{\text{cluster}}} = \sum_{i=1}^{N_{\text{cluster}}} p_i"
            r"\end{equation*}",
            font_size=CONTENT_FONT_SIZE,
            t2c={
                r"m_{N_{\text{cluster}}}": CustomColors.GOLD.value,
                r"p_i": CustomColors.RED.value,
            },
        )
        text_where = TexText(
            "where the \\textit{Chebyshev degrees} $p_i$, and \\textit{cluster factors} $\gamma_i$ satisfy",
            font_size=CONTENT_FONT_SIZE,
            t2c={
                "\\textit{Chebyshev degrees} $p_i$": CustomColors.RED.value,
                "\\textit{cluster factors} $\gamma_i$": CustomColors.RED.value,
            },
        ).next_to(multi_cluster_cg_bound, DOWN, buff=0.5)
        p_i_equation = TexText(
            r"\begin{equation*}"
            r"  p_i \leq \left\lceil\log_{\gamma_i}{\frac{\epsilon}{2}} + \sum_{j=1}^{i-1} p_j\log_{\gamma_i}\left(\frac{\zeta^{(j)}_2(a_j, b_j)}{\zeta^{(i,j)}_1(a_j, b_j, b_i)}\right)\right\rceil"
            r"\end{equation*}",
            font_size=CONTENT_FONT_SIZE,
            t2c={
                "p_i": CustomColors.RED.value,
                r"\gamma_i": CustomColors.RED.value,
                r"a_j": CustomColors.SKY.value,
                r"b_j": CustomColors.SKY.value,
                r"b_i": CustomColors.SKY.value,
            },
        )
        gamma_i_equation = TexText(
            r"\begin{equation*}"
            r"\gamma_i = \frac{\sqrt{b_i} - \sqrt{a_i}}{\sqrt{b_i} + \sqrt{a_i}}, \quad i=1,\ldots,N_{\text{cluster}}"
            r"\end{equation*}",
            font_size=CONTENT_FONT_SIZE,
            t2c={
                r"\gamma_i": CustomColors.RED.value,
                "b_i": CustomColors.SKY.value,
                "a_i": CustomColors.SKY.value,
            },
        )
        sub_equations = (
            VGroup(p_i_equation, gamma_i_equation)
            .arrange(RIGHT, buff=0.5)
            .next_to(text_where, DOWN, buff=0.5)
        )
        og_spectrum_backup = og_spectrum.copy()
        self.update_slide(
            subtitle="General CG Iteration Bound",
            notes="leading to the most general CG iteration bound.",
            additional_animations=[
                ReplacementTransform(text_why_stop, text_result),
                FadeOut(og_spectrum),
                FadeOut(og_respoly),
                Write(multi_cluster_cg_bound),
                Write(text_where),
                Write(sub_equations),
            ],
        )

        # slide: general CG iteration bound as a function of clusters
        multi_cluster_cg_bound_simple = TexText(
            r"\[" r"m_{N_{\text{cluster}}}(\text{clusters})" r"\]",
            font_size=2.0 * CONTENT_FONT_SIZE,
            t2c={
                r"m_{N_{\text{cluster}}}": CustomColors.GOLD.value,
                r"\text{clusters}": CustomColors.RED.value,
            },
        ).move_to(multi_cluster_cg_bound.get_center())
        og_spectrum_backup.clear_updaters()
        og_spectrum_backup.next_to(multi_cluster_cg_bound_simple, DOWN, buff=0.5)
        bars_and_labels = og_spectrum_backup[-16:]
        for label in bars_and_labels:  # get the cluster labels
            label.set_color(CustomColors.RED.value)
        self.update_slide(
            subtitle="General CG Iteration Bound",
            notes="This is a simplified view of the general CG iteration bound.",
            additional_animations=[
                ReplacementTransform(
                    multi_cluster_cg_bound, multi_cluster_cg_bound_simple
                ),
                FadeInFromPoint(
                    og_spectrum_backup, multi_cluster_cg_bound_simple.get_center()
                ),
                FadeOut(text_where),
                FadeOut(sub_equations),
                FadeOut(text_result),
            ],
        )

        # slide: but how do we get from spectrum of A to clusters?
        text_how_to_get_clusters = TexText(
            "But how do we get from the spectrum of A to a set of clusters?",
            font_size=CONTENT_FONT_SIZE,
            t2c={
                "spectrum of A": CustomColors.SKY.value,
                "clusters": CustomColors.RED.value,
            },
        ).next_to(multi_cluster_cg_bound_simple, UP, buff=0.3)
        for bar_label in bars_and_labels:
            bar_label.generate_target()
            bar_label.target.shift(0.5 * DOWN)
        og_spectrum_pure = og_spectrum_backup[:-16]
        for mobj in og_spectrum_pure:
            mobj.generate_target()
            mobj.target.shift(0.5 * UP)
        self.next_slide(
            notes="But how do we get from the spectrum of A to a set of clusters?",
            title="Partitioning Algorithms",
            subtitle="From Spectrum to Clusters",
            additional_animations=[
                ReplacementTransform(
                    multi_cluster_cg_bound_simple, text_how_to_get_clusters
                ),
                *[MoveToTarget(bar_label) for bar_label in bars_and_labels],
                *[MoveToTarget(mobj) for mobj in og_spectrum_pure],
            ],
        )

        # slide: partitioning algorithm
        text_partitioning = TexText(
            "Let's try to do this for a sample spectrum.",
            font_size=CONTENT_FONT_SIZE,
        )
        num_clusters = 3
        arrow_length = 7
        spectrum_for_partitioning, all_eigs_part = self.generate_clustered_spectrum(
            [(0.1, 0.25), (0.6, 0.7), (0.75, 0.95)],
            ["a_1", "b_1", "a_2", "b_2", "a_3", "b_3"],
            randomize=True,
            resolution=100,
            arrow_length=arrow_length,
            return_eigs=True
        )
        spectrum_for_partitioning[0].next_to(text_partitioning, DOWN, buff=0.5)
        self.next_slide(
            subtitle="Two-Cluster Case",
            additional_animations=[
                ReplacementTransform(text_how_to_get_clusters, text_partitioning),
                FadeOut(og_spectrum_backup),
                FadeIn(spectrum_for_partitioning),
            ],
        )

        # slide: two-cluster partitioning setup
        text_algorithm = TexText(
            "We start by assuming no knowledge of the cluster boundaries and then find the largest relative gap in the spectrum.",
            font_size=CONTENT_FONT_SIZE,
            t2c={"largest relative gap": CustomColors.RED.value},
        ).move_to(text_partitioning.get_center())
        spectrum_arrow = spectrum_for_partitioning[0]
        spectrum_arrow.generate_target()
        spectrum_arrow.target.stretch_to_fit_width(FRAME_WIDTH)
        spectrum_arrow.target.align_to(ORIGIN, LEFT)
        spectrum_arrow.target.shift(LEFT)
        spectrum_for_partitioning_eigs = spectrum_for_partitioning[
            2 : -4 * num_clusters
        ]
        spectrum_for_partitioning_bars_labels = spectrum_for_partitioning[
            -4 * num_clusters :
        ]
        self.update_slide(
            additional_animations=[
                ReplacementTransform(text_partitioning, text_algorithm),
                MoveToTarget(spectrum_arrow),
                *[
                    FadeOut(bar_label)
                    for bar_label in spectrum_for_partitioning_bars_labels
                ],
            ],
            notes="We start by assuming no knowledge of the cluster boundaries...",
            transition_time=2 * self.RUN_TIME,
            subtitle="Partitioning: two-cluster case",
        )

        # slide: partitioning animation (start)
        text_largest_gap = TexText(
            "Step.1 Find index of largest relative gap: ",
            font_size=CONTENT_FONT_SIZE,
            t2c={"index": CustomColors.RED.value},
        )
        text_largest_gap_eq = Tex(
            r"k^* = \max_i \left\{\frac{\lambda_{i+1}}{\lambda_i}, i=1,\ldots,n\right\} =",
            font_size=CONTENT_FONT_SIZE,
            t2c={
                r"k^*": CustomColors.RED.value,
                r"\lambda_i": CustomColors.GOLD.value,
                r"\lambda_{i+1}": CustomColors.GOLD.value,
            },
        ).next_to(text_largest_gap, RIGHT, buff=0.1)
        text_curr_k = DecimalNumber(
            1,
            font_size=CONTENT_FONT_SIZE,
            color=CustomColors.RED.value,
            num_decimal_places=0,
        ).next_to(text_largest_gap_eq, RIGHT, buff=0.1)
        step_1 = VGroup(text_largest_gap, text_largest_gap_eq, text_curr_k).move_to(
            text_algorithm.get_center()
        )
        l1 = (
            spectrum_for_partitioning_eigs[0].get_center()[0]
            - spectrum_arrow.get_start()[0]
        )
        l2 = (
            spectrum_for_partitioning_eigs[1].get_center()[0]
            - spectrum_arrow.get_start()[0]
        )
        relative_gap = l2 / l1
        largest_gap_curr_val = DecimalNumber(
            relative_gap,
            font_size=CONTENT_FONT_SIZE,
            num_decimal_places=2,
        )
        lambda_i_arrow_label = VGroup(
            arrow_i := Arrow(
                start=spectrum_for_partitioning_eigs[0].get_center()
                - (1.0 * UP + 0.1 * RIGHT),
                end=spectrum_for_partitioning_eigs[0].get_center(),
                buff=0.0,
                fill_color=CustomColors.GOLD.value,
            ),
            label_i := Tex(
                r"\lambda",
                font_size=CONTENT_FONT_SIZE,
                t2c={r"\lambda": CustomColors.GOLD.value},
            ).next_to(arrow_i, DOWN + LEFT, buff=0.0),
        )
        lambda_ip1_arrow_label = VGroup(
            arrow_ip1 := Arrow(
                start=spectrum_for_partitioning_eigs[1].get_center()
                - (1.0 * UP + 0.1 * LEFT),
                end=spectrum_for_partitioning_eigs[1].get_center(),
                buff=0.0,
                fill_color=CustomColors.GOLD.value,
            ),
            label_ip1 := Tex(
                r"\lambda",
                font_size=CONTENT_FONT_SIZE,
                t2c={r"\lambda": CustomColors.GOLD.value},
            ).next_to(arrow_ip1, DOWN + RIGHT, buff=0.0),
        )
        index_i = DecimalNumber(
            1,
            font_size=0.5 * CONTENT_FONT_SIZE,
            color=CustomColors.GOLD.value,
            num_decimal_places=0,
        ).next_to(label_i, 0.5 * DOWN + RIGHT, buff=0.05)
        index_ip1 = DecimalNumber(
            2,
            font_size=0.5 * CONTENT_FONT_SIZE,
            color=CustomColors.GOLD.value,
            num_decimal_places=0,
        ).next_to(label_ip1, 0.5 * DOWN + RIGHT, buff=0.05)
        always(index_i.next_to, label_i, 0.5 * DOWN + RIGHT, buff=0.05)
        always(index_ip1.next_to, label_ip1, 0.5 * DOWN + RIGHT, buff=0.05)
        label_i_cp = label_i.copy()
        index_i_cp = index_i.copy()
        label_ip1_cp = label_ip1.copy()
        index_ip1_cp = index_ip1.copy()
        always(index_i_cp.next_to, label_i_cp, 0.5 * DOWN + RIGHT, buff=0.05)
        always(index_ip1_cp.next_to, label_ip1_cp, 0.5 * DOWN + RIGHT, buff=0.05)
        text_curr_relative_gap = VGroup(
            label_ip1_cp,
            TexText(r"\ /", font_size=CONTENT_FONT_SIZE),
            label_i_cp,
            Tex(r"=", font_size=CONTENT_FONT_SIZE),
            largest_gap_curr_val,
        )

        def label_i_updater(mobj: Mobject, dt: float):
            mobj.next_to(arrow_i, DOWN + LEFT, buff=0.0)
            return mobj

        label_i.add_updater(lambda m, dt: label_i_updater(m, dt))

        def label_ip1_updater(mobj: Mobject, dt: float):
            mobj.next_to(arrow_ip1, DOWN + RIGHT, buff=0.0)
            return mobj

        label_ip1.add_updater(lambda m, dt: label_ip1_updater(m, dt))

        always(text_curr_relative_gap.arrange, RIGHT, buff=0.3)
        always(text_curr_relative_gap.next_to, arrow_ip1, RIGHT, buff=0.5)
        self.update_slide(
            notes="we find the largest relative gap in the spectrum...",
            # loop=True,
            subtitle="Two-Cluster Case",
            additional_animations=[
                Write(lambda_i_arrow_label),
                Write(lambda_ip1_arrow_label),
                Write(index_i),
                Write(index_ip1),
                spectrum_for_partitioning_eigs[0].animate.set_fill(
                    CustomColors.GOLD.value
                ),
                spectrum_for_partitioning_eigs[1].animate.set_fill(
                    CustomColors.GOLD.value
                ),
                ReplacementTransform(text_algorithm, step_1),
                Write(text_curr_relative_gap),
                Write(index_i_cp),
                Write(index_ip1_cp),
            ],
        )

        # # slide: partitioning animation (run) NOTE: The animation used to work but now causes a render error
        # iteration_time = 0.2  # seconds
        # sim_time = (len(spectrum_for_partitioning_eigs) - 2) * iteration_time
        # self.start = self.time

        # def get_index():
        #     return int(np.floor((self.time - self.start) / iteration_time))

        # def arrow_i_updater(mobj, dt: float):
        #     index = get_index()
        #     mobj.set_points_by_ends(
        #         spectrum_for_partitioning_eigs[index].get_center()
        #         - (1.0 * UP + 0.1 * RIGHT),
        #         spectrum_for_partitioning_eigs[index].get_center(),
        #     )
        #     return mobj

        # arrow_i.add_updater(lambda m, dt: arrow_i_updater(m, dt))

        # def arrow_ip1_updater(mobj, dt: float):
        #     index = get_index()
        #     mobj.set_points_by_ends(
        #         spectrum_for_partitioning_eigs[index + 1].get_center()
        #         - (1.0 * UP + 0.1 * LEFT),
        #         spectrum_for_partitioning_eigs[index + 1].get_center(),
        #     )
        #     return mobj

        # arrow_ip1.add_updater(lambda m, dt: arrow_ip1_updater(m, dt))

        # def decimal_updater_i(mobj: Mobject, dt: float):
        #     index = get_index()
        #     mobj.set_value(index + 1)
        #     return mobj

        # always(index_i.next_to, label_i, 0.5 * DOWN + RIGHT, buff=0.05)
        # index_i.add_updater(lambda m, dt: decimal_updater_i(m, dt))

        # def decimal_updater_ip1(mobj: Mobject, dt: float):
        #     index = get_index()
        #     mobj.set_value(index + 2)
        #     return mobj

        # always(index_ip1.next_to, label_ip1, 0.5 * DOWN + RIGHT, buff=0.05)
        # index_ip1.add_updater(lambda m, dt: decimal_updater_ip1(m, dt))

        # def decimal_updater_i_cp(mobj: Mobject, dt: float):
        #     index = get_index()
        #     mobj.set_value(index + 1)
        #     return mobj

        # always(index_i_cp.next_to, label_i_cp, 0.5 * DOWN + RIGHT, buff=0.05)
        # index_i_cp.add_updater(lambda m, dt: decimal_updater_i_cp(m, dt))

        # def decimal_updater_ip1_cp(mobj: Mobject, dt: float):
        #     index = get_index()
        #     mobj.next_to(label_ip1_cp, 0.5 * DOWN + RIGHT, buff=0.05)
        #     mobj.set_value(index + 2)
        #     return mobj

        # always(index_ip1_cp.next_to, label_ip1_cp, 0.5 * DOWN + RIGHT, buff=0.05)
        # index_ip1_cp.add_updater(lambda m, dt: decimal_updater_ip1_cp(m, dt))

        # def eig_dot_color_updater(mobj: Mobject, dt: float):
        #     index = get_index()
        #     if (
        #         mobj == spectrum_for_partitioning_eigs[index]
        #         or mobj == spectrum_for_partitioning_eigs[index + 1]
        #     ):
        #         mobj.set_fill(CustomColors.GOLD.value)
        #     return mobj

        # for eig_dot in spectrum_for_partitioning_eigs:
        #     eig_dot.add_updater(lambda m, dt: eig_dot_color_updater(m, dt))

        # curr_k = 1
        # max_gap = 0

        # def decimal_updater_relative_gap(mobj: Mobject, dt: float):
        #     nonlocal max_gap, curr_k
        #     index = get_index()
        #     l1 = (
        #         spectrum_for_partitioning_eigs[index].get_center()[0]
        #         - spectrum_arrow.get_start()[0]
        #     )
        #     l2 = (
        #         spectrum_for_partitioning_eigs[index + 1].get_center()[0]
        #         - spectrum_arrow.get_start()[0]
        #     )
        #     relative_gap = l2 / l1
        #     if max_gap < relative_gap:
        #         max_gap = relative_gap
        #         curr_k = index
        #     mobj.set_value(relative_gap)
        #     return mobj

        # largest_gap_curr_val.add_updater(
        #     lambda m, dt: decimal_updater_relative_gap(m, dt)
        # )
        # text_curr_k.add_updater(lambda m, dt: m.set_value(curr_k + 1))
        # super().next_slide(loop=True)
        # self.play(
        #     spectrum_arrow.animate.align_to(ORIGIN, RIGHT),
        #     run_time=sim_time,
        #     rate_func=smooth,
        # )
        # super().next_slide()

    #     # slide: go to largest gap
    #     arrow_i.remove_updater(arrow_i.get_updaters()[-1])
    #     arrow_ip1.remove_updater(arrow_ip1.get_updaters()[-1])
    #     index_i.remove_updater(index_i.get_updaters()[-1])
    #     index_ip1.remove_updater(index_ip1.get_updaters()[-1])
    #     index_i_cp.remove_updater(index_i_cp.get_updaters()[-1])
    #     index_ip1_cp.remove_updater(index_ip1_cp.get_updaters()[-1])
    #     largest_gap_curr_val.remove_updater(largest_gap_curr_val.get_updaters()[-1])
    #     for eig_dot in spectrum_for_partitioning_eigs:
    #         eig_dot.remove_updater(eig_dot.get_updaters()[-1])
        eigs = np.array(all_eigs_part)
        rel_distances = eigs[1:] / eigs[:-1]
        max_gap = np.max(rel_distances)
        curr_k = int(np.argmax(rel_distances))
        def arrow_i_updater_final(mobj, dt: float):
            mobj.set_points_by_ends(
                spectrum_for_partitioning_eigs[curr_k].get_center()
                - (1.0 * UP + 0.1 * RIGHT),
                spectrum_for_partitioning_eigs[curr_k].get_center(),
            )
            return mobj

        arrow_i.add_updater(lambda m, dt: arrow_i_updater_final(m, dt))

        def arrow_ip1_updater_final(mobj, dt: float):
            mobj.set_points_by_ends(
                spectrum_for_partitioning_eigs[curr_k + 1].get_center()
                - (1.0 * UP + 0.1 * LEFT),
                spectrum_for_partitioning_eigs[curr_k + 1].get_center(),
            )
            return mobj

        arrow_ip1.add_updater(lambda m, dt: arrow_ip1_updater_final(m, dt))
        spectrum_arrow.generate_target()
        spectrum_arrow.target.align_on_border(LEFT)
        spectrum_arrow.target.shift(0.2 * DOWN)
        spectrum_arrow.target.stretch_to_fit_width(0.9 * FRAME_WIDTH)
        step_2 = (
            TexText(
                r"Step.2 Check performance condition. Let $\kappa_l = \frac{\lambda_{k^*}}{\lambda_1}$, $\kappa_r = \frac{\lambda_{n}}{\lambda_{k^*+1}}$, and $W_{-1}(\cdot)$ be the \textit{Lambert} $W$ function. Then $m_{2} < m_1$ if"
                r"\[\kappa \geq 4\kappa_l\kappa_r W_{-1}\left(-\frac{1}{2\sqrt{\kappa_r}\exp\left(\frac{1}{\sqrt{\kappa_r}}\right)}\right)^2.\]",
                font_size=CONTENT_FONT_SIZE,
                t2c={
                    r"\kappa_l": CustomColors.SKY.value,
                    r"\kappa_r": CustomColors.SKY.value,
                    r"\textit{Lambert}": CustomColors.GOLD.value,
                    r"W": CustomColors.GOLD.value,
                    r"m_{2}": CustomColors.GOLD.value,
                    r"m_1": CustomColors.GOLD.value,
                },
            )
            .next_to(spectrum_arrow, UP, buff=0.2)
            .shift(-spectrum_arrow.get_center()[0] * RIGHT)
        )
        left_cluster = VGroup(
            *spectrum_for_partitioning_eigs[: curr_k + 1],
        )
        right_cluster = VGroup(
            *spectrum_for_partitioning_eigs[curr_k + 1 :],
        )
        lambda_1_label = Tex(
            r"\lambda_1",
            font_size=CONTENT_FONT_SIZE,
            t2c={r"\lambda_1": CustomColors.SKY.value},
        )
        f_always(
            lambda_1_label.move_to, lambda: left_cluster[0].get_center() + 0.3 * UP
        )
        lambda_k_label = Tex(
            r"\lambda_{k^*}",
            font_size=CONTENT_FONT_SIZE,
            t2c={r"\lambda_{k^*}": CustomColors.SKY.value},
        )
        f_always(
            lambda_k_label.move_to, lambda: left_cluster[-1].get_center() + 0.3 * UP
        )
        lambda_kp1_label = Tex(
            r"\lambda_{k^*+1}",
            font_size=CONTENT_FONT_SIZE,
            t2c={r"\lambda_{k^*+1}": CustomColors.SKY.value},
        )
        f_always(
            lambda_kp1_label.move_to, lambda: right_cluster[0].get_center() + 0.3 * UP
        )
        lambda_n_label = Tex(
            r"\lambda_{n}",
            font_size=CONTENT_FONT_SIZE,
            t2c={r"\lambda_{n}": CustomColors.SKY.value},
        )
        f_always(
            lambda_n_label.move_to, lambda: right_cluster[-1].get_center() + 0.3 * UP
        )
        self.update_slide(
            additional_animations=[
                MoveToTarget(spectrum_arrow),
                largest_gap_curr_val.animate.set_value(max_gap),
                step_1.animate.next_to(step_2, 0.5 * UP),
                Write(step_2),
                index_i.animate.set_value(curr_k + 1),
                index_ip1.animate.set_value(curr_k + 2),
                index_i_cp.animate.set_value(curr_k + 1),
                index_ip1_cp.animate.set_value(curr_k + 2),
                Write(lambda_1_label),
                Write(lambda_k_label),
                Write(lambda_kp1_label),
                Write(lambda_n_label),
            ],
            subtitle="Split Condition",
            notes="we find the largest relative gap in the spectrum.",
        )

        # slide: two-cluster partitioning result
        step_3 = TexText(
            "Step.3 If this condition is satisfied, we perform recursion on the two partitions.",
            font_size=CONTENT_FONT_SIZE,
            t2c={"recursion": CustomColors.GOLD.value},
        ).next_to(step_2, DOWN, buff=0.5)
        lambda_i_arrow_label.clear_updaters()
        lambda_ip1_arrow_label.clear_updaters()
        index_i.clear_updaters()
        index_ip1.clear_updaters()
        index_i_cp.clear_updaters()
        index_ip1_cp.clear_updaters()
        largest_gap_curr_val.clear_updaters()
        brace_left = always_redraw(
            Brace,
            left_cluster,
            DOWN,
            buff=0.1,
        )
        label_left = always_redraw(
            brace_left.get_text, "Left Partition", font_size=0.8 * CONTENT_FONT_SIZE
        )
        brace_right = always_redraw(
            Brace,
            right_cluster,
            DOWN,
            buff=0.1,
        )
        label_right = always_redraw(
            brace_right.get_text, "Right Partition", font_size=0.8 * CONTENT_FONT_SIZE
        )
        spectrum_arrow.generate_target()
        spectrum_arrow.target.move_to(ORIGIN)
        spectrum_arrow.target.scale(arrow_length / spectrum_arrow.get_length())
        spectrum_arrow.target.next_to(step_3, DOWN, buff=0.5)
        self.next_slide(
            subtitle="Recursion",
            additional_animations=[
                MoveToTarget(spectrum_arrow),
                Write(step_3),
                FadeOut(lambda_i_arrow_label),
                FadeOut(lambda_ip1_arrow_label),
                FadeOut(index_i),
                FadeOut(index_ip1),
                FadeOut(index_i_cp),
                FadeOut(index_ip1_cp),
                FadeOut(text_curr_relative_gap),
                Write(brace_left),
                Write(label_left),
                Write(brace_right),
                Write(label_right),
            ],
            notes="If this condition is satisfied, we perform recursion on the two partitions.",
        )

        # slide: zoom in on left partition
        spectrum_arrow.generate_target()
        diff_vec = left_cluster.get_center() - spectrum_arrow.get_center()
        width_factor = spectrum_arrow.get_length() / left_cluster.get_width()
        spectrum_arrow.target.stretch_to_fit_width(
            width_factor * spectrum_arrow.get_length()
        )
        spectrum_arrow.target.shift(-width_factor * diff_vec[0] * RIGHT)
        self.update_slide(
            subtitle="Recursion (Left Partition)",
            additional_animations=[MoveToTarget(spectrum_arrow)],
            notes="This results in two partitions.",
        )

        # slide: highlight steps (left partition)
        step_1.generate_target()
        step_1.target.scale(1.2)
        self.play(
            MoveToTarget(step_1), run_time=self.RUN_TIME, rate_func=there_and_back
        )
        step_2.generate_target()
        step_2.target.scale(1.2)
        self.play(
            MoveToTarget(step_2), run_time=self.RUN_TIME, rate_func=there_and_back
        )
        step_3.generate_target()
        step_3.target.scale(1.2)
        self.play(
            MoveToTarget(step_3), run_time=self.RUN_TIME, rate_func=there_and_back
        )
        self.update_slide(
            subtitle="Recursion (Left Partition)",
            notes="This results in two partitions.",
        )

        # slide: zoom in on right partition
        spectrum_arrow.generate_target()
        diff_vec = right_cluster.get_center() - spectrum_arrow.get_center()
        width_factor_arrow_length = arrow_length / spectrum_arrow.get_width()
        width_factor_right_cluster = (
            spectrum_arrow.get_width() / right_cluster.get_width()
        )
        width_factor = width_factor_arrow_length * width_factor_right_cluster
        spectrum_arrow.target.stretch_to_fit_width(
            width_factor * spectrum_arrow.get_length()
        )
        new_location_right_cluster = (
            spectrum_arrow.get_center() + width_factor * diff_vec
        )
        spectrum_arrow.target.shift(new_location_right_cluster[0] * LEFT)
        self.update_slide(
            subtitle="Recursion (Right Partition)",
            additional_animations=[
                MoveToTarget(spectrum_arrow),
            ],
            notes="This results in two partitions.",
        )

        # slide: highlight steps (right partition)
        step_1.generate_target()
        step_1.target.scale(1.2)
        self.play(
            MoveToTarget(step_1), run_time=self.RUN_TIME, rate_func=there_and_back
        )
        step_2.generate_target()
        step_2.target.scale(1.2)
        self.play(
            MoveToTarget(step_2), run_time=self.RUN_TIME, rate_func=there_and_back
        )
        step_3.generate_target()
        step_3.target.scale(1.2)
        self.play(
            MoveToTarget(step_3), run_time=self.RUN_TIME, rate_func=there_and_back
        )
        self.update_slide(
            subtitle="Recursion (Right Partition)",
            notes="This results in two partitions.",
        )

        # slide: final step
        step_4 = TexText(
            r"Step.4 Convert set of split indices $\{k^*_i\}_{i=1}^{N_{\text{cluster}}-1}$ to cluster boundaries $\{(a_i, b_i)\}_{i=1}^{N_{\text{cluster}}}$ \& feed into $m_{N_{\text{cluster}}}$.",
            font_size=CONTENT_FONT_SIZE,
            t2c={
                r"cluster boundaries": CustomColors.RED.value,
                r"m_{N_{\text{cluster}}}": CustomColors.GOLD.value,
            },
        ).next_to(step_3, DOWN, buff=0.5)
        brace_left.clear_updaters()
        label_left.clear_updaters()
        brace_right.clear_updaters()
        label_right.clear_updaters()
        self.update_slide(
            subtitle="Feeding Clusters into CG Bound",
            additional_animations=[
                FadeOut(spectrum_for_partitioning),
                FadeOut(brace_left),
                FadeOut(label_left),
                FadeOut(brace_right),
                FadeOut(label_right),
                FadeOut(lambda_1_label),
                FadeOut(lambda_k_label),
                FadeOut(lambda_kp1_label),
                FadeOut(lambda_n_label),
                Write(step_4),
            ],
        )

        # slide: refine research question
        self.slide_contents = [
            step_1,
            step_2,
            step_3,
            step_4,
        ]
        self.main_question.move_to(ORIGIN).shift(2*UP)
        self.sub_questions.generate_target()
        self.sub_questions.target.set_opacity(1)
        self.update_slide(
            notes="So we need to refine our research question further.",
            title="Research Question (Revised)",
            subtitle="Predicting Preconditioned CG Performance",
            new_contents=[self.main_question],
            additional_animations=[
                MoveToTarget(self.sub_questions)
            ],
            transition_time=2 * self.RUN_TIME,
        )

        self.slide_contents = [self.main_question]
        self.sub_questions.generate_target()
        self.sub_questions.target.set_opacity(0)
        self.play(MoveToTarget(self.sub_questions), run_time=self.RUN_TIME)

    def generate_clustered_spectrum(
        self,
        clusters: list[tuple[float, float]],
        cluster_labels: list[str],
        arrow_length: float = 7.0,
        resolution: int = 150,
        dot_size: float = 0.7 * DEFAULT_DOT_RADIUS,
        bar_buff: float = 0.01,
        randomize: bool = False,
        return_eigs: bool = False,
    ) -> VGroup:
        arrow = Arrow(
            start=ORIGIN,
            end=arrow_length * RIGHT,
            buff=0.0,
            color=WHITE,
        ).move_to(ORIGIN)
        always(arrow.set_color, WHITE)

        def arrow_updater(arrow: Mobject):
            arrow.buff = 0.5
            arrow.stroke_width = 1
            return arrow

        arrow.add_updater(lambda m: arrow_updater(m))
        cluster_eigs = []

        def eig_dot_updater(mobj: Mobject, eig):
            mobj.set_fill(CustomColors.SKY.value)
            mobj.scale(1)
            mobj.move_to(arrow.get_start() + [eig * arrow.get_length(), 0, 0])
            return mobj

        all_eigs = []
        for cluster in clusters:
            if not randomize:
                num_eigs = int(resolution * (cluster[1] - cluster[0]))
                eigs = np.linspace(cluster[0], cluster[1], num_eigs)
            else:
                num_eigs = int(
                    resolution * (cluster[1] - cluster[0]) * np.random.uniform(0.8, 1.2)
                )
                eigs = np.sort(np.random.uniform(cluster[0], cluster[1], size=num_eigs))
            all_eigs.extend(eigs)
            for eig in eigs:
                eig_dot = Dot(
                    arrow.get_start() + [eig * arrow.get_length(), 0, 0],
                    fill_color=CustomColors.SKY.value,
                    radius=dot_size,
                )
                eig_dot.add_updater(lambda mobj, eig=eig: eig_dot_updater(mobj, eig))
                cluster_eigs.append(eig_dot)
        cluster_bars = []

        def line_updater(line: Mobject, x: float):
            line.set_color(WHITE)
            line.put_start_and_end_on(
                arrow.get_start() + [x * arrow.get_length() - bar_buff, -0.1, 0],
                arrow.get_start() + [x * arrow.get_length() - bar_buff, 0.1, 0],
            )
            return line

        for x, y in clusters:
            line1 = Line(
                start=arrow.get_start() + [x * arrow.get_length() - bar_buff, -0.1, 0],
                end=arrow.get_start() + [x * arrow.get_length() - bar_buff, 0.1, 0],
                color=WHITE,
            )
            line1.add_updater(lambda mobj, x=x: line_updater(mobj, x))
            line2 = Line(
                start=arrow.get_start() + [y * arrow.get_length() + bar_buff, -0.1, 0],
                end=arrow.get_start() + [y * arrow.get_length() + bar_buff, 0.1, 0],
                color=WHITE,
            )
            line2.add_updater(lambda mobj, y=y: line_updater(mobj, y))
            cluster_bars.extend([line1, line2])
        cluster_bar_labels = []

        def label_updater(label: Mobject, bar: Mobject):
            label.next_to(bar, DOWN, buff=0.1)
            label.align_to(arrow.get_center() + 0.4 * DOWN, DOWN)
            return label

        for bar, label in zip(cluster_bars, cluster_labels):
            label = (
                Tex(label, font_size=CONTENT_FONT_SIZE)
                .next_to(bar, DOWN, buff=0.1)
                .align_to(arrow.get_center() + 0.4 * DOWN, DOWN)
            )
            label.add_updater(lambda mobj, bar=bar: label_updater(mobj, bar))
            cluster_bar_labels.append(label)
        lambda_label = Tex(
            r"\lambda",
            font_size=CONTENT_FONT_SIZE,
        ).next_to(arrow, RIGHT, buff=0.1)
        always(lambda_label.next_to, arrow, RIGHT, buff=0.1)
        out = VGroup(
            arrow,
            lambda_label,
            *cluster_eigs,
            *cluster_bars,
            *cluster_bar_labels,
        )
        if return_eigs:
            return out, all_eigs
        return out

    def generate_multiple_random_spectra(
        self,
        num_spectra,
        min_clusters: int = 2,
        max_clusters: int = 4,
        rng: np.random.Generator = np.random.default_rng(42),
        randomize: bool = False,
    ) -> tuple[list[VGroup], list[manimlib.Tex]]:
        spectra = []
        res_polys = []
        for _ in range(num_spectra):
            num_cluster_ends = rng.choice(
                [2 * i for i in range(min_clusters, max_clusters + 1)]
            )
            ends = np.sort(rng.random(size=num_cluster_ends))
            clusters = [(ends[i], ends[i + 1]) for i in range(0, len(ends), 2)]
            labels = [
                f"a_{i//2+1}" if i % 2 == 0 else f"b_{i//2+1}" for i in range(len(ends))
            ]
            spectra.append(
                self.generate_clustered_spectrum(clusters, labels, randomize=randomize)
            )
            res_polys.append(
                Tex(
                    f"r(\lambda) \\approx \prod_{{i=1}}^{{{len(clusters)}}} \hat{{C}}_{{p_i}}^{{[a_i,b_i]}}(\lambda) \in \mathcal{{P}}_m",
                    font_size=CONTENT_FONT_SIZE,
                ).next_to(spectra[-1], DOWN, buff=0.3)
            )
        return spectra, res_polys

    def multi_cluster_spectrum_updater(
        self, spectrum: VGroup, dt, update_frequency: float = 0.5
    ) -> VGroup:
        tolerance = 1 / FPS  # Allowable tolerance based on frame rate
        dt = self.time - self.now
        if abs((dt % update_frequency)) <= tolerance:
            num_cluster_ends = random.choice([4, 6, 8, 10, 12])
            ends = np.sort(np.random.sample(size=num_cluster_ends))
            clusters = [(ends[i], ends[i + 1]) for i in range(0, len(ends), 2)]
            labels = [
                f"a_{i//2+1}" if i % 2 == 0 else f"b_{i//2+1}" for i in range(len(ends))
            ]
            new_spectrum = self.generate_clustered_spectrum(clusters, labels)
            spectrum.clear()
            spectrum += new_spectrum
        return spectrum

    def level_5_results(self):
        # slide: coefficient functions
        image_height = (FRAME_HEIGHT - self.BOTTOM_MARGIN - self.TOP_MARGIN) * 0.7
        coefficient_functions = (
            ImageMobject("coefficient_functions")
            .set_height(image_height)
            .set_z_index(1)
        ).align_to(self.TOP_MARGIN, UP)
        rectangle_coefficient_functions = (
            BackgroundRectangle(
                coefficient_functions,
                color=CustomColors.BLUE.value,
                buff=0.1,
            )
            .set_z_index(-1)
            .round_corners(0.1)
            .set_opacity(1.0)
        )

        M_1 = TexText(
            r"$M_1 = M_{\text{2-OAS-GDSW}}$",
            font_size=1.0 * CONTENT_FONT_SIZE,
        )
        M_2 = TexText(
            r"$M_2 = M_{\text{2-OAS-RGDSW}}$",
            font_size=1.0 * CONTENT_FONT_SIZE,
        )
        M_3 = TexText(
            r"$M_3 = M_{\text{2-OAS-AMS}}$",
            font_size=1.0 * CONTENT_FONT_SIZE,
        )
        for i, M in enumerate([M_1, M_2, M_3]):
            M.next_to(rectangle_coefficient_functions, DOWN, buff=0.1).shift(
                (i - 1) * 3.0 * RIGHT
            )

        self.update_slide(
            "Absolute Performance",
            subtitle="Coefficient Functions \& Preconditioners",
            notes="Discuss results on absolute sharpness of the new bounds",
            additional_animations=[
                FadeIn(coefficient_functions),
                FadeIn(rectangle_coefficient_functions),
                Write(M_1),
                Write(M_2),
                Write(M_3),
            ],
            index="Sharpness \& Practical Use of New Bound",
        )

        # slide: absolute performance (GDSW)
        gdsw_results = (
            ImageMobject(
                "absolute_performance_2-OAS-GDSW.png",
                height=image_height,
            )
            .set_z_index(1)
            .align_to(self.TOP_MARGIN, UP)
        )
        rectangle_gdsw_results = (
            BackgroundRectangle(
                gdsw_results,
                buff=0.1,
                color=CustomColors.BLUE.value,
                fill_color=CustomColors.BLUE.value,
            )
            .round_corners(0.1)
            .set_opacity(1.0)
            .set_z_index(-1)
        )
        M_1.generate_target()
        M_1.target.next_to(rectangle_gdsw_results, DOWN, buff=0.1)
        M_2.generate_target()
        M_2.target.set_opacity(0.0)
        M_3.generate_target()
        M_3.target.set_opacity(0.0)
        self.update_slide(
            subtitle="New iteration bounds for $M_1 = M_{\\text{2-OAS-GDSW}}$",
            additional_animations=[
                FadeOut(coefficient_functions),
                FadeOut(rectangle_coefficient_functions),
                FadeIn(gdsw_results),
                FadeIn(rectangle_gdsw_results),
                MoveToTarget(M_1),
                MoveToTarget(M_2),
                MoveToTarget(M_3),
            ],
        )

        # slide: absolute performance (RGDSW)
        rgdsw_results = (
            ImageMobject(
                "absolute_performance_2-OAS-RGDSW.png",
                height=image_height,
            )
            .set_z_index(1)
            .align_to(self.TOP_MARGIN, UP)
        )
        rectangle_rgdsw_results = (
            BackgroundRectangle(
                rgdsw_results,
                buff=0.1,
                color=CustomColors.BLUE.value,
                fill_color=CustomColors.BLUE.value,
            )
            .round_corners(0.1)
            .set_opacity(1.0)
            .set_z_index(-1)
        )
        M_1.generate_target()
        M_1.target.set_opacity(0.0)
        M_2.generate_target()
        M_2.target.next_to(rectangle_rgdsw_results, DOWN, buff=0.1)
        M_2.target.set_opacity(1.0)
        self.update_slide(
            subtitle="New iteration bounds for $M_2 = M_{\\text{2-OAS-RGDSW}}$",
            additional_animations=[
                FadeOut(gdsw_results),
                FadeOut(rectangle_gdsw_results),
                FadeIn(rgdsw_results),
                FadeIn(rectangle_rgdsw_results),
                MoveToTarget(M_1),
                MoveToTarget(M_2),
                MoveToTarget(M_3),
            ],
        )
        # slide: absolute performance (AMS)
        ams_results = (
            ImageMobject(
                "absolute_performance_2-OAS-AMS.png",
                height=image_height,
            )
            .set_z_index(1)
            .align_to(self.TOP_MARGIN, UP)
        )
        rectangle_ams_results = (
            BackgroundRectangle(
                ams_results,
                buff=0.1,
                color=CustomColors.BLUE.value,
                fill_color=CustomColors.BLUE.value,
            )
            .round_corners(0.1)
            .set_opacity(1.0)
            .set_z_index(-1)
        )
        M_2.generate_target()
        M_2.target.set_opacity(0.0)
        M_3.generate_target()
        M_3.target.next_to(rectangle_ams_results, DOWN, buff=0.1)
        M_3.target.set_opacity(1.0)
        self.update_slide(
            subtitle="New iteration bounds for $M_3 = M_{\\text{2-OAS-AMS}}$",
            additional_animations=[
                FadeOut(rgdsw_results),
                FadeOut(rectangle_rgdsw_results),
                FadeIn(ams_results),
                FadeIn(rectangle_ams_results),
                MoveToTarget(M_1),
                MoveToTarget(M_2),
                MoveToTarget(M_3),
            ],
        )

        # slide: partitioning output
        spectra_width = 0.3 * FRAME_WIDTH
        buffer_width = 2
        shift_distance = 0.5 * (spectra_width + buffer_width)
        partitioning_output = self.generate_partitioning_output(spectra_width)
        coef_1 = TexText(
            r"$\mathcal{C}_{\text{3layer, vert}}$",
            font_size=CONTENT_FONT_SIZE,
        ).shift(shift_distance * LEFT + 1.5 * UP)
        coef_2 = TexText(
            r"$\mathcal{C}_{\text{edge slabs, around vertices}}$",
            font_size=CONTENT_FONT_SIZE,
        ).move_to(shift_distance * RIGHT + 1.5 * UP)

        all_spectra = {i: VGroup() for i in range(len(MESHES))}
        labels = [
            Tex(f"\sigma(M_{i+1}^{{-1}}A)", font_size=CONTENT_FONT_SIZE)
            for i in range(3)
        ]
        for i, _ in enumerate(MESHES):
            for j, coef_func in enumerate([coef_1, coef_2]):
                for k, spectrum in enumerate(partitioning_output[i][j]):
                    offset = (0.5 + k) * DOWN
                    spectrum.align_to(coef_func, UP)
                    spectrum.shift(offset).shift(
                        shift_distance * (LEFT if coef_func == coef_1 else RIGHT)
                    )
                    if j == 0:
                        labels[k].next_to(spectrum, LEFT, buff=0.1)
                    all_spectra[i].add(spectrum)
        M_2.generate_target()
        M_2.target.set_opacity(1.0)
        M_2.target.next_to(all_spectra[0], DOWN, buff=0.5)
        M_1.generate_target()
        M_1.target.set_opacity(1.0)
        M_1.target.next_to(M_2.target, LEFT, buff=1.0)
        M_3.generate_target()
        M_3.target.set_opacity(1.0)
        M_3.target.next_to(M_2.target, RIGHT, buff=1.0)
        self.update_slide(
            subtitle="Partitioning Output",
            additional_animations=[
                FadeOut(ams_results),
                FadeOut(rectangle_ams_results),
                MoveToTarget(M_1),
                MoveToTarget(M_2),
                MoveToTarget(M_3),
                Write(coef_1),
                Write(coef_2),
                Write(all_spectra[len(MESHES) - 1]),  # show only first mesh initially
                *[Write(label) for label in labels],
            ],
            transition_time=2.0 * self.RUN_TIME,
        )

        # TODO: takes long time to render! Works great tho
        # slide: show other meshes
        super().next_slide(notes="Show other meshes", loop=True)
        for spectra in all_spectra.values():
            spectra.suspend_updating(True)
            spectra.set_opacity(0.0, recurse=True)
        all_spectra[0].set_opacity(1.0)
        for i in [i + 1 for i in range(len(MESHES) - 1)] + [0]:
            all_spectra[i].generate_target()
            all_spectra[i].target.set_opacity(1.0)
            for j in range(len(MESHES)):
                if j != i:
                    all_spectra[j].generate_target()
                    all_spectra[j].target.set_opacity(0.0)
            self.play(
                *[MoveToTarget(m) for m in all_spectra.values()],
                run_time=2.0 * self.RUN_TIME,
            )
        super().next_slide()

        # # TODO: same thing here, takes long time to render
        # # slide: show biggest mesh
        # all_spectra[len(MESHES) - 1].generate_target()
        # all_spectra[len(MESHES) - 1].target.set_opacity(1.0)
        # for i in range(len(MESHES)-1):
        #     all_spectra[i].generate_target()
        #     all_spectra[i].target.set_opacity(0.0)
        #     # all_spectra[i].set_opacity(0.0, recurse=True)
        # self.update_slide(
        #     subtitle="Partitioning Output: Largest Problem Size",
        #     additional_animations=[
        #         ,
        #     ],
        #     notes="Show the largest mesh",
        #     transition_time=2.0 * self.RUN_TIME,
        # )

        # slide:

        # # mimick end of above TODO
        # for spectra in all_spectra.values():
        #     spectra.suspend_updating(True)
        # all_spectra[len(MESHES) - 1].set_opacity(1.0)

        # self.update_slide(
        #     subtitle="Early Estimation of New Bound for $M_{\\text{2-OAS-GDSW}}$",
        #     additional_animations=[
        #         MoveToTarget(gdsw_spectrum[0]),
        #         *[MoveToTarget(m) for m in remaining_spectra],
        #         FadeOut(coef_1, remover=False),
        #         FadeOut(coef_2, remover=False),
        #         FadeOut(M_1, remover=False),
        #         FadeOut(M_2, remover=False),
        #         FadeOut(M_3, remover=False),
        #         *[FadeOut(label, remover=False) for label in labels],
        #         ShowCreation(axes),
        #         ShowCreation(x_label),
        #         ShowCreation(y_label),
        #     ],
        #     notes="Show the largest mesh",
        #     transition_time=2.0 * self.RUN_TIME,
        # )
        # slide: ritz migration (AMS)
        self.slide_contents = (
            [
                coef_1,
                coef_2,
            ]
            + [f for f in all_spectra.values()]
            + labels
        )
        ams_ritz_migration = (
            ImageMobject(
                "bound_and_spectrum_vs_iterations_2-OAS-AMS.png",
                height=image_height,
            )
            .set_z_index(1)
            .align_to(self.TOP_MARGIN, UP)
        )
        rectangle_ams_ritz_migration = (
            BackgroundRectangle(
                ams_ritz_migration,
                buff=0.1,
                color=CustomColors.BLUE.value,
                fill_color=CustomColors.BLUE.value,
            )
            .round_corners(0.1)
            .set_opacity(1.0)
            .set_z_index(-1)
        )
        M_1.generate_target()
        M_1.target.set_opacity(0.0)
        M_2.generate_target()
        M_2.target.set_opacity(0.0)
        M_3.generate_target()
        M_3.target.next_to(rectangle_ams_results, DOWN, buff=0.1)
        M_3.target.set_opacity(1.0)
        self.update_slide(
            subtitle="Early Estimation of New Bound for $M_{\\text{2-OAS-AMS}}$",
            new_contents=[
                ams_ritz_migration,
                rectangle_ams_ritz_migration,
            ],
            additional_animations=[
                MoveToTarget(M_1),
                MoveToTarget(M_2),
                MoveToTarget(M_3),
            ],
        )

        # slide: ritz migration (RGDSW)
        self.slide_contents = [
            ams_ritz_migration,
            rectangle_ams_ritz_migration,
        ]
        rgdsw_ritz_migration = (
            ImageMobject(
                "bound_and_spectrum_vs_iterations_2-OAS-RGDSW.png",
                height=image_height,
            )
            .set_z_index(1)
            .align_to(self.TOP_MARGIN, UP)
        )
        rectangle_rgdsw_ritz_migration = (
            BackgroundRectangle(
                rgdsw_ritz_migration,
                buff=0.1,
                color=CustomColors.BLUE.value,
                fill_color=CustomColors.BLUE.value,
            )
            .round_corners(0.1)
            .set_opacity(1.0)
            .set_z_index(-1)
        )
        M_2.generate_target()
        M_2.target.set_opacity(1.0)
        M_2.target.next_to(rectangle_rgdsw_results, DOWN, buff=0.1)
        M_3.generate_target()
        M_3.target.set_opacity(0.0)
        self.update_slide(
            subtitle="Early Estimation of New Bound for $M_{\\text{2-OAS-RGDSW}}$",
            new_contents=[
                rgdsw_ritz_migration,
                rectangle_rgdsw_ritz_migration,
            ],
            additional_animations=[
                MoveToTarget(M_1),
                MoveToTarget(M_2),
                MoveToTarget(M_3),
            ],
        )

        self.slide_contents = [
            rgdsw_ritz_migration,
            rectangle_rgdsw_ritz_migration,
            M_2,
        ]

    def generate_partitioning_output(self, spectra_width: float) -> dict:
        out = {}
        for i, mesh_params in enumerate(MESHES):
            out[i] = {}
            for j, coef_func in enumerate(COEF_FUNCS):
                min_eig = float("inf")
                max_eig = float("-inf")
                spectra_list = []
                cluster_indices_list = []
                for preconditioner_cls, coarse_space_cls in PRECONDITIONERS:
                    # Load the eigenvalues from the saved numpy file
                    fp = get_spectrum_save_path(
                        mesh_params, coef_func, preconditioner_cls, coarse_space_cls
                    )
                    if fp.exists():
                        eigenvalues = np.load(fp)["eigenvalues"]
                        spectra_list.append(eigenvalues)

                        # store the split indices for the sharpened bound (except for the last one)
                        cluster_indices = []
                        split_idxs = partition_eigenspectrum(eigenvalues)
                        start = 0
                        for end in split_idxs:
                            if start == end:
                                # cluster_indices.append((start, start))
                                pass
                            else:
                                cluster_indices.append((start, end))
                            start = end + 1
                        cluster_indices_list.append(cluster_indices)

                        # keep track of the min and max eigenvalues across all spectra for consistent x_range
                        _max_eig = np.max(eigenvalues)
                        _min_eig = np.min(eigenvalues)
                        max_eig = max(max_eig, _max_eig)
                        min_eig = min(min_eig, _min_eig)
                    else:
                        # Provide a clickable link to the script in the repo using Rich markup with absolute path
                        approx_path = Path(__file__).parent / "approximate_spectra.py"
                        LOGGER.error(
                            f"File %s does not exist. Run '[link=file:{approx_path}]approximate_spectra.py[/link]' first.",
                            fp,
                        )
                        exit()
                out[i][j] = [
                    self.generate_spectrum(
                        eigs,
                        cluster_indices,
                        x_range=(min_eig, max_eig),
                        arrow_length=spectra_width,
                    )
                    for eigs, cluster_indices in zip(spectra_list, cluster_indices_list)
                ]
        return out

    def generate_spectrum(
        self,
        eigs: np.ndarray,
        cluster_indices: list[tuple[int, int]],
        arrow_length: float = 7.0,
        dot_size: float = 0.7 * DEFAULT_DOT_RADIUS,
        bar_buff: float = 0.01,
        edge_buffer: float = 0.1,
        x_range: tuple[float, float] = None,
    ) -> VGroup:
        arrow = Arrow(
            start=ORIGIN,
            end=arrow_length * RIGHT,
            buff=0.0,
            color=WHITE,
        ).move_to(ORIGIN)
        always(arrow.set_color, WHITE)

        def arrow_updater(arrow: Mobject):
            arrow.buff = 0.5
            arrow.stroke_width = 1
            return arrow

        arrow.add_updater(lambda m: arrow_updater(m))

        def buffer():
            return [arrow.get_length() * edge_buffer, 0, 0]

        # eig dots
        def eig_dot_updater(mobj: Mobject, eig):
            mobj.set_fill(CustomColors.SKY.value)
            mobj.scale(1)
            mobj.move_to(
                arrow.get_start()
                + buffer()
                + [eig * (arrow.get_length() * (1 - 2 * edge_buffer)), 0, 0]
            )
            return mobj

        eig_dots = []
        log_eigs = np.log10(eigs)
        if x_range is not None:
            min_eig, max_eig = x_range
            log_min_eig = np.log10(min_eig)
            log_max_eig = np.log10(max_eig)
        else:
            log_max_eig = np.max(log_eigs)
            log_min_eig = np.min(log_eigs)

        normalized_eigs = (log_eigs - log_min_eig) / (log_max_eig - log_min_eig)
        for eig in normalized_eigs:
            eig_dot = Dot(
                arrow.get_start()
                + buffer()
                + [eig * arrow.get_length() * (1 - 2 * edge_buffer), 0, 0],
                fill_color=CustomColors.SKY.value,
                radius=dot_size,
            )
            eig_dot.add_updater(lambda mobj, eig=eig: eig_dot_updater(mobj, eig))
            eig_dots.append(eig_dot)

        # cluster bars
        def line1_updater(line: Mobject, x: float):
            line.set_color(WHITE)
            line.put_start_and_end_on(
                arrow.get_start()
                + buffer()
                + [
                    x * (arrow.get_length() * (1 - 2 * edge_buffer))
                    - bar_buff * arrow.get_length(),
                    -0.1,
                    0,
                ],
                arrow.get_start()
                + buffer()
                + [
                    x * (arrow.get_length() * (1 - 2 * edge_buffer))
                    - bar_buff * arrow.get_length(),
                    0.1,
                    0,
                ],
            )
            return line

        def line2_updater(line: Mobject, y: float):
            line.set_color(WHITE)
            line.put_start_and_end_on(
                arrow.get_start()
                + buffer()
                + [
                    y * (arrow.get_length() * (1 - 2 * edge_buffer))
                    + bar_buff * arrow.get_length(),
                    -0.1,
                    0,
                ],
                arrow.get_start()
                + buffer()
                + [
                    y * (arrow.get_length() * (1 - 2 * edge_buffer))
                    + bar_buff * arrow.get_length(),
                    0.1,
                    0,
                ],
            )
            return line

        cluster_bars = []
        for i1, i2 in cluster_indices:
            x, y = normalized_eigs[i1], normalized_eigs[i2]
            line1 = Line(
                start=arrow.get_start()
                + buffer()
                + [
                    x * (arrow.get_length() * (1 - 2 * edge_buffer))
                    - bar_buff * arrow.get_length(),
                    -0.1,
                    0,
                ],
                end=arrow.get_start()
                + buffer()
                + [
                    x * (arrow.get_length() * (1 - 2 * edge_buffer))
                    - bar_buff * arrow.get_length(),
                    0.1,
                    0,
                ],
                color=WHITE,
            )
            line1.add_updater(lambda mobj, x=x: line1_updater(mobj, x))
            line2 = Line(
                start=arrow.get_start()
                + buffer()
                + [
                    y * (arrow.get_length() * (1 - 2 * edge_buffer))
                    + bar_buff * arrow.get_length(),
                    -0.1,
                    0,
                ],
                end=arrow.get_start()
                + buffer()
                + [
                    y * (arrow.get_length() * (1 - 2 * edge_buffer))
                    + bar_buff * arrow.get_length(),
                    0.1,
                    0,
                ],
                color=WHITE,
            )
            line2.add_updater(lambda mobj, y=y: line2_updater(mobj, y))
            cluster_bars.extend([line1, line2])

        # cluster bar labels
        def label_updater(label: Mobject, bar: Mobject):
            label.next_to(bar, DOWN, buff=0.1)
            label.align_to(arrow.get_center() + 0.4 * DOWN, DOWN)
            return label

        def left_label_updater(label: Mobject, bar: Mobject):
            label.align_to(bar.get_center(), RIGHT)
            return label

        def right_label_updater(label: Mobject, bar: Mobject):
            label.align_to(bar.get_center(), LEFT)
            return label

        cluster_bar_labels = []
        cluster_labels = [
            f"a_{i//2+1}" if i % 2 == 0 else f"b_{i//2+1}"
            for i in range(len(cluster_indices) * 2)
        ]
        for i, (bar, label) in enumerate(zip(cluster_bars, cluster_labels)):
            label = (
                Tex(label, font_size=CONTENT_FONT_SIZE)
                .next_to(bar, DOWN, buff=0.1)
                .align_to(arrow.get_center() + 0.4 * DOWN, DOWN)
            )
            label.add_updater(lambda mobj, bar=bar: label_updater(mobj, bar))
            if i % 2 == 0:
                label.add_updater(lambda mobj, bar=bar: left_label_updater(mobj, bar))
            else:
                label.add_updater(lambda mobj, bar=bar: right_label_updater(mobj, bar))
            cluster_bar_labels.append(label)

        def format_cond(c):
            if np.isnan(c):
                return "n/a"
            mantissa, exp = f"{c:.1e}".split("e")
            exp = int(exp)
            if exp == 0:
                return mantissa
            return rf"{mantissa} \times 10^{{{exp}}}"

        condition_number = Tex(
            f"\kappa = {format_cond(np.max(eigs) / np.min(eigs))}",
            font_size=CONTENT_FONT_SIZE,
        ).next_to(arrow, RIGHT, buff=0.1)
        always(condition_number.next_to, arrow, RIGHT, buff=0.1)
        return VGroup(
            arrow,
            condition_number,
            *eig_dots,
            *cluster_bars,
            *cluster_bar_labels,
        )

    def ritz_value_animation(self, mesh_params, coef_func, preconditioner):
        fp = get_spectrum_save_path(
            mesh_params, coef_func, preconditioner[0], preconditioner[1]
        )
        spectra = []
        cluster_list = []
        min_eig = float("inf")
        max_eig = float("-inf")
        if fp.exists():
            # Load the alpha and beta arrays from the saved numpy file
            array_zip = np.load(fp)
            alpha = array_zip["alpha"]
            beta = array_zip["beta"]

            # loop over CG iterations
            num_iterations = min(N_ITERATIONS, len(alpha) - 1)
            niters_multi_cluster = np.zeros(num_iterations, dtype=int)
            for k in range(num_iterations):
                # compute Ritz values
                lanczos_matrix = CustomCG.get_lanczos_matrix_from_coefficients(
                    alpha[: k + 1], beta[:k]
                )
                eigenvalues = eigs(lanczos_matrix)
                spectra.append(eigenvalues)

                # get max and min eigenvalues for consistent x_range
                _max_eig = np.max(eigenvalues)
                _min_eig = np.min(eigenvalues)
                max_eig = max(max_eig, _max_eig)
                min_eig = min(min_eig, _min_eig)

                # calculate sharpened bound
                niter_multi_cluster = multi_cluster_cg_iteration_bound(
                    eigenvalues, log_rtol=LOG_RTOL, exact_convergence=False
                )
                niters_multi_cluster[k] = niter_multi_cluster

                # get clusters
                split_idxs = partition_eigenspectrum(eigenvalues)
                start = 0
                cluster_indices = []
                for end in split_idxs:
                    if start == end:
                        # cluster_indices.append((start, start))
                        pass
                    else:
                        cluster_indices.append((start, end))
                    start = end + 1
                cluster_list.append(cluster_indices)
        else:
            # Provide a clickable link to the script in the repo using Rich markup with absolute path
            approx_path = Path(__file__).parent / "approximate_spectra.py"
            LOGGER.error(
                f"File %s does not exist. Run '[link=file:{approx_path}]approximate_spectra.py[/link]' first.",
                fp,
            )
            exit()

        # generate spectrum + bound
        spectrum_mobjs = []
        for spectrum, clusters in zip(spectra, cluster_list):
            spectrum_mobjs.append(
                self.generate_spectrum(
                    spectrum,
                    clusters,
                    x_range=(min_eig, max_eig),
                )
            )
        return spectrum_mobjs, niters_multi_cluster

    def level_6_conclusion(self):
        cover = ImageMobject("presentation_cover", height=FRAME_HEIGHT).set_z_index(-2)
        rectangle = Rectangle(
            width=FRAME_WIDTH,
            height=FRAME_HEIGHT,
            fill_color=BLACK,
            fill_opacity=0.65,
            stroke_width=0,
        ).set_z_index(-1)
        self.update_slide(
            "Conclusion",
            notes="Summarize the main points and contributions of the work",
            new_contents=[cover, rectangle],
            index="Conclusion",
        )
        item = Item()
        bullets = VGroup()
        bullet_1 = (
            defense.paragraph(
                f"{item}. Classical CG iteration bound $m_1$ is too coarse for high-contrast problems",
                font_size=CONTENT_FONT_SIZE,
                t2c={"high-contrast": CustomColors.RED.value},
                width=0.8 * FRAME_WIDTH,
            )
            .align_to(self.TOP_MARGIN, UP)
            .shift(1.0 * DOWN)
        )
        bullets.add(bullet_1)
        background_rect_1 = (
            BackgroundRectangle(
                bullet_1,
                color=CustomColors.BLUE.value,
                buff=0.2,
                fill_opacity=1.0,
            )
            .round_corners(0.1)
            .set_z_index(-1)
        )
        self.update_slide(
            additional_animations=[Write(bullet_1), FadeIn(background_rect_1)],
            transition_time=self.RUN_TIME,
        )

        # slide: 2
        bullet_2 = defense.paragraph(
            f"{item}. We developed $m_{{N_{{\\text{{cluster}}}}}}$ building on Axelsson's $m_2$",
            font_size=CONTENT_FONT_SIZE,
            width=0.3 * FRAME_WIDTH,
            t2c={
                "$m_{{N_{{\\text{{cluster}}}}}}$": CustomColors.GOLD.value,
                "$m_2$": CustomColors.SKY.value,
            },
            alignment=ALIGN.LEFT,
        ).next_to(bullet_1, DOWN, buff=0.5)
        bullets.add(bullet_2)
        background_rect_2 = (
            BackgroundRectangle(
                bullet_2,
                color=CustomColors.BLUE.value,
                buff=0.2,
                fill_opacity=1.0,
            )
            .round_corners(0.1)
            .set_z_index(-1)
        )
        self.update_slide(
            additional_animations=[Write(bullet_2), FadeIn(background_rect_2)],
            transition_time=self.RUN_TIME,
        )

        # slide: 3
        bullet_3 = defense.paragraph(
            f"{item}. Relative improvement of 10x-1000x (on fully developed spectra)",
            font_size=CONTENT_FONT_SIZE,
            width=0.3 * FRAME_WIDTH,
            t2c={"10x-1000x": CustomColors.GOLD.value},
            alignment=ALIGN.LEFT,
        ).next_to(bullet_2, DOWN, buff=0.5)
        bullets.add(bullet_3)
        background_rect_3 = (
            BackgroundRectangle(
                bullet_3,
                color=CustomColors.BLUE.value,
                buff=0.2,
                fill_opacity=1.0,
            )
            .round_corners(0.1)
            .set_z_index(-1)
        )
        self.update_slide(
            additional_animations=[Write(bullet_3), FadeIn(background_rect_3)],
            transition_time=self.RUN_TIME,
        )

        # slide: 4
        bullet_4 = defense.paragraph(
            f"{item}. Early PCG-based estimation of bound achieved, though hampered by slowly converging Ritz values",
            font_size=CONTENT_FONT_SIZE,
            width=0.3 * FRAME_WIDTH,
            t2c={"slowly converging Ritz values": CustomColors.RED.value},
            alignment=ALIGN.LEFT,
        ).next_to(bullet_3, DOWN, buff=0.5)
        bullets.add(bullet_4)
        background_rect_4 = (
            BackgroundRectangle(
                bullet_4,
                color=CustomColors.BLUE.value,
                buff=0.2,
                fill_opacity=1.0,
            )
            .round_corners(0.1)
            .set_z_index(-1)
        )
        self.update_slide(
            additional_animations=[Write(bullet_4), FadeIn(background_rect_4)],
            transition_time=self.RUN_TIME,
        )

        # slide: 5
        bullet_5 = defense.paragraph(
            f"{item}. Future work: refine cluster partitioning, improve a priori spectral estimation, and apply bounds to broader problem classes",
            font_size=CONTENT_FONT_SIZE,
            t2c={
                "refine": CustomColors.GOLD.value,
                "a priori spectral estimation": CustomColors.RED.value,
                "apply bounds to broader problem classes": CustomColors.SKY.value,
            },
            width=0.3 * FRAME_WIDTH,
            alignment=ALIGN.LEFT,
        ).next_to(bullet_4, DOWN, buff=0.5)
        bullets.add(bullet_5)
        background_rect_5 = (
            BackgroundRectangle(
                bullet_5,
                color=CustomColors.BLUE.value,
                buff=0.2,
                fill_opacity=1.0,
            )
            .round_corners(0.1)
            .set_z_index(-1)
        )
        self.update_slide(
            additional_animations=[Write(bullet_5), FadeIn(background_rect_5)],
            transition_time=self.RUN_TIME,
        )

        self.slide_contents = [
            bullets,
            background_rect_1,
            background_rect_2,
            background_rect_3,
            background_rect_4,
            background_rect_5, 
        ]

    def backup(self):
        # slide: zoom-in on GDSW + edge slabs spectrum + axes
        # gdsw_spectrum = all_spectra[len(MESHES) - 1][0]
        # gdsw_spectrum.resume_updating()
        # remaining_spectra = all_spectra[len(MESHES) - 1][1:]
        # gdsw_original_loc = gdsw_spectrum.get_center()
        # gdsw_spectrum[0].generate_target()
        # gdsw_spectrum[0].target.stretch_to_fit_width(0.6 * FRAME_WIDTH)
        # gdsw_spectrum[0].target.move_to(ORIGIN)
        # gdsw_spectrum[0].target.shift(2 * DOWN)
        # for spectrum in remaining_spectra:
        #     # spectrum.generate_target()
        #     # spectrum.target.set_opacity(0.0)
        #     spectrum.set_opacity(0.0)

        # slide: ritz value migration animation
        ritz_spectra, bounds = self.ritz_value_animation(
            DefaultQuadMeshParams.Nc64, COEF_FUNCS[1], PRECONDITIONERS[0]
        )
        sim_time = 8.0 * self.RUN_TIME
        update_freq = sim_time / N_ITERATIONS
        ritz_update_freq = 1
        all_dots = VGroup()
        for ritz_spectrum in ritz_spectra:
            ritz_spectrum.set_opacity(0.0)
            ritz_spectrum[0].shift(2 * DOWN)
            ritz_spectrum[0].stretch_to_fit_width(0.6 * FRAME_WIDTH)
            ritz_spectrum.generate_target()
            ritz_spectrum.target.set_opacity(1.0)
        axes = Axes(
            x_range=(0, N_ITERATIONS, N_ITERATIONS // 10),
            y_range=(0, 4, 1),
            height=0.4 * FRAME_HEIGHT,
            width=0.7 * FRAME_WIDTH,
            axis_config={
                "stroke_color": CustomColors.RED.value,
                "stroke_width": 2,
                "include_tip": True,
            },
        ).next_to(ritz_spectra[0], UP, buff=0.5)
        x_label = TexText(r"Iteration $i$", font_size=CONTENT_FONT_SIZE).next_to(
            axes, DOWN, buff=0.5
        )
        y_label = (
            Tex(r"m_{N_{\text{cluster}}}", font_size=CONTENT_FONT_SIZE)
            .rotate(90 * DEGREES)
            .next_to(axes, LEFT, buff=0.5)
        )

        axes.get_x_axis().add_numbers(
            x_values=[i for i in range(0, N_ITERATIONS + 1, N_ITERATIONS // 10)],
            font_size=0.7 * CONTENT_FONT_SIZE,
        )
        for i in range(5):
            axes.get_y_axis().add(
                Tex(f"10^{i}", font_size=CONTENT_FONT_SIZE).move_to(
                    axes.c2p(0, i, 0) + 0.4 * LEFT
                )
            )
        self.update_slide(
            "Backup Slides",
            notes="",
            additional_animations=[Write(axes), Write(x_label), Write(y_label)],
        )
        super().next_slide(notes="Show ritz value animation", loop=True)
        self.play(MoveToTarget(ritz_spectrum), run_time=update_freq)
        for i in range(N_ITERATIONS):
            bound_dot = Dot(
                axes.c2p(i, np.log10(bounds[i]), 0),
                color=CustomColors.GOLD.value,
                radius=DEFAULT_DOT_RADIUS,
            )
            all_dots.add(bound_dot)
            self.play(
                FadeIn(bound_dot),
                run_time=update_freq,
                rate_func=linear,
            )
            if i % ritz_update_freq == 0 and i + ritz_update_freq < N_ITERATIONS:
                ritz_spectra[i].generate_target()
                ritz_spectra[i].target.set_opacity(0.0)
                self.play(
                    MoveToTarget(ritz_spectra[i]),
                    MoveToTarget(ritz_spectra[i + ritz_update_freq]),
                    run_time=update_freq,
                    rate_func=linear,
                )
        super().next_slide()

        # # slide: back to overview
        # gdsw_spectrum.resume_updating()
        # gdsw_spectrum[0].generate_target()
        # gdsw_spectrum[0].target.stretch_to_fit_width(spectra_width)
        # gdsw_spectrum[0].target.move_to(gdsw_original_loc)
        # gdsw_spectrum.target.set_opacity(1.0)
        # self.update_slide(
        #     subtitle="Partitioning Output",
        #     additional_animations=[
        #         MoveToTarget(gdsw_spectrum[0]),
        #         FadeOut(axes, remover=False),
        #         FadeOut(x_label, remover=False),
        #         FadeOut(y_label, remover=False),
        #         *[FadeOut(ritz_spectrum) for ritz_spectrum in ritz_spectra],
        #         FadeOut(all_dots),
        #         FadeIn(coef_1),
        #         FadeIn(coef_2),
        #         FadeIn(M_1),
        #         FadeIn(M_2),
        #         FadeIn(M_3),
        #         *[FadeIn(label) for label in labels],
        #     ],
        #     transition_time=2.0 * self.RUN_TIME,
        #     notes="Back to overview",
        # )

    def references(self):
        refs = list(CITED_REFERENCES.values())
        chunk_size = 6
        chunks = (len(refs) + chunk_size - 1) // chunk_size  # ceiling division
        for c in range(0, chunks):
            start = c * chunk_size
            end = min((c + 1) * chunk_size, len(refs))
            refs_text = [f"{ref}" for ref in refs[start:end]]
            refs_mobj = (
                defense.paragraph(
                    *refs_text,
                    font_size=FOOTNOTE_FONT_SIZE,
                    additional_preamble="\\usepackage{hyperref}",
                    alignment=ALIGN.LEFT,
                    width=0.6 * FRAME_WIDTH,
                )
                .next_to(self.TOP_MARGIN, DOWN, buff=0.5)
                .align_to(self.LEFT_MARGIN, LEFT)
            )
            self.update_slide(
                f"References ({start+1}-{end})",
                new_contents=[refs_mobj],
                notes="References",
            )
            self.slide_contents = [refs_mobj]

    # full construct
    def construct(self):
        # self.wait_time_between_slides = 0.10
        # self.title_slide()
        # self.level_0_opening()
        # self.toc()
        # self.level_1_intro_cg()
        # self.level_2_cg_convergence()
        # self.level_3_preconditioning()
        self.level_4_two_clusters()
        # self.level_5_results()
        # self.level_6_conclusion()
        # # self.backup()
        # self.references()
