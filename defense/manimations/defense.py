# NOTE the structure and functionality of this script and its main class `defense` are
# adopted from https://github.com/jeertmans/jeertmans.github.io/blob/main/_slides/2023-12-07-confirmation/main.py
# next to that of course the manim_slides documentation https://manim-slides.readthedocs.io/en/latest/
from enum import Enum

from manimlib import *

pass
# import cv2 # install for video playback
import os
from json import load

from manim_slides import Slide

os.environ["NO_HCMSFEM_CLI_ARGS"] = ""
from hcmsfem.plot_utils import CustomColors
from hcmsfem.root import get_venv_root
from hcmsfem.solvers import CustomCG, classic_cg_iteration_bound

# Manim render settings
FPS = 30


class QUALITY(Enum):
    FOUR_K = (3840, 2160)
    HD = (1920, 1080)
    P720 = (1280, 720)
    P480 = (854, 480)
    P360 = (640, 360)
    P240 = (426, 240)


manim_config.camera.fps = FPS
manim_config.camera.resolution = QUALITY.HD.value
manim_config.background_color = WHITE
manim_config.directories.raster_images = (get_venv_root() / "images").as_posix()
manim_config.camera.background_color = CustomColors.NAVY.value
SCENE_WIDTH_CM = FRAME_WIDTH * 2.54

# font settings
TITLE_FONT_SIZE = 48
CONTENT_FONT_SIZE = 0.6 * TITLE_FONT_SIZE
SOURCE_FONT_SIZE = 0.2 * TITLE_FONT_SIZE
FOOTNOTE_FONT_SIZE = 0.75 * CONTENT_FONT_SIZE


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
    Thesis committee: & Prof. H. Schuttelaars,                            & TU Delft, responsible supervisor \\
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

        # Mutable variables
        self.counter = 0
        self.slide_number = Integer(1).set_color(WHITE).to_corner(DR)
        self.slide_title = Text("Contents", font_size=TITLE_FONT_SIZE).to_corner(UL)
        self.slide_subtitle = Text(
            "Subcontents", font_size=0.5 * TITLE_FONT_SIZE
        ).next_to(self.slide_title, DOWN)
        self.slide_subtitle_visible = False

        # slide contents (everything except title, subtitle, slide number)
        self.slide_contents: list[Mobject] = []

        # TU delft logo
        self.tu_delft_logo = (
            ImageMobject("TUDelft_logo_white", height=0.1 * FRAME_HEIGHT)
            .to_corner(DL)
            .shift(0.5 * DOWN)
        )
        self.tu_delft_logo.set_z_index(10)  # make sure logo is always on top

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

    # utility functions
    def next_slide(self, **kwargs):
        slide_number_update, new_slide_number = self.update_slide_number()
        self.play(slide_number_update)
        self.slide_number = new_slide_number
        super().next_slide(**kwargs)

    def update_slide_number(self):
        self.counter += 1
        new_slide_number = TexText(f"{self.counter}").move_to(self.slide_number)
        slide_number_update = ReplacementTransform(self.slide_number, new_slide_number)
        return slide_number_update, new_slide_number

    def update_slide_titles(self, title, subtitle):
        title_animations = []

        # construct new title
        if title is not None:
            new_title = (
                TexText(title, font_size=TITLE_FONT_SIZE)
                .move_to(self.slide_title)
                .align_to(self.slide_title, LEFT)
            )
            title_animations.append(ReplacementTransform(self.slide_title, new_title))

        # check for new subtitle
        if subtitle is not None and self.slide_subtitle_visible:
            new_subtitle = (
                TexText(subtitle, font_size=0.5 * TITLE_FONT_SIZE)
                .move_to(self.slide_subtitle)
                .align_to(self.slide_title, LEFT)
            )
            title_animations.append(
                ReplacementTransform(self.slide_subtitle, new_subtitle)
            )
            self.slide_subtitle_visible = True
        elif subtitle is not None and not self.slide_subtitle_visible:
            new_subtitle = (
                TexText(subtitle, font_size=0.5 * TITLE_FONT_SIZE)
                .move_to(self.slide_subtitle)
                .align_to(self.slide_title, LEFT)
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
        )

    def update_slide(
        self,
        title=None,
        subtitle=None,
        new_contents: list[Mobject] = [],
        transition_time: float = 0.75,
        notes: str = "",
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
        # wipe_animation += [FadeOut(m) for m in self.slide_contents]

        # add new content to scene but out of view
        for m in new_contents:
            m.move_to(m.get_center() + np.array([FRAME_WIDTH, 0, 0]))

        # make animations for bringing new content in
        wipe_animation += [
            m.animate.move_to(m.get_center() - np.array([FRAME_WIDTH, 0, 0]))
            for m in new_contents
        ]

        # animate slide number change and optional content wipe
        title_update, new_title, new_subtitle = self.update_slide_titles(
            title, subtitle
        )
        slide_number_update, new_slide_number = self.update_slide_number()
        self.play(
            *title_update,
            slide_number_update,
            *wipe_animation,
            run_time=transition_time,
        )

        # update new_contents
        self.slide_contents = []
        self.slide_title = new_title
        if subtitle:
            self.slide_subtitle = new_subtitle
        self.slide_number = new_slide_number

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
        texts = VGroup(*texts).arrange(direction)

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
            fill_opacity=0.6,
            stroke_width=0,
        ).set_z_index(-1)
        self.play(FadeIn(cover), FadeIn(rectangle), run_time=0.5)

        # Title slide
        super().next_slide(notes="title slide")
        title = TexText(
            "Sharpened CG Iteration Bound for High-contrast Heterogeneous Scalar Elliptic PDEs",
            font_size=0.8 * TITLE_FONT_SIZE,
            t2c={"High-contrast": RED},
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
        ]
        super().next_slide()

    def level_0_opening(self):
        image_height = 0.6 * FRAME_HEIGHT
        shift_direction = 0.1 * DOWN + 1.5 * LEFT
        scale = 0.4

        # slide: pcb simulation
        pcb_img = ImageMobject("pcb_temp_sim", height=image_height)
        pcb_text = TexText(
            "PCB Temperature Simulation", font_size=FOOTNOTE_FONT_SIZE
        ).next_to(pcb_img, 0.5 * DOWN)
        pcb_cite = cite("pcb_simulation").next_to(pcb_text, RIGHT)
        self.update_slide(
            "Opening",
            new_contents=[pcb_img, pcb_text, pcb_cite],
            subtitle="Motivation",
            notes="What do Printed Circuit Board simulations...",
        )

        # slide: subsurface modelling
        pcb_img.generate_target()
        pcb_img.target.scale(scale)
        pcb_img.target.next_to(self.slide_subtitle, DOWN)
        pcb_img.target.align_to(self.slide_title, LEFT)
        self.play(
            MoveToTarget(pcb_img), FadeOut(pcb_text), FadeOut(pcb_cite), run_time=0.5
        )
        subsurface_img = ImageMobject("subsurface_modelling", height=image_height)
        subsurface_text = TexText(
            "Subsurface Modelling", font_size=FOOTNOTE_FONT_SIZE
        ).next_to(subsurface_img, 0.5 * DOWN)
        subsurface_cite = cite("subsurface_modelling").next_to(subsurface_text, RIGHT)
        self.update_slide_contents(
            [subsurface_img, subsurface_text, subsurface_cite],
            notes="subsurface modelling...",
        )

        # slide: eit
        subsurface_img.generate_target()
        subsurface_img.target.scale(scale)
        subsurface_img.target.next_to(pcb_img, RIGHT)
        subsurface_img.target.shift(shift_direction)
        self.play(
            MoveToTarget(subsurface_img),
            FadeOut(subsurface_text),
            FadeOut(subsurface_cite),
            run_time=0.5,
        )
        eit_img = ImageMobject("eit", height=image_height)
        eit_text = TexText(
            "Electrical Impedance Tomography", font_size=FOOTNOTE_FONT_SIZE
        ).next_to(eit_img, 0.5 * DOWN)
        eit_cite = cite("eit").next_to(eit_text, RIGHT)
        self.update_slide_contents(
            [eit_img, eit_text, eit_cite], notes="Electrical Impedance Tomography..."
        )

        # slide: composite materials
        eit_img.generate_target()
        eit_img.target.scale(scale)
        eit_img.target.next_to(subsurface_img, RIGHT)
        eit_img.target.shift(shift_direction)
        self.play(
            MoveToTarget(eit_img), FadeOut(eit_text), FadeOut(eit_cite), run_time=0.5
        )
        composite_img = ImageMobject("composite_material", height=image_height)
        composite_text = TexText(
            "Composite Materials", font_size=FOOTNOTE_FONT_SIZE
        ).next_to(composite_img, 0.5 * DOWN)
        composite_cite = cite("composite_materials").next_to(composite_text, RIGHT)
        self.update_slide_contents(
            [composite_img, composite_text, composite_cite],
            notes="Composite materials...",
        )

        # slide: pem
        composite_img.generate_target()
        composite_img.target.scale(scale)
        composite_img.target.next_to(eit_img, RIGHT)
        composite_img.target.shift(shift_direction)
        self.play(
            MoveToTarget(composite_img),
            FadeOut(composite_text),
            FadeOut(composite_cite),
            run_time=0.5,
        )
        pem_img = ImageMobject("pem", height=image_height)
        pem_text = TexText(
            "Proton Exchange Membranes", font_size=FOOTNOTE_FONT_SIZE
        ).next_to(pem_img, 0.5 * DOWN)
        pem_cite = cite("pem").next_to(pem_text, RIGHT)
        self.update_slide_contents(
            [pem_img, pem_text, pem_cite],
            notes="and Proton Exchange Membranes have in common?",
        )

        # slide: model problem
        pem_img.generate_target()
        pem_img.target.scale(scale)
        pem_img.target.next_to(composite_img, RIGHT)
        pem_img.target.shift(shift_direction)
        self.play(
            MoveToTarget(pem_img, run_time=0.5),
            FadeOut(pem_text),
            FadeOut(pem_cite),
            run_time=0.5,
        )

        # slide: high contrast brace
        images = [
            pcb_img,
            subsurface_img,
            eit_img,
            composite_img,
            pem_img,
        ]
        images_vg = []
        bboxes = []
        for img in images:
            bbox = always_redraw(SurroundingRectangle, img)
            always(bbox.set_opacity, 0)
            bboxes.append(bbox)
            images_vg.append(bbox)
        images_vg = always_redraw(VGroup, *images_vg)
        high_contrast_brace = always_redraw(Brace, images_vg, direction=DOWN)
        high_contrast_label = always_redraw(
            TexText, "High-contrast", font_size=1.5 * CONTENT_FONT_SIZE
        )
        always(high_contrast_label.next_to, high_contrast_brace, DOWN, buff=0.1)
        self.play(
            Write(high_contrast_brace),
            Write(high_contrast_label),
            run_time=1.0,
        )
        self.next_slide(notes="They all involve high-contrast!")

        # slide: images move to the left and reveal high-contrast equation
        for i in range(1, len(images)):
            images[i].generate_target()
            bboxes[i].generate_target()
            images[i].target.align_to(self.slide_title, LEFT)
            bboxes[i].target.align_to(self.slide_title, LEFT)
            images[i].target.shift(2 * DOWN * i / len(images))
            bboxes[i].target.shift(2 * DOWN * i / len(images))
        self.play(
            *[MoveToTarget(img) for img in images[1:]],
            *[MoveToTarget(bbox) for bbox in bboxes[1:]],
            run_time=0.5,
        )
        self.next_slide(notes="This leads us to the model problem...")

        # slide: model problem
        model_problem = defense.paragraph(
            r"Let $u_D\in H^{3/2}(\partial\Omega)$, $f\in L^2(\Omega)$, $\mathcal{C}\in L^\infty(\Omega)$. Find $u\in H^2(\Omega)$ such that"
            + self.strong_equation,
            width=0.2 * FRAME_WIDTH,
            font_size=CONTENT_FONT_SIZE,
            alignment=ALIGN.CENTER,
        ).next_to(images_vg, RIGHT, buff=1.0)
        self.play(Write(model_problem), run_time=1.0)
        self.next_slide(
            notes="We want to find the solution u. To do so we discretize..."
        )

        # slide: variational formulation & discretization
        self.play(
            model_problem.animate.shift(1.5 * UP),
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
        domain_width = 0.25 * FRAME_WIDTH
        domain = Rectangle(
            width=domain_width, height=domain_width, color=WHITE
        ).next_to(model_problem_weak, 0.2 * DOWN, buff=1.0)
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
        ).next_to(model_problem_weak, 0.2 * DOWN, buff=1.0)
        self.play(ShowCreation(grid_lines), run_time=1.0)

        # slide: FEM formulation
        self.next_slide(notes="Through FEM")
        fem_formulation = defense.paragraph(
            r"FEM basis $V_h = \text{span}\{\phi_i\}_{i=1}^{n}$" + self.fem_formulation,
            width=0.2 * FRAME_WIDTH,
            font_size=CONTENT_FONT_SIZE,
            alignment=ALIGN.CENTER,
        ).move_to(model_problem_weak)
        self.play(
            ReplacementTransform(model_problem_weak[0], fem_formulation[0]),
            run_time=1.0,
        )
        self.next_slide(notes="we get a linear system..")
        linear_system_text = TexText(
            r"\begin{equation*}A\mathbf{u}=\mathbf{b}\end{equation*}",
            font_size=2.0 * CONTENT_FONT_SIZE,
        ).move_to(fem_formulation.get_center())
        self.play(
            FadeOut(domain),
            FadeOut(domain_label),
            FadeOut(grid_lines),
            ReplacementTransform(fem_formulation[0], linear_system_text),
            run_time=1.0,
        )

        # slide: CG
        self.next_slide(notes="We solve this system using Conjugate Gradient method...")
        arrow = Arrow(
            start=ORIGIN,
            end=DOWN,
            buff=0.0,
            stroke_width=2,
            color=WHITE,
        ).next_to(linear_system_text, DOWN, buff=0.5)
        cg_text = defense.paragraph(
            "Conjugate Gradient (CG) Method",
            font_size=1.5 * CONTENT_FONT_SIZE,
            alignment=ALIGN.CENTER,
            width=0.2 * FRAME_WIDTH,
        ).next_to(arrow, DOWN, buff=0.5)
        self.play(Write(arrow), Write(cg_text), run_time=1.0)
        succesive_approximations = VGroup()
        num_approximations = 5
        for i in range(num_approximations):
            approx = TexText(
                f"$\\mathbf{{u}}_{{{i}}}$,",
                font_size=1.5 * CONTENT_FONT_SIZE,
            )
            succesive_approximations.add(approx)
        succesive_approximations.add(TexText("...", font_size=1.5 * CONTENT_FONT_SIZE))
        succesive_approximations.arrange(RIGHT, buff=0.5)
        succesive_approximations.next_to(cg_text, DOWN, buff=0.5)
        approximations_brace = Brace(succesive_approximations, DOWN)
        approximations_label = TexText(
            "Successive Approximations", font_size=1.5 * CONTENT_FONT_SIZE
        )
        approximations_label.next_to(approximations_brace, DOWN)
        self.play(
            Write(succesive_approximations),
            Write(approximations_brace),
            Write(approximations_label),
            run_time=1.0,
        )
        cg_stuff = VGroup(
            cg_text,
            arrow,
            succesive_approximations,
            approximations_brace,
            approximations_label,
        )

        # slide: main research question
        self.next_slide(notes="But how fast does CG converge? Main Research Question.")
        main_question = defense.paragraph(
            "\\textit{How can we determine the total number of necessary CG approximations?}",
            font_size=2.0 * CONTENT_FONT_SIZE,
            alignment=ALIGN.CENTER,
            t2c={
                "necessary": CustomColors.RED.value,
            },
            width=0.22 * FRAME_WIDTH,
        )
        self.slide_contents = images + bboxes + [linear_system_text, cg_stuff]

        # move everything up
        self.update_slide(
            new_contents=main_question, subtitle="Main Research Question", notes=""
        )
        self.slide_contents = [main_question]

    def toc(self):
        item = Item()
        contents = defense.paragraph(
            f"{item}. Introducing CG"
            + f"\\\\{item}. How Does CG Converge? The Role of Eigenvalues"
            + f"\\\\{item}. Preconditioning: Taming High-Contrast Problems"
            + f"\\\\{item}. Towards Sharper Iteration Bounds: Two-Cluster Spectra"
            + f"\\\\{item}. Multi-Cluster Spectra"
            + f"\\\\{item}. How Sharp Are the New Bounds?"
            + f"\\\\{item}. New Bounds in Practice: Using Ritz Values"
            + f"\\\\{item}. Conclusion: Key Takeaways \& Future Directions",
            font_size=1.5 * CONTENT_FONT_SIZE,
            alignment=ALIGN.LEFT,
            width=0.22 * FRAME_WIDTH,
        ).align_to(self.slide_title, LEFT)
        self.update_slide("Contents", new_contents=contents, notes="Table of Contents")
        self.slide_contents = [contents]

    def level_1_intro_cg(self):
        linear_system_text = TexText(
            r"\begin{equation*}A\mathbf{u}=\mathbf{b}\end{equation*}",
            font_size=2.0 * CONTENT_FONT_SIZE,
        )
        self.update_slide(
            "Introducing CG",
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
        self.play(Write(initial_guess), Write(arrow), run_time=self.RUN_TIME)
        self.next_slide(notes="We provide an initial guess...")

        # slide: residual
        initial_system = VGroup(linear_system_text, initial_guess, arrow)
        initial_residual = TexText(
            r"\begin{equation*}\mathbf{r}_0 = \mathbf{b} - A\mathbf{u}_0\end{equation*}",
            font_size=2.0 * CONTENT_FONT_SIZE,
        ).move_to(initial_system.get_center())
        self.play(
            ReplacementTransform(initial_system, initial_residual),
            run_time=self.RUN_TIME,
        )
        self.next_slide(notes="which gives an initial residual.")

        # slide: desired error tolerance + CG black box
        initial_residual.generate_target()
        initial_residual.target.align_to(self.slide_title, LEFT)
        initial_residual.target.shift(0.5 * UP)
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
        self.play(
            Write(error_tol),
            Write(inputs_brace),
            Write(arrow2),
            Write(cg_rectangle),
            Write(cg_text),
            run_time=self.RUN_TIME,
        )
        self.next_slide(
            notes="We specify a desired error tolerance and feed everything into CG..."
        )

        # slide: cg algorithm
        cg_algorithm = TexText(
            self.cg_algorithm,
            additional_preamble=r"\usepackage{algorithmic}",
            font_size=2 * CONTENT_FONT_SIZE,
        )
        cg_alg_width = cg_algorithm.get_corner(UR)[0] - cg_algorithm.get_corner(DL)[0]
        cg_alg_height = cg_algorithm.get_corner(UR)[1] - cg_algorithm.get_corner(DL)[1]
        cg_rectangle_og = cg_rectangle.copy()
        cg_text_og = cg_text.copy()
        cg_rectangle.generate_target()
        cg_rectangle.target.move_to(ORIGIN)
        cg_rectangle.target.stretch_to_fit_width(cg_alg_width + 0.5)
        cg_rectangle.target.stretch_to_fit_height(cg_alg_height + 0.5)
        cg_rectangle.target.set_fill(CustomColors.BLUE.value, opacity=1.0)
        inputs_brace_arrow = VGroup(inputs, inputs_brace, arrow2)
        self.play(
            MoveToTarget(cg_rectangle),
            ReplacementTransform(cg_text, cg_algorithm),
            inputs_brace_arrow.animate.set_opacity(0.4),
            run_time=self.RUN_TIME,
        )
        self.next_slide(notes="which runs the following algorithm...")

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
        self.play(
            FadeTransform(cg_algorithm, cg_algorithm_colored),
            run_time=self.RUN_TIME,
        )
        self.next_slide(
            notes="We recognize the initial residual and error tolerance as inputs. Also note the error at the jth iteration."
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
        self.play(
            Write(outputs),
            Write(outputs_brace),
            Write(arrow3),
            inputs_brace_arrow.animate.set_opacity(1.0),
            run_time=self.RUN_TIME,
        )
        self.next_slide(
            notes="CG outputs the jth approximation and the corresponding error."
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
        self.play(
            ReplacementTransform(u_j_text, succesive_approximations),
            ReplacementTransform(epsilon_j_text, succesive_errors),
            FadeOut(inputs_brace_arrow),
            FadeOut(cg_text_og),
            FadeOut(cg_rectangle_og),
            FadeOut(outputs_brace),
            FadeOut(arrow3),
            run_time=self.RUN_TIME,
        )
        self.next_slide(notes="We get succesive approximations and errors...")

        # slide: CG error
        max_scale = 3.0
        min_scale = 0.5
        for i, err in enumerate(succesive_errors):
            center = err.get_center()
            center_on_origin = center - [0, center[1], 0]
            err.generate_target()
            err.target.scale(min_scale + (max_scale - min_scale) / (2**i))
            err.target.move_to(center_on_origin)
        self.play(
            FadeOut(succesive_approximations),
            *[MoveToTarget(err) for err in succesive_errors],
            run_time=self.RUN_TIME,
        )
        self.next_slide(notes="with errors decreasing as j increases.")

        # slide: CG classical error bound
        citation = cite("iter_method_saad").next_to(
            self.slide_title, RIGHT, buff=0.2
        )
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
        self.play(
            Write(citation),
            ReplacementTransform(succesive_errors, classical_error_bound),
            Write(classical_error_bound_box),
            Write(classical_error_bound_text),
            run_time=self.RUN_TIME,
        )
        self.next_slide(
            notes="The classical CG error bound relates the error to the condition number kappa(A) and the number of iterations m."
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
        self.play(
            FadeOut(citation),
            ReplacementTransform(classical_error_bound, classical_iteration_bound),
            ReplacementTransform(
                classical_error_bound_text, classical_iteration_bound_text
            ),
            ReplacementTransform(
                classical_error_bound_box, classical_iteration_bound_box
            ),
            run_time=self.RUN_TIME,
        )
        self.next_slide(
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
        self.play(
            MoveToTarget(m_1_text),
            ReplacementTransform(m_text, m_text_new),
            Write(high_contrast_text),
            run_time=2 * self.RUN_TIME,
        )
        self.next_slide(
            notes="The problem is that for high-contrast problems, kappa is very large. Leading to very pessimistic bounds on m.",
        )

        # slide: update research question
        self.slide_contents = [
            cg_iteration_bound_simplified,
            classical_iteration_bound_text,
            m_text_new,
        ]
        previous_main_question = defense.paragraph(
            "\\textit{How can we determine the total number of necessary CG approximations?}",
            font_size=2.0 * CONTENT_FONT_SIZE,
            alignment=ALIGN.CENTER,
            t2c={
                "necessary": CustomColors.RED.value,
            },
            width=0.22 * FRAME_WIDTH,
        )
        self.update_slide(
            new_contents=[previous_main_question],
            subtitle="Research Question (Revised)",
            notes="So we update our research question.",
        )
        new_main_question = defense.paragraph(
            "\\textit{How can we construct a sharp CG iteration bound for high-contrast problems?}",
            font_size=2.0 * CONTENT_FONT_SIZE,
            alignment=ALIGN.CENTER,
            t2c={
                "high-contrast": CustomColors.GOLD.value,
                "sharp": CustomColors.GOLD.value,
            },
            width=0.22 * FRAME_WIDTH,
        )
        self.play(
            ReplacementTransform(previous_main_question, new_main_question),
            run_time=2 * self.RUN_TIME,
        )
        self.next_slide(notes="This is the main research question of this thesis.")
        self.slide_contents = [new_main_question]

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
            .align_to(self.slide_subtitle, LEFT)
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
        self.play(
            ReplacementTransform(linear_system_text, plot),
            FadeIn(speaker_img),
            run_time=self.RUN_TIME,
        )
        self.update_slide(
            subtitle="The Role of Eigenvalues: Sound Example",
            notes="We can view the matrix A as a signal...",
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
            subtitle="The Role of Eigenvalues",
            new_contents=[linear_system_text],
            notes="Returning to our original system...",
        )

        # slide: eigenvalues of A
        spectrum_A = TexText(
            r"$\sigma(A) = \{\lambda_1, \lambda_2, \ldots, \lambda_n\}$",
            font_size=2.0 * CONTENT_FONT_SIZE,
            t2c={
                r"\lambda_1": CustomColors.RED.value,
                r"\lambda_2": CustomColors.GOLD.value,
                r"\lambda_n": CustomColors.BLUE.value,
            },
        ).move_to(linear_system_text.get_center())
        condition_number_text = TexText(
            r"$\kappa(A) = \frac{\lambda_{\text{max}}}{\lambda_{\text{min}}}$",
            font_size=2.0 * CONTENT_FONT_SIZE,
        )
        always(condition_number_text.next_to, spectrum_A, DOWN, buff=0.5)
        self.play(
            ReplacementTransform(linear_system_text, spectrum_A),
            Write(condition_number_text, lag_ratio=0.5),
            run_time=2 * self.RUN_TIME,
        )
        self.next_slide(
            notes="We can do a similar decomposition of A resulting in its eigenvalues or spectrum...",
        )

        # slide: CG solution polynmomial
        spectrum_A.generate_target()
        spectrum_A.target.next_to(self.slide_subtitle, DOWN, buff=0.5).align_to(
            self.slide_subtitle, LEFT
        )
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
        self.play(
            Write(arrow),
            Write(cg_solution),
            Write(cg_solution_text),
            run_time=2 * self.RUN_TIME,
        )
        self.next_slide(
            notes="Why is this important? To see this, we look at the CG solution polynomial..."
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
        self.play(
            ReplacementTransform(cg_solution, cg_residual),
            ReplacementTransform(cg_solution_text, cg_residual_text),
        )
        self.next_slide(notes="Which is related to the residual polynomial...")

        # slide: General CG error bound
        cg_residual.generate_target()
        cg_residual.target.next_to(condition_number_text, DOWN, buff=0.5).align_to(
            self.slide_subtitle, LEFT
        )
        cg_residual_and_spectrum = VGroup(
            spectrum_A, condition_number_text, cg_residual
        )
        brace = Brace(cg_residual_and_spectrum, RIGHT, buff=0.1)
        always(brace.next_to, cg_residual_and_spectrum, RIGHT, buff=0.1)
        cg_error_bound = TexText(
            r"$\epsilon_m \leq \underset{r\in\mathcal{P}_m,\ r(0)=1}{\min} \ \underset{\lambda \in \sigma(A)}{\max}|r(\lambda)|$",
            font_size=2.0 * CONTENT_FONT_SIZE,
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
        self.play(
            FadeOut(cg_residual_text),
            FadeOut(cg_rectangle_and_text),
            FadeOut(arrow),
            MoveToTarget(cg_residual),
            Write(brace),
            Write(cg_error_bound),
            Write(cg_error_bound_text),
            run_time=2 * self.RUN_TIME,
        )
        self.next_slide(
            notes="The CG error can be bounded in terms of the residual polynomial..."
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
        self.play(
            FadeOut(brace),
            FadeOut(cg_residual_and_spectrum),
            ReplacementTransform(cg_error_bound, cg_error_bound_uniform),
            ReplacementTransform(cg_error_bound_text, cg_error_bound_text_uniform),
            Write(arrow),
            Write(m_1_text),
            run_time=2 * self.RUN_TIME,
        )
        self.next_slide(
            notes="which can recover the classical bound by assuming a uniform distribution of eigenvalues...",
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
        )
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
            full_classical_bound.animate.to_edge(DOWN).shift(0.5 * UP),
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
        )
        self.next_slide(
            notes="For a uniform spectrum, the classical bound is sharp.",
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
        )
        self.next_slide(
            notes="For a split spectrum, the classical bound is pessimistic.",
        )

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
        old_research_question = defense.paragraph(
            "\\textit{How can we construct a sharp CG iteration bound for high-contrast problems?}",
            font_size=2.0 * CONTENT_FONT_SIZE,
            alignment=ALIGN.CENTER,
            width=0.22 * FRAME_WIDTH,
        )
        self.update_slide(
            new_contents=[old_research_question],
            subtitle="Research Question (Revisited)",
            notes="Revisiting the research question in light of the new findings.",
        )

        # slide: new research question
        new_research_question = defense.paragraph(
            "\\textit{How can we construct a sharp CG iteration bound for high-contrast problems using the full spectrum of A?}",
            t2c={
                "full spectrum of A": CustomColors.GOLD.value,
            },
            font_size=2.0 * CONTENT_FONT_SIZE,
            alignment=ALIGN.CENTER,
            width=0.22 * FRAME_WIDTH,
        )
        self.play(
            ReplacementTransform(old_research_question, new_research_question),
            run_time=self.RUN_TIME,
        )
        self.next_slide(notes="This leads to the refined research question.")
        self.slide_contents = [new_research_question]

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
            self.play(
                FadeIn(cg_residual_poly_graph, scale=0.5),
                FadeIn(spectrum_group, scale=0.5),
                ReplacementTransform(self.y_label, y_label),
                FadeIn(degree_tex, scale=0.5),
                run_time=spectra_transition_time,
            )
        else:
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
        self.update_slide(
            "Preconditioning",
            new_contents=[high_contrast_issue],
            subtitle="Taming High-Contrast Problems",
            notes="Recap: High-contrast leads to",
        )
        always(high_contrast_issue.arrange, RIGHT, buff=0.5)
        always(high_contrast_issue.move_to, ORIGIN)

        # slide: continued
        self.play(
            condition_number.animate.scale(2.0),
            arrow_to_m_1.animate.set_opacity(1),
            m_1.animate.set_opacity(1).scale(2.0),
            m.animate.set_opacity(1),
            run_time=2 * self.RUN_TIME,
        )
        self.next_slide(
            notes="high values for kappa and thus pessimistic bound m_1 on m.",
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
        self.play(
            ReplacementTransform(system, preconditioned_system), run_time=self.RUN_TIME
        )
        self.next_slide(
            notes="The problem of large condition numbers can be mitigated by using preconditioning. We transform the system...",
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
            FadeOutToPoint(preconditioned_system, conditioner_number_center),
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
        self.play(
            ReplacementTransform(high_contrast_issue, high_contrast_issue2),
            run_time=self.RUN_TIME,
        )
        self.play(
            preconditioned_condition_number.animate.scale(0.5),
            m_1.animate.scale(0.5),
            run_time=self.RUN_TIME,
        )
        m_sharp = TexText(
            r"$ \geq m$",
            font_size=2.0 * CONTENT_FONT_SIZE,
            t2c={r"\kappa": CustomColors.RED.value},
        ).move_to(m.get_center())
        self.play(ReplacementTransform(m, m_sharp), run_time=self.RUN_TIME)
        self.next_slide(
            notes="and hope that the preconditioned system has a smaller condition number.",
        )

        # slide: high coefficient functions from Filipe and Alexander
        high_contrast_issue2.clear_updaters()
        self.slide_contents = [
            high_contrast_text,
            arrow_hc_to_kappa,
            preconditioned_condition_number,
            arrow_to_m_1,
            m_1,
            m_sharp,
        ]
        coefficient_function_image = ImageMobject("coefficient_functions").set_height(
            0.6 * FRAME_HEIGHT
        )
        citations = [
            cite("ams_coarse_space_comp_study_Alves2024"),
            cite("ams_framework_Wang2014"),
            cite("gdsw_coarse_space_Dohrmann2008"),
            cite("rgdsw_coarse_space_Dohrmann2017"),
        ]
        for i, c in enumerate(citations):
            c.next_to(self.slide_subtitle, RIGHT, buff=0.1).shift(i * 0.3 * RIGHT)
        self.update_slide(
            subtitle="Taming High-Contrast Problems",
            new_contents=[coefficient_function_image, citations[0]],
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
            M.next_to(coefficient_function_image, DOWN, buff=0).shift(
                (i - 1) * 3.0 * RIGHT
            )
        self.play(
            Write(M_1),
            Write(M_2),
            Write(M_3),
            *[Write(citation) for citation in citations[1:]],
            run_time=2 * self.RUN_TIME,
        )
        self.next_slide(
            notes="We consider three different preconditioners for these problems...",
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
            M.next_to(coefficient_function_image, DOWN, buff=0).shift(
                (i - 1) * 1.0 * RIGHT
            )
        self.play(
            ReplacementTransform(M_1, M_1_simple),
            ReplacementTransform(M_2, M_2_simple),
            ReplacementTransform(M_3, M_3_simple),
            run_time=self.RUN_TIME,
        )
        self.next_slide(notes="which we will refer to as M1, M2, and M3.")

        # slide: all preconditioners had similar condition numbers
        M_2_kappa = TexText(
            r"$\sim\kappa(M_2^{-1}A)\sim$",
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
        self.play(
            *[FadeOut(citation) for citation in citations],
            FadeOut(coefficient_function_image),
            ReplacementTransform(M_1_simple, M_1_kappa),
            ReplacementTransform(M_2_simple, M_2_kappa),
            ReplacementTransform(M_3_simple, M_3_kappa),
            run_time=2 * self.RUN_TIME,
        )
        self.next_slide(
            notes="All three preconditioners resulted in similar condition numbers...",
        )

        # slide: results from study
        vertex_func_results = (
            ImageMobject("cg_iterations_coefficient_vertex_Alves")
            .stretch_to_fit_width(0.4 * FRAME_WIDTH)
            .align_to(self.slide_subtitle, LEFT)
            .shift(0.9 * RIGHT)
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
                r"$\mathcal{C}_{\text{3layer, edge}}$",
                font_size=CONTENT_FONT_SIZE,
            ).next_to(edge_func_results, UP, buff=0.2),
        ]
        legend = (
            ImageMobject("method_legend_Alves")
            .stretch_to_fit_height(0.05 * FRAME_HEIGHT)
            .stretch_to_fit_width(0.4 * FRAME_WIDTH)
            .to_edge(DOWN)
            .shift(0.5 * UP)
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

        self.play(
            FadeOut(M_1_kappa),
            FadeOut(M_2_kappa),
            FadeOut(M_3_kappa),
            *[FadeIn(mobj) for mobj in vertex_func_plot],
            *[FadeIn(mobj) for mobj in edge_func_plot],
            FadeIn(legend),
            FadeIn(M_1_legend),
            FadeIn(M_2_legend),
            FadeIn(M_3_legend),
            run_time=2 * self.RUN_TIME,
        )
        self.next_slide(
            notes="However the number of CG iterations varied significantly..."
        )

        # slide: M_2 (RGDSW) took more iterations
        M_2_legend.generate_target()
        M_2_legend.target.scale(2.0)
        M_2_legend.background_rectangle.generate_target()
        M_2_legend.background_rectangle.target.stretch_to_fit_width(1.3)
        M_2_legend.background_rectangle.target.stretch_to_fit_height(
            legend.get_height()
        )
        self.play(
            MoveToTarget(M_2_legend),
            MoveToTarget(M_2_legend.background_rectangle),
            run_time=self.RUN_TIME,
        )
        self.next_slide(
            notes="with M2 (RGDSW) consistently taking more iterations.",
        )

        # slide: restate research question
        self.slide_contents = (
            vertex_func_plot
            + edge_func_plot
            + [legend, M_1_legend, M_2_legend, M_3_legend]
        )
        old_research_question = defense.paragraph(
            "\\textit{How can we sharpen the classical CG iteration bound $m_1$ for high-contrast problems using the full spectrum of A?}",
            font_size=2.0 * CONTENT_FONT_SIZE,
            alignment=ALIGN.CENTER,
            width=0.22 * FRAME_WIDTH,
        )
        self.update_slide(
            new_contents=[old_research_question],
            subtitle="Research Question (Revisited)",
            notes="Revisiting the research question in light of the new findings.",
        )

        # slide: new research question
        new_research_question = defense.paragraph(
            "\\textit{How can we construct a sharp CG iteration bound for high-contrast problems using the full spectrum of A, such that it can distinguish between $M_1,M_2,M_3$?}",
            t2c={
                "can distinguish between $M_1,M_2,M_3$": CustomColors.GOLD.value,
            },
            font_size=2.0 * CONTENT_FONT_SIZE,
            alignment=ALIGN.CENTER,
            width=0.22 * FRAME_WIDTH,
        )
        self.play(
            ReplacementTransform(old_research_question, new_research_question),
            run_time=self.RUN_TIME,
        )
        self.slide_contents = [new_research_question]

    def level_4_two_clusters(self):
        self.update_slide(
            "Towards Sharper Iteration Bounds",
            subtitle="Two-Cluster Spectra",
            notes="Explain two-cluster bound from Axelsson",
        )

    def level_5_multi_clusters(self):
        self.update_slide(
            "Multi-Cluster Spectra",
            notes="Extension to multi-cluster spectra",
        )

    def level_6_sharpness(self):
        self.update_slide(
            "How Sharp Are the New Bounds?",
            notes="Discuss results on absolute sharpness of the new bounds",
        )

    def level_7_early_bounds(self):
        self.update_slide(
            "Early Bounds",
            notes="Discuss results on using Ritz values to get early bounds",
        )

    def level_8_conclusion(self):
        self.update_slide(
            "Conclusion",
            subtitle="Key Takeaways & Future Directions",
            notes="Conclusion and future directions",
        )

    def backup(self):
        self.update_slide("Backup Slides", notes="Backup slides", clean_up=True)

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
                .next_to(self.slide_title, DOWN, buff=0.5)
                .align_to(self.slide_title, LEFT)
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
        self.title_slide()
        self.level_0_opening()
        self.toc()
        self.level_1_intro_cg()
        self.level_2_cg_convergence()
        self.level_3_preconditioning()
        self.level_4_two_clusters()
        # self.level_5_multi_clusters()
        # self.level_6_sharpness()
        # self.level_7_early_bounds()
        # self.level_8_conclusion()
        # self.backup()
        self.references()

    # TODO: miscellaneous
    # Optional: Video playback function
    # def play_video(file):
    #     cap = cv2.VideoCapture(file)
    #     flag = True

    #     while flag:
    #         flag, frame = cap.read()
    #         fps = cap.get(cv2.CAP_PROP_FPS)
    #         delay = 1 / fps

    #         if flag:
    #             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #             frame_img = ImageMobject(frame, *args, **kwargs)
    #             self.add(frame_img)
    #             self.wait(delay)
    #             self.remove(frame_img)

    #     cap.release()
    #     cap.release()
