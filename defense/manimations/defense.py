# NOTE the structure and functionality of this script and its main class `defense` are
# adopted from https://github.com/jeertmans/jeertmans.github.io/blob/main/_slides/2023-12-07-confirmation/main.py
# next to that of course the manim_slides documentation https://manim-slides.readthedocs.io/en/latest/
from manimlib import *

pass
# import cv2 # install for video playback
import os
from json import load

from manim_slides import Slide

os.environ["NO_HCMSFEM_CLI_ARGS"] = ""
from hcmsfem.plot_utils import CustomColors
from hcmsfem.root import get_venv_root

# Manim render settings
FPS = 30
QUALITY = (1920, 1080)  # 4k = (3840,2160)
manim_config.camera.fps = FPS
manim_config.camera.resolution = QUALITY
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


class defense(Slide):

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
        self.counter = 1
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

        # self.tex_template.add_to_preamble(
        #     r"""
        # \usepackage{siunitx}
        # \usepackage{amsmath}
        # \usepackage[colorlinks=true, urlcolor=blue]{hyperref}
        # """
        # )

    # utility functions
    def update_slide_number(self):
        self.counter += 1
        new_slide_number = TexText(f"{self.counter}").move_to(self.slide_number)
        slide_number_update = ReplacementTransform(self.slide_number, new_slide_number)
        return slide_number_update, new_slide_number

    def update_slide_titles(self, title, subtitle):
        title_animations = []

        # construct new title
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

        return title_animations, new_title, new_subtitle if subtitle else None

    def update_slide(
        self,
        title,
        subtitle=None,
        new_contents: list[Mobject] = [],
        transition_time: float = 0.75,
        notes: str = None,
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
        self.next_slide(notes=notes)

    def update_slide_contents(self, new_contents: list[Mobject], notes: str = None):
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

        # Remove old content from slide
        wipe_animation += [FadeOut(m) for m in self.slide_contents]

        # update slide number
        slide_number_update, new_slide_number = self.update_slide_number()

        # animate optional content wipe
        self.play(*wipe_animation, slide_number_update)

        # update new_contents
        # self.slide_contents = new_contents
        self.slide_constents = []
        self.slide_number = new_slide_number

        self.next_slide(notes=notes)

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

    @staticmethod
    def paragraph(
        *strs, alignment=LEFT, direction=DOWN, width=0.5 * FRAME_WIDTH, **kwargs
    ):
        texts = []
        for s in strs:
            print(s)
            texts.append(
                TexText(
                    f"\\begin{{minipage}}{{{width}in}}{{{s}}}\\end{{minipage}}",
                    alignment=R"\raggedright",
                    **kwargs,
                )
            )
        texts = VGroup(*texts).arrange(direction)

        if len(strs) > 1:
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
        self.next_slide(notes="title slide")
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
        self.next_slide()

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
        high_contrast_label = always_redraw(TexText, "High-contrast")
        always(high_contrast_label.next_to, high_contrast_brace, DOWN, buff=0.1)
        self.play(
            ShowCreation(high_contrast_brace),
            ShowCreation(high_contrast_label),
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

        # model problem

        # collect remaining slide contents
        self.slide_contents = images + bboxes

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
            font_size=CONTENT_FONT_SIZE,
        ).align_to(self.slide_title, LEFT)
        self.update_slide("Contents", new_contents=contents, notes="Table of Contents")
        self.slide_contents = [contents]

    def level_1_intro_cg(self):
        self.update_slide(
            "Introducing CG",
            notes="Explain CG as an iterative method for solving Ax=b",
        )

    def level_2_cg_convergence(self):
        self.update_slide(
            "How Does CG Converge?",
            subtitle="The Role of Eigenvalues",
            notes="Explain CGs dependence on eigenvalues",
        )

    def level_3_preconditioning(self):
        self.update_slide(
            "Preconditioning",
            subtitle="Taming High-Contrast Problems",
            notes="Explain (motivation for) preconditioning",
        )

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
        for i in range(0, len(refs), chunk_size):
            refs_text = [f"{ref}" for ref in refs[i : i + chunk_size]]
            refs_mobj = (
                defense.paragraph(
                    *refs_text,
                    font_size=FOOTNOTE_FONT_SIZE,
                    additional_preamble="\\usepackage{hyperref}",
                    alignment=LEFT,
                    width=0.6 * FRAME_WIDTH,
                )
                .next_to(self.slide_title, DOWN, buff=0.5)
                .align_to(self.slide_title, LEFT)
            )
            self.update_slide(
                f"References ({i+1}-{min(i+chunk_size, len(refs))})",
                new_contents=refs_mobj,
                notes="References",
            )

    # full construct
    def construct(self):
        # self.wait_time_between_slides = 0.10
        self.title_slide()
        self.level_0_opening()
        self.toc()
        # self.level_1_intro_cg()
        # self.level_2_cg_convergence()
        # self.level_3_preconditioning()
        # self.level_4_two_clusters()
        # self.level_5_multi_clusters()
        # self.level_6_sharpness()
        # self.level_7_early_bounds()
        # self.level_8_conclusion()
        # self.backup()
        self.references()
