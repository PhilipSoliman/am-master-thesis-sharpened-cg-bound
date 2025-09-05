# NOTE the structure and functionality of this script and its main class `defense` are
# adopted from https://github.com/jeertmans/jeertmans.github.io/blob/main/_slides/2023-12-07-confirmation/main.py
# next to that of course the manim_slides documentation https://manim-slides.readthedocs.io/en/latest/
from manimlib import *

# from manim import *

pass
# import cv2 # install for video playback
import os

from manim_slides import Slide

os.environ["NO_HCMSFEM_CLI_ARGS"] = ""
from hcmsfem.plot_utils import CustomColors

# Manim render settings
FPS = 30
QUALITY = (1920, 1080)  # 4k = (3840,2160)
manim_config.camera.fps = FPS
manim_config.camera.resolution = QUALITY
manim_config.background_color = WHITE


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

        # Font sizes
        self.TITLE_FONT_SIZE = 48
        self.CONTENT_FONT_SIZE = 0.6 * self.TITLE_FONT_SIZE
        self.SOURCE_FONT_SIZE = 0.2 * self.TITLE_FONT_SIZE

        # Mutable variables
        self.counter = 1
        self.slide_number = Integer(1).set_color(WHITE).to_corner(DR)
        self.slide_title = Text("Contents", font_size=self.TITLE_FONT_SIZE).to_corner(
            UL
        )
        self.slide_subtitle = Text(
            "Subcontents", font_size=0.5 * self.TITLE_FONT_SIZE
        ).next_to(self.slide_title, DOWN)
        self.slide_subtitle_visible = False

        # slide contents (everything except title, subtitle, slide number)
        self.slide_contents: list[Mobject] = []

        # self.tex_template.add_to_preamble(
        #     r"""
        # \usepackage{siunitx}
        # \usepackage{amsmath}
        # \newcommand{\ts}{\textstyle}
        # """
        # )

    # utility functions
    def update_slide_number(self):
        self.counter += 1
        new_slide_number = Text(f"{self.counter}").move_to(self.slide_number)
        slide_number_update = ReplacementTransform(self.slide_number, new_slide_number)
        return slide_number_update, new_slide_number

    def update_slide_titles(self, title, subtitle):
        title_animations = []

        # construct new title
        new_title = (
            Text(title, font_size=self.TITLE_FONT_SIZE)
            .move_to(self.slide_title)
            .align_to(self.slide_title, LEFT)
        )
        title_animations.append(ReplacementTransform(self.slide_title, new_title))

        # check for new subtitle
        if subtitle is not None and self.slide_subtitle_visible:
            new_subtitle = (
                Text(subtitle, font_size=0.5 * self.TITLE_FONT_SIZE)
                .move_to(self.slide_subtitle)
                .align_to(self.slide_title, LEFT)
            )
            title_animations.append(
                ReplacementTransform(self.slide_subtitle, new_subtitle)
            )
            self.slide_subtitle_visible = True
        elif subtitle is not None and not self.slide_subtitle_visible:
            new_subtitle = (
                Text(subtitle, font_size=0.5 * self.TITLE_FONT_SIZE)
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
        mobjects: list[Mobject] = [],
        transition_time: float = 0.75,
        clean_up: bool = False,
        notes: str = None,
    ):
        """
        Update the slide with new mobjects. If clean_up is True, remove all existing mobjects from the slide.
        """

        wipe_animation = []
        if mobjects or clean_up:
            # make animations for wiping old content out
            wipe_animation += [
                m.animate.move_to(m.get_center() - np.array([FRAME_WIDTH, 0, 0]))
                for m in self.slide_contents
            ]

            if not clean_up:
                # add new content to scene but out of view
                for m in mobjects:
                    m.move_to(m.get_center() + np.array([FRAME_WIDTH, 0, 0]))

                # make animations for bringing new content in
                wipe_animation += [
                    m.animate.move_to(m.get_center() - np.array([FRAME_WIDTH, 0, 0]))
                    for m in mobjects
                ]

            # Remove old content from slide
            # wipe_animation.append(FadeOut(*self.slide_contents))

            # overwrite slide content
            self.slide_contents = mobjects

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

        # update mobjects
        self.slide_title = new_title
        if subtitle:
            self.slide_subtitle = new_subtitle
        self.slide_number = new_slide_number

        # go to next slide
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
    def paragraph(*strs, alignment=LEFT, direction=DOWN, **kwargs):
        texts = VGroup(*[Text(s, **kwargs) for s in strs]).arrange(direction)

        if len(strs) > 1:
            for text in texts[1:]:
                text.align_to(texts[0], direction=alignment)

        return texts

    # level constructs
    def title_slide(self):
        # Add initial slide (maybe high-contrast coefficient func?)

        # Title slide
        self.next_slide(notes="title slide")
        title = Text(
            "Sharpened CG Iteration Bound\nfor High-contrast Heterogeneous\nScalar Elliptic PDEs",
            font_size=0.8 * self.TITLE_FONT_SIZE,
            t2c={"High-contrast": RED},
            alignment="CENTER",
        )
        subtitle = Text(
            "Going Beyond Condition Number",
            font_size=0.5 * self.TITLE_FONT_SIZE,
            alignment="CENTER",
        ).next_to(title, DOWN)
        author = Text(
            "Philip M. Soliman",
            font_size=0.4 * self.TITLE_FONT_SIZE,
            alignment="CENTER",
        ).next_to(subtitle, DOWN)
        affiliation = Text(
            "Master's Thesis Defense\nSupervised by Prof. A. Heinlein and\n F. Cumaru",
            color=BLACK,
            font_size=0.25 * self.TITLE_FONT_SIZE,
            t2c={"Prof. A. Heinlein and\n F. Cumaru": CustomColors.RED.value},
            alignment="RIGHT",
        ).next_to(author, DOWN)
        self.play(
            Write(title),
            Write(subtitle),
            Write(author),
            Write(affiliation),
        )
        self.slide_contents += VGroup(title, subtitle, author, affiliation)
        self.next_slide()

    def level_0_opening(self):
        self.update_slide(
            "Opening",
            subtitle="Motivation",
            notes="What do all these applications have in common?",
            clean_up=True,
        )

    def toc(self):
        item = Item()
        contents = defense.paragraph(
            f"{item}. Introducing CG",
            f"{item}. How Does CG Converge? The Role of Eigenvalues",
            f"{item}. Preconditioning: Taming High-Contrast Problems",
            f"{item}. Towards Sharper Iteration Bounds: Two-Cluster Spectra",
            f"{item}. Multi-Cluster Spectra",
            f"{item}. How Sharp Are the New Bounds?",
            f"{item}. New Bounds in Practice: Using Ritz Values",
            f"{item}. Conclusion: Key Takeaways \& Future Directions",
            # color=BLACK,
            font_size=self.CONTENT_FONT_SIZE,
        ).align_to(self.slide_title, LEFT)
        self.update_slide("Contents", mobjects=contents, notes="Table of Contents")

    def level_1_intro_cg(self):
        self.update_slide(
            "Introducing CG",
            notes="Explain CG as an iterative method for solving Ax=b",
            clean_up=True,
        )

    def level_2_cg_convergence(self):
        self.update_slide(
            "How Does CG Converge?",
            subtitle="The Role of Eigenvalues",
            notes="Explain CGs dependence on eigenvalues",
            clean_up=True,
        )

    def level_3_preconditioning(self):
        self.update_slide(
            "Preconditioning",
            subtitle="Taming High-Contrast Problems",
            notes="Explain (motivation for) preconditioning",
            clean_up=True,
        )

    def level_4_two_clusters(self):
        self.update_slide(
            "Towards Sharper Iteration Bounds",
            subtitle="Two-Cluster Spectra",
            notes="Explain two-cluster bound from Axelsson",
            clean_up=True,
        )

    def level_5_multi_clusters(self):
        self.update_slide(
            "Multi-Cluster Spectra",
            notes="Extension to multi-cluster spectra",
            clean_up=True,
        )

    def level_6_sharpness(self):
        self.update_slide(
            "How Sharp Are the New Bounds?",
            notes="Discuss results on absolute sharpness of the new bounds",
            clean_up=True,
        )

    def level_7_early_bounds(self):
        self.update_slide(
            "Early Bounds",
            notes="Discuss results on using Ritz values to get early bounds",
            clean_up=True,
        )

    def level_8_conclusion(self):
        self.update_slide(
            "Conclusion",
            subtitle="Key Takeaways & Future Directions",
            notes="Conclusion and future directions",
            clean_up=True,
        )

    def level_9_backup(self):
        self.update_slide("Backup Slides", notes="Backup slides", clean_up=True)

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
        self.level_5_multi_clusters()
        self.level_6_sharpness()
        self.level_7_early_bounds()
        self.level_8_conclusion()
        self.level_9_backup()
