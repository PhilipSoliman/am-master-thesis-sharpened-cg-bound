from manimlib import *

pass
from manim_slides import Slide

FPS = 30
PIXEL_WIDTH = 1920
PIXEL_HEIGHT = 1080
QUALITY = (PIXEL_WIDTH, PIXEL_HEIGHT)  # 4k = (3840,2160)
manim_config.camera.fps = FPS
manim_config.camera.resolution = QUALITY


class slides_test(Slide):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.counter = 1
        self.canvas_contents: list[Mobject] = []

    def update_canvas(
        self,
        mobjects: list[Mobject] = [],
        transition_time: float = 0.75,
        clean_up: bool = False,
    ):
        self.counter += 1
        old_slide_number = self.canvas["slide_number"]
        new_slide_number = Text(f"{self.counter}").move_to(old_slide_number)
        wipe_animation = []
        if mobjects or clean_up:
            # make animations for wiping old content out
            wipe_animation += [
                m.animate.move_to(m.get_center() - np.array([FRAME_WIDTH, 0, 0]))
                for m in self.canvas_contents
            ]

            if not clean_up:
                # add new content to scene but out of view
                for m in mobjects:
                    m.move_to(m.get_center() + np.array([FRAME_WIDTH, 0, 0]))
                self.add(*mobjects)

                # make animations for bringing new content in
                wipe_animation += [
                    m.animate.move_to(m.get_center() - np.array([FRAME_WIDTH, 0, 0]))
                    for m in mobjects
                ]

            # update canvas content
            self.canvas_contents = mobjects

        # animate slide number change and optional content wipe
        self.play(
            Transform(old_slide_number, new_slide_number),
            *wipe_animation,
            run_time=transition_time,
        )

    def construct(self):
        title = Text("My Title").to_corner(UL)
        slide_number = Text("1").to_corner(DL)
        self.add_to_canvas(title=title, slide_number=slide_number)
        self.play(FadeIn(title), FadeIn(slide_number))

        self.next_slide()
        circle = Circle(radius=2)
        dot = Dot()
        self.canvas_contents += [circle, dot]
        self.update_canvas()
        self.play(ShowCreation(circle))
        self.play(MoveAlongPath(dot, circle))

        self.next_slide()
        self.update_canvas()
        square = Square()
        self.update_canvas(mobjects=[square])

        self.next_slide()
        self.update_canvas()
        self.play(Transform(self.canvas["title"], Text("New Title").to_corner(UL)))

        self.next_slide()
        self.update_canvas(clean_up=True)
        self.remove_from_canvas("title", "slide_number")
