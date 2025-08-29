from manim import *
from manim_slides import Slide

class DefaultTemplate(Slide):
    def construct(self):
        self.next_slide(loop=True)  # Start looping

        circle = Circle()  # create a circle
        circle.set_fill(PINK, opacity=0.5)  # set color and transparency

        square = Square()  # create a square
        # square.flip(RIGHT)  # flip horizontally
        square.rotate(-3 * TAU / 8)  # rotate a certain amount

        self.play(Create(square))  # animate the creation of the square
        self.play(Transform(square, circle))  # interpolate the square into the circle
        self.play(FadeOut(square))  # fade out animation

class CGResidualPolynomial(Slide):
    lambda_min = 0.01
    lambda_max = 1

    def construct(self):
        self.next_slide(loop=True)  # Start looping
        axes = Axes(
            x_range=[0, self.lambda_max * 1.1, 2],
            y_range=[-0.2, 1.1, 0.5],
            x_length=10,
            y_length=5,
            axis_config={"color": BLUE},
            x_axis_config={"numbers_to_include": np.arange(0, 11, 2)},
            y_axis_config={"numbers_to_include": np.arange(0, 1.1, 0.5)},
        ).to_edge(DOWN)
        
        axes_labels = axes.get_axis_labels(x_label=r"\lambda", y_label=r"|P_k(\lambda)|")

        self.play(Create(axes), Write(axes_labels))
