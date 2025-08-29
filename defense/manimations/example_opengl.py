from manim import *
from manim.opengl import *


class OpenGLExample(Scene):
    def construct(self):
        # Create a 3D axes
        axes = ThreeDAxes()

        # Create a surface
        self.surface = OpenGLSurface(
            lambda u, v: np.array([u, v, np.sin(u) * np.cos(v)]),
            u_range=[-PI, PI],
            v_range=[-PI, PI],
            resolution=(15, 32),
        )
        surface_mesh = OpenGLSurfaceMesh(self.surface)
        self.play(Create(surface_mesh))
        self.play(FadeTransform(surface_mesh, self.surface))
        self.wait(1)
        self.play(
            self.camera.animate.set_euler_angles(phi=50 * DEGREES, theta=-10 * DEGREES),
            run_time=3,
        )

        # change lighting
        light = self.camera.light_source
        self.play(light.animate.move_to([5, 5, 5]), run_time=2)
        self.play(light.animate.move_to([-5, -5, 5]), run_time=2)
        self.play(
            self.camera.animate.set_euler_angles(phi=70 * DEGREES, theta=70 * DEGREES),
            run_time=3,
        )
        self.interactive_embed()

    def on_key_press(self, symbol, modifiers):
        from pyglet.window import key as pyglet_key

        if symbol == pyglet_key.SPACE:
            self.play(self.surface.animate.set_opacity(0.5), duration=4)

        return super().on_key_press(symbol, modifiers)


class NewtonIteration(Scene):
    def construct(self):
        self.axes = Axes()
        self.f = lambda x: (x + 6) * (x + 3) * x * (x - 3) * (x - 6) / 300
        curve = self.axes.plot(self.f, x_range=[-7, 7], color=BLUE)
        self.cursor_dot = Dot(color=YELLOW)
        self.play(Create(self.axes), Create(curve), FadeIn(self.cursor_dot))
        self.interactive_embed()

    def on_key_press(self, symbol, modifiers):
        from pyglet.window import key as pyglet_key
        from scipy.differentiate import derivative

        if symbol == pyglet_key.RIGHT:
            x, y = self.axes.point_to_coords(self.mouse_point.get_location())
            self.play(
                self.cursor_dot.animate.move_to(self.axes.c2p(x, self.f(x))),
            )
        if symbol == pyglet_key.I:
            x, y = self.axes.point_to_coords(self.cursor_dot.get_center())
            # newton
            x_new = x - self.f(x) / derivative(self.f, x, atol=1e-6)
            curve_point = self.cursor_dot.get_center()
            axes_point = self.axes.c2p(x_new, 0)
            tangent = Line(
                curve_point + (curve_point - axes_point) * 0.25,
                axes_point + (axes_point - curve_point) * 0.25,
                color=RED,
                stroke_width=2,
            )
            self.play(Create(tangent))
            self.play(self.cursor_dot.animate.move_to(self.axes.c2p(x_new, 0)))
            self.play(
                self.cursor_dot.animate.move_to(self.axes.c2p(x_new, self.f(x_new))),
                FadeOut(tangent),
            )
        return super().on_key_press(symbol, modifiers)
