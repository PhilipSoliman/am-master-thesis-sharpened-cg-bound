from manim import *


class Scene(Scene):
    def interactive_embed(self):
        if not config.write_to_movie:
            super().interactive_embed()
