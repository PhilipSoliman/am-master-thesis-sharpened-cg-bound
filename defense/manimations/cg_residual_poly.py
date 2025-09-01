import os

import numpy as np

os.environ["NO_HCMSFEM_CLI_ARGS"] = ""
from hcmsfem.plot_utils import CustomColors
from hcmsfem.solvers import CustomCG

pass  # manim should be imported last

from manim_slides import Slide
from manimlib import *
from manimlib.config import manim_config

# Manim render settings
FPS = 60
QUALITY = (1920, 1080)  # 4k = (3840,2160)
manim_config.camera.fps = FPS
manim_config.camera.resolution = QUALITY

# plot settings
DOMAIN = (0.0, 1.01, 0.2)
DOMAIN_RANGE = DOMAIN[1] - DOMAIN[0]
CODOMAIN = (-1.5, 1.51, 0.3)
CODOMAIN_RANGE = CODOMAIN[1] - CODOMAIN[0]
HEIGHT = 7
WIDTH = 10
NUM_POINTS = 100
X_VALUES = np.linspace(DOMAIN[0], DOMAIN[1], NUM_POINTS)
SPECTRA_TRANSITION_TIME = 1.5

# initiate spectra
NUM_SPECTRA = 15
NUM_CLUSTERS = 3
EIGV_SAMPLE_RANGE = (
    0.01,
    1.0,
)
PROBLEM_SIZE = 300
RNG = np.random.default_rng(42)
RHS = RNG.random(PROBLEM_SIZE)
X0 = np.zeros(PROBLEM_SIZE)


def generate_spectra(
    eigenvalue_range: tuple[float, float],
    cluster_width_limits: tuple[float, float],
    num_clusters: int,
    num_spectra: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    spectra = []
    matrices = []
    n_interior_eigenvalues = PROBLEM_SIZE - 2  # reserving 2 for min and max eigenvalues
    for _ in range(num_spectra):
        n_interior_eigenvalues = (
            PROBLEM_SIZE - 2
        )  # reserving 2 for min and max eigenvalues
        clusters = []
        for i in range(num_clusters):
            # determine number of eigenvalues in this cluster
            if i < num_clusters - 1:
                n_cluster_eigenvalues = int(n_interior_eigenvalues * RNG.random(1))
            else:
                n_cluster_eigenvalues = n_interior_eigenvalues

            # construct cluster
            cluster_width = RNG.uniform(*cluster_width_limits)
            min_eig = eigenvalue_range[0] + cluster_width
            max_eig = eigenvalue_range[1] - cluster_width
            cluster_center = RNG.uniform(min_eig, max_eig)
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
    spectra: list[np.ndarray], diagonal_matrices: list[np.ndarray]
) -> list[np.poly1d]:
    residual_polys = []
    for spectrum, matrix in zip(spectra, diagonal_matrices):
        cg = CustomCG(matrix, RHS, X0, tol=1e-8, maxiter=PROBLEM_SIZE)
        x_exact = RHS / spectrum
        _, _ = cg.solve(x_exact=x_exact)
        cg.residual_polynomials()
        coeffs = cg.residual_polynomials_coefficients[-1]
        rp = np.poly1d(coeffs)
        residual_polys.append(rp)
    return residual_polys


class cg_residual_poly(Slide):
    def construct(self):
        # slide: create axes
        self.next_slide()
        self.axes = Axes(
            x_range=DOMAIN,
            y_range=CODOMAIN,
            height=HEIGHT,
            width=WIDTH,
            axis_config={
                "stroke_color": CustomColors.RED.value,
                "stroke_width": 2,
                "include_tip": True,
            },
        )
        self.x_label = Tex(r"\lambda").move_to(
            self.axes.c2p(DOMAIN[1] + 0.1 * DOMAIN_RANGE, CODOMAIN[0] + 0.5 * CODOMAIN_RANGE)
        )
        self.y_label = Tex(r"r(\lambda)").move_to(
            self.axes.c2p(DOMAIN[0] - 0.15 * DOMAIN_RANGE, 0.8 * CODOMAIN[1])
        )
        self.axes.add_coordinate_labels(
            font_size=20,
            num_decimal_places=1,
        )
        self.play(
            ShowCreation(self.axes),
            ShowCreation(self.x_label),
            ShowCreation(self.y_label),
        )

        # generate spectra and polynomials for two clusters
        spectra, matrices = generate_spectra(
            EIGV_SAMPLE_RANGE,
            (1e-2, 0.2),
            num_clusters=NUM_CLUSTERS,
            num_spectra=NUM_SPECTRA,
        )
        cg_residual_polys = generate_cg_residual_polys(spectra, matrices)

        # plot the first spectrum and its residual polynomial
        self.plot_spectrum_and_poly(spectra[0], cg_residual_polys[0])

        # slide: loop through spectra and polys
        self.next_slide(loop=True)
        for spectrum, cg_residual_poly in zip(spectra[1:], cg_residual_polys[1:]):
            self.plot_spectrum_and_poly(spectrum, cg_residual_poly)
        self.plot_spectrum_and_poly(spectra[0], cg_residual_polys[0])

        # slide: fade out previous graphs
        self.next_slide()
        self.play(FadeOut(VGroup(self.cg_residual_poly_graph, self.spectrum_group)))

        # # Similarly, you can call self.axes.point_to_coords, or self.axes.p2c
        # # print(self.axes.p2c(dot.get_center()))

        # # We can draw lines from the self.axes to better mark the coordinates
        # # of a given point.
        # # Here, the always_redraw command means that on each new frame
        # # the lines will be redrawn
        # h_line = always_redraw(lambda: self.axes.get_h_line(dot.get_left()))
        # v_line = always_redraw(lambda: self.axes.get_v_line(dot.get_bottom()))

        # self.play(
        #     ShowCreation(h_line),
        #     ShowCreation(v_line),
        # )
        # self.play(dot.animate.move_to(self.axes.c2p(3, -2)))
        # self.next_slide()
        # self.play(dot.animate.move_to(self.axes.c2p(1, 1)))

        # self.next_slide()
        # # If we tie the dot to a particular set of coordinates, notice
        # # that as we move the self.axes around it respects the coordinate
        # # system defined by them.
        # f_always(dot.move_to, lambda: self.axes.c2p(1, 1))
        # self.play(
        #     self.axes.animate.scale(0.75).to_corner(UL),
        #     run_time=2,
        # )

        # self.next_slide()
        # self.play(FadeOut(VGroup(self.axes, dot, h_line, v_line)))
        # self.play(FadeOut(VGroup(self.axes, dot, h_line, v_line)))
        # self.play(FadeOut(VGroup(self.axes, dot, h_line, v_line)))
        # self.play(FadeOut(VGroup(self.axes, dot, h_line, v_line)))
        # self.play(FadeOut(VGroup(self.axes, dot, h_line, v_line)))
        # self.play(FadeOut(VGroup(self.axes, dot, h_line, v_line)))

    def plot_spectrum_and_poly(self, spectrum: np.ndarray, cg_residual_poly: np.poly1d):
        # create dots from spectrum
        spectrum_dots = [
            Dot(
                self.axes.c2p(eig, 0),
                fill_color=CustomColors.BLUE.value,
                radius=DEFAULT_DOT_RADIUS,
            )
            for eig in spectrum
        ]
        spectrum_group = VGroup(*spectrum_dots)

        # create graph from cg_residual_poly
        cg_residual_poly_graph = self.axes.get_graph(
            lambda x: cg_residual_poly(x),
            x_range=(DOMAIN[0], DOMAIN[1], DOMAIN[1] / NUM_POINTS),
            color=CustomColors.GOLD.value,
        )

        # get degree of polynomial (= number of CG iterations)
        degree = cg_residual_poly.coefficients.shape[0] - 1
        y_label = self.axes.get_y_axis_label(f"r_{{{degree}}}(\\lambda)").move_to(
            self.y_label.get_center()
        )

        # add degree in seperate texbox
        degree_tex = Tex(f"m = {degree}")
        degree_tex.move_to(
            self.axes.c2p(DOMAIN[0] + 0.5 * DOMAIN_RANGE, 0.9 * CODOMAIN[1])
        )

        if not hasattr(self, "cg_residual_poly_graph"):
            self.play(
                FadeIn(cg_residual_poly_graph, scale=0.5),
                FadeIn(spectrum_group, scale=0.5),
                ReplacementTransform(self.y_label, y_label),
                FadeIn(degree_tex, scale=0.5),
                run_time=SPECTRA_TRANSITION_TIME,
            )
        else:
            self.play(
                ReplacementTransform(
                    self.cg_residual_poly_graph, cg_residual_poly_graph
                ),
                ReplacementTransform(self.spectrum_group, spectrum_group),
                ReplacementTransform(self.y_label, y_label),
                ReplacementTransform(self.degree_tex, degree_tex),
                run_time=SPECTRA_TRANSITION_TIME,
            )

        # update mobjects
        self.cg_residual_poly_graph = cg_residual_poly_graph
        self.spectrum_group = spectrum_group
        self.y_label = y_label
        self.degree_tex = degree_tex
