#!/usr/bin/env python3
"""Reproduce the two stochastic Neumann boundary-control examples.

This is a self-contained reference implementation.  It uses Q1 finite
elements on a uniform unit-square mesh, a lumped-mass KL discretization of
the stated two-dimensional Gaussian covariance, fixed standard-normal
samples, and a primal-dual active-set Newton solve for the box constraints.
The linear active-set systems are solved by sparse direct factorization.
This is deliberately a reference solver: the paper's indefinite saddle
matrix must not be passed to ordinary Euclidean PCG.
"""

from __future__ import annotations

import argparse
import csv
import json
import struct
import time
import zlib
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse.linalg import spsolve


ROOT = Path(__file__).resolve().parent
FIGDIR = ROOT / "figures"


def _pdf_escape(text: str) -> str:
    return str(text).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _fmt(value: float) -> str:
    """Compact decimal formatting for axes and color bars."""
    if abs(value) >= 100 or (0 < abs(value) < 1e-3):
        return f"{value:.1e}"
    return f"{value:.3g}"


def _viridis(t: float) -> tuple[float, float, float]:
    """A compact, color-blind-friendly approximation of viridis."""
    stops = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    colors = np.array([[68, 1, 84], [59, 82, 139], [33, 145, 140],
                       [94, 201, 98], [183, 222, 48], [253, 231, 37]], dtype=float) / 255.0
    t = float(np.clip(t, 0.0, 1.0))
    return tuple(float(np.interp(t, stops, colors[:, k])) for k in range(3))


def _pdf_text(parts: list[str], x: float, y: float, text: str, size: float = 8.0,
              angle: float = 0.0, font: str = "F1") -> None:
    if angle:
        radians = np.deg2rad(angle)
        c, s = float(np.cos(radians)), float(np.sin(radians))
        matrix = f"{c:.5f} {s:.5f} {-s:.5f} {c:.5f} {x:.2f} {y:.2f}"
    else:
        matrix = f"1 0 0 1 {x:.2f} {y:.2f}"
    parts.append(f"0 0 0 rg BT /{font} {size:.2f} Tf {matrix} Tm ({_pdf_escape(text)}) Tj ET")


def _write_pdf(path: Path, width: float, height: float, content: str) -> None:
    """Write a single-page, vector PDF without external plotting packages."""
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width:.2f} {height:.2f}] "
         "/Resources << /Font << /F1 4 0 R /F2 5 0 R >> >> /Contents 6 0 R >>").encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",
        None,
    ]
    stream = content.encode("ascii", "replace")
    objects[5] = f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream"
    output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{index} 0 obj\n".encode())
        output.extend(obj)
        output.extend(b"\nendobj\n")
    xref = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    output.extend("".join(f"{offset:010d} 00000 n \n" for offset in offsets[1:]).encode())
    output.extend(f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
                  f"startxref\n{xref}\n%%EOF\n".encode())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(output)


def _draw_colorbar(parts: list[str], x: float, y: float, width: float, height: float,
                   vmin: float, vmax: float, label: str) -> None:
    bands = 48
    for k in range(bands):
        t0, t1 = k / bands, (k + 1) / bands
        r, g, b = _viridis((t0 + t1) / 2.0)
        yy = y + t0 * height
        parts.append(f"{r:.4f} {g:.4f} {b:.4f} rg {x:.2f} {yy:.2f} {width:.2f} "
                     f"{height / bands:.2f} re f")
    parts.append(f"0 0 0 RG 0.45 w {x:.2f} {y:.2f} {width:.2f} {height:.2f} re S")
    for t in (0.0, 0.5, 1.0):
        yy = y + t * height
        parts.append(f"0 0 0 RG 0.45 w {x + width:.2f} {yy:.2f} m {x + width + 3:.2f} {yy:.2f} l S")
        _pdf_text(parts, x + width + 6, yy - 2.5, _fmt(vmin + t * (vmax - vmin)), 7.0)
    _pdf_text(parts, x - 1, y + height + 8, label, 7.2)


def write_field_pdf(path: Path, values: np.ndarray, *, vmin=None, vmax=None,
                    title: str, colorbar_label: str) -> None:
    """Write a labeled vector heatmap with axes, ticks, and a color bar."""
    a = np.asarray(values, dtype=float)
    if a.ndim != 2:
        raise ValueError("field figure requires a 2-D array")
    finite = np.isfinite(a)
    vmin = float(np.nanmin(a[finite])) if vmin is None and np.any(finite) else (0.0 if vmin is None else vmin)
    vmax = float(np.nanmax(a[finite])) if vmax is None and np.any(finite) else (1.0 if vmax is None else vmax)
    if vmax <= vmin:
        vmax = vmin + 1.0
    width, height = 320.0, 250.0
    x0, y0, plot_w, plot_h = 42.0, 38.0, 190.0, 174.0
    cbx, cbw = 246.0, 11.0
    parts = ["1 J 1 j 0.25 w"]
    ny_cells, nx_cells = a.shape[0] - 1, a.shape[1] - 1
    for j in range(ny_cells):
        for i in range(nx_cells):
            value = float(np.mean(a[j:j + 2, i:i + 2]))
            r, g, b = _viridis((value - vmin) / (vmax - vmin))
            xx = x0 + i * plot_w / nx_cells
            yy = y0 + j * plot_h / ny_cells
            parts.append(f"{r:.4f} {g:.4f} {b:.4f} rg {xx:.3f} {yy:.3f} "
                         f"{plot_w / nx_cells + 0.15:.3f} {plot_h / ny_cells + 0.15:.3f} re f")
    parts.append(f"0 0 0 RG 0.65 w {x0:.2f} {y0:.2f} {plot_w:.2f} {plot_h:.2f} re S")
    for t in (0.0, 0.5, 1.0):
        xx = x0 + t * plot_w
        yy = y0 + t * plot_h
        parts.append(f"0 0 0 RG 0.45 w {xx:.2f} {y0:.2f} m {xx:.2f} {y0 - 3:.2f} l S")
        parts.append(f"0 0 0 RG 0.45 w {x0:.2f} {yy:.2f} m {x0 - 3:.2f} {yy:.2f} l S")
        _pdf_text(parts, xx - 4 if t else xx - 2, y0 - 14, _fmt(t), 7.0)
        _pdf_text(parts, x0 - 25, yy - 2.5, _fmt(t), 7.0)
    _pdf_text(parts, x0 + plot_w / 2 - 8, 15, "x1", 8.0)
    _pdf_text(parts, 10, y0 + plot_h / 2 - 12, "x2", 8.0, angle=90)
    _pdf_text(parts, x0, height - 17, title, 9.0, font="F2")
    _draw_colorbar(parts, cbx, y0, cbw, plot_h, vmin, vmax, colorbar_label)
    _write_pdf(path, width, height, "\n".join(parts))


def write_control_pdf(path: Path, values: np.ndarray, *, vmin=None, vmax=None,
                      title: str, segments: tuple[str, ...]) -> None:
    """Write a boundary-control strip with arclength and segment labels."""
    a = np.asarray(values, dtype=float).ravel()
    finite = np.isfinite(a)
    vmin = float(np.nanmin(a[finite])) if vmin is None and np.any(finite) else (0.0 if vmin is None else vmin)
    vmax = float(np.nanmax(a[finite])) if vmax is None and np.any(finite) else (1.0 if vmax is None else vmax)
    if vmax <= vmin:
        vmax = vmin + 1.0
    width, height = 320.0, 250.0
    x0, y0, plot_w, plot_h = 42.0, 78.0, 190.0, 84.0
    cbx, cbw = 246.0, 11.0
    parts = ["1 J 1 j 0.25 w", f"0.96 0.96 0.96 rg {x0:.2f} {y0:.2f} {plot_w:.2f} {plot_h:.2f} re f"]
    dx = plot_w / max(1, len(a))
    for i, value in enumerate(a):
        r, g, b = _viridis((value - vmin) / (vmax - vmin))
        parts.append(f"{r:.4f} {g:.4f} {b:.4f} rg {x0 + i * dx:.3f} {y0:.3f} {dx + 0.25:.3f} {plot_h:.3f} re f")
    parts.append(f"0 0 0 RG 0.65 w {x0:.2f} {y0:.2f} {plot_w:.2f} {plot_h:.2f} re S")
    for t in (0.0, 0.5, 1.0):
        xx = x0 + t * plot_w
        parts.append(f"0 0 0 RG 0.45 w {xx:.2f} {y0:.2f} m {xx:.2f} {y0 - 3:.2f} l S")
        _pdf_text(parts, xx - 4, y0 - 14, _fmt(t), 7.0)
    parts.append(f"0 0 0 RG 0.55 w {x0:.2f} {y0 - 17:.2f} m {x0 + plot_w:.2f} {y0 - 17:.2f} l S")
    _pdf_text(parts, x0 + plot_w / 2 - 28, y0 - 31, "normalized boundary coordinate s/|Gamma_u|", 7.5)
    if len(segments) > 1:
        for k, segment in enumerate(segments):
            boundary = x0 + k * plot_w / len(segments)
            if k:
                parts.append(f"0 0 0 RG 0.8 w [{2} {2}] 0 d {boundary:.2f} {y0:.2f} m {boundary:.2f} {y0 + plot_h:.2f} l S [] 0 d")
            _pdf_text(parts, boundary + plot_w / len(segments) / 2 - 7, y0 + plot_h + 10, segment, 7.5, font="F2")
    _pdf_text(parts, x0 + plot_w / 2 - 18, height - 17, title, 9.0, font="F2")
    _draw_colorbar(parts, cbx, y0, cbw, plot_h, vmin, vmax, "control value")
    _write_pdf(path, width, height, "\n".join(parts))


def q1_element_matrices(hx: float, hy: float) -> tuple[np.ndarray, np.ndarray]:
    """Return local stiffness and mass matrices for bilinear Q1 elements."""
    ke = np.zeros((4, 4))
    me = np.zeros((4, 4))
    g = 1.0 / np.sqrt(3.0)
    for xi in (-g, g):
        for eta in (-g, g):
            dxi = np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)]) / 4
            deta = np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)]) / 4
            grads = np.column_stack((2 * dxi / hx, 2 * deta / hy))
            shape = np.array([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta),
                              (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)]) / 4
            jac = hx * hy / 4
            ke += jac * grads @ grads.T
            me += jac * np.outer(shape, shape)
    return ke, me


class Q1BoundaryProblem:
    def __init__(self, nx: int = 12, ny: int = 12, sides: tuple[str, ...] = ("B1",)):
        self.nx, self.ny = nx, ny
        self.hx, self.hy = 1.0 / nx, 1.0 / ny
        self.coords = np.array([[i * self.hx, j * self.hy]
                                for j in range(ny + 1) for i in range(nx + 1)])
        self.n = self.coords.shape[0]
        self.sides = sides
        self.K, self.M = self._assemble_volume()
        self.nodes, self.C, self.R = self._assemble_control()
        self.mlump = np.asarray(self.M.sum(axis=1)).ravel()

    def idx(self, i: int, j: int) -> int:
        return j * (self.nx + 1) + i

    def _assemble_volume(self):
        K = sparse.lil_matrix((self.n, self.n))
        M = sparse.lil_matrix((self.n, self.n))
        ke, me = q1_element_matrices(self.hx, self.hy)
        for j in range(self.ny):
            for i in range(self.nx):
                ids = [self.idx(i, j), self.idx(i + 1, j),
                       self.idx(i + 1, j + 1), self.idx(i, j + 1)]
                for a in range(4):
                    for b in range(4):
                        K[ids[a], ids[b]] += ke[a, b]
                        M[ids[a], ids[b]] += me[a, b]
        edge = self.hx / 6.0 * np.array([[2.0, 1.0], [1.0, 2.0]])
        for i in range(self.nx):
            for j in (0, self.ny):
                ids = [self.idx(i, j), self.idx(i + 1, j)]
                for a in range(2):
                    for b in range(2):
                        K[ids[a], ids[b]] += edge[a, b]
        return K.tocsr(), M.tocsr()

    def _edge_nodes(self, side: str) -> list[int]:
        if side == "B1":
            return [self.idx(self.nx, j) for j in range(self.ny + 1)]
        if side == "B2":
            return [self.idx(0, j) for j in range(self.ny + 1)]
        raise ValueError(side)

    def _assemble_control(self):
        nodes = [node for side in self.sides for node in self._edge_nodes(side)]
        m = len(nodes)
        node_to_col = {node: k for k, node in enumerate(nodes)}
        C = sparse.lil_matrix((m, m))
        R = sparse.lil_matrix((self.n, m))
        edge = self.hy / 6.0 * np.array([[2.0, 1.0], [1.0, 2.0]])
        for side in self.sides:
            edge_nodes = self._edge_nodes(side)
            for a in range(len(edge_nodes) - 1):
                ids = edge_nodes[a:a + 2]
                cols = [node_to_col[x] for x in ids]
                for i in range(2):
                    for j in range(2):
                        C[cols[i], cols[j]] += edge[i, j]
                        R[ids[i], cols[j]] += edge[i, j]
        return nodes, C.tocsr(), R.tocsr()

    def covariance_modes(self, ell: float, sigma: float, N: int):
        d = self.coords[:, None, :] - self.coords[None, :, :]
        cov = sigma ** 2 * np.exp(-np.linalg.norm(d, axis=2) / ell)
        mass = np.maximum(self.mlump, 1e-14)
        root = np.sqrt(mass)
        sym = root[:, None] * cov * root[None, :]
        vals, vecs = eigh(sym, subset_by_index=(self.n - N, self.n - 1))
        order = np.argsort(vals)[::-1]
        vals, vecs = vals[order], vecs[:, order]
        modes = vecs / root[:, None]
        return vals, modes

    def target(self) -> np.ndarray:
        x, y = self.coords[:, 0], self.coords[:, 1]
        return (0.5 * (((0.2 <= x) & (x <= 0.4) | (0.6 <= x) & (x <= 0.8)) &
                       (0.4 <= y) & (y <= 0.8))).astype(float)

    def solve_active_set(self, f: np.ndarray, wd: np.ndarray, beta: float, c: float,
                         ua: float = 0.0, ub: float = 0.2, maxit: int = 40):
        n, m = self.n, len(self.nodes)
        K, M, C, R = self.K, self.M, self.C, self.R
        rhs = M @ f

        def solve_for(active: np.ndarray, bounds: np.ndarray):
            na = int(active.size)
            E = sparse.csc_matrix((np.ones(na), (active, np.arange(na))), shape=(m, na))
            z_nm = sparse.csc_matrix((n, m))
            z_na = sparse.csc_matrix((n, na))
            z_an = sparse.csc_matrix((na, n))
            z_aa = sparse.csc_matrix((na, na))
            A = sparse.bmat([[M, z_nm, K, z_na],
                             [z_nm.T, beta * C, -R.T, E],
                             [K, -R, None, z_na],
                             [z_an, E.T, z_an, None]], format="csc")
            b = np.concatenate([M @ wd, np.zeros(m), rhs, bounds])
            sol = spsolve(A, b)
            w = sol[:n]
            u = sol[n:n + m]
            lam = sol[n + m:2 * n + m]
            eta = np.zeros(m)
            eta[active] = sol[2 * n + m:]
            return w, u, lam, eta

        w, u, lam, eta = solve_for(np.array([], dtype=int), np.array([]))
        active = np.array([], dtype=int)
        for it in range(1, maxit + 1):
            upper = eta + c * (u - ub) > 1e-10
            lower = eta - c * (ua - u) < -1e-10
            lower &= ~upper
            new_active = np.flatnonzero(upper | lower)
            bounds = np.where(upper[new_active], ub, ua)
            w, u, lam, eta = solve_for(new_active, bounds)
            if np.array_equal(new_active, active):
                break
            active = new_active
        # Four-block KKT residual.  The first three blocks are the discrete
        # state, adjoint, and control-stationarity equations.  The fourth is
        # the projection form of complementarity.  This code uses the active
        # test eta + c*(u-bound), hence the matching projection is
        # u = clip(u + eta/c, ua, ub).
        state_vec = K @ w - R @ u - rhs
        adjoint_vec = M @ w + K @ lam - M @ wd
        control_vec = beta * C @ u - R.T @ lam + eta
        projected_u = np.clip(u + eta / c, ua, ub)
        complementarity_vec = u - projected_u

        state_res = float(np.linalg.norm(state_vec))
        adj_res = float(np.linalg.norm(adjoint_vec))
        ctrl_res = float(np.linalg.norm(control_vec))
        comp_res = float(np.linalg.norm(complementarity_vec))

        # Each block is scaled by one plus the sum of the norms of its terms.
        state_scale = 1.0 + np.linalg.norm(K @ w) + np.linalg.norm(R @ u) + np.linalg.norm(rhs)
        adj_scale = 1.0 + np.linalg.norm(M @ w) + np.linalg.norm(K @ lam) + np.linalg.norm(M @ wd)
        ctrl_scale = 1.0 + np.linalg.norm(beta * C @ u) + np.linalg.norm(R.T @ lam) + np.linalg.norm(eta)
        comp_scale = 1.0 + np.linalg.norm(u) + np.linalg.norm(projected_u)

        nr_state = float(state_res / state_scale)
        nr_adj = float(adj_res / adj_scale)
        nr_ctrl = float(ctrl_res / ctrl_scale)
        nr_comp = float(comp_res / comp_scale)
        # Euclidean norm of the four individually normalized residual blocks.
        full_kkt = float(np.sqrt(nr_state**2 + nr_adj**2 + nr_ctrl**2 + nr_comp**2))

        lower_violation = np.maximum(ua - u, 0.0)
        upper_violation = np.maximum(u - ub, 0.0)
        violation_vector = lower_violation + upper_violation
        box_violation_linf = float(np.max(violation_vector)) if violation_vector.size else 0.0
        box_violation_l2 = float(np.linalg.norm(violation_vector))

        return {"w": w, "u": u, "lambda": lam, "eta": eta,
                "active": new_active, "outer_iterations": it,
                "state_residual": state_res, "adjoint_residual": adj_res,
                "control_residual": ctrl_res, "complementarity_residual": comp_res,
                "normalized_state_residual": nr_state,
                "normalized_adjoint_residual": nr_adj,
                "normalized_control_residual": nr_ctrl,
                "normalized_complementarity_residual": nr_comp,
                "normalized_full_kkt_residual": full_kkt,
                "box_violation": box_violation_l2,
                "box_violation_l2": box_violation_l2,
                "box_violation_linf": box_violation_linf}


def edge_image(problem: Q1BoundaryProblem, values: np.ndarray) -> np.ndarray:
    a = np.zeros(problem.ny + 1)
    for j, node in enumerate(problem._edge_nodes("B1")):
        if node in problem.nodes:
            a[j] = values[problem.nodes.index(node)]
    return a


def save_example_figures(problem, name, mean_w, var_w, source, wd, mean_u, var_u, residual):
    nx, ny = problem.nx, problem.ny
    def field(x):
        return np.asarray(x).reshape(ny + 1, nx + 1)
    prefix = "Fig_1_" if name == "example1" else "Fig_2_"
    write_field_pdf(FIGDIR / f"{prefix}a.pdf", field(mean_w), vmin=0, vmax=0.5,
                    title="mean state", colorbar_label="state")
    write_field_pdf(FIGDIR / f"{prefix}b.pdf", field(wd), vmin=0, vmax=0.5,
                    title="target state", colorbar_label="target")
    write_field_pdf(FIGDIR / f"{prefix}c.pdf", field(var_w), vmin=0,
                    title="state variance", colorbar_label="variance")
    write_field_pdf(FIGDIR / f"{prefix}d.pdf", field(source), vmin=0,
                    title="source realization", colorbar_label="source")
    write_control_pdf(FIGDIR / f"{prefix}e.pdf", mean_u, vmin=0, vmax=0.2,
                      title="mean boundary control", segments=problem.sides)
    write_control_pdf(FIGDIR / f"{prefix}f.pdf", var_u, vmin=0,
                      title="control variance", segments=problem.sides)


def save_comparison_figures(first, second):
    """Write figures whose panels use the same data scales for both examples."""
    p = first["problem"]
    nx, ny = p.nx, p.ny

    def field(x):
        return np.asarray(x).reshape(ny + 1, nx + 1)

    mean_1, mean_2 = field(first["mean_w"]), field(second["mean_w"])
    var_1, var_2 = field(first["var_w"]), field(second["var_w"])
    err_1 = np.abs(mean_1 - field(first["wd"]))
    err_2 = np.abs(mean_2 - field(second["wd"]))
    mean_difference = np.abs(mean_2 - mean_1)
    variance_difference = np.abs(var_2 - var_1)
    error_max = max(float(err_1.max()), float(err_2.max()), 1e-12)
    mean_difference_max = max(float(mean_difference.max()), 1e-12)
    variance_difference_max = max(float(variance_difference.max()), 1e-12)

    write_field_pdf(FIGDIR / "Fig_3_a.pdf", mean_1, vmin=0, vmax=0.5,
                    title="mean state: control on B1", colorbar_label="state")
    write_field_pdf(FIGDIR / "Fig_3_b.pdf", mean_2, vmin=0, vmax=0.5,
                    title="mean state: control on B1+B2", colorbar_label="state")
    write_field_pdf(FIGDIR / "Fig_3_c.pdf", err_1, vmin=0, vmax=error_max,
                    title="absolute tracking error: B1", colorbar_label="abs. error")
    write_field_pdf(FIGDIR / "Fig_3_d.pdf", err_2, vmin=0, vmax=error_max,
                    title="absolute tracking error: B1+B2", colorbar_label="abs. error")
    write_field_pdf(FIGDIR / "Fig_4_a.pdf", mean_difference, vmin=0, vmax=mean_difference_max,
                    title="change in mean state", colorbar_label="abs. difference")
    write_field_pdf(FIGDIR / "Fig_4_b.pdf", variance_difference, vmin=0, vmax=variance_difference_max,
                    title="change in state variance", colorbar_label="abs. difference")


def run_example(name, sides, ell, sigma, beta, c, normals, nx, ny, N,
                covariance_data=None, save_figures=True):
    problem = Q1BoundaryProblem(nx=nx, ny=ny, sides=sides)
    if covariance_data is None:
        lambdas, modes = problem.covariance_modes(ell, sigma, N)
    else:
        lambdas, modes = covariance_data
    wd = problem.target()
    mean_source = 0.1 * (((0.2 <= problem.coords[:, 0]) & (problem.coords[:, 0] <= 0.8))).astype(float)
    ws, us, active_counts, residuals, sample_metrics = [], [], [], [], []
    t0 = time.perf_counter()
    source0 = None
    for k, z in enumerate(normals):
        source = mean_source + modes @ (np.sqrt(np.maximum(lambdas, 0)) * z[:N])
        if source0 is None:
            source0 = source.copy()
        out = problem.solve_active_set(source, wd, beta=beta, c=c)
        ws.append(out["w"])
        us.append(out["u"])
        active_counts.append(out["outer_iterations"])
        residuals.append(out)
        sample_metrics.append({
            "sample": k + 1,
            "newton_iterations": out["outer_iterations"],
            "active_nodes": int(len(out["active"])),
            "normalized_state_residual": out["normalized_state_residual"],
            "normalized_adjoint_residual": out["normalized_adjoint_residual"],
            "normalized_control_residual": out["normalized_control_residual"],
            "normalized_complementarity_residual": out["normalized_complementarity_residual"],
            "normalized_full_kkt_residual": out["normalized_full_kkt_residual"],
            "box_violation_linf": out["box_violation_linf"],
            "box_violation_l2": out["box_violation_l2"],
        })
    elapsed = time.perf_counter() - t0
    W, U = np.asarray(ws), np.asarray(us)
    mean_w, var_w = W.mean(axis=0), W.var(axis=0)
    mean_u, var_u = U.mean(axis=0), U.var(axis=0)
    tracking = float(np.mean([((w - wd) @ (problem.M @ (w - wd))) for w in W]))
    regularization = float(beta * np.mean([u @ (problem.C @ u) for u in U]))
    max_kkt_residual = float(max(x["normalized_full_kkt_residual"] for x in residuals))
    mean_kkt_residual = float(np.mean([x["normalized_full_kkt_residual"] for x in residuals]))
    row = {"example": name, "samples": len(normals), "nx": nx, "ny": ny, "N": N,
           "beta": beta, "c": c, "ell": ell, "sigma": sigma,
           "retained_variance": float(lambdas.sum()),
           "retained_ratio": float(lambdas.sum() / (sigma ** 2)),
           "tracking_error": tracking,
           "regularization_term": regularization, "objective": 0.5 * (tracking + regularization),
           "max_box_violation_linf": float(max(x["box_violation_linf"] for x in residuals)),
           "max_box_violation_l2": float(max(x["box_violation_l2"] for x in residuals)),
           "mean_box_violation_linf": float(np.mean([x["box_violation_linf"] for x in residuals])),
           "max_kkt_residual": max_kkt_residual,
           "mean_kkt_residual": mean_kkt_residual,
           "mean_newton_iterations": float(np.mean(active_counts)),
           "max_newton_iterations": int(max(active_counts)),
           "seconds": elapsed}
    if save_figures:
        save_example_figures(problem, name, mean_w, var_w, source0, wd, mean_u, var_u,
                             np.abs(mean_w - wd))
    artifacts = {"problem": problem, "mean_w": mean_w, "var_w": var_w,
                 "source": source0, "wd": wd, "mean_u": mean_u, "var_u": var_u,
                 "residual": np.abs(mean_w - wd), "sample_metrics": sample_metrics}
    return row, lambdas, artifacts


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(rows[0])
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def restrict_field(values: np.ndarray, fine_n: int, coarse_n: int) -> np.ndarray:
    stride = fine_n // coarse_n
    return np.asarray(values).reshape(fine_n + 1, fine_n + 1)[::stride, ::stride].ravel()


def boundary_restriction(values: np.ndarray, fine_n: int, coarse_n: int, sides: tuple[str, ...]) -> np.ndarray:
    stride = fine_n // coarse_n
    block = fine_n + 1
    out = []
    for k in range(len(sides)):
        out.append(np.asarray(values)[k * block:(k + 1) * block][::stride])
    return np.concatenate(out)


def rms_difference(coarse: np.ndarray, fine: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(coarse) - np.asarray(fine)) /
                 np.sqrt(np.asarray(coarse).size))


def run_studies(normals: np.ndarray, samples: int, N_main: int) -> dict:
    """Run paired h, KL, variance, beta, and c studies with fixed samples."""
    sides_by_name = {"example1": ("B1",), "example2": ("B1", "B2")}
    mesh_levels = [6, 12, 24]
    deterministic_normals = np.zeros((1, 0), dtype=float)
    mesh_rows, mesh_rates = [], []
    mesh_artifacts = {name: {} for name in sides_by_name}
    mesh_results = {name: {} for name in sides_by_name}
    for nx in mesh_levels:
        problem = Q1BoundaryProblem(nx=nx, ny=nx)
        deterministic_cov = (np.zeros(0), np.zeros((problem.n, 0)))
        for name, sides in sides_by_name.items():
            row, _, artifact = run_example(name, sides, 1.0, 0.9, 0.2, 0.9,
                                            normals[:, :0], nx, nx, 0,
                                            deterministic_cov, save_figures=False)
            mesh_artifacts[name][nx] = artifact
            mesh_results[name][nx] = row
    for name, sides in sides_by_name.items():
        ref = mesh_artifacts[name][mesh_levels[-1]]
        for nx in mesh_levels:
            artifact = mesh_artifacts[name][nx]
            row = mesh_results[name][nx]
            state_error = 0.0 if nx == 24 else rms_difference(
                artifact["mean_w"], restrict_field(ref["mean_w"], 24, nx))
            control_error = 0.0 if nx == 24 else rms_difference(
                artifact["mean_u"], boundary_restriction(ref["mean_u"], 24, nx, sides))
            mesh_rows.append({"example": name, "nx": nx, "elements": nx * nx,
                              "state_nodes": (nx + 1) ** 2,
                              "state_error_to_24": state_error,
                              "control_error_to_24": control_error,
                              "tracking_error": row["tracking_error"],
                              "objective": row["objective"],
                              "max_box_violation_linf": row["max_box_violation_linf"],
                              "max_kkt_residual": row["max_kkt_residual"],
                              "mean_newton_iterations": row["mean_newton_iterations"]})
        e6 = next(r["state_error_to_24"] for r in mesh_rows
                  if r["example"] == name and r["nx"] == 6)
        e12 = next(r["state_error_to_24"] for r in mesh_rows
                   if r["example"] == name and r["nx"] == 12)
        c6 = next(r["control_error_to_24"] for r in mesh_rows
                  if r["example"] == name and r["nx"] == 6)
        c12 = next(r["control_error_to_24"] for r in mesh_rows
                   if r["example"] == name and r["nx"] == 12)
        mesh_rates.append({"example": name,
                           "state_observed_rate": np.log(e6 / e12) / np.log(2.0),
                           "control_observed_rate": np.log(c6 / c12) / np.log(2.0),
                           "state_error_6_to_24": e6,
                           "state_error_12_to_24": e12})

    ref_problem = Q1BoundaryProblem(nx=24, ny=24)
    kl_all = ref_problem.covariance_modes(1.0, 0.9, 7)
    kl_rows, kl_artifacts = [], {name: {} for name in sides_by_name}
    for N in [7, 1, 3, 5]:
        cov = (kl_all[0][:N], kl_all[1][:, :N])
        for name, sides in sides_by_name.items():
            row, _, artifact = run_example(name, sides, 1.0, 0.9, 0.2, 0.9,
                                            normals[:, :N], 24, 24, N, cov,
                                            save_figures=False)
            kl_artifacts[name][N] = artifact
            kl_rows.append({"example": name, "N": N,
                            "retained_variance": row["retained_variance"],
                            "retained_ratio": row["retained_ratio"],
                            "state_error_to_N7": 0.0 if N == 7 else rms_difference(
                                artifact["mean_w"], kl_artifacts[name][7]["mean_w"]),
                            "tracking_error": row["tracking_error"],
                            "objective": row["objective"],
                            "max_kkt_residual": row["max_kkt_residual"]})
    kl_rates = []
    for name in sides_by_name:
        vals = {r["N"]: r for r in kl_rows if r["example"] == name}
        for low, high in [(1, 3), (3, 5)]:
            kl_rates.append({"example": name, "N_low": low, "N_high": high,
                             "observed_rate": np.log(vals[low]["state_error_to_N7"] /
                                                       vals[high]["state_error_to_N7"]) /
                                               np.log(high / low)})

    variance_rows = []
    base_cov = ref_problem.covariance_modes(1.0, 1.0, N_main)
    for sigma in [0.3, 0.6, 0.9, 1.2]:
        cov = (base_cov[0] * sigma ** 2, base_cov[1])
        for name, sides in sides_by_name.items():
            row, _, _ = run_example(name, sides, 1.0, sigma, 0.2, 0.9,
                                     normals[:, :N_main], 24, 24, N_main, cov,
                                     save_figures=False)
            variance_rows.append({"example": name, "sigma": sigma,
                                  "pointwise_variance_scale": sigma ** 2,
                                  "retained_variance": row["retained_variance"],
                                  "retained_ratio": row["retained_ratio"],
                                  "tracking_error": row["tracking_error"],
                                  "regularization_term": row["regularization_term"],
                                  "objective": row["objective"],
                                  "max_kkt_residual": row["max_kkt_residual"]})

    sensitivity_rows = []
    cov = ref_problem.covariance_modes(1.0, 0.9, N_main)
    for beta in [0.02, 0.2, 2.0]:
        for name, sides in sides_by_name.items():
            row, _, _ = run_example(name, sides, 1.0, 0.9, beta, 0.9,
                                     normals[:, :N_main], 24, 24, N_main, cov,
                                     save_figures=False)
            sensitivity_rows.append({"example": name, "study": "beta", "value": beta,
                                     "beta": beta, "c": 0.9,
                                     "tracking_error": row["tracking_error"],
                                     "regularization_term": row["regularization_term"],
                                     "objective": row["objective"],
                                     "mean_newton_iterations": row["mean_newton_iterations"],
                                     "max_kkt_residual": row["max_kkt_residual"]})
    for c in [0.1, 0.9, 1.5]:
        for name, sides in sides_by_name.items():
            row, _, _ = run_example(name, sides, 1.0, 0.9, 0.2, c,
                                     normals[:, :N_main], 24, 24, N_main, cov,
                                     save_figures=False)
            sensitivity_rows.append({"example": name, "study": "c", "value": c,
                                     "beta": 0.2, "c": c,
                                     "tracking_error": row["tracking_error"],
                                     "regularization_term": row["regularization_term"],
                                     "objective": row["objective"],
                                     "mean_newton_iterations": row["mean_newton_iterations"],
                                     "max_kkt_residual": row["max_kkt_residual"]})

    write_csv(ROOT / "mesh_convergence.csv", mesh_rows)
    write_csv(ROOT / "mesh_convergence_rates.csv", mesh_rates)
    write_csv(ROOT / "kl_convergence.csv", kl_rows)
    write_csv(ROOT / "kl_convergence_rates.csv", kl_rates)
    write_csv(ROOT / "variance_sweep.csv", variance_rows)
    kl_rows.sort(key=lambda r: (r["example"], r["N"]))
    write_csv(ROOT / "beta_c_sensitivity.csv", sensitivity_rows)
    return {"mesh": mesh_rows, "mesh_rates": mesh_rates, "kl": kl_rows,
            "kl_rates": kl_rates, "variance": variance_rows,
            "sensitivity": sensitivity_rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--nx", type=int, default=24)
    ap.add_argument("--ny", type=int, default=24)
    ap.add_argument("--N", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20261017)
    ap.add_argument("--core-only", action="store_true",
                    help="compute baseline sample-wise KKT/box metrics and deterministic h-refinement only")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    normals = rng.standard_normal((args.samples, max(args.N, 7)))
    settings = vars(args) | {"solver": "sparse_direct_active_set", "target": "two rectangular bumps",
                              "covariance": "sigma^2 exp(-||x-x'||_2/ell)",
                              "figures": str(FIGDIR)}
    rows = []
    eigenvalues = {}
    reference_problem = Q1BoundaryProblem(nx=args.nx, ny=args.ny)
    covariance_data = reference_problem.covariance_modes(1.0, 0.9, args.N)
    artifacts = []
    for spec in [("example1", ("B1",)), ("example2", ("B1", "B2"))]:
        row, vals, artifact = run_example(spec[0], spec[1], 1.0, 0.9, 0.2, 0.9,
                                    normals, args.nx, args.ny, args.N, covariance_data)
        rows.append(row)
        eigenvalues[spec[0]] = vals.tolist()
        artifacts.append(artifact)
    save_comparison_figures(artifacts[0], artifacts[1])
    sample_rows = []
    for artifact, spec in zip(artifacts, ["example1", "example2"]):
        for metric in artifact["sample_metrics"]:
            sample_rows.append({"example": spec, **metric})
    write_csv(ROOT / "samplewise_kkt_box_metrics.csv", sample_rows)
    out_csv = ROOT / "reproduction_metrics.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if args.core_only:
        # The deterministic refinement table is evaluated with one
        # deterministic solve per mesh (not duplicated as repeated samples).
        mesh_levels = [6, 12, 24]
        deterministic_normals = np.zeros((1, 0), dtype=float)
        mesh_artifacts = {"example1": {}, "example2": {}}
        mesh_rows = []
        mesh_rates = []
        for name, sides in [("example1", ("B1",)), ("example2", ("B1", "B2"))]:
            for nmesh in mesh_levels:
                problem = Q1BoundaryProblem(nx=nmesh, ny=nmesh)
                zero_covariance = (np.zeros(0), np.zeros((problem.n, 0)))
                row, _, artifact = run_example(
                    name, sides, 1.0, 0.9, 0.2, 0.9,
                    deterministic_normals, nmesh, nmesh, 0,
                    zero_covariance, save_figures=False)
                mesh_artifacts[name][nmesh] = (row, artifact)
            ref_row, ref = mesh_artifacts[name][24]
            for nmesh in mesh_levels:
                row, artifact = mesh_artifacts[name][nmesh]
                state_error = 0.0 if nmesh == 24 else rms_difference(
                    artifact["mean_w"], restrict_field(ref["mean_w"], 24, nmesh))
                control_error = 0.0 if nmesh == 24 else rms_difference(
                    artifact["mean_u"], boundary_restriction(ref["mean_u"], 24, nmesh, sides))
                mesh_rows.append({
                    "example": name, "nx": nmesh, "elements": nmesh * nmesh,
                    "state_nodes": (nmesh + 1) ** 2,
                    "state_error_to_24": state_error,
                    "control_error_to_24": control_error,
                    "tracking_error": row["tracking_error"],
                    "objective": row["objective"],
                    "max_box_violation_linf": row["max_box_violation_linf"],
                    "max_kkt_residual": row["max_kkt_residual"],
                    "mean_newton_iterations": row["mean_newton_iterations"],
                })
            by_n = {r["nx"]: r for r in mesh_rows if r["example"] == name}
            mesh_rates.append({
                "example": name,
                "state_observed_rate": float(np.log(by_n[6]["state_error_to_24"] /
                                                     by_n[12]["state_error_to_24"]) / np.log(2.0)),
                "control_observed_rate": float(np.log(by_n[6]["control_error_to_24"] /
                                                       by_n[12]["control_error_to_24"]) / np.log(2.0)),
            })
        write_csv(ROOT / "mesh_convergence.csv", mesh_rows)
        write_csv(ROOT / "mesh_convergence_rates.csv", mesh_rates)
        settings["studies"] = {"mesh_levels": mesh_levels, "mode": "core-only deterministic h refinement"}
        (ROOT / "reproduction_settings.json").write_text(json.dumps(settings, indent=2) + "\n")
        (ROOT / "kl_eigenvalues.json").write_text(json.dumps(eigenvalues, indent=2) + "\n")
        print(json.dumps({"settings": settings, "results": rows,
                          "mesh": mesh_rows, "mesh_rates": mesh_rates,
                          "samplewise_metrics_file": "samplewise_kkt_box_metrics.csv"}, indent=2))
        return

    studies = run_studies(normals, args.samples, args.N)
    settings["studies"] = {
        "mesh_levels": [6, 12, 24],
        "kl_levels": [1, 3, 5, 7],
        "sigma_levels": [0.3, 0.6, 0.9, 1.2],
        "beta_levels": [0.02, 0.2, 2.0],
        "c_levels": [0.1, 0.9, 1.5],
        "study_samples": args.samples,
    }
    (ROOT / "reproduction_settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    (ROOT / "kl_eigenvalues.json").write_text(json.dumps(eigenvalues, indent=2) + "\n")
    print(json.dumps({"settings": settings, "results": rows, "eigenvalues": eigenvalues,
                      "samplewise_metrics_file": "samplewise_kkt_box_metrics.csv",
                      "study_files": ["mesh_convergence.csv", "mesh_convergence_rates.csv",
                                      "kl_convergence.csv", "kl_convergence_rates.csv",
                                      "variance_sweep.csv", "beta_c_sensitivity.csv"]}, indent=2))


if __name__ == "__main__":
    main()
