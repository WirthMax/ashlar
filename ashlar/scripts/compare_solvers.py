"""Compare ASHLAR's spanning-tree and global least-squares position solvers.

The two solvers are compared on the *identical* cached pairwise alignments:
alignment / phase-correlation is run exactly once, then MST positions and
global-solve positions are both derived from that single cached state. This
isolates the position solver as the only changed variable and keeps each
comparison fast.

Two input modes are supported:

  * ``--input``      real image data, read through the same reader path as the
                     ``ashlar`` CLI.
  * ``--synthetic``  a generated tiled dataset with known ground-truth tile
                     positions, so absolute error is measurable. Optionally
                     degrades a fraction of tiles (blur / blank) to emulate
                     out-of-focus or low-signal regions.
"""

import sys
import argparse
import pathlib

import numpy as np
import scipy.ndimage
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .. import reg


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

class SyntheticMetadata(reg.Metadata):
    """In-memory metadata for a synthetic tile grid (no BioFormats/Java)."""

    def __init__(self, tiles, positions, pixel_size=1.0):
        self._tiles = tiles
        self._recorded = np.asarray(positions, dtype=float)
        self._pixel_size = pixel_size

    @property
    def _num_images(self):
        return len(self._tiles)

    @property
    def num_channels(self):
        return 1

    @property
    def pixel_size(self):
        return self._pixel_size

    @property
    def pixel_dtype(self):
        return np.uint16

    def tile_position(self, i):
        return self._recorded[i]

    def tile_size(self, i):
        return np.array(self._tiles[i].shape, dtype=int)


class SyntheticReader(reg.Reader):
    """In-memory reader serving pre-rendered synthetic tiles."""

    def __init__(self, tiles, positions, pixel_size=1.0):
        self.tiles = [np.ascontiguousarray(t) for t in tiles]
        self.metadata = SyntheticMetadata(self.tiles, positions, pixel_size)

    def read(self, series, c):
        return self.tiles[series]


def render_canvas(shape, n_blobs, rng):
    """Render a canvas of random Gaussian "nucleus-like" blobs."""
    canvas = np.zeros(shape, dtype=np.float64)
    ys = rng.uniform(0, shape[0], n_blobs)
    xs = rng.uniform(0, shape[1], n_blobs)
    for y, x in zip(ys, xs):
        sigma = rng.uniform(3, 9)
        amp = rng.uniform(0.3, 1.0)
        r = int(np.ceil(sigma * 3))
        y0, y1 = int(max(0, y - r)), int(min(shape[0], y + r + 1))
        x0, x1 = int(max(0, x - r)), int(min(shape[1], x + r + 1))
        if y0 >= y1 or x0 >= x1:
            continue
        gy, gx = np.mgrid[y0:y1, x0:x1]
        canvas[y0:y1, x0:x1] += amp * np.exp(
            -(((gy - y) ** 2 + (gx - x) ** 2) / (2 * sigma ** 2))
        )
    return canvas


def make_synthetic(grid, overlap, jitter, degrade_frac, seed, tile_size=256):
    """Build a synthetic tiled dataset with known ground-truth positions.

    Returns ``(reader, true_positions, degraded_mask)`` where ``true_positions``
    are the exact (noise-free) tile origins the tiles were cut from, and the
    reader's metadata holds the jittered "recorded stage" positions.
    """
    rng = np.random.RandomState(seed)
    rows, cols = grid
    step = tile_size * (1.0 - overlap)
    margin = tile_size
    canvas_shape = (
        int(margin * 2 + step * (rows - 1) + tile_size),
        int(margin * 2 + step * (cols - 1) + tile_size),
    )
    n_blobs = int(canvas_shape[0] * canvas_shape[1] / 1500)
    canvas = render_canvas(canvas_shape, n_blobs, rng)
    canvas += rng.normal(0, 0.01, canvas_shape).clip(0, None)

    true_positions = []
    recorded_positions = []
    tiles = []
    n = rows * cols
    degrade_idx = set(
        rng.choice(n, int(round(degrade_frac * n)), replace=False).tolist()
    ) if degrade_frac > 0 else set()
    degraded_mask = np.zeros(n, dtype=bool)

    for idx in range(n):
        r, c = divmod(idx, cols)
        ty = margin + r * step
        tx = margin + c * step
        # Cut tile from canvas at the integer true position.
        iy, ix = int(round(ty)), int(round(tx))
        tile = canvas[iy:iy + tile_size, ix:ix + tile_size].copy()
        true_positions.append([float(iy), float(ix)])
        # Recorded (stage) position carries per-tile jitter error.
        jit = rng.uniform(-jitter, jitter, 2)
        recorded_positions.append([iy + jit[0], ix + jit[1]])
        if idx in degrade_idx:
            degraded_mask[idx] = True
            tile = scipy.ndimage.gaussian_filter(tile, sigma=6)
            if rng.rand() < 0.5:
                tile *= 0.05  # near-blank, low-signal region
        # Scale to uint16.
        m = tile.max()
        if m > 0:
            tile = tile / m
        tiles.append((tile * 60000).astype(np.uint16))

    reader = SyntheticReader(tiles, recorded_positions, pixel_size=1.0)
    return reader, np.array(true_positions), degraded_mask


# ---------------------------------------------------------------------------
# Shared single-pass alignment, then both solvers
# ---------------------------------------------------------------------------

def align_once(reader, channel, max_shift, alpha, filter_sigma, weight,
               anchor_lambda, verbose):
    """Run alignment + pruning exactly once and return the prepared aligner.

    The permutation-test RNG is forced deterministic (``randomize=False``) so
    pruning is reproducible across runs.
    """
    aligner = reg.EdgeAligner(
        reader, channel=channel, max_shift=max_shift, alpha=alpha,
        filter_sigma=filter_sigma, randomize=False, do_make_thumbnail=False,
        position_weight=weight, anchor_lambda=anchor_lambda, verbose=verbose,
    )
    aligner.make_thumbnail()
    aligner.check_overlaps()
    aligner.compute_threshold()
    aligner.register_all()
    aligner.build_spanning_tree()
    return aligner


def pruned_edges(aligner):
    """List of ``(t1, t2, shift, error)`` for surviving (finite-error) edges."""
    out = []
    for (t1, t2), (shift, error) in aligner._cache.items():
        if np.isfinite(error):
            out.append((t1, t2, np.asarray(shift, dtype=float), error))
    return out


def weighted_sse(aligner, shifts, edges):
    """Total weighted SSE of edge residuals r_ij = s_j - s_i - c_ij."""
    total = 0.0
    residuals = {}
    for t1, t2, shift, error in edges:
        r = shifts[t2] - shifts[t1] - shift
        w = aligner._edge_weight(error)
        residuals[(t1, t2)] = r
        total += w * float(r @ r)
    return total, residuals


def tree_path_lengths(aligner):
    """Per-tile graph distance from its component root in the spanning tree."""
    lengths = np.full(aligner.metadata.num_images, np.nan)
    st = aligner.spanning_tree
    for component in nx.connected_components(st):
        cc = st.subgraph(component)
        center = nx.center(cc)[0]
        for node, d in nx.shortest_path_length(cc, center).items():
            lengths[node] = d
    return lengths


def centroid_align(positions, reference):
    """Translate ``positions`` to minimize SSE vs ``reference`` (gauge match)."""
    return positions - positions.mean(axis=0) + reference.mean(axis=0)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def summarize(name, values):
    v = np.asarray(values)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return dict(max=np.nan, median=np.nan, p90=np.nan, rmse=np.nan)
    return dict(
        max=float(np.max(v)),
        median=float(np.median(v)),
        p90=float(np.percentile(v, 90)),
        rmse=float(np.sqrt(np.mean(v ** 2))),
    )


def main(argv=sys.argv):
    parser = argparse.ArgumentParser(
        description="Compare ASHLAR's MST and global least-squares position"
        " solvers on a single cached set of pairwise alignments.",
    )
    parser.add_argument(
        "--input", nargs="+", metavar="FILE",
        help="Real image input(s), as accepted by the ashlar CLI.",
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Generate a synthetic dataset with known ground-truth positions.",
    )
    parser.add_argument("--grid", default="8x8", metavar="RxC",
                        help="Synthetic grid dimensions (rows x cols).")
    parser.add_argument("--overlap", type=float, default=0.1,
                        help="Synthetic tile overlap fraction.")
    parser.add_argument("--jitter", type=float, default=5.0, metavar="PX",
                        help="Synthetic per-tile stage jitter (pixels).")
    parser.add_argument("--degrade-frac", type=float, default=0.0,
                        help="Fraction of synthetic tiles to blur/blank.")
    parser.add_argument("--tile-size", type=int, default=256,
                        help="Synthetic tile size (pixels).")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for synthetic generation.")
    parser.add_argument("-c", "--align-channel", type=int, default=0,
                        help="Reference channel for alignment.")
    parser.add_argument("-m", "--maximum-shift", type=float, default=15.0,
                        help="Maximum corrective shift in microns.")
    parser.add_argument("--stitch-alpha", type=float, default=0.01,
                        help="Permutation-test significance level.")
    parser.add_argument("--filter-sigma", type=float, default=0.0,
                        help="Pre-alignment Gaussian filter sigma.")
    parser.add_argument("--position-weight", choices=["inv-encc", "ncc",
                        "uniform"], default="inv-encc",
                        help="Edge weighting for the global solver.")
    parser.add_argument("--anchor-lambda", type=float, default=0.0,
                        help="Tikhonov anchor strength for the global solver.")
    parser.add_argument("--out", default="results", metavar="DIR",
                        help="Output directory for plots and artifacts.")
    args = parser.parse_args(argv[1:])

    if args.synthetic == bool(args.input):
        parser.error("specify exactly one of --synthetic or --input")

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    true_positions = None
    degraded_mask = None
    if args.synthetic:
        rows, cols = (int(x) for x in args.grid.lower().split("x"))
        print(f"Generating synthetic {rows}x{cols} grid (overlap={args.overlap},"
              f" jitter={args.jitter}px, degrade={args.degrade_frac}) ...")
        reader, true_positions, degraded_mask = make_synthetic(
            (rows, cols), args.overlap, args.jitter, args.degrade_frac,
            args.seed, tile_size=args.tile_size,
        )
    else:
        from .ashlar import build_reader
        print(f"Reading {args.input[0]} ...")
        reader = build_reader(args.input[0])

    print("Running alignment + pruning once ...")
    aligner = align_once(
        reader, args.align_channel, args.maximum_shift, args.stitch_alpha,
        args.filter_sigma, args.position_weight, args.anchor_lambda,
        verbose=True,
    )

    edges = pruned_edges(aligner)
    tree_edges = {tuple(sorted(e)) for e in aligner.spanning_tree.edges}
    n_redundant = len(edges) - len(tree_edges)
    print(f"\n{aligner.metadata.num_images} tiles, {len(edges)} surviving edges"
          f" ({len(tree_edges)} tree, {n_redundant} redundant).")

    # Derive both solutions from the identical cached alignments.
    aligner.calculate_positions()
    mst_shifts = aligner.shifts.copy()
    mst_positions = aligner.positions.copy()
    aligner.calculate_positions_global()
    glb_shifts = aligner.shifts.copy()
    glb_positions = aligner.positions.copy()

    pixel_size = aligner.metadata.pixel_size

    # Agreement between the two solutions (gauge-matched by centroid).
    mst_c = centroid_align(mst_positions, mst_positions)
    glb_c = centroid_align(glb_positions, mst_positions)
    agreement = np.linalg.norm(glb_c - mst_c, axis=1)

    # Edge residuals and total weighted SSE.
    mst_sse, _ = weighted_sse(aligner, mst_shifts, edges)
    glb_sse, _ = weighted_sse(aligner, glb_shifts, edges)

    paths = tree_path_lengths(aligner)

    rows_out = []
    if true_positions is not None:
        mst_err = np.linalg.norm(
            centroid_align(mst_positions, true_positions) - true_positions,
            axis=1,
        )
        glb_err = np.linalg.norm(
            centroid_align(glb_positions, true_positions) - true_positions,
            axis=1,
        )
        for name, err in (("mst", mst_err), ("global", glb_err)):
            s = summarize(name, err)
            deg = summarize(name, err[degraded_mask]) if degraded_mask.any() \
                else dict(median=np.nan)
            sse = mst_sse if name == "mst" else glb_sse
            rows_out.append((name, sse, s["median"], s["p90"], deg["median"]))
        np.savez(
            out / "synthetic_errors.npz",
            true_positions=true_positions, mst_positions=mst_positions,
            global_positions=glb_positions, degraded_mask=degraded_mask,
            path_lengths=paths,
        )
    else:
        rows_out.append(("mst", mst_sse, np.nan, np.nan, np.nan))
        rows_out.append(("global", glb_sse, np.nan, np.nan, np.nan))

    # ---- print summary table ----
    def um(px):
        return px * pixel_size

    print()
    print("method   | weighted SSE | vs-GT median (px) | vs-GT p90 (px) |"
          " degraded-tile median (px)")
    for name, sse, med, p90, degmed in rows_out:
        print(f"{name:8s} | {sse:12.4g} | {_fmt(med):>17s} | {_fmt(p90):>14s} |"
              f" {_fmt(degmed):>25s}")
    print()
    print(f"MST<->global per-tile displacement: max={agreement.max():.3f}px"
          f" ({um(agreement.max()):.3f}um), median={np.median(agreement):.3f}px,"
          f" p90={np.percentile(agreement, 90):.3f}px")
    print(f"Total weighted SSE: mst={mst_sse:.4g}  global={glb_sse:.4g}"
          f"  (global <= mst by construction: {glb_sse <= mst_sse + 1e-6})")
    print("  MST puts zero residual on its {} tree edges and dumps all residual"
          " onto the {} redundant edges; the global solve spreads residual"
          " across all edges.".format(len(tree_edges), n_redundant))

    _make_plots(out, aligner, mst_positions, glb_positions, paths,
                true_positions, degraded_mask)
    print(f"\nArtifacts written to {out}/")
    return 0


def _fmt(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) \
        else f"{x:.3f}"


def _make_plots(out, aligner, mst_positions, glb_positions, paths,
                true_positions, degraded_mask):
    size = aligner.metadata.size

    # Displacement field: each method relative to recorded stage positions.
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    nominal = aligner.metadata.positions
    for ax, pos, title in (
        (axs[0], mst_positions, "MST"),
        (axs[1], glb_positions, "global"),
    ):
        d = pos - nominal
        ax.quiver(nominal[:, 1], nominal[:, 0], d[:, 1], -d[:, 0],
                  angles="xy", scale_units="xy", scale=1, width=0.003)
        ax.set_title(f"{title} corrective shifts")
        ax.set_aspect("equal")
        ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out / "displacement_field.png", dpi=120)
    plt.close(fig)

    # Accumulation signature: per-tile error vs tree-path-length.
    if true_positions is not None:
        mst_err = np.linalg.norm(
            centroid_align(mst_positions, true_positions) - true_positions,
            axis=1)
        glb_err = np.linalg.norm(
            centroid_align(glb_positions, true_positions) - true_positions,
            axis=1)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(paths, mst_err, label="mst", alpha=0.7)
        ax.scatter(paths, glb_err, label="global", alpha=0.7)
        if degraded_mask is not None and degraded_mask.any():
            ax.scatter(paths[degraded_mask], mst_err[degraded_mask],
                       facecolors="none", edgecolors="red", s=90,
                       label="degraded (mst)")
        ax.set_xlabel("spanning-tree path length from root")
        ax.set_ylabel("Euclidean error vs ground truth (px)")
        ax.set_title("Accumulation signature")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "accumulation_scatter.png", dpi=120)
        plt.close(fig)

        # Error heatmaps.
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        vmax = max(mst_err.max(), glb_err.max())
        for ax, err, title in (
            (axs[0], mst_err, "MST"), (axs[1], glb_err, "global")):
            sc = ax.scatter(true_positions[:, 1], true_positions[:, 0],
                            c=err, s=80, vmin=0, vmax=vmax, cmap="viridis")
            ax.set_title(f"{title} error (px)")
            ax.set_aspect("equal")
            ax.invert_yaxis()
            fig.colorbar(sc, ax=ax)
        fig.tight_layout()
        fig.savefig(out / "error_heatmaps.png", dpi=120)
        plt.close(fig)


if __name__ == "__main__":
    main()
