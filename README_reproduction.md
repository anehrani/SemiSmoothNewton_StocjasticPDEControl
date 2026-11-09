# Reproducing the two examples

The reference implementation is `reproduce_examples.py`. It uses Q1 finite elements on the unit square, the two-dimensional covariance

\[
C_f(x,x')=\sigma^2\exp(-\|x-x'\|_2/\ell),
\]

five KL modes, 64 fixed standard-normal samples, and a primal-dual active-set Newton method for the pointwise boundary-control bounds. Both examples use the same covariance, target, samples, mesh, bounds, beta, and active-set parameter; they differ only in whether control is applied on $B_1$ or on $B_1\cup B_2$. Each active-set KKT system is solved by a sparse direct method. This is appropriate for a reproducible reference calculation; ordinary Euclidean PCG is not applied to the indefinite KKT matrix.

Run from this directory:

```bash
python3 reproduce_examples.py --samples 64 --nx 24 --ny 24 --N 5
```

The command writes:

- `reproduction_metrics.csv` with tracking, regularization, objective, KL-retention, feasibility, residual, iteration, and timing data;
- `reproduction_settings.json` and `kl_eigenvalues.json`;
- `mesh_convergence.csv` and `mesh_convergence_rates.csv` for the deterministic spatial refinement study;
- `kl_convergence.csv` and `kl_convergence_rates.csv` for the fixed-sample KL-dimension study;
- `variance_sweep.csv` for the fixed-sample covariance-amplitude study;
- `beta_c_sensitivity.csv` for the fixed-sample regularization and active-set-parameter studies;
- Vector PDF figures under `figures/`, using the filenames consumed by the LaTeX manuscript. Field panels include physical $x_1,x_2$ axes, labeled color bars, and fixed paired color limits; boundary-control panels include normalized boundary arclength, segment labels, and a control-value color bar.

The refinement and sensitivity studies reuse the same seeded standard-normal array. The mesh study uses the deterministic mean source to isolate spatial discretization error; the KL study uses the $N=7$ result as its reference and reports retained variance; the variance study changes $\sigma$ while holding the normalized KL modes and samples fixed.

The supplied historical manuscript did not include the original stochastic-Galerkin implementation, quadrature, mesh log, or PCG log. The script therefore reproduces the stated model and examples with a fixed-sample reference discretization; it does not claim bitwise recovery of unavailable historical runs.
