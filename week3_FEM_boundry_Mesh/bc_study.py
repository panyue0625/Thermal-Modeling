"""
Week 3 - Boundary-condition study for a 2D Gaussian hotspot (FEniCSx / dolfinx >= 0.10)

Same heat source as week 2, uniform mesh, only the boundary conditions change.
  Case A : all four edges Dirichlet T = 300 K               (week-2 baseline)
  Case B : left edge Dirichlet, other edges adiabatic        (Neumann q_n = 0)
  Case C : all four edges Robin  -k dT/dn = h (T - T_amb)
  Case D : bottom edge Robin (heat sink), other edges adiabatic
  Sweep  : all-Robin with h from 0.1 to 1e4 -> shows Neumann <-> Dirichlet limits

Outputs: bc_cases_temperature.png, bc_profiles.png, bc_h_sweep.png
"""
import numpy as np
import matplotlib.pyplot as plt
import ufl
from mpi4py import MPI
from dolfinx import mesh, fem, geometry, default_scalar_type
from dolfinx.fem.petsc import LinearProblem

# ------------------------------------------------------------------
# 1. Parameters (same as week 2)
# ------------------------------------------------------------------
length = 1.0
thermal_k = 1.0
T0 = 300.0        # Dirichlet temperature
T_amb = 300.0     # ambient temperature used by Robin edges
xc, yc = 0.68, 0.38
q0, sigma = 100.0, 0.08
N = 80            # uniform mesh: isolate the effect of the BCs

# ------------------------------------------------------------------
# 2. Mesh + tag the four edges with integer markers
# ------------------------------------------------------------------
domain = mesh.create_rectangle(
    MPI.COMM_WORLD,
    [np.array([0.0, 0.0]), np.array([length, length])],
    [N, N],
    cell_type=mesh.CellType.triangle,
)
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)

LEFT, RIGHT, BOTTOM, TOP = 1, 2, 3, 4
edge_locators = [
    (LEFT,   lambda x: np.isclose(x[0], 0.0)),
    (RIGHT,  lambda x: np.isclose(x[0], length)),
    (BOTTOM, lambda x: np.isclose(x[1], 0.0)),
    (TOP,    lambda x: np.isclose(x[1], length)),
]
facet_ids, facet_vals = [], []
for marker, locator in edge_locators:
    facets = mesh.locate_entities_boundary(domain, fdim, locator)
    facet_ids.append(facets)
    facet_vals.append(np.full_like(facets, marker))
facet_ids = np.hstack(facet_ids).astype(np.int32)
facet_vals = np.hstack(facet_vals).astype(np.int32)
order = np.argsort(facet_ids)
facet_tag = mesh.meshtags(domain, fdim, facet_ids[order], facet_vals[order])

# ds(LEFT), ds(RIGHT), ... now integrate over one edge only
ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tag)

# ------------------------------------------------------------------
# 3. Function space, heat source, constants
# ------------------------------------------------------------------
V = fem.functionspace(domain, ("Lagrange", 1))
x = ufl.SpatialCoordinate(domain)
n = ufl.FacetNormal(domain)
k = fem.Constant(domain, default_scalar_type(thermal_k))
q_expr = q0 * ufl.exp(-((x[0] - xc) ** 2 + (x[1] - yc) ** 2) / (2 * sigma ** 2))


def integrate(expr):
    """Assemble a scalar integral and sum over MPI ranks."""
    return domain.comm.allreduce(fem.assemble_scalar(fem.form(expr)), op=MPI.SUM)


Q_total = integrate(q_expr * ufl.dx)   # total heat generated (per unit depth)

# ------------------------------------------------------------------
# 4. Generic solver: spec = {edge_marker: (type, value)}
#      ("D", T_wall)        Dirichlet   T = T_wall
#      ("N", q_n)           Neumann     -k dT/dn = q_n  (outward flux; 0 = adiabatic)
#      ("R", (h, T_inf))    Robin       -k dT/dn = h (T - T_inf)
# ------------------------------------------------------------------
_solve_count = 0


def solve(spec):
    global _solve_count
    _solve_count += 1

    T = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Volume part of the weak form:  ∫ k ∇T·∇v dx - ∫ q v dx
    F = k * ufl.dot(ufl.grad(T), ufl.grad(v)) * ufl.dx - q_expr * v * ufl.dx
    bcs = []

    for marker, (kind, value) in spec.items():
        if kind == "D":                       # essential BC: constrain DOFs
            dofs = fem.locate_dofs_topological(V, fdim, facet_tag.find(marker))
            bcs.append(fem.dirichletbc(default_scalar_type(value), dofs, V))
        elif kind == "N":                     # natural BC: boundary term + ∫ q_n v ds
            if value != 0.0:                  # adiabatic -> term vanishes, add nothing
                F += fem.Constant(domain, default_scalar_type(value)) * v * ds(marker)
        elif kind == "R":                     # natural BC: + ∫ h (T - T_inf) v ds
            h, T_inf = value
            h_c = fem.Constant(domain, default_scalar_type(h))
            F += h_c * (T - T_inf) * v * ds(marker)
        else:
            raise ValueError(f"unknown BC type {kind}")

    problem = LinearProblem(
        ufl.lhs(F), ufl.rhs(F), bcs=bcs,
        petsc_options_prefix=f"bc_study_{_solve_count}_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )
    return problem.solve()


def heat_out(Th, spec):
    """Total heat leaving through the boundary -> should equal Q_total."""
    out = 0.0
    for marker, (kind, value) in spec.items():
        if kind == "D":     # flux recovered from the gradient: -k ∇T·n
            out += integrate(-k * ufl.dot(ufl.grad(Th), n) * ds(marker))
        elif kind == "N":
            out += value * length
        elif kind == "R":
            h, T_inf = value
            out += integrate(h * (Th - T_inf) * ds(marker))
    return out


def t_max(Th):
    return domain.comm.allreduce(Th.x.array.max(), op=MPI.MAX)


# ------------------------------------------------------------------
# 5. The four cases
# ------------------------------------------------------------------
h_ref = 10.0
all_edges = (LEFT, RIGHT, BOTTOM, TOP)
cases = {
    "A: all Dirichlet 300 K":
        {m: ("D", T0) for m in all_edges},
    "B: left Dirichlet, rest adiabatic":
        {LEFT: ("D", T0), RIGHT: ("N", 0.0), BOTTOM: ("N", 0.0), TOP: ("N", 0.0)},
    f"C: all Robin (h={h_ref:g})":
        {m: ("R", (h_ref, T_amb)) for m in all_edges},
    f"D: bottom Robin (h={h_ref:g}), rest adiabatic":
        {LEFT: ("N", 0.0), RIGHT: ("N", 0.0), BOTTOM: ("R", (h_ref, T_amb)), TOP: ("N", 0.0)},
}

results = {}
if domain.comm.rank == 0:
    print(f"Total heat generated Q = {Q_total:.4f}\n")
    print(f"{'case':45s} {'T_max [K]':>10s} {'heat out':>10s} {'balance err':>12s}")
for name, spec in cases.items():
    Th = solve(spec)
    out = heat_out(Th, spec)
    results[name] = Th
    if domain.comm.rank == 0:
        print(f"{name:45s} {t_max(Th):10.4f} {out:10.4f} {abs(out - Q_total) / Q_total:11.2%}")

# ------------------------------------------------------------------
# 6. Plot: temperature fields (own colour scale per case)
# ------------------------------------------------------------------
coords = V.tabulate_dof_coordinates()
fig, axes = plt.subplots(2, 2, figsize=(11, 10))
for ax, (name, Th) in zip(axes.flat, results.items()):
    cf = ax.tricontourf(coords[:, 0], coords[:, 1], Th.x.array.real, levels=40, cmap="hot")
    fig.colorbar(cf, ax=ax, label="T (K)")
    ax.plot(xc, yc, "c+", ms=10)
    ax.set_title(f"{name}\nTmax = {t_max(Th):.3f} K", fontsize=10)
    ax.set_aspect("equal")
fig.tight_layout()
fig.savefig("bc_cases_temperature.png", dpi=200)

# ------------------------------------------------------------------
# 7. Plot: temperature along the horizontal line y = yc through the hotspot
# ------------------------------------------------------------------
xs = np.linspace(0.0, length, 201)
pts = np.zeros((xs.size, 3))
pts[:, 0], pts[:, 1] = xs, yc
tree = geometry.bb_tree(domain, tdim)
cand = geometry.compute_collisions_points(tree, pts)
coll = geometry.compute_colliding_cells(domain, cand, pts)
keep, cells = [], []
for i in range(pts.shape[0]):
    if len(coll.links(i)) > 0:
        keep.append(i)
        cells.append(coll.links(i)[0])
keep = np.array(keep, dtype=np.int32)

plt.figure(figsize=(8, 5))
for name, Th in results.items():
    vals = Th.eval(pts[keep], np.array(cells, dtype=np.int32)).flatten()
    plt.plot(xs[keep], vals, label=name)
plt.axvline(xc, color="gray", ls=":", lw=1)
plt.xlabel("x  (along y = yc)")
plt.ylabel("T (K)")
plt.title("Temperature profile through the hotspot")
plt.legend(fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("bc_profiles.png", dpi=200)

# ------------------------------------------------------------------
# 8. Robin sweep: Tmax vs h  (h -> 0: adiabatic, h -> inf: Dirichlet)
# ------------------------------------------------------------------
h_values = np.logspace(-1, 4, 11)
tmax_h = []
for h in h_values:
    Th = solve({m: ("R", (h, T_amb)) for m in all_edges})
    tmax_h.append(t_max(Th))
    if domain.comm.rank == 0:
        print(f"h = {h:9.3g}   Tmax = {tmax_h[-1]:.4f} K")

T_dirichlet = t_max(results["A: all Dirichlet 300 K"])
perimeter = 4 * length
lumped = T_amb + Q_total / (h_values * perimeter)   # small-h estimate: T ~ uniform

plt.figure(figsize=(7, 5))
plt.semilogx(h_values, tmax_h, "o-", label="FEM, all-Robin")
plt.axhline(T_dirichlet, color="k", ls="--", label="all-Dirichlet limit (h → ∞)")
plt.semilogx(h_values, lumped, ":", label="lumped estimate T_amb + Q/(h·P)")
plt.ylim(T_amb - 0.5, max(tmax_h) + 1)
plt.xlabel("heat-transfer coefficient h   (Bi = hL/k)")
plt.ylabel("peak temperature (K)")
plt.title("Robin BC interpolates between adiabatic and Dirichlet")
plt.legend()
plt.grid(alpha=0.3, which="both")
plt.tight_layout()
plt.savefig("bc_h_sweep.png", dpi=200)
plt.show()