"""
Week 3 - Mesh refinement study (FEniCSx / dolfinx >= 0.10)

Question: how do we choose a "good" mesh for a Gaussian hotspot?
We compare three strategies on the SAME problem (all edges Dirichlet 300 K,
same Gaussian source as week 2) using error-vs-DOFs:

  1. Uniform meshes            N = 10 ... 160
  2. Concentric refinement     radii = c * sigma  (week-2 idea, now parametrised)
  3. Adaptive refinement (AMR) solve -> estimate -> mark -> refine, automatic

Quantity of interest: temperature RISE at the hotspot centre, dT = T(xc,yc) - 300.
Reference value (finite differences up to 800x800 + Richardson): 1.0782645 K

Outputs: mesh_error_vs_dofs.png, mesh_amr_meshes.png, mesh_concentric_vs_amr.png
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import ufl
from mpi4py import MPI
from dolfinx import mesh, fem, geometry, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
from dolfinx.plot import vtk_mesh

# ------------------------------------------------------------------
# 1. Problem parameters (same as week 2)
# ------------------------------------------------------------------
length = 1.0
thermal_k = 1.0
T0 = 300.0
xc, yc = 0.68, 0.38
q0, sigma = 100.0, 0.08
DT_REF = 1.0782645          # reference temperature rise at (xc, yc)

comm = MPI.COMM_WORLD
tdim, fdim = 2, 1


def uniform_mesh(N):
    return mesh.create_rectangle(
        comm, [np.array([0.0, 0.0]), np.array([length, length])], [N, N],
        cell_type=mesh.CellType.triangle)


def source(domain):
    x = ufl.SpatialCoordinate(domain)
    return q0 * ufl.exp(-((x[0] - xc) ** 2 + (x[1] - yc) ** 2) / (2 * sigma ** 2))


# ------------------------------------------------------------------
# 2. Solver on any mesh
# ------------------------------------------------------------------
_count = 0


def solve(domain):
    global _count
    _count += 1
    V = fem.functionspace(domain, ("Lagrange", 1))
    domain.topology.create_connectivity(fdim, tdim)
    bfacets = mesh.exterior_facet_indices(domain.topology)
    bdofs = fem.locate_dofs_topological(V, fdim, bfacets)
    bc = fem.dirichletbc(default_scalar_type(T0), bdofs, V)

    T, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    k = fem.Constant(domain, default_scalar_type(thermal_k))
    a = k * ufl.dot(ufl.grad(T), ufl.grad(v)) * ufl.dx
    L = source(domain) * v * ufl.dx
    problem = LinearProblem(
        a, L, bcs=[bc], petsc_options_prefix=f"mesh_study_{_count}_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    return problem.solve()


def n_dofs(Th):
    im = Th.function_space.dofmap.index_map
    return im.size_global * Th.function_space.dofmap.index_map_bs


def value_at(Th, px, py):
    """Evaluate the FEM solution at one point (serial use)."""
    domain = Th.function_space.mesh
    p = np.array([[px, py, 0.0]])
    tree = geometry.bb_tree(domain, tdim)
    cand = geometry.compute_collisions_points(tree, p)
    cell = geometry.compute_colliding_cells(domain, cand, p).links(0)[0]
    return Th.eval(p, np.array([cell], dtype=np.int32))[0]


def rel_error(Th):
    return abs((value_at(Th, xc, yc) - T0) - DT_REF) / DT_REF


# ------------------------------------------------------------------
# 3. Refinement helpers
# ------------------------------------------------------------------
def refine_cells(domain, cells):
    """Refine the given cells (and whatever neighbours are needed to keep the mesh conforming)."""
    domain.topology.create_entities(1)
    domain.topology.create_connectivity(tdim, 1)
    edges = mesh.compute_incident_entities(domain.topology, cells.astype(np.int32), tdim, 1)
    new_domain, _, _ = mesh.refine(domain, edges)
    return new_domain


def concentric_mesh(base_N, radii):
    """Week-2 strategy: refine every cell whose midpoint lies inside each radius in turn."""
    domain = uniform_mesh(base_N)
    for r in radii:
        nc = domain.topology.index_map(tdim).size_local
        cells = np.arange(nc, dtype=np.int32)
        mids = mesh.compute_midpoints(domain, tdim, cells)
        inside = cells[(mids[:, 0] - xc) ** 2 + (mids[:, 1] - yc) ** 2 < r ** 2]
        if inside.size:
            domain = refine_cells(domain, inside)
    return domain


def error_indicator(Th):
    """
    Residual-based a-posteriori error indicator, one value per cell:
      eta_K^2 = h_K^2 ||q + k ΔT_h||^2_K  +  sum_edges h_E/2 ||[k ∂T_h/∂n]||^2_E
    term 1: how badly the PDE is violated inside the cell
    term 2: jump of the heat flux between neighbouring cells (exact solution has none)
    (Dirichlet boundary -> no boundary term. For Robin/Neumann edges add one.)
    """
    domain = Th.function_space.mesh
    W = fem.functionspace(domain, ("DG", 0))
    w = ufl.TestFunction(W)
    hK = ufl.CellDiameter(domain)
    n = ufl.FacetNormal(domain)
    k = fem.Constant(domain, default_scalar_type(thermal_k))
    flux_jump = ufl.jump(k * ufl.grad(Th), n)
    G = (hK ** 2 * (source(domain) + ufl.div(k * ufl.grad(Th))) ** 2 * w * ufl.dx
         + hK("+") / 2 * flux_jump ** 2 * w("+") * ufl.dS
         + hK("-") / 2 * flux_jump ** 2 * w("-") * ufl.dS)
    eta2_dofs = fem.assemble_vector(fem.form(G)).array
    nc = domain.topology.index_map(tdim).size_local
    cell_to_dof = W.dofmap.list[:nc, 0]          # DG0: one dof per cell
    return eta2_dofs[cell_to_dof]                # eta_K^2 ordered by cell index


def dorfler_mark(eta2, theta=0.5):
    """Smallest set of cells that together carry a fraction theta of the total error."""
    order = np.argsort(eta2)[::-1]
    cumulative = np.cumsum(eta2[order])
    n_mark = np.searchsorted(cumulative, theta * cumulative[-1]) + 1
    return order[:n_mark].astype(np.int32)


# ------------------------------------------------------------------
# 4. Strategy 1: uniform meshes
# ------------------------------------------------------------------
print("\n=== Uniform ===")
uni = []
for N in [10, 20, 40, 80, 160]:
    Th = solve(uniform_mesh(N))
    uni.append((n_dofs(Th), rel_error(Th)))
    print(f"N={N:4d}  dofs={uni[-1][0]:7d}  rel.err={uni[-1][1]:.3e}")

# ------------------------------------------------------------------
# 5. Strategy 2: concentric rings, radii = c * sigma
#    Each ring halves the element size: n_levels = log2(h_base / h_target)
# ------------------------------------------------------------------
print("\n=== Concentric (radii = c*sigma) ===")
ring_factors = [3.0, 2.0, 1.0, 0.5]           # 3σ, 2σ, 1σ, 0.5σ
conc = {}                                     # base_N -> list of (dofs, err)
for base_N in [10, 20, 40]:
    conc[base_N] = []
    for n_levels in range(0, 5):
        radii = [c * sigma for c in ring_factors[:n_levels]]
        Th = solve(concentric_mesh(base_N, radii))
        conc[base_N].append((n_dofs(Th), rel_error(Th)))
        h_min = (length / base_N) / 2 ** n_levels
        print(f"base_N={base_N:3d} levels={n_levels}  h_min={h_min:.4f} (sigma/h_min={sigma / h_min:5.1f})"
              f"  dofs={conc[base_N][-1][0]:7d}  rel.err={conc[base_N][-1][1]:.3e}")

# ------------------------------------------------------------------
# 6. Strategy 3: adaptive mesh refinement
# ------------------------------------------------------------------
print("\n=== Adaptive (Dorfler theta=0.5) ===")
domain = uniform_mesh(10)
amr, amr_eta, snapshots = [], [], {}
for it in range(14):
    Th = solve(domain)
    eta2 = error_indicator(Th)
    eta_total = np.sqrt(eta2.sum())
    amr.append((n_dofs(Th), rel_error(Th)))
    amr_eta.append(eta_total)
    print(f"iter={it:2d}  dofs={amr[-1][0]:7d}  rel.err={amr[-1][1]:.3e}  eta={eta_total:.3e}")
    snapshots[it] = domain
    if amr[-1][0] > 40000:
        snapshots[it] = domain
        break
    domain = refine_cells(domain, dorfler_mark(eta2, theta=0.5))

# ------------------------------------------------------------------
# 7. Plot: error vs DOFs (the key figure)
# ------------------------------------------------------------------
plt.figure(figsize=(8, 6))
d, e = np.array(uni).T
plt.loglog(d, e, "ko-", lw=2, label="uniform")
for base_N, data in conc.items():
    d, e = np.array(data).T
    plt.loglog(d, e, "s--", alpha=0.8, label=f"concentric, base N={base_N} (0-4 rings)")
d, e = np.array(amr).T
plt.loglog(d, e, "r^-", lw=2, label="adaptive (AMR)")
# reference slope: error ~ h^2 ~ DOFs^-1
d_ref = np.array([1e2, 1e5])
plt.loglog(d_ref, 2.0 * e[0] * (d_ref / d_ref[0]) ** -1.0, "k:", lw=1, label="slope -1  (error ∝ h²)")
plt.xlabel("number of DOFs")
plt.ylabel("relative error of ΔT at hotspot centre")
plt.title("Accuracy vs cost for three meshing strategies")
plt.grid(alpha=0.3, which="both")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("mesh_error_vs_dofs.png", dpi=200)

# ------------------------------------------------------------------
# 8. Plot: estimator vs true error (does eta track the real error?)
# ------------------------------------------------------------------
plt.figure(figsize=(7, 5))
d = np.array([x[0] for x in amr])
plt.loglog(d, np.array([x[1] for x in amr]) * DT_REF, "r^-", label="true error |ΔT_h - ΔT_ref| (K)")
plt.loglog(d, amr_eta, "bo-", label="estimator η (no exact solution needed)")
plt.xlabel("number of DOFs")
plt.title("AMR: error estimator vs true error")
plt.grid(alpha=0.3, which="both")
plt.legend()
plt.tight_layout()
plt.savefig("mesh_estimator.png", dpi=200)


# ------------------------------------------------------------------
# 9. Plot: how the adaptive mesh grows
# ------------------------------------------------------------------
def draw_mesh(ax, domain, title):
    topo, _, geom = vtk_mesh(domain, tdim)
    tri = Triangulation(geom[:, 0], geom[:, 1], topo.reshape(-1, 4)[:, 1:4])
    ax.triplot(tri, color="steelblue", lw=0.3)
    ax.plot(xc, yc, "r.", ms=8)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)

show = list(snapshots.items())[::3]          # every 3rd AMR iteration
if show[-1][0] != list(snapshots)[-1]:       # always include the final mesh
    show.append(list(snapshots.items())[-1])
fig, axes = plt.subplots(1, len(show), figsize=(4 * len(show), 4.3), layout="constrained")
for ax, (it, dom) in zip(np.atleast_1d(axes), show):
    draw_mesh(ax, dom, f"AMR iter {it}: {dom.topology.index_map(tdim).size_global} cells")
fig.savefig("mesh_amr_meshes.png", dpi=200)

# Side by side: week-2 concentric mesh vs AMR mesh of similar size
fig, axes = plt.subplots(1, 2, figsize=(10, 5), layout="constrained")
week2 = concentric_mesh(20, [0.22, 0.12, 0.06])
draw_mesh(axes[0], week2, f"week 2 concentric: {week2.topology.index_map(tdim).size_global} cells")
target = week2.topology.index_map(tdim).size_global
closest = min(snapshots.values(), key=lambda m: abs(m.topology.index_map(tdim).size_global - target))
draw_mesh(axes[1], closest, f"AMR: {closest.topology.index_map(tdim).size_global} cells")
fig.savefig("mesh_concentric_vs_amr.png", dpi=200)
plt.show()