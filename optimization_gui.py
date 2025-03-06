import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import math
import matplotlib as mpl
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from stl import mesh

mpl.use("TkAgg")

def lk_H8(nu):
    """
    Calcula la matriz de rigidez de un elemento hexaédrico (H8), considerando la relación de Poisson.
    """
    A = np.array([
        [32, 6, -8, 6, -6, 4, 3, -6, -10, 3, -3, -3, -4, -8],
        [-48, 0, 0, -24, 24, 0, 0, 0, 12, -12, 0, 12, 12, 12]
    ], dtype=float)
    k = (1 / 144) * (A.T @ np.array([1, nu], dtype=float))

    K1 = np.array([
        [k[0], k[1], k[1], k[2], k[4], k[4]],
        [k[1], k[0], k[1], k[3], k[5], k[6]],
        [k[1], k[1], k[0], k[3], k[6], k[5]],
        [k[2], k[3], k[3], k[0], k[7], k[7]],
        [k[4], k[5], k[6], k[7], k[0], k[1]],
        [k[4], k[6], k[5], k[7], k[1], k[0]]
    ], dtype=float)

    K2 = np.array([
        [k[8], k[7], k[11], k[5], k[3], k[6]],
        [k[7], k[8], k[11], k[4], k[2], k[4]],
        [k[9], k[9], k[12], k[6], k[3], k[5]],
        [k[5], k[4], k[10], k[8], k[1], k[9]],
        [k[3], k[2], k[4], k[1], k[8], k[11]],
        [k[10], k[3], k[5], k[11], k[9], k[12]]
    ], dtype=float)

    K3 = np.array([
        [k[5], k[6], k[3], k[8], k[11], k[7]],
        [k[6], k[5], k[3], k[9], k[12], k[9]],
        [k[4], k[4], k[2], k[7], k[11], k[8]],
        [k[8], k[9], k[1], k[5], k[10], k[4]],
        [k[11], k[12], k[9], k[10], k[5], k[3]],
        [k[1], k[11], k[8], k[3], k[4], k[2]]
    ], dtype=float)

    K4 = np.array([
        [k[13], k[10], k[10], k[12], k[9], k[9]],
        [k[10], k[13], k[10], k[11], k[8], k[7]],
        [k[10], k[10], k[13], k[11], k[7], k[8]],
        [k[12], k[11], k[11], k[13], k[6], k[6]],
        [k[9], k[8], k[7], k[6], k[13], k[10]],
        [k[9], k[7], k[8], k[6], k[10], k[13]]
    ], dtype=float)

    K5 = np.array([
        [k[0], k[1], k[7], k[2], k[4], k[3]],
        [k[1], k[0], k[7], k[3], k[5], k[10]],
        [k[7], k[7], k[0], k[4], k[10], k[5]],
        [k[2], k[3], k[4], k[0], k[7], k[1]],
        [k[4], k[5], k[10], k[7], k[0], k[7]],
        [k[3], k[10], k[5], k[1], k[7], k[0]]
    ], dtype=float)

    K6 = np.array([
        [k[13], k[10], k[6], k[12], k[9], k[11]],
        [k[10], k[13], k[6], k[11], k[8], k[1]],
        [k[6], k[6], k[13], k[9], k[1], k[8]],
        [k[12], k[11], k[9], k[13], k[6], k[10]],
        [k[9], k[8], k[1], k[6], k[13], k[6]],
        [k[11], k[1], k[8], k[10], k[6], k[13]]
    ], dtype=float)

    factor = 1 / ((nu + 1) * (1 - 2 * nu))
    top = np.hstack([K1, K2, K3, K4])
    bottom = np.hstack([K2.T, K5, K6, K3.T])
    third = np.hstack([K3.T, K6, K5.T, K2.T])
    fourth = np.hstack([K4, K3, K2, K1])
    KE = factor * np.vstack([top, bottom, third, fourth])

    return KE

def get_load_and_fixed_dofs(nelx, nely, nelz, x_load, y_load, z_load):
    """
    Determina los grados de libertad donde se aplican cargas y las condiciones de frontera fijas.
    """
    # Load DOFs: nodo en (x_load, y_load, z_load)
    il, jl, kl = np.meshgrid(np.array([x_load]), np.array([y_load]), np.array([z_load]), indexing='ij')

    load_nid = (kl * ((nelx + 1) * (nely + 1)) + il * (nely + 1) + (nely + 1 - jl)) - 1
    load_nid = load_nid.flatten()
    load_dof = 3 * load_nid + 1  # Aplica la carga en la dirección Y

    # Fixed DOFs: nodos en x = 0, en toda la base y y en toda la altura z
    iif = np.array([0, 0, nelx, nelx])
    jf = np.array([0, 0, 0, 0])
    kf = np.array([0, nelz, 0, nelz])

    fixed_nid = (kf * ((nelx + 1) * (nely + 1)) + iif * (nely + 1) + (nely + 1 - jf)) - 1
    fixed_nid = fixed_nid.flatten()
    fixed_dof = np.unique(np.concatenate([3 * fixed_nid, 3 * fixed_nid + 1, 3 * fixed_nid + 2]))

    return load_dof, fixed_dof

def setup_FE(nelx, nely, nelz, load_dof, fixed_dof):
    """
    Configura el análisis de elementos finitos, incluyendo el vector de fuerzas y los desplazamientos iniciales.
    """
    nele = nelx * nely * nelz
    ndof = 3 * (nelx + 1) * (nely + 1) * (nelz + 1)
    F = np.zeros(ndof)
    F[load_dof] = -1.0  # Se definen las fuerzas aplicadas
    U = np.zeros(ndof)
    free_dof = np.setdiff1d(np.arange(ndof), fixed_dof)
    return F, U, free_dof, ndof, nele

def generate_edofMat(nelx, nely, nelz):
    """
    Genera la matriz de conectividad de los elementos, necesaria para ensamblar la matriz de rigidez global.
    """
    Nx = nelx + 1
    Ny = nely + 1
    nele = nelx * nely * nelz
    edofMat = np.zeros((nele, 24), dtype=int)
    elem = 0
    for k in range(nelz):
        base = k * (Nx * Ny)
        for i in range(nelx):
            for j in range(nely):
                # Bottom face nodes:
                n1 = base + i * Ny + j
                n2 = base + (i + 1) * Ny + j
                n3 = base + (i + 1) * Ny + (j + 1)
                n4 = base + i * Ny + (j + 1)
                # Top face nodes:
                n5 = n1 + (Nx * Ny)
                n6 = n2 + (Nx * Ny)
                n7 = n3 + (Nx * Ny)
                n8 = n4 + (Nx * Ny)
                nodes = [n1, n2, n3, n4, n5, n6, n7, n8]
                dofs = []
                for node in nodes:
                    dofs.extend([3 * node, 3 * node + 1, 3 * node + 2])
                edofMat[elem, :] = dofs
                elem += 1
    return edofMat

def get_global_stiffness_indices(edofMat):
    """
    Precalcular los índices globales de filas y columnas para ensamblar la matriz de rigidez.
    """
    nele = edofMat.shape[0]
    total_entries = nele * 24 * 24
    iK = np.empty(total_entries, dtype=int)
    jK = np.empty(total_entries, dtype=int)
    index = 0
    for e in range(nele):
        edofs = edofMat[e, :]
        block_i = np.repeat(edofs, 24)
        block_j = np.tile(edofs, 24)
        iK[index:index + 24 * 24] = block_i
        jK[index:index + 24 * 24] = block_j
        index += 24 * 24
    return iK, jK

def create_filter(nelx, nely, nelz, rmin):
    """
    Crea un filtro de sensibilidad para suavizar resultados y evitar patrones irregulares.
    """
    nele = nelx * nely * nelz
    rmin_int = math.ceil(rmin) - 1
    max_entries = nele * ((2 * rmin_int + 1) ** 3)
    iH = np.empty(max_entries, dtype=int)
    jH = np.empty(max_entries, dtype=int)
    sH = np.empty(max_entries, dtype=float)
    counter = 0
    for k1 in range(nelz):
        for i1 in range(nelx):
            for j1 in range(nely):
                e1 = k1 * nelx * nely + i1 * nely + j1
                for k2 in range(max(k1 - rmin_int, 0), min(k1 + rmin_int, nelz - 1) + 1):
                    for i2 in range(max(i1 - rmin_int, 0), min(i1 + rmin_int, nelx - 1) + 1):
                        for j2 in range(max(j1 - rmin_int, 0), min(j1 + rmin_int, nely - 1) + 1):
                            e2 = k2 * nelx * nely + i2 * nely + j2
                            dist = np.sqrt((i1 - i2) ** 2 + (j1 - j2) ** 2 + (k1 - k2) ** 2)
                            val = max(0.0, rmin - dist)
                            iH[counter] = e1
                            jH[counter] = e2
                            sH[counter] = val
                            counter += 1
    iH = iH[:counter]
    jH = jH[:counter]
    sH = sH[:counter]
    H = sp.coo_matrix((sH, (iH, jH)), shape=(nele, nele)).tocsr()
    Hs = np.array(H.sum(axis=1)).flatten()
    return H, Hs

def optimize_topology(ax1,
                      nelx, nely, nelz,
                      x_load, y_load, z_load,
                      volfrac, penal, rmin,
                      maxloop=200, tolx=0.01,
                      displayflag=False,
                      e0_material=1.0,
                      nu_material=0.3):
    """
    Realiza el proceso iterativo de optimización usando SIMP. Actualiza la figura ax1 en cada iteración.
    """
    # Material properties
    E0 = e0_material
    Emin = 1e-9
    nu = nu_material
    KE = lk_H8(nu)

    # FE setup
    load_dof, fixed_dof = get_load_and_fixed_dofs(nelx, nely, nelz, x_load, y_load, z_load)
    F, U, free_dof, ndof, nele = setup_FE(nelx, nely, nelz, load_dof, fixed_dof)
    edofMat = generate_edofMat(nelx, nely, nelz)
    iK, jK = get_global_stiffness_indices(edofMat)
    H, Hs = create_filter(nelx, nely, nelz, rmin)

    # Initialize design variables
    x = volfrac * np.ones((nely, nelx, nelz))
    xPhys = x.copy()

    loop = 0
    change = 1.0
    history = []

    while change > tolx and loop < maxloop:
        loop += 1
        # FE Analysis
        xPhys_vec = xPhys.ravel(order='F')
        stiffness = Emin + (xPhys_vec ** penal) * (E0 - Emin)
        sK = (KE.flatten()[:, None] * stiffness[None, :]).flatten(order='F')
        K = sp.coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsr()
        K = (K + K.T) * 0.5  # Enforce symmetry
        U[free_dof] = spsolve(K[free_dof, :][:, free_dof], F[free_dof])

        # Objective and sensitivity analysis
        ce = np.zeros(nele)
        for e in range(nele):
            ue = U[edofMat[e, :]]
            ce[e] = ue @ (KE @ ue)
        ce = ce.reshape((nely, nelx, nelz), order='F')
        c = np.sum((Emin + xPhys ** penal * (E0 - Emin)) * ce)
        dc = -penal * (E0 - Emin) * (xPhys ** (penal - 1)) * ce
        dv = np.ones_like(xPhys)

        # Filtering sensitivities
        dc_flat = dc.ravel(order='F')
        dv_flat = dv.ravel(order='F')
        dc_filtered = np.array(H.dot(dc_flat / Hs)).flatten()
        dv_filtered = np.array(H.dot(dv_flat / Hs)).flatten()
        dc = dc_filtered.reshape((nely, nelx, nelz), order='F')
        dv = dv_filtered.reshape((nely, nelx, nelz), order='F')

        # Optimality criteria update
        l1 = 0.0
        l2 = 1e9
        move = 0.2
        while (l2 - l1) / (l1 + l2) > 1e-3:
            lmid = 0.5 * (l1 + l2)
            xnew = np.maximum(0, np.maximum(x - move,
                                            np.minimum(1, np.minimum(x + move, x * np.sqrt(-dc / dv / lmid)))))
            xnew_flat = xnew.ravel(order='F')
            xPhys_flat = np.array(H.dot(xnew_flat) / Hs).flatten()
            xPhys_new = xPhys_flat.reshape((nely, nelx, nelz), order='F')
            if np.sum(xPhys_new) > volfrac * nele:
                l1 = lmid
            else:
                l2 = lmid

        change = np.max(np.abs(xnew - x))
        x = xnew.copy()
        xPhys = xPhys_new.copy()
        history.append(c)
        print(f" It.: {loop:3d} Obj.: {c:11.4f} Vol.: {np.mean(xPhys):7.3f} ch.: {change:7.3f}")

        # Update the plot in ax1 (already embedded in Tk window).
        ax1.clear()
        display_solid_3D(ax1, xPhys)  # custom function below that plots on the given axes
        log_message = f"Iteration: {loop} | Obj.: {c:.4f} | Vol.: {np.mean(xPhys):.3f} | Ch.: {change:.3f}"
        ax1.set_title(log_message, fontsize=10)

        # Instead of plt.pause, just flush the Tk canvas
        canvas.draw()
        root.update_idletasks()
        root.update()
        # No plt.show() or plt.pause() to avoid new windows or blocking

    return xPhys, c, history

def display_solid_3D(ax, rho, threshold=0.5):
    """
    Plot the solid in the provided Axes3D (ax) without creating a new figure.
    """
    ax.clear()
    nely, nelx, nelz = rho.shape
    for i in range(nelx):
        for j in range(nely):
            for k in range(nelz):
                if rho[j, i, k] > threshold:
                    # Each voxel is drawn as a small cube
                    verts = [
                        [i,   j,   k],
                        [i+1, j,   k],
                        [i+1, j+1, k],
                        [i,   j+1, k],
                        [i,   j,   k+1],
                        [i+1, j,   k+1],
                        [i+1, j+1, k+1],
                        [i,   j+1, k+1]
                    ]
                    faces = [
                        [0, 1, 2, 3],
                        [4, 5, 6, 7],
                        [0, 1, 5, 4],
                        [2, 3, 7, 6],
                        [0, 3, 7, 4],
                        [1, 2, 6, 5]
                    ]
                    poly3d = [[verts[idx] for idx in face] for face in faces]
                    ax.add_collection3d(
                        Poly3DCollection(poly3d, facecolors='gray',
                                         linewidths=0.1, edgecolors='k')
                    )
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([0, nelx])
    ax.set_ylim([0, nely])
    ax.set_zlim([0, nelz])
    ax.set_box_aspect([nelx, nely, nelz])

def display_iso_surface_3D(ax, rho, threshold=0.5):
    """
    Displays a non-rotated iso-surface of 'rho' on the provided 3D axes.
    Fixes the orientation by reordering the array for marching_cubes.
    
    Parameters:
        ax        : A 3D matplotlib axes object (Axes3D).
        rho       : A 3D numpy array with shape (nely, nelx, nelz).
        threshold : Density threshold for the isosurface.
    """
    ax.clear()

    # 1) Transpose (nely, nelx, nelz) -> (nelz, nely, nelx) so the first dimension is "Z"
    reorder_rho = np.transpose(rho, (2, 0, 1))
    
    # 2) Optionally pad to avoid partial surfaces at the edges
    reorder_rho = np.pad(reorder_rho, pad_width=1, mode='constant', constant_values=0)
    
    # 3) Run marching cubes on the (z, y, x) volume
    verts, faces, _, _ = measure.marching_cubes(reorder_rho, level=threshold)
    
    # marching_cubes returns vertices in (z, y, x). Subtract 1 to remove padding offset
    verts -= 1
    
    # 4) Reorder (z, y, x) -> (x, y, z)
    verts = verts[:, [2, 1, 0]]
    
    # 5) Create a Poly3DCollection and add to the axes
    mesh_poly = Poly3DCollection(verts[faces], alpha=0.7, edgecolor='k')
    ax.add_collection3d(mesh_poly)
    
    # 6) Match axis limits and aspect ratio to your voxel dimensions
    nely, nelx, nelz = rho.shape
    ax.set_xlim([0, nelx])
    ax.set_ylim([0, nely])
    ax.set_zlim([0, nelz])
    ax.set_box_aspect([nelx, nely, nelz])
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

def export_optimized_stl(rho, threshold=0.5, filename='optimized.stl', hx=1, hy=1, hz=1):
    """
    Export the optimized topology as an STL, using cubes for each voxel > threshold.
    """
    nely, nelx, nelz = rho.shape
    triangles = []
    faces_idx = [
        [0, 1, 2, 3],
        [1, 5, 6, 2],
        [3, 2, 6, 7],
        [0, 4, 7, 3],
        [0, 1, 5, 4],
        [4, 5, 6, 7]
    ]
    for k in range(nelz):
        z = k * hz
        for i in range(nelx):
            x = i * hx
            for j in range(nely):
                y = j * hy
                if rho[j, i, k] > threshold:
                    # 8 vertices of a cube
                    v1 = [x,   y,   z]
                    v2 = [x+hx, y,   z]
                    v3 = [x+hx, y+hy, z]
                    v4 = [x,   y+hy, z]
                    v5 = [x,   y,   z+hz]
                    v6 = [x+hx, y,   z+hz]
                    v7 = [x+hx, y+hy, z+hz]
                    v8 = [x,   y+hy, z+hz]
                    verts = np.array([v1, v2, v3, v4, v5, v6, v7, v8], dtype=float)

                    # Build triangles
                    for face in faces_idx:
                        tri1 = [verts[face[0]], verts[face[1]], verts[face[2]]]
                        tri2 = [verts[face[0]], verts[face[2]], verts[face[3]]]
                        triangles.append(tri1)
                        triangles.append(tri2)

    if not triangles:
        print("No elements above threshold. STL not created.")
        return

    triangles_array = np.array(triangles, dtype=np.float32)
    stl_mesh = mesh.Mesh(np.zeros(triangles_array.shape[0], dtype=mesh.Mesh.dtype))
    for i, tri in enumerate(triangles_array):
        stl_mesh.vectors[i] = tri
    stl_mesh.save(filename)
    print(f"STL file saved as '{filename}'.")

def export_iso_stl(rho, threshold=0.5, filename='optimized_iso.stl'):
    """
    Export a smoothed iso-surface STL using marching_cubes.
    """
    padded_rho = np.pad(rho, pad_width=1, mode='constant', constant_values=0)
    verts, faces, _, _ = measure.marching_cubes(padded_rho, level=threshold)
    verts -= 1
    stl_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            stl_mesh.vectors[i][j] = verts[f[j]]
    stl_mesh.save(filename)
    print(f"Smoothed STL file saved as '{filename}'")

# ---------------- TKINTER GUI SETUP ----------------
root = tk.Tk()
root.title("Optimización de Geometrías para Impresión 3D")

# Create a single figure with two subplots
fig = Figure(figsize=(10, 7))
fig.suptitle("Optimización de Geometrías para Impresión 3D", fontsize=14, fontweight="bold")
canvas = FigureCanvasTkAgg(fig, master=root)

canvas.get_tk_widget().grid(row=0, column=1, rowspan=6, padx=10, pady=5)

# Subplots for the single window
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

# Frames for input
dimensions_frame = ttk.LabelFrame(root, text="Dimensiones")
dimensions_frame.grid(row=0, column=0, padx=10, pady=5, sticky="ew")

material_frame = ttk.LabelFrame(root, text="Parametros del Material")
material_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")

load_frame = ttk.LabelFrame(root, text="Posición de la Carga")
load_frame.grid(row=2, column=0, padx=10, pady=5, sticky="ew")

optimization_frame = ttk.LabelFrame(root, text="Optimización")
optimization_frame.grid(row=3, column=0, padx=10, pady=5, sticky="ew")

# Dimensions inputs
entry_nelx = ttk.Entry(dimensions_frame)
entry_nely = ttk.Entry(dimensions_frame)
entry_nelz = ttk.Entry(dimensions_frame)

ttk.Label(dimensions_frame, text="X:").grid(row=0, column=0)
entry_nelx.grid(row=0, column=1, padx=5, pady=2)
ttk.Label(dimensions_frame, text="Y:").grid(row=1, column=0)
entry_nely.grid(row=1, column=1, padx=5, pady=2)
ttk.Label(dimensions_frame, text="Z:").grid(row=2, column=0)
entry_nelz.grid(row=2, column=1, padx=5, pady=2)

# Material inputs
entry_E0 = ttk.Entry(material_frame)
entry_nu = ttk.Entry(material_frame)

ttk.Label(material_frame, text="Modulo de Young:").grid(row=0, column=0)
entry_E0.grid(row=0, column=1, padx=5, pady=2)
ttk.Label(material_frame, text="Coeficiente Poisson:").grid(row=1, column=0)
entry_nu.grid(row=1, column=1, padx=5, pady=2)

# Load Position inputs
entry_x_load = ttk.Entry(load_frame)
entry_y_load = ttk.Entry(load_frame)
entry_z_load = ttk.Entry(load_frame)

ttk.Label(load_frame, text="Posición x:").grid(row=0, column=0)
entry_x_load.grid(row=0, column=1, padx=5, pady=2)
ttk.Label(load_frame, text="Posición y:").grid(row=1, column=0)
entry_y_load.grid(row=1, column=1, padx=5, pady=2)
ttk.Label(load_frame, text="Posición z:").grid(row=2, column=0)
entry_z_load.grid(row=2, column=1, padx=5, pady=2)

# Optimization Settings inputs
entry_volfrac = ttk.Entry(optimization_frame)
entry_penal = ttk.Entry(optimization_frame)
entry_rmin = ttk.Entry(optimization_frame)
ttk.Label(optimization_frame, text="% Volumen a Usar:").grid(row=0, column=0)
entry_volfrac.grid(row=0, column=1, padx=5, pady=2) 
ttk.Label(optimization_frame, text="Factor de Penalización:").grid(row=1, column=0) 
entry_penal.grid(row=1, column=1, padx=5, pady=2) 
ttk.Label(optimization_frame, text="Parámetro de Sensibilidad:").grid(row=2, column=0) 
entry_rmin.grid(row=2, column=1, padx=5, pady=2)

# Default Values
entry_nelx.insert(0, "10") 
entry_nely.insert(0, "5") 
entry_nelz.insert(0, "10") 
entry_E0.insert(0, "1.0") 
entry_nu.insert(0, "0.3") 
entry_x_load.insert(0, "5") 
entry_y_load.insert(0, "5") 
entry_z_load.insert(0, "5") 
entry_volfrac.insert(0, "0.2") 
entry_penal.insert(0, "3.0") 
entry_rmin.insert(0, "1.5")

xPhys = None

def start_optimization(): 
    ax1.clear()
    ax2.clear()
    try: 
        nelx_val = int(entry_nelx.get()) 
        nely_val = int(entry_nely.get()) 
        nelz_val = int(entry_nelz.get()) 
        volfrac_val = float(entry_volfrac.get()) 
        penal_val = float(entry_penal.get()) 
        rmin_val = float(entry_rmin.get()) 
        x_load_val = int(entry_x_load.get()) 
        y_load_val = int(entry_y_load.get()) 
        z_load_val = int(entry_z_load.get()) 
        e0_material_val = float(entry_E0.get()) 
        nu_material_val = float(entry_nu.get())
        # Run the topology optimization
        global xPhys
        xPhys, c, history = optimize_topology(
            ax1,
            nelx_val, nely_val, nelz_val,
            x_load_val, y_load_val, z_load_val,
            volfrac_val, penal_val, rmin_val,
            maxloop=200, tolx=0.01,
            displayflag=False,
            e0_material=e0_material_val,
            nu_material=nu_material_val
        )

        # Show final iso-surface in ax2
        ax2.clear()
        display_iso_surface_3D(ax2, xPhys)
        canvas.draw()

    except ValueError:
        print("Invalid input. Please enter valid numerical values.")

def save_stl(): 
    if xPhys is None: 
        print("No topology to save. Run the optimization first.") 
    else:
        export_optimized_stl(xPhys, filename='optimized.stl') 
        export_iso_stl(xPhys, filename='optimized_iso.stl') 
        print("STL files saved.")

start_button = ttk.Button(root, text="Optimizar", command=start_optimization) 
start_button.grid(row=4, column=0, pady=10)

save_stl_button = ttk.Button(root, text="Guardar STL", command=save_stl) 
save_stl_button.grid(row=5, column=0, pady=10)

root.mainloop()