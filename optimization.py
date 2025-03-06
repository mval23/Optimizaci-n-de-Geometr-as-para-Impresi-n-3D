import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import math
import matplotlib as plt

plt.use("TkAgg")
from skimage import measure
import trimesh


def lk_H8(nu):
    """
    Calcula la matriz de rigidez de un elemento hexaédrico (H8), considerando la relación de Poisson. Define cómo se
    deforma el elemento bajo carga.
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


def get_load_and_fixed_dofs(nelx, nely, nelz):
    """
    Determina los grados de libertad donde se aplican cargas y las condiciones de frontera fijas.
    """
    # Load DOFs: nodes at x = nelx, y = 0, z = 0:nelz.
    il, jl, kl = np.meshgrid(np.array([nelx//2]), np.array([nely]), np.array([nelz//2]), indexing='ij')
    #il, jl, kl = np.meshgrid(np.array([nelx / 2]), np.array([0]), np.array([nelz / 2]), indexing='ij')

    load_nid = (kl * ((nelx + 1) * (nely + 1)) + il * (nely + 1) + (nely + 1 - jl)) - 1
    load_nid = load_nid.flatten()
    load_dof = 3 * load_nid + 1

    # Fixed DOFs: nodes at x = 0, for y = 0:nely and z = 0:nelz.
    # iif, jf, kf = np.meshgrid(np.array([0]), np.arange(nely + 1), np.arange(nelz + 1), indexing='ij')
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
    F[load_dof] = -1.0  # Apply load # Se definen las fuerzas aplicadas
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
    Crea un filtro de sensibilidad para suavizar los resultados y evitar patrones irregulares en la distribución del material."""
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


def optimize_topology(nelx, nely, nelz, volfrac, penal, rmin, maxloop=200, tolx=0.01, displayflag=False, e0_material= 1.0, nu_material= 0.3):
    """
    Función principal que realiza el proceso iterativo de optimización usando el método SIMP. Calcula la distribución
    de material óptima para minimizar la complacencia.
    """
    # Material properties
    E0 = e0_material # 1.0 modulo de young 
    Emin = 1e-9
    nu = nu_material # 0.3 coeficiente poission 
    KE = lk_H8(nu)

    # FE setup
    load_dof, fixed_dof = get_load_and_fixed_dofs(nelx, nely, nelz)
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

        if displayflag:
            display_3D(xPhys)

    if displayflag:
        display_3D(xPhys)

    return xPhys, c, history


def display_3D(rho, threshold=0.5):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import numpy as np

    # Get the array dimensions: note the MATLAB ordering (nely, nelx, nelz)
    nely, nelx, nelz = rho.shape

    # Define unit element sizes
    hx = 1
    hy = 1
    hz = 1

    # Define the connectivity of the cube faces (MATLAB indices start at 1; subtract 1 for Python)
    # MATLAB face matrix:
    #   [1 2 3 4; 2 6 7 3; 4 3 7 8; 1 5 8 4; 1 2 6 5; 5 6 7 8]
    faces = [
        [0, 1, 2, 3],
        [1, 5, 6, 2],
        [3, 2, 6, 7],
        [0, 4, 7, 3],
        [0, 1, 5, 4],
        [4, 5, 6, 7]
    ]

    # Create a new figure and 3D axes.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('ISO display')

    # Loop over the 3D grid. In MATLAB, the loops are: for k=1:nelz, for i=1:nelx, for j=1:nely.
    # We use 0-based indexing here.
    for k in range(nelz):
        z = k * hz
        for i in range(nelx):
            x = i * hx
            for j in range(nely):
                # MATLAB: y = nely*hy - (j-1)*hy, so for j in Python (0-based) do:
                y = nely * hy - j * hy
                # Check if the density in this element is above the threshold.
                if rho[j, i, k] > threshold:
                    # Define the vertices of a unit cube at (x, y, z).
                    # The MATLAB code builds:
                    #   vert = [x y z;
                    #           x y-hx z;
                    #           x+hx y-hx z;
                    #           x+hx y z;
                    #           x y z+hx;
                    #           x y-hx z+hx;
                    #           x+hx y-hx z+hx;
                    #           x+hx y z+hx];
                    # Then it swaps columns 2 and 3 and negates the new second column.
                    v1 = [x, y, z]
                    v2 = [x, y - hx, z]
                    v3 = [x + hx, y - hx, z]
                    v4 = [x + hx, y, z]
                    v5 = [x, y, z + hx]
                    v6 = [x, y - hx, z + hx]
                    v7 = [x + hx, y - hx, z + hx]
                    v8 = [x + hx, y, z + hx]
                    verts = np.array([v1, v2, v3, v4, v5, v6, v7, v8])

                    # Swap the 2nd and 3rd coordinates:
                    verts[:, [1, 2]] = verts[:, [2, 1]]
                    # Multiply the new 2nd column by -1:
                    verts[:, 1] = -verts[:, 1]

                    # Determine face color. MATLAB uses:
                    #   [0.2+0.8*(1 - rho(j,i,k))] for each RGB channel.
                    col_val = 0.2 + 0.8 * (1 - rho[j, i, k])
                    face_color = (col_val, col_val, col_val)

                    # Create a Poly3DCollection for this cube using the defined faces.
                    poly3d = Poly3DCollection(verts[faces], facecolors=face_color, edgecolors='k')
                    ax.add_collection3d(poly3d)

    # Set axis limits.
    ax.set_xlim(0, nelx * hx)
    # After swapping, the y-axis corresponds to -z, so set limits accordingly.
    ax.set_ylim(-nelz * hz, 0)
    ax.set_zlim(0, nely * hy)

    # Set an equal aspect ratio. (Note: aspect handling in 3D may be a bit approximate.)
    ax.set_box_aspect((nelx * hx, nelz * hz, nely * hy))

    # Turn off the axis, and set a view angle similar to MATLAB's view([30,30]).
    ax.set_axis_off()
    ax.view_init(elev=30, azim=30)
    plt.tight_layout()
    plt.show()
    plt.pause(1)  # Pause briefly to allow the figure to update
    # plt.close()       # Close the figure window after display


def display_solid_3D(rho, threshold=0.5):
    """
    Display the optimized 3D topology as a solid made of cubes, one per element
    with density above the threshold.

    The routine is based on the MATLAB function you provided. It loops through
    the (nely, nelx, nelz) array, and for each element with density > threshold,
    it creates a unit cube (with user-defined element sizes hx, hy, hz = 1) and
    draws its six faces. The vertices are transformed (swapping the 2nd and 3rd
    coordinates and negating the new 2nd coordinate) to match the MATLAB behavior.

    Parameters:
        rho       : 3D numpy array with shape (nely, nelx, nelz).
                    (Note: The first dimension is the y-index, the second is x,
                    and the third is z—as in the MATLAB code.)
        threshold : Density threshold (default 0.5) above which the element is drawn.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import numpy as np

    # Get the array dimensions: note the MATLAB ordering (nely, nelx, nelz)
    nely, nelx, nelz = rho.shape

    # Define unit element sizes
    hx = 1
    hy = 1
    hz = 1

    # Define the connectivity of the cube faces (MATLAB indices start at 1; subtract 1 for Python)
    # MATLAB face matrix:
    #   [1 2 3 4; 2 6 7 3; 4 3 7 8; 1 5 8 4; 1 2 6 5; 5 6 7 8]
    faces = [
        [0, 1, 2, 3],
        [1, 5, 6, 2],
        [3, 2, 6, 7],
        [0, 4, 7, 3],
        [0, 1, 5, 4],
        [4, 5, 6, 7]
    ]

    # Create a new figure and 3D axes.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('ISO display')

    # Loop over the 3D grid. In MATLAB, the loops are: for k=1:nelz, for i=1:nelx, for j=1:nely.
    # We use 0-based indexing here.
    for k in range(nelz):
        z = k * hz
        for i in range(nelx):
            x = i * hx
            for j in range(nely):
                # MATLAB: y = nely*hy - (j-1)*hy, so for j in Python (0-based) do:
                y = nely * hy - j * hy
                # Check if the density in this element is above the threshold.
                if rho[j, i, k] > threshold:
                    # Define the vertices of a unit cube at (x, y, z).
                    # The MATLAB code builds:
                    #   vert = [x y z;
                    #           x y-hx z;
                    #           x+hx y-hx z;
                    #           x+hx y z;
                    #           x y z+hx;
                    #           x y-hx z+hx;
                    #           x+hx y-hx z+hx;
                    #           x+hx y z+hx];
                    # Then it swaps columns 2 and 3 and negates the new second column.
                    v1 = [x, y, z]
                    v2 = [x, y - hx, z]
                    v3 = [x + hx, y - hx, z]
                    v4 = [x + hx, y, z]
                    v5 = [x, y, z + hx]
                    v6 = [x, y - hx, z + hx]
                    v7 = [x + hx, y - hx, z + hx]
                    v8 = [x + hx, y, z + hx]
                    verts = np.array([v1, v2, v3, v4, v5, v6, v7, v8])

                    # Swap the 2nd and 3rd coordinates:
                    verts[:, [1, 2]] = verts[:, [2, 1]]
                    # Multiply the new 2nd column by -1:
                    verts[:, 1] = -verts[:, 1]

                    # Determine face color. MATLAB uses:
                    #   [0.2+0.8*(1 - rho(j,i,k))] for each RGB channel.
                    col_val = 0.2 + 0.8 * (1 - rho[j, i, k])
                    face_color = (col_val, col_val, col_val)

                    # Create a Poly3DCollection for this cube using the defined faces.
                    poly3d = Poly3DCollection(verts[faces], facecolors=face_color, edgecolors='k')
                    ax.add_collection3d(poly3d)

    # Set axis limits.
    ax.set_xlim(0, nelx * hx)
    # After swapping, the y-axis corresponds to -z, so set limits accordingly.
    ax.set_ylim(-nelz * hz, 0)
    ax.set_zlim(0, nely * hy)

    # Set an equal aspect ratio. (Note: aspect handling in 3D may be a bit approximate.)
    ax.set_box_aspect((nelx * hx, nelz * hz, nely * hy))

    # Turn off the axis, and set a view angle similar to MATLAB's view([30,30]).
    ax.set_axis_off()
    ax.view_init(elev=30, azim=30)
    plt.tight_layout()
    plt.show()


# --- Function to Save the Optimized Topology as an STL File ---

def export_optimized_stl(rho, threshold=0.5, filename='optimized.stl', hx=1, hy=1, hz=1):
    """
    Export the optimized topology (rho) as an STL file using the same
    cube-based geometry as the display_3D function.

    Parameters:
        rho       : 3D numpy array of densities with shape (nely, nelx, nelz).
                    (The array ordering is: first dimension = y, second = x, third = z.)
        threshold : Density threshold (default 0.5) above which an element is considered solid.
        filename  : Name of the output STL file.
        hx, hy, hz: Unit element sizes in x, y, and z directions (default = 1).

    The routine uses the same vertex generation and transformation as your MATLAB code:
      1. For each element with rho[j,i,k] > threshold, compute the eight vertices of a unit cube.
      2. Swap the 2nd and 3rd coordinates and negate the new 2nd coordinate.
      3. For each of the 6 faces (each defined as a quadrilateral), split into two triangles.
      4. Accumulate all triangles and export them using numpy-stl.
    """
    import numpy as np
    from stl import mesh

    nely, nelx, nelz = rho.shape
    triangles = []

    # Face connectivity for a cube (MATLAB indices: [1 2 3 4; 2 6 7 3; 4 3 7 8; 1 5 8 4; 1 2 6 5; 5 6 7 8])
    # Convert to 0-based indexing:
    faces_idx = [
        [0, 1, 2, 3],
        [1, 5, 6, 2],
        [3, 2, 6, 7],
        [0, 4, 7, 3],
        [0, 1, 5, 4],
        [4, 5, 6, 7]
    ]

    # Loop over each element in the grid.
    for k in range(nelz):
        z = k * hz
        for i in range(nelx):
            x = i * hx
            for j in range(nely):
                # MATLAB: y = nely*hy - (j-1)*hy; for Python 0-based j:
                y = nely * hy - j * hy
                if rho[j, i, k] > threshold:
                    # Compute the 8 vertices of the cube at this element.
                    # (They are defined exactly as in your MATLAB code.)
                    v1 = [x, y, z]
                    v2 = [x, y - hx, z]
                    v3 = [x + hx, y - hx, z]
                    v4 = [x + hx, y, z]
                    v5 = [x, y, z + hx]
                    v6 = [x, y - hx, z + hx]
                    v7 = [x + hx, y - hx, z + hx]
                    v8 = [x + hx, y, z + hx]
                    verts = np.array([v1, v2, v3, v4, v5, v6, v7, v8], dtype=float)

                    # Swap the 2nd and 3rd columns:
                    verts[:, [1, 2]] = verts[:, [2, 1]]
                    # Negate the new second column:
                    verts[:, 1] = -verts[:, 1]

                    # For each face, split the quad into two triangles.
                    for face in faces_idx:
                        # A quadrilateral face: vertices v0, v1, v2, v3
                        # Triangle 1: [v0, v1, v2]
                        tri1 = [verts[face[0]], verts[face[1]], verts[face[2]]]
                        # Triangle 2: [v0, v2, v3]
                        tri2 = [verts[face[0]], verts[face[2]], verts[face[3]]]
                        triangles.append(tri1)
                        triangles.append(tri2)

    # If no triangles were generated, warn and exit.
    if len(triangles) == 0:
        print("No elements above threshold. STL not created.")
        return

    triangles_array = np.array(triangles, dtype=np.float32)

    # Create an stl.Mesh object.
    stl_mesh = mesh.Mesh(np.zeros(triangles_array.shape[0], dtype=mesh.Mesh.dtype))
    for i, tri in enumerate(triangles_array):
        stl_mesh.vectors[i] = tri

    stl_mesh.save(filename)
    print(f"STL file saved as '{filename}'.")

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
from stl import mesh

def display_iso_surface_3D(rho, threshold=0.5):
    """
    Muestra la topología optimizada como una isosuperficie suave en 3D.
    Usa el algoritmo Marching Cubes para generar una superficie continua.
    """
    # Agregar padding para mejorar la generación en los bordes
    padded_rho = np.pad(rho, pad_width=1, mode='constant', constant_values=0)
    
    verts, faces, _, _ = measure.marching_cubes(padded_rho, level=threshold)
    verts -= 1  # Ajustar por el padding
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Isosuperficie de la Topología Optimizada')
    
    mesh = Poly3DCollection(verts[faces], alpha=0.7, edgecolor='k')
    ax.add_collection3d(mesh)
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    
    plt.show()

def export_iso_stl(rho, threshold=0.5, filename='optimized_iso.stl'):
    """
    Exporta la topología optimizada en una representación suavizada (ISO) en formato STL.
    """
    # Agregar padding para mejorar la generación en los bordes
    padded_rho = np.pad(rho, pad_width=1, mode='constant', constant_values=0)
    
    verts, faces, _, _ = measure.marching_cubes(padded_rho, level=threshold)
    verts -= 1  # Ajustar por el padding
    
    stl_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            stl_mesh.vectors[i][j] = verts[f[j]]
    
    stl_mesh.save(filename)
    print(f"STL suavizado guardado como '{filename}'")





# --- Main Script ---

if __name__ == "__main__":
    # Define optimization p arameters.
    nelx, nely, nelz = 20, 10, 20  # You may want higher values for a more detailed solid
    volfrac = 0.2
    penal = 3.0
    rmin = 1
    maxloop = 200
    tolx = 0.01

    # Turn off intermediate display to avoid multiple windows.
    displayflag = True

    # Run topology optimization.
    xPhys, c, history = optimize_topology(nelx, nely, nelz, volfrac, penal, rmin, maxloop=maxloop, tolx=tolx,
                                          displayflag=displayflag)

    # Display the final optimized design as a solid.
    display_solid_3D(xPhys, threshold=0.5)

    # Optionally, save the final design as an STL file.
    export_optimized_stl(xPhys, threshold=0.5, filename='optimized.stl', hx=1, hy=1, hz=1)

    # Display the final optimized design as an isosurface.
    display_iso_surface_3D(xPhys, threshold=0.5)

    # Optionally, save the final design as an ISO STL file.
    export_iso_stl(xPhys, threshold=0.5, filename='optimized_iso.stl')



    print("\nOptimization, final solid display, and STL export completed.")
