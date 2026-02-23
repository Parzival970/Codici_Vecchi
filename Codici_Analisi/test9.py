import numpy as np
from scipy.spatial import KDTree

# Caricamento della nuvola di punti (sostituire con il path del file o array di punti già ottenuti)
# Esempio con Open3D (se disponibile):
import open3d as o3d
pcd = o3d.io.read_point_cloud("ball_pointclouds/ball_frame_0005.ply")
points = np.asarray(pcd.points)

# Costruzione di un albero k-d per ricerche veloci dei vicini
tree = KDTree(points)

# Parametri
k_neighbors = 50   # numero di vicini da considerare per ogni punto (escluso il punto stesso)
curvatures = []    # lista per salvare (k1, k2, R1, R2) di ogni punto

for i, p in enumerate(points):
    # Trova i k vicini più prossimi (incluso il punto stesso)
    dists, idx = tree.query(p, k=k_neighbors+1)  # k+1 per includere il punto corrente
    neighbor_idx = idx[1:]   # escludi il punto stesso (prima entry)
    neighbors = points[neighbor_idx]
    
    # Trasla i vicini per portare p all'origine
    neighbors_centered = neighbors - p  # p diventa (0,0,0)
    
    # Stima della normale locale via PCA (autovalore minore)
    cov = np.cov(neighbors_centered.T)  
    eigvals, eigvecs = np.linalg.eigh(cov)
    normal = eigvecs[:, np.argmin(eigvals)]  
    normal = normal / np.linalg.norm(normal)  # normalizzazione
    
    # Definisci assi tangenti u, v ortogonali a normal
    # Si sceglie un vettore di riferimento non parallelo a normal
    ref_axis = np.array([1, 0, 0]) if abs(normal[2]) > 0.9 else np.array([0, 0, 1])
    u_axis = np.cross(normal, ref_axis)
    u_axis /= np.linalg.norm(u_axis)
    v_axis = np.cross(normal, u_axis)
    v_axis /= np.linalg.norm(v_axis)
    
    # Coordinate dei vicini nel sistema locale (u,v,n)
    U = neighbors_centered.dot(u_axis)   # proiezione su u
    V = neighbors_centered.dot(v_axis)   # proiezione su v
    W = neighbors_centered.dot(normal)   # coordinata lungo n (uscente)
    
    # Costruisci matrice del sistema lineare per il fit: W = A*U^2 + B*U*V + C*V^2 + D*U + E*V
    A_mat = np.column_stack([U**2, U*V, V**2, U, V])
    b_vec = W
    # Risolve i minimi quadrati per i coefficienti [A, B, C, D, E]
    coeff, *_ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
    A, B, C, D, E = coeff  # estrai i coefficienti
    
    # Calcola la matrice Hessiana H = [[2A, B], [B, 2C]]
    H = np.array([[2*A, B],
                  [B, 2*C]])
    # Autovalori di H -> curvature principali k1, k2
    k1, k2 = np.linalg.eigvals(H)
    # Ordina k1, k2 per convenzione (k1 = maggiore curvatura)
    k1, k2 = sorted([k1, k2], key=lambda x: abs(x), reverse=True)
    
    # Calcola i raggi di curvatura (valori assoluti, evitando divisioni per 0)
    R1 = np.inf if abs(k1) < 1e-8 else 1.0/ k1
    R2 = np.inf if abs(k2) < 1e-8 else 1.0/ k2
    
    curvatures.append((k1, k2, R1, R2))

# 'curvatures' ora contiene per ogni punto la coppia di curvature e raggi di curvatura.
# Estrai k1 e k2
k1_vals = np.array([k[0] for k in curvatures])
k2_vals = np.array([k[1] for k in curvatures])

# Calcola curvatura gaussiana
K_vals = k1_vals * k2_vals

# Normalizza K a [0,1] per colormap
K_min, K_max = np.percentile(K_vals, [1, 99])  # evita outlier estremi
K_norm = np.clip((K_vals - K_min) / (K_max - K_min), 0, 1)

# Mappa a colori RGB con una colormap (es. viridis)
import matplotlib.pyplot as plt
colormap = plt.cm.get_cmap('viridis')
colors = colormap(K_norm)[:, :3]  # prendi solo RGB

# Assegna i colori al point cloud di Open3D
pcd.colors = o3d.utility.Vector3dVector(colors)

# Visualizza
o3d.visualization.draw_geometries([pcd])
