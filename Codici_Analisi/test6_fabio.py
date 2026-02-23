import open3d as o3d
import numpy as np
import os
import sys
from sklearn.cluster import DBSCAN
from scipy.optimize import minimize

# === 1. CONFIGURAZIONE PATH ===
dir_path = "ball_pointclouds" 
frame = sys.argv[1] if len(sys.argv) > 1 else "0000"
ball_path = os.path.join(dir_path, f"ball_frame_{frame}.ply")

if not os.path.exists(ball_path):
    print(f"[ERRORE] File non trovato: {os.path.abspath(ball_path)}")
    sys.exit(1)

# === 2. CARICAMENTO E FILTRAGGIO ===
pcd_ball = o3d.io.read_point_cloud(ball_path)
points = np.asarray(pcd_ball.points)

if points.shape[0] < 20:
    print(f"[ERRORE] Punti insufficienti nel file.")
    sys.exit(1)

# Filtro radiale iniziale
center_init = points.mean(axis=0)
dists = np.linalg.norm(points - center_init, axis=1)
points = points[dists < 0.08]

# === 3. CLUSTERING DBSCAN ===
labels = DBSCAN(eps=0.0085, min_samples=20).fit(points).labels_

if np.any(labels != -1):
    majority = np.argmax(np.bincount(labels[labels != -1]))
    points_clean = points[labels == majority]
    
    pcd_final = o3d.geometry.PointCloud()
    pcd_final.points = o3d.utility.Vector3dVector(points_clean)
    
    # === 4. RICOSTRUZIONE SUPERFICIE (BPA) ===
    print("Inizio ricostruzione BPA...")
    pcd_final.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
    pcd_final.orient_normals_consistent_tangent_plane(10)

    distances = pcd_final.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radii = [avg_dist * r for r in [1.5, 2.0, 3.0]]

    bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd_final, o3d.utility.DoubleVector(radii)
    )
    print(f"Ricostruzione completata. Triangoli: {len(bpa_mesh.triangles)}")

    # === 5. ANALISI GEOMETRICA (Fitting) ===
    
    def fit_sphere(pts):
        centroid = np.mean(pts, axis=0)
        def objective(params):
            c, r = params[:3], params[3]
            dists = np.linalg.norm(pts - c, axis=1)
            return np.mean((dists - r)**2)
        init = np.append(centroid, np.mean(np.linalg.norm(pts - centroid, axis=1)))
        res = minimize(objective, init)
        return res.x[:3], res.x[3], res.fun

    def fit_ellipsoid(pts):
        centroid = np.mean(pts, axis=0)
        centered_pts = pts - centroid
        cov = np.cov(centered_pts.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        initial_radii = np.sqrt(np.abs(eigenvalues)) * 2.0 
        def objective(params):
            c, r = params[:3], params[3:]
            if np.any(r < 0.0001): return 1e10
            local_pts = (pts - c) @ eigenvectors
            val = np.sum((local_pts / r)**2, axis=1)
            return np.std(val) + np.abs(np.mean(val) - 1.0)
        res = minimize(objective, np.concatenate([centroid, initial_radii]), method='Nelder-Mead')
        return res.x[:3], res.x[3:], eigenvectors

    points_final = np.asarray(pcd_final.points)
    s_center, s_radius, s_error = fit_sphere(points_final)
    e_center, e_radii, e_rot = fit_ellipsoid(points_final)

    print(f"\n[SFERA] Raggio: {s_radius:.5f}, Errore: {s_error:.6f}")
    print(f"[ELLISSOIDE] Semi-assi: {e_radii}")

    # === 6. CREAZIONE VISUALE ===

    # Sfera (Gialla) - CORRETTO: raggio come float
    sphere_vis = o3d.geometry.TriangleMesh.create_sphere(radius=s_radius)
    sphere_vis.translate(s_center)
    sphere_vis.paint_uniform_color([1, 0.7, 0.2])
    sphere_vis.compute_vertex_normals()

    # Ellissoide (Verde) - CORRETTO: applicazione rotazione e scala
    ell_vis = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
    e_verts = (np.asarray(ell_vis.vertices) * e_radii) @ e_rot.T + e_center
    ell_vis.vertices = o3d.utility.Vector3dVector(e_verts)
    ell_vis.paint_uniform_color([0.2, 0.8, 0.4])
    ell_vis.compute_vertex_normals()

    # Mesh BPA (Grigia)
    bpa_mesh.paint_uniform_color([0.6, 0.6, 0.6])
    bpa_mesh.compute_vertex_normals()
    
    # Nuvola punti (Rossa)
    pcd_final.paint_uniform_color([1, 0, 0])

    # === 7. VISUALIZZAZIONE FINALE ===
    print("\n[INFO] Visualizzazione aperta. Premi 'W' per il wireframe.")
    # Visualizziamo tutto insieme per il confronto
    # o3d.visualization.draw_geometries([pcd_final, bpa_mesh, sphere_vis, ell_vis], 
    #                                 window_name=f"Analisi Frame {frame}",
    #                                 mesh_show_back_face=True)
    # o3d.visualization.draw_geometries([pcd_final, bpa_mesh, ell_vis], 
    #                                 window_name=f"Analisi Frame {frame}",
    #                                 mesh_show_back_face=True)

    o3d.visualization.draw_geometries([pcd_final, bpa_mesh], 
                            window_name=f"Analisi Frame {frame}",
                            mesh_show_back_face=True)
else:
    print("[WARN] Nessun cluster trovato.")