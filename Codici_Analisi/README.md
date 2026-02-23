# Point Cloud Ball Analysis – Overview

Questo progetto raccoglie diversi script sperimentali per analizzare una pallina (o oggetti simili) a partire da point cloud 3D.

Gli script esplorano approcci differenti:

* segmentazione e pulizia della point cloud
* stima del raggio tramite **fit di sfera**
* fit di **ellissoide** per rilevare deformazioni
* confronto tra modelli geometrici
* ricostruzione di mesh
* analisi della **curvatura** locale o globale

---

## 🔹 Requisiti principali

Gli script utilizzano alcune di queste librerie:

* Open3D
* NumPy
* SciPy
* Trimesh
* PyVista / Vedo (visualizzazione)
* Matplotlib (color map)

Installazione tipica:

```bash
pip install open3d numpy scipy trimesh pyvista vedo matplotlib
```

---

# CLASSI DI SCRIPT

## 1️⃣ Segmentazione + Fit Sfera

Stimano centro e raggio della pallina.

### `test2.py`

Pipeline completa:

* filtro radiale
* DBSCAN
* fit sfera
* rimozione outlier post-fit
* RMSE e visualizzazione

**Quando usarlo:** stima robusta del raggio.

---

### `prova_codice1.py`

Segmenta tutta la scena e fa fit sfera per ogni cluster.

**Quando usarlo:** esplorare oggetti multipli.

---

## 2️⃣ Confronto modelli geometrici

### `prova_codice2.py`

Fit sfera ed ellissoide e scelta del modello migliore (in base all’errore).

⚠️ Richiede correzione del codice (scope della funzione ellissoide).

**Quando usarlo:** verificare deformazioni.

---

## 3️⃣ Fit Ellissoide

Servono per rilevare deformazioni o oggetti non perfettamente sferici.

### `test_fitting_ell.py`

Fit ellissoide diretto con inizializzazione semplice.

### `test_fitting_ell2.py`

Fit ellissoide con inizializzazione migliore e mesh più pulita.

### `test_fitting_ell3.py`

Fit ellissoide con normalizzazione per stabilità numerica.

**Quando usarli:** analisi di deformazioni.

---

## 4️⃣ Mesh + Curvatura

Ricostruiscono una superficie e stimano la curvatura.

### `test6.py`

Mesh Poisson + curvatura media.

### `test9.py`

Mesh + curvatura gaussiana/media con visualizzazione.

**Quando usarli:** analisi della superficie.

---

## 5️⃣ Curvatura direttamente sulla Point Cloud

### `test6_fabio.py`

Curvatura stimata tramite PCA locale.

### `test7.py`

Stima curvature principali tramite fit quadratico locale.

### `test8.py`

Come test7 ma con preprocessing e diagnostica migliorata.

**Quando usarli:** analisi locale della geometria.

---

## 6️⃣ Mesh BPA + confronto modelli

### `test10.py`

* DBSCAN
* Mesh Ball Pivoting
* Fit sfera vs ellissoide

**Quando usarlo:** confronto avanzato tra modelli geometrici.

---

# 📊 Quale script scegliere?

### Per stimare il raggio

👉 `test2.py`

### Per verificare deformazioni

👉 `test10.py` oppure script ellissoide

### Per analizzare la curvatura

👉 `test8.py` (senza mesh)
👉 `test9.py` (con mesh)

---

# ⚠️ Note importanti

* I risultati dipendono fortemente da:

  * densità della point cloud
  * rumore
  * parametri DBSCAN
  * scelta dei vicini nelle curvature

* La ricostruzione mesh può introdurre artefatti.

---

# 📁 Struttura dati attesa

Gli script assumono file del tipo:

```
pointcloud_0005.ply
ball_frame_0005.ply
```

Modificare i percorsi se necessario.

---

# ✨ Scopo del progetto

Questo insieme di script serve per:

* stimare il raggio di una pallina
* valutare deformazioni
* analizzare la curvatura
* confrontare metodi geometrici

---

**Autore:** esperimenti di analisi geometrica su point cloud
