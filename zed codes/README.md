ZED + YOLO 3D Vision Toolkit

Questo progetto contiene script per acquisizione, segmentazione e analisi 3D di oggetti utilizzando:

ZED stereo camera

YOLO (Ultralytics)

Point cloud 3D

Segmentazione semantica

Analisi geometrica

🔹 Funzionalità principali

✔ acquisizione immagini e depth
✔ detection e segmentazione in tempo reale
✔ estrazione point cloud degli oggetti
✔ visualizzazione 3D live
✔ analisi geometrica (curvatura, raggio)
✔ registrazione dataset sincronizzati

📦 Requisiti
pip install ultralytics opencv-python numpy open3d

Installa anche:

ZED SDK

pyzed

CUDA (consigliato)

📁 CLASSI DI SCRIPT
1️⃣ Test e acquisizione dati
hello_zed.py

Test connessione ZED.

zed_opencv.py

Acquisizione e salvataggio immagini, depth e point cloud.

👉 utile per creare dataset raw.

2️⃣ Detection YOLO (2D)
zed_yolo.py

Detection con bounding box.

zed_yolo_fin_v*.py

Versioni migliorate e più robuste.

👉 utile per test veloci.

3️⃣ Segmentazione YOLO
zed_yolo_seg_apple.py

Segmentazione e overlay maschere.

👉 utile per debugging visivo.

4️⃣ Segmentazione + estrazione point cloud
zed_pc_yolo.py

Estrazione punti 3D dagli oggetti segmentati.

zed_pc_yolo_v2.py

Filtra solo oggetto target.

zed_pc_yolo_v3.py

Salva point cloud segmentata.

👉 primo passo verso analisi 3D.

5️⃣ Visualizzazione point cloud filtrata
zed_seg_yolo.py

Visualizza cloud filtrata tramite maschere.

zed_yolo_seg_off*.py

Filtraggio tramite bounding box (più veloce).

👉 utile per debugging e visualizzazione live.

6️⃣ Segmentazione + analisi geometrica
zed_yolo_seg+pc_apple.py

Stima curvatura e raggio.

zed_yolo_seg+pc_applev2.py

Mesh e identificazione area di grasp.

👉 utile per robotica e manipolazione.

7️⃣ Registrazione dataset
zed_yolo_seg_off4.py

Registra video + maschere + point cloud.

zed_yolo_seg_offv2.py

Salva cloud segmentate.

zed_yolo_seg+pc_applev3.py

Salva bounding box e point cloud.

👉 ideale per training AI e analisi offline.

8️⃣ Integrazione avanzata
yolo11_zed.py

YOLO in thread separato + tracking ZED.

👉 pipeline real-time professionale.

🧭 Quale script usare?
✔ detection semplice

→ zed_yolo.py

✔ estrarre point cloud oggetti

→ zed_pc_yolo_v2.py

✔ visualizzare oggetti 3D live

→ zed_seg_yolo.py

✔ analisi geometrica

→ zed_yolo_seg+pc_apple.py

✔ creare dataset

→ zed_yolo_seg_off4.py

✔ sistema real-time avanzato

→ yolo11_zed.py
