import cv2
import mediapipe as mp
from deepface import DeepFace
from ultralytics import YOLO
import numpy as np
from collections import Counter
import math

# --- Configurações ---
input_video_path = "Unlocking Facial Recognition_ Diverse Activities Analysis.mp4"
output_video_path = "resultado_final.mp4"
report_path = "relatorio_analise.txt"

MOVEMENT_THRESHOLD = 40 

# --- Inicialização ---
print("1. Carregando modelo YOLO (Atividades)...")
yolo_model = YOLO('yolov8n.pt') 

print("2. Configurando MediaPipe (Rostos)...")
mp_face_detection = mp.solutions.face_detection

cap = cv2.VideoCapture(input_video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_count_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# --- Mapeamento ---
activity_map = {
    67: "Usando Celular", 77: "Usando Celular",
    63: "Trabalhando (PC)", 64: "Trabalhando (Mouse)", 66: "Trabalhando (Teclado)",
    73: "Lendo / Estudando", 41: "Bebendo / Pausa"
}

# --- Variáveis ---
stats = {
    'total_frames': 0,
    'anomalies': 0,
    'emotions': Counter(),
    'activities': Counter()
}

prev_faces_centers = []
current_activity = "Analisando..."
print("Iniciando processamento com interface transparente...")

with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        stats['total_frames'] += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        
        # --- 1. Detecção de Atividade (YOLO - a cada 10 frames) ---
        if stats['total_frames'] % 10 == 0:
            results = yolo_model.predict(frame, classes=list(activity_map.keys()), conf=0.4, verbose=False)
            found_activities = []
            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    if cls_id in activity_map:
                        found_activities.append(activity_map[cls_id])
            
            if found_activities:
                current_activity = max(set(found_activities), key=found_activities.count)
            else:
                current_activity = "Conversando / Ocioso"
        
        stats['activities'][current_activity] += 1

        # --- 2. Interface Transparente (NOVO CÓDIGO) ---
        # Cria uma cópia da imagem original para desenhar o retângulo
        overlay = frame.copy()
        
        # Desenha o retângulo preto na cópia (altura reduzida para 55px)
        cv2.rectangle(overlay, (0, 0), (width, 55), (0, 0, 0), -1)
        
        # Aplica a transparência: 0.6 = 60% opacidade (vidro fumê)
        alpha = 0.6 
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        # Escreve o texto menor (fontScale 0.5)
        # Texto Branco
        cv2.putText(frame, f"ATIVIDADE: {current_activity}", (10, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Texto Vermelho (Anomalias)
        cv2.putText(frame, f"ANOMALIAS: {stats['anomalies']}", (10, 45), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # --- 3. Detecção Facial e Anomalias ---
        face_results = face_detection.process(frame_rgb)
        current_faces_centers = []
        is_anomaly_frame = False

        if face_results.detections:
            if current_activity == "Conversando / Ocioso" and len(face_results.detections) > 1:
                current_activity = "Interacao Social"
            
            for detection in face_results.detections:
                bboxC = detection.location_data.relative_bounding_box
                x, y, w, h = int(bboxC.xmin * width), int(bboxC.ymin * height), \
                             int(bboxC.width * width), int(bboxC.height * height)
                x, y = max(0, x), max(0, y)
                
                # Anomalia de Movimento
                center_x, center_y = x + w//2, y + h//2
                current_faces_centers.append((center_x, center_y))
                
                color = (0, 255, 0) 
                thickness = 1 # Linha mais fina para ser discreto
                
                if prev_faces_centers:
                    min_dist = min([math.hypot(center_x - px, center_y - py) for px, py in prev_faces_centers])
                    if min_dist > MOVEMENT_THRESHOLD:
                        is_anomaly_frame = True
                        color = (0, 0, 255)
                        thickness = 2
                        # Texto de anomalia menor também
                        cv2.putText(frame, "MOVIMENTO!", (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)

                # Análise de Emoção
                face_roi = frame[y:y+h, x:x+w]
                if face_roi.size > 0:
                    try:
                        if stats['total_frames'] % 2 == 0:
                            analysis = DeepFace.analyze(img_path=face_roi, actions=['emotion'], 
                                                      enforce_detection=False, detector_backend='skip', silent=True)
                            emotion = analysis[0]['dominant_emotion']
                            stats['emotions'][emotion] += 1
                            # Texto da emoção reduzido (0.5)
                            cv2.putText(frame, emotion, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    except:
                        pass

        prev_faces_centers = current_faces_centers

        if is_anomaly_frame:
            stats['anomalies'] += 1
            # Alerta no canto direito, menor
            cv2.putText(frame, "ALERTA: ANOMALIA", (width - 150, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        out.write(frame)
        cv2.imshow('Monitoramento Inteligente', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
out.release()
cv2.destroyAllWindows()

# --- GERA RELATORIO ---
print("Gerando relatório...")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("RELATORIO FINAL DE MONITORAMENTO\n")
    f.write("================================\n")
    f.write(f"Frames: {stats['total_frames']} | Anomalias: {stats['anomalies']}\n\n")
    f.write("ATIVIDADES:\n")
    for act, count in stats['activities'].most_common():
        f.write(f"- {act}: {count} frames\n")
    f.write("\nEMOCOES:\n")
    for emo, count in stats['emotions'].most_common():
        f.write(f"- {emo}: {count}\n")

print(f"Concluído! Vídeo: {output_video_path}")