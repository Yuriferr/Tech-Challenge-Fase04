import os

# --- CORREÇÃO DE ERRO PROTOBUF (Essencial para não travar o DeepFace/MediaPipe) ---
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import cv2
import mediapipe as mp
from deepface import DeepFace
from ultralytics import YOLO
from collections import Counter
import math
import numpy as np
from moviepy.editor import VideoFileClip # Requer moviepy==1.0.3

# --- CONFIGURAÇÕES GERAIS ---
INPUT_VIDEO = "Unlocking Facial Recognition_ Diverse Activities Analysis.mp4"
TEMP_VIDEO = "temp_video_mudo.mp4"       # Arquivo temporário (sem áudio)
OUTPUT_VIDEO = "resultado_final_audio.mp4" # Arquivo final (com áudio)
REPORT_FILE = "relatorio_analise.txt"

# Sensibilidade para movimento brusco (quanto menor, mais sensível)
MOVEMENT_THRESHOLD = 40 

# --- 1. INICIALIZAÇÃO DOS MODELOS ---
print(">>> Carregando modelos de IA... (Isso pode demorar alguns segundos)")

# YOLOv8 (Detecta objetos para inferir atividades)
yolo_model = YOLO('yolov8n.pt') 

# MediaPipe (Detecta rostos com alta precisão)
mp_face_detection = mp.solutions.face_detection

# Preparação do Vídeo
cap = cv2.VideoCapture(INPUT_VIDEO)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Gravador temporário (OpenCV não grava áudio)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(TEMP_VIDEO, fourcc, fps, (width, height))

# --- MAPEAMENTO DE OBJETOS -> ATIVIDADES ---
# IDs baseados no COCO Dataset
activity_map = {
    67: "Usando Celular", 77: "Usando Celular", # Cell phone
    63: "Trabalhando (PC)",  # Laptop
    64: "Trabalhando (Mouse)", 
    66: "Trabalhando (Teclado)",
    73: "Lendo / Estudando", # Book
    41: "Bebendo / Pausa"    # Cup
}

# --- VARIÁVEIS DE ESTATÍSTICA ---
stats = {
    'frames': 0,
    'anomalies': 0,
    'emotions': Counter(),
    'activities': Counter()
}

prev_faces_centers = []
current_activity = "Analisando..."

print(">>> Iniciando processamento visual...")

# --- LOOP DE PROCESSAMENTO ---
with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        stats['frames'] += 1
        
        # Converte para RGB para o MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        
        # ---------------------------------------------------------
        # 1. DETECÇÃO DE ATIVIDADE (Contexto via YOLO)
        # ---------------------------------------------------------
        # Roda a cada 10 frames para performance
        if stats['frames'] % 10 == 0:
            results = yolo_model.predict(frame, classes=list(activity_map.keys()), conf=0.4, verbose=False)
            found_activities = []
            
            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    if cls_id in activity_map:
                        found_activities.append(activity_map[cls_id])
            
            if found_activities:
                # Pega a atividade mais comum encontrada na cena
                current_activity = max(set(found_activities), key=found_activities.count)
            else:
                current_activity = "Conversando / Ocioso"
        
        stats['activities'][current_activity] += 1

        # ---------------------------------------------------------
        # 2. INTERFACE (Barra Transparente e Texto Discreto)
        # ---------------------------------------------------------
        overlay = frame.copy()
        # Barra preta de 55px de altura
        cv2.rectangle(overlay, (0, 0), (width, 55), (0, 0, 0), -1)
        
        # Mistura imagem original com a barra preta (60% opacidade)
        alpha = 0.6 
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        # Texto Informativo (Branco e Vermelho)
        cv2.putText(frame, f"ATIVIDADE: {current_activity}", (15, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.putText(frame, f"ANOMALIAS (MOVIMENTO): {stats['anomalies']}", (15, 45), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)

        # ---------------------------------------------------------
        # 3. DETECÇÃO FACIAL, EMOÇÕES E ANOMALIAS
        # ---------------------------------------------------------
        face_results = face_detection.process(frame_rgb)
        current_faces_centers = []
        is_anomaly_frame = False

        if face_results.detections:
            # Regra Social: Se não tem objetos (Ocioso) mas tem várias pessoas, é interação
            if current_activity == "Conversando / Ocioso" and len(face_results.detections) > 1:
                current_activity = "Interacao Social"

            for detection in face_results.detections:
                # Extrai coordenadas do rosto
                bboxC = detection.location_data.relative_bounding_box
                x, y, w, h = int(bboxC.xmin * width), int(bboxC.ymin * height), \
                             int(bboxC.width * width), int(bboxC.height * height)
                x, y = max(0, x), max(0, y) # Garante que não é negativo

                # --- Checagem de Anomalia (Movimento Brusco) ---
                center_x, center_y = x + w//2, y + h//2
                current_faces_centers.append((center_x, center_y))
                
                box_color = (0, 255, 0) # Verde
                thickness = 1           # Fino
                
                if prev_faces_centers:
                    # Calcula distância para o rosto mais próximo no frame anterior
                    min_dist = min([math.hypot(center_x - px, center_y - py) for px, py in prev_faces_centers])
                    
                    if min_dist > MOVEMENT_THRESHOLD:
                        is_anomaly_frame = True
                        box_color = (0, 0, 255) # Vermelho
                        thickness = 2
                        cv2.putText(frame, "MOVIMENTO!", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

                # Desenha o retângulo no rosto
                cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, thickness)

                # --- Análise de Emoção (DeepFace) ---
                # Apenas analisa se o rosto for grande o suficiente
                face_roi = frame[y:y+h, x:x+w]
                if face_roi.size > 0:
                    try:
                        # Analisa a cada 2 frames para não travar
                        if stats['frames'] % 2 == 0:
                            analysis = DeepFace.analyze(img_path=face_roi, actions=['emotion'], 
                                                      enforce_detection=False, detector_backend='skip', silent=True)
                            
                            emotion = analysis[0]['dominant_emotion']
                            stats['emotions'][emotion] += 1
                            
                            # Escreve a emoção pequena acima do rosto
                            cv2.putText(frame, emotion, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)
                    except:
                        pass # Ignora erros de detecção em frames borrados

        # Atualiza histórico para detectar movimento no próximo frame
        prev_faces_centers = current_faces_centers

        if is_anomaly_frame:
            stats['anomalies'] += 1
            cv2.putText(frame, "ALERTA: ANOMALIA", (width - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Grava frame
        out.write(frame)
        
        # Mostra na tela (Opcional, comente se estiver rodando em servidor)
        cv2.imshow('Analise de IA', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
out.release()
cv2.destroyAllWindows()

# ---------------------------------------------------------
# 4. PÓS-PROCESSAMENTO: INSERÇÃO DE ÁUDIO
# ---------------------------------------------------------
print("\n>>> Processamento visual concluído. Adicionando áudio original...")

try:
    if os.path.exists(TEMP_VIDEO):
        # Carrega clips
        video_clip = VideoFileClip(TEMP_VIDEO)
        original_audio = VideoFileClip(INPUT_VIDEO).audio
        
        # Combina
        final_clip = video_clip.set_audio(original_audio)
        
        # Salva (codec libx264 é o padrão universal)
        final_clip.write_videofile(OUTPUT_VIDEO, codec='libx264', audio_codec='aac')
        
        # Limpa arquivos
        video_clip.close()
        del original_audio
        del final_clip
        
        # Remove temporário
        if os.path.exists(TEMP_VIDEO):
            os.remove(TEMP_VIDEO)
            print("Arquivo temporário removido.")
    else:
        print("Erro: Arquivo temporário de vídeo não foi gerado.")

except Exception as e:
    print(f"ERRO CRÍTICO NO ÁUDIO: {e}")
    print("Verifique se o MoviePy está na versão 1.0.3")

# ---------------------------------------------------------
# 5. GERAÇÃO DE RELATÓRIO TXT
# ---------------------------------------------------------
print(f"\n>>> Gerando relatório em: {REPORT_FILE}")

with open(REPORT_FILE, "w", encoding="utf-8") as f:
    f.write("RELATORIO FINAL DE MONITORAMENTO DE VIDEO\n")
    f.write("=======================================\n")
    f.write(f"Total de Frames Analisados: {stats['frames']}\n")
    f.write(f"Total de Anomalias (Movimentos Bruscos): {stats['anomalies']}\n\n")
    
    f.write("--- ATIVIDADES DETECTADAS (Contexto) ---\n")
    if stats['activities']:
        for act, count in stats['activities'].most_common():
            pct = (count / stats['frames']) * 100
            f.write(f"[ {pct:.1f}% ] {act} ({count} frames)\n")
    else:
        f.write("Nenhuma atividade específica detectada.\n")
            
    f.write("\n--- EMOCOES PREDOMINANTES ---\n")
    total_emotions = sum(stats['emotions'].values())
    if total_emotions > 0:
        for emo, count in stats['emotions'].most_common():
            pct = (count / total_emotions) * 100
            f.write(f"[ {pct:.1f}% ] {emo}\n")
    else:
        f.write("Nenhuma emoção detectada com clareza.\n")

print("\n=======================================")
print(f"CONCLUÍDO COM SUCESSO!")
print(f"Vídeo: {OUTPUT_VIDEO}")
print(f"Relatório: {REPORT_FILE}")
print("=======================================")