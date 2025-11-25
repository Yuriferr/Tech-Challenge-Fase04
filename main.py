import cv2
import mediapipe as mp
from deepface import DeepFace
import numpy as np

# --- Configurações ---
input_video_path = "Unlocking Facial Recognition_ Diverse Activities Analysis.mp4"
output_video_path = "resultado_emocoes.mp4"

# Inicializa MediaPipe (Para encontrar ONDE está o rosto)
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(input_video_path)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

print("Iniciando análise de emoções... (Isso pode ser um pouco mais lento)")

# Usamos model_selection=1 para pegar rostos em várias distâncias
with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Cópia para o MediaPipe (RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        
        # 1. MediaPipe encontra os rostos
        results = face_detection.process(frame_rgb)

        if results.detections:
            for detection in results.detections:
                # Extrair coordenadas
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                             int(bboxC.width * iw), int(bboxC.height * ih)
                
                # Correção de limites (para não quebrar se o rosto sair da tela)
                if x < 0: x = 0
                if y < 0: y = 0
                if x + w > iw: w = iw - x
                if y + h > ih: h = ih - y

                # Desenha o retângulo (mantendo verde como pediu)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # 2. Recorte do Rosto (ROI - Region of Interest)
                # Recortamos apenas a parte do rosto para enviar ao DeepFace
                face_roi = frame[y:y+h, x:x+w]

                # Se o recorte for válido (tem tamanho maior que 0)
                if face_roi.size > 0:
                    try:
                        # 3. DeepFace analisa APENAS o recorte
                        # actions=['emotion']: só queremos emoção
                        # enforce_detection=False: confiamos no recorte do MediaPipe
                        analysis = DeepFace.analyze(
                            img_path=face_roi, 
                            actions=['emotion'], 
                            enforce_detection=False, 
                            detector_backend='skip', # Importante: 'skip' pois já recortamos o rosto
                            silent=True
                        )
                        
                        # DeepFace retorna uma lista, pegamos o primeiro item
                        dominant_emotion = analysis[0]['dominant_emotion']
                        
                        # Escreve a emoção (Texto verde, sem tradução)
                        cv2.putText(frame, dominant_emotion, (x, y - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    except Exception as e:
                        # Se o DeepFace falhar em classificar (rosto muito borrado), ignoramos
                        pass

        cv2.imshow('Analise de Emocoes', frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
out.release()
cv2.destroyAllWindows()
print("Processamento finalizado.")