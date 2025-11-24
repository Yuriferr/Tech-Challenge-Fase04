import cv2
import numpy as np
from deepface import DeepFace
from ultralytics import YOLO
from tqdm import tqdm
import os

# --- CONFIGURAÇÕES INICIAIS ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Dicionário de Tradução (Adicione mais se quiser)
TRADUCAO_EMOCOES = {
    'angry': 'RAIVA',
    'disgust': 'NOJO',
    'fear': 'MEDO',
    'happy': 'FELIZ',
    'sad': 'TRISTE',
    'surprise': 'SURPRESA',
    'neutral': 'NEUTRO'
}

# Cores para cada emoção (BGR)
CORES_EMOCOES = {
    'RAIVA': (0, 0, 255),      # Vermelho
    'FELIZ': (0, 255, 255),    # Amarelo
    'TRISTE': (255, 0, 0),     # Azul
    'NEUTRO': (200, 200, 200), # Cinza
    'SURPRESA': (0, 165, 255)  # Laranja
}

# Carregar Modelo YOLO Pose
print("Carregando modelo YOLOv8-Pose...")
model_pose = YOLO('yolov8n-pose.pt') 

def inferir_acao_corpo(keypoints):
    """
    Lógica focada EXCLUSIVAMENTE no corpo (Ombros, Cotovelos, Punhos).
    Ignora cabeça (nariz, olhos, orelhas).
    
    Indices COCO Keypoints usados:
    5: Ombro Esq, 6: Ombro Dir
    7: Cotovelo Esq, 8: Cotovelo Dir
    9: Punho Esq, 10: Punho Dir
    """
    if keypoints is None or len(keypoints) == 0:
        return "", (0,0,0)

    # Pega a primeira pessoa (índice 0)
    kp = keypoints.cpu().numpy()[0]

    # Extrair Y (Altura) - Lembre-se: Y menor = Mais alto na tela
    y_ombro_esq = kp[5][1]
    y_ombro_dir = kp[6][1]
    y_punho_esq = kp[9][1]
    y_punho_dir = kp[10][1]
    
    # Extrair X (Posição horizontal)
    x_punho_esq = kp[9][0]
    x_punho_dir = kp[10][0]

    # Confianças (para garantir que o braço existe na imagem)
    conf_ombro = kp[5][2]
    conf_punho = kp[9][2]

    if conf_ombro < 0.5: return "...", (100,100,100)

    # --- LÓGICA 1: BRAÇOS LEVANTADOS (Acima dos Ombros) ---
    # Não depende mais do nariz, e sim da linha do ombro
    media_altura_ombros = (y_ombro_esq + y_ombro_dir) / 2
    
    braco_esq_levantado = (y_punho_esq < media_altura_ombros) and (conf_punho > 0.5)
    braco_dir_levantado = (y_punho_dir < media_altura_ombros) and (kp[10][2] > 0.5)

    if braco_esq_levantado and braco_dir_levantado:
        return "BRACOS AO AR / COMEMORANDO", (0, 255, 0) # Verde
    elif braco_esq_levantado or braco_dir_levantado:
        return "ACENANDO / MAO ALTA", (0, 165, 255) # Laranja

    # --- LÓGICA 2: BRAÇOS CRUZADOS / FECHADOS ---
    # Se a distância horizontal entre os punhos for pequena
    distancia_punhos = abs(x_punho_esq - x_punho_dir)
    if distancia_punhos < 50 and y_punho_esq > media_altura_ombros: # 50 pixels de tolerância
        return "BRACOS FECHADOS / CRUZADOS", (255, 0, 255) # Magenta

    return "POSTURA RELAXADA", (200, 200, 200)

# --- PIPELINE PRINCIPAL ---
input_video = 'Unlocking Facial Recognition_ Diverse Activities Analysis.mp4'
output_video = 'video_final_traduzido_corpo.mp4'

cap = cv2.VideoCapture(input_video)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

print(f"Iniciando processamento de {total_frames} frames...")

for _ in tqdm(range(total_frames), desc="Processando"):
    ret, frame = cap.read()
    if not ret:
        break

    # 1. YOLO POSE (Corpo)
    results_pose = model_pose(frame, verbose=False)
    
    for r in results_pose:
        # Plotar esqueleto
        frame = r.plot(boxes=False) 
        
        # Inferir Ação Corporal
        if r.keypoints is not None and len(r.keypoints) > 0:
            texto_acao, cor_acao = inferir_acao_corpo(r.keypoints.data)
            
            # Desenhar barra de status da ação no topo
            if texto_acao != "...":
                cv2.rectangle(frame, (0, 0), (400, 45), (30, 30, 30), -1)
                cv2.putText(frame, f"CORPO: {texto_acao}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor_acao, 2)

    # 2. DEEPFACE (Emoção Traduzida)
    try:
        face_analysis = DeepFace.analyze(img_path=frame, 
                                       actions=['emotion'], 
                                       detector_backend='yolov8', 
                                       enforce_detection=False, 
                                       align=False,
                                       silent=True)
        
        for face in face_analysis:
            region = face['region']
            emo_en = face['dominant_emotion']
            
            # TRADUÇÃO AQUI
            emo_pt = TRADUCAO_EMOCOES.get(emo_en, "NEUTRO")
            cor_emo = CORES_EMOCOES.get(emo_pt, (255, 255, 255))
            
            x, y, w, h = region['x'], region['y'], region['w'], region['h']
            
            # Desenhar caixa da face
            cv2.rectangle(frame, (x, y), (x+w, y+h), cor_emo, 2)
            
            # Fundo do texto para leitura
            cv2.rectangle(frame, (x, y-30), (x+w, y), cor_emo, cv2.FILLED)
            cv2.putText(frame, emo_pt, (x + 5, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    except Exception:
        pass

    out.write(frame)
    
    # Preview (Redimensionado para performance visual)
    preview = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
    cv2.imshow('IA Multimodal - PT-BR', preview)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("Processamento concluído com sucesso!")