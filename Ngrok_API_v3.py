"""
Ngrok_API_v3.py
---------------
API unificada – Soja (10 classes) + PlantVillage (38 classes) = 48 classes total.
Compatível com o site do seu amigo (mesma estrutura de resposta JSON).

Uso:
  python Ngrok_API_v3.py
"""

import base64
import io
import os
import traceback
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from ultralytics import YOLO
from pyngrok import ngrok

# ─── Configuração ─────────────────────────────────────────────────────────────
MODEL_PATH = os.environ.get("MODEL_PATH", "best.pt")

# ─── Mapa completo: nome interno YOLO → português ─────────────────────────────
MAPA_CLASSES = {
    # ── 10 classes de SOJA ────────────────────────────────────────────────────
    "Soja___Queima_Bacteriana": "Soja – Queima Bacteriana",
    "Soja___Mancha_Parda":      "Soja – Mancha Parda",
    "Soja___Crestamento":       "Soja – Crestamento",
    "Soja___Ferrugem":          "Soja – Ferrugem Asiática",
    "Soja___Virus_Mosaico":     "Soja – Vírus do Mosaico",
    "Soja___Oidio":             "Soja – Oídio",
    "Soja___Septoriose":        "Soja – Septoriose",
    "Soja___Podridao_Sul":      "Soja – Podridão do Sul",
    "Soja___Morte_Subita":      "Soja – Síndrome da Morte Súbita",
    "Soja___Mosaico_Amarelo":   "Soja – Mosaico Amarelo",

    # Compatibilidade com nomes antigos do modelo de soja
    "bacterial_blight":        "Soja – Queima Bacteriana",
    "brown_spot":              "Soja – Mancha Parda",
    "crestamento":             "Soja – Crestamento",
    "ferrugem":                "Soja – Ferrugem Asiática",
    "Mosaic Virus":            "Soja – Vírus do Mosaico",
    "powdery_mildew":          "Soja – Oídio",
    "septoria":                "Soja – Septoriose",
    "Southern blight":         "Soja – Podridão do Sul",
    "Sudden Death Syndrome":   "Soja – Síndrome da Morte Súbita",
    "Yellow Mosaic":           "Soja – Mosaico Amarelo",

    # ── 38 classes PlantVillage ───────────────────────────────────────────────
    "Apple___Apple_Scab":               "Maçã – Sarna",
    "Apple___Black_Rot":                "Maçã – Podridão Negra",
    "Apple___Cedar_Apple_Rust":         "Maçã – Ferrugem Cedar",
    "Apple___Healthy":                  "Maçã – Saudável",
    "Blueberry___Healthy":              "Mirtilo – Saudável",
    "Cherry___Powdery_Mildew":          "Cereja – Oídio",
    "Cherry___Healthy":                 "Cereja – Saudável",
    "Corn___Cercospora_Gray_Leaf_Spot": "Milho – Mancha Cinza (Cercospora)",
    "Corn___Common_Rust":               "Milho – Ferrugem Comum",
    "Corn___Northern_Leaf_Blight":      "Milho – Requeima (Northern Leaf Blight)",
    "Corn___Healthy":                   "Milho – Saudável",
    "Grape___Black_Rot":                "Uva – Podridão Negra",
    "Grape___Esca_Black_Measles":       "Uva – Esca (Sarampo Negro)",
    "Grape___Leaf_Blight":              "Uva – Queima das Folhas",
    "Grape___Healthy":                  "Uva – Saudável",
    "Orange___Huanglongbing":           "Laranja – Huanglongbing (Greening)",
    "Peach___Bacterial_Spot":           "Pêssego – Mancha Bacteriana",
    "Peach___Healthy":                  "Pêssego – Saudável",
    "Pepper_Bell___Bacterial_Spot":     "Pimentão – Mancha Bacteriana",
    "Pepper_Bell___Healthy":            "Pimentão – Saudável",
    "Potato___Early_Blight":            "Batata – Pinta Preta",
    "Potato___Late_Blight":             "Batata – Requeima",
    "Potato___Healthy":                 "Batata – Saudável",
    "Raspberry___Healthy":              "Framboesa – Saudável",
    "Soybean___Healthy":                "Soja (PlantVillage) – Saudável",
    "Squash___Powdery_Mildew":          "Abobrinha – Oídio",
    "Strawberry___Leaf_Scorch":         "Morango – Queima das Folhas",
    "Strawberry___Healthy":             "Morango – Saudável",
    "Tomato___Bacterial_Spot":          "Tomate – Mancha Bacteriana",
    "Tomato___Early_Blight":            "Tomate – Pinta Preta",
    "Tomato___Late_Blight":             "Tomate – Requeima",
    "Tomato___Leaf_Mold":               "Tomate – Mofo das Folhas",
    "Tomato___Septoria_Leaf_Spot":      "Tomate – Septoriose",
    "Tomato___Spider_Mites":            "Tomate – Ácaros (Dois Pontos)",
    "Tomato___Target_Spot":             "Tomate – Mancha Alvo",
    "Tomato___Yellow_Leaf_Curl_Virus":  "Tomate – Vírus do Enrolamento Amarelo",
    "Tomato___Mosaic_Virus":            "Tomate – Vírus do Mosaico",
    "Tomato___Healthy":                 "Tomate – Saudável",
}

# ─── Recomendações por palavra-chave ──────────────────────────────────────────
RECOMENDACOES = {
    "Queima Bacteriana":       "Aplique bactericida cúprico. Evite irrigação por aspersão e alta umidade.",
    "Mancha Parda":            "Use fungicidas à base de trifloxistrobina. Faça rotação de culturas.",
    "Crestamento":             "Remova e destrua partes afetadas. Evite ferimentos nas plantas.",
    "Ferrugem":                "Aplique fungicidas específicos (tebuconazol ou azoxistrobina) preventivamente.",
    "Vírus do Mosaico":        "Controle pulgões e outros insetos vetores. Remova plantas infectadas.",
    "Mosaico":                 "Controle insetos vetores. Use sementes certificadas e livres de vírus.",
    "Oídio":                   "Aplique enxofre molhável ou fungicidas sistêmicos. Melhore ventilação.",
    "Septoriose":              "Use fungicidas e evite molhar as folhas. Faça rotação de culturas.",
    "Podridão do Sul":         "Melhore a drenagem do solo. Aplique fungicidas no solo preventivamente.",
    "Morte Súbita":            "Rotação de culturas. Evite compactação e encharcamento do solo.",
    "Mosaico Amarelo":         "Controle mosca-branca. Use variedades resistentes.",
    "Sarna":                   "Aplique fungicidas cúpricos preventivamente. Remova folhas infectadas.",
    "Podridão Negra":          "Use fungicidas à base de captan. Evite ferimentos nas frutas.",
    "Greening":                "Sem cura disponível. Remova plantas infectadas e controle o psilídeo.",
    "Mancha Bacteriana":       "Use bactericidas cúpricos. Evite irrigação por aspersão.",
    "Requeima":                "Aplique fungicidas protetores (mancozeb). Evite excesso de umidade.",
    "Pinta Preta":             "Rotação de culturas e fungicidas à base de cobre.",
    "Mofo":                    "Melhore a ventilação. Aplique fungicidas preventivos.",
    "Ácaros":                  "Use acaricidas específicos. Evite excesso de nitrogênio.",
    "Mancha Alvo":             "Use fungicidas protetores. Remova restos de cultura.",
    "Enrolamento Amarelo":     "Controle mosca-branca. Use variedades resistentes e tolerantes.",
    "Queima das Folhas":       "Aplique fungicidas. Melhore a drenagem do solo.",
    "Mancha Cinza":            "Aplique fungicidas específicos. Faça rotação de culturas.",
    "Esca":                    "Poda e destruição de ramos infectados. Não há controle efetivo.",
    "Saudável":                "Planta saudável! Continue com as boas práticas de manejo.",
}

def get_recomendacao(nome_pt: str) -> str:
    for key, rec in RECOMENDACOES.items():
        if key.lower() in nome_pt.lower():
            return rec
    return "Consulte um agrônomo para recomendações específicas da sua região."


# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="AgroVision IA – Unificado", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("uploads", exist_ok=True)
os.makedirs("static",  exist_ok=True)

# ─── Carrega modelo ───────────────────────────────────────────────────────────
yolo_model = None
try:
    if os.path.exists(MODEL_PATH):
        yolo_model = YOLO(MODEL_PATH)
        n_classes = len(yolo_model.names)
        print(f"✔ Modelo carregado: {MODEL_PATH} | {n_classes} classes")
    else:
        print(f"⚠  Modelo não encontrado: {MODEL_PATH}")
        print(    "   Rode trainmaster6.py e copie o best.pt para cá.")
except Exception as e:
    print(f"✘ Erro ao carregar modelo: {e}")

# ─── Ngrok ────────────────────────────────────────────────────────────────────
try:
    ngrok.kill()
    public_url = ngrok.connect(8000)
    BASE_URL = str(public_url)
    print(f"🔥 URL pública: {BASE_URL}")
except Exception as e:
    BASE_URL = "http://localhost:8000"
    print(f"⚠  Ngrok não iniciado: {e}")


# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/")
def home():
    classes_carregadas = len(yolo_model.names) if yolo_model else 0
    return {
        "status":  "AgroVision IA rodando 🌱",
        "versao":  "3.0 – Soja + PlantVillage",
        "classes": classes_carregadas,
        "url":     BASE_URL,
    }

@app.get("/url")
def get_url():
    return {"url": BASE_URL}

@app.get("/health")
def health():
    return {
        "modelo":  "carregado" if yolo_model is not None else "não carregado",
        "classes": len(yolo_model.names) if yolo_model else 0,
        "url":     BASE_URL,
    }

@app.get("/classes")
def listar_classes():
    """Retorna todas as classes disponíveis no modelo."""
    if yolo_model is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado.")
    classes = {
        idx: {
            "interno": name,
            "portugues": MAPA_CLASSES.get(name, name),
        }
        for idx, name in yolo_model.names.items()
    }
    return {"total": len(classes), "classes": classes}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Recebe imagem → retorna classificação.
    Resposta 100% compatível com o site do seu amigo.
    Nunca crasha: todo erro vira JSON com campo 'detail'.
    """
    # ── Valida modelo ────────────────────────────────────────────────────────
    if yolo_model is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo não carregado. Verifique se best.pt existe e reinicie a API."
        )

    # ── Valida extensão ──────────────────────────────────────────────────────
    allowed_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ext = os.path.splitext((file.filename or "").lower())[1]
    if ext not in allowed_ext:
        raise HTTPException(
            status_code=415,
            detail=f"Formato '{ext}' não suportado. Use: {', '.join(allowed_ext)}"
        )

    temp_path   = os.path.join("uploads", f"{uuid.uuid4()}.jpg")
    result_path = os.path.join("static",  f"{uuid.uuid4()}.jpg")

    try:
        # ── Salva e converte para RGB ────────────────────────────────────────
        content = await file.read()
        img = Image.open(io.BytesIO(content)).convert("RGB")
        img.save(temp_path, "JPEG", quality=95)

        # ── Inferência ───────────────────────────────────────────────────────
        results = yolo_model(temp_path)
        results[0].save(filename=result_path)

        # ── Extrai resultados ────────────────────────────────────────────────
        probs     = results[0].probs
        classe_id = int(probs.top1)
        confianca = float(probs.top1conf)

        nome_interno = yolo_model.names[classe_id]
        nome_pt      = MAPA_CLASSES.get(nome_interno, nome_interno)
        recomendacao = get_recomendacao(nome_pt)

        # ── Imagem como base64 (nunca quebra por URL) ────────────────────────
        with open(result_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")

        # ── Top-5 ────────────────────────────────────────────────────────────
        top5_idx  = probs.top5
        top5_conf = probs.top5conf.tolist()
        top5 = [
            {
                "classe":    MAPA_CLASSES.get(yolo_model.names[i], yolo_model.names[i]),
                "confianca": round(float(c) * 100, 2),
            }
            for i, c in zip(top5_idx, top5_conf)
        ]

        return {
            # Campos originais (compatibilidade com site do amigo)
            "classe":           nome_pt,
            "confianca":        round(confianca * 100, 2),
            "recomendacao":     recomendacao,
            # Imagem em base64 – sem depender de URL ngrok
            "imagem_resultado": f"data:image/jpeg;base64,{img_b64}",
            # Extras
            "top5":             top5,
            "classe_interna":   nome_interno,
        }

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erro na predição: {str(e)}")

    finally:
        for path in [temp_path, result_path]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass


# ─── Inicialização ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
