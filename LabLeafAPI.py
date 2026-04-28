"""
API AgroVision IA (versão deploy - Render)
"""

import base64
import io
import os
import traceback
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from ultralytics import YOLO

# ─── Configuração ─────────────────────────────────────────────
MODEL_PATH = os.environ.get("MODEL_PATH", "best.pt")

# ─── App ──────────────────────────────────────────────────────
app = FastAPI(title="AgroVision IA", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("uploads", exist_ok=True)
os.makedirs("static", exist_ok=True)

# ─── Modelo (lazy load) ───────────────────────────────────────
yolo_model = None

def get_model():
    global yolo_model
    if yolo_model is None:
        print("🔄 Carregando modelo YOLO...")
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError("Modelo best.pt não encontrado")
        yolo_model = YOLO(MODEL_PATH)
    return yolo_model

# ─── Rotas ────────────────────────────────────────────────────
@app.get("/")
def home(request: Request):
    return {
        "status": "AgroVision IA rodando 🌱",
        "url": str(request.base_url)
    }

@app.get("/health")
def health():
    return {
        "modelo": "carregado" if yolo_model else "não carregado"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        model = get_model()

        # salvar imagem temporária
        temp_path = f"uploads/{uuid.uuid4()}.jpg"

        content = await file.read()
        img = Image.open(io.BytesIO(content)).convert("RGB")
        img.save(temp_path)

        # inferência
        results = model(temp_path)

        # pegar resultado
        probs = results[0].probs
        classe_id = int(probs.top1)
        confianca = float(probs.top1conf)

        nome = model.names[classe_id]

        # imagem resultado
        result_path = f"static/{uuid.uuid4()}.jpg"
        results[0].save(filename=result_path)

        with open(result_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")

        return {
            "classe": nome,
            "confianca": round(confianca * 100, 2),
            "imagem_resultado": f"data:image/jpeg;base64,{img_b64}"
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # limpar arquivos
        for path in ["uploads", "static"]:
            for file in os.listdir(path):
                try:
                    os.remove(os.path.join(path, file))
                except:
                    pass


# ─── Inicialização ────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)