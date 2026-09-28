import os
import sys
import modal

model_volume = modal.Volume.from_name("saarthi-model-cache", create_if_missing=True)

saarthi_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements.txt")
    .pip_install("fastapi", "uvicorn[standard]", "websockets", "httpx", "peft", "accelerate", "torchvision", "qwen-vl-utils")
    .env({
        "HF_HOME": "/cache/hf",
        "SAARTHI_LOCAL_DATA_DIR": "/cache/local_data",
        "PRELOAD_MODEL_ON_STARTUP": "true",
        "WEB_SEARCH_ENABLED": "false",
        "WEB_LOCAL_SHORTCUTS": "true",
        "WEB_STREAM_TIMEOUT_SECONDS": "300",
        "WEB_MAX_TOKENS": "250",
    })
    .add_local_dir(
        "/kaggle/working/SAARTHI_SERVER",
        remote_path="/root/SAARTHI_SERVER",
        ignore=[".git", "__pycache__", "*.log", "*.zip", "local_data", "saarthi_v2_perfect/checkpoint-*"],
    )
)

app = modal.App("saarthi-server", image=saarthi_image)

@app.function(
    cpu=8.0,
    memory=16384,
    volumes={"/cache": model_volume},
    scaledown_window=180,
    timeout=600,
)
@modal.concurrent(max_inputs=10)
@modal.asgi_app()
def web_app():
    sys.path.insert(0, "/root/SAARTHI_SERVER")
    os.chdir("/root/SAARTHI_SERVER")
    from server import app as fastapi_app
    from fastapi.middleware.cors import CORSMiddleware

    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return fastapi_app
