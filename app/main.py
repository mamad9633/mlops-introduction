from fastapi import FastAPI
from contextlib import asynccontextmanager
import pickle
from pydantic import BaseModel
from fastapi import HTTPException
import time
import asyncio
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv
from fastapi import BackgroundTasks
load_dotenv()

# Create a FastAPI instance
app = FastAPI()

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/")
async def root():
    return {"message": "Hello World"}
    

ml_models = {} # Global dictionary to hold the models.
def load_model(path: str):
    model = None
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models when the app starts
    ml_models['logistic_model'] = load_model("models/logistic_regression.pkl")
    ml_models['rf_model'] = load_model("models/random_forest.pkl")

    yield
    # This code will be executed after the application finishes handling requests, right before the shutdown
    # Clean up the ML models and release the resources
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root():
    return {"message": "Models loaded and FastAPI is ready!"}

@app.get("/models")
async def list_models():
    # Return the list of available models' names
    return {"available_models": list(ml_models.keys())}





class IrisData(BaseModel):
    sepal_length: float = Field(..., gt=0, lt=10, description="Sepal length must be between 0 and 10", example=5.1)
    sepal_width: float = Field(..., gt=0, lt=10, description="Sepal width must be between 0 and 10", example=3.5)
    petal_length: float = Field(..., gt=0, lt=10, description="Petal length must be between 0 and 10", example=1.4)
    petal_width: float = Field(..., gt=0, lt=10, description="Petal width must be between 0 and 10", example=0.2)



CACHE = {}
"""
key ---> TIMESTAMP, prediction
rf_model:{speal_length:3,sepal_widht:4.1..} ---> 1
logistic_model:{.....} ---> 0
"""

TTL = 30  # seconds


def make_cache_key(model_name, iris: IrisData):
    return f"{model_name}:{iris.json()}"


@app.post("/predict_cached/{model_name}")
async def predict_cached(model_name: str, iris: IrisData):
    key = make_cache_key(model_name, iris)
    now = time.time()

    if key in CACHE and now - CACHE[key][0] < TTL:
        return {"model": model_name, "prediction": CACHE[key][1], "cache": "HIT"}

    X = [[iris.sepal_length, iris.sepal_width, iris.petal_length, iris.petal_width]]
    pred = int(ml_models[model_name].predict(X)[0])

    CACHE[key] = (now, pred)
    return {"model": model_name, "prediction": pred, "cache": "MISS"}





@app.post("/predict/{model_name}")
async def predict(model_name: str, iris: IrisData):
    await asyncio.sleep(5) # Mimic heavy workload.
     
    input_data = [[iris.sepal_length, iris.sepal_width, iris.petal_length, iris.petal_width]]
     
    if model_name not in ml_models.keys():
        raise HTTPException(status_code=404, detail="Model not found")
     
    ml_model = ml_models[model_name]
    prediction = ml_model.predict(input_data)

    return {"model": model_name, "prediction": int(prediction[0])}




def log_prediction(data: dict):
    print(f"Logging prediction: {data}")

@app.post("/predict_log/{model_name}")
async def predict_log(model_name: str, iris: IrisData, background_tasks: BackgroundTasks):
    X = [[iris.sepal_length, iris.sepal_width, iris.petal_length, iris.petal_width]]
    pred = int(ml_models[model_name].predict(X)[0])
    background_tasks.add_task(log_prediction, {"model": model_name, "features": iris.dict(), "pred": pred})
    return {"model": model_name, "prediction": pred}


from fastapi import Security, HTTPException, Depends
from fastapi.security import APIKeyHeader
import os

API_KEY_NAME = "X-API-Key"
API_KEY = os.getenv("XAPIKEY", "dev-secret")
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def require_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key

@app.post("/predict_secure/{model_name}")
async def predict_secure(model_name: str, iris: IrisData, _: str = Depends(require_api_key)):
    ...
    X = [[iris.sepal_length, iris.sepal_width, iris.petal_length, iris.petal_width]]
    pred = int(ml_models[model_name].predict(X)[0])
    return {"model": model_name, "prediction": pred}

from fastapi import Request
from starlette.responses import JSONResponse
"""
@app.middleware("http")
async def check_auth(request: Request, call_next):
    if request.headers.get("X-API-Key") != os.getenv("XAPIKEY", "dev-secret"):
        return JSONResponse({"detail": "Unauthorized"}, status_code=401)
    return await call_next(request)

from fastapi import Request
from starlette.responses import JSONResponse

WHITELIST = {"/docs", "/openapi.json", "/health"}

@app.middleware("http")
async def check_auth(request: Request, call_next):
    if request.url.path in WHITELIST:
        return await call_next(request)
    if request.headers.get("X-API-Key") != os.getenv("XAPIKEY", "dev-secret"):
        return JSONResponse({"detail": "Unauthorized"}, status_code=401)
    return await call_next(request)


import time, uuid, json

@app.middleware("http")
async def log_timing(request: Request, call_next):
    start = time.perf_counter()
    req_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    response = await call_next(request)
    duration = time.perf_counter() - start
    response.headers["X-Process-Time"] = f"{duration:.3f}s"
    response.headers["X-Request-ID"] = req_id
    print(json.dumps({
        "id": req_id,
        "method": request.method,
        "path": request.url.path,
        "status": response.status_code,
        "duration": round(duration, 3)
    }))
    return response


import time

CACHE = {}
TTL = 30  # seconds

def make_cache_key(model_name, iris: IrisData):
    return f"{model_name}:{iris.json()}"

@app.post("/predict_cached/{model_name}")
async def predict_cached(model_name: str, iris: IrisData):
    key = make_cache_key(model_name, iris)
    now = time.time()

    if key in CACHE and now - CACHE[key][0] < TTL:
        return {"model": model_name, "prediction": CACHE[key][1], "cache": "HIT"}

    X = [[iris.sepal_length, iris.sepal_width, iris.petal_length, iris.petal_width]]
    pred = int(ml_models[model_name].predict(X)[0])
    CACHE[key] = (now, pred)
    return {"model": model_name, "prediction": pred, "cache": "MISS"}

"""