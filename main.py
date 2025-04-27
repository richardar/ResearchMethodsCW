from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware
import app  

fastapi_app = FastAPI()
fastapi_app.mount("/", WSGIMiddleware(app.server))  
