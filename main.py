from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware
import app  # This imports your app.py

fastapi_app = FastAPI()
fastapi_app.mount("/", WSGIMiddleware(app.server))  # uses app.server
