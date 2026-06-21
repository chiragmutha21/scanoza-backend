print("Importing os...")
import os
print("Importing traceback...")
import traceback
print("Importing tempfile...")
import tempfile
print("Importing contextlib...")
from contextlib import asynccontextmanager

print("Importing fastapi...")
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
print("Importing dotenv...")
from dotenv import load_dotenv

print("Importing database...")
from database import connect_db, disconnect_db, get_images_collection
print("Importing routes...")
from routes import router

print("All imports completed successfully!")
