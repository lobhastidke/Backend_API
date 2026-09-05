from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Configure CORS
origins = [
    "http://localhost:5173",       # Allow local Vite dev server
    "https://testprojectfrontend.vercel.app/" # ALLOW YOUR VERCEL DOMAIN HERE (update this after you deploy to Vercel)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
