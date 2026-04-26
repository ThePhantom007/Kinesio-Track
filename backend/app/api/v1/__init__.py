"""
Assembles all v1 routers into a single APIRouter mounted at /api/v1.
"""

from fastapi import APIRouter

from app.api.v1.auth       import router as auth_router
from app.api.v1.clinicians import router as clinicians_router
from app.api.v1.intake     import router as intake_router
from app.api.v1.media      import router as media_router
from app.api.v1.patients   import router as patients_router
from app.api.v1.plans      import router as plans_router
from app.api.v1.progress   import router as progress_router
from app.api.v1.sessions   import router as sessions_router

v1_router = APIRouter(prefix="/api/v1")

v1_router.include_router(auth_router)
v1_router.include_router(intake_router)
v1_router.include_router(plans_router)
v1_router.include_router(sessions_router)
v1_router.include_router(patients_router)
v1_router.include_router(clinicians_router)
v1_router.include_router(media_router)
v1_router.include_router(progress_router)