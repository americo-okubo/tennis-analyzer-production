#!/usr/bin/env python3
"""
Tennis Analyzer Web API - MVP
FastAPI application for table tennis technique analysis
"""

import os
import sys
import uvicorn
import hashlib
import secrets
import tempfile
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, status, Form
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr
import jwt

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import our analysis systems
from tennis_comparison_backend import TableTennisAnalyzer, TennisAnalyzerAPI
from real_time_analysis_engine import RealTimeAnalysisEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "FIXED_SECRET_KEY_FOR_DEBUG_12345678901234567890")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

# Global analyzer instances
analyzer_api = None
real_time_engine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global analyzer_api, real_time_engine
    
    logger.info("Starting Tennis Analyzer API...")
    
    # Initialize analyzers
    try:
        analyzer_api = TennisAnalyzerAPI()
        real_time_engine = RealTimeAnalysisEngine()
        logger.info("Analysis engines initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize analysis engines: {e}")
        raise
    
    yield
    
    logger.info("Shutting down Tennis Analyzer API...")


# FastAPI app
app = FastAPI(
    title="Tennis Analyzer API",
    description="Table Tennis technique analysis with real-time capabilities",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Pydantic models
class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int

class AnalysisMetadata(BaseModel):
    maoDominante: str  # D or E
    ladoCamera: str    # D or E  
    ladoRaquete: str   # F or B
    tipoMovimento: str # D or P

class ComparisonRequest(BaseModel):
    user_metadata: AnalysisMetadata
    professional_name: Optional[str] = None

class AnalysisResult(BaseModel):
    success: bool
    analysis_id: str
    timestamp: datetime
    final_score: Optional[float] = None
    analysis_type: str
    detailed_analysis: Optional[Dict] = None
    recommendations: Optional[List[str]] = None
    professional_comparisons: Optional[List[Dict]] = None
    movement_statistics: Optional[Dict] = None
    error: Optional[str] = None


# Simple in-memory user storage (replace with database in production)
users_db = {
    "demo": {
        "username": "demo",
        "email": "demo@example.com",
        "password_hash": hashlib.sha256("demo123".encode()).hexdigest(),
        "created_at": datetime.now()
    }
}

# Authentication functions
def verify_password(plain_password: str, password_hash: str) -> bool:
    """Verify password against hash"""
    return hashlib.sha256(plain_password.encode()).hexdigest() == password_hash

def get_user(username: str) -> Optional[Dict]:
    """Get user from database"""
    return users_db.get(username)

def authenticate_user(username: str, password: str) -> Optional[Dict]:
    """Authenticate user credentials"""
    user = get_user(username)
    if not user or not verify_password(password, user["password_hash"]):
        return None
    return user

def create_access_token(data: Dict) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict:
    """Verify JWT token"""
    try:
        logger.info(f"[AUTH_DEBUG] Verifying token: {credentials.credentials[:50]}...")
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        logger.info(f"[AUTH_DEBUG] Token decoded successfully: {payload}")
        username: str = payload.get("sub")
        if username is None:
            logger.warning(f"[AUTH_DEBUG] No username in token payload")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials"
            )
        user = get_user(username)
        if user is None:
            logger.warning(f"[AUTH_DEBUG] User {username} not found")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )
        logger.info(f"[AUTH_DEBUG] User {username} authenticated successfully")
        return user
    except jwt.PyJWTError as e:
        logger.error(f"[AUTH_DEBUG] JWT decode error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )

def validate_video_file(file: UploadFile) -> None:
    """Validate uploaded video file"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"File type {file_ext} not allowed. Allowed types: {ALLOWED_VIDEO_EXTENSIONS}"
        )

async def save_uploaded_file(file: UploadFile) -> Path:
    """Save uploaded file to temporary location"""
    validate_video_file(file)
    
    # Create temporary file
    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)
    
    file_ext = Path(file.filename).suffix
    temp_file = temp_dir / f"{secrets.token_hex(16)}{file_ext}"
    
    # Save file
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large")
    
    with open(temp_file, "wb") as f:
        f.write(content)
    
    return temp_file


# API Routes

@app.get("/")
async def root():
    """API health check"""
    return {
        "message": "Tennis Analyzer API",
        "version": "1.0.0",
        "status": "active",
        "features": ["video_analysis", "real_time", "comparison", "authentication"]
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "components": {
            "analyzer_api": analyzer_api is not None,
            "real_time_engine": real_time_engine is not None,
            "temp_directory": Path("temp_uploads").exists()
        }
    }

# Authentication endpoints
@app.post("/auth/register", response_model=Dict)
async def register(user: UserCreate):
    """Register new user"""
    if user.username in users_db:
        raise HTTPException(status_code=400, detail="Username already exists")
    
    password_hash = hashlib.sha256(user.password.encode()).hexdigest()
    users_db[user.username] = {
        "username": user.username,
        "email": user.email,
        "password_hash": password_hash,
        "created_at": datetime.now()
    }
    
    return {"message": "User created successfully", "username": user.username}

@app.post("/auth/login", response_model=Token)
async def login(user: UserLogin):
    """User login"""
    authenticated_user = authenticate_user(user.username, user.password)
    if not authenticated_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password"
        )
    
    access_token = create_access_token(data={"sub": user.username})
    return Token(
        access_token=access_token,
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )

@app.get("/auth/me")
async def get_current_user(current_user: Dict = Depends(verify_token)):
    """Get current user info"""
    return {
        "username": current_user["username"],
        "email": current_user["email"],
        "created_at": current_user["created_at"]
    }

# Analysis endpoints

# TEMPORARY PUBLIC ENDPOINT FOR TESTING
@app.get("/professionals-public")
async def get_professionals_public(
    movement_type: str = "forehand_drive"
):
    """Get available professional players (PUBLIC - NO AUTH REQUIRED)"""
    try:
        professionals = analyzer_api.engine.get_available_professionals(movement_type)
        return {
            "success": True,
            "movement_type": movement_type,
            "professionals": professionals,
            "count": len(professionals)
        }
    except Exception as e:
        logger.error(f"Error getting professionals: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/professionals")
async def get_professionals(
    movement_type: str = "forehand_drive"
):
    """Get available professional players (without validation)"""
    try:
        professionals = analyzer_api.engine.get_available_professionals(movement_type)
        return {
            "success": True,
            "movement_type": movement_type,
            "professionals": professionals,
            "count": len(professionals)
        }
    except Exception as e:
        logger.error(f"Error getting professionals: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/validate-and-get-professionals")
async def validate_and_get_professionals(
    file: UploadFile = File(...),
    metadata: str = Form(...),
    current_user: Dict = Depends(verify_token)
):
    """Validate video and get available professional players"""
    try:
        # Parse metadata
        import json
        metadata_dict = json.loads(metadata)
        logger.info(f"[VALIDATION_DEBUG] Metadata recebido: {metadata_dict}")
        
        # Create metadata object manually for now
        class AnalysisMetadataTemp:
            def __init__(self, data):
                self.maoDominante = data.get('maoDominante', 'Destro')
                self.ladoCamera = data.get('ladoCamera', 'Direita')
                self.ladoRaquete = data.get('ladoRaquete', 'F')
                self.tipoMovimento = data.get('tipoMovimento', 'D')
                self._data = data
            
            def dict(self):
                return self._data
        
        analysis_metadata = AnalysisMetadataTemp(metadata_dict)
        
        # Save uploaded file temporarily for validation
        temp_file = await save_uploaded_file(file)
        
        logger.info(f"[VALIDATION_BEFORE_PROS] Validating video before listing professionals...")
        
        # CRITICAL VALIDATION: Use REAL video content analysis
        user_config_movement = f"{'forehand' if analysis_metadata.ladoRaquete == 'F' else 'backhand'}_{'drive' if analysis_metadata.tipoMovimento == 'D' else 'push'}"
        
        # ALWAYS use biomechanical classifier for REAL content validation
        try:
            logger.info(f"[VALIDATION_BEFORE_PROS] Analyzing video content with classifier...")
            validation_result = analyzer_api.engine.validate_user_video(str(temp_file), analysis_metadata.dict())
            
            if validation_result.get('success') and 'detected_info' in validation_result:
                detected_info = validation_result['detected_info']
                validation_passed = validation_result.get('validation_passed', False)
                
                logger.info(f"[VALIDATION_BEFORE_PROS] Validation result: {validation_result.get('message', 'No message')}")
                
                # Check if validation actually passed
                if not validation_passed:
                    # Cleanup temp file before error
                    try:
                        temp_file.unlink()
                    except:
                        pass
                    
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Video validation failed: {validation_result.get('message', 'Configuration does not match video content')}"
                    )
                    
                logger.info(f"[VALIDATION_BEFORE_PROS] Content validation PASSED! Loading professionals...")
            else:
                error_msg = validation_result.get('error', 'Could not validate video content')
                logger.warning(f"[VALIDATION_BEFORE_PROS] Validation failed: {error_msg}")
                # Cleanup temp file before error
                try:
                    temp_file.unlink()
                except:
                    pass
                raise HTTPException(
                    status_code=400, 
                    detail=f"Video analysis failed: {error_msg}"
                )
                
        except HTTPException:
            raise  # Re-raise HTTP exceptions
        except Exception as e:
            logger.error(f"[VALIDATION_BEFORE_PROS] Classifier validation error: {e}")
            # Cleanup temp file before error
            try:
                temp_file.unlink()
            except:
                pass
            raise HTTPException(
                status_code=500, 
                detail=f"Video analysis system error: {str(e)}"
            )
        
        # If validation passed, get professionals
        movement_key = analyzer_api.engine._build_movement_key(analysis_metadata.dict())
        professionals = analyzer_api.engine.get_available_professionals(movement_key)
        
        # Cleanup temp file
        try:
            temp_file.unlink()
        except:
            pass
        
        return {
            "success": True,
            "movement_type": movement_key,
            "professionals": professionals,
            "count": len(professionals),
            "validation_passed": True,
            "message": "Video validated successfully"
        }
    except Exception as e:
        logger.error(f"Error getting professionals: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    metadata: str = Form(...),
    current_user: Dict = Depends(verify_token)
):
    """Upload and validate user video"""
    try:
        # Parse metadata
        import json
        metadata_dict = json.loads(metadata)
        analysis_metadata = AnalysisMetadata(**metadata_dict)
        
        # Save uploaded file
        temp_file = await save_uploaded_file(file)
        
        # Validate video
        result = analyzer_api.process_upload(
            temp_file.read_bytes(),
            file.filename,
            analysis_metadata.dict()
        )
        
        # Generate analysis ID
        analysis_id = secrets.token_hex(16)
        
        return {
            "success": True,
            "analysis_id": analysis_id,
            "filename": file.filename,
            "file_path": str(temp_file),
            "validation": result,
            "metadata": analysis_metadata.dict()
        }
        
    except Exception as e:
        logger.error(f"Error uploading video: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze", response_model=AnalysisResult)
async def analyze_technique(
    user_video: UploadFile = File(...),
    metadata: str = Form(...),
    professional_name: Optional[str] = Form(None),
    cycle_index: int = Form(1)
):
    """Enhanced biomechanical analysis - NO AUTH"""
    try:
        logger.info(f"[BIOMECH_API] Enhanced biomechanical analysis (cycle {cycle_index})")
        from enhanced_single_cycle_analysis import EnhancedSingleCycleAnalyzer
        enhanced_analyzer = EnhancedSingleCycleAnalyzer()
        
        # Parse metadata
        import json
        metadata_dict = json.loads(metadata)
        analysis_metadata = AnalysisMetadata(**metadata_dict)
        
        # Save user video
        user_video_path = await save_uploaded_file(user_video)
        
        # Get professional video (optional for independent analysis)
        user_config_movement = f"{'forehand' if analysis_metadata.ladoRaquete == 'F' else 'backhand'}_{'drive' if analysis_metadata.tipoMovimento == 'D' else 'push'}"
        
        pro_video_path = None
        if professional_name:
            try:
                professionals = analyzer_api.engine.get_available_professionals(user_config_movement)
                selected_pro = next((p for p in professionals if p["name"] == professional_name), None)
                if selected_pro:
                    pro_video_path = selected_pro.get("file_path")
                    logger.info(f"[BIOMECH_API] Using professional: {professional_name}")
                else:
                    logger.warning(f"[BIOMECH_API] Professional {professional_name} not found, continuing with independent analysis")
            except Exception as e:
                logger.warning(f"[BIOMECH_API] Error finding professional: {e}, continuing with independent analysis")
        else:
            logger.info(f"[BIOMECH_API] Independent biomechanical analysis")
        
        # Get professional metadata if we have a professional video
        pro_metadata = {}
        if pro_video_path:
            try:
                pro_metadata = analyzer_api.get_professional_metadata(pro_video_path, user_config_movement)
            except Exception as e:
                logger.warning(f"[BIOMECH_API] Could not get professional metadata: {e}")
                pro_metadata = {}
        
        # Run enhanced analysis (comparative or independent)
        result = enhanced_analyzer.compare_enhanced_single_cycles(
            str(user_video_path),
            str(pro_video_path) if pro_video_path else None,
            analysis_metadata.dict(),
            pro_metadata,
            cycle_index=cycle_index
        )
        
        if result['success']:
            logger.info(f"[BIOMECH_API] Enhanced analysis completed: {result.get('final_score', 0):.1f}%")
            
            # Add optimized professional comparisons
            try:
                from optimized_professional_comparison import OptimizedProfessionalComparator
                comparator = OptimizedProfessionalComparator()
                
                # Determine detected movement
                detected_movement = result.get('detailed_analysis', {}).get('movement_classification', {}).get('detected_movement', user_config_movement)
                
                # Find best matches
                best_matches = comparator.find_best_matches(result, detected_movement, max_results=3)
                
                # Add comparisons to result
                if best_matches:
                    logger.info(f"[PROFESSIONAL_COMPARISON] Found {len(best_matches)} professional matches")
                    result['professional_comparisons'] = []
                    
                    for match in best_matches:
                        comparison_data = {
                            'professional_name': match.professional_name,
                            'professional_video': match.professional_video,
                            'similarity_score': match.similarity_score,
                            'similarity_percentage': f"{match.similarity_score * 100:.1f}%",
                            'detailed_comparison': match.detailed_comparison,
                            'recommendations': match.recommendations,
                            'comparison_confidence': match.confidence
                        }
                        result['professional_comparisons'].append(comparison_data)
                        
                    # Add movement statistics
                    movement_stats = comparator.get_movement_statistics(detected_movement)
                    result['movement_statistics'] = movement_stats
                    
                    logger.info(f"[PROFESSIONAL_COMPARISON] Best match: {best_matches[0].professional_name} ({best_matches[0].similarity_score*100:.1f}% similarity)")
                else:
                    logger.warning(f"[PROFESSIONAL_COMPARISON] No matches found for {detected_movement}")
                    result['professional_comparisons'] = []
                    result['movement_statistics'] = {}
                    
            except Exception as e:
                logger.error(f"[PROFESSIONAL_COMPARISON] Error in optimized comparison: {e}")
                result['professional_comparisons'] = []
                result['movement_statistics'] = {}
            
            return AnalysisResult(
                success=True,
                analysis_id=secrets.token_hex(16),
                timestamp=datetime.now(),
                analysis_type='enhanced_single_cycle_biomechanical_with_comparisons',
                final_score=result.get('final_score'),
                detailed_analysis=result,
                recommendations=result.get('recommendations', []),
                professional_comparisons=result.get('professional_comparisons', []),
                movement_statistics=result.get('movement_statistics', {})
            )
        else:
            logger.error(f"[BIOMECH_API] Enhanced analysis failed: {result.get('error')}")
            raise HTTPException(status_code=500, detail=result.get('error', 'Enhanced analysis failed'))
        
    except Exception as e:
        logger.error(f"Error in biomechanical analysis: {e}")
        return AnalysisResult(
            success=False,
            analysis_id=secrets.token_hex(16),
            timestamp=datetime.now(),
            analysis_type='enhanced_single_cycle_biomechanical_with_comparisons',
            error=str(e)
        )



# Development endpoints
@app.get("/dev/test-components")
async def test_components():
    """Test system components (development only)"""
    if os.getenv("ENVIRONMENT") != "development":
        raise HTTPException(status_code=404, detail="Not found")
    
    results = {}
    
    # Test analyzer
    try:
        professionals = analyzer_api.engine.get_available_professionals('forehand_drive')
        results["analyzer"] = {"status": "ok", "professionals_count": len(professionals)}
    except Exception as e:
        results["analyzer"] = {"status": "error", "error": str(e)}
    
    # Test real-time engine
    try:
        engine_test = RealTimeAnalysisEngine()
        results["real_time_engine"] = {"status": "ok", "initialized": True}
    except Exception as e:
        results["real_time_engine"] = {"status": "error", "error": str(e)}
    
    return results

@app.post("/test-real-analysis")
async def test_real_analysis():
    """Test endpoint with exact same code that worked in direct test"""
    try:
        # Use exact same paths and metadata that worked in direct test
        user_video = r"C:\Users\aokub\OneDrive\tennis-analyzer-production\videos\Zhang_Jike_FD_D_D.mp4"
        pro_video = r"C:\Users\aokub\OneDrive\tennis-analyzer-production\profissionais\forehand_drive\Zhang_Jike_FD_D_D.mp4"
        
        user_metadata = {
            'maoDominante': 'Destro',
            'ladoCamera': 'Direita',
            'ladoRaquete': 'F',
            'tipoMovimento': 'D'
        }
        
        prof_metadata = {
            'maoDominante': 'D',
            'ladoCamera': 'D',
            'ladoRaquete': 'F',
            'tipoMovimento': 'D'
        }
        
        print(f"[TEST_REAL] Starting test with direct backend call...")
        print(f"[TEST_REAL] User video exists: {os.path.exists(user_video)}")
        print(f"[TEST_REAL] Pro video exists: {os.path.exists(pro_video)}")
        
        # Call tennis_comparison_backend directly (exactly like successful direct test)
        result = analyzer_api.engine.compare_techniques(
            user_video, pro_video, user_metadata, prof_metadata
        )
        
        print(f"[TEST_REAL] Backend result success: {result.get('success')}")
        print(f"[TEST_REAL] Backend result score: {result.get('final_score')}")
        print(f"[TEST_REAL] Backend result user_analysis keys: {list(result.get('user_analysis', {}).keys())}")
        print(f"[TEST_REAL] Backend result professional_analysis keys: {list(result.get('professional_analysis', {}).keys())}")
        print(f"[TEST_REAL] Backend result comparison keys: {list(result.get('comparison', {}).keys())}")
        
        return result
        
    except Exception as e:
        logger.error(f"[TEST_REAL] Error: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    # Development server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )