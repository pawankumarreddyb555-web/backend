from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
import io
import json
import uuid
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import numpy as np
import librosa
import soundfile as sf
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import tempfile
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import base64

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI(title="Emotion Detection API", description="Text and Speech Emotion Detection using Machine Learning")

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Global variables for models
text_emotion_model = None
tokenizer = None

# Initialize models at startup
@app.on_event("startup")
async def startup_event():
    global text_emotion_model, tokenizer
    try:
        # Initialize text emotion detection model
        model_name = "j-hartmann/emotion-english-distilroberta-base"
        text_emotion_model = pipeline("text-classification", 
                                    model=model_name, 
                                    tokenizer=model_name,
                                    device=0 if torch.cuda.is_available() else -1)
        logger.info("Text emotion detection model loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        # Continue without GPU if CUDA not available
        try:
            text_emotion_model = pipeline("text-classification", 
                                        model=model_name, 
                                        tokenizer=model_name,
                                        device=-1)  # Force CPU
            logger.info("Text emotion detection model loaded on CPU")
        except Exception as e2:
            logger.error(f"Failed to load model on CPU: {str(e2)}")

# Define Models
class TextEmotionRequest(BaseModel):
    text: str
    
class TextEmotionResponse(BaseModel):
    text: str
    emotion: str
    confidence: float
    all_emotions: Dict[str, float]
    processing_time: float

class SpeechEmotionResponse(BaseModel):
    filename: str
    emotion: str
    confidence: float
    all_emotions: Dict[str, float]
    processing_time: float
    duration: float

class AnalysisHistory(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: str  # "text" or "speech"
    input_data: str
    emotion: str
    confidence: float
    all_emotions: Dict[str, float]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

# Audio processing functions
def extract_audio_features(audio_data, sample_rate):
    """Extract basic audio features for emotion classification"""
    try:
        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1)
        
        # Extract spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)
        spectral_centroids_mean = np.mean(spectral_centroids)
        
        # Extract zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio_data)
        zcr_mean = np.mean(zcr)
        
        # Extract spectral rolloff
        rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate)
        rolloff_mean = np.mean(rolloff)
        
        # Extract chroma features
        chroma = librosa.feature.chroma_stft(y=audio_data, sr=sample_rate)
        chroma_mean = np.mean(chroma, axis=1)
        
        # Combine all features
        features = np.concatenate([
            mfccs_mean,
            [spectral_centroids_mean],
            [zcr_mean],
            [rolloff_mean],
            chroma_mean
        ])
        
        return features
    except Exception as e:
        logger.error(f"Error extracting audio features: {str(e)}")
        return None

def classify_speech_emotion(features):
    """Simple rule-based emotion classification for speech"""
    if features is None:
        return {"neutral": 0.5, "unknown": 0.5}
    
    # This is a simplified approach - in production, you'd use a trained model
    # For now, we'll use basic heuristics based on audio features
    
    # Get key features
    mfcc_mean = np.mean(features[:13])
    spectral_centroid = features[13]
    zcr = features[14]
    
    emotions = {
        "happy": 0.0,
        "sad": 0.0,
        "angry": 0.0,
        "fearful": 0.0,
        "neutral": 0.0,
        "surprised": 0.0
    }
    
    # Simple heuristic-based classification
    if spectral_centroid > 2000 and zcr > 0.1:
        emotions["happy"] = 0.4
        emotions["surprised"] = 0.3
        emotions["neutral"] = 0.3
    elif spectral_centroid > 2500 and mfcc_mean > 0:
        emotions["angry"] = 0.5
        emotions["fearful"] = 0.3
        emotions["neutral"] = 0.2
    elif mfcc_mean < -10:
        emotions["sad"] = 0.6
        emotions["neutral"] = 0.4
    else:
        emotions["neutral"] = 0.8
        emotions["happy"] = 0.2
    
    return emotions

# API Routes
@api_router.get("/")
async def root():
    return {"message": "Emotion Detection API", "status": "running"}

@api_router.post("/analyze-text", response_model=TextEmotionResponse)
async def analyze_text_emotion(request: TextEmotionRequest):
    """Analyze emotion from text input"""
    import time
    start_time = time.time()
    
    try:
        if not text_emotion_model:
            raise HTTPException(status_code=500, detail="Text emotion model not loaded")
        
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        # Get emotion predictions
        predictions = text_emotion_model(request.text)
        
        # Format results
        all_emotions = {}
        top_emotion = predictions[0]['label'].lower()
        top_confidence = predictions[0]['score']
        
        for pred in predictions:
            emotion_name = pred['label'].lower()
            all_emotions[emotion_name] = round(pred['score'], 4)
        
        processing_time = round(time.time() - start_time, 4)
        
        # Store in database
        history_entry = AnalysisHistory(
            type="text",
            input_data=request.text[:200],  # Truncate for storage
            emotion=top_emotion,
            confidence=top_confidence,
            all_emotions=all_emotions
        )
        
        try:
            await db.analysis_history.insert_one(history_entry.model_dump())
        except Exception as db_error:
            logger.warning(f"Failed to store analysis in database: {str(db_error)}")
        
        return TextEmotionResponse(
            text=request.text,
            emotion=top_emotion,
            confidence=round(top_confidence, 4),
            all_emotions=all_emotions,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error in text emotion analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@api_router.post("/analyze-speech", response_model=SpeechEmotionResponse)
async def analyze_speech_emotion(file: UploadFile = File(...)):
    """Analyze emotion from uploaded audio file"""
    import time
    start_time = time.time()
    
    try:
        # Validate file type
        allowed_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.ogg']
        file_extension = os.path.splitext(file.filename)[1].lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file format. Allowed formats: {', '.join(allowed_extensions)}"
            )
        
        # Read audio file
        audio_data = await file.read()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            temp_file.write(audio_data)
            temp_file_path = temp_file.name
        
        try:
            # Load audio with librosa
            y, sr = librosa.load(temp_file_path, sr=16000)  # Resample to 16kHz
            duration = librosa.get_duration(y=y, sr=sr)
            
            if duration < 0.5:
                raise HTTPException(status_code=400, detail="Audio file too short (minimum 0.5 seconds)")
            if duration > 60:
                # Truncate to first 60 seconds
                y = y[:60*sr]
                duration = 60
            
            # Extract features
            features = extract_audio_features(y, sr)
            
            # Classify emotion
            emotion_probs = classify_speech_emotion(features)
            
            # Get top emotion
            top_emotion = max(emotion_probs.items(), key=lambda x: x[1])
            emotion_name = top_emotion[0]
            confidence = top_emotion[1]
            
            processing_time = round(time.time() - start_time, 4)
            
            # Store in database
            history_entry = AnalysisHistory(
                type="speech",
                input_data=file.filename,
                emotion=emotion_name,
                confidence=confidence,
                all_emotions=emotion_probs
            )
            
            try:
                await db.analysis_history.insert_one(history_entry.model_dump())
            except Exception as db_error:
                logger.warning(f"Failed to store analysis in database: {str(db_error)}")
            
            return SpeechEmotionResponse(
                filename=file.filename,
                emotion=emotion_name,
                confidence=round(confidence, 4),
                all_emotions={k: round(v, 4) for k, v in emotion_probs.items()},
                processing_time=processing_time,
                duration=round(duration, 2)
            )
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in speech emotion analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@api_router.get("/history")
async def get_analysis_history(limit: int = 20):
    """Get recent analysis history"""
    try:
        history = await db.analysis_history.find().sort("timestamp", -1).limit(limit).to_list(length=None)
        # Convert ObjectId to string for JSON serialization
        for item in history:
            if "_id" in item:
                item["_id"] = str(item["_id"])
        return {"history": history}
    except Exception as e:
        logger.error(f"Error fetching history: {str(e)}")
        return {"history": []}

@api_router.post("/generate-report")
async def generate_report(analysis_ids: List[str] = None):
    """Generate a PDF report of emotion analysis results"""
    try:
        # Create temporary PDF file
        temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        temp_pdf.close()
        
        # Create PDF document
        doc = SimpleDocTemplate(temp_pdf.name, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title = Paragraph("Emotion Detection Analysis Report", styles['Title'])
        story.append(title)
        story.append(Spacer(1, 20))
        
        # Get analysis data
        if analysis_ids:
            # Get specific analyses
            analyses = await db.analysis_history.find({"id": {"$in": analysis_ids}}).to_list(length=None)
        else:
            # Get recent analyses
            analyses = await db.analysis_history.find().sort("timestamp", -1).limit(10).to_list(length=None)
        
        # Add content
        for i, analysis in enumerate(analyses, 1):
            # Analysis header
            header = Paragraph(f"Analysis #{i} - {analysis['type'].title()}", styles['Heading2'])
            story.append(header)
            story.append(Spacer(1, 10))
            
            # Details
            details = [
                f"Type: {analysis['type'].title()}",
                f"Emotion: {analysis['emotion'].title()}",
                f"Confidence: {analysis['confidence']:.2%}",
                f"Timestamp: {analysis['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}",
            ]
            
            if analysis['type'] == 'text':
                details.append(f"Text: {analysis['input_data'][:100]}...")
            else:
                details.append(f"File: {analysis['input_data']}")
            
            for detail in details:
                para = Paragraph(detail, styles['Normal'])
                story.append(para)
            
            story.append(Spacer(1, 15))
        
        # Build PDF
        doc.build(story)
        
        return FileResponse(
            temp_pdf.name,
            media_type='application/pdf',
            filename=f"emotion_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        )
        
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")

@api_router.get("/stats")
async def get_emotion_stats():
    """Get emotion analysis statistics"""
    try:
        # Get all analyses
        all_analyses = await db.analysis_history.find().to_list(length=None)
        
        if not all_analyses:
            return {"total_analyses": 0, "emotion_distribution": {}, "type_distribution": {}}
        
        # Calculate statistics
        total_analyses = len(all_analyses)
        
        # Emotion distribution
        emotion_counts = {}
        type_counts = {}
        
        for analysis in all_analyses:
            emotion = analysis['emotion']
            analysis_type = analysis['type']
            
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            type_counts[analysis_type] = type_counts.get(analysis_type, 0) + 1
        
        return {
            "total_analyses": total_analyses,
            "emotion_distribution": emotion_counts,
            "type_distribution": type_counts,
            "recent_analyses": len([a for a in all_analyses if a.get('timestamp') and 
                                  (datetime.now(timezone.utc) - 
                                   (a['timestamp'] if a['timestamp'].tzinfo else a['timestamp'].replace(tzinfo=timezone.utc))).days < 7])
        }
        
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        return {"total_analyses": 0, "emotion_distribution": {}, "type_distribution": {}}

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
   