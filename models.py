"""
Pydantic models for request/response schemas.
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class ContentMetadata(BaseModel):
    """Metadata embedded in the content document."""
    keypointsCount: int = 0  # We repurpose this as embedding dimension
    fileSize: int = 0


class ARContentResponse(BaseModel):
    """Response schema for a content item (mirrors reference API contract)."""
    id: str = Field(alias="_id", default="")
    contentId: str
    originalImageName: str
    imagePath: str
    videoPath: str = ""
    videoLink: Optional[str] = None
    videoType: str = "file"
    descriptorPath: str = ""
    metadata: Optional[ContentMetadata] = None
    createdAt: str = ""
    type: str = "video"
    title: Optional[str] = ""
    text: Optional[str] = ""
    url: Optional[str] = ""

    class Config:
        populate_by_name = True


class UploadResponse(BaseModel):
    """Response after successful upload."""
    message: str = "Upload successful"
    contentId: str
    videoUrl: str = ""
    descriptorUrl: str = ""


class VideoLookupResponse(BaseModel):
    """Response for video URL lookup."""
    videoUrl: str


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str


# ── Step 2: Multi-content attachment models ────────────────────────────────

ALLOWED_CONTENT_TYPES = ["video", "audio", "image", "text", "pdf"]


class AttachedContentResponse(BaseModel):
    """Response schema for an attached content item."""
    attachmentId: str
    contentId: str  # Reference to the parent image's contentId
    type: str       # video / audio / image / text / pdf
    url: Optional[str] = None
    text: Optional[str] = None
    title: str = ""
    order: int = 1
    createdAt: str = ""


class ScanResponse(BaseModel):
    """Response schema for a scan (recognition) result."""
    matchFound: bool
    confidence: float
    matchPercentage: int = 0
    content: Optional[ARContentResponse] = None
    attachments: List[AttachedContentResponse] = []
    message: str = ""


class AttachContentRequest(BaseModel):
    """Request schema for attaching content (used for JSON-only requests)."""
    contentId: str
    contentType: str
    url: Optional[str] = None
    text: Optional[str] = None
    title: Optional[str] = ""
