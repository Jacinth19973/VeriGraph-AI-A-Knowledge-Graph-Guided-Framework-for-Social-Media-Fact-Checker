"""Social Media API endpoints."""
from __future__ import annotations
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal
from urllib.parse import urlparse
import re

from services.social_media_service import (
    extract_facebook_post, extract_instagram_post, extract_x_post,
    extract_public_metadata, SOCIAL_PLATFORMS
)

router = APIRouter()


class SocialMediaPostRequest(BaseModel):
    platform: Literal["facebook", "instagram", "x"] = Field(..., description="Social media platform")
    post_url: str = Field(..., description="Full URL to the social media post")


class SocialMediaPostResponse(BaseModel):
    platform: str
    post_url: str
    post_id: str = ""
    author: str = ""
    author_id: str = ""
    caption: str = ""
    full_text: str = ""
    image_url: str = ""
    thumbnail_url: str = ""
    video_url: str = ""
    published_at: str = ""
    location: str = ""
    extraction_method: Literal["official_api", "public_metadata", "manual_input"] = "manual_input"
    extraction_status: Literal["success", "partial", "failed", "unsupported"] = "unsupported"
    error: str = ""
    metadata: Dict[str, Any] = {}


class SocialMediaVerificationRequest(BaseModel):
    platform: Literal["facebook", "instagram", "x"]
    post_url: str
    use_extracted_content: bool = True


PLATFORM_DISPLAY = {
    "facebook": "Facebook",
    "instagram": "Instagram", 
    "x": "X",
}


@router.get("/platforms")
async def get_supported_platforms():
    """Get list of supported social media platforms."""
    return {
        "platforms": [
            {
                "id": "facebook",
                "name": "Facebook",
                "description": "Facebook posts via Graph API or public metadata",
                "requires_auth": True,
                "supports_images": True,
                "supports_video": True,
            },
            {
                "id": "instagram",
                "name": "Instagram",
                "description": "Instagram posts via Meta Graph API or public metadata",
                "requires_auth": True,
                "supports_images": True,
                "supports_video": True,
            },
            {
                "id": "x",
                "name": "X (formerly Twitter)",
                "description": "X posts via X API or public metadata",
                "requires_auth": True,
                "supports_images": True,
                "supports_video": True,
            },
        ]
    }


@router.post("/fetch", response_model=SocialMediaPostResponse)
async def fetch_social_media_post(request: SocialMediaPostRequest):
    """Fetch a social media post and extract its content."""
    platform = request.platform.lower()
    post_url = request.post_url.strip()
    
    if not post_url:
        raise HTTPException(status_code=400, detail="Post URL is required")
    
    # Validate URL format
    parsed = urlparse(post_url)
    if not parsed.scheme or not parsed.netloc:
        raise HTTPException(status_code=400, detail="Invalid URL format")
    
    try:
        if platform == "facebook":
            result = await extract_facebook_post(post_url)
        elif platform == "instagram":
            result = await extract_instagram_post(post_url)
        elif platform == "x":
            result = await extract_x_post(post_url)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported platform: {platform}")
        
        return SocialMediaPostResponse(**result)
    
    except HTTPException:
        raise
    except Exception as e:
        return SocialMediaPostResponse(
            platform=PLATFORM_DISPLAY.get(platform, platform),
            post_url=post_url,
            extraction_method="manual_input",
            extraction_status="failed",
            error=str(e),
        )


@router.post("/verify")
async def verify_social_media_post(request: SocialMediaVerificationRequest, background_tasks: BackgroundTasks):
    """Fetch social media post and run verification pipeline."""
    # First fetch the post
    fetch_result = await fetch_social_media_post(SocialMediaPostRequest(
        platform=request.platform,
        post_url=request.post_url,
    ))
    
    if fetch_result.extraction_status in ("failed", "unsupported") and not fetch_result.image_url:
        raise HTTPException(
            status_code=400, 
            detail=f"Could not extract content from {fetch_result.platform} post. Please use manual input."
        )
    
    # Prepare verification input
    from pipeline.orchestrator import verify_claim
    import uuid
    from services.time_service import india_now

    case_id = f"SOCIAL-{india_now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8]}"
    
    claim = fetch_result.caption or fetch_result.full_text or "No text content extracted"
    image_path = fetch_result.image_url if fetch_result.image_url else ""
    
    case_input = {
        "case_id": case_id,
        "claim": claim,
        "image_path": image_path,
        "promote_knowledge": False,
        "metadata": {
            "source_platform": fetch_result.platform,
            "source_url": fetch_result.post_url,
            "post_id": fetch_result.post_id,
            "author": fetch_result.author,
            "published_at": fetch_result.published_at,
            "location": fetch_result.location,
            "extraction_method": fetch_result.extraction_method,
            "extraction_status": fetch_result.extraction_status,
            "original_metadata": fetch_result.metadata,
        },
    }
    
    # Run verification
    try:
        result = verify_claim(case_input)
        return {
            "case_id": case_id,
            "verification_result": result.dict(),
            "social_media_extraction": fetch_result.dict(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Verification failed: {e}")