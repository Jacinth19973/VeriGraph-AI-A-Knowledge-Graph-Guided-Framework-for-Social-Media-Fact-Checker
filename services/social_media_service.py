"""VeriGraph AI — Social Media ingestion service."""
from __future__ import annotations
import re
import json
import asyncio
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, parse_qs
from datetime import datetime

import requests
from bs4 import BeautifulSoup

# Platform configuration
SOCIAL_PLATFORMS = {
    "facebook": {
        "name": "Facebook",
        "domains": ["facebook.com", "fb.com", "m.facebook.com"],
        "api_base": "https://graph.facebook.com/v18.0",
    },
    "instagram": {
        "name": "Instagram",
        "domains": ["instagram.com", "www.instagram.com"],
        "api_base": "https://graph.facebook.com/v18.0",
    },
    "x": {
        "name": "X",
        "domains": ["x.com", "twitter.com", "mobile.twitter.com"],
        "api_base": "https://api.twitter.com/2",
    },
}


def _get_platform_from_url(url: str) -> Optional[str]:
    """Detect platform from URL."""
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    for platform, config in SOCIAL_PLATFORMS.items():
        if any(d in domain for d in config["domains"]):
            return platform
    return None


async def extract_facebook_post(url: str) -> Dict[str, Any]:
    """
    Extract Facebook post content.
    
    Attempts:
    1. Official Graph API (requires access token)
    2. Public metadata extraction (og: tags)
    """
    post_id = _extract_facebook_post_id(url)
    
    # Try official API if credentials available
    from config.settings import FACEBOOK_ACCESS_TOKEN
    if FACEBOOK_ACCESS_TOKEN:
        try:
            api_result = await _fetch_facebook_graph_api(post_id, FACEBOOK_ACCESS_TOKEN)
            if api_result:
                return api_result
        except Exception:
            pass  # Fall through to public metadata
    
    # Fallback: Public metadata extraction
    return await _extract_public_facebook_metadata(url, post_id)


async def _fetch_facebook_graph_api(post_id: str, access_token: str) -> Optional[Dict[str, Any]]:
    """Fetch post via Facebook Graph API."""
    import aiohttp
    
    fields = "id,message,full_picture,created_time,permalink_url,attachments,place,from"
    url = f"https://graph.facebook.com/v18.0/{post_id}"
    params = {"fields": fields, "access_token": access_token}
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as resp:
            if resp.status == 200:
                data = await resp.json()
                return _format_facebook_response(data, "official_api")
            elif resp.status == 403:
                # Permission denied
                return None
    return None


async def _extract_public_facebook_metadata(url: str, post_id: str) -> Dict[str, Any]:
    """Extract Facebook post metadata from public page."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=15, verify=False)
        if response.status_code != 200:
            return _failed_response("facebook", url, f"HTTP {response.status_code}")
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Extract OpenGraph metadata
        og_title = soup.find("meta", property="og:title")
        og_description = soup.find("meta", property="og:description")
        og_image = soup.find("meta", property="og:image")
        og_url = soup.find("meta", property="og:url")
        
        # Try to find JSON-LD
        json_ld = soup.find("script", type="application/ld+json")
        structured_data = {}
        if json_ld:
            try:
                structured_data = json.loads(json_ld.string)
            except Exception:
                pass
        
        return {
            "platform": "Facebook",
            "post_url": url,
            "post_id": post_id,
            "author": og_title.get("content", "") if og_title else "",
            "caption": og_description.get("content", "") if og_description else "",
            "full_text": og_description.get("content", "") if og_description else "",
            "image_url": og_image.get("content", "") if og_image else "",
            "thumbnail_url": og_image.get("content", "") if og_image else "",
            "published_at": "",
            "location": "",
            "extraction_method": "public_metadata",
            "extraction_status": "partial" if og_title else "failed",
            "metadata": {"structured_data": structured_data},
        }
    except Exception as e:
        return _failed_response("facebook", url, str(e))


def _extract_facebook_post_id(url: str) -> str:
    """Extract post ID from Facebook URL."""
    parsed = urlparse(url)
    path = parsed.path
    
    # Pattern: /posts/{post_id} or /photos/{photo_id} or /videos/{video_id}
    patterns = [
        r"/posts/(\d+)",
        r"/photos/(\d+)", 
        r"/videos/(\d+)",
        r"/(\d{15,})",  # Long numeric IDs
    ]
    for pattern in patterns:
        match = re.search(pattern, path)
        if match:
            return match.group(1)
    
    # Try query parameters
    query = parse_qs(parsed.query)
    for param in ["story_fbid", "id", "post_id"]:
        if param in query:
            return query[param][0]
    
    return ""


async def extract_instagram_post(url: str) -> Dict[str, Any]:
    """
    Extract Instagram post content.
    
    Attempts:
    1. Official Graph API (requires access token + permissions)
    2. Public metadata extraction (og: tags)
    """
    post_id = _extract_instagram_shortcode(url)
    
    # Try official API if credentials available
    from config.settings import INSTAGRAM_ACCESS_TOKEN
    if INSTAGRAM_ACCESS_TOKEN:
        try:
            api_result = await _fetch_instagram_graph_api(post_id, INSTAGRAM_ACCESS_TOKEN)
            if api_result:
                return api_result
        except Exception:
            pass
    
    # Fallback: Public metadata
    return await _extract_public_instagram_metadata(url, post_id)


async def _fetch_instagram_graph_api(shortcode: str, access_token: str) -> Optional[Dict[str, Any]]:
    """Fetch Instagram post via Graph API."""
    import aiohttp
    
    # First get media ID from shortcode
    url = f"https://graph.facebook.com/v18.0/{shortcode}"
    params = {"fields": "id,media_type,media_url,thumbnail_url,caption,timestamp,permalink,username", "access_token": access_token}
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as resp:
            if resp.status == 200:
                data = await resp.json()
                return _format_instagram_response(data, "official_api")
    return None


async def _extract_public_instagram_metadata(url: str, shortcode: str) -> Dict[str, Any]:
    """Extract Instagram post metadata from public page."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=15, verify=False)
        if response.status_code != 200:
            return _failed_response("instagram", url, f"HTTP {response.status_code}")
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # OpenGraph
        og_title = soup.find("meta", property="og:title")
        og_description = soup.find("meta", property="og:description")
        og_image = soup.find("meta", property="og:image")
        og_video = soup.find("meta", property="og:video")
        
        # Instagram also has meta tags
        insta_title = soup.find("meta", property="instapp:title")
        
        return {
            "platform": "Instagram",
            "post_url": url,
            "post_id": shortcode,
            "author": insta_title.get("content", "") if insta_title else (og_title.get("content", "") if og_title else ""),
            "caption": og_description.get("content", "") if og_description else "",
            "full_text": og_description.get("content", "") if og_description else "",
            "image_url": og_image.get("content", "") if og_image else "",
            "thumbnail_url": og_image.get("content", "") if og_image else "",
            "video_url": og_video.get("content", "") if og_video else "",
            "published_at": "",
            "location": "",
            "extraction_method": "public_metadata",
            "extraction_status": "partial" if og_title else "failed",
            "metadata": {},
        }
    except Exception as e:
        return _failed_response("instagram", url, str(e))


def _extract_instagram_shortcode(url: str) -> str:
    """Extract shortcode from Instagram URL."""
    parsed = urlparse(url)
    path = parsed.path
    
    # Pattern: /p/{shortcode}/ or /reel/{shortcode}/ or /tv/{shortcode}/
    patterns = [
        r"/p/([A-Za-z0-9_-]+)",
        r"/reel/([A-Za-z0-9_-]+)",
        r"/tv/([A-Za-z0-9_-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, path)
        if match:
            return match.group(1)
    
    return ""


async def extract_x_post(url: str) -> Dict[str, Any]:
    """
    Extract X (Twitter) post content.
    
    Attempts:
    1. Official X API v2 (requires Bearer token)
    2. Public metadata extraction (og: tags, twitter:card)
    """
    tweet_id = _extract_x_tweet_id(url)
    
    # Try official API if credentials available
    from config.settings import X_BEARER_TOKEN
    if X_BEARER_TOKEN:
        try:
            api_result = await _fetch_x_api(tweet_id, X_BEARER_TOKEN)
            if api_result:
                return api_result
        except Exception:
            pass
    
    # Fallback: Public metadata
    return await _extract_public_x_metadata(url, tweet_id)


async def _fetch_x_api(tweet_id: str, bearer_token: str) -> Optional[Dict[str, Any]]:
    """Fetch tweet via X API v2."""
    import aiohttp
    
    url = f"https://api.twitter.com/2/tweets/{tweet_id}"
    params = {
        "tweet.fields": "created_at,author_id,public_metrics,attachments,geo,entities",
        "expansions": "author_id,attachments.media_keys",
        "media.fields": "url,preview_image_url,type",
        "user.fields": "username,name,profile_image_url",
    }
    headers = {"Authorization": f"Bearer {bearer_token}"}
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params, headers=headers) as resp:
            if resp.status == 200:
                data = await resp.json()
                return _format_x_response(data, "official_api")
    return None


async def _extract_public_x_metadata(url: str, tweet_id: str) -> Dict[str, Any]:
    """Extract X post metadata from public page."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=15, verify=False)
        if response.status_code != 200:
            return _failed_response("x", url, f"HTTP {response.status_code}")
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Twitter card metadata
        twitter_card = soup.find("meta", attrs={"name": "twitter:card"})
        twitter_title = soup.find("meta", attrs={"name": "twitter:title"})
        twitter_description = soup.find("meta", attrs={"name": "twitter:description"})
        twitter_image = soup.find("meta", attrs={"name": "twitter:image"})
        twitter_player = soup.find("meta", attrs={"name": "twitter:player"})
        
        # OpenGraph fallback
        og_title = soup.find("meta", property="og:title")
        og_description = soup.find("meta", property="og:description")
        og_image = soup.find("meta", property="og:image")
        
        # JSON-LD
        json_ld = soup.find("script", type="application/ld+json")
        structured_data = {}
        if json_ld:
            try:
                structured_data = json.loads(json_ld.string)
            except Exception:
                pass
        
        title = twitter_title.get("content", "") if twitter_title else (og_title.get("content", "") if og_title else "")
        description = twitter_description.get("content", "") if twitter_description else (og_description.get("content", "") if og_description else "")
        image = twitter_image.get("content", "") if twitter_image else (og_image.get("content", "") if og_image else "")
        
        return {
            "platform": "X",
            "post_url": url,
            "post_id": tweet_id,
            "author": "",
            "caption": description,
            "full_text": description,
            "image_url": image,
            "thumbnail_url": image,
            "video_url": twitter_player.get("content", "") if twitter_player else "",
            "published_at": "",
            "location": "",
            "extraction_method": "public_metadata",
            "extraction_status": "partial" if title else "failed",
            "metadata": {"structured_data": structured_data},
        }
    except Exception as e:
        return _failed_response("x", url, str(e))


def _extract_x_tweet_id(url: str) -> str:
    """Extract tweet ID from X URL."""
    parsed = urlparse(url)
    path = parsed.path
    
    # Pattern: /username/status/{tweet_id}
    match = re.search(r"/status/(\d+)", path)
    if match:
        return match.group(1)
    
    return ""


async def extract_public_metadata(url: str) -> Dict[str, Any]:
    """Generic public metadata extraction for any URL."""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    
    try:
        response = requests.get(url, headers=headers, timeout=15, verify=False)
        if response.status_code != 200:
            return _failed_response("unknown", url, f"HTTP {response.status_code}")
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # OpenGraph
        og_title = soup.find("meta", property="og:title")
        og_description = soup.find("meta", property="og:description")
        og_image = soup.find("meta", property="og:image")
        og_url = soup.find("meta", property="og:url")
        
        # Twitter card
        twitter_title = soup.find("meta", attrs={"name": "twitter:title"})
        twitter_description = soup.find("meta", attrs={"name": "twitter:description"})
        twitter_image = soup.find("meta", attrs={"name": "twitter:image"})
        
        title = og_title.get("content", "") if og_title else (twitter_title.get("content", "") if twitter_title else "")
        description = og_description.get("content", "") if og_description else (twitter_description.get("content", "") if twitter_description else "")
        image = og_image.get("content", "") if og_image else (twitter_image.get("content", "") if twitter_image else "")
        
        platform = _get_platform_from_url(url) or "unknown"
        
        return {
            "platform": SOCIAL_PLATFORMS.get(platform, {}).get("name", "Unknown"),
            "post_url": url,
            "post_id": "",
            "author": "",
            "caption": description,
            "full_text": description,
            "image_url": image,
            "thumbnail_url": image,
            "published_at": "",
            "location": "",
            "extraction_method": "public_metadata",
            "extraction_status": "partial" if title else "failed",
            "metadata": {},
        }
    except Exception as e:
        return _failed_response("unknown", url, str(e))


def _format_facebook_response(data: Dict[str, Any], method: str) -> Dict[str, Any]:
    """Format Facebook Graph API response."""
    return {
        "platform": "Facebook",
        "post_url": data.get("permalink_url", ""),
        "post_id": data.get("id", ""),
        "author": data.get("from", {}).get("name", ""),
        "author_id": data.get("from", {}).get("id", ""),
        "caption": data.get("message", ""),
        "full_text": data.get("message", ""),
        "image_url": data.get("full_picture", ""),
        "thumbnail_url": data.get("full_picture", ""),
        "published_at": data.get("created_time", ""),
        "location": data.get("place", {}).get("name", "") if data.get("place") else "",
        "extraction_method": method,
        "extraction_status": "success",
        "metadata": data,
    }


def _format_instagram_response(data: Dict[str, Any], method: str) -> Dict[str, Any]:
    """Format Instagram Graph API response."""
    return {
        "platform": "Instagram",
        "post_url": data.get("permalink", ""),
        "post_id": data.get("id", ""),
        "author": data.get("username", ""),
        "author_id": data.get("id", ""),
        "caption": data.get("caption", ""),
        "full_text": data.get("caption", ""),
        "image_url": data.get("media_url", "") if data.get("media_type") == "IMAGE" else "",
        "thumbnail_url": data.get("thumbnail_url", "") if data.get("thumbnail_url") else data.get("media_url", ""),
        "video_url": data.get("media_url", "") if data.get("media_type") == "VIDEO" else "",
        "published_at": data.get("timestamp", ""),
        "location": "",
        "extraction_method": method,
        "extraction_status": "success",
        "metadata": data,
    }


def _format_x_response(data: Dict[str, Any], method: str) -> Dict[str, Any]:
    """Format X API v2 response."""
    tweet = data.get("data", {})
    users = {u["id"]: u for u in data.get("includes", {}).get("users", [])}
    media = {m["media_key"]: m for m in data.get("includes", {}).get("media", [])}
    
    author = users.get(tweet.get("author_id", ""), {})
    attachments = tweet.get("attachments", {}).get("media_keys", [])
    
    image_url = ""
    video_url = ""
    for key in attachments:
        m = media.get(key, {})
        if m.get("type") == "photo":
            image_url = m.get("url", "")
        elif m.get("type") in ("video", "animated_gif"):
            video_url = m.get("url", "")
    
    return {
        "platform": "X",
        "post_url": f"https://x.com/{author.get('username', '')}/status/{tweet.get('id', '')}",
        "post_id": tweet.get("id", ""),
        "author": author.get("name", ""),
        "author_id": author.get("username", ""),
        "caption": tweet.get("text", ""),
        "full_text": tweet.get("text", ""),
        "image_url": image_url,
        "thumbnail_url": image_url,
        "video_url": video_url,
        "published_at": tweet.get("created_at", ""),
        "location": "",
        "extraction_method": method,
        "extraction_status": "success",
        "metadata": data,
    }


def _failed_response(platform: str, url: str, error: str) -> Dict[str, Any]:
    """Create a failed extraction response."""
    platform_names = {"facebook": "Facebook", "instagram": "Instagram", "x": "X"}
    return {
        "platform": platform_names.get(platform, platform.capitalize()),
        "post_url": url,
        "post_id": "",
        "author": "",
        "caption": "",
        "full_text": "",
        "image_url": "",
        "thumbnail_url": "",
        "video_url": "",
        "published_at": "",
        "location": "",
        "extraction_method": "public_metadata",
        "extraction_status": "failed",
        "error": error,
        "metadata": {},
    }