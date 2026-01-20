"""YouTube transcript crawler with cleaning and error handling."""

import re
from typing import Any
from urllib.parse import parse_qs, urlparse

from loguru import logger

from hermes.core import ArticleDocument
from hermes.collectors.base import BaseCrawler

try:
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api._errors import (
        NoTranscriptFound,
        TranscriptsDisabled,
        VideoUnavailable,
        TooManyRequests,
        YouTubeRequestFailed
    )
    YOUTUBE_API_AVAILABLE = True
except ImportError:
    YOUTUBE_API_AVAILABLE = False
    logger.warning(
        "youtube-transcript-api not installed. "
        "Install with: pip install youtube-transcript-api"
    )


class YouTubeTranscriptCrawler(BaseCrawler):
    """Crawler for YouTube video transcripts."""
    
    model = ArticleDocument
    
    def __init__(self, preferred_languages: list[str] | None = None):
        """
        Initialize YouTube transcript crawler.
        
        Args:
            preferred_languages: List of preferred language codes (e.g., ['en', 'es'])
                                Default: ['en']
        """
        if not YOUTUBE_API_AVAILABLE:
            raise ImportError(
                "youtube-transcript-api is required for YouTube transcript crawler. "
                "Install with: pip install youtube-transcript-api"
            )
        
        self.preferred_languages = preferred_languages or ["en"]
    
    def extract_video_id(self, url: str) -> str | None:
        """
        Extract video ID from YouTube URL.
        
        Args:
            url: YouTube video URL
            
        Returns:
            Video ID or None if invalid URL
        """
        try:
            parsed_url = urlparse(url)
            
            # Handle youtu.be short links
            if parsed_url.netloc in ["youtu.be", "www.youtu.be"]:
                return parsed_url.path.lstrip("/").split("/")[0].split("?")[0]
            
            # Handle youtube.com links
            if "youtube.com" in parsed_url.netloc:
                # Check for /watch?v= format
                if parsed_url.path == "/watch":
                    query_params = parse_qs(parsed_url.query)
                    if "v" in query_params:
                        return query_params["v"][0]
                
                # Check for /embed/ format
                if parsed_url.path.startswith("/embed/"):
                    return parsed_url.path.split("/embed/")[1].split("?")[0]
                
                # Check for /v/ format
                if parsed_url.path.startswith("/v/"):
                    return parsed_url.path.split("/v/")[1].split("?")[0]
            
            logger.error(f"Could not extract video ID from URL: {url}")
            return None
            
        except Exception as e:
            logger.error(f"Error parsing YouTube URL {url}: {e}")
            return None
    
    def clean_transcript(self, transcript_data: list[dict[str, Any]]) -> str:
        """
        Clean and format raw transcript data.
        
        Args:
            transcript_data: Raw transcript data from API
            
        Returns:
            Cleaned transcript text
        """
        if not transcript_data:
            return ""
        
        # Combine all text segments
        raw_text = " ".join([segment.get("text", "") for segment in transcript_data])
        
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', raw_text)
        
        # Remove common YouTube auto-caption artifacts
        cleaned = re.sub(r'\[Music\]', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\[Applause\]', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\[Laughter\]', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\[.*?\]', '', cleaned)  # Remove other bracketed content
        
        # Fix common transcription issues
        cleaned = re.sub(r'\s+([.,!?;:])', r'\1', cleaned)  # Fix punctuation spacing
        cleaned = re.sub(r'([.,!?;:])\s*', r'\1 ', cleaned)  # Normalize after punctuation
        
        # Remove duplicate spaces again after all replacements
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Create paragraphs based on sentence endings (improved readability)
        sentences = re.split(r'([.!?]+\s+)', cleaned)
        paragraphs = []
        current_paragraph = []
        
        for i in range(0, len(sentences), 2):
            if i < len(sentences):
                sentence = sentences[i]
                if i + 1 < len(sentences):
                    sentence += sentences[i + 1]
                
                current_paragraph.append(sentence)
                
                # Create new paragraph every 3-5 sentences
                if len(current_paragraph) >= 4:
                    paragraphs.append("".join(current_paragraph).strip())
                    current_paragraph = []
        
        # Add remaining sentences
        if current_paragraph:
            paragraphs.append("".join(current_paragraph).strip())
        
        return "\n\n".join(paragraphs).strip()
    
    def get_video_metadata(self, video_id: str) -> dict[str, Any]:
        """
        Get basic video metadata (would require additional API).
        For now, returns minimal metadata.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            Metadata dictionary
        """
        # In a full implementation, you could use YouTube Data API v3
        # For now, we'll return minimal info
        return {
            "video_id": video_id,
            "url": f"https://www.youtube.com/watch?v={video_id}"
        }
    
    def extract(self, link: str, **kwargs) -> None:
        """
        Extract transcript from YouTube video.
        
        Args:
            link: YouTube video URL
            **kwargs: Must contain:
                - 'user': UserDocument
                Optional:
                - 'languages': list[str] - Preferred languages
                - 'preserve_formatting': bool - Keep timestamps (default: False)
        """
        # Check if already exists
        old_model = self.model.find_one(link=link)
        if old_model:
            logger.info(f"YouTube transcript already exists: {link}")
            return
        
        # Extract user
        user = kwargs.get("user")
        if not user:
            logger.error("User parameter is required for YouTube transcript")
            return
        
        logger.info(f"Extracting YouTube transcript: {link}")
        
        # Extract video ID
        video_id = self.extract_video_id(link)
        if not video_id:
            logger.error(f"Failed to extract video ID from: {link}")
            return
        
        # Override languages if provided
        languages = kwargs.get("languages", self.preferred_languages)
        
        try:
            # Fetch transcript
            logger.debug(f"Fetching transcript for video ID: {video_id}")
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            
            # Try to get transcript in preferred language
            transcript = None
            transcript_language = None
            
            # First try manually created transcripts
            try:
                for lang in languages:
                    try:
                        transcript = transcript_list.find_manually_created_transcript([lang])
                        transcript_language = lang
                        logger.info(f"Found manual transcript in {lang}")
                        break
                    except NoTranscriptFound:
                        continue
            except Exception as e:
                logger.debug(f"No manual transcripts found: {e}")
            
            # If no manual transcript, try auto-generated
            if not transcript:
                try:
                    for lang in languages:
                        try:
                            transcript = transcript_list.find_generated_transcript([lang])
                            transcript_language = lang
                            logger.info(f"Found auto-generated transcript in {lang}")
                            break
                        except NoTranscriptFound:
                            continue
                except Exception as e:
                    logger.debug(f"No auto-generated transcripts found: {e}")
            
            # If still no transcript, try any available
            if not transcript:
                logger.warning(f"No transcript in preferred languages {languages}, trying any available")
                try:
                    available = list(transcript_list)
                    if available:
                        transcript = available[0]
                        transcript_language = transcript.language_code
                        logger.info(f"Using available transcript in {transcript_language}")
                except Exception as e:
                    logger.error(f"No transcripts available at all: {e}")
                    return
            
            if not transcript:
                logger.error(f"No transcript found for video: {video_id}")
                return
            
            # Fetch transcript data
            transcript_data = transcript.fetch()
            
            # Clean transcript
            cleaned_transcript = self.clean_transcript(transcript_data)
            
            if not cleaned_transcript:
                logger.error(f"Transcript cleaning resulted in empty content for: {video_id}")
                return
            
            # Get metadata
            metadata = self.get_video_metadata(video_id)
            
            # Build content structure
            content_data = {
                "title": f"YouTube Video Transcript - {video_id}",
                "content": cleaned_transcript,
                "video_id": video_id,
                "language": transcript_language or "unknown",
                "transcript_type": "manual" if transcript.is_generated is False else "auto-generated"
            }
            
            # Save document
            instance = self.model(
                platform="youtube",
                link=link,
                title=f"YouTube Transcript - {video_id}",
                content=content_data,
                author_id=user.id,
                author_full_name=user.full_name,
                tags=["youtube", "transcript", "video"]
            )
            instance.save()
            
            logger.info(
                f"Successfully extracted and cleaned YouTube transcript: {video_id} "
                f"({len(cleaned_transcript)} chars, language: {transcript_language})"
            )
            
        except TranscriptsDisabled:
            logger.error(f"Transcripts are disabled for video: {video_id}")
        except VideoUnavailable:
            logger.error(f"Video is unavailable: {video_id}")
        except TooManyRequests:
            logger.error(f"Too many requests to YouTube API. Please wait and try again later.")
        except YouTubeRequestFailed as e:
            logger.error(f"YouTube API request failed for {video_id}: {e}")
        except NoTranscriptFound:
            logger.error(f"No transcript found for video: {video_id}")
        except Exception as e:
            logger.error(f"Unexpected error extracting transcript for {video_id}: {e}")
            logger.exception(e)
    
    @classmethod
    def extract_transcript(
        cls,
        user,
        url: str,
        languages: list[str] | None = None
    ) -> ArticleDocument | None:
        """
        Convenience method for extracting YouTube transcript.
        
        Args:
            user: UserDocument instance
            url: YouTube video URL
            languages: Preferred language codes
            
        Returns:
            ArticleDocument instance or None if failed
        """
        crawler = cls(preferred_languages=languages)
        crawler.extract(link=url, user=user, languages=languages)
        
        # Return the saved document
        return cls.model.find_one(link=url)
