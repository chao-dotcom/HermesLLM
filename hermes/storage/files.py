"""File storage utilities."""

import json
from pathlib import Path
from typing import Any

from loguru import logger


class FileStorage:
    """File-based storage for artifacts and data."""
    
    def __init__(self, base_path: str | Path = "data"):
        """
        Initialize file storage.
        
        Args:
            base_path: Base directory for storage
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def save_json(self, data: Any, filename: str, subdir: str | None = None) -> Path:
        """
        Save data as JSON file.
        
        Args:
            data: Data to save
            filename: Filename (with or without .json)
            subdir: Optional subdirectory
            
        Returns:
            Path to saved file
        """
        if not filename.endswith(".json"):
            filename = f"{filename}.json"
        
        if subdir:
            file_path = self.base_path / subdir / filename
            file_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            file_path = self.base_path / filename
        
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved JSON to {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Failed to save JSON: {e}")
            raise
    
    def load_json(self, filename: str, subdir: str | None = None) -> Any:
        """
        Load data from JSON file.
        
        Args:
            filename: Filename to load
            subdir: Optional subdirectory
            
        Returns:
            Loaded data
        """
        if not filename.endswith(".json"):
            filename = f"{filename}.json"
        
        if subdir:
            file_path = self.base_path / subdir / filename
        else:
            file_path = self.base_path / filename
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            logger.info(f"Loaded JSON from {file_path}")
            return data
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Failed to load JSON: {e}")
            raise
    
    def save_text(self, text: str, filename: str, subdir: str | None = None) -> Path:
        """
        Save text to file.
        
        Args:
            text: Text content
            filename: Filename
            subdir: Optional subdirectory
            
        Returns:
            Path to saved file
        """
        if subdir:
            file_path = self.base_path / subdir / filename
            file_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            file_path = self.base_path / filename
        
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(text)
            
            logger.info(f"Saved text to {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Failed to save text: {e}")
            raise
    
    def load_text(self, filename: str, subdir: str | None = None) -> str:
        """
        Load text from file.
        
        Args:
            filename: Filename to load
            subdir: Optional subdirectory
            
        Returns:
            Text content
        """
        if subdir:
            file_path = self.base_path / subdir / filename
        else:
            file_path = self.base_path / filename
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            
            logger.info(f"Loaded text from {file_path}")
            return text
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Failed to load text: {e}")
            raise
    
    def exists(self, filename: str, subdir: str | None = None) -> bool:
        """Check if file exists."""
        if subdir:
            file_path = self.base_path / subdir / filename
        else:
            file_path = self.base_path / filename
        return file_path.exists()
    
    def list_files(self, subdir: str | None = None, pattern: str = "*") -> list[Path]:
        """
        List files in directory.
        
        Args:
            subdir: Optional subdirectory
            pattern: Glob pattern for filtering
            
        Returns:
            List of file paths
        """
        if subdir:
            dir_path = self.base_path / subdir
        else:
            dir_path = self.base_path
        
        if not dir_path.exists():
            return []
        
        return list(dir_path.glob(pattern))
