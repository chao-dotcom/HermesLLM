"""GitHub crawler for repositories."""

import os
import subprocess
from pathlib import Path

from loguru import logger

from hermes.core import RepositoryDocument
from hermes.collectors.base import BaseCrawler


class GithubCrawler(BaseCrawler):
    """Crawler for GitHub repositories."""
    
    model = RepositoryDocument
    
    def extract(self, link: str, **kwargs) -> None:
        """
        Extract GitHub repository data.
        
        Args:
            link: Repository URL
            **kwargs: Must contain 'user' key with UserDocument
        """
        # Check if already exists
        old_model = self.model.find_one(link=link)
        if old_model:
            logger.info(f"Repository already exists: {link}")
            return
        
        logger.info(f"Cloning GitHub repository: {link}")
        
        # Extract repo name from URL
        repo_name = link.rstrip("/").split("/")[-1]
        if repo_name.endswith(".git"):
            repo_name = repo_name[:-4]
        
        # Clone repository
        clone_dir = Path.cwd() / "temp_repos" / repo_name
        clone_dir.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            subprocess.run(
                ["git", "clone", link, str(clone_dir)],
                check=True,
                capture_output=True,
                text=True
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to clone repository: {e.stderr}")
            return
        
        # Extract file tree and content
        tree = self._get_file_tree(clone_dir)
        content = self._extract_content(clone_dir)
        
        # Save repository
        user = kwargs["user"]
        instance = self.model(
            platform="github",
            link=link,
            name=repo_name,
            content={
                "tree": tree,
                "files": content
            },
            author_id=user.id,
            author_full_name=user.full_name
        )
        instance.save()
        
        # Cleanup
        self._cleanup(clone_dir)
        
        logger.info(f"Successfully scraped GitHub repository: {repo_name}")
    
    def _get_file_tree(self, repo_path: Path) -> str:
        """Get file tree structure."""
        try:
            result = subprocess.run(
                ["tree", "-L", "3", "-I", "*.git"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=False
            )
            return result.stdout if result.returncode == 0 else ""
        except FileNotFoundError:
            # Fallback if tree command not available
            return self._manual_tree(repo_path)
    
    def _manual_tree(self, repo_path: Path, prefix: str = "", level: int = 0) -> str:
        """Manually create tree structure."""
        if level > 3:
            return ""
        
        tree_str = ""
        items = sorted(repo_path.iterdir(), key=lambda x: (not x.is_dir(), x.name))
        
        for item in items:
            if item.name.startswith('.'):
                continue
            
            tree_str += f"{prefix}{item.name}\n"
            if item.is_dir():
                tree_str += self._manual_tree(item, prefix + "  ", level + 1)
        
        return tree_str
    
    def _extract_content(self, repo_path: Path) -> dict:
        """Extract content from important files."""
        content = {}
        
        # Look for important files
        important_files = [
            "README.md",
            "README.rst",
            "README.txt",
            "setup.py",
            "pyproject.toml",
            "requirements.txt",
            "package.json"
        ]
        
        for file_name in important_files:
            file_path = repo_path / file_name
            if file_path.exists():
                try:
                    content[file_name] = file_path.read_text(encoding="utf-8")
                except Exception as e:
                    logger.warning(f"Could not read {file_name}: {e}")
        
        return content
    
    def _cleanup(self, clone_dir: Path) -> None:
        """Remove cloned repository."""
        try:
            import shutil
            shutil.rmtree(clone_dir)
            logger.debug(f"Cleaned up: {clone_dir}")
        except Exception as e:
            logger.warning(f"Could not clean up {clone_dir}: {e}")
