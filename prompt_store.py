"""
prompt_store.py
--------------
Helpers for resolving prompt files from the immutable base tree and the
editable current overlay.
"""

from __future__ import annotations

from pathlib import Path

from config import (
    AI_LANGUAGE,
    BASE_PROMPTS_DIR,
    CURRENT_PROMPTS_DIR,
    get_prompt_language_dir,
)


def _normalize_relative_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        candidate = candidate.relative_to(BASE_PROMPTS_DIR)
    return candidate


def base_path(path: str | Path) -> Path:
    return BASE_PROMPTS_DIR / _normalize_relative_path(path)


def current_path(path: str | Path) -> Path:
    return CURRENT_PROMPTS_DIR / _normalize_relative_path(path)


def resolve_prompt_path(path: str | Path) -> Path:
    base = base_path(path)
    current = current_path(path)
    if current.exists():
        return current
    return base


def resolve_prompt_path_by_language(path: str | Path, language: str | None = None) -> Path:
    """Resolve prompt path from language-specific directory or base.
    
    Priority:
    1. Current overlay (DATA_DIR/prompts/current/{language}/...)
    2. Language-specific base (prompts/{lang}/...)
    3. Default base (prompts/base/...)
    
    Args:
        path: Relativní cesta k prompt souboru
        language: jazyk (en, cs), vypustí AI_LANGUAGE pokud None
    """
    if language is None:
        language = AI_LANGUAGE
    
    # Nejprve zkusit current overlay pro daný jazyk
    language_current = CURRENT_PROMPTS_DIR / language / path
    if language_current.exists():
        return language_current
    
    # Pak language-specific base
    language_dir = get_prompt_language_dir(language)
    language_base = language_dir / path
    if language_base.exists():
        return language_base
    
    # Nakonec fallback na default base
    return base_path(path)


def prompt_exists_in_base(path: str | Path) -> bool:
    return base_path(path).exists()


def list_base_prompt_names() -> list[str]:
    if not BASE_PROMPTS_DIR.exists():
        return []
    names: list[str] = []
    for file_path in sorted(BASE_PROMPTS_DIR.rglob("*")):
        if file_path.is_file():
            names.append(file_path.relative_to(BASE_PROMPTS_DIR).as_posix())
    return names


def list_prompt_names_by_language(language: str) -> list[str]:
    """List prompt names for a specific language directory.
    
    Falls back to base directory if language-specific directory doesn't exist.
    """
    from config import get_prompt_language_dir
    
    lang_dir = get_prompt_language_dir(language)
    if not lang_dir.exists():
        return list_base_prompt_names()
    
    names: list[str] = []
    for file_path in sorted(lang_dir.rglob("*")):
        if file_path.is_file():
            names.append(file_path.relative_to(lang_dir).as_posix())
    return names


def write_current_prompt(path: str | Path, content: str) -> Path:
    destination = current_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8")
    return destination


def delete_current_prompt(path: str | Path) -> bool:
    destination = current_path(path)
    if not destination.exists():
        return False
    destination.unlink()
    return True


def clear_current_prompts() -> int:
    if not CURRENT_PROMPTS_DIR.exists():
        return 0

    removed = 0
    for file_path in sorted(CURRENT_PROMPTS_DIR.rglob("*"), reverse=True):
        if file_path.is_file():
            file_path.unlink()
            removed += 1

    for dir_path in sorted(CURRENT_PROMPTS_DIR.rglob("*"), reverse=True):
        if dir_path.is_dir() and not any(dir_path.iterdir()):
            dir_path.rmdir()

    return removed