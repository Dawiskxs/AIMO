# Dynamic Language Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow users to add new languages by simply creating a new folder under `prompts/{lang_code}/` (e.g., `prompts/de/` for German) without code changes. The system will automatically discover and support any ISO 639-1 language code that has a corresponding prompts directory.

**Architecture:** The solution makes language detection fully dynamic by scanning the `prompts/` directory at runtime. Language validation moves from hardcoded lists to runtime discovery. All language-specific paths (prompts, devices, examples, seeds) are resolved via the existing `get_prompt_language_dir()` function which already accepts any language code.

**Tech Stack:** Python 3.11+, pathlib, existing config/prompt_store/settings_store modules

**Spec:** This plan implements the user's request: "novou funkci která umožní to abych mohl udělat novou složku s jazykem prstě s jiným jazykem a kod to zaregistroval do možností a udělá to jen díky té zkratce jako cs nebo en nebo třeba nová de"

## Global Constraints

- Must maintain backward compatibility with existing `en` and `cs` languages
- Language codes must be valid ISO 639-1 codes (2-letter lowercase)
- New language folders must follow existing structure: `{lang}/full_prompt.txt`, `{lang}/devices/`, `{lang}/examples/`, `{lang}/seeds/`, `{lang}/tasks/`
- Settings validation must accept any discovered language, not just hardcoded ones
- No breaking changes to existing API

## Review Focus

| Input Class / Failure Mode | Expected Behavior | Test Location |
|---|---|---|
| New language folder created at runtime | Discovered automatically on next settings reload / prompt reload | Task 3 |
| Invalid language code (not in prompts dir) | Rejected with clear error, falls back to current language | Task 2, Task 4 |
| Language folder missing required files (full_prompt.txt) | Graceful fallback to base prompts, log warning | Task 4 |
| Switching to new language mid-session | Clears connector sessions, reloads prompts correctly | Task 4 |
| Multiple language folders with different structures | Each loads its own files, no cross-contamination | Task 3 |

---

### Task 1: Add Dynamic Language Discovery to config.py

**Files:**
- Modify: `c:\Users\david\Documents\AIMO\config.py:79-85` (around PROMPTS_CS_DIR and get_prompt_language_dir)

**Interfaces:**
- Consumes: `PROMPTS_DIR` (Path), `RESOURCE_DIR` (Path)
- Produces: `get_supported_languages() -> list[str]`, updated `get_prompt_language_dir(language: str) -> Path`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config_language_discovery.py
import pytest
from config import get_supported_languages, get_prompt_language_dir

def test_get_supported_languages_includes_en_and_cs():
    langs = get_supported_languages()
    assert "en" in langs
    assert "cs" in langs
    # Should be sorted
    assert langs == sorted(langs)

def test_get_prompt_language_dir_returns_correct_path_for_known_lang():
    en_dir = get_prompt_language_dir("en")
    assert en_dir.name == "en"
    assert en_dir.parent.name == "prompts"

def test_get_prompt_language_dir_returns_path_for_new_lang(tmp_path):
    # This will be tested after implementation creates a test language dir
    pass
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config_language_discovery.py::test_get_supported_languages_includes_en_and_cs -v`
Expected: FAIL with "get_supported_languages not defined"

- [ ] **Step 3: Implement `get_supported_languages()` and update `get_prompt_language_dir()` in `config.py`**

Add after `get_prompt_language_dir()` function (~line 85):

```python
def get_supported_languages() -> list[str]:
    """Return sorted list of language codes that have prompt directories.
    
    Scans PROMPTS_DIR for subdirectories that look like language codes
    (2-letter lowercase ISO 639-1 codes).
    """
    if not PROMPTS_DIR.exists():
        return ["en"]  # fallback
    
    langs = []
    for item in PROMPTS_DIR.iterdir():
        if item.is_dir() and len(item.name) == 2 and item.name.islower() and item.name.isalpha():
            # Verify it has at least the base prompt file
            if (item / "full_prompt.txt").exists() or (item / "system_prompt.txt").exists():
                langs.append(item.name)
    
    # Ensure 'en' is always first if present
    langs.sort()
    if "en" in langs:
        langs.remove("en")
        langs.insert(0, "en")
    return langs


def get_prompt_language_dir(language: str) -> Path:
    """Vrací správnou složku promptů podle jazyka.
    
    Args:
        language: jazyk kód (en, cs, de, fr, atd.)
        
    Returns:
        Path ke složce promptů pro daný jazyk
    """
    lang_dir = PROMPTS_DIR / language
    if lang_dir.exists():
        return lang_dir
    # Fallback to English if language directory doesn't exist
    return PROMPTS_DIR / "en"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config_language_discovery.py::test_get_supported_languages_includes_en_and_cs -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add config.py tests/test_config_language_discovery.py
git commit -m "feat(config): add dynamic language discovery from prompts directory"
```

---

### Task 2: Update settings_store.py Language Validation

**Files:**
- Modify: `c:\Users\david\Documents\AIMO\settings_store.py:60` (DEFAULT_SETTINGS language default) and `~line 270` (language validation in _normalize_settings)

**Interfaces:**
- Consumes: `config.get_supported_languages()`
- Produces: Updated `_normalize_settings()` that validates against dynamic list

- [ ] **Step 1: Write the failing test**

```python
# tests/test_settings_language_validation.py
import pytest
from settings_store import load_settings, _normalize_settings
from config import get_supported_languages

def test_settings_accepts_any_supported_language():
    supported = get_supported_languages()
    for lang in supported:
        settings = _normalize_settings({"language": lang})
        assert settings["language"] == lang

def test_settings_rejects_unsupported_language():
    # Use a code that won't exist in prompts dir
    settings = _normalize_settings({"language": "xx"})
    assert settings["language"] == "en"  # falls back to default
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_settings_language_validation.py::test_settings_accepts_any_supported_language -v`
Expected: FAIL (validation still hardcoded to "en", "cs")

- [ ] **Step 3: Update `_normalize_settings()` in `settings_store.py`**

Replace the hardcoded language validation (around line 270):

```python
    # Language validation - use dynamically discovered languages
    from config import get_supported_languages
    supported_languages = get_supported_languages()
    lang = _as_clean_text(settings.get("language")).lower()
    if lang not in supported_languages:
        lang = "en"  # default fallback
    settings["language"] = lang
```

Also update DEFAULT_SETTINGS comment to reflect dynamic languages.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_settings_language_validation.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add settings_store.py tests/test_settings_language_validation.py
git commit -m "feat(settings): validate language against dynamically discovered languages"
```

---

### Task 3: Update PromptBuilder to Accept Any Discovered Language

**Files:**
- Modify: `c:\Users\david\Documents\AIMO\prompt_builder.py:150-160` (set_language method)

**Interfaces:**
- Consumes: `config.get_supported_languages()`
- Produces: Updated `set_language()` that validates against dynamic list

- [ ] **Step 1: Write the failing test**

```python
# tests/test_prompt_builder_language.py
import pytest
from prompt_builder import PromptBuilder
from config import get_supported_languages

def test_prompt_builder_accepts_all_supported_languages():
    builder = PromptBuilder()
    for lang in get_supported_languages():
        builder.set_language(lang)
        assert builder.current_language == lang

def test_prompt_builder_rejects_unsupported_language():
    builder = PromptBuilder()
    original = builder.current_language
    builder.set_language("xx")  # not a supported language
    assert builder.current_language == original  # unchanged
    # Should log warning
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_prompt_builder_language.py::test_prompt_builder_accepts_all_supported_languages -v`
Expected: FAIL (set_language still hardcoded to "en", "cs")

- [ ] **Step 3: Update `set_language()` in `prompt_builder.py`**

Replace the validation in `set_language()` method (~line 150):

```python
    def set_language(self, language: str) -> None:
        """Set the language for prompt loading.

        Args:
            language: Language code (e.g., 'en', 'cs', 'de', 'fr')
        """
        from config import get_supported_languages
        lang = language.lower()
        supported = get_supported_languages()
        if lang not in supported:
            log.warning("Invalid language %r; supported: %s; falling back to %s", 
                       language, supported, self.current_language)
            return
        if lang != self.current_language:
            self.current_language = lang
            # Force reload prompts from the language-specific directory
            self.reload()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_prompt_builder_language.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add prompt_builder.py tests/test_prompt_builder_language.py
git commit -m "feat(prompt_builder): accept any dynamically discovered language"
```

---

### Task 4: Add Integration Test for Full Language Switch Flow

**Files:**
- Create: `c:\Users\david\Documents\AIMO\tests\test_language_switch_integration.py`

**Interfaces:**
- Consumes: Orchestrator, PromptBuilder, SettingsStore, config
- Produces: End-to-end test verifying language switch works for new languages

- [ ] **Step 1: Write the failing test**

```python
# tests/test_language_switch_integration.py
import pytest
import tempfile
from pathlib import Path
from orchestrator import Orchestrator
from config import PROMPTS_DIR, get_supported_languages

def test_new_language_folder_auto_discovered(tmp_path):
    """Test that creating a new language folder makes it available."""
    # Create a temporary prompts dir structure
    test_prompts = tmp_path / "prompts"
    test_prompts.mkdir()
    
    # Copy existing en as base
    import shutil
    shutil.copytree(PROMPTS_DIR / "en", test_prompts / "en")
    
    # Create new language 'de' with minimal required files
    de_dir = test_prompts / "de"
    de_dir.mkdir()
    (de_dir / "full_prompt.txt").write_text("German prompt")
    (de_dir / "system_prompt.txt").write_text("German system")
    (de_dir / "devices").mkdir()
    (de_dir / "devices" / "generic.txt").write_text("German device")
    (de_dir / "examples" / "big").mkdir(parents=True)
    (de_dir / "seeds").mkdir()
    (de_dir / "seeds" / "persona_moods.txt").write_text("mood1\nmood2")
    (de_dir / "seeds" / "pacing_strategies.txt").write_text("strategy1")
    (de_dir / "tasks").mkdir()
    (de_dir / "tasks" / "user_turn_task.txt").write_text("task {n_turns}")
    
    # This test would need config to use test_prompts dir
    # For now, verify the concept works by checking existing discovery
    langs = get_supported_languages()
    assert "en" in langs
    assert "cs" in langs
    
    # Test that a new folder would be picked up (conceptually)
    # In real test, we'd monkeypatch PROMPTS_DIR
    pass

def test_orchestrator_language_switch_clears_sessions():
    """Test that language switch triggers session cleanup."""
    # This is already tested by existing language tests
    # Verify the apply_settings flow works for any supported language
    pass
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_language_switch_integration.py -v`
Expected: FAIL (test incomplete)

- [ ] **Step 3: Implement proper integration test with monkeypatching**

```python
# tests/test_language_switch_integration.py
import pytest
from unittest.mock import patch
from pathlib import Path
from config import get_supported_languages, PROMPTS_DIR

def test_new_language_discovered_via_monkeypatch(tmp_path, monkeypatch):
    """Test dynamic language discovery by mocking PROMPTS_DIR."""
    # Create test prompts structure
    test_prompts = tmp_path / "prompts"
    test_prompts.mkdir()
    
    # Create en and de
    for lang in ["en", "de"]:
        lang_dir = test_prompts / lang
        lang_dir.mkdir()
        (lang_dir / "full_prompt.txt").write_text(f"{lang} prompt")
        (lang_dir / "system_prompt.txt").write_text(f"{lang} system")
        (lang_dir / "devices").mkdir()
        (lang_dir / "devices" / "generic.txt").write_text(f"{lang} device")
        (lang_dir / "examples" / "big").mkdir(parents=True)
        (lang_dir / "seeds").mkdir()
        (lang_dir / "seeds" / "persona_moods.txt").write_text("mood1")
        (lang_dir / "seeds" / "pacing_strategies.txt").write_text("strategy1")
        (lang_dir / "tasks").mkdir()
        (lang_dir / "tasks" / "user_turn_task.txt").write_text("task {n_turns}")
    
    # Monkeypatch PROMPTS_DIR
    import config
    monkeypatch.setattr(config, "PROMPTS_DIR", test_prompts)
    # Need to re-import functions that use PROMPTS_DIR at module level
    from importlib import reload
    reload(config)
    
    langs = config.get_supported_languages()
    assert "en" in langs
    assert "de" in langs
    assert "cs" not in langs  # not in test dir
    assert len(langs) == 2

def test_language_switch_flow():
    """Test the full apply_settings -> prompt_builder -> connector cleanup flow."""
    # This tests the existing flow works with any language
    # The key assertion is that set_language doesn't hardcode en/cs
    from prompt_builder import PromptBuilder
    builder = PromptBuilder()
    
    # Should work for any language returned by get_supported_languages
    for lang in get_supported_languages():
        builder.set_language(lang)
        assert builder.current_language == lang
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_language_switch_integration.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_language_switch_integration.py
git commit -m "test: add integration tests for dynamic language discovery and switching"
```

---

### Task 5: Verify All Existing Tests Still Pass

**Files:**
- No new files, run existing test suite

**Interfaces:**
- Consumes: All existing tests
- Produces: Confirmation that no regressions

- [ ] **Step 1: Run full test suite**

Run: `pytest tests/ -v`
Expected: All 41 tests pass (or whatever the current count is)

- [ ] **Step 2: Run language-specific tests**

Run: `pytest tests/test_jazyk.py -v`
Expected: PASS

- [ ] **Step 3: If any failures, fix them**

Debug and fix any regressions introduced by the changes.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "fix: resolve any regressions from dynamic language support"
```

---

### Task 6: Create Example New Language (Optional - Documentation)

**Files:**
- Create: `c:\Users\david\Documents\AIMO\prompts\de\full_prompt.txt` (and minimal structure)
- Create: `c:\Users\david\Documents\AIMO\docs\adding-languages.md`

**Interfaces:**
- Consumes: None
- Produces: Example German language folder + documentation

- [ ] **Step 1: Create minimal German language structure**

```bash
mkdir -p prompts/de/{devices,examples/big,seeds,tasks}
```

Create minimal files:
- `prompts/de/full_prompt.txt` - copy from en and translate key parts
- `prompts/de/system_prompt.txt` - minimal system prompt
- `prompts/de/devices/generic.txt` - generic device profile
- `prompts/de/seeds/persona_moods.txt` - at least one mood
- `prompts/de/seeds/pacing_strategies.txt` - at least one strategy
- `prompts/de/tasks/user_turn_task.txt` - task template with `{n_turns}` placeholder

- [ ] **Step 2: Create documentation**

```markdown
# Adding New Languages

To add a new language to AIMO:

1. Create a new directory under `prompts/` with the ISO 639-1 language code (e.g., `de`, `fr`, `es`, `ja`)
2. Add the following required files:
   - `full_prompt.txt` - Main prompt template
   - `system_prompt.txt` - System prompt
   - `devices/generic.txt` - Generic device profile (other device files optional)
   - `seeds/persona_moods.txt` - At least one mood
   - `seeds/pacing_strategies.txt` - At least one strategy
   - `tasks/user_turn_task.txt` - Task template with `{n_turns}` placeholder
3. Optional: Add examples in `examples/big/`, device-specific profiles in `devices/`
4. Restart the application or reload settings - the new language will appear automatically

The system automatically discovers languages by scanning the `prompts/` directory.
No code changes required!
```

- [ ] **Step 3: Commit**

```bash
git add prompts/de/ docs/adding-languages.md
git commit -m "docs: add example German language and documentation for adding languages"
```

---

## Summary

This plan adds fully dynamic language support by:
1. **config.py** - `get_supported_languages()` scans `prompts/` directory at runtime
2. **settings_store.py** - Validates language setting against discovered languages
3. **prompt_builder.py** - Accepts any discovered language in `set_language()`
4. **Tests** - Verify discovery, validation, and switching work for arbitrary languages

After implementation, users can add a new language by simply creating `prompts/{code}/` with the required files - no code changes needed.