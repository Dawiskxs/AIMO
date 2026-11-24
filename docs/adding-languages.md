# Adding New Languages

To add a new language to AIMO:

1. Create a new directory under `prompts/` with the ISO 639-1 language code (e.g., `de`, `fr`, `es`, `ja`)
2. Add the following required files:
   - `full_prompt.txt` - Main prompt template
   - `system_prompt.txt` - System prompt (legacy runtime)
   - `devices/generic.txt` - Generic device profile (other device files optional)
   - `seeds/persona_moods.txt` - At least one mood
   - `seeds/pacing_strategies.txt` - At least one strategy
   - `tasks/user_turn_task.txt` - Task template with `{n_turns}` placeholder
3. Optional: Add examples in `examples/big/`, device-specific profiles in `devices/`
4. Restart the application or reload settings - the new language will appear automatically

The system automatically discovers languages by scanning the `prompts/` directory.
No code changes required!

## Required Directory Structure

```
prompts/
├── en/
├── cs/
├── de/              # <-- your new language
│   ├── full_prompt.txt
│   ├── system_prompt.txt
│   ├── devices/
│   │   └── generic.txt
│   ├── examples/
│   │   └── big/
│   ├── seeds/
│   │   ├── persona_moods.txt
│   │   └── pacing_strategies.txt
│   └── tasks/
│       └── user_turn_task.txt
```

## Language Code Requirements

- Must be a valid ISO 639-1 2-letter code (lowercase)
- Examples: `de` (German), `fr` (French), `es` (Spanish), `ja` (Japanese), `zh` (Chinese), `ru` (Russian)
- Directory name MUST match the language code exactly

## File Descriptions

### `full_prompt.txt` (Required)
The main prompt template for the new AI runtime. This is the most important file - it defines:
- Output format (JSON array with intent, intensity, duration_ms, speech)
- Available intents and their meanings
- Speech rules and delivery guidelines
- Device profile placeholder (`{{DEVICE_BLOCK}}`)
- Examples placeholder (`{{EXAMPLES_BLOCK}}`)
- Video block placeholder (`{{#VIDEO}}...{{/VIDEO}}`)

### `system_prompt.txt` (Required for legacy)
Legacy prompt format used by older runtime versions. Should contain similar content adapted for single JSON object output with pattern/speed/depth/base/intensity commands.

### `devices/generic.txt` (Required)
Generic device profile used as fallback when no device-specific profile exists. Defines:
- Contact & anatomy (abstract)
- What intensity means for this device
- Sensation vocabulary guidelines
- What "stopped" means

### `seeds/persona_moods.txt` (Required)
List of persona moods, one per line. These define the AI's dominant affect for a session.
Example:
```
Clinical and detached - cold, precise, almost scientific
Gleefully sadistic - enjoying every second
Impatient and aggressive - wants results now
```

### `seeds/pacing_strategies.txt` (Required)
List of pacing strategies, one per line. These define how intensity progresses through a session.
Example:
```
The Linear Build: Steady, incremental increases over 4-5 turns.
The Staircase: Three steps forward, one step back.
```

### `tasks/user_turn_task.txt` (Required)
Task template for user turns. Must contain `{n_turns}` placeholder.
Example:
```
You are given {n_turns} turns to continue the session. Maintain continuity with the previous context and keep the action coherent.
```

## Optional Files

### `examples/big/*.txt`
Example sessions showing expected output format and style. The system loads all `.txt` and `.json` files from this directory.

### `devices/{device_type}.txt`
Device-specific profiles (e.g., `ossm.txt`, `coyote.txt`, `buttplug.txt`, `none.txt`). These override the generic profile when that device is active.

## Verification

After creating the language folder:
1. Restart the application, OR
2. Go to Settings and reload, OR
3. Run: `python -c "from config import get_supported_languages; print(get_supported_languages())"`

Your new language should appear in the list.

## Testing a New Language

```bash
# Test that the language is discovered
python -c "from config import get_supported_languages; print('Supported:', get_supported_languages())"

# Test prompt builder
python -c "
from prompt_builder import PromptBuilder
b = PromptBuilder()
b.set_language('de')  # replace with your language code
print('Language set to:', b.current_language)
print('System prompt length:', len(b.get_system_prompt()))
"

# Test settings validation
python -c "
from settings_store import _normalize_settings
result = _normalize_settings({'language': 'de'})  # replace with your code
print('Settings language:', result['language'])
```