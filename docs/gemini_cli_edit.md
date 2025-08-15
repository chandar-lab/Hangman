I’ll locate the in-place edit tool in this repo, extract its full spec, and explain how it works internally with precise file paths and line numbers.

Checklist
- Identify all implementations/references of the in-place edit tool.
- Provide the most relevant tool’s complete spec (name, description, parameters, JSON schema, examples/comments).
- Explain internal behavior (what’s written, how diffs are produced, where implemented).
- Include file paths + line numbers for all relevant definitions.
- Document any aliases/variants.

Progress update
- Scanned core tools and docs; the in-place file edit tool is EditTool (tool name replace) in packages/core/src/tools/edit.ts. Collected spec, schema, and internals; also found related docs and helpers.

## The in-place edit tool

### Tool identity and aliases
- Internal class: EditTool
  - File: packages/core/src/tools/edit.ts (class definition at line 65)
- Tool name (LLM-visible): replace (static readonly Name)
  - Line: 69
- Display name: Edit (passed to BaseTool constructor)
  - Lines: 79–109 block includes description

Other related tool:
- WriteFile tool (for full file overwrite): WriteFileTool, tool name write_file
  - File: packages/core/src/tools/write-file.ts (class at line 45)
  - Included below under “Related/variants”.

### Full specification (from code)

Where: packages/core/src/tools/edit.ts

- Name and description (constructor)
  - Lines: 79–109
  - Description text (as implemented):

  Replaces text within a file. By default, replaces a single occurrence, but can replace multiple occurrences when `expected_replacements` is specified. This tool requires providing significant context around the change to ensure precise targeting. Always use the ReadFile tool to examine the file's current content before attempting a text replacement.

  Expectation for required parameters:
  1. `file_path` MUST be an absolute path; otherwise an error will be thrown.
  2. `old_string` MUST be the exact literal text to replace (including all whitespace, indentation, newlines, and surrounding code etc.).
  3. `new_string` MUST be the exact literal text to replace `old_string` with (also including all whitespace, indentation, newlines, and surrounding code etc.). Ensure the resulting code is correct and idiomatic.
  4. NEVER escape `old_string` or `new_string`, that would break the exact literal text requirement.
  Important: If ANY of the above are not satisfied, the tool will fail. CRITICAL for `old_string`: Must uniquely identify the single instance to change. Include at least 3 lines of context BEFORE and AFTER the target text, matching whitespace and indentation precisely. If this string matches multiple locations, or does not match exactly, the tool will fail.
  Multiple replacements: Set `expected_replacements` to the number of occurrences you want to replace. The tool will replace ALL occurrences that match `old_string` exactly. Ensure the number of replacements matches your expectation.

- JSON schema for parameters (constructor)
  - Lines: 92–113 and 139–143
  - Schema:
    - type: object
    - required: ['file_path', 'old_string', 'new_string']
    - properties:
      - file_path (string)
        - description: The absolute path to the file to modify. Must start with '/'.
        - Lines: 97–100
      - old_string (string)
        - description: The exact literal text to replace, preferably unescaped. For single replacements (default), include at least 3 lines of context BEFORE and AFTER the target text, matching whitespace and indentation precisely. For multiple replacements, specify expected_replacements parameter. If this string is not the exact literal text (i.e. you escaped it) or does not match exactly, the tool will fail.
        - Lines: 101–105
      - new_string (string)
        - description: The exact literal text to replace `old_string` with, preferably unescaped. Provide the EXACT text. Ensure the resulting code is correct and idiomatic.
        - Lines: 106–109
      - expected_replacements (number, minimum: 1)
        - description: Number of replacements expected. Defaults to 1 if not specified. Use when you want to replace multiple occurrences.
        - Lines: 110–113

- Parameter TypeScript interface (EditToolParams)
  - Lines: 18–38

- Pre-execution description string builder (getDescription)
  - Lines: 338–361

- Parameter validation (validateToolParams)
  - Checks JSON schema (if provided), absolute path, and path within root directory.
  - Lines: 146–164
  - Note: Path containment check uses this.rootDirectory set from Config.getTargetDir() (constructor at lines 116–118) and helper isWithinRoot at lines 120–133.

- Confirmation flow (shouldConfirmExecute)
  - Lines: 286–336
  - Behavior: If approval mode is AUTO_EDIT, no confirmation. Otherwise, validates, computes the proposed edit, and returns ToolEditConfirmationDetails with a unified diff created via Diff.createPatch (lines 316–324) and DEFAULT_DIFF_OPTIONS (diffOptions.ts lines 7–10: context=3, ignoreWhitespace=true).

- Execution (execute)
  - Lines: 365–436
  - Behavior:
    - Validate params.
    - Calculate the edit (lines 379–392).
    - If no error, ensure parent directories exist (line 396), then write the final new content to disk with fs.writeFileSync(file_path, newContent, 'utf8') (line 397).
    - For display, generate a unified diff (Diff.createPatch) when editing an existing file (lines 406–414).
    - Returns llmContent indicating success with occurrence count or creation.

- Modification support for user adjustments (ModifiableTool.getModifyContext)
  - Lines: 444–466
  - Provides a way for the UI to open a diff editor and let the user change the proposed new content before executing. It recomputes params by setting:
    - old_string: oldContent from the temp file
    - new_string: modifiedProposedContent
    - Lines: 462–466
  - This is used by the generic modifier in packages/core/src/tools/modifiable-tool.ts.

### How it works internally

At a high level, EditTool replaces literal text within a single file, writes the entire resulting file content, and produces a unified diff for display/confirmation.

- Occurrence counting and correction:
  - The tool first reads the current file content (calculateEdit) and calls ensureCorrectEdit to correct over-escaped inputs and to count occurrences reliably.
    - ensureCorrectEdit is implemented in packages/core/src/utils/editCorrector.ts, function defined at line 63.
    - It:
      - Unescapes likely over-escaped strings (unescapeStringForGeminiBug, lines 510–576).
      - Counts occurrences (countOccurrences, lines 592–607).
      - Optionally calls the Gemini model to correct old_string (correctOldStringMismatch, lines 115–173) and adjust new_string (correctNewString, lines 179–233; correctNewStringEscaping, lines 236–279).
      - Caches results (LruCache).
  - The calculated occurrences must match expected_replacements (default 1); otherwise, the tool returns an error and does not write.
    - Mismatch errors set in edit.ts lines 249–261.

- Replacement mechanics:
  - If creating a new file (old_string === '' and file doesn’t exist), the new content is exactly new_string.
  - Otherwise, new content is computed via _applyReplacement(currentContent, oldString, newString, isNewFile):
    - For existing files, it returns currentContent.replaceAll(oldString, newString).
    - Lines: 168–192, specific replaceAll at line 191.
  - Important: Because replaceAll replaces all occurrences, the occurrence validation ensures this equals expected_replacements before writing. If expected_replacements=1 and occurrences>1, it fails; if expected_replacements=N and occurrences=N, then all N will be replaced.

- Writing and diff:
  - Writes are full-file writes using fs.writeFileSync (line 397), not line-based patches. No unified diff is applied to the file; diffs are for display only.
  - Diff for confirmation and result display is created via Diff.createPatch (lines 316–324, 406–414), using DEFAULT_DIFF_OPTIONS (context=3, ignoreWhitespace=true) at packages/core/src/tools/diffOptions.ts lines 7–10.

- Safety:
  - Path must be absolute and within the configured root directory (validateToolParams, lines 146–164).
  - If old_string is empty and the file exists, it errors (cannot create over an existing file) — set during calculateEdit based on fileExists (lines 240–247).
  - If file not found and old_string isn’t empty, it errors (lines 222–230).
  - If occurrences=0 (after correction), it errors (lines 247–253).
  - If occurrences mismatch expected_replacements, it errors (lines 254–261).

- Confirmation flow:
  - Unless ApprovalMode.AUTO_EDIT, shouldConfirmExecute returns ToolEditConfirmationDetails with a preview diff (lines 316–336).
  - The UI uses this to prompt the user; on proceeding “always”, approval mode is set to AUTO_EDIT (line 331–334).

- User modify-then-execute path (optional):
  - Using ModifiableTool and modifyWithEditor (packages/core/src/tools/modifiable-tool.ts lines 77–139), the CLI can open a diff in the user’s editor, let them adjust the proposed replacement, and then re-run with updated params via createUpdatedParams from EditTool (lines 462–466).

### What gets applied vs. what’s displayed
- Applied to disk: Full file content replacement via fs.writeFileSync of newContent computed by replaceAll (line 397).
- Displayed to user: A unified diff via Diff.createPatch (lines 316–324, 406–414). It is not used to patch the file; it’s only visual.

### External documentation (repo docs)
- docs/tools/file-system.md section “6. replace (Edit)”
  - File: docs/tools/file-system.md
  - Lines: 126–222 (primary block)
  - Includes:
    - Tool name: replace
    - Display name: Edit
    - File: edit.ts
    - Parameters with notes, behavior, failure conditions, confirmation behavior
    - Explicit “Enhanced Reliability (Multi-Stage Edit Correction)” explanation (matches editCorrector.ts behavior)

## Definitions and code locations (with line numbers)

Primary tool
- EditTool class
  - File: packages/core/src/tools/edit.ts
  - Class definition: line 65
  - static Name = 'replace': line 69
  - Constructor (name, displayName, description, schema): lines 79–143
  - validateToolParams: lines 146–164
  - _applyReplacement: lines 168–192
  - calculateEdit (core logic including ensureCorrectEdit, errors, newContent): lines 194–273
  - shouldConfirmExecute (diff preview): lines 286–336
  - getDescription: lines 338–361
  - execute (validate, calculate, write file, diff result): lines 365–436
  - ensureParentDirectoriesExist: lines 437–442
  - getModifyContext (user-edit support): lines 444–466

Helpers
- Diff options
  - File: packages/core/src/tools/diffOptions.ts
  - DEFAULT_DIFF_OPTIONS: lines 7–10

- Edit corrector (LLM-assisted correction and escaping fixes)
  - File: packages/core/src/utils/editCorrector.ts
  - ensureCorrectEdit(): defined at line 63
  - correctOldStringMismatch(): lines 115–173
  - correctNewString(): lines 179–233
  - correctNewStringEscaping(): lines 236–279
  - correctStringEscaping(): lines 296–354
  - unescapeStringForGeminiBug(): lines 510–576
  - countOccurrences(): lines 592–607

- Modifiable tool infra (for user modification of proposed changes)
  - File: packages/core/src/tools/modifiable-tool.ts
  - Interfaces and modifyWithEditor() pipeline: lines 12–139

Exports and references
- Export of EditTool
  - File: packages/core/src/index.ts
  - Line: 36 (export * from './tools/edit.js';)

- CLI references (uses EditTool.Name in non-interactive excludes and import)
  - File: packages/cli/src/gemini.tsx
  - Import includes EditTool: line 30
  - Use in non-interactive exclusion: line 248

Docs
- File: docs/tools/file-system.md
  - Section “6. replace (Edit)”: lines 126–222

## Related/variants

- WriteFile tool (WriteFileTool; tool name write_file)
  - File: packages/core/src/tools/write-file.ts
  - Purpose: Write/overwrite entire file content (create or overwrite), showing diff for confirmation. Not a targeted in-place substring replacement; it writes the full new content.
  - Class definition: line 45
  - Parameter schema: lines 67–84
  - shouldConfirmExecute (diff preview): lines 113–154
  - execute (writes full content): lines 156–255

No other in-repo edit aliases (like “apply_patch” or “update_file”) exist. The unified diff is produced solely for display via Diff.createPatch; it’s not a separate “apply patch” tool.

## Summary of behavior
- Mode: literal text replacement inside a single file; full-file write.
- Matching: old_string must exactly match and occur exactly expected_replacements times; otherwise error (with correction heuristics first).
- Write method: fs.writeFileSync(newContent), not line-based edits or patch application.
- Diff: unified diff (context=3, ignoreWhitespace=true) for confirmation/UI.

Requirements coverage
- Find tool(s) and references: Done (EditTool/replace; references and docs listed).
- Full spec for most relevant: Done (name, description, schema, parameters, behavior).
- Internal workings explained with code pointers: Done (replaceAll + full file write, diff generation).
- File paths + line numbers: Done.
- Multiple versions/aliases: Documented replace/Edit; related write_file noted; no other aliases found.
