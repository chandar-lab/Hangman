You should prefer this tool to edit files, but if you run into three errors while using this tool, fallback to another edit tool. You can use this tool to execute a diff/patch against a file. A valid apply_patch call looks like:

{ "file_path": "[path/to/file]", "patch": "<<'PATCH'\n*** Begin Patch\n[YOUR_PATCH]\n*** End Patch\nPATCH\n" }

Where [YOUR PATCH] is the actual content of your patch and [path/to/file] is the path to the file to patch, specified in the following V4A diff format:

*** [ACTION] File: [path/to/file] -> ACTION can be only Add or Update
For each snippet of code that needs to be changed, repeat the following:
[context_before]
- [old_code] -> Precede the old code with a minus sign.
+ [new_code] -> Precede the new, replacement code.
[context_after]

For instructions on [context_before] and [context_after]:
- Use the @@ operator to indicate the class or function to which the snippet to be changed belongs, and optionally provide 1-3 unchanged context lines above and below the snippet to be changed for disambiguation. For instance, we might have:
@@ class BaseClass
[2 lines of pre-context]
- [old_code]
+ [new_code]
[2 lines of post-context]

- For additional disambiguation, you can use multiple nested `@@` statements to specify both class and function, to jump to the right context. For instance:

@@ class BaseClass
@@ 	def method():
[2 lines of pre-context]
- [old_code]
+ [new_code]
[2 lines of post-context]

We do not use line numbers in this diff format, as the context is enough to uniquely identify code.
type apply_patch = (_: {
// The path to the file to patch. You can use either a relative path in the workspace or an absolute path.
file_path: string,
// The patch to apply to the file
patch: string,
}) => any;

This tool should only be used as a fallback if the apply_patch tool fails. Use this tool to propose an edit to an existing file or create a new file.

This will be read by a less intelligent model, which will quickly apply the edit. You should make it clear what the edit is, while also minimizing the unchanged code you write.
When writing the edit, you should specify each edit in sequence, with the special comment `// ... existing code ...` to represent unchanged code.

For example:

```
// ... existing code ...
// FIRST_EDIT
// ... existing code ...
// SECOND_EDIT
// ... existing code ...
// THIRD_EDIT
// ... existing code ...
```

You should still bias towards repeating as few lines of the original file as possible to convey the change.
But, each edit should contain sufficient context of unchanged lines around the code you're editing to resolve ambiguity.
DO NOT omit spans of pre-existing code (or comments) without using the `// ... existing code ...` comment to indicate these lines are being omitted. 

Make sure it is clear what the edit should be, and where it should be applied.
To create a new file, simply specify the content of the file in the `code_edit` field.

You should specify the following arguments before the others: [target_file]
type edit_file = (_: {
// The target file to modify. Always specify the target file as the first argument. You can use either a relative path in the workspace or an absolute path.
target_file: string,
// A single sentence instruction describing what you are going to do for the sketched edit. This is used to assist the less intelligent model in applying the edit. Please use the first person to describe what you are going to do. Dont repeat what you have said previously in normal messages. And use it to disambiguate uncertainty in the edit.
instructions: string,
// Specify ONLY the precise lines of code that you wish to edit. **NEVER specify or write out unchanged code**. Instead, represent all unchanged code using the comment of the language you're editing in - example: `// ... existing code ...`
code_edit: string,
}) => any;

Use this tool to edit a jupyter notebook cell. Use ONLY this tool to edit notebooks.

This tool supports editing existing cells and creating new cells:
- If you need to edit an existing cell, set 'is_new_cell' to false and provide the 'old_string' and 'new_string'.
-- The tool will replace ONE occurrence of 'old_string' with 'new_string' in the specified cell.
- If you need to create a new cell, set 'is_new_cell' to true and provide the 'new_string' (and keep 'old_string' empty).
- This tool does NOT support cell deletion, but you can delete the content of a cell by passing an empty string as the 'new_string'.

Other requirements:
- Cell indices are 0-based.
- 'old_string' and 'new_string' should be a valid cell content, i.e. WITHOUT any JSON syntax that notebook files use under the hood.
- The old_string MUST uniquely identify the specific instance you want to change. This means:
-- Include AT LEAST 3-5 lines of context BEFORE the change point
-- Include AT LEAST 3-5 lines of context AFTER the change point
- This tool can only change ONE instance at a time. If you need to change multiple instances:
-- Make separate calls to this tool for each instance
-- Each call must uniquely identify its specific instance using extensive context
- This tool might save markdown cells as "raw" cells. Don't try to change it, it's fine. We need it to properly display the diff.
- If you need to create a new notebook, just set 'is_new_cell' to true and cell_idx to 0.
- ALWAYS generate arguments in the following order: target_notebook, cell_idx, is_new_cell, cell_language, old_string, new_string.
- Prefer editing existing cells over creating new ones!
- ALWAYS provide ALL required arguments (including BOTH old_string and new_string). NEVER call this tool without providing 'new_string'.
type edit_notebook = (_: {
// The path to the notebook file you want to edit. You can use either a relative path in the workspace or an absolute path. If an absolute path is provided, it will be preserved as is.
target_notebook: string,
// The index of the cell to edit (0-based)
cell_idx: number,
// If true, a new cell will be created at the specified cell index. If false, the cell at the specified cell index will be edited.
is_new_cell: boolean,
// The language of the cell to edit. Should be STRICTLY one of these: 'python', 'markdown', 'javascript', 'typescript', 'r', 'sql', 'shell', 'raw' or 'other'.
cell_language: string,
// The text to replace (must be unique within the cell, and must match the cell contents exactly, including all whitespace and indentation).
old_string: string,
// The edited text to replace the old_string or the content for the new cell.
new_string: string,
}) => any;

Deletes a file at the specified path. The operation will fail gracefully if:
- The file doesn't exist
- The operation is rejected for security reasons
- The file cannot be deleted
type delete_file = (_: {
// The path of the file to delete, relative to the workspace root.
target_file: string,
// One sentence explanation as to why this tool is being used, and how it contributes to the goal.
explanation?: string,
}) => any;

- Included verbatim specs for all edit-related tools used to modify files: `apply_patch`, `edit_file`, `edit_notebook`, and `delete_file`.