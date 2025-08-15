// Edit text files. Do not use this tool to edit Jupyter notebooks. `apply_patch` allows you to execute a diff/patch against a text file, but the format of the diff specification is unique to this task, so pay careful attention to these instructions. To use the `apply_patch` command, you should pass a message of the following structure as "input":
//
// *** Begin Patch
// [YOUR_PATCH]
// *** End Patch
//
// Where [YOUR_PATCH] is the actual content of your patch, specified in the following V4A diff format.
//
// *** [ACTION] File: [/absolute/path/to/file] -> ACTION can be one of Add, Update, or Delete.
// An example of a message that you might pass as "input" to this function, in order to apply a patch, is shown below.
//
// *** Begin Patch
// *** Update File: /Users/someone/pygorithm/searching/binary_search.py
// @@class BaseClass
// @@    def search():
// -        pass
// +        raise NotImplementedError()
//
// @@class Subclass
// @@    def search():
// -        pass
// +        raise NotImplementedError()
//
// *** End Patch
// Do not use line numbers in this diff format.

type apply_patch = (_: {
// A short description of what the tool call is aiming to achieve.
explanation: string,
// The edit patch to apply.
input: string,
}) => any;


// Insert new code into an existing file in the workspace. Use this tool once per file that needs to be modified, even if there are multiple changes for a file. Generate the "explanation" property first.
// The system is very smart and can understand how to apply your edits to the files, you just need to provide minimal hints.
// Avoid repeating existing code, instead use comments to represent regions of unchanged code. Be as concise as possible. For example:
// // ...existing code...
// { changed code }
// // ...existing code...
// { changed code }
// // ...existing code...
//
// Here is an example of how you should use format an edit to an existing Person class:
// class Person {
// // ...existing code...
// age: number;
// // ...existing code...
// getAge() {
// return this.age;
// }
// }

type insert_edit_into_file = (_: {
// The code change to apply to the file.
// The system is very smart and can understand how to apply your edits to the files, you just need to provide minimal hints.
// Avoid repeating existing code, instead use comments to represent regions of unchanged code. Be as concise as possible. For example:
// // ...existing code...
// { changed code }
// // ...existing code...
// { changed code }
// // ...existing code...
//
// Here is an example of how you should use format an edit to an existing Person class:
// class Person {
// // ...existing code...
// age: number;
// // ...existing code...
// getAge() {
// return this.age;
// }
// }
code: string,
// A short explanation of the edit being made.
explanation: string,
// An absolute path to the file to edit.
filePath: string,
}) => any;


// This is a tool for editing an existing Notebook file in the workspace. Generate the "explanation" property first.
// The system is very smart and can understand how to apply your edits to the notebooks.
// When updating the content of an existing cell, ensure newCode includes at least 3-5 lines of context both before and after the new changes, preserving whitespace and indentation exactly.

type edit_notebook_file = (_: {
// Id of the cell that needs to be deleted or edited. Use the value `TOP`, `BOTTOM` when inserting a cell at the top or bottom of the notebook, else provide the id of the cell after which a new cell is to be inserted. Remember, if a cellId is provided and editType=insert, then a cell will be inserted after the cell with the provided cellId.
cellId?: string,
// The operation peformed on the cell, whether `insert`, `delete` or `edit`.
// Use the `editType` field to specify the operation: `insert` to add a new cell, `edit` to modify an existing cell's content, and `delete` to remove a cell.
editType: "insert" | "delete" | "edit",
// A one-sentence description of edit operation. This will be shown to the user before the tool is run.
explanation: string,
// An absolute path to the notebook file to edit, or the URI of a untitled, not yet named, file, such as `untitled:Untitled-1.
filePath: string,
// The language of the cell. `markdown`, `python`, `javascript`, `julia`, etc.
language?: string,
newCode?:
// The code for the new or existing cell to be edited. Code should not be wrapped within <VSCode.Cell> tags
 | string
 | Array<
// The code for the new or existing cell to be edited. Code should not be wrapped within <VSCode.Cell> tags
string
>
,
}) => any;
