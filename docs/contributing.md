# Contributing

Welcome. If you want to learn and build a transformer in pure C, this is the right project.

We do not use big frameworks, libraries, or automatic code generation for the C implementation, so the contribution rules are strict.

## AI and LLM Rule

- **BANNED:** Do not use AI to write the C code. The point of the project is to understand the math and memory layout yourself. AI-generated C will be rejected.
- **ALLOWED:** You can use AI to research math formulas, fix typos, write documentation, or ask for better code practices, as long as you review the result carefully.

## Memory Discipline

Free every allocation you create. If you change a code path, check that the matching cleanup path still owns the memory it should own.

## Working Style

- Keep changes small and local.
- Prefer explicit math and predictable control flow.
- Preserve the existing style of the C codebase.
- Update tests when behavior changes.

If you open a pull request, include the relevant reasoning for the change and make sure the implementation still matches the derivations in the math docs.