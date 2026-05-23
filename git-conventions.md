# Git Conventions

## Branches

Main branch with working version.


Feature branches for integration:

```text
feature/<name>
research/<topic>
fix/<name>
docs/<name>
refactor/<name>
```

## PR Flow

- PR required for `main`.
- Rebase merge by default.
- Squash commit history when it makes the project history clearer.
- Delete merged branches.

## Commit Convention

Format:

```text
type: summary
```

Types:

- `feat`
- `research`
- `docs`
- `fix`
- `refactor`
- `release`

Examples:

```text
research: analyze data and simulation results
feat: add new evaluation module
docs: update README
fix: clean links
```