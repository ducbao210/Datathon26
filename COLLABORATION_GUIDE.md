# Git Collaboration Guide

This guide outlines the recommended Git workflow for contributing to our project, taking into account the following repository rules:

* **Require linear history**: Every merge into the `main` branch must result in a straight line of commits. Merge commits are disallowed; use **Rebase** to keep the history clean.
* **Require a pull request before merging**: Direct pushes to protected branches (e.g., `main`) are disallowed. All changes must be submitted via a Pull Request (PR) to integrate into the main codebase.
* **Block force pushes**: `git push --force` is strictly prohibited. This rule safeguards the project's commit history, preventing accidental overwrites and ensuring a stable, shared development timeline for all team members.

Following these guidelines will ensure a smooth, secure, and collaborative development process.

## Contributing Guide

### 1. Clone Repository

First, clone the project repository to your local machine:

```bash
git clone <your-repository-url>
cd <your-repository-name>
```

### 2. Pull Latest Code

Before starting any new work, ensure your local `main` branch is up-to-date with the remote repository:

```bash
git checkout main
git pull origin main
```

### 3. Create a New Branch

Create a new feature branch for your work. Use a descriptive name that reflects the purpose of your changes (e.g., `feature/add-user-authentication`, `bugfix/fix-login-issue`, `docs/update-readme`).

```bash
git checkout -b feature/your-feature-name # Example: git checkout -b feature/add-login-page
```

### 4. Commit and Sync (Linear History)

Make your changes within your feature branch. Stage and commit them as usual. 

**Crucial:** Because we require a **linear history**, if the remote `main` branch has moved forward while you were working, you must **rebase** your branch instead of merging.

```bash
# Stage and commit
git add .
git commit -m "feat: Implement new feature X"

# To maintain linear history before pushing:
git fetch origin
git rebase origin/main
```
*Note: If there are conflicts during rebase, resolve them, then run `git rebase --continue`.*

### 5. Push and Create Pull Request

After ensuring your history is linear, push your branch to the remote repository and create a PR via the web interface.

```bash
git push origin feature/your-feature-name
```

**Example Pull Request Flow:**

`feature/your-feature-name` (Rebased) → `main`


### 6. Post-Merge Cleanup

Once your Pull Request has been successfully merged into the `main` branch, it is good practice to clean up your local and remote environments.

1.  **Update your local `main` branch:**
    ```bash
    git checkout main
    git pull origin main
    ```
2.  **Delete your local feature branch:**
    ```bash
    git branch -d feature/your-feature-name
    ```
3.  **Delete the remote feature branch:**
    ```bash
    git push origin --delete feature/your-feature-name
    ```

## Summary of Key Commands

| Command                                   | Description                                                                 |
| :---------------------------------------- | :-------------------------------------------------------------------------- |
| `git clone <url>`                         | Clone a repository into a new directory.                                    |
| `git pull origin main`                    | Fetch and integrate changes from the remote `main` branch.                  |
| `git checkout -b <branch-name>`           | Create and switch to a new branch.                                          |
| `git commit -m "<message>"`               | Record staged changes with a descriptive message.                           |
| `git rebase main`                         | Re-apply your commits on top of the latest main (for Linear History).       |
| `git push origin <branch-name>`           | Push changes to the remote repository.                                      |
| `git branch -d <branch-name>`             | Delete a local branch.                                                      |
| `git push origin --delete <branch-name>`  | Delete a remote branch.                                                     |

By following this workflow, collaborators can effectively contribute to the project while respecting the defined repository rules. If you have any questions, please consult with team lead developers.