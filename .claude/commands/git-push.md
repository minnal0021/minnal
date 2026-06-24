Stage, commit, and push the current changes.

If the user supplied a message use it verbatim (available as $ARGUMENTS).
Otherwise generate a concise commit message from the diff.

Steps:

1. Run `git status` and `git diff HEAD` in parallel to see what has changed.
   Also run `git log --oneline -5` to understand the project's commit-message style.

2. Decide the commit message:
   - If $ARGUMENTS is non-empty, use it as the commit subject line.
   - Otherwise write a short imperative summary (≤72 chars) that describes
     the "why" not the "what", matching the style seen in recent commits.

3. Stage only relevant files — prefer naming them explicitly rather than
   `git add -A`. Never stage files that look like secrets (.env, *.pem,
   credentials*, *secret*, *.key). Warn the user and stop if any such file
   would be included.

4. Confirm there is something to commit (`git diff --cached --stat`).
   If nothing is staged, report "Nothing to commit." and stop.

5. Create the commit using a HEREDOC so multi-line messages format correctly:
   ```
   git commit -m "$(cat <<'EOF'
   <subject line>

   Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
   EOF
   )"
   ```

6. Push to the current branch's tracking remote:
   `git push`
   If the branch has no upstream yet, use:
   `git push --set-upstream origin $(git branch --show-current)`

7. Report the short commit SHA and the remote URL the push targeted.
   If any step fails, explain the error and stop — do not retry destructive
   operations silently.
