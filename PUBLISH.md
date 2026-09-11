# Publish RAID-SQL to GitHub (do this yourself)

Replace `YOUR_USERNAME` everywhere after the repo exists.

## 1. Safety check (secrets)

```bash
cd RAID-SQL
# Must NOT list .env
git check-ignore -v .env

# Dry-run: nothing sensitive should appear
git status --ignored
```

Confirm these stay **untracked**:

- `.env`
- `.venv/`
- `.chroma/`
- `*credentials*.json` / service-account files

Confirm these **are** tracked:

- `raid_sql/`, `scripts/`, `config.py`, `requirements.txt`
- `.env.example`, `LICENSE`, `README.md`, `CITATION.cff`, `METHOD_FLOW.md`
- `outputs/README.md` and `outputs/test_raid_v2_values/` (locked claim)

## 2. Create the GitHub repo

On GitHub: **New repository** → name `RAID-SQL` → public → **do not** add README/licence (this folder already has them).

## 3. First push

```bash
cd RAID-SQL
git init
git add .
git status    # review carefully
git commit -m "$(cat <<'EOF'
Initial public release of RAID-SQL with locked Spider-test package.

EOF
)"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/RAID-SQL.git
git push -u origin main
```

Optional release tag for citations:

```bash
git tag -a v1.0.0 -m "Locked Flash claim: 87.38% official EX"
git push origin v1.0.0
```

## 4. After the URL is live

1. In this repo, replace `YOUR_USERNAME` in `README.md` and `CITATION.cff`.
2. In dissertation / presentation / ICAIT sources, replace the same placeholder (see `final/` docs).
3. Commit and push the username fix.

## 5. Suggested citation URL

`https://github.com/YOUR_USERNAME/RAID-SQL`

Tagged form (after step 3):

`https://github.com/YOUR_USERNAME/RAID-SQL/releases/tag/v1.0.0`
