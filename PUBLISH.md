# Publish / update RAID-SQL on GitHub

Public repository: https://github.com/AungKaungMyat9604/RAID-SQL

## Safety check (secrets)

```bash
cd RAID-SQL
git check-ignore -v .env
git status
```

Confirm `.env`, `.venv/`, `.chroma/`, and credential JSON files stay **untracked**.

## Push updates

```bash
cd RAID-SQL
git add -A
git status
git commit -m "Your message"
git push origin main
```

Optional release tag:

```bash
git tag -a v1.0.0 -m "Locked Flash claim: 87.38% official EX"
git push origin v1.0.0
```

## Citation URL

https://github.com/AungKaungMyat9604/RAID-SQL
