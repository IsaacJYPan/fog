# Retrieve a saved Kaggle notebook

For new immutable experiments, use [fog-experiment collect](kaggle-workflow.md). The
`fog-kaggle` helper supports earlier notebook exports and manual imports.

Copy `configs/kaggle_results.example.json` to `configs/kaggle_results.json`, then set the notebook
owner/slug, saved version, exact `scriptVersionId`, and export filename. The account-local file
is ignored by Git. Run setup and login from the repository root:

```powershell
.\fog-kaggle.cmd setup
.\fog-kaggle.cmd login
.\fog-kaggle.cmd status
.\fog-kaggle.cmd pull
```

The helper uses the official Kaggle client. Login requests the CLI's resource-management scope;
review it in the browser. Credentials stay outside the repository. Existing `KAGGLE_API_TOKEN`
or supported legacy credentials also work. See the
[upstream authentication reference](https://github.com/Kaggle/kaggle-cli/blob/main/skills/references/auth.md).

The tested API serves latest outputs rather than reliably returning a requested historical
version. The downloader therefore checks the signed download URL's run ID against the configured
`scriptVersionId`. If a newer run replaces it, downloading fails closed; use the saved version's
Output tab and manual import. The status command reports the latest notebook session.

To select a different saved version, supply both identifiers:

```powershell
.\fog-kaggle.cmd pull --version 2 --run-id <scriptVersionId>
```

Only the named export archive is downloaded. Imports go to
`runs/kaggle/<owner>--<notebook>/v<version>-<hash>/`; original archives remain under
`runs/kaggle/downloads/`. Identical imports are reused; different archives get separate folders.
`runs/kaggle/latest-import.json` points to the latest `HANDOFF.md` and `RESULTS.json`, which list
checkpoints, sidecars, hashes, metrics, audits, and pipeline status. Importing never promotes a model.

For manual import after downloading from the Output tab:

```powershell
.\fog-kaggle.cmd import --archive C:\path\fog-candidates-training-output.zip --version 1
```

Manual version metadata is supplied by the caller; these legacy ZIPs do not independently prove
the Kaggle version. Original configs preserve Kaggle paths. Continuing locally requires a new
job with valid local paths and refreshed audits; optimizer sidecars alone do not establish an
exact cross-platform continuation.
