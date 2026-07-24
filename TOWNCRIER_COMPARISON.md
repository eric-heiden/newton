# Changelog workflow comparison

This spike is based on the same `main` commit as pull request #3609 so the two
implementations can be compared directly.

## Summary

| Concern | Current custom implementation | Towncrier spike |
|---|---|---|
| PR format | One `.md` file can hold every category and entry | One file per entry and category; files share one identifier |
| Naming | Readable slug plus random suffix | GitHub issue number, or `+` readable slug plus random suffix |
| PR creation before a number exists | Always supported | Use the issue number when known; otherwise use an orphan identifier |
| Development preview | `scripts/changelog.py build --dry-run` | `towncrier build --draft` |
| Development consolidation | Supported; writes `[Unreleased]` and provenance comments | Deliberately omitted; preview is non-mutating |
| Release build | Custom `release` command | Native `towncrier build` |
| Synchronization to `main` | Custom `reconcile` command | Cherry-pick the release build commit |
| `.skip` and PR policy | Custom script | Small Newton policy script around Towncrier |
| Project dependency | None | None; Towncrier 25.8.0 is pinned only in `uvx` and pre-commit |
| First release | Fully automated transition | One-time merge of duplicate category headings during the release audit |

## Fragment examples

The custom implementation keeps a whole pull request in one file:

```text
changelog.d/camera-rays-a1b2c3d4.md
```

```markdown
### Added

- Add pinhole camera rays.
- Add fisheye camera rays.

### Deprecated

- Deprecate `old_ray()` in favor of `camera_ray()`.
```

Towncrier represents those entries as separate files:

```text
changelog.d/3607.added.md
changelog.d/3607.added.1.md
changelog.d/3607.deprecated.md
```

Each file contains only its entry text, without a heading or bullet. Using the
same identifier lets CI enforce one logical fragment set per pull request.

## What Towncrier owns

Towncrier supplies the parsing, category ordering, Markdown rendering, issue
links, orphan handling, dated release insertion, draft mode, and deletion of
rendered fragments. The spike does not reimplement those behaviors.

Newton still needs repository policy that Towncrier does not provide:

- exactly one logical identifier for a normal pull request to `main`;
- multiple identifiers for a release backport pull request;
- one-line `.skip` reasons;
- no direct `CHANGELOG.md` edits in normal pull requests;
- changelog-only scope for the `release-management` label;
- Newton's filename, punctuation, and content rules.

That policy is isolated in `scripts/changelog_policy.py`.

## Release-branch result

The integration test creates diverged `main` and `release-X.Y` branches:

1. A fragment exists at the branch point.
2. `main` receives a main-only fragment and a later backported fragment.
3. The later change is cherry-picked to the release branch.
4. Towncrier builds the release and deletes the two shipped fragments.
5. The exact build commit is cherry-picked back to `main`.

The cherry-pick removes only the two fragments present in the release and keeps
the main-only fragment. This replaces the custom reconciliation algorithm with
normal Git behavior.

## Maintenance surface

At the time of this spike:

- custom engine: 883 lines;
- custom engine tests: 523 lines;
- Towncrier policy adapter: 286 lines;
- Towncrier policy and integration tests: 336 lines.

The totals exclude shared documentation, CI, and skill edits. Those are needed
for either workflow. The Towncrier route owns less than half as much Python and,
more importantly, does not own changelog parsing, rendering, release insertion,
fragment deletion, or branch reconciliation.

## Tradeoffs and recommendation

The strongest argument for the custom implementation is the single file per
pull request. It is less cluttered and makes multi-category changes pleasant to
edit. It also automates the first release transition and permits mutating
consolidation during development.

The strongest argument for Towncrier is a smaller long-term maintenance surface.
Its fragment-per-entry format is more verbose in the tree, but the files are
short-lived, easy for agents to generate, and naturally survive release-branch
cherry-picks. Avoiding development consolidation also removes ambiguity about
which release owns a change.

Recommendation: use Towncrier unless one-file-per-pull-request is a hard
requirement. Accept the extra fragment files, keep mutation release-only, and
perform the documented one-time category cleanup on the first Towncrier
release. This preserves the workflow advantages discussed on #3609 while
removing the highest-risk custom code: rendering and reconciliation.
