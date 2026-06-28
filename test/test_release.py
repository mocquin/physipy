"""Unit tests for the pure logic in ``scripts/release.py``.

Only the side-effect-free helpers are exercised here (version parsing/bumping,
the tag<->version guard, the changelog roll). The subprocess-driven release
orchestration is not invoked.
"""

import importlib.util
import pathlib
import unittest

# Load scripts/release.py without making scripts/ an importable package.
_RELEASE_PATH = (
    pathlib.Path(__file__).resolve().parent.parent / "scripts" / "release.py"
)
_spec = importlib.util.spec_from_file_location("release", _RELEASE_PATH)
release = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(release)


class TestReadVersion(unittest.TestCase):
    def test_reads_double_quotes(self):
        self.assertEqual(
            release.read_version('__version__ = "0.2.9"'), "0.2.9"
        )

    def test_reads_single_quotes(self):
        self.assertEqual(
            release.read_version("__version__ = '1.0.0'"), "1.0.0"
        )

    def test_reads_amongst_other_lines(self):
        text = '# header\n# -*- coding: utf-8 -*-\n\n__version__ = "0.2.8"\n'
        self.assertEqual(release.read_version(text), "0.2.8")

    def test_missing_raises(self):
        with self.assertRaises(ValueError):
            release.read_version("nope = 1")


class TestParseVersion(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(release.parse_version("0.3.1"), (0, 3, 1))

    def test_invalid(self):
        for bad in ("0.3", "abc", "1.2.3.4", "v1.2.3", "1.2.x"):
            with self.assertRaises(ValueError):
                release.parse_version(bad)


class TestBumpVersion(unittest.TestCase):
    def test_patch(self):
        self.assertEqual(release.bump_version("0.2.9", "patch"), "0.2.10")

    def test_minor_resets_patch(self):
        self.assertEqual(release.bump_version("0.2.9", "minor"), "0.3.0")

    def test_major_resets_minor_patch(self):
        self.assertEqual(release.bump_version("0.2.9", "major"), "1.0.0")

    def test_explicit_greater_ok(self):
        self.assertEqual(release.bump_version("0.2.9", "0.3.1"), "0.3.1")

    def test_explicit_not_greater_raises(self):
        with self.assertRaises(ValueError):
            release.bump_version("0.2.9", "0.2.9")
        with self.assertRaises(ValueError):
            release.bump_version("0.2.9", "0.2.8")

    def test_explicit_invalid_raises(self):
        with self.assertRaises(ValueError):
            release.bump_version("0.2.9", "0.3")


class TestReplaceVersion(unittest.TestCase):
    def test_replaces_and_preserves(self):
        text = '# c\n# -*- coding: utf-8 -*-\n\n__version__ = "0.2.8"\n'
        out = release.replace_version(text, "0.2.9")
        self.assertIn('__version__ = "0.2.9"', out)
        self.assertNotIn("0.2.8", out)
        self.assertTrue(out.startswith("# c\n"))
        self.assertEqual(release.read_version(out), "0.2.9")

    def test_missing_raises(self):
        with self.assertRaises(ValueError):
            release.replace_version("nothing here", "1.0.0")


class TestTagHelpers(unittest.TestCase):
    def test_tag_for_is_bare_version(self):
        # physipy tags are bare versions (no 'v' prefix)
        self.assertEqual(release.tag_for("0.3.1"), "0.3.1")

    def test_version_from_tag(self):
        self.assertEqual(release.version_from_tag("0.3.1"), "0.3.1")
        self.assertEqual(
            release.version_from_tag("v0.3.1"), "0.3.1"
        )  # tolerant

    def test_guard_matches(self):
        release.assert_tag_matches_version("0.3.1", "0.3.1")  # no raise

    def test_guard_mismatch_raises(self):
        with self.assertRaises(ValueError):
            release.assert_tag_matches_version("0.3.0", "0.3.1")
        with self.assertRaises(ValueError):
            release.assert_tag_matches_version("v0.3.1", "0.3.1")  # stray 'v'


_CHANGELOG = """# Changelog

## [Unreleased]

### Added
- A shiny new feature.

## [0.2.9]

- Baseline.
"""

_CHANGELOG_EMPTY = """# Changelog

## [Unreleased]

<!-- placeholder comment -->

## [0.2.9]

- Baseline.
"""


class TestRollChangelog(unittest.TestCase):
    def test_moves_entries_under_versioned_heading(self):
        out = release.roll_changelog(_CHANGELOG, "0.3.0", "2026-06-28")
        self.assertIn("## [0.3.0] - 2026-06-28", out)
        self.assertIn("- A shiny new feature.", out)
        # the moved entry sits under the new version, not Unreleased
        unreleased_idx = out.index("## [Unreleased]")
        version_idx = out.index("## [0.3.0]")
        feature_idx = out.index("- A shiny new feature.")
        self.assertLess(unreleased_idx, version_idx)
        self.assertLess(version_idx, feature_idx)
        # a fresh, empty Unreleased remains and the old baseline is intact
        self.assertIn("## [Unreleased]", out)
        self.assertIn("## [0.2.9]", out)

    def test_empty_unreleased_still_creates_section(self):
        out = release.roll_changelog(_CHANGELOG_EMPTY, "0.3.0", "2026-06-28")
        self.assertIn("## [0.3.0] - 2026-06-28", out)

    def test_placeholder_comment_not_copied_into_dated_section(self):
        text = (
            "# Changelog\n\n## [Unreleased]\n\n"
            f"{release._UNRELEASED_PLACEHOLDER}\n\n"
            "### Added\n- A feature.\n\n## [0.2.9]\n\n- Baseline.\n"
        )
        out = release.roll_changelog(text, "0.3.0", "2026-06-28")
        # split into the fresh Unreleased part and the dated section
        dated = out[out.index("## [0.3.0]") : out.index("## [0.2.9]")]
        self.assertIn("- A feature.", dated)
        self.assertNotIn("Add notes here as you work", dated)
        # the placeholder stays in the fresh Unreleased
        fresh = out[out.index("## [Unreleased]") : out.index("## [0.3.0]")]
        self.assertIn("Add notes here as you work", fresh)
        # a blank line separates the dated section from the next heading
        self.assertIn("- A feature.\n\n## [0.2.9]", out)

    def test_missing_unreleased_raises(self):
        with self.assertRaises(ValueError):
            release.roll_changelog("# Changelog\n\n## [0.2.9]\n", "1.0.0", "x")


class TestChangelogHasEntries(unittest.TestCase):
    def test_with_entries(self):
        self.assertTrue(release.changelog_has_entries(_CHANGELOG))

    def test_only_comment_is_empty(self):
        self.assertFalse(release.changelog_has_entries(_CHANGELOG_EMPTY))


if __name__ == "__main__":
    unittest.main(verbosity=2)
