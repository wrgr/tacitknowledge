import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path


APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


class AssessmentExportSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import database
        except ImportError as e:
            raise unittest.SkipTest(f"database.py not importable in this environment: {e}")
        cls.database = database

    def setUp(self):
        self._orig_db_file = self.database.DB_FILE
        self.database.DB_FILE = Path(tempfile.mktemp(suffix=".db"))

    def tearDown(self):
        self.database.DB_FILE = self._orig_db_file

    def _assessment_columns(self):
        with sqlite3.connect(str(self.database.DB_FILE)) as conn:
            return {row[1] for row in conn.execute("PRAGMA table_info(assessments)").fetchall()}

    def test_init_db_creates_all_canonical_assessment_fields(self):
        self.database.init_db()

        columns = self._assessment_columns()

        for field in self.database.ASSESSMENT_FIELDS:
            self.assertIn(field, columns)

    def test_init_db_migrates_existing_assessments_table_with_missing_export_fields(self):
        old_fields = [
            field for field in self.database.ASSESSMENT_FIELDS
            if field not in {"export_schema_version", "closing_nudge_used", "process_caution"}
        ]
        with sqlite3.connect(str(self.database.DB_FILE)) as conn:
            conn.execute(
                "CREATE TABLE assessments ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                + ", ".join(f"{field} TEXT NOT NULL DEFAULT ''" for field in old_fields)
                + ", UNIQUE(username, report_file, task_title))"
            )
            conn.commit()

        self.database.init_db()

        columns = self._assessment_columns()
        self.assertIn("closing_nudge_used", columns)
        self.assertIn("process_caution", columns)
        self.assertIn("export_schema_version", columns)

    def test_report_export_versions_groups_versions_by_report_file(self):
        self.database.init_db()
        rows = []
        for version, task in [("1", "Task A"), ("2", "Task B")]:
            row = {field: "" for field in self.database.ASSESSMENT_FIELDS}
            row.update({
                "username": "emma",
                "report_file": "fr_report_20260707_120000.md",
                "task_title": task,
                "report_type": "free_response",
                "export_schema_version": version,
            })
            rows.append(row)
        self.database.upsert_assessment_rows(rows)

        versions = self.database.assessment_report_export_versions()

        self.assertEqual(
            versions[("emma", "fr_report_20260707_120000.md")],
            {"1", "2"},
        )


class AssessmentExportDictionaryTests(unittest.TestCase):
    def test_new_process_fields_are_documented_for_research_export(self):
        doc = (APP_DIR / "docs" / "research_export_data_dictionary.md").read_text(encoding="utf-8")

        self.assertIn("`closing_nudge_used`", doc)
        self.assertIn("`process_caution`", doc)
        self.assertIn("`export_schema_version`", doc)

    def test_research_rows_populate_closing_nudge_and_process_caution(self):
        source = (APP_DIR / "app.py").read_text(encoding="utf-8")

        self.assertIn('_ASSESSMENT_EXPORT_SCHEMA_VERSION = "2"', source)
        self.assertIn('"export_schema_version": _ASSESSMENT_EXPORT_SCHEMA_VERSION', source)
        self.assertIn('"closing_nudge_used": (', source)
        self.assertIn('"process_caution": overlay.get("caution", "") if overlay else ""', source)
        self.assertIn("assessment_report_export_versions", source)


if __name__ == "__main__":
    unittest.main()
