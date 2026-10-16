"""Unit tests: typed messages and DOI normalisation (the basis of deduplication)."""

import unittest

from pydantic import ValidationError

from research_agent.models import Constraints, Paper, SubQuestion, normalise_doi


class TestNormaliseDoi(unittest.TestCase):
    def test_strips_url_prefixes_and_case(self):
        for raw in ["https://doi.org/10.1038/S42256-020-00287-7",
                    "http://dx.doi.org/10.1038/s42256-020-00287-7",
                    "doi:10.1038/s42256-020-00287-7", " 10.1038/S42256-020-00287-7. "]:
            self.assertEqual(normalise_doi(raw), "10.1038/s42256-020-00287-7", raw)

    def test_rejects_non_dois(self):
        for raw in [None, "", "not a doi", "11.1234/x", "10.12/too-short-prefix"]:
            self.assertIsNone(normalise_doi(raw), raw)

    def test_keeps_unusual_but_valid_suffix(self):
        doi = "10.1002/(sici)1097-4571(199806)49:8<693::aid-asi4>3.0.co;2-0"
        self.assertEqual(normalise_doi(doi), doi)


class TestPaper(unittest.TestCase):
    def test_doi_is_normalised_on_construction(self):
        p = Paper(paper_id="x", title="T", doi="https://doi.org/10.1/ABC")
        self.assertIsNone(p.doi)  # 10.1 has too short a registrant code
        p = Paper(paper_id="x", title="T", doi="https://doi.org/10.1234/ABC")
        self.assertEqual(p.doi, "10.1234/abc")

    def test_dedup_key_prefers_doi_then_title(self):
        a = Paper(paper_id="a", title="Agents: A Survey!", doi="10.1234/x")
        b = Paper(paper_id="b", title="Different", doi="10.1234/X")
        c = Paper(paper_id="c", title="agents a survey")
        d = Paper(paper_id="d", title="AGENTS - a survey")
        self.assertEqual(a.dedup_key, b.dedup_key)
        self.assertEqual(c.dedup_key, d.dedup_key)
        self.assertNotEqual(a.dedup_key, c.dedup_key)


class TestSubQuestionAndConstraints(unittest.TestCase):
    def test_query_quotes_are_stripped(self):
        s = SubQuestion(id=1, text="What is studied?", search_query='  "agent planning" ')
        self.assertEqual(s.search_query, "agent planning")

    def test_invalid_values_rejected(self):
        with self.assertRaises(ValidationError):
            SubQuestion(id=0, text="valid text", search_query="q q")
        with self.assertRaises(ValidationError):
            Constraints(max_papers_per_subquestion=0)
        with self.assertRaises(ValidationError):
            Constraints(year_from=1500)


if __name__ == "__main__":
    unittest.main()
