"""Unit tests: scholarly API parsing, fallback, retry and Crossref verification."""

import unittest
from unittest import mock

import requests

from research_agent.models import Constraints, Paper
from research_agent.sources import (CrossrefValidator, FallbackSearch, OpenAlexSearch,
                                    SemanticScholarSearch, SourceUnavailable, title_similarity)
from tests.helpers import FakeResponse, FakeSession, load_fixture


class TestSemanticScholar(unittest.TestCase):
    def test_parses_records_and_skips_malformed(self):
        session = FakeSession([FakeResponse(200, load_fixture("s2_search.json"))])
        papers = SemanticScholarSearch(session=session).search(
            "q", Constraints(year_from=2020, max_papers_per_subquestion=5))
        self.assertEqual([p.paper_id for p in papers], ["S2:abc123", "S2:noabs1"])
        self.assertEqual(papers[0].doi, "10.1038/s42256-020-00287-7")
        params = session.requests[0]["params"]
        self.assertEqual(params["year"], "2020-")
        self.assertEqual(params["limit"], 5)

    def test_api_key_header_only_when_set(self):
        self.assertEqual(SemanticScholarSearch().headers, {})
        self.assertEqual(SemanticScholarSearch("k").headers, {"x-api-key": "k"})

    @mock.patch("research_agent.sources.time.sleep")
    def test_retries_429_then_raises_unavailable(self, _sleep):
        session = FakeSession([FakeResponse(429)] * 4)
        with self.assertRaises(SourceUnavailable):
            SemanticScholarSearch(session=session).search("q", Constraints())
        self.assertEqual(len(session.requests), 4)

    def test_client_error_is_not_retried(self):
        session = FakeSession([FakeResponse(400)])
        with self.assertRaises(SourceUnavailable):
            SemanticScholarSearch(session=session).search("q", Constraints())
        self.assertEqual(len(session.requests), 1)


class TestOpenAlex(unittest.TestCase):
    def test_parse_rebuilds_abstract_and_doi(self):
        session = FakeSession([FakeResponse(200, load_fixture("openalex_works.json"))])
        [p] = OpenAlexSearch(mailto="me@example.org", session=session).search(
            "q", Constraints(year_from=1990, year_to=2000))
        self.assertEqual(p.paper_id, "OA:W123")
        self.assertEqual(p.abstract, "Agents act autonomously and react and plan")
        self.assertEqual(p.doi, "10.1017/s0269888900008122")
        self.assertIn("from_publication_date:1990-01-01", session.requests[0]["params"]["filter"])


class TestFallback(unittest.TestCase):
    def test_falls_back_only_on_transport_failure(self):
        primary, secondary = mock.Mock(), mock.Mock()
        primary.name, secondary.name = "a", "b"
        primary.search.side_effect = SourceUnavailable("down")
        secondary.search.return_value = ["x"]
        self.assertEqual(FallbackSearch(primary, secondary).search("q", Constraints()), ["x"])
        # An empty (but successful) primary result must NOT trigger fallback.
        primary.search.side_effect, primary.search.return_value = None, []
        self.assertEqual(FallbackSearch(primary, secondary).search("q", Constraints()), [])

    @mock.patch("research_agent.sources.time.sleep")
    def test_network_exception_becomes_unavailable(self, _sleep):
        session = FakeSession([requests.ConnectionError("x")] * 4)
        with self.assertRaises(SourceUnavailable):
            OpenAlexSearch(session=session).search("q", Constraints())


class TestCrossref(unittest.TestCase):
    P = Paper(paper_id="p", year=2021, doi="10.1038/s42256-020-00287-7",
              title="An Open Source Machine Learning Framework for Efficient and Transparent Systematic Reviews")

    def test_verified_when_title_and_year_match(self):
        meta = load_fixture("crossref_work.json")["message"]
        out = CrossrefValidator.compare(self.P, meta)
        self.assertTrue(out.doi_verified)
        self.assertIsNone(out.metadata_flag)

    def test_title_mismatch_flagged(self):
        out = CrossrefValidator.compare(self.P, {"title": ["Soil microbiology"],
                                                 "issued": {"date-parts": [[2021]]}})
        self.assertFalse(out.doi_verified)
        self.assertTrue(out.metadata_flag.startswith("title_mismatch"))

    def test_year_tolerance_of_one(self):
        meta = {"title": [self.P.title], "issued": {"date-parts": [[2020]]}}
        self.assertTrue(CrossrefValidator.compare(self.P, meta).doi_verified)
        meta["issued"] = {"date-parts": [[2017]]}
        self.assertIn("year_mismatch", CrossrefValidator.compare(self.P, meta).metadata_flag)

    def test_not_found_no_doi_and_unavailable(self):
        session = FakeSession([FakeResponse(404)])
        out = CrossrefValidator(session=session).validate(self.P)
        self.assertEqual(out.metadata_flag, "doi_not_found")
        no_doi = Paper(paper_id="n", title="t")
        self.assertEqual(CrossrefValidator().validate(no_doi).metadata_flag, "no_doi")
        with mock.patch.object(CrossrefValidator, "fetch", side_effect=SourceUnavailable("x")):
            out = CrossrefValidator().validate(self.P)
        self.assertFalse(out.doi_verified)
        self.assertTrue(out.metadata_flag.startswith("crossref_unavailable"))

    def test_markup_in_crossref_titles_is_ignored(self):
        self.assertGreater(title_similarity("Learning <i>in situ</i> agents", "Learning in situ agents"), 0.95)


if __name__ == "__main__":
    unittest.main()
