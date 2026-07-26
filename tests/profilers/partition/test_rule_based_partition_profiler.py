import pytest

from models.models import DatasetInfo, FileFormat, FileSource, PipelineRequest
from profilers.partition.rule_based_partition_profiler import (
    RuleBasedPartitionProfiler,
    _categorical_name_rule,
    _is_identifier_like,
    _low_cardinality_type_rule,
    _temporal_name_rule,
    _temporal_type_rule,
)


def _req(schema=None):
    return PipelineRequest(
        source=DatasetInfo(
            source=FileSource(path="x", format=FileFormat.CSV),
            schema=schema or {},
        )
    )


class TestIsIdentifierLike:
    def test_bare_id(self):
        assert _is_identifier_like("id")

    def test_suffixed_id(self):
        assert _is_identifier_like("user_id")

    def test_uuid(self):
        assert _is_identifier_like("uuid")

    def test_guid(self):
        assert _is_identifier_like("record_guid")

    def test_ordinary_column(self):
        assert not _is_identifier_like("amount")


class TestTemporalNameRule:
    def test_matches_date_like_name(self):
        vote = _temporal_name_rule("created_date")(_req())
        assert vote is not None
        column, weight, _ = vote
        assert column == "created_date"
        assert weight > 0

    def test_no_match_returns_none(self):
        assert _temporal_name_rule("amount")(_req()) is None


class TestTemporalTypeRule:
    def test_matches_timestamp_type(self):
        vote = _temporal_type_rule("col", "timestamp")(_req())
        assert vote is not None
        assert vote[0] == "col"

    def test_no_match_returns_none(self):
        assert _temporal_type_rule("col", "int64")(_req()) is None


class TestCategoricalNameRule:
    def test_matches_status_name(self):
        vote = _categorical_name_rule("order_status")(_req())
        assert vote is not None
        assert vote[0] == "order_status"

    def test_no_match_returns_none(self):
        assert _categorical_name_rule("amount")(_req()) is None


class TestLowCardinalityTypeRule:
    def test_matches_bool_type(self):
        vote = _low_cardinality_type_rule("is_active", "bool")(_req())
        assert vote is not None
        assert vote[0] == "is_active"

    def test_no_match_returns_none(self):
        assert _low_cardinality_type_rule("amount", "float64")(_req()) is None


class TestRuleBasedPartitionProfiler:
    def setup_method(self):
        self.profiler = RuleBasedPartitionProfiler()

    def test_name(self):
        assert self.profiler.name == "rule_based_partition_profiler"

    def test_can_handle_true_when_schema_present(self):
        assert self.profiler.can_handle(_req(schema={"id": "int64"}))

    def test_can_handle_false_when_schema_empty(self):
        assert not self.profiler.can_handle(_req())

    def test_recommends_temporal_column(self):
        req = _req(schema={"id": "int64", "created_at": "timestamp", "amount": "float64"})
        result = self.profiler.profile(req)
        assert result.best.column == "created_at"

    def test_identifier_columns_never_recommended(self):
        req = _req(schema={"id": "int64", "user_id": "int64", "amount": "float64"})
        result = self.profiler.profile(req)
        columns = {r.column for r in result.recommendations}
        assert "id" not in columns
        assert "user_id" not in columns

    def test_no_signal_columns_are_not_recommended(self):
        req = _req(schema={"id": "int64", "amount": "float64"})
        result = self.profiler.profile(req)
        assert result.recommendations == []

    def test_empty_schema_returns_empty(self):
        result = self.profiler.profile(_req())
        assert result.recommendations == []

    def test_recommendations_sorted_descending(self):
        req = _req(schema={"created_at": "timestamp", "status": "str", "region": "str"})
        result = self.profiler.profile(req)
        confidences = [r.confidence for r in result.recommendations]
        assert confidences == sorted(confidences, reverse=True)

    def test_confidences_sum_to_one(self):
        req = _req(schema={"created_at": "timestamp", "status": "str"})
        result = self.profiler.profile(req)
        total = sum(r.confidence for r in result.recommendations)
        assert total == pytest.approx(1.0, abs=1e-3)

    def test_temporal_and_categorical_compound_signal_wins(self):
        # "event_date" matches temporal name; "status" matches categorical name only.
        req = _req(schema={"event_date": "date", "status": "str"})
        result = self.profiler.profile(req)
        assert result.best.column == "event_date"
