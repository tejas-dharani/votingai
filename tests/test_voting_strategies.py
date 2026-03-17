"""Tests for voting strategy implementations."""

import pytest

from votingai.core.voting_strategies import (
    MajorityVotingStrategy,
    PluralityVotingStrategy,
    QualifiedMajorityStrategy,
    UnanimousVotingStrategy,
    VotingStrategyFactory,
    extract_confidence_scores,
    validate_weighted_votes,
)
from votingai.core.voting_protocols import VotingMethod


# ---------------------------------------------------------------------------
# MajorityVotingStrategy
# ---------------------------------------------------------------------------

class TestMajorityVotingStrategy:
    def setup_method(self):
        self.strategy = MajorityVotingStrategy()

    def test_approved_when_majority_approve(self):
        result = self.strategy.calculate_result(
            {"approve": 3, "reject": 1, "abstain": 0}, 4, [1.0, 1.0, 1.0, 1.0]
        )
        assert result.result == "approved"
        assert result.is_approved

    def test_rejected_when_majority_reject(self):
        result = self.strategy.calculate_result(
            {"approve": 1, "reject": 3, "abstain": 0}, 4, [1.0, 1.0, 1.0, 1.0]
        )
        assert result.result == "rejected"
        assert result.is_rejected

    def test_no_consensus_on_tie(self):
        result = self.strategy.calculate_result(
            {"approve": 2, "reject": 2, "abstain": 0}, 4, [1.0, 1.0, 1.0, 1.0]
        )
        assert result.result == "no_consensus"
        assert not result.has_consensus

    def test_no_consensus_on_empty_votes(self):
        result = self.strategy.calculate_result({}, 3, [])
        assert result.result == "no_consensus"

    def test_participation_rate_calculated(self):
        result = self.strategy.calculate_result(
            {"approve": 3, "reject": 0, "abstain": 0}, 4, [1.0, 1.0, 1.0]
        )
        assert result.participation_rate == pytest.approx(3 / 4)

    def test_method_name(self):
        assert self.strategy.method_name == "majority"

    def test_does_not_require_threshold(self):
        assert self.strategy.requires_threshold is False


# ---------------------------------------------------------------------------
# QualifiedMajorityStrategy
# ---------------------------------------------------------------------------

class TestQualifiedMajorityStrategy:
    def test_approved_at_67_percent(self):
        strategy = QualifiedMajorityStrategy(threshold=0.67)
        result = strategy.calculate_result(
            {"approve": 2, "reject": 1, "abstain": 0}, 3, [1.0, 1.0, 1.0]
        )
        assert result.result == "approved"

    def test_no_consensus_below_threshold(self):
        strategy = QualifiedMajorityStrategy(threshold=0.75)
        result = strategy.calculate_result(
            {"approve": 2, "reject": 2, "abstain": 0}, 4, [1.0, 1.0, 1.0, 1.0]
        )
        assert result.result == "no_consensus"

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError):
            QualifiedMajorityStrategy(threshold=0.0)

    def test_threshold_above_one_raises(self):
        with pytest.raises(ValueError):
            QualifiedMajorityStrategy(threshold=1.5)

    def test_method_name_includes_threshold(self):
        strategy = QualifiedMajorityStrategy(threshold=0.67)
        assert "67" in strategy.method_name

    def test_requires_threshold(self):
        assert QualifiedMajorityStrategy().requires_threshold is True


# ---------------------------------------------------------------------------
# UnanimousVotingStrategy
# ---------------------------------------------------------------------------

class TestUnanimousVotingStrategy:
    def setup_method(self):
        self.strategy = UnanimousVotingStrategy()

    def test_approved_when_all_approve(self):
        result = self.strategy.calculate_result(
            {"approve": 3, "reject": 0, "abstain": 0}, 3, [1.0, 1.0, 1.0]
        )
        assert result.result == "approved"

    def test_rejected_when_all_reject(self):
        result = self.strategy.calculate_result(
            {"approve": 0, "reject": 3, "abstain": 0}, 3, [1.0, 1.0, 1.0]
        )
        assert result.result == "rejected"

    def test_no_consensus_with_split_vote(self):
        result = self.strategy.calculate_result(
            {"approve": 2, "reject": 1, "abstain": 0}, 3, [1.0, 1.0, 1.0]
        )
        assert result.result == "no_consensus"

    def test_no_consensus_with_abstention(self):
        result = self.strategy.calculate_result(
            {"approve": 2, "reject": 0, "abstain": 1}, 3, [1.0, 1.0, 1.0]
        )
        assert result.result == "no_consensus"

    def test_method_name(self):
        assert self.strategy.method_name == "unanimous"


# ---------------------------------------------------------------------------
# PluralityVotingStrategy
# ---------------------------------------------------------------------------

class TestPluralityVotingStrategy:
    def setup_method(self):
        self.strategy = PluralityVotingStrategy()

    def test_approved_with_plurality(self):
        result = self.strategy.calculate_result(
            {"approve": 2, "reject": 1, "abstain": 0}, 3, [1.0, 1.0, 1.0]
        )
        assert result.result == "approved"

    def test_rejected_with_plurality(self):
        result = self.strategy.calculate_result(
            {"approve": 1, "reject": 2, "abstain": 0}, 3, [1.0, 1.0, 1.0]
        )
        assert result.result == "rejected"

    def test_no_consensus_on_tie(self):
        result = self.strategy.calculate_result(
            {"approve": 2, "reject": 2, "abstain": 0}, 4, [1.0, 1.0, 1.0, 1.0]
        )
        assert result.result == "no_consensus"

    def test_method_name(self):
        assert self.strategy.method_name == "plurality"


# ---------------------------------------------------------------------------
# VotingStrategyFactory
# ---------------------------------------------------------------------------

class TestVotingStrategyFactory:
    def setup_method(self):
        self.factory = VotingStrategyFactory()

    def test_creates_majority_strategy(self):
        strategy = self.factory.create_strategy(VotingMethod.MAJORITY)
        assert isinstance(strategy, MajorityVotingStrategy)

    def test_creates_qualified_majority_strategy(self):
        strategy = self.factory.create_strategy(VotingMethod.QUALIFIED_MAJORITY, threshold=0.75)
        assert isinstance(strategy, QualifiedMajorityStrategy)
        assert strategy.threshold == 0.75

    def test_creates_unanimous_strategy(self):
        strategy = self.factory.create_strategy(VotingMethod.UNANIMOUS)
        assert isinstance(strategy, UnanimousVotingStrategy)

    def test_creates_plurality_strategy(self):
        strategy = self.factory.create_strategy(VotingMethod.PLURALITY)
        assert isinstance(strategy, PluralityVotingStrategy)

    def test_unsupported_method_raises(self):
        with pytest.raises(ValueError):
            self.factory.create_strategy(VotingMethod.RANKED_CHOICE)

    def test_available_methods_returned(self):
        methods = self.factory.get_available_methods()
        assert VotingMethod.MAJORITY in methods
        assert VotingMethod.UNANIMOUS in methods
        assert VotingMethod.PLURALITY in methods
        assert VotingMethod.QUALIFIED_MAJORITY in methods

    def test_qualified_majority_default_threshold(self):
        strategy = self.factory.create_strategy(VotingMethod.QUALIFIED_MAJORITY)
        assert isinstance(strategy, QualifiedMajorityStrategy)
        assert strategy.threshold == pytest.approx(0.67)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

class TestUtilityFunctions:
    def test_extract_confidence_scores(self):
        votes = {
            "Alice": {"confidence": 0.9},
            "Bob": {"confidence": 0.7},
            "Carol": {"confidence": 1.0},
        }
        scores = extract_confidence_scores(votes)
        assert scores == pytest.approx([0.9, 0.7, 1.0])

    def test_extract_confidence_default_for_missing(self):
        votes = {"Alice": {}}
        scores = extract_confidence_scores(votes)
        assert scores == [1.0]

    def test_validate_weighted_votes_passes(self):
        validate_weighted_votes({"approve": 2.0, "reject": 1.0, "abstain": 0.0})

    def test_validate_weighted_votes_missing_key_raises(self):
        with pytest.raises(ValueError):
            validate_weighted_votes({"approve": 2.0, "reject": 1.0})

    def test_validate_weighted_votes_negative_raises(self):
        with pytest.raises(ValueError):
            validate_weighted_votes({"approve": -1.0, "reject": 0.0, "abstain": 0.0})
