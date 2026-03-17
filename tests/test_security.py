"""Tests for security and cryptographic services."""

import pytest

from votingai.security.cryptographic_services import (
    AuditLogger,
    CryptographicIntegrity,
    SecurityValidator,
)


# ---------------------------------------------------------------------------
# SecurityValidator
# ---------------------------------------------------------------------------

class TestSecurityValidator:
    def test_valid_agent_name(self):
        assert SecurityValidator.validate_agent_name("Alice") == "Alice"
        assert SecurityValidator.validate_agent_name("Agent_1") == "Agent_1"
        assert SecurityValidator.validate_agent_name("bot-99") == "bot-99"

    def test_agent_name_too_long_raises(self):
        with pytest.raises(ValueError, match="too long"):
            SecurityValidator.validate_agent_name("A" * 51)

    def test_agent_name_invalid_chars_raises(self):
        with pytest.raises(ValueError, match="invalid characters"):
            SecurityValidator.validate_agent_name("Alice@Domain")

    def test_agent_name_with_space_raises(self):
        with pytest.raises(ValueError):
            SecurityValidator.validate_agent_name("Alice Bob")

    def test_validate_proposal_text_strips_xss(self):
        result = SecurityValidator.validate_proposal_text("<script>alert('xss')</script>")
        assert "<" not in result
        assert ">" not in result
        assert "script" in result  # content kept, tags removed

    def test_validate_proposal_text_too_long_raises(self):
        with pytest.raises(ValueError):
            SecurityValidator.validate_proposal_text("x" * 10001)

    def test_validate_vote_reasoning_strips_dangerous_chars(self):
        result = SecurityValidator.validate_vote_reasoning("Good proposal & safe")
        assert "&" not in result

    def test_validate_vote_reasoning_too_long_raises(self):
        with pytest.raises(ValueError):
            SecurityValidator.validate_vote_reasoning("x" * 5001)

    def test_generate_secure_nonce_is_unique(self):
        nonce1 = SecurityValidator.generate_secure_nonce()
        nonce2 = SecurityValidator.generate_secure_nonce()
        assert nonce1 != nonce2
        assert len(nonce1) == 32  # 16 bytes = 32 hex chars

    def test_generate_proposal_id_starts_with_prefix(self):
        pid = SecurityValidator.generate_proposal_id()
        assert pid.startswith("proposal_")

    def test_generate_proposal_id_is_unique(self):
        assert SecurityValidator.generate_proposal_id() != SecurityValidator.generate_proposal_id()

    def test_vote_signature_verification(self):
        vote_data = {"vote": "approve", "proposal_id": "p1", "reasoning": "looks good"}
        key = "secret-key"
        sig = SecurityValidator.create_vote_signature(vote_data, key)
        assert SecurityValidator.verify_vote_signature(vote_data, key, sig) is True

    def test_vote_signature_fails_with_wrong_key(self):
        vote_data = {"vote": "approve", "proposal_id": "p1", "reasoning": "looks good"}
        sig = SecurityValidator.create_vote_signature(vote_data, "key1")
        assert SecurityValidator.verify_vote_signature(vote_data, "wrong-key", sig) is False

    def test_validate_vote_options_too_many_raises(self):
        with pytest.raises(ValueError):
            SecurityValidator.validate_vote_options(["option"] * 21)

    def test_validate_vote_options_valid(self):
        options = ["Option A", "Option B"]
        result = SecurityValidator.validate_vote_options(options)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# CryptographicIntegrity
# ---------------------------------------------------------------------------

class TestCryptographicIntegrity:
    def setup_method(self):
        self.crypto = CryptographicIntegrity()

    def test_register_agent_returns_key(self):
        key = self.crypto.register_agent("Alice")
        assert isinstance(key, str)
        assert len(key) == 64  # 32 bytes = 64 hex chars

    def test_sign_and_verify_vote(self):
        self.crypto.register_agent("Alice")
        vote_data = {"vote": "approve", "proposal_id": "p1"}
        sig = self.crypto.sign_vote("Alice", vote_data)
        assert self.crypto.verify_vote_signature("Alice", vote_data, sig) is True

    def test_tampered_vote_fails_verification(self):
        self.crypto.register_agent("Bob")
        vote_data = {"vote": "approve", "proposal_id": "p1"}
        sig = self.crypto.sign_vote("Bob", vote_data)
        tampered = {"vote": "reject", "proposal_id": "p1"}
        assert self.crypto.verify_vote_signature("Bob", tampered, sig) is False

    def test_sign_unregistered_agent_raises(self):
        with pytest.raises(ValueError, match="not registered"):
            self.crypto.sign_vote("Unknown", {"vote": "approve"})

    def test_replay_attack_detection(self):
        used_nonces = {"nonce-123", "nonce-456"}
        assert self.crypto.detect_replay_attack("nonce-123", used_nonces) is True
        assert self.crypto.detect_replay_attack("nonce-new", used_nonces) is False

    def test_different_agents_get_different_keys(self):
        key1 = self.crypto.register_agent("Alice")
        key2 = self.crypto.register_agent("Bob")
        assert key1 != key2


# ---------------------------------------------------------------------------
# AuditLogger
# ---------------------------------------------------------------------------

class TestAuditLogger:
    def setup_method(self):
        self.logger = AuditLogger(enable_file_logging=False)

    def test_starts_empty(self):
        assert len(self.logger.audit_entries) == 0

    def test_log_proposal_created(self):
        self.logger.log_proposal_created("p1", "Alice", "Test Proposal")
        assert len(self.logger.audit_entries) == 1
        assert self.logger.audit_entries[0]["event_type"] == "proposal_created"

    def test_log_vote_cast(self):
        self.logger.log_vote_cast("p1", "Alice", "approve", True)
        assert len(self.logger.audit_entries) == 1
        assert self.logger.audit_entries[0]["event_type"] == "vote_cast"

    def test_log_voting_result(self):
        self.logger.log_voting_result("p1", "approved", 1.0)
        entry = self.logger.audit_entries[0]
        assert entry["result"] == "approved"
        assert entry["participation_rate"] == 1.0

    def test_log_security_violation(self):
        self.logger.log_security_violation("replay_attack", "nonce reused")
        entry = self.logger.audit_entries[0]
        assert entry["event_type"] == "security_violation"
        assert entry["violation_type"] == "replay_attack"

    def test_audit_summary_counts_events(self):
        self.logger.log_proposal_created("p1", "Alice", "Test")
        self.logger.log_vote_cast("p1", "Bob", "approve", True)
        self.logger.log_vote_cast("p1", "Carol", "reject", True)
        summary = self.logger.get_audit_summary()
        assert summary["total_entries"] == 3

    def test_multiple_votes_logged(self):
        for i in range(5):
            self.logger.log_vote_cast(f"p{i}", "Alice", "approve", False)
        assert len(self.logger.audit_entries) == 5
