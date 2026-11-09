"""Tests for :mod:`api.progress` — Redis pub/sub progress publisher.

All tests mock the Redis client to avoid requiring a running Redis instance.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from api.progress import channel_for, make_progress_publisher, publish_sentinel


# ---------------------------------------------------------------------------
# channel_for
# ---------------------------------------------------------------------------


class TestChannelFor:
    """Tests for the channel name helper."""

    def test_returns_expected_format(self) -> None:
        assert channel_for("abc123") == "run:abc123:progress"

    def test_different_ids_produce_different_channels(self) -> None:
        assert channel_for("a") != channel_for("b")

    def test_empty_id(self) -> None:
        assert channel_for("") == "run::progress"


# ---------------------------------------------------------------------------
# make_progress_publisher
# ---------------------------------------------------------------------------


class TestMakeProgressPublisher:
    """Tests for the progress publisher factory."""

    @patch("api.progress._redis.from_url")
    @patch("api.progress.get_config")
    def test_returns_client_and_callback(
        self,
        mock_config: MagicMock,
        mock_from_url: MagicMock,
    ) -> None:
        mock_config.return_value.redis.url = "redis://localhost:6379/0"
        mock_client = MagicMock()
        mock_from_url.return_value = mock_client

        client, callback = make_progress_publisher("run123")

        assert client is mock_client
        assert callable(callback)
        mock_from_url.assert_called_once_with("redis://localhost:6379/0")

    @patch("api.progress._redis.from_url")
    @patch("api.progress.get_config")
    def test_callback_publishes_json(
        self,
        mock_config: MagicMock,
        mock_from_url: MagicMock,
    ) -> None:
        mock_config.return_value.redis.url = "redis://localhost:6379/0"
        mock_client = MagicMock()
        mock_from_url.return_value = mock_client

        _, callback = make_progress_publisher("run123")
        callback(0.5, "Halfway there")

        expected_channel = "run:run123:progress"
        published = mock_client.publish.call_args
        assert published[0][0] == expected_channel
        payload = json.loads(published[0][1])
        assert payload == {"fraction": 0.5, "message": "Halfway there"}

    @patch("api.progress._redis.from_url")
    @patch("api.progress.get_config")
    def test_callback_suppresses_publish_errors(
        self,
        mock_config: MagicMock,
        mock_from_url: MagicMock,
    ) -> None:
        mock_config.return_value.redis.url = "redis://localhost:6379/0"
        mock_client = MagicMock()
        mock_client.publish.side_effect = ConnectionError("Redis down")
        mock_from_url.return_value = mock_client

        _, callback = make_progress_publisher("run123")
        # Should not raise
        callback(0.1, "test")

    @patch("api.progress._redis.from_url")
    def test_uses_explicit_redis_url(
        self,
        mock_from_url: MagicMock,
    ) -> None:
        mock_from_url.return_value = MagicMock()
        client, _ = make_progress_publisher("run1", redis_url="redis://custom:1234/5")
        mock_from_url.assert_called_once_with("redis://custom:1234/5")


# ---------------------------------------------------------------------------
# publish_sentinel
# ---------------------------------------------------------------------------


class TestPublishSentinel:
    """Tests for the sentinel publisher."""

    @patch("api.progress._redis.from_url")
    @patch("api.progress.get_config")
    def test_completed_sentinel(
        self,
        mock_config: MagicMock,
        mock_from_url: MagicMock,
    ) -> None:
        mock_config.return_value.redis.url = "redis://localhost:6379/0"
        mock_client = MagicMock()
        mock_from_url.return_value = mock_client

        publish_sentinel("run42", status="completed")

        expected_channel = "run:run42:progress"
        published = mock_client.publish.call_args
        assert published[0][0] == expected_channel
        payload = json.loads(published[0][1])
        assert payload == {"fraction": 1.0, "message": "completed"}
        mock_client.close.assert_called_once()

    @patch("api.progress._redis.from_url")
    @patch("api.progress.get_config")
    def test_failed_sentinel_with_error(
        self,
        mock_config: MagicMock,
        mock_from_url: MagicMock,
    ) -> None:
        mock_config.return_value.redis.url = "redis://localhost:6379/0"
        mock_client = MagicMock()
        mock_from_url.return_value = mock_client

        publish_sentinel("run42", status="failed", error="timeout")

        payload = json.loads(mock_client.publish.call_args[0][1])
        assert payload == {"fraction": 1.0, "message": "failed: timeout"}
        mock_client.close.assert_called_once()

    @patch("api.progress._redis.from_url")
    @patch("api.progress.get_config")
    def test_sentinel_closes_client_on_error(
        self,
        mock_config: MagicMock,
        mock_from_url: MagicMock,
    ) -> None:
        """Client should be closed even if publish raises."""
        mock_config.return_value.redis.url = "redis://localhost:6379/0"
        mock_client = MagicMock()
        mock_client.publish.side_effect = ConnectionError("boom")
        mock_from_url.return_value = mock_client

        with pytest.raises(ConnectionError):
            publish_sentinel("run42", status="completed")

        mock_client.close.assert_called_once()

    @patch("api.progress._redis.from_url")
    def test_sentinel_uses_explicit_redis_url(
        self,
        mock_from_url: MagicMock,
    ) -> None:
        mock_client = MagicMock()
        mock_from_url.return_value = mock_client

        publish_sentinel("r1", status="completed", redis_url="redis://custom:9999/0")
        mock_from_url.assert_called_once_with("redis://custom:9999/0")
