from __future__ import annotations

import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rethinkskill.utils.integrity import (
    freeze_manifest,
    freeze_receipt,
)
from rethinkskill.utils.serde import (
    RegularFileSnapshot,
    atomic_copy,
    atomic_write,
    atomic_write_json,
    atomic_write_json_snapshot,
    atomic_write_snapshot,
    canonical_json_bytes,
    read_json,
    read_json_snapshot,
    sha256_bytes,
    sha256_file,
    snapshot_regular_file,
    strict_json_loads,
)


class HashingAndIOTests(unittest.TestCase):
    def test_regular_file_snapshot_rejects_forged_metadata(self) -> None:
        with self.assertRaisesRegex(ValueError, "digest does not match"):
            RegularFileSnapshot(
                payload=b"evidence",
                sha256="0" * 64,
                size=len(b"evidence"),
            )
        with self.assertRaisesRegex(ValueError, "size does not match"):
            RegularFileSnapshot(
                payload=b"evidence",
                sha256=sha256_bytes(b"evidence"),
                size=0,
            )

    def test_regular_file_snapshot_binds_payload_size_and_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "evidence.jsonl"
            path.write_bytes(b'{"id":"before"}\n')

            snapshot = snapshot_regular_file(path)
            path.write_bytes(b'{"id":"after"}\n')

            self.assertEqual(snapshot.payload, b'{"id":"before"}\n')
            self.assertEqual(snapshot.size, len(snapshot.payload))
            self.assertEqual(snapshot.sha256, sha256_bytes(snapshot.payload))

    def test_regular_file_snapshot_does_not_follow_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "evidence.jsonl"
            target.write_bytes(b'{"id":"private"}\n')
            link = root / "link.jsonl"
            link.symlink_to(target)

            with self.assertRaises((OSError, ValueError)):
                snapshot_regular_file(link)

    def test_regular_file_snapshot_rejects_in_place_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "evidence.jsonl"
            path.write_bytes(b'{"id":"before"}\n')
            original_read = os.read
            mutated = False

            def drifting_read(descriptor: int, size: int) -> bytes:
                nonlocal mutated
                payload = original_read(descriptor, size)
                if payload and not mutated:
                    mutated = True
                    path.write_bytes(b'{"id":"replacement-longer"}\n')
                return payload

            with (
                patch("rethinkskill.utils.serde.os.read", side_effect=drifting_read),
                self.assertRaisesRegex(ValueError, "changed while reading"),
            ):
                snapshot_regular_file(path)

    def test_canonical_json_is_order_independent(self) -> None:
        self.assertEqual(
            canonical_json_bytes({"b": 2, "a": 1}),
            canonical_json_bytes({"a": 1, "b": 2}),
        )

    def test_canonical_json_rejects_non_finite_numbers(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "Out of range float values",
        ):
            canonical_json_bytes({"score": float("nan")})

    def test_canonical_json_rejects_non_text_object_keys(self) -> None:
        for value in (
            {1: "integer"},
            {"nested": {2: "integer"}},
            [{"nested": {3: "integer"}}],
        ):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(
                    TypeError,
                    "object keys must be strings",
                ),
            ):
                canonical_json_bytes(value)

    def test_manifest_freeze_detaches_nested_values(self) -> None:
        source = {"status": "PASS", "nested": {"value": 1}}
        manifest = freeze_manifest(source)
        source["nested"]["value"] = 99
        self.assertEqual(manifest["nested"]["value"], 1)
        self.assertNotIn("manifest_sha256", manifest)
        with self.assertRaises(TypeError):
            manifest["nested"]["value"] = 2

    def test_receipt_freeze_is_immutable_without_self_hash(self) -> None:
        receipt = freeze_receipt(
            {
                "schema_version": 3,
                "status": "VALIDATED",
                "calls": {"attempted": 1, "completed": 1},
            }
        )
        self.assertNotIn("receipt_sha256", receipt)
        with self.assertRaises(TypeError):
            receipt["calls"]["completed"] = 0

    def test_strict_json_loading_rejects_non_standard_constants(self) -> None:
        for value in (
            '{"score": NaN}',
            '{"score": Infinity}',
            "[-Infinity]",
            '{"score": 1e999}',
        ):
            with (
                self.subTest(value=value),
                self.assertRaises(json.JSONDecodeError),
            ):
                strict_json_loads(value)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "invalid.json"
            path.write_text('{"score": NaN}', encoding="utf-8")
            with self.assertRaises(json.JSONDecodeError):
                read_json(path)

    def test_strict_json_loading_rejects_duplicate_object_keys(self) -> None:
        for value in (
            '{"name": 1, "name": 2}',
            '{"nested": {"name": 1, "name": 2}}',
        ):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(
                    json.JSONDecodeError,
                    "duplicate JSON object key",
                ),
            ):
                strict_json_loads(value)

    def test_json_parsing_and_hashing_share_one_file_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "evidence.json"
            original = canonical_json_bytes({"state": "original"})
            path.write_bytes(original)

            snapshot, value = read_json_snapshot(path)
            path.write_bytes(canonical_json_bytes({"state": "drifted"}))

            self.assertEqual(value, {"state": "original"})
            self.assertEqual(snapshot.payload, original)
            self.assertEqual(snapshot.sha256, sha256_bytes(original))

    def test_atomic_json_is_canonical(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "nested" / "value.json"
            atomic_write_json(path, {"b": 2, "a": 1})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"a": 1, "b": 2})
            self.assertEqual(sha256_file(path), sha256_file(path))

    def test_atomic_write_snapshot_binds_exact_published_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "nested/evidence.jsonl"
            payload = b'{"id":"bound"}\n'

            snapshot = atomic_write_snapshot(path, payload)

            self.assertEqual(path.read_bytes(), payload)
            self.assertEqual(snapshot.payload, payload)
            self.assertEqual(snapshot.sha256, sha256_bytes(payload))
            self.assertEqual(snapshot.size, len(payload))

    def test_atomic_write_snapshot_requires_exact_bytes(self) -> None:
        class BytesSubclass(bytes):
            pass

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "evidence.bin"
            with self.assertRaisesRegex(TypeError, "exact bytes"):
                atomic_write_snapshot(path, BytesSubclass(b"unsafe"))
            self.assertFalse(path.exists())

    def test_atomic_json_snapshot_binds_canonical_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"

            snapshot = atomic_write_json_snapshot(
                path,
                {"b": 2, "a": {"value": 1}},
            )

            expected = canonical_json_bytes({"a": {"value": 1}, "b": 2})
            self.assertEqual(path.read_bytes(), expected)
            self.assertEqual(snapshot.payload, expected)
            self.assertEqual(snapshot.sha256, sha256_bytes(expected))

    def test_atomic_write_does_not_follow_predictable_temp_symlink(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "value.json"
            victim = root / "victim.txt"
            victim.write_text("preserve", encoding="utf-8")
            predictable = root / f".value.json.tmp.{os.getpid()}"
            predictable.symlink_to(victim)
            before = set(root.glob(".value.json.tmp.*"))

            atomic_write(destination, b"replacement")

            self.assertEqual(destination.read_bytes(), b"replacement")
            self.assertEqual(victim.read_text(encoding="utf-8"), "preserve")
            self.assertEqual(set(root.glob(".value.json.tmp.*")), before)
            self.assertEqual(
                stat.S_IMODE(destination.stat().st_mode),
                0o600,
            )

    def test_atomic_write_cleans_temporary_after_replace_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "value.json"
            with (
                patch(
                    "rethinkskill.utils.serde.os.replace",
                    side_effect=OSError("synthetic replace failure"),
                ),
                self.assertRaisesRegex(OSError, "synthetic replace failure"),
            ):
                atomic_write(destination, b"replacement")

            self.assertFalse(destination.exists())
            self.assertEqual(tuple(root.glob(".value.json.tmp.*")), ())

    def test_atomic_copy_verifies_frozen_digest_before_publish(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.bin"
            destination = root / "destination.bin"
            source.write_bytes(b"actual")

            with self.assertRaisesRegex(
                ValueError,
                "frozen SHA-256",
            ):
                atomic_copy(
                    source,
                    destination,
                    expected_sha256="0" * 64,
                )

            self.assertFalse(destination.exists())
            self.assertEqual(
                tuple(root.glob(".destination.bin.tmp.*")),
                (),
            )

    def test_atomic_copy_rejects_symlink_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target.bin"
            target.write_bytes(b"evidence")
            source = root / "source.bin"
            source.symlink_to(target)
            destination = root / "destination.bin"

            with self.assertRaisesRegex(ValueError, "must not be a symlink"):
                atomic_copy(
                    source,
                    destination,
                    expected_sha256=sha256_file(target),
                )

            self.assertFalse(destination.exists())
