"""Losslessly retain every <qq8d contact record in bounded private gzip chunks."""

import gzip
import hashlib
from pathlib import Path
import re
import struct


RECORD_FORMAT = "<qq8d"
RECORD_BYTES = struct.calcsize(RECORD_FORMAT)


def file_sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


class BoundedContactArchive:
    def __init__(self, workspace, trial_id, max_raw_bytes=4 * 1024 * 1024):
        if not re.fullmatch(r"[a-f0-9-]{36}", trial_id):
            raise ValueError("invalid trial identifier")
        if not RECORD_BYTES <= max_raw_bytes <= 4 * 1024 * 1024:
            raise ValueError("invalid chunk size")
        self.workspace = Path(workspace).resolve()
        self.directory = self.workspace / "private-experimental-evidence" / "sdf-contact" / trial_id
        self.directory.mkdir(parents=True, exist_ok=False)
        self.max_raw_bytes = max_raw_bytes
        self.records = 0
        self.stream_hash = hashlib.sha256()
        self.verified_hash = hashlib.sha256()
        self.chunks = []
        self.closed = False
        self._file = self._gzip = self._path = None
        self._chunk_bytes = 0

    def append(self, record):
        if self.closed:
            raise RuntimeError("archive is already finalized")
        if not isinstance(record, bytes) or len(record) != RECORD_BYTES:
            raise ValueError("one exact <qq8d record is required")
        if self._gzip is not None and self._chunk_bytes + len(record) > self.max_raw_bytes:
            self._finish_chunk()
        if self._gzip is None:
            self._path = self.directory / ("contacts-%05d.bin.gz" % len(self.chunks))
            self._file = self._path.open("xb")
            self._gzip = gzip.GzipFile(filename="", mode="wb", fileobj=self._file, mtime=0)
        if self._gzip.write(record) != len(record):
            raise OSError("short raw-contact write")
        self.stream_hash.update(record)
        self._chunk_bytes += len(record)
        self.records += 1

    def _finish_chunk(self):
        if self._gzip is None:
            return
        self._gzip.close()
        self._file.close()
        total = 0
        raw_hash = hashlib.sha256()
        with gzip.open(self._path, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                total += len(chunk)
                raw_hash.update(chunk)
                self.verified_hash.update(chunk)
        if total != self._chunk_bytes or total % RECORD_BYTES:
            raise OSError("raw-contact chunk failed complete round-trip verification")
        if self._path.stat().st_size >= 50_000_000:
            raise OSError("compressed contact chunk exceeds reporting threshold")
        self.chunks.append({
            "path": self._path.relative_to(self.workspace).as_posix(),
            "compressed_bytes": self._path.stat().st_size,
            "compressed_sha256": file_sha256(self._path),
            "raw_bytes": total, "raw_sha256": raw_hash.hexdigest(),
            "records": total // RECORD_BYTES,
        })
        self._file = self._gzip = self._path = None
        self._chunk_bytes = 0

    def finish(self):
        if not self.closed:
            self._finish_chunk()
            if self.verified_hash.hexdigest() != self.stream_hash.hexdigest():
                raise OSError("whole raw-contact stream round-trip mismatch")
            if sum(item["records"] for item in self.chunks) != self.records:
                raise OSError("raw-contact record count mismatch")
            self.closed = True
        return {
            "scope": "Exact raw contacts retained privately on Brev and copied with the full "
                     "workspace to Mac; these private chunks are not in the GitHub checkpoint.",
            "record_format": RECORD_FORMAT,
            "fields": ["step", "native_contact_index", "normal_force_n", "point_x", "point_y",
                       "point_z", "normal_x", "normal_y", "normal_z", "separation_m"],
            "record_bytes": RECORD_BYTES, "records": self.records,
            "stream_sha256": self.stream_hash.hexdigest(), "chunks": self.chunks,
            "all_appended_records_round_trip_verified": self.closed,
        }
