# Novelty dispatch trial, 2026-11-16

The full comparison corpus stays mandatory. This trial changes scheduling only:
up to 100 complete records per dispatch, with a 400,000-byte budget for the
serialized records. A single larger definition remains whole in its own batch.
Per-record decision criteria, original sources, coverage checks, overlap stops,
repair behavior and cache identity are unchanged. No parallel agents are added.

On the actual 2,374-record catalog, fixed groups of 25 require 95 groups. The
trial grouping needs 34; both contain all 2,374 complete records. Maximum
serialized record payload changes from 382,100 to 398,717 bytes. This is a
dispatch-count measurement, **not an established model wall-time improvement**.
Source attachments read later are not included in that serialized-size budget.

Field validation must compare completed records per minute on new, uncached
reviews, inspect full comparison/repair coverage, and retain the prior receipts.
If model throughput does not improve beyond normal variation, revert the
scheduling change. Already completed decisions can be reused in either grouping;
regrouping must neither invent results nor invalidate equivalent evidence.

Regression checks cover all-record identity/coverage, intact oversized Unicode
definitions, a late overlap, invalid limits and cache reuse after regrouping.
