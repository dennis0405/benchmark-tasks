# Novelty dispatch trial, 2026-11-16

The full comparison corpus stays mandatory. This trial changes scheduling only:
up to 100 complete records per dispatch, with a 400,000-byte budget for the
serialized records. A single larger definition remains whole in its own batch.
Per-record decision criteria, original sources, coverage checks, overlap stops,
repair behavior and cache identity are unchanged. No parallel agents are added.

Initial sizing in the 2,374-record catalog's source order gave 95 versus 34
groups. **That order is not the engine's relevance order.** After two new
submissions arrived, the actual engine ordering for 2,376 records gives:

| Schedule | Groups | Complete records | Maximum serialized record payload |
| --- | ---: | ---: | ---: |
| Fixed 25 | 96 | 2,376 | 992,467 bytes |
| Trial count/byte budget | 43 | 2,376 | 399,953 bytes |

These are scheduling/payload measurements, **not an established model wall-time
improvement**. Source attachments read later are not included in that
serialized-size budget. Actual dispatches also depend on exact cache hits and
required coverage repairs. No model was called to measure grouping.

Field validation must compare completed records per minute on new, uncached
reviews, inspect full comparison/repair coverage, and retain the prior receipts.
If model throughput does not improve beyond normal variation, revert the
scheduling change. Already completed decisions can be reused in either grouping;
regrouping must neither invent results nor invalidate equivalent evidence.

Regression checks cover all-record identity/coverage, intact oversized Unicode
definitions, a late overlap, invalid limits and cache reuse after regrouping.

## Cache-aware follow-up

Observed during the live trial: unchanged, cached large definitions still used
the byte budget of a batch, even though the model was asked only about the few
changed records left in it. This fragmented fresh work across small calls.

Resolve all exact per-record cache evidence before grouping the remaining work.
The same context, record, note and schema validation still apply. A cached
overlap blocks immediately; it cannot be resampled through an unrelated review.
Only genuinely missing/changed definitions consume a new dispatch's budget.

A fixed regression fixture with 20 unchanged large definitions and 20 changed
short definitions previously made 20 fresh calls; it now makes one, retaining
all 40 full comparisons. This is a measured request-count result with a test
reviewer, not a production-model wall-time result. Existing partial-coverage
repairs, changed-note invalidation and overlap guards remain covered.
