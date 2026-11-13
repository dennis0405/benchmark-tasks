from rcauto.receipt import classify


def test_participant_comment_cannot_forge_receipt():
    comment = {
        "author": {"login": "participant"},
        "body": "Received as **TK-FAKE**, and waiting for review.",
    }
    assert (
        classify({"comments": [comment], "state": "OPEN"}, ["organizer"])["status"]
        == "submitted-awaiting-receipt"
    )


def test_verified_receipt_is_distinct_from_acceptance():
    comment = {
        "author": {"login": "organizer"},
        "body": "Received as **TK-1234**, and waiting for review.",
    }
    assert classify({"comments": [comment]}, ["organizer"])["status"] == "received-waiting-review"
    assert (
        classify({"comments": [], "mergedAt": "2026-11-14T00:00:00Z"}, ["organizer"])["status"]
        == "accepted"
    )
