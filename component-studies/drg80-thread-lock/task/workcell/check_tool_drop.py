"""Controls at both supported integration rates; no simulator is needed."""

from tool_drop import ToolDrop


def example():
    return ToolDrop((-0.05, -0.60), (0.75, 0.60), 0.75)


for hz in (240, 480):
    m = example()
    assert not m.observe(1, (0, 0, 0.95), True, False)
    assert not m.observe(2, (0, 0, 0.91), False, True)
    assert not m.observe(3, (0, 0, 0.899), False, False)
    assert m.observe(4, (0, 0, 0.91), False, True) == (
        4,
        "unsupported_fall_with_impact",
    )
    assert m.observe(5, (0, 0, 0.95), True, False) == m.failure
    m = example()
    assert not m.observe(1, (0, 0, 0.95), True, False)
    assert not m.observe(2, (0, 0, 0.89), True, True)
    assert not m.observe(3, (0, 0, 0.85), False, True)
    assert not m.observe(4, (0.799, 0, 0.80), False, False)
    assert m.observe(5, (0.801, 0, 0.80), False, False) == (5, "tool_outside_table")
    assert example().observe(1, (0, 0, 0.699), False, False)
    print("tool drop controls passed at", hz, "Hz")
