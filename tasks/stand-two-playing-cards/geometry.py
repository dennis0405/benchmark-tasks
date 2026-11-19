"""SI dimensions for the two free jumbo cards, their pickup slots and the mat."""
import math

WIDTH, HEIGHT, THICKNESS = .089, .127, .00065
MASS = .008
MAT_Z = .775
CENTER_X = .30
STAGGER = .028
RACK_Y = .165
NAMES = ('coral', 'blue')
SIGNS = (1, -1)
LEAN = math.radians(16)
HOLD_SECONDS = 3.0


def stand_boxes():
    yield 'mat', (CENTER_X, 0, MAT_Z-.01375), (.29, .44, .0275)
    for name, sign in zip(NAMES, SIGNS):
        for jaw in (-1, 1):
            yield name+('_front' if jaw == 1 else '_back'), (
                CENTER_X+jaw*.0035, sign*RACK_Y, MAT_Z+.01), (.005,.077,.02)
