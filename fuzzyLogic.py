import sympy as sp

class fuzzySet:
    def __init__(self, inputValue):
        x = sp.Symbol('x')
        self.NL = sp.Piecewise(
            (0, x < 0),
            (0, x > .5),
            (1 - 2*x, True)
        )
        self.NM = sp.Piecewise(
            (4*x, x < .25),
            (0, x > .75),
            (1.5 - 2*x, True)
        )
        self.Z = sp.Piecewise(
            (2*x, x < .5),
            (2 - 2*x, x >= .5)
        )
        self.PM = sp.Piecewise(
            (0, x < .25),
            (4 - 4*x, x > .75),
            (-0.5 + 2*x, True)
        )
        self.PL = sp.Piecewise(
            (0, x < .5),
            (0, x > 1),
            (-1 + 2*x, True)
        )


mySet = fuzzySet(0.8)
