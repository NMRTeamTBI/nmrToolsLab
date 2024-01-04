
class UtilsHandler():
    def __init__(self):
        pass

    def cm2inch(self,*tupl):
        inch = 2.54
        if isinstance(tupl[0], tuple):
            return tuple(i/inch for i in tupl[0])
        else:
            return tuple(i/inch for i in tupl)