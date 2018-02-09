
class Rectangle:
    def __init__(self, x=-1, y=-1, width=-1, height=-1):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

        self.isEmpty = (x == -1 or y == -1 or width == -1 or height == -1)
        self.topLeft = (x, y)
        self.bottomRight = (x+width, y+height)


