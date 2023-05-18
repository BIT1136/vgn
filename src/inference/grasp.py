class ParallelJawGrasp(object):
    def __init__(self, pose, width):
        self.pose = pose
        self.width = width

    def __str__(self) -> str:
        return f"pose:{self.pose},width:{self.width}"
