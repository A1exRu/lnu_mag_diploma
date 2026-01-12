class PathPlanner:
    def plan(self, detections, depth_map):

        if len(detections) == 0:
            return "FORWARD"
        else:
            return "SLOW DOWN"
