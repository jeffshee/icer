class Face:
    def __init__(self, frame_number, location, encoding, is_detected=True):
        self.frame_number = frame_number
        self.location = location
        self.encoding = encoding
        self.is_detected = is_detected

    def __repr__(self):
        return "<Face frame_number:%s location:%s is_detected:%s>" % (
            self.frame_number, self.location, self.is_detected)

    def __str__(self):
        return "frame_number:%s location:%s is_detected:%s" % (self.frame_number, self.location, self.is_detected)


