from sys import argv
import cv2

default_box = (50, 50, 600, 800)

class MouseEnterBox():
    def __init__(self, video_path, bx=default_box[0], by=default_box[1], bw=default_box[2], bh=default_box[3], display_video=False):
        self.cap = cv2.VideoCapture(video_path)
        self.display = display_video

        # Box coordinates
        self.box_x, self.box_y, self.box_w, self.box_h = bx, by, bw, bh
        self.box = (self.box_x, self.box_y, self.box_x + self.box_w, self.box_y + self.box_h)

        # Background subtractor
        self.fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold=50, detectShadows=False)

        # Frame tracking
        self.frame_count = 0
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps == 0:
            raise ValueError("Failed to read FPS from video.")

        # Mouse state
        self.in_box = False
        self.entry_frame = None
        self.entry_times = []
        self.durations = []

    def get_times(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            self.frame_count += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            fgmask = self.fgbg.apply(gray)
            _, thresh = cv2.threshold(fgmask, 50, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            mouse_inside = False
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 500:
                    continue

                x, y, w, h = cv2.boundingRect(cnt)
                if self.box[0] <= x and self.box[1] <= y and (x + w) <= self.box[2] and (y + h) <= self.box[3]:
                    mouse_inside = True
                    break

            if mouse_inside and not self.in_box:
                self.entry_frame = self.frame_count
                self.in_box = True

            elif not mouse_inside and self.in_box:
                exit_frame = self.frame_count
                entry_time = self.entry_frame / self.fps
                duration = (exit_frame - self.entry_frame) / self.fps
                self.entry_times.append(entry_time)
                self.durations.append(duration)
                self.in_box = False
                self.entry_frame = None

            if self.display:
                if self.display_image(frame):
                    break

        # Handle case where video ends with mouse still inside
        if self.in_box and self.entry_frame is not None:
            entry_time = self.entry_frame / self.fps
            duration = (self.frame_count - self.entry_frame) / self.fps
            self.entry_times.append(entry_time)
            self.durations.append(duration)

        self.cap.release()
        cv2.destroyAllWindows()

        return list(zip(self.entry_times, self.durations))

    def display_image(self, image):
        cv2.rectangle(image, (self.box[0], self.box[1]), (self.box[2], self.box[3]), (0, 255, 0), 2)

        if self.in_box and self.entry_frame is not None:
            current_time = self.frame_count / self.fps
            time_in = current_time - (self.entry_frame / self.fps)
            cv2.putText(image, f"In box: {time_in:.2f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        frame_out = cv2.resize(image, (900, 600))
        cv2.imshow("Frame", frame_out)

        if cv2.waitKey(int((1/self.fps) * 1000)) & 0xFF == 27:
            return True
        return False


if __name__ == "__main__":
    bx = by = bw = bh = None
    flags = []
    filepath = argv[1]

    if len(argv) > 5:
        bx = int(argv[2])
        by = int(argv[3])
        bw = int(argv[4])
        bh = int(argv[5])
        flags = argv[6:]

    if bx is None: bx = default_box[0]
    if by is None: by = default_box[1]
    if bw is None: bw = default_box[2]
    if bh is None: bh = default_box[3]

    display_video = "-V" in flags
    MEB = MouseEnterBox(filepath, bx=bx, by=by, bw=bw, bh=bh, display_video=display_video)
    results = MEB.get_times()
    

    if results:
        print("Mouse entry times and durations:")
        for i, (entry_time, duration) in enumerate(results):
            print(f"{i+1}. Entered at {entry_time:.2f}s, stayed for {duration:.2f}s")
    else:
        print("Mouse never fully entered the box.")
