from sys import argv
import numpy as np
import cv2

default_box = (50,50,600,800)

class MouseEnterBox():
    def __init__(self, video_path, bx=default_box[0],by=default_box[1],bw=default_box[2],bh=default_box[3], display_video=False):
        # Capture
        self.cap = cv2.VideoCapture(video_path)
        self.display = display_video

        # Box
        self.box_x, self.box_y, self.box_w, self.box_h = bx, by, bw, bh 
        self.box = (self.box_x, self.box_y, self.box_x + self.box_w, self.box_y + self.box_h)

        # Background Subtractor
        self.fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold=50, detectShadows=False)

        self.frame_count = 0
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.mouse_enter_time = None

    def get_time(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            self.frame_count += 1

            #Gray
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            #Background Subtraction
            fgmask = self.fgbg.apply(gray)

            #Threshold
            _, thresh = cv2.threshold(fgmask, 50, 255, cv2.THRESH_BINARY)

            #Contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            #Process
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 500:  # Adjust threshold based on mouse size
                    continue

                x, y, w, h = cv2.boundingRect(cnt)

                # cv2.putText(frame, f"{frame_count / fps:.2f} sec", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                # Check if bounding box is inside the predefined box
                if self.box[0] <= x and self.box[1] <= y and (x + w) <= (self.box[2]) and (y + h) <= (self.box[3]):
                    if self.mouse_enter_time is None:
                        self.mouse_enter_time = self.frame_count / self.fps
                    break  # Mouse is inside, no need to check further

            if self.display:
                if (self.display_image(frame)):
                    break

        self.cap.release()
        cv2.destroyAllWindows()

        return self.mouse_enter_time

    def display_image(self, image):
        
                # Draw the box for visualization
                cv2.rectangle(image, (self.box[0],self.box[1]), (self.box[2],self.box[3]), (0, 255, 0), 2)

                if self.mouse_enter_time is not None:
                    cv2.putText(image, f"{self.mouse_enter_time} sec", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Show the frame (for debugging)
                new_size = (900, 600)
                frame_out = cv2.resize(image,new_size)

                cv2.imshow("Frame", frame_out)

                # thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
                # cv2.rectangle(thresh, (self.box[0],self.box[1]), (self.box[2],self.box[3]), (0, 255, 0), 2)
                # thresh_out = cv2.resize(thresh,new_size)
                # cv2.imshow("Mask", thresh_out)

                if cv2.waitKey(int((1/self.fps) *1000)) & 0xFF == 27:  # Press ESC to exit
                    return True
                return False



if __name__ == "__main__":    
    bx = None
    by = None
    bw = None
    bh = None
    flags = []

    filepath = argv[1]
    if (len(argv) > 5):
        bx = int(argv[2])    # Must all exist together
        by = int(argv[3])    # Must all exist together
        bw = int(argv[4])    # Must all exist together
        bh = int(argv[5])    # Must all exist together
        flags = argv[6:]    # Additional Flags (-V)
    
    if bx is None: bx = default_box[0]
    if by is None: by = default_box[1]
    if bw is None: bw = default_box[2]
    if bh is None: bh = default_box[3]

    if ("-V" in flags): display_video=True
    else: display_video = False
    
    MEB = MouseEnterBox("mouse_video.mp4",bx=bx, by=by, bw=bw, bh=bh, display_video=display_video)

    enter = MEB.get_time()

    if enter is not None:
        print(f"Mouse fully entered the box at {enter:.2f} seconds.")
    else:
        print("Mouse never fully entered the box.")