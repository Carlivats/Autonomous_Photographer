import tkinter as tk
from picamera2 import Picamera2
from PIL import Image, ImageTk

class Picamera2Preview:
    def __init__(self, root):
        self.root = root
        self.root.title("Autonomous Camera Preview")
        
        self.root.attributes('-fullscreen', True)
        self.root.configure(bg="black")
        
        self.picam2 = Picamera2()
        config = self.picam2.create_video_configuration(
            {"size": (640, 480), "format": "RGB888"}
        )
        self.picam2.configure(config)
        self.picam2.start()

        # --- UI Layout ---
        self.video_label = tk.Label(self.root, bg="black")
        self.video_label.pack(expand=True, fill="both")

        self.exit_button = tk.Button(
            self.root, 
            text="Exit Preview", 
            font=("Helvetica", 16, "bold"), 
            bg="#e74c3c", 
            fg="white", 
            command=self.exit_app, 
            height=2
        )
        self.exit_button.pack(fill="x")

        # Start the continuous frame capture loop
        self.update_frame()

    def update_frame(self):
        """Captures a frame from Picamera2 and updates the Tkinter label."""
        try:
            # Grab the current frame directly as a numpy array
            frame = self.picam2.capture_array("main")
            
            # Convert the array directly to a PIL Image, then to Tkinter format
            pil_img = Image.fromarray(frame)
            tk_img = ImageTk.PhotoImage(image=pil_img)
            
            # Update the label
            self.video_label.imgtk = tk_img # Prevent garbage collection
            self.video_label.configure(image=tk_img)
            
        except Exception as e:
            print(f"Error capturing frame: {e}")

        # Schedule the next frame update (~60 fps target)
        self.root.after(15, self.update_frame)

    def exit_app(self):
        """Cleanly shut down the camera and close the UI."""
        self.picam2.stop()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = Picamera2Preview(root)
    root.mainloop()