import time
import numpy as np
import cv2
from picamera2 import Picamera2, MappedArray
from picamera2.devices.imx500 import IMX500

model_path = "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"
model = IMX500(model_path)
intrinsics = model.network_intrinsics

picam2 = Picamera2(model.camera_num)

# Drawing Callback
def draw_boxes(request):
    metadata = request.get_metadata()
    np_outputs = model.get_outputs(metadata, add_batch=True)
    
    if np_outputs is not None:
        # Access the image array directly using MappedArray
        with MappedArray(request, "main") as m:
            boxes, scores, labels = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]
            
            for i, score in enumerate(scores):
                if score > 0.6:
                    # Convert IMX500 coordinates to screen pixels
                    # imx500.convert_inference_coords returns (x, y, w, h)
                    x, y, w, h = model.convert_inference_coords(boxes[i], metadata, picam2)
                    label_name = intrinsics.labels[int(labels[i])]
                    
                    # Draw the bounding box
                    cv2.rectangle(m.array, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Add label text
                    text = f"{label_name} {score:.2f}"
                    cv2.putText(m.array, text, (x, y - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Using "XRGB8888" format for easier OpenCV compatibility
config = picam2.create_preview_configuration(
    main={"format": "XRGB8888", "size": (640, 480)},
    controls={"FrameRate": intrinsics.inference_rate}
)

# Register the drawing function as a pre-callback
picam2.pre_callback = draw_boxes

model.show_network_fw_progress_bar()
picam2.start(config, show_preview=True)

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    picam2.stop()
