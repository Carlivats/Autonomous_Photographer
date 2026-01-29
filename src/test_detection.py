import time
import numpy as np
from picamera2 import Picamera2
from picamera2.devices.imx500 import IMX500

model_path = "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"
imx500 = IMX500(model_path)
intrinsics = imx500.network_intrinsics

picam2 = Picamera2(imx500.camera_num)

config = picam2.create_preview_configuration(controls={"FrameRate": intrinsics.inference_rate})
imx500.show_network_fw_progress_bar()
picam2.start(config, show_preview=True)

try:
    while True:
        metadata = picam2.capture_metadata()
        
        # Get raw inference results (outputs list of NumPy arrays)
        # For MobileNetV2 SSD, outputs should be: [boxes, scores, classes]
        np_outputs = imx500.get_outputs(metadata, add_batch=True)
        
        if np_outputs is not None:
            boxes, scores, labels = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]
            
            for i, score in enumerate(scores):
                if score > 0.6:  # 60% confidence threshold
                    # Use the labels loaded from the model intrinsics
                    label_name = intrinsics.labels[int(labels[i])]
                    print(f"Detected: {label_name} with {score:.2f} confidence")
        
        time.sleep(0.1)
        
except KeyboardInterrupt:
    picam2.stop()
