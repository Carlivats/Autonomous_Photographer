# Autonomous Photographer
*Developed by Carlos Vasquez Trochez | Supervised by Komalpreet Kaur, PhD**

*Salem State University | Computer Science Department**

The Autonomous Photographer is an intelligent, robotic camera system **designed to eliminate the compromises of taking photos during travel, outdoor activities, or public events.** By automatically recognizing subjects, analyzing scene composition, and executing motor movements, it captures visually balanced, professional-quality photographs independently.

**Built for portability and affordability,** the system allows users to step into their environment while the camera evaluates the composition, repositions itself, and takes the perfect shot.

---

## 🚀 Key Features

* **Real-Time AI Vision (YOLOv8):** Deploys OpenCV and lightweight deep learning models (YOLOv8) for continuous subject detection. Tracking logic has been custom-engineered to prioritize facial keypoints (specifically centering on the nose)
* **Autonomous Hardware Reframing:** Evaluates scene layout based on established photographic principles (like the Rule of Thirds or Centered Symmetry). The control system translates composition scores into precise pan-tilt motor commands, actively adjusting the frame in under **2 seconds per iteration**.
* **Quality Assurance Analysis:** Continuously evaluates sharpness, contrast, exposure, and motion blur. The automated rating engine discards blurry or poorly lit shots and retains only high-quality photos.
* **Highly Portable & Wireless:** Equipped with a 5V UPS Shield and batteries for continuous, untethered power. 
* **Intuitive User Interface:** A desktop-class interface built with PyQt5 that mimics an everyday camera app. Users can initiate sessions, adjust framing preferences, check live statistics, and browse the integrated gallery.

---

## 🛠️ Hardware Architecture

The physical build was designed for accessibility and reproducibility using cost-effective components. **All custom 3D-printed chassis files (.stl / LycheeSlicer files) are available in this repository!!**

* **Compute & AI Acceleration:**
  * Raspberry Pi 5
  * Hailo AI HAT+ (handles computationally expensive object-detection models)
* **Vision:**
  * Raspberry Pi AI Camera Module (IMX500)
* **Motion Control:**
  * Custom 3D-printed pan-tilt mount
  * Two precision Servo Motors
  * **PCA9685 Servo Driver:** Isolates motor power from the logic board, resolving voltage-related trembling to ensure smooth, stable camera motion.
* **Power:**
  * UPS 5V Shield + Batteries for field portability

---

## 💻 Software Stack

* **Programming Language:** Python
* **Libraries & Frameworks:** OpenCV, NumPy, PyQt5 (User Interface), RPi.GPIO (Hardware Control)
* **Machine Learning:** YOLOv8
* **3D Modeling & Printing:** Blender, LycheeSlicer
* **Project Management:** Miro, GitHub

---

## 📂 Data Storage Strategy

Captured images are saved directly to local storage alongside corresponding JSON metadata files (using the exact same basename). This sidecar metadata logs:
* Timestamps
* Sharpness & Motion Blur scores
* Exposure & Contrast ratings
* Overall Composition score
* Detected subject coordinates

---

## 🔮 Future Scope

While the current prototype successfully bridges the gap between casual photography and professional composition, future iterations will focus on:
1. **Model Compression:** Optimizing the object-detection models to run entirely on the native Raspberry Pi 5 hardware, removing the need for the Hailo AI HAT+ to further reduce hardware costs.
2. **Mobile Application:** Transitioning the UI to a dedicated mobile application to provide a more accessible wireless connection and control experience for everyday users.

---

## 📬 Contact
**Carlos Vasquez Trochez**
* Email: carlosvasqueztrochez@gmail.com
* GitHub: [Carlivats](https://github.com/Carlivats)
