# Facial Recognition Security System

<p align="center">
  A Python and OpenCV-based facial detection and recognition system for security monitoring.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv" alt="OpenCV">
</p>

---

## Overview

The **Facial Recognition Security System** is a computer vision project built with **Python** and **OpenCV**. It detects faces from images, compares them against a predefined facial database, and identifies recognized individuals.

The project also includes a webcam-based security monitoring component that can be used for detecting and recognizing faces during surveillance.

## Features

* **Face Detection** — Detects human faces using OpenCV Haar Cascade classifiers.
* **Face Recognition** — Compares detected faces with images stored in the local database.
* **Identity Matching** — Identifies recognized individuals based on the database images.
* **Face Annotation** — Draws bounding boxes and labels around detected/recognized faces.
* **Image Processing** — Processes input images and generates an annotated output.
* **Webcam Security** — Provides webcam-based facial monitoring through the security module.
* **Custom Database** — Allows additional individuals to be added to the recognition database.

## Technologies Used

* **Python**
* **OpenCV**
* **face_recognition**
* **NumPy**
* **Matplotlib**

## Project Structure

```text
facial-recognition-security-system/
│
├── database/
│   ├── M.Freeman.jpeg
│   └── S.Stallone.jpeg
│
├── images/
│   └── actors.jpeg
│
├── main.py
├── webcam_security.py
├── README.md
└── result.jpg
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/karthikvemula23/facial-recognition-security-system.git
cd facial-recognition-security-system
```

### 2. Install the required dependencies

```bash
pip install face_recognition opencv-python numpy matplotlib
```

## Usage

### Image-Based Face Recognition

Run:

```bash
python main.py
```

The program will allow you to select an image and process it for face detection and recognition.

The recognized faces are annotated with their corresponding names.

### Webcam Security

To run the webcam-based security component:

```bash
python webcam_security.py
```

The webcam module can be used for real-time facial monitoring and security-related detection.

## Face Database

The `database` directory contains the reference images used for facial recognition.

To add a new person:

1. Add an image of the person to the `database` directory.
2. Use the person's name as the filename.
3. Make sure the image contains a clear view of a single face.

Example:

```text
database/
├── M.Freeman.jpeg
├── S.Stallone.jpeg
└── John_Doe.jpeg
```

The filename is used as the person's identity during recognition.

## Output

After processing an image, the system generates an annotated image containing:

* Detected face bounding boxes
* Recognized person's name
* Face recognition results

The generated image is saved as:

```text
result.jpg
```

## How It Works

The system follows a basic computer vision pipeline:

```text
Input Image / Webcam
        ↓
   Face Detection
        ↓
   Face Recognition
        ↓
 Compare with Database
        ↓
 Identify Person
        ↓
 Annotate / Security Response
```

## Future Improvements

Potential improvements include:

* Real-time facial recognition optimization
* Improved recognition accuracy
* Support for multiple face databases
* Video file processing
* Unknown-person detection and alerts
* Email or notification-based security alerts
* Web-based monitoring dashboard
* Improved performance for larger databases


