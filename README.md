*Lane Detection System Using OpenCV*

A computer vision–based Lane Detection System implemented using Python and OpenCV.
This project detects lane markings from videos or live webcam feed, applies edge detection and region masking, and overlays detected lanes on the original frames.

#Features
- Supports video files and webcam input
- Edge detection using Canny
- Region of Interest (ROI) masking
- Lane overlay on original frames
- Output video generation

#Tech Stack
- Python
- OpenCV
- NumPy
- Matplotlib
- imutils

#Lane-Detection-System-Using-OpenCV/
    run.py                # Main entry point
    lane_detector.py      # Lane detection logic
    utils.py              # Helper functions
    test2.mp4             # Sample input video
    lane_out.avi          # Output video (generated)
    requirements.txt
    README.md

#How to Run the Project
1. Clone the Repository
git clone https://github.com/Vaishnavinichal/Lane-Detection-System-Using-OpenCV.git
cd Lane-Detection-System-Using-OpenCV

2. Create and Activate Virtual Environment
python -m venv .venv
.venv\Scripts\activate

3. Install Dependencies
pip install -r requirements.txt

4. Run 
python run.py --display
or
python run.py -i 0 --display

5. Save images
python run.py --display --debug


#Notes
- Use --display to enable visualization
- Without --display, the project runs in headless mode
- Press q to exit when display is enabled
