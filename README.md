# Estimating circles from corners

Welcome to this lab in the computer vision course [TEK5030] at the University of Oslo.

In this lab we will
- Create our own corner keypoint detector `CornerDetector`.
- Use detected corner keypoints and RANSAC to find a circle with the `CircleEstimator` class.

Start by cloning this repository on your machine.
Then open the lab project in your editor.

The lab is carried out by following these steps:

1. [Get an overview](lab-guide/1-get-an-overview.md)
2. [Implement a corner feature detector](lab-guide/2-implement-a-corner-feature-detector.md)
3. [Detect circles from corners with RANSAC](lab-guide/3-detect-circles-from-corners-with-ransac.md)

Start the lab by going to the [first step](lab-guide/1-get-an-overview.md).

## Prerequisites

Here is a quick reference if you need to set up a Python virtual environment manually:

```bash
python3.8 -m venv venv  # any python version > 3.8 is OK
source venv/bin/activate.
# expect to see (venv) at the beginning of your prompt.
pip install -U pip  # <-- Important step for Ubuntu 18.04!
pip install -r requirements.txt
```

Please consult the [resource pages] if you need more help with the setup.

[TEK5030]: https://www.uio.no/studier/emner/matnat/its/TEK5030/
[resource pages]: https://tek5030.github.io