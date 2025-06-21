# Coach Harmony: Your AI-Powered Vocal Coach

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-deployment-link-here.streamlit.app/)

Coach Harmony is a sophisticated web application built with Streamlit that provides aspiring singers with personalized vocal training and professional-grade performance analysis. Leveraging advanced AI and signal processing techniques, this tool helps users unlock their full singing potential by offering actionable feedback on pitch, tempo, timbre, and more.

## ✨ Key Features

- **🎤 Advanced Vocal Analysis**: Get detailed metrics on pitch accuracy, stability, vocal range, and timbre consistency.
- **🎼 Genre-Specific Feedback**: The AI compares your performance against a dataset of various genres (Pop, Rock, Classical, etc.) to provide style-specific advice.
- **📈 Progress Tracking**: Visualize your improvement over time with historical charts for pitch accuracy and overall vocal scores.
- **🗣️ Actionable AI Feedback**: Receive clear, constructive tips and targeted exercises to improve your technique after each session.
- **📁 Easy File Upload**: Simply upload your `.wav` or `.mp3` recordings to get an instant analysis.

## 🛠️ Installation & Setup

To run this application on your local machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/AIExecute-Vocal-Coach.git
    cd AIExecute-Vocal-Coach
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 How to Run the App

Once you have completed the installation, you can run the app with a single command:

```bash
streamlit run app.py
```

The application will open in your default web browser.

## 📂 Project Structure

The table below outlines the essential files and directories required for the application to function correctly.

| Path               | Description                                                                                              |
| ------------------ | -------------------------------------------------------------------------------------------------------- |
| `app.py`           | The main Python script containing the Streamlit application logic and UI.                                |
| `requirements.txt` | A list of all the Python packages required to run the application.                                       |
| `data/`            | Contains the dataset files (`.csv`) and audio samples used for genre comparison and analysis.              |
| `models/`          | Stores the pre-trained machine learning models and scalers used for predictions and feature transformation.|
| `README.md`        | The file you are currently reading, providing an overview and instructions for the project.              |
| `download_data.py` | Script to automatically download and extract the data folder from Google Drive.                          |

## 📥 Downloading the Data

The full dataset is too large to be included in this repository.

To download and extract the data, run:

```bash
pip install gdown
python download_data.py
```

This will download and extract the data into the `data/` folder automatically from [Google Drive](https://drive.google.com/file/d/1Xb9q79ZbSVSKErvazg7BQEKXvuiYTzJ-/view?usp=drive_link).

---