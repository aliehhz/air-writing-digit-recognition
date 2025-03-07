# Air-Writing Digit Recognition

This project allows users to draw digits in the air using their index finger, and the system recognizes the drawn number using a deep learning model.


## Features
- **Real-time hand tracking** using MediaPipe
- **Air-writing on a virtual canvas**
- **Digit recognition** using a trained CNN model
- **Web-based UI** using Flask
- **Deployment on Render**

---

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/real-time-digit-recognition.git
cd real-time-digit-recognition
```

### 2. Create a Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application Locally
```bash
python app.py
```
Then, open your browser and go to:
```
http://127.0.0.1:5000/

```

## Future Improvements
- Improve model accuracy using more training data
- Implement gesture-based commands
- Support for letters in addition to digits


---

## Author
Developed by [Alieh Hassanzadeh](https://github.com/aliehhz).
Feel free to contribute and improve!

