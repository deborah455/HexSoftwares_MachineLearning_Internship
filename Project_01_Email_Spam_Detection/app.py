from flask import Flask, render_template, request, redirect, url_for
import joblib
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from datetime import datetime
import textwrap


app = Flask(__name__)


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load("models/spam_classifier.joblib")
vectorizer = joblib.load("models/tfidf_vectorizer.joblib")


# ============================================================
# DIRECTORIES
# ============================================================

SCREENSHOTS_DIR = Path("screenshots")
SCREENSHOTS_DIR.mkdir(exist_ok=True)


# ============================================================
# PREDICTION HISTORY
# ============================================================

prediction_history = []


# ============================================================
# FONT HELPER
# ============================================================

def get_font(size, bold=False):
    """
    Load a Windows font.
    Falls back to Pillow's default font if unavailable.
    """

    if bold:
        font_path = "C:/Windows/Fonts/arialbd.ttf"
    else:
        font_path = "C:/Windows/Fonts/arial.ttf"

    try:
        return ImageFont.truetype(font_path, size)
    except OSError:
        return ImageFont.load_default()


# ============================================================
# CREATE PREDICTION SCREENSHOT
# ============================================================

def create_prediction_screenshot(message, prediction, confidence):
    """
    Create and save a PNG screenshot of the prediction result.
    """

    # --------------------------------------------------------
    # Generate sequential filename
    # --------------------------------------------------------

    existing_files = list(
        SCREENSHOTS_DIR.glob("test_*.png")
    )

    test_number = len(existing_files) + 1

    filename = (
        f"test_{test_number:03d}.png"
    )

    filepath = SCREENSHOTS_DIR / filename


    # --------------------------------------------------------
    # Canvas
    # --------------------------------------------------------

    width = 1200
    height = 850

    image = Image.new(
        "RGB",
        (width, height),
        "#f5f7fb"
    )

    draw = ImageDraw.Draw(image)


    # --------------------------------------------------------
    # Fonts
    # --------------------------------------------------------

    title_font = get_font(42, bold=True)
    subtitle_font = get_font(22)
    heading_font = get_font(25, bold=True)
    body_font = get_font(20)
    result_font = get_font(32, bold=True)
    small_font = get_font(17)


    # --------------------------------------------------------
    # Main white card
    # --------------------------------------------------------

    draw.rounded_rectangle(
        (70, 55, 1130, 795),
        radius=25,
        fill="white"
    )


    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    draw.text(
        (600, 105),
        "📧",
        font=get_font(50),
        anchor="mm"
    )

    draw.text(
        (600, 175),
        "Email Spam Detector",
        font=title_font,
        fill="#1f2937",
        anchor="mm"
    )

    draw.text(
        (600, 220),
        "Machine Learning Prediction Result",
        font=subtitle_font,
        fill="#6b7280",
        anchor="mm"
    )


    # --------------------------------------------------------
    # Email section
    # --------------------------------------------------------

    draw.rounded_rectangle(
        (120, 275, 1080, 500),
        radius=18,
        fill="#f9fafb",
        outline="#e5e7eb",
        width=2
    )

    draw.text(
        (155, 310),
        "Email Tested",
        font=heading_font,
        fill="#374151"
    )


    # --------------------------------------------------------
    # Wrap email text
    # --------------------------------------------------------

    wrapped_lines = textwrap.wrap(
        message,
        width=78
    )

    max_lines = 7

    wrapped_lines = wrapped_lines[:max_lines]

    if len(
        textwrap.wrap(message, width=78)
    ) > max_lines:

        wrapped_lines[-1] += "..."


    y_position = 355

    for line in wrapped_lines:

        draw.text(
            (155, y_position),
            line,
            font=body_font,
            fill="#4b5563"
        )

        y_position += 30


    # --------------------------------------------------------
    # Result colors
    # --------------------------------------------------------

    if prediction == "Spam":

        result_background = "#fee2e2"
        result_border = "#fecaca"
        result_color = "#991b1b"
        result_icon = "🚨"
        result_text = "SPAM DETECTED"

    else:

        result_background = "#dcfce7"
        result_border = "#bbf7d0"
        result_color = "#166534"
        result_icon = "✓"
        result_text = "NOT SPAM"


    # --------------------------------------------------------
    # Result box
    # --------------------------------------------------------

    draw.rounded_rectangle(
        (120, 525, 1080, 685),
        radius=18,
        fill=result_background,
        outline=result_border,
        width=2
    )

    draw.text(
        (600, 565),
        result_icon,
        font=get_font(38, bold=True),
        fill=result_color,
        anchor="mm"
    )

    draw.text(
        (600, 610),
        result_text,
        font=result_font,
        fill=result_color,
        anchor="mm"
    )

    draw.text(
        (600, 650),
        f"Model Confidence: {confidence}%",
        font=body_font,
        fill=result_color,
        anchor="mm"
    )


    # --------------------------------------------------------
    # Footer
    # --------------------------------------------------------

    draw.text(
        (600, 735),
        "TF-IDF + Multinomial Naive Bayes",
        font=small_font,
        fill="#9ca3af",
        anchor="mm"
    )

    draw.text(
        (600, 765),
        "Hex Software Machine Learning Internship",
        font=small_font,
        fill="#9ca3af",
        anchor="mm"
    )


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    image.save(
        filepath,
        "PNG"
    )

    print(
        f"Screenshot saved to: {filepath}"
    )


# ============================================================
# HOME
# ============================================================

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    confidence = None
    message = ""


    if request.method == "POST":

        message = request.form.get(
            "message",
            ""
        ).strip()


        if message:

            # ------------------------------------------------
            # Transform message
            # ------------------------------------------------

            features = vectorizer.transform(
                [message]
            )


            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            prediction_value = model.predict(
                features
            )[0]


            probabilities = model.predict_proba(
                features
            )[0]


            # ------------------------------------------------
            # Confidence
            # ------------------------------------------------

            confidence = round(
                float(max(probabilities)) * 100,
                2
            )


            # ------------------------------------------------
            # Label
            # ------------------------------------------------

            prediction = (
                "Spam"
                if prediction_value == 1
                else "Not Spam"
            )


            # ------------------------------------------------
            # Save prediction history
            # ------------------------------------------------

            prediction_history.insert(
                0,
                {
                    "message": (
                        message[:80]
                        + ("..." if len(message) > 80 else "")
                    ),
                    "prediction": prediction,
                    "confidence": confidence,
                },
            )


            # Keep latest 10
            del prediction_history[10:]


            # ------------------------------------------------
            # SAVE SCREENSHOT
            # ------------------------------------------------

            create_prediction_screenshot(
                message=message,
                prediction=prediction,
                confidence=confidence
            )


    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        message=message,
        history=prediction_history,
    )


# ============================================================
# CLEAR HISTORY
# ============================================================

@app.route(
    "/clear-history",
    methods=["POST"]
)
def clear_history():

    prediction_history.clear()

    return redirect(
        url_for("home")
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )