import os
import time
from datetime import datetime

from flask import Flask, render_template, session, request, abort, redirect, url_for, flash
from flask_wtf import FlaskForm
from flask_wtf.file import FileRequired, FileField
from wtforms import SubmitField, TextAreaField
from wtforms.validators import InputRequired, ValidationError
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

from utils import ImageCaptionPipline

# Example loading
# https://github.com/devtonic-net/flask-loading-app-message-and-spinner/blob/main/main.py


# Instantiates Flask Application
# Run the app by typing 'flask run' in the virtual environment terminal
app = Flask(__name__)
load_dotenv()
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY")

# Configures file upload configurations
if not os.path.exists("static"):
    os.mkdir("static")
elif not os.path.exists(os.path.join("static", "uploads")):
    os.mkdir(os.path.join("static", "uploads"))
app.config["UPLOAD_DIRECTORY"] = "static/uploads"
app.config["MAX_UPLOAD_SIZE"] = 1048576  # 2 MB

# Sets up model pipeline
CaptionGenerator = ImageCaptionPipline("text_transformer_saved.pkl", "large_bilstm.h5")


# Validates file upload type and size
def validate_file_upload(file_field):
    if not file_field.data:
        return " File is invalid."
    elif not file_field.data.content_type.startswith("image"):
        return " Invalid file type (must be an image)."

    file_size = len(file_field.data.read())
    file_field.data.seek(0)  # Since validation read the file, reset the head

    if file_size > app.config["MAX_UPLOAD_SIZE"]:
        return f" File exceeds maximum size ({app.config['MAX_UPLOAD_SIZE']} bytes)."
    return None


# Main page form
class MainForm(FlaskForm):
    select_file = FileField("File", validators=[FileRequired()])
    text_field = TextAreaField("Creative Tuning")
    submit = SubmitField("Generate Caption")


# Main Webpage
# Contains basic information about the project
@app.route("/", methods=["GET", "POST"])
def home():
    form = MainForm()
    if form.validate_on_submit():
        # Ensure valid uploaded file
        validation = validate_file_upload(form.select_file)
        if validation:
            return render_template("home.html", form=form, flash_message=validation)

        context = form.text_field.data
        filename = secure_filename(form.select_file.data.filename)
        form.select_file.data.save(os.path.join(app.config["UPLOAD_DIRECTORY"], filename))
        session["filename"] = filename
        session['current_time'] = time.time()
        session["resulting_caption"] = CaptionGenerator.predict(os.path.join(app.config["UPLOAD_DIRECTORY"], filename))

        return redirect("/result")
    return render_template("home.html", form=form, flash_message=None)


# Result Webpage
# Displays the generated captions
@app.route("/result")
def result():
    filename = session.get("filename", None)
    current_time = session.get("current_time", None)
    resulting_caption = session.get("resulting_caption", None)
    if filename and resulting_caption:
        time_string = None
        if current_time:
            time_string = "Inference Time: {:.2f}s".format(time.time() - current_time)

        captions_list = []
        captions_list.append(resulting_caption + "1")
        captions_list.append(resulting_caption + "2")
        captions_list.append(resulting_caption + "3")

        filepath = "uploads/" + filename
        return render_template("result.html", image=filepath, captions=captions_list, time_string=time_string)


# Handles 404 error and returns a page html
@app.errorhandler(404)
def page_not_found(err):
    return "404 Page Not Found", 404


# Handles 500 error and returns a page html
@app.errorhandler(500)
def page_not_found(err):
    return "500 Internal Server Error", 500


# Runs the application. Alternatively it can be run by typing "flask run" in the terminal after activating the
# virtual environment
if __name__ == "__main__":
    app.run(debug=True)
