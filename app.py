import os
import time

import openai
from flask import Flask, render_template, session, request, abort, redirect, url_for, flash
from flask_wtf import FlaskForm
from flask_wtf.file import FileRequired, FileField
from wtforms import SubmitField, TextAreaField
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

from utils import ImageCaptionPipline


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
print("~ Reminder that I have to pay for API usage ~")
print("https://platform.openai.com/usage")
openai.api_key = os.getenv("OPENAI_API_KEY")
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
        session["context"] = context

        return redirect("/loading")
    return render_template("home.html", form=form, flash_message=None)


# Loading page that is displayed to the user when the caption is generating
@app.route("/loading", methods=["GET", "POST"])
def loading():
    return render_template("loading.html")


# Result Webpage
# Displays the generated captions
@app.route("/result")
def result():
    filename = session.get("filename", None)
    context = session.get("context", None)

    if filename is None:
        print("ERROR: Session File Name is Invalid!")
        return redirect("/")
    session.pop("filename")
    session.pop("context")

    # Times and generates the captions using the ML generation pipeline
    current_time = time.time()
    resulting_caption = CaptionGenerator.predict(os.path.join(app.config["UPLOAD_DIRECTORY"], filename), context, 3)
    current_time = time.time() - current_time

    if resulting_caption:
        time_string = "Inference Time: {:.2f}s".format(current_time)
        context = f"Context: {context}" if context else None

        filepath = "uploads/" + filename
        return render_template("result.html", image=filepath, captions=resulting_caption, time_string=time_string, context=context)


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
    app.run(debug=False)
