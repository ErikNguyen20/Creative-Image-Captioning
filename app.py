import os
from flask import Flask, render_template, session, request, abort, redirect, url_for, flash
from flask_wtf import FlaskForm
from flask_wtf.file import FileRequired, FileField
from wtforms import SubmitField, TextAreaField
from wtforms.validators import InputRequired, ValidationError
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# Example loading
# https://github.com/devtonic-net/flask-loading-app-message-and-spinner/blob/main/main.py


# Instantiates Flask Application
# Run the app by typing 'flask run' in the virtual environment terminal
app = Flask(__name__)

# Loads dot env variables
load_dotenv()
# Sets secret key for form validation (CSRF)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY")

# Configures file upload configurations
if not os.path.exists("static"):
    os.mkdir("static")
elif not os.path.exists(os.path.join("static", "uploads")):
    os.mkdir(os.path.join("static", "uploads"))
app.config["UPLOAD_DIRECTORY"] = "static/uploads"
app.config["MAX_UPLOAD_SIZE"] = 1048576  # 2 MB


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

        return redirect("/result")
    return render_template("home.html", form=form, flash_message=None)


@app.route("/result")
def result():
    filename = session.get("filename", None)
    if filename:
        return f"{filename} has been uploaded!"
        # filepath = os.path.join(app.config["UPLOAD_DIRECTORY"], filename)


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
