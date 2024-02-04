import os
from flask import Flask, render_template, request, abort, redirect, url_for
from flask_wtf import FlaskForm
from wtforms import SubmitField, SelectField, FileField, TextAreaField
from wtforms.validators import InputRequired
from werkzeug.utils import secure_filename
from dotenv import load_dotenv


# Instantiates Flask Application
# Run the app by typing 'flask run' in the virtual environment terminal
app = Flask(__name__)

# Loads dot env variables
load_dotenv()
# Sets secret key for form validation (CSRF)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY")
uploads_directory = "static/uploads"


class MainForm(FlaskForm):
    select_file = FileField("File", validators=[InputRequired()])
    text_field = TextAreaField("Creative Tuning")
    submit = SubmitField("Generate Caption")

#STYLESHEET: <link rel="stylesheet" href="{{ url_for('static', filename='css/page.css') }}">
# Main Webpage
# Contains basic information about the project
@app.route("/", methods=["GET", "POST"])
def home():
    form = MainForm()
    if form.validate_on_submit():
        file = form.select_file.data
        context = form.text_field.data
        filename = secure_filename(file.filename)
        file.save(uploads_directory + filename)

        return "File has been uploaded."
    return render_template("home.html", form=form)


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
