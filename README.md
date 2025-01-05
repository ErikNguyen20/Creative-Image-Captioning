# Senior Capstone Project
 CPSC 491

## Installation

To get started with this web application, follow these steps:

1. Ensure that you have [Python >= 3.9](https://www.python.org/downloads/) installed.

2. Clone this repository to a directory on your local machine:

   ```
   mkdir captioncrafterdir
   cd captioncrafterdir
   https://github.com/ErikNguyen20/Creative-Image-Captioning.git
   ```

3. Create and activate a Python virtual environment:

   ```
   python -m venv .venv
   .venv\Scripts\activate
   ```

4. Install the required Python packages from `requirements.txt`:

   ```
   pip install -r requirements.txt
   ```
   
5. Create an `.env` file and create your secret key, add your [OpenAI API Key](https://platform.openai.com/), and optional [Ngrok Key](https://ngrok.com/):

   ```
   SECRET_KEY = could_be_any_custom_secret_key_here
   OPENAI_API_KEY = <INSERT OPENAI API KEY HERE>
   GROK_KEY = <(OPTIONAL) INSERT NGROK AUTH KEY HERE>
   ```

6. Launch/run the web application using the flask command. Copy or Click on the URL printed in the terminal window to navigate to the web page:

   ```
   flask run
   ```
