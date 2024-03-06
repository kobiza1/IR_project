import search_frontend as se
from pyngrok import ngrok
from nltk.corpus import stopwords
import nltk

# Download stopwords resource
nltk.download('stopwords')

english_stopwords = frozenset(stopwords.words("english"))
ngrok.set_auth_token("2d9SiyKzJhyFceSoCw6IPeovnYa_2njdFfvkSTwfsoaPFCCrN")
public_url = ngrok.connect("5000").public_url
print(public_url)
# Update any base URLs to use the public ngrok URL
se.app.config["BASE_URL"] = public_url
se.app.run()
