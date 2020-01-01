FROM python:3.7.3

# Upgrade pip
RUN pip install --upgrade pip

# Install NLTK [REMOVE]
RUN pip install -U nltk==3.4.1
RUN python -m nltk.downloader rslp
RUN python -m nltk.downloader punkt

# install Spacy
RUN pip install -U spacy==2.1.4
RUN pip install -U spacy-lookups-data
RUN python -m spacy download es_core_news_md
RUN python -m spacy link es_core_news_md es
RUN python -m spacy download en_core_web_md
RUN python -m spacy link en_core_web_md en

# Install Gensim
RUN pip install -U gensim==3.7.3

# Install NLTK
RUN pip install -U nltk==3.4.1
RUN python -m nltk.downloader rslp
RUN python -m nltk.downloader punkt
RUN python -m nltk.downloader stopwords
# Other
RUN pip install gitpython

WORKDIR /usr/src/dataset

