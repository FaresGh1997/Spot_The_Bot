# Spot-the-Bot

Spotting Bot-written text for 5 languages (Arabic , English, Russian, Thai and Atikamekw ) using graph theory and semantic paths Analysis.

[Additional resources](https://drive.google.com/drive/folders/1UlUlQJD7eCQFL42PyB8tkrx2mkOGm2WQ?usp=sharing)

### The problem:
Bots can be used a variety of purposes, from amplifying propaganda to spreading misinformation.

### The solution:
The purpose of this project is to develop a sufficient method for generic bot detection task that uses complex networks and word embedding techniques to capture semantic patterns across different languages. The approach is based on the idea that the structure of a text can reveals information about its authorship if the text is represented as paths on a graph of the language.

### The Tasks:

1. Data collection: Gathering a diverse range of texts which represent the language usage for Thai and Atikamekw languages.
2. Data Preprocessing: Tokenization, removing stop words and lemmatization for Thai and Atikamekw languages.
3. Vectorization and Word-embeddings: using two approaches
   - TF-IDF (term frequency- inverse document frequency) + SVD (single value decomposition).
   - CBOW (continues bag of words).
4. Graph Construction: construct Gabriel graphs which represents the semantic structures of a language. Two graphs for each language, one from SVD embeddings, the other from CBOW embeddings.
5. Bot side and Language model text: training deep learning language models for each language and use these models to generate bot text, which would be used in the comparison with the human text to validate the efficiency of the approach.
6. Analysis of the approach: spotting bot text using the graph and characteristics of human and bot text when represented as a path on those graphs.


### The Findings:
- graphs can be used as an effective way for spotting bot generated text by analyzing path characteristics on language graphs (accuracy 97.81% using weighted path length characteristics).
- Datasets and tools for Thai and Atikamekw languages have been produced in this work, such as cleaned corpuses, lemmatizer and language models.
- In general, CBOW-based graphs are sparser and capture less general structural information than their SVD counterparts.
- Weighted path length analysis shows that bots text yield higher weighted path length score than humans.
- Closeness and betweenness centralities analysis show higher scores for human texts in comparison to the bot text.


# Further Details:

## Thai Language:

1. we start by literature extraction from websites for Thai language, multiple links for files were extracted from multiple websites [(code)](https://github.com/faresGh97/Spot-the-Bot/blob/main/Thai/Thai_Text_Extraction.ipynb), mainly links for Thai literature from websites where extracted from [libgen](https://libgen.is/search.php?req=thai&open=0&res=25&view=simple&phrase=1&column=language) and from [archive](https://archive.org/details/booksbylanguage_thai?&sort=-week&page=126), those links represents literature in multiple files format (txt or Pdf).

2. we download literature files associated with the extracted links mentioned in the previous step [(code)](https://github.com/faresGh97/Spot-the-Bot/blob/main/Thai/Thai_Text.ipynb), in this step around 6000 literature text were prepared in order to be used in the next step which is cleaning and stemming.

3. in this step [(code)](https://github.com/faresGh97/Spot-the-Bot/blob/main/Thai/Thai_preprocessing.ipynb), cleaning for the extracted files from the previous text has been done based on the multiple criteria’s, first, we start by removing the unwanted text tokens from the extracted text (foreign symbols, numbers, punctuations). Second, we preform stop-words removal. Third: we preform lemmatization: since Thai language does not have the concept of spaces to separate the words in a sentence, rather a Thai speaker can understand Thai text based on the context, and also Thai language does not have the concept of prefixes and suffixes to differentiate the time of the verbs for example, hence, we decided to preform word tokenization task instead, multiple models were tested to preform word tokenization (pythainlp , deepcut, attcut), we decided to use [attcut](https://pythainlp.github.io/attacut/overview.html) library to preform word tokenization  , which is a transformer model that achieved a fast and solid output when concerned about word tokenization task for Thai language, hence for each file we extracted from the step above, a processed file was generated that contains tokens extracted from the base file separated by white spaces. Finally, we generate a clean Corpus file that contains in each line the title of the literature file and the tokens extracted from the literature file in one line, meaning one line in the Corpus = one literature file for each file from the 6000 literatures extracted in the previous step.

4. we preform TF-IDF matrixes and then SVDs matrixes in order to get the final embeddings of words in Thai [(code)](https://github.com/faresGh97/Spot-the-Bot/blob/main/Thai/Thai_Embeddings.ipynb) based on the clean corpus file extracted from the step above. Multiple matrixes have been extracted in this step.


## Atikamekw Language:

1- since this language is a tribal language in north America region (in Montreal, Canada specifically), there was no literature to be found for this language, thus we extracted Wikipedia articles for this languages, thus for each article in [Atikamekw wikipedia,](https://atj.wikipedia.org/wiki/Kotakahi:Toutes_les_pages) a text file representing the content of this article has been extracted [(code)](https://github.com/faresGh97/Spot-the-Bot/blob/main/Atikamekw/Atikamekw_Text_Extraction.ipynb), which resulted in 1583 files that contains Atikamekw text.

2- cleaning and lemmatization of the text has been done in this step [(code)](https://github.com/faresGh97/Spot-the-Bot/blob/main/Atikamekw/Atikamekw%20Limma.ipynb). First, we preform text cleaning the same way we did for Thai language (without the stop-words removal step), then we preform lemmatization, for the same reasons mentioned above we couldn’t find a lemmatization tool for this language, hence a self-made lemmatization class has been made based on the lexicon literature and websites that explained the grammatical rules for this language, we mentioned:
 - Atikamekw Morphology and Lexicon, 1978, Beland, Jean Pierre [eng]
 - Manuel D'Initation a la Langue Atikamekw, 2020, Cercle Kisis [frn]
 - [verbs conjunction](https://verbes.atikamekw.atlas-ling.ca/) [frn]

after lemmatization a preprocessed text file for each of the articles mentioned above was generated, and a corpus file from the processed files was generated.

3- we preform TF-IDF matrixes and then SVDs matrixes in order to get the final embeddings of words in Atikamekw [(code)](https://github.com/faresGh97/Spot-the-Bot/blob/main/Atikamekw/Atikamekw_embeddings.ipynb) based on the clean corpus file extracted from the step above. Multiple matrixes have been extracted in this step.

## Graph construction for the 5 languages:

after obtaining the vectorized corpuses for the 5 languages, 2 Gabriel Graphs were constructed for each language. First a Delaunay triangulation was generated for the each vectorized corpus in parallel, and then using a from-scratch code, fully connected weighted undirected Gabriel Graphs are constructed for each Delaunay triangulations.

## Bot implementation:
GPT (Generative Pre-trained Transformer) based models were used to represent bots in this study. various models were used to ensure the generic of the solution (GPT2 , GPT3 , mGPT ....).

for Atikamekw language: since there are no text generation models for this language, from scratch GPT-2 model was trained to represent the bot in this project.


Skills developed: Scipy | Huggingface | Tensorflow | numpy | GPT-based models | Classification | networkx | Data Processing | Feature engineering | Dimensionality Reduction | igraph | python | matplotlib | Data Visualization | web scraping | model finetuning | Network Science | NLP.
