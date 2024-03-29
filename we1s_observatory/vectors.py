"""vectors.py."""

import json
import os
import sys
from collections import Counter

import regex as re

from we1s_observatory.libs import regex

# Avoid loading spaCy if we already have it loaded
if ("spacy" not in sys.modules) and ("spacy" not in dir()):
    import spacy
    from spacy.tokenizer import Tokenizer


class Vectors:
    """Configure a vectors object."""

    def __init__(
        self,
        index,
        manifest_path,
        vectors_path,
        model="en_core_web_sm",
        stoplist="we1s_standard_stoplist.txt",
        log_file="log.txt",
    ):
        """Initialize the class.

        Note: The default model should be changed to 'en_core_web_lg'
        in the production environment.

        """
        self.log = ""
        # Get stoplist
        with open(stoplist, "r", encoding="utf-8") as f:
            self.stoplist = f.read().split("\n")
        # Read the manifest
        try:
            self.manifest = self.read_manifest(manifest_path)
            # Get the index and filename
            self.index = str(index)
            self.filename = self.manifest["name"] + ".json"
            # Get the Bag of Words
            if "bag_of_words" in self.manifest:
                self.bag = self.manifest["bag_of_words"]
            elif "features" in self.manifest:
                self.tokens = [feature[0] for feature in self.manifest["features"][1:]]
                self.bag = self.bagify()
            else:
                # Load the language model with custom tokenizer and entity merger
                self.nlp = spacy.load(model)
                self.nlp.tokenizer = self.custom_tokenizer()
                self.nlp.add_pipe(self.skip_ents, after="ner")
                # Create a spaCy document, extract tokens, then bagify
                self.doc = self.nlp(self.manifest["content"])
                self.tokens = self.get_tokens()
                self.bag = self.bagify()
            # Create a row of vectors and save it to the vectors file
            # self.vectors = self.get_vectors()
            self.vector_sequence = self.get_vector_sequence()
            self.vectors_path = vectors_path
            # self.save()
        except RuntimeError:
            pass
        if self.log is not "":
            with open(log_file, "a") as f:
                f.write(self.log)

    def read_manifest(self, filepath):
        """Read the manifest file."""
        try:
            with open(filepath, "rb") as f:
                return json.loads(f.read())
        except IOError:
            self.log += filepath + ",Could not read file.\n"

    def custom_tokenizer(self):
        """Add custom tokenizer settings."""
        return Tokenizer(
            self.nlp.vocab,
            prefix_search=regex.PREFIX_RE.search,
            suffix_search=regex.SUFFIX_RE.search,
            infix_finditer=regex.INFIX_RE.finditer,
            token_match=regex.SIMPLE_URL_RE.match,
        )

    # Custom entity merging filter
    def skip_ents(self, skip=["CARDINAL", "DATE", "QUANTITY", "TIME"]):
        """Duplicate spaCy's ner pipe, but with additional filters.

        Parameters:
        - doc (Doc): The Doc object.
        - ignore (list): A list of spaCy ner categories to ignore (e.g. DATE) when merging entities.

        # RETURNS (Doc): The Doc object with merged entities.

        """
        # Match months
        months = re.compile(
            r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sept(?:ember)?|oct(?:ober)?|nov(?:ember)?|Dec(?:ember)?)"
        )
        with self.doc.retokenize() as retokenizer:
            for ent in self.doc.ents:
                merge = True
                if ent.label_ in skip:
                    merge = False
                if ent.label_ == "DATE" and re.match(months, ent.text.lower()):
                    merge = True
                if merge == True:
                    attrs = {
                        "tag": ent.root.tag,
                        "dep": ent.root.dep,
                        "ent_type": ent.label,
                    }
                    retokenizer.merge(ent, attrs=attrs)
        # return doc

    def get_tokens(self):
        """Return a list of tokens."""
        return [token.text for token in self.doc]

    def bagify(self, trim_punct=True):
        """Convert a list of values to a dict of value frequencies.

        Parameters:
        - trim_punct: If True, strips attached punctuation that may have survived tokenisation.

        """
        # An attempt to strip predictably meaningless stray punctuation
        punct = re.compile(r"\.\W|\W\.|^[\!\?\(\),;:\[\]\{\}]|[\!\?\(\),;:\[\]\{\}]$")
        # Make sure we are working with a list of values
        if trim_punct == True:
            tokens = [re.sub(punct, "", token) for token in self.tokens]
        else:
            tokens = [token for token in self.tokens]
        return dict(Counter(tokens))

    def get_vectors(self, strip_stopwords=True):
        """Convert a dictionary bag of words to MALLET vectors format.

        Note: This is deprecated code which probably won't work.

        Parameters:
        - strip_stopwords: Boolean to remove words from a custom list.

        """
        # Build a row containing the key-value pairs
        row = self.index + " " + self.filename + " "
        for k, v in self.bag.items():
            # Another check on stray punctuation
            if k.isalnum() == True:
                if strip_stopwords == False:
                    row += k.replace(" ", "_") + ":" + str(v) + " "
                elif strip_stopwords == True and k not in self.stoplist:
                    row += k.replace(" ", "_") + ":" + str(v) + " "
                else:
                    pass
        return row.strip()

    def get_vector_sequence(self, strip_stopwords=True):
        """Convert a dictionary bag of words to a sequence of terms based on term counts.

        Parameters:
        - strip_stopwords: Boolean to remove words from a custom list.

        """
        row = self.index + " " + self.filename + " "
        for k, v in self.bag.items():
            # Another check on stray punctuation
            if k.isalnum() == True:
                if strip_stopwords == False:
                    term = k.replace(" ", "_") + " "
                    terms = term * v
                    row += terms
                elif strip_stopwords == True and k not in self.stoplist:
                    term = k.replace(" ", "_") + " "
                    term = re.sub("the_|a_|an_", "", term)
                    terms = term * v
                    row += terms
                else:
                    pass
        return row.strip()

    def save(self):
        """Append the row to the vectors file."""
        with open(self.vectors_path, "a", encoding="utf-8") as f:
            f.write(self.vector_sequence.strip() + "\n")


def vectorize_dir(json_directory, vectors_file, model, stoplist, log_file):
    """Vectorize a directory of json manifests.

    Sample usage:
    vectorize_dir('caches/json',
                  'model/vectors.txt',
                  'en_core_web_lg',
                  stoplist='libs/vectors/we1s_standard_stoplist.txt'
                  logfile='libs/vectors/log.txt;
                  )

    """
    # Delete pre-existing file
    if os.path.isfile(vectors_file):
        os.remove(vectors_file)
    files = sorted(
        file for file in os.listdir(json_directory) if file.endswith(".json")
    )
    for i, file in enumerate(files):
        file = json_directory + "/" + file
        vectors = Vectors(i, file, vectors_file, model, stoplist=stoplist)
        vectors.save()
