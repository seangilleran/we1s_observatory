import csv
import json
import os
import time
from collections import Counter

import fire
import pandas as pd
import regex as re
import spacy
from nltk.util import ngrams
from spacy.symbols import LEMMA, ORTH, POS, TAG
from spacy.tokenizer import Tokenizer

from we1s_observatory.document import Document
from we1s_observatory.libs import regex


class Preprocessor:
    """Configure a preprocessor object."""

    def __init__(
        self,
        model="en_core_web_sm",
        sources_csv=None,
        wikifier_output_dir="",
        max_length=3000000,
    ):
        """Initialize the preprocessor."""

        # Save wikifier option
        self.wikifier_output_dir = wikifier_output_dir

        # Load the language model
        # print('Preparing language model...')
        self.nlp = spacy.load(model)
        self.nlp.max_length = max_length

        # Import readability
        # print('Testing readability...')
        try:
            from spacy_readability import Readability

            self.collect_readability_scores = True
        except:
            msg = """The spacy-readability module is not installed on your system.
            Readability scores will be unavailable unless you `pip install spacy-_readability`."""
            # print(msg)
            self.collect_readability_scores = False
            pass

        # Configure language model options
        self.add_stopwords = []
        self.remove_stopwords = []
        self.skip_entities = ["CARDINAL", "DATE (except months)", "QUANTITY", "TIME"]
        self.lemmatization_cases = {
            "humanities": [
                {ORTH: u"humanities", LEMMA: u"humanities", POS: u"NOUN", TAG: u"NNS"}
            ]
        }

        # Configure entity categories to be skipped when merging entities
        self.options = {
            "merge_noun_chunks": False,
            "merge_subtokens": False,
            "skip_ents": self.skip_entities,
            "collect_readability_scores": self.collect_readability_scores,
        }

        # Handle lemmatisation exceptions
        for k, v in self.lemmatization_cases.items():
            self.nlp.tokenizer.add_special_case(k, v)

        # Add and remove custom stop words - disabled for optimisation
        # for word in self.add_stopwords:
        #     self.nlp.vocab[word].is_stop = True
        # for word in self.remove_stopwords:
        #     self.nlp.vocab[word].is_stop = False

        self.nlp.add_pipe(self.skip_ents, after="ner")

        # Add readability to pipeline
        if self.collect_readability_scores == True:
            self.nlp.add_pipe(Readability())

        # Load the sources file - disabled for optimisation
        self.sources = ""
        if sources_csv:
            with open(sources_csv, "r") as f:
                self.sources = [dict(line) for line in csv.DictReader(f)]

    # Add Custom Tokenizer if needed
    def custom_tokenizer(self, nlp):
        """Add custom tokenizer settings."""
        # nlp.tokenizer = custom_tokenizer(nlp)
        return Tokenizer(
            nlp.vocab,
            prefix_search=regex.PREFIX_RE.search,
            suffix_search=regex.SUFFIX_RE.search,
            infix_finditer=regex.INFIX_RE.finditer,
            token_match=regex.SIMPLE_URL_RE.match,
        )

    # Custom entity merging filter
    def skip_ents(self, doc, skip=["CARDINAL", "DATE", "QUANTITY", "TIME"]):
        """Duplicate spaCy's ner pipe, but with additional filters.

        Parameters:
        - doc (Doc): The Doc object.
        - ignore (list): A list of spaCy ner categories to ignore (e.g. DATE) when merging entities.

        RETURNS (Doc): The Doc object with merged entities.

        """
        # Match months
        with doc.retokenize() as retokenizer:
            for ent in doc.ents:
                merge = True
                if ent.label_ in skip:
                    merge = False
                if ent.label_ == "DATE" and re.match(regex.MONTHS_RE, ent.text.lower()):
                    merge = True
                if merge == True:
                    attrs = {
                        "tag": ent.root.tag,
                        "dep": ent.root.dep,
                        "ent_type": ent.label,
                    }
                    retokenizer.merge(ent, attrs=attrs)
        return doc

    # Not part of the Document class for ease of access.
    # Create bags as separate dicts and then save them to the manifest.
    def bagify(self, series, trim_punct=True, as_counter=False):
        """Convert a list of values to a dict of value frequencies.

        Parameters:
        - trim_punct: If True, strips attached punctuation that may have survived tokenisation.
        - as_counter: If True, returns a Python Counter object enabling its most_common() method.

        """
        # An attempt to strip predictably meaningless stray punctuation
        punct = re.compile(r"\.\W|\W\.|^[\!\?\(\),;:\[\]\{\}]|[\!\?\(\),;:\[\]\{\}]$")
        # Make sure we are working with a list of values
        if isinstance(series, pd.DataFrame):
            print("Please select only one columns from the dataframe.")
        if isinstance(series, pd.Series):
            if trim_punct == True:
                series = [re.sub(punct, "", term) for term in list(series.values)]
            else:
                series = [term for term in list(series.values)]
        if as_counter == True:
            return Counter(series)
        else:
            return dict(Counter(series))

    def preprocess_dir(self, manifest_dir, content_property, kwargs=None):
        """Walk through a directory of folders and preprocess the json files.

        Parameters:
        - content_property: The manifest property to be used as the source of the content.
        - add_properties: A list of properties to add to the manifest. Default is `None`.
          If the property includes options the values should be separated by colons (e.g.
          `ngrams:2` for bigrams or `stems:snowball` for the Snowball stemmer.)
        - remove_properties: A list of properties to remove from the manifest. Default is `None`.
        - kwargs: A dict of options to pass to the main preprocessing function.

        """
        # Start the timer
        start = time.time()
        # Walk the directory and preprocess each file
        all_files = [
            os.path.join(r, file)
            for r, d, f in os.walk(manifest_dir)
            for file in f
            if file.endswith(".json") and not file.startswith("._")
        ]
        for file in all_files:
            file = file.replace("\\", "/")  # Handle Windows paths
            tmp = file.split("/")
            path = "/".join(tmp[:-1])
            filename = tmp[-1]
            self.preprocess(path, filename, content_property, kwargs=None)
        # Print time to completion
        end = time.time()
        t = end - start
        print("Processed all files in " + str(t) + " seconds.")

    def preprocess_file(self, manifest_dir, filename, content_property, kwargs=None):
        """Preprocess a specific json file.

        Parameters:
        - content_property: The manifest property to be used as the source of the content.
        - add_properties: A list of properties to add to the manifest. Default is `None`.
          If the property includes options the values should be separated by colons (e.g.
          `ngrams:2` for bigrams or `stems:snowball` for the Snowball stemmer.)
        - remove_properties: A list of properties to remove from the manifest. Default is `None`.
        - kwargs: A dict of options to pass to the main preprocessing function.

        """
        self.preprocess(manifest_dir, filename, content_property, kwargs)

    def preprocess(
        self,
        manifest_dir,
        filename,
        content_property,
        kwargs=None,
        add_properties=None,
        remove_properties=None,
        ppversion="0.1",
    ):
        """Start the main preprocessing function."""

        # Start doc timer
        doc_start = time.time()

        # Initialise the Document object
        try:
            doc = Document(
                manifest_dir,
                filename,
                content_property=content_property,
                model=self.nlp,
                kwargs=kwargs,
            )
        except UnicodeDecodeError as err:
            print("Document failed:", filename, ":", err)
            return False

        # short-circuit and skip if JSON was already processed by version
        try:
            if doc.manifest_dict["ppversion"] == ppversion:
                return True
        except KeyError:
            doc.manifest_dict["ppversion"] = ppversion

        # export the wikifier document if the directory is set
        if self.wikifier_output_dir:
            doc.export_content(output_dir=self.wikifier_output_dir)

        # Remove manifest properties if the remove_properties list is submitted
        if remove_properties is not None:
            doc.remove_property(remove_properties, save=False)

        # Sort and serialise the features table
        features = doc.get_features()
        features.sort_values(by=["TOKEN"], inplace=True)
        features_list = json.loads(pd.DataFrame.to_json(features, orient="values"))
        features_list.insert(0, list(features.columns))
        doc.manifest_dict["features"] = features_list

        # Bagify the normed tokens (skipping punctuation and line breaks)
        # Attempt to remove stray punctuation
        punct = re.compile(r"\.\W|\W\.|^[\!\?\(\),;:\[\]\{\}]|[\!\?\(\),;:\[\]\{\}]$")
        filtered = [
            re.sub(punct, "", token.norm_)
            for token in doc.content
            if token.norm_ != "_"
            and token.is_punct != True
            and token.is_space != True
            and token.is_digit != True
        ]
        filtered = sorted(filtered)
        doc.manifest_dict["bag_of_words"] = dict(Counter(filtered))

        # Add any additional properties to the manifest:
        if add_properties is not None:
            for property in add_properties:
                if property == "lemmas":
                    doc.manifest_dict["lemmas"] = doc.lemmas(as_list=True)
                if property == "punctuation":
                    doc.manifest_dict["punctuation"] = doc.punctuation(as_list=True)
                if property == "pos":
                    doc.manifest_dict["pos"] = doc.pos(as_list=True)
                if property == "tags":
                    doc.manifest_dict["tags"] = doc.tags(as_list=True)
                if property.startswith("stems"):
                    options = property.split(":")
                    doc.manifest_dict["stems"] = doc.stems(
                        stemmer=options[1], as_list=True
                    )
                if property.startswith("ngrams"):
                    doc.manifest_dict["ngrams"] = doc.ngrams(n=options[1], as_list=True)

        # Add the readability scores to the manifest
        doc.manifest_dict["readability_scores"] = doc.readability_scores(as_list=True)[
            0
        ]

        # Add the total word count (skipping punctuation and line breaks) to the manifest
        doc.manifest_dict["word_count"] = len(
            doc.filter(
                column="TOKEN",
                skip_punct=True,
                skip_stopwords=False,
                skip_linebreaks=True,
            )
        )

        # Add the country in which the document was published
        if self.sources:
            doc.manifest_dict["country"] = [
                x for x in self.sources if x["source_title"] == doc.manifest_dict["pub"]
            ][0]["country"]

        # Add language model metadata
        doc.manifest_dict["language_model"] = self.nlp.meta
        custom = {
            "linebreak_regex": str(regex.LINEBREAK_REGEX),
            "nonbreak_regex": str(regex.NONBREAKING_SPACE_REGEX),
            "prefix_re": str(regex.PREFIX_RE),
            "suffix_re": str(regex.SUFFIX_RE),
            "infix_re": str(regex.INFIX_RE),
            "simple_url_re": str(regex.SIMPLE_URL_RE),
            "add_stopwords": self.add_stopwords,
            "remove_stopwords": self.remove_stopwords,
            "lemmatization_cases": self.lemmatization_cases,
            "skip_entities": self.skip_entities,
        }
        doc.manifest_dict["language_model"]["custom"] = custom

        # Save the changes to the manifest
        with open(doc.manifest_filepath, "w", encoding="utf-8") as f:
            f.write(json.dumps(doc.manifest_dict))

        # Print time to completion
        doc_end = time.time()
        doc_t = doc_end - doc_start
        # print('Processed ' + doc.manifest_filepath + ' in ' + str(doc_t) + ' seconds.')


# def main(**kwargs):
#     """Capture the command line arguments and fire the preprocessor."""
#     pp = Preprocessor()
#
#     add_properties = None
#     remove_properties = None
#     # Because we don't want to pass a long dict on the command line, we set options here.
#     options = {'merge_noun_chunks': False, 'merge_subtokens': False, 'collect_readability_scores': pp.collect_readability_scores}
#     try:
#         manifest_dir = kwargs['path']
#     except:
#         raise KeyError('Please supply a directory path with `--path`.')
#     if 'filename' in kwargs:
#         manifest_file = kwargs['filename']
#     else:
#         manifest_file = None
#     try:
#         content_property = kwargs['property']
#     except:
#         raise KeyError("Please supply a JSON property where the document's content is found with `--property`.")
#     if 'add_properties' in kwargs and kwargs['add_properties'] is not None:
#         try:
#             if isinstance(kwargs['add_properties'], tuple):
#                 add_properties = list(kwargs['add_properties'])
#             else:
#                 add_properties = [kwargs['add_properties']]
#             if len(add_properties) == 0:
#                 add_properties = None
#         except:
#             raise ValueError('The `add-properties` parameter must contain multiple values separated by commas.')
#     if 'remove_properties' in kwargs and kwargs['remove_properties'] is not None:
#         try:
#             if isinstance(kwargs['remove_properties'], tuple):
#                 remove_properties = list(kwargs['remove_properties'])
#             else:
#                 remove_properties = [kwargs['remove_properties']]
#             if len(remove_properties) == 0:
#                 remove_properties = None
#         except:
#             raise ValueError('The `remove-properties` parameter must contain multiple values separated by commas.')
#
#     if manifest_file is not None:
#         print('Starting preprocessing...')
#         pp.preprocess_file(manifest_dir, manifest_file, content_property, kwargs=options)
#         print('Finished preprocessing file.')
#     else:
#         print('Starting preprocessing...')
#         pp.preprocess_dir(manifest_dir, content_property, kwargs=options)
#         print('Finished preprocessing directory.')
#
#
# if __name__ == '__main__':
#   fire.Fire(main)
