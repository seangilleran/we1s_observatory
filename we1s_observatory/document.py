import json
import os

import pandas as pd
import spacy
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.util import ngrams
from spacy.symbols import LEMMA, POS, TAG

from we1s_observatory.libs import clean, regex

# The Document class
class Document:
    """Model a document's features.

    Parameters:
    - manifest_dir: the path to the manifest directory
    - manifest_file: the name of the manifest file.
    - content_property: the name of the property from which to extract the content
    Returns a dataframe.
    """

    def __init__(self, manifest_dir, manifest_file, content_property, model, **kwargs):
        """Initialize the object."""
        self.nlp = model
        self.manifest_filepath = os.path.join(manifest_dir, manifest_file)
        self.content_property = content_property
        self.manifest_dict = self._read_manifest()
        self.doc_string = clean.scrub(self._get_docstring(content_property))
        self.content = self.nlp(self.doc_string)
        # self.options = kwargs['kwargs']
        self.options = kwargs
        # Re-do this to deserialise a list of lists.
        if "features" in self.manifest_dict:
            # self.features = self.get_features()
            self.features = self.deserialize(json.dumps(self.manifest_dict["features"]))
        else:
            self.features = self.get_features()

    def _read_manifest(self):
        """Read a JSON file and return a Python dict."""
        with open(self.manifest_filepath, "r", encoding="utf-8") as f:
            return json.loads(f.read())

    def _get_docstring(self, content_property):
        """Extract a document string from a manifest property."""
        return self.manifest_dict[content_property]

    def get_features(self):
        """Process the document with the spaCy pipeline into a pandas dataframe.

        If `collect_readability_scores` is set, Flesch-Kincaid Readability,
        Flesch-Kincaid Reading Ease and Dale-Chall formula scores are collected
        in a tuple in that order. Other formulas are available (see
        https://github.com/mholtzscher/spacy_readability).

        Parameters:
        - as_list: Return the features as a list instead of a dataframe.

        """
        # Handle optional pipes - disabled for optimisation
        # if 'merge_noun_chunks' in self.options and self.options['merge_noun_chunks'] == True:
        #     merge_nps = self.nlp.create_pipe('merge_noun_chunks')
        #     self.nlp.add_pipe(merge_nps)
        # if 'merge_subtokens' in self.options and self.options['merge_subtokens'] == True:
        #     merge_subtok = self.nlp.create_pipe('merge_subtokens')
        #     self.nlp.add_pipe(merge_subtok)

        # Build the feature list
        feature_list = []
        columns = ["TOKEN", "NORM", "LEMMA", "POS", "TAG", "STOPWORD", "ENTITIES"]
        for token in self.content:
            # Get named entity info (I=Inside, O=Outside, B=Begin)
            ner = (token.ent_iob_, token.ent_type_)
            t = [
                token.text,
                token.norm_,
                token.lemma_,
                token.pos_,
                token.tag_,
                str(token.is_stop),
                ner,
            ]
            feature_list.append(tuple(t))
        return pd.DataFrame(feature_list, columns=columns)

    def filter(
        self,
        pattern=None,
        column="TOKEN",
        skip_punct=False,
        skip_stopwords=False,
        skip_linebreaks=False,
        case=True,
        flags=0,
        na=False,
        regex=True,
    ):
        """Return a new dataframe with filtered rows.

        Parameters:
        - pattern: The string or regex pattern on which to filter.
        - column: The column where the string is to be searched.
        - skip_punct: Do not include punctuation marks.
        - skip_stopwords: Do not include stopwords.
        - skip_linebreaks: Do not include linebreaks.
        - case: Perform a case-sensitive match.
        - flags: Regex flags.
        - na: Filler for empty cells.
        - regex: Set to True; otherwise absolute values will be matched.
        The last four parameters are from `pandas.Series.str.contains`.

        """
        # Filter based on column content
        new_df = self.features
        if pattern is not None:
            new_df = new_df[
                new_df[column].str.contains(
                    pattern, case=case, flags=flags, na=na, regex=regex
                )
            ]
        # Filter based on token type
        if skip_punct == True:
            new_df = new_df[
                ~new_df["POS"].str.contains(
                    "PUNCT", case=True, flags=0, na=False, regex=True
                )
            ]
        if skip_stopwords == True:
            new_df = new_df[
                ~new_df["STOPWORD"].str.contains(
                    "TRUE", case=False, flags=0, na=False, regex=True
                )
            ]
        if skip_linebreaks == True:
            new_df = new_df[
                ~new_df["POS"].str.contains(
                    "SPACE", case=True, flags=0, na=False, regex=True
                )
            ]
        return new_df

    def lemmas(self, as_list=False):
        """Return a dataframe containing just the lemmas."""
        if as_list == True:
            return [token.lemma_ for token in self.content]
        else:
            return pd.DataFrame(
                [token.lemma_ for token in self.content], columns=["LEMMA"]
            )

    def punctuation(self, as_list=False):
        """Return a dataframe containing just the punctuation marks."""
        if as_list == True:
            return [token.text for token in self.content if token.is_punct]
        else:
            return pd.DataFrame(
                [token.text for token in self.content if token.is_punct],
                columns=["PUNCTUATION"],
            )

    def pos(self, as_list=False):
        """Return a dataframe containing just the parts of speech."""
        if as_list == True:
            return [token.pos_ for token in self.content]
        else:
            return pd.DataFrame([token.pos_ for token in self.content], columns=["POS"])

    def tags(self, as_list=False):
        """Return a dataframe containing just the tags."""
        if as_list == True:
            return [token.tag_ for token in self.content]
        else:
            return pd.DataFrame([token.tag_ for token in self.content], columns=["TAG"])

    def entities(self, options=["text", "label"], as_list=False):
        """Return a dataframe containing just the entities from the document.

        Parameters:
        - options: a list of attributes ('text', 'start', 'end', 'label')
        - as_list: return the entities as a list of tuples.

        """
        ents = []
        for ent in self.content.ents:
            e = []
            if "text" in options:
                e.append(ent.text)
            if "start" in options:
                e.append(ent.start)
            if "end" in options:
                e.append(ent.end)
            if "label" in options:
                e.append(ent.label_)
            ents.append(tuple(e))
        if as_list == True:
            return ents
        else:
            return pd.DataFrame(ents, columns=[option.title() for option in options])

    def readability_scores(
        self,
        columns=[
            "Flesch-Kincaid Readability",
            "Flesch-Kincaid Reading Ease",
            "Dale-Chall",
        ],
        as_list=False,
    ):
        """Get a list of readability scores from the document.

        Parameters:
        - columns: a list of labels for the score types
        - as_df: return the list as a dataframe.

        """
        fkr = self.content._.flesch_kincaid_reading_ease
        fkg = self.content._.flesch_kincaid_grade_level
        dc = self.content._.dale_chall
        scores = [(fkr, fkg, dc)]
        if as_list == True:
            return scores
        else:
            return pd.DataFrame(scores, columns=columns)

    def stems(self, stemmer="porter", as_list=False):
        """Convert the tokens in a spaCy document to stems.

        Parameters:
        - stemmer: the stemming algorithm ('porter' or 'snowball').
        - as_list: return the dataframe as a list.

        """
        if stemmer == "snowball":
            stemmer = SnowballStemmer(language="english")
        else:
            stemmer = PorterStemmer()
        stems = [stemmer.stem(token.text) for token in self.content]
        if as_list == True:
            return stems
        else:
            return pd.DataFrame(stems, columns=["Stems"])

    def ngrams(self, n=2, as_list=False):
        """Convert the tokens in a spaCy document to ngrams.

        Parameters:
        - n: The number of tokens in an ngram.
        - as_list: return the dataframe as a list.

        """
        ngram_tokens = list(ngrams([token.text for token in self.content], n))
        if as_list == True:
            return ngram_tokens
        else:
            prefix = str(n) + "-"
            if n == 2:
                prefix = "Bi"
            if n == 3:
                prefix = "Tri"
            label = prefix + "grams"
            return pd.DataFrame({label: pd.Series(ngram_tokens)})

    def remove_property(self, properties, save=False):
        """Remove a property from the manifest.

        Parameters:
        - property: The property or a list of properties to be removed from the manifest.
        - save: Save the deletion to the manifest.

        """
        for property in properties:
            del self.manifest_dict[property]
        # Write the json to the manifest file
        # IMPORTANT: May not work if the manifest file has binary content
        with open(self.manifest_filepath, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.manifest_dict))

    def serialize(self, df, indent=None):
        """Serialize a dataframe as a list of lists with the column headers as the first element.

        Parameters:
        - indent: An integer indicating the number of spaces to indent the json string. Default is None.

        """
        j = json.loads(pd.DataFrame.to_json(df, orient="values"))
        j.insert(0, list(df.columns))
        return json.dumps(j, indent=indent)

    def deserialize(self, j):
        """Deserialize a list of lists to a dataframe using the first element as the headers."""
        df = pd.read_json(j, orient="values")
        headers = df.iloc[0]
        return pd.DataFrame(df.values[1:], columns=headers)

    def save(self, property=None, series=None, sort=False):
        """Convert a series of values and save them to the manifest file.

        Overwrites the original manifest file, so not to be used lightly.
        IMPORTANT: May not work if the manifest file has binary content.
        Parameters:
        - property: A string naming the JSON property to save to.
        - series: The list or dataframe to save.
        - sort: Alphabetically sort the series to lose token order.

        """
        with open(self.manifest_filepath, "w", encoding="utf-8") as f:
            if isinstance(series, dict) or isinstance(series, list):
                self.manifest_dict[property] = series
                f.write(json.dumps(self.manifest_dict))
            else:
                if sort == True:
                    col = list(series.columns)[0]
                    series.sort_values(by=[col], inplace=True)
                json_str = self.serialize(series)
                self.manifest_dict[property] = json.loads(json_str)
                f.write(json_str)

    def export_content(self, output_dir=None):
        """Save a copy of the content to the Wikifier text folder."""
        docstring = (
            self._get_docstring(self.content_property)
            .replace("\[\.\]", "")
            .replace("\\r\\n", "\n")
        )
        filename = os.path.basename(self.manifest_filepath).rsplit(".json")[0] + ".txt"
        output_filepath = os.path.join(output_dir, filename)
        # Or use this for multiple directories
        # output_dir = os.path.join(output_dir, data_dir)
        # output_filepath = os.path.join(output_dir, filename)
        # if not os.path.exists(output_dir):
        #     os.makedirs(output_dir)
        with open(output_filepath, "w", encoding="utf-8") as wf:
            wf.write(docstring)
