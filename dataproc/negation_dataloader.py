import csv
import re
from base_dataloader import BaseDataLoader


class DataLoaderNegation(BaseDataLoader):
    """
    Data loader for negation dataset.
    """

    def __init__(self, corpus_path, label_path=None) -> None:
        super().__init__(data_path=corpus_path)
        self.corpus_path = corpus_path
        self.label_path = label_path
        self.data = None
        self.labels = None
        self.read_data()
        
    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        with open(input_file, "r", encoding="utf-8-sig") as f:
            return list(csv.reader(f, delimiter="\t", quotechar=quotechar))

    def read_data(self):
        """
        Read the raw data from files.
        """
        # Read corpus
        lines_example = self._read_tsv(self.corpus_path)
        self.data = [' '.join(line) for line in lines_example]
        
        # Read labels if available
        if self.label_path is not None:
            lines_label = self._read_tsv(self.label_path)
            self.labels = [int(line[0]) for line in lines_label]

    @staticmethod
    def extract_event(s):
        pattern = r'<e>(.*?)<\/e>'
        event = re.findall(pattern, s)
        assert len(event) == 1
        return event[0]

    def get_processed_data(self):
        """Generate question-answering examples from the data."""
        template = "Is event <e> {} </e> negated by the context?"
        questions_with_context = []
        
        for context in self.data:
            event = self.extract_event(context)
            question = template.format(event)
            questions_with_context.append(f"question: {question}  context: {context}")

        if self.labels is not None:
            answers = ["Yes" if label == 1 else "No" for label in self.labels]
            return questions_with_context, answers
        else:
            return questions_with_context
