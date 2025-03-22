from abc import ABC, abstractmethod


class BaseDataLoader(ABC):
    """
    Abstract base class for data loaders.
    """

    def __init__(self, data_path, **kwargs) -> None:
        """
        Initialize the data loader.
        
        Args:
            data_path: Path to the main data file
            **kwargs: Additional parameters specific to each data loader
        """
        self.data_path = data_path
        self.kwargs = kwargs

    @abstractmethod
    def read_data(self):
        """
        Read the raw data from files.
        """
        pass

    @abstractmethod
    def get_processed_data(self):
        """
        Process the data and return the processed output.
        This method should return the data in the final format needed for the task.
        """
        pass