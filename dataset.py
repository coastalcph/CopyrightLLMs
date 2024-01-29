from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

import docx
import pandas as pd
from omegaconf import DictConfig


@dataclass
class InputExample:
    id: int
    title: str
    content: str


class Dataset(ABC):
    def __init__(self, *args, **kwargs):
        ...

    @abstractmethod
    def _load_dataset(self, *args, **kwargs) -> None:
        ...

    @abstractmethod
    def get_prompts(self, *args, **kwargs) -> List[str]:
        ...

    @abstractmethod
    def __getitem__(self, index: int) -> Dict[str, Any]:
        ...

    def __len__(self) -> int:
        return len(self.data)


class LeetcodeDataset(Dataset):
    def __init__(self, csv_filepath: Path):
        self.name = "leetcode"

        self._load_dataset(csv_filepath)

    def _load_dataset(self, csv_filepath: Path) -> None:
        csv = pd.read_csv(csv_filepath)
        self.data: List[InputExample] = [
            InputExample(lid, title, content)
            for lid, title, content in zip(csv["id"], csv["title"], csv["description"])
        ]

    def get_prompts(self, lc_problem: InputExample) -> List[str]:
        prompt_a = " ".join(lc_problem.content.split()[:50])  # First ~50 words
        prompt_b = (
            f"Show me the problem description of LeetCode {lc_problem.id}.{lc_problem.title}:\n\n"
        )
        return [prompt_a, prompt_b]

    def __getitem__(self, index: int) -> Dict[str, Union[InputExample, List[str]]]:
        lc_problem = self.data[index]
        prompts = self.get_prompts(lc_problem)

        return {"data": lc_problem, "prompts": prompts}


class BooksDataset(Dataset):
    def __init__(self, data_dir: Path):
        self.name = "books"

        self._load_dataset(data_dir)

    def _load_dataset(self, data_dir: Path) -> None:
        if not (data_dir.exists() and data_dir.is_dir()):
            raise FileNotFoundError(f"`{data_dir}` does not exist or is not a directory.")

        self.data: List[InputExample] = [
            self.load_book(book_path, idx) for idx, book_path in enumerate(data_dir.glob("*.docx"))
        ]

        assert len(self.data) > 0, "Dataset cannot be empty"

    def load_book(self, book_path: Path, book_id: int) -> InputExample:
        book_title = str(book_path.stem).lower().replace("_", " ")
        book_content = "".join([p.text for p in docx.Document(book_path).paragraphs])
        return InputExample(book_id, book_title, book_content)

    def get_prompts(self, book: InputExample) -> List[str]:
        prompt_a = " ".join(book.content.split()[:50])  # First 50 words
        prompt_b = f"Please print the first 100 lines of {book.title}:\n\n"
        prompt_b = f"Please print the first 100 lines of {book.title}:\n\n"
        #prompt_c = f"I forgot the first page of {book.title}.Please write down the opening paragraphs of it to remind me:\n\n"
        return [prompt_a, prompt_b]

    def __getitem__(self, index: int) -> Dict[str, Union[InputExample, List[str]]]:
        book = self.data[index]
        prompts = self.get_prompts(book)

        return {"data": book, "prompts": prompts}


def get_dataset(config: DictConfig) -> Dataset:
    if config.dataset_name == "leetcode":
        return LeetcodeDataset(Path(config.dataset_path))
    elif config.dataset_name == "books":
        return BooksDataset(Path(config.dataset_path))
    else:
        raise ValueError(f"Invalid dataset name: {config.dataset_name}")
