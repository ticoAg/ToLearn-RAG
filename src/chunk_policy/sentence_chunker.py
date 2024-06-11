import re
from pathlib import Path
from typing import List

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import TextSplitter
from pydantic import BaseModel


class SentenceObj(BaseModel):
    sentence: str  # 当前句子
    index: int  # 索引
    used: bool  # 是否在main_sent中用过
    combine: str  # 混合后的句子


class SentenceWisedTextSplitter:
    def __init__(
        self, sentences: List[SentenceObj], sentence_length: int, overlap_length: int
    ) -> None:
        """
        # Declare the strategy to combine sentences
        1. 分为三部分 prev + main + next
        2. 对于main, 每次叠加短句, 最大不得超过sentence_length * 1.2
        3. 对于prev,
        """
        self.sentences: List[SentenceObj] = sentences
        self.main_sentence_length = sentence_length
        self.overlap_length = overlap_length
        self.sent_lth = len(self.sentences)

    def _combine_main_sentence(self, idx, curr_obj: SentenceObj):
        main_sentence = ""
        # 先遍历最近的几句,组装最近几轮的主句
        main_sentence += curr_obj.sentence
        if len(main_sentence) >= self.main_sentence_length:
            curr_obj.used = True
        else:
            _idx = idx
            while True:
                _idx += 1
                if _idx >= self.sent_lth:
                    break
                next_obj: SentenceObj = self.sentences[_idx]
                if (
                    len(main_sentence + next_obj.sentence)
                    > 1.2 * self.main_sentence_length
                ):
                    break
                else:
                    main_sentence += next_obj.sentence + " "
                    next_obj.used = True
        return main_sentence

    def _combine_prev_sentence(self, idx):
        """前向拼接"""
        _idx = idx
        prev_overlap_sentence = ""
        while True:
            _idx -= 1
            if _idx < 0:
                break
            overlap_obj: SentenceObj = self.sentences[_idx]
            if len(prev_overlap_sentence) > 1.2 * self.overlap_length:
                break
            else:
                prev_overlap_sentence = (
                    overlap_obj.sentence + " " + prev_overlap_sentence
                )
        return prev_overlap_sentence

    def _combine_next_sentence(self, idx):
        """后向拼接"""
        _idx = idx
        next_overlap_sentence = ""
        while True:
            _idx += 1
            if _idx >= self.sent_lth:
                break
            next_overlap_obj: SentenceObj = self.sentences[_idx]
            if (
                len(next_overlap_sentence + next_overlap_obj.sentence)
                > 1.2 * self.overlap_length
            ):
                break
            else:
                next_overlap_sentence += " " + next_overlap_obj.sentence
        return next_overlap_sentence

    def _split(self):
        # 遍历每个句子字典
        for idx in range(self.sent_lth):
            curr_obj = self.sentences[idx]
            if curr_obj.used:
                continue

            main_sentence = self._combine_main_sentence(idx, curr_obj)
            prev_overlap_sentence = self._combine_prev_sentence(idx)
            next_overlap_sentence = self._combine_next_sentence(idx)

            # 更新组装后的句子 总长度不大于 1.2 * overlap_length + self.main_sentence_length + 1.2 * overlap_length
            curr_obj.combine = (
                prev_overlap_sentence + main_sentence + next_overlap_sentence
            )
        return [i.combine for i in self.sentences if i.combine]


if __name__ == "__main__":
    path = Path.cwd() / "docs" / "example_files" / "WizardLM_OCR_RESULT.txt"
    text = open(path, "r").read().replace("\n\n", " ")
    single_sentences_list = re.split(r"(?<=[.?!。？！])\s+", text)
    sentences = [
        SentenceObj(sentence=x, index=i, used=False, combine="")
        for i, x in enumerate(single_sentences_list)
    ]
    splitter = SentenceWisedTextSplitter(
        sentences, sentence_length=450, overlap_length=100
    )
    combined_sentences = splitter._split()
    print(combined_sentences[:10])
