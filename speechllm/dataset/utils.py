import copy
import re
from dataclasses import dataclass
from typing import Dict, Sequence

import torch
import transformers

import logging
from speechllm import conversation as conversation_lib
from speechllm.constants import IGNORE_INDEX, DEFAULT_AUDIO_TOKEN, DEFAULT_AUDIO_TOKEN_IDX

logger = logging.getLogger(__name__)

def _tokenize_fn(
    strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx + 2 : cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "assistant":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = "unknown"
        sentence["value"] = (
            BEGIN_SIGNAL + from_str + ": " + sentence["value"] + END_SIGNAL
        )
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation

def preprocess_llama3(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_audio: bool = False,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    assert conv.version == "llama_3_1"
    roles = {"human": conv.roles[0], "assistant": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    if has_audio:
        # we need to skip the audio placeholder token and put it back as -300
        input_ids = torch.stack([
            tokenizer_mm_token(prompt, tokenizer, return_tensors="pt") 
            for prompt in conversations
        ])
    else:
        input_ids = tokenizer(conversations, return_tensors="pt", padding="longest", max_length=tokenizer.model_max_length, truncation=True).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])]
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx : conv_idx + 2]))
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            round_len = len(tokenizer_mm_token(rou, tokenizer))+1
            instruction_len = len(tokenizer_mm_token(parts[0], tokenizer))

            if i > 0:
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX # ignore the instruction
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            cur_len += round_len


        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    conversation_version: str,
    has_audio: bool = False,
) -> Dict:
    """
    Given a list of sources, each is a conversation list with a system message and a user message.
    """
    conversation = conversation_lib.conv_templates[conversation_version]
    conversation_lib.default_conversation = conversation

    assert conversation.version == "llama_3_1"

    if conversation.version == "llama_3_1":
        return preprocess_llama3(
            sources,
            tokenizer,
            has_audio=has_audio,
        )

    # default
    conversations = []
    for source in sources:
        header = f"{conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)

    conversations_tokenized = _tokenize_fn(conversations, tokenizer)
    input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_lens = _tokenize_fn(
            [header] + [s["value"] for s in source], tokenizer
        )["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        batch_audios_srs = [None] * len(instances)

        for i, instance in enumerate(instances):
            if "audio" in instance:
                batch_audios_srs[i] = (
                    instance["audio"][0],
                    instance["audio_sr"][0],
                )


        # if not all none add to the batch
        def all_none(batch):
            return all([x is None for x in batch])

        if not all_none(batch_audios_srs):
            batch["audios_srs"] = batch_audios_srs

        return batch

def tokenizer_mm_token(
    prompt,
    tokenizer,
    DEFAULT_AUDIO_TOKEN_IDX=DEFAULT_AUDIO_TOKEN_IDX,
    return_tensors=None,
):
    matches = [
        (m.start(), m.end(), DEFAULT_AUDIO_TOKEN_IDX)
        for m in re.finditer(DEFAULT_AUDIO_TOKEN, prompt)
    ]

    # Initialize input_ids list with BOS token if present
    input_ids = []

    # Tokenize and split prompt by audio tokens
    prev_end = 0
    for start, end, mm_token in matches:
        # Tokenize text between previous match and current match
        if prev_end < start:
            new_ids = (
                tokenizer(prompt[prev_end:start]).input_ids[1:]
                if tokenizer.bos_token_id is not None
                else tokenizer(prompt[prev_end:start]).input_ids
            )
            input_ids.extend(new_ids)
        # Add audio token
        input_ids.append(mm_token)
        prev_end = end

    # Add any remaining part of the prompt after the last match
    if prev_end < len(prompt):
        new_ids = (
            tokenizer(prompt[prev_end:]).input_ids[1:]
            if tokenizer.bos_token_id is not None
            else tokenizer(prompt[prev_end:]).input_ids
        )
        input_ids.extend(new_ids)

    if tokenizer.bos_token_id is not None:
        input_ids = [tokenizer.bos_token_id] + input_ids

    # Return tensors if specified
    if return_tensors == "pt":
        return torch.tensor(input_ids, dtype=torch.long)
    elif return_tensors is not None:
        raise ValueError(f"Unsupported tensor type: {return_tensors}")

    return input_ids