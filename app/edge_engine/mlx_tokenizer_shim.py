from __future__ import annotations

import importlib
import json
from functools import partial
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Dict, List, Optional

from tokenizers import Tokenizer


class StreamingDetokenizer:
    __slots__ = ("text", "tokens", "offset")

    def reset(self):
        raise NotImplementedError()

    def add_token(self, _token):
        raise NotImplementedError()

    def finalize(self):
        raise NotImplementedError()

    @property
    def last_segment(self):
        text = self.text
        segment = text[self.offset :]
        self.offset = len(text)
        return segment


class NaiveStreamingDetokenizer(StreamingDetokenizer):
    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self._tokenizer.decode([0])
        self.reset()

    def reset(self):
        self.offset = 0
        self.tokens = []
        self._text = ""
        self._current_tokens = []
        self._current_text = ""

    def add_token(self, token):
        self._current_tokens.append(token)
        self.tokens.append(token)

    def finalize(self):
        self._text += self._tokenizer.decode(self._current_tokens)
        self._current_tokens = []
        self._current_text = ""

    @property
    def text(self):
        if self._current_tokens:
            self._current_text = self._tokenizer.decode(self._current_tokens)
            if self._current_text.endswith("\ufffd") or (
                self._tokenizer.clean_up_tokenization_spaces
                and len(self._current_text) > 0
                and self._current_text[-1] == " "
            ):
                self._current_text = self._current_text[:-1]
        if self._current_text and self._current_text[-1] == "\n":
            self._text += self._current_text
            self._current_tokens.clear()
            self._current_text = ""
        return self._text + self._current_text


class SPMStreamingDetokenizer(StreamingDetokenizer):
    def __init__(self, tokenizer, trim_space=True):
        self.trim_space = trim_space
        self._sep = "\u2581".encode()
        self.tokenmap = [""] * (max(tokenizer.vocab.values()) + 1)
        for value, tokenid in tokenizer.vocab.items():
            if value.startswith("<0x"):
                self.tokenmap[tokenid] = bytes([int(value[3:5], 16)])
            else:
                self.tokenmap[tokenid] = value.encode()
        self.reset()

    def reset(self):
        self.offset = 0
        self._unflushed = b""
        self.text = ""
        self.tokens = []

    def _try_flush(self, force=False):
        text = self._unflushed.replace(self._sep, b" ").decode("utf-8", "replace")
        if not force and text.endswith("\ufffd"):
            return
        if not self.text and self.trim_space and text and text[0] == " ":
            text = text[1:]
        self.text += text
        self._unflushed = b""

    def add_token(self, token):
        self.tokens.append(token)
        self._unflushed += self.tokenmap[token]
        self._try_flush()

    def finalize(self):
        self._try_flush(force=True)
        self._unflushed = b""


class BPEStreamingDetokenizer(StreamingDetokenizer):
    _byte_decoder = None
    _space_matches = (".", "?", "!", ",", "n't", "'m", "'s", "'ve", "'re")

    def __init__(self, tokenizer):
        self.clean_spaces = tokenizer.clean_up_tokenization_spaces
        self.tokenmap = [None] * len(tokenizer.vocab)
        for value, tokenid in tokenizer.vocab.items():
            self.tokenmap[tokenid] = value
        self.reset()
        self.make_byte_decoder()

    def reset(self):
        self.offset = 0
        self._unflushed = ""
        self.text = ""
        self.tokens = []

    def _decode_bytes(self, seq):
        barr = bytearray()
        for c in seq:
            res = self._byte_decoder.get(c, False)
            if res:
                barr.append(res)
            else:
                barr.extend(bytes(c, "utf-8"))
        return barr.decode("utf-8", "replace")

    def _maybe_trim_space(self, current_text):
        if len(current_text) == 0:
            return current_text
        if current_text[0] != " ":
            return current_text
        if not self.text:
            return current_text[1:]
        if self.clean_spaces and current_text[1:].startswith(self._space_matches):
            return current_text[1:]
        return current_text

    def add_token(self, token):
        self.tokens.append(token)
        value = self.tokenmap[token] if token < len(self.tokenmap) else "!"
        self._unflushed += value
        text = self._decode_bytes(self._unflushed)
        if not text.endswith("\ufffd") and not (
            len(value) == 1 and self._byte_decoder.get(value[0]) == 32
        ):
            self.text += self._maybe_trim_space(text)
            self._unflushed = ""

    def finalize(self):
        current_text = bytearray(self._byte_decoder[c] for c in self._unflushed).decode(
            "utf-8",
            "replace",
        )
        self.text += self._maybe_trim_space(current_text)
        self._unflushed = ""

    @classmethod
    def make_byte_decoder(cls):
        if cls._byte_decoder is not None:
            return
        char_to_bytes = {}
        limits = [0, ord("!"), ord("~") + 1, ord("¡"), ord("¬") + 1, ord("®"), ord("ÿ") + 1]
        n = 0
        for i, (start, stop) in enumerate(zip(limits, limits[1:])):
            if i % 2 == 0:
                for b in range(start, stop):
                    char_to_bytes[chr(2**8 + n)] = b
                    n += 1
            else:
                for b in range(start, stop):
                    char_to_bytes[chr(b)] = b
        cls._byte_decoder = char_to_bytes


class LocalTokenizer:
    def __init__(self, model_path: Path, tokenizer_config: Dict[str, Any]):
        self.model_path = model_path
        self.init_kwargs = dict(tokenizer_config)
        self.clean_up_tokenization_spaces = bool(
            tokenizer_config.get("clean_up_tokenization_spaces", False)
        )
        self.chat_template = tokenizer_config.get("chat_template")
        self._tokenizer = Tokenizer.from_file(str(model_path / "tokenizer.json"))
        self.vocab = self._tokenizer.get_vocab(with_added_tokens=True)

        special_map = _read_json_dict(model_path / "special_tokens_map.json")
        self.bos_token = _extract_token_string(
            tokenizer_config.get("bos_token", special_map.get("bos_token"))
        )
        self.eos_token = _extract_token_string(
            tokenizer_config.get("eos_token", special_map.get("eos_token"))
        )
        self.pad_token = _extract_token_string(
            tokenizer_config.get("pad_token", special_map.get("pad_token"))
        )
        self.unk_token = _extract_token_string(
            tokenizer_config.get("unk_token", special_map.get("unk_token"))
        )

        self.bos_token_id = _extract_token_id(tokenizer_config.get("bos_token_id"), self.bos_token, self.vocab)
        self.eos_token_id = _extract_token_id(tokenizer_config.get("eos_token_id"), self.eos_token, self.vocab)
        self.pad_token_id = _extract_token_id(tokenizer_config.get("pad_token_id"), self.pad_token, self.vocab)
        self.unk_token_id = _extract_token_id(tokenizer_config.get("unk_token_id"), self.unk_token, self.vocab)

    def get_vocab(self):
        return dict(self.vocab)

    def encode(self, text, add_special_tokens=True, **_kwargs):
        if isinstance(text, list):
            return [self.encode(item, add_special_tokens=add_special_tokens) for item in text]
        return self._tokenizer.encode(str(text), add_special_tokens=add_special_tokens).ids

    def encode_batch(self, texts, add_special_tokens=True, **_kwargs):
        return [encoding.ids for encoding in self._tokenizer.encode_batch(list(texts), add_special_tokens=add_special_tokens)]

    def decode(self, token_ids, skip_special_tokens=False, **_kwargs):
        return self._tokenizer.decode(list(token_ids), skip_special_tokens=skip_special_tokens)

    def batch_decode(self, batch_token_ids, skip_special_tokens=False, **_kwargs):
        return [
            self.decode(token_ids, skip_special_tokens=skip_special_tokens)
            for token_ids in batch_token_ids
        ]

    def convert_tokens_to_ids(self, token):
        return self.vocab.get(str(token))

    def apply_chat_template(self, conversation, tokenize=True, add_generation_prompt=False, **_kwargs):
        rendered = _render_messages(conversation, add_generation_prompt=add_generation_prompt)
        if tokenize:
            return self.encode(rendered, add_special_tokens=False)
        return rendered


class TokenizerWrapper:
    def __init__(
        self,
        tokenizer,
        detokenizer_class=NaiveStreamingDetokenizer,
        eos_token_ids=None,
        chat_template=None,
        tool_call_start=None,
        tool_call_end=None,
        tool_parser=None,
    ):
        self._tokenizer = tokenizer
        self._detokenizer_class = detokenizer_class
        self._eos_token_ids = set(eos_token_ids) if eos_token_ids is not None else {tokenizer.eos_token_id}
        self._think_start = None
        self._think_end = None
        self._think_start_id = None
        self._think_end_id = None
        self._chat_template = chat_template
        self.has_chat_template = tokenizer.chat_template is not None or chat_template is not None
        self._tool_parser = tool_parser
        self._tool_call_start = tool_call_start
        self._tool_call_end = tool_call_end

        vocab = tokenizer.get_vocab()
        think_tokens = [("<think>", "</think>"), ("<longcat_think>", "</longcat_think>")]
        for think_start, think_end in think_tokens:
            if think_start in vocab and think_end in vocab:
                self._think_start = think_start
                self._think_end = think_end
                self._think_start_id = vocab[think_start]
                self._think_end_id = vocab[think_end]
                break

        if (tool_call_start and tool_call_start not in vocab) or (
            tool_call_end and tool_call_end not in vocab
        ):
            self._tool_call_start = None
            self._tool_call_end = None
            self._tool_parser = None

    def apply_chat_template(self, *args, tokenize=True, **kwargs):
        if self._chat_template is not None:
            out = self._chat_template(*args, **kwargs)
            if tokenize:
                out = self._tokenizer.encode(out, add_special_tokens=False)
            return out
        kwargs["return_dict"] = False
        return self._tokenizer.apply_chat_template(*args, tokenize=tokenize, **kwargs)

    @property
    def detokenizer(self):
        return self._detokenizer_class(self)

    def __getattr__(self, attr):
        if attr == "eos_token_ids":
            return self._eos_token_ids
        if attr.startswith("_"):
            return self.__getattribute__(attr)
        return getattr(self._tokenizer, attr)

    def __setattr__(self, attr, value):
        if attr == "eos_token_ids":
            self._eos_token_ids = set(value) if value is not None else set()
        elif attr.startswith("_") or attr == "has_chat_template":
            super().__setattr__(attr, value)
        else:
            setattr(self._tokenizer, attr, value)


def _read_json_dict(path: Path) -> Dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return raw if isinstance(raw, dict) else {}


def _extract_token_string(value: Any) -> Optional[str]:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        content = value.get("content")
        return str(content) if isinstance(content, str) else None
    return None


def _extract_token_id(raw_id: Any, token: Optional[str], vocab: Dict[str, int]) -> Optional[int]:
    if isinstance(raw_id, int):
        return raw_id
    if token is None:
        return None
    return vocab.get(token)


def _match(a, b):
    if type(a) != type(b):
        return False
    if isinstance(a, dict):
        return len(a) == len(b) and all(k in b and _match(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_match(ai, bi) for ai, bi in zip(a, b))
    return a == b


def _is_spm_decoder(decoder):
    target = {
        "type": "Sequence",
        "decoders": [
            {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
            {"type": "ByteFallback"},
            {"type": "Fuse"},
            {"type": "Strip", "content": " ", "start": 1, "stop": 0},
        ],
    }
    return _match(target, decoder)


def _is_spm_decoder_no_space(decoder):
    target = {
        "type": "Sequence",
        "decoders": [
            {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
            {"type": "ByteFallback"},
            {"type": "Fuse"},
        ],
    }
    return _match(target, decoder)


def _is_bpe_decoder(decoder):
    return isinstance(decoder, dict) and decoder.get("type") == "ByteLevel"


def _infer_tool_parser(chat_template):
    if not isinstance(chat_template, str):
        return None
    if "<minimax:tool_call>" in chat_template:
        return "minimax_m2"
    if "<start_function_call>" in chat_template:
        return "function_gemma"
    if "<longcat_tool_call>" in chat_template:
        return "longcat"
    if "<arg_key>" in chat_template:
        return "glm47"
    if "<|tool_list_start|>" in chat_template:
        return "pythonic"
    if "<tool_call>\\n<function=" in chat_template or "<tool_call>\n<function=" in chat_template:
        return "qwen3_coder"
    if "<|tool_calls_section_begin|>" in chat_template:
        return "kimi_k2"
    if "[TOOL_CALLS]" in chat_template:
        return "mistral"
    if "<tool_call>" in chat_template and "tool_call.name" in chat_template:
        return "json_tools"
    return None


def _render_messages(conversation, *, add_generation_prompt: bool) -> str:
    if isinstance(conversation, str):
        rendered = conversation
    elif isinstance(conversation, list):
        parts: List[str] = []
        for item in conversation:
            if isinstance(item, dict):
                role = str(item.get("role") or "").strip()
                content = item.get("content", "")
                if isinstance(content, list):
                    text_parts = []
                    for chunk in content:
                        if isinstance(chunk, dict) and chunk.get("type") == "text":
                            text_parts.append(str(chunk.get("text") or ""))
                        else:
                            text_parts.append(str(chunk))
                    content = "".join(text_parts)
                if role:
                    parts.append(f"{role}: {content}")
                else:
                    parts.append(str(content))
            else:
                parts.append(str(item))
        rendered = "\n".join(part for part in parts if part)
    else:
        rendered = str(conversation)
    if add_generation_prompt:
        rendered = f"{rendered}\nassistant:"
    return rendered


def load(
    model_path,
    tokenizer_config_extra: Optional[Dict[str, Any]] = None,
    eos_token_ids=None,
) -> TokenizerWrapper:
    detokenizer_class = NaiveStreamingDetokenizer
    model_path = Path(model_path)
    tokenizer_file = model_path / "tokenizer.json"

    if not tokenizer_file.exists():
        raise FileNotFoundError(f"Missing tokenizer.json in {model_path}")

    with open(tokenizer_file, "r", encoding="utf-8") as fid:
        try:
            tokenizer_content = json.load(fid)
        except JSONDecodeError as exc:
            raise JSONDecodeError("Failed to parse tokenizer.json", exc.doc, exc.pos) from exc

    decoder = tokenizer_content.get("decoder")
    if decoder is not None:
        if _is_spm_decoder(decoder):
            detokenizer_class = SPMStreamingDetokenizer
        elif _is_spm_decoder_no_space(decoder):
            detokenizer_class = partial(SPMStreamingDetokenizer, trim_space=False)
        elif _is_bpe_decoder(decoder):
            detokenizer_class = BPEStreamingDetokenizer

    if isinstance(eos_token_ids, int):
        eos_token_ids = [eos_token_ids]

    tokenizer_config = _read_json_dict(model_path / "tokenizer_config.json")
    if tokenizer_config_extra:
        tokenizer_config.update(tokenizer_config_extra)

    tokenizer = LocalTokenizer(model_path, tokenizer_config)

    chat_template = None
    if chat_template_type := tokenizer.init_kwargs.get("chat_template_type"):
        chat_template = importlib.import_module(
            f"mlx_lm.chat_templates.{chat_template_type}"
        ).apply_chat_template

    tool_parser_type = tokenizer.init_kwargs.get(
        "tool_parser_type", _infer_tool_parser(tokenizer.chat_template)
    )
    if tool_parser_type is not None:
        tool_module = importlib.import_module(f"mlx_lm.tool_parsers.{tool_parser_type}")
        tool_parser = tool_module.parse_tool_call
        tool_call_start = tool_module.tool_call_start
        tool_call_end = tool_module.tool_call_end
    else:
        tool_parser = None
        tool_call_start = None
        tool_call_end = None

    return TokenizerWrapper(
        tokenizer,
        detokenizer_class,
        eos_token_ids=eos_token_ids,
        chat_template=chat_template,
        tool_parser=tool_parser,
        tool_call_start=tool_call_start,
        tool_call_end=tool_call_end,
    )
