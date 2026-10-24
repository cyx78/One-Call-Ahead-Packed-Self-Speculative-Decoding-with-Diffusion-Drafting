from transformers import PreTrainedTokenizer


def _tokenizer_vocab(tokenizer: PreTrainedTokenizer) -> dict[str, int]:
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if callable(get_vocab):
        return get_vocab()
    return getattr(tokenizer, "vocab", {}) or {}


def _tokenizer_added_vocab(tokenizer: PreTrainedTokenizer) -> dict[str, int]:
    get_added_vocab = getattr(tokenizer, "get_added_vocab", None)
    if callable(get_added_vocab):
        return get_added_vocab()
    return {}


def _tokenizer_token_text(token) -> str | None:
    if token is None:
        return None
    content = getattr(token, "content", None)
    if content is not None:
        return str(content)
    return str(token)


def _get_tokenizer_special_token(tokenizer: PreTrainedTokenizer, name: str):
    token = getattr(tokenizer, name, None)
    if token is not None:
        return _tokenizer_token_text(token)
    return _tokenizer_token_text(getattr(tokenizer, f"_{name}", None))


def _ensure_tokenizer_eos_token(tokenizer: PreTrainedTokenizer):
    eos_token = _get_tokenizer_special_token(tokenizer, "eos_token")
    if eos_token is None and "<|endoftext|>" in _tokenizer_vocab(tokenizer):
        eos_token = "<|endoftext|>"
    if eos_token is None:
        eos_token = _get_tokenizer_special_token(tokenizer, "sep_token")
    if eos_token is None:
        raise AttributeError(
            "Tokenizer must define eos_token, _eos_token, sep_token, or _sep_token before missing special tokens can be filled."
        )
    if getattr(tokenizer, "eos_token", None) is None:
        tokenizer.eos_token = eos_token
    return eos_token


def _lookup_token_in_mapping(mapping, token: str) -> int | None:
    if not mapping:
        return None
    if token in mapping:
        return int(mapping[token])
    for key, value in mapping.items():
        if _tokenizer_token_text(key) == token:
            return int(value)
    return None


def _tokenizer_token_id(
    tokenizer: PreTrainedTokenizer, token: str | None
) -> int | None:
    token = _tokenizer_token_text(token)
    if token is None:
        return None
    for mapping in (
        _tokenizer_added_vocab(tokenizer),
        _tokenizer_vocab(tokenizer),
        getattr(tokenizer, "added_tokens_encoder", None),
        getattr(tokenizer, "_added_tokens_encoder", None),
        getattr(tokenizer, "encoder", None),
    ):
        token_id = _lookup_token_in_mapping(mapping, token)
        if token_id is not None:
            return token_id
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if not callable(convert):
        return None
    try:
        token_id = convert(token)
    except Exception:
        return None
    if not isinstance(token_id, int):
        return None
    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    unk_token = _get_tokenizer_special_token(tokenizer, "unk_token")
    if unk_token_id is not None and token_id == unk_token_id and (token != unk_token):
        return None
    convert_back = getattr(tokenizer, "convert_ids_to_tokens", None)
    if callable(convert_back):
        try:
            roundtrip_token = convert_back(token_id)
        except Exception:
            roundtrip_token = None
        if isinstance(roundtrip_token, list):
            roundtrip_token = roundtrip_token[0] if len(roundtrip_token) == 1 else None
        roundtrip_token = _tokenizer_token_text(roundtrip_token)
        if roundtrip_token is not None and roundtrip_token != token:
            return None
    return token_id


def _set_tokenizer_special_token_id(
    tokenizer: PreTrainedTokenizer, token_name: str, token_id: int
) -> None:
    token_id = int(token_id)
    id_name = f"{token_name}_id"
    for attr_name in (id_name, f"_{id_name}"):
        try:
            setattr(tokenizer, attr_name, token_id)
        except Exception:
            pass


def _ensure_tokenizer_special_token_id(
    tokenizer: PreTrainedTokenizer, token_name: str
) -> None:
    token = _get_tokenizer_special_token(tokenizer, token_name)
    if token is None:
        return
    id_name = f"{token_name}_id"
    if getattr(tokenizer, id_name, None) is not None:
        return
    token_id = _tokenizer_token_id(tokenizer, token)
    if token_id is not None:
        _set_tokenizer_special_token_id(tokenizer, token_name, token_id)


def _ensure_tokenizer_mask_token_registered(
    tokenizer: PreTrainedTokenizer, target_vocab_size: int | None = None
) -> int:
    mask_token = _get_tokenizer_special_token(tokenizer, "mask_token")
    if mask_token is None:
        mask_token = "<|fim_middle|>"
    tokenizer.mask_token = mask_token
    token_id = _tokenizer_token_id(tokenizer, mask_token)
    if token_id is not None:
        _set_tokenizer_special_token_id(tokenizer, "mask_token", token_id)
        return int(token_id)
    try:
        before_len = len(tokenizer)
    except Exception:
        before_len = None
    if target_vocab_size is not None and before_len is not None:
        target_vocab_size = int(target_vocab_size)
        if before_len >= target_vocab_size:
            token_id = target_vocab_size - 1
            _set_tokenizer_special_token_id(tokenizer, "mask_token", token_id)
            return int(token_id)
    added_count = 0
    can_grow = (
        target_vocab_size is None
        or before_len is None
        or before_len < target_vocab_size
    )
    if can_grow:
        add_special_tokens = getattr(tokenizer, "add_special_tokens", None)
        if callable(add_special_tokens):
            added = add_special_tokens({"mask_token": mask_token})
            added_count += int(added or 0)
        token_id = _tokenizer_token_id(tokenizer, mask_token)
        if token_id is None:
            add_tokens = getattr(tokenizer, "add_tokens", None)
            if callable(add_tokens):
                try:
                    added = add_tokens([mask_token], special_tokens=True)
                except TypeError:
                    added = add_tokens([mask_token])
                added_count += int(added or 0)
                token_id = _tokenizer_token_id(tokenizer, mask_token)
    if token_id is None:
        try:
            after_len = len(tokenizer)
        except Exception:
            after_len = None
        if before_len is not None and (
            after_len is not None and after_len > before_len or added_count > 0
        ):
            token_id = before_len
        elif target_vocab_size is not None:
            token_id = int(target_vocab_size) - 1
    if token_id is None or token_id < 0:
        raise ValueError("Tokenizer must have mask_token_id.")
    _set_tokenizer_special_token_id(tokenizer, "mask_token", token_id)
    return int(token_id)


def _ensure_tokenizer_save_pretrained_state(tokenizer: PreTrainedTokenizer) -> None:
    special_token_attrs = getattr(
        tokenizer,
        "SPECIAL_TOKENS_ATTRIBUTES",
        [
            "bos_token",
            "eos_token",
            "unk_token",
            "sep_token",
            "pad_token",
            "cls_token",
            "mask_token",
            "additional_special_tokens",
        ],
    )
    special_tokens_map = getattr(tokenizer, "_special_tokens_map", None)
    if not isinstance(special_tokens_map, dict):
        special_tokens_map = {}
    for attr in special_token_attrs:
        if attr == "additional_special_tokens":
            special_tokens_map.setdefault(attr, [])
            continue
        special_tokens_map.setdefault(attr, None)
        token = _get_tokenizer_special_token(tokenizer, attr)
        if token is not None:
            special_tokens_map[attr] = token
    tokenizer._special_tokens_map = special_tokens_map
    init_kwargs = getattr(tokenizer, "init_kwargs", None)
    if not isinstance(init_kwargs, dict):
        init_kwargs = {}
    for attr, value in special_tokens_map.items():
        if value:
            init_kwargs[attr] = value
    tokenizer.init_kwargs = init_kwargs
    if "extra_special_tokens" not in tokenizer.__dict__:
        tokenizer.extra_special_tokens = {}
    if "chat_template" not in tokenizer.__dict__:
        tokenizer.chat_template = None
    if "init_inputs" not in tokenizer.__dict__:
        tokenizer.init_inputs = ()
    if "_processor_class" not in tokenizer.__dict__:
        tokenizer._processor_class = None
    if "_auto_class" not in tokenizer.__dict__:
        tokenizer._auto_class = None
    if not hasattr(tokenizer, "verbose"):
        tokenizer.verbose = False
    if not hasattr(tokenizer, "clean_up_tokenization_spaces"):
        tokenizer.clean_up_tokenization_spaces = False
    if not hasattr(tokenizer, "model_max_length"):
        tokenizer.model_max_length = int(1e30)


def maybe_add_missing_special_tokens(
    tokenizer: PreTrainedTokenizer, target_vocab_size: int | None = None
):
    eos_token = _ensure_tokenizer_eos_token(tokenizer)
    if getattr(tokenizer, "bos_token", None) is None:
        tokenizer.bos_token = eos_token
    if getattr(tokenizer, "pad_token", None) is None:
        added_vocab = _tokenizer_added_vocab(tokenizer)
        if "<|finetune_right_pad_id|>" in added_vocab:
            tokenizer.pad_token = "<|finetune_right_pad_id|>"
        else:
            tokenizer.pad_token = eos_token
    if getattr(tokenizer, "mask_token", None) is None:
        added_vocab = _tokenizer_added_vocab(tokenizer)
        vocab = _tokenizer_vocab(tokenizer)
        if "<|reserved_special_token_0|>" in added_vocab:
            tokenizer.mask_token = "<|reserved_special_token_0|>"
        elif "<|fim_middle|>" in added_vocab:
            tokenizer.mask_token = "<|fim_middle|>"
        elif "_MASK" in vocab:
            tokenizer.mask_token = "_MASK"
        else:
            tokenizer.mask_token = "<|fim_middle|>"
    for token_name in ("eos_token", "bos_token", "pad_token"):
        _ensure_tokenizer_special_token_id(tokenizer, token_name)
    _ensure_tokenizer_save_pretrained_state(tokenizer)
    _ensure_tokenizer_mask_token_registered(
        tokenizer, target_vocab_size=target_vocab_size
    )
    _ensure_tokenizer_save_pretrained_state(tokenizer)
    return tokenizer
