#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2024-11-14 14:28:52
# @Last modified by: ArthurBernard
# @Last modified time: 2024-11-14 19:07:13

""" Test `data/chat.py` script. """

# Built-in packages

# Third party packages
from llama_cpp import LlamaTokenizer
import pytest

# Local packages
from pyllmsol.data.chat import Chat, Message, ROLES, SEP

__all__ = []


class MockTokenizer(LlamaTokenizer):
    def __init__(self):
        self.pad_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return [i for i, _ in enumerate(text)]

    def __bool__(self):
        return True

    def encode(self, text, add_special_tokens=False):
        return [i for i, _ in enumerate(text.decode('utf-8'))]


@pytest.fixture
def mock_tokenizer():
    return MockTokenizer()


@pytest.fixture
def sample_message(mock_tokenizer):
    return Message(role="user", content="Hello, world!", tokenizer=mock_tokenizer)


@pytest.fixture
def sample_chat(mock_tokenizer):
    items = [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "How can I help?"},
    ]
    return Chat(items=items, tokenizer=mock_tokenizer)


# Test Message object

def test_valid_role_initialization(mock_tokenizer):
    # Test valid role initialization
    msg = Message(role="user", content="Hi!", tokenizer=mock_tokenizer)
    assert msg.role == "user"
    assert msg.header == ROLES["user"]
    assert msg.footer == SEP


def test_invalid_role_initialization(mock_tokenizer):
    # Test that initializing with an invalid role raises ValueError
    with pytest.raises(ValueError):
        Message(role="invalid_role", content="Hi!", tokenizer=mock_tokenizer)


def test_text_property(sample_message):
    # Test that the text property returns the formatted message
    expected_text = ROLES["user"] + "Hello, world!" + SEP
    assert sample_message.text == expected_text


def test_tokens_property(sample_message):
    # Test that tokens property returns the correct token IDs
    tokens = sample_message.tokens
    # Expected tokens correspond to the word positions in the mock tokenizer
    assert tokens == [i for i in range(len(sample_message.text))]  # header + "Hello," + "world!"


def test_mask_property_for_user_role(sample_message):
    # Test mask for a user role where all tokens should be unmasked (1)
    mask = sample_message.mask
    assert mask == [1] * len(sample_message.tokens)


def test_mask_property_for_assistant_role(mock_tokenizer):
    # Test mask for an assistant role where content tokens are masked (0)
    msg = Message(role="assistant", content="How can I help?", tokenizer=mock_tokenizer)
    header_tokens = msg.tokenize(msg.header, add_special_tokens=False)
    expected_mask = [1] * len(header_tokens) + [0] * (len(msg.tokens) - len(header_tokens))
    assert msg.mask == expected_mask


def test_str_method(sample_message):
    # Test __str__ method for formatted message text
    assert str(sample_message) == sample_message.text


def test_repr_method(sample_message):
    # Test __repr__ method
    assert repr(sample_message) == "Message(from=user, content=Hello, world!)"


def test_format_method(sample_message):
    # Test __format__ with custom format specification
    formatted = format(sample_message)
    assert formatted == "user: Hello, world!"


def test_getitem_role(sample_message):
    # Test __getitem__ for 'role'
    assert sample_message["role"] == "user"


def test_getitem_content(sample_message):
    # Test __getitem__ for 'content'
    assert sample_message["content"] == "Hello, world!"


def test_getitem_invalid_key(sample_message):
    # Test that __getitem__ raises KeyError for an invalid key
    with pytest.raises(KeyError):
        sample_message["invalid_key"]


def test_contains_role(sample_message):
    # Test __contains__ for 'role'
    assert "role" in sample_message


def test_contains_content(sample_message):
    # Test __contains__ for 'content'
    assert "content" in sample_message


def test_contains_invalid_key(sample_message):
    # Test that __contains__ returns False for an invalid key
    assert "invalid_key" not in sample_message


# Test Chat object

def test_chat_text_property(sample_chat):
    msg1 = "<|start_header_id|>user<|end_header_id|>\n\nHello!<|eot_id|>"
    msg2 = "<|start_header_id|>assistant<|end_header_id|>\n\nHow can I help?<|eot_id|>"
    assert f"{msg1}{msg2}" == sample_chat.text


def test_chat_tokens_property(sample_chat):
    assert sample_chat.tokens == [i for i in range(len(sample_chat.text))]


def test_chat_mask_property(sample_chat):
    expected_mask = [1 for i in range(len(sample_chat.items[0].text))]
    expected_mask += [1 for i in range(len(sample_chat.items[1].header))]
    expected_mask += [0 for i in range(len(sample_chat.items[1].content))]
    expected_mask += [0 for i in range(len(sample_chat.items[1].footer))]
    assert sample_chat.mask == expected_mask


def test_pad_valid_length(sample_chat):
    # Test padding with sufficient `total_tokens`
    total_tokens = len(sample_chat.tokens) + 5
    tokens, mask = sample_chat.pad(total_tokens)
    assert len(tokens) == total_tokens
    assert tokens[-5:] == [0] * 5
    assert len(tokens) == len(mask)
    assert len(mask) == total_tokens


def test_pad_insufficient_length(sample_chat):
    # Test padding with insufficient `total_tokens`, should raise ValueError
    total_tokens = len(sample_chat.tokens) - 1
    with pytest.raises(ValueError, match="must be greater than or equal to"):
        sample_chat.pad(total_tokens)


def test_add_message_dict(sample_chat):
    # Test adding a message as a dictionary to the chat
    message = {"role": "user", "content": "New user message"}
    sample_chat.add(message, inplace=True)
    assert sample_chat.items[-1].content == "New user message"


def test_add_message_object(sample_chat, mock_tokenizer):
    # Test adding a Message object to the chat
    new_message = Message(role="assistant", content="New assistant message", tokenizer=mock_tokenizer)
    sample_chat.add(new_message, inplace=True)
    assert sample_chat.items[-1].content == "New assistant message"


def test_add_invalid_role_in_chat(sample_chat):
    # Test adding a message with an invalid role, should raise KeyError
    invalid_message = {"role": "invalid_role", "content": "Should fail"}
    with pytest.raises(ValueError, match="Invalid role"):
        sample_chat.add(invalid_message, inplace=True)


if __name__ == "__main__":
    pass
