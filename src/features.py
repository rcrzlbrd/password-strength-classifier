import math
import string

COMMON_PASSWORDS = {
    "password", "123456", "qwerty", "abc123", "letmein",
    "monkey", "1234567", "dragon", "111111", "baseball",
    "iloveyou", "trustno1", "sunshine", "master", "welcome",
    "shadow", "superman", "michael", "football", "password1"
}

def shannon_entropy(password: str) -> float:
    if not password:
        return 0.0
    freq = {}
    for ch in password:
        freq[ch] = freq.get(ch, 0) + 1
    n = len(password)
    return -sum((c / n) * math.log2(c / n) for c in freq.values())

def has_consecutive(password: str, n: int = 3) -> int:
    for i in range(len(password) - n + 1):
        chunk = password[i:i + n]
        if len(set(chunk)) == 1:
            return 1
        if all(ord(chunk[j+1]) - ord(chunk[j]) == 1 for j in range(n-1)):
            return 1
        if all(ord(chunk[j]) - ord(chunk[j+1]) == 1 for j in range(n-1)):
            return 1
    return 0

def extract_features(password: str) -> dict:
    if not isinstance(password, str):
        password = str(password)

    length = len(password)
    unique_chars = len(set(password))

    return {
        "length": length,
        "entropy": shannon_entropy(password),
        "num_uppercase": sum(1 for c in password if c.isupper()),
        "num_lowercase": sum(1 for c in password if c.islower()),
        "num_digits": sum(1 for c in password if c.isdigit()),
        "num_special": sum(1 for c in password if c in string.punctuation),
        "unique_char_ratio": unique_chars / length if length > 0 else 0,
        "has_consecutive": has_consecutive(password),
        "is_common": int(password.lower() in COMMON_PASSWORDS),
    }
