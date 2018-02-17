WHITELIST = 'abcdefghijklmnopqrstuvwxyz1234567890?.,'


def in_white_list(_word):
    valid_word = False
    for char in _word:
        if char in WHITELIST:
            valid_word = True
            break

    if valid_word is False:
        return False

    return True
