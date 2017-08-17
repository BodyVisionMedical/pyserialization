def dict_keys_to_str(dict_orig):
    dict_str = {}
    for key in dict_orig.keys():
        dict_str[key.name] = dict_orig[key]
    return dict_str


def dict_str_keys_to_enum_keys(dict_str, enum_type):
    dict_final = {}
    for key_str in dict_str.keys():
        dict_final[enum_type[key_str]] = dict_str[key_str]
    return dict_final
