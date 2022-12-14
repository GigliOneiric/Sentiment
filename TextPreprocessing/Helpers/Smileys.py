import Config.text


def remove_smiley(text):
    """
    Remove smileys

    Sources: https://de.wiktionary.org/wiki/Verzeichnis:International/Smileys
             https://en.wiktionary.org/wiki/Appendix:Emoticons
    """

    SMILEYS = {
        ":)": "",
        ":-)": "",
        ":^)": "",
        ":-]": "",
        "=]": "",
        ":]": "",
        ":D": "",
        ":-D": "",
        ":))": "",
        ";-]": "",
        ";o)": "",
        "¦)": "",
        "=:)": "",
        ":9": "",
        "c:": "",
        ":'D": "",
        "xD": "",
        "XD": "",
        "B)": "",
        "B-)": "",
        "8)": "",
        "8-)": "",
        "=8)": "",
        "=8^)": "",
        "=B)": "",
        "=B^)": "",
        "~8D": "",
        "y=)": "",
        ">:)": "",
        ">:D": "",
        ">:>": "",
        ">:[]": "",
        "^_^": "",
        "^-^": "",
        "^.^": "",
        "^,^": "",
        "^^": "",
        "^^'": "",
        "^^°": "",
        "^////^": "",
        "^o^": "",
        "^O^": "",
        "^0^": "",
        "\o/": "",
        "<o/": "",
        "<(^.^)>": "",
        "-^_^-": "",
        "*(^_^)*": "",
        "*0*": "",
        "Ü": "",
        "*~*": "",
        ":>": "",
        ":i": "",
        "l:": "",
        ":(": "",
        ":c": "",
        ":[": "",
        "=(": "",
        "=[": "",
        ":'(": "",
        ":,(": "",
        ";(": "",
        ";_;": "",
        "T.T": "",
        "T_T": "",
        "Q_Q": "",
        ":S": "",
        ":-/": "",
        ":/": "",
        ":-I": "",
        ">:(": "",
        ">:o": "",
        ">:O": "",
        ">:@": "",
        "DX": "",
        ":-E3": "",
        "x_X": "",
        "X_x": "",
        "x_x": "",
        "x.X": "",
        "X.x": "",
        "x.x": "",
        "°_°": "",
        ">.<": "",
        ">,<": "",
        "-.-": "",
        "-,-": "",
        "-_-": "",
        "._.": "",
        "^_°'": "",
        "^,°'": "",
        "Oo": "",
        "oO": "",
        "O.o'": "",
        "cO": "",
        "ô_o": "",
        "Ô_ô": "",
        "D:": "",
        "D8<": "",
        "O_O": "",
        "Ò_Ó": "",
        "U_U": "",
        "v_v": "",
        ":<": "",
        "m(": "",
        "°^°": "",
        "(@_@)": "",
        ";.;": "",
        ";)": "",
        ";-)": "",
        "^.-": "",
        ":§": "",
        ";D": "",
        ";-D": "",
        ":P": "",
        ":p": "",
        "c[=": "",
        ":p~~~~~~": "",
        ":-*": "",
        ":*": "",
        ";*": "",
        ":-x": "",
        "C:": "",
        ":o": "",
        ":-o": "",
        ":O": "",
        "0:-)": "",
        "O:-)": "",
        "3:)": "",
        "3:D": "",
        "-.-zZz": "",
        "(o)_(o)": "",
        "($)_($)": "",
        "^_-": "",
        "//.o": "",
        "^w^": "",
        "=^_^=": "",
        "x3": "",
        "*_*": "",
        "#-)": "",
        "`*,...ò_Ó...,*´": "",
        ":-{}": "",
        ":ö": "",
        "û_û": "",
        "Ö_Ö": "",
        ":o)": "",
        "cB": "",
        "BD": "",
        "Y_": "",
        ":-€": "",
        ":3": "",
        "x'DD": "",
        "l/l": "",
        ":o)>": "",
        "(_8(I)": "",
        "//:=|": "",
        "<3": "",
        "</3": "",
        "<'3": "",
        "<°(((><": "",
        "<°{{{><": "",
        "<°++++<": "",
        ">)))°>": "",
        "o=(====>": "",
        "@>--}---": "",
        "@>-`-,--": "",
        "(_|::|_)": "",
        "c(_)": "",
        "[:|]": "",
        "(°oo°)": "",
        "(.)(.)": "",
        "( . Y . )": "",
        "( . )": "",
        "| . |": "",
        ").(": "",
        "(_i_)": "",
        "( Y )": "",
        "8===D": ""
    }
    text = text.split()
    reformed = [SMILEYS[word] if word in SMILEYS else word for word in text]
    return " ".join(reformed)


def replace_smiley(text):
    """
    Replace smileys by their meaning

    Sources: https://de.wiktionary.org/wiki/Verzeichnis:International/Smileys
             https://en.wiktionary.org/wiki/Appendix:Emoticons
    """
    SMILEYS = {
        ":)": " " + Config.text.smile + " ",
        ":-)": " " + Config.text.smile + " ",
        ":^)": "",
        ":-]": " " + Config.text.smile + " ",
        "=]": " " + Config.text.smile + " ",
        ":]": " " + Config.text.smile + " ",
        ":D": "",
        ":-D": "",
        ":))": "",
        ";-]": "",
        ";o)": "",
        "¦)": "",
        "=:)": "",
        ":9": "",
        "c:": "",
        ":'D": "",
        "xD": " " + Config.text.laugh + " ",
        "XD": " " + Config.text.laugh + " ",
        "B)": "",
        "B-)": "",
        "8)": "",
        "8-)": "",
        "=8)": "",
        "=8^)": "",
        "=B)": "",
        "=B^)": "",
        "~8D": "",
        "y=)": "",
        ">:)": "",
        ">:D": "",
        ">:>": "",
        ">:[]": "",
        "^_^": "",
        "^-^": "",
        "^.^": "",
        "^,^": "",
        "^^": "",
        "^^'": "",
        "^^°": "",
        "^////^": "",
        "^o^": "",
        "^O^": "",
        "^0^": "",
        "\o/": " " + Config.text.cheering + " ",
        "<o/": "",
        "<(^.^)>": "",
        "-^_^-": "",
        "*(^_^)*": "",
        "*0*": "",
        "Ü": "",
        "*~*": "",
        ":>": "",
        ":i": "",
        "l:": "",
        ":(": "",
        ":c": " " + Config.text.sad + " ",
        ":[": " " + Config.text.sad + " ",
        "=(": " " + Config.text.sad + " ",
        "=[": " " + Config.text.sad + " ",
        ":'(": "",
        ":,(": "",
        ";(": "",
        ";_;": "",
        "T.T": "",
        "T_T": "",
        "Q_Q": "",
        ":S": "",
        ":-/": "",
        ":/": "",
        ":-I": "",
        ">:(": "",
        ">:o": "",
        ">:O": "",
        ">:@": "",
        "DX": "",
        ":-E3": "",
        "x_X": "",
        "X_x": "",
        "x_x": "",
        "x.X": "",
        "X.x": "",
        "x.x": "",
        ">.<": "",
        ">,<": "",
        "-.-": "",
        "-,-": "",
        "-_-": "",
        "._.": "",
        "^_°'": "",
        "^,°'": "",
        "Oo": "",
        "oO": "",
        "O.o'": "",
        "cO": "",
        "ô_o": "",
        "Ô_ô": "",
        "D:": "",
        "D8<": "",
        "O_O": " " + Config.text.surprised + " ",
        "Ò_Ó": "",
        "U_U": "",
        "v_v": "",
        ":<": "",
        "°_°": "",
        "m(": "",
        "°^°": "",
        "(@_@)": "",
        ";.;": "",
        ";)": " " + Config.text.wink + " ",
        ";-)": " " + Config.text.wink + " ",
        "^.-": " " + Config.text.wink + " ",
        ":§": "",
        ";D": "",
        ";-D": "",
        ":P": " " + Config.text.stuck_out_tongue + " ",
        ":p": " " + Config.text.stuck_out_tongue + " ",
        "c[=": "",
        ":p~~~~~~": "",
        ":-*": " " + Config.text.kiss + " ",
        ":*": " " + Config.text.kiss + " ",
        ";*": "",
        ":-x": "",
        "C:": "",
        ":o": "",
        ":-o": "",
        ":O": "",
        "0:-)": "",
        "O:-)": " " + Config.text.innocent + " ",
        "3:)": " " + Config.text.devilish + " ",
        "3:D": "",
        "-.-zZz": "",
        "(o)_(o)": "",
        "($)_($)": "",
        "^_-": "",
        "//.o": "",
        "^w^": "",
        "=^_^=": "",
        "x3": "",
        "*_*": "",
        "#-)": "",
        "`*,...ò_Ó...,*´": "",
        ":-{}": "",
        ":ö": "",
        "û_û": "",
        "Ö_Ö": "",
        ":o)": "",
        "cB": "",
        "BD": "",
        "Y_": "",
        ":-€": "",
        ":3": "",
        "x'DD": "",
        "l/l": "",
        ":o)>": "",
        "@>--}---": " " + Config.text.rose + " ",
        "@>-`-,--": " " + Config.text.rose + " ",
        "(_|::|_)": "",
        "c(_)": "",
        "[:|]": "",
        "(°oo°)": "",
        "(.)(.)": " " + Config.text.boobs + " ",
        "( . Y . )": " " + Config.text.boobs + " ",
        "( . )": "",
        "| . |": "",
        ").(": "",
        "(_i_)": " " + Config.text.butt + " ",
        "( Y )": "",
        "8===D": " " + Config.text.penis + " ",
        "(_8(I)": " " + Config.text.homer_simpson + " ",
        "//:=|": " " + Config.text.hitler + " ",
        "<3": " " + Config.text.heart + " ",
        "</3": " " + Config.text.broken_heart + " ",
        "<'3": " " + Config.text.broken_heart + " ",
        "<°(((><": "",
        "<°{{{><": "",
        "<°++++<": "",
        ">)))°>": "",
        "o=(====>": "",
    }
    text = text.split()
    reformed = [SMILEYS[word] if word in SMILEYS else word for word in text]
    return " ".join(reformed)