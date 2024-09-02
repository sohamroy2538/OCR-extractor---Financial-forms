import re
from typing import Optional, Dict
import Levenshtein
import os




import re
from typing import Optional, Dict




def find_years(text , geo_dict: Optional[Dict] = None):
    pattern = r'(2[0-9]{3})'
    
    # Find all matches in the text
    matches = re.findall(pattern, text)

    if len(matches) > 0:
        return matches[0]
    return ""
    

def identity(text, geo_dict: Optional[Dict] = None):
    return text

def name_rl1(text, geo_dict: Optional[Dict] = None):
    return find_words_after_match(text , "particulier" , 2)

def name_t4(text, geo_dict: Optional[Dict] = None):
    return find_words_after_match(text , "defemployeur" , 4)
    
def find_words_after_match(sentence, target_word , match_threshold, geo_dict: Optional[Dict] = None):
    
    words = sentence.split()
    rem_word = []
    for i, word in enumerate(words):
        if Levenshtein.distance(word, target_word) <= match_threshold:
            print(Levenshtein.distance(word, target_word))
            try:
                rem_word = words[i+1 :]
            except:
                rem_word = [""]
            break

    for i, word in enumerate(rem_word):
         if Levenshtein.distance(word, target_word) <= match_threshold:
            return sentence
    
    return " ".join(rem_word)


def process_value(text, geo_dict: Optional[Dict] = None):
    # Define the regex patterns
    pattern1 = re.compile(r'\b(\d{1,})/(\d{2})\b')
    pattern2 = re.compile(r'\b(\d{1,}) (\d{2})\b')
    pattern3 = re.compile(r'\b(\d{1,})\.(\d{2})\b')
    pattern4 = re.compile(r'\b(\d{1,})(?: ?)\.(?: ?)(\d{3})\.(\d{2})\b')
    pattern5 = re.compile(r'\b(\d{1,})(?: ?),(?: ?)(\d{3})\.(\d{2})\b')
    pattern6 = re.compile(r'\b(\d{1,})(?: ?),(?: ?)(\d{2})\b')
    pattern7 = re.compile(r'\b(\d{3,})\b')
    
    match4 = pattern4.search(text)
    if match4:
        return f"{match4.group(1)}.{match4.group(2)}.{match4.group(3)}."


    match5 = pattern5.search(text)
    if match5:
        return f"{match5.group(1)}.{match5.group(2)}.{match5.group(3)}."


    match6 = pattern6.search(text)
    if match6:
        return f"{match6.group(1)}.{match6.group(2)}"


    match1 = pattern1.search(text)
    if match1:
        return f"{match1.group(1)}.{match1.group(2)}"
    
    match2 = pattern2.search(text)
    if match2:
        return f"{match2.group(1)}.{match2.group(2)}"
    
    # Check for pattern3
    match3 = pattern3.search(text)
    if match3:
        return f"{match3.group(1)}.{match3.group(2)}"


    # Check for pattern3
    if not (match1 and match2 and match3 and match4 and match5 and match6):
        match7 = pattern7.findall(text)

        if len(match7) > 0:
            match7 = sorted(match7 , key = lambda x : len(x), reverse= True)
            match7 = match7[0]
    
        if len(match7) >= 3:
            return f"{match7[:-2]}.{match7[-2:]}"
    
    # If no digits found, return original text
    return ""

def t4_optional(text,  geo_dict: Optional[Dict] = None):
    pattern = r'\b([3-9][0-9])\b'
    
    # Find all matches in the text
    matches = re.findall(pattern, text)
    
    return matches[0] if len(matches) != 0  else ""


def check_10(text, geo_dict: Optional[Dict] = None):
    # Define the pattern list
    patterns = ["ab", "bc", "mb", "nb", "nl", "nt", "ns", "nu", "on", "pe", "qc", "sk", "yt", "us", "zz"]
    pattern_found = []
    
    regex_pattern = r'\s?(' + '|'.join(patterns) + r')\s?'
    compiled_pattern = re.compile(regex_pattern, re.IGNORECASE)
    
    # Find all matches in the text
    matches = compiled_pattern.findall(text)
    
    for match in matches:
        if match.lower() in patterns:
            pattern_found.append(match.lower())
    
    return pattern_found[0] if len(pattern_found) > 0 else ""

def t4_20(text, geo_dict: Optional[Dict] = None):
    pattern = r'(?<!\d)\s?207\s?(?!\d)'
    result = re.sub(pattern, ' ', text)
    # Return the resulting string
    return process_value(result.strip())

def t4_18(text, geo_dict: Optional[Dict] = None):
    pattern = r'(?<!\d)\s?312\s?(?!\d)'
    
    result = re.sub(pattern, ' ', text)
    
    # Return the resulting string
    return process_value(result.strip())

def t4_16(text, geo_dict: Optional[Dict] = None):
    pattern = r'(?<!\d)\s?306\s?(?!\d)'
    
    result = re.sub(pattern, ' ', text)
    
    # Return the resulting string
    return process_value(result.strip())

def t4_22(text, geo_dict: Optional[Dict] = None):
    pattern = r'(?<!\d)\s?437\s?(?!\d)'
    
    result = re.sub(pattern, ' ', text)
    
    # Return the resulting string
    return process_value(result.strip())

def t4_24(text, geo_dict: Optional[Dict] = None):
    pattern = r'(?<!\d)\s?24\s?(?!\d)'
    result = text

    if re.search(pattern, text):
        value = geo_dict.get("24", None)
        
        if value is None:
            result =  re.sub(pattern, ' ', text)
        
        numeric_values = [geo_dict[key] for key in geo_dict if key.isdigit() and key != "24"]
        if all(value < val for val in numeric_values):
            result =  re.sub(pattern, ' ', text)
        
    return process_value(result.strip())

def t4_26(text, geo_dict: Optional[Dict] = None):
    pattern = r'(?<!\d)\s?26\s?(?!\d)'
    result = text

    if re.search(pattern, text):
        value = geo_dict.get("26", None)
        if value is None:
            result =  re.sub(pattern, ' ', text)
        
        numeric_values = [geo_dict[key] for key in geo_dict if key.isdigit() and key != "26"]
        
        if all(value < val for val in numeric_values):
            result =  re.sub(pattern, ' ', text)
        
    return process_value(result.strip())

def t4_44(text, geo_dict: Optional[Dict] = None):

    pattern = r'(?<!\d)\s?212\s?(?!\d)'
    result = re.sub(pattern, ' ', text)
    return process_value(result.strip())

def t4_45(text, geo_dict: Optional[Dict] = None):
    pattern = r'(?<!\d)[1-5](?!\d)'
    matches = re.findall(pattern, text)
    return matches[0].strip() if len(matches) !=0 else ""



def numero_assurance(input_string, geo_dict: Optional[Dict] = None):
    
    pattern = re.compile(r'\b(\d{3}-\d{3}-\d{3})|(\d{9})|(\d{3} \d{3} \d{3})|(\d{3} \d{6})|(\d{6} \d{3})\b')
    match = pattern.search(input_string)
    # Return the matched string or None if no match
    if match:
        return match.group(0)
    else:
        return ""
