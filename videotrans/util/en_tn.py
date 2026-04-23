# This code is copied from https://github.com/OpenDocCN/python-code-anls/blob/master/docs/hf-tfm/models----clvp----number_normalizer.py.md
# The following is the copyright statement attached to the original document
#
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""English Normalizer class for CLVP."""

import re


class EnglishNormalizer:
    def __init__(self):
        # List of (regular expression, replacement) pairs for abbreviations:
        self._abbreviations = [
            # Compile regular expressions for abbreviations and their replacements
            (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
            for x in [
                ("mrs", "misess"),
                ("mr", "mister"),
                ("dr", "doctor"),
                ("st", "saint"),
                ("co", "company"),
                ("jr", "junior"),
                ("maj", "major"),
                ("gen", "general"),
                ("drs", "doctors"),
                ("rev", "reverend"),
                ("lt", "lieutenant"),
                ("hon", "honorable"),
                ("sgt", "sergeant"),
                ("capt", "captain"),
                ("esq", "esquire"),
                ("ltd", "limited"),
                ("col", "colonel"),
                ("ft", "fort"),
            ]
        ]

        # List of English words for numbers
        self.ones = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
        self.teens = [
            "ten",
            "eleven",
            "twelve",
            "thirteen",
            "fourteen",
            "fifteen",
            "sixteen",
            "seventeen",
            "eighteen",
            "nineteen",
        ]
        self.tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

    def number_to_words(self, num: int) -> str:
        """
        Converts numbers(`int`) to words(`str`).

        Please note that it only supports up to - "'nine hundred ninety-nine quadrillion, nine hundred ninety-nine
        trillion, nine hundred ninety-nine billion, nine hundred ninety-nine million, nine hundred ninety-nine
        thousand, nine hundred ninety-nine'" or `number_to_words(999_999_999_999_999_999)`.
        """
        # If the input number is 0, return the string "zero"
        if num == 0:
            return "zero"
        # If the input number is less than 0, return the English representation of the negative number and call itself recursively to process the absolute value
        elif num < 0:
            return "minus " + self.number_to_words(abs(num))
        # Process numbers between 0 and 9 and directly return the corresponding English representation
        elif num < 10:
            return self.ones[num]
        # Process numbers between 10 and 19 and directly return the corresponding English representation
        elif num < 20:
            return self.teens[num - 10]
        # Process numbers between 20 and 99, decompose them into tens and ones digits, and recursively call itself to process the ones digits
        elif num < 100:
            return self.tens[num // 10] + ("-" + self.number_to_words(num % 10) if num % 10 != 0 else "")
        # Process numbers between 100 and 999, decompose them into hundreds and the remaining part, and call itself recursively to process the remaining part
        elif num < 1000:
            return (
                    self.ones[num // 100] + " hundred" + (
                " " + self.number_to_words(num % 100) if num % 100 != 0 else "")
            )
        # Process numbers between 1000 and 999999, decompose them into thousands and the remaining part, and call itself recursively to process the remaining part
        elif num < 1_000_000:
            return (
                    self.number_to_words(num // 1000)
                    + " thousand"
                    + (", " + self.number_to_words(num % 1000) if num % 1000 != 0 else "")
            )
        # Process numbers between 1000000 and 999999999, decompose them into millions of digits and the remaining part, and call itself recursively to process the remaining part
        elif num < 1_000_000_000:
            return (
                    self.number_to_words(num // 1_000_000)
                    + " million"
                    + (", " + self.number_to_words(num % 1_000_000) if num % 1_000_000 != 0 else "")
            )
        # Process numbers between 1000000000 and 999999999999, decompose them into billions of digits and the remaining parts, and call itself recursively to process the remaining parts
        elif num < 1_000_000_000_000:
            return (
                    self.number_to_words(num // 1_000_000_000)
                    + " billion"
                    + (", " + self.number_to_words(num % 1_000_000_000) if num % 1_000_000_000 != 0 else "")
            )
        # Process numbers between 1000000000000 and 999999999999999, decompose them into trillions of digits and the remaining parts, and call itself recursively to process the remaining parts
        elif num < 1_000_000_000_000_000:
            return (
                    self.number_to_words(num // 1_000_000_000_000)
                    + " trillion"
                    + (", " + self.number_to_words(num % 1_000_000_000_000) if num % 1_000_000_000_000 != 0 else "")
            )
        # Process numbers between 1000000000000000 and 9999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999.
        elif num < 1_000_000_000_000_000_000:
            return (
                    self.number_to_words(num // 1_000_000_000_000_000)
                    + " quadrillion"
                    + (
                        ", " + self.number_to_words(num % 1_000_000_000_000_000)
                        if num % 1_000_000_000_000_000 != 0
                        else ""
                    )
            )
        # Process numbers out of range and return the string "number out of range"
        else:
            return "number out of range"

    def convert_to_ascii(self, text: str) -> str:
        """
        Converts unicode to ascii
        """
        # Convert Unicode text to ASCII encoding, ignoring non-ASCII characters
        return text.encode("ascii", "ignore").decode("utf-8")

    def _expand_dollars(self, m: str) -> str:
        """
        This method is used to expand numerical dollar values into spoken words.
        """
        # Matched numeric string, that is, currency value
        match = m.group(1)
        # Split the currency value into an integer part and a decimal part according to the decimal point
        parts = match.split(".")
        if len(parts) > 2:
            return match + " dollars"  # If there are more than one decimal point, return the original string plus "dollars" to indicate an abnormal format

        # Parse the integer part and decimal part
        dollars = int(parts[0]) if parts[0] else 0
        cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
        # According to the integer part and decimal part of the currency value, construct the corresponding English expression form
        if dollars and cents:
            dollar_unit = "dollar" if dollars == 1 else "dollars"
            cent_unit = "cent" if cents == 1 else "cents"
            return "%s %s, %s %s" % (dollars, dollar_unit, cents, cent_unit)
        elif dollars:
            dollar_unit = "dollar" if dollars == 1 else "dollars"
            return "%s %s" % (dollars, dollar_unit)
        elif cents:
            cent_unit = "cent" if cents == 1 else "cents"
            return "%s %s" % (cents, cent_unit)
        else:
            return "zero dollars"

    def _remove_commas(self, m: str) -> str:
        """
        This method is used to remove commas from sentences.
        """
        # Remove commas from input string
        return m.group(1).replace(",", "")

    def _expand_decimal_point(self, m: str) -> str:
        """
        This method is used to expand '.' into spoken word ' point '.
        """
        # Replace the period '.' in the input string with the word " point "
        return m.group(1).replace(".", " point ")

    def _expand_ordinal(self, num: str) -> str:
        """
        This method is used to expand ordinals such as '1st', '2nd' into spoken words.
        """
        # Define the suffix mapping table for English ordinal words
        ordinal_suffixes = {1: "st", 2: "nd", 3: "rd"}

        # Extract the numeric part of the ordinal word and convert it to an integer
        num = int(num.group(0)[:-2])
        #Choose the correct suffix according to different situations of ordinal numbers
        if 10 <= num % 100 <= 20:
            suffix = "th"
        else:
            suffix = ordinal_suffixes.get(num % 10, "th")
        # Convert the integer to the corresponding English ordinal word form and add a suffix
        return self.number_to_words(num) + suffix

    def _expand_number(self, m: str) -> str:
        """
        This method acts as a preprocessing step for numbers between 1000 and 3000 (same as the original repository,
        link :
        https://github.com/neonbjb/tortoise-tts/blob/4003544b6ff4b68c09856e04d3eff9da26d023c2/tortoise/utils/tokenizer.py#L86)
        """
        # Extract the matched numeric string and convert it to an integer
        num = int(m.group(0))

        # If the number is between 1000 and 3000, expand the English numbers according to specific rules
        if 1000 < num < 3000:
            if num == 2000:
                return "two thousand"
            elif 2000 < num < 2010:
                return "two thousand " + self.number_to_words(num % 100)
            elif num % 100 == 0:
                return self.number_to_words(num // 100) + " hundred"
            else:
                return self.number_to_words(num)
        else:
            return self.number_to_words(num)

    # This method is used to normalize numbers in text, such as converting numbers to words, removing commas, etc.
    def normalize_numbers(self, text: str) -> str:
        # Use regular expressions to replace matching numbers and commas and call the self._remove_commas method
        text = re.sub(re.compile(r"([0-9][0-9\,]+[0-9])"), self._remove_commas, text)
        # Replace matching pound amounts with their word representations
        text = re.sub(re.compile(r"£([0-9\,]*[0-9]+)"), r"\1 pounds", text)
        # Replace the matching dollar amount with its complete dollar amount expression and call the self._expand_dollars method
        text = re.sub(re.compile(r"\$([0-9\.\,]*[0-9]+)"), self._expand_dollars, text)
        # Replace the matching decimal form with its complete numerical expression and call the self._expand_decimal_point method
        text = re.sub(re.compile(r"([0-9]+\.[0-9]+)"), self._expand_decimal_point, text)
        # Replace the matching ordinal word (such as 1st, 2nd) with its complete ordinal word form and call the self._expand_ordinal method
        text = re.sub(re.compile(r"[0-9]+(st|nd|rd|th)"), self._expand_ordinal, text)
        # Replace the matching number with its complete numerical expression and call the self._expand_number method
        text = re.sub(re.compile(r"[0-9]+"), self._expand_number, text)
        # Return the normalized text
        return text

    # Expand abbreviations
    def expand_abbreviations(self, text: str) -> str:
        # Traverse abbreviations and their corresponding replacement rules, and use regular expressions to replace them.
        for regex, replacement in self._abbreviations:
            text = re.sub(regex, replacement, text)
        # Return the expanded text
        return text

    # Remove extra whitespace characters
    def collapse_whitespace(self, text: str) -> str:
        # Use regular expressions to replace multiple consecutive whitespace characters with one space
        return re.sub(re.compile(r"\s+"), " ", text)

    # Object callable methods to convert text to ASCII, convert numbers to full form, and expand abbreviations
    def __call__(self, text):
        # Convert text to ASCII representation
        text = self.convert_to_ascii(text)
        #Convert text to lowercase
        text = text.lower()
        # Normalize numbers in text
        text = self.normalize_numbers(text)
        # Expand abbreviations in text
        text = self.expand_abbreviations(text)
        # Remove extra whitespace characters from text
        text = self.collapse_whitespace(text)
        # Remove double quotes from text
        text = text.replace('"', "")

        # Return the processed text
        return text
