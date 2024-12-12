import re

text = "The best answer is D"
# pattern = r"((?<=The answer is )(.*)(?=.)|(?<=answer is )(.*)(?=.)|(?<=The answer: )(.*)(?=.)|(?<=The final answer: )(.*)(?=.))"
pattern=r"answer is \(?([A-J])\)?"
match = re.search(pattern, text)

if match:
    print("Match found:", match.group(1))  # Output: Match found: D
else:
    print("No match found")