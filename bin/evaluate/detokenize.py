import sys

import tokenizer

for line in sys.stdin:
    sys.stdout.write(tokenizer.correct_spaces(line.strip()) + "\n")
