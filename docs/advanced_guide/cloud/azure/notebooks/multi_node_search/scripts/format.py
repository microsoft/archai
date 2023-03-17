import sys
import json
error = json.load(open(sys.argv[1], encoding='utf-8'))
print(error['message'])