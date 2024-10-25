from dotenv import load_dotenv
load_dotenv()

import os
print(f"[API KEY]\n{os.environ['OPENAI_API_KEY']}")