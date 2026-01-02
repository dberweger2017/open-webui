import time
import sys
from openai import OpenAI

# Force unbuffered output so you see text instantly
sys.stdout.reconfigure(line_buffering=True)

client = OpenAI(base_url='http://localhost:8000/v1', api_key='EMPTY')
MODEL = 'mistralai/Ministral-3-8B-Instruct-2512'

print(f'--- Connecting to {MODEL} ---')

# Speed Test with Visual Streaming
print('\n--- STREAMING OUTPUT BELOW ---\n')

start_time = time.time()
first_token_time = None

stream = client.chat.completions.create(
    model=MODEL,
    messages=[{'role': 'user', 'content': 'Write a short poem about a robot learning to love.'}],
    max_tokens=200,
    stream=True
)

token_count = 0

for chunk in stream:
    content = chunk.choices[0].delta.content
    if content:
        # Measure time to first token
        if first_token_time is None:
            first_token_time = time.time()
        
        # Print the word immediately without a newline
        print(content, end='', flush=True)
        token_count += 1

end_time = time.time()

# Calculate stats
ttft = first_token_time - start_time
total_gen_time = end_time - first_token_time
speed = token_count / total_gen_time

print(f'\n\n-----------------------------')
print(f'Time to First Token (Latency): {ttft:.4f}s')
print(f'Generation Speed:              {speed:.2f} tokens/sec')