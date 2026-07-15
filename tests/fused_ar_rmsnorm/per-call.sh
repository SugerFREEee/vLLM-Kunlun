curl -X POST http://127.0.0.1:8806/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "messages": [{"role": "user", "content": "什么是黑洞"}],
      "temperature": 0,
      "max_tokens": 64,
      "model": "Qwen3-32B"
    }'