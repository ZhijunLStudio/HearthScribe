import openai

# 创建一个客户端，指向您本地的服务
client = openai.OpenAI(
    base_url="http://127.0.0.1:8081/v1",  # 确保URL正确
    api_key="sk-no-key-required"        # API key是必需的，但内容随意
)

# 发起聊天请求
response = client.chat.completions.create(
    model="ernie-model",
    messages=[
        {"role": "system", "content": "你是一个乐于助人的AI助手。"},
        {"role": "user", "content": "你好！请解释一下什么是黑洞。"}
    ]
)

# 打印结果
print(response.choices[0].message.content)