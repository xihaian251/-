# 第二个版本：增强的问候程序，支持多种语言
def say_hello(name, language="zh"):
    greetings = {
        "zh": f"你好，{name}！",
        "en": f"Hello, {name}!",
        "ja": f"こんにちは、{name}！",
        "fr": f"Bonjour, {name} !"
    }
    return greetings.get(language, greetings["zh"])

def say_goodbye(name, language="zh"):
    goodbyes = {
        "zh": f"再见，{name}！",
        "en": f"Goodbye, {name}!",
        "ja": f"さようなら、{name}！",
        "fr": f"Au revoir, {name} !"
    }
    return goodbyes.get(language, goodbyes["zh"])

if __name__ == "__main__":
    print(say_hello("世界", "zh"))
    print(say_hello("World", "en"))
    print(say_goodbye("世界", "zh"))
