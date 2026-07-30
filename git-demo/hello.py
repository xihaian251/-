# 第一个版本：简单的问候程序
def say_hello(name):
    return f"你好，{name}！"

if __name__ == "__main__":
    message = say_hello("世界")
    print(message)
