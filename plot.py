import pandas as pd
import matplotlib.pyplot as plt

# 1. 读取数据
try:
    df = pd.read_csv('data.csv') # 确保文件名对得上
except FileNotFoundError:
    print("找不到 CSV 文件，请检查文件名！")
    exit()

# 2. 创建画布
plt.figure(figsize=(10, 6))

# 3. 绘制数据 (假设列名是 x 和 y，如果不是请修改)
plt.plot(df['x'], df['y'], label='Unicycle Path', color='blue', linewidth=2)

# 4. 修饰
plt.title('Unicycle Simulation Result')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.grid(True)
plt.legend()

# 5. 重点：先保存，再尝试显示
plt.savefig('final_comparison.png')
print("图片已保存至 final_comparison.png")
plt.show()