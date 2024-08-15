import pandas as pd
import openai
import time
import torch
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# torch.cuda.set_device(1)  # 选择第二个GPU
os.environ['CUDA_VISIBLE_DEVICES']='2' #此处选择你要用的GPU序号 0，1，2，3
os.environ['CUDA_LAUNCH_BLOCKING'] = '2' # 下面老是报错 shape 不一致
os.environ['TORCH_USE_CUDA_DSA'] = '2'
from transformers import AutoTokenizer, AutoModelForCausalLM
ckpt_path = "internlm/internlm-xcomposer2-vl-7b"
# ckpt_path = "internlm/internlm-xcomposer2d5-7b"
tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True, cache_dir="/data/internlm")
# Set `torch_dtype=torch.float16` to load model in float16, otherwise it will be loaded as float32 and might cause OOM Error.
model = AutoModelForCausalLM.from_pretrained(ckpt_path, torch_dtype=torch.float32, trust_remote_code=True, cache_dir="/data/internlm").cuda()
model = model.eval()



def generate_solution_from_problem(math_problem, model=model, attempt=0, max_retries=3):
    # 生成解题过程的提示
    prompt = f"""用中文一步一步给出以下数学题的详细解题过程。
                        示例：
                        数学题：求函数$f(x) = x^2 - 4x + 3$的顶点，并画出其图像。
                        解题过程: 1.Derive the Function: First, we find the derivative of $f(x)$, which is $f'(x) = 2x - 4$.
                            2.Find the Zero of the Derivative: Setting $f'(x) = 0$, we solve for $x$ and get $x = 2$. This means the function has an extremum at $x = 2$.
                            3.Compute the $y$ Coordinate of the Vertex: Substituting $x = 2$ into the original function, we get $f(2) = 2^2 - 4*2 + 3 = -1$. Thus, the vertex is at $(2, -1)$.
                            4.Determine the Type of the Vertex: Since the coefficient of the quadratic term is positive, we know this vertex represents the minimum point of the function, i.e., the lowest point on the graph.
                        数学题：{math_problem}
                        解题过程：
                        """

    # # 调用OpenAI API生成解题过程
    # response = openai.ChatCompletion.create(
    #     model=model,
    #     messages=[
    #         {"role": "user", "content": prompt}
    #     ]
    # )
    # 生成响应
    with torch.cuda.amp.autocast():
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        output = model.generate(**inputs, max_length=4096)
        response = tokenizer.decode(output[0], skip_special_tokens=True)

    # 返回生成的解题过程
    return response





def generate_manim_code_from_solution(problem, solution, model=model, attempt=0, max_retries=3):
    # 生成Manim代码的提示
    prompt = f"""根据下面的题目与解题过程,生成相应的Manim代码，来展示这个解题过程中的数学函数图像和解题步骤。视频具体要求如下：
                        1.每个视频展开的样式分为两幕，第一幕为展示题目，第二幕为展示解析。
                        2.第一幕的目标为展示题目信息，文字不能超过边界，且文字不能重叠。
                        3.第二幕要求分布展示解题过程，且文字内容不能重叠，文字内容不能超过边界。画图符合题目意思，且要素完整。文字与图像不能重叠
                        4.第二幕屏幕左边展示画图区域，屏幕右边展示分布讲解区域
                        5.视频效果要连贯，第二幕先展示左边画图部分，再展示右边解析部分

                        数学题目：{problem}
                        解题过程：{solution}
                        """

    # # 调用OpenAI API生成Manim代码
    # response = openai.ChatCompletion.create(
    #     model=model,
    #     messages=[
    #         {"role": "user", "content": prompt}
    #     ]
    # )
    # 生成响应
    with torch.cuda.amp.autocast():
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        output = model.generate(**inputs, max_length=4096)
        response = tokenizer.decode(output[0], skip_special_tokens=True)

    # 返回生成的Manim代码
    return response





#新添加的一个prompt代码
def generate_manim_code_from_problem_and_solution(math_problem, solution, is_physics=False, model="gpt-4", attempt=0, max_retries=3):
    # Encode the problem and solution into UTF-8 strings
    math_problem_utf8 = math_problem.encode('utf-8')
    solution_utf8 = solution.encode('utf-8')

    problem_type = "physics" if is_physics else "mathematics"
    prompt = f"""
    Given the following {problem_type} problem and its solution, create Manim code to visually represent both the solution process and the key concepts. The code should use Manim's capabilities to create any necessary diagrams or models directly in the animation, such as cylinders or cones for geometric problems, without relying on any external images. Here are specific guidelines:

    1. Start with the problem statement at the top, aligning it to the upper edge of the animation.
    2. Automatically include visual diagrams or models at each step where they help visualize the concept being explained.
    3. Use next_to for positioning equations and explanations, maintaining proper spacing.
    4. Set the font size to 24 for all text and math elements.
    5. Clearly distinguish between the problem statement and the solution process.
    6. Use smooth visual transitions to ensure viewer comprehension without haste.
    7. Space out text and graphics evenly to avoid clutter.
    8. Intuitively interpret solution steps to include relevant diagrams or graphics such as force diagrams in physics or geometric constructions in mathematics.
    9. Ensure mathematical expressions and shapes are properly displayed.
    10. Manage text overflow to keep all elements on-screen.

    For physics problems:
    - Automatically draw force diagrams, free-body diagrams, or motion paths where applicable.

    For mathematics problems:
    - Automatically generate graphs, geometric constructions, or other illustrative diagrams as needed to elucidate the solution.

    Enhance clarity with:
    - Concise text explanations alongside relevant visuals, split into multiple lines if necessary.
    - Limit each line to 10 words and each page to 10 lines.

    Example Solution: {solution_utf8}

    Manim code requirements:
    - Interpret and visualize crucial steps using diagrams.
    - Include FadeIn and corresponding FadeOut animations to prevent overlapping.
    """

    # response = openai.ChatCompletion.create(
    #     model=model,
    #     messages=[{"role": "user", "content": prompt}]
    # )
    # code = response['choices'][0]['message']['content']
    # 生成响应
    with torch.cuda.amp.autocast():
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        output = model.generate(**inputs, max_length=4096)
        response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response




math_problem = "已知一次函数f(x)=mx+c和g(x)=nx+d,且当x=1时，f(x)=g(x)，当x=2时，f(x)-g(x)=4.求m、n、c、d之间的关系"
solution = generate_solution_from_problem(math_problem)
manim_code = generate_manim_code_from_solution(math_problem, solution)
# manim_code = generate_manim_code_from_problem_and_solution(math_problem, solution)

print("解题过程：", solution)
print("Manim代码:", manim_code)