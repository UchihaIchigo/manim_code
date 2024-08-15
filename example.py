from manim import *
from sympy import *
import numpy as np
# class SquareToCircle(Scene):
#     def construct(self):
#         circle = Circle()
#         square = Square()
#         square.flip(RIGHT)
#         square.rotate(-3 * TAU / 8)
#         circle.set_fill(PINK, opacity=0.5)

#         self.play(Create(square))
#         self.play(Transform(square, circle))
#         self.play(FadeOut(square))


# class SineCurveUnitCircle(Scene):
#     # contributed by heejin_park, https://infograph.tistory.com/230
#     def construct(self):
#         self.show_axis()
#         self.show_circle()
#         self.move_dot_and_draw_curve()
#         self.wait()

#     def show_axis(self):
#         x_start = np.array([-6,0,0])
#         x_end = np.array([6,0,0])

#         y_start = np.array([-4,-2,0])
#         y_end = np.array([-4,2,0])

#         x_axis = Line(x_start, x_end)
#         y_axis = Line(y_start, y_end)

#         self.add(x_axis, y_axis)
#         self.add_x_labels()

#         self.origin_point = np.array([-4,0,0])
#         self.curve_start = np.array([-3,0,0])

#     def add_x_labels(self):
#         x_labels = [
#             MathTex("\pi"), MathTex("2 \pi"),
#             MathTex("3 \pi"), MathTex("4 \pi"),
#         ]

#         for i in range(len(x_labels)):
#             x_labels[i].next_to(np.array([-1 + 2*i, 0, 0]), DOWN)
#             self.add(x_labels[i])

#     def show_circle(self):
#         circle = Circle(radius=1)
#         circle.move_to(self.origin_point)
#         self.add(circle)
#         self.circle = circle

#     def move_dot_and_draw_curve(self):
#         orbit = self.circle
#         origin_point = self.origin_point

#         dot = Dot(radius=0.08, color=YELLOW)
#         dot.move_to(orbit.point_from_proportion(0))
#         self.t_offset = 0
#         rate = 0.25

#         def go_around_circle(mob, dt):
#             self.t_offset += (dt * rate)
#             # print(self.t_offset)
#             mob.move_to(orbit.point_from_proportion(self.t_offset % 1))

#         def get_line_to_circle():
#             return Line(origin_point, dot.get_center(), color=BLUE)

#         def get_line_to_curve():
#             x = self.curve_start[0] + self.t_offset * 4
#             y = dot.get_center()[1]
#             return Line(dot.get_center(), np.array([x,y,0]), color=YELLOW_A, stroke_width=2 )


#         self.curve = VGroup()
#         self.curve.add(Line(self.curve_start,self.curve_start))
#         def get_curve():
#             last_line = self.curve[-1]
#             x = self.curve_start[0] + self.t_offset * 4
#             y = dot.get_center()[1]
#             new_line = Line(last_line.get_end(),np.array([x,y,0]), color=YELLOW_D)
#             self.curve.add(new_line)

#             return self.curve

#         dot.add_updater(go_around_circle)

#         origin_to_circle_line = always_redraw(get_line_to_circle)
#         dot_to_curve_line = always_redraw(get_line_to_curve)
#         sine_curve_line = always_redraw(get_curve)

#         self.add(dot)
#         self.add(orbit, origin_to_circle_line, dot_to_curve_line, sine_curve_line)
#         self.wait(8.5)

#         dot.remove_updater(go_around_circle)

# class MovingFrameBox(Scene):
#     def construct(self):
#         text=MathTex(
#             "\\frac{d}{dx}f(x)g(x)=","f(x)\\frac{d}{dx}g(x)","+",
#             "g(x)\\frac{d}{dx}f(x)"
#         )
#         self.play(Write(text))
#         framebox1 = SurroundingRectangle(text[1], buff = .1)
#         framebox2 = SurroundingRectangle(text[3], buff = .1)
#         self.play(
#             Create(framebox1),
#         )
#         self.wait()
#         self.play(
#             ReplacementTransform(framebox1,framebox2),
#         )
#         self.wait()


# class FollowingGraphCamera(MovingCameraScene):
#     def construct(self):
#         self.camera.frame.save_state()

#         # create the axes and the curve
#         ax = Axes(x_range=[-1, 10], y_range=[-1, 10])
#         graph = ax.plot(lambda x: np.sin(x), color=BLUE, x_range=[0, 3 * PI])

#         # create dots based on the graph
#         moving_dot = Dot(ax.i2gp(graph.t_min, graph), color=ORANGE)
#         dot_1 = Dot(ax.i2gp(graph.t_min, graph))
#         dot_2 = Dot(ax.i2gp(graph.t_max, graph))

#         self.add(ax, graph, dot_1, dot_2, moving_dot)
#         self.play(self.camera.frame.animate.scale(0.5).move_to(moving_dot))

#         def update_curve(mob):
#             mob.move_to(moving_dot.get_center())

#         self.camera.frame.add_updater(update_curve)
#         self.play(MoveAlongPath(moving_dot, graph, rate_func=linear))
#         self.camera.frame.remove_updater(update_curve)

#         self.play(Restore(self.camera.frame))



# class SolutionAnimation(Scene):
#     def construct(self):
#         # 步骤 1: 展示问题
#         self.show_problem()

#         # 步骤 2: 讨论 x 的可能值
#         self.discuss_x_values()

#         # 步骤 3: 因子分解和 y，z 的关系
#         self.factor_decomposition()

#         # 步骤 4: 展示解决方案
#         self.show_solution()

#     def show_problem(self):
#         # 展示问题
#         problem = MathTex("x^3", "(y^3", "+", "z^3)", "=", "2012", "(xyz", "+", "z)")
#         self.play(Write(problem))
#         self.wait(2)
#         self.play(FadeOut(problem))

#     def discuss_x_values(self):
#         # 讨论 x 的可能值
#         discuss_x = Tex("Considering the divisibility by $503$, $x$ must be $2^m$ for $m \in \{0,1,2,3\}$...")
#         self.play(Write(discuss_x))
#         self.wait(2)
#         self.play(FadeOut(discuss_x))

#     def factor_decomposition(self):
#         # 因子分解和 y，z 的关系
#         factor_discussion = Tex("Factorizing and considering $y$ and $z$ relationship...")
#         self.play(Write(factor_discussion))
#         self.wait(2)
#         self.play(FadeOut(factor_discussion))

#     def show_solution(self):
#         # 展示解决方案
#         final_solution = Tex("The only solution is $(2, 251, 252)$")
#         self.play(Write(final_solution))
#         self.wait(2)
#         self.play(FadeOut(final_solution))

# class PlotFunctions(Scene):
#     def construct(self):
#         # 创建坐标系
#         axes = Axes(
#             x_range=[0, 300, 50],
#             y_range=[0, 3e7, 5e6],
#             x_length=7,
#             y_length=6,
#             axis_config={"color": BLUE},
#         )

#         # 标签坐标系
#         x_label = axes.get_x_axis_label("y or z")
#         y_label = axes.get_y_axis_label("Value")

#         # 定义 y^3 + z^3 函数
#         def cubic_function(y):
#             return y**3

#         # 定义 2012(2y + 2 + 2z) 函数
#         def linear_function(y):
#             return 2012 * (2*y + 2 + 2*y)

#         # 绘制函数 y^3 + z^3
#         cubic_graph = axes.plot(cubic_function, color=GREEN)
#         cubic_label = axes.get_graph_label(cubic_graph, label="y^3 + z^3")

#         # 绘制函数 2012(2y + 2 + 2z)
#         linear_graph = axes.plot(linear_function, color=RED)
#         linear_label = axes.get_graph_label(linear_graph, label="2012(2y + 2 + 2z)")

#         # 将对象添加到场景
#         self.play(Create(axes), Create(x_label), Create(y_label))
#         self.play(Create(cubic_graph), Write(cubic_label))
#         self.play(Create(linear_graph), Write(linear_label))

#         # 等待3秒
#         self.wait(3)

# class SolutionWithGraph(Scene):
#     def construct(self):
#         # 展示原始问题
#         self.show_problem()

#         # 解释 x 的可能值
#         self.explain_x_values()

#         # 假设 x = 2 并简化问题
#         self.assume_x_and_simplify()

#         # 展示方程图像和解释推导
#         self.show_graphs_and_derivation()

#     def show_problem(self):
#         # 展示问题
#         problem = MathTex("x^3(y^3 + z^3) = 2012(xyz + z)", substrings_to_isolate="xyz")
#         problem.set_color_by_tex("x", YELLOW)
#         problem.set_color_by_tex("y", GREEN)
#         problem.set_color_by_tex("z", RED)
#         self.play(Write(problem))
#         self.wait(2)
#         self.play(FadeOut(problem))

#     def explain_x_values(self):
#         # 解释 x 的可能值
#         explanation = Text("Since 2012 is divisible by 2 but not 503, x must be a power of 2.", font_size=24)
#         self.play(Write(explanation))
#         self.wait(2)
#         self.play(FadeOut(explanation))

#     def assume_x_and_simplify(self):
#         # 假设 x = 2 并简化问题
#         assumption = Tex("Assume $x = 2$. Then the equation simplifies to:")
#         simplified_eq = MathTex("2^3(y^3 + z^3) = 2012(2yz + z)", substrings_to_isolate="yz")
#         simplified_eq.set_color_by_tex("y", GREEN)
#         simplified_eq.set_color_by_tex("z", RED)
#         self.play(Write(assumption))
#         self.wait(1)
#         self.play(Transform(assumption, simplified_eq))
#         self.wait(2)
#         self.play(FadeOut(assumption))

#     def show_graphs_and_derivation(self):
#         # 展示方程图像和解释推导
#         axes = Axes(
#             x_range=[0, 260, 20],
#             y_range=[0, 2e7, 2e6],
#             x_length=7,
#             y_length=4,
#             axis_config={"color": BLUE}
#         )

#         # 定义函数 y^3 + z^3
#         def cubic_function(y):
#             return 8 * y**3

#         # 定义函数 2012(2yz + z)
#         def product_function(y):
#             return 2012 * (4*y + 1)

#         # 绘制函数 y^3 + z^3
#         cubic_graph = axes.plot(cubic_function, color=GREEN, x_range=[1, 260])
#         cubic_label = axes.get_graph_label(cubic_graph, label="8(y^3 + z^3)", x_val=100)

#         # 绘制函数 2012(2yz + z)
#         product_graph = axes.plot(product_function, color=RED, x_range=[1, 260])
#         product_label = axes.get_graph_label(product_graph, label="2012(2yz + z)", x_val=50)

#         self.play(Create(axes), Write(cubic_label), Write(product_label))
#         self.play(Create(cubic_graph), Create(product_graph))
#         self.wait(2)

#         # 找到图像的交点，并解释这表示什么
#         intersection_dot = Dot(color=ORANGE).move_to(axes.c2p(251, product_function(251)))
#         self.play(FadeIn(intersection_dot, scale=0.5))
#         intersection_label = Tex("(251, 2012(2yz + z))").next_to(intersection_dot, UP)
#         self.play(Write(intersection_label))
#         self.wait(3)

#         # 清除所有对象
#         self.play(FadeOut(axes), FadeOut(cubic_graph), FadeOut(product_graph), FadeOut(intersection_dot), FadeOut(intersection_label), FadeOut(cubic_label), FadeOut(product_label))

# class CubicEquationScene(Scene):
#     def construct(self):
#         # 展示问题
#         self.setup_scene()
#         # 分解x的可能值
#         self.decompose_x()
#         # 分解方程并分析结果
#         self.analyze_equation()
#         # 展示解
#         self.present_solution()
    
#     def setup_scene(self):
#         # 显示原问题
#         problem = MathTex("x^3(y^3 + z^3) = 2012(xy + x + 2z)")
#         self.play(Write(problem))
#         self.wait(2)
#         self.play(FadeOut(problem))
    
#     def decompose_x(self):
#         # 显示x的可能值
#         x_values = MathTex("x = 2^m", ",", "m \in \{0,1,2,3\}")
#         self.play(Write(x_values))
#         self.wait(2)
#         self.play(FadeOut(x_values))

#     def analyze_equation(self):
#         # 解析方程
#         analysis = Tex(r"If $x = 1$, the equation becomes $y^3 + z^3 = 2012(y + z)$. \
#                         Let's analyze this equation graphically.")
#         self.play(Write(analysis))
#         self.wait(2)
#         self.play(FadeOut(analysis))
        
#         # 绘制方程图像
#         axes = Axes(x_range=[0, 10], y_range=[0, 10000], axis_config={"include_tip": True})
#         graph_y3_plus_z3 = axes.plot(lambda y: y**3 + y**3, color=BLUE, x_range=[0, 8])
#         graph_2012y_plus_z = axes.plot(lambda y: 2012 * (y + 1), color=GREEN, x_range=[0, 8])
#         labels = VGroup(
#             axes.get_graph_label(graph_y3_plus_z3, label='y^3 + z^3'),
#             axes.get_graph_label(graph_2012y_plus_z, label='2012(y + z)')
#         )
#         self.play(Create(axes), Create(graph_y3_plus_z3), Create(graph_2012y_plus_z), Write(labels))
#         self.wait(2)
#         self.play(FadeOut(axes), FadeOut(graph_y3_plus_z3), FadeOut(graph_2012y_plus_z), FadeOut(labels))
    
#     def present_solution(self):
#         # 展示最终解
#         solution = MathTex("(x,y,z) = (2,251,252)")
#         self.play(Write(solution))
#         self.wait(2)
#         self.play(FadeOut(solution))

# class SolveAndVisualizeFunction(Scene):
#     def construct(self):
#         self.show_solution_steps()
#         self.plot_function()

#     def show_solution_steps(self):
#         steps = VGroup(
#             Tex(r"1. Derive the function: $f'(x) = 2x - 4$"),
#             Tex(r"2. Find the zero of the derivative: $x = 2$"),
#             Tex(r"3. Compute the $y$ coordinate of the vertex: $f(2) = -1$"),
#             Tex(r"4. Determine the type of the vertex: minimum point"),
#         )
#         steps.arrange(DOWN, aligned_edge=LEFT)
#         for step in steps:
#             self.play(Write(step))
#             self.wait(1)
#         self.play(*[FadeOut(step) for step in steps])

#     def plot_function(self):
#         axes = Axes(
#             x_range=[-1, 5],
#             y_range=[-2, 6],
#         )
#         quadratic_graph = axes.plot(lambda x: x**2 - 4*x + 3, color=GREEN)
#         graph_label = axes.get_graph_label(quadratic_graph, label='f(x)=x^2-4x+3')
        
#         vertex_dot = Dot(axes.c2p(2, -1), color=RED)
#         vertex_label = MathTex("(2, -1)").next_to(vertex_dot, DOWN)
        
#         self.play(Create(axes), Create(quadratic_graph), Write(graph_label))
#         self.play(FadeIn(vertex_dot), Write(vertex_label))
#         self.wait(5)

# class BinomialTheorem(Scene):
#     def construct(self):
#         self.show_solution_steps()
#         self.plot_function()

#     def show_solution_steps(self):
#         steps = VGroup(
#             Tex(r"1. Explanation of Binomial Theorem: Expansion of $(x+y)^n$ into sum terms of the form of $a_{i}x^{n-i}y^i$.",font_size=24),
#             Tex(r"2. Expansion of $(n+1)^3$ is $n^3 + 3n^2 + 3n + 1$. This is obtained by expanding each term in the expression $(n+1)^3$.",font_size=24),
#             Tex(r"3. Now, we can graph the function $y = n^3 + 3n^2 + 3n + 1$. For this, label the 'n' and 'y' axes and plot the function using a set of n values.",font_size=24),
#         )
#         steps.arrange(DOWN, aligned_edge=LEFT)
#         for step in steps:
#             self.play(Write(step))
#             self.wait(1)
#         self.play(*[FadeOut(step) for step in steps])

#     def plot_function(self):
#         axes = Axes(
#             x_range=[-3, 3],
#             y_range=[-10, 15],
#             x_axis_config={"numbers_to_include": [-2, -1, 1, 2]},
#             y_axis_config={"numbers_to_include": [-5, 0, 5, 10]},
#         )

#         # 创建函数图像
#         func_graph = axes.plot(lambda x: x**3 + 3*x**2 + 3*x + 1, color=BLUE)
#         graph_label = axes.get_graph_label(func_graph, label='(n + 1)^3')

#         # 创建轴标签
#         x_axis_label = MathTex("n").next_to(axes.x_axis, RIGHT)
#         y_axis_label = MathTex("y").next_to(axes.y_axis, UP)

#         # 在场景中绘制所有元素
#         self.play(Create(axes), Create(func_graph), Write(graph_label), Write(x_axis_label), Write(y_axis_label))
#         self.wait(5)

# 1
# from manim import *
# import numpy as np

# class InequalitySolution(Scene):
#     def construct(self):
#         self.show_solution_steps()
#         self.plot_functions()

#     def show_solution_steps(self):
#         steps = VGroup(
#             Tex(r"1. Express the functions in piecewise form: $g(x) = [-4,4]$"),
#             Tex(r"2. Determine the conditions for a: Solve the inequality $f(x+a)\geq g(x)$"),
#             Tex(r"3. Combine the inequalities: $-2 \leq a \leq 6$"),
#             Tex(r"4. Sketch the graphs to confirm the results"),
#         )
#         steps.arrange(DOWN, aligned_edge=LEFT)
#         for step in steps:
#             self.play(Write(step))
#             self.wait(1)
#         self.play(*[FadeOut(step) for step in steps])

#     def plot_functions(self):
#         axes = Axes(
#             x_range=[-8, 10],
#             y_range=[-5, 5],
#         )

#         fx_graph = axes.plot(lambda x: abs(x - 2),color=BLUE)
#         fx_label = axes.get_graph_label(fx_graph, label='f(x)=|x-2|')

#         gx_func = lambda x: 4 if (-1.5 < x < 0.5) else -4
#         gx_graph = axes.plot(gx_func, color=RED)
#         gx_label = axes.get_graph_label(gx_graph, label=r'g(x)', color=RED, direction=UP)

#         a_values = [axes.get_vertical_line(axes.input_to_graph_point(a, gx_graph), color=YELLOW) for a in np.arange(-2, 7, 1)]
#         a_values_animations = [Create(line) for line in a_values]

#         self.play(Create(axes), Create(fx_graph), Write(fx_label))
#         self.wait(1)
#         self.play(Create(gx_graph), Write(gx_label))
#         self.wait(1)
#         self.play(*a_values_animations)
#         self.wait(2)

# 2
# from manim import *
# class SolveAndVisualizeInequality(Scene):
#     def construct(self):
#         self.show_solution_steps()
#         self.plot_functions()
    
#     def show_solution_steps(self):
#         steps = VGroup(
#             Tex(r"1. Simplify the function: $f(x)$ = {-4x - 13, 4x - 9, -1}"),
#             Tex(r"2. Combine f(x) and g(x) Inequality: $-(x+t-1) \leq -4x - 13$, $-(x+t-1) \leq 4x - 9$, $-(x+t-1) \leq -1$"),
#             Tex(r"3. Solve for the Range of t: $t \geq 3x + 14$, $t \geq 5x - 8$, $t \geq x - 1$"),
#             Tex(r"4. Figure out the Value of t: $3 \leq t$"),  
#             Tex(r"5. Draw the Graph: $f(x)$ and $g(x)$")
#         )
#         steps.arrange(DOWN, aligned_edge=LEFT)
#         for step in steps:
#             self.play(Write(step))
#             self.wait(1)
#         self.play(*[FadeOut(step) for step in steps])

#     def plot_functions(self):
#         axes = Axes(
#             x_range=[-4, 3],
#             y_range=[-15, 6],
#         )
#         f1_graph = axes.plot(lambda x: -4*x - 13, color=GREEN, x_range=[-4,-3])
#         f2_graph = axes.plot(lambda x: 4*x - 9, color=GREEN, x_range=[-3,2])
#         f3_graph = axes.plot(lambda x: -1, color=GREEN, x_range=[2,3])

#         self.play(Create(axes), Create(f1_graph), Create(f2_graph), Create(f3_graph))
#         self.wait(2)

#         g_graph = axes.plot(lambda x: -(x-1), color=RED)
#         g_label = axes.get_graph_label(g_graph, label='g(x)', direction=UP)
#         self.play(Create(g_graph), Write(g_label))
#         self.wait(2)

#         for t in range(-1, 4):
#             new_g_graph = axes.plot(lambda x: -(x-(1+t)), color=YELLOW)
#             self.play(Transform(g_graph, new_g_graph))
#             self.wait(1)
#         self.wait(2)


# 3
# from manim import *

# class SolveAndVisualizeEllipse(Scene):
#     def construct(self):
#         self.show_solution_steps()
#         self.plot_ellipse()

#     def show_solution_steps(self):
#         steps = VGroup(
#             Tex(r"1. Derive the line equation: $\\frac{y^2}{3-K^2} + \\frac{(ky+x)^2}{a^2} = 1$"),
#             Tex(r"2. Compute the slopes: $k_{PM} * k_{PN} = -1/4$"),
#             Tex(r"3. Solve for k from the product of slopes"),
#             Tex(r"4. Prove F is inside the circle with MN as diameter"),
#             Tex(r"5. Find possible values for k"),
#         )
#         steps.arrange(DOWN, aligned_edge=LEFT)
#         for step in steps:
#             self.play(Write(step))
#             self.wait(1)
#         self.play(*[FadeOut(step) for step in steps])

#     def plot_ellipse(self):
#         axes = Axes(
#             x_range=[-4, 4],
#             y_range=[-4, 4],
#         )
#         ellipse = Ellipse(width=6,height=4)
#         F = Dot(point=[-3, 0, 0], color=GREEN)
#         F_label = MathTex("-a,0").next_to(F, LEFT)
#         F_name = Text("F").next_to(F, DOWN)
#         M = Dot(point=[1.5, 2, 0], color=BLUE)
#         M_label = MathTex("M").next_to(M, RIGHT)
#         N = Dot(point=[4, 1, 0], color=BLUE)
#         N_label = MathTex("N").next_to(N, RIGHT)
#         self.play(FadeIn(axes), DrawBorderThenFill(ellipse))
#         self.play(FadeIn(F), FadeIn(M), FadeIn(N),Write(F_label), Write(F_name), Write(M_label), Write(N_label))
#         self.wait(2)

# 4
# from manim import *

# class SolveAndVisualizeEllipse(Scene):
#     def construct(self):
#         self.show_solution_steps()
#         self.plot_ellipse()

#     def show_solution_steps(self):
#         steps = VGroup(
#             Tex(r"1. Derive the line equation: $\\frac{y^2}{3-K^2} + \\frac{(ky+x)^2}{a^2} = 1$"),
#             Tex(r"2. Compute the slopes: $k_{PM} * k_{PN} = -1/4$"),
#             Tex(r"3. Solve for k from the product of slopes"),
#             Tex(r"4. Prove F is inside the circle with MN as diameter"),
#             Tex(r"5. Find possible values for k"),
#         )
#         steps.arrange(DOWN, aligned_edge=LEFT)
#         for step in steps:
#             self.play(Write(step))
#             self.wait(1)
#         self.play(*[FadeOut(step) for step in steps])

#     def plot_ellipse(self):
#         axes = Axes(
#             x_range=[-4, 4],
#             y_range=[-4, 4],
#         )
#         ellipse = Ellipse(width=6,height=4)
#         F = Dot(point=[-3, 0, 0], color=GREEN)
#         F_label = MathTex("-a,0").next_to(F, LEFT)
#         F_name = Text("F").next_to(F, DOWN)
#         M = Dot(point=[1.5, 2, 0], color=BLUE)
#         M_label = MathTex("M").next_to(M, RIGHT)
#         N = Dot(point=[4, 1, 0], color=BLUE)
#         N_label = MathTex("N").next_to(N, RIGHT)
#         self.play(FadeIn(axes), DrawBorderThenFill(ellipse))
#         self.play(FadeIn(F), FadeIn(M), FadeIn(N),Write(F_label), Write(F_name), Write(M_label), Write(N_label))
#         self.wait(2)



# from manim import *
#
# class SolveAndVisualizeFunction(Scene):
#     def construct(self):
#         self.show_solution_steps()
#         self.plot_function()
#
#     def show_solution_steps(self):
#         steps = VGroup(
#             Tex(r"1. Derive the function: $f'(x) = 2x - 4$"),
#             Tex(r"2. Find the zero of the derivative: $x = 2$"),
#             Tex(r"3. Compute the $y$ coordinate of the vertex: $f(2) = -1$"),
#             Tex(r"4. Determine the type of the vertex: minimum point"),
#         ).arrange(DOWN, center=True, aligned_edge=LEFT)
#
#         self.play(Write(steps))
#         self.wait(1)
#
#     def plot_function(self):
#         axes = Axes(
#             x_range=[-2, 8], y_range=[-3, 4],
#             axis_config={"include_tip": False},
#             x_length=7, y_length=5,
#         )
#
#         # Updated method to create and add graph to axes
#         quadratic_graph = axes.plot(lambda x: x**2 - 4*x + 3, color=BLUE)
#         graph_label = axes.get_graph_label(quadratic_graph, label='f(x)=x^2-4x+3', x_val=3)
#
#         vertex_dot = Dot(axes.c2p(2, -1), color=RED)
#         vertex_label = MathTex("(2, -1)").next_to(vertex_dot, DOWN)
#
#         self.add(axes, quadratic_graph, graph_label, vertex_dot, vertex_label)
#         self.play(Create(quadratic_graph), Write(graph_label))
#         self.play(FadeIn(vertex_dot), Write(vertex_label))
#         self.wait(2)


# from manim import *
#
# from manimlib import ShowCreation
#
#
# class RepresentMathProblem(Scene):
#     def construct(self):
#         # 题目
#         title = Tex("Math Problem:").scale(1.2)
#         title.to_edge(UP)
#         self.play(Write(title))
#
#         # 题目具体内容
#         problem = MathTex(r"\text{Given } y=ax+b, \text{ when } x=c, \text{ find the value of } y")
#         problem.next_to(title, DOWN, buff=1)
#         self.play(Write(problem))
#
#         self.wait(2)
#
#         # 清除题目
#         self.play(*[FadeOut(mob) for mob in self.mobjects])
#         self.wait()
#
#         # 解题过程标题
#         solution_title = Tex("Solution:").scale(1.2)
#         solution_title.to_edge(UP)
#         self.play(Write(solution_title))
#
#         # 图画区域标题
#         graph_title = Tex("Graph Area:").scale(1.2)
#         graph_title.to_edge(UP, buff=0.5)
#         self.play(Write(graph_title))
#
#         # 分割线
#         line = Line(np.array([-1, -3, 0]), np.array([-1, 3, 0]))
#         # self.play(ShowCreation(line))
#
#         # 显示函数图像
#         plot = FunctionGraph(lambda x: 2 * x + 1, color=YELLOW)
#         self.play(Create(plot))
#
#         point = Dot(point=np.array([2, 5, 0]))
#         self.play(Create(point))
#
#         # 解题步骤
#         step1 = Tex("Step 1: Determine the parameter values:",
#                     " We know that the function is y=ax+b, in which a and b are constants and x is variable.")
#         step2 = Tex("Step 2: Substitute x into the function:",
#                     " We know x=c, so we can substitute this value into the function, y=ac+b.")
#         step3 = Tex("Step 3: Get the value of y:", " This is the value of y when x=c.")
#
#         for step in [step1, step2, step3]:
#             step.to_edge(RIGHT)
#             self.play(Write(step))
#             self.wait(2)
#
#         self.wait(5)


from manim import *
class run2(Scene):
    def construct(self):
        # 创建标题文本
        title1 = Text("已知一次函数y=kx+b的图像与直线y=2x+1平行，", font_size=24)
        title2 = Text("且与y=3x+1的纵坐标为-2,求k，b的值。", font_size=24)
        title_group = VGroup(title1, title2).arrange(DOWN, center=False).move_to(ORIGIN)
        # 播放标题显示动画
        self.play(Write(title_group))
        self.wait(2)
        self.play(FadeOut(title_group))

        # 第二幕：展示解析过程
        axes1 = Axes(
            x_range=[-6, 3, 1],
            y_range=[-8, 6, 1],
            x_length=7,
            y_length=6,
            axis_config={"include_numbers": True},
        )
        labels = axes1.get_axis_labels(x_label="x", y_label="y")
        # 画出两条一次函数的图像
        graph1 = axes1.plot(lambda x: 2 * x + 1, color=RED)
        graph2 = axes1.plot(lambda x: 3 * x + 1, color=BLUE)
        graph3 = axes1.plot(lambda x: 2 * x, color=YELLOW)
        graph_label1 = axes1.get_graph_label(graph1, label="y = 2x + 1", x_val=1, direction=UR)
        graph_label2 = axes1.get_graph_label(graph2, label="y = 3x + 1", x_val=3, direction=UR)
        graph_label3 = axes1.get_graph_label(graph3, label="y = 2x", x_val=3, direction=UR)
        point = Dot(axes1.coords_to_point(-1, -2), color=GREEN)
        points = VGroup(point)
        # 播放创建坐标轴、标签和图像的动画
        self.play(Create(axes1), Write(labels))
        self.play(Create(graph1), Write(graph_label1))
        self.play(Create(graph2), Write(graph_label2))
        self.play(Create(graph3), Write(graph_label3))
        self.play(Create(points))
        # 创建解释文本
        explanation = VGroup(
            Text("1.确定一次函数的形式", font_size=24),
            Text("2.应用平行线性质，斜率应该相等，即k=2", font_size=24),
            Text("3.确定函数的交点，代入y=3x+1=-2：", font_size=24),
            MathTex("3x+1=-2\Rightarrow x=-1", font_size=24),
            Text("4.将点(-1,-2)代入平行线y=2x+b：", font_size=24),
            MathTex("2*(-1)+b=-2\Rightarrow b=0", font_size=24),
        )

        # 逐步显示解释文本
        for step in explanation:
            self.play(Write(step))
            self.wait(2)
        self.wait(3)
        # 将解释文本移动到右半画布中
        explanation.move_to(RIGHT)
        # 将坐标系移动到左半画布中
        axes1.move_to(LEFT)
        # # 统一所有文本的大小
        # for step in explanation:
        #     step.set_width(FRAME_WIDTH - 2)
        # # 修改 get_graph_label 中的参数 x_val 的值，使生成的标签在视觉上不会产生重叠
        # graph_label1 = axes1.get_graph_label(graph1, label="y = 2x + 1", x_val=1, direction=UR)
        # graph_label2 = axes1.get_graph_label(graph2, label="y = 3x + 1", x_val=3, direction=UR)
        # graph_label3 = axes1.get_graph_label(graph3, label="y = 2x", x_val=3, direction=UR)[UNUSED_TOKEN_145]