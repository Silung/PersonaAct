"""
Interview Agent Gradio 前端
"""
import gradio as gr
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from interview_agent import InterviewAgent
from behavior_analyzer import BehaviorAnalyzer


def create_app(
    data_dir: str = "data/yqg",
    raw_data_dir: str = "raw_data/yqg",
    api_base: str = "http://127.0.0.1:8012/v1",
    api_key: str = "1234567890",
    model: str = "qwen"
):
    agent = None
    
    def format_progress(agent):
        """格式化进度显示为 TODO list"""
        if not agent or not agent.interview_plan:
            return "📋 访谈进度: 未开始"
        
        progress = agent.get_current_progress()
        lines = [f"## 📋 访谈进度: {progress['current']}/{progress['total']}"]
        lines.append("")
        
        for i, section in enumerate(agent.interview_plan, 1):
            if i < progress['current']:
                # 已完成
                status = "✅"
            elif i == progress['current']:
                # 进行中
                status = f"🔄 ({progress['section_turn']}/{progress['section_max_turns']}轮)"
            else:
                # 未开始
                status = "⏳"
            
            lines.append(f"{status} {i}. {section['title']}")
        
        return "\n".join(lines)
    
    def initialize_interview(dataset):
        nonlocal agent
        
        # 构建数据集路径
        selected_data_dir = f"data/{dataset}"
        selected_raw_dir = f"raw_data/{dataset}"
        
        agent = InterviewAgent(
            data_dir=selected_data_dir,
            raw_data_dir=selected_raw_dir,
            api_base=api_base,
            api_key=api_key,
            model=model
        )
        init_result = agent.initialize()
        first_question = agent.get_first_question()
        
        progress_md = format_progress(agent)
        
        return (
            [[None, first_question]],
            init_result.get("behavior_summary", ""),
            json.dumps(init_result.get("analysis", {}), ensure_ascii=False, indent=2),
            "访谈进行中...",
            progress_md,
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True),
        )
    
    def chat_step(user_input, chat_history):
        nonlocal agent
        
        if not agent:
            return chat_history + [[user_input, "请先点击「开始访谈」"]], "", "📋 访谈进度: 未开始", gr.update()
        
        if not user_input.strip():
            return chat_history, "", format_progress(agent), gr.update()
        
        # 检查是否是生成人设的命令
        if user_input.strip() in ["生成人设", "生成画像", "生成persona", "完成访谈"]:
            chat_history = chat_history + [[user_input, None]]
            response = agent.generate_final_persona()
            chat_history[-1][1] = response
            persona = agent.get_current_persona()
            progress_md = format_progress(agent) + "\n\n✅ **访谈已完成！**"
            return chat_history, persona, progress_md, gr.update(interactive=False)
        
        chat_history = chat_history + [[user_input, None]]
        result = agent.chat(user_input)
        
        # 解析返回结果
        response = result["response"]
        section_changed = result["section_changed"]
        auto_ended = result.get("auto_ended", False)
        interview_completed = result.get("interview_completed", False)
        
        # 如果访谈完成，自动生成人设
        if interview_completed:
            response += f"\n\n---\n🎉 **所有环节已完成！正在生成用户画像...**"
            chat_history[-1][1] = response
            
            # 自动生成人设
            persona_text = agent.generate_final_persona()
            chat_history.append([None, f"**用户画像生成完成：**\n\n{persona_text}"])
            
            persona = agent.get_current_persona()
            progress_md = format_progress(agent) + "\n\n✅ **访谈已完成！**"
            
            return chat_history, persona, progress_md, gr.update(interactive=False)
        
        # 如果环节切换，生成新环节的第一个问题
        if section_changed and not interview_completed:
            section_info = result["section_info"]
            if section_info["current"] <= section_info["total"]:
                next_section = agent.interview_plan[agent.current_section]
                if auto_ended:
                    response += f"\n\n---\n⏰ **当前环节已达到最大轮数**\n💡 **进入下一环节**: {next_section['title']}"
                else:
                    response += f"\n\n---\n💡 **进入下一环节**: {next_section['title']}"
                
                # 生成新环节的第一个问题
                next_question = agent.generate_section_question()
                chat_history[-1][1] = response
                chat_history.append([None, next_question])
            else:
                chat_history[-1][1] = response
        else:
            chat_history[-1][1] = response
        
        persona = agent.get_current_persona()
        progress_md = format_progress(agent)
        
        return chat_history, persona, progress_md, gr.update()
    
    def export_persona():
        nonlocal agent
        if not agent:
            return "{}"
        return json.dumps(agent.get_structured_persona(), ensure_ascii=False, indent=2)
    
    def generate_persona_click(chat_history):
        nonlocal agent
        if not agent:
            return chat_history, "", gr.update()
        
        # 生成人设，只更新 Persona 栏，不改变对话历史
        agent.generate_final_persona()
        persona = agent.get_current_persona()
        
        # 保持输入框可用，允许继续对话
        return chat_history, persona, gr.update()
    
    def reset_interview():
        nonlocal agent
        agent = None
        return [], "点击「开始访谈」加载数据", "{}", "", "📋 访谈进度: 未开始", gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False)
    
    with gr.Blocks(title="🧠 Interview Agent", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🧠 User Interview Agent\n基于行为数据的用户访谈与画像生成")
        
        with gr.Row():
            dataset_selector = gr.Radio(
                choices=["yqg", "zsl"],
                value="yqg",
                label="选择数据集",
                info="选择要分析的用户数据"
            )
        
        # 访谈进度显示
        with gr.Row():
            progress_display = gr.Markdown(value="📋 访谈进度: 未开始")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📊 行为摘要")
                behavior_display = gr.Markdown(value="点击「开始访谈」加载数据")
                
                with gr.Accordion("详细数据", open=False):
                    analysis_display = gr.Code(value="{}", language="json")
            
            with gr.Column(scale=2):
                gr.Markdown("### 💬 访谈")
                chatbot = gr.Chatbot(height=400)
                
                with gr.Row():
                    msg = gr.Textbox(placeholder="输入回答...", show_label=False, scale=4, interactive=False)
                    submit_btn = gr.Button("发送", scale=1, interactive=False)
                
                with gr.Row():
                    start_btn = gr.Button("🚀 开始访谈", variant="primary")
                    generate_btn = gr.Button("✨ 生成人设", variant="secondary", interactive=False)
                    reset_btn = gr.Button("🔄 重置")
            
            with gr.Column(scale=1):
                gr.Markdown("### 🧾 Persona")
                persona_display = gr.Markdown(value="")
                export_btn = gr.Button("📤 导出")
                export_output = gr.Code(value="", language="json")
        
        start_btn.click(initialize_interview, inputs=[dataset_selector], outputs=[chatbot, behavior_display, analysis_display, persona_display, progress_display, msg, submit_btn, generate_btn])
        reset_btn.click(reset_interview, outputs=[chatbot, behavior_display, analysis_display, persona_display, progress_display, msg, submit_btn, generate_btn])
        
        msg.submit(chat_step, [msg, chatbot], [chatbot, persona_display, progress_display, msg]).then(lambda: "", outputs=[msg])
        submit_btn.click(chat_step, [msg, chatbot], [chatbot, persona_display, progress_display, msg]).then(lambda: "", outputs=[msg])
        generate_btn.click(generate_persona_click, [chatbot], [chatbot, persona_display, msg])
        export_btn.click(export_persona, outputs=[export_output])
    
    return demo


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/yqg")
    parser.add_argument("--raw_data_dir", default="raw_data/yqg")
    parser.add_argument("--api_base", default="http://127.0.0.1:8012/v1")
    parser.add_argument("--api_key", default="1234567890")
    parser.add_argument("--model", default="qwen")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    

    
    demo = create_app(args.data_dir, args.raw_data_dir, args.api_base, args.api_key, args.model)
    demo.launch(server_port=args.port, share=args.share, server_name="0.0.0.0")


if __name__ == "__main__":
    main()
