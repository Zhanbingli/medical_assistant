"""
Streamlit UI 组件库
提供更好的 UI 元素，减少 unsafe_allow_html 使用
"""

import streamlit as st
from typing import Dict, Any, List, Optional


class MedicalUI:
    """医学 AI 助手 UI 组件库"""
    
    @staticmethod
    def render_header(title: str, subtitle: str):
        """渲染页面头部"""
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #1a73e8, #4285f4); 
                    padding: 1.5rem; border-radius: 15px; color: white; 
                    text-align: center; margin-bottom: 1.5rem;'>
            <h1 style='margin: 0; font-size: 1.8rem;'>{title}</h1>
            <p style='margin: 0.5rem 0 0 0; opacity: 0.9; font-size: 1rem;'>{subtitle}</p>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_welcome_message():
        """渲染欢迎消息"""
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #1a73e8, #4285f4); 
                    color: white; padding: 1.5rem; border-radius: 20px; margin: 1rem 0;'>
            <div style='display: flex; align-items: center; margin-bottom: 1rem;'>
                <span style='font-size: 2rem; margin-right: 0.5rem;'>👨‍⚕️</span>
                <strong style='font-size: 1.2rem;'>AI 医学助手</strong>
            </div>
            <div>你好，我是你的专业医学助手。我会基于本地知识库为你提供循证诊断建议。</div>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_diagnosis_card(title: str, content: str):
        """渲染诊断结果卡片"""
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #1a73e8, #4285f4); 
                    color: white; border-radius: 15px; padding: 1.5rem; 
                    margin: 1rem 0; border-left: 5px solid white;'>
            <h4 style='margin-top: 0; color: white; margin-bottom: 1rem;'>
                📊 {title}
            </h4>
            <div style='color: white; line-height: 1.8;'>{content}</div>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_checklist_card(title: str, content: str):
        """渲染检查建议卡片"""
        st.markdown(f"""
        <div style='background: white; border: 2px solid #34a853; 
                    border-radius: 15px; padding: 1.5rem; margin: 1rem 0;'>
            <h4 style='margin-top: 0; color: #34a853; margin-bottom: 1rem;'>
                🔬 {title}
            </h4>
            <div style='line-height: 1.8; color: #333;'>{content}</div>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_file_item(filename: str, icon: str = "📖"):
        """渲染文件列表项"""
        st.markdown(f"""
        <div style='background: rgba(255, 255, 255, 0.9); 
                    border-radius: 10px; padding: 0.8rem; 
                    margin: 0.3rem 0; border-left: 4px solid #1a73e8;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.05);'>
            {icon} <strong>{filename}</strong>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_upload_area():
        """渲染上传区域"""
        st.markdown("""
        <div style='border: 2px dashed #1a73e8; border-radius: 15px; 
                    padding: 2rem; text-align: center; 
                    background: rgba(26, 115, 232, 0.05);'>
            <p style='color: #1a73e8; font-size: 1.1rem; margin-bottom: 0.5rem;'>
                📚 拖拽文件到此处或点击选择
            </p>
            <p style='color: #666; font-size: 0.9rem;'>支持 Markdown 格式 (.md)</p>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_model_info(embedding_model: str, llm_model: str, reranker_model: str):
        """渲染模型信息"""
        with st.expander("🤖 模型信息", expanded=False):
            st.markdown(f"""
            - **嵌入模型**: {embedding_model}
            - **推理模型**: {llm_model}
            - **重排序**: {reranker_model}
            """)
    
    @staticmethod
    def render_usage_guide():
        """渲染使用指南"""
        with st.expander("📖 使用说明", expanded=False):
            st.markdown("""
            ### 如何获得最佳诊断建议？
            
            1. **描述主要症状** (如：发热、咳嗽、胸痛)
            2. **说明持续时间** (如：3天、1周)
            3. **提供关键体征** (如：体温38.5°C、血压120/80)
            4. **提及既往病史** (如：高血压病史5年)
            
            ### 🌐 网络搜索
            当知识库内容不足时，系统会自动使用网络搜索补充信息。
            """)


class ChatComponents:
    """聊天界面组件"""

    @staticmethod
    def render_thinking_process(thought: str):
        """渲染思考过程"""
        st.markdown(f"""
        <div style='background: rgba(255, 152, 0, 0.1); padding: 1rem; 
                    border-radius: 10px; margin: 0.5rem 0; 
                    border-left: 3px solid #ff9800;'>
            <strong style='color: #ff9800;'>🤔 思考过程</strong>
            <div style='margin-top: 0.5rem; color: #666;'>{thought}</div>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_searching(keyword: str):
        """渲染搜索状态"""
        st.markdown(f"""
        <div style='background: rgba(76, 175, 80, 0.1); padding: 1rem; 
                    border-radius: 10px; margin: 0.5rem 0;
                    border-left: 3px solid #4caf50;'>
            <strong style='color: #4caf50;'>🔍 正在检索</strong>
            <div style='margin-top: 0.5rem;'>关键词: <code>{keyword}</code></div>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_observation(content: str):
        """渲染观察结果"""
        with st.expander("📋 检索详情", expanded=False):
            st.markdown(f"<div style='font-family: monospace; padding: 0.5rem; background: #f5f5f5; border-radius: 5px;'>{content}</div>", unsafe_allow_html=True)
    
    @staticmethod
    def render_web_search_results(results: List[Dict[str, str]]):
        """渲染网络搜索结果"""
        st.markdown("### 🌐 网络搜索结果")
        for result in results:
            with st.expander(result.get('title', '无标题'), expanded=True):
                st.markdown(result.get('snippet', ''))
                if result.get('url'):
                    st.link_button("查看原文", result['url'])


class SidebarComponents:
    """侧边栏组件"""

    @staticmethod
    def render_knowledge_base_section(db) -> str:
        """渲染知识库管理区域，返回选中文件"""
        st.subheader("📚 知识库管理")
        
        current_files = db.get_existing_files()
        if not current_files:
            st.info("暂无数据，请上传教材")
            return None
        
        # 显示文件列表
        for f in sorted(current_files):
            ChatComponents.render_file_item(f)
        
        st.markdown("---")
        
        # 删除文件功能
        st.subheader("🗑️ 管理")
        file_to_delete = st.selectbox(
            "选择要删除的文件",
            options=[""] + sorted(list(current_files)),
            index=0
        )
        
        if file_to_delete and st.button("删除文件", type="secondary"):
            success, error = db.delete_file(file_to_delete)
            if success:
                st.success(f"已删除 {file_to_delete}")
                st.rerun()
            else:
                st.error(f"删除失败: {error}")
        
        return file_to_delete
    
    @staticmethod
    def render_debug_settings() -> tuple:
        """渲染调试设置，返回 (debug_mode, web_search_enabled)"""
        debug_mode = st.toggle('🛠️ 调试模式', value=False, help="显示详细的检索过程")
        web_search_enabled = st.toggle('🌐 启用网络搜索', value=True, help="当知识库为空时自动搜索网络")
        return debug_mode, web_search_enabled
    
    @staticmethod
    def render_upload_section(db, embedder) -> int:
        """渲染文件上传区域，返回成功上传数量"""
        st.subheader("📤 上传新书")
        
        # 优化选项
        optimize_md = st.toggle("🔧 自动优化Markdown", value=True, help="自动清理格式并优化结构")
        
        # 文件上传
        uploaded_files = st.file_uploader(
            "选择文件",
            type=["md"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        
        success_count = 0
        
        if uploaded_files:
            # 统计文件大小
            total_size = sum(file.size for file in uploaded_files) / 1024 / 1024
            st.info(f"共选择了 {len(uploaded_files)} 个文件，总计 {total_size:.2f} MB")
            
            if st.button("🚀 开始学习", type="primary", use_container_width=True):
                for file in uploaded_files:
                    with st.status(f"📖 正在学习 {file.name}...", expanded=True) as status:
                        try:
                            content = file.read().decode("utf-8", errors='ignore')
                            
                            if optimize_md:
                                from src.utils.markdown_optimizer import optimize_markdown_for_rag
                                content = optimize_markdown_for_rag(content)
                            
                            progress_text = status.empty()
                            
                            def update_progress(progress, text):
                                progress_text.progress(progress, text=f"{file.name}: {text}")
                            
                            success, info = embedder.process_file(content, file.name, db, update_progress)
                            
                            if success:
                                success_count += 1
                                status.update(
                                    label=f"✅ 学习成功！添加了 {info} 条知识",
                                    state="complete"
                                )
                                st.toast(f"📚 成功学习《{file.name}》", icon="🎉")
                            elif info == "EXIST":
                                status.update(
                                    label=f"⚠️ 《{file.name}》 已存在",
                                    state="complete"
                                )
                            else:
                                status.update(
                                    label=f"❌ 学习失败: {info}",
                                    state="error"
                                )
                        except Exception as e:
                            status.update(
                                label=f"❌ 处理失败: {str(e)}",
                                state="error"
                            )
                
                if success_count > 0:
                    st.success(f"🎉 成功学习 {success_count} 本新书！")
                    time.sleep(1)
                    st.rerun()
        
        return success_count
