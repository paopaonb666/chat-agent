import pytest
from unittest.mock import patch, AsyncMock


class TestWebSearchTool:
    def test_tool_has_correct_metadata(self):
        from app.langgraph_agent.tools.web_search import web_search_tool

        assert web_search_tool.name == "web_search_tool"
        assert "搜索互联网获取最新信息" in web_search_tool.description
        assert "query" in web_search_tool.args_schema.model_fields

    @pytest.mark.asyncio
    async def test_tool_calls_web_search_service(self):
        from app.langgraph_agent.tools.web_search import web_search_tool

        mock_sources = [
            {"title": "Test", "url": "http://test.com", "snippet": "A test result", "position": 1}
        ]

        with patch("app.langgraph_agent.tools.web_search.web_search", new=AsyncMock(return_value=mock_sources)):
            with patch("app.langgraph_agent.tools.web_search.get_stream_writer") as mock_writer:
                mock_writer.return_value = lambda x: None
                result = await web_search_tool.ainvoke({"query": "test query"})

        assert "Test" in result
        assert "http://test.com" in result

    @pytest.mark.asyncio
    async def test_tool_handles_empty_results(self):
        from app.langgraph_agent.tools.web_search import web_search_tool

        with patch("app.langgraph_agent.tools.web_search.web_search", new=AsyncMock(return_value=[])):
            with patch("app.langgraph_agent.tools.web_search.get_stream_writer") as mock_writer:
                mock_writer.return_value = lambda x: None
                result = await web_search_tool.ainvoke({"query": "no results"})

        assert "未找到相关搜索结果" in result

    @pytest.mark.asyncio
    async def test_tool_handles_exception_gracefully(self):
        from app.langgraph_agent.tools.web_search import web_search_tool

        with patch("app.langgraph_agent.tools.web_search.web_search", new=AsyncMock(side_effect=Exception("boom"))):
            with patch("app.langgraph_agent.tools.web_search.get_stream_writer") as mock_writer:
                mock_writer.return_value = lambda x: None
                result = await web_search_tool.ainvoke({"query": "error"})

        assert "未找到相关搜索结果" in result

    @pytest.mark.asyncio
    async def test_tool_enhances_bad_query(self):
        from app.langgraph_agent.tools.web_search import web_search_tool

        mock_sources = [
            {"title": "拉康 精神分析", "url": "http://test.com", "snippet": "理论", "position": 1}
        ]

        with patch("app.langgraph_agent.tools.web_search.web_search", new=AsyncMock(return_value=mock_sources)):
            with patch("app.langgraph_agent.tools.web_search.get_stream_writer") as mock_writer:
                mock_writer.return_value = lambda x: None
                result = await web_search_tool.ainvoke({"query": "拉"})

        assert "拉康" in result or "未找到" in result

    @pytest.mark.asyncio
    async def test_tool_warns_on_empty_results(self):
        from app.langgraph_agent.tools.web_search import web_search_tool

        with patch("app.langgraph_agent.tools.web_search.web_search", new=AsyncMock(return_value=[])):
            with patch("app.langgraph_agent.tools.web_search.get_stream_writer") as mock_writer:
                mock_writer.return_value = lambda x: None
                result = await web_search_tool.ainvoke({"query": "无结果测试"})

        assert "未找到相关搜索结果" in result
        assert "专有名词" in result
