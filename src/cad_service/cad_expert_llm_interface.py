import logging
from pathlib import Path

from llm_utils.openai_api.text_message_content import TextMessageContent
from llm_utils.prompt_generation.prompt import Prompt
from llm_utils.textgen_api.textgen_api import TextGenApi
from python_utils.string_utils import get_markup_from_text

logger = logging.getLogger(__name__)


class CADExpertLLMInterface:

    def __init__(self, llm_service: TextGenApi) -> None:
        self.llm_service = llm_service
        self.resources = Path(__file__).parent.parent.parent / "prompts"
        assert self.resources.exists()

    def query_cad_object(self, query: str, log: bool = False):
        """given user prompt `query` retrieve information about cad shape `shape`"""
        prompt = Prompt.load_from_file(self.resources / "prompt-query-matching-part.xml")
        prompt.replace("{instruction}", query)
        chat = prompt.to_chat()

        response_message = self.llm_service.do_call(chat=chat)

        logger.info(str(response_message))

        content = response_message.content[0]
        assert isinstance(content, TextMessageContent)
        response_text = content.text

        if log:
            with open("out/response.txt", "w") as f:
                f.write(response_text)

        return get_markup_from_text(response_text, ["python"])
