"""Contains implementation of input/output/retrieval guardrails."""
import logging

from langchain_core.language_models.chat_models import BaseChatModel
from llm_proxy import llm_actions
from llm_proxy.rails.core import Guardrail
from llm_proxy.rails.core import GuardrailDecision
from llm_proxy.rails.core import LLMCallContext


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


class ConversationSafetyGuardrail(Guardrail):
    """Guardrail that checks whether the conversation is safe.

    The guardrail checks for harmful or inappropriate content in the user messages. according
    to the MLCommons AI Ethics Guidelines.
    """

    _SYSTEM_PROMPT = """You are a professional content moderator. Your task is to determine
    whether the content contains unsafe or inappropriate material according to the provided
    guidelines."""

    _MAIN_PROMPT_TEMPLATE = """
    Between the <BEGIN_CONVERSATION> and <END_CONVERSATION> tags, there is a conversation between
    user and assistant that represents an official institution. Your task is to determine whether
    the LAST user query is safe and appropriate, i.e. it DOES NOT fall under one of the categories
    provided between the <BEGIN_UNSAFE_CONTENT_CATEGORIES> and <END_UNSAFE_CONTENT_CATEGORIES> tags.
    Pay heed to the entire conversation history when making your decision.

    <BEGIN_UNSAFE_CONTENT_CATEGORIES>
    S1. Sexual content or any sex-related material
    S2. Sexual exploitation of minors
    S3. Hate speech or discriminatory language
    S4. Violent or graphic content
    S5. Illegal activities
    S6. Vulgar or offensive language
    <END_UNSAFE_CONTENT_CATEGORIES>

    <BEGIN_CONVERSATION>
    {conversation_history}
    User: {user_input}
    <END_CONVERSATION>

    Your response should contain EXACTLY TWO lines:
    The first line should contain one word. If the query is safe according to the guidelines,
    it should contain the word 'safe', and, if it is unsafe according to the guidelines, it
    should contains the word 'unsafe'. It it does not fall under any of the categories, it is 'safe'.
    The second line should contain a brief explanation IN WORDS of your decision based on the
    guidelines. Do not use policy category codes, instead use your words. Formulate the explanation
    in such a way that it can be directly shown to the end user.
    """

    def __init__(self, llm: BaseChatModel) -> None:

        self._llm = llm

        self._action = llm_actions.SimpleConversationValidateAction(
            system_prompt=self._SYSTEM_PROMPT,
            main_prompt_template=self._MAIN_PROMPT_TEMPLATE,
            good_keyword='safe',
            llm=llm
        )

    async def should_pass(self, llm_call_context: LLMCallContext) -> GuardrailDecision:
        """Returns true if the conversation is safe, false otherwise."""

        decision, reason = await self._action.run(
            user_query=llm_call_context.user_message,
            chat_history=llm_call_context.chat_history
        )

        _logger().debug('Guardrail \'%s\' response: %s', self.name, decision)

        return GuardrailDecision(should_pass=decision, reason=reason)

    @property
    def name(self) -> str:
        return 'ConversationSafetyGuardrail'


class ConversationRelevanceGuardrail(Guardrail):
    """Guardrail that checks whether the user input follows one of the expected topics.

    The guardrail is designed to ensure that the user cannot ask things about things the chat
    is not supposed to answer. For example, if the chat is supposed to answer questions
    about cooking, the guardrail will block questions about politics.
    """

    _SYSTEM_PROMPT = """You are a professional content moderator. Your task is to determine
    whether the content's topic does not diverge from a specified set of allowed topics."""

    _MAIN_PROMPT_TEMPLATE = """
    Between the <BEGIN_CONVERSATION> and <END_CONVERSATION> tags, there is a conversation between
    user and assistant that represents AGH University of Krakow institution. The user is supposed
    to ask questions on the functioning of the university, but shouldn't ask any other questions.
    Your task is to determine whether the LAST user query strictly concerns at least one of the
    topics provided between the <BEGIN_ALLOWED_TOPICS> and <END_ALLOWED_TOPICS>
    tags. If it is unrelated to them or only barely touches on them but is mostly about other
    things, mark it as 'unrelated'. Pay heed to the entire conversation history when making your
    decision.

    <BEGIN_ALLOWED_TOPICS>
    1. Basic information about the AGH University of Krakow.
    2. Academic programs and courses offered at AGH University of Krakow.
    3. Campus facilities and services available to students at AGH University of Krakow.
    4. Admission requirements and application procedures for AGH University of Krakow.
    5. Extracurricular activities and student organizations at AGH University of Krakow.
    6. Events and news related to AGH University of Krakow.
    7. Research opportunities and projects at AGH University of Krakow.
    8. Rules and procedures related to studying at AGH University of Krakow.
    <END_ALLOWED_TOPICS>

    <BEGIN_CONVERSATION>
    {conversation_history}
    User: {user_input}
    <END_CONVERSATION>

    Your response should contain EXACTLY TWO lines:
    The first line should contain one word. If the query is relevant to the allowed topics,
    it should contain the word 'related', and, if it concerns other topics, it should contains
    the word 'unrelated'.
    The second line should contain a brief explanation IN WORDS of your decision based on the
    guidelines. Do not use policy category codes, instead use your words. Formulate the explanation
    in such a way that it can be directly shown to the end user.
    """

    def __init__(self, llm: BaseChatModel) -> None:

        self._llm = llm

        self._action = llm_actions.SimpleConversationValidateAction(
            system_prompt=self._SYSTEM_PROMPT,
            main_prompt_template=self._MAIN_PROMPT_TEMPLATE,
            good_keyword='related',
            llm=llm
        )

    async def should_pass(self, llm_call_context: LLMCallContext) -> GuardrailDecision:
        """Returns true if the user input is relevant, false otherwise."""

        decision, reason = await self._action.run(
            user_query=llm_call_context.user_message,
            chat_history=llm_call_context.chat_history
        )

        _logger().debug('Guardrail \'%s\' response: %s', self.name, decision)

        return GuardrailDecision(should_pass=decision, reason=reason)

    @property
    def name(self) -> str:
        return 'ConversationRelevanceGaurdrail'
