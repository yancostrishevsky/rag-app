"""Contains GUI related utils."""
import logging
from typing import Iterator
from typing import List
from typing import Tuple
from typing import Optional
from typing import Dict
from typing import Any

import gradio as gr

from web_app.backend import utils
from web_app.backend import (context_retriever, llm_proxy)


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


class MainController:
    """Renders GUI elements and handles controller-view interactions."""

    _DOC_MD_TEMPLATE = """
### {title}

{content}
"""

    _DOCS_LIST_MD_TEMPLATE = """
## Round {round_nr}

{docs}
"""

    def __init__(self,
                 context_retriever_service: context_retriever.ContextRetrieverService,
                 llm_proxy_service: llm_proxy.LLMProxyService):

        self._context_retriever_service = context_retriever_service
        self._llm_proxy_service = llm_proxy_service

        self._documents_retrieval_history: List[List[utils.ContextDocument]] = []

    def render_gui(self) -> None:
        """Renders the UI for application and assigns the necessary callbacks."""

        with gr.Row(elem_id='agh_header_row', height='70vh'):

            with gr.Column(elem_id='context_column', scale=1):
                gr.Markdown(
                    """
                    # Context
                    """
                )

                gr.Markdown(
                    """
                    ## Retrieved Documents
                    """
                )

                with gr.Column(elem_id='retrieved_docs',
                               variant='panel',
                               elem_classes='retrieved-docs'):

                    docs_list = gr.Markdown('Retrieved documents will be displayed here.')

            with gr.Column(elem_id='chat_column', scale=3):

                gr.ChatInterface(
                    self._chat_interface_callback,
                    chatbot=gr.Chatbot(elem_id='agh_chat',
                                       type='messages',
                                       height='70vh',
                                       show_copy_button=True),
                    title='AGH Chat',
                    type='messages',
                    textbox=gr.Textbox(
                        placeholder='Type a message...', label='Your message'),
                    additional_outputs=[docs_list],
                )

    def _chat_interface_callback(self,
                                 user_message: str,
                                 raw_history: Optional[utils.UnstructuredChatHistory],
                                 ) -> Iterator[Tuple[Dict[str, Any], gr.Markdown]]:
        """Main callback for the chat interface.

        It first retrieves context docs from the context-retriever, updates the internal documents
        storage and streams the llm-proxy response together with the markdown representation
        of the retrieval history.
        """

        raw_history = raw_history or []

        chat_history = utils.ChatHistory(
            [utils.ChatMessage(message['role'], message['content'])
             for message in raw_history]
        )

        try:
            context_docs = self._context_retriever_service.collect_context_info(
                user_message=user_message,
                chat_history=chat_history
            )

            self._documents_retrieval_history.append(context_docs)

        except Exception as e:
            _logger().error('Failed to collect context info from backend.')
            raise gr.Error('Failed to collect context info from backend.', duration=5) from e

        context_docs_repr = self._create_retrieved_docs_representation()

        for chat_message in self._stream_llm_response(user_message,
                                                      chat_history,
                                                      context_docs):
            yield chat_message, context_docs_repr

    def _stream_llm_response(self,
                             user_message: str,
                             chat_history=utils.ChatHistory,
                             context_docs=List[utils.ContextDocument]) -> Iterator[Dict[str, Any]]:
        """Yields concatenated chunks retrieved from the llm."""

        chat_response = self._llm_proxy_service.stream_chat_response(
            user_message=user_message,
            chat_history=chat_history,
            context_docs=context_docs
        )

        full_text_response = ''
        for chunk in chat_response:

            full_text_response += chunk['content']

            yield {
                'role': 'assistant',
                'content': full_text_response
            }

    def _create_retrieved_docs_representation(self) -> gr.Markdown:
        """Concatenates the documents retrieved till now and returns their Markdown repr."""

        retrieval_history_reprs: List[str] = []

        for docs in self._documents_retrieval_history:

            docs_repr = (self._DOC_MD_TEMPLATE.format(title=doc.title,
                                                      content=doc.content)
                         for doc in docs)

            retrieval_history_reprs.append('\n\n'.join(docs_repr))

        docs_list_repr = (self._DOCS_LIST_MD_TEMPLATE.format(round_nr=i + 1,
                                                             docs=repr)
                          for i, repr in enumerate(retrieval_history_reprs))

        return gr.Markdown('\n---\n'.join(docs_list_repr))
