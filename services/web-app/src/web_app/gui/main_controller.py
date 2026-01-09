"""Contains GUI related utils."""
import logging
from typing import Any
from typing import Iterator
import requests

import gradio as gr
from web_app.backend import context_retriever
from web_app.backend import llm_proxy
from web_app.backend import utils


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

        self._documents_retrieval_history: list[list[utils.ContextDocument]] = []

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

                gr.Label('AGH Chat', elem_id='agh_chat_label', show_label=False)

                chatbot = gr.Chatbot(elem_id='agh_chat',
                                     type='messages',
                                     height='70vh',
                                     show_copy_button=True,
                                     label='AGH Chat')
                msg = gr.Textbox(placeholder='Type a message...', show_label=False)

                msg.submit(  # pylint: disable=no-member
                    self._move_user_msg_to_chat, [msg, chatbot], [msg, chatbot]
                ).success(
                    self._retrieve_and_store_docs, chatbot, None
                ).success(
                    self._stream_chat_response, chatbot, chatbot
                ).success(
                    self._create_retrieved_docs_representation, None, docs_list
                )

    def _stream_chat_response(self,
                              chat_history: utils.UnstructuredChatHistory,
                              ) -> Iterator[utils.UnstructuredChatHistory]:
        """Streams the chat response based on the chat history with the latest user msg."""

        chat_history, user_message = chat_history[:-1], chat_history[-1]['content']

        structured_history = utils.ChatHistory(
            [utils.ChatMessage(message['role'], message['content'])
             for message in chat_history]
        )

        full_response = ''

        for chunk in self._llm_proxy_service.stream_chat_response(
            user_message=user_message,
            chat_history=structured_history,
            context_docs=self._documents_retrieval_history[-1]
        ):

            if 'error' in chunk:
                _logger().debug('Received error from llm-proxy: %s', chunk['error'])

                yield chat_history
                self._documents_retrieval_history.pop()

                raise gr.Error(chunk['error'], duration=None)

            full_response += chunk['content']

            yield chat_history + [{'role': 'user', 'content': user_message},
                                  {'role': 'assistant', 'content': full_response}]

    def _create_retrieved_docs_representation(self) -> gr.Markdown:
        """Concatenates the documents retrieved till now and returns their Markdown repr."""

        retrieval_history_reprs: list[str] = []

        for docs in self._documents_retrieval_history:

            docs_repr = (self._DOC_MD_TEMPLATE.format(title=doc.metadata['title'],
                                                      content=doc.content)
                         for doc in docs)

            retrieval_history_reprs.append('\n\n'.join(docs_repr))

        docs_list_repr = (self._DOCS_LIST_MD_TEMPLATE.format(round_nr=i + 1,
                                                             docs=repr)
                          for i, repr in enumerate(retrieval_history_reprs))

        return gr.Markdown('\n---\n'.join(docs_list_repr))

    def _move_user_msg_to_chat(self,
                               user_message: str,
                               chat_history: utils.UnstructuredChatHistory | None,
                               ) -> tuple[str, utils.UnstructuredChatHistory]:
        """Migrates the submitted user message to the chat history and resets the msg input."""

        chat_history = chat_history or []

        return '', chat_history + [{'role': 'user', 'content': user_message}]

    def _retrieve_and_store_docs(self,
                                 chat_history: utils.UnstructuredChatHistory,
                                 ) -> None:
        """Retrieves context documents and stores them internally."""

        chat_history, user_message = chat_history[:-1], chat_history[-1]['content']

        gr.Info('Collecting context documents...', duration=5)

        try:
            context_docs = self._context_retriever_service.collect_context_info(
                user_message=user_message,
                chat_history=utils.ChatHistory(
                    [utils.ChatMessage(message['role'], message['content'])
                     for message in chat_history]
                )
            )

        except requests.HTTPError as e:
            _logger().error('Failed to collect context info from backend: %s', e)

            raise gr.Error('Failed to collect context info from backend.', duration=None)

        self._documents_retrieval_history.append(context_docs)
