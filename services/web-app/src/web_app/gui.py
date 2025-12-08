"""Contains GUI related utils."""
import logging
from typing import Iterator
from typing import List
from typing import Tuple

import gradio as gr
import web_app
from web_app.backend_communication import ChatHistoryType


def _logger() -> logging.Logger:
    return logging.getLogger('web_app')


def _create_retrieved_docs_representation(docs: List[Tuple[str, str]]) -> gr.Markdown:
    """Concatenates the retrieved documents and returns their Markdown representation."""

    content = '\n'.join(
        f"""
        ### {title}
        {content}
        """
        for title, content in docs
    )

    return gr.Markdown(content)


class MainController:
    """Renders GUI elements and handles controller-view interactions."""

    def __init__(self,
                 backend_service: web_app.backend_communication.BackendService):

        self._backend_service = backend_service

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

                    docs_list = gr.Markdown('Retrieved documents will be displayed here.',)

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
                                 history: ChatHistoryType
                                 ) -> Iterator[Tuple[ChatHistoryType, gr.Markdown]]:
        """Main callback for the chat interface."""

        history = history or []

        try:
            context_docs = self._backend_service.collect_context_info(
                user_message=user_message,
                chat_history=history
            )
        except Exception as e:
            _logger().error('Failed to collect context info from backend.')
            raise gr.Error('Failed to collect context info from backend.', duration=5) from e

        context_docs_repr = _create_retrieved_docs_representation(context_docs)

        chat_response = self._backend_service.stream_chat_response(
            user_message=user_message,
            chat_history=history,
            context_docs=context_docs
        )

        full_text_response = ''
        for chunk in chat_response:

            token = chunk.get('content', '')
            full_text_response += token

            chat_message = [{
                'role': 'assistant',
                'content': full_text_response
            }]

            yield chat_message, context_docs_repr
