"""Contains GUI related utils."""
import logging
from typing import Iterator

import gradio as gr
import requests
from web_app.backend import context_retriever
from web_app.backend import llm_proxy
from web_app.backend import utils


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


class MainController:
    """Renders GUI elements and handles controller-view interactions."""

    _NO_CONTEXT_RESPONSE = (
        'I could not find supporting information for this question in the uploaded document '
        'database. Please upload relevant documents or ask about information that is present in '
        'the knowledge base.'
    )

    _DOC_MD_TEMPLATE = """
### {title}

Page: {page}
Score: {score}

{content}
"""

    _DOCS_LIST_MD_TEMPLATE = """
## Round {round_nr}

{docs}
"""

    _DEBUG_TURN_MD_TEMPLATE = """
## Round {round_nr}

### User Message
{user_message}

### Safety Check
Status: {safety_status}
Reason: {safety_reason}

```text
{safety_raw}
```

### Relevance Check
Status: {relevance_status}
Reason: {relevance_reason}

```text
{relevance_raw}
```

### Rewritten Query
```text
{rewritten_query}
```

### Prompt Preview
```text
{prompt_preview}
```

### Final Answer
```text
{final_response}
```
"""

    def __init__(self,
                 context_retriever_service: context_retriever.ContextRetrieverService,
                 llm_proxy_service: llm_proxy.LLMProxyService):

        self._context_retriever_service = context_retriever_service
        self._llm_proxy_service = llm_proxy_service

        self._documents_retrieval_history: list[list[utils.ContextDocument]] = []
        self._debug_history: list[utils.TurnDebugInfo] = []

    def render_gui(self) -> None:
        """Renders the UI for application and assigns the necessary callbacks."""

        with gr.Row(elem_id='agh_header_row', height='70vh'):

            with gr.Column(elem_id='context_column', scale=1):
                gr.Markdown(
                    """
                    # Context
                    """
                )

                file_upload = gr.File(label='Upload Files for Context')

                file_upload.upload(  # pylint: disable=no-member
                    self._upload_file, file_upload, file_upload,
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

                gr.Markdown(
                    """
                    ## Debug Trace
                    """
                )

                with gr.Column(elem_id='debug_trace',
                               variant='panel',
                               elem_classes='debug-trace'):

                    debug_trace = gr.Markdown('Debug trace will be displayed here.')

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
                    self._validate_user_msg, chatbot, chatbot
                ).success(
                    self._retrieve_and_store_docs, chatbot, None
                ).success(
                    self._stream_chat_response, chatbot, chatbot
                ).success(
                    self._refresh_side_panels, None, [docs_list, debug_trace]
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

        if not self._documents_retrieval_history[-1]:
            self._debug_history[-1].final_response = self._NO_CONTEXT_RESPONSE
            yield chat_history + [{'role': 'user', 'content': user_message},
                                  {'role': 'assistant', 'content': self._NO_CONTEXT_RESPONSE}]
            return

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
                self._debug_history.pop()

                raise gr.Error(chunk['error'],
                               title='Error while generating chat response',
                               duration=None)

            full_response += chunk['content']

            yield chat_history + [{'role': 'user', 'content': user_message},
                                  {'role': 'assistant', 'content': full_response}]

        self._debug_history[-1].final_response = full_response

    def _create_retrieved_docs_representation(self) -> str:
        """Concatenates the documents retrieved till now and returns their Markdown repr."""

        retrieval_history_reprs: list[str] = []

        for docs in self._documents_retrieval_history:

            docs_repr = (self._DOC_MD_TEMPLATE.format(
                title=doc.metadata.get('title', 'Unknown'),
                page=doc.metadata.get('page', 'Unknown'),
                score=(f'{doc.retrieval_score:.4f}'
                       if doc.retrieval_score is not None else 'n/a'),
                content=doc.content)
                         for doc in docs)

            retrieval_history_reprs.append('\n\n'.join(docs_repr))

        docs_list_repr = (self._DOCS_LIST_MD_TEMPLATE.format(round_nr=i + 1,
                                                             docs=repr)
                          for i, repr in enumerate(retrieval_history_reprs))

        return '\n---\n'.join(docs_list_repr) or 'Retrieved documents will be displayed here.'

    def _create_debug_trace_representation(self) -> str:
        """Returns Markdown representation of the debug trace history."""

        turns_repr = [
            self._DEBUG_TURN_MD_TEMPLATE.format(
                round_nr=i + 1,
                user_message=turn.user_message,
                safety_status=('PASS'
                               if turn.safety_check is not None and turn.safety_check.is_ok
                               else ('FAIL' if turn.safety_check is not None else 'PENDING')),
                safety_reason=(turn.safety_check.reason
                               if turn.safety_check is not None and turn.safety_check.reason
                               else 'None'),
                safety_raw=(turn.safety_check.raw_response
                            if turn.safety_check is not None and turn.safety_check.raw_response
                            else 'No raw response captured.'),
                relevance_status=('PASS'
                                  if turn.relevance_check is not None and turn.relevance_check.is_ok
                                  else ('FAIL'
                                        if turn.relevance_check is not None else 'PENDING')),
                relevance_reason=(turn.relevance_check.reason
                                  if turn.relevance_check is not None and turn.relevance_check.reason
                                  else 'None'),
                relevance_raw=(turn.relevance_check.raw_response
                               if (turn.relevance_check is not None and
                                   turn.relevance_check.raw_response)
                               else 'No raw response captured.'),
                rewritten_query=turn.rewritten_query or 'Not available.',
                prompt_preview=turn.prompt_preview or 'Not available.',
                final_response=turn.final_response or 'Not available.',
            )
            for i, turn in enumerate(self._debug_history)
        ]

        return '\n---\n'.join(turns_repr) or 'Debug trace will be displayed here.'

    def _refresh_side_panels(self) -> tuple[str, str]:
        """Returns refreshed content for retrieved docs and debug trace panels."""

        return (
            self._create_retrieved_docs_representation(),
            self._create_debug_trace_representation()
        )

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
            retrieval_result = self._context_retriever_service.collect_context_info(
                user_message=user_message,
                chat_history=utils.ChatHistory(
                    [utils.ChatMessage(message['role'], message['content'])
                     for message in chat_history]
                )
            )

        except requests.HTTPError as e:
            _logger().error('Failed to collect context info from backend: %s', e)

            raise gr.Error('Failed to collect context info from backend.', duration=None)

        self._documents_retrieval_history.append(retrieval_result.context_docs)
        self._debug_history[-1].rewritten_query = retrieval_result.rewritten_query
        self._debug_history[-1].context_docs = retrieval_result.context_docs

        try:
            self._debug_history[-1].prompt_preview = self._llm_proxy_service.build_chat_debug_prompt(
                user_message=user_message,
                chat_history=utils.ChatHistory(
                    [utils.ChatMessage(message['role'], message['content'])
                     for message in chat_history]
                ),
                context_docs=retrieval_result.context_docs
            )
        except requests.HTTPError as e:
            _logger().error('Failed to build chat debug prompt: %s', e)
            self._debug_history[-1].prompt_preview = 'Failed to build prompt preview.'

    def _validate_user_msg(self,
                           chat_history: utils.UnstructuredChatHistory,
                           ) -> Iterator[utils.UnstructuredChatHistory]:
        """Validates the user message for safety and relevance."""

        chat_history, user_message = chat_history[:-1], chat_history[-1]['content']

        structured_history = utils.ChatHistory(
            [utils.ChatMessage(message['role'], message['content'])
             for message in chat_history]
        )

        gr.Info('Validating user message...', duration=5)

        try:
            safety_check = self._llm_proxy_service.check_input_safety(
                user_message, structured_history)

            self._debug_history.append(utils.TurnDebugInfo(
                user_message=user_message,
                safety_check=safety_check
            ))

            if not safety_check.is_ok:
                yield chat_history
                raise gr.Error(safety_check.reason,
                               title='Input Safety Check Failed',
                               duration=None)

            relevance_check = self._llm_proxy_service.check_input_relevance(
                user_message, structured_history)

            self._debug_history[-1].relevance_check = relevance_check

            if not relevance_check.is_ok:
                yield chat_history
                raise gr.Error(relevance_check.reason,
                               title='Input Relevance Check Failed',
                               duration=None)

        except requests.HTTPError as e:
            _logger().error('Failed to validate user message: %s', e)

            raise gr.Error('Failure while validating user message.', duration=None)

        yield chat_history + [{'role': 'user', 'content': user_message}]

    def _upload_file(self,
                     uploaded_file_path: str) -> None:
        """Uploads a file to the context retriever service."""

        try:
            upload_error = self._context_retriever_service.upload_file(uploaded_file_path)

        except requests.HTTPError as e:
            _logger().error('Failed to upload file to context retriever: %s', e)

            raise gr.Error('Failed to upload a file.', duration=None)

        if upload_error is not None:
            _logger().error('Failed to upload file to context retriever: %s', upload_error)

            raise gr.Error(upload_error,
                           title='Failed to upload a file',
                           duration=None)

        gr.Success('File uploaded successfully!', duration=5)
