"""Contains the main entrypoint for the web application."""
import logging.config
import os

import gradio as gr
import hydra
import omegaconf

from web_app.backend import (context_retriever, llm_proxy)
from web_app import gui


def _logger() -> logging.Logger:
    return logging.getLogger('web_app')


with hydra.initialize(version_base=None, config_path='cfg'):
    cfg = hydra.compose(config_name='main_dev')

    _logger().info('Starting web application with configuration:\n%s',
                   omegaconf.OmegaConf.to_yaml(cfg))

os.makedirs(os.path.join(cfg.persist_data_path, 'log'), exist_ok=True)

logging.config.dictConfig({
    'version': 1,
    'loggers': {
        'root': {
            'level': 'NOTSET',
            'handlers': ['console'],
            'propagate': True
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'DEBUG',
            'formatter': 'default'
        }
    },
    'formatters': {
        'default': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        }
    }
})

context_retriever_service = context_retriever.ContextRetrieverService(
    cfg.context_retriever_url
)

llm_proxy_service = llm_proxy.LLMProxyService(
    cfg.llm_proxy_url
)

CUSTOM_CSS = """
.retrieved-docs {
  max-height: 50vh;
  overflow-y: auto;
}
"""

with gr.Blocks(fill_height=True, title='AGH Chat', css=CUSTOM_CSS) as web_application:
    gui.MainController(context_retriever_service, llm_proxy_service).render_gui()

web_application.launch(server_name=cfg.web_app_host,
                       server_port=cfg.web_app_port)
