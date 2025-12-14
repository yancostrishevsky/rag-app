"""Contains the main entrypoint for the web application."""
import logging.config
import os
from datetime import datetime

import gradio as gr
import hydra
import omegaconf
from web_app.backend import context_retriever
from web_app.backend import llm_proxy
from web_app.gui import main_controller
from web_app.gui import utils as gui_utils


def _logger() -> logging.Logger:
    return logging.getLogger()


@hydra.main(version_base=None, config_path='cfg', config_name='main')
def main(cfg: omegaconf.DictConfig) -> None:
    """Initializes and serves the web app."""

    os.makedirs(os.path.join(cfg.persist_data_path, 'log'), exist_ok=True)

    logging.config.dictConfig({
        'version': 1,
        'loggers': {
            'root': {
                'level': 'NOTSET',
                'handlers': ['console', 'file'],
                'propagate': True
            },
            'httpx': {
                'level': 'WARNING',
                'propagate': False
            },
            'httpcore': {
                'level': 'WARNING',
                'propagate': False
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': 'INFO',
                'formatter': 'default'
            },
            'file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'DEBUG',
                'formatter': 'default',
                'filename': os.path.join(cfg.persist_data_path,
                                         'log',
                                         f'{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.log'),
                'maxBytes': 5_000_000,
                'backupCount': 5,
                'encoding': 'utf-8',
            }
        },
        'formatters': {
            'default': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            }
        }
    })

    _logger().info('Starting web application with configuration:\n%s',
                   omegaconf.OmegaConf.to_yaml(cfg))

    _logger().info('Creating services for backend communication...')

    context_retriever_service = context_retriever.ContextRetrieverService(
        cfg.context_retriever_url
    )

    llm_proxy_service = llm_proxy.LLMProxyService(
        cfg.llm_proxy_url
    )

    _logger().info('Rendering GUI...')

    with gr.Blocks(fill_height=True,
                   title='AGH Chat',
                   css=gui_utils.CUSTOM_CSS) as web_application:
        main_controller.MainController(context_retriever_service, llm_proxy_service).render_gui()

    web_application.launch(server_name=cfg.web_app_host,
                           server_port=cfg.web_app_port)


if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter
