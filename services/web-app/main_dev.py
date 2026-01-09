"""Contains the main entrypoint for the web application."""
import logging.config

import gradio as gr
import hydra
import omegaconf
from web_app.backend import context_retriever
from web_app.backend import llm_proxy
from web_app.gui import main_controller
from web_app.gui import utils as gui_utils
from web_app.backend import utils as backend_utils


def _logger() -> logging.Logger:
    return logging.getLogger('web_app')


with hydra.initialize(version_base=None, config_path='cfg'):
    cfg = hydra.compose(config_name='main_dev')

    _logger().info('Starting web application with configuration:\n%s',
                   omegaconf.OmegaConf.to_yaml(cfg))

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
    endpoint_cfg=backend_utils.EndpointConnectionCfg(**cfg.context_retriever_cfg)
)

llm_proxy_service = llm_proxy.LLMProxyService(
    endpoint_cfg=backend_utils.EndpointConnectionCfg(**cfg.llm_proxy_cfg)
)

with gr.Blocks(fill_height=True, title='AGH Chat', css=gui_utils.CUSTOM_CSS) as web_application:
    main_controller.MainController(context_retriever_service, llm_proxy_service).render_gui()

web_application.launch(server_name=cfg.web_app_host,
                       server_port=cfg.web_app_port)
