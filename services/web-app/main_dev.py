"""Contains the main entrypoint for the web application."""
import logging.config

import gradio as gr
import hydra
import omegaconf
import web


def _logger():
    return logging.getLogger('web_app')


with hydra.initialize(version_base=None, config_path='cfg'):
    cfg = hydra.compose(config_name='main')

    _logger().info('Starting web application with configuration: %s',
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
            'level': 'INFO',
            'formatter': 'default'
        }
    },
    'formatters': {
        'default': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        }
    }
})

backend_service = web.backend_communication.BackendService(
    backend_url=cfg.backend_entrypoint_url
)

CUSTOM_CSS = """
.retrieved-docs {
  max-height: 30vh;
  overflow-y: auto;
}
"""

with gr.Blocks(fill_height=True, title='AGH Chat', css=CUSTOM_CSS) as web_app:
    web.gui.MainController(backend_service).render_gui()

web_app.launch(server_name=cfg.web_app_host,
               server_port=cfg.web_app_port)
