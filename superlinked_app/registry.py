# registry.py
from superlinked_app.apps.sota_app import config as config_sota
from superlinked_app.apps.sota_app import data_prep as data_prep_sota
from superlinked_app.apps.sota_app import index as index_sota
from superlinked_app.apps.sota_app import query as query_sota
from superlinked_app.apps.sota_app_baseline import config as config_baseline
from superlinked_app.apps.sota_app_baseline import \
    data_prep as data_prep_baseline
from superlinked_app.apps.sota_app_baseline import index as index_baseline
from superlinked_app.apps.sota_app_baseline import query as query_baseline

MODULE_REGISTRY = {
    "sota_app": {
        "data_prep": data_prep_sota,
        "config": config_sota,
        "index": index_sota,
        "query": query_sota,
    },
    "sota_app_baseline": {
        "data_prep": data_prep_baseline,
        "config": config_baseline,
        "index": index_baseline,
        "query": query_baseline,
    },
}
