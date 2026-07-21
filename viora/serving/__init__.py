"""Serving layer: the FastAPI app and the static Viora Studio web client.

The HTTP API (``viora/serving/api.py``) is added in a later milestone; the web
client under ``web/`` is self-contained and works today against a mock inference
adapter, and will call the real endpoints once the API lands.
"""
