import asyncio
import os
import unittest

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from fastapi import HTTPException

import api
import generator


class _Response:
    def __init__(self, content):
        message = type("Message", (), {"content": content})()
        choice = type("Choice", (), {"message": message})()
        self.choices = [choice]


class _Completions:
    def __init__(self, content):
        self.content = content

    def create(self, **_kwargs):
        return _Response(self.content)


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.original_client = generator.client

    def tearDown(self):
        generator.client = self.original_client

    def _set_generation_response(self, content):
        completions = _Completions(content)
        generator.client = type(
            "Client", (), {"chat": type("Chat", (), {"completions": completions})()}
        )()

    def test_successful_generation_returns_final_answer(self):
        self._set_generation_response("  FastAPI and React.  ")

        answer = generator.generate_answer("What was used?", "FastAPI, React", [])

        self.assertEqual(answer, "FastAPI and React.")

    def test_empty_generation_returns_document_fallback(self):
        self._set_generation_response(None)

        answer = generator.generate_answer("What was used?", "FastAPI, React", [])

        self.assertEqual(answer, "I don't know based on the provided document.")

    def test_generation_exception_becomes_bad_gateway(self):
        original_rewrite = api.rewrite_query
        original_embed = api.embed_texts
        original_search = api.search
        original_generate = api.generate_answer
        try:
            api.rewrite_query = lambda question, history: question
            api.embed_texts = lambda texts: "query-vector"
            api.search = lambda embedding, top_k: {"documents": [["context"]]}

            def fail_generation(*_args):
                raise RuntimeError("OpenRouter unavailable")

            api.generate_answer = fail_generation

            with self.assertRaises(HTTPException) as raised:
                asyncio.run(api.ask_question(api.AskRequest(question="Question", history=[])))

            self.assertEqual(raised.exception.status_code, 502)
            self.assertEqual(raised.exception.detail, "Unable to generate an answer at this time.")
        finally:
            api.rewrite_query = original_rewrite
            api.embed_texts = original_embed
            api.search = original_search
            api.generate_answer = original_generate


if __name__ == "__main__":
    unittest.main()
