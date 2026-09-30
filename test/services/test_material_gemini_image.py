# -*- coding: utf-8 -*-
import io
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from google.genai import errors as genai_errors
from PIL import Image

from app.config import config
from app.models.schema import VideoAspect
from app.services import material


def _png_bytes(width=64, height=112):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (10, 120, 200)).save(buffer, format="PNG")
    return buffer.getvalue()


def _image_response(data):
    part = SimpleNamespace(inline_data=SimpleNamespace(data=data))
    candidate = SimpleNamespace(
        content=SimpleNamespace(parts=[part]), finish_reason="STOP"
    )
    return SimpleNamespace(candidates=[candidate])


def _api_error(code):
    return genai_errors.APIError(
        code, {"error": {"code": code, "message": f"error {code}"}}
    )


class TestGeminiImageProvider(unittest.TestCase):
    """Gemini 文生图素材源，全部替换 google-genai Client，不依赖真实网络与计费。"""

    def setUp(self):
        self.original_app_config = dict(config.app)
        self.save_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.save_dir, ignore_errors=True)
        config.app["gemini_api_key"] = "gm-test-key"
        config.app.pop("gemini_image_model", None)
        config.app.pop("gemini_image_prompt_template", None)

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original_app_config)

    def _patch_client(self, side_effect):
        client = MagicMock()
        client.__enter__.return_value = client
        client.models.generate_content.side_effect = side_effect
        return patch("google.genai.Client", return_value=client), client

    def test_is_gemini_image_enabled_requires_api_key(self):
        self.assertTrue(material.is_gemini_image_enabled({"gemini_api_key": "k"}))
        self.assertFalse(material.is_gemini_image_enabled({"gemini_api_key": " "}))
        self.assertFalse(material.is_gemini_image_enabled({}))

    def test_generate_images_gemini_saves_image_with_aspect_and_prompt(self):
        client_patch, client = self._patch_client([_image_response(_png_bytes())])
        with client_patch:
            items = material.generate_images_gemini(
                search_term="Netflix red envelope on a doormat",
                minimum_duration=5,
                video_aspect=VideoAspect.portrait,
                save_dir=self.save_dir,
            )

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].provider, "gemini_image")
        self.assertEqual(items[0].duration, 5)
        self.assertEqual(items[0].source_info["rendition"]["height"], 112)
        kwargs = client.models.generate_content.call_args.kwargs
        self.assertEqual(kwargs["model"], material.GEMINI_IMAGE_DEFAULT_MODEL)
        self.assertIn("Netflix red envelope on a doormat", kwargs["contents"])
        self.assertEqual(kwargs["config"].image_config.aspect_ratio, "9:16")

    def test_generate_images_gemini_uses_configured_model_and_template(self):
        config.app["gemini_image_model"] = "custom-image-model"
        config.app["gemini_image_prompt_template"] = "flat illustration of {term}"
        client_patch, client = self._patch_client([_image_response(_png_bytes())])
        with client_patch:
            material.generate_images_gemini("a mailbox", 5, VideoAspect.landscape)

        kwargs = client.models.generate_content.call_args.kwargs
        self.assertEqual(kwargs["model"], "custom-image-model")
        self.assertEqual(kwargs["contents"], "flat illustration of a mailbox")
        self.assertEqual(kwargs["config"].image_config.aspect_ratio, "16:9")

    def test_generate_images_gemini_retries_503_then_succeeds(self):
        client_patch, client = self._patch_client(
            [_api_error(503), _image_response(_png_bytes())]
        )
        with client_patch, patch("app.services.material.time.sleep") as sleep:
            items = material.generate_images_gemini("term", 5, save_dir=self.save_dir)

        self.assertEqual(len(items), 1)
        self.assertEqual(client.models.generate_content.call_count, 2)
        sleep.assert_called_once()

    def test_generate_images_gemini_skips_term_on_rejection(self):
        client_patch, client = self._patch_client([_api_error(400)])
        with client_patch, patch("app.services.material.time.sleep") as sleep:
            items = material.generate_images_gemini("term", 5, save_dir=self.save_dir)

        self.assertEqual(items, [])
        self.assertEqual(client.models.generate_content.call_count, 1)
        sleep.assert_not_called()

    def test_generate_images_gemini_skips_response_without_image(self):
        empty = SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(parts=[]), finish_reason="SAFETY"
                )
            ]
        )
        client_patch, _ = self._patch_client([empty])
        with client_patch:
            self.assertEqual(material.generate_images_gemini("term", 5), [])

    def test_generate_images_gemini_does_not_retry_unconfirmed_errors(self):
        client_patch, client = self._patch_client([ConnectionError("reset")])
        with client_patch, self.assertRaises(material.OpenAIImageUnconfirmedError):
            material.generate_images_gemini("term", 5, save_dir=self.save_dir)
        self.assertEqual(client.models.generate_content.call_count, 1)

    def test_download_videos_gemini_image_generates_on_demand_and_stops(self):
        def fake_generate(search_term, minimum_duration, video_aspect, save_dir=""):
            item = material.MaterialInfo()
            item.provider = "gemini_image"
            item.url = f"/tmp/{search_term}.png"
            item.duration = minimum_duration
            item.source_info = {"provider": "gemini_image"}
            return [item]

        with (
            patch(
                "app.services.material.generate_images_gemini",
                side_effect=fake_generate,
            ) as generate,
            patch(
                "app.services.material._render_openai_image_video",
                side_effect=lambda path, duration: f"{path}.mp4",
            ),
            patch("app.services.material.generate_images_openai") as openai_generate,
        ):
            result = material.download_videos(
                task_id="test-gemini-image-lazy",
                search_terms=["a", "b", "c"],
                source="gemini_image",
                audio_duration=8,
                max_clip_duration=5,
            )

        self.assertEqual(generate.call_count, 2)
        openai_generate.assert_not_called()
        self.assertEqual(result, ["/tmp/a.png.mp4", "/tmp/b.png.mp4"])


if __name__ == "__main__":
    unittest.main()


class TestGeminiLlmFallback(unittest.TestCase):
    """Gemini 文本模型繁忙时先退避重试，再切换备用模型。"""

    def _genai(self, side_effect):
        client = MagicMock()
        client.__enter__.return_value = client
        client.models.generate_content.side_effect = side_effect
        return SimpleNamespace(Client=MagicMock(return_value=client)), client

    def test_falls_back_to_next_model_after_retries(self):
        from app.services import llm

        genai, client = self._genai(
            [_api_error(503), _api_error(503), _api_error(503), "ok"]
        )
        with patch("app.services.llm.time.sleep"):
            result = llm._gemini_generate_with_fallback(
                genai,
                api_key="k",
                http_options=None,
                model_name="main",
                prompt="p",
                generation_config=None,
                app_config={"gemini_fallback_model_names": ["backup"]},
            )

        self.assertEqual(result, "ok")
        models = [c.kwargs["model"] for c in client.models.generate_content.call_args_list]
        self.assertEqual(models, ["main", "main", "main", "backup"])

    def test_non_retryable_error_is_raised_immediately(self):
        from app.services import llm

        genai, client = self._genai([_api_error(404)])
        with self.assertRaises(genai_errors.APIError):
            llm._gemini_generate_with_fallback(
                genai,
                api_key="k",
                http_options=None,
                model_name="main",
                prompt="p",
                generation_config=None,
                app_config={"gemini_fallback_model_names": ["backup"]},
            )
        self.assertEqual(client.models.generate_content.call_count, 1)
