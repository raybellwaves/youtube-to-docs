import os
import unittest
from unittest.mock import MagicMock, patch

from youtube_to_docs import infographic


class TestInfographic(unittest.TestCase):
    def setUp(self):
        # Mock environment variables
        self.env_patcher = patch.dict(
            os.environ,
            {
                "GEMINI_API_KEY": "fake_gemini_key",
            },
        )
        self.env_patcher.start()

    def tearDown(self):
        self.env_patcher.stop()

    @patch("youtube_to_docs.infographic.genai.Client")
    def test_generate_infographic_gemini(self, mock_client_cls):
        mock_client = mock_client_cls.return_value
        mock_resp = MagicMock()

        # Mocking the response structure for generate_images
        mock_image = MagicMock()
        mock_image.image.image_bytes = b"fake_image_bytes"
        mock_resp.generated_images = [mock_image]

        mock_client.models.generate_images.return_value = mock_resp

        image_bytes = infographic.generate_infographic(
            "gemini-2.5-flash-image", "Summary text", "Video Title"
        )

        self.assertEqual(image_bytes, b"fake_image_bytes")
        mock_client.models.generate_images.assert_called_once()

    def test_generate_infographic_none_model(self):
        image_bytes = infographic.generate_infographic(
            None, "Summary text", "Video Title"
        )
        self.assertIsNone(image_bytes)

    def test_generate_infographic_unsupported_model(self):
        image_bytes = infographic.generate_infographic(
            "unsupported-model", "Summary text", "Video Title"
        )
        self.assertIsNone(image_bytes)

    @patch("youtube_to_docs.infographic.genai.Client")
    def test_generate_infographic_no_images(self, mock_client_cls):
        mock_client = mock_client_cls.return_value
        mock_resp = MagicMock()
        mock_resp.generated_images = []
        mock_client.models.generate_images.return_value = mock_resp

        image_bytes = infographic.generate_infographic(
            "gemini-2.5-flash-image", "Summary text", "Video Title"
        )
        self.assertIsNone(image_bytes)


if __name__ == "__main__":
    unittest.main()
