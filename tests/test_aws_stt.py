import json
import os
import unittest
from unittest.mock import MagicMock, patch

from youtube_to_docs import llms


class TestAWSSTT(unittest.TestCase):
    def setUp(self):
        self.env_patcher = patch.dict(os.environ, {"YTD_S3_BUCKET_NAME": "test-bucket"})
        self.env_patcher.start()

    def tearDown(self):
        self.env_patcher.stop()

    @patch("youtube_to_docs.llms._transcribe_aws")
    def test_aws_dispatch(self, mock_transcribe):
        """Test that aws-transcribe is dispatched to _transcribe_aws."""
        mock_transcribe.return_value = ("transcript", "srt_content", 0, 0)
        llms.generate_transcript("aws-transcribe", "audio.m4a", "http://url")
        mock_transcribe.assert_called_once()

    @patch("boto3.client")
    @patch("time.sleep", return_value=None)
    def test_transcribe_aws_success(self, mock_sleep, mock_boto_client):
        """Test _transcribe_aws success path."""
        mock_s3 = MagicMock()
        mock_transcribe = MagicMock()

        def side_effect(service_name, **kwargs):
            if service_name == "s3":
                return mock_s3
            if service_name == "transcribe":
                return mock_transcribe
            return MagicMock()

        mock_boto_client.side_effect = side_effect

        # Mock Transcribe job status
        mock_transcribe.get_transcription_job.return_value = {
            "TranscriptionJob": {"TranscriptionJobStatus": "COMPLETED"}
        }

        # Mock S3 get_object for transcript
        mock_body = MagicMock()
        transcript_data = {
            "results": {
                "transcripts": [{"transcript": "Hello world"}],
                "items": [
                    {
                        "type": "pronunciation",
                        "alternatives": [{"content": "Hello"}],
                        "start_time": "0.0",
                        "end_time": "0.5",
                    },
                    {
                        "type": "pronunciation",
                        "alternatives": [{"content": "world"}],
                        "start_time": "0.6",
                        "end_time": "1.0",
                    },
                    {"type": "punctuation", "alternatives": [{"content": "!"}]},
                ],
            }
        }
        mock_body.read.return_value = json.dumps(transcript_data).encode("utf-8")
        mock_s3.get_object.return_value = {"Body": mock_body}

        transcript, srt, in_tok, out_tok = llms._transcribe_aws(
            "aws-transcribe", "audio.m4a", "http://url"
        )

        self.assertEqual(transcript, "Hello world")
        self.assertIn("Hello world!", srt)
        self.assertIn("00:00:00,000 --> 00:00:01,000", srt)

    @patch("boto3.client")
    def test_transcribe_aws_no_bucket(self, mock_boto_client):
        """Test _transcribe_aws failure when bucket is missing."""
        with patch.dict(os.environ, {}, clear=True):
            transcript, srt, in_tok, out_tok = llms._transcribe_aws(
                "aws-transcribe", "audio.m4a", "http://url"
            )
            self.assertTrue(transcript.startswith("Error: YTD_S3_BUCKET_NAME"))


if __name__ == "__main__":
    unittest.main()
